"""beam daemon: the control-plane hub, in Python (asyncio).

One daemon per node. Exactly one node is the head; it is the routing hub and the
authority on membership and placement. Non-head daemons keep a single connection
to the head and execute create/call/get/kill requests pushed down it.

This is pure control plane: it launches actor subprocesses, assigns GPUs, routes
method-call RPCs, and gathers small results. Tensor traffic goes over NCCL,
never through here. See docs/DESIGN.md.
"""

from __future__ import annotations  # keep `X | None` valid on py3.9

import asyncio
import glob
import json
import os
import secrets
import shlex
import struct
import subprocess
from collections.abc import Awaitable, Callable
from typing import Any


def _terminate(proc: subprocess.Popen | None) -> bool:
    """Best-effort kill of an actor worker subprocess that hasn't already exited.

    SIGTERM, brief non-blocking polls, then SIGKILL + short wait. Avoids multi-
    second blocking waits on the asyncio thread while still reaping stubborn
    workers before callers free GPU indices.

    Returns True if the process is gone (or was already gone / None), False if
    it may still be alive after best-effort kill (callers should not free GPUs).
    """
    if proc is None:
        return True
    if proc.poll() is not None:
        return True
    try:
        proc.terminate()
    except OSError:
        pass
    # Brief poll loop (no long blocking wait on the event-loop thread).
    for _ in range(10):
        if proc.poll() is not None:
            return True
        try:
            proc.wait(timeout=0.05)
            return True
        except Exception:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=0.5)
    except Exception:
        pass
    return proc.poll() is not None


# ---- wire framing: [4-byte big-endian len][JSON header][plen raw payload] ----


def encode_frame(header: dict, payload: bytes = b"") -> bytes:
    header = dict(header)
    header["plen"] = len(payload)
    h = json.dumps(header).encode()
    return struct.pack(">I", len(h)) + h + payload


_MAX_FRAME = 512 * 1024 * 1024  # corrupt-length guard; see _proto._MAX_FRAME


async def read_frame(reader: asyncio.StreamReader) -> tuple[dict, bytes]:
    n = struct.unpack(">I", await reader.readexactly(4))[0]
    if n == 0 or n > _MAX_FRAME:
        raise ConnectionError("bad frame header length %d" % n)
    header = json.loads(await reader.readexactly(n))
    if not isinstance(header, dict):  # valid JSON but not an object (e.g. a bare int)
        raise ConnectionError("frame header is not a JSON object")
    plen = header.get("plen", 0)
    if plen < 0 or plen > _MAX_FRAME:
        raise ConnectionError("bad frame payload length %d" % plen)
    payload = await reader.readexactly(plen) if plen else b""
    return header, payload


class Peer:
    """Bidirectional RPC mux over one connection. Either side can issue call();
    the other side's handler answers. Responses match requests by reqid."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handler: Callable[[Peer, dict, bytes], Awaitable[tuple[dict, bytes]]],
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.handler = handler
        self.pending: dict[int, asyncio.Future] = {}
        self.next_id = 0
        self.wlock = asyncio.Lock()
        self.on_close: Callable[[], Any] | None = None
        self.closed = False
        # Set when close() finishes so concurrent await close() joins release.
        self._close_done: asyncio.Future | None = None
        self._tasks: set[asyncio.Task] = (
            set()
        )  # keep handler task refs so they aren't GC'd mid-flight
        # per-client ownership, for cleanup on disconnect
        self.created_pgs: list[str] = []
        self.created_actors: list[str] = []
        # in-flight create_actor/create_pg RPCs (worker driver peers): release
        # must not free PGs while a create is still running against them.
        self.in_flight = 0

    async def serve(self) -> None:
        try:
            while True:
                header, payload = await read_frame(self.reader)
                if header.get("resp"):
                    rid: Any = header.get("reqid")
                    fut = self.pending.pop(rid, None)
                    if fut and not fut.done():
                        fut.set_result((header, payload))
                else:
                    # Do not start work after close began: release_client would
                    # miss ownership updates from handlers scheduled too late.
                    if self.closed:
                        continue
                    t = asyncio.create_task(self._handle(header, payload))
                    self._tasks.add(t)
                    t.add_done_callback(self._tasks.discard)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except Exception:  # unexpected: surface it instead of dying silently
            import sys
            import traceback

            traceback.print_exc(file=sys.stderr)
        finally:
            await self.close()

    async def _handle(self, header: dict, payload: bytes) -> None:
        try:
            resp, rpl = await self.handler(self, header, payload)
        except Exception as e:  # never let a handler kill the read loop
            resp, rpl = {"err": str(e)}, b""
        if resp is None:
            resp = {"t": header.get("t", "") + "_ok"}
        resp["reqid"] = header.get("reqid")
        resp["resp"] = True
        try:
            await self.send(resp, rpl)
        except (ConnectionError, OSError):
            pass

    async def send(self, header: dict, payload: bytes = b"") -> None:
        async with self.wlock:
            if self.closed:
                raise ConnectionError("connection closed")
            self.writer.write(encode_frame(header, payload))
            await self.writer.drain()

    async def call(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        # After close(), write/drain can appear to succeed while nothing will
        # ever answer; fail fast so callers (e.g. _dispatch) set slot.err.
        if self.closed:
            raise ConnectionError("connection closed")
        self.next_id += 1
        rid = self.next_id
        # copy: never mutate the caller's dict. A routing handler passes the
        # message it received straight to call(); mutating reqid here would
        # corrupt the reqid _handle echoes back on its own response.
        header = dict(header)
        header["reqid"] = rid
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        try:
            await self.send(header, payload)
            rheader, rpayload = await fut
        except BaseException:
            # send failed before a response: drop the orphan future so close()
            # does not set_exception on a waiter nobody is awaiting.
            self.pending.pop(rid, None)
            raise
        if rheader.get("err"):
            raise RuntimeError(rheader["err"])
        return rheader, rpayload

    async def close(self) -> None:
        if self.closed:
            # Another close is in flight: join it so on_hello's await close()
            # does not return before release_client finishes.
            done = self._close_done
            if done is not None and not done.done():
                await done
            return
        self.closed = True
        loop = asyncio.get_running_loop()
        self._close_done = loop.create_future()
        # Fail pending RPCs *before* draining handlers. A handler blocked in
        # same-peer call() (e.g. head create pushed to the requesting worker)
        # waits on pending; gathering first deadlocks and on_close never runs.
        for fut in list(self.pending.values()):
            if not fut.done():
                fut.set_exception(ConnectionError("connection closed"))
        self.pending.clear()
        # Always close the socket and fire on_close even if drain is cancelled
        # (nested close from on_kill mid-drain would otherwise no-op forever).
        try:
            # Drain framed handlers so ownership / closed-path cleanup finishes
            # before release_client. Bound wait so a hung get/create cannot block
            # forever; cancel stragglers after the budget.
            tasks = list(self._tasks)
            if tasks:
                _done, pending = await asyncio.wait(tasks, timeout=30.0)
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        finally:
            try:
                self.writer.close()
            except OSError:
                pass
            cb = self.on_close
            self.on_close = None  # fire at most once
            try:
                if cb is not None:
                    try:
                        r = cb()
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:
                        pass
            finally:
                if self._close_done is not None and not self._close_done.done():
                    self._close_done.set_result(None)


def detect_gpus(override: int | None = None) -> int:
    if override is not None and override >= 0:
        return override
    env = os.environ.get("BEAM_NUM_GPUS")
    if env:
        return int(env)
    return len(glob.glob("/dev/nvidia[0-9]*"))


def new_node_id() -> str:
    return "n" + secrets.token_hex(4)


def owner_of(obj_id: str) -> str:
    i = obj_id.rfind("-o")
    return obj_id[:i] if i >= 0 else ""


class ActorProc:
    def __init__(
        self,
        actor_id: str,
        peer: Peer,
        gpus: list[int],
        proc: subprocess.Popen | None = None,
    ) -> None:
        self.id = actor_id
        self.peer = peer
        self.gpus = gpus
        self.proc = proc  # the python -m ray._worker subprocess
        self.lock = asyncio.Lock()  # Ray actors are single-threaded


class ObjSlot:
    def __init__(self) -> None:
        self.ev = asyncio.Event()
        self.data = b""
        self.err = ""


class Daemon:
    def __init__(self, is_head: bool, node_id: str, ip: str, num_gpus: int) -> None:
        self.self_info: dict[str, Any] = {
            "node": node_id,
            "ip": ip,
            "ngpu": num_gpus,
            "alive": True,
            "head": is_head,
        }
        self.is_head = is_head
        self.node_id = node_id
        self.num_gpus = num_gpus
        self.head_peer: Peer | None = None
        self.sock_path: str | None = None
        # strong refs for fire-and-forget tasks (dispatch, head serve) so the
        # event loop cannot GC them mid-flight (see Peer._tasks for the same).
        self._bg_tasks: set[asyncio.Task] = set()

        self.gpu_used = [False] * num_gpus
        self.actors: dict[str, ActorProc] = {}
        self.objects: dict[str, ObjSlot] = {}
        self.pending_workers: dict[str, asyncio.Future] = {}
        # actor_id -> (Popen, Peer|None, gpus) while _host_actor is in flight
        self._hosting: dict[str, tuple[subprocess.Popen, Peer | None, list[int]]] = {}
        # ids killed by head before the worker entered _hosting (remote create race)
        self._kill_pending: set[str] = set()
        # actor_id -> owner node for kills that failed after the owner peer closed
        self._orphans: dict[str, str] = {}
        # actor_id -> live reaper task (so we can restart after cancel)
        self._orphan_tasks: dict[str, asyncio.Task] = {}
        # PGs deferred until orphan actors are reaped (avoid double-booking GPUs)
        self._orphan_pgs: set[str] = set()
        self.obj_seq = 0
        self.id_seq = 0

        # head-only state, but typed/empty on every node so attribute access and
        # mypy are uniform. Only the head seeds itself into `nodes`.
        self.nodes: dict[str, dict[str, Any]] = {}
        self.pgs: dict[str, list[dict[str, Any]]] = {}
        self.actor_loc: dict[str, str] = {}
        if is_head:
            self.nodes[node_id] = {"info": dict(self.self_info), "peer": None}

    def _track(self, task: asyncio.Task) -> asyncio.Task:
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ---- ids ----
    def _next_obj(self) -> str:
        self.obj_seq += 1
        return f"{self.node_id}-o{self.obj_seq}"

    def _next_id(self, kind: str) -> str:
        self.id_seq += 1
        return f"{self.node_id}-{kind}{self.id_seq}"

    # ---- servers ----
    async def serve_unix(self, path: str) -> None:
        self.sock_path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        await asyncio.start_unix_server(self._on_conn, path=path)

    async def serve_tcp(self, host: str, port: int) -> None:
        await asyncio.start_server(self._on_conn, host=host, port=port)

    async def _on_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = Peer(reader, writer, self.handle)
        # default: if this turns out to be a driver, free its resources on close.
        # on_hello overwrites this for joining worker daemons (-> _drop_node).
        peer.on_close = lambda: self.release_client(peer)
        await peer.serve()

    async def join_head(self, host: str, port: int, retries: int = 60) -> None:
        # the worker daemon may start before the head's TCP listener is up
        # (e.g. both launched together), so retry the dial with backoff.
        for attempt in range(retries):
            try:
                reader, writer = await asyncio.open_connection(host, port)
                break
            except OSError:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(1)
        self.head_peer = Peer(reader, writer, self.handle)

        # If the head link dies, reap local actors: the head cannot RPC-kill us
        # anymore, and without this island workers hold GPUs forever.
        def _on_head_lost() -> None:
            self.shutdown()

        self.head_peer.on_close = _on_head_lost
        self._track(asyncio.create_task(self.head_peer.serve()))
        await self.head_peer.call(
            {"t": "hello", "node": self.node_id, "ip": self.self_info["ip"], "ngpu": self.num_gpus}
        )

    # ---- dispatch ----
    async def handle(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        t = m.get("t")
        fn = getattr(self, "on_" + t, None) if t else None
        if fn is None:
            return {"err": "unknown message type: %s" % t}, b""
        return await fn(peer, m, payload)

    async def _forward_head(self, m: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        if self.head_peer is None:
            return {"err": "no head connection"}, b""
        r, pl = await self.head_peer.call(m, payload)
        return r, pl

    def _peer_for(self, node: str) -> Peer | None:
        rec = self.nodes.get(node)
        return rec["peer"] if rec else None

    # ---- membership (head) ----
    async def on_hello(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if not self.is_head:
            return {"err": "not the head node"}, b""
        node = m["node"]
        # Re-hello with the same node id: move driver-on-worker ownership to the
        # new peer, mark the old peer superseded (so an in-flight release_client
        # skips transferred ids), then close it.
        old = self.nodes.get(node)
        if old is not None:
            old_peer = old.get("peer")
            if old_peer is not None and old_peer is not peer:
                old_peer.superseded = True
                # So in-flight release_client can hand claimed PGs to the new peer
                # after transfer cleared old lists (avoids free-under-live-actor).
                old_peer.superseded_by = peer
                peer.created_actors.extend(old_peer.created_actors)
                peer.created_pgs.extend(old_peer.created_pgs)
                old_peer.created_actors.clear()
                old_peer.created_pgs.clear()
                # stragglers after transfer: release only what is still on old_peer
                old_peer.on_close = lambda p=old_peer: self.release_client(p)
                await old_peer.close()

        # Free PGs/actors this worker connection owns (driver-on-worker forwards
        # track ownership on this peer), then drop membership. Without the
        # release, a dead worker leaked placement groups that still reserved
        # GPUs on live nodes.
        async def _on_worker_close() -> None:
            rec = self.nodes.get(node)
            # Stale close after reconnect: live peer already replaced this node.
            if rec is not None and rec.get("peer") is not peer:
                return
            await self.release_client(peer)
            self._drop_node(node, peer)

        # New peer already closed (raced with disconnect): never install a dead
        # peer as the live node without a second on_close. Release transferred
        # ownership now, then drop any stale membership still pointing at old.
        if peer.closed:
            await self.release_client(peer)
            if old is not None:
                old_p = old.get("peer")
                rec = self.nodes.get(node)
                if rec is not None and rec.get("peer") is old_p:
                    self._drop_node(node, old_p)
            return {"t": "hello_ok"}, b""

        self.nodes[node] = {
            "info": {
                "node": node,
                "ip": m.get("ip", ""),
                "ngpu": m.get("ngpu", 0),
                "alive": True,
                "head": False,
            },
            "peer": peer,
        }
        peer.on_close = _on_worker_close
        return {"t": "hello_ok"}, b""

    def _drop_node(self, node: str, peer: Peer | None = None) -> None:
        rec = self.nodes.get(node)
        # stale close from an old connection after the node already reconnected:
        # ignore it, don't drop the live node.
        if peer is not None and rec is not None and rec["peer"] is not peer:
            return
        # remove the node entirely (a reconnect makes a fresh entry). Leaving a
        # phantom "down" node behind would fail `ray status` health checks forever,
        # since a restarted worker gets a new node id.
        self.nodes.pop(node, None)
        # actors on a gone node are unreachable; drop their routing so calls fail
        # cleanly ("unknown actor") instead of hanging on a dead connection.
        for aid in [a for a, n in self.actor_loc.items() if n == node]:
            del self.actor_loc[aid]

    def _drop_actor(self, actor_id: str) -> None:
        """Reclaim an actor whose worker subprocess died (crash, not just kill):
        free its GPUs and drop it from the local + routing tables. Idempotent, so
        it is safe to also fire after an explicit on_kill."""
        ap = self.actors.pop(actor_id, None)
        if ap:
            for g in ap.gpus:
                if 0 <= g < len(self.gpu_used):
                    self.gpu_used[g] = False
        self.actor_loc.pop(actor_id, None)

    def _used_on_node(self, node: str) -> int:
        used = 0
        for pg in self.pgs.values():
            for b in pg:
                if b["node"] == node and b["gpu"] >= 0:
                    used += 1
        return used

    async def on_status(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if not self.is_head:
            r, _ = await self._forward_head({"t": "status"})
            return r, b""
        out = []
        for node, rec in self.nodes.items():
            info = dict(rec["info"])
            info["used"] = self._used_on_node(node)
            if node == self.node_id:
                info["used"] += sum(self.gpu_used)
            out.append(info)
        return {"t": "status_ok", "nodes": out}, b""

    # ---- placement groups (head) ----
    async def on_create_pg(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if not self.is_head:
            if peer is not None:
                peer.in_flight += 1
            try:
                # Explicit task + shield so Peer.close cancel cannot drop a
                # head-side PG that already completed while we were waiting.
                fwd_task = asyncio.create_task(self._forward_head(m, payload))
                try:
                    r, pl = await asyncio.shield(fwd_task)
                except asyncio.CancelledError:
                    rpl = await self._await_fwd_after_cancel(fwd_task, peer, "pg")
                    if rpl is not None and peer is not None and rpl[0].get("pg"):
                        await asyncio.shield(self._worker_cleanup_pg(peer, rpl[0]["pg"]))
                    raise
                if peer is not None and r.get("pg"):
                    if peer.closed:
                        # Shield so Peer.close cancel cannot abort remove_pg
                        # before the in_flight pin/retry is installed.
                        await asyncio.shield(self._worker_cleanup_pg(peer, r["pg"]))
                        return (
                            {"err": "client disconnected during placement group create"},
                            b"",
                        )
                    peer.created_pgs.append(r["pg"])
                return r, pl
            finally:
                if peer is not None:
                    peer.in_flight = max(0, peer.in_flight - 1)
        free = {}
        for node, rec in self.nodes.items():
            if not rec["info"]["alive"]:
                continue
            used = set()
            for pg in self.pgs.values():
                for b in pg:
                    if b["node"] == node and b["gpu"] >= 0:
                        used.add(b["gpu"])
            if node == self.node_id:
                used.update(i for i, u in enumerate(self.gpu_used) if u)
            free[node] = [i for i in range(rec["info"]["ngpu"]) if i not in used]

        bundles = []
        for spec in m.get("specs", []):
            if not spec.get("GPU", 0):
                bundles.append({"node": self.node_id, "gpu": -1})
                continue
            placed = False
            for node in self.nodes:
                if free.get(node):
                    bundles.append({"node": node, "gpu": free[node].pop(0)})
                    placed = True
                    break
            if not placed:
                return {"err": "placement group needs more GPUs than the cluster has free"}, b""

        pg_id = self._next_id("pg")
        self.pgs[pg_id] = bundles
        if peer is not None:
            if peer.closed:
                self.pgs.pop(pg_id, None)
                return {"err": "client disconnected during placement group create"}, b""
            peer.created_pgs.append(pg_id)
        return {"t": "create_pg_ok", "pg": pg_id}, b""

    async def on_remove_pg(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if not self.is_head:
            return await self._forward_head(m)
        pg_id: Any = m.get("pg")
        self.pgs.pop(pg_id, None)
        return {"t": "remove_pg_ok"}, b""

    async def on_pg_table(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if not self.is_head:
            return await self._forward_head(m)

        def encode(pg: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = []
            for b in pg:
                spec = {"GPU": 1} if b["gpu"] >= 0 else {}
                out.append({"node": b["node"], "spec": spec})
            return out

        pg_id = m.get("pg")
        if pg_id:
            pg = self.pgs.get(pg_id)
            if pg is None:
                return {"err": "unknown placement group %s" % pg_id}, b""
            data: dict[str, Any] = {"bundles": encode(pg)}
        else:
            data = {"pgs": {k: encode(v) for k, v in self.pgs.items()}}
        return {"t": "pg_table_ok", "data": data}, b""

    async def on_resources(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if not self.is_head:
            return await self._forward_head({"t": "resources"})
        out = {}
        for node, rec in self.nodes.items():
            used = self._used_on_node(node)
            if node == self.node_id:
                used += sum(self.gpu_used)
            free = max(0, rec["info"]["ngpu"] - used)
            out[node] = {"GPU": float(free), "CPU": 1.0}
        return {"t": "resources_ok", "data": out}, b""

    # ---- actors ----
    async def on_create_actor(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if self.is_head:
            node, gpus, err = self._place_actor(m)
            if err:
                return {"err": err}, b""
            # _place_actor returns non-None node/gpus whenever err is falsy
            assert node is not None and gpus is not None
            actor_id = self._next_id("a")
            self.actor_loc[actor_id] = node
            if peer is not None:
                peer.created_actors.append(actor_id)
            req = {"t": "create_actor", "actor": actor_id, "gpus": gpus}
            try:
                if node == self.node_id:
                    resp, rpl = await self._host_actor(req, payload)
                    if resp.get("err"):
                        raise RuntimeError(resp["err"])
                else:
                    p = self._peer_for(node)
                    if p is None:
                        raise RuntimeError("node %s is not available" % node)
                    await p.call(req, payload)
                    resp, rpl = (
                        {
                            "t": "create_actor_ok",
                            "actor": actor_id,
                            "gpus": gpus,
                            "node": node,
                        },
                        b"",
                    )
                # Client disconnected while create was in flight.
                # Re-hello supersede: ownership usually moved to the new peer.
                # A late _handle that only appended after transfer still has the
                # id on this dead peer — orphan-reap instead of silent leak.
                if peer is not None and peer.closed:
                    if getattr(peer, "superseded", False):
                        if actor_id in peer.created_actors:
                            peer.created_actors.remove(actor_id)
                            owned_elsewhere = False
                            for rec in self.nodes.values():
                                p = rec.get("peer")
                                if (
                                    p is not None
                                    and p is not peer
                                    and actor_id in getattr(p, "created_actors", ())
                                ):
                                    owned_elsewhere = True
                                    break
                            if not owned_elsewhere and self._actor_still_tracked(actor_id):
                                self._schedule_orphan_reap(actor_id, node)
                        return resp, rpl
                    await self._rollback_failed_create(peer, actor_id, node, gpus)
                    return {"err": "client disconnected during actor create"}, b""
                return resp, rpl
            except Exception as e:
                # Best-effort kill on the owner first: the worker may already
                # have registered the actor even though the create RPC failed.
                await self._rollback_failed_create(peer, actor_id, node, gpus)
                return {"err": str(e)}, b""
        # worker node: a head push carries a pre-assigned "actor" id; a request
        # from a local driver does not, so route it to the head for placement.
        # This is what lets the vLLM driver run on a worker node (CPU head, GPU
        # workers, engine on a GPU worker).
        if "actor" not in m:
            if peer is not None:
                peer.in_flight += 1
            try:
                # Explicit task + shield so close-cancel still observes the
                # head result and can kill a live actor the head already owns.
                fwd_task = asyncio.create_task(self._forward_head(m, payload))
                try:
                    r, pl = await asyncio.shield(fwd_task)
                except asyncio.CancelledError:
                    fwd_res = await self._await_fwd_after_cancel(fwd_task, peer, "actor")
                    if fwd_res is not None and peer is not None and fwd_res[0].get("actor"):
                        await asyncio.shield(self._worker_cleanup_actor(peer, fwd_res[0]["actor"]))
                    raise
                if peer is not None and r.get("actor"):
                    aid = r["actor"]
                    # Driver left during forward: head finished create but ownership
                    # was never recorded. Kill while still in_flight so release
                    # cannot free PGs under a live actor process.
                    if peer.closed:
                        await asyncio.shield(self._worker_cleanup_actor(peer, aid))
                        return {"err": "client disconnected during actor create"}, b""
                    peer.created_actors.append(aid)
                return r, pl
            finally:
                if peer is not None:
                    peer.in_flight = max(0, peer.in_flight - 1)
        return await self._host_actor(m, payload)

    async def _await_fwd_after_cancel(
        self,
        fwd_task: asyncio.Task,
        peer: Peer | None,
        kind: str,
    ) -> tuple[dict, bytes] | None:
        """Finish a shielded forward after the outer task was cancelled.

        Bound wait so Peer.close is not blocked forever. If the head is still
        working past the budget, do NOT cancel the forward (create may already
        own GPUs on the head). Pin in_flight and finish cleanup in the background.
        """
        try:
            return await asyncio.wait_for(asyncio.shield(fwd_task), timeout=30)
        except Exception:
            if peer is not None and not fwd_task.done():
                peer.in_flight += 1
                self._track(asyncio.create_task(self._bg_finish_fwd_cleanup(peer, fwd_task, kind)))
            elif peer is not None and fwd_task.done() and not fwd_task.cancelled():
                try:
                    return fwd_task.result()
                except Exception:
                    return None
            return None

    async def _bg_finish_fwd_cleanup(self, peer: Peer, fwd_task: asyncio.Task, kind: str) -> None:
        """Wait for a late head create and kill/remove if it succeeded."""
        try:
            r, _pl = await fwd_task
            if kind == "actor" and r.get("actor"):
                await self._worker_cleanup_actor(peer, r["actor"])
            elif kind == "pg" and r.get("pg"):
                await self._worker_cleanup_pg(peer, r["pg"])
        except Exception:
            pass
        finally:
            peer.in_flight = max(0, peer.in_flight - 1)

    async def _worker_cleanup_actor(self, peer: Peer, aid: str) -> None:
        """Kill a head-created actor after driver disconnect / cancel mid-forward.

        Pin in_flight before any await so CancelledError still leaves a
        background retry (_pinned_retry_kill drops the pin) for release_client.
        """
        peer.in_flight += 1
        try:
            await asyncio.wait_for(
                self._forward_head({"t": "kill", "actor": aid}),
                timeout=30,
            )
            if aid in self.actors:
                try:
                    await self.on_kill(None, {"actor": aid}, b"")
                except Exception:
                    pass
            peer.in_flight = max(0, peer.in_flight - 1)
        except asyncio.CancelledError:
            self._track(asyncio.create_task(self._pinned_retry_kill(peer, aid)))
            raise
        except Exception:
            if aid in self.actors:
                try:
                    await self.on_kill(None, {"actor": aid}, b"")
                except Exception:
                    pass
            self._track(asyncio.create_task(self._pinned_retry_kill(peer, aid)))

    async def _worker_cleanup_pg(self, peer: Peer, pg_id: str) -> None:
        """Remove a head-created PG after driver disconnect / cancel mid-forward."""
        peer.in_flight += 1
        try:
            await asyncio.wait_for(
                self._forward_head({"t": "remove_pg", "pg": pg_id}),
                timeout=30,
            )
            peer.in_flight = max(0, peer.in_flight - 1)
        except asyncio.CancelledError:
            self._track(asyncio.create_task(self._pinned_retry_remove_pg(peer, pg_id)))
            raise
        except Exception:
            self._track(asyncio.create_task(self._pinned_retry_remove_pg(peer, pg_id)))

    async def _force_kill_actor(self, actor_id: str, node: str) -> None:
        """Kill an actor by known owner node (works even if actor_loc was cleared).

        Only drops actor_loc after a successful remote kill so a failed attempt
        remains retriable (same rule as on_kill).
        """
        try:
            if node == self.node_id:
                # ensure local kill path can find it
                self.actor_loc[actor_id] = node
                await self.on_kill(None, {"actor": actor_id}, b"")
                return
            p = self._peer_for(node)
            if p is None:
                # owner unreachable: drop stale routing (cannot retry via RPC)
                self.actor_loc.pop(actor_id, None)
                return
            try:
                await asyncio.wait_for(p.call({"t": "kill", "actor": actor_id}), timeout=30)
            except Exception:
                return  # keep actor_loc for retry
            self.actor_loc.pop(actor_id, None)
        except Exception:
            pass

    def _actor_still_tracked(self, actor_id: str) -> bool:
        return actor_id in self.actor_loc or actor_id in self.actors or actor_id in self._hosting

    async def _rollback_failed_create(
        self,
        peer: Peer | None,
        actor_id: str,
        node: str,
        gpus: list[int],
    ) -> None:
        """Kill best-effort, then drop ownership only if the actor is gone.

        If the owner kill fails, keep actor_loc + created_actors so a later
        release_client/kill can still reach the live process.
        """
        await self._force_kill_actor(actor_id, node)
        if self._actor_still_tracked(actor_id):
            # Owner still alive but the driver peer is dead: do not rely on a
            # later release_client (it will not run). Retry kill in the background.
            if peer is None or peer.closed:
                self._schedule_orphan_reap(actor_id, node)
            return
        if peer is not None and actor_id in peer.created_actors:
            peer.created_actors.remove(actor_id)
        self._orphans.pop(actor_id, None)
        # free local greedy indices only if no *other* live actor/hosting holds them
        # (a concurrent create may have re-placed the same GPU after our kill freed it)
        if node == self.node_id:
            held = {g for ap in self.actors.values() for g in ap.gpus}
            held.update(g for _p, _w, gs in self._hosting.values() for g in gs)
            for g in gpus:
                if 0 <= g < len(self.gpu_used) and g not in held:
                    self.gpu_used[g] = False

    def _schedule_orphan_reap(self, actor_id: str, node: str) -> None:
        """Register an orphan and ensure a reaper task is running for it.

        Re-entry while a task is already live is a no-op (refresh node). If the
        previous task finished/cancelled while the actor is still tracked, start
        a new one so the slot is never sticky without a worker.
        """
        self._orphans[actor_id] = node
        t = self._orphan_tasks.get(actor_id)
        if t is not None and not t.done():
            return
        try:
            task = asyncio.get_running_loop().create_task(self._reap_orphan(actor_id, node))
        except RuntimeError:
            # No running loop (unit tests calling sync helpers): id is still
            # registered in _orphans for a later schedule under a live loop.
            return
        self._orphan_tasks[actor_id] = task
        self._track(task)

    def _free_orphan_pgs_if_idle(self) -> None:
        """Drop deferred PG reservations once no orphan actors remain."""
        if self._orphans:
            return
        for pg_id in list(self._orphan_pgs):
            self.pgs.pop(pg_id, None)
        self._orphan_pgs.clear()

    async def _reap_orphan(self, actor_id: str, node: str) -> None:
        """Retry force-kill for actors left after a dead driver peer disconnect.

        Backs off until the actor is untracked. On success (or when the actor
        disappears), clears the orphan slot and any deferred PGs. On cancel
        while still tracked, leaves the slot so a later _schedule_orphan_reap
        can restart the task (never sticky-without-worker: schedule checks
        task.done()).
        """
        delay = 0.25
        try:
            while self._actor_still_tracked(actor_id):
                if actor_id not in self._orphans:
                    return  # ownership released elsewhere
                await self._force_kill_actor(actor_id, node)
                if not self._actor_still_tracked(actor_id):
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
        finally:
            self._orphan_tasks.pop(actor_id, None)
            if not self._actor_still_tracked(actor_id):
                self._orphans.pop(actor_id, None)
                self._free_orphan_pgs_if_idle()

    def _place_actor(self, m: dict) -> tuple[str | None, list[int] | None, str | None]:
        pg_id = m.get("pg")
        if pg_id:
            pg = self.pgs.get(pg_id)
            if pg is None:
                return None, None, "unknown placement group %s" % pg_id
            bundle = m.get("bundle", 0)
            if bundle < 0 or bundle >= len(pg):
                return None, None, "bundle index %d out of range" % bundle
            b = pg[bundle]
            return b["node"], ([b["gpu"]] if b["gpu"] >= 0 else []), None
        if m.get("ngpu", 0) <= 0:
            return self.node_id, [], None
        # exclude GPUs already owned by pg bundles on this node, so a non-pg GPU
        # actor can't grab an index a placement-group bundle is using.
        pg_used = {
            b["gpu"]
            for pg in self.pgs.values()
            for b in pg
            if b["node"] == self.node_id and b["gpu"] >= 0
        }
        for i in range(self.num_gpus):
            if not self.gpu_used[i] and i not in pg_used:
                self.gpu_used[i] = True
                return self.node_id, [i], None
        return None, None, "no free GPU for actor"

    async def _host_actor(self, m: dict, payload: bytes) -> tuple[dict, bytes]:
        actor_id = m["actor"]
        gpus = m.get("gpus", []) or []
        # head kill arrived before we entered hosting (remote create vs kill race)
        if actor_id in self._kill_pending:
            self._kill_pending.discard(actor_id)
            return {"err": "actor %s killed during create" % actor_id}, b""
        fut = asyncio.get_running_loop().create_future()
        self.pending_workers[actor_id] = fut
        proc: subprocess.Popen | None = None
        try:
            proc = self._spawn_worker(actor_id, gpus)
            # peer filled in by on_worker_hello so mid-create kill can close the socket
            self._hosting[actor_id] = (proc, None, list(gpus))
            if actor_id in self._kill_pending:
                self._kill_pending.discard(actor_id)
                self.pending_workers.pop(actor_id, None)
                if _terminate(proc):
                    self._free_local_gpus(gpus)
                    return {"err": "actor %s killed during create" % actor_id}, b""
                # still alive: keep hosting (finally will not drop)
                return {"err": "actor %s process still alive after kill" % actor_id}, b""
            try:
                peer = await asyncio.wait_for(fut, timeout=120)
            except asyncio.TimeoutError:
                self.pending_workers.pop(actor_id, None)
                if _terminate(proc):
                    self._free_local_gpus(gpus)
                    return {"err": "worker for %s did not attach" % actor_id}, b""
                return {"err": "worker for %s did not attach (process still alive)" % actor_id}, b""
            except Exception as e:
                # e.g. kill during create set_exception on the attach future
                self.pending_workers.pop(actor_id, None)
                if _terminate(proc):
                    self._free_local_gpus(gpus)
                    return {"err": "actor %s create aborted: %s" % (actor_id, e)}, b""
                return {
                    "err": "actor %s create aborted (process still alive): %s" % (actor_id, e)
                }, b""
            try:
                await peer.call({"t": "init"}, payload)  # instantiate the pickled class
            except Exception as e:  # ctor raised / peer closed mid-init
                try:
                    await peer.close()
                except Exception:
                    pass
                if _terminate(proc):
                    self._free_local_gpus(gpus)
                    return {"err": "actor %s init failed: %s" % (actor_id, e)}, b""
                return {
                    "err": "actor %s init failed (process still alive): %s" % (actor_id, e)
                }, b""
            # kill raced create / worker died after init: do not publish a doomed actor
            if actor_id not in self._hosting or actor_id in self._kill_pending or peer.closed:
                self._kill_pending.discard(actor_id)
                try:
                    peer.on_close = None
                    if not peer.closed:
                        await peer.close()
                except Exception:
                    pass
                if _terminate(proc):
                    self._free_local_gpus(gpus)
                    return {"err": "actor %s killed during create" % actor_id}, b""
                return {"err": "actor %s process still alive after kill" % actor_id}, b""
            self.actors[actor_id] = ActorProc(actor_id, peer, gpus, proc)
            return {"t": "create_actor_ok", "actor": actor_id, "gpus": gpus}, b""
        except asyncio.CancelledError:
            # Peer.close drain cancelled us: never leave a zombie worker untracked.
            self.pending_workers.pop(actor_id, None)
            host = self._hosting.get(actor_id)
            gone = False
            if host is not None:
                p, wpeer, _gs = host
                if wpeer is not None:
                    wpeer.on_close = None
                gone = _terminate(p)
            elif proc is not None:
                gone = _terminate(proc)
                if not gone and actor_id not in self._hosting:
                    self._hosting[actor_id] = (proc, None, list(gpus))
            if gone:
                self._hosting.pop(actor_id, None)
                self._free_local_gpus(gpus)
            # if not gone: leave _hosting so process stays tracked
            raise
        except Exception as e:
            # spawn OSError etc. before the attach wait: drop pending so a late
            # worker_hello cannot pair with a dead create.
            self.pending_workers.pop(actor_id, None)
            if proc is not None:
                if _terminate(proc):
                    self._free_local_gpus(gpus)
                elif actor_id not in self._hosting:
                    self._hosting[actor_id] = (proc, None, list(gpus))
            return {"err": "actor %s spawn failed: %s" % (actor_id, e)}, b""
        finally:
            self._kill_pending.discard(actor_id)
            self.pending_workers.pop(actor_id, None)
            # Drop hosting only if the proc is gone or the actor was published.
            # A live unreaped worker (or concurrent on_kill restore) must stay
            # tracked so GPU free/rollback cannot run under a still-running process.
            host = self._hosting.get(actor_id)
            if host is not None:
                p = host[0]
                if actor_id in self.actors or p is None or p.poll() is not None:
                    self._hosting.pop(actor_id, None)

    def _free_local_gpus(self, gpus: list[int]) -> None:
        """Clear greedy GPU bits only if no other live actor/hosting holds them."""
        held = {g for ap in self.actors.values() for g in ap.gpus}
        held.update(g for _p, _w, gs in self._hosting.values() for g in gs)
        for g in gpus:
            if 0 <= g < len(self.gpu_used) and g not in held:
                self.gpu_used[g] = False

    def _spawn_worker(self, actor_id: str, gpus: list[int]) -> subprocess.Popen:
        cmdline = os.environ.get("BEAM_WORKER_CMD", "python3 -m ray._worker")
        ids = ",".join(str(g) for g in gpus)
        assert self.sock_path is not None  # serve_unix runs before any actor spawn
        env = dict(os.environ)
        env.update(
            {
                "BEAM_SOCK": self.sock_path,
                "BEAM_ACTOR_ID": actor_id,
                "BEAM_NODE_ID": self.node_id,
                "BEAM_GPU_IDS": ids,
                "CUDA_VISIBLE_DEVICES": ids,  # NVIDIA
                "HIP_VISIBLE_DEVICES": ids,  # AMD ROCm
                "ROCR_VISIBLE_DEVICES": ids,  # AMD ROCr runtime
            }
        )
        return subprocess.Popen(shlex.split(cmdline), env=env)

    async def on_worker_hello(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        actor_id = m["actor"]
        fut = self.pending_workers.pop(actor_id, None)
        if fut and not fut.done():
            fut.set_result(peer)
        # attach peer for mid-create kill (close socket, not only SIGTERM)
        host = self._hosting.get(actor_id)
        if host is not None:
            self._hosting[actor_id] = (host[0], peer, host[2])

        # when this actor's subprocess dies, reclaim locally and tell the head
        # so actor_loc does not keep routing to a dead worker forever.
        async def _on_actor_close() -> None:
            self._drop_actor(actor_id)
            if not self.is_head:
                try:
                    await self._forward_head({"t": "actor_gone", "actor": actor_id})
                except Exception:
                    pass

        peer.on_close = _on_actor_close
        return {"t": "worker_hello_ok"}, b""

    async def on_actor_gone(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        """Worker reports an actor process died or was reaped locally. Head
        drops routing only; PG reservations stay until remove_pg/release."""
        if not self.is_head:
            return await self._forward_head(m)
        self.actor_loc.pop(m["actor"], None)
        return {"t": "actor_gone_ok"}, b""

    async def on_call(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        if self.is_head:
            node = self.actor_loc.get(m["actor"])
            if node and node != self.node_id:
                p = self._peer_for(node)
                # p is peer => the call came from the very node we'd forward to
                # (actor died there mid-flight): bounce-back loop, fail cleanly.
                if p is None or p is peer:
                    return {"err": "unknown actor %s" % m["actor"]}, b""
                r, _ = await p.call(m, payload)
                return {"t": "call_ok", "obj": r.get("obj")}, b""
        elif m["actor"] not in self.actors:
            # worker node, actor lives elsewhere: the head knows where (driver
            # running on a worker node).
            return await self._forward_head(m, payload)
        ap = self.actors.get(m["actor"])
        if ap is None:
            return {"err": "unknown actor %s" % m["actor"]}, b""
        obj_id = self._next_obj()
        slot = ObjSlot()
        self.objects[obj_id] = slot
        self._track(asyncio.create_task(self._dispatch(ap, m["method"], payload, slot)))
        return {"t": "call_ok", "obj": obj_id}, b""

    async def _dispatch(self, ap: ActorProc, method: str, payload: bytes, slot: ObjSlot) -> None:
        try:
            async with ap.lock:  # serialize per actor
                _, rpl = await ap.peer.call({"t": "method", "method": method}, payload)
            slot.data = rpl
        except Exception as e:
            slot.err = str(e)
        except asyncio.CancelledError:
            # shutdown cancels bg tasks: never leave a ready slot with empty
            # data/err (that looks like a successful empty result on get).
            slot.err = "actor dispatch cancelled"
            raise
        finally:
            slot.ev.set()

    async def on_kill(self, peer: Peer | None, m: dict, payload: bytes) -> tuple[dict, bytes]:
        actor_id = m["actor"]
        if self.is_head:
            node = self.actor_loc.get(actor_id)
            if node and node != self.node_id:
                p = self._peer_for(node)
                # p is peer => kill bounced back from the owner (actor not local
                # there yet / already gone): drop routing, do not loop.
                if p is None or p is peer:
                    self.actor_loc.pop(actor_id, None)
                    return {"t": "kill_ok"}, b""
                # build a clean message: m may be a synthetic dict (e.g. from
                # release_client) that lacks a well-formed type field.
                try:
                    await asyncio.wait_for(p.call({"t": "kill", "actor": actor_id}), timeout=30)
                except Exception as e:
                    # keep actor_loc so a later kill/retry can still route.
                    # TimeoutError() stringifies to "" which is falsy; always
                    # return a non-empty err so release_client schedules reaps.
                    msg = str(e) or "kill timed out or failed"
                    return {"err": msg}, b""
                self.actor_loc.pop(actor_id, None)
                return {"t": "kill_ok"}, b""
            self.actor_loc.pop(actor_id, None)

        if actor_id not in self.actors:
            # Mid-create: hosting but not yet in self.actors. Reap here; never
            # forward to the head (that would bounce forever with actor_loc).
            hosting = self._hosting.pop(actor_id, None)
            if hosting is not None:
                proc, wpeer, gpus = hosting
                fut = self.pending_workers.pop(actor_id, None)
                if fut is not None and not fut.done():
                    fut.set_exception(RuntimeError("actor killed during create"))
                try:
                    if wpeer is not None:
                        wpeer.on_close = None
                        try:
                            await wpeer.close()  # abort in-flight init RPC
                        except Exception:
                            pass
                finally:
                    # Always reap even if close is cancelled (CancelledError).
                    # Only free GPU indices when the process is confirmed gone.
                    if _terminate(proc):
                        held = {g for ap2 in self.actors.values() for g in ap2.gpus}
                        held.update(g for _p, _w, gs in self._hosting.values() for g in gs)
                        for g in gpus:
                            if 0 <= g < len(self.gpu_used) and g not in held:
                                self.gpu_used[g] = False
                    else:
                        # Still alive: restore hosting + routing so still tracked.
                        self._hosting[actor_id] = (proc, wpeer, list(gpus))
                        if self.is_head:
                            self.actor_loc[actor_id] = self.node_id
                        return {"err": "actor %s process still alive after kill" % actor_id}, b""
                return {"t": "kill_ok"}, b""
            if not self.is_head:
                # Kill from head for an unknown local id: tombstone so a create
                # still in flight aborts instead of publishing an orphan.
                if peer is self.head_peer:
                    self._kill_pending.add(actor_id)
                    return {"t": "kill_ok"}, b""
                # worker node, actor lives elsewhere: route via the head
                return await self._forward_head({"t": "kill", "actor": actor_id})
            return {"t": "kill_ok"}, b""

        ap = self.actors.pop(actor_id, None)
        if ap:
            # explicit kill: do not also fire actor_gone via peer.on_close
            # (head already owns the routing update for remote kills).
            ap.peer.on_close = None
            try:
                await ap.peer.close()  # closing the socket makes the worker exit
            except Exception:
                pass
            finally:
                # Always reap even if peer.close is cancelled mid-drain.
                # Only free GPU indices when the process is confirmed gone.
                if _terminate(ap.proc):
                    held = {g for a in self.actors.values() for g in a.gpus}
                    held.update(g for _p, _w, gs in self._hosting.values() for g in gs)
                    for g in ap.gpus:
                        if 0 <= g < len(self.gpu_used) and g not in held:
                            self.gpu_used[g] = False
                else:
                    # Process still alive: restore tracking so we do not free
                    # GPUs/PGs under a live worker (release will orphan-reap).
                    self.actors[actor_id] = ap
                    if self.is_head:
                        self.actor_loc[actor_id] = self.node_id
                    return {"err": "actor %s process still alive after kill" % actor_id}, b""
        # Driver-on-worker ray.kill: head still has actor_loc until told.
        if not self.is_head:
            try:
                await self._forward_head({"t": "actor_gone", "actor": actor_id})
            except Exception:
                pass
        return {"t": "kill_ok"}, b""

    # ---- objects ----
    async def on_put(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        obj_id = self._next_obj()
        slot = ObjSlot()
        slot.data = payload
        slot.ev.set()
        self.objects[obj_id] = slot
        return {"t": "put_ok", "obj": obj_id}, b""

    async def on_get(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        obj_id = m["obj"]
        owner = owner_of(obj_id)
        if owner != self.node_id:
            if self.is_head:
                p = self._peer_for(owner)
                if p is None:
                    return {"err": "cannot locate object %s" % obj_id}, b""
                r, pl = await p.call(m)
                return r, pl
            return await self._forward_head(m)
        # keep the slot (not pop): a ref can be get/stat'd more than once, like
        # real ray. vLLM's hot path uses its own MessageQueue, not ray objects,
        # so the store does not accumulate during inference (see DESIGN scope).
        slot = self.objects.get(obj_id)
        if slot is None:
            return {"err": "unknown object %s" % obj_id}, b""
        timeout = m.get("timeout")
        try:
            await asyncio.wait_for(slot.ev.wait(), timeout)
        except asyncio.TimeoutError:
            return {"err": "GetTimeoutError: object %s not ready in %ss" % (obj_id, timeout)}, b""
        if slot.err:
            return {"err": slot.err}, b""
        return {"t": "get_ok", "obj": obj_id}, slot.data

    async def on_stat(self, peer: Peer, m: dict, payload: bytes) -> tuple[dict, bytes]:
        """Report readiness without blocking (backs ray.wait). 'ready' is a plain
        bool, never an error, so the client does not raise on not-ready."""
        obj_id = m["obj"]
        owner = owner_of(obj_id)
        if owner != self.node_id:
            if self.is_head:
                p = self._peer_for(owner)
                if p is None:
                    return {"t": "stat_ok", "ready": False}, b""
                r, _ = await p.call(m)
                return r, b""
            return await self._forward_head(m)
        slot = self.objects.get(obj_id)
        return {"t": "stat_ok", "ready": bool(slot and slot.ev.is_set())}, b""

    # ---- cleanup ----
    async def release_client(self, peer: Peer) -> None:
        """When a driver disconnects, free the placement groups and actors it
        created so their GPUs return to the pool (no leak across runs).

        Claim each id off the peer list before any await so re-hello cannot
        transfer an id mid-kill. A single missing id (concurrent rollback) is
        skipped with continue, not return (which would leak later PGs/actors).

        Kill soft-errors (on_kill returns err without raising) and hard
        exceptions both schedule orphan reaping when the actor is still
        tracked. Re-append to a dying peer is a dead letter; the reaper is
        the recovery path. PGs are deferred while orphans remain so GPU
        bundles are not double-booked with live actor processes.
        """
        if getattr(peer, "superseded", False):
            return
        if not self.is_head:
            # a local driver on a worker node: the head owns placement and
            # routing, so forward the releases there (drops the head-side leak
            # for the CPU-head / GPU-worker, driver-on-worker topology).
            # Claim PGs before kill awaits so they cannot be transferred away
            # while actors are still being reaped.
            pending_pgs: list[str] = []
            for pg_id in list(peer.created_pgs):
                if getattr(peer, "superseded", False):
                    break
                if pg_id not in peer.created_pgs:
                    continue
                peer.created_pgs.remove(pg_id)
                pending_pgs.append(pg_id)
            failed_actors: list[str] = []
            for actor_id in list(peer.created_actors):
                if getattr(peer, "superseded", False):
                    break  # remaining ids transferred; still free claimed ones
                if actor_id not in peer.created_actors:
                    continue
                peer.created_actors.remove(actor_id)  # claim before await
                forward_ok = False
                try:
                    await asyncio.wait_for(
                        self._forward_head({"t": "kill", "actor": actor_id}),
                        timeout=30,
                    )
                    forward_ok = True
                except Exception:
                    pass
                # head unreachable: still reap any actor hosted on this node
                if actor_id in self.actors:
                    try:
                        await self.on_kill(None, {"actor": actor_id}, b"")
                    except Exception:
                        pass
                if not forward_ok:
                    # Always re-forward until head acks (even after local reap):
                    # local kill's actor_gone is best-effort and may have failed.
                    failed_actors.append(actor_id)
            # Defer PG free while creates are still in flight (actor may still
            # land on a bundle GPU) or kills need retry.
            if failed_actors or peer.in_flight > 0:
                self._track(
                    asyncio.create_task(
                        self._retry_worker_release(failed_actors, pending_pgs, peer)
                    )
                )
            else:
                for pg_id in pending_pgs:
                    try:
                        await asyncio.wait_for(
                            self._forward_head({"t": "remove_pg", "pg": pg_id}),
                            timeout=30,
                        )
                    except Exception:
                        self._track(asyncio.create_task(self._retry_forward_remove_pg(pg_id)))
            return
        # Claim PGs before any kill await so re-hello cannot transfer them while
        # actors are still being reaped (would double-book bundle GPUs).
        claimed_pgs: list[str] = []
        for pg_id in list(peer.created_pgs):
            if getattr(peer, "superseded", False):
                break
            if pg_id not in peer.created_pgs:
                continue
            peer.created_pgs.remove(pg_id)
            claimed_pgs.append(pg_id)
        orphaned: list[str] = []
        for actor_id in list(peer.created_actors):
            if getattr(peer, "superseded", False):
                break
            if actor_id not in peer.created_actors:
                continue
            peer.created_actors.remove(actor_id)  # claim before await
            node = self.actor_loc.get(actor_id) or self.node_id
            kill_failed = False
            try:
                r, _ = await self.on_kill(None, {"actor": actor_id}, b"")
                if r.get("err"):
                    kill_failed = True
            except Exception:
                kill_failed = True
            if kill_failed and self._actor_still_tracked(actor_id):
                # Peer is disconnecting: do not re-append (dead letter). Reap.
                self._schedule_orphan_reap(actor_id, node)
                orphaned.append(actor_id)
        # Real re-hello clears created_actors before this resumes.
        if getattr(peer, "superseded", False) and claimed_pgs:
            target = getattr(peer, "superseded_by", None)
            if target is not None and not target.closed:
                # Live replacement peer will release these on disconnect.
                target.created_pgs.extend(claimed_pgs)
                claimed_pgs = []
            elif target is not None and target.closed:
                # Target cannot run release_client again. Reap any leftovers it
                # still lists, then free/defer claimed PGs via the normal path
                # (do not attach to a dead peer or park without orphan actors).
                for aid in list(target.created_actors):
                    if self._actor_still_tracked(aid):
                        node = self.actor_loc.get(aid) or self.node_id
                        self._schedule_orphan_reap(aid, node)
                        orphaned.append(aid)
                # claimed_pgs fall through to defer/free below
            elif peer.created_actors:
                # Edge path without superseded_by: keep with remaining.
                peer.created_pgs.extend(claimed_pgs)
                claimed_pgs = []
        # Defer PG free while ANY orphan remains (global, not just this call).
        defer_pgs = bool(self._orphans) or any(self._actor_still_tracked(a) for a in orphaned)
        for pg_id in claimed_pgs:
            if defer_pgs:
                # Keep PG reservation until orphan actors are gone.
                self._orphan_pgs.add(pg_id)
            else:
                self.pgs.pop(pg_id, None)
                self._orphan_pgs.discard(pg_id)
        self._free_orphan_pgs_if_idle()

    async def _retry_worker_release(
        self,
        actor_ids: list[str],
        pg_ids: list[str],
        peer: Peer | None = None,
    ) -> None:
        """Worker: wait for in-flight creates/closed kills, then kill+remove_pg."""
        delay = 0.25
        # in_flight covers open forwards AND pinned closed-path kill/remove retries
        while peer is not None and peer.in_flight > 0:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
        # Creates that finished after claim may have appended new actors; claim them.
        if peer is not None:
            for actor_id in list(peer.created_actors):
                if actor_id not in actor_ids:
                    peer.created_actors.remove(actor_id)
                    actor_ids.append(actor_id)
            for pg_id in list(peer.created_pgs):
                if pg_id not in pg_ids:
                    peer.created_pgs.remove(pg_id)
                    pg_ids.append(pg_id)
        for actor_id in actor_ids:
            await self._retry_forward_kill(actor_id)
        for pg_id in pg_ids:
            await self._retry_forward_remove_pg(pg_id)

    async def _pinned_retry_kill(self, peer: Peer, actor_id: str) -> None:
        """Retry head kill while holding peer.in_flight (released in finally)."""
        try:
            await self._retry_forward_kill(actor_id)
        finally:
            peer.in_flight = max(0, peer.in_flight - 1)

    async def _pinned_retry_remove_pg(self, peer: Peer, pg_id: str) -> None:
        """Retry remove_pg while holding peer.in_flight (released in finally)."""
        try:
            await self._retry_forward_remove_pg(pg_id)
        finally:
            peer.in_flight = max(0, peer.in_flight - 1)

    async def _retry_forward_kill(self, actor_id: str) -> None:
        """Worker-side: re-forward kill to head until acknowledged.

        Always requires a successful head kill even after a local reap, so a
        failed actor_gone cannot leave stale actor_loc on the head forever.
        Stops if the head link is gone (no infinite spin after head loss).
        """
        delay = 0.25
        while True:
            if self.head_peer is None or self.head_peer.closed:
                return
            if actor_id in self.actors:
                try:
                    await self.on_kill(None, {"actor": actor_id}, b"")
                except Exception:
                    pass
            try:
                await asyncio.wait_for(
                    self._forward_head({"t": "kill", "actor": actor_id}),
                    timeout=30,
                )
                return
            except Exception:
                if self.head_peer is None or self.head_peer.closed:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    async def _retry_forward_remove_pg(self, pg_id: str) -> None:
        """Worker-side: re-forward remove_pg until acknowledged."""
        delay = 0.25
        while True:
            if self.head_peer is None or self.head_peer.closed:
                return
            try:
                await asyncio.wait_for(
                    self._forward_head({"t": "remove_pg", "pg": pg_id}),
                    timeout=30,
                )
                return
            except Exception:
                if self.head_peer is None or self.head_peer.closed:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    def shutdown(self) -> None:
        """Reap every actor worker subprocess this daemon spawned. Called on a
        clean daemon stop so workers don't orphan to init."""
        for ap in list(self.actors.values()):
            ap.peer.on_close = None
            ap.peer.closed = True
            for fut in list(ap.peer.pending.values()):
                if not fut.done():
                    fut.set_exception(ConnectionError("connection closed"))
            ap.peer.pending.clear()
            try:
                ap.peer.writer.close()
            except Exception:
                pass
            if _terminate(ap.proc):
                for g in ap.gpus:
                    if 0 <= g < len(self.gpu_used):
                        self.gpu_used[g] = False
        self.actors.clear()
        # mid-create workers live only in _hosting until init finishes
        for proc, wpeer, gpus in list(self._hosting.values()):
            if wpeer is not None:
                wpeer.on_close = None
                wpeer.closed = True
                for fut in list(wpeer.pending.values()):
                    if not fut.done():
                        fut.set_exception(ConnectionError("connection closed"))
                wpeer.pending.clear()
                try:
                    wpeer.writer.close()
                except Exception:
                    pass
            if _terminate(proc):
                for g in gpus:
                    if 0 <= g < len(self.gpu_used):
                        self.gpu_used[g] = False
        self._hosting.clear()
        self._kill_pending.clear()
        self._orphans.clear()
        self._orphan_pgs.clear()
        for t in list(self._orphan_tasks.values()):
            t.cancel()
        self._orphan_tasks.clear()
        for t in list(self._bg_tasks):
            t.cancel()
        for fut in list(self.pending_workers.values()):
            if not fut.done():
                fut.set_exception(RuntimeError("daemon shutdown"))
        self.pending_workers.clear()
