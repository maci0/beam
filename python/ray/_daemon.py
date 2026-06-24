"""beam daemon: the control-plane hub, in Python (asyncio).

One daemon per node. Exactly one node is the head; it is the routing hub and the
authority on membership and placement. Non-head daemons keep a single connection
to the head and execute create/call/get/kill requests pushed down it.

This is pure control plane: it launches actor subprocesses, assigns GPUs, routes
method-call RPCs, and gathers small results. Tensor traffic goes over NCCL,
never through here. See DESIGN.md.
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


def _terminate(proc: subprocess.Popen | None) -> None:
    """Best-effort kill of an actor worker subprocess that hasn't already exited."""
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass


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
        self._tasks: set[asyncio.Task] = (
            set()
        )  # keep handler task refs so they aren't GC'd mid-flight
        # per-client ownership, for cleanup on disconnect
        self.created_pgs: list[str] = []
        self.created_actors: list[str] = []

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
            self.writer.write(encode_frame(header, payload))
            await self.writer.drain()

    async def call(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        self.next_id += 1
        rid = self.next_id
        # copy: never mutate the caller's dict. A routing handler passes the
        # message it received straight to call(); mutating reqid here would
        # corrupt the reqid _handle echoes back on its own response.
        header = dict(header)
        header["reqid"] = rid
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        await self.send(header, payload)
        rheader, rpayload = await fut
        if rheader.get("err"):
            raise RuntimeError(rheader["err"])
        return rheader, rpayload

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("connection closed"))
        try:
            self.writer.close()
        except OSError:
            pass
        if self.on_close:
            r = self.on_close()
            if asyncio.iscoroutine(r):
                await r


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

        self.gpu_used = [False] * num_gpus
        self.actors: dict[str, ActorProc] = {}
        self.objects: dict[str, ObjSlot] = {}
        self.pending_workers: dict[str, asyncio.Future] = {}
        self.obj_seq = 0
        self.id_seq = 0

        self.nodes: dict[str, dict[str, Any]] = {}
        self.pgs: dict[str, list[dict[str, Any]]] = {}
        self.actor_loc: dict[str, str] = {}
        if is_head:
            self.nodes = {node_id: {"info": dict(self.self_info), "peer": None}}
            self.pgs = {}
            self.actor_loc = {}

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
        asyncio.create_task(self.head_peer.serve())
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
        peer.on_close = lambda: self._drop_node(node, peer)
        return {"t": "hello_ok"}, b""

    def _drop_node(self, node: str, peer: Peer | None = None) -> None:
        rec = self.nodes.get(node)
        # if the node already reconnected with a fresh peer, this is a stale
        # close from the old connection: ignore it, don't drop the live node.
        if peer is not None and rec is not None and rec["peer"] is not peer:
            return
        if rec:
            rec["info"]["alive"] = False
            rec["peer"] = None  # don't route to a closed peer
        # actors on a dead node are unreachable; drop their routing so calls fail
        # cleanly ("unknown actor") instead of hanging on a dead connection.
        for aid in [a for a, n in self.actor_loc.items() if n == node]:
            del self.actor_loc[aid]

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
            return await self._forward_head(m, payload)
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
                    return resp, rpl
                p = self._peer_for(node)
                if p is None:
                    raise RuntimeError("node %s is not available" % node)
                await p.call(req, payload)
                return {"t": "create_actor_ok", "actor": actor_id, "gpus": gpus, "node": node}, b""
            except Exception as e:
                # roll back placement so a failed create leaks neither the
                # actor_loc routing entry nor a greedily-reserved GPU.
                self.actor_loc.pop(actor_id, None)
                if peer is not None and actor_id in peer.created_actors:
                    peer.created_actors.remove(actor_id)
                for g in gpus:
                    if 0 <= g < len(self.gpu_used):
                        self.gpu_used[g] = False
                return {"err": str(e)}, b""
        # worker node: a head push carries a pre-assigned "actor" id; a request
        # from a local driver does not, so route it to the head for placement.
        # This is what lets the vLLM driver run on a worker node (CPU head, GPU
        # workers, engine on a GPU worker).
        if "actor" not in m:
            return await self._forward_head(m, payload)
        return await self._host_actor(m, payload)

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
        fut = asyncio.get_running_loop().create_future()
        self.pending_workers[actor_id] = fut
        proc = self._spawn_worker(actor_id, gpus)
        try:
            peer = await asyncio.wait_for(fut, timeout=120)
        except asyncio.TimeoutError:
            self.pending_workers.pop(actor_id, None)
            _terminate(proc)  # don't leave an orphaned subprocess holding a GPU
            return {"err": "worker for %s did not attach" % actor_id}, b""
        try:
            await peer.call({"t": "init"}, payload)  # instantiate the pickled class
        except Exception as e:  # constructor raised (common with vLLM) -> reap it
            await peer.close()
            _terminate(proc)
            return {"err": "actor %s init failed: %s" % (actor_id, e)}, b""
        self.actors[actor_id] = ActorProc(actor_id, peer, gpus, proc)
        return {"t": "create_actor_ok", "actor": actor_id, "gpus": gpus}, b""

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
        fut = self.pending_workers.pop(m["actor"], None)
        if fut and not fut.done():
            fut.set_result(peer)
        return {"t": "worker_hello_ok"}, b""

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
        asyncio.create_task(self._dispatch(ap, m["method"], payload, slot))
        return {"t": "call_ok", "obj": obj_id}, b""

    async def _dispatch(self, ap: ActorProc, method: str, payload: bytes, slot: ObjSlot) -> None:
        try:
            async with ap.lock:  # serialize per actor
                _, rpl = await ap.peer.call({"t": "method", "method": method}, payload)
            slot.data = rpl
        except Exception as e:
            slot.err = str(e)
        finally:
            slot.ev.set()

    async def on_kill(self, peer: Peer | None, m: dict, payload: bytes) -> tuple[dict, bytes]:
        actor_id = m["actor"]
        if self.is_head:
            node = self.actor_loc.pop(actor_id, None)
            if node and node != self.node_id:
                p = self._peer_for(node)
                if p:
                    # build a clean message: m may be a synthetic dict (e.g. from
                    # release_client) that lacks a well-formed type field.
                    await p.call({"t": "kill", "actor": actor_id})
                return {"t": "kill_ok"}, b""
        elif actor_id not in self.actors:
            # worker node, actor lives elsewhere: route via the head
            return await self._forward_head({"t": "kill", "actor": actor_id})
        ap = self.actors.pop(actor_id, None)
        if ap:
            await ap.peer.close()  # closing the socket makes the worker exit
            _terminate(ap.proc)  # ...and reap it if it doesn't
            for g in ap.gpus:  # free any greedily-reserved GPU (no-op for pg gpus)
                if 0 <= g < len(self.gpu_used):
                    self.gpu_used[g] = False
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
        created so their GPUs return to the pool (no leak across runs)."""
        if not self.is_head:
            return
        for actor_id in list(peer.created_actors):
            try:
                await self.on_kill(None, {"actor": actor_id}, b"")
            except Exception:
                pass
        for pg_id in list(peer.created_pgs):
            self.pgs.pop(pg_id, None)

    def shutdown(self) -> None:
        """Reap every actor worker subprocess this daemon spawned. Called on a
        clean daemon stop so workers don't orphan to init."""
        for ap in list(self.actors.values()):
            _terminate(ap.proc)
        self.actors.clear()
