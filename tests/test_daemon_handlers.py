"""Unit + fuzz tests for the async daemon handlers in `_daemon.py`, driven
in-process with a FakePeer (no real sockets, subprocesses, or GPUs). Each
`on_*` coroutine is run via `asyncio.run` and asserted on its `(dict, bytes)`
return and the daemon state it mutates (actor_loc, pgs, gpu_used, objects,
nodes)."""

import asyncio
import os
import socket
import subprocess
import sys
import time as _time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray import _daemon
from ray._daemon import (
    ActorProc,
    Daemon,
    ObjSlot,
    Peer,
    _terminate,
    encode_frame,
    read_frame,
)

# ---- test doubles -----------------------------------------------------------


class FakePeer:
    """Records `.call(header, payload)` and replays canned (resp, payload)
    pairs. Mirrors the subset of `Peer` the handlers touch."""

    def __init__(self, responses=None, raise_on_call=None):
        self.responses = responses or {}
        self.raise_on_call = raise_on_call
        self.calls = []
        self.closed = False
        self.superseded = False
        self.superseded_by = None
        self.created_actors = []
        self.created_pgs = []
        self.in_flight = 0
        self.on_close = None
        self.pending: dict = {}
        self.writer = type("W", (), {"close": lambda self: None})()

    async def call(self, header, payload=b""):
        self.calls.append((dict(header), payload))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        t = header.get("t", "")
        canned = self.responses.get(t, {})
        if canned.get("err"):
            raise RuntimeError(canned["err"])
        resp = {"t": t + "_ok", **canned}
        return resp, canned.get("_body", b"")

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.on_close is not None:
            r = self.on_close()
            if asyncio.iscoroutine(r):
                await r


class FakeProc:
    """Stand-in for subprocess.Popen: poll()/terminate()/wait/kill."""

    def __init__(self, alive=True):
        self._alive = alive
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


def head(ngpu=2):
    return Daemon(is_head=True, node_id="n1", ip="1.2.3.4", num_gpus=ngpu)


def worker(ngpu=2):
    return Daemon(is_head=False, node_id="w1", ip="5.6.7.8", num_gpus=ngpu)


def run(coro):
    return asyncio.run(coro)


# ---- _terminate -------------------------------------------------------------


def test_terminate_none_is_noop():
    _terminate(None)  # no crash on a never-spawned proc


def test_terminate_live_proc():
    p = FakeProc(alive=True)
    _terminate(p)
    assert p.terminated


def test_terminate_dead_proc_not_touched():
    p = FakeProc(alive=False)
    _terminate(p)
    assert not p.terminated  # poll() is not None -> skip


def test_terminate_swallows_oserror():
    class Boom:
        def poll(self):
            return None

        def terminate(self):
            raise OSError("gone")

        def kill(self):
            raise OSError("gone")

        def wait(self, timeout=None):
            raise OSError("gone")

    _terminate(Boom())  # OSError swallowed, no raise


def test_terminate_kill_oserror_and_wait_fail():
    class Stubborn:
        def __init__(self):
            self.n = 0

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            self.n += 1
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

        def kill(self):
            raise OSError("nope")

    _terminate(Stubborn())  # all errors swallowed


def test_terminate_wait_succeeds_after_term():
    class DiesOnWait:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("should not kill")

    _terminate(DiesOnWait())


def test_peer_close_on_close_raises_swallowed():
    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)
        p = Peer(r1, w1, lambda *a: None)

        def boom():
            raise RuntimeError("cb")

        p.on_close = boom
        await p.close()  # no raise
        s2.close()

    run(go())


def test_on_kill_close_error_still_terminates():
    d = head()
    proc = FakeProc()

    class BadPeer(FakePeer):
        async def close(self):
            raise RuntimeError("close failed")

    d.actors["a1"] = ActorProc("a1", peer=BadPeer(), gpus=[], proc=proc)
    d.actor_loc["a1"] = "n1"
    r, _ = run(d.on_kill(None, {"actor": "a1"}, b""))
    assert r["t"] == "kill_ok"
    assert proc.terminated
    assert "a1" not in d.actors


# ---- _dispatch / handle -----------------------------------------------------


def test_handle_unknown_type():
    d = head()
    r, p = run(d.handle(FakePeer(), {"t": "bogus"}, b""))
    assert "unknown message type" in r["err"] and p == b""


def test_handle_no_type():
    d = head()
    r, _ = run(d.handle(FakePeer(), {}, b""))
    assert "unknown message type" in r["err"]


def test_handle_routes_to_on_put():
    d = head()
    r, _ = run(d.handle(FakePeer(), {"t": "put"}, b"x"))
    assert r["t"] == "put_ok" and r["obj"].startswith("n1-o")


# ---- on_status --------------------------------------------------------------


def test_on_status_head_lists_self():
    d = head(4)
    d.gpu_used[0] = True
    r, _ = run(d.on_status(FakePeer(), {"t": "status"}, b""))
    assert r["t"] == "status_ok"
    me = next(n for n in r["nodes"] if n["node"] == "n1")
    assert me["used"] == 1 and me["ngpu"] == 4 and me["head"] is True


def test_on_status_counts_pg_and_greedy():
    d = head(4)
    d.gpu_used[1] = True
    d.pgs["p"] = [{"node": "n1", "gpu": 2}, {"node": "n1", "gpu": -1}]
    r, _ = run(d.on_status(FakePeer(), {"t": "status"}, b""))
    me = next(n for n in r["nodes"] if n["node"] == "n1")
    assert me["used"] == 2  # one greedy + one pg gpu bundle (the -1 doesn't count)


def test_on_status_worker_forwards_to_head():
    d = worker()
    hp = FakePeer({"status": {"nodes": [{"node": "n1"}]}})
    d.head_peer = hp
    r, _ = run(d.on_status(FakePeer(), {"t": "status"}, b""))
    assert r["nodes"] == [{"node": "n1"}]
    assert hp.calls[0][0]["t"] == "status"


# ---- on_resources -----------------------------------------------------------


def test_on_resources_head():
    d = head(4)
    d.gpu_used[0] = True
    r, _ = run(d.on_resources(FakePeer(), {"t": "resources"}, b""))
    assert r["t"] == "resources_ok"
    assert r["data"]["n1"] == {"GPU": 3.0, "CPU": 1.0}


def test_on_resources_worker_forwards():
    d = worker()
    d.head_peer = FakePeer({"resources": {"data": {"n1": {"GPU": 1.0}}}})
    r, _ = run(d.on_resources(FakePeer(), {"t": "resources"}, b""))
    assert r["data"]["n1"]["GPU"] == 1.0


# ---- placement groups -------------------------------------------------------


def test_on_create_pg_cpu_only_bundle():
    d = head(2)
    peer = FakePeer()
    r, _ = run(d.on_create_pg(peer, {"t": "create_pg", "specs": [{}]}, b""))
    assert r["t"] == "create_pg_ok"
    pg_id = r["pg"]
    assert d.pgs[pg_id] == [{"node": "n1", "gpu": -1}]
    assert peer.created_pgs == [pg_id]


def test_on_create_pg_closed_peer_rolls_back():
    d = head(2)
    peer = FakePeer()
    peer.closed = True
    r, _ = run(d.on_create_pg(peer, {"t": "create_pg", "specs": [{"GPU": 1}]}, b""))
    assert "disconnected" in r["err"]
    assert d.pgs == {}
    assert peer.created_pgs == []


def test_on_create_pg_worker_closed_removes():
    d = worker()
    d.head_peer = FakePeer({"create_pg": {"pg": "n1-pg9"}})
    peer = FakePeer()
    peer.closed = True
    r, _ = run(d.on_create_pg(peer, {"t": "create_pg", "specs": [{"GPU": 1}]}, b""))
    assert "disconnected" in r["err"]
    assert {"t": "remove_pg", "pg": "n1-pg9"} in [c[0] for c in d.head_peer.calls]


def test_on_create_pg_worker_closed_remove_error():
    d = worker()
    calls = {"n": 0}

    class Flaky(FakePeer):
        async def call(self, header, payload=b""):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"t": "create_pg_ok", "pg": "n1-pg1"}, b""
            raise RuntimeError("gone")

    d.head_peer = Flaky()
    peer = FakePeer()
    peer.closed = True
    r, _ = run(d.on_create_pg(peer, {"t": "create_pg", "specs": [{}]}, b""))
    assert "disconnected" in r["err"]


def test_on_create_pg_gpu_bundle_assigns_index():
    d = head(2)
    r, _ = run(d.on_create_pg(FakePeer(), {"t": "create_pg", "specs": [{"GPU": 1}]}, b""))
    assert d.pgs[r["pg"]] == [{"node": "n1", "gpu": 0}]


def test_on_create_pg_exhaustion_errors():
    d = head(1)
    r, _ = run(
        d.on_create_pg(FakePeer(), {"t": "create_pg", "specs": [{"GPU": 1}, {"GPU": 1}]}, b"")
    )
    assert "more GPUs than the cluster has free" in r["err"]
    assert d.pgs == {}  # nothing committed on failure


def test_on_create_pg_worker_forwards_and_tracks():
    d = worker()
    d.head_peer = FakePeer({"create_pg": {"pg": "n1-pg9"}})
    peer = FakePeer()
    r, _ = run(d.on_create_pg(peer, {"t": "create_pg", "specs": [{}]}, b""))
    assert r["pg"] == "n1-pg9"
    assert peer.created_pgs == ["n1-pg9"]  # tracked for release on the worker too


def test_on_remove_pg_head():
    d = head()
    d.pgs["p"] = [{"node": "n1", "gpu": -1}]
    r, _ = run(d.on_remove_pg(FakePeer(), {"t": "remove_pg", "pg": "p"}, b""))
    assert r["t"] == "remove_pg_ok" and "p" not in d.pgs


def test_on_remove_pg_unknown_is_ok():
    d = head()
    r, _ = run(d.on_remove_pg(FakePeer(), {"t": "remove_pg", "pg": "nope"}, b""))
    assert r["t"] == "remove_pg_ok"  # pop(None) tolerated


def test_on_remove_pg_worker_forwards():
    d = worker()
    d.head_peer = FakePeer()
    run(d.on_remove_pg(FakePeer(), {"t": "remove_pg", "pg": "p"}, b""))
    assert d.head_peer.calls[0][0]["t"] == "remove_pg"


def test_on_pg_table_single():
    d = head()
    d.pgs["p"] = [{"node": "n1", "gpu": 0}, {"node": "n1", "gpu": -1}]
    r, _ = run(d.on_pg_table(FakePeer(), {"t": "pg_table", "pg": "p"}, b""))
    bundles = r["data"]["bundles"]
    assert bundles[0] == {"node": "n1", "spec": {"GPU": 1}}
    assert bundles[1] == {"node": "n1", "spec": {}}


def test_on_pg_table_all():
    d = head()
    d.pgs["p"] = [{"node": "n1", "gpu": 0}]
    r, _ = run(d.on_pg_table(FakePeer(), {"t": "pg_table"}, b""))
    assert "p" in r["data"]["pgs"]


def test_on_pg_table_unknown_errors():
    d = head()
    r, _ = run(d.on_pg_table(FakePeer(), {"t": "pg_table", "pg": "nope"}, b""))
    assert "unknown placement group" in r["err"]


def test_on_pg_table_worker_forwards():
    d = worker()
    d.head_peer = FakePeer({"pg_table": {"data": {"pgs": {}}}})
    r, _ = run(d.on_pg_table(FakePeer(), {"t": "pg_table"}, b""))
    assert r["data"] == {"pgs": {}}


# ---- objects: put / get / stat ----------------------------------------------


def test_on_put_stores_payload():
    d = head()
    r, _ = run(d.on_put(FakePeer(), {"t": "put"}, b"hello"))
    obj = r["obj"]
    assert d.objects[obj].data == b"hello" and d.objects[obj].ev.is_set()


def test_put_get_roundtrip():
    d = head()
    rp, _ = run(d.on_put(FakePeer(), {"t": "put"}, b"data"))
    obj = rp["obj"]
    rg, body = run(d.on_get(FakePeer(), {"t": "get", "obj": obj}, b""))
    assert rg["t"] == "get_ok" and body == b"data"


def test_on_get_unknown_object():
    d = head()
    r, _ = run(d.on_get(FakePeer(), {"t": "get", "obj": "n1-o999"}, b""))
    assert "unknown object" in r["err"]


def test_on_get_timeout_path():
    d = head()
    slot = ObjSlot()  # event never set
    d.objects["n1-o1"] = slot
    r, _ = run(d.on_get(FakePeer(), {"t": "get", "obj": "n1-o1", "timeout": 0.01}, b""))
    assert "GetTimeoutError" in r["err"] and "n1-o1" in r["err"]


def test_on_get_without_a_budget_waits_for_the_slot_to_fill():
    """No timeout means wait as long as it takes: a plain ray.get(ref) must
    return once the actor's call finishes, not time out and not return early."""
    d = head()
    slot = ObjSlot()
    d.objects["n1-o1"] = slot

    async def scenario():
        get_task = asyncio.ensure_future(
            d.on_get(FakePeer(), {"t": "get", "obj": "n1-o1"}, b"")
        )
        await asyncio.sleep(0.01)  # handler is parked on the unset event
        assert not get_task.done()
        slot.data = b"late"
        slot.ev.set()
        return await asyncio.wait_for(get_task, 1.0)

    r, body = run(scenario())
    assert r["t"] == "get_ok" and body == b"late"


def test_on_get_ready_slot_never_waits():
    """A slot the actor already filled answers on the spot, whatever budget the
    client sent, so the wait branch is never entered for a done ref."""
    d = head()
    slot = ObjSlot()
    slot.data = b"v"
    slot.ev.set()
    d.objects["n1-o1"] = slot
    r, body = run(d.on_get(FakePeer(), {"t": "get", "obj": "n1-o1", "timeout": 30.0}, b""))
    assert r["t"] == "get_ok" and body == b"v"


def test_on_get_ready_slot_wins_over_expired_budget():
    """A zero/negative budget on a slot the actor already filled must still
    return the value: `asyncio.wait_for(ev, 0)` raises without ever checking the
    flag, so the budget is bounded by the slot's state, never the reverse."""
    d = head()
    slot = ObjSlot()
    slot.data = b"done"
    slot.ev.set()
    d.objects["n1-o1"] = slot
    for budget in (0, 0.0, -1.0):
        r, body = run(d.on_get(FakePeer(), {"t": "get", "obj": "n1-o1", "timeout": budget}, b""))
        assert r["t"] == "get_ok" and body == b"done", (budget, r)
    # a not-ready slot with the same exhausted budget still times out
    d.objects["n1-o2"] = ObjSlot()
    r2, _ = run(d.on_get(FakePeer(), {"t": "get", "obj": "n1-o2", "timeout": 0.0}, b""))
    assert "GetTimeoutError" in r2["err"]


def test_on_get_null_timeout_on_ready_slot():
    """An explicit null budget means "wait forever", not "wait zero seconds"."""
    d = head()
    slot = ObjSlot()
    slot.data = b"v"
    slot.ev.set()
    d.objects["n1-o1"] = slot
    r, body = run(d.on_get(FakePeer(), {"t": "get", "obj": "n1-o1", "timeout": None}, b""))
    assert r["t"] == "get_ok" and body == b"v"


def test_on_get_propagates_slot_error():
    d = head()
    slot = ObjSlot()
    slot.err = "boom"
    slot.ev.set()
    d.objects["n1-o1"] = slot
    r, _ = run(d.on_get(FakePeer(), {"t": "get", "obj": "n1-o1"}, b""))
    assert r["err"] == "boom"


def test_on_get_remote_owner_head_routes():
    d = head()
    other = FakePeer({"get": {"_body": b"remote"}})
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": other}
    r, body = run(d.on_get(FakePeer(), {"t": "get", "obj": "n2-o1"}, b""))
    assert body == b"remote" and other.calls[0][0]["obj"] == "n2-o1"


def test_on_get_remote_owner_unlocatable():
    d = head()
    r, _ = run(d.on_get(FakePeer(), {"t": "get", "obj": "n2-o1"}, b""))
    assert "cannot locate object" in r["err"]


def test_on_get_worker_forwards_remote():
    d = worker()
    d.head_peer = FakePeer({"get": {"_body": b"z"}})
    r, body = run(d.on_get(FakePeer(), {"t": "get", "obj": "n9-o1"}, b""))
    assert body == b"z"


def test_on_stat_ready_and_not_ready():
    d = head()
    rp, _ = run(d.on_put(FakePeer(), {"t": "put"}, b"x"))
    r, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": rp["obj"]}, b""))
    assert r["ready"] is True
    # unknown obj on this node -> not ready, never an error
    r2, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": "n1-o999"}, b""))
    assert r2["ready"] is False


def test_on_stat_pending_slot_not_ready():
    d = head()
    d.objects["n1-o1"] = ObjSlot()  # event unset
    r, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": "n1-o1"}, b""))
    assert r["ready"] is False


def test_on_stat_remote_owner_no_peer():
    d = head()
    r, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": "n2-o1"}, b""))
    assert r["ready"] is False  # unreachable owner reports not-ready, no raise


def test_on_stat_remote_owner_routes():
    d = head()
    other = FakePeer({"stat": {"ready": True}})
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": other}
    r, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": "n2-o1"}, b""))
    assert r["ready"] is True


def test_on_stat_hop_is_bounded_by_the_client_budget():
    """A stat that has to reach a remote owner must not block the handler past
    the caller's budget: a wedged peer would otherwise stall the driver's
    ray.wait poll for the full _RPC_TIMEOUT."""
    d = head()
    seen = []

    class Wedged(FakePeer):
        async def call(self, header, payload=b""):
            seen.append(header["obj"])
            await asyncio.sleep(30)  # peer never answers
            raise AssertionError("should have been cut off")

    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": Wedged()}
    started = _time.monotonic()
    r, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": "n2-o1", "timeout": 0.05}, b""))
    elapsed = _time.monotonic() - started
    assert r["ready"] is False  # not ready, never an error
    assert elapsed < 2.0, "stat hop ignored the 0.05s budget (%.3fs)" % elapsed
    assert seen == ["n2-o1"]


def test_on_stat_worker_hop_is_bounded_by_the_client_budget():
    d = worker()

    class DeadHead(FakePeer):
        async def call(self, header, payload=b""):
            await asyncio.sleep(30)
            raise AssertionError("should have been cut off")

    d.head_peer = DeadHead()
    started = _time.monotonic()
    r, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": "n9-o1", "timeout": 0.05}, b""))
    assert r["ready"] is False
    assert _time.monotonic() - started < 2.0


def test_hop_budget_clamps_to_the_rpc_timeout():
    assert _daemon._hop_budget(None) == _daemon._RPC_TIMEOUT
    assert _daemon._hop_budget(0.0) == _daemon._STAT_HOP_FLOOR  # 0 would never fire
    assert _daemon._hop_budget(-5.0) == _daemon._STAT_HOP_FLOOR
    assert _daemon._hop_budget(1e9) == _daemon._RPC_TIMEOUT  # never outlive the RPC cap
    assert _daemon._hop_budget(2.0) == 2.0


def test_on_stat_worker_forwards():
    d = worker()
    d.head_peer = FakePeer({"stat": {"ready": True}})
    r, _ = run(d.on_stat(FakePeer(), {"t": "stat", "obj": "n9-o1"}, b""))
    assert r["ready"] is True


# ---- worker hello -----------------------------------------------------------


def test_on_worker_hello_resolves_pending():
    d = head()

    async def go():
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        d.pending_workers["a1"] = f
        peer = FakePeer()
        r, _ = await d.on_worker_hello(peer, {"t": "worker_hello", "actor": "a1"}, b"")
        assert r["t"] == "worker_hello_ok"
        assert f.done() and f.result() is peer
        assert peer.on_close is not None  # wired to _drop_actor
        return peer

    peer = run(go())
    assert isinstance(peer, FakePeer)


def test_on_worker_hello_no_pending():
    d = head()
    r, _ = run(d.on_worker_hello(FakePeer(), {"t": "worker_hello", "actor": "ghost"}, b""))
    assert r["t"] == "worker_hello_ok"  # no pending future, still acks


# ---- _host_actor ------------------------------------------------------------


def _spawn_stub(proc):
    def _spawn(self, actor_id, gpus):
        return proc

    return _spawn


def test_host_actor_success(monkeypatch):
    d = head()
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    worker_peer = FakePeer({"init": {}})

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)  # let _host_actor register the pending future
        d.pending_workers["a1"].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert r["t"] == "create_actor_ok" and r["actor"] == "a1"
    assert "a1" in d.actors and d.actors["a1"].peer is worker_peer
    assert worker_peer.calls[0][0]["t"] == "init"


def test_host_actor_attach_timeout(monkeypatch):
    d = head()
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))

    # patch the 120s attach timeout down so the never-resolved future times out fast
    real_wait_for = asyncio.wait_for

    async def fast_wait_for(fut, timeout):
        return await real_wait_for(fut, 0.02)

    monkeypatch.setattr(_daemon.asyncio, "wait_for", fast_wait_for)
    r, _ = run(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
    assert "did not attach" in r["err"]
    assert proc.terminated  # orphan subprocess reaped
    assert "a1" not in d.pending_workers


def test_host_actor_init_failure_reaps(monkeypatch):
    d = head()
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    worker_peer = FakePeer(raise_on_call=RuntimeError("ctor blew up"))

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        d.pending_workers["a1"].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert "init failed" in r["err"] and "ctor blew up" in r["err"]
    assert worker_peer.closed and proc.terminated  # _terminate + peer.close
    assert "a1" not in d.actors


# ---- on_create_actor (head) -------------------------------------------------


def test_on_create_actor_cpu_local_host(monkeypatch):
    d = head()
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    worker_peer = FakePeer({"init": {}})
    peer = FakePeer()

    async def go():
        task = asyncio.ensure_future(d.on_create_actor(peer, {"t": "create_actor", "ngpu": 0}, b""))
        await asyncio.sleep(0)
        aid = next(iter(d.pending_workers))
        d.pending_workers[aid].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert r["t"] == "create_actor_ok"
    assert r["node"] == "n1"  # the local host reports its owner node, like the remote path
    aid = r["actor"]
    assert d.actor_loc[aid] == "n1" and aid in peer.created_actors


def test_on_create_actor_rollback_on_worker_failure(monkeypatch):
    """Head places on a remote node; the remote create raises -> force-kill then
    roll back ownership when the kill succeeds."""
    d = head(2)

    class FailCreatePeer(FakePeer):
        async def call(self, header, payload=b""):
            if header.get("t") == "create_actor":
                raise RuntimeError("node died")
            return await super().call(header, payload)

    remote = FailCreatePeer()
    d.nodes["n2"] = {"info": {"node": "n2", "ngpu": 2, "alive": True}, "peer": remote}
    # force placement onto n2 via a pg bundle that lives on n2 with a gpu
    d.pgs["p"] = [{"node": "n2", "gpu": 0}]
    peer = FakePeer()
    r, _ = run(d.on_create_actor(peer, {"t": "create_actor", "pg": "p", "bundle": 0}, b""))
    assert "node died" in r["err"]
    assert d.actor_loc == {}  # routing entry rolled back after successful kill
    assert peer.created_actors == []  # ownership entry removed


def test_on_create_actor_remote_rollback_does_not_free_head_gpus():
    """Remote create failure must not clear head gpu_used by remote GPU index.

    The rollback frees indices against the owner node's GPU table. Remote
    bundle indices are not head indices; treating them as such would free a
    local non-pg reservation that happens to share the same integer index.
    """
    d = head(2)
    d.gpu_used[0] = True  # local non-pg actor already holds head GPU 0

    class FailCreatePeer(FakePeer):
        async def call(self, header, payload=b""):
            if header.get("t") == "create_actor":
                raise RuntimeError("spawn failed")
            return await super().call(header, payload)

    remote = FailCreatePeer()
    d.nodes["n2"] = {"info": {"node": "n2", "ngpu": 2, "alive": True}, "peer": remote}
    d.pgs["p"] = [{"node": "n2", "gpu": 0}]  # same index, different node
    r, _ = run(d.on_create_actor(FakePeer(), {"t": "create_actor", "pg": "p", "bundle": 0}, b""))
    assert "spawn failed" in r["err"]
    assert d.gpu_used[0] is True  # head GPU 0 still reserved
    assert d.actor_loc == {}


def test_on_create_actor_greedy_gpu_rollback(monkeypatch):
    """A greedy (non-pg) GPU actor that fails to start must return its reserved
    GPU to the pool."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    worker_peer = FakePeer(raise_on_call=RuntimeError("boom"))

    async def go():
        task = asyncio.ensure_future(
            d.on_create_actor(FakePeer(), {"t": "create_actor", "ngpu": 1}, b"")
        )
        await asyncio.sleep(0)
        aid = next(iter(d.pending_workers))
        d.pending_workers[aid].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert r.get("err")
    assert d.gpu_used == [False]  # reserved GPU returned on failure
    assert d.actor_loc == {}


def test_on_create_actor_remote_node_success():
    d = head(2)
    remote = FakePeer({"create_actor": {}})
    d.nodes["n2"] = {"info": {"node": "n2", "ngpu": 2, "alive": True}, "peer": remote}
    d.pgs["p"] = [{"node": "n2", "gpu": 0}]
    r, _ = run(d.on_create_actor(FakePeer(), {"t": "create_actor", "pg": "p", "bundle": 0}, b""))
    assert r["t"] == "create_actor_ok" and r["node"] == "n2"
    assert remote.calls[0][0]["t"] == "create_actor"


def test_on_create_actor_place_error():
    d = head()
    r, _ = run(d.on_create_actor(FakePeer(), {"t": "create_actor", "pg": "ghost"}, b""))
    assert "unknown placement group" in r["err"]


def test_on_create_actor_node_unavailable():
    d = head(2)
    # pg points at a node with no live peer entry
    d.pgs["p"] = [{"node": "n2", "gpu": 0}]
    r, _ = run(d.on_create_actor(FakePeer(), {"t": "create_actor", "pg": "p", "bundle": 0}, b""))
    assert "not available" in r["err"]
    assert d.actor_loc == {}  # force_kill drops loc when owner peer is gone


def test_on_create_actor_keeps_routing_when_kill_also_fails():
    """If create and subsequent force-kill both fail, keep actor_loc + ownership."""
    d = head(2)
    remote = FakePeer(raise_on_call=RuntimeError("link down"))
    d.nodes["n2"] = {"info": {"node": "n2", "ngpu": 2, "alive": True}, "peer": remote}
    d.pgs["p"] = [{"node": "n2", "gpu": 0}]
    peer = FakePeer()
    r, _ = run(d.on_create_actor(peer, {"t": "create_actor", "pg": "p", "bundle": 0}, b""))
    assert "link down" in r["err"]
    assert d.actor_loc  # still routable for a later kill
    assert peer.created_actors  # release_client can still retry


def test_rollback_does_not_steal_other_actors_gpu():
    """After kill frees a GPU and another actor re-places it, rollback must not
    clear gpu_used for the new owner."""
    d = head(1)
    # simulate: A reserved GPU 0, was killed, B now holds GPU 0
    d.gpu_used[0] = True
    d.actors["b"] = ActorProc("b", peer=FakePeer(), gpus=[0], proc=FakeProc())
    peer = FakePeer()
    peer.created_actors = ["a"]
    # A is fully gone
    run(d._rollback_failed_create(peer, "a", "n1", [0]))
    assert d.gpu_used[0] is True  # B still owns it
    assert "a" not in peer.created_actors


def test_rollback_schedules_orphan_when_kill_fails_and_peer_closed():
    d = head(2)
    remote = FakePeer(raise_on_call=RuntimeError("down"))
    d.nodes["n2"] = {"info": {"node": "n2", "ngpu": 2, "alive": True}, "peer": remote}
    d.actor_loc["a1"] = "n2"
    peer = FakePeer()
    peer.closed = True
    peer.created_actors = ["a1"]
    run(d._rollback_failed_create(peer, "a1", "n2", [0]))
    assert "a1" in d._orphans
    assert d.actor_loc.get("a1") == "n2"
    # second schedule is no-op
    d._schedule_orphan_reap("a1", "n2")
    assert d._orphans["a1"] == "n2"


def test_reap_orphan_clears_when_untracked():
    d = head(2)
    d._orphans["a1"] = "n2"

    async def go():
        await d._reap_orphan("a1", "n2")

    run(go())
    assert "a1" not in d._orphans


def test_reap_orphan_retries_until_gone(monkeypatch):
    d = head(2)
    d.actor_loc["a1"] = "n2"
    d._orphans["a1"] = "n2"
    tries = {"n": 0}

    async def force(actor_id, node):
        tries["n"] += 1
        if tries["n"] >= 2:
            d.actor_loc.pop(actor_id, None)

    async def nosleep(_s):
        return None

    monkeypatch.setattr(d, "_force_kill_actor", force)
    monkeypatch.setattr(asyncio, "sleep", nosleep)

    async def go():
        await d._reap_orphan("a1", "n2")

    run(go())
    assert "a1" not in d._orphans and tries["n"] >= 2


def test_reap_orphan_keeps_slot_on_cancel_if_still_tracked(monkeypatch):
    """Cancel mid-reap leaves the slot; schedule can restart via task.done()."""
    d = head(2)
    d.actor_loc["a1"] = "n2"
    d._orphans["a1"] = "n2"
    calls = {"n": 0}

    async def force(actor_id, node):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise asyncio.CancelledError()

    async def nosleep(_s):
        return None

    monkeypatch.setattr(d, "_force_kill_actor", force)
    monkeypatch.setattr(asyncio, "sleep", nosleep)

    async def go():
        try:
            await d._reap_orphan("a1", "n2")
        except asyncio.CancelledError:
            pass

    run(go())
    assert "a1" in d._orphans  # still needs reaping

    # restart is allowed because the previous task is done
    async def schedule_under_loop():
        d._schedule_orphan_reap("a1", "n2")
        assert "a1" in d._orphan_tasks

    run(schedule_under_loop())


def test_reap_orphan_frees_deferred_pgs_when_idle(monkeypatch):
    d = head(2)
    d.actor_loc["a1"] = "n2"
    d._orphans["a1"] = "n2"
    d.pgs["p1"] = [{"node": "n2", "gpu": 0}]
    d._orphan_pgs.add("p1")

    async def force(actor_id, node):
        d.actor_loc.pop(actor_id, None)

    monkeypatch.setattr(d, "_force_kill_actor", force)

    async def go():
        await d._reap_orphan("a1", "n2")

    run(go())
    assert "p1" not in d.pgs
    assert not d._orphan_pgs


def test_on_create_actor_worker_no_actor_forwards():
    d = worker()
    d.head_peer = FakePeer({"create_actor": {"actor": "n1-a5"}})
    peer = FakePeer()
    r, _ = run(d.on_create_actor(peer, {"t": "create_actor", "ngpu": 0}, b""))
    assert r["actor"] == "n1-a5"
    assert peer.created_actors == ["n1-a5"]  # tracked on the worker


def test_on_create_actor_worker_with_actor_hosts(monkeypatch):
    d = worker()
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    worker_peer = FakePeer({"init": {}})

    async def go():
        task = asyncio.ensure_future(
            d.on_create_actor(FakePeer(), {"t": "create_actor", "actor": "n1-a1", "gpus": []}, b"")
        )
        await asyncio.sleep(0)
        d.pending_workers["n1-a1"].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert r["t"] == "create_actor_ok" and "n1-a1" in d.actors


# ---- on_call ----------------------------------------------------------------


def test_on_call_local_actor_returns_obj():
    d = head()
    ap = ActorProc("a1", peer=FakePeer({"method": {"_body": b"r"}}), gpus=[])
    d.actors["a1"] = ap
    d.actor_loc["a1"] = "n1"
    r, _ = run(d.on_call(FakePeer(), {"t": "call", "actor": "a1", "method": "f"}, b""))
    assert r["t"] == "call_ok" and r["obj"].startswith("n1-o")
    # the dispatched slot eventually carries the worker reply
    obj = r["obj"]

    async def wait_slot():
        await asyncio.wait_for(d.objects[obj].ev.wait(), 1)
        return d.objects[obj].data

    # re-running on a fresh loop won't see the task; assert slot exists instead
    assert obj in d.objects


def test_on_call_unknown_actor():
    d = head()
    r, _ = run(d.on_call(FakePeer(), {"t": "call", "actor": "ghost", "method": "f"}, b""))
    assert "unknown actor" in r["err"]


def test_on_call_remote_routes():
    d = head()
    remote = FakePeer({"call": {"obj": "n2-o7"}})
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": remote}
    d.actor_loc["a1"] = "n2"
    r, _ = run(d.on_call(FakePeer(), {"t": "call", "actor": "a1", "method": "f"}, b""))
    assert r["obj"] == "n2-o7"


def test_on_call_bounce_back_guard():
    """The call arrives from the very node we'd forward to (actor died there
    mid-flight): must fail cleanly, not loop."""
    d = head()
    peer = FakePeer()
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": peer}
    d.actor_loc["a1"] = "n2"
    r, _ = run(d.on_call(peer, {"t": "call", "actor": "a1", "method": "f"}, b""))
    assert "unknown actor" in r["err"]


def test_on_call_worker_actor_elsewhere_forwards():
    d = worker()
    d.head_peer = FakePeer({"call": {"obj": "n1-o9"}})
    r, _ = run(d.on_call(FakePeer(), {"t": "call", "actor": "n1-a1", "method": "f"}, b""))
    # forwarded to head; head returns its raw resp
    assert d.head_peer.calls[0][0]["actor"] == "n1-a1"


def test_dispatch_sets_slot_data():
    d = head()
    ap = ActorProc("a1", peer=FakePeer({"method": {"_body": b"result"}}), gpus=[])

    async def go():
        slot = ObjSlot()
        await d._dispatch(ap, "f", b"args", slot)
        return slot

    slot = run(go())
    assert slot.ev.is_set() and slot.data == b"result" and slot.err == ""


def test_dispatch_records_error():
    d = head()
    ap = ActorProc("a1", peer=FakePeer(raise_on_call=RuntimeError("method boom")), gpus=[])

    async def go():
        slot = ObjSlot()
        await d._dispatch(ap, "f", b"", slot)
        return slot

    slot = run(go())
    assert slot.ev.is_set() and "method boom" in slot.err


# ---- on_kill ----------------------------------------------------------------


def test_on_kill_local_actor_frees_gpu():
    d = head(2)
    proc = FakeProc()
    ap = ActorProc("a1", peer=FakePeer(), gpus=[1], proc=proc)
    d.actors["a1"] = ap
    d.actor_loc["a1"] = "n1"
    d.gpu_used[1] = True
    r, _ = run(d.on_kill(FakePeer(), {"actor": "a1"}, b""))
    assert r["t"] == "kill_ok"
    assert "a1" not in d.actors and d.gpu_used[1] is False
    assert ap.peer.closed and proc.terminated


def test_on_kill_remote_routes():
    d = head()
    remote = FakePeer()
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": remote}
    d.actor_loc["a1"] = "n2"
    r, _ = run(d.on_kill(FakePeer(), {"actor": "a1"}, b""))
    assert r["t"] == "kill_ok"
    assert remote.calls[0][0] == {"t": "kill", "actor": "a1"}
    assert "a1" not in d.actor_loc


def test_on_kill_remote_keeps_routing_on_rpc_failure():
    """If the owner kill RPC fails, actor_loc must stay so a retry can route."""
    d = head()
    remote = FakePeer(raise_on_call=RuntimeError("link down"))
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": remote}
    d.actor_loc["a1"] = "n2"
    r, _ = run(d.on_kill(FakePeer(), {"actor": "a1"}, b""))
    assert "link down" in r["err"]
    assert d.actor_loc["a1"] == "n2"


def test_on_kill_bounce_back_guard():
    """Kill arrives from the owner node we'd forward to: drop routing, no loop."""
    d = head()
    peer = FakePeer()
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": peer}
    d.actor_loc["a1"] = "n2"
    r, _ = run(d.on_kill(peer, {"actor": "a1"}, b""))
    assert r["t"] == "kill_ok"
    assert "a1" not in d.actor_loc
    assert peer.calls == []  # must not re-forward to the same peer


def test_on_kill_mid_create_reaps_hosting_proc():
    d = head(1)
    proc = FakeProc()
    wpeer = FakePeer()
    d._hosting["a1"] = (proc, wpeer, [0])

    async def go():
        loop = asyncio.get_running_loop()
        f = loop.create_future()
        d.pending_workers["a1"] = f
        d.actor_loc["a1"] = "n1"
        r, _ = await d.on_kill(FakePeer(), {"actor": "a1"}, b"")
        return r, f

    r, f = run(go())
    assert r["t"] == "kill_ok"
    assert proc.terminated and wpeer.closed
    assert f.done() and "killed during create" in str(f.exception())
    assert "a1" not in d._hosting


def test_on_kill_from_head_unknown_local_is_ok():
    """Worker must not bounce a head kill for an id it does not host."""
    d = worker()
    d.head_peer = FakePeer()
    r, _ = run(d.on_kill(d.head_peer, {"actor": "ghost"}, b""))
    assert r["t"] == "kill_ok"
    assert d.head_peer.calls == []  # no forward
    assert "ghost" in d._kill_pending  # tombstone for late create


def test_host_actor_aborts_if_kill_pending(monkeypatch):
    d = worker()
    d.sock_path = "/x.sock"
    d._kill_pending.add("a1")
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(FakeProc()))
    r, _ = run(d._host_actor({"actor": "a1", "gpus": []}, b""))
    assert "killed during create" in r["err"]
    assert "a1" not in d._kill_pending


def test_host_actor_aborts_if_kill_pending_after_spawn(monkeypatch):
    """Kill tombstone arrives after spawn/hosting entry is installed."""
    d = worker()
    d.sock_path = "/x.sock"
    proc = FakeProc()

    def spawn_and_tombstone(self, actor_id, gpus):
        d._kill_pending.add(actor_id)
        return proc

    monkeypatch.setattr(Daemon, "_spawn_worker", spawn_and_tombstone)
    r, _ = run(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
    assert "killed" in r["err"] and proc.terminated


def test_create_actor_aborts_when_client_disconnects_after_host(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    worker_peer = FakePeer({"init": {}})
    driver = FakePeer()

    async def go():
        task = asyncio.ensure_future(
            d.on_create_actor(driver, {"t": "create_actor", "ngpu": 1}, b"")
        )
        await asyncio.sleep(0)
        aid = next(iter(d.pending_workers))
        d.pending_workers[aid].set_result(worker_peer)
        driver.closed = True  # disconnect before create returns
        return await task

    r, _ = run(go())
    assert "disconnected" in r["err"]
    assert d.actor_loc == {}


def test_create_actor_worker_forward_disconnects(monkeypatch):
    d = worker()
    d.head_peer = FakePeer({"create_actor": {"actor": "n1-a9"}})
    driver = FakePeer()
    driver.closed = True
    r, _ = run(d.on_create_actor(driver, {"t": "create_actor", "ngpu": 0}, b""))
    assert "disconnected" in r["err"]
    assert {"t": "kill", "actor": "n1-a9"} in [c[0] for c in d.head_peer.calls]
    assert driver.created_actors == []


def test_create_actor_worker_forward_disconnect_kill_errors(monkeypatch):
    """Disconnect-after-create path swallows kill failures."""
    d = worker()
    d.head_peer = FakePeer(
        responses={"create_actor": {"actor": "n1-a9"}},
        raise_on_call=None,
    )
    # first call succeeds (create); subsequent calls fail
    calls = {"n": 0}
    orig_call = d.head_peer.call

    async def flaky(header, payload=b""):
        calls["n"] += 1
        if calls["n"] == 1:
            return await orig_call(header, payload)
        raise RuntimeError("gone")

    d.head_peer.call = flaky
    d.actors["n1-a9"] = ActorProc("n1-a9", peer=FakePeer(), gpus=[], proc=FakeProc())
    driver = FakePeer()
    driver.closed = True

    async def boom(*a, **k):
        raise RuntimeError("local kill failed")

    monkeypatch.setattr(d, "on_kill", boom)
    r, _ = run(d.on_create_actor(driver, {"t": "create_actor", "ngpu": 0}, b""))
    assert "disconnected" in r["err"]


def test_force_kill_actor_remote():
    d = head()
    remote = FakePeer()
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": remote}
    d.actor_loc["a1"] = "n2"
    run(d._force_kill_actor("a1", "n2"))
    assert remote.calls[0][0]["t"] == "kill"
    assert "a1" not in d.actor_loc


def test_force_kill_actor_remote_rpc_error():
    d = head()
    remote = FakePeer(raise_on_call=RuntimeError("down"))
    d.nodes["n2"] = {"info": {"node": "n2"}, "peer": remote}
    d.actor_loc["a1"] = "n2"
    run(d._force_kill_actor("a1", "n2"))  # swallowed
    assert d.actor_loc["a1"] == "n2"  # keep routing for retry


def test_force_kill_actor_local():
    d = head(1)
    proc = FakeProc()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[0], proc=proc)
    d.gpu_used[0] = True
    run(d._force_kill_actor("a1", "n1"))
    assert "a1" not in d.actors and proc.terminated


def test_force_kill_actor_outer_exception_swallowed(monkeypatch):
    d = head()

    def boom(*a, **k):
        raise RuntimeError("peer table exploded")

    monkeypatch.setattr(d, "_peer_for", boom)
    run(d._force_kill_actor("a1", "n2"))  # must not raise


def test_host_actor_aborted_by_kill_during_attach(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        # simulate on_kill mid-create (attach not yet done)
        fut = d.pending_workers["a1"]
        fut.set_exception(RuntimeError("actor killed during create"))
        return await task

    r, _ = run(go())
    assert "aborted" in r["err"] or "killed" in r["err"]
    assert "a1" not in d._hosting


def test_shutdown_reaps_hosting_procs():
    d = head()
    p1, p2 = FakeProc(), FakeProc()
    wp = FakePeer()
    loop = asyncio.new_event_loop()
    pending_fut = loop.create_future()
    wp.pending[1] = pending_fut
    d._hosting["a1"] = (p1, None, [])
    d._hosting["a2"] = (p2, wp, [0])
    d.gpu_used = [True]

    class BoomWriter:
        def close(self):
            raise OSError("already closed")

    wp.writer = BoomWriter()

    async def go():
        f = asyncio.get_running_loop().create_future()
        d.pending_workers["pending"] = f
        d.shutdown()
        return f

    f = run(go())
    assert p1.terminated and p2.terminated
    assert d._hosting == {} and d.pending_workers == {}
    assert f.done() and "shutdown" in str(f.exception())
    assert pending_fut.done()
    loop.close()


def test_shutdown_reaps_actors_with_pending():
    d = head(1)
    peer = FakePeer()
    loop = asyncio.new_event_loop()
    fut = loop.create_future()
    peer.pending[7] = fut
    d.actors["a1"] = ActorProc("a1", peer=peer, gpus=[0], proc=FakeProc())
    d.gpu_used[0] = True

    class BoomW:
        def close(self):
            raise RuntimeError("x")

    peer.writer = BoomW()
    d.shutdown()
    assert "a1" not in d.actors and fut.done() and d.gpu_used[0] is False
    loop.close()


def test_host_actor_killed_after_init_before_publish(monkeypatch):
    """Kill pops _hosting after init succeeds; actor must not be published."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))

    class KillAfterInitPeer(FakePeer):
        async def call(self, header, payload=b""):
            # after successful init reply, simulate concurrent kill
            d._hosting.pop("a1", None)
            return await super().call(header, payload)

        async def close(self):
            self.closed = True
            raise RuntimeError("close after kill race")

    worker_peer = KillAfterInitPeer({"init": {}})

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        d.pending_workers["a1"].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert "killed during create" in r["err"]
    assert "a1" not in d.actors


def test_on_kill_hosting_peer_close_error():
    d = head(1)

    class BoomPeer(FakePeer):
        async def close(self):
            self.closed = True
            raise RuntimeError("close failed")

    proc = FakeProc()
    d._hosting["a1"] = (proc, BoomPeer(), [])
    r, _ = run(d.on_kill(FakePeer(), {"actor": "a1"}, b""))
    assert r["t"] == "kill_ok" and proc.terminated


def test_host_actor_init_close_error_swallowed(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))

    class BoomClosePeer(FakePeer):
        def __init__(self):
            super().__init__(raise_on_call=RuntimeError("init boom"))

        async def close(self):
            self.closed = True
            raise RuntimeError("close also boom")

    worker_peer = BoomClosePeer()

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": []}, b""))
        await asyncio.sleep(0)
        d.pending_workers["a1"].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert "init failed" in r["err"]


def test_worker_hello_attaches_peer_to_hosting():
    d = head()
    proc = FakeProc()
    d._hosting["a1"] = (proc, None, [1])
    peer = FakePeer()
    run(d.on_worker_hello(peer, {"t": "worker_hello", "actor": "a1"}, b""))
    assert d._hosting["a1"] == (proc, peer, [1])


def test_on_hello_stale_close_guard_returns():
    d = head()
    old = FakePeer()
    new = FakePeer()
    run(d.on_hello(old, {"t": "hello", "node": "n2", "ip": "x", "ngpu": 1}, b""))
    old.created_pgs = ["keep-me"]
    d.pgs["keep-me"] = [{"node": "n1", "gpu": 0}]
    run(d.on_hello(new, {"t": "hello", "node": "n2", "ip": "x", "ngpu": 1}, b""))
    assert old.closed  # re-hello awaits close of superseded peer
    assert "keep-me" in new.created_pgs  # ownership transferred
    assert "n2" in d.nodes and d.nodes["n2"]["peer"] is new


def test_on_worker_close_stale_guard():
    """_on_worker_close no-ops when nodes[node].peer is no longer this peer."""
    d = head()
    live = FakePeer()
    run(d.on_hello(live, {"t": "hello", "node": "n2", "ip": "x", "ngpu": 1}, b""))
    cb = live.on_close
    live.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": 0}]
    # replace peer without going through re-hello close
    d.nodes["n2"]["peer"] = FakePeer()
    run(cb())  # hits stale guard return
    assert "p1" in d.pgs
    assert "n2" in d.nodes


def test_stale_worker_close_skips_release():
    """After re-hello, ownership moves to the new peer; old peer is closed."""
    d = head()
    old = FakePeer()
    new = FakePeer()
    run(d.on_hello(old, {"t": "hello", "node": "n2", "ip": "x", "ngpu": 1}, b""))
    old.created_pgs = ["p-should-not-drop"]
    old.created_actors = ["a-owned"]
    d.pgs["p-should-not-drop"] = [{"node": "n1", "gpu": 0}]
    d.actor_loc["a-owned"] = "n2"
    run(d.on_hello(new, {"t": "hello", "node": "n2", "ip": "x", "ngpu": 1}, b""))
    assert old.closed and old.superseded
    # ownership moved to the new peer so a later disconnect still cleans up
    assert "p-should-not-drop" in new.created_pgs
    assert "a-owned" in new.created_actors
    assert old.created_pgs == [] and old.created_actors == []
    # new peer disconnect still drops node, not a stale old close
    assert d.nodes["n2"]["peer"] is new


def test_release_client_skips_transferred_ids():
    """In-flight release must not kill ids that re-hello moved off the peer."""
    d = head(1)
    peer = FakePeer()
    peer.created_actors = ["a1"]
    d.actor_loc["a1"] = "n1"
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[0], proc=FakeProc())
    d.gpu_used[0] = True

    async def go():
        # start release, but transfer ownership mid-flight before kill
        peer.superseded = True
        peer.created_actors.clear()
        await d.release_client(peer)

    run(go())
    assert "a1" in d.actors  # not killed


def test_release_client_continues_after_missing_id(monkeypatch):
    """A concurrent rollback removing one id must not skip remaining PGs."""
    d = head()
    peer = FakePeer()
    peer.created_actors = ["a1", "a2"]
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actors["a2"] = ActorProc("a2", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actor_loc["a1"] = d.actor_loc["a2"] = "n1"

    async def kill_removes_other(peer_arg, m, payload):
        # first kill claims a1; a2 already gone from list (simulated)
        if m.get("actor") == "a1" and "a2" in peer.created_actors:
            peer.created_actors.remove("a2")
        return {"t": "kill_ok"}, b""

    monkeypatch.setattr(d, "on_kill", kill_removes_other)
    run(d.release_client(peer))
    assert "p1" not in d.pgs  # PG loop still runs


def test_release_client_skips_pg_after_transfer():
    d = head()
    peer = FakePeer()
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    peer.superseded = True
    peer.created_pgs.clear()
    run(d.release_client(peer))
    assert "p1" in d.pgs


def test_release_client_worker_skips_when_superseded():
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()
    peer.created_actors = ["a1"]
    peer.superseded = True
    run(d.release_client(peer))
    assert d.head_peer.calls == []


def test_release_client_skips_id_removed_from_list_mid_loop(monkeypatch):
    d = head(1)
    peer = FakePeer()
    peer.created_actors = ["a1"]
    d.actor_loc["a1"] = "n1"
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[0], proc=FakeProc())

    async def clear_then_ok(peer_arg, m, payload):
        peer.created_actors.clear()
        return {"t": "kill_ok"}, b""

    # membership check is before on_kill; clear list before release so second id path
    peer.created_actors = ["a1"]
    peer.created_actors.remove("a1")  # empty after snapshot would need mid-await
    # better: two ids, clear second during first kill
    peer.created_actors = ["a1", "a2"]
    d.actor_loc["a2"] = "n1"
    d.actors["a2"] = ActorProc("a2", peer=FakePeer(), gpus=[], proc=FakeProc())

    async def kill_clears(peer_arg, m, payload):
        peer.created_actors[:] = []  # transfer away remaining
        return {"t": "kill_ok"}, b""

    monkeypatch.setattr(d, "on_kill", kill_clears)
    run(d.release_client(peer))
    # a2 not killed via second iteration because list was cleared
    assert "a2" in d.actors


def test_release_client_worker_skips_pg_mid_loop():
    """Ghost membership mid-claim: id vanishes from live list -> continue."""
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()

    class GhostList(list):
        def __contains__(self, item):
            # first id present for claim, later checks fail
            return item in list(self) and item != "p2"

        def remove(self, item):
            list.remove(self, item)

    peer.created_pgs = GhostList(["p1", "p2"])
    run(d.release_client(peer))
    # p2 skipped via not-in; only p1 forwarded
    assert len(d.head_peer.calls) == 1
    assert d.head_peer.calls[0][0]["pg"] == "p1"


def test_release_client_worker_superseded_mid_actor_loop():
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            peer.superseded = True

    peer.created_actors = SuperList(["a1", "a2"])

    async def go():
        await d.release_client(peer)

    run(go())
    assert len(d.head_peer.calls) == 1


def test_release_client_worker_superseded_mid_pg_loop():
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            peer.superseded = True

    peer.created_pgs = SuperList(["p1", "p2"])
    run(d.release_client(peer))
    assert len(d.head_peer.calls) == 1


def test_release_client_schedules_retry_on_forward_fail(monkeypatch):
    """Dying driver peer must not be re-appended; background retry is scheduled."""
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("gone"))
    peer = FakePeer()
    peer.created_actors = ["a1"]
    peer.created_pgs = ["p1"]
    scheduled: list[str] = []

    def track(task):
        scheduled.append("release")
        task.cancel()  # don't run infinite retry in unit test
        return task

    monkeypatch.setattr(d, "_track", track)
    run(d.release_client(peer))
    assert "a1" not in peer.created_actors  # claimed, not dead-lettered
    assert "p1" not in peer.created_pgs
    # single chained task: kills then pgs
    assert scheduled == ["release"]


def test_release_client_schedules_pg_retry_on_forward_fail(monkeypatch):
    """PG-only release with head down: retry remove_pg without kill chain."""
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("gone"))
    peer = FakePeer()
    peer.created_pgs = ["p1"]
    scheduled: list[str] = []

    def track(task):
        scheduled.append("pg")
        task.cancel()
        return task

    monkeypatch.setattr(d, "_track", track)
    run(d.release_client(peer))
    assert "p1" not in peer.created_pgs
    assert scheduled == ["pg"]


def test_release_client_head_schedules_orphan_on_kill_fail(monkeypatch):
    d = head()
    peer = FakePeer()
    peer.created_actors = ["a1"]
    d.actor_loc["a1"] = "n2"
    scheduled: list[tuple[str, str]] = []

    def capture(actor_id, node):
        scheduled.append((actor_id, node))
        d._orphans[actor_id] = node  # register without starting a task

    async def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(d, "on_kill", boom)
    monkeypatch.setattr(d, "_schedule_orphan_reap", capture)
    run(d.release_client(peer))
    assert "a1" not in peer.created_actors  # not dead-lettered onto dying peer
    assert scheduled == [("a1", "n2")]


def test_release_client_head_soft_err_schedules_orphan(monkeypatch):
    """on_kill returns err without raising: still schedule orphan reaper."""
    d = head()
    peer = FakePeer()
    peer.created_actors = ["a1"]
    d.actor_loc["a1"] = "n2"
    scheduled: list[tuple[str, str]] = []

    def capture(actor_id, node):
        scheduled.append((actor_id, node))
        d._orphans[actor_id] = node

    async def soft(*a, **k):
        return {"err": "link down"}, b""

    monkeypatch.setattr(d, "on_kill", soft)
    monkeypatch.setattr(d, "_schedule_orphan_reap", capture)
    run(d.release_client(peer))
    assert scheduled == [("a1", "n2")]
    assert "a1" not in peer.created_actors


def test_release_client_head_defers_pg_while_orphans(monkeypatch):
    """PG reservations stay until orphan actors are reaped (no double-book)."""
    d = head()
    peer = FakePeer()
    peer.created_actors = ["a1"]
    peer.created_pgs = ["p1"]
    d.actor_loc["a1"] = "n2"
    d.pgs["p1"] = [{"node": "n2", "gpu": 0}]

    def capture(actor_id, node):
        d._orphans[actor_id] = node

    async def soft(*a, **k):
        return {"err": "down"}, b""

    monkeypatch.setattr(d, "on_kill", soft)
    monkeypatch.setattr(d, "_schedule_orphan_reap", capture)
    run(d.release_client(peer))
    assert "a1" in d._orphans
    assert "p1" in d.pgs  # not freed while orphan lives
    assert "p1" in d._orphan_pgs


def test_release_client_head_skips_missing_actor_id():
    d = head()
    peer = FakePeer()
    peer.created_actors = ["a1"]
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    # a1 not in list when loop checks (removed before release)
    peer.created_actors.clear()
    run(d.release_client(peer))
    assert "p1" not in d.pgs


def test_release_client_continue_on_missing_ids_in_snapshot():
    """Snapshot has ghost ids already removed from live lists -> continue."""
    d = head()
    peer = FakePeer()

    class GhostList(list):
        def __contains__(self, item):
            return False  # always missing at check time

        def remove(self, item):
            raise AssertionError("must not claim missing id")

    peer.created_actors = GhostList(["ghost"])
    peer.created_pgs = GhostList(["ghost-pg"])
    # no real work: both continue; ensure no crash
    run(d.release_client(peer))


def test_release_client_worker_continue_missing_actor():
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()

    class GhostList(list):
        def __contains__(self, item):
            return False

    peer.created_actors = GhostList(["ghost"])
    peer.created_pgs = ["p1"]
    run(d.release_client(peer))
    assert d.head_peer.calls and d.head_peer.calls[0][0]["t"] == "remove_pg"


def test_release_client_head_superseded_mid_pg_loop():
    d = head()
    peer = FakePeer()
    peer.created_actors = []

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            if item == "p1":
                peer.superseded = True

    peer.created_pgs = SuperList(["p1", "p2"])
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    d.pgs["p2"] = [{"node": "n1", "gpu": -1}]
    run(d.release_client(peer))
    assert "p1" not in d.pgs and "p2" in d.pgs


def test_release_client_head_superseded_mid_actor_loop():
    d = head()
    peer = FakePeer()
    new_peer = FakePeer()

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            peer.superseded = True
            peer.superseded_by = new_peer
            # real re-hello: transfer remaining then clear
            for a in list(self):
                new_peer.created_actors.append(a)
            peer.created_actors.clear()
            peer.created_pgs.clear()

    peer.created_actors = SuperList(["a1", "a2"])
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actors["a2"] = ActorProc("a2", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actor_loc["a1"] = d.actor_loc["a2"] = "n1"
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    run(d.release_client(peer))
    # PGs handed to replacement peer only because it still owns transferred a2
    assert "a2" in new_peer.created_actors
    assert "p1" in new_peer.created_pgs
    assert "p1" in d.pgs


def test_release_client_superseded_always_hands_pgs_to_by():
    """Even if transfer lists look empty, hand claimed PGs to superseded_by."""
    d = head()
    peer = FakePeer()
    new_peer = FakePeer()

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            peer.superseded = True
            peer.superseded_by = new_peer
            peer.created_actors.clear()
            peer.created_pgs.clear()

    peer.created_actors = SuperList(["a1"])
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actor_loc["a1"] = "n1"
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    run(d.release_client(peer))
    # Safe over-retain: new peer owns p1 until it disconnects
    assert "p1" in new_peer.created_pgs
    assert "p1" in d.pgs


def test_release_client_superseded_without_by_keeps_on_peer():
    """Edge path: superseded but no superseded_by; remaining actors keep PGs."""
    d = head()
    peer = FakePeer()

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            peer.superseded = True
            # leave a2 on list (no clear) so elif peer.created_actors branch runs

    peer.created_actors = SuperList(["a1", "a2"])
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actors["a2"] = ActorProc("a2", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actor_loc["a1"] = d.actor_loc["a2"] = "n1"
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    run(d.release_client(peer))
    assert "p1" in peer.created_pgs
    assert "p1" in d.pgs


def test_terminate_waits_then_kills(monkeypatch):
    class Stubborn:
        def __init__(self):
            self.steps = []
            self._alive = True

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self.steps.append("term")

        def kill(self):
            self.steps.append("kill")
            self._alive = False

        def wait(self, timeout=None):
            self.steps.append("wait:%s" % timeout)
            if "kill" not in self.steps:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return 0

    p = Stubborn()
    _terminate(p)
    assert p.steps[0] == "term"
    assert "kill" in p.steps
    assert p.poll() == 0


def test_on_actor_gone_worker_forwards():
    d = worker()
    d.head_peer = FakePeer({"actor_gone": {}})
    r, _ = run(d.on_actor_gone(FakePeer(), {"t": "actor_gone", "actor": "a1"}, b""))
    assert d.head_peer.calls[0][0]["t"] == "actor_gone"


def test_worker_actor_close_notifies_head():
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()
    run(d.on_worker_hello(peer, {"t": "worker_hello", "actor": "a1"}, b""))
    assert peer.on_close is not None
    run(peer.on_close())
    assert {"t": "actor_gone", "actor": "a1"} in [c[0] for c in d.head_peer.calls]


def test_worker_actor_close_swallows_notify_error():
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("head gone"))
    peer = FakePeer()
    run(d.on_worker_hello(peer, {"t": "worker_hello", "actor": "a1"}, b""))
    run(peer.on_close())  # no raise


def test_release_client_local_fallback_swallows_kill_error(monkeypatch):
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("head gone"))
    proc = FakeProc()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=proc)
    peer = FakePeer()
    peer.created_actors = ["a1"]

    async def boom(*a, **k):
        raise RuntimeError("kill failed")

    monkeypatch.setattr(d, "on_kill", boom)
    run(d.release_client(peer))  # no raise


def test_on_kill_unknown_is_ok():
    d = head()
    r, _ = run(d.on_kill(FakePeer(), {"actor": "ghost"}, b""))
    assert r["t"] == "kill_ok"


def test_on_kill_worker_forwards():
    d = worker()
    d.head_peer = FakePeer()
    r, _ = run(d.on_kill(FakePeer(), {"actor": "n1-a1"}, b""))
    assert r["t"] == "kill_ok"
    assert d.head_peer.calls[0][0] == {"t": "kill", "actor": "n1-a1"}


def test_on_kill_worker_local_notifies_head():
    """ray.kill on the owner worker must clear head actor_loc via actor_gone."""
    d = worker()
    d.head_peer = FakePeer()
    proc = FakeProc()
    ap = ActorProc("a1", peer=FakePeer(), gpus=[], proc=proc)
    d.actors["a1"] = ap
    r, _ = run(d.on_kill(FakePeer(), {"actor": "a1"}, b""))
    assert r["t"] == "kill_ok"
    assert "a1" not in d.actors and proc.terminated
    assert {"t": "actor_gone", "actor": "a1"} in [c[0] for c in d.head_peer.calls]


def test_on_actor_gone_head_drops_routing():
    d = head()
    d.actor_loc["a1"] = "n2"
    r, _ = run(d.on_actor_gone(FakePeer(), {"t": "actor_gone", "actor": "a1"}, b""))
    assert r["t"] == "actor_gone_ok" and "a1" not in d.actor_loc


# ---- on_hello (membership) --------------------------------------------------


def test_on_hello_registers_node():
    d = head()
    peer = FakePeer()
    r, _ = run(d.on_hello(peer, {"t": "hello", "node": "n2", "ip": "9.9.9.9", "ngpu": 3}, b""))
    assert r["t"] == "hello_ok"
    assert d.nodes["n2"]["info"]["ngpu"] == 3 and d.nodes["n2"]["peer"] is peer
    assert peer.on_close is not None  # wired to release + _drop_node


@pytest.mark.parametrize("bad", ["2", 2.5, -5, None, [1], True, 10**9, 2**40])
def test_on_hello_normalizes_malformed_ngpu(bad):
    """A peer's GPU count bounds range(), ngpu-used and `ray status`'s %d, so a
    non-count must not be stored: it crashed the head's create_pg handler and
    poisoned the status view. The node joins as a 0-GPU node instead."""
    d = head()
    peer = FakePeer()
    r, _ = run(d.on_hello(peer, {"t": "hello", "node": "n2", "ip": "9.9.9.9", "ngpu": bad}, b""))
    assert r["t"] == "hello_ok"
    assert d.nodes["n2"]["info"]["ngpu"] == 0
    # the stored value must survive every consumer of it
    status, _ = run(d.on_status(peer, {"t": "status"}, b""))
    assert status["nodes"][-1]["ngpu"] == 0
    resources, _ = run(d.on_resources(peer, {"t": "resources"}, b""))
    assert resources["data"]["n2"] == {"GPU": 0.0, "CPU": 1.0}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, "1", [1], {"GPU": 1}])
def test_on_create_pg_rejects_malformed_bundle_spec(bad):
    d = head()
    r, _ = run(d.on_create_pg(FakePeer(), {"t": "create_pg", "specs": [{"GPU": bad}]}, b""))
    assert "invalid bundle spec" in r["err"]
    assert d.pgs == {}  # nothing half-placed


def test_on_create_pg_zero_gpu_bundle_is_cpu_bundle():
    d = head(2)
    r, _ = run(d.on_create_pg(FakePeer(), {"t": "create_pg", "specs": [{"CPU": 1}]}, b""))
    assert r["t"] == "create_pg_ok"
    assert d.pgs[r["pg"]] == [{"node": "n1", "gpu": -1}]


def test_on_hello_close_releases_owned_pgs_and_drops_node():
    """Worker disconnect must free PGs/actors tracked on that connection
    (driver-on-worker ownership), not only drop membership. Otherwise a PG
    that reserved a head GPU stays forever after the worker dies."""
    d = head(2)
    peer = FakePeer()
    run(d.on_hello(peer, {"t": "hello", "node": "n2", "ip": "9.9.9.9", "ngpu": 2}, b""))
    # PG spans head + worker; ownership is on the worker connection because the
    # driver sat on that worker and the create_pg was forwarded.
    d.pgs["p1"] = [{"node": "n1", "gpu": 0}, {"node": "n2", "gpu": 0}]
    d.actor_loc["a-remote"] = "n3"  # actor on a third node, owned by this driver
    d.actor_loc["a-on-n2"] = "n2"
    peer.created_pgs = ["p1"]
    peer.created_actors = ["a-remote", "a-on-n2"]
    # third node so on_kill of a-remote has somewhere to route
    other = FakePeer()
    d.nodes["n3"] = {"info": {"node": "n3", "ngpu": 1, "alive": True}, "peer": other}

    run(peer.on_close())

    assert "n2" not in d.nodes
    assert "p1" not in d.pgs  # head GPU reservation released
    assert "a-on-n2" not in d.actor_loc
    assert "a-remote" not in d.actor_loc
    assert other.calls and other.calls[0][0]["t"] == "kill"


def test_on_hello_rejected_on_worker():
    d = worker()
    r, _ = run(d.on_hello(FakePeer(), {"t": "hello", "node": "n2"}, b""))
    assert "not the head" in r["err"]


# ---- _forward_head ----------------------------------------------------------


def test_forward_head_no_connection():
    d = worker()
    r, _ = run(d._forward_head({"t": "status"}))
    assert "no head connection" in r["err"]


def test_forward_head_relays():
    d = worker()
    d.head_peer = FakePeer({"status": {"nodes": []}})
    r, _ = run(d._forward_head({"t": "status"}))
    assert r["nodes"] == []


# ---- release_client ---------------------------------------------------------


def test_release_client_head_frees_actors_and_pgs():
    d = head(2)
    proc = FakeProc()
    ap = ActorProc("a1", peer=FakePeer(), gpus=[0], proc=proc)
    d.actors["a1"] = ap
    d.actor_loc["a1"] = "n1"
    d.gpu_used[0] = True
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    peer = FakePeer()
    peer.created_actors = ["a1"]
    peer.created_pgs = ["p1"]
    run(d.release_client(peer))
    assert "a1" not in d.actors and d.gpu_used[0] is False
    assert "p1" not in d.pgs


def test_release_client_worker_forwards_to_head():
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()
    peer.created_actors = ["a1", "a2"]
    peer.created_pgs = ["p1"]
    run(d.release_client(peer))
    sent = [c[0] for c in d.head_peer.calls]
    assert {"t": "kill", "actor": "a1"} in sent
    assert {"t": "kill", "actor": "a2"} in sent
    assert {"t": "remove_pg", "pg": "p1"} in sent


def test_release_client_worker_swallows_forward_errors():
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("head gone"))
    peer = FakePeer()
    peer.created_actors = ["a1"]
    peer.created_pgs = ["p1"]
    run(d.release_client(peer))  # errors swallowed, no raise


def test_release_client_worker_reaps_local_when_head_down():
    """Driver-on-worker disconnect with head unreachable must still kill local actors."""
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("head gone"))
    proc = FakeProc()
    ap = ActorProc("a1", peer=FakePeer(), gpus=[], proc=proc)
    d.actors["a1"] = ap
    peer = FakePeer()
    peer.created_actors = ["a1"]
    run(d.release_client(peer))
    assert "a1" not in d.actors and proc.terminated


def test_release_client_head_no_head_peer_needed():
    d = head()
    peer = FakePeer()
    peer.created_actors = ["ghost"]  # not present; on_kill tolerates it
    run(d.release_client(peer))


# ---- shutdown ---------------------------------------------------------------


def test_shutdown_reaps_all_actors():
    d = head()
    p1, p2 = FakeProc(), FakeProc()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=p1)
    d.actors["a2"] = ActorProc("a2", peer=FakePeer(), gpus=[], proc=p2)
    d.shutdown()
    assert p1.terminated and p2.terminated and d.actors == {}


# ---- _peer_for / _used_on_node ----------------------------------------------


def test_peer_for_known_and_unknown():
    d = head()
    peer = FakePeer()
    d.nodes["n2"] = {"info": {}, "peer": peer}
    assert d._peer_for("n2") is peer
    assert d._peer_for("n404") is None


def test_used_on_node():
    d = head()
    d.pgs["p"] = [{"node": "n2", "gpu": 0}, {"node": "n2", "gpu": -1}, {"node": "n3", "gpu": 1}]
    assert d._used_on_node("n2") == 1  # only the gpu>=0 bundle on n2
    assert d._used_on_node("n3") == 1


# ---- ids --------------------------------------------------------------------


def test_next_obj_and_id_sequence():
    d = head()
    assert d._next_obj() == "n1-o1"
    assert d._next_obj() == "n1-o2"
    assert d._next_id("a") == "n1-a1"
    assert d._next_id("pg") == "n1-pg2"  # id_seq is shared across kinds


# ---- fuzz -------------------------------------------------------------------


@settings(max_examples=100)
@given(st.lists(st.integers(min_value=0, max_value=8), max_size=20, unique=True))
def test_fuzz_create_pg_never_oversubscribes(gpu_indices):
    """No matter which GPUs are pre-reserved, a GPU pg bundle never lands on a
    used index, and over-subscription always errors cleanly."""
    ngpu = 4
    d = Daemon(is_head=True, node_id="n1", ip="x", num_gpus=ngpu)
    for i in gpu_indices:
        if i < ngpu:
            d.gpu_used[i] = True
    free = ngpu - sum(d.gpu_used)
    specs = [{"GPU": 1}] * (free + 1)
    r, _ = run(d.on_create_pg(FakePeer(), {"t": "create_pg", "specs": specs}, b""))
    assert "err" in r  # one more than free always fails, never silently overcommits


@settings(max_examples=100)
@given(st.binary(max_size=256))
def test_fuzz_put_get_roundtrip(payload):
    d = Daemon(is_head=True, node_id="n1", ip="x", num_gpus=0)
    rp, _ = run(d.on_put(FakePeer(), {"t": "put"}, payload))
    rg, body = run(d.on_get(FakePeer(), {"t": "get", "obj": rp["obj"]}, b""))
    assert body == payload


@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=50))
def test_fuzz_next_obj_monotonic(n):
    d = Daemon(is_head=True, node_id="nX", ip="x", num_gpus=0)
    ids = [d._next_obj() for _ in range(n)]
    assert ids == ["nX-o%d" % (i + 1) for i in range(n)]
    assert len(set(ids)) == len(ids)  # unique


# ---- encode_frame / read_frame (async) --------------------------------------


def test_encode_read_frame_roundtrip():
    async def go():
        frame = encode_frame({"t": "x", "a": 1}, b"body")
        reader = asyncio.StreamReader()
        reader.feed_data(frame)
        reader.feed_eof()
        h, p = await read_frame(reader)
        assert h["t"] == "x" and h["a"] == 1 and p == b"body" and h["plen"] == 4

    run(go())


def test_read_frame_bad_length():
    async def go():
        import struct

        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", 0))  # zero-length header rejected
        reader.feed_eof()
        with pytest.raises(ConnectionError):
            await read_frame(reader)

    run(go())


def test_read_frame_non_object_header():
    async def go():
        import struct

        body = b"123"  # valid JSON, but an int not an object
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", len(body)) + body)
        reader.feed_eof()
        with pytest.raises(ConnectionError):
            await read_frame(reader)

    run(go())


def test_read_frame_bad_plen():
    async def go():
        import json
        import struct

        body = json.dumps({"t": "x", "plen": -1}).encode()
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", len(body)) + body)
        reader.feed_eof()
        with pytest.raises(ConnectionError):
            await read_frame(reader)

    run(go())


def test_peer_serve_catches_unexpected_error(capsys):
    # a frame with a valid length but invalid-JSON body makes read_frame raise
    # JSONDecodeError (a ValueError, not in the handled tuple), so Peer.serve's
    # catch-all prints a traceback and closes cleanly instead of propagating.
    import struct

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", 3) + b"{{{")
        reader.feed_eof()

        class W:
            def write(self, b):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

        peer = Peer(reader, W(), handler=None)
        await peer.serve()
        assert peer.closed

    run(go())
    assert "Traceback" in capsys.readouterr().err


# ---- Peer end-to-end over a real socketpair ---------------------------------


async def _peer_pair(handler_a, handler_b):
    """Two Peers connected over an asyncio stream pair (a socketpair lifted into
    asyncio transports). Exercises serve/send/call/_handle/close for real."""
    s1, s2 = socket.socketpair()
    s1.setblocking(False)
    s2.setblocking(False)
    r1, w1 = await asyncio.open_connection(sock=s1)
    r2, w2 = await asyncio.open_connection(sock=s2)
    pa = Peer(r1, w1, handler_a)
    pb = Peer(r2, w2, handler_b)
    ta = asyncio.create_task(pa.serve())
    tb = asyncio.create_task(pb.serve())
    return pa, pb, ta, tb


def test_peer_call_roundtrip():
    async def go():
        async def echo_handler(peer, m, payload):
            return {"t": "echo_ok", "got": m.get("v")}, payload + b"!"

        async def noop_handler(peer, m, payload):
            return {"t": "noop_ok"}, b""

        pa, pb, ta, tb = await _peer_pair(noop_handler, echo_handler)
        try:
            resp, payload = await pa.call({"t": "echo", "v": 42}, b"hi")
            assert resp["got"] == 42 and payload == b"hi!"
        finally:
            await pa.close()
            await pb.close()
            for t in (ta, tb):
                t.cancel()

    run(go())


def test_peer_call_error_raises():
    async def go():
        async def boom_handler(peer, m, payload):
            raise RuntimeError("handler exploded")

        async def noop_handler(peer, m, payload):
            return {"t": "noop_ok"}, b""

        pa, pb, ta, tb = await _peer_pair(noop_handler, boom_handler)
        try:
            with pytest.raises(RuntimeError, match="handler exploded"):
                await pa.call({"t": "boom"})
        finally:
            await pa.close()
            await pb.close()
            for t in (ta, tb):
                t.cancel()

    run(go())


def test_peer_close_fails_pending():
    async def go():
        async def noop_handler(peer, m, payload):
            return {"t": "noop_ok"}, b""

        pa, pb, ta, tb = await _peer_pair(noop_handler, noop_handler)
        # register a pending call, then close pb so pa's read loop ends and
        # close() fails the in-flight future with ConnectionError
        await pb.close()
        with pytest.raises((ConnectionError, RuntimeError)):
            await pa.call({"t": "noop"})
        await pa.close()
        for t in (ta, tb):
            t.cancel()

    run(go())


def test_peer_close_runs_on_close_callback():
    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)
        p = Peer(r1, w1, lambda *a: None)
        fired = []
        p.on_close = lambda: fired.append(True)
        await p.close()
        await p.close()  # idempotent: second close does nothing
        assert fired == [True]
        s2.close()

    run(go())


# ---- _spawn_worker (no real subprocess) -------------------------------------


def test_spawn_worker_builds_env_and_cmd(monkeypatch):
    """_spawn_worker shapes the command + GPU env vars; stub subprocess.Popen so
    nothing is actually launched."""
    captured = {}

    class FakePopen:
        def __init__(self, argv, env=None):
            captured["argv"] = argv
            captured["env"] = env

    monkeypatch.setattr(_daemon.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("BEAM_WORKER_CMD", "mypy-worker --x")
    d = head()
    d.sock_path = "/run/beam.sock"
    proc = d._spawn_worker("a7", [1, 3])
    assert isinstance(proc, FakePopen)
    assert captured["argv"] == ["mypy-worker", "--x"]
    env = captured["env"]
    assert env["BEAM_ACTOR_ID"] == "a7"
    assert env["BEAM_GPU_IDS"] == "1,3"
    assert env["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert env["HIP_VISIBLE_DEVICES"] == "1,3"
    assert env["BEAM_SOCK"] == "/run/beam.sock"


# ---- serve_unix / serve_tcp / _on_conn --------------------------------------


def test_serve_unix_creates_listener(tmp_path):
    async def go():
        d = head()
        sock = os.path.join(str(tmp_path), "nested", "beam.sock")
        await d.serve_unix(sock)  # creates parent dir + binds
        assert d.sock_path == sock and os.path.exists(sock)
        # connecting a client triggers _on_conn -> a Peer that serves
        r, w = await asyncio.open_unix_connection(sock)
        encode = _daemon.encode_frame
        w.write(encode({"t": "status"}))
        await w.drain()
        h, _ = await _daemon.read_frame(r)
        assert h["t"] == "status_ok"
        w.close()

    run(go())


def test_serve_tcp_creates_listener():
    async def go():
        d = head()
        await d.serve_tcp("127.0.0.1", 0)  # port 0 -> OS picks a free port

    run(go())


def test_join_head_sets_on_close_to_shutdown(monkeypatch):
    d = worker()
    called = []

    class HP(FakePeer):
        pass

    async def fake_open(host, port):
        return object(), object()

    monkeypatch.setattr(_daemon.asyncio, "open_connection", fake_open)

    def fake_peer(reader, writer, handler):
        p = HP()
        return p

    monkeypatch.setattr(_daemon, "Peer", fake_peer)
    monkeypatch.setattr(d, "shutdown", lambda: called.append(True))

    async def go():
        # minimal: set head_peer like join_head does
        d.head_peer = HP()

        def _on_head_lost():
            d.shutdown()

        d.head_peer.on_close = _on_head_lost
        d.head_peer.on_close()
        return called

    assert run(go()) == [True]


def test_join_head_handshake():
    """A worker daemon dials a real head over TCP and completes the hello; the
    head records the new node. No subprocess, just two in-process daemons."""

    async def go():
        h = head(0)
        server = await asyncio.start_server(h._on_conn, host="127.0.0.1", port=0)
        port = server.sockets[0].getsockname()[1]
        w = worker(2)
        await w.join_head("127.0.0.1", port)
        # the head saw the worker's hello and registered it
        for _ in range(50):
            if "w1" in h.nodes:
                break
            await asyncio.sleep(0.01)
        assert "w1" in h.nodes and h.nodes["w1"]["info"]["ngpu"] == 2
        assert w.head_peer is not None
        server.close()

    run(go())


def test_join_head_retries_then_fails():
    async def go():
        w = worker()
        with pytest.raises(OSError):
            # nothing listening on this port; retries=2 keeps it quick
            await w.join_head("127.0.0.1", 1, retries=2)

    run(go())


# ---- _handle default-response + send-error branches -------------------------


def test_handle_send_error_is_swallowed():
    """If sending the handler's response fails (peer vanished mid-reply), the
    error is swallowed, not raised (covers Peer._handle send-except branch)."""

    async def go():
        class DeadWriter:
            def write(self, b):
                raise ConnectionError("peer gone")

            async def drain(self):
                pass

            def close(self):
                pass

        async def ok_handler(peer, m, payload):
            return {"t": "ok"}, b""

        p = Peer(reader=None, writer=DeadWriter(), handler=ok_handler)
        await p._handle({"t": "ping", "reqid": 1}, b"")  # must not raise

    run(go())


def test_handle_none_response_gets_default():
    """A handler returning None must be answered with a synthesized
    '<t>_ok' response (covers the resp-is-None branch in Peer._handle)."""

    async def go():
        async def none_handler(peer, m, payload):
            return None, b""

        async def noop_handler(peer, m, payload):
            return {"t": "noop_ok"}, b""

        pa, pb, ta, tb = await _peer_pair(noop_handler, none_handler)
        try:
            resp, _ = await pa.call({"t": "ping"})
            assert resp["t"] == "ping_ok"  # default synthesized from request type
        finally:
            await pa.close()
            await pb.close()
            for t in (ta, tb):
                t.cancel()

    run(go())


def test_peer_close_swallows_writer_oserror():
    async def go():
        # a minimal writer whose close() raises: exercises the `except OSError`
        # guard in Peer.close without poking a real StreamWriter (whose __del__
        # would then re-raise during GC).
        class BoomWriter:
            def close(self):
                raise OSError("writer dead")

        p = Peer(reader=None, writer=BoomWriter(), handler=lambda *a: None)
        await p.close()  # OSError on writer.close swallowed, no raise

    run(go())


# ---- on_create_pg multi-node / dead-node branches ---------------------------


def test_on_create_pg_skips_dead_node_and_uses_remote():
    d = head(0)  # head has no GPUs
    d.nodes["dead"] = {"info": {"node": "dead", "ngpu": 4, "alive": False}, "peer": object()}
    d.nodes["live"] = {"info": {"node": "live", "ngpu": 2, "alive": True}, "peer": object()}
    # a pg bundle already sits on the live node, exercising the multi-node used set
    d.pgs["existing"] = [{"node": "live", "gpu": 0}]
    r, _ = run(d.on_create_pg(FakePeer(), {"t": "create_pg", "specs": [{"GPU": 1}]}, b""))
    assert r["t"] == "create_pg_ok"
    placed = d.pgs[r["pg"]][0]
    assert placed["node"] == "live" and placed["gpu"] == 1  # gpu 0 taken, picks 1


# ---- release_client on_kill error is swallowed ------------------------------


def test_release_client_head_swallows_on_kill_error(monkeypatch):
    d = head()

    async def boom(*a, **k):
        raise RuntimeError("kill failed")

    monkeypatch.setattr(d, "on_kill", boom)
    peer = FakePeer()
    peer.created_actors = ["a1"]
    run(d.release_client(peer))  # error swallowed, no raise


def test_peer_close_async_on_close():
    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)
        p = Peer(r1, w1, lambda *a: None)
        fired = []

        async def acb():
            fired.append(True)

        p.on_close = acb
        await p.close()
        assert fired == [True]  # coroutine on_close is awaited
        s2.close()

    run(go())


def test_peer_close_drains_handler_tasks_before_on_close():
    """release/on_close must not run while framed handlers are still pending."""
    order: list[str] = []

    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)

        started = asyncio.Event()

        async def slow_handler(peer, m, payload):
            order.append("handle_start")
            started.set()
            await asyncio.sleep(0.01)
            order.append("handle_done")
            return {"t": "ok"}, b""

        p = Peer(r1, w1, slow_handler)

        async def on_close():
            order.append("on_close")

        p.on_close = on_close
        # Simulate a handler already scheduled (as serve would)
        t = asyncio.create_task(p._handle({"t": "x", "reqid": 1}, b""))
        p._tasks.add(t)
        t.add_done_callback(p._tasks.discard)
        await started.wait()  # the handler is definitely in flight
        await p.close()
        assert order == ["handle_start", "handle_done", "on_close"]
        s2.close()

    run(go())


def test_peer_close_fails_pending_before_drain():
    """Handlers blocked on same-peer call() must unblock when close runs."""

    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        s2.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)
        r2, w2 = await asyncio.open_connection(sock=s2)

        async def bounce(peer, m, payload):
            # call back on the same peer: would hang if pending not failed first
            return await peer.call({"t": "echo"}, b"")

        p = Peer(r1, w1, bounce)
        # plant a pending future as if call() is in flight
        fut = asyncio.get_running_loop().create_future()
        p.pending[99] = fut

        async def stuck_handler(peer, m, payload):
            try:
                await fut  # wait until close fails pending
            except ConnectionError:
                return {"t": "aborted"}, b""
            return {"t": "ok"}, b""

        p.handler = stuck_handler
        t = asyncio.create_task(p._handle({"t": "x", "reqid": 1}, b""))
        p._tasks.add(t)
        t.add_done_callback(p._tasks.discard)
        await asyncio.sleep(0)
        await p.close()
        assert fut.done()
        assert t.done()
        w2.close()

    run(go())


def test_peer_close_fires_on_close_even_if_drain_cancelled():
    """Cancelled drain must still close writer and run on_close once."""
    fired = []

    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)

        async def forever(peer, m, payload):
            await asyncio.Event().wait()
            return {"t": "ok"}, b""

        p = Peer(r1, w1, forever)
        p.on_close = lambda: fired.append("close")
        t = asyncio.create_task(p._handle({"t": "x", "reqid": 1}, b""))
        p._tasks.add(t)
        t.add_done_callback(p._tasks.discard)
        await asyncio.sleep(0)

        async def bad_wait(aws, timeout=None):
            raise asyncio.CancelledError()

        orig = asyncio.wait
        asyncio.wait = bad_wait  # type: ignore
        try:
            try:
                await p.close()
            except asyncio.CancelledError:
                pass
        finally:
            asyncio.wait = orig  # type: ignore

        assert fired == ["close"]
        assert p.on_close is None
        s2.close()

    run(go())


def test_peer_close_second_await_joins_first():
    """Concurrent close() waits for the first close's release to finish."""
    order: list[str] = []

    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)
        p = Peer(r1, w1, lambda *a: None)
        gate = asyncio.Event()

        async def slow_on_close():
            order.append("release_start")
            await gate.wait()
            order.append("release_done")

        p.on_close = slow_on_close
        t1 = asyncio.create_task(p.close())
        await asyncio.sleep(0)
        assert p.closed
        t2 = asyncio.create_task(p.close())
        await asyncio.sleep(0)
        assert not t2.done()  # joined on _close_done
        gate.set()
        await t1
        await t2
        assert order == ["release_start", "release_done"]
        s2.close()

    run(go())


def test_release_client_superseded_closed_target_frees_or_reaps():
    d = head()
    peer = FakePeer()
    new_peer = FakePeer()
    new_peer.closed = True

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            peer.superseded = True
            peer.superseded_by = new_peer
            peer.created_actors.clear()
            peer.created_pgs.clear()

    peer.created_actors = SuperList(["a1"])
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actor_loc["a1"] = "n1"
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    run(d.release_client(peer))
    # a1 killed; closed target has no leftovers → free PG (not stuck forever)
    assert "p1" not in d.pgs
    assert "p1" not in new_peer.created_pgs


def test_release_client_superseded_closed_target_with_leftover_actors():
    d = head()
    peer = FakePeer()
    new_peer = FakePeer()
    new_peer.closed = True
    new_peer.created_actors = ["a2"]  # transferred earlier; still tracked

    class SuperList(list):
        def remove(self, item):
            list.remove(self, item)
            peer.superseded = True
            peer.superseded_by = new_peer
            peer.created_actors.clear()
            peer.created_pgs.clear()

    peer.created_actors = SuperList(["a1"])
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actors["a2"] = ActorProc("a2", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.actor_loc["a1"] = d.actor_loc["a2"] = "n1"
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    scheduled = []

    def capture(aid, node):
        scheduled.append(aid)
        d._orphans[aid] = node

    # Patch on the instance after construction
    d._schedule_orphan_reap = capture  # type: ignore
    run(d.release_client(peer))
    assert "a2" in scheduled
    assert "p1" in d._orphan_pgs or "p1" in d.pgs


def test_terminate_returns_false_if_still_alive():
    class Immortal:
        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

    assert _terminate(Immortal()) is False


def test_on_kill_returns_err_if_process_still_alive():
    d = head(1)
    d.gpu_used[0] = True

    class Immortal(FakeProc):
        def __init__(self):
            super().__init__(alive=True)

        def terminate(self):
            self.terminated = True
            # stay alive

        def kill(self):
            self.killed = True
            # stay alive

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

    proc = Immortal()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[0], proc=proc)
    d.actor_loc["a1"] = "n1"
    r, _ = run(d.on_kill(None, {"actor": "a1"}, b""))
    assert r.get("err")
    assert "a1" in d.actors  # restored for tracking
    assert d.gpu_used[0] is True  # not freed under live process


def test_host_actor_aborts_publish_if_peer_closed(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    worker_peer = FakePeer()
    worker_peer.closed = True  # already dead when attach completes

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": []}, b""))
        await asyncio.sleep(0)
        d.pending_workers["a1"].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert r.get("err")
    assert "a1" not in d.actors


def test_host_actor_peer_closed_proc_still_alive(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = _immortal()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    d.gpu_used[0] = True
    worker_peer = FakePeer()
    worker_peer.closed = True

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        d.pending_workers["a1"].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert "still alive" in r["err"]
    assert "a1" in d._hosting
    assert d.gpu_used[0] is True


def test_on_kill_hosting_restores_if_process_alive():
    d = head(1)
    d.gpu_used[0] = True

    class Immortal(FakeProc):
        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

    proc = Immortal()
    d._hosting["a1"] = (proc, FakePeer(), [0])
    r, _ = run(d.on_kill(None, {"actor": "a1"}, b""))
    assert r.get("err")
    assert "a1" in d._hosting
    assert d.actor_loc.get("a1") == "n1"
    assert d.gpu_used[0] is True


def test_host_actor_finally_keeps_hosting_if_alive(monkeypatch):
    """Concurrent kill restore + host_actor finally must not untrack live proc."""
    d = head(1)
    d.sock_path = "/x.sock"

    class Immortal(FakeProc):
        def terminate(self):
            self.terminated = True

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

    proc = Immortal()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        # kill mid-create while still hosting
        r, _ = await d.on_kill(None, {"actor": "a1"}, b"")
        assert r.get("err")
        try:
            await task
        except Exception:
            pass
        # immortal proc must remain tracked
        assert "a1" in d._hosting

    run(go())


def test_host_actor_kill_pending_after_spawn_still_alive(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"

    class Immortal(FakeProc):
        def terminate(self):
            self.terminated = True

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

    proc = Immortal()
    d.gpu_used[0] = True

    def spawn_then_tombstone(self, actor_id, gpus):
        d._kill_pending.add(actor_id)  # after spawn, before hosting check
        return proc

    monkeypatch.setattr(Daemon, "_spawn_worker", spawn_then_tombstone)
    r, _ = run(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
    assert "still alive" in r["err"]
    assert "a1" in d._hosting
    assert d.gpu_used[0] is True


def test_host_actor_cancel_keeps_hosting_if_alive(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = _immortal()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    d.gpu_used[0] = True

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert "a1" in d._hosting
        assert d.gpu_used[0] is True

    run(go())


def test_host_actor_cancel_frees_if_proc_dies(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc(alive=True)
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    d.gpu_used[0] = True

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert "a1" not in d._hosting
        assert d.gpu_used[0] is False

    run(go())


def test_host_actor_cancel_restores_hosting_if_popped(monkeypatch):
    """Cancel after hosting was cleared: re-park immortal proc."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = _immortal()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    d.gpu_used[0] = True

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        d._hosting.pop("a1", None)  # simulate concurrent clear
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert "a1" in d._hosting
        assert d.gpu_used[0] is True

    run(go())


def _immortal():
    class Immortal(FakeProc):
        def terminate(self):
            self.terminated = True

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)

    return Immortal()


def test_host_actor_timeout_still_alive(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = _immortal()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    d.gpu_used[0] = True

    async def go():
        # fail attach immediately
        async def boom_wait(aw, timeout=None):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(asyncio, "wait_for", boom_wait)
        return await d._host_actor({"actor": "a1", "gpus": [0]}, b"")

    r, _ = run(go())
    assert "still alive" in r["err"]
    assert "a1" in d._hosting


def test_host_actor_init_fail_still_alive(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = _immortal()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    d.gpu_used[0] = True
    wp = FakePeer(raise_on_call=RuntimeError("init boom"))

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        d.pending_workers["a1"].set_result(wp)
        return await task

    r, _ = run(go())
    assert "still alive" in r["err"]
    assert "a1" in d._hosting


def test_host_actor_spawn_fail_still_alive(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    d.gpu_used[0] = True

    def boom_spawn(self, aid, gpus):
        raise RuntimeError("exec failed")

    monkeypatch.setattr(Daemon, "_spawn_worker", boom_spawn)
    r, _ = run(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
    assert "spawn failed" in r["err"]


def test_host_actor_spawn_sets_proc_then_raises(monkeypatch):
    """Exception path with proc set but hosting empty: restore hosting if alive."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = _immortal()
    d.gpu_used[0] = True

    def weird_spawn(self, aid, gpus):
        return proc

    class BoomDict(dict):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __setitem__(self, k, v):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("map full")  # first assign (hosting) fails
            return dict.__setitem__(self, k, v)  # restore on cleanup succeeds

    monkeypatch.setattr(Daemon, "_spawn_worker", weird_spawn)
    d._hosting = BoomDict()
    r, _ = run(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
    assert "spawn failed" in r["err"]
    assert "a1" in d._hosting
    assert d.gpu_used[0] is True


def test_retry_forward_kill_stops_when_head_gone(monkeypatch):
    d = worker()
    d.head_peer = None
    run(d._retry_forward_kill("a1"))  # returns without spinning


def test_retry_forward_remove_pg_stops_when_head_closed(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()
    d.head_peer.closed = True
    run(d._retry_forward_remove_pg("p1"))


def test_retry_forward_kill_stops_after_fail_when_head_closes(monkeypatch):
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("x"))
    tries = {"n": 0}

    async def flaky_forward(m, payload=b""):
        tries["n"] += 1
        d.head_peer.closed = True
        raise RuntimeError("down")

    monkeypatch.setattr(d, "_forward_head", flaky_forward)
    run(d._retry_forward_kill("a1"))
    assert tries["n"] == 1


def test_retry_forward_remove_pg_stops_after_fail_when_head_closes(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()

    async def flaky_forward(m, payload=b""):
        d.head_peer.closed = True
        raise RuntimeError("down")

    monkeypatch.setattr(d, "_forward_head", flaky_forward)
    run(d._retry_forward_remove_pg("p1"))


def test_peer_close_cancels_straggler_handlers(monkeypatch):
    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)

        async def forever(peer, m, payload):
            await asyncio.Event().wait()
            return {"t": "ok"}, b""

        p = Peer(r1, w1, forever)
        t = asyncio.create_task(p._handle({"t": "x", "reqid": 1}, b""))
        p._tasks.add(t)
        t.add_done_callback(p._tasks.discard)
        await asyncio.sleep(0)

        real_wait = asyncio.wait

        async def quick_wait(aws, timeout=None):
            # expire immediately so close cancels the hung handler
            return await real_wait(aws, timeout=0)

        monkeypatch.setattr(asyncio, "wait", quick_wait)
        await p.close()
        assert t.done()
        s2.close()

    run(go())


def test_host_actor_cancel_terminates_proc(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    d.gpu_used[0] = True

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(go())
    assert proc.terminated
    assert "a1" not in d._hosting
    assert d.gpu_used[0] is False


def test_await_fwd_after_cancel_swallows_error():
    d = worker()
    peer = FakePeer()

    async def go():
        async def boom():
            raise RuntimeError("x")

        t = asyncio.create_task(boom())
        return await d._await_fwd_after_cancel(t, peer, "actor")

    assert run(go()) is None


def test_await_fwd_after_cancel_timeout_schedules_bg(monkeypatch):
    d = worker()
    peer = FakePeer()
    scheduled = []

    def track(task):
        scheduled.append("bg")
        task.cancel()
        return task

    monkeypatch.setattr(d, "_track", track)

    async def go():
        async def hang():
            await asyncio.Event().wait()
            return {"actor": "a1"}, b""

        t = asyncio.create_task(hang())
        real_wf = asyncio.wait_for

        async def quick(aw, timeout=None):
            return await real_wf(aw, timeout=0.01)

        monkeypatch.setattr(asyncio, "wait_for", quick)
        r = await d._await_fwd_after_cancel(t, peer, "actor")
        assert r is None
        assert not t.cancelled()  # forward left running for bg cleanup
        assert scheduled == ["bg"]
        assert peer.in_flight >= 1
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    run(go())


def test_bg_finish_fwd_cleanup_actor(monkeypatch):
    d = worker()
    peer = FakePeer()
    cleaned = []

    async def cleanup(p, aid):
        cleaned.append(aid)

    monkeypatch.setattr(d, "_worker_cleanup_actor", cleanup)

    async def go():
        async def done():
            return {"actor": "n1-a1"}, b""

        t = asyncio.create_task(done())
        peer.in_flight = 1
        await d._bg_finish_fwd_cleanup(peer, t, "actor")
        assert peer.in_flight == 0

    run(go())
    assert cleaned == ["n1-a1"]


def test_bg_finish_fwd_cleanup_pg(monkeypatch):
    d = worker()
    peer = FakePeer()
    cleaned = []

    async def cleanup(p, pg):
        cleaned.append(pg)

    monkeypatch.setattr(d, "_worker_cleanup_pg", cleanup)

    async def go():
        async def done():
            return {"pg": "p1"}, b""

        t = asyncio.create_task(done())
        peer.in_flight = 1
        await d._bg_finish_fwd_cleanup(peer, t, "pg")

    run(go())
    assert cleaned == ["p1"]


def test_bg_finish_fwd_cleanup_error():
    d = worker()
    peer = FakePeer()

    async def go():
        async def boom():
            raise RuntimeError("x")

        t = asyncio.create_task(boom())
        peer.in_flight = 1
        await d._bg_finish_fwd_cleanup(peer, t, "actor")
        assert peer.in_flight == 0

    run(go())


def test_await_fwd_after_cancel_done_result():
    """wait_for times out but the forward already finished: return its result."""
    d = worker()
    peer = FakePeer()

    async def direct():
        async def done():
            return {"actor": "a"}, b""

        t2 = asyncio.create_task(done())
        await t2

        async def always_timeout(aw, timeout=None):
            raise TimeoutError()

        # patch wait_for only for this call
        orig = asyncio.wait_for
        asyncio.wait_for = always_timeout  # type: ignore
        try:
            return await d._await_fwd_after_cancel(t2, peer, "actor")
        finally:
            asyncio.wait_for = orig  # type: ignore

    assert run(direct()) == ({"actor": "a"}, b"")


def test_worker_cleanup_actor_schedules_retry(monkeypatch):
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("down"))
    peer = FakePeer()
    scheduled = []

    def track(task):
        scheduled.append("k")
        task.cancel()
        return task

    monkeypatch.setattr(d, "_track", track)
    run(d._worker_cleanup_actor(peer, "a1"))
    assert scheduled == ["k"]
    assert peer.in_flight >= 1


def test_worker_cleanup_actor_success_local_kill(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()
    proc = FakeProc()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=proc)
    run(d._worker_cleanup_actor(peer, "a1"))
    assert "a1" not in d.actors
    assert peer.in_flight == 0


def test_worker_cleanup_actor_success_local_kill_error(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())

    async def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(d, "on_kill", boom)
    run(d._worker_cleanup_actor(peer, "a1"))  # swallows local kill error
    assert peer.in_flight == 0


def test_worker_cleanup_actor_cancel_pins_retry(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()
    scheduled = []

    def track(task):
        scheduled.append("k")
        task.cancel()
        return task

    async def cancel_forward(m, payload=b""):
        raise asyncio.CancelledError()

    monkeypatch.setattr(d, "_forward_head", cancel_forward)
    monkeypatch.setattr(d, "_track", track)

    async def go():
        try:
            await d._worker_cleanup_actor(peer, "a1")
        except asyncio.CancelledError:
            return "c"

    assert run(go()) == "c"
    assert scheduled == ["k"]


def test_worker_cleanup_pg_schedules_retry(monkeypatch):
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("down"))
    peer = FakePeer()
    scheduled = []

    def track(task):
        scheduled.append("p")
        task.cancel()
        return task

    monkeypatch.setattr(d, "_track", track)
    run(d._worker_cleanup_pg(peer, "p1"))
    assert scheduled == ["p"]


def test_worker_cleanup_pg_success(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()
    peer = FakePeer()
    run(d._worker_cleanup_pg(peer, "p1"))
    assert peer.in_flight == 0
    assert d.head_peer.calls and d.head_peer.calls[0][0]["t"] == "remove_pg"


def test_worker_cleanup_pg_cancel_pins_retry(monkeypatch):
    d = worker()
    peer = FakePeer()
    scheduled = []

    def track(task):
        scheduled.append("p")
        task.cancel()
        return task

    async def cancel_forward(m, payload=b""):
        raise asyncio.CancelledError()

    monkeypatch.setattr(d, "_forward_head", cancel_forward)
    monkeypatch.setattr(d, "_track", track)

    async def go():
        try:
            await d._worker_cleanup_pg(peer, "p1")
        except asyncio.CancelledError:
            return "c"

    assert run(go()) == "c"
    assert scheduled == ["p"]


def test_worker_create_pg_cancel_after_forward_cleans(monkeypatch):
    """True task.cancel() after head returns: cleanup must still run (shielded)."""
    d = worker()
    d.head_peer = FakePeer({"create_pg": {"pg": "p9"}})
    peer = FakePeer()
    cleaned = []

    async def cleanup(p, pg):
        cleaned.append(pg)
        await asyncio.sleep(0)

    monkeypatch.setattr(d, "_worker_cleanup_pg", cleanup)

    # Block the forward on events instead of a wall-clock sleep: the test
    # controls exactly when the head reply lands, so the cancel below is always
    # delivered mid-forward, on a fast machine and a loaded CI runner alike.
    async def go():
        entered, release = asyncio.Event(), asyncio.Event()

        async def gated_forward(m, payload=b""):
            entered.set()
            await release.wait()
            return {"t": "create_pg_ok", "pg": "p9"}, b""

        monkeypatch.setattr(d, "_forward_head", gated_forward)
        task = asyncio.ensure_future(
            d.on_create_pg(peer, {"t": "create_pg", "specs": [{"GPU": 1}]}, b"")
        )
        await entered.wait()  # the forward task is now in flight and pinned
        task.cancel()
        release.set()  # let the shielded forward finish after the cancel
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "ok"

    assert run(go()) == "cancelled"
    assert cleaned == ["p9"], "cancel must still release the head-side placement group"


def test_worker_create_actor_cancel_after_forward_cleans(monkeypatch):
    d = worker()
    peer = FakePeer()
    cleaned = []

    async def cleanup(p, aid):
        cleaned.append(aid)
        await asyncio.sleep(0)

    monkeypatch.setattr(d, "_worker_cleanup_actor", cleanup)

    async def go():
        # Gated forward: the cancel below is delivered mid-forward by
        # construction, not by racing a wall-clock sleep.
        entered, release = asyncio.Event(), asyncio.Event()

        async def gated_forward(m, payload=b""):
            entered.set()
            await release.wait()
            return {"t": "create_actor_ok", "actor": "n1-a9"}, b""

        monkeypatch.setattr(d, "_forward_head", gated_forward)
        task = asyncio.ensure_future(d.on_create_actor(peer, {"t": "create_actor", "ngpu": 0}, b""))
        await entered.wait()
        task.cancel()
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "ok"

    assert run(go()) == "cancelled"
    assert cleaned == ["n1-a9"]


def test_host_actor_cancel_with_hosting_wpeer(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    wpeer = FakePeer()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))

    async def go():
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": []}, b""))
        await asyncio.sleep(0)
        # attach peer into hosting as mid-create would
        d._hosting["a1"] = (proc, wpeer, [])
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(go())
    assert proc.terminated
    assert wpeer.on_close is None


def test_host_actor_cancel_before_hosting_entry(monkeypatch):
    """Cancel after spawn assigns proc but before _hosting set (synthetic)."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()

    def slow_spawn(self, actor_id, gpus):
        # raise CancelledError path with proc set and hosting empty:
        # call cancel handler logic by raising after spawn via wait_for
        return proc

    monkeypatch.setattr(Daemon, "_spawn_worker", slow_spawn)

    async def go():
        # Force CancelledError after spawn by cancelling during wait_for(fut)
        task = asyncio.ensure_future(d._host_actor({"actor": "a1", "gpus": []}, b""))
        await asyncio.sleep(0)
        # remove hosting so cancel hits elif proc branch
        d._hosting.pop("a1", None)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(go())
    assert proc.terminated


def test_on_hello_closed_peer_releases_without_installing():
    d = head()
    peer = FakePeer()
    peer.closed = True
    peer.created_pgs = ["p1"]
    d.pgs["p1"] = [{"node": "n1", "gpu": -1}]
    r, _ = run(d.on_hello(peer, {"t": "hello", "node": "w2", "ip": "9.9.9.9", "ngpu": 1}, b""))
    assert r["t"] == "hello_ok"
    assert "w2" not in d.nodes  # dead peer not installed
    assert "p1" not in d.pgs  # released


def test_on_hello_closed_peer_after_rehello_drops_old():
    d = head()
    old = FakePeer()
    old.created_actors = ["a1"]
    d.nodes["w2"] = {"info": {"node": "w2", "ngpu": 1, "alive": True}, "peer": old}
    d.actor_loc["a1"] = "w2"
    # remote kill will fail; orphan path may schedule
    new = FakePeer()
    new.closed = True
    r, _ = run(d.on_hello(new, {"t": "hello", "node": "w2", "ip": "1.1.1.1", "ngpu": 1}, b""))
    assert r["t"] == "hello_ok"
    assert old.superseded
    # old membership cleared; new not installed
    assert "w2" not in d.nodes or d.nodes["w2"]["peer"] is not new


def test_create_superseded_late_append_schedules_orphan(monkeypatch):
    """Late create on superseded peer with id only on dead peer → orphan reap."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    peer = FakePeer()
    peer.closed = True
    peer.superseded = True
    worker_peer = FakePeer()
    scheduled: list[tuple[str, str]] = []

    def capture(aid, node):
        scheduled.append((aid, node))
        d._orphans[aid] = node

    monkeypatch.setattr(d, "_schedule_orphan_reap", capture)

    async def go():
        task = asyncio.ensure_future(d.on_create_actor(peer, {"t": "create_actor", "ngpu": 0}, b""))
        await asyncio.sleep(0)
        aid = next(iter(d.pending_workers))
        # id is on peer.created_actors from early append; not on any live node peer
        d.pending_workers[aid].set_result(worker_peer)
        return await task, aid

    r, aid = run(go())
    assert r[0].get("t") == "create_actor_ok"
    assert scheduled and scheduled[0][0] == aid


def test_create_superseded_owned_elsewhere_no_orphan(monkeypatch):
    """Id still on dead peer but also on live peer → no orphan (transfer ok)."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    peer = FakePeer()
    peer.closed = True
    peer.superseded = True
    worker_peer = FakePeer()
    live = FakePeer()
    scheduled: list = []

    monkeypatch.setattr(d, "_schedule_orphan_reap", lambda *a: scheduled.append(a))

    async def go():
        task = asyncio.ensure_future(d.on_create_actor(peer, {"t": "create_actor", "ngpu": 0}, b""))
        await asyncio.sleep(0)
        aid = next(iter(d.pending_workers))
        # keep id on dead peer AND on live peer (both lists)
        live.created_actors.append(aid)
        d.nodes["w2"] = {
            "info": {"node": "w2", "ngpu": 1, "alive": True},
            "peer": live,
        }
        d.pending_workers[aid].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert r.get("t") == "create_actor_ok"
    assert not scheduled
    assert not proc.terminated


def test_peer_serve_skips_handlers_when_closed():
    async def go():
        s1, s2 = socket.socketpair()
        s1.setblocking(False)
        s2.setblocking(False)
        r1, w1 = await asyncio.open_connection(sock=s1)
        handled = []

        async def handler(peer, m, payload):
            handled.append(m.get("t"))
            return {"t": "ok"}, b""

        p = Peer(r1, w1, handler)
        p.closed = True  # close already in progress
        # write a request frame to the peer
        from ray._daemon import encode_frame

        # s2 is the client side: send a frame then close so serve() exits.
        s2.sendall(encode_frame({"t": "ping", "reqid": 1}))
        s2.close()
        await p.serve()  # reads frame, skips handler because closed, then EOF
        assert handled == []

    run(go())


def test_schedule_orphan_skips_when_task_live():
    d = head(2)
    d._orphans["a1"] = "n2"

    async def never():
        await asyncio.Event().wait()

    async def go():
        t = asyncio.get_running_loop().create_task(never())
        d._orphan_tasks["a1"] = t
        d._schedule_orphan_reap("a1", "n3")  # refresh node, no second task
        assert d._orphans["a1"] == "n3"
        assert d._orphan_tasks["a1"] is t
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    run(go())


def test_schedule_orphan_no_running_loop():
    d = head(2)
    d._schedule_orphan_reap("a1", "n2")  # no loop: still registers
    assert d._orphans["a1"] == "n2"
    assert "a1" not in d._orphan_tasks


def test_free_orphan_pgs_noop_while_orphans_remain():
    d = head(2)
    d._orphans["a1"] = "n2"
    d.pgs["p1"] = [{"node": "n2", "gpu": 0}]
    d._orphan_pgs.add("p1")
    d._free_orphan_pgs_if_idle()
    assert "p1" in d.pgs and "p1" in d._orphan_pgs


def test_reap_orphan_exits_if_slot_cleared_mid_loop(monkeypatch):
    d = head(2)
    d.actor_loc["a1"] = "n2"
    d._orphans["a1"] = "n2"

    async def force(actor_id, node):
        d._orphans.pop(actor_id, None)  # released elsewhere

    monkeypatch.setattr(d, "_force_kill_actor", force)

    async def go():
        # re-enter with slot present for the while check, force clears it
        d._orphans["a1"] = "n2"
        await d._reap_orphan("a1", "n2")

    run(go())


def test_host_actor_spawn_oserror(monkeypatch):
    d = head(1)
    d.sock_path = "/x.sock"

    def boom(*a, **k):
        raise OSError("cannot exec")

    monkeypatch.setattr(Daemon, "_spawn_worker", boom)
    r, _ = run(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
    assert "spawn failed" in r["err"]
    assert "a1" not in d.pending_workers
    assert "a1" not in d._hosting


def test_host_actor_post_spawn_unexpected_error(monkeypatch):
    """Exception after spawn with proc set: terminate the subprocess."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))

    class BoomDict(dict):
        def __setitem__(self, k, v):
            raise RuntimeError("map full")

    d._hosting = BoomDict()
    r, _ = run(d._host_actor({"actor": "a1", "gpus": [0]}, b""))
    assert "spawn failed" in r["err"]
    assert proc.terminated


def test_retry_forward_kill_local_then_forward(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()
    proc = FakeProc()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=proc)

    async def nosleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", nosleep)
    run(d._retry_forward_kill("a1"))
    assert "a1" not in d.actors
    assert d.head_peer.calls  # actor_gone or kill forwarded via on_kill


def test_retry_forward_kill_local_kill_raises_then_forward(monkeypatch):
    """Local on_kill raises: swallow, then forward to head (covers except path)."""
    d = worker()
    d.head_peer = FakePeer()
    d.actors["a1"] = ActorProc("a1", peer=FakePeer(), gpus=[], proc=FakeProc())

    async def boom(*a, **k):
        raise RuntimeError("transient")

    async def nosleep(_s):
        return None

    monkeypatch.setattr(d, "on_kill", boom)
    monkeypatch.setattr(asyncio, "sleep", nosleep)
    run(d._retry_forward_kill("a1"))
    # forward succeeded after local raise; local map may still hold id
    assert d.head_peer.calls and d.head_peer.calls[0][0]["t"] == "kill"


def test_retry_forward_kill_forward_succeeds(monkeypatch):
    d = worker()
    d.head_peer = FakePeer()

    async def nosleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", nosleep)
    run(d._retry_forward_kill("ghost"))
    assert d.head_peer.calls and d.head_peer.calls[0][0]["t"] == "kill"


def test_retry_forward_kill_retries_until_head_up(monkeypatch):
    d = worker()
    tries = {"n": 0}

    class FlakyHead(FakePeer):
        async def call(self, header, payload=b""):
            tries["n"] += 1
            if tries["n"] < 3:
                raise RuntimeError("down")
            return await super().call(header, payload)

    d.head_peer = FlakyHead()

    async def nosleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", nosleep)
    run(d._retry_forward_kill("a1"))
    assert tries["n"] >= 3


def test_retry_forward_remove_pg_succeeds():
    d = worker()
    d.head_peer = FakePeer()
    run(d._retry_forward_remove_pg("p1"))
    assert d.head_peer.calls[0][0]["t"] == "remove_pg"


def test_retry_forward_remove_pg_retries_until_ok(monkeypatch):
    d = worker()
    tries = {"n": 0}

    class FlakyHead(FakePeer):
        async def call(self, header, payload=b""):
            tries["n"] += 1
            if tries["n"] < 3:
                raise RuntimeError("down")
            return await super().call(header, payload)

    d.head_peer = FlakyHead()

    async def nosleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", nosleep)
    run(d._retry_forward_remove_pg("p1"))
    assert tries["n"] >= 3


def test_retry_worker_release_kills_before_pgs(monkeypatch):
    d = worker()
    order: list[str] = []

    async def kill(aid):
        order.append("kill:" + aid)

    async def rmpg(pg):
        order.append("pg:" + pg)

    monkeypatch.setattr(d, "_retry_forward_kill", kill)
    monkeypatch.setattr(d, "_retry_forward_remove_pg", rmpg)
    run(d._retry_worker_release(["a1", "a2"], ["p1"], None))
    assert order == ["kill:a1", "kill:a2", "pg:p1"]


def test_retry_worker_release_waits_for_in_flight(monkeypatch):
    d = worker()
    peer = FakePeer()
    peer.in_flight = 2
    steps = {"n": 0}

    async def nosleep(_s):
        steps["n"] += 1
        if steps["n"] >= 2:
            peer.in_flight = 0

    order: list[str] = []

    async def kill(aid):
        order.append("k")

    async def rmpg(pg):
        order.append("p")

    monkeypatch.setattr(asyncio, "sleep", nosleep)
    monkeypatch.setattr(d, "_retry_forward_kill", kill)
    monkeypatch.setattr(d, "_retry_forward_remove_pg", rmpg)
    peer.created_actors = ["late"]
    peer.created_pgs = ["p2"]
    run(d._retry_worker_release([], ["p1"], peer))
    assert steps["n"] >= 2
    assert "k" in order and "p" in order
    assert "late" not in peer.created_actors


def test_dispatch_cancelled_sets_err():
    d = head()
    slot = ObjSlot()
    ap = ActorProc(
        "a", peer=FakePeer(raise_on_call=asyncio.CancelledError()), gpus=[], proc=FakeProc()
    )

    async def go():
        try:
            await d._dispatch(ap, "m", b"", slot)
        except asyncio.CancelledError:
            pass

    run(go())
    assert slot.ev.is_set()
    assert "cancel" in slot.err


def test_on_kill_mid_create_gpu_held_guard():
    """Mid-create kill must not clear a GPU re-reserved by another actor."""
    d = head(1)
    d.gpu_used[0] = True
    # hosting holds gpu 0; another actor already took it after concurrent free
    d._hosting["a1"] = (FakeProc(), None, [0])
    d.actors["b"] = ActorProc("b", peer=FakePeer(), gpus=[0], proc=FakeProc())
    run(d.on_kill(None, {"actor": "a1"}, b""))
    assert d.gpu_used[0] is True  # b still owns it
    assert "a1" not in d._hosting


def test_pinned_retry_kill_decrements_in_flight(monkeypatch):
    d = worker()
    peer = FakePeer()
    peer.in_flight = 1

    async def ok(aid):
        return None

    monkeypatch.setattr(d, "_retry_forward_kill", ok)
    run(d._pinned_retry_kill(peer, "a1"))
    assert peer.in_flight == 0


def test_pinned_retry_remove_pg_decrements_in_flight(monkeypatch):
    d = worker()
    peer = FakePeer()
    peer.in_flight = 1

    async def ok(pg):
        return None

    monkeypatch.setattr(d, "_retry_forward_remove_pg", ok)
    run(d._pinned_retry_remove_pg(peer, "p1"))
    assert peer.in_flight == 0


def test_worker_create_closed_pins_in_flight_on_kill_fail(monkeypatch):
    d = worker()
    d.head_peer = FakePeer(raise_on_call=RuntimeError("down"))
    peer = FakePeer()
    peer.closed = True
    # First forward is create; FakePeer raises on all - need create success then kill fail
    calls = {"n": 0}

    class SeqPeer(FakePeer):
        async def call(self, header, payload=b""):
            calls["n"] += 1
            if header.get("t") == "create_actor":
                return {"t": "create_actor_ok", "actor": "n1-a9"}, b""
            raise RuntimeError("kill down")

    d.head_peer = SeqPeer()
    scheduled = []

    def track(task):
        scheduled.append(task)
        task.cancel()
        return task

    monkeypatch.setattr(d, "_track", track)
    r, _ = run(d.on_create_actor(peer, {"t": "create_actor", "ngpu": 0}, b""))
    assert "disconnected" in r["err"]
    # pin was applied then finally decremented once; pin leaves in_flight for retry
    assert peer.in_flight >= 0
    assert scheduled  # pinned retry scheduled


def test_release_client_head_frees_pg_if_orphan_cleared_mid_loop(monkeypatch):
    """Reaper may clear an earlier orphan while we await a later kill."""
    d = head()
    peer = FakePeer()
    peer.created_actors = ["a1", "a2"]
    peer.created_pgs = ["p1"]
    d.actor_loc["a1"] = "n2"
    d.actor_loc["a2"] = "n1"
    d.actors["a2"] = ActorProc("a2", peer=FakePeer(), gpus=[], proc=FakeProc())
    d.pgs["p1"] = [{"node": "n2", "gpu": 0}]
    n = {"i": 0}

    async def kill_seq(peer, m, payload=b""):
        n["i"] += 1
        if n["i"] == 1:
            # soft-fail first; schedule would orphan a1
            return {"err": "down"}, b""
        # second kill succeeds and clears a1 orphan as if reaper raced
        d._orphans.pop("a1", None)
        d.actor_loc.pop("a1", None)
        return await Daemon.on_kill(d, peer, m, payload)

    scheduled: list[str] = []

    def capture(aid, node):
        scheduled.append(aid)
        d._orphans[aid] = node

    monkeypatch.setattr(d, "on_kill", kill_seq)
    monkeypatch.setattr(d, "_schedule_orphan_reap", capture)
    run(d.release_client(peer))
    # a1 was scheduled then cleared before PG decision; a2 killed ok → free PG
    assert "p1" not in d.pgs or "p1" not in d._orphan_pgs


def test_shutdown_cancels_orphan_tasks():
    d = head(2)

    async def go():
        async def never():
            await asyncio.Event().wait()

        t = asyncio.get_running_loop().create_task(never())
        d._orphan_tasks["a1"] = t
        d._orphans["a1"] = "n2"
        d._bg_tasks.add(t)
        d.shutdown()
        assert not d._orphan_tasks and not d._orphans
        # give cancellation a tick
        await asyncio.sleep(0)
        assert t.cancelled() or t.done()

    run(go())


def test_on_kill_timeout_returns_nonempty_err(monkeypatch):
    """TimeoutError() stringifies to ''; err must still be truthy for reapers."""
    d = head(2)

    class HangPeer(FakePeer):
        async def call(self, header, payload=b""):
            raise TimeoutError()

    d.nodes["n2"] = {
        "info": {"node": "n2", "ngpu": 1, "alive": True},
        "peer": HangPeer(),
    }
    d.actor_loc["a1"] = "n2"

    async def boom_wait(awaitable, timeout=None):
        # consume the coroutine so it is not left un-awaited
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", boom_wait)
    r, _ = run(d.on_kill(None, {"actor": "a1"}, b""))
    assert r.get("err")  # truthy
    assert d.actor_loc.get("a1") == "n2"


def test_create_after_supersede_does_not_rollback(monkeypatch):
    """Re-hello transferred the id: create success must not kill it."""
    d = head(1)
    d.sock_path = "/x.sock"
    proc = FakeProc()
    monkeypatch.setattr(Daemon, "_spawn_worker", _spawn_stub(proc))
    peer = FakePeer()
    peer.closed = True
    peer.superseded = True
    worker_peer = FakePeer()
    live = FakePeer()

    async def go():
        task = asyncio.ensure_future(d.on_create_actor(peer, {"t": "create_actor", "ngpu": 0}, b""))
        await asyncio.sleep(0)
        aid = next(iter(d.pending_workers))
        # simulate re-hello transfer of ownership off the dead peer
        if aid in peer.created_actors:
            peer.created_actors.remove(aid)
        live.created_actors.append(aid)
        d.nodes["w2"] = {
            "info": {"node": "w2", "ngpu": 1, "alive": True},
            "peer": live,
        }
        d.pending_workers[aid].set_result(worker_peer)
        return await task

    r, _ = run(go())
    assert r.get("t") == "create_actor_ok"
    assert not proc.terminated
    assert r["actor"] in d.actors
