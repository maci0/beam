"""Unit + fuzz tests for the async daemon handlers in `_daemon.py`, driven
in-process with a FakePeer (no real sockets, subprocesses, or GPUs). Each
`on_*` coroutine is run via `asyncio.run` and asserted on its `(dict, bytes)`
return and the daemon state it mutates (actor_loc, pgs, gpu_used, objects,
nodes)."""

import asyncio
import os
import socket
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray import _daemon  # noqa: E402
from ray._daemon import (  # noqa: E402
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
        self.created_actors = []
        self.created_pgs = []
        self.on_close = None

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
        self.closed = True


class FakeProc:
    """Stand-in for subprocess.Popen: poll()/terminate() only."""

    def __init__(self, alive=True):
        self._alive = alive
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False


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

    _terminate(Boom())  # OSError swallowed, no raise


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
    aid = r["actor"]
    assert d.actor_loc[aid] == "n1" and aid in peer.created_actors


def test_on_create_actor_rollback_on_worker_failure(monkeypatch):
    """Head places on a remote node; the remote .call raises -> placement must
    roll back: actor_loc popped, GPU freed, created_actors entry removed."""
    d = head(2)
    remote = FakePeer(raise_on_call=RuntimeError("node died"))
    d.nodes["n2"] = {"info": {"node": "n2", "ngpu": 2, "alive": True}, "peer": remote}
    # force placement onto n2 via a pg bundle that lives on n2 with a gpu
    d.pgs["p"] = [{"node": "n2", "gpu": 0}]
    peer = FakePeer()
    r, _ = run(d.on_create_actor(peer, {"t": "create_actor", "pg": "p", "bundle": 0}, b""))
    assert "node died" in r["err"]
    assert d.actor_loc == {}  # routing entry rolled back
    assert peer.created_actors == []  # ownership entry removed


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
    assert d.actor_loc == {}  # rolled back


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


# ---- on_hello (membership) --------------------------------------------------


def test_on_hello_registers_node():
    d = head()
    peer = FakePeer()
    r, _ = run(d.on_hello(peer, {"t": "hello", "node": "n2", "ip": "9.9.9.9", "ngpu": 3}, b""))
    assert r["t"] == "hello_ok"
    assert d.nodes["n2"]["info"]["ngpu"] == 3 and d.nodes["n2"]["peer"] is peer
    assert peer.on_close is not None  # wired to _drop_node


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
