"""Unit + fuzz tests for the daemon's pure logic (placement, id parsing, node
membership) and the Peer reqid-copy regression. Networked/async paths are
covered by the shell harnesses in test/."""

import asyncio
import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray._daemon import ActorProc, Daemon, Peer, detect_gpus, new_node_id, owner_of


def head(ngpu=4):
    return Daemon(is_head=True, node_id="n1", ip="1.2.3.4", num_gpus=ngpu)


# ---- placement ----
def test_place_pg_bundle():
    d = head()
    d.pgs["p"] = [{"node": "n1", "gpu": 2}]
    assert d._place_actor({"pg": "p", "bundle": 0}) == ("n1", [2], None)


def test_place_pg_bad_bundle():
    d = head()
    d.pgs["p"] = [{"node": "n1", "gpu": 0}]
    node, _, err = d._place_actor({"pg": "p", "bundle": 9})
    assert node is None and err


def test_place_unknown_pg():
    node, _, err = head()._place_actor({"pg": "nope", "bundle": 0})
    assert node is None and "unknown placement group" in err


def test_place_cpu_actor():
    assert head()._place_actor({"ngpu": 0}) == ("n1", [], None)


def test_place_greedy_excludes_pg_gpu():
    d = head(2)
    d.pgs["p"] = [{"node": "n1", "gpu": 0}]  # pg owns GPU 0
    _, gpus, err = d._place_actor({"ngpu": 1})
    assert gpus == [1] and err is None  # greedy must skip the pg-owned index


def test_place_exhaustion():
    d = head(1)
    d.gpu_used[0] = True
    node, _, err = d._place_actor({"ngpu": 1})
    assert node is None and "no free GPU" in err


# ---- malformed quantities on the wire --------------------------------------
# ngpu/bundle/GPU arrive from a peer over the control socket. Used raw they are
# not quantities: a str or float raises TypeError out of the placement handler
# mid-create, and NaN slips past every comparison it is checked against.
BAD_COUNTS = ["1", float("nan"), float("inf"), -0.5, -5, None, [1], True]


@pytest.mark.parametrize("bad", BAD_COUNTS)
def test_place_rejects_malformed_ngpu(bad):
    node, gpus, err = head()._place_actor({"ngpu": bad})
    assert node is None and gpus is None
    assert "invalid num_gpus" in err


@pytest.mark.parametrize("bad", BAD_COUNTS)
def test_place_rejects_malformed_bundle(bad):
    d = head()
    d.pgs["p"] = [{"node": "n1", "gpu": 0}]
    node, gpus, err = d._place_actor({"pg": "p", "bundle": bad})
    assert node is None and gpus is None
    assert "bundle index" in err


def test_place_rejects_ngpu_above_node_capacity():
    # 3 GPUs on a 2-GPU node can never be satisfied; placing it anyway would
    # hand the actor one device for a three-device request.
    node, gpus, err = head(2)._place_actor({"ngpu": 3})
    assert node is None and gpus is None and "exceeds" in err


def test_place_fractional_ngpu_still_takes_one_device():
    # num_gpus=0.5 is preserved on the wire (vLLM splits workers that way) and
    # the actor still gets exactly one CUDA_VISIBLE_DEVICES entry.
    assert head(4)._place_actor({"ngpu": 0.5}) == ("n1", [0], None)


@given(st.one_of(st.integers(), st.floats(), st.text()))
def test_place_never_raises_on_any_ngpu(value):
    d = head()
    try:
        d._place_actor({"ngpu": value})
    except Exception as e:  # noqa: BLE001 - the point is that nothing escapes
        raise AssertionError("ngpu=%r raised %r" % (value, e)) from None


# ---- id parsing ----
def test_owner_of():
    assert owner_of("nabc-o5") == "nabc"
    assert owner_of("n1-o123") == "n1"
    assert owner_of("garbage") == ""
    assert owner_of("") == ""


@given(st.text())
def test_owner_of_never_crashes(s):
    owner_of(s)  # any string in, no exception


def test_node_id_format():
    nid = new_node_id()
    assert nid[0] == "n" and len(nid) == 9
    # the suffix must be lowercase hex: membership keys and owner_of match on it
    assert nid[1:] == nid[1:].lower()
    int(nid[1:], 16)  # raises if the suffix is not hex


def test_detect_gpus(monkeypatch):
    monkeypatch.setenv("BEAM_NUM_GPUS", "7")
    assert detect_gpus() == 7
    assert detect_gpus(override=3) == 3  # explicit override wins
    monkeypatch.delenv("BEAM_NUM_GPUS")
    assert detect_gpus(override=0) == 0


# ---- membership ----
def test_drop_node_removes_node_and_routing():
    d = head()
    d.nodes["nX"] = {"info": {"node": "nX", "alive": True}, "peer": object()}
    d.actor_loc["a1"] = "nX"
    d.actor_loc["a2"] = "n1"
    d._drop_node("nX")
    assert "nX" not in d.nodes  # gone, not a phantom DOWN node forever
    assert "a1" not in d.actor_loc and "a2" in d.actor_loc  # only the dead node's


def test_drop_node_ignores_stale_peer():
    d = head()
    live = object()
    d.nodes["nX"] = {"info": {"node": "nX", "alive": True}, "peer": live}
    d._drop_node("nX", peer=object())  # a different (old) peer closing
    assert "nX" in d.nodes and d.nodes["nX"]["peer"] is live  # live node untouched


def test_drop_actor_frees_gpu_and_routing():
    d = head(2)
    d.gpu_used[1] = True
    d.actor_loc["a1"] = "n1"
    d.actors["a1"] = ActorProc("a1", peer=object(), gpus=[1])  # type: ignore[arg-type]
    d._drop_actor("a1")
    assert "a1" not in d.actors and "a1" not in d.actor_loc
    assert d.gpu_used[1] is False  # crashed actor's GPU reclaimed


# ---- the reqid-copy regression (the bug that hung driver-on-worker) ----
def test_peer_call_does_not_mutate_caller_header():
    async def run():
        class FakeWriter:
            def write(self, b):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

        p = Peer(reader=None, writer=FakeWriter(), handler=None)
        original = {"t": "call", "actor": "a1"}
        task = asyncio.ensure_future(p.call(original))
        await asyncio.sleep(0)  # let call() send + register its pending future
        assert "reqid" not in original  # caller's dict must be untouched
        rid = next(iter(p.pending))
        p.pending[rid].set_result(({"resp": True, "reqid": rid}, b""))
        await task

    asyncio.run(run())


def test_peer_call_on_closed_raises():
    async def run():
        class FakeWriter:
            def write(self, b):
                raise AssertionError("must not write after close")

            async def drain(self):
                pass

            def close(self):
                pass

        p = Peer(reader=None, writer=FakeWriter(), handler=None)
        p.closed = True
        try:
            await p.call({"t": "x"})
            raise AssertionError("expected ConnectionError")
        except ConnectionError as e:
            assert "closed" in str(e)
        assert p.pending == {}

    asyncio.run(run())


def test_peer_send_on_closed_raises():
    async def run():
        class FakeWriter:
            def write(self, b):
                raise AssertionError("must not write")

            async def drain(self):
                pass

            def close(self):
                pass

        p = Peer(reader=None, writer=FakeWriter(), handler=None)
        p.closed = True
        try:
            await p.send({"t": "x"})
            raise AssertionError("expected ConnectionError")
        except ConnectionError:
            pass

    asyncio.run(run())
