"""Unit + fuzz tests for the daemon's pure logic (placement, id parsing, node
membership) and the Peer reqid-copy regression. Networked/async paths are
covered by the shell harnesses in test/."""

import asyncio
import os
import sys

from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray._daemon import Daemon, Peer, detect_gpus, new_node_id, owner_of  # noqa: E402


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
    assert nid[0] == "n" and len(nid) == 9 and int(nid[1:], 16) >= 0


def test_detect_gpus(monkeypatch):
    monkeypatch.setenv("BEAM_NUM_GPUS", "7")
    assert detect_gpus() == 7
    assert detect_gpus(override=3) == 3  # explicit override wins
    monkeypatch.delenv("BEAM_NUM_GPUS")
    assert detect_gpus(override=0) == 0


# ---- membership ----
def test_drop_node_clears_routing_and_peer():
    d = head()
    d.nodes["nX"] = {"info": {"node": "nX", "alive": True}, "peer": object()}
    d.actor_loc["a1"] = "nX"
    d.actor_loc["a2"] = "n1"
    d._drop_node("nX")
    assert "a1" not in d.actor_loc and "a2" in d.actor_loc  # only the dead node's
    assert d.nodes["nX"]["peer"] is None
    assert d.nodes["nX"]["info"]["alive"] is False


def test_drop_node_ignores_stale_peer():
    d = head()
    live = object()
    d.nodes["nX"] = {"info": {"node": "nX", "alive": True}, "peer": live}
    d._drop_node("nX", peer=object())  # a different (old) peer closing
    assert d.nodes["nX"]["peer"] is live  # live connection untouched
    assert d.nodes["nX"]["info"]["alive"] is True


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
