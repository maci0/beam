"""Unit + fuzz tests for the `ray` top-level lifecycle and runtime context:
init/is_initialized/shutdown, kill, _RemoteMethod.remote, ActorHandle dunder
guard, _RuntimeContext accessors, cluster/available resources, nodes, gpu-id
parsing, and ip lookup. All daemon traffic goes through the FakeClient pattern
(monkeypatch `ray._need`); no sockets or daemon."""

import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import ray  # noqa: E402


class FakeClient:
    def __init__(self, responses=None, body=b""):
        self.responses = responses or {}
        self.body = body
        self.sent = []
        self.closed = False

    def request(self, header, payload=b""):
        self.sent.append((header, payload))
        t = header["t"]
        canned = self.responses.get(t, {})
        if canned.get("err"):
            raise RuntimeError(canned["err"])
        resp = {"t": t + "_ok", "resp": True, **canned}
        return resp, canned.get("_body", self.body)

    def close(self):
        self.closed = True


def use(monkeypatch, fc):
    monkeypatch.setattr(ray, "_need", lambda: fc)
    return fc


# ---- lifecycle --------------------------------------------------------------


def test_init_is_initialized_shutdown(monkeypatch):
    fc = FakeClient()
    monkeypatch.setattr(ray, "DaemonClient", lambda *a, **k: fc)
    monkeypatch.setattr(ray, "_client", None)
    assert ray.is_initialized() is False
    ctx = ray.init()
    assert ctx is not None and ray.is_initialized() is True
    ray.shutdown()
    assert ray.is_initialized() is False and fc.closed


def test_init_reinit_raises(monkeypatch):
    monkeypatch.setattr(ray, "DaemonClient", lambda *a, **k: FakeClient())
    monkeypatch.setattr(ray, "_client", None)
    ray.init()
    try:
        with pytest.raises(RuntimeError, match="already initialized"):
            ray.init()
        assert ray.init(ignore_reinit_error=True) is None  # tolerated
    finally:
        ray.shutdown()


def test_shutdown_when_not_initialized(monkeypatch):
    monkeypatch.setattr(ray, "_client", None)
    ray.shutdown()  # no-op, no raise
    assert ray.is_initialized() is False


def test_need_raises_when_uninitialized(monkeypatch):
    monkeypatch.setattr(ray, "_client", None)
    with pytest.raises(RuntimeError, match="not initialized"):
        ray._need()


def test_need_returns_client_when_initialized(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ray, "_client", sentinel)
    assert ray._need() is sentinel


# ---- kill -------------------------------------------------------------------


def test_kill_sends_kill(monkeypatch):
    fc = use(monkeypatch, FakeClient({"kill": {}}))
    ray.kill(ray.ActorHandle("n1-a1"))
    assert fc.sent[0][0] == {"t": "kill", "actor": "n1-a1"}


def test_kill_non_handle_is_noop(monkeypatch):
    fc = use(monkeypatch, FakeClient())
    ray.kill("not an actor")
    assert fc.sent == []


# ---- _RemoteMethod / ActorHandle --------------------------------------------


def test_remote_method_remote_sends_call(monkeypatch):
    fc = use(monkeypatch, FakeClient({"call": {"obj": "n1-o1"}}))
    h = ray.ActorHandle("n1-a1")
    ref = h.do_work.remote(1, 2, k="v")
    assert ref.id == "n1-o1"
    header, payload = fc.sent[0]
    assert header == {"t": "call", "actor": "n1-a1", "method": "do_work"}
    args, kwargs = ray._proto.loads(payload)
    assert args == (1, 2) and kwargs == {"k": "v"}


def test_actor_handle_dunder_raises():
    h = ray.ActorHandle("n1-a1")
    name = "__deepcopy__"  # via a var so ruff doesn't rewrite getattr to attr access
    with pytest.raises(AttributeError):
        getattr(h, name)  # dunder lookups must not become remote methods


def test_actor_handle_normal_attr_is_remote_method():
    h = ray.ActorHandle("n1-a1")
    m = h.some_method
    assert isinstance(m, ray._RemoteMethod) and m._name == "some_method"


# ---- _RuntimeContext --------------------------------------------------------


def test_runtime_context_node_id_from_env(monkeypatch):
    monkeypatch.setenv("BEAM_NODE_ID", "nENV")
    assert ray._RuntimeContext().get_node_id() == "nENV"


def test_runtime_context_node_id_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("BEAM_NODE_ID", raising=False)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))  # no daemon.json
    assert ray._RuntimeContext().get_node_id() == "driver"


def test_runtime_context_node_id_from_runtime_file(monkeypatch, tmp_path):
    import json

    monkeypatch.delenv("BEAM_NODE_ID", raising=False)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(os.path.join(str(tmp_path), "daemon.json"), "w") as f:
        json.dump({"node": "nFILE"}, f)
    assert ray._RuntimeContext().get_node_id() == "nFILE"


def test_runtime_context_accelerator_ids(monkeypatch):
    monkeypatch.setenv("BEAM_GPU_IDS", "0,2,3")
    assert ray._RuntimeContext().get_accelerator_ids() == {"GPU": ["0", "2", "3"]}


def test_runtime_context_accelerator_ids_empty(monkeypatch):
    monkeypatch.setenv("BEAM_GPU_IDS", "")
    assert ray._RuntimeContext().get_accelerator_ids() == {"GPU": []}


def test_runtime_context_gpu_ids_property(monkeypatch):
    monkeypatch.setenv("BEAM_GPU_IDS", "1,4")
    assert ray._RuntimeContext().gpu_ids == [1, 4]


def test_get_runtime_context_returns_context():
    assert isinstance(ray.get_runtime_context(), ray._RuntimeContext)


# ---- get_gpu_ids ------------------------------------------------------------


def test_get_gpu_ids(monkeypatch):
    monkeypatch.setenv("BEAM_GPU_IDS", "0,1,2")
    assert ray.get_gpu_ids() == [0, 1, 2]


def test_get_gpu_ids_empty(monkeypatch):
    monkeypatch.delenv("BEAM_GPU_IDS", raising=False)
    assert ray.get_gpu_ids() == []


# ---- resources --------------------------------------------------------------


def _status_fc(monkeypatch, nodes):
    return use(monkeypatch, FakeClient({"status": {"nodes": nodes}}))


def test_cluster_resources(monkeypatch):
    _status_fc(monkeypatch, [{"node": "n1", "ngpu": 4}, {"node": "n2", "ngpu": 2}])
    assert ray.cluster_resources() == {"GPU": 6.0, "CPU": 2.0}


def test_available_resources(monkeypatch):
    _status_fc(
        monkeypatch, [{"node": "n1", "ngpu": 4, "used": 1}, {"node": "n2", "ngpu": 2, "used": 2}]
    )
    assert ray.available_resources() == {"GPU": 3.0, "CPU": 2.0}


def test_available_resources_clamps_negative(monkeypatch):
    _status_fc(monkeypatch, [{"node": "n1", "ngpu": 1, "used": 5}])  # over-used
    assert ray.available_resources()["GPU"] == 0  # max(0, ...) clamps


def test_nodes(monkeypatch):
    _status_fc(monkeypatch, [{"node": "n1", "ngpu": 2, "ip": "1.2.3.4", "alive": True}])
    out = ray.nodes()
    assert out[0]["NodeID"] == "n1" and out[0]["Alive"] is True
    assert out[0]["NodeManagerAddress"] == "1.2.3.4"
    assert out[0]["Resources"] == {"GPU": 2.0, "CPU": 1.0}


def test_status_nodes_empty(monkeypatch):
    use(monkeypatch, FakeClient({"status": {}}))
    assert ray._status_nodes() == []


# ---- _get_ip ----------------------------------------------------------------


def test_get_ip_returns_str():
    ip = ray._get_ip()
    assert isinstance(ip, str) and ip.count(".") == 3


def test_get_ip_fallback(monkeypatch):
    monkeypatch.delenv("BEAM_NODE_IP", raising=False)
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)

    class DeadSock:
        def connect(self, addr):
            raise OSError("no route")

        def getsockname(self):
            raise AssertionError

        def close(self):
            pass

    monkeypatch.setattr(ray.socket, "socket", lambda *a, **k: DeadSock())
    assert ray._get_ip() == "127.0.0.1"


def test_get_ip_prefers_beam_node_ip(monkeypatch):
    # an explicitly configured cluster IP wins over the socket heuristic
    monkeypatch.setenv("BEAM_NODE_IP", "10.1.2.3")

    def _no_socket(*a, **k):
        raise AssertionError("socket must not be used when the IP is configured")

    monkeypatch.setattr(ray.socket, "socket", _no_socket)
    assert ray._get_ip() == "10.1.2.3"


def test_get_ip_falls_back_to_vllm_host_ip(monkeypatch):
    monkeypatch.delenv("BEAM_NODE_IP", raising=False)
    monkeypatch.setenv("VLLM_HOST_IP", "10.4.5.6")
    assert ray._get_ip() == "10.4.5.6"


@given(st.text(min_size=1).filter(lambda s: s.strip() and "\x00" not in s))
def test_get_ip_returns_configured_value(ip):
    import os

    prev_b = os.environ.get("BEAM_NODE_IP")
    prev_v = os.environ.get("VLLM_HOST_IP")
    os.environ.pop("VLLM_HOST_IP", None)
    os.environ["BEAM_NODE_IP"] = ip
    try:
        assert ray._get_ip() == ip
    finally:
        os.environ.pop("BEAM_NODE_IP", None)
        if prev_b is not None:
            os.environ["BEAM_NODE_IP"] = prev_b
        if prev_v is not None:
            os.environ["VLLM_HOST_IP"] = prev_v


# ---- remote with placement-group scheduling strategy ------------------------


def test_remote_with_pg_scheduling_strategy(monkeypatch):
    from ray.util.placement_group import PlacementGroup
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    fc = use(monkeypatch, FakeClient({"create_actor": {"actor": "n1-a1"}}))
    pg = PlacementGroup("n1-pg1", [])
    strat = PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=1)

    class W:
        pass

    ray.remote(W).options(scheduling_strategy=strat).remote()
    header = fc.sent[0][0]
    assert header["pg"] == "n1-pg1" and header["bundle"] == 1


def test_remote_bare_decorator(monkeypatch):
    fc = use(monkeypatch, FakeClient({"create_actor": {"actor": "n1-a1"}}))

    @ray.remote
    class W:
        pass

    h = W.remote()
    assert isinstance(h, ray.ActorHandle)
    assert fc.sent[0][0]["ngpu"] == 0.0


def test_remote_redecoration_merges():
    class W:
        pass

    rc = ray.remote(num_gpus=1)(W)
    rc2 = ray.remote(num_gpus=2)(rc)  # re-decorate
    assert rc2._options["num_gpus"] == 2 and rc2._cls is W


# ---- wait loop --------------------------------------------------------------


def test_wait_local_value_is_ready(monkeypatch):
    use(monkeypatch, FakeClient())  # never consulted for a local-value ref
    ref = ray.ObjectRef("x", value=1, has_value=True)
    ready, not_ready = ray.wait([ref], num_returns=1, timeout=1)
    assert ready == [ref] and not_ready == []


def test_wait_polls_until_ready(monkeypatch):
    """First stat says not-ready, second says ready: the loop retries (covers
    the time.sleep poll path) and returns once num_returns is met."""
    calls = {"n": 0}

    class Flaky:
        def request(self, header, payload=b""):
            calls["n"] += 1
            ready = calls["n"] >= 2  # not ready on the first poll
            return {"t": "stat_ok", "ready": ready}, b""

    use(monkeypatch, Flaky())
    monkeypatch.setattr(ray.time, "sleep", lambda s: None)  # don't actually sleep
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=None)
    assert len(ready) == 1 and calls["n"] >= 2


# ---- types / actor re-exports -----------------------------------------------


def test_types_reexports_objectref():
    from ray import types

    assert types.ObjectRef is ray.ObjectRef


def test_actor_reexports_handle():
    from ray import actor

    assert actor.ActorHandle is ray.ActorHandle


def test_objectref_repr():
    assert repr(ray.ObjectRef("n1-o1")) == "ObjectRef(n1-o1)"


# ---- fuzz -------------------------------------------------------------------


@given(st.lists(st.integers(min_value=0, max_value=15), max_size=8))
def test_fuzz_get_gpu_ids_parses(ids):
    import unittest.mock as mock

    env_val = ",".join(str(i) for i in ids)
    with mock.patch.dict(os.environ, {"BEAM_GPU_IDS": env_val}, clear=False):
        assert ray.get_gpu_ids() == ids


@given(st.text(alphabet="0123456789,", max_size=20))
def test_fuzz_accelerator_ids_never_crashes(env_val):
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {"BEAM_GPU_IDS": env_val}, clear=False):
        out = ray._RuntimeContext().get_accelerator_ids()
    assert all(g for g in out["GPU"])  # no empty strings survive the split filter


@given(
    st.lists(
        st.fixed_dictionaries({"node": st.text(min_size=1, max_size=4), "ngpu": st.integers(0, 8)}),
        max_size=6,
    )
)
def test_fuzz_cluster_resources_sums(nodes):
    import unittest.mock as mock

    fc = FakeClient({"status": {"nodes": nodes}})
    with mock.patch.object(ray, "_need", lambda: fc):
        res = ray.cluster_resources()
    if nodes:
        assert res["GPU"] == float(sum(n["ngpu"] for n in nodes))
        assert res["CPU"] == float(len(nodes))
    else:
        assert res == {}
