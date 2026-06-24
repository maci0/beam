"""Unit + fuzz tests for the `ray` shim's translation logic. A fake daemon
client records requests and returns canned responses, so these run with no
daemon, no sockets, no GPUs."""

import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import ray  # noqa: E402
from ray import _proto  # noqa: E402
from ray.exceptions import GetTimeoutError  # noqa: E402
from ray.runtime_env import RuntimeEnv  # noqa: E402


class FakeClient:
    """Mirrors DaemonClient.request: an "err" key in the canned response raises
    RuntimeError, exactly like the real client does for a daemon error frame."""

    def __init__(self, responses=None, body=b""):
        self.responses = responses or {}
        self.body = body
        self.sent = []

    def request(self, header, payload=b""):
        self.sent.append((header, payload))
        t = header["t"]
        canned = self.responses.get(t, {})
        if canned.get("err"):
            raise RuntimeError(canned["err"])  # same contract as DaemonClient
        resp = {"t": t + "_ok", "resp": True, **canned}
        return resp, canned.get("_body", self.body)


def use(monkeypatch, fc):
    monkeypatch.setattr(ray, "_need", lambda: fc)
    return fc


# ---- ObjectRef ----
def test_objectref_has_value_shortcuts_get(monkeypatch):
    fc = use(monkeypatch, FakeClient())
    ref = ray.ObjectRef("x", value=99, has_value=True)
    assert ray.get(ref) == 99
    assert fc.sent == []  # no daemon round-trip for a local value


def test_objectref_eq_hash():
    a, b = ray.ObjectRef("n1-o1"), ray.ObjectRef("n1-o1")
    assert a == b and hash(a) == hash(b)
    assert a != ray.ObjectRef("n1-o2")


@given(st.text(min_size=1))
def test_objectref_id_roundtrip(s):
    assert ray.ObjectRef(s).id == s  # never crashes, id preserved


# ---- put / get ----
def test_put_pickles_and_returns_ref(monkeypatch):
    fc = use(monkeypatch, FakeClient({"put": {"obj": "n1-o1"}}))
    ref = ray.put({"a": 1})
    assert ref.id == "n1-o1"
    assert _proto.loads(fc.sent[0][1]) == {"a": 1}


def test_get_unpickles_body(monkeypatch):
    use(monkeypatch, FakeClient({"get": {"_body": _proto.dumps([1, 2, 3])}}))
    assert ray.get(ray.ObjectRef("n1-o5")) == [1, 2, 3]


def test_get_timeout_maps_to_GetTimeoutError(monkeypatch):
    # drive the real path: daemon returns an "err" with the exact on_get string,
    # the client raises RuntimeError, the shim re-raises GetTimeoutError
    use(
        monkeypatch,
        FakeClient({"get": {"err": "GetTimeoutError: object n1-o1 not ready in 0.01s"}}),
    )
    with pytest.raises(GetTimeoutError):
        ray.get(ray.ObjectRef("n1-o1"), timeout=0.01)


def test_get_other_error_reraises(monkeypatch):
    use(monkeypatch, FakeClient({"get": {"err": "unknown object n1-o1"}}))
    with pytest.raises(RuntimeError):
        ray.get(ray.ObjectRef("n1-o1"))


def test_err_response_propagates_from_put(monkeypatch):
    use(monkeypatch, FakeClient({"put": {"err": "daemon exploded"}}))
    with pytest.raises(RuntimeError, match="daemon exploded"):
        ray.put(123)


def test_get_list_preserves_order(monkeypatch):
    use(monkeypatch, FakeClient({"get": {"_body": _proto.dumps(7)}}))
    assert ray.get([ray.ObjectRef("a"), ray.ObjectRef("b")]) == [7, 7]


# ---- wait ----
def test_wait_num_returns_capped(monkeypatch):
    use(monkeypatch, FakeClient({"stat": {"ready": True}}))
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=99, timeout=1)
    assert len(ready) == 1 and not_ready == []  # cap at len(refs), no hang


def test_wait_timeout_returns_partial(monkeypatch):
    use(monkeypatch, FakeClient({"stat": {"ready": False}}))
    ready, not_ready = ray.wait([ray.ObjectRef("a")], num_returns=1, timeout=0)
    assert ready == [] and len(not_ready) == 1


# ---- remote / options ----
def test_remote_sends_float_num_gpus(monkeypatch):
    fc = use(monkeypatch, FakeClient({"create_actor": {"actor": "n1-a1"}}))

    class W:
        def __init__(self, x=0):
            self.x = x

    h = ray.remote(num_gpus=0.5)(W).remote()
    assert h is not None
    assert fc.sent[0][0]["ngpu"] == 0.5  # fractional preserved, not truncated to 0


def test_options_merges(monkeypatch):
    use(monkeypatch, FakeClient({"create_actor": {"actor": "n1-a1"}}))

    class W:
        pass

    rc = ray.remote(W).options(num_gpus=1)
    assert rc._options.get("num_gpus") == 1


# ---- RuntimeEnv ----
def test_runtime_env_kwargs_and_positional():
    assert RuntimeEnv(env_vars={"A": "1"})["env_vars"] == {"A": "1"}
    assert RuntimeEnv({"working_dir": "/x"})["working_dir"] == "/x"  # positional dict


@given(st.dictionaries(st.text(), st.text(), max_size=6))
def test_runtime_env_fuzz(d):
    assert dict(RuntimeEnv(d)) == d
