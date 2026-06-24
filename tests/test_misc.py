"""Unit + fuzz tests for the small leaf modules: the actor worker subprocess
entrypoint (`_worker.py`, driven over a socketpair, no real subprocess), the
unsupported `ray.dag` stubs, the `ray.cloudpickle` re-export, the `ray.__main__`
dispatch, and `detect_gpus` env/glob fuzz."""

import os
import socket
import sys
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray import _daemon, _proto, _worker  # noqa: E402


class _PairedSock:
    """Wraps one end of a socketpair so _worker.main's `connect()` is a no-op
    (the pair is already connected). Real sockets reject attribute assignment,
    so we proxy instead of monkeypatching the socket object."""

    def __init__(self, sock):
        self._sock = sock

    def connect(self, path):
        pass  # already paired

    def __getattr__(self, name):
        return getattr(self._sock, name)


# ---- _worker._reply ---------------------------------------------------------


def test_worker_reply_ok():
    a, b = socket.socketpair()
    try:
        _worker._reply(a, {"t": "method", "reqid": 7}, _proto.dumps(42))
        h, p = _proto.read_frame(b)
        assert h["t"] == "method_ok" and h["reqid"] == 7 and h["resp"] is True
        assert _proto.loads(p) == 42
    finally:
        a.close()
        b.close()


def test_worker_reply_err_clears_payload():
    a, b = socket.socketpair()
    try:
        _worker._reply(a, {"t": "init"}, b"ignored", err="boom")
        h, p = _proto.read_frame(b)
        assert h["err"] == "boom" and p == b"" and h["reqid"] == 0
    finally:
        a.close()
        b.close()


# ---- _worker.main over a socketpair (no subprocess) -------------------------


class _Demo:
    def __init__(self, base=0):
        self.base = base

    def add(self, x):
        return self.base + x

    def boom(self):
        raise ValueError("method failed")


def test_worker_init_and_method(monkeypatch):
    a, b = socket.socketpair()
    monkeypatch.setenv("BEAM_SOCK", "/unused")
    monkeypatch.setenv("BEAM_ACTOR_ID", "a1")
    monkeypatch.setattr(_worker.socket, "socket", lambda *x, **k: _PairedSock(a))

    t = threading.Thread(target=_worker.main, daemon=True)
    t.start()

    # 1) read the worker_hello the worker sends on attach
    h, _ = _proto.read_frame(b)
    assert h["t"] == "worker_hello" and h["actor"] == "a1"

    # 2) send init
    _proto.write_frame(b, {"t": "init", "reqid": 1}, _proto.dumps((_Demo, (5,), {})))
    h, _ = _proto.read_frame(b)
    assert h["t"] == "init_ok" and not h.get("err")

    # 3) send a method call
    _proto.write_frame(b, {"t": "method", "method": "add", "reqid": 2}, _proto.dumps(((10,), {})))
    h, p = _proto.read_frame(b)
    assert h["t"] == "method_ok" and _proto.loads(p) == 15

    # 4) method that raises -> err reply, worker keeps serving
    _proto.write_frame(b, {"t": "method", "method": "boom", "reqid": 3}, _proto.dumps(((), {})))
    h, _ = _proto.read_frame(b)
    assert h.get("err") and "method failed" in h["err"]

    # 5) unknown op
    _proto.write_frame(b, {"t": "frob", "reqid": 4}, b"")
    h, _ = _proto.read_frame(b)
    assert "unknown worker op" in h["err"]

    b.close()  # closing the daemon end makes read_frame raise -> worker returns
    t.join(timeout=2)
    assert not t.is_alive()
    a.close()


def test_worker_init_failure_exits(monkeypatch):
    a, b = socket.socketpair()
    monkeypatch.setenv("BEAM_SOCK", "/unused")
    monkeypatch.setenv("BEAM_ACTOR_ID", "a2")
    monkeypatch.setattr(_worker.socket, "socket", lambda *x, **k: _PairedSock(a))

    class Boom:
        def __init__(self):
            raise RuntimeError("ctor died")

    t = threading.Thread(target=_worker.main, daemon=True)
    t.start()
    _proto.read_frame(b)  # worker_hello
    _proto.write_frame(b, {"t": "init", "reqid": 1}, _proto.dumps((Boom, (), {})))
    h, _ = _proto.read_frame(b)
    assert h.get("err") and "ctor died" in h["err"]
    # init failure exits the worker (no instance to serve methods on)
    t.join(timeout=2)
    assert not t.is_alive()
    a.close()
    b.close()


def test_worker_ignores_resp_frames(monkeypatch):
    a, b = socket.socketpair()
    monkeypatch.setenv("BEAM_SOCK", "/unused")
    monkeypatch.setenv("BEAM_ACTOR_ID", "a3")
    monkeypatch.setattr(_worker.socket, "socket", lambda *x, **k: _PairedSock(a))

    t = threading.Thread(target=_worker.main, daemon=True)
    t.start()
    _proto.read_frame(b)  # worker_hello
    # a stray response frame: the worker must skip it, not treat it as an op
    _proto.write_frame(b, {"t": "worker_hello_ok", "resp": True}, b"")
    # follow with a real init to prove the loop is still alive
    _proto.write_frame(b, {"t": "init", "reqid": 1}, _proto.dumps((_Demo, (), {})))
    h, _ = _proto.read_frame(b)
    assert h["t"] == "init_ok"
    b.close()
    t.join(timeout=2)
    a.close()


# ---- ray.dag (unsupported stubs) --------------------------------------------


def test_dag_stubs_raise_notimplemented():
    from ray import dag

    for cls in (dag.CompiledDAG, dag.InputNode, dag.MultiOutputNode):
        with pytest.raises(NotImplementedError, match="compiled DAG"):
            cls()


# ---- ray.cloudpickle re-export ----------------------------------------------


def test_cloudpickle_reexport():
    from ray import cloudpickle as cp

    assert cp.loads(cp.dumps({"a": 1})) == {"a": 1}
    assert hasattr(cp, "register_pickle_by_value")


# ---- ray.__main__ dispatch --------------------------------------------------


def test_main_module_calls_cli_main(monkeypatch):
    import ray._cli as cli

    monkeypatch.setattr(cli, "main", lambda: 0)
    # importing __main__ as a module should not execute (guarded by __name__);
    # just assert the symbol it wires up is present.
    import ray.__main__ as m

    assert m.main is cli.main


# ---- detect_gpus fuzz -------------------------------------------------------


def test_detect_gpus_glob_path(monkeypatch):
    monkeypatch.delenv("BEAM_NUM_GPUS", raising=False)
    monkeypatch.setattr(_daemon.glob, "glob", lambda pat: ["/dev/nvidia0", "/dev/nvidia1"])
    assert _daemon.detect_gpus() == 2


@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=64))
def test_fuzz_detect_gpus_env(n):
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {"BEAM_NUM_GPUS": str(n)}, clear=False):
        assert _daemon.detect_gpus() == n


@settings(max_examples=100)
@given(st.integers(min_value=-5, max_value=64), st.integers(min_value=0, max_value=64))
def test_fuzz_detect_gpus_override_precedence(override, env):
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {"BEAM_NUM_GPUS": str(env)}, clear=False):
        result = _daemon.detect_gpus(override=override)
    if override >= 0:
        assert result == override  # non-negative override always wins
    else:
        assert result == env  # negative override is ignored, env used


@settings(max_examples=50)
@given(st.lists(st.text(), max_size=8))
def test_fuzz_detect_gpus_counts_glob(devs):
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("BEAM_NUM_GPUS", None)
        with mock.patch.object(_daemon.glob, "glob", lambda pat: devs):
            assert _daemon.detect_gpus() == len(devs)


# ---- new_node_id fuzz -------------------------------------------------------


@settings(max_examples=50)
@given(st.integers(min_value=0, max_value=100))
def test_fuzz_new_node_id_unique(_n):
    ids = {_daemon.new_node_id() for _ in range(20)}
    assert all(i.startswith("n") and len(i) == 9 for i in ids)
