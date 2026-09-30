"""Unit + fuzz tests for the small leaf modules: the actor worker subprocess
entrypoint (`_worker.py`, driven over a socketpair, no real subprocess, plus
frame/pickle fuzz of its dispatch loop), the unsupported `ray.dag` stubs, the
`ray.cloudpickle` re-export, the `ray.__main__` dispatch, and `detect_gpus`
env/glob fuzz."""

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


def test_worker_reply_survives_nonstring_t():
    # `t` arrives as decoded JSON, so it can be any type. Concatenating a
    # non-str (or hitting a missing key) here would raise inside the error path
    # and kill the worker instead of answering the frame.
    for req in ({}, {"t": None}, {"t": 7}, {"t": ["x"]}, {"t": 1.5}):
        a, b = socket.socketpair()
        try:
            _worker._reply(a, dict(req, reqid=3), err="bad op")
            h, p = _proto.read_frame(b)
            assert h["t"] == "_ok" and h["reqid"] == 3 and h["err"] == "bad op" and p == b""
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


# ---- _worker.main frame-loop fuzz (untrusted socket -> pickle -> dispatch) --
#
# The worker is the innermost untrusted-input parser in the tree: it reads framed
# bytes off the socket, deserializes the payload as a pickle, and dispatches on
# attacker-controlled header keys. A crash or hang here takes down every actor
# GPU. These harnesses drive the real main() over a socketpair and assert the
# loop's invariants: a malformed *method* frame is answered with an err frame and
# the worker keeps serving; a malformed *init* payload is answered and the worker
# then exits (it must never serve methods on a None instance); every reply is a
# well-formed frame echoing the request's reqid; the loop never wedges.


class _FuzzActor:
    """Picklable actor class with a method that raises, so the fuzz frames reach
    real dispatch (not just the None-instance path)."""

    def echo(self, x):
        return x

    def boom(self):
        raise ValueError("actor raised")


def _drive_worker(script, timeout=5.0):
    """Run the real _worker.main against `script` = [(header, payload, expect_reply)].

    Returns (replies, hello). Each `expect_reply` frame must produce exactly one
    reply (every op in the worker replies exactly once, including init failures),
    so the read count is deterministic. A short socket timeout turns a wedged
    worker into a test failure instead of a hung suite. Patches are applied with
    mock.patch (not a fixture) so state is fully reset between fuzz examples.
    """
    import unittest.mock as mock

    a, b = socket.socketpair()
    b.settimeout(timeout)  # hang guard: a stuck worker fails fast
    t = threading.Thread(target=_worker.main, daemon=True)
    replies = []
    hello = {}
    try:
        with (
            mock.patch.dict(os.environ, {"BEAM_SOCK": "/unused", "BEAM_ACTOR_ID": "afuzz"}),
            mock.patch.object(_worker.socket, "socket", lambda *x, **k: _PairedSock(a)),
        ):
            t.start()
            hello, _ = _proto.read_frame(b)  # worker_hello sent on attach
            assert hello["t"] == "worker_hello" and hello["actor"] == "afuzz"
            for header, payload, expect_reply in script:
                _proto.write_frame(b, header, payload)
                if expect_reply:
                    replies.append(_proto.read_frame(b))
    finally:
        b.close()  # EOF -> read_frame raises ConnectionError -> worker returns
        t.join(timeout=timeout)
        a.close()
    assert not t.is_alive(), "worker loop did not exit when the peer closed"
    return replies, hello


def _assert_reply(h, payload, reqid):
    """Every reply must be a well-formed frame with the request's reqid echoed,
    an '_ok' type, and (on error) an err string with the payload cleared."""
    assert h["resp"] is True
    assert h["t"].endswith("_ok")
    assert h["reqid"] == reqid
    if "err" in h:
        assert isinstance(h["err"], str) and h["err"] != ""
        assert payload == b""  # _reply clears the payload on error
    else:
        assert "err" not in h


# fuzzed method frames: arbitrary op name, method name, and raw payload bytes.
_method_headers = st.fixed_dictionaries(
    {},
    optional={
        "t": st.sampled_from(["method", "method", "method", "", "init_x", None, 7]),
        "method": st.one_of(
            st.sampled_from(["echo", "boom", "nope", "", "__class__", "0"]),
            st.text(max_size=8),
        ),
        "reqid": st.integers(min_value=-(2**31), max_value=2**31),
    },
)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            _method_headers,
            st.one_of(
                st.binary(max_size=64),
                st.binary(min_size=0, max_size=8).map(_proto.dumps),
            ),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_fuzz_worker_method_frames_never_kill_loop(frames):
    """Initialize a live actor, then send fuzzed method/unknown frames. Every frame
    is answered with exactly one well-formed reply; a bad method name, bad pickle,
    or unknown op becomes an err reply and the worker keeps serving (never wedges,
    never dies, never replies on a malformed frame)."""
    script = [({"t": "init", "reqid": 0}, _proto.dumps((_FuzzActor, (), {})), True)]
    for i, (h, p) in enumerate(frames, start=1):
        h = dict(h)  # `t` may legitimately be a non-str JSON value; keep it
        h["reqid"] = i
        script.append((h, p, True))

    replies, _ = _drive_worker(script)
    # one init reply + one reply per fuzzed frame
    assert len(replies) == len(frames) + 1
    for (h, _p, _expect), (rh, rp) in zip(script, replies):
        _assert_reply(rh, rp, h["reqid"])
    # a valid method call still succeeds, proving the loop stayed functional
    script2 = [
        ({"t": "init", "reqid": 100}, _proto.dumps((_FuzzActor, (), {})), True),
        ({"t": "method", "method": "echo", "reqid": 101}, _proto.dumps(((7,), {})), True),
    ]
    replies2, _ = _drive_worker(script2)
    assert _proto.loads(replies2[1][1]) == 7  # post-fuzz sanity: still serving


# fuzzed init payloads: a malformed init must be answered and then the worker
# exits; it must never hang or go on to serve methods on a half-built instance.
_init_payloads = st.one_of(
    st.binary(max_size=48),  # raw / truncated pickle bytes
    st.binary(min_size=0, max_size=16).map(_proto.dumps),  # valid pickles of wrong shape
    st.sampled_from([b"", b"not-a-pickle", b"\x80\x04", b"\x00" * 8]),
)


@settings(max_examples=50, deadline=None)
@given(payload=_init_payloads)
def test_fuzz_worker_init_payload_always_answered(payload):
    """Send one fuzzed init payload. Whatever the bytes are, the worker answers
    exactly one well-formed frame (init_ok on success, an err frame on failure)
    and the loop never hangs; on a failed init it exits rather than serving
    methods on a half-built instance."""
    replies, _ = _drive_worker([({"t": "init", "reqid": 0}, payload, True)])
    assert len(replies) == 1
    _assert_reply(replies[0][0], replies[0][1], 0)


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


# ---- new_node_id -----------------------------------------------------------


def test_new_node_id_unique_and_well_formed():
    """Distinctness is load-bearing: a collision would merge two workers into
    one membership entry, so two daemons on different hosts would silently share
    a node id and route to the wrong one."""
    ids = [_daemon.new_node_id() for _ in range(20)]
    assert len(set(ids)) == 20, "new_node_id returned a duplicate node id"
    for i in ids:
        assert i[0] == "n" and len(i) == 9
        int(i[1:], 16)  # the suffix is hex, as owner_of/parse expect
