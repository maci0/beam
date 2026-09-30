"""Unit tests for the synchronous DaemonClient (`_client.py`). Driven over a
real `socket.socketpair()` (an in-process pipe, not a network socket): a canned
response frame is written into one end and the client reads it off the other,
so request/close/err-mapping are exercised without a daemon."""

import json
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import ray._client  # noqa: E402,F401  (ensure the submodule is loaded)
from ray import _proto  # noqa: E402
from ray._client import DaemonClient, DaemonNotRunning, _runtime_sock  # noqa: E402

# `ray._client` the attribute is the module-level `_client = None` global, which
# shadows the submodule on the package. Reach the real module via sys.modules.
_client = sys.modules["ray._client"]


def _make_client(server_sock, client_sock):
    """Build a DaemonClient bound to a pre-connected socketpair end, bypassing
    the unix connect in __init__ (no daemon to dial)."""
    c = DaemonClient.__new__(DaemonClient)
    c.sock_path = "<pair>"
    c._sock = client_sock
    import threading

    c._lock = threading.Lock()
    c._timeout = None  # default budget, as __init__ would set it
    return c


def test_request_roundtrip():
    srv, cli = socket.socketpair()
    try:
        c = _make_client(srv, cli)
        # daemon side: write a canned response frame onto srv
        _proto.write_frame(srv, {"t": "put_ok", "obj": "n1-o1"}, b"body")
        resp, body = c.request({"t": "put"}, b"payload")
        assert resp["obj"] == "n1-o1" and body == b"body"
        # the client actually sent our request frame onto cli -> srv reads it
        hdr, pl = _proto.read_frame(srv)
        assert hdr["t"] == "put" and pl == b"payload"
    finally:
        srv.close()
        cli.close()


def test_request_err_maps_to_runtimeerror():
    srv, cli = socket.socketpair()
    try:
        c = _make_client(srv, cli)
        _proto.write_frame(srv, {"t": "get_ok", "err": "boom"}, b"")
        with pytest.raises(RuntimeError, match="boom"):
            c.request({"t": "get", "obj": "x"})
    finally:
        srv.close()
        cli.close()


def test_close_is_idempotent():
    srv, cli = socket.socketpair()
    try:
        c = _make_client(srv, cli)
        c.close()
        c.close()  # second close swallows OSError, no raise
    finally:
        srv.close()


def test_close_swallows_oserror():
    c = DaemonClient.__new__(DaemonClient)

    class Boom:
        def close(self):
            raise OSError("already gone")

    c._sock = Boom()
    c.close()  # OSError swallowed, no raise


def test_request_after_close_raises():
    srv, cli = socket.socketpair()
    try:
        c = _make_client(srv, cli)
        c.close()
        with pytest.raises(OSError):
            c.request({"t": "put"})
    finally:
        srv.close()


def test_request_timeout_bounds_a_wedged_daemon():
    """A request issued with a budget must not block on an unbounded socket
    read: this is what lets ray.wait honor its timeout when the daemon is slow
    (or the owner node has dropped and the head's stat is stuck)."""
    srv, cli = socket.socketpair()
    try:
        c = _make_client(srv, cli)
        with pytest.raises(TimeoutError, match="timed out"):
            c.request({"t": "stat", "obj": "n1-o1"}, b"", timeout=0.05)
    finally:
        srv.close()
        cli.close()


def test_request_timeout_is_cleared_for_the_next_call():
    """A budget applies to one round-trip only; the next request is unbounded
    again (a per-call deadline must not leak into later calls)."""
    srv, cli = socket.socketpair()
    try:
        c = _make_client(srv, cli)
        with pytest.raises(TimeoutError):
            c.request({"t": "stat", "obj": "n1-o1"}, b"", timeout=0.05)
        assert c._sock.gettimeout() is None
        _proto.write_frame(srv, {"t": "stat_ok", "ready": False}, b"")
        resp, _ = c.request({"t": "stat", "obj": "n1-o1"})
        assert resp["ready"] is False
    finally:
        srv.close()
        cli.close()


def test_request_uses_client_default_timeout():
    srv, cli = socket.socketpair()
    try:
        c = _make_client(srv, cli)
        c._timeout = 0.05
        with pytest.raises(TimeoutError):
            c.request({"t": "stat", "obj": "n1-o1"})  # no per-call budget
        # an explicit budget overrides the default
        _proto.write_frame(srv, {"t": "stat_ok", "ready": True}, b"")
        resp, _ = c.request({"t": "stat", "obj": "n1-o1"}, timeout=5.0)
        assert resp["ready"] is True
    finally:
        srv.close()
        cli.close()


# ---- _runtime_sock ----------------------------------------------------------


def test_runtime_sock_from_env(monkeypatch):
    monkeypatch.setenv("BEAM_SOCK", "/explicit.sock")
    assert _runtime_sock() == "/explicit.sock"


def test_runtime_sock_from_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("BEAM_SOCK", raising=False)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(os.path.join(str(tmp_path), "daemon.json"), "w") as f:
        json.dump({"sock": "/from/dir.sock"}, f)
    assert _runtime_sock() == "/from/dir.sock"


def test_runtime_sock_missing_file_raises(tmp_path, monkeypatch):
    # a missing runtime document now says so, instead of leaking FileNotFoundError
    monkeypatch.delenv("BEAM_SOCK", raising=False)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(DaemonNotRunning, match="no local daemon found"):
        _runtime_sock()


def test_runtime_sock_malformed_doc_raises(tmp_path, monkeypatch):
    """A half-written / hand-edited daemon.json must not raise KeyError/JSONDecodeError."""
    monkeypatch.delenv("BEAM_SOCK", raising=False)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(os.path.join(str(tmp_path), "daemon.json"), "w") as f:
        f.write("{not json")
    with pytest.raises(DaemonNotRunning):
        _runtime_sock()

    with open(os.path.join(str(tmp_path), "daemon.json"), "w") as f:
        json.dump({"pid": 7}, f)  # valid JSON, no "sock"
    with pytest.raises(DaemonNotRunning):
        _runtime_sock()


def test_client_init_reports_unreachable_daemon(monkeypatch):
    """Connecting to a dead socket path reports it, not a bare OSError."""

    class DeadSock:
        def connect(self, path):
            raise OSError("no such file or directory")

        def close(self):
            pass

    monkeypatch.setattr(_client.socket, "socket", lambda *a, **k: DeadSock())
    with pytest.raises(DaemonNotRunning, match="cannot reach the local daemon"):
        DaemonClient(sock_path="/gone.sock")


def test_client_init_dials_sock_path(monkeypatch):
    """__init__ connects a real AF_UNIX socket; stub socket.socket so we can
    assert the path without a live daemon."""
    connected = {}

    class FakeSock:
        def connect(self, path):
            connected["path"] = path

        def close(self):
            pass

    monkeypatch.setattr(_client.socket, "socket", lambda *a, **k: FakeSock())
    c = DaemonClient(sock_path="/my.sock")
    assert connected["path"] == "/my.sock" and c.sock_path == "/my.sock"
