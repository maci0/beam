"""Unit + fuzz tests for the `ray`/`beam` command line (`_cli.py`): runtime
path resolution, local-ip lookup, usage, `start` argument parsing, and the
status/stop flows driven against a fake runtime dir with monkeypatched
socket/os.kill (no real daemon, sockets, or signals)."""

import json
import os
import signal
import struct
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray import _cli  # noqa: E402

# ---- runtime dir / path -----------------------------------------------------


def test_runtime_dir_env_override(monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", "/custom/dir")
    assert _cli._runtime_dir() == "/custom/dir"
    assert _cli._runtime_path() == "/custom/dir/daemon.json"


def test_runtime_dir_default(monkeypatch):
    monkeypatch.delenv("BEAM_RUNTIME_DIR", raising=False)
    d = _cli._runtime_dir()
    assert d.endswith(".beam")


# ---- _local_ip --------------------------------------------------------------


def test_local_ip_returns_str():
    ip = _cli._local_ip()
    assert isinstance(ip, str) and ip.count(".") == 3


def test_local_ip_falls_back_on_oserror(monkeypatch):
    class DeadSock:
        def connect(self, addr):
            raise OSError("no route")

        def getsockname(self):
            raise AssertionError("must not be called")

        def close(self):
            pass

    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: DeadSock())
    assert _cli._local_ip() == "127.0.0.1"


# ---- _usage / main ----------------------------------------------------------


def test_usage_returns_2(capsys):
    assert _cli._usage() == 2
    assert "drop-in subset of ray" in capsys.readouterr().err


def test_main_no_args_is_usage():
    assert _cli.main([]) == 2


def test_main_help():
    assert _cli.main(["--help"]) == 2
    assert _cli.main(["help"]) == 2


def test_main_unknown_command(capsys):
    assert _cli.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_main_bootstrap_dispatch(monkeypatch):
    called = []
    monkeypatch.setattr(_cli, "bootstrap_env", lambda: called.append(True))
    assert _cli.main(["bootstrap"]) == 0 and called == [True]


def test_main_dispatches_start_status_stop(monkeypatch):
    seen = []
    monkeypatch.setattr(_cli, "_start", lambda rest: seen.append(("start", rest)) or 0)
    monkeypatch.setattr(_cli, "_status", lambda: seen.append(("status",)) or 0)
    monkeypatch.setattr(_cli, "_stop", lambda: seen.append(("stop",)) or 0)
    assert _cli.main(["start", "--head"]) == 0
    assert _cli.main(["status"]) == 0
    assert _cli.main(["stop"]) == 0
    assert seen == [("start", ["--head"]), ("status",), ("stop",)]


def test_start_block_flag_accepted(monkeypatch):
    """--block is a no-op accepted for ray compatibility (covers that branch)."""
    captured = {}

    def fake_run(coro):
        coro.close()
        return 0

    def fake_run_daemon(head, node_id, ip, gpus, port, address):
        captured["node_ip"] = ip

        async def _c():
            return 0

        return _c()

    monkeypatch.setattr(_cli, "maybe_bootstrap", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli._daemon, "detect_gpus", lambda n: 0)
    # space-separated --port and --node-ip forms, plus --block
    rc = _cli._start(["--head", "--block", "--port", "6400", "--node-ip", "3.3.3.3"])
    assert rc == 0 and captured["node_ip"] == "3.3.3.3"


def test_start_node_ip_equals_form(monkeypatch):
    captured = {}

    def fake_run(coro):
        coro.close()
        return 0

    def fake_run_daemon(head, node_id, ip, gpus, port, address):
        captured["node_ip"] = ip

        async def _c():
            return 0

        return _c()

    monkeypatch.setattr(_cli, "maybe_bootstrap", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli._daemon, "detect_gpus", lambda n: 0)
    _cli._start(["--head", "--node-ip=4.4.4.4"])
    assert captured["node_ip"] == "4.4.4.4"


# ---- _start arg parsing -----------------------------------------------------


def test_start_needs_head_or_address(capsys):
    assert _cli._start([]) == 2
    assert "need --head or --address" in capsys.readouterr().err


def test_start_bad_port_exits_2():
    with pytest.raises(SystemExit) as e:
        _cli._start(["--head", "--port", "notaport"])
    assert e.value.code == 2


def test_start_port_missing_value_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        _cli._start(["--head", "--port"])
    assert e.value.code == 2
    assert "--port expects a value" in capsys.readouterr().err


def test_start_unknown_flag(capsys):
    assert _cli._start(["--head", "--bogus"]) == 2
    assert "unknown flag" in capsys.readouterr().err


def test_start_negative_num_gpus(capsys):
    assert _cli._start(["--head", "--num-gpus", "-1"]) == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_start_bad_num_gpus_value():
    with pytest.raises(SystemExit) as e:
        _cli._start(["--head", "--num-gpus", "x"])
    assert e.value.code == 2


def test_start_bad_address_port(capsys):
    assert _cli._start(["--address", "host:notnum"]) == 2
    assert "port must be numeric" in capsys.readouterr().err


def test_start_help_returns_usage():
    assert _cli._start(["--help"]) == 2


def test_start_dispatches_to_run_daemon(monkeypatch):
    """A well-formed --head invocation should reach asyncio.run(_run_daemon...)
    with parsed args; stub it out so no real daemon starts."""
    captured = {}

    def fake_run(coro):
        coro.close()  # don't actually await the daemon
        return 0

    def fake_run_daemon(head, node_id, ip, gpus, port, address):
        captured.update(head=head, node_id=node_id, ip=ip, gpus=gpus, port=port, address=address)

        async def _c():
            return 0

        return _c()

    monkeypatch.setattr(_cli, "maybe_bootstrap", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli._daemon, "detect_gpus", lambda n: 8)
    monkeypatch.setattr(_cli, "_local_ip", lambda: "1.1.1.1")
    rc = _cli._start(["--head", "--port=7000", "--num-gpus=8", "--node-ip", "2.2.2.2"])
    assert rc == 0
    assert captured["head"] is True and captured["port"] == 7000
    assert captured["gpus"] == 8 and captured["ip"] == "2.2.2.2"


def test_start_address_form_parses(monkeypatch):
    captured = {}

    def fake_run(coro):
        coro.close()
        return 0

    def fake_run_daemon(head, node_id, ip, gpus, port, address):
        captured.update(head=head, address=address)

        async def _c():
            return 0

        return _c()

    monkeypatch.setattr(_cli, "maybe_bootstrap", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli._daemon, "detect_gpus", lambda n: 0)
    monkeypatch.setattr(_cli, "_local_ip", lambda: "1.1.1.1")
    _cli._start(["--address=h:6379"])
    assert captured["head"] is False and captured["address"] == "h:6379"


# ---- _run_daemon (head + worker, no real listeners) -------------------------


def _patch_daemon(monkeypatch, *, head_serve_exc=None, join_exc=None):
    """Replace Daemon's networking with no-ops so _run_daemon can run end to end
    in-process: no unix/tcp listeners, no signal-driven block."""

    async def fake_serve_unix(self, path):
        self.sock_path = path

    async def fake_serve_tcp(self, host, port):
        if head_serve_exc:
            raise head_serve_exc

    async def fake_join_head(self, host, port, retries=60):
        if join_exc:
            raise join_exc

    monkeypatch.setattr(_cli._daemon.Daemon, "serve_unix", fake_serve_unix)
    monkeypatch.setattr(_cli._daemon.Daemon, "serve_tcp", fake_serve_tcp)
    monkeypatch.setattr(_cli._daemon.Daemon, "join_head", fake_join_head)

    # make the blocking stop.wait() return at once, and skip signal wiring
    class InstantEvent:
        def set(self):
            pass

        async def wait(self):
            return

    monkeypatch.setattr(_cli.asyncio, "Event", InstantEvent)

    class NoSigLoop:
        def add_signal_handler(self, sig, cb):
            pass

    monkeypatch.setattr(_cli.asyncio, "get_running_loop", lambda: NoSigLoop())


def test_run_daemon_head(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _patch_daemon(monkeypatch)
    import asyncio as aio

    rc = aio.run(_cli._run_daemon(True, "n1", "1.2.3.4", 4, 6379, None))
    assert rc == 0
    # runtime file written then cleaned on shutdown
    assert not os.path.exists(_cli._runtime_path())
    out = capsys.readouterr().out
    assert "beam head started" in out and "shutting down" in out


def test_run_daemon_head_bind_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _patch_daemon(monkeypatch, head_serve_exc=OSError("addr in use"))
    import asyncio as aio

    rc = aio.run(_cli._run_daemon(True, "n1", "1.2.3.4", 4, 6379, None))
    assert rc == 1
    assert "cannot bind port" in capsys.readouterr().err


def test_run_daemon_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _patch_daemon(monkeypatch)
    import asyncio as aio

    rc = aio.run(_cli._run_daemon(False, "w1", "5.6.7.8", 2, 6379, "head:6379"))
    assert rc == 0
    assert "beam worker joined" in capsys.readouterr().out


def test_run_daemon_worker_unreachable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _patch_daemon(monkeypatch, join_exc=OSError("no route"))
    import asyncio as aio

    rc = aio.run(_cli._run_daemon(False, "w1", "5.6.7.8", 2, 6379, "head:6379"))
    assert rc == 1
    assert "head not reachable" in capsys.readouterr().err


# ---- _status ----------------------------------------------------------------


def _write_runtime(tmp_path, monkeypatch, rt):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(os.path.join(str(tmp_path), "daemon.json"), "w") as f:
        json.dump(rt, f)


class FakeStatusSock:
    """A unix socket stand-in that replays one framed status response."""

    def __init__(self, resp):
        body = json.dumps(resp).encode()
        self._buf = struct.pack(">I", len(body)) + body
        self._pos = 0
        self.connected = False

    def connect(self, addr):
        self.connected = True

    def sendall(self, data):
        pass

    def recv(self, n):
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass


def test_status_no_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))  # empty dir, no daemon.json
    assert _cli._status() == 1
    assert "no running daemon" in capsys.readouterr().err


def test_status_reports_nodes(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock", "pid": 1})
    resp = {
        "nodes": [
            {"node": "n1", "ip": "1.2.3.4", "used": 1, "ngpu": 4, "alive": True, "head": True}
        ]
    }
    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: FakeStatusSock(resp))
    assert _cli._status() == 0
    out = capsys.readouterr().out
    assert "n1" in out and "1/4" in out and "1 nodes" in out


def test_status_down_node_returns_1(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock"})
    resp = {"nodes": [{"node": "n2", "ngpu": 2, "alive": False}]}
    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: FakeStatusSock(resp))
    assert _cli._status() == 1
    assert "DOWN" in capsys.readouterr().err


def test_status_daemon_error(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock"})
    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: FakeStatusSock({"err": "kaboom"}))
    assert _cli._status() == 1
    assert "kaboom" in capsys.readouterr().err


def test_status_connect_refused(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock"})

    class Refused:
        def connect(self, addr):
            raise OSError("refused")

    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: Refused())
    assert _cli._status() == 1
    assert "cannot reach daemon" in capsys.readouterr().err


# ---- _recv ------------------------------------------------------------------


def test_recv_exact():
    s = FakeStatusSock({"x": 1})
    # first 4 bytes are the length prefix
    n = struct.unpack(">I", _cli._recv(s, 4))[0]
    body = _cli._recv(s, n)
    assert json.loads(body) == {"x": 1}


def test_recv_short_read_raises():
    class EofSock:
        def recv(self, n):
            return b""

    with pytest.raises(ConnectionError):
        _cli._recv(EofSock(), 4)


# ---- _stop ------------------------------------------------------------------


def test_stop_no_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    assert _cli._stop() == 1
    assert "no running daemon" in capsys.readouterr().err


def test_stop_signals_and_cleans(tmp_path, monkeypatch, capsys):
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    _write_runtime(tmp_path, monkeypatch, {"sock": sock, "pid": 4242})
    signals = []

    def fake_kill(pid, sig):
        signals.append((pid, sig))
        if sig == 0:
            raise OSError("process gone")  # alive check: report dead -> stop polling

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    assert _cli._stop() == 0
    assert (4242, signal.SIGTERM) in signals
    assert not os.path.exists(sock)  # stale socket cleaned
    assert "stopped pid 4242" in capsys.readouterr().out


def test_stop_already_dead_pid(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/nope.sock", "pid": 999})

    def fake_kill(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    assert _cli._stop() == 0
    assert "no live daemon" in capsys.readouterr().out


def test_stop_no_pid(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/nope.sock"})
    assert _cli._stop() == 0
    assert "no live daemon" in capsys.readouterr().out


def test_stop_escalates_to_sigkill(tmp_path, monkeypatch, capsys):
    """A process that ignores SIGTERM stays alive through the poll loop, then
    gets SIGKILL (covers the escalation branch)."""
    _write_runtime(tmp_path, monkeypatch, {"sock": "/nope.sock", "pid": 5})
    sigs = []

    def fake_kill(pid, sig):
        sigs.append(sig)  # alive for SIGTERM and every os.kill(pid, 0) probe

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)  # don't wait ~5s
    assert _cli._stop() == 0
    assert signal.SIGKILL in sigs


def test_stop_signal_oserror(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/nope.sock", "pid": 6})

    def fake_kill(pid, sig):
        raise PermissionError("not allowed")  # an OSError subclass

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    assert _cli._stop() == 0
    assert "cannot signal pid" in capsys.readouterr().err


# ---- maybe_bootstrap / bootstrap_env ----------------------------------------


def test_maybe_bootstrap_skips_on_host(monkeypatch):
    monkeypatch.delenv("BEAM_BOOTSTRAP", raising=False)
    monkeypatch.setattr(_cli.os.path, "exists", lambda p: False)  # no /.dockerenv
    called = []
    monkeypatch.setattr(_cli, "bootstrap_env", lambda: called.append(True))
    _cli.maybe_bootstrap()
    assert called == []


def test_maybe_bootstrap_runs_when_forced(monkeypatch):
    monkeypatch.setenv("BEAM_BOOTSTRAP", "1")
    called = []
    monkeypatch.setattr(_cli, "bootstrap_env", lambda: called.append(True))
    _cli.maybe_bootstrap()
    assert called == [True]


def test_bootstrap_env_into_tmp(tmp_path, monkeypatch):
    """bootstrap_env writes launchers + .pth files; point both targets into a
    tmp dir so it never touches /usr/local/bin or the system site dirs.
    `glob` is imported locally inside bootstrap_env, so patch the real glob
    module; os.path.join is shimmed to redirect the /usr/local/bin prefix."""
    import glob as glob_mod

    bindir = tmp_path / "bin"
    sitedir = tmp_path / "site"
    bindir.mkdir()
    sitedir.mkdir()

    real_join = os.path.join

    def fake_join(*parts):
        if parts and parts[0] == "/usr/local/bin":
            return real_join(str(bindir), *parts[1:])
        return real_join(*parts)

    monkeypatch.setattr(_cli.os.path, "join", fake_join)
    monkeypatch.setattr(glob_mod, "glob", lambda pat: [str(sitedir)])
    _cli.bootstrap_env()  # must not raise
    assert (bindir / "ray").exists() and (bindir / "beam").exists()
    assert (sitedir / "beam.pth").exists()
    launcher = (bindir / "ray").read_text()
    assert "-m ray" in launcher


def test_bootstrap_env_tolerates_oserror(monkeypatch):
    """Unwritable targets (the host case) must be swallowed, never raise."""
    import glob as glob_mod

    def boom(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr("builtins.open", boom)
    monkeypatch.setattr(glob_mod, "glob", lambda pat: [])
    _cli.bootstrap_env()  # no raise


# ---- fuzz -------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=65535))
def test_fuzz_start_valid_port_parses(port):
    """Any in-range integer port string parses without SystemExit at the arg
    stage (we stop before binding by stubbing run)."""
    seen = {}

    def fake_run(coro):
        coro.close()
        return 0

    def fake_run_daemon(head, node_id, ip, gpus, port_, address):
        seen["port"] = port_

        async def _c():
            return 0

        return _c()

    import unittest.mock as mock

    with (
        mock.patch.object(_cli, "maybe_bootstrap", lambda: None),
        mock.patch.object(_cli.asyncio, "run", fake_run),
        mock.patch.object(_cli, "_run_daemon", fake_run_daemon),
        mock.patch.object(_cli._daemon, "detect_gpus", lambda n: 0),
        mock.patch.object(_cli, "_local_ip", lambda: "1.1.1.1"),
    ):
        rc = _cli._start(["--head", "--port=%d" % port])
    assert rc == 0 and seen["port"] == port


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=10))
def test_fuzz_start_nonint_port_exits(garbage):
    """A non-integer --port value always exits 2, never parses to a bogus int."""
    import unittest.mock as mock

    if garbage.lstrip("-").isdigit():
        return  # this branch is for non-numeric strings only
    with mock.patch.object(_cli, "maybe_bootstrap", lambda: None):
        try:
            rc = _cli._start(["--head", "--port=%s" % garbage])
        except SystemExit as e:
            assert e.code == 2
            return
    assert rc == 2  # unknown-flag fallthrough also returns 2
