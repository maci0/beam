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
from ray import _cli

# ---- runtime dir / path -----------------------------------------------------


def test_runtime_dir_env_override(monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", "/custom/dir")
    assert _cli._runtime_dir() == "/custom/dir"
    assert _cli._runtime_path() == "/custom/dir/daemon.json"


def test_runtime_dir_default(monkeypatch):
    monkeypatch.delenv("BEAM_RUNTIME_DIR", raising=False)
    d = _cli._runtime_dir()
    assert d.endswith(".beam")


# ---- socket path length preflight -------------------------------------------


def test_check_sock_path_accepts_ordinary_paths():
    _cli._check_sock_path("/tmp/beam/daemon.sock")
    _cli._check_sock_path("/home/someone/.beam/daemon.sock")


def test_check_sock_path_rejects_overlong_path(capsys):
    """AF_UNIX caps sun_path; a deep BEAM_RUNTIME_DIR must be named, not crash."""
    long = "/" + "a" * 120 + "/daemon.sock"
    with pytest.raises(SystemExit) as e:
        _cli._check_sock_path(long)
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "AF_UNIX" in err and "BEAM_RUNTIME_DIR" in err and long in err


def test_start_refuses_overlong_runtime_dir(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path / ("d" * 80)))
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: 12345)
    with pytest.raises(SystemExit) as e:
        _cli._start(["--head"])
    assert e.value.code == 1
    assert "AF_UNIX" in capsys.readouterr().err


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

    monkeypatch.delenv("BEAM_NODE_IP", raising=False)
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: DeadSock())
    assert _cli._local_ip() == "127.0.0.1"


def test_local_ip_prefers_beam_node_ip(monkeypatch):
    monkeypatch.setenv("BEAM_NODE_IP", "10.1.2.3")
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)

    class BoomSock:
        def connect(self, addr):
            raise AssertionError("must not probe default route when env is set")

        def close(self):
            pass

    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: BoomSock())
    assert _cli._local_ip() == "10.1.2.3"


def test_local_ip_falls_back_to_vllm_host_ip(monkeypatch):
    monkeypatch.delenv("BEAM_NODE_IP", raising=False)
    monkeypatch.setenv("VLLM_HOST_IP", "10.4.5.6")
    assert _cli._local_ip() == "10.4.5.6"


# ---- _usage / main ----------------------------------------------------------


def test_usage_help_goes_to_stdout_and_exits_0(capsys):
    """`--help` is a successful request: stdout, exit 0 (git/docker/kubectl)."""
    assert _cli._usage(0) == 0
    out = capsys.readouterr()
    assert "drop-in subset of ray" in out.out and out.err == ""


def test_usage_error_goes_to_stderr_and_exits_2(capsys):
    """Bad usage is an error: stderr, exit 2."""
    assert _cli._usage(2) == 2
    out = capsys.readouterr()
    assert "drop-in subset of ray" in out.err and out.out == ""


def test_main_no_args_is_usage():
    assert _cli.main([]) == 2


def test_main_help():
    assert _cli.main(["--help"]) == 0
    assert _cli.main(["-h"]) == 0
    assert _cli.main(["help"]) == 0


def test_main_version(capsys):
    """--version prints to stdout and exits 0, like every well-behaved CLI."""
    from ray import __version__

    assert _cli.main(["--version"]) == 0
    assert _cli.main(["-V"]) == 0
    assert __version__ in capsys.readouterr().out


def test_main_unknown_command(capsys):
    assert _cli.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_main_unknown_command_suggests(capsys):
    """A near miss names a real command instead of just rejecting it."""
    assert _cli.main(["stat"]) == 2
    err = capsys.readouterr().err
    assert "did you mean" in err and ("'start'" in err or "'status'" in err)


def test_main_unknown_command_no_suggestion_when_distant(capsys):
    """A totally unrelated word gets no misleading suggestion."""
    assert _cli.main(["zzzzqqq"]) == 2
    assert "did you mean" not in capsys.readouterr().err


def test_status_rejects_stray_args(capsys):
    """`status` has no options; a stray flag must not be silently ignored."""
    assert _cli.main(["status", "--json"]) == 2
    assert "unexpected argument" in capsys.readouterr().err


def test_stop_rejects_stray_args(capsys):
    assert _cli.main(["stop", "--force"]) == 2
    assert "unexpected argument" in capsys.readouterr().err


def test_status_stop_bootstrap_help_exit_0(capsys):
    """Every command answers --help on stdout with 0, not a usage error."""
    for cmd in ("status", "stop", "bootstrap"):
        assert _cli.main([cmd, "--help"]) == 0
        assert "drop-in subset of ray" in capsys.readouterr().out


def test_bootstrap_rejects_stray_args(capsys):
    assert _cli.main(["bootstrap", "--yes"]) == 2
    assert "unexpected argument" in capsys.readouterr().err


def test_main_reports_config_error(capsys, monkeypatch):
    """A bad env var reaches the operator as one 'beam:' line and exit 2, not a
    traceback out of the middle of `ray start`."""
    from ray import _config

    def _bad_start(_args):
        raise _config.ConfigError("beam: BEAM_NUM_GPUS must be a non-negative integer, got 'x'")

    monkeypatch.setattr(_cli, "_start", _bad_start)
    assert _cli.main(["start", "--head"]) == 2
    assert "BEAM_NUM_GPUS must be" in capsys.readouterr().err


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
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli._daemon, "detect_gpus", lambda n: 0)
    # space-separated --port and --node-ip forms, plus --block
    monkeypatch.delenv("BEAM_NODE_IP", raising=False)
    rc = _cli._start(["--head", "--block", "--port", "6400", "--node-ip", "3.3.3.3"])
    assert rc == 0 and captured["node_ip"] == "3.3.3.3"
    assert os.environ.get("BEAM_NODE_IP") == "3.3.3.3"  # workers inherit advertised IP


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
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli._daemon, "detect_gpus", lambda n: 0)
    _cli._start(["--head", "--node-ip=4.4.4.4"])
    assert captured["node_ip"] == "4.4.4.4"


# ---- process-level contract (real `python -m ray`, exit code + stream) -------

_PY_DIR = os.path.join(os.path.dirname(__file__), "..", "python")


def _run_ray(*args):
    """Run `python -m ray` as a real process; return (rc, stdout, stderr)."""
    import subprocess

    env = dict(os.environ, PYTHONPATH=_PY_DIR, BEAM_RUNTIME_DIR="/nonexistent-beam-cli-test")
    p = subprocess.run(
        [sys.executable, "-m", "ray", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return p.returncode, p.stdout, p.stderr


@pytest.mark.parametrize("flag", ["--help", "-h", "help"])
def test_process_help_is_stdout_exit_0(flag):
    """`ray --help` must be pipeable: text on stdout, nothing on stderr, exit 0."""
    rc, out, err = _run_ray(flag)
    assert rc == 0
    assert "usage:" in out and "ray start --head" in out
    assert err == ""


@pytest.mark.parametrize("cmd", ["start", "status", "stop", "bootstrap"])
def test_process_subcommand_help_exit_0(cmd):
    rc, out, err = _run_ray(cmd, "--help")
    assert rc == 0, (out, err)
    assert "usage:" in out and err == ""


def test_process_version_stdout_exit_0():
    from ray import __version__

    rc, out, err = _run_ray("--version")
    assert rc == 0 and __version__ in out and err == ""


def test_process_no_args_is_usage_error_on_stderr():
    """No args = missing command: stderr, exit 2, and stdout stays clean for pipes."""
    rc, out, err = _run_ray()
    assert rc == 2 and out == "" and "usage:" in err


def test_process_unknown_command_exit_2_stderr():
    rc, out, err = _run_ray("frobnicate")
    assert rc == 2 and out == "" and "unknown command" in err


def test_process_start_bad_flag_value_exit_2_not_crash():
    """A bad flag value exits 2 cleanly (no traceback), with the flag named."""
    rc, out, err = _run_ray("start", "--head", "--port", "notaport")
    assert rc == 2 and out == "" and "--port expects an integer" in err
    assert "Traceback" not in err


# ---- _start arg parsing -----------------------------------------------------


def test_start_needs_head_or_address(capsys):
    assert _cli._start([]) == 2
    assert "need --head or --address" in capsys.readouterr().err


def test_start_bad_port_exits_2(capsys):
    """A bad value returns 2 like every other error path (no SystemExit)."""
    assert _cli._start(["--head", "--port", "notaport"]) == 2
    assert "--port expects an integer" in capsys.readouterr().err


def test_start_port_missing_value_exits_2(capsys):
    assert _cli._start(["--head", "--port"]) == 2
    assert "--port expects a value" in capsys.readouterr().err


def test_start_unknown_flag(capsys):
    assert _cli._start(["--head", "--bogus"]) == 2
    assert "unknown flag" in capsys.readouterr().err


def test_start_negative_num_gpus(capsys):
    assert _cli._start(["--head", "--num-gpus", "-1"]) == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_start_bad_num_gpus_value(capsys):
    assert _cli._start(["--head", "--num-gpus", "x"]) == 2
    assert "--num-gpus expects an integer" in capsys.readouterr().err


def test_start_bad_beam_num_gpus_env(capsys, monkeypatch):
    """BEAM_NUM_GPUS is the env form of --num-gpus: a typo must fail with a
    message, not a ValueError traceout from detect_gpus."""
    monkeypatch.setenv("BEAM_NUM_GPUS", "abc")
    assert _cli._start(["--head"]) == 2
    assert "BEAM_NUM_GPUS must be an integer" in capsys.readouterr().err


def test_start_negative_beam_num_gpus_env(capsys, monkeypatch):
    monkeypatch.setenv("BEAM_NUM_GPUS", "-2")
    assert _cli._start(["--head"]) == 2
    assert "BEAM_NUM_GPUS must be >= 0" in capsys.readouterr().err


def test_start_beam_num_gpus_env_applies(monkeypatch):
    captured = {}

    def fake_run(coro):
        coro.close()
        return 0

    def fake_run_daemon(head, node_id, ip, gpus, port, address):
        captured["gpus"] = gpus

        async def _c():
            return 0

        return _c()

    monkeypatch.setenv("BEAM_NUM_GPUS", "3")
    monkeypatch.setattr(_cli, "maybe_bootstrap", lambda: None)
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli, "_local_ip", lambda: "1.1.1.1")
    assert _cli._start(["--head"]) == 0
    assert captured["gpus"] == 3


def test_start_bad_address_port(capsys):
    assert _cli._start(["--address", "host:notnum"]) == 2
    assert "port must be numeric" in capsys.readouterr().err


def test_start_help_returns_usage(capsys):
    assert _cli._start(["--help"]) == 0
    assert "drop-in subset of ray" in capsys.readouterr().out


def test_start_unknown_flag_suggests(capsys):
    assert _cli._start(["--head", "--prot", "1"]) == 2
    assert "did you mean '--port'" in capsys.readouterr().err


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
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
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
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(_cli.asyncio, "run", fake_run)
    monkeypatch.setattr(_cli, "_run_daemon", fake_run_daemon)
    monkeypatch.setattr(_cli._daemon, "detect_gpus", lambda n: 0)
    monkeypatch.setattr(_cli, "_local_ip", lambda: "1.1.1.1")
    _cli._start(["--address=h:6379"])
    assert captured["head"] is False and captured["address"] == "h:6379"


def test_start_refuses_if_daemon_already_running(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: 12345)
    assert _cli._start(["--head"]) == 1
    assert "already running" in capsys.readouterr().err


def test_live_daemon_pid_corrupt_json_is_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "daemon.json").write_text("{not json")
    assert _cli._live_daemon_pid() is None


def test_live_daemon_pid_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    assert _cli._live_daemon_pid() is None


def test_live_daemon_pid_dead_process(tmp_path, monkeypatch):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock", "pid": 1})

    def fake_kill(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    assert _cli._live_daemon_pid() is None


def test_live_daemon_pid_eperm_treated_live(tmp_path, monkeypatch):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock", "pid": 9})

    def fake_kill(pid, sig):
        raise PermissionError("nope")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    assert _cli._live_daemon_pid() == 9


def test_live_daemon_pid_alive(tmp_path, monkeypatch):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock", "pid": 42})
    monkeypatch.setattr(_cli.os, "kill", lambda pid, sig: None)
    assert _cli._live_daemon_pid() == 42


def test_live_daemon_pid_bad_pid_type(tmp_path, monkeypatch):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock", "pid": "nope"})
    assert _cli._live_daemon_pid() is None


def test_live_daemon_pid_zero_pid(tmp_path, monkeypatch):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock", "pid": 0})
    assert _cli._live_daemon_pid() is None


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


def test_run_daemon_exit_other_pid_leaves_claim(tmp_path, monkeypatch, capsys):
    """If claim was rewritten to another pid, exit must not delete the sock."""
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _patch_daemon(monkeypatch)
    import asyncio as aio

    orig_shutdown = _cli._daemon.Daemon.shutdown

    def rewrite_then_shutdown(self):
        path = _cli._runtime_path()
        with open(path) as f:
            rt = json.load(f)
        rt["pid"] = os.getpid() + 999
        with open(path, "w") as f:
            json.dump(rt, f)
        return orig_shutdown(self)

    monkeypatch.setattr(_cli._daemon.Daemon, "shutdown", rewrite_then_shutdown)
    rc = aio.run(_cli._run_daemon(True, "n1", "1.2.3.4", 0, 6379, None))
    assert rc == 0
    assert os.path.exists(_cli._runtime_path())


def test_run_daemon_exit_missing_claim(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _patch_daemon(monkeypatch)
    import asyncio as aio

    orig_shutdown = _cli._daemon.Daemon.shutdown

    def drop_claim_then_shutdown(self):
        try:
            os.remove(_cli._runtime_path())
        except OSError:
            pass
        return orig_shutdown(self)

    monkeypatch.setattr(_cli._daemon.Daemon, "shutdown", drop_claim_then_shutdown)
    assert aio.run(_cli._run_daemon(True, "n1", "1.2.3.4", 0, 6379, None)) == 0


def test_run_daemon_head_binds_configured_address(tmp_path, monkeypatch, capsys):
    """BEAM_BIND_ADDRESS decides which interface the unauthenticated control port
    listens on; the startup line says which one that was."""
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("BEAM_BIND_ADDRESS", "10.0.0.5")
    bound = {}
    _patch_daemon(monkeypatch)
    orig = _cli._daemon.Daemon.serve_tcp

    async def record(self, host, port):
        bound["host"] = host
        await orig(self, host, port)

    monkeypatch.setattr(_cli._daemon.Daemon, "serve_tcp", record)
    import asyncio as aio

    assert aio.run(_cli._run_daemon(True, "n1", "10.0.0.5", 4, 6379, None)) == 0
    assert bound["host"] == "10.0.0.5"
    assert "bound to 10.0.0.5" in capsys.readouterr().out


def test_run_daemon_head_bind_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _patch_daemon(monkeypatch, head_serve_exc=OSError("addr in use"))
    import asyncio as aio

    rc = aio.run(_cli._run_daemon(True, "n1", "1.2.3.4", 4, 6379, None))
    assert rc == 1
    assert "cannot bind" in capsys.readouterr().err


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


def test_stop_concurrent_live_stop_waits(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock), "stopped": 1}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def fake_kill(pid, sig):
        probes["n"] += 1
        if probes["n"] < 3:
            return  # still "alive"
        raise ProcessLookupError()  # peer stop exited

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    # peer stop died but left an abandoned hold: seize-clean leftovers
    out = capsys.readouterr().out
    assert "cleaned leftover stop state" in out
    assert not sock.exists()


def test_stop_concurrent_stop_fully_cleaned(tmp_path, monkeypatch, capsys):
    """When the peer stop finishes and removes the claim, return cleanly."""
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def fake_kill(pid, sig):
        probes["n"] += 1
        if probes["n"] >= 2:
            # peer stop cleaned the claim
            try:
                os.remove(tmp_path / "daemon.json")
            except OSError:
                pass
            raise OSError("gone")
        return None

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_concurrent_live_stop_timeout(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli.os, "kill", lambda pid, sig: None)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 1
    assert "another stop in progress" in capsys.readouterr().err


def test_stop_abandoned_stop_hold_cleans(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))

    def dead(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(_cli.os, "kill", dead)
    assert _cli._stop() == 0
    assert "cleaned leftover stop state" in capsys.readouterr().out
    assert not sock.exists()
    assert not (tmp_path / "daemon.json").exists()


def test_stop_abandoned_does_not_unlink_live_claim(tmp_path, monkeypatch, capsys):
    """After seize, if doc is no longer a stop hold, restore and report carefully."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    claim = tmp_path / "daemon.json"
    with open(claim, "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    # Stay dead for pre-seize probes; after restore, pid 99 is "live".
    state = {"n": 0}

    def kill_fn(pid, sig):
        state["n"] += 1
        if pid == 99:
            return None  # live daemon after restore
        raise ProcessLookupError()

    real_rename = os.rename

    def rename_swap(src, dst):
        real_rename(src, dst)
        # concurrent rewrite of seized doc to a non-stopping daemon claim
        with open(dst, "w") as f:
            json.dump({"pid": 99, "sock": str(sock)}, f)

    monkeypatch.setattr(_cli.os, "kill", kill_fn)
    monkeypatch.setattr(_cli.os, "rename", rename_swap)
    # Restored non-stopping claim with live pid → must not claim success
    assert _cli._stop() == 1
    err = capsys.readouterr().err
    assert "live daemon" in err or "re-run ray stop" in err
    assert claim.exists()


def test_stop_abandoned_does_not_clobber_new_claim(tmp_path, monkeypatch, capsys):
    """Restore must not rename-over a concurrent start's claim."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    claim = tmp_path / "daemon.json"
    with open(claim, "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_rename = os.rename

    def rename_and_reclaim(src, dst):
        real_rename(src, dst)
        # concurrent start claims path while we hold seized
        with open(claim, "w") as f:
            json.dump({"pid": 99, "sock": str(sock)}, f)

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "rename", rename_and_reclaim)
    # hold re-check sees live pid 55? seized still has stopping:55 but path is 99
    # After seize, doc is stopping with pid 55; kill(55) ProcessLookupError → clean
    # But path exists (new claim) → must not unlink sock
    assert _cli._stop() == 0
    assert claim.exists()
    with open(claim) as f:
        assert json.load(f)["pid"] == 99
    assert sock.exists()  # new daemon's sock preserved


def test_stop_concurrent_probe_eperm(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        _cli.os, "kill", lambda pid, sig: (_ for _ in ()).throw(PermissionError("x"))
    )
    assert _cli._stop() == 1
    assert "cannot probe stop pid" in capsys.readouterr().err


def test_stop_abandoned_cleanup_remove_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        _cli.os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError())
    )
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 0
    assert "cleaned leftover stop state" in capsys.readouterr().out


def test_stop_refuses_seize_when_claim_pid_changed(tmp_path, monkeypatch, capsys):
    """After old daemon dies, do not rename-seize a new daemon's claim."""
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    _write_runtime(tmp_path, monkeypatch, {"sock": sock, "pid": 4242})
    probes = {"n": 0}

    def fake_kill(pid, sig):
        probes["n"] += 1
        if sig == 0:
            # first alive checks for 4242: dead; claim already rewritten to 99
            if pid == 4242:
                with open(tmp_path / "daemon.json", "w") as f:
                    json.dump({"pid": 99, "sock": sock}, f)
                raise OSError("gone")
            return None  # 99 live
        return None

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "owned by another daemon" in capsys.readouterr().out
    with open(tmp_path / "daemon.json") as f:
        assert json.load(f)["pid"] == 99  # not seized


def test_stop_precheck_same_pid_stopping_live(tmp_path, monkeypatch, capsys):
    """Pre-seize re-read: same pid with stopping and live process → do not seize."""
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    _write_runtime(tmp_path, monkeypatch, {"sock": sock, "pid": 4242})
    reads = {"n": 0}

    def read_fn():
        reads["n"] += 1
        if reads["n"] == 1:
            return {"pid": 4242, "sock": sock}
        # still pid 4242 but marked stopping (another stop/reclaim race)
        return {"pid": 4242, "stopping": True, "sock": sock}

    def fake_kill(pid, sig):
        if sig == 0 and reads["n"] == 1:
            raise OSError("gone")  # daemon gone during wait
        return None  # pre-seize: pid 4242 appears live as stop hold

    monkeypatch.setattr(_cli, "_read_runtime", read_fn)
    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "owned by another daemon" in capsys.readouterr().out


def test_stop_precheck_same_pid_stopping_dead_seizes(tmp_path, monkeypatch, capsys):
    """Same pid stopping but process dead: fall through and seize."""
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    path = tmp_path / "daemon.json"
    with open(path, "w") as f:
        json.dump({"pid": 4242, "sock": sock}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    reads = {"n": 0}

    def read_fn():
        reads["n"] += 1
        if reads["n"] == 1:
            return {"pid": 4242, "sock": sock}
        return {"pid": 4242, "stopping": True, "sock": sock}

    def fake_kill(pid, sig):
        raise ProcessLookupError()  # always dead

    monkeypatch.setattr(_cli, "_read_runtime", read_fn)
    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    # After seize, hold install needs real files: mock rename of path
    real_rename = os.rename

    def rename_ok(src, dst):
        # materialize seized content
        if str(src) == str(path):
            with open(dst, "w") as f:
                json.dump({"pid": 4242, "stopping": True, "sock": sock}, f)
            try:
                os.remove(src)
            except OSError:
                pass
            return
        return real_rename(src, dst)

    monkeypatch.setattr(_cli.os, "rename", rename_ok)
    rc = _cli._stop()
    assert rc == 0


def test_stop_precheck_same_pid_stopping_eperm(tmp_path, monkeypatch, capsys):
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    _write_runtime(tmp_path, monkeypatch, {"sock": sock, "pid": 4242})
    reads = {"n": 0}
    phase = {"precheck": False}

    def read_fn():
        reads["n"] += 1
        if reads["n"] == 1:
            return {"pid": 4242, "sock": sock}
        phase["precheck"] = True
        return {"pid": 4242, "stopping": True, "sock": sock}

    def fake_kill(pid, sig):
        if phase["precheck"]:
            raise PermissionError("eperm")
        if sig == 0:
            raise OSError("gone")
        return None  # SIGTERM ok

    monkeypatch.setattr(_cli, "_read_runtime", read_fn)
    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "owned by another daemon" in capsys.readouterr().out


def test_stop_precheck_read_fails_after_death(tmp_path, monkeypatch, capsys):
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    _write_runtime(tmp_path, monkeypatch, {"sock": sock, "pid": 4242})
    reads = {"n": 0}
    real_read = _cli._read_runtime

    def read_fn():
        reads["n"] += 1
        if reads["n"] == 1:
            return real_read()
        raise OSError("gone")

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")
        return None

    monkeypatch.setattr(_cli, "_read_runtime", read_fn)
    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "stopped pid" in capsys.readouterr().out


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


def test_stop_does_not_unlink_new_daemon_files(tmp_path, monkeypatch, capsys):
    """After the old pid dies, a new daemon may rewrite daemon.json; stop must
    not delete the new runtime (atomic rename seize + restore)."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 111})
    real_rename = os.rename

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    def fake_rename(src, dst):
        # first rename seizes the old file; inject a new daemon's runtime as
        # if it claimed the path before we could read the seized content...
        # instead: after seize, rewrite doomed content to new pid and restore path
        real_rename(src, dst)
        # simulate: new daemon already owns the path under a different name
        # by writing pid 222 into the seized file so stop restores it
        with open(dst) as f:
            data = json.load(f)
        data["pid"] = 222
        with open(dst, "w") as f:
            json.dump(data, f)

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "rename", fake_rename)
    assert _cli._stop() == 0
    assert (tmp_path / "daemon.json").exists()
    assert "another daemon" in capsys.readouterr().out


def test_stop_reread_runtime_error_falls_back(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 7})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    real_rename = os.rename

    def seize_then_corrupt(src, dst):
        real_rename(src, dst)
        with open(dst, "w") as f:
            f.write("{bad")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "rename", seize_then_corrupt)
    assert _cli._stop() == 0
    assert "stopped pid 7" in capsys.readouterr().out


def test_stop_rename_missing_runtime(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/nope.sock", "pid": 3})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "rename", lambda s, d: (_ for _ in ()).throw(OSError("gone")))
    assert _cli._stop() == 0
    assert "stopped pid 3" in capsys.readouterr().out


def test_stop_restore_rename_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 111})
    real_rename = os.rename
    state = {"n": 0}

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    def fake_rename(src, dst):
        state["n"] += 1
        if state["n"] == 1:
            real_rename(src, dst)
            with open(dst) as f:
                data = json.load(f)
            data["pid"] = 222
            with open(dst, "w") as f:
                json.dump(data, f)
            return
        raise OSError("cannot restore")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "rename", fake_rename)
    assert _cli._stop() == 0
    assert "another daemon" in capsys.readouterr().out


def test_stop_leaves_sock_if_new_claim_appeared(tmp_path, monkeypatch, capsys):
    """Hold-path link fails when concurrent start already claimed; leave sock."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("live")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 111})
    real_rename = os.rename

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    def seize_then_new_claim(src, dst):
        real_rename(src, dst)
        # concurrent start claims the free path again
        with open(src, "w") as f:
            json.dump({"sock": str(sock), "pid": 222}, f)

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "rename", seize_then_new_claim)
    assert _cli._stop() == 0
    assert sock.exists() and sock.read_text() == "live"
    assert "another daemon" in capsys.readouterr().out


def test_stop_hold_link_exists_unlink_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("live")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 111})
    real_rename = os.rename

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    def seize_then_new_claim(src, dst):
        real_rename(src, dst)
        with open(src, "w") as f:
            json.dump({"sock": str(sock), "pid": 222}, f)

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "rename", seize_then_new_claim)
    monkeypatch.setattr(_cli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("y")))
    assert _cli._stop() == 0
    assert "another daemon" in capsys.readouterr().out


def test_stop_hold_oserror_does_not_remove_sock(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("keep")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 9})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("nospace")))
    assert _cli._stop() == 0
    assert sock.exists() and sock.read_text() == "keep"
    assert "stopped pid 9" in capsys.readouterr().out


def test_stop_hold_oserror_cleanup_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("keep")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 9})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("nospace")))
    monkeypatch.setattr(_cli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("y")))
    assert _cli._stop() == 0


def test_stop_hold_ownership_lost_before_sock_unlink(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("live")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 9})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    real_link = os.link

    def link_then_steal(src, dst):
        real_link(src, dst)
        # concurrent start rewrites claim
        with open(dst, "w") as f:
            json.dump({"pid": 999, "stopping": False}, f)

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "link", link_then_steal)
    assert _cli._stop() == 0
    assert sock.exists()
    assert "another daemon" in capsys.readouterr().out


def test_stop_hold_ownership_lost_doomed_remove_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("live")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 9})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    real_link = os.link

    def link_then_steal(src, dst):
        real_link(src, dst)
        with open(dst, "w") as f:
            json.dump({"pid": 999, "stopping": False}, f)

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "link", link_then_steal)
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 0


def test_stop_hold_claim_unreadable_after_hold(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 9})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    real_link = os.link

    def link_then_corrupt(src, dst):
        real_link(src, dst)
        with open(dst, "w") as f:
            f.write("{bad")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "link", link_then_corrupt)
    assert _cli._stop() == 0
    assert "stopped pid 9" in capsys.readouterr().out


def test_stop_hold_unreadable_doomed_remove_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 9})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    real_link = os.link

    def link_then_corrupt(src, dst):
        real_link(src, dst)
        with open(dst, "w") as f:
            f.write("{bad")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(_cli.os, "link", link_then_corrupt)
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 0


def test_stop_hold_success_unlink_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 9})

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("gone")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    real_unlink = os.unlink

    def flaky_unlink(p):
        if "stophold" in str(p):
            raise OSError("busy")
        return real_unlink(p)

    monkeypatch.setattr(_cli.os, "unlink", flaky_unlink)
    assert _cli._stop() == 0
    assert "stopped pid 9" in capsys.readouterr().out


def test_claim_runtime_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    rt = {"sock": "/s", "node": "n", "head": True, "pid": 1}
    assert _cli._claim_runtime(rt) is None
    assert (tmp_path / "daemon.json").exists()
    # second claim with live pid refused
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: 1)
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 2})
    assert err is not None and "already running" in err


def test_claim_runtime_replaces_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _write_runtime(tmp_path, monkeypatch, {"sock": "/s", "pid": 9})
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    # The seize path kill(0)s the pid out of the file. Say it is dead rather
    # than depending on whether pid 9 happens to exist on the host (it does
    # inside a CI container, where kill(0) raises EPERM and reads as live).
    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    assert _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3}) is None


def test_claim_runtime_stale_rename_fails(tmp_path, monkeypatch):
    """Stale reclaim uses rename-seize; if rename fails, retry then give up."""
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _write_runtime(tmp_path, monkeypatch, {"sock": "/s", "pid": 9})
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(
        _cli.os,
        "link",
        lambda src, dst: (_ for _ in ()).throw(FileExistsError()),
    )
    monkeypatch.setattr(
        _cli.os,
        "rename",
        lambda s, d: (_ for _ in ()).throw(OSError("busy")),
    )
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3})
    assert err is not None and "cannot claim" in err


def test_claim_runtime_seized_still_live(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _write_runtime(tmp_path, monkeypatch, {"sock": "/s", "pid": 42})
    # first live check says dead so we try seize; kill(0) then says live
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    real_link = os.link
    links = {"n": 0}

    def link_fn(src, dst):
        links["n"] += 1
        # first link is claim attempt (tmp → path): conflict
        if links["n"] == 1:
            raise FileExistsError()
        # restore link(seized → path): allow
        return real_link(src, dst)

    monkeypatch.setattr(_cli.os, "link", link_fn)
    monkeypatch.setattr(_cli.os, "kill", lambda pid, sig: None)  # process "alive"
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3})
    assert err is not None and "already running" in err
    assert (tmp_path / "daemon.json").exists()  # restored


def test_claim_runtime_seized_active_stop_hold(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 42, "stopping": True}, f)
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(_cli.os, "link", lambda src, dst: (_ for _ in ()).throw(FileExistsError()))
    monkeypatch.setattr(_cli.os, "kill", lambda pid, sig: None)  # stop still alive
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3})
    assert err is not None and "already running" in err


def test_claim_runtime_seized_abandoned_stop_hold(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 42, "stopping": True}, f)
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    links = {"n": 0}
    real_link = os.link

    def flaky_link(src, dst):
        links["n"] += 1
        if links["n"] == 1:
            raise FileExistsError()
        return real_link(src, dst)

    monkeypatch.setattr(_cli.os, "link", flaky_link)

    def dead(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(_cli.os, "kill", dead)
    assert _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3}) is None


def test_claim_runtime_seized_eperm_treated_live(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _write_runtime(tmp_path, monkeypatch, {"sock": "/s", "pid": 42})
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(
        _cli.os,
        "link",
        lambda src, dst: (_ for _ in ()).throw(FileExistsError()),
    )

    def eperm(pid, sig):
        raise PermissionError("nope")

    monkeypatch.setattr(_cli.os, "kill", eperm)
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3})
    assert err is not None and "already running" in err


def test_claim_runtime_seized_stop_hold_eperm(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 42, "stopping": True}, f)
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    monkeypatch.setattr(_cli.os, "link", lambda src, dst: (_ for _ in ()).throw(FileExistsError()))
    monkeypatch.setattr(
        _cli.os, "kill", lambda pid, sig: (_ for _ in ()).throw(PermissionError("x"))
    )
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3})
    assert err is not None and "already running" in err


def test_claim_runtime_seized_dead_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _write_runtime(tmp_path, monkeypatch, {"sock": "/s", "pid": 42})
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    links = {"n": 0}
    real_link = os.link

    def flaky_link(src, dst):
        links["n"] += 1
        if links["n"] == 1:
            raise FileExistsError()
        return real_link(src, dst)

    monkeypatch.setattr(_cli.os, "link", flaky_link)

    def dead(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(_cli.os, "kill", dead)
    assert _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3}) is None


def test_claim_runtime_seized_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "daemon.json").write_text("{bad")
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    links = {"n": 0}
    real_link = os.link

    def flaky_link(src, dst):
        links["n"] += 1
        if links["n"] == 1:
            raise FileExistsError()
        return real_link(src, dst)

    monkeypatch.setattr(_cli.os, "link", flaky_link)
    assert _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3}) is None


def test_claim_runtime_seized_live_restore_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _write_runtime(tmp_path, monkeypatch, {"sock": "/s", "pid": 42})
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    real_rename = os.rename
    n = {"c": 0}

    def rename_once(src, dst):
        n["c"] += 1
        if n["c"] == 1:
            return real_rename(src, dst)
        raise OSError("cannot restore")

    monkeypatch.setattr(_cli.os, "rename", rename_once)
    monkeypatch.setattr(_cli.os, "link", lambda src, dst: (_ for _ in ()).throw(FileExistsError()))
    monkeypatch.setattr(_cli.os, "kill", lambda pid, sig: None)
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3})
    assert err is not None and "already running" in err


def test_claim_runtime_seized_remove_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    _write_runtime(tmp_path, monkeypatch, {"sock": "/s", "pid": 42})
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    links = {"n": 0}
    real_link = os.link

    def flaky_link(src, dst):
        links["n"] += 1
        if links["n"] == 1:
            raise FileExistsError()
        return real_link(src, dst)

    monkeypatch.setattr(_cli.os, "link", flaky_link)

    def dead(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(_cli.os, "kill", dead)
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("busy")))
    # still succeeds overall (remove of seized is best-effort)
    assert _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 3}) is None


def test_claim_runtime_write_failure_cleans(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    try:
        _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 1})
        raise AssertionError("expected OSError")
    except OSError:
        pass
    assert not (tmp_path / "daemon.json").exists()


def test_claim_runtime_exhausted_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    # path always "exists" for link; remove is a no-op so retries exhaust
    monkeypatch.setattr(
        _cli.os,
        "link",
        lambda src, dst: (_ for _ in ()).throw(FileExistsError()),
    )
    monkeypatch.setattr(_cli.os, "remove", lambda p: None)
    err = _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 1})
    assert err is not None and "cannot claim" in err


def test_claim_runtime_tmp_unlink_failure_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_unlink = os.unlink
    calls = {"n": 0}

    def flaky_unlink(p):
        calls["n"] += 1
        if str(p).endswith(".tmp." + str(os.getpid())) or ".tmp." in str(p):
            raise OSError("busy tmp")
        return real_unlink(p)

    monkeypatch.setattr(_cli.os, "unlink", flaky_unlink)
    assert _cli._claim_runtime({"sock": "/s", "node": "n", "head": True, "pid": 1}) is None
    assert (tmp_path / "daemon.json").exists()


def test_run_daemon_claim_conflict(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli, "_claim_runtime", lambda rt: "beam: daemon already running (pid 1)\n")
    import asyncio as aio

    rc = aio.run(_cli._run_daemon(True, "n1", "1.2.3.4", 0, 6379, None))
    assert rc == 1
    assert "already running" in capsys.readouterr().err


def test_status_framing_error(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/x.sock"})

    class BadSock:
        def connect(self, addr):
            pass

        def sendall(self, b):
            pass

        def recv(self, n):
            return b""  # short read -> ConnectionError

        def close(self):
            pass

    monkeypatch.setattr(_cli.socket, "socket", lambda *a, **k: BadSock())
    assert _cli._status() == 1
    assert "cannot reach daemon" in capsys.readouterr().err


def test_stop_no_pid(tmp_path, monkeypatch, capsys):
    _write_runtime(tmp_path, monkeypatch, {"sock": "/nope.sock"})
    assert _cli._stop() == 0
    assert "no live daemon" in capsys.readouterr().out


def test_stop_escalates_to_sigkill(tmp_path, monkeypatch, capsys):
    """A process that ignores SIGTERM stays alive through the poll loop, then
    gets SIGKILL (covers the escalation branch)."""
    _write_runtime(tmp_path, monkeypatch, {"sock": "/nope.sock", "pid": 5})
    sigs = []
    probes = {"n": 0}

    def fake_kill(pid, sig):
        sigs.append(sig)
        if sig == 0:
            probes["n"] += 1
            # stay "alive" for the SIGTERM poll (50) + a few post-SIGKILL probes
            if probes["n"] > 55:
                raise OSError("gone")

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)  # don't wait ~5s
    assert _cli._stop() == 0
    assert signal.SIGKILL in sigs


def test_stop_sigkill_still_alive_leaves_files(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 5})

    def always_alive(pid, sig):
        pass  # kill(0) never fails

    monkeypatch.setattr(_cli.os, "kill", always_alive)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 1
    assert sock.exists()
    assert "still alive" in capsys.readouterr().err


def test_stop_signal_oserror(tmp_path, monkeypatch, capsys):
    """EPERM on signal: leave runtime files so the live daemon stays findable."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("")
    _write_runtime(tmp_path, monkeypatch, {"sock": str(sock), "pid": 6})

    def fake_kill(pid, sig):
        raise PermissionError("not allowed")  # an OSError subclass

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    assert _cli._stop() == 1
    assert "cannot signal pid" in capsys.readouterr().err
    assert sock.exists()  # runtime files not unlinked under a live, unsignalable pid
    assert (tmp_path / "daemon.json").exists()


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
        mock.patch.object(_cli, "_live_daemon_pid", lambda: None),
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

    # Accept only pure non-integers (no trailing space that int() also rejects
    # after strip in some paths; exclude anything isdigit after strip).
    token = garbage.strip().lstrip("-")
    if not garbage.strip() or token.isdigit():
        return
    with (
        mock.patch.object(_cli, "maybe_bootstrap", lambda: None),
        mock.patch.object(_cli, "_live_daemon_pid", lambda: None),
    ):
        try:
            rc = _cli._start(["--head", "--port=%s" % garbage])
        except SystemExit as e:
            assert e.code == 2
            return
    assert rc == 2  # unknown-flag fallthrough also returns 2


def test_stop_concurrent_peer_cleaned_claim(tmp_path, monkeypatch, capsys):
    """Peer stop finishes and leaves a non-stopping claim -> done."""
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def fake_kill(pid, sig):
        probes["n"] += 1
        if probes["n"] >= 2:
            with open(tmp_path / "daemon.json", "w") as f:
                json.dump({"pid": 99, "sock": "/x"}, f)  # new daemon, no stopping
            raise ProcessLookupError()
        return None

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_concurrent_peer_different_hold_pid(tmp_path, monkeypatch, capsys):
    """After peer dies, a different stop hold is present -> leave it."""
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def fake_kill(pid, sig):
        probes["n"] += 1
        if probes["n"] >= 2:
            with open(tmp_path / "daemon.json", "w") as f:
                json.dump({"pid": 77, "stopping": True, "sock": "/x"}, f)
            raise ProcessLookupError()
        return None

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_rename_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "rename", lambda s, d: (_ for _ in ()).throw(OSError("busy")))
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_restore_rename_fails(tmp_path, monkeypatch, capsys):
    """Seized doc is not a stop hold; restore rename fails still returns 0."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    claim = tmp_path / "daemon.json"
    with open(claim, "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_rename = os.rename
    renames = {"n": 0}

    def rename_then_fail_restore(src, dst):
        renames["n"] += 1
        if renames["n"] == 1:
            real_rename(src, dst)
            with open(dst, "w") as f:
                json.dump({"pid": 99, "sock": str(sock)}, f)
            return
        raise OSError("restore busy")

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "rename", rename_then_fail_restore)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_hold_eperm(tmp_path, monkeypatch, capsys):
    """EPERM probing seized hold pid: restore claim, report in progress."""
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_seq(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            raise ProcessLookupError()  # initial abandoned check
        raise PermissionError("eperm")  # re-check after seize

    monkeypatch.setattr(_cli.os, "kill", kill_seq)
    assert _cli._stop() == 1
    assert "another stop in progress" in capsys.readouterr().err
    assert (tmp_path / "daemon.json").exists()


def test_stop_abandoned_hold_still_live(tmp_path, monkeypatch, capsys):
    """Seized hold pid becomes live again: restore and leave it."""
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_seq(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            raise ProcessLookupError()
        return None  # live on re-check

    monkeypatch.setattr(_cli.os, "kill", kill_seq)
    assert _cli._stop() == 1
    assert "another stop in progress" in capsys.readouterr().err
    assert (tmp_path / "daemon.json").exists()


def test_stop_abandoned_eperm_restore_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_seq(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            raise ProcessLookupError()
        raise PermissionError("eperm")

    monkeypatch.setattr(_cli.os, "kill", kill_seq)
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("nope")))
    assert _cli._stop() == 1


def test_stop_abandoned_live_restore_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_seq(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            raise ProcessLookupError()
        return None

    monkeypatch.setattr(_cli.os, "kill", kill_seq)
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("nope")))
    assert _cli._stop() == 1


def test_stop_abandoned_restore_file_exists(tmp_path, monkeypatch, capsys):
    """link restore hits FileExistsError: drop seized, leave new claim."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_rename = os.rename

    def kill_fn(pid, sig):
        # stay dead for pre-seize probes; after rename path has pid 99 live
        if pid == 99:
            return None
        raise ProcessLookupError()

    def rename_then_reclaim(src, dst):
        real_rename(src, dst)
        with open(tmp_path / "daemon.json", "w") as f:
            json.dump({"pid": 99, "sock": str(sock)}, f)

    monkeypatch.setattr(_cli.os, "kill", kill_fn)
    monkeypatch.setattr(_cli.os, "rename", rename_then_reclaim)
    assert _cli._stop() == 1
    err = capsys.readouterr().err
    assert "in progress" in err or "live" in err or "owned" in err
    with open(tmp_path / "daemon.json") as f:
        assert json.load(f)["pid"] == 99


def test_stop_abandoned_restore_exists_unlink_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_seq(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            raise ProcessLookupError()
        return None

    monkeypatch.setattr(_cli.os, "kill", kill_seq)
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(FileExistsError()))
    monkeypatch.setattr(_cli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 1


def test_stop_abandoned_restore_oserror_unlink_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_seq(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            raise ProcessLookupError()
        return None

    monkeypatch.setattr(_cli.os, "kill", kill_seq)
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("link")))
    monkeypatch.setattr(_cli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("un")))
    assert _cli._stop() == 1


def test_stop_abandoned_path_reclaimed_remove_seized_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_rename = os.rename

    def rename_and_reclaim(src, dst):
        real_rename(src, dst)
        with open(tmp_path / "daemon.json", "w") as f:
            json.dump({"pid": 99, "sock": str(sock)}, f)

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "rename", rename_and_reclaim)
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_bad_seized_json(tmp_path, monkeypatch, capsys):
    """Corrupt seized file: still best-effort clean sock/seized."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    claim = tmp_path / "daemon.json"
    with open(claim, "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_rename = os.rename

    def rename_corrupt(src, dst):
        real_rename(src, dst)
        with open(dst, "w") as f:
            f.write("{not json")

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "rename", rename_corrupt)
    assert _cli._stop() == 0
    assert "cleaned leftover stop state" in capsys.readouterr().out


def test_stop_abandoned_reclaim_link_exists(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    links = {"n": 0}

    def link_exists(src, dst):
        links["n"] += 1
        raise FileExistsError()

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", link_exists)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out
    assert links["n"] >= 1


def test_stop_abandoned_reclaim_link_exists_unlink_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(FileExistsError()))
    monkeypatch.setattr(_cli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 0


def test_stop_abandoned_reclaim_link_oserror(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("e")))
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_reclaim_link_oserror_unlink_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("e")))
    monkeypatch.setattr(_cli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 0


def test_stop_abandoned_reclaim_then_not_stopping(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_link = os.link

    def link_then_mutate(src, dst):
        real_link(src, dst)
        with open(dst, "w") as f:
            json.dump({"pid": 99, "sock": str(sock)}, f)  # no stopping

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", link_then_mutate)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_reclaim_ownership_lost_remove_fails(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_link = os.link

    def link_then_mutate(src, dst):
        real_link(src, dst)
        with open(dst, "w") as f:
            json.dump({"pid": 99, "stopping": True, "sock": str(sock)}, f)

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", link_then_mutate)
    monkeypatch.setattr(_cli.os, "remove", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_reclaim_then_bad_json(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_link = os.link

    def link_corrupt(src, dst):
        real_link(src, dst)
        with open(dst, "w") as f:
            f.write("{bad")

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", link_corrupt)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_hold_finally_unlink_fails(tmp_path, monkeypatch, capsys):
    """finally unlink of hold tmp fails: still clean sock/claim."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_unlink = os.unlink
    unlinks = {"n": 0}

    def unlink_flaky(p):
        unlinks["n"] += 1
        if "stopclean" in str(p):
            raise OSError("busy")
        return real_unlink(p)

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "unlink", unlink_flaky)
    assert _cli._stop() == 0
    assert "cleaned leftover stop state" in capsys.readouterr().out


def test_link_restore_success_and_unlink_fail(tmp_path, monkeypatch):
    path = str(tmp_path / "daemon.json")
    seized = path + ".seized"
    with open(seized, "w") as f:
        f.write("{}")
    real_unlink = os.unlink
    unlinks = {"n": 0}

    def unlink_once(p):
        unlinks["n"] += 1
        if unlinks["n"] == 1:
            raise OSError("x")
        return real_unlink(p)

    monkeypatch.setattr(_cli.os, "unlink", unlink_once)
    assert _cli._link_restore(seized, path) is True
    assert os.path.exists(path)


def test_link_restore_exists_unlink_fail(tmp_path, monkeypatch):
    path = str(tmp_path / "daemon.json")
    seized = path + ".seized"
    with open(path, "w") as f:
        f.write('{"pid":1}')
    with open(seized, "w") as f:
        f.write("{}")
    monkeypatch.setattr(_cli.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert _cli._link_restore(seized, path) is False


def test_link_restore_oserror_keeps_seized(tmp_path, monkeypatch):
    """Non-FileExistsError on link must not destroy the last claim copy."""
    path = str(tmp_path / "daemon.json")
    seized = path + ".seized"
    with open(seized, "w") as f:
        f.write("{}")
    monkeypatch.setattr(_cli.os, "link", lambda s, d: (_ for _ in ()).throw(OSError("link")))
    assert _cli._link_restore(seized, path) is False
    assert os.path.exists(seized)


def test_stop_concurrent_peer_left_live_daemon(tmp_path, monkeypatch, capsys):
    """Peer stop finished; non-stopping claim with live pid → exit 1."""
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_fn(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            return None  # concurrent stop live
        if probes["n"] == 2:
            # peer stop exits; rewrite claim as live daemon
            with open(tmp_path / "daemon.json", "w") as f:
                json.dump({"pid": 99, "sock": "/x"}, f)
            raise ProcessLookupError()
        return None  # pid 99 live for _live_daemon_pid

    monkeypatch.setattr(_cli.os, "kill", kill_fn)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 1
    assert "still running" in capsys.readouterr().err


def test_stop_concurrent_different_hold_live(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_fn(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            return None
        if probes["n"] == 2:
            with open(tmp_path / "daemon.json", "w") as f:
                json.dump({"pid": 77, "stopping": True, "sock": "/x"}, f)
            raise ProcessLookupError()
        return None  # 77 live

    monkeypatch.setattr(_cli.os, "kill", kill_fn)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 1
    assert "another stop in progress" in capsys.readouterr().err


def test_stop_concurrent_different_hold_eperm(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    probes = {"n": 0}

    def kill_fn(pid, sig):
        probes["n"] += 1
        if probes["n"] == 1:
            return None
        if probes["n"] == 2:
            with open(tmp_path / "daemon.json", "w") as f:
                json.dump({"pid": 77, "stopping": True, "sock": "/x"}, f)
            raise ProcessLookupError()
        raise PermissionError("eperm")

    monkeypatch.setattr(_cli.os, "kill", kill_fn)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 1


def test_stop_abandoned_precheck_not_stopping_live(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    reads = {"n": 0}
    real_read = _cli._read_runtime

    def read_fn():
        reads["n"] += 1
        if reads["n"] == 1:
            return real_read()
        return {"pid": 99, "sock": "/x"}  # no stopping on pre-seize re-read

    monkeypatch.setattr(_cli, "_read_runtime", read_fn)
    monkeypatch.setattr(
        _cli.os,
        "kill",
        lambda p, s: None if p == 99 else (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert _cli._stop() == 1
    assert "still running" in capsys.readouterr().err


def test_stop_abandoned_precheck_not_stopping_no_live(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    reads = {"n": 0}

    def read_fn():
        reads["n"] += 1
        if reads["n"] == 1:
            return {"pid": 55, "stopping": True, "sock": "/x"}
        return {"pid": 0, "sock": "/x"}  # not stopping, no live pid

    monkeypatch.setattr(_cli, "_read_runtime", read_fn)
    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli, "_live_daemon_pid", lambda: None)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_precheck_read_fails(tmp_path, monkeypatch, capsys):
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": "/x"}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    reads = {"n": 0}

    def read_fn():
        reads["n"] += 1
        if reads["n"] == 1:
            return {"pid": 55, "stopping": True, "sock": "/x"}
        raise OSError("gone")

    monkeypatch.setattr(_cli, "_read_runtime", read_fn)
    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_doc_live_restore(tmp_path, monkeypatch, capsys):
    """After seize, hold pid is live → restore and exit 1."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    # Pre-check: dead. Post-seize hold check: live.
    state = {"phase": "pre"}

    def kill_fn(pid, sig):
        if state["phase"] == "pre":
            raise ProcessLookupError()
        return None

    real_rename = os.rename

    def rename_flip(src, dst):
        real_rename(src, dst)
        state["phase"] = "post"

    monkeypatch.setattr(_cli.os, "kill", kill_fn)
    monkeypatch.setattr(_cli.os, "rename", rename_flip)
    assert _cli._stop() == 1
    assert "in progress" in capsys.readouterr().err


def test_stop_abandoned_doc_eperm_restore(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    state = {"phase": "pre"}

    def kill_fn(pid, sig):
        if state["phase"] == "pre":
            raise ProcessLookupError()
        raise PermissionError("eperm")

    real_rename = os.rename

    def rename_flip(src, dst):
        real_rename(src, dst)
        state["phase"] = "post"

    monkeypatch.setattr(_cli.os, "kill", kill_fn)
    monkeypatch.setattr(_cli.os, "rename", rename_flip)
    assert _cli._stop() == 1
    assert "in progress" in capsys.readouterr().err


def test_stop_abandoned_toctou_ownership_lost(tmp_path, monkeypatch, capsys):
    """Between hold install and final unlink, claim is stolen."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    real_open = open
    real_link = os.link
    links = {"n": 0}

    def link_then_steal(src, dst):
        real_link(src, dst)
        links["n"] += 1
        if links["n"] >= 1 and "stopclean" in str(src):
            with real_open(dst, "w") as f:
                json.dump({"pid": 99, "sock": str(sock)}, f)

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(_cli.os, "link", link_then_steal)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_second_toctou_foreign(tmp_path, monkeypatch, capsys):
    """Second TOCTOU check on abandoned path sees foreign claim."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    path = tmp_path / "daemon.json"
    with open(path, "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    import builtins

    real_open = builtins.open
    path_s = str(path)
    own_reads = {"n": 0}

    def open_spy(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        f = real_open(file, *a, **k)
        if str(file) == path_s and "r" in str(mode):
            own_reads["n"] += 1
            # After the stopclean hold is in place, the later ownership
            # re-check reads a foreign claim.
            if own_reads["n"] >= 3:
                f.close()

                class Fake:
                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        pass

                    def read(self, *a):
                        return json.dumps({"pid": 1, "sock": str(sock)})

                return Fake()
        return f

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(builtins, "open", open_spy)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_abandoned_second_toctou_bad_json(tmp_path, monkeypatch, capsys):
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    path = tmp_path / "daemon.json"
    with open(path, "w") as f:
        json.dump({"pid": 55, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    import builtins

    real_open = builtins.open
    path_s = str(path)
    own_reads = {"n": 0}

    def open_spy(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if str(file) == path_s and "r" in str(mode):
            own_reads["n"] += 1
            if own_reads["n"] >= 3:

                class Fake:
                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        pass

                    def read(self, *a):
                        return "{bad"

                return Fake()
        return real_open(file, *a, **k)

    monkeypatch.setattr(_cli.os, "kill", lambda p, s: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(builtins, "open", open_spy)
    assert _cli._stop() == 0
    assert "another stop finished" in capsys.readouterr().out


def test_stop_normal_second_ownership_check_fails(tmp_path, monkeypatch, capsys):
    """Normal stop: second TOCTOU ownership check sees foreign claim."""
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    _write_runtime(tmp_path, monkeypatch, {"sock": sock, "pid": 4242})
    path = str(tmp_path / "daemon.json")

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("process gone")

    import builtins

    open_count = {"n": 0}
    real_builtins_open = builtins.open

    def open_spy(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        f = real_builtins_open(file, *a, **k)
        if str(file) == path and "r" in str(mode):
            open_count["n"] += 1
            if open_count["n"] >= 3:
                f.close()

                class Fake:
                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        pass

                    def read(self, *a):
                        return json.dumps({"pid": 1, "sock": sock})

                return Fake()
        return f

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(builtins, "open", open_spy)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    rc = _cli._stop()
    assert rc == 0
    out = capsys.readouterr().out
    assert "owned by another" in out or "stopped pid" in out


def test_stop_normal_second_ownership_bad_json(tmp_path, monkeypatch, capsys):
    sock = os.path.join(str(tmp_path), "daemon.sock")
    open(sock, "w").close()
    _write_runtime(tmp_path, monkeypatch, {"sock": sock, "pid": 4242})
    path = str(tmp_path / "daemon.json")

    def fake_kill(pid, sig):
        if sig == 0:
            raise OSError("process gone")

    import builtins

    open_count = {"n": 0}
    real_builtins_open = builtins.open

    def open_spy(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if str(file) == path and "r" in str(mode):
            open_count["n"] += 1
            if open_count["n"] >= 3:

                class Fake:
                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        pass

                    def read(self, *a):
                        return "{not-json"

                return Fake()
        return real_builtins_open(file, *a, **k)

    monkeypatch.setattr(_cli.os, "kill", fake_kill)
    monkeypatch.setattr(builtins, "open", open_spy)
    import time as time_mod

    monkeypatch.setattr(time_mod, "sleep", lambda s: None)
    assert _cli._stop() == 0
    assert "stopped pid" in capsys.readouterr().out


def test_stop_abandoned_zero_hold_pid(tmp_path, monkeypatch, capsys):
    """stopping hold with pid 0: skip live probe, clean seized."""
    sock = tmp_path / "daemon.sock"
    sock.write_text("x")
    with open(tmp_path / "daemon.json", "w") as f:
        json.dump({"pid": 0, "stopping": True, "sock": str(sock)}, f)
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    # stop_pid is 0/falsy so we skip the live concurrent wait branch
    assert _cli._stop() == 0
    assert "cleaned leftover stop state" in capsys.readouterr().out
    assert not sock.exists()
