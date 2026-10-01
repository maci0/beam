"""`ray` / `beam` command line: start/status/stop/bootstrap.

`ray start` runs the daemon (this process blocks, like `ray start --block`).
Everything is Python now; there is no separate binary.
"""

from __future__ import annotations  # keep `X | None` valid on py3.9

import asyncio
import difflib
import json
import os
import signal
import socket
import struct
import sys
import time
from collections.abc import Callable, Sequence

from . import _config, _daemon, _proto, _runtime

# ---- tuning ----

# Wait budgets when stopping a daemon: poll for a clean exit, then after SIGKILL.
_EXIT_POLLS = 50
_EXIT_POLL_INTERVAL = 0.1
_SIGKILL_POLLS = 20
_SIGKILL_POLL_INTERVAL = 0.05
# sockaddr_un.sun_path is a fixed 108-byte field: a socket path holds at most
# 107 bytes on Linux (103 on macOS). Capped a little lower so the check below
# also catches paths that only blow up once something appends to them.
_SOCK_PATH_MAX = 100


def _check_sock_path(sock: str) -> None:
    """Refuse a unix-socket path the kernel cannot hold, and say how to fix it.

    sockaddr_un.sun_path is a fixed 108-byte field, so the path plus its NUL
    must fit in 107 bytes on Linux, 103 on macOS. Past that, bind/connect fails
    as a bare OSError("AF_UNIX path too long") from deep inside
    socket.connect(), long after the caller thought its setup had worked.
    """
    n = len(sock.encode())
    if n <= _SOCK_PATH_MAX:
        return
    sys.stderr.write(
        "beam: the daemon socket path is %d bytes, over the %d-byte AF_UNIX limit:\n"
        "  %s\n"
        "Point BEAM_RUNTIME_DIR at a shorter directory (it holds only daemon.sock\n"
        "and daemon.json), e.g. BEAM_RUNTIME_DIR=/tmp/beam, and start again.\n"
        % (n, _SOCK_PATH_MAX, sock)
    )
    sys.exit(1)


def _local_ip() -> str:
    # Prefer an explicit cluster IP (same vars as ray._get_ip /
    # get_node_ip_address) so multi-homed hosts don't advertise the
    # default-route interface via membership / `ray status` while the shim
    # advertises the LAN address.
    return _config.node_ip() or _config.route_probe_ip()


_COMMANDS = ("start", "status", "stop", "bootstrap")
_HELP_FLAGS = ("-h", "--help", "help")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _usage(2)
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd in _HELP_FLAGS:
            return _usage(0)
        if cmd in ("-V", "--version"):
            from . import __version__

            print("ray (beam) %s" % __version__)
            return 0
        if cmd == "start":
            return _start(rest)
        if cmd == "status":
            return _no_args("status", _status, rest)
        if cmd == "stop":
            return _no_args("stop", _stop, rest)
        if cmd == "bootstrap":
            return _no_args("bootstrap", _bootstrap, rest)
    except _config.ConfigError as e:
        # bad environment variable: one clear line, not a traceback
        sys.stderr.write("%s\n" % e)
        return 2
    sys.stderr.write("beam: unknown command %r%s\n" % (cmd, _suggest(cmd, _COMMANDS)))
    return _usage(2)


def _bootstrap() -> int:
    bootstrap_env()
    return 0


def _suggest(word: str, options: Sequence[str]) -> str:
    """`; did you mean 'x'?` for a near miss, '' when nothing is close."""
    close = difflib.get_close_matches(word, options, n=1)
    return "; did you mean '%s'?" % close[0] if close else ""


def _no_args(cmd: str, run: Callable[[], int], args: list[str]) -> int:
    """Run a command that takes no options, or explain why the args were rejected.

    Stray args used to be discarded, so `ray stop --force` looked like it had
    forced something when it had done nothing.
    """
    for a in args:
        if a in _HELP_FLAGS:
            return _usage(0)
    if args:
        sys.stderr.write("beam %s: unexpected argument %r\n" % (cmd, args[0]))
        return 2
    return run()


def _usage(code: int) -> int:
    """Print usage. 0 -> stdout (explicit `--help`), 2 -> stderr (bad usage)."""
    text = (
        "beam: a drop-in subset of ray for vLLM distributed inference\n\n"
        "usage:\n"
        "  ray start --head [--port 6379] [--num-gpus N] [--node-ip IP]   start head (blocks)\n"
        "  ray start --address HOST:PORT [--num-gpus N]                   join cluster (blocks)\n"
        "  ray status                                  show cluster nodes/GPUs (exit 1 if down)\n"
        "  ray stop                                    stop the local daemon, clean runtime files\n"
        "  ray bootstrap                              install the ray/beam launcher + beam.pth\n"
        "  ray --version                               print the version and exit\n\n"
        "flags:\n"
        "  -h, --help                 show this help (same for every command)\n"
        "      --block                accepted for ray compatibility; start always blocks\n\n"
        "exit codes:\n"
        "  0  success (--help, --version, clean start/stop)\n"
        "  1  runtime error: daemon not running/unreachable, a node is down, stop refused\n"
        "  2  usage error: unknown command/flag, bad or missing flag value\n\n"
        "environment:\n"
        "  BEAM_NUM_GPUS     override detected GPU count (set on boxes without /dev/nvidia*)\n"
        "  BEAM_NODE_IP      advertise this address (else VLLM_HOST_IP, else default-route IP)\n"
        "  BEAM_RUNTIME_DIR  daemon state dir (default ~/.beam; the socket path\n"
        "                     must stay under ~100 bytes, so keep this dir short)\n"
        "  BEAM_SOCK         override the actor/worker daemon socket (default:\n"
        "                     the path recorded in the runtime dir's daemon.json)\n"
        "  BEAM_WORKER_CMD   how to launch an actor (default 'python3 -m ray._worker')\n"
        "  BEAM_BOOTSTRAP    force bootstrap outside a container (auto inside one)\n"
        "  BEAM_BIND_ADDRESS address the head's control port binds (default 0.0.0.0, i.e. every\n"
        "                    interface; set the cluster LAN address to narrow it)\n"
        "  BEAM_TIMEOUT      cap every control-plane timeout, in seconds (test/sim seam)\n"
        "  BEAM_SLEEP        'module:callable' delay hook: hook(seconds) -> awaitable (daemon)\n"
        "                    or -> None (shim); sync in the shim, the event loop in the daemon\n"
        "  BEAM_CLOCK        'module:callable' returning the current time in seconds\n"
        "  BEAM_SEED         derive node ids from this seed instead of OS entropy\n"
    )
    if code:
        sys.stderr.write(text)
    else:
        sys.stdout.write(text)
    return code


_START_FLAGS = ("--head", "--block", "--port", "--address", "--num-gpus", "--node-ip")


class _UsageError(Exception):
    """A bad or missing flag value; `_start` prints it and exits 2."""


def _port_ok(port: int) -> bool:
    """A TCP port number is a 16-bit field, so only 0-65535 can be bound or
    connected to (sockaddr_in.sin_port is uint16)."""
    return 0 <= port <= 65535


def _start(args: list[str]) -> int:
    head = False
    port = 6379
    address = None
    num_gpus = None
    node_ip = None
    i = 0

    def grab(name: str) -> str:
        nonlocal i
        if i + 1 >= len(args):
            sys.stderr.write("beam start: %s expects a value\n" % name)
            raise _UsageError
        i += 1
        return args[i]

    def as_int(val: str, name: str) -> int:
        try:
            return int(val)
        except ValueError:
            sys.stderr.write("beam start: %s expects an integer, got %r\n" % (name, val))
            raise _UsageError from None

    try:
        while i < len(args):
            a = args[i]
            if a == "--head":
                head = True
            elif a == "--block":
                pass  # always blocks; accepted for compatibility
            elif a == "--port":
                port = as_int(grab("--port"), "--port")
            elif a.startswith("--port="):
                port = as_int(a.split("=", 1)[1], "--port")
            elif a == "--address":
                address = grab("--address")
            elif a.startswith("--address="):
                address = a.split("=", 1)[1]
            elif a == "--num-gpus":
                num_gpus = as_int(grab("--num-gpus"), "--num-gpus")
            elif a.startswith("--num-gpus="):
                num_gpus = as_int(a.split("=", 1)[1], "--num-gpus")
            elif a == "--node-ip":
                node_ip = grab("--node-ip")
            elif a.startswith("--node-ip="):
                node_ip = a.split("=", 1)[1]
            elif a in ("-h", "--help"):
                return _usage(0)
            else:
                sys.stderr.write("beam start: unknown flag %r%s\n" % (a, _suggest(a, _START_FLAGS)))
                return 2
            i += 1
    except _UsageError:
        return 2
    if not head and not address:
        sys.stderr.write("beam start: need --head or --address HOST:PORT\n")
        return 2
    if address:
        _, _, ap = address.partition(":")
        if ap and not ap.isdigit():
            sys.stderr.write("beam start: --address port must be numeric, got %r\n" % ap)
            return 2
        if ap and not _port_ok(int(ap)):
            sys.stderr.write("beam start: --address port must be 0-65535, got %s\n" % ap)
            return 2
    if not _port_ok(port):
        # A TCP port is a 16-bit field: bind() raises OverflowError on anything
        # wider, which would reach the operator as a bare traceback.
        sys.stderr.write("beam start: --port must be 0-65535, got %d\n" % port)
        return 2
    if num_gpus is not None and num_gpus < 0:
        sys.stderr.write("beam start: --num-gpus must be >= 0, got %d\n" % num_gpus)
        return 2
    # BEAM_NUM_GPUS is the same setting as --num-gpus; validate it here so a typo
    # fails with a message instead of a ValueError traceout (or a negative GPU count).
    env_gpus = os.environ.get("BEAM_NUM_GPUS")
    if num_gpus is None and env_gpus is not None:
        try:
            num_gpus = int(env_gpus)
        except ValueError:
            sys.stderr.write("beam: BEAM_NUM_GPUS must be an integer, got %r\n" % env_gpus)
            return 2
        if num_gpus < 0:
            sys.stderr.write("beam: BEAM_NUM_GPUS must be >= 0, got %d\n" % num_gpus)
            return 2

    # Refuse an unusable socket path, then a live daemon's socket/runtime dir.
    # Checked here so a deep BEAM_RUNTIME_DIR is a named error instead of a
    # bare OSError("AF_UNIX path too long") from deep inside connect().
    _check_sock_path(_config.runtime_sock_path())
    live = _runtime.live_daemon_pid()
    if live is not None:
        sys.stderr.write(
            "beam start: daemon already running (pid %d). Run 'ray stop' first.\n" % live
        )
        return 1

    maybe_bootstrap()
    gpus = _daemon.detect_gpus(num_gpus)
    if gpus == 0 and num_gpus is None and not os.environ.get("BEAM_NUM_GPUS"):
        sys.stderr.write(
            "beam start: detected 0 GPUs (no /dev/nvidia*). If this node has GPUs, set "
            "--num-gpus N or BEAM_NUM_GPUS (e.g. GB10/ROCm device nodes differ).\n"
        )
    node_id = _daemon.new_node_id()
    ip = node_ip or _local_ip()
    # So actor workers inherit the advertised address (matches membership / status).
    os.environ["BEAM_NODE_IP"] = ip
    return asyncio.run(_run_daemon(head, node_id, ip, gpus, port, address))


async def _run_daemon(
    head: bool,
    node_id: str,
    ip: str,
    gpus: int,
    port: int,
    address: str | None,
) -> int:
    d = _daemon.Daemon(head, node_id, ip, gpus)
    sock = _config.runtime_sock_path()

    # Exclusive claim before unlinking the sock, so two concurrent starts cannot
    # both pass the live check and steal each other's socket path.
    os.makedirs(_config.runtime_dir(), exist_ok=True)
    rt: dict = {"sock": sock, "node": node_id, "head": head, "pid": os.getpid()}
    claim = _runtime.claim(rt)
    if claim is not None:
        sys.stderr.write(claim)
        return 1

    def _fail(msg: str) -> int:
        sys.stderr.write(msg)
        for path in (sock, _config.runtime_json_path()):
            try:
                os.remove(path)  # don't leave a stale socket / pidfile behind
            except OSError:
                pass
        return 1

    try:
        await d.serve_unix(sock)
    except OSError as e:
        return _fail("beam: cannot bind the daemon socket at %s (%s)\n" % (sock, e))

    if head:
        bind = _config.bind_address()
        try:
            await d.serve_tcp(bind, port)
        except OSError as e:
            return _fail(
                "beam head: cannot bind %s:%d (%s). Another head running? "
                "Run 'ray stop' first, or pick another --port.\n" % (bind, port, e)
            )
        rt["addr"] = "%s:%d" % (ip, port)
        # the control plane is unauthenticated (see SECURITY in README): keep :port
        # on a trusted/private network only. It binds every interface unless
        # BEAM_BIND_ADDRESS narrows it.
        print(
            "beam head started on %s:%d (%d GPUs), control port bound to %s"
            % (ip, port, gpus, bind)
        )
        print("join with:  ray start --address %s:%d" % (ip, port))
    else:
        assert address is not None  # _start guarantees --address when not --head
        host, _, p = address.partition(":")
        print("beam worker: joining head at %s (retrying until it answers)..." % address)
        try:
            await d.join_head(host, int(p or 6379))
        except OSError as e:
            return _fail(
                "beam worker: head not reachable at %s (%s). Check it is up and the "
                "port is open between nodes.\n" % (address, e)
            )
        except RuntimeError as e:
            # The head answered and refused: on_hello rejects a join aimed at a
            # daemon that is not the head with {"err": "not the head node"},
            # which Peer.call raises as a RuntimeError. Without this arm the
            # refusal escapes as a traceout and _fail never runs, so the claimed
            # daemon.json and the bound daemon.sock stay behind and the next
            # `ray start` refuses with "daemon already running" until an
            # operator deletes them by hand.
            return _fail(
                "beam worker: head at %s refused the join: %s\n"
                "--address must name the head (the node started with --head).\n" % (address, e)
            )
        rt["addr"] = address
        print("beam worker joined %s (%d GPUs)" % (address, gpus))

    _runtime.write(rt)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    print("beam: shutting down")
    d.shutdown()  # reap actor worker subprocesses instead of orphaning them
    # Unlink sock only while our claim still names this pid (keeps concurrent
    # restarts from rebinding daemon.sock and then having it deleted). Remove
    # the claim last so _runtime.live_daemon_pid stays true for the whole
    # teardown.
    path = _config.runtime_json_path()
    rt2 = _runtime.read()
    if not rt2 or _pid_of(rt2) != os.getpid():
        return 0  # unreadable, or another daemon owns the claim; leave sock alone
    try:
        os.remove(sock)
    except OSError:  # pragma: no cover
        pass
    try:
        os.remove(path)
    except OSError:  # pragma: no cover
        pass
    return 0


def _status() -> int:
    sock_path = _runtime.read().get("sock")
    if not isinstance(sock_path, str) or not sock_path:
        sys.stderr.write("beam status: no running daemon found\n")
        return 1
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            s.connect(sock_path)
        except OSError as e:
            sys.stderr.write("beam status: cannot reach daemon: %s\n" % e)
            return 1
        try:
            # Same framing as _proto.write_frame/read_frame, which the daemon
            # speaks: a header-only frame with no payload ("plen" absent reads
            # back as 0), decoded by the shared reader so the two copies of the
            # framing cannot drift apart.
            _proto.write_frame(s, {"t": "status"})
            resp, _body = _proto.read_frame(s)
        except (OSError, ConnectionError, json.JSONDecodeError, struct.error) as e:
            sys.stderr.write("beam status: cannot reach daemon: %s\n" % e)
            return 1
    finally:
        try:
            s.close()
        except Exception:
            pass
    # _proto.read_frame already rejected a reply whose header is not a JSON
    # object, so `resp` is a dict here.
    if resp.get("err"):
        sys.stderr.write("beam status: %s\n" % resp["err"])
        return 1
    nodes = resp.get("nodes") or []
    tot = used = down = 0
    print("node                 ip                 GPUs   state  role")
    for nd in nodes:
        role = "head" if nd.get("head") else "worker"
        alive = nd.get("alive", True)
        down += 0 if alive else 1
        print(
            "%-20s %-18s %d/%-4d %-6s %s"
            % (
                nd["node"],
                nd.get("ip", ""),
                nd.get("used", 0),
                nd.get("ngpu", 0),
                "up" if alive else "DOWN",
                role,
            )
        )
        tot += nd.get("ngpu", 0)
        used += nd.get("used", 0)
    print("\ncluster: %d nodes, %d/%d GPUs used" % (len(nodes), used, tot))
    if down:
        sys.stderr.write("beam status: %d node(s) DOWN\n" % down)
        return 1  # so `ray status && ...` health checks fail on a dropped node
    return 0


def _pid_of(doc: dict) -> int:
    """The `pid` field of a runtime document, or 0 when missing or not an int.

    A pid that cannot be parsed is not a process: callers treat 0 as "there is
    nothing to signal, but the claim's files are still cleaned".
    """
    try:
        return int(doc.get("pid") or 0)
    except (ValueError, TypeError):
        return 0


def _stop() -> int:
    rt = _runtime.read()
    if not rt:
        sys.stderr.write("beam stop: no running daemon found\n")
        return 1
    pid = _pid_of(rt)  # 0 for a malformed pid: nothing to signal, files still cleaned
    # Another stop is holding the runtime: never SIGTERM that stop process.
    if rt.get("stopping"):
        stop_pid = pid
        sock_path = rt.get("sock")
        if stop_pid:
            try:
                os.kill(stop_pid, 0)
            except ProcessLookupError:
                pass  # abandoned hold; clean leftovers below
            except OSError as e:
                sys.stderr.write("beam stop: cannot probe stop pid %s: %s\n" % (stop_pid, e))
                return 1
            else:
                # live concurrent stop: wait briefly for it to finish
                for _ in range(_EXIT_POLLS):
                    try:
                        os.kill(stop_pid, 0)
                    except OSError:
                        break  # peer exit or crash; re-verify cleanup below
                    time.sleep(_EXIT_POLL_INTERVAL)
                else:
                    sys.stderr.write("beam stop: another stop in progress (pid %s)\n" % stop_pid)
                    return 1
                # Peer stop gone: if claim cleaned, done; if abandoned hold
                # remains, fall through to rename-seize cleanup.
                rt2 = _runtime.read()
                if not rt2 or not rt2.get("stopping"):
                    # Peer stop finished; a new daemon may own the claim.
                    live = _runtime.live_daemon_pid()
                    if live:
                        sys.stderr.write(
                            "beam stop: another stop finished but daemon pid %s "
                            "is still running; re-run ray stop\n" % live
                        )
                        return 1
                    print("beam stop: another stop finished")
                    return 0
                # still a stopping hold; only clean if that stop pid is dead
                hold_pid = _pid_of(rt2)
                if hold_pid and hold_pid != stop_pid:
                    # different stop/daemon hold: do not seize their claim
                    try:
                        os.kill(hold_pid, 0)
                        sys.stderr.write(
                            "beam stop: another stop in progress (pid %s)\n" % hold_pid
                        )
                        return 1
                    except ProcessLookupError:
                        pass  # dead hold with different pid; fall through
                    except OSError:
                        sys.stderr.write(
                            "beam stop: another stop in progress (pid %s)\n" % hold_pid
                        )
                        return 1
                    sock_path = rt2.get("sock") or sock_path
                else:
                    sock_path = rt2.get("sock") or sock_path
                # fall through to abandoned seize-clean
        # abandoned stop hold: only seize if the current doc is still a dead
        # stopping hold (never rename-seize a live stopclean/daemon claim).
        path = _config.runtime_json_path()
        cur = _runtime.read()
        if not cur or not cur.get("stopping"):
            live = _runtime.live_daemon_pid()
            if live:
                sys.stderr.write("beam stop: daemon pid %s still running; re-run ray stop\n" % live)
                return 1
            print("beam stop: another stop finished")
            return 0
        cur_pid = _pid_of(cur)  # 0: no usable hold pid, so nothing live to protect
        if cur_pid:
            try:
                os.kill(cur_pid, 0)
            except ProcessLookupError:
                pass  # dead hold; safe to seize
            except OSError:
                sys.stderr.write("beam stop: another stop in progress (pid %s)\n" % cur_pid)
                return 1
            else:
                sys.stderr.write("beam stop: another stop in progress (pid %s)\n" % cur_pid)
                return 1
        sock_path = cur.get("sock") or sock_path
        seized = path + ".stopabandoned.%d" % os.getpid()
        try:
            os.rename(path, seized)
        except OSError:
            print("beam stop: another stop finished")
            return 0

        def _drop_seized() -> None:
            try:
                os.remove(seized)
            except OSError:
                pass

        doc = _runtime.read(seized)
        if not doc or not doc.get("stopping"):
            # not an abandoned hold: restore if free; never report success if live
            _runtime.link_restore(seized, path)
            live = _runtime.live_daemon_pid()
            if live:
                sys.stderr.write(
                    "beam stop: runtime owned by live daemon pid %s; re-run ray stop\n" % live
                )
                return 1
            print("beam stop: another stop finished")
            return 0
        hold_pid = _pid_of(doc)
        if hold_pid:
            try:
                os.kill(hold_pid, 0)
            except ProcessLookupError:
                pass  # confirmed abandoned
            except OSError:
                _runtime.link_restore(seized, path)
                sys.stderr.write("beam stop: another stop in progress (pid %s)\n" % hold_pid)
                return 1
            else:
                _runtime.link_restore(seized, path)
                sys.stderr.write("beam stop: another stop in progress (pid %s)\n" % hold_pid)
                return 1
        sock_path = doc.get("sock") or sock_path
        # Install a *live* stop hold (this process's pid) before unlinking the
        # sock so concurrent ray start sees a live claim and refuses.
        hold = path + ".stopclean.%d" % os.getpid()
        hold_linked = False
        try:
            with open(hold, "w") as f:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "stopping": True,
                        "sock": sock_path,
                        "cleaned": True,
                    },
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            os.link(hold, path)
            hold_linked = True
        except FileExistsError:
            try:
                os.unlink(hold)
            except OSError:
                pass
            _drop_seized()
            live = _runtime.live_daemon_pid()
            if live:
                sys.stderr.write("beam stop: runtime owned by live process pid %s\n" % live)
                return 1
            print("beam stop: another stop finished")
            return 0
        except OSError:
            try:
                os.unlink(hold)
            except OSError:
                pass
            _drop_seized()
            print("beam stop: another stop finished")
            return 0
        finally:
            if hold_linked:
                try:
                    os.unlink(hold)
                except OSError:
                    pass
        # Confirm we still own the live hold immediately before sock unlink.
        cur = _runtime.read()
        # A hold we cannot parse is not a hold we own: never unlink a path we
        # cannot prove is ours.
        owned = _pid_of(cur) == os.getpid() and bool(cur.get("stopping"))
        if not owned:
            # Do not unlink path we do not own; only drop seized.
            _drop_seized()
            print("beam stop: another stop finished")
            return 0
        for p in (sock_path, seized, path):
            try:
                if p:
                    os.remove(p)
            except OSError:
                pass
        print("beam stop: cleaned leftover stop state")
        return 0
    alive = False
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            alive = True
        except ProcessLookupError:
            pass  # already gone; fall through to clean up its stale files
        except OSError as e:
            # Process exists but we cannot signal it (e.g. EPERM). Leave
            # runtime files alone so the live daemon stays discoverable.
            sys.stderr.write("beam stop: cannot signal pid %s: %s\n" % (pid, e))
            return 1
    if alive:
        for _ in range(_EXIT_POLLS):  # clean-exit budget, then SIGKILL
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(_EXIT_POLL_INTERVAL)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:  # pragma: no cover (pid race)
                pass
            # wait for SIGKILL to take effect before seizing the runtime
            for _ in range(_SIGKILL_POLLS):
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(_SIGKILL_POLL_INTERVAL)
            else:
                # still alive (D-state / stuck): do not steal its runtime files
                sys.stderr.write(
                    "beam stop: pid %s still alive after SIGKILL; leaving runtime files\n" % pid
                )
                return 1
    # Atomically seize daemon.json so a concurrent restart cannot lose its claim.
    # Re-read first: only rename when the claim still names the target pid (a
    # new daemon may have claimed the path after the old process died).
    path = _config.runtime_json_path()
    cur = _runtime.read()
    cur_pid = _pid_of(cur)
    if cur_pid != pid:
        print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
        return 0
    # Live stop hold for another process: do not seize it.
    if cur.get("stopping") and cur_pid and cur_pid != os.getpid():
        try:
            os.kill(cur_pid, 0)
        except OSError:
            # probe says the holder is gone; the seize below still re-checks the
            # renamed document before it touches anything
            pass
        else:
            print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
            return 0
    doomed = path + ".stopping.%d" % os.getpid()
    try:
        os.rename(path, doomed)
    except OSError:
        print(
            "beam stop: stopped pid %s" % pid
            if alive
            else "beam stop: no live daemon (cleaned stale files)"
        )
        return 0
    rt2 = _runtime.read(doomed)
    if _pid_of(rt2) != pid:
        # give it back only if path is free; never rename-over a new claim
        _runtime.link_restore(doomed, path)
        print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
        return 0
    sock_path = rt2.get("sock") or rt.get("sock")
    # Hold the runtime path with *this* stop process's pid so concurrent
    # `ray start` sees a live claim via _runtime.live_daemon_pid / the marker.
    hold = path + ".stophold.%d" % os.getpid()
    hold_ok = False
    try:
        with open(hold, "w") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "stopping": True,
                    "stopped": pid,
                    "sock": sock_path,
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        os.link(hold, path)
        hold_ok = True
    except FileExistsError:
        try:
            os.unlink(hold)
        except OSError:
            pass
        try:
            os.remove(doomed)
        except OSError:
            pass
        print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
        return 0
    except OSError:
        # Cannot install hold: only drop the seized doomed file, never the sock
        # or a path we do not exclusively own.
        try:
            os.unlink(hold)
        except OSError:
            pass
        try:
            os.remove(doomed)
        except OSError:
            pass
        print(
            "beam stop: stopped pid %s" % pid
            if alive
            else "beam stop: no live daemon (cleaned stale files)"
        )
        return 0
    finally:
        if hold_ok:
            try:
                os.unlink(hold)
            except OSError:
                pass
    # Confirm we still own the claim immediately before unlinking the sock.
    cur = _runtime.read()
    if _pid_of(cur) != os.getpid() or not cur.get("stopping"):
        try:
            os.remove(doomed)
        except OSError:
            pass
        print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
        return 0
    for p in (sock_path, doomed, path):
        try:
            if p:
                os.remove(p)
        except OSError:
            pass
    print(
        "beam stop: stopped pid %s" % pid
        if alive
        else "beam stop: no live daemon (cleaned stale files)"
    )
    return 0


def maybe_bootstrap() -> None:
    """Bootstrap only inside a container (or when forced), so running on a host
    never touches /usr/local/bin or the system python site dirs."""
    if os.environ.get("BEAM_BOOTSTRAP") or os.path.exists("/.dockerenv"):
        bootstrap_env()


def bootstrap_env() -> None:
    """Make `ray`/`beam` commands available and the shim importable, so a single
    bind mount of the beam dir is all a container needs."""
    py = sys.executable
    py_pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    launcher = '#!/bin/sh\nexec %s -m ray "$@"\n' % py
    for name in ("ray", "beam"):
        path = os.path.join("/usr/local/bin", name)
        try:
            with open(path, "w") as f:
                f.write(launcher)
            os.chmod(path, 0o755)
        except OSError:
            pass

    import glob

    for pat in (
        "/usr/lib/python3*/site-packages",
        "/usr/lib/python3*/dist-packages",
        "/usr/local/lib/python3*/site-packages",
        "/usr/local/lib/python3*/dist-packages",
    ):
        for d in glob.glob(pat):
            try:
                with open(os.path.join(d, "beam.pth"), "w") as f:
                    f.write(py_pkg_parent + "\n")
            except OSError:  # pragma: no cover (unwritable site dir)
                pass
