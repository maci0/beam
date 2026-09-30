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

from . import _config, _daemon

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


def _runtime_dir() -> str:
    return _config.runtime_dir()


def _runtime_path() -> str:
    return _config.runtime_json_path()


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
        "  ray status                                  show cluster nodes/GPUs (exit 1 if any down)\n"
        "  ray stop                                    stop the local daemon, clean its runtime files\n"
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
        "  BEAM_SOCK         actor/worker daemon socket (the CLI reads it from the runtime dir)\n"
        "  BEAM_WORKER_CMD   how to launch an actor (default 'python3 -m ray._worker')\n"
        "  BEAM_BOOTSTRAP    force bootstrap outside a container (auto inside one)\n"
        "  BEAM_BIND_ADDRESS address the head's control port binds (default 0.0.0.0, i.e. every\n"
        "                    interface; set the cluster LAN address to narrow it)\n"
    )
    if code:
        sys.stderr.write(text)
    else:
        sys.stdout.write(text)
    return code


_START_FLAGS = ("--head", "--block", "--port", "--address", "--num-gpus", "--node-ip")


class _UsageError(Exception):
    """A bad or missing flag value; `_start` prints it and exits 2."""


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
    if num_gpus is not None and num_gpus < 0:
        sys.stderr.write("beam start: --num-gpus must be >= 0, got %d\n" % num_gpus)
        return 2

    # Refuse an unusable socket path, then a live daemon's socket/runtime dir.
    # Checked here so a deep BEAM_RUNTIME_DIR is a named error instead of a
    # bare OSError("AF_UNIX path too long") from deep inside connect().
    _check_sock_path(os.path.join(_runtime_dir(), "daemon.sock"))
    live = _live_daemon_pid()
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


def _write_runtime_atomic(rt: dict) -> None:
    """Write daemon.json via temp+replace so readers never see a partial file."""
    path = _runtime_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(rt, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
    os.makedirs(_runtime_dir(), exist_ok=True)
    rt: dict = {"sock": sock, "node": node_id, "head": head, "pid": os.getpid()}
    claim = _claim_runtime(rt)
    if claim is not None:
        sys.stderr.write(claim)
        return 1

    await d.serve_unix(sock)

    def _fail(msg: str) -> int:
        sys.stderr.write(msg)
        for path in (sock, _runtime_path()):
            try:
                os.remove(path)  # don't leave a stale socket / pidfile behind
            except OSError:
                pass
        return 1

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
        rt["addr"] = address
        print("beam worker joined %s (%d GPUs)" % (address, gpus))

    _write_runtime_atomic(rt)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    print("beam: shutting down")
    d.shutdown()  # reap actor worker subprocesses instead of orphaning them
    # Unlink sock only while our claim still names this pid (keeps concurrent
    # restarts from rebinding daemon.sock and then having it deleted). Remove
    # the claim last so _live_daemon_pid stays true for the whole teardown.
    path = _runtime_path()
    try:
        with open(path) as f:
            rt2 = json.load(f)
        if int(rt2.get("pid") or 0) != os.getpid():
            return 0  # another daemon owns the claim; leave sock alone
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError, KeyError):
        return 0
    try:
        os.remove(sock)
    except OSError:  # pragma: no cover
        pass
    try:
        os.remove(path)
    except OSError:  # pragma: no cover
        pass
    return 0


def _read_runtime() -> dict:
    with open(_runtime_path()) as f:
        return json.load(f)


def _link_restore(seized: str, path: str) -> bool:
    """Restore seized claim to path only if path is free (never clobber).

    Returns True if restored, False if path is already owned (seized dropped)
    or restore failed. On non-FileExistsError link failures, leave seized on
    disk so the last copy of a live claim is not destroyed.
    """
    try:
        os.link(seized, path)
        try:
            os.unlink(seized)
        except OSError:
            pass
        return True
    except FileExistsError:
        # path already has a claim: drop our seized copy only
        try:
            os.unlink(seized)
        except OSError:
            pass
        return False
    except OSError:
        # path still free but link failed (EPERM/ENOSPC/...): keep seized
        return False


def _claim_runtime(rt: dict) -> str | None:
    """Atomically publish a complete daemon.json for this process.

    Writes the full document to a temp file first, then links it into place so
    another start never sees an empty/partial claim and steals the runtime.
    Stale pidfiles are rename-seized (not remove-then-link) to avoid a window
    where a concurrent claim is deleted out from under a live daemon.
    Returns an error message if another live daemon owns the runtime, else None.
    """
    path = _runtime_path()
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(rt, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        for _ in range(2):
            try:
                os.link(tmp, path)  # fails if path already exists
                return None
            except FileExistsError:
                live = _live_daemon_pid()
                if live is not None:
                    return "beam: daemon already running (pid %d). Run 'ray stop' first.\n" % live
                # Seize the existing path atomically, confirm it is still dead
                # (or not an in-progress stop hold), then discard and retry.
                seized = path + ".stale.%d" % os.getpid()
                try:
                    os.rename(path, seized)
                except OSError:
                    continue  # raced with another claim/stop; retry link
                still_live = False
                old_pid: int | None = None
                try:
                    with open(seized) as f:
                        old = json.load(f)
                    old_pid = int(old.get("pid") or 0)
                    # Active ray stop holds the path with the stop process pid.
                    if old.get("stopping") and old_pid:
                        try:
                            os.kill(old_pid, 0)
                            still_live = True
                        except ProcessLookupError:
                            still_live = False  # abandoned stop; reclaim
                        except OSError:
                            still_live = True
                    elif old_pid:
                        try:
                            os.kill(old_pid, 0)
                            still_live = True
                        except ProcessLookupError:
                            still_live = False
                        except OSError:
                            still_live = True  # EPERM: treat as live
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                    AttributeError,
                    KeyError,
                ):
                    still_live = False
                if still_live:
                    _link_restore(seized, path)
                    return "beam: daemon already running (pid %s). Run 'ray stop' first.\n" % (
                        old_pid if old_pid else "?"
                    )
                try:
                    os.remove(seized)
                except OSError:
                    pass
                continue
        return "beam: cannot claim runtime dir %s\n" % path
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _live_daemon_pid() -> int | None:
    """Return the pid of a still-running local daemon, or None if none/stale."""
    try:
        rt = _read_runtime()
        pid = int(rt.get("pid") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError, KeyError):
        return None
    if not pid:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except OSError:
        # EPERM etc: process exists but is not signalable; treat as live so we
        # do not unlink its socket out from under it.
        return pid
    return pid


def _status() -> int:
    try:
        rt = _read_runtime()
        sock_path = rt["sock"]
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError, KeyError):
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
            hdr = json.dumps({"t": "status"}).encode()
            s.sendall(struct.pack(">I", len(hdr)) + hdr)
            (n,) = struct.unpack(">I", _recv(s, 4))
            resp = json.loads(_recv(s, n))
        except (OSError, ConnectionError, json.JSONDecodeError, struct.error) as e:
            sys.stderr.write("beam status: cannot reach daemon: %s\n" % e)
            return 1
    finally:
        try:
            s.close()
        except Exception:
            pass
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


def _recv(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("short read")
        buf.extend(chunk)
    return bytes(buf)


def _stop() -> int:

    try:
        rt = _read_runtime()
        pid = int(rt.get("pid") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError, KeyError):
        sys.stderr.write("beam stop: no running daemon found\n")
        return 1
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
                try:
                    rt2 = _read_runtime()
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                    AttributeError,
                    KeyError,
                ):
                    print("beam stop: another stop finished")
                    return 0
                if not rt2.get("stopping"):
                    # Peer stop finished; a new daemon may own the claim.
                    live = _live_daemon_pid()
                    if live:
                        sys.stderr.write(
                            "beam stop: another stop finished but daemon pid %s "
                            "is still running; re-run ray stop\n" % live
                        )
                        return 1
                    print("beam stop: another stop finished")
                    return 0
                # still a stopping hold; only clean if that stop pid is dead
                hold_pid = int(rt2.get("pid") or 0)
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
        path = _runtime_path()
        try:
            cur = _read_runtime()
            cur_pid = int(cur.get("pid") or 0)
            if not cur.get("stopping"):
                live = _live_daemon_pid()
                if live:
                    sys.stderr.write(
                        "beam stop: daemon pid %s still running; re-run ray stop\n" % live
                    )
                    return 1
                print("beam stop: another stop finished")
                return 0
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
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            AttributeError,
            KeyError,
        ):
            print("beam stop: another stop finished")
            return 0
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

        doc: dict = {}
        doc_ok = False
        try:
            with open(seized) as f:
                doc = json.load(f)
            doc_ok = isinstance(doc, dict)
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            AttributeError,
            KeyError,
        ):
            doc = {}
        if doc_ok and not doc.get("stopping"):
            # not an abandoned hold: restore if free; never report success if live
            _link_restore(seized, path)
            live = _live_daemon_pid()
            if live:
                sys.stderr.write(
                    "beam stop: runtime owned by live daemon pid %s; re-run ray stop\n" % live
                )
                return 1
            print("beam stop: another stop finished")
            return 0
        if doc_ok:
            hold_pid = int(doc.get("pid") or 0)
            if hold_pid:
                try:
                    os.kill(hold_pid, 0)
                except ProcessLookupError:
                    pass  # confirmed abandoned
                except OSError:
                    _link_restore(seized, path)
                    sys.stderr.write("beam stop: another stop in progress (pid %s)\n" % hold_pid)
                    return 1
                else:
                    _link_restore(seized, path)
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
            live = _live_daemon_pid()
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
        try:
            with open(path) as f:
                cur = json.load(f)
            if int(cur.get("pid") or 0) != os.getpid() or not cur.get("stopping"):
                # Do not unlink path we do not own; only drop seized.
                _drop_seized()
                print("beam stop: another stop finished")
                return 0
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            AttributeError,
            KeyError,
        ):
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
    path = _runtime_path()
    try:
        cur = _read_runtime()
        cur_pid = int(cur.get("pid") or 0)
        if cur_pid != pid:
            print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
            return 0
        # Live stop hold for another process: do not seize it.
        if cur.get("stopping") and cur_pid and cur_pid != os.getpid():
            try:
                os.kill(cur_pid, 0)
                print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
                return 0
            except ProcessLookupError:
                pass  # abandoned hold with matching stopped target; seize below
            except OSError:
                print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
                return 0
    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        AttributeError,
        KeyError,
    ):
        print(
            "beam stop: stopped pid %s" % pid
            if alive
            else "beam stop: no live daemon (cleaned stale files)"
        )
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
    try:
        with open(doomed) as f:
            rt2 = json.load(f)
        if int(rt2.get("pid") or 0) != pid:
            # give it back only if path is free; never rename-over a new claim
            _link_restore(doomed, path)
            print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
            return 0
        sock_path = rt2.get("sock") or rt.get("sock")
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError, KeyError):
        sock_path = rt.get("sock")
    # Hold the runtime path with *this* stop process's pid so concurrent
    # `ray start` sees a live claim via _live_daemon_pid / stopping marker.
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
    try:
        with open(path) as f:
            cur = json.load(f)
        if int(cur.get("pid") or 0) != os.getpid() or not cur.get("stopping"):
            try:
                os.remove(doomed)
            except OSError:
                pass
            print("beam stop: stopped pid %s (runtime now owned by another daemon)" % pid)
            return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError, KeyError):
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
