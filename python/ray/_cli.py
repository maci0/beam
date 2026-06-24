"""`ray` / `beam` command line: start/status/stop/bootstrap.

`ray start` runs the daemon (this process blocks, like `ray start --block`).
Everything is Python now; there is no separate binary.
"""

from __future__ import annotations  # keep `X | None` valid on py3.9

import asyncio
import json
import os
import signal
import socket
import struct
import sys
from collections.abc import Sequence

from . import _daemon


def _runtime_dir() -> str:
    return os.environ.get("BEAM_RUNTIME_DIR") or os.path.join(os.path.expanduser("~"), ".beam")


def _runtime_path() -> str:
    return os.path.join(_runtime_dir(), "daemon.json")


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _usage()
    cmd, rest = argv[0], argv[1:]
    if cmd == "start":
        return _start(rest)
    if cmd == "status":
        return _status()
    if cmd == "stop":
        return _stop()
    if cmd == "bootstrap":
        bootstrap_env()
        return 0
    if cmd in ("-h", "--help", "help"):
        return _usage()
    sys.stderr.write("beam: unknown command %r\n" % cmd)
    return _usage()


def _usage() -> int:
    sys.stderr.write(
        "beam: a drop-in subset of ray for vLLM distributed inference\n\n"
        "usage:\n"
        "  ray start --head [--port 6379] [--num-gpus N]   start the head (blocks)\n"
        "  ray start --address HOST:PORT [--num-gpus N]     join a cluster (blocks)\n"
        "  ray status                                       show cluster nodes/GPUs\n"
        "  ray stop                                         stop the local daemon\n"
    )
    return 2


def _start(args: list[str]) -> int:
    head = False
    port = 6379
    address = None
    num_gpus = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--head":
            head = True
        elif a == "--block":
            pass  # always blocks; accepted for compatibility
        elif a == "--port":
            i += 1
            port = int(args[i])
        elif a.startswith("--port="):
            port = int(a.split("=", 1)[1])
        elif a == "--address":
            i += 1
            address = args[i]
        elif a.startswith("--address="):
            address = a.split("=", 1)[1]
        elif a == "--num-gpus":
            i += 1
            num_gpus = int(args[i])
        elif a.startswith("--num-gpus="):
            num_gpus = int(a.split("=", 1)[1])
        else:
            sys.stderr.write("beam: unknown flag %r\n" % a)
            return 2
        i += 1
    if not head and not address:
        sys.stderr.write("beam start: need --head or --address HOST:PORT\n")
        return 2

    maybe_bootstrap()
    gpus = _daemon.detect_gpus(num_gpus)
    node_id = _daemon.new_node_id()
    ip = _local_ip()
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
    sock = os.path.join(_runtime_dir(), "daemon.sock")
    await d.serve_unix(sock)

    rt = {"sock": sock, "node": node_id, "head": head, "pid": os.getpid()}
    if head:
        await d.serve_tcp("0.0.0.0", port)
        rt["addr"] = "%s:%d" % (ip, port)
        print("beam head started on %s:%d (%d GPUs)" % (ip, port, gpus))
        print("join with:  ray start --address %s:%d" % (ip, port))
    else:
        assert address is not None  # _start guarantees --address when not --head
        host, _, p = address.partition(":")
        await d.join_head(host, int(p or 6379))
        rt["addr"] = address
        print("beam worker joined %s (%d GPUs)" % (address, gpus))

    os.makedirs(_runtime_dir(), exist_ok=True)
    with open(_runtime_path(), "w") as f:
        json.dump(rt, f)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    print("beam: shutting down")
    d.shutdown()  # reap actor worker subprocesses instead of orphaning them
    try:
        os.remove(_runtime_path())
    except OSError:
        pass
    return 0


def _read_runtime() -> dict:
    with open(_runtime_path()) as f:
        return json.load(f)


def _status() -> int:
    try:
        rt = _read_runtime()
    except OSError:
        sys.stderr.write("beam status: no running daemon found\n")
        return 1
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(rt["sock"])
    except OSError as e:
        sys.stderr.write("beam status: cannot reach daemon: %s\n" % e)
        return 1
    hdr = json.dumps({"t": "status"}).encode()
    s.sendall(struct.pack(">I", len(hdr)) + hdr)
    (n,) = struct.unpack(">I", _recv(s, 4))
    resp = json.loads(_recv(s, n))
    s.close()
    if resp.get("err"):
        sys.stderr.write("beam status: %s\n" % resp["err"])
        return 1
    nodes = resp.get("nodes") or []
    tot = used = 0
    print("node                 ip                 GPUs   alive  role")
    for nd in nodes:
        role = "head" if nd.get("head") else "worker"
        print(
            "%-20s %-18s %d/%-4d %-6s %s"
            % (
                nd["node"],
                nd.get("ip", ""),
                nd.get("used", 0),
                nd.get("ngpu", 0),
                nd.get("alive", True),
                role,
            )
        )
        tot += nd.get("ngpu", 0)
        used += nd.get("used", 0)
    print("\ncluster: %d nodes, %d/%d GPUs used" % (len(nodes), used, tot))
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
    except OSError:
        sys.stderr.write("beam stop: no running daemon found\n")
        return 1
    pid = rt.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
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
            except OSError:
                pass
