"""Synchronous client to the local beamd over its unix socket.

The vLLM driver issues one request at a time per logical operation, so a single
locked connection is enough; the heavy parallelism lives in the daemons and the
actor subprocesses, not here.
"""

from __future__ import annotations  # keep `str | None` valid on py3.9

import json
import os
import socket
import threading

from . import _proto


def _runtime_sock() -> str:
    if os.environ.get("BEAM_SOCK"):
        return os.environ["BEAM_SOCK"]
    rt_dir = os.environ.get("BEAM_RUNTIME_DIR") or os.path.join(os.path.expanduser("~"), ".beam")
    with open(os.path.join(rt_dir, "daemon.json")) as f:
        return json.load(f)["sock"]


class DaemonClient:
    def __init__(self, sock_path: str | None = None):
        self.sock_path = sock_path or _runtime_sock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.sock_path)
        self._lock = threading.Lock()

    def request(self, header: dict, payload: bytes = b""):
        with self._lock:
            _proto.write_frame(self._sock, header, payload)
            resp, body = _proto.read_frame(self._sock)
        if resp.get("err"):
            raise RuntimeError(resp["err"])
        return resp, body

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
