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

# Smallest budget applied to the socket. A caller-supplied budget measures
# elapsed waiting, not syscall precision: Python rounds a sub-millisecond
# settimeout down to 0, and a 0 timeout on a socket reports "timed out"
# without having read anything. Sub-millisecond budgets are only ever asked
# for by a deadline that has already expired, so honoring 1ms costs nothing.
_SOCKET_TIMEOUT_FLOOR = 0.001


def _runtime_sock() -> str:
    if os.environ.get("BEAM_SOCK"):
        return os.environ["BEAM_SOCK"]
    rt_dir = os.environ.get("BEAM_RUNTIME_DIR") or os.path.join(os.path.expanduser("~"), ".beam")
    with open(os.path.join(rt_dir, "daemon.json")) as f:
        return json.load(f)["sock"]


class DaemonClient:
    def __init__(self, sock_path: str | None = None, timeout: float | None = None) -> None:
        self.sock_path = sock_path or _runtime_sock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.sock_path)
        self._lock = threading.Lock()
        # Default round-trip budget. None = block until the daemon answers,
        # which is what put/get/call want.
        self._timeout = timeout

    def request(
        self, header: dict, payload: bytes = b"", timeout: float | None = None
    ) -> tuple[dict, bytes]:
        """One request/response round-trip.

        `timeout` (seconds) bounds only how long this call blocks; it raises
        TimeoutError instead of leaving the caller stuck behind an unbounded
        socket read, so a caller holding a deadline budget can honor it (see
        ray.wait). None falls back to the client's default budget.
        """
        budget = self._timeout if timeout is None else timeout
        with self._lock:
            if budget is None:
                self._sock.settimeout(None)
            else:
                self._sock.settimeout(max(_SOCKET_TIMEOUT_FLOOR, budget))
            try:
                _proto.write_frame(self._sock, header, payload)
                resp, body = _proto.read_frame(self._sock)
            except (socket.timeout, TimeoutError) as e:
                raise TimeoutError(
                    "request %r timed out after %ss" % (header.get("t"), budget)
                ) from e
            finally:
                self._sock.settimeout(None)  # a late stat must not inherit a stale budget
        if resp.get("err"):
            raise RuntimeError(resp["err"])
        return resp, body

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
