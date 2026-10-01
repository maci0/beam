"""Synchronous client to the local beamd over its unix socket.

The vLLM driver issues one request at a time per logical operation, so a single
locked connection is enough; the heavy parallelism lives in the daemons and the
actor subprocesses, not here.
"""

from __future__ import annotations  # keep `str | None` valid on py3.9

import socket
import threading

from . import _config, _proto


class DaemonNotRunning(RuntimeError):
    """No local beam daemon to talk to (no runtime document in BEAM_RUNTIME_DIR)."""


# Smallest budget applied to the socket. A caller-supplied budget measures
# elapsed waiting, not syscall precision: Python rounds a sub-millisecond
# settimeout down to 0, and a 0 timeout on a socket reports "timed out"
# without having read anything. Sub-millisecond budgets are only ever asked
# for by a deadline that has already expired, so honoring 1ms costs nothing.
_SOCKET_TIMEOUT_FLOOR = 0.001


def _runtime_sock() -> str:
    """Socket of the local daemon: BEAM_SOCK, else daemon.json's recorded path.

    Raises DaemonNotRunning (a RuntimeError, like every other beam failure the
    driver sees) instead of the bare FileNotFoundError / KeyError / JSONDecodeError
    that reading the runtime document used to raise.
    """
    sock = _config.runtime_sock()
    if sock is None:
        raise DaemonNotRunning(
            "beam: no local daemon found (looked for %s). Start one with "
            "'ray start --head', or point BEAM_SOCK / BEAM_RUNTIME_DIR at it."
            % _config.runtime_json_path()
        )
    return sock


class DaemonClient:
    def __init__(self, sock_path: str | None = None, timeout: float | None = None) -> None:
        self.sock_path = sock_path or _runtime_sock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.connect(self.sock_path)
        except OSError as e:
            # The connect failed, so no caller can ever reach this socket: close
            # it here instead of dropping it to the GC, which would leak the fd
            # for every failed ray.init() attempt (ray.init() only assigns the
            # global _client after a successful construction).
            try:
                self._sock.close()
            except OSError:
                pass
            raise DaemonNotRunning(
                "beam: cannot reach the local daemon on %s (%s). Is it running?"
                % (self.sock_path, e)
            ) from None
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
            # A protocol break (a truncated frame, a header that is not a JSON
            # object, an fd-level reset) leaves the daemon's answer still
            # pending on this stream: the next request would read that stale
            # answer as its own response and hand the caller the wrong object.
            # End the session instead so the next call fails loudly.
            if self._sock.fileno() < 0:
                raise ConnectionError(
                    "beam: daemon connection on %s is closed; re-run ray.init()" % self.sock_path
                )
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
