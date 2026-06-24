"""Wire framing shared by the client shim and the actor worker.

Frame layout (must match encode_frame/read_frame in _daemon.py):
    [4-byte big-endian length][JSON header][optional raw payload]
The header's ``plen`` field gives the length of the raw payload that follows.
"""

from __future__ import annotations  # keep PEP585 generics valid on py3.9

import json
import socket
import struct
from typing import Any

# sanity bound so a corrupt length can't trigger a multi-GB allocation/hang.
# headers are small JSON; payloads are pickled actor results (a few MB at most).
_MAX_FRAME = 512 * 1024 * 1024

try:  # cloudpickle handles vLLM's classes/closures; pickle is the fallback
    import cloudpickle as _pickle
except ImportError:  # pragma: no cover
    import pickle as _pickle


def dumps(obj: Any) -> bytes:
    return _pickle.dumps(obj)


def loads(buf: bytes) -> Any:
    return _pickle.loads(buf)


def write_frame(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header["plen"] = len(payload)
    hdr = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(hdr)) + hdr + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("daemon connection closed")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[dict, bytes]:
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    if n == 0 or n > _MAX_FRAME:
        raise ConnectionError("bad frame header length %d" % n)
    header = json.loads(_recv_exact(sock, n))
    plen = header.get("plen", 0)
    if plen < 0 or plen > _MAX_FRAME:
        raise ConnectionError("bad frame payload length %d" % plen)
    payload = _recv_exact(sock, plen) if plen else b""
    return header, payload
