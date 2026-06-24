"""Wire-protocol unit + fuzz tests. The framing parses untrusted bytes off a
socket, so it is the highest-value fuzz target: random input must fail cleanly
(a known exception), never hang or allocate unboundedly."""

import json
import os
import socket
import struct
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray import _proto  # noqa: E402
from ray._daemon import encode_frame  # noqa: E402


class BytesSock:
    """Minimal socket for read_frame: serves a fixed buffer, EOF -> b''."""

    def __init__(self, data):
        self.buf = data
        self.pos = 0

    def recv(self, n):
        chunk = self.buf[self.pos : self.pos + n]
        self.pos += len(chunk)
        return chunk


def roundtrip(header, payload):
    return _proto.read_frame(BytesSock(encode_frame(header, payload)))


def test_roundtrip_basic():
    h, p = roundtrip({"t": "call", "x": 1}, b"hello")
    assert h["t"] == "call" and h["x"] == 1 and p == b"hello"


def test_empty_payload():
    _, p = roundtrip({"t": "ping"}, b"")
    assert p == b""


# cover the values the real wire carries: floats (num_gpus=0.5), big ints
# (reqids), control/surrogate-free unicode, not just small ints.
_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
_PARSE_ERRORS = (ConnectionError, json.JSONDecodeError, UnicodeDecodeError, struct.error)


@settings(max_examples=400)
@given(st.dictionaries(st.text(min_size=1), _scalars, max_size=8), st.binary(max_size=8192))
def test_roundtrip_fuzz(header, payload):
    header = {**header, "t": "x"}  # a type key is always present on the wire
    h, p = roundtrip(header, payload)
    assert p == payload
    for k, v in header.items():
        assert h[k] == v


@settings(max_examples=500)
@given(st.binary(max_size=64))
def test_garbage_rejected_or_valid(data):
    # random bytes: either raise a known parse/EOF error, or (rarely) parse to a
    # structurally valid frame. Never hang, OOM, or return a malformed shape.
    try:
        h, p = _proto.read_frame(BytesSock(data))
    except _PARSE_ERRORS:
        return
    assert isinstance(h, dict) and isinstance(p, bytes)


@settings(max_examples=500)
@given(st.binary(max_size=300))
def test_structured_frame_fuzz(body):
    # a well-formed 4-byte length over `body`, but body is arbitrary, so this
    # actually exercises the JSON / unicode / plen-bound branches (not just EOF).
    frame = struct.pack(">I", len(body)) + body
    try:
        h, p = _proto.read_frame(BytesSock(frame))
    except _PARSE_ERRORS:
        return
    assert isinstance(h, dict) and isinstance(p, bytes)


def test_class_and_closure_roundtrip():
    # the actual reason cloudpickle is used: classes and closures, not just data
    class W:
        def __init__(self, r):
            self.r = r

    obj = _proto.loads(_proto.dumps(W(7)))
    assert obj.r == 7
    n = 3
    fn = _proto.loads(_proto.dumps(lambda x: x + n))
    assert fn(4) == 7


def test_oversize_rejected():
    frame = struct.pack(">I", _proto._MAX_FRAME + 1) + b"{}"
    with pytest.raises(ConnectionError):
        _proto.read_frame(BytesSock(frame))


def test_zero_length_header_rejected():
    with pytest.raises(ConnectionError):
        _proto.read_frame(BytesSock(struct.pack(">I", 0)))


def test_bad_plen_rejected():
    body = json.dumps({"t": "x", "plen": _proto._MAX_FRAME + 1}).encode()
    frame = struct.pack(">I", len(body)) + body
    with pytest.raises(ConnectionError):
        _proto.read_frame(BytesSock(frame))


@settings(max_examples=200)
@given(st.recursive(_scalars, lambda c: st.lists(c) | st.dictionaries(st.text(), c), max_leaves=20))
def test_pickle_roundtrip(obj):
    assert _proto.loads(_proto.dumps(obj)) == obj


# ---- write_frame over a real socketpair (the wire path the client uses) -----


def test_write_frame_socketpair_roundtrip():
    a, b = socket.socketpair()
    try:
        _proto.write_frame(a, {"t": "call", "n": 5}, b"body")
        h, p = _proto.read_frame(b)
        assert h["t"] == "call" and h["n"] == 5 and p == b"body"
        assert h["plen"] == 4  # write_frame stamps the payload length
    finally:
        a.close()
        b.close()


def test_read_frame_socket_eof_raises():
    a, b = socket.socketpair()
    a.close()  # peer gone -> recv returns b'' -> ConnectionError
    try:
        with pytest.raises(ConnectionError):
            _proto.read_frame(b)
    finally:
        b.close()


@settings(max_examples=150)
@given(st.dictionaries(st.text(min_size=1), _scalars, max_size=6), st.binary(max_size=2048))
def test_write_frame_roundtrip_fuzz(header, payload):
    header = {**header, "t": "x"}
    a, b = socket.socketpair()
    try:
        _proto.write_frame(a, header, payload)
        h, p = _proto.read_frame(b)
        assert p == payload
        for k, v in header.items():
            assert h[k] == v
    finally:
        a.close()
        b.close()
