"""Wire-protocol unit + fuzz tests. The framing parses untrusted bytes off a
socket, so it is the highest-value fuzz target: random input must fail cleanly
(a known exception), never hang or allocate unboundedly."""

import json
import os
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


_scalars = st.none() | st.booleans() | st.integers(-(10**9), 10**9) | st.text()


@settings(max_examples=300)
@given(st.dictionaries(st.text(min_size=1), _scalars, max_size=8), st.binary(max_size=8192))
def test_roundtrip_fuzz(header, payload):
    header = {**header, "t": "x"}  # a type key is always present on the wire
    h, p = roundtrip(header, payload)
    assert p == payload
    for k, v in header.items():
        assert h[k] == v


@settings(max_examples=500)
@given(st.binary(max_size=64))
def test_garbage_no_crash(data):
    # random bytes: must raise a known parse/EOF error, not hang or OOM
    try:
        _proto.read_frame(BytesSock(data))
    except (ConnectionError, json.JSONDecodeError, UnicodeDecodeError, struct.error):
        pass


def test_oversize_rejected():
    frame = struct.pack(">I", _proto._MAX_FRAME + 1) + b"{}"
    with pytest.raises(ConnectionError):
        _proto.read_frame(BytesSock(frame))


def test_zero_length_header_rejected():
    with pytest.raises(ConnectionError):
        _proto.read_frame(BytesSock(struct.pack(">I", 0)))


@settings(max_examples=200)
@given(st.recursive(_scalars, lambda c: st.lists(c) | st.dictionaries(st.text(), c), max_leaves=20))
def test_pickle_roundtrip(obj):
    assert _proto.loads(_proto.dumps(obj)) == obj
