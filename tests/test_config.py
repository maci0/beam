"""Unit + fuzz tests for the environment-variable loaders (`_config.py`).

Every operator-facing env var is read through here, so these tests pin the two
properties the loaders exist for: a valid value flows through unchanged (same
precedence chain as before the refactor), and an unusable value is rejected with
a ConfigError naming the variable instead of being used.
"""

import ipaddress
import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ray import _config  # noqa: E402

_ALL = (
    "BEAM_NUM_GPUS",
    "BEAM_NODE_IP",
    "VLLM_HOST_IP",
    "BEAM_RUNTIME_DIR",
    "BEAM_SOCK",
    "BEAM_GPU_IDS",
    "BEAM_WORKER_CMD",
    "BEAM_BIND_ADDRESS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ALL:
        monkeypatch.delenv(name, raising=False)


# ---- num_gpus --------------------------------------------------------------


def test_num_gpus_unset_is_none():
    assert _config.num_gpus() is None  # caller falls back to /dev/nvidia*


@pytest.mark.parametrize("raw", ["0", "1", "8", " 4 ", "+2"])
def test_num_gpus_parses(monkeypatch, raw):
    monkeypatch.setenv("BEAM_NUM_GPUS", raw)
    assert _config.num_gpus() == int(raw)


def test_num_gpus_empty_is_unset(monkeypatch):
    monkeypatch.setenv("BEAM_NUM_GPUS", "")
    assert _config.num_gpus() is None


@pytest.mark.parametrize("raw", ["abc", "8gpus", "1.5", "0x2", "1,2", " "])
def test_num_gpus_rejects_garbage(monkeypatch, raw):
    """A typo used to raise a bare ValueError from inside detect_gpus()."""
    monkeypatch.setenv("BEAM_NUM_GPUS", raw)
    with pytest.raises(_config.ConfigError, match="BEAM_NUM_GPUS"):
        _config.num_gpus()


@pytest.mark.parametrize("raw", ["-1", "-8"])
def test_num_gpus_rejects_negative(monkeypatch, raw):
    """-1 would publish a negative GPU count and break membership accounting."""
    monkeypatch.setenv("BEAM_NUM_GPUS", raw)
    with pytest.raises(_config.ConfigError, match=">= 0"):
        _config.num_gpus()


def test_num_gpus_override_wins_and_skips_env(monkeypatch):
    monkeypatch.setenv("BEAM_NUM_GPUS", "not-a-number")  # never parsed: override is first
    assert _config.num_gpus(4) == 4
    assert _config.num_gpus(0) == 0  # 0 is a real count (CPU head), not "unset"


# ---- node_ip ---------------------------------------------------------------


def test_node_ip_unset_is_none():
    assert _config.node_ip() is None  # caller probes the default route


def test_node_ip_precedence(monkeypatch):
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.2")
    assert _config.node_ip() == "10.0.0.2"
    monkeypatch.setenv("BEAM_NODE_IP", "10.0.0.1")
    assert _config.node_ip() == "10.0.0.1"  # beam's own var wins over vLLM's
    assert _config.node_ip("10.0.0.9") == "10.0.0.9"  # the CLI flag wins over both


def test_node_ip_empty_falls_through(monkeypatch):
    """`docker run -e BEAM_NODE_IP` expands to empty: advertise nothing beats
    advertising "" to every peer."""
    monkeypatch.setenv("BEAM_NODE_IP", "")
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.2")
    assert _config.node_ip() == "10.0.0.2"


def test_node_ip_canonicalises(monkeypatch):
    monkeypatch.setenv("BEAM_NODE_IP", "fd00:0000:0000:0000:0000:0000:0000:0001")
    assert _config.node_ip() == str(ipaddress.ip_address("fd00::1"))


@pytest.mark.parametrize("raw", ["10.0.0.256", "not-an-ip", "10.0.0.1:8080", "example.com", "1"])
def test_node_ip_rejects_non_address(monkeypatch, raw):
    """The old code advertised any string, so a typo hung the cluster at
    connect time instead of failing at startup."""
    monkeypatch.setenv("BEAM_NODE_IP", raw)
    with pytest.raises(_config.ConfigError, match="BEAM_NODE_IP"):
        _config.node_ip()


def test_node_ip_override_rejected(monkeypatch):
    with pytest.raises(_config.ConfigError, match="--node-ip"):
        _config.node_ip("nonsense")


@given(st.text(min_size=1).filter(lambda s: s.strip() and "\x00" not in s))
def test_fuzz_node_ip_either_returns_address_or_raises(raw):
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {"BEAM_NODE_IP": raw}, clear=False):
        try:
            got = _config.node_ip()
        except _config.ConfigError:
            return
        assert got == str(ipaddress.ip_address(got))  # whatever we return is an address


# ---- bind_address ----------------------------------------------------------


def test_bind_address_default_is_all_interfaces():
    assert _config.bind_address() == "0.0.0.0"


def test_bind_address_empty_is_default(monkeypatch):
    monkeypatch.setenv("BEAM_BIND_ADDRESS", "")
    assert _config.bind_address() == "0.0.0.0"


def test_bind_address_narrows(monkeypatch):
    monkeypatch.setenv("BEAM_BIND_ADDRESS", "10.0.0.5")
    assert _config.bind_address() == "10.0.0.5"


@pytest.mark.parametrize("raw", ["localhost", "10.0.0.5:6379", "*", "example.com"])
def test_bind_address_rejects_non_address(monkeypatch, raw):
    monkeypatch.setenv("BEAM_BIND_ADDRESS", raw)
    with pytest.raises(_config.ConfigError, match="BEAM_BIND_ADDRESS"):
        _config.bind_address()


def test_bind_address_rejects_multicast(monkeypatch):
    monkeypatch.setenv("BEAM_BIND_ADDRESS", "239.1.1.1")
    with pytest.raises(_config.ConfigError, match="unicast"):
        _config.bind_address()


# ---- runtime dir / sock ----------------------------------------------------


def test_runtime_dir_env(monkeypatch):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", "/var/run/beam")
    assert _config.runtime_dir() == "/var/run/beam"
    assert _config.runtime_json_path() == "/var/run/beam/daemon.json"
    assert _config.runtime_sock_path() == "/var/run/beam/daemon.sock"


def test_runtime_dir_default(monkeypatch):
    monkeypatch.setenv("HOME", "/home/op")
    assert _config.runtime_dir() == os.path.join("/home/op", ".beam")


def test_runtime_sock_env_wins(monkeypatch):
    monkeypatch.setenv("BEAM_SOCK", "/explicit.sock")
    monkeypatch.setenv("BEAM_RUNTIME_DIR", "/ignored")
    assert _config.runtime_sock() == "/explicit.sock"


def test_runtime_sock_from_runtime_dir(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(os.path.join(str(tmp_path), "daemon.json"), "w") as f:
        json.dump({"sock": "/from/dir.sock"}, f)
    assert _config.runtime_sock() == "/from/dir.sock"


@pytest.mark.parametrize("body", ["{not json", "[1, 2]", '{"pid": 7}', '{"sock": ""}'])
def test_runtime_sock_malformed_doc_is_none(monkeypatch, tmp_path, body):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path))
    with open(os.path.join(str(tmp_path), "daemon.json"), "w") as f:
        f.write(body)
    assert _config.runtime_sock() is None  # caller reports "no daemon running"


def test_runtime_sock_missing_doc_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("BEAM_RUNTIME_DIR", str(tmp_path / "nope"))
    assert _config.runtime_sock() is None


# ---- worker handoff ids ----------------------------------------------------


def test_gpu_ids_unset_is_empty():
    assert _config.gpu_ids() == []
    assert _config.accelerator_ids() == []


def test_gpu_ids_parses(monkeypatch):
    monkeypatch.setenv("BEAM_GPU_IDS", "0,1,2")
    assert _config.gpu_ids() == [0, 1, 2]
    assert _config.accelerator_ids() == ["0", "1", "2"]


@pytest.mark.parametrize("raw", ["gpu0", "0,x", "0,,x", "-", "1.5"])
def test_gpu_ids_rejects_garbage(monkeypatch, raw):
    monkeypatch.setenv("BEAM_GPU_IDS", raw)
    with pytest.raises(_config.ConfigError, match="BEAM_GPU_IDS"):
        _config.gpu_ids()


@given(st.lists(st.integers(min_value=0, max_value=15), max_size=8))
def test_fuzz_gpu_ids_roundtrip(ids):
    import unittest.mock as mock

    env_val = ",".join(str(i) for i in ids)
    with mock.patch.dict(os.environ, {"BEAM_GPU_IDS": env_val}, clear=False):
        assert _config.gpu_ids() == ids
        assert _config.accelerator_ids() == [str(i) for i in ids]


# ---- worker_cmd ------------------------------------------------------------


def test_worker_cmd_default_and_override(monkeypatch):
    assert _config.worker_cmd() == "python3 -m ray._worker"
    monkeypatch.setenv("BEAM_WORKER_CMD", "/opt/venv/bin/python -m ray._worker")
    assert _config.worker_cmd() == "/opt/venv/bin/python -m ray._worker"
    monkeypatch.setenv("BEAM_WORKER_CMD", "")
    assert _config.worker_cmd() == "python3 -m ray._worker"  # empty is unset
