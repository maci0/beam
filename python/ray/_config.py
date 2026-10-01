"""Every environment variable beam reads for a deploy, in one place, validated once.

Config used to be read ad hoc (`os.environ.get(...)` at each call site), so a
mistyped value surfaced only at the moment it mattered: `BEAM_NUM_GPUS=8 ` with
a stray character silently fell back to "count /dev/nvidia*", a non-numeric one
crashed with a bare ValueError inside a traceback, and a garbage `BEAM_NODE_IP`
was advertised to every peer until the cluster hung. Read each value here
instead, so a bad value is rejected with a message naming the variable.

Each loader takes an `override` that the caller already parsed (a CLI flag the
caller validated) and returns it untouched; otherwise it reads and validates the
variable. `ConfigError` messages start with "beam:" so every config failure
reaches the operator the same way.

The determinism seams (BEAM_TIMEOUT / BEAM_SLEEP / BEAM_CLOCK / BEAM_SEED) are
not here: they are test/simulation hooks read directly by the daemon and the shim
rather than through a loader, so BEAM_TIMEOUT, BEAM_SEED and the shim's
BEAM_SLEEP can be changed mid-process while BEAM_CLOCK and the daemon's BEAM_SLEEP
are captured once at import. See docs/DEVELOPMENT.md.

Documented in README.md ("Environment") and docs/OPERATIONS.md. The pure handoff
vars (BEAM_NODE_ID / BEAM_GPU_IDS / BEAM_ACTOR_ID) are produced by the daemon on
the actor subprocess and have no parser here. BEAM_SOCK is the one exception: the
daemon sets it per actor, and runtime_sock() also honors it as an operator
override on the `sock` path recorded in daemon.json.
"""

from __future__ import annotations  # keep PEP604 annotations valid on py3.9

import ipaddress
import json
import os
import socket

__all__ = [
    "ConfigError",
    "accelerator_ids",
    "bind_address",
    "gpu_ids",
    "local_node_id",
    "node_ip",
    "num_gpus",
    "read_runtime_doc",
    "route_probe_ip",
    "runtime_dir",
    "runtime_json_path",
    "runtime_sock",
    "runtime_sock_path",
    "worker_cmd",
]

# Daemon state dir used when BEAM_RUNTIME_DIR is unset: the "~/.beam" the usage
# text, README, and OPERATIONS all document.
_RUNTIME_DIR = ".beam"
# Read in this order, so beam advertises the same address vLLM and ray's own
# get_node_ip_address see (vLLM sets VLLM_HOST_IP).
_NODE_IP_VARS = ("BEAM_NODE_IP", "VLLM_HOST_IP")
# SOCK_DGRAM connect sends no packet: it only reads back the kernel's chosen
# source address for the default route.
_ROUTE_PROBE_ADDR = ("8.8.8.8", 80)
_LOCALHOST = "127.0.0.1"
_ANY = "0.0.0.0"


class ConfigError(Exception):
    """An environment variable is set to a value beam cannot use."""


def _doc_str(doc: dict, key: str) -> str | None:
    """A non-empty string field of a runtime document, else None."""
    val = doc.get(key)
    return val if isinstance(val, str) and val else None


def _env(name: str) -> str | None:
    """Value of `name`, or None when unset or set to the empty string.

    Empty is treated as unset on purpose: `docker run -e BEAM_NODE_IP` and
    `BEAM_NODE_IP=` both expand to nothing, and advertising an empty address
    would break every peer connection, so the documented fallback chain is the
    only useful reading of an empty value.
    """
    val = os.environ.get(name)
    return val if val else None


def _address(raw: str, name: str) -> str:
    """Validated literal IPv4/IPv6 address, returned canonically."""
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        raise ConfigError(
            "beam: %s must be an IP address (e.g. 10.0.0.5 or fd00::1), got %r" % (name, raw)
        ) from None


# ---- node inventory ----


def num_gpus(override: int | None = None) -> int | None:
    """GPUs on this node: `override`, else BEAM_NUM_GPUS, else None for "detect
    from /dev/nvidia*" (the caller decides).

    The override is the caller's own count and is returned unchecked. BEAM_NUM_GPUS
    is operator input and is validated: non-numeric is a typo, negative means the
    node believes it has GPUs it does not have, and either way the daemon ends up
    publishing a wrong count that placement silently trusts.
    """
    if override is not None:
        return override
    raw = _env("BEAM_NUM_GPUS")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(
            "beam: BEAM_NUM_GPUS must be a non-negative integer, got %r "
            "(omit it to count /dev/nvidia*)" % raw
        ) from None
    if value < 0:
        raise ConfigError("beam: BEAM_NUM_GPUS must be >= 0, got %d" % value)
    return value


def node_ip(override: str | None = None) -> str | None:
    """Address this node advertises to its peers.

    `override` (ray start --node-ip) wins, then BEAM_NODE_IP, then VLLM_HOST_IP.
    None means nothing is configured: the caller probes the default route, which
    on a multi-homed host is often not the address peers can reach.
    """
    if override:
        return _address(override, "--node-ip")
    for name in _NODE_IP_VARS:
        raw = _env(name)
        if raw:
            return _address(raw, name)
    return None


def route_probe_ip() -> str:
    """Source address the kernel picks for the default route, else 127.0.0.1."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(_ROUTE_PROBE_ADDR)
        return str(sock.getsockname()[0])
    except OSError:
        return _LOCALHOST
    finally:
        sock.close()


# ---- daemon runtime state ----


def runtime_dir() -> str:
    """Directory holding daemon.json / daemon.sock, default ~/.beam."""
    return _env("BEAM_RUNTIME_DIR") or os.path.join(os.path.expanduser("~"), _RUNTIME_DIR)


def runtime_sock_path() -> str:
    """Unix socket path derived from runtime_dir()."""
    return os.path.join(runtime_dir(), "daemon.sock")


def runtime_json_path() -> str:
    """Path of the daemon.json runtime document."""
    return os.path.join(runtime_dir(), "daemon.json")


def runtime_sock() -> str | None:
    """BEAM_SOCK when set, else the socket recorded in daemon.json.

    None means "no runtime document" (missing or malformed), so the caller can
    report that no local daemon is running instead of leaking a bare
    FileNotFoundError or a KeyError from `["sock"]`.
    """
    sock = _env("BEAM_SOCK")
    return sock if sock else _doc_str(read_runtime_doc(), "sock")


def read_runtime_doc(path: str | None = None) -> dict:
    """A runtime document, or {} when it is missing or unreadable.

    `path` defaults to the live daemon.json. Every reader of that document (the
    shim's socket and node-id lookups, the CLI's status/stop/claim paths, and the
    seized or held copies they rename it to) goes through here, so the layout of
    the document and the "unreadable means empty" rule are defined once and one
    parse cannot hand a socket to one caller and a KeyError to another.
    """
    try:
        with open(path or runtime_json_path()) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def local_node_id(fallback: str = "driver") -> str:
    """Node id the local daemon published, else `fallback`.

    Read from the same runtime document as runtime_sock(), so a caller that
    found the socket also finds a node id, and one that found neither does not
    have to know the document's layout to say so.
    """
    return _doc_str(read_runtime_doc(), "node") or fallback


# ---- actor worker handoff ----


def gpu_ids() -> list[int]:
    """Device ids assigned to this actor (BEAM_GPU_IDS), [] when unset/empty.

    A malformed id raises ConfigError: get_gpu_ids() has no channel for "bad
    value", so accepting one would hand vLLM a silently wrong device selection.
    """
    raw = _env("BEAM_GPU_IDS")
    ids = _split_ids(raw)
    if not ids:
        return []
    try:
        return [int(g) for g in ids]
    except ValueError:
        raise ConfigError(
            "beam: BEAM_GPU_IDS must be comma-separated non-negative integers, got %r" % raw
        ) from None


def accelerator_ids() -> list[str]:
    """The same ids as strings, the form vLLM logs and compares."""
    return _split_ids(_env("BEAM_GPU_IDS"))


def _split_ids(raw: str | None) -> list[str]:
    return [g.strip() for g in raw.split(",") if g.strip()] if raw else []


def worker_cmd() -> str:
    """Command the daemon runs to launch an actor, default 'python3 -m ray._worker'."""
    return _env("BEAM_WORKER_CMD") or "python3 -m ray._worker"


# ---- network ----


def bind_address() -> str:
    """Address the head's TCP control port binds to, default 0.0.0.0.

    The control plane is unauthenticated, so 0.0.0.0 exposes it on every
    interface; set BEAM_BIND_ADDRESS to the cluster LAN address to narrow that.
    Empty is treated as unset, like every other variable here.
    """
    raw = _env("BEAM_BIND_ADDRESS")
    if not raw:
        return _ANY
    addr = _address(raw, "BEAM_BIND_ADDRESS (%s listens on every interface)" % _ANY)
    if ipaddress.ip_address(addr).is_multicast:
        raise ConfigError("beam: BEAM_BIND_ADDRESS must be a unicast address, got %r" % raw)
    return addr
