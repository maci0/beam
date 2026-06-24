"""beam: a drop-in subset of the ``ray`` API, scoped to what vLLM's
RayDistributedExecutor uses for distributed inference.

Only the surface vLLM imports is implemented. See DESIGN.md for the contract.
"""

from __future__ import annotations  # keep PEP604 annotations valid on py3.9

import os
import socket
import time
from collections.abc import Iterable
from typing import Any

from . import (
    _proto,
    util,  # noqa: F401  (exposes ray.util.*)
)
from ._client import DaemonClient
from .util import (  # noqa: F401
    PlacementGroup,
    get_current_placement_group,
    placement_group,
    remove_placement_group,
)

_client: DaemonClient | None = None
# Report a recent ray version so vLLM's `ray.__version__` / metadata checks pass.
# beam tracks ray's distributed-executor API surface, not its release number.
__version__ = "2.43.0"


# ---- lifecycle ----


def init(
    address: str | None = None,
    *args: Any,
    ignore_reinit_error: bool = False,
    **kwargs: Any,
) -> _RuntimeContext | None:
    global _client
    if _client is not None:
        if ignore_reinit_error:
            return None
        raise RuntimeError("ray already initialized")
    _client = DaemonClient()
    return _RuntimeContext()


def is_initialized() -> bool:
    return _client is not None


def shutdown(*args: Any, **kwargs: Any) -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def _need() -> DaemonClient:
    if _client is None:
        raise RuntimeError("ray is not initialized; call ray.init() first")
    return _client


# ---- object refs ----


class ObjectRef:
    __slots__ = ("id", "_value", "_has_value")

    def __init__(self, obj_id: str, value: Any = None, has_value: bool = False) -> None:
        self.id = obj_id
        self._value = value
        self._has_value = has_value

    # equal by id, like real ray, so refs work as dict keys / in membership tests
    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjectRef) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return "ObjectRef(%s)" % self.id


def put(obj: Any) -> ObjectRef:
    resp, _ = _need().request({"t": "put"}, _proto.dumps(obj))
    return ObjectRef(resp["obj"])


def get(refs: ObjectRef | Iterable[ObjectRef], timeout: float | None = None) -> Any:
    from .exceptions import GetTimeoutError

    single = isinstance(refs, ObjectRef)
    items: list[ObjectRef] = [refs] if isinstance(refs, ObjectRef) else list(refs)
    deadline = None if timeout is None else time.time() + timeout
    out = []
    for ref in items:
        if getattr(ref, "_has_value", False):
            out.append(ref._value)
            continue
        req: dict[str, Any] = {"t": "get", "obj": ref.id}
        if deadline is not None:
            req["timeout"] = max(0.0, deadline - time.time())  # one global deadline
        try:
            _, body = _need().request(req)
        except RuntimeError as e:
            if "GetTimeoutError" in str(e):
                raise GetTimeoutError(str(e)) from None
            raise
        out.append(_proto.loads(body) if body else None)
    return out[0] if single else out


def wait(
    refs: Iterable[ObjectRef],
    *,
    num_returns: int = 1,
    timeout: float | None = None,
    **kwargs: Any,
) -> tuple[list[ObjectRef], list[ObjectRef]]:
    refs = list(refs)
    num_returns = min(num_returns, len(refs))  # never block waiting for more than exist
    deadline = None if timeout is None else time.time() + timeout
    while True:
        ready: list[ObjectRef] = []
        not_ready: list[ObjectRef] = []
        for ref in refs:
            if getattr(ref, "_has_value", False):
                ready.append(ref)
                continue
            resp, _ = _need().request({"t": "stat", "obj": ref.id})
            (ready if resp.get("ready") else not_ready).append(ref)
        if len(ready) >= num_returns or (deadline and time.time() >= deadline):
            return ready, not_ready
        time.sleep(0.005)


# ---- actors ----


class _RemoteMethod:
    def __init__(self, handle: ActorHandle, name: str) -> None:
        self._handle = handle
        self._name = name

    def remote(self, *args: Any, **kwargs: Any) -> ObjectRef:
        payload = _proto.dumps((args, kwargs))
        resp, _ = _need().request(
            {"t": "call", "actor": self._handle._actor_id, "method": self._name},
            payload,
        )
        return ObjectRef(resp["obj"])


class ActorHandle:
    def __init__(self, actor_id: str) -> None:
        self._actor_id = actor_id

    def __getattr__(self, name: str) -> _RemoteMethod:
        if name.startswith("__"):
            raise AttributeError(name)
        return _RemoteMethod(self, name)


class _RemoteClass:
    def __init__(self, cls: type, options: dict) -> None:
        self._cls = cls
        self._options = options

    def options(self, **opts: Any) -> _RemoteClass:
        merged = dict(self._options)
        merged.update(opts)
        return _RemoteClass(self._cls, merged)

    def remote(self, *args: Any, **kwargs: Any) -> ActorHandle:
        opts = self._options
        num_gpus = float(opts.get("num_gpus", 0) or 0)  # keep fractional (0.5) intact
        pg_id, bundle = "", 0
        strategy = opts.get("scheduling_strategy")
        if strategy is not None and getattr(strategy, "placement_group", None):
            pg_id = strategy.placement_group.id
            bundle = strategy.placement_group_bundle_index
        header = {
            "t": "create_actor",
            "ngpu": num_gpus,
            "pg": pg_id,
            "bundle": bundle,
        }
        payload = _proto.dumps((self._cls, args, kwargs))
        resp, _ = _need().request(header, payload)
        return ActorHandle(resp["actor"])


def kill(actor: Any, *args: Any, **kwargs: Any) -> None:
    """Terminate an actor's worker subprocess (ray.kill)."""
    if isinstance(actor, ActorHandle):
        _need().request({"t": "kill", "actor": actor._actor_id})


def remote(*args: Any, **options: Any) -> Any:
    """``ray.remote`` as a bare decorator or with options.

    Supports the two forms vLLM uses:
        @ray.remote
        class W: ...
    and
        ray.remote(num_gpus=1, scheduling_strategy=...)(W).remote(...)
    """

    def wrap(cls: Any) -> _RemoteClass:
        if isinstance(cls, _RemoteClass):  # tolerate re-decoration
            merged = dict(cls._options)
            merged.update(options)
            return _RemoteClass(cls._cls, merged)
        return _RemoteClass(cls, options)

    if len(args) == 1 and callable(args[0]) and not options:
        return wrap(args[0])
    return wrap


# ---- runtime context / resources ----


class _RuntimeContext:
    def get_node_id(self) -> str:
        return os.environ.get("BEAM_NODE_ID") or _local_node_id()

    def get_accelerator_ids(self) -> dict[str, list[str]]:
        ids = os.environ.get("BEAM_GPU_IDS", "")
        return {"GPU": [g for g in ids.split(",") if g]}

    # some vLLM paths read .gpu_ids directly
    @property
    def gpu_ids(self) -> list[int]:
        return get_gpu_ids()


def get_runtime_context() -> _RuntimeContext:
    return _RuntimeContext()


def get_gpu_ids() -> list[int]:
    ids = os.environ.get("BEAM_GPU_IDS", "")
    return [int(g) for g in ids.split(",") if g]


def _status_nodes() -> list[dict]:
    resp, _ = _need().request({"t": "status"})
    return resp.get("nodes") or []


def cluster_resources() -> dict[str, float]:
    res: dict[str, float] = {}
    for n in _status_nodes():
        res["GPU"] = res.get("GPU", 0.0) + n.get("ngpu", 0)
        res["CPU"] = res.get("CPU", 0.0) + 1.0
    return res


def available_resources() -> dict[str, float]:
    res: dict[str, float] = {}
    for n in _status_nodes():
        free = n.get("ngpu", 0) - n.get("used", 0)
        res["GPU"] = res.get("GPU", 0.0) + max(0, free)
        res["CPU"] = res.get("CPU", 0.0) + 1.0
    return res


def nodes() -> list[dict]:
    out = []
    for n in _status_nodes():
        out.append(
            {
                "NodeID": n.get("node"),
                "Alive": n.get("alive", True),
                "NodeManagerAddress": n.get("ip", ""),
                "Resources": {"GPU": float(n.get("ngpu", 0)), "CPU": 1.0},
            }
        )
    return out


def _local_node_id() -> str:
    import json

    rt_dir = os.environ.get("BEAM_RUNTIME_DIR") or os.path.join(os.path.expanduser("~"), ".beam")
    try:
        with open(os.path.join(rt_dir, "daemon.json")) as f:
            return json.load(f)["node"]
    except OSError:
        return "driver"


def _get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
