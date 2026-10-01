"""beam: a drop-in subset of the ``ray`` API, scoped to what vLLM's
RayDistributedExecutor uses for distributed inference.

Only the surface vLLM imports is implemented. See docs/DESIGN.md for the contract.
"""

from __future__ import annotations  # keep PEP604 annotations valid on py3.9

import os
import time
from collections.abc import Iterable
from typing import Any

from . import (
    _config,
    _proto,
    util,
)
from ._client import DaemonClient
from .util import (
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


# ray.wait has no push notification; poll the daemon at this interval.
_WAIT_POLL_INTERVAL = 0.005

# Timeouts are elapsed-time budgets, so they are measured on the monotonic
# clock: time.time() moves under an NTP step, a manual clock change, a
# leap-second smear, or a host suspend, which would make a deadline jump
# (an hour of the caller's wait skipped) or never arrive (a hang past the
# timeout). time.monotonic() is unaffected by those and never goes backwards.


def _deadline(timeout: float | None) -> float | None:
    """Deadline on the monotonic clock, or None for "wait forever"."""
    return None if timeout is None else time.monotonic() + timeout


def _remaining(deadline: float | None) -> float | None:
    """Seconds left before `deadline`, clamped at 0. None means no deadline."""
    return None if deadline is None else max(0.0, deadline - time.monotonic())


# ---- object refs ----


class ObjectRef:
    __slots__ = ("_has_value", "_value", "id")

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
    deadline = _deadline(timeout)  # one global deadline, monotonic
    out = []
    for ref in items:
        if ref._has_value:
            out.append(ref._value)
            continue
        req: dict[str, Any] = {"t": "get", "obj": ref.id}
        left = _remaining(deadline)
        if left is not None:
            req["timeout"] = left
        try:
            # Bound the socket read by the same budget the daemon gets, so an
            # unreachable daemon ends the get at the deadline instead of
            # hanging past it.
            _, body = _budgeted(_need(), req, left)
        except RuntimeError as e:
            if "GetTimeoutError" in str(e):
                raise GetTimeoutError(str(e)) from None
            raise
        except TimeoutError as e:
            raise GetTimeoutError(
                "GetTimeoutError: object %s not ready in %ss" % (ref.id, left)
            ) from e
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
    deadline = _deadline(timeout)
    client = _need()
    while True:
        ready: list[ObjectRef] = []
        not_ready: list[ObjectRef] = []
        for ref in refs:
            if ref._has_value:
                ready.append(ref)
                continue
            resp, _ = _stat(client, ref, _remaining(deadline))
            (ready if resp.get("ready") else not_ready).append(ref)
        if len(ready) >= num_returns or (deadline and time.monotonic() >= deadline):
            return ready, not_ready
        # Never sleep past the deadline: ray.wait is polled on a fixed cadence
        # (vLLM's liveness thread), so oversleeping here compounds every cycle.
        left = _remaining(deadline)
        if left is not None and left <= 0:
            return ready, not_ready
        time.sleep(_WAIT_POLL_INTERVAL if left is None else min(_WAIT_POLL_INTERVAL, left))


def _budgeted(client: Any, header: dict[str, Any], left: float | None) -> tuple[dict, bytes]:
    """One round-trip, bounded by the caller's remaining deadline budget.

    A request issued under a deadline must not block longer than that deadline,
    so the budget is passed to the client and applied to the socket read: a
    daemon that is slow, or a hop to a peer node that has dropped, would
    otherwise hold the caller well past the budget it was promised. Clients
    without a timeout parameter (test doubles) are simply called unbudgeted.
    """
    try:
        return client.request(header, timeout=left)
    except TypeError:  # client.request takes no budget
        return client.request(header)


def _stat(client: Any, ref: ObjectRef, left: float | None) -> tuple[dict, bytes]:
    """A `stat` that the caller can time out on: a daemon that misses the
    budget is reported not-ready (never an error), and the wait loop's own
    deadline check decides whether to poll again."""
    try:
        return _budgeted(client, {"t": "stat", "obj": ref.id}, left)
    except TimeoutError:
        return {"t": "stat_ok", "ready": False}, b""


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
        return {"GPU": _config.accelerator_ids()}

    # some vLLM paths read .gpu_ids directly
    @property
    def gpu_ids(self) -> list[int]:
        return get_gpu_ids()


def get_runtime_context() -> _RuntimeContext:
    return _RuntimeContext()


def get_gpu_ids() -> list[int]:
    return _config.gpu_ids()


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

    try:
        with open(os.path.join(_config.runtime_dir(), "daemon.json")) as f:
            return json.load(f)["node"]
    except OSError:
        return "driver"


def _get_ip() -> str:
    # Prefer an explicitly configured cluster IP. The route probe below returns
    # the default-route interface, which on a multi-homed host (router, VM
    # bridges) is often not the cluster LAN the other nodes reach us on.
    return _config.node_ip() or _config.route_probe_ip()
