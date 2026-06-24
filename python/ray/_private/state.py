"""``ray._private.state``: per-node resource views vLLM reads during cluster
initialization. Backed by the daemon's status, keyed by node id.
"""


def available_resources_per_node() -> dict:
    from .. import _need

    resp, _ = _need().request({"t": "resources"})
    return resp.get("data") or {}


def total_resources_per_node() -> dict:
    from .. import _status_nodes

    return {n["node"]: {"GPU": float(n.get("ngpu", 0)), "CPU": 1.0} for n in _status_nodes()}


class _State:
    _available_resources_per_node = staticmethod(available_resources_per_node)


# vLLM's fallback path: `from ray._private.state import state as _state`
state = _State()
