"""``ray.util.state.list_nodes`` (used only on vLLM's elastic data-parallel
scaling path). Backed by the daemon status.
"""

from __future__ import annotations  # keep PEP585 generics valid on py3.9

from typing import Any


class _NodeState:
    def __init__(self, node_id: str, node_ip: str, state: str = "ALIVE") -> None:
        self.node_id = node_id
        self.node_ip = node_ip
        self.state = state


def list_nodes(*args: Any, **kwargs: Any) -> list[_NodeState]:
    from .. import _status_nodes

    return [
        _NodeState(n["node"], n.get("ip", ""), "ALIVE" if n.get("alive") else "DEAD")
        for n in _status_nodes()
    ]
