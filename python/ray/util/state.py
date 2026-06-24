"""``ray.util.state.list_nodes`` (used only on vLLM's elastic data-parallel
scaling path). Backed by the daemon status.
"""


class _NodeState:
    def __init__(self, node_id, node_ip, state="ALIVE"):
        self.node_id = node_id
        self.node_ip = node_ip
        self.state = state


def list_nodes(*args, **kwargs):
    from .. import _status_nodes

    return [
        _NodeState(n["node"], n.get("ip", ""), "ALIVE" if n.get("alive") else "DEAD")
        for n in _status_nodes()
    ]
