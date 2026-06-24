"""``ray.util`` subset: placement groups, scheduling strategies, helpers."""

from __future__ import annotations  # keep `X | None` valid on py3.9

from typing import Any

from . import scheduling_strategies  # noqa: F401
from .placement_group import (  # noqa: F401
    PlacementGroup,
    get_current_placement_group,
    placement_group,
    remove_placement_group,
)


def get_node_ip_address() -> str:
    from .. import _get_ip

    return _get_ip()


def placement_group_table(pg: Any = None) -> dict:
    """ray.util.placement_group_table: bundle index -> node id and spec.

    With a placement group, returns one table; without, all tables keyed by id.
    """
    from .. import _need

    header = {"t": "pg_table"}
    if pg is not None:
        header["pg"] = str(pg.id)
    resp, _ = _need().request(header)
    data = resp.get("data") or {}

    def table(bundle_list: list[dict]) -> dict:
        return {
            "bundles_to_node_id": {i: b["node"] for i, b in enumerate(bundle_list)},
            "bundles": {i: b["spec"] for i, b in enumerate(bundle_list)},
            "state": "CREATED",
        }

    if pg is not None:
        return table(data.get("bundles", []))
    return {pgid: table(bl) for pgid, bl in (data.get("pgs") or {}).items()}
