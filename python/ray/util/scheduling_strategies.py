"""Scheduling strategies vLLM passes to ``ray.remote(...)``."""

from __future__ import annotations  # keep `X | None` valid on py3.9

from typing import Any


class PlacementGroupSchedulingStrategy:
    def __init__(
        self,
        placement_group: Any,
        placement_group_bundle_index: int = -1,
        placement_group_capture_child_tasks: bool | None = None,
    ) -> None:
        self.placement_group = placement_group
        self.placement_group_bundle_index = placement_group_bundle_index
        self.placement_group_capture_child_tasks = placement_group_capture_child_tasks


class NodeAffinitySchedulingStrategy:
    def __init__(self, node_id: str, soft: bool = False, **kwargs: Any) -> None:
        self.node_id = node_id
        self.soft = soft
