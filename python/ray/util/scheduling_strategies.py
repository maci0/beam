"""Scheduling strategies vLLM passes to ``ray.remote(...)``."""


class PlacementGroupSchedulingStrategy:
    def __init__(
        self,
        placement_group,
        placement_group_bundle_index=-1,
        placement_group_capture_child_tasks=None,
    ):
        self.placement_group = placement_group
        self.placement_group_bundle_index = placement_group_bundle_index
        self.placement_group_capture_child_tasks = placement_group_capture_child_tasks


class NodeAffinitySchedulingStrategy:
    def __init__(self, node_id, soft=False, **kwargs):
        self.node_id = node_id
        self.soft = soft
