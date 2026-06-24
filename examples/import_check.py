"""Import-only smoke test: every ray symbol vLLM touches on the supported path
resolves. No daemon, no GPUs. Run inside the built image to catch a broken shim
install before launching a cluster.
"""

import importlib

import ray

REQUIRED = {
    "ray": [
        "init",
        "is_initialized",
        "shutdown",
        "remote",
        "get",
        "put",
        "wait",
        "kill",
        "get_gpu_ids",
        "get_runtime_context",
        "cluster_resources",
        "available_resources",
        "nodes",
        "ActorHandle",
    ],
    "ray.util": [
        "placement_group",
        "get_current_placement_group",
        "remove_placement_group",
        "placement_group_table",
        "get_node_ip_address",
    ],
    "ray.util.placement_group": ["PlacementGroup", "placement_group"],
    "ray.util.scheduling_strategies": ["PlacementGroupSchedulingStrategy"],
    "ray.util.state": ["list_nodes"],
    "ray.util.metrics": ["Metric", "Gauge", "Counter", "Histogram"],
    "ray._private.state": ["available_resources_per_node", "total_resources_per_node", "state"],
    "ray.runtime_env": ["RuntimeEnv"],
    "ray.exceptions": ["RayActorError", "GetTimeoutError"],
    "ray.actor": ["ActorHandle"],
    "ray.types": ["ObjectRef"],
    "ray.dag": ["CompiledDAG", "InputNode", "MultiOutputNode"],
}


def main():
    assert ray.__version__, "ray.__version__ missing"
    for mod_name, names in REQUIRED.items():
        mod = importlib.import_module(mod_name)
        for n in names:
            assert hasattr(mod, n), f"{mod_name}.{n} missing"
    print("import_check OK: ray %s, %d modules" % (ray.__version__, len(REQUIRED)))


if __name__ == "__main__":
    main()
