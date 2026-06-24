"""End-to-end demo of the beam `ray` shim, mirroring how vLLM's
RayDistributedExecutor drives workers: build a placement group with one GPU
bundle per worker, spawn one actor per bundle, then broadcast method calls and
gather results.

No GPUs or torch required: BEAM_NUM_GPUS fakes the device count and the actor
just reports the GPU id beam assigned it.
"""

import os

import ray
from ray.util import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

WORLD_SIZE = int(os.environ.get("BEAM_DEMO_WORLD", "4"))


class Worker:
    def __init__(self, rank):
        self.rank = rank

    def identity(self):
        return {
            "rank": self.rank,
            "node": ray.get_runtime_context().get_node_id(),
            "gpu_ids": ray.get_gpu_ids(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def echo(self, x):
        return x * 10 + self.rank


def main():
    ray.init()
    assert ray.is_initialized()

    print("cluster_resources:", ray.cluster_resources())

    pg = placement_group([{"GPU": 1}] * WORLD_SIZE)
    ray.get(pg.ready())

    workers = []
    for i in range(WORLD_SIZE):
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=i,
            placement_group_capture_child_tasks=True,
        )
        w = ray.remote(num_gpus=1, scheduling_strategy=strategy)(Worker).remote(i)
        workers.append(w)

    # broadcast + gather, the core vLLM control-plane pattern
    infos = ray.get([w.identity.remote() for w in workers])
    for info in infos:
        print("worker:", info)

    echoes = ray.get([w.echo.remote(i) for i, w in enumerate(workers)])
    print("echoes:", echoes)

    # one actor per bundle, each with exactly one GPU; results gathered in order
    assert echoes == [i * 10 + i for i in range(WORLD_SIZE)], echoes
    assert all(len(info["gpu_ids"]) == 1 for info in infos), infos
    assert ray.cluster_resources()["GPU"] >= float(WORLD_SIZE)
    nodes_used = {info["node"] for info in infos}
    print("nodes used:", len(nodes_used), sorted(nodes_used))
    # multinode/3node harnesses set this to assert actors actually spread across
    # nodes (not all piled on one), making cross-node placement a real gate.
    expect = int(os.environ.get("BEAM_DEMO_EXPECT_NODES", "0"))
    if expect:
        assert len(nodes_used) == expect, "expected %d nodes, used %d" % (expect, len(nodes_used))

    for w in workers:
        ray.kill(w)
    ray.shutdown()
    print("OK")


if __name__ == "__main__":
    main()
