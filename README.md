<p align="center">
  <img src="docs/logo.svg" alt="beam" width="320">
</p>

<p align="center">
  <b>A drop-in alternative to <a href="https://ray.io">Ray</a>, scoped to one job:<br>making vLLM distributed inference work across nodes.</b>
</p>

<p align="center">
  <a href="https://github.com/maci0/beam/actions/workflows/ci.yml"><img src="https://github.com/maci0/beam/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="coverage">
  <img src="https://img.shields.io/badge/tests-273-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/mypy-strict-blue" alt="mypy strict">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python 3.9+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="license AGPL-3.0"></a>
</p>

It implements only the slice of Ray that vLLM's `RayDistributedExecutor` uses:
cluster membership, GPU accounting, placement groups, and an actor-call hub.
The heavy tensor-parallel traffic still goes over NCCL/torch.distributed, exactly
as with real Ray, so beam stays small. Pure Python, no build step, one dependency.
([vLLM parallelism & scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/).)

**~1,470 lines, 124 KB, 1 dependency** vs Ray's 644k Python LoC / 183 MB install
(see [docs/DESIGN.md](docs/DESIGN.md#size-vs-ray)).

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — rationale and scope (why it's this small, the contract)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, topology, end-to-end startup walkthrough
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — wire format and every message type
- [docs/API.md](docs/API.md) — the full ray surface implemented, mapped to daemon ops
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — multi-node deploy, NCCL/RoCE, memory, troubleshooting
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — file map, tests, keeping the shim in sync with vLLM
- [test/dgx/README.md](test/dgx/README.md) — the two-node DGX Spark harness

## Deploy into the stock vllm-openai image (no rebuild)

The `vllm/vllm-openai` image no longer ships ray, so nothing shadows the shim,
and beam runs on the image's own python. Inject it with **one read-only bind
mount**; `PYTHONPATH` points python at the shim, and beam writes a `ray` launcher
itself on first start.

    mkdir -p /opt/beam
    cp -r python examples /opt/beam/      # /opt/beam/{python,examples}
    # or grab the bundle from a release:
    #   curl -sL https://github.com/maci0/beam/releases/latest/download/beam-bindmount-<tag>.tar.gz | tar xz -C /opt/beam --strip-components=0

    # head node (stock image, one mount, image python as entrypoint)
    docker run -d --name beam-head --network host --gpus all --ipc host \
        -v /opt/beam:/opt/beam:ro -e PYTHONPATH=/opt/beam/python \
        --entrypoint python3 vllm/vllm-openai:latest \
        -m ray start --head --port 6379 --block

    # each worker node
    docker run -d --name beam-worker --network host --gpus all --ipc host \
        -v /opt/beam:/opt/beam:ro -e PYTHONPATH=/opt/beam/python \
        --entrypoint python3 vllm/vllm-openai:latest \
        -m ray start --address <HEAD_IP>:6379 --block

    docker exec beam-head python3 -m ray status

    # serve, pointing vLLM at the ray backend (runs in the head container)
    docker exec beam-head vllm serve <model> \
        --tensor-parallel-size 2 \
        --distributed-executor-backend ray

    # then hit it
    docker exec beam-head curl -s localhost:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"<model>","prompt":"The capital of France is","max_tokens":8}'

vLLM's `import ray`, placement-group creation, and per-worker RPCs are served by
beam. GPUs are detected from `/dev/nvidia*`; override with `--num-gpus N` or
`BEAM_NUM_GPUS` (DGX Spark / GB10 needs the override, its device nodes are not
`/dev/nvidia*`).

beam implements vLLM's MessageQueue ray executor, not Ray Compiled Graph.
Recent vLLM selects that executor by default; only **older vLLM** versions need
it forced with `-e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`.

`test/dgx/` is a ready-made harness for two DGX Spark nodes (deploy, up,
control-plane test, vllm-test) and is verified end-to-end on real GB10 nodes. See
`test/dgx/README.md`.

### Bare metal (no container)

    uv pip install ./python       # provides `import ray` and the `ray` command
    ray start --head --port 6379          # head, blocks
    ray start --address <HEAD_IP>:6379    # each worker

## Verify without GPUs

Both checks fake the GPU count and need no torch/CUDA (only `uv` + cloudpickle):

    bash test/run_e2e.sh          # single head, 4 fake GPUs
    bash test/run_multinode.sh    # GPU-less head + a 4-GPU worker, routed through the hub

## Validated topologies

Each row ran end to end on real hardware (or fake-GPU control plane where noted).
The data plane is NCCL/RCCL on GPU backends, or gloo (torch.distributed) on the
CPU/Vulkan backend; beam carries only the control plane either way.

| topology | hardware | result | harness |
|---|---|---|---|
| Single node, TP=1 | 1× GB10 (NVIDIA DGX Spark) | inference ✓ | `test/dgx/dgx.sh` |
| 2 nodes, TP=2 | 2× DGX Spark, NCCL over **RoCE + TCP** | inference ✓ | `test/dgx/dgx.sh` |
| **CPU head + 2 GPU workers**, TP=2 | AMD CPU head (0 GPU, no vLLM) + 2 NVIDIA sparks over RoCE | inference ✓, head is pure control plane on a *different arch*; driver runs on a worker | `test/run_cpuhead_gpuworkers.sh` |
| 3-node CPU control plane | 3 machines (AMD + 2 NVIDIA), fake GPUs | 1 actor/node, cross-node placement ✓ | `test/run_cpu_cluster.sh` (N=3) |
| 4-node CPU control plane | 4 machines, fake GPUs | 1 actor/node, cross-node placement ✓ | `test/run_cpu_cluster.sh` |
| Single node AMD ROCm, TP=1 | RX 7900 XTX, vLLM 0.23 | inference ✓ | `test/run_rocm.sh` |
| Cross-node AMD control plane | 2 AMD nodes | placement + RCCL **init** on both ranks ✓ | `test/run_rocm_2node.sh` |
| **2 nodes, TP=2, Vulkan over gloo** | RX 7900 XTX (gfx1100) + RX 6900 XT (gfx1030), heterogeneous, vllm-vulkan, all-reduce over **gloo/TCP** | inference ✓ (correct output, full data plane closed) | manual 2-node vllm-vulkan |
| **2 nodes, TP=2, cross-arch + cross-vendor** | **x86_64 AMD RX 6900 XT (RDNA2)** + **aarch64 NVIDIA GB10 (Blackwell)**, vllm-vulkan, all-reduce over **gloo/TCP** | inference ✓ (correct output) | manual 2-node vllm-vulkan |

CPU-only rows fake the GPU count (`BEAM_NUM_GPUS`) to exercise membership,
placement, and actor RPC without devices.

The **Vulkan-over-gloo** rows close the full cross-node data plane end to end
(MessageQueue broadcast/response over zmq **and** the tensor-parallel all-reduce
over gloo). The last row does it across **two CPU architectures and two GPU
vendors at once** — half the model sharded onto an x86 AMD RDNA2 card, half onto
an aarch64 NVIDIA Blackwell GPU, each computing on Vulkan, reducing over gloo;
beam ships the same actor bytecode to both (same Python minor version) and the
node-agnostic control plane does not care about arch or vendor. Cross-node
**RCCL** *completion*
specifically is the remaining open item: blocked every time by hardware/cloud
availability (homogeneous 2-node NVIDIA/AMD with a real RDMA network), never by
beam. Harnesses for it are ready: `test/run_rocm_cluster.sh` (SSH-into-node) and
`test/run_rocm_azure.sh` (VM + docker).

Two gotchas surfaced closing the gloo row, both worth knowing for any multi-node
deploy (see [docs/OPERATIONS.md](docs/OPERATIONS.md)):

- **Node IP on multi-homed hosts.** vLLM advertises its zmq queues at
  `ray.util.get_node_ip_address()`. The default heuristic picks the default-route
  interface, which on a router/multi-NIC box is often not the cluster LAN. Set
  `BEAM_NODE_IP` (or `VLLM_HOST_IP`) per node to the reachable address.
- **Bidirectional reachability.** The all-reduce and the per-worker response
  queues need the driver to reach the workers (driver→worker), not just
  worker→driver. A default-deny host firewall (e.g. ufw) that allows the
  worker→head direction can still silently stall the head→worker queues. Open the
  cluster subnet between nodes.
- **Per-node interpreter for heterogeneous images.** beam launches each actor
  with `BEAM_WORKER_CMD` (default `python3 -m ray._worker`). When a node's image
  keeps vLLM in a venv rather than the system Python (so `python3` can't import
  it), set `BEAM_WORKER_CMD` on that node's daemon container to the venv's
  interpreter, e.g. `-e BEAM_WORKER_CMD='/opt/venv/bin/python -m ray._worker'`.
  Nodes may run different images/arches; only the Python *minor* version must
  match across the cluster (cloudpickle bytecode).

## Keep the shim in sync with vLLM

`scripts/scan_vllm_ray.py` statically scans a vLLM checkout for every `ray.*`
symbol it uses and checks the shim covers it (out-of-scope features like
ray.data / ray.serve / compiled-DAG are reported, not failed). Run it on a vLLM
bump as a CI gate:

    git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm
    uv run --with cloudpickle python scripts/scan_vllm_ray.py --src /tmp/vllm

## Environment

| var | meaning |
|-----|---------|
| `BEAM_NUM_GPUS`    | override detected GPU count |
| `BEAM_RUNTIME_DIR` | daemon state dir (default `~/.beam`) |
| `BEAM_SOCK`        | daemon unix socket (else read from the runtime dir) |
| `BEAM_WORKER_CMD`  | how to launch a python actor (default `python3 -m ray._worker`) |

## Not implemented (by design)

Object store for large data, fault tolerance/actor restart, autoscaling, the
dashboard, non-actor tasks, and `VLLM_USE_RAY_COMPILED_DAG`. vLLM's distributed
inference path needs none of these.

## Security / trust model

beam's control plane is **unauthenticated**, exactly like Ray's. The head binds
its TCP port (default 6379) on `0.0.0.0`, and the protocol carries cloudpickled
payloads that worker daemons unpickle and execute. Anyone who can reach the port
can run code as the daemon user. **Run it only on a trusted, private network**
(a cluster subnet / VPC), never exposed to the internet. This is the same
posture Ray documents for its own 6379. Set `--node-ip` to advertise a specific
address; keep the port behind your firewall/security group.

## License

[GNU AGPL-3.0-or-later](LICENSE). Copyleft: changes to beam stay open, and
because it runs as an inference service, the AGPL network clause means anyone
offering a modified beam over a network must make their source available. beam
imports cleanly into other-licensed code (vLLM is Apache-2.0); the copyleft
covers beam and its derivatives, not the model you serve or the rest of your
stack.
