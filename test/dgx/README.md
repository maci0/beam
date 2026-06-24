# DGX Spark test harness

Runs beam across two DGX Spark nodes using the **stock `vllm/vllm-openai` image,
unmodified**. beam is injected at runtime with a single read-only bind mount; the
image is never rebuilt.

## Why bind-mount, not a custom image or a sidecar

- The `ray` shim must be importable **in vLLM's own process** (`import ray`), and
  the actor subprocesses beam spawns (`python -m ray._worker`) need vLLM's torch
  env. A sidecar daemon container can provide neither. So beam has to live inside
  the vLLM container.
- Rebuilding the image to bake it in works but is heavyweight and you asked not
  to. Instead we mount one directory and let beam wire itself in on startup.

One mount: `-v /opt/beam:/opt/beam:ro`, with `-e PYTHONPATH=/opt/beam/python` so
`import ray` resolves to the mounted shim (no pip install). The container's
entrypoint is the image's own `python3` running `-m ray start`. On `ray start`,
beam (detecting the container via `/.dockerenv`) self-bootstraps by writing
`ray` and `beam` launcher scripts into `/usr/local/bin` so `docker exec ... ray
status` works too.

Both head and worker run the same stock image with the same mount.

## Layout inside the container

    /opt/beam/python/ray      the import-ray shim (pure Python, no binary)
    /opt/beam/examples        driver_demo.py, import_check.py

## Usage

Edit `config.sh` (node IPs, model), then from your workstation (which has SSH
access to both nodes):

    ./dgx.sh all          # deploy -> up -> status -> imports -> cp-test
    ./dgx.sh vllm-test    # vllm serve, TP across both nodes over RoCE, + a request
    ./dgx.sh down

Step by step:

    ./dgx.sh deploy       # rsync the shim + examples to /opt/beam on both nodes
    ./dgx.sh up           # docker run head + worker (stock image, one :ro mount each)
    ./dgx.sh status       # beam status from the head
    ./dgx.sh imports      # import-only shim smoke test in the head container
    ./dgx.sh cp-test      # 2 actors, one per node: cross-node control plane
    ./dgx.sh vllm-test    # the real thing: vllm serve --distributed-executor-backend ray

## What each test proves

- **imports**: every ray symbol vLLM touches resolves in the mounted shim.
- **cp-test**: the head places one actor per node, broadcasts calls, and gathers
  results across the network. This is beam's whole job; it runs without a model.
- **vllm-test**: vLLM brings up tensor-parallel workers (one rank per node) using
  beam as the ray backend; NCCL/RoCE carries the tensors, beam carries the
  control plane. A completion request confirms the full path.

## NCCL / RoCE and memory (vllm-test only)

The control-plane tests (`cp-test`) need no NCCL and pass over plain TCP. `vllm-test`
runs the real tensor-parallel collectives, which needs two things sorted on the
nodes (both are cluster config, not beam, which never touches the data plane):

- **RDMA access in the container.** `--network host` is not enough, and a
  `-v /dev/infiniband` bind mount exposes the device nodes but docker's
  device-cgroup still blocks them (which is what made `--privileged` look
  necessary). The fix is `--device /dev/infiniband` (the flag recurses the
  directory, granting the cgroup permission for every node inside, same as
  NVIDIA's recipes) plus `--cap-add IPC_LOCK --ulimit memlock=-1:-1`. `config.sh`
  sets `RDMA_ARGS` accordingly, no privileged. Without device access NCCL logs
  `Unable to open device rocep*` and falls back to TCP; with it:
  `NCCL INFO NET/IB : Using rocep1s0f1/RoCE roceP2p1s0f1/RoCE`.
- **The right HCA.** `NCCL_EXTRA` names the two cabled RoCE ports
  (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`) with bootstrap over the mgmt net
  (`NCCL_SOCKET_IFNAME=enP7s7`). `NCCL_DEBUG=INFO` shows the selection.
- **Unified memory.** GB10 shares 128 GiB between GPU and system, so the default
  `--gpu-memory-utilization 0.9` (≈107 GiB KV cache) gets the worker OOM-killed
  (exit 137). `SERVE_EXTRA` caps it to 0.5 and adds `--enforce-eager`.

## Validated locally

The bind-mount + self-bootstrap flow and cross-node actor placement were verified
in two stock `python:3.12-slim` containers on a docker network (see the project
README). On the Spark nodes the only differences are real GPUs and NCCL over RoCE
for `vllm-test`.
