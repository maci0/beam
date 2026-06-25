# Operations

Running beam + vLLM across multiple nodes, and the cluster-config gotchas found
bringing it up on real hardware (2× DGX Spark / GB10, RoCE).

## Deploy: bind-mount into the stock image (no rebuild)

`vllm/vllm-openai` no longer ships ray, so nothing shadows the shim, and beam
runs on the image's own python. Inject it with one read-only bind mount.

Lay the package down on each node (host path is bind-mounted to `/opt/beam`):

```
cp -r python examples /opt/beam/        # /opt/beam/{python,examples}
```

Head node:

```
docker run -d --name beam-head --network host --gpus all --ipc host \
    -v /opt/beam:/opt/beam:ro -e PYTHONPATH=/opt/beam/python \
    --entrypoint python3 vllm/vllm-openai:latest \
    -m ray start --head --port 6379 --block
```

Each worker node:

```
docker run -d --name beam-worker --network host --gpus all --ipc host \
    -v /opt/beam:/opt/beam:ro -e PYTHONPATH=/opt/beam/python \
    --entrypoint python3 vllm/vllm-openai:latest \
    -m ray start --address <HEAD_IP>:6379 --block
```

On startup beam self-bootstraps (it detects the container via `/.dockerenv`):
it writes a `ray` launcher onto `PATH` and a `.pth` so `import ray` resolves to
the mounted shim. So inside the container `ray status`, `ray start`, and
`import ray` all work with no install.

Check the cluster, then serve:

```
docker exec beam-head python3 -m ray status
docker exec beam-head vllm serve <model> \
    --tensor-parallel-size 2 --distributed-executor-backend ray
```

`test/dgx/dgx.sh` automates all of this over SSH; see `test/dgx/README.md`.

## GPU count

beam detects GPUs from `/dev/nvidia*`. On platforms whose device nodes are not
`/dev/nvidia*` (DGX Spark / GB10), set `BEAM_NUM_GPUS` explicitly (e.g. `-e
BEAM_NUM_GPUS=1`).

## NCCL over RoCE in containers

The control plane is plain TCP and needs nothing special. The vLLM data plane
(NCCL all-reduce) needs the RDMA fabric, which is a separate concern from beam.
Three things must be right:

### 1. RDMA device access

`--network host --gpus all` is not enough. The NVIDIA container runtime injects
**GPUs**, not RDMA NICs — the verbs devices (`/dev/infiniband/uverbs*`,
`rdma_cm`) are a different kernel subsystem and are not auto-injected. A plain
`-v /dev/infiniband` bind mount exposes the device nodes but docker's
device-cgroup still denies opening them (which is what makes `--privileged`
appear necessary).

The fix is to pass the devices with `--device`, which grants the cgroup
permission. `--device` recurses a directory, so one flag covers all of them:

```
--cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband
```

This is what NVIDIA's own multi-node recipes do; no `--privileged`. Without it,
NCCL logs `Unable to open device rocep*` → `NET/IB : No device found` and falls
back to TCP sockets.

### 2. The right HCA

Name the RoCE ports that are actually cabled between the nodes, and keep the
NCCL/Gloo bootstrap on a reachable interface:

```
-e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
-e NCCL_SOCKET_IFNAME=enP7s7
-e GLOO_SOCKET_IFNAME=enP7s7
-e NCCL_DEBUG=INFO          # shows the transport it selects
```

Find the active RoCE links with `rdma link show` (look for `state ACTIVE`) and
their netdevs with `ip -br link`. Success looks like:

```
NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE [1]roceP2p1s0f1:1/RoCE
NCCL INFO Using network IB
```

### 3. Worker environment

Actor workers inherit their environment from the **daemon** process, not from the
`docker exec` that runs `vllm serve`. So cluster-wide NCCL env must be set on the
daemon container at `docker run` time (the harness puts it in `NCCL_EXTRA`), not
on the exec.

## Uniform Python across the cluster

Every node must run the **same Python minor version**. beam ships actor classes
to workers with cloudpickle, which serializes methods as version-specific
bytecode; a 3.13 driver talking to 3.12 workers cannot be unpickled and the
worker dies ("connection closed" at actor creation). This is the same constraint
real Ray has. When the driver and workers are all in the vLLM container they
already match; if you run a daemon outside the container (e.g. a head on a
laptop), pin its interpreter to the container's (`uv venv --python 3.12`).

## TCP transport (no RoCE)

beam's control plane is always plain TCP, so it needs no fabric. To run the vLLM
data plane over TCP sockets too (e.g. no RoCE, or to isolate a fabric problem),
disable IB so NCCL falls back to sockets on a chosen interface:

```
-e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1
```

Slower than RoCE, but it proves the whole path end to end without RDMA.

## gloo data plane (CPU / Vulkan backends)

On the CPU platform (which the [vllm-vulkan](https://github.com/ericcurtin/vllm-vulkan)
plugin rides) the tensor-parallel all-reduce runs over **gloo** instead of
NCCL/RCCL. beam carries the control plane identically; two extra things matter
for the gloo data plane across nodes:

- **Advertise the right node IP.** vLLM publishes its zmq message queues at
  `ray.util.get_node_ip_address()`. The default heuristic returns the
  default-route interface, which on a multi-homed host (a router, or a box with
  several NICs / a secondary address on one NIC) is frequently not the address
  the other nodes reach you on. Set `BEAM_NODE_IP` (preferred) or `VLLM_HOST_IP`
  per node to the cluster-LAN address, and point gloo at the same interface with
  `GLOO_SOCKET_IFNAME`.
- **Reachability must be bidirectional.** Workers connect to the driver's
  broadcast queue (worker→driver), but the driver also connects to each worker's
  *response* queue and the all-reduce is peer-to-peer (driver→worker). A
  default-deny host firewall that only permits the worker→driver direction lets
  the cluster form and the broadcast queue subscribe, then **silently stalls** at
  the response-queue/all-reduce barrier. Open the cluster subnet between nodes
  (e.g. `ufw allow from <subnet>`); ICMP/ssh passing is not enough.

## AMD GPUs (ROCm)

beam works on AMD the same way it does on NVIDIA: it sets the per-actor device
env so vLLM's ROCm path picks the right GPU. When it assigns an actor its
GPU id, it exports `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and
`ROCR_VISIBLE_DEVICES` together, so the same daemon serves either vendor.

Use the `rocm/vllm` image instead of `vllm/vllm-openai`, and give the container
the ROCm devices: `--device /dev/kfd --device /dev/dri/renderD<N> --group-add
video --group-add render --security-opt seccomp=unconfined`.

**Avoiding the integrated GPU.** A box with an AMD CPU often exposes the CPU's
integrated graphics (e.g. "Raphael") as a second ROCm device. Do not let vLLM
land on it. The clean way is to expose only the **discrete** card's render node
to the container, so ROCm enumerates a single GPU:

```
lspci | grep -iE 'vga|display'        # find the discrete card (e.g. Navi 31), note its PCI id
ls -l /sys/class/drm/renderD*/device  # map renderD<N> -> PCI id
# expose only that render node:  --device /dev/dri/renderD<N>
```

`rocm-smi --showproductname` inside the container should then list only the
discrete GPU. `test/run_rocm.sh` does exactly this (single node, TP=1) and
prints what ROCm sees as a check.

Note: ROCm tensor-parallel across nodes assumes uniform GPUs (same gfx arch);
mixing, say, gfx1100 and gfx1030 is not a supported vLLM/RCCL configuration even
though beam will place the actors fine.

## Cross-node AMD on Azure (ND MI300X v5)

Cloud AMD multi-node needs a provider where the nodes reach each other directly
on all ports (NCCL advertises private addresses and uses dynamic ports, so
NAT'd / proxied container hosts like RunPod standard pods cannot carry it).
Azure ND MI300X v5 VMs sit in a VNet with SR-IOV InfiniBand and satisfy that.

Provision two VMs in one VNet + proximity placement group (for IB):

```
az group create -n beam-amd -l eastus2
az ppg create -n beam-ppg -g beam-amd
az network vnet create -g beam-amd -n beam-vnet --subnet-name s0 --address-prefixes 10.0.0.0/16 --subnet-prefixes 10.0.0.0/24
for i in 0 1; do
  az vm create -g beam-amd -n nd$i \
    --size Standard_ND96isr_MI300X_v5 \
    --image microsoft-dsvm:ubuntu-hpc:2204-rocm:latest \
    --ppg beam-ppg --vnet-name beam-vnet --subnet s0 \
    --admin-username azureuser --ssh-key-values ~/.ssh/id_ed25519.pub \
    --public-ip-sku Standard
done
az vm list-ip-addresses -g beam-amd -o table   # note public + private (10.0.0.x) IPs
```

Then run the harness (`docker` + ROCm device access are present on the HPC
image):

```
VM0='azureuser@<pub0>' VM1='azureuser@<pub1>' \
VM0_PRIV=10.0.0.4 VM1_PRIV=10.0.0.5 \
GPUS_PER_NODE=1 MODEL=meta-llama/Llama-3.1-8B-Instruct \
bash test/run_rocm_azure.sh
```

`run_rocm_azure.sh` deploys beam to both VMs, runs the rocm/vllm container on
each (KFD + /dev/dri + /dev/infiniband + IPC_LOCK), starts beam head on vm0 /
worker on vm1 over the private IP, then `vllm serve` TP across both with RCCL
over InfiniBand. Set `IB_DISABLE=1 SOCKET_IF=eth0` to force RCCL onto the VNet
TCP socket instead, and bump `GPUS_PER_NODE`/`TP_SIZE` to use all 8 GPUs/node.

## CPU inference

vLLM's CPU backend (gloo, `device=cpu`) lets nodes without a GPU participate, but
the stock `vllm/vllm-openai` image is CUDA-only — CPU inference needs a
CPU-built image (vLLM's `Dockerfile.cpu`) or a CPU wheel. beam itself is
indifferent to the device: the control plane is identical, only the actors'
`torch.distributed` backend changes (gloo instead of nccl). With a CPU image,
the same bind-mount deploy works, dropping `--gpus all` and setting
`BEAM_NUM_GPUS` to the desired worker count per node.

## Cluster homogeneity (what vLLM requires of the nodes)

beam will spread actors across any mix of nodes (proven: a driver on a GPU-less
x86 laptop placing workers on two arm64 GPU nodes, control plane fully working).
vLLM is stricter, and these are vLLM/hardware limits, not beam's:

- **The beam head can be CPU-only; the node that runs `vllm serve` needs a GPU.**
  beam supports running the driver on any node (it routes create/call/kill from a
  worker node to the head), so a real "CPU head + GPU workers" cluster works: run
  the head daemon on a CPU box and launch `vllm serve` on a GPU worker. What does
  NOT work is launching `vllm serve` itself on a GPU-less node: the EngineCore
  initializes the platform locally and fails with `No CUDA runtime is found →
  Failed to infer device type`. There is no driver=cpu / workers=cuda split.
- **Same architecture.** vLLM's CPU image is published amd64-only; it will not
  run natively on arm64 nodes (e.g. DGX Spark/GB10). A real multi-node CPU run
  needs all nodes on the same arch with a matching image.
- **Same Python** (see above) and the same vLLM version across nodes.

Consequence on a heterogeneous set (e.g. 1 x86 CPU box + 2 arm64 GPU boxes):
real distributed inference spans only the homogeneous GPU pair; the odd node can
host the beam head's daemon but not a vLLM engine or worker. beam is not the
limiter here.

## Memory on unified-memory GPUs

GB10 shares its 128 GiB between GPU and system. vLLM's default
`--gpu-memory-utilization 0.9` sizes a ~107 GiB KV cache, which pushes the host
over and the worker gets OOM-killed (container exit code **137**, with
`OOMKilled=false` because it is host-level, not cgroup). Cap it and skip
CUDA-graph capture:

```
vllm serve … --gpu-memory-utilization 0.5 --max-model-len 8192 --enforce-eager
```

## Troubleshooting

| symptom | cause | fix |
|---------|-------|-----|
| `ray status` shows 1 node | worker didn't join | check head IP/port reachable, `--address` correct, head started first |
| NCCL `Unable to open device rocep*`, falls back to Socket | RDMA verbs not accessible in container | add `--device /dev/infiniband` + `IPC_LOCK` + `memlock` |
| worker container exits 137, `OOMKilled=false` | host memory pressure (unified memory) | lower `--gpu-memory-utilization`, add `--enforce-eager` |
| `placement group needs more GPUs than the cluster has free` | GPUs reserved by a previous run's pg | restart the daemons, or let the driver disconnect (beam frees pgs on disconnect) |
| `RayWorkerMonitor … connection closed` | a worker actor died; beam correctly detected it | look at that worker's container logs for the real crash (often NCCL or OOM) |
| engine init hangs at `MessageQueue.wait_until_ready` / first all-reduce, cluster otherwise formed | driver can't reach a worker (one-way firewall, or wrong advertised IP) | open the cluster subnet head→workers; set `BEAM_NODE_IP`/`VLLM_HOST_IP` per node to the LAN address |
| workers connect but queues never become ready on a multi-homed host | node advertised its default-route IP, not the cluster LAN | set `BEAM_NODE_IP` per node |
| `Tensor parallel size (N) exceeds available GPUs (1)` warning | benign: vLLM compares TP to per-node GPUs; beam spreads ranks across nodes | ignore (or add GPUs per node) |
| `import ray` finds the real ray | a real ray is installed in the image | use the stock vllm-openai image (no ray), or uninstall ray |

## Validated configuration

Confirmed end to end on 2× DGX Spark (GB10, arm64), `vllm/vllm-openai:v0.23.0`,
`Qwen/Qwen2.5-0.5B-Instruct`, `--tensor-parallel-size 2`:

- beam control plane across both nodes (one worker per node)
- NCCL all-reduce ring over both RoCE links (`rocep1s0f1`, `roceP2p1s0f1`)
- no `--privileged`, no image rebuild
- OpenAI completion returned (`system_fingerprint: vllm-0.23.0-tp2`)

The exact flags are in `test/dgx/config.sh`.
