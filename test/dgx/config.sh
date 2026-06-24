# DGX Spark cluster config. Edit to match your nodes, then use ./dgx.sh.
HEAD_IP=192.168.0.211
WORKER_IP=192.168.0.212
SSH_USER=maci
SSH_KEY=/home/maci/.config/NVIDIA/Sync/config/nvsync.key

# Stock vLLM image, used unmodified (no rebuild). beam is bind-mounted in.
IMAGE=vllm/vllm-openai:latest

# One GB10 GPU per node. (beam is pure Python, so no arch/cross-compile concerns.)
NUM_GPUS=1                      # GB10 device nodes aren't /dev/nvidia*, so set it explicitly

REMOTE_DIR=/home/maci/beam      # host path on each node (no sudo); bind-mounted to /opt/beam
HEAD_PORT=6379                  # beam head TCP (ray's default port)

# Control-plane test: one GPU bundle per node.
DEMO_WORLD=2

# vLLM serve test.
MODEL=Qwen/Qwen2.5-0.5B-Instruct
SERVE_PORT=8000
TP_SIZE=2                       # 2 nodes x 1 GPU
# GB10 shares 128 GiB between GPU and system, so cap KV cache well under the
# default 0.9 and skip CUDA-graph capture, or the worker gets OOM-killed (137).
SERVE_EXTRA="--gpu-memory-utilization 0.5 --max-model-len 8192 --enforce-eager"
HF_CACHE=/home/maci/.cache/huggingface   # remote path, mounted into the container

# NCCL over RoCE inside containers needs the RDMA verbs devices passed through
# with --device, which grants the docker device-cgroup permission (a plain -v
# bind mount exposes the nodes but the cgroup still blocks them, which is why
# --privileged appeared necessary). --device recurses a directory, so the whole
# /dev/infiniband goes in at once, same as NVIDIA's multi-node recipes. Plus
# locked memory for RDMA pinning.
RDMA_ARGS="--cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband"

# The two nodes are linked by two ACTIVE RoCE ports: rocep1s0f1 (10.0.1.x) and
# roceP2p1s0f1 (10.0.2.x). Use both for NCCL data; bootstrap/gloo over the mgmt
# net (enP7s7). NCCL_DEBUG=INFO shows the transport it selects.
NCCL_EXTRA="-e NCCL_DEBUG=INFO -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7"
# Fallback to plain TCP sockets over a fabric link if RoCE refuses to come up:
#   NCCL_EXTRA="-e NCCL_DEBUG=INFO -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1"
