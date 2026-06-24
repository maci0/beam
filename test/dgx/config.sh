# shellcheck shell=bash
# Two-node GPU cluster config (example: 2x DGX Spark / GB10). Every value is
# env-overridable, so you can edit here or export before running ./dgx.sh.
HEAD_IP="${HEAD_IP:-10.0.0.1}"          # node 1
WORKER_IP="${WORKER_IP:-10.0.0.2}"      # node 2
SSH_USER="${SSH_USER:-$USER}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# Stock vLLM image, used unmodified (no rebuild). beam is bind-mounted in.
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"

# GPUs per node. (beam is pure Python, so no arch/cross-compile concerns.)
NUM_GPUS="${NUM_GPUS:-1}"               # set explicitly if device nodes aren't /dev/nvidia* (e.g. GB10)

REMOTE_DIR="${REMOTE_DIR:-$HOME/beam}"  # host path on each node (no sudo); bind-mounted to /opt/beam
HEAD_PORT="${HEAD_PORT:-6379}"          # beam head TCP (ray's default port)

# Control-plane test: one GPU bundle per node.
DEMO_WORLD="${DEMO_WORLD:-2}"

# vLLM serve test.
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVE_PORT="${SERVE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"                 # 2 nodes x 1 GPU
# Unified-memory parts (e.g. GB10 shares 128 GiB between GPU and system) OOM-kill
# the worker at the default 0.9 util, so cap KV cache and skip CUDA-graph capture.
SERVE_EXTRA="${SERVE_EXTRA:---gpu-memory-utilization 0.5 --max-model-len 8192 --enforce-eager}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"   # remote path, mounted into the container

# NCCL over RoCE inside containers needs the RDMA verbs devices passed through
# with --device, which grants the docker device-cgroup permission (a plain -v
# bind mount exposes the nodes but the cgroup still blocks them, which is why
# --privileged appeared necessary). --device recurses a directory, so the whole
# /dev/infiniband goes in at once, same as NVIDIA's multi-node recipes. Plus
# locked memory for RDMA pinning. Empty it out for a non-RDMA cluster.
RDMA_ARGS="${RDMA_ARGS:---cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband}"

# Point NCCL at your fabric: NCCL_IB_HCA = the RoCE/IB HCA(s) carrying the data,
# NCCL_SOCKET_IFNAME = the interface for bootstrap/gloo. Check names with
# `ibv_devices` / `ip -o link`. NCCL_DEBUG=INFO prints the transport it selects.
NCCL_EXTRA="${NCCL_EXTRA:--e NCCL_DEBUG=INFO}"
# Examples:
#   RoCE:  NCCL_EXTRA="-e NCCL_DEBUG=INFO -e NCCL_IB_HCA=<hca0>,<hca1> -e NCCL_SOCKET_IFNAME=<mgmt-if> -e GLOO_SOCKET_IFNAME=<mgmt-if>"
#   TCP:   NCCL_EXTRA="-e NCCL_DEBUG=INFO -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=<data-if>"
