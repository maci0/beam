#!/usr/bin/env bash
# 2-node ROCm tensor-parallel inference on Azure ND MI300X v5 VMs, beam as the
# ray backend. Unlike RunPod containers, Azure VMs sit in a VNet and reach each
# other directly on every port (plus SR-IOV InfiniBand), so RCCL runs natively.
#
# Model: SSH into each VM HOST, run the rocm/vllm container there (docker), beam
# head on vm0 + worker on vm1, vLLM TP across both. This is the dgx.sh pattern
# with ROCm devices + the VNet/IB network.
#
# Prereqs on each VM: docker, the AMD GPU + KFD devices, and (for IB) the
# /dev/infiniband devices. Two ND VMs in the same VNet/subnet.
#
#   VM0='azureuser@20.1.2.3' VM1='azureuser@20.1.2.4' \
#   VM0_PRIV=10.0.0.4 VM1_PRIV=10.0.0.5 GPUS_PER_NODE=1 \
#   bash test/run_rocm_azure.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- fill these ------------------------------------------------------------
VM0="${VM0:-}"                 # head VM ssh target  e.g. azureuser@<public-ip>
VM1="${VM1:-}"                 # worker VM ssh target
VM0_PRIV="${VM0_PRIV:-}"       # head VNet private IP (worker dials this; RCCL bootstrap)
VM1_PRIV="${VM1_PRIV:-}"       # worker VNet private IP
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
# ---------------------------------------------------------------------------

IMAGE="${IMAGE:-rocm/vllm-dev:nightly}"   # vLLM 0.23 main, gfx942 kernels
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"        # MI300X VMs have 8; use 1 for a quick cross-node TP=2
TP_SIZE="${TP_SIZE:-$((GPUS_PER_NODE * 2))}"
PORT="${PORT:-8001}"
REMOTE_DIR="${REMOTE_DIR:-/opt/beam}"
# IB interface that carries the VNet/IB traffic; empty = let NCCL autodetect IB.
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
# Socket fallback iface (used only if you set IB_DISABLE=1), e.g. eth0.
SOCKET_IF="${SOCKET_IF:-eth0}"
IB_DISABLE="${IB_DISABLE:-0}"

SSH="ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
[ -z "$VM0" ] || [ -z "$VM1" ] || [ -z "$VM0_PRIV" ] || [ -z "$VM1_PRIV" ] && {
  echo "set VM0, VM1, VM0_PRIV, VM1_PRIV (see header)"; exit 2; }

n0() { $SSH "$VM0" "$@"; }
n1() { $SSH "$VM1" "$@"; }

cleanup() {
  n0 "docker rm -f beam-head 2>/dev/null" >/dev/null 2>&1 || true
  n1 "docker rm -f beam-worker 2>/dev/null" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

echo "=== deploy beam to both VMs ($REMOTE_DIR) ==="
for VM in "$VM0" "$VM1"; do
  $SSH "$VM" "sudo mkdir -p $REMOTE_DIR && sudo chown \$(whoami) $REMOTE_DIR" 2>/dev/null || $SSH "$VM" "mkdir -p $REMOTE_DIR"
  rsync -a --delete -e "$SSH" "$ROOT/python/" "$VM:$REMOTE_DIR/python/"
  rsync -a --delete -e "$SSH" "$ROOT/examples/" "$VM:$REMOTE_DIR/examples/"
done

# container args: ROCm GPUs + KFD + InfiniBand + locked memory
run_args="--network host --ipc host --shm-size 16g \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v $REMOTE_DIR:/opt/beam:ro -e PYTHONPATH=/opt/beam/python \
  -e BEAM_NUM_GPUS=$GPUS_PER_NODE -e PYTHONUNBUFFERED=1 \
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1"
[ "$IB_DISABLE" = "1" ] && run_args="$run_args -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=$SOCKET_IF -e GLOO_SOCKET_IFNAME=$SOCKET_IF"
[ -n "$NCCL_IB_HCA" ] && run_args="$run_args -e NCCL_IB_HCA=$NCCL_IB_HCA"

echo "=== beam head on vm0 ($VM0_PRIV) ==="
n0 "docker rm -f beam-head 2>/dev/null; docker run -d --name beam-head $run_args \
  --entrypoint python3 $IMAGE -m ray start --head --port 6379 --block"
for _ in $(seq 1 50); do n0 "docker exec beam-head test -S /root/.beam/daemon.sock 2>/dev/null" && break; sleep 0.5; done

echo "=== beam worker on vm1 -> $VM0_PRIV:6379 ==="
n1 "docker rm -f beam-worker 2>/dev/null; docker run -d --name beam-worker $run_args \
  --entrypoint python3 $IMAGE -m ray start --address $VM0_PRIV:6379 --block"

for _ in $(seq 1 60); do n0 "docker exec beam-head python3 -m ray status 2>/dev/null" | grep -q '2 nodes' && break; sleep 1; done
echo "=== ray status ==="
n0 "docker exec beam-head python3 -m ray status"

echo "=== vllm serve TP=$TP_SIZE across both ND nodes (RCCL over IB) ==="
n0 "docker exec -d -e VLLM_HOST_IP=$VM0_PRIV beam-head bash -lc \"vllm serve $MODEL \
  --distributed-executor-backend ray --tensor-parallel-size $TP_SIZE --port $PORT \
  > /tmp/vllm.log 2>&1\""

echo "=== waiting for health ==="
for i in $(seq 1 120); do
  n0 "docker exec beam-head curl -sf http://localhost:$PORT/health >/dev/null 2>&1" && { echo "HEALTHY ~$((i*5))s"; break; }
  n0 "docker exec beam-head grep -qiE 'no kernel image|HIP error|Traceback|RuntimeError|NCCL error' /tmp/vllm.log 2>/dev/null" && { echo "ERROR ~$((i*5))s"; n0 "docker exec beam-head tail -20 /tmp/vllm.log"; exit 1; }
  sleep 5
done
echo "=== completion ==="
n0 "docker exec beam-head curl -s http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}'"
echo; echo "rocm-azure: done"
