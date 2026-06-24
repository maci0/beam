#!/usr/bin/env bash
# 2-node ROCm tensor-parallel inference on a real multi-node cluster (e.g. a
# RunPod Instant Cluster of 2x MI300X). Unlike the standalone-pod path, the nodes
# here share a real private network, so RCCL runs over it directly.
#
# RunPod Instant Cluster nodes ARE the container (rocm/vllm-dev:nightly), so we
# SSH straight in -- no docker-in-docker. Fill the endpoints below (or pass as
# env), then run. beam head on node0, worker on node1, vLLM TP=2 across both.
#
#   NODE0='root@1.2.3.4 -p 22001' NODE1='root@1.2.3.5 -p 22002' \
#   HEAD_IP=10.65.0.2 NCCL_IF=eth1 bash test/run_rocm_cluster.sh
# ssh option/target strings (SSHOPTS, NODE="user@host -p port") are intentionally
# word-split, and remote commands intentionally expand on the client side.
# shellcheck disable=SC2086,SC2029
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- fill these (SSH targets + the head's PRIVATE/cluster IP that the worker and
#     RCCL use; NCCL_IF is the interface that carries HEAD_IP, auto-detected if
#     left empty) -------------------------------------------------------------
NODE0="${NODE0:-}"          # head    e.g. 'root@213.173.x.x -p 19201'
NODE1="${NODE1:-}"          # worker  e.g. 'root@213.173.y.y -p 19202'
HEAD_IP="${HEAD_IP:-}"      # node0 cluster/private IP, e.g. 10.65.0.2
NCCL_IF="${NCCL_IF:-}"      # cluster iface name, e.g. eth1 (auto if empty)
# ---------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8001}"
SSHOPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

[ -z "$NODE0" ] || [ -z "$NODE1" ] || [ -z "$HEAD_IP" ] && {
  echo "set NODE0, NODE1, HEAD_IP (and optionally NCCL_IF). See header."; exit 2; }

n0() { ssh $SSHOPTS $NODE0 "$@"; }
n1() { ssh $SSHOPTS $NODE1 "$@"; }

cleanup() {
  n0 "pkill -f 'ray._daemon|ray start|vllm serve' 2>/dev/null" || true
  n1 "pkill -f 'ray._daemon|ray start' 2>/dev/null" || true
}
trap cleanup EXIT

echo "=== sanity: GPUs + vLLM on both nodes ==="
n0 "rocm-smi --showproductname 2>/dev/null | grep -i 'GFX Version' | head -1; python3 -c 'import vllm;print(\"vllm\",vllm.__version__)'"
n1 "rocm-smi --showproductname 2>/dev/null | grep -i 'GFX Version' | head -1; python3 -c 'import vllm;print(\"vllm\",vllm.__version__)'"

# auto-detect the cluster interface carrying HEAD_IP on node0 if not given
if [ -z "$NCCL_IF" ]; then
  NCCL_IF=$(n0 "ip -o -4 addr show | awk '/$HEAD_IP/{print \$2; exit}'")
  echo "auto-detected NCCL_IF=$NCCL_IF (carries $HEAD_IP)"
fi

echo "=== deploy beam to both nodes (/opt/beam) ==="
for N in "$NODE0" "$NODE1"; do
  ssh $SSHOPTS $N "mkdir -p /opt/beam"
  rsync -a --delete -e "ssh $SSHOPTS" "$ROOT/python/" "$N:/opt/beam/python/" 2>/dev/null \
    || tar -C "$ROOT" -cf - python | ssh $SSHOPTS $N "tar -C /opt/beam -xf -"
done

env_common="PYTHONPATH=/opt/beam/python BEAM_NUM_GPUS=1 VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
  NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=$NCCL_IF GLOO_SOCKET_IFNAME=$NCCL_IF NCCL_DEBUG=WARN"

echo "=== beam head on node0 ($HEAD_IP) ==="
n0 "pkill -f 'ray start' 2>/dev/null; sleep 1; \
    $env_common BEAM_RUNTIME_DIR=/root/.beam nohup python3 -m ray start --head --port 6379 \
    > /tmp/beamd.log 2>&1 & sleep 3; cat /tmp/beamd.log"

echo "=== beam worker on node1 -> $HEAD_IP:6379 ==="
n1 "pkill -f 'ray start' 2>/dev/null; sleep 1; \
    $env_common nohup python3 -m ray start --address $HEAD_IP:6379 \
    > /tmp/beamd.log 2>&1 & sleep 3; cat /tmp/beamd.log"

echo "=== ray status ==="
for _ in $(seq 1 30); do n0 "python3 -m ray status 2>/dev/null" | grep -q '2 nodes' && break; sleep 1; done
n0 "python3 -m ray status"

echo "=== vllm serve TP=2 across both MI300X (RCCL over cluster net) ==="
n0 "$env_common VLLM_HOST_IP=$HEAD_IP nohup vllm serve $MODEL \
    --distributed-executor-backend ray --tensor-parallel-size 2 --port $PORT \
    --enforce-eager --max-model-len 4096 --gpu-memory-utilization 0.85 \
    > /tmp/vllm.log 2>&1 & echo launched"

echo "=== waiting for health ==="
for i in $(seq 1 120); do
  n0 "curl -sf http://localhost:$PORT/health >/dev/null 2>&1" && { echo "HEALTHY ~$((i*5))s"; break; }
  n0 "grep -qiE 'no kernel image|HIP error|Traceback|RuntimeError|NCCL error' /tmp/vllm.log 2>/dev/null" && { echo "ERROR ~$((i*5))s"; n0 "tail -15 /tmp/vllm.log"; exit 1; }
  sleep 5
done

echo "=== completion ==="
n0 "curl -s http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}'"
echo; echo "rocm-cluster: done"
