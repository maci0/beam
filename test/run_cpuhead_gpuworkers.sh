#!/usr/bin/env bash
# CPU head + GPU workers: the beam head runs on THIS host (no GPU, no vLLM, pure
# control plane). The two GPU spark nodes are workers, and the vLLM engine runs
# on one of them (spark1). beam places both TP bundles on the GPU nodes (this
# host has 0 GPUs) and routes the driver's RPCs to the head from a worker node.
# NCCL runs spark1<->spark2 over RoCE. Mirrors a real Ray "CPU head, GPU workers"
# cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/test/dgx/config.sh"
SSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
THIS_IP=$(ip -4 route get 1.1.1.1 | grep -oP 'src \K\S+')
RUN="$ROOT/.cpuhead-run"; rm -rf "$RUN"; mkdir -p "$RUN"

# head daemon on this host (Python pinned to the workers' 3.12; it only routes,
# but ray.init in the shim must import cleanly)
uv venv --python 3.12 "$RUN/venv" >/dev/null
VENVPY="$RUN/venv/bin/python"
uv pip install --python "$VENVPY" cloudpickle >/dev/null

cleanup() {
  kill "${HEAD_PID:-}" 2>/dev/null || true
  $SSH "$SSH_USER@$HEAD_IP"   "docker rm -f beam-w1 2>/dev/null" >/dev/null 2>&1 || true
  $SSH "$SSH_USER@$WORKER_IP" "docker rm -f beam-w2 2>/dev/null" >/dev/null 2>&1 || true
  rm -rf "$RUN"
}
trap cleanup EXIT

echo "=== deploy beam to sparks ==="
bash "$ROOT/test/dgx/dgx.sh" deploy >/dev/null

echo "=== beam head on this host ($THIS_IP), CPU only, no vLLM ==="
BEAM_RUNTIME_DIR="$RUN" PYTHONPATH="$ROOT/python" BEAM_NUM_GPUS=0 \
  "$VENVPY" -m ray start --head --port "$HEAD_PORT" &
HEAD_PID=$!
for _ in $(seq 1 50); do [ -S "$RUN/daemon.sock" ] && break; sleep 0.1; done

wr="--network host --gpus all --ipc host --shm-size 10g ${RDMA_ARGS:-} \
  -v $REMOTE_DIR:/opt/beam:ro -e PYTHONPATH=/opt/beam/python -e BEAM_NUM_GPUS=1 \
  -e PYTHONUNBUFFERED=1 $NCCL_EXTRA --entrypoint python3"
echo "=== GPU worker daemons on both sparks join the CPU head ==="
$SSH "$SSH_USER@$HEAD_IP"   "docker rm -f beam-w1 2>/dev/null; docker run -d --name beam-w1 $wr $IMAGE -m ray start --address $THIS_IP:$HEAD_PORT --block" >/dev/null
$SSH "$SSH_USER@$WORKER_IP" "docker rm -f beam-w2 2>/dev/null; docker run -d --name beam-w2 $wr $IMAGE -m ray start --address $THIS_IP:$HEAD_PORT --block" >/dev/null

for _ in $(seq 1 60); do
  BEAM_RUNTIME_DIR="$RUN" PYTHONPATH="$ROOT/python" "$VENVPY" -m ray status 2>/dev/null | grep -q "3 nodes" && break; sleep 0.5
done
echo "=== ray status (head has 0 GPUs) ==="
BEAM_RUNTIME_DIR="$RUN" PYTHONPATH="$ROOT/python" "$VENVPY" -m ray status

echo "=== vLLM engine on spark1 (a GPU worker), TP=2, driver routes via the CPU head ==="
$SSH "$SSH_USER@$HEAD_IP" "docker exec -d beam-w1 bash -lc \"vllm serve $MODEL \
  --distributed-executor-backend ray --tensor-parallel-size $TP_SIZE \
  --port $SERVE_PORT ${SERVE_EXTRA:-} > /tmp/vllm.log 2>&1\""

echo "=== waiting for health ==="
for i in $(seq 1 120); do
  if $SSH "$SSH_USER@$HEAD_IP" "docker exec beam-w1 curl -sf http://localhost:$SERVE_PORT/health" >/dev/null 2>&1; then
    echo "HEALTHY after ~$((i*5))s"; break
  fi
  sleep 5
done
echo "=== completion request ==="
$SSH "$SSH_USER@$HEAD_IP" "docker exec beam-w1 curl -s http://localhost:$SERVE_PORT/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}'"
echo; echo "cpuhead-gpuworkers: done"
