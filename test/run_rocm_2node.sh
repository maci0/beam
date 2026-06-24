#!/usr/bin/env bash
# 2-node ROCm tensor-parallel inference: head on this host (.77, RX 7900 XTX /
# gfx1100), worker on .99 (RX 6800/6900 XT / gfx1030), beam as the ray backend.
# Consumer cards have no RoCE, so RCCL runs over TCP sockets (NCCL_IB_DISABLE=1).
# Heterogeneous archs (gfx1100 + gfx1030) are not an officially supported vLLM TP
# config; beam will place the actors fine, RCCL/vLLM may or may not cope.
set -euo pipefail

HEAD_IP=192.168.0.77
WORK_IP=192.168.0.99
HEAD_IF=enp125s0f4u1
WORK_IF=enp38s0
HEAD_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # repo on this host
WORK_DIR=/home/maci/beam                        # beam deployed on .99
IMAGE=rocm/vllm-dev:nightly
RENDER=/dev/dri/renderD128                       # discrete GPU on both nodes
MODEL=Qwen/Qwen2.5-0.5B-Instruct
PORT=8001
SSH="ssh -o BatchMode=yes"

cleanup() {
  docker rm -f beam-rh 2>/dev/null || true
  $SSH maci@$WORK_IP 'bash -lc "docker rm -f beam-rw 2>/dev/null"' >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

common="--network host --ipc host --shm-size 10g --device /dev/kfd --device $RENDER \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -e PYTHONPATH=/opt/beam/python -e BEAM_NUM_GPUS=1 -e PYTHONUNBUFFERED=1 \
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN"

echo "=== head on $HEAD_IP (gfx1100) ==="
docker run -d --name beam-rh $common -e NCCL_SOCKET_IFNAME=$HEAD_IF -e GLOO_SOCKET_IFNAME=$HEAD_IF \
  -v "$HEAD_DIR:/opt/beam:ro" --entrypoint python3 "$IMAGE" \
  -m ray start --head --port 6379 --block >/dev/null
for _ in $(seq 1 50); do docker exec beam-rh test -S /root/.beam/daemon.sock 2>/dev/null && break; sleep 0.3; done

echo "=== worker on $WORK_IP (gfx1030), joins head over TCP ==="
$SSH maci@$WORK_IP "bash -lc 'docker rm -f beam-rw 2>/dev/null; docker run -d --name beam-rw $common \
  -e NCCL_SOCKET_IFNAME=$WORK_IF -e GLOO_SOCKET_IFNAME=$WORK_IF \
  -v $WORK_DIR:/opt/beam:ro --entrypoint python3 $IMAGE \
  -m ray start --address $HEAD_IP:6379 --block'" >/dev/null

for _ in $(seq 1 60); do
  docker exec beam-rh python3 -m ray status 2>/dev/null | grep -q "2 nodes" && break; sleep 0.5
done
echo "=== ray status (2 AMD nodes) ==="
docker exec beam-rh python3 -m ray status

echo "=== vllm serve TP=2 across both AMD GPUs (RCCL over TCP) ==="
docker exec -d beam-rh bash -lc "vllm serve $MODEL --distributed-executor-backend ray \
  --tensor-parallel-size 2 --port $PORT --enforce-eager --max-model-len 4096 \
  --gpu-memory-utilization 0.4 > /tmp/v.log 2>&1"

for i in $(seq 1 96); do
  docker exec beam-rh curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { echo "HEALTHY ~$((i*5))s"; break; }
  docker exec beam-rh grep -qaiE "no kernel image|not supported|HIP error|Traceback|RuntimeError|gfx1030.*not|incompatible" /tmp/v.log 2>/dev/null && { echo "ERROR ~$((i*5))s"; docker exec beam-rh grep -aiE "Error|gfx|kernel|not supported|RCCL|NCCL" /tmp/v.log | tail -10; exit 1; }
  sleep 5
done
echo "=== completion ==="
docker exec beam-rh curl -s http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}"
echo; echo "rocm-2node: done"
