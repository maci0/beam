#!/usr/bin/env bash
# ROCm (AMD GPU) smoke test on this host, using the rocm/vllm image and beam as
# the ray backend. Single node (one discrete GPU here), so TP=1: it validates
# that beam drives vLLM on ROCm and that only the DISCRETE GPU is used, never the
# CPU's integrated graphics.
#
# Avoiding the iGPU: we expose ONLY the discrete card's render node + /dev/kfd to
# the container, so ROCm enumerates a single device. Find the discrete render
# node with:  for d in /sys/class/drm/renderD*; do echo "$d ->
#   $(cat $(dirname $d/device)/uevent 2>/dev/null)"; done  and `lspci | grep VGA`
# (the discrete is the dedicated card, e.g. Navi 31 / RX 7900 XTX, not "Granite
# Ridge"/"Raphael" / the APU). The renderD<->card numbering is NOT fixed: on this
# host renderD128 is the discrete RX 7900 XTX (pci 03:00.0) and renderD129 is the
# iGPU, so always verify the mapping before trusting the number.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# rocm/vllm-dev:nightly tracks vLLM main (0.23+). Avoid rocm/vllm:latest (old
# vLLM whose ray executor hard-requires Ray Compiled Graph). beam needs the
# MessageQueue executor, forced below with VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1.
# (The arch-matched stable tag rocm/vllm:rocm7.13.0_gfx110X-..._vllm_0.19.1 is a
# fallback if nightly lacks your GPU's gfx kernels.)
IMAGE="${ROCM_IMAGE:-rocm/vllm-dev:nightly}"
RENDER="${ROCM_RENDER:-/dev/dri/renderD128}"   # discrete GPU render node (verify per host!)
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT=8001
RUN="$ROOT/.rocm-run"; rm -rf "$RUN"; mkdir -p "$RUN"

cleanup() { docker rm -f beam-rocm 2>/dev/null || true; rm -rf "$RUN"; }
trap cleanup EXIT
cleanup

echo "=== exposing ONLY $RENDER (discrete) + /dev/kfd, iGPU not mapped ==="
docker run -d --name beam-rocm --network host --ipc host --shm-size 10g \
  --device /dev/kfd --device "$RENDER" \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -v "$ROOT:/opt/beam:ro" -e PYTHONPATH=/opt/beam/python -e BEAM_NUM_GPUS=1 \
  -e PYTHONUNBUFFERED=1 -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
  --entrypoint python3 "$IMAGE" -m ray start --head --port 6379 --block >/dev/null

for _ in $(seq 1 50); do docker exec beam-rocm test -S /root/.beam/daemon.sock 2>/dev/null && break; sleep 0.2; done

echo "=== which GPU does ROCm see in the container? (must be the discrete only) ==="
docker exec beam-rocm rocm-smi --showproductname 2>/dev/null | grep -E 'GPU|Card Series' | head

echo "=== beam status ==="
docker exec beam-rocm python3 -m ray status

# If the discrete card also drives this host's display, some VRAM is already in
# use, so keep gpu-memory-utilization well under 0.9 (override with MEM_UTIL).
MEM_UTIL="${MEM_UTIL:-0.4}"
echo "=== vllm serve on ROCm (TP=1, ray backend, mem-util=$MEM_UTIL) ==="
docker exec -d beam-rocm bash -lc "vllm serve $MODEL --distributed-executor-backend ray \
  --tensor-parallel-size 1 --port $PORT --enforce-eager --max-model-len 8192 \
  --gpu-memory-utilization $MEM_UTIL > /tmp/vllm.log 2>&1"

echo "=== waiting for health ==="
for i in $(seq 1 120); do
  if docker exec beam-rocm curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "HEALTHY after ~$((i*5))s"; break
  fi
  if docker exec beam-rocm grep -qiE "Failed to infer device|HIP error|no ROCm|hipError" /tmp/vllm.log 2>/dev/null; then
    echo "ROCm device error:"; docker exec beam-rocm tail -8 /tmp/vllm.log; exit 1
  fi
  sleep 5
done
echo "=== completion ==="
docker exec beam-rocm curl -s "http://localhost:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16}"
echo; echo "rocm: done"
