#!/usr/bin/env bash
# Validates that the driver can run on a WORKER node, not just the head (the
# "CPU head, GPU workers, engine on a worker" topology). All local, fake GPUs.
# Head has 0 GPUs; two workers have 1 each; the driver connects to worker1's
# socket. Exercises create_pg / create_actor / call / get all forwarded from a
# worker to the head and routed back.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=test/lib.sh
. "$ROOT/test/lib.sh"
RUN="$ROOT/.dow-run"; rm -rf "$RUN"; mkdir -p "$RUN"
# one runtime dir per daemon: the socket path is AF_UNIX-capped at ~100 bytes
RT_HEAD="$(beam_runtime_dir dow-h)"
RT_W1="$(beam_runtime_dir dow-w1)"
RT_W2="$(beam_runtime_dir dow-w2)"

uv venv "$RUN/venv" >/dev/null
VENVPY="$RUN/venv/bin/python"
uv pip install --python "$VENVPY" cloudpickle >/dev/null

export PYTHONPATH="$ROOT/python"
export BEAM_WORKER_CMD="$VENVPY -m ray._worker"

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
  rm -rf "$RUN" "$RT_HEAD" "$RT_W1" "$RT_W2"
}
trap cleanup EXIT

BEAM_RUNTIME_DIR="$RT_HEAD" BEAM_NUM_GPUS=0 "$VENVPY" -m ray start --head --port 6385 & pids+=($!)
for _ in $(seq 1 50); do [ -S "$RT_HEAD/daemon.sock" ] && break; sleep 0.1; done
BEAM_RUNTIME_DIR="$RT_W1" BEAM_NUM_GPUS=1 "$VENVPY" -m ray start --address 127.0.0.1:6385 & pids+=($!)
BEAM_RUNTIME_DIR="$RT_W2" BEAM_NUM_GPUS=1 "$VENVPY" -m ray start --address 127.0.0.1:6385 & pids+=($!)

for _ in $(seq 1 50); do
  BEAM_RUNTIME_DIR="$RT_HEAD" "$VENVPY" -m ray status 2>/dev/null | grep -q "3 nodes" && break; sleep 0.2
done
echo "--- ray status ---"
BEAM_RUNTIME_DIR="$RT_HEAD" "$VENVPY" -m ray status

# driver connects to WORKER1's socket, not the head
echo "--- driver on worker1 (2 actors across the 2 GPU workers) ---"
BEAM_SOCK="$RT_W1/daemon.sock" BEAM_DEMO_WORLD=2 "$VENVPY" "$ROOT/examples/driver_demo.py"
echo "driver-on-worker: PASS"
