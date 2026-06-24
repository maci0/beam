#!/usr/bin/env bash
# Multi-node check, colocated on one host: a GPU-less head plus a worker with 4
# fake GPUs. The driver talks only to the head; actors must be spawned on the
# worker and their results routed back through the head hub.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.mn-run"
rm -rf "$RUN"; mkdir -p "$RUN/head" "$RUN/worker"

uv venv "$RUN/venv" >/dev/null
VENVPY="$RUN/venv/bin/python"
uv pip install --python "$VENVPY" cloudpickle >/dev/null

export PYTHONPATH="$ROOT/python"
export BEAM_WORKER_CMD="$VENVPY -m ray._worker"

cleanup() {
  kill "${HEAD_PID:-}" 2>/dev/null || true
  kill "${WORK_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

# head: no GPUs, listens on 6380
BEAM_RUNTIME_DIR="$RUN/head" BEAM_NUM_GPUS=0 \
  "$VENVPY" -m ray start --head --port 6380 &
HEAD_PID=$!
for _ in $(seq 1 50); do [ -S "$RUN/head/daemon.sock" ] && break; sleep 0.1; done

# worker: 4 GPUs, joins the head
BEAM_RUNTIME_DIR="$RUN/worker" BEAM_NUM_GPUS=4 \
  "$VENVPY" -m ray start --address 127.0.0.1:6380 &
WORK_PID=$!

for _ in $(seq 1 50); do
  if BEAM_RUNTIME_DIR="$RUN/head" "$VENVPY" -m ray status 2>/dev/null | grep -q "2 nodes"; then
    break
  fi
  sleep 0.2
done

echo "--- ray status ---"
BEAM_RUNTIME_DIR="$RUN/head" "$VENVPY" -m ray status

# driver connects to the HEAD socket; all 4 GPU actors land on the one worker
# (the only node with GPUs), routed cross-process through the hub
BEAM_SOCK="$RUN/head/daemon.sock" BEAM_DEMO_EXPECT_NODES=1 \
  "$VENVPY" "$ROOT/examples/driver_demo.py"

echo "multinode: PASS"
