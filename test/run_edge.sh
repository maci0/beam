#!/usr/bin/env bash
# Edge-case suite: error propagation, put/get, wait, parallelism, large payloads,
# pg exhaustion, GPU-leak-fix regression. CPU-only (fake GPUs), no torch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.edge-run"
rm -rf "$RUN"; mkdir -p "$RUN"

uv venv "$RUN/venv" >/dev/null
VENVPY="$RUN/venv/bin/python"
uv pip install --python "$VENVPY" cloudpickle >/dev/null

export BEAM_RUNTIME_DIR="$RUN"
export PYTHONPATH="$ROOT/python"
export BEAM_NUM_GPUS=4
export BEAM_WORKER_CMD="$VENVPY -m ray._worker"

cleanup() { [ -n "${HEAD_PID:-}" ] && kill "$HEAD_PID" 2>/dev/null || true; rm -rf "$RUN"; }
trap cleanup EXIT

"$VENVPY" -m ray start --head >/dev/null &
HEAD_PID=$!
for _ in $(seq 1 50); do [ -S "$RUN/daemon.sock" ] && break; sleep 0.1; done

"$VENVPY" "$ROOT/examples/edge_cases.py"
echo "edge: PASS"
