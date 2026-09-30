#!/usr/bin/env bash
# Edge-case suite: error propagation, put/get, wait, parallelism, large payloads,
# pg exhaustion, GPU-leak-fix regression. CPU-only (fake GPUs), no torch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=test/lib.sh
. "$ROOT/test/lib.sh"
RUN="$ROOT/.edge-run"
RT="$(beam_runtime_dir edge)"   # short: AF_UNIX caps the socket path
rm -rf "$RUN"; mkdir -p "$RUN"

uv venv "$RUN/venv" >/dev/null
VENVPY="$RUN/venv/bin/python"
uv pip install --python "$VENVPY" cloudpickle >/dev/null

export BEAM_RUNTIME_DIR="$RT"
export PYTHONPATH="$ROOT/python"
export BEAM_NUM_GPUS=4
export BEAM_WORKER_CMD="$VENVPY -m ray._worker"

cleanup() { kill "${HEAD_PID:-}" 2>/dev/null || true; rm -rf "$RT"; }
trap cleanup EXIT

"$VENVPY" -m ray start --head >/dev/null &
HEAD_PID=$!
for _ in $(seq 1 50); do [ -S "$RT/daemon.sock" ] && break; sleep 0.1; done

"$VENVPY" "$ROOT/examples/edge_cases.py"
echo "edge: PASS"
