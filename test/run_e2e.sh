#!/usr/bin/env bash
# End-to-end check: start a head daemon, run the vLLM-style driver demo against
# it, assert it prints OK. CPU-only (BEAM_NUM_GPUS fakes the devices).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.e2e-run"        # disk-backed scratch (not tmpfs)
rm -rf "$RUN"; mkdir -p "$RUN"

uv venv "$RUN/venv" >/dev/null
VENVPY="$RUN/venv/bin/python"
uv pip install --python "$VENVPY" cloudpickle >/dev/null

export BEAM_RUNTIME_DIR="$RUN"
export PYTHONPATH="$ROOT/python"
export BEAM_NUM_GPUS=4
export BEAM_WORKER_CMD="$VENVPY -m ray._worker"

cleanup() { kill "${HEAD_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

"$VENVPY" -m ray start --head &
HEAD_PID=$!
for _ in $(seq 1 50); do [ -S "$RUN/daemon.sock" ] && break; sleep 0.1; done

"$VENVPY" "$ROOT/examples/driver_demo.py"

echo "--- ray status ---"
"$VENVPY" -m ray status

echo "e2e: PASS"
