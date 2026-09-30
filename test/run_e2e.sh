#!/usr/bin/env bash
# End-to-end check: start a head daemon, run the vLLM-style driver demo against
# it, assert it prints OK. CPU-only (BEAM_NUM_GPUS fakes the devices).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=test/lib.sh
. "$ROOT/test/lib.sh"
RUN="$ROOT/.e2e-run"        # venv scratch, disk-backed (not tmpfs)
RT="$(beam_runtime_dir e2e)"  # daemon state: short path, AF_UNIX caps it
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

"$VENVPY" -m ray start --head &
HEAD_PID=$!
for _ in $(seq 1 50); do [ -S "$RT/daemon.sock" ] && break; sleep 0.1; done

"$VENVPY" "$ROOT/examples/driver_demo.py"

echo "--- ray status ---"
"$VENVPY" -m ray status

echo "e2e: PASS"
