#!/usr/bin/env bash
# Three-node beam control plane over TCP across real machines: this host as the
# head (native python, no GPU needed) plus both DGX Spark nodes as worker daemons
# in the stock vLLM container. Validates >2 nodes, cross-machine routing, and the
# head running outside a container. CPU-only (fake GPUs), no model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/test/dgx/config.sh"
SSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

THIS_IP=$(ip -4 route get 1.1.1.1 | grep -oP 'src \K\S+')
RUN="$ROOT/.3node-run"
rm -rf "$RUN"; mkdir -p "$RUN"

# Local python for the head daemon, its actor worker, and the driver. It MUST
# match the workers' Python (the vLLM container is 3.12): cloudpickle ships class
# methods as version-specific bytecode, so a mismatched interpreter cannot
# unpickle them. This is the same "uniform Python across the cluster" rule real
# Ray has.
uv venv --python 3.12 "$RUN/venv" >/dev/null
VENVPY="$RUN/venv/bin/python"
uv pip install --python "$VENVPY" cloudpickle >/dev/null

export BEAM_RUNTIME_DIR="$RUN"
export PYTHONPATH="$ROOT/python"
export BEAM_NUM_GPUS=1
export BEAM_WORKER_CMD="$VENVPY -m ray._worker"

worker_run="--network host -e PYTHONPATH=/opt/beam/python -e BEAM_NUM_GPUS=1 \
  -v $REMOTE_DIR:/opt/beam:ro --entrypoint python3"

cleanup() {
  kill "${HEAD_PID:-}" 2>/dev/null || true
  $SSH "$SSH_USER@$HEAD_IP"   "docker rm -f beam-n1 2>/dev/null" >/dev/null 2>&1 || true
  $SSH "$SSH_USER@$WORKER_IP" "docker rm -f beam-n2 2>/dev/null" >/dev/null 2>&1 || true
  rm -rf "$RUN"
}
trap cleanup EXIT

echo "=== deploy beam to both sparks ==="
bash "$ROOT/test/dgx/dgx.sh" deploy >/dev/null

echo "=== head on this host ($THIS_IP) ==="
"$VENVPY" -m ray start --head --port "$HEAD_PORT" &
HEAD_PID=$!
for _ in $(seq 1 50); do [ -S "$RUN/daemon.sock" ] && break; sleep 0.1; done

echo "=== worker daemons on both sparks join $THIS_IP:$HEAD_PORT ==="
$SSH "$SSH_USER@$HEAD_IP"   "docker rm -f beam-n1 2>/dev/null; docker run -d --name beam-n1 $worker_run $IMAGE -m ray start --address $THIS_IP:$HEAD_PORT --block" >/dev/null
$SSH "$SSH_USER@$WORKER_IP" "docker rm -f beam-n2 2>/dev/null; docker run -d --name beam-n2 $worker_run $IMAGE -m ray start --address $THIS_IP:$HEAD_PORT --block" >/dev/null

for _ in $(seq 1 60); do
  "$VENVPY" -m ray status 2>/dev/null | grep -q "3 nodes" && break
  sleep 0.5
done
echo "=== ray status ==="
"$VENVPY" -m ray status

echo "=== driver: 3 actors, one per node (this host + 2 sparks) ==="
BEAM_DEMO_WORLD=3 BEAM_DEMO_EXPECT_NODES=3 "$VENVPY" "$ROOT/examples/driver_demo.py"
echo "3node: PASS"
