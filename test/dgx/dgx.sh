#!/usr/bin/env bash
# Test harness for running beam across two DGX Spark nodes, injecting beam into
# the stock vllm-openai image via a single bind mount (no image rebuild, no
# binary: beam is pure Python and runs on the image's own python).
#
#   ./dgx.sh deploy       copy the beam package + examples to both nodes
#   ./dgx.sh up           start head + worker daemon containers (stock image)
#   ./dgx.sh status       beam status from the head
#   ./dgx.sh imports      import-only shim smoke test inside the head container
#   ./dgx.sh cp-test      cross-node control-plane test (actors on both nodes)
#   ./dgx.sh vllm-test    vllm serve TP across nodes + a completion request
#   ./dgx.sh down         stop and remove both containers
#   ./dgx.sh all          deploy -> up -> status -> imports -> cp-test
set -euo pipefail
cd "$(dirname "$0")"
SELF="$PWD/$(basename "$0")"
# shellcheck source=test/dgx/config.sh
source ./config.sh
ROOT="$(cd ../.. && pwd)"

H="$SSH_USER@$HEAD_IP"
W="$SSH_USER@$WORKER_IP"
SSH="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
RSH="$SSH"

# Container run args shared by head and worker. One read-only bind mount of the
# beam dir; PYTHONPATH points the image's python at the shim; beam writes the
# `ray` launcher itself on startup (bootstrap, container-gated).
run_args() {
  echo "--network host --gpus all --ipc host --shm-size 10g ${RDMA_ARGS:-} \
    -v $REMOTE_DIR:/opt/beam:ro \
    -v $HF_CACHE:/root/.cache/huggingface \
    -e PYTHONPATH=/opt/beam/python -e BEAM_NUM_GPUS=$NUM_GPUS $NCCL_EXTRA \
    --entrypoint python3"
}

case "${1:-}" in
deploy)
  for HOST in "$H" "$W"; do
    echo "=== deploy -> $HOST ==="
    $SSH "$HOST" "mkdir -p $REMOTE_DIR"
    rsync -a --delete -e "$RSH" "$ROOT/python/" "$HOST:$REMOTE_DIR/python/"
    rsync -a --delete -e "$RSH" "$ROOT/examples/" "$HOST:$REMOTE_DIR/examples/"
  done
  ;;

up)
  echo "=== head on $HEAD_IP ==="
  $SSH "$H" "docker rm -f beam-head 2>/dev/null; docker run -d --name beam-head \
    $(run_args) $IMAGE -m ray start --head --port $HEAD_PORT --block"
  sleep 2
  echo "=== worker on $WORKER_IP ==="
  $SSH "$W" "docker rm -f beam-worker 2>/dev/null; docker run -d --name beam-worker \
    $(run_args) $IMAGE -m ray start --address $HEAD_IP:$HEAD_PORT --block"
  sleep 2
  echo "containers up; run ./dgx.sh status"
  ;;

status)
  $SSH "$H" "docker exec beam-head python3 -m ray status"
  ;;

imports)
  $SSH "$H" "docker exec beam-head python3 /opt/beam/examples/import_check.py"
  ;;

cp-test)
  echo "control-plane test: $DEMO_WORLD actors across the cluster"
  $SSH "$H" "docker exec -e BEAM_DEMO_WORLD=$DEMO_WORLD beam-head \
    python3 /opt/beam/examples/driver_demo.py"
  ;;

vllm-test)
  echo "starting vllm serve ($MODEL, TP=$TP_SIZE) on the head..."
  $SSH "$H" "docker exec -d beam-head vllm serve $MODEL \
    --distributed-executor-backend ray --tensor-parallel-size $TP_SIZE \
    --port $SERVE_PORT ${SERVE_EXTRA:-}"
  echo "waiting for the model to load (poll /health)..."
  for _ in $(seq 1 120); do
    if $SSH "$H" "curl -sf http://localhost:$SERVE_PORT/health" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  echo "=== completion request ==="
  $SSH "$H" "curl -s http://localhost:$SERVE_PORT/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"$MODEL\",\"prompt\":\"Hello from beam, the capital of France is\",\"max_tokens\":16}'"
  echo
  ;;

down)
  $SSH "$H" "docker rm -f beam-head 2>/dev/null || true"
  $SSH "$W" "docker rm -f beam-worker 2>/dev/null || true"
  ;;

all)
  bash "$SELF" deploy && bash "$SELF" up && sleep 3 \
    && bash "$SELF" status && bash "$SELF" imports && bash "$SELF" cp-test
  echo
  echo "control plane verified across both nodes."
  echo "run './dgx.sh vllm-test' for the full TP-over-RoCE inference test."
  ;;

*)
  grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
