#!/usr/bin/env bash
# N-node CPU control-plane test across real machines. No GPUs, no vLLM: each node
# fakes 1 GPU (BEAM_NUM_GPUS=1) so the driver places one actor per node, then
# broadcasts calls and gathers results. Proves cluster membership + cross-node
# placement + actor RPC on real hardware.
#
# Uniform python is required (cloudpickle bytecode compat), so every node runs
# the same python:3.12-slim container on the host network. Nodes can have
# different host python / arch; the container makes them uniform.
#
# Edit NODES below (first entry = head). Each entry: "name|ssh-opts|host-ip"
# where ssh-opts is "LOCAL" for the local host (head runs docker locally), else
# the ssh flags (no target). SSH_USER is the login on the remote nodes. Override
# the whole list with the BEAM_NODES env var (newline-separated, same format).
# Then: bash test/run_cpu_cluster.sh   (set N to use only the first N nodes.)
# the $o ssh-option string is intentionally word-split; remote commands expand locally.
# shellcheck disable=SC2086,SC2029
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SSH_USER="${SSH_USER:-$USER}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSHO="-i $SSH_KEY -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"
if [ -n "${BEAM_NODES:-}" ]; then
  mapfile -t NODES <<< "$BEAM_NODES"
else
  NODES=(
    "n0|LOCAL|10.0.0.1"   # this host, head (local docker)
    "n1|$SSHO|10.0.0.2"
    "n2|$SSHO|10.0.0.3"
    "n3|$SSHO|10.0.0.4"
  )
fi
N="${N:-${#NODES[@]}}"
IMAGE="${IMAGE:-python:3.12-slim}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/beam}"
PORT="${PORT:-6379}"
HEAD_IP="$(echo "${NODES[0]}" | cut -d'|' -f3)"

opts_of() { echo "${NODES[$1]}" | cut -d'|' -f2; }
ip_of() { echo "${NODES[$1]}" | cut -d'|' -f3; }
# run a command on node i (local if opts == LOCAL)
on() { local o; o="$(opts_of "$1")"; local ip; ip="$(ip_of "$1")"; shift
  if [ "$o" = "LOCAL" ]; then bash -c "$*"; else ssh $o "$SSH_USER@$ip" "$*"; fi; }

cleanup() { for i in $(seq 0 $((N-1))); do on "$i" "docker rm -f beam-cpu 2>/dev/null" >/dev/null 2>&1 || true; done; }
trap cleanup EXIT
cleanup

echo "=== deploy beam to $N nodes ($REMOTE_DIR) ==="
for i in $(seq 0 $((N-1))); do
  o="$(opts_of "$i")"; ip="$(ip_of "$i")"
  if [ "$o" = "LOCAL" ]; then
    mkdir -p "$REMOTE_DIR"; cp -r "$ROOT/python" "$ROOT/examples" "$REMOTE_DIR/"
  else
    ssh $o "$SSH_USER@$ip" "mkdir -p $REMOTE_DIR"
    rsync -a --delete -e "ssh $o" "$ROOT/python/" "$SSH_USER@$ip:$REMOTE_DIR/python/"
    rsync -a --delete -e "ssh $o" "$ROOT/examples/" "$SSH_USER@$ip:$REMOTE_DIR/examples/"
  fi
done

RUN="docker run -d --name beam-cpu --network host -e BEAM_NUM_GPUS=1 \
  -e PYTHONPATH=/opt/beam/python -e PYTHONUNBUFFERED=1 -v $REMOTE_DIR:/opt/beam:ro \
  --entrypoint bash $IMAGE -c"
PREP="pip install -q cloudpickle >/dev/null 2>&1"

echo "=== head on ${HEAD_IP} ==="
on 0 "$RUN \"$PREP && python3 -m ray start --head --port $PORT --block\"" >/dev/null
sleep 6  # pip install + daemon up

for i in $(seq 1 $((N-1))); do
  ip="$(echo "${NODES[$i]}" | cut -d'|' -f3)"
  echo "=== worker n$i ($ip) -> $HEAD_IP:$PORT ==="
  on "$i" "$RUN \"$PREP && python3 -m ray start --address $HEAD_IP:$PORT --block\"" >/dev/null
done

echo "=== wait for $N nodes ==="
for _ in $(seq 1 40); do
  on 0 "docker exec beam-cpu python3 -m ray status 2>/dev/null" | grep -q "$N nodes" && break; sleep 1
done
on 0 "docker exec beam-cpu python3 -m ray status"

echo "=== driver: $N actors, one per node ==="
on 0 "docker exec -e BEAM_DEMO_WORLD=$N -e BEAM_DEMO_EXPECT_NODES=$N beam-cpu \
  python3 /opt/beam/examples/driver_demo.py"
echo "cpu-cluster ($N nodes): PASS"
