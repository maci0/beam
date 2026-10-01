# shellcheck shell=bash
# Shared helpers for the harnesses in this directory. Source it, don't run it:
#
#   . "$(dirname "$0")/lib.sh"
#
# beam's control socket is an AF_UNIX socket, so its path is capped at 107 bytes
# on Linux (103 on macOS); `ray start` now refuses a longer one up front. A
# checkout nested deeply enough blows that budget before the harness gets
# anywhere, so harnesses keep the daemon runtime dir short: it holds only
# daemon.sock and daemon.json (a few hundred bytes), so tmpfs is fine. The venv
# stays in the repo, where there is disk space.
beam_free_port() {  # prints a currently-free TCP port for a head to listen on
  python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()'
}

# cloudpickle is beam's one runtime dependency (see python/pyproject.toml).
# Harnesses install it into a scratch venv, so they pin the same range the
# manifest declares: a bare name floats to whatever cloudpickle was newest on
# the day, and actor payloads are serialized per cloudpickle version on both
# ends of the socket.
CLOUDPICKLE_SPEC='cloudpickle>=3.1.2,<4'

beam_runtime_dir() {  # $1 = tag; prints a fresh short dir, safe to delete
  local dir="${TMPDIR:-/tmp}/beam-$1-$$"
  rm -rf "$dir"; mkdir -p "$dir"
  printf '%s\n' "$dir"
}

# Install the runtime dependency into the interpreter named by $1. If $2 names a
# readable requirements file, install from it with --require-hashes (the
# uv.lock export, where every artifact carries its sha256); otherwise fall back
# to the manifest's range, which is what installing the wheel resolves anyway.
beam_install_dep() {  # $1 = python; $2 = optional hash-pinned requirements file
  if [ -n "${2:-}" ] && [ -f "$2" ]; then
    # --no-deps: the file pins cloudpickle with hashes; the shim itself is on
    # PYTHONPATH, and letting pip drag in `-e ./python` would install the very
    # tree under test.
    uv pip install --python "$1" --no-deps --require-hashes -r "$2" >/dev/null
  else
    uv pip install --python "$1" "$CLOUDPICKLE_SPEC" >/dev/null
  fi
}
