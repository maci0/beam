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

beam_runtime_dir() {  # $1 = tag; prints a fresh short dir, safe to delete
  local dir="${TMPDIR:-/tmp}/beam-$1-$$"
  rm -rf "$dir"; mkdir -p "$dir"
  printf '%s\n' "$dir"
}
