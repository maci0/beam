# Changelog

All notable changes to beam are recorded here, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and beam follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) for its own tags
(`vX.Y.Z`). See [docs/RELEASING.md](docs/RELEASING.md) for how a release is cut.

## How to read the versions

beam publishes two version numbers and they mean different things:

- The **tag** (`v0.2.0`, this file's headings) is beam's release line. It is the
  only version that says anything about beam's own compatibility, and it is the
  version `SECURITY.md` calls "the current release".
- The **distribution version** (`ray 2.43.0`, in `python/pyproject.toml` and
  `ray.__version__`) is pinned to a real Ray release on purpose. beam shadows the
  `ray` distribution so vLLM's version and metadata checks resolve; the number
  tracks the Ray release whose executor API beam implements, not beam's own
  history. Do not read it as "which beam is this" — read the tag.

Both numbers are checked against each other and against the tag by
`scripts/check_release.py`, which `make check` and the release workflow run.

## [Unreleased]

Ten commits since v0.2.0, none of them released yet. The behavioral changes a
consumer can notice:

### Added

- `BEAM_BIND_ADDRESS`: bind the head's control port to one address instead of
  `0.0.0.0`. The control plane is unauthenticated, so narrowing the bind was
  the one lever the operator did not have.
- `ray --version`, and `-h`/`--help` on every subcommand.
- New environment-variable validation at startup: `BEAM_NUM_GPUS`,
  `BEAM_NODE_IP`, `VLLM_HOST_IP` and `BEAM_BIND_ADDRESS` are parsed and
  range-checked once (`_config.py`) and a bad value exits 2 with a message naming
  the variable instead of crashing at the moment it mattered. A multicast
  `BEAM_BIND_ADDRESS` is rejected.
- `docs/THREAT_MODEL.md`, `SECURITY.md`, and a supported-versions statement
  (one line: the current release; no backports).

### Changed

- `ray.get(timeout=...)` and `ray.wait(timeout=...)` are elapsed-time budgets on
  the monotonic clock, and the budget now bounds the whole call: each round-trip
  carries the time remaining, so a daemon that stops answering ends `get` with
  `GetTimeoutError` and returns from `wait` at the timeout instead of hanging
  past it. Before, both measured the deadline on `time.time()` (an NTP step or a
  suspend could end a call early or stretch it) and both could block well past
  the budget they were promised.
- `ray.get` on an already-resolved ref returns the value even when the budget has
  expired, and `timeout=0` is a non-blocking poll rather than "always fail". The
  daemon used to answer `wait_for(ev, 0)` without ever looking at the flag and
  reported a resolved object as not ready.
- `create_actor` now answers with the owning `node` on the local-host path too,
  matching the remote path. `ray.get_runtime_context().get_node_id()` and vLLM's
  actor-placement logging saw the owner only for remotely hosted actors.
- `ray --help` prints to stdout and exits 0; bad usage still goes to stderr and
  exits 2. Help and version are documented as exit code 0 in the usage text.

### Fixed

- `ray start` refuses a `BEAM_RUNTIME_DIR` whose unix-socket path exceeds the
  AF_UNIX limit with a named error, instead of a bare `OSError` out of
  `socket.connect()`.
- `DaemonNotRunning` replaces the `FileNotFoundError` / `KeyError` /
  `JSONDecodeError` a driver saw when no daemon was running.
- A `return` inside `finally` in the daemon's actor-kill path (a `SyntaxError`
  from CPython 3.14) is gone; the outcome is flagged and returned after.
- Client, daemon, and CLI read every environment variable through one module, so
  a value is validated the same way everywhere it is read.

### Breaking

- `ray status`, `ray stop`, and `ray bootstrap` now reject stray arguments
  (exit 2) instead of silently ignoring them. `ray stop --force` used to look
  like it had forced something when it had done nothing. There is no `--force`
  flag on any beam command.
- A `ray` invocation with no arguments still prints usage on stderr and exits 2,
  but the text now documents the exit codes, the `BEAM_*` variables, and
  `BEAM_BIND_ADDRESS`. Any script that matched the old usage text breaks.

Upgrade: nothing in the `ray` Python surface vLLM imports changed shape. Scripts
that invoke the CLI must drop stray arguments and may rely on `--help` exiting 0.

## [0.2.0] - 2026-08-26

### Fixed

- The stale-runtime claim test no longer depends on ambient pids, which had been
  failing inside the CI container since 15 Aug: it wrote a runtime file claiming
  pid 9, and the seize path's `kill(0)` on a pid it does not own raises `EPERM`
  inside the container where pid 9 exists, so the claim read as live.

## [0.1.3] - 2026-08-26

### Changed

- Correctness and accuracy pass over the tree: `Peer.superseded` /
  `superseded_by` are declared fields instead of attributes attached through a
  `type: ignore` and read back with `getattr` defaults; `_terminate` narrows its
  excepts to `(subprocess.TimeoutExpired, OSError)` (eight tests had raised
  `TimeoutExpired` without importing `subprocess`, so the `NameError` was
  swallowed and the timeout branch was never tested); repeated timeouts and poll
  budgets are named constants.
- Size and test-count claims in README/DESIGN.md re-measured;
  `BEAM_NODE_IP` and `BEAM_BOOTSTRAP` documented.

No change to the `ray` API surface.

## [0.1.2] - 2026-08-04

### Fixed

- Daemon ownership cleanup and stop/claim races: fail-pending close drain, orphan
  reapers, GPUs freed only after process death, re-`hello` placement-group
  handoff, and CLI rename-seize/stop holds.
- A failed remote actor create no longer frees the head's GPUs twice.

## [0.1.1] - 2026-06-25

### Added

- The node IP can be set explicitly (`BEAM_NODE_IP` / `VLLM_HOST_IP`) instead of
  being guessed from the default route, which on a multi-homed host (router, VM
  bridges) is often not the address peers can reach.

### Fixed

- Cross-node tensor parallelism validated over gloo; the IP and firewall
  gotchas recorded in the docs.

## [0.1.0] - 2026-06-25

### Added

- First release. Pure-Python drop-in subset of Ray's API for vLLM multi-node
  distributed inference: cluster membership, GPU accounting, placement groups,
  and an actor-call hub. Bind-mount into the stock vLLM image; tensor traffic
  stays on NCCL/RCCL. Fully typed, 100% test coverage, validated cross-node.

[Unreleased]: https://github.com/maci0/beam/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/maci0/beam/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/maci0/beam/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/maci0/beam/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/maci0/beam/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maci0/beam/releases/tag/v0.1.0