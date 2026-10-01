# Contributing to beam

beam is pure Python with one dependency (`cloudpickle`); the installable
package lives in `python/` (named `ray` on purpose, so it shadows the real
package), the tools live in this repo.

## Setup

You need [`uv`](https://docs.astral.sh/uv/) and `shellcheck`. Nothing else:
no venv to create, no global installs, no GPU, no torch. Every command below
resolves its own dependencies into a cached environment.

    make          # list the targets
    make check    # everything CI runs: lint, types, shellcheck, yamllint, tests, import, e2e

## The edit-test loop

    make test                                  # unit + fuzz suite, ~10s
    make test-one T=tests/test_cli.py::test_x   # one test, file, or -k pattern
    make e2e                                   # control-plane harnesses (no GPU)

`make test` enforces the 100% coverage gate CI enforces, so a new line without
a test fails locally, not after a push. To iterate faster, run one file or
pattern without the gate:

    make test-one T=tests/test_cli.py -k test_name

Tool versions (pytest, hypothesis, ruff, mypy, …) are pinned in one block at
the top of the `Makefile`; bump them there when you want a newer tool, so a
release cannot change a run you did not ask it to change.

Conventions live in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): the file map,
the wire protocol being the source of truth, and the vLLM-sync scanner to run
when vLLM changes which `ray.*` symbols it imports.

## Adding to the shim

- a pure data/type/exception symbol: add a stub module or attribute, following
  `runtime_env.py`, `types.py`, `exceptions.py`;
- something needing cluster state: add an `on_<t>` handler in `_daemon.py` and
  a thin call in the shim, following `placement_group_table` or `resources`;
- anything vLLM starts importing: run the scanner (below) and cover what it
  reports as MISSING. CI's `vllm-surface` job runs the same scan but is allowed
  to fail, so the check that keeps it honest is a local one.

    git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm
    uv run --with 'cloudpickle>=3.1.2,<4' python scripts/scan_vllm_ray.py --src /tmp/vllm

## Before you open a PR

1. `make check` is green. It runs the same commands as CI, so a green run means
   CI is green.
2. New code in `python/ray` is fully typed (mypy strict) and covered; new
   shell code passes `make shell`.
3. If a change alters a protocol message, update `docs/PROTOCOL.md` and
   `docs/API.md` in the same commit.
4. Nothing generated is committed by hand: `make build` makes the wheel/sdist
   (`cd python && uv build`), `make bundle` makes the bind-mount tarball, both
   with the same flags the release workflow uses. `make repro` builds each of
   them twice from perturbed copies and fails unless the bytes match; CI's
   `artifacts` job runs it, so a timestamp or a path leaking into an artifact
   is a red build, not a surprise.
5. `requirements.lock` is generated (`make lockfile-export`), not hand-edited;
   refresh it in the same commit as any `uv.lock` change.
6. If the change alters what a caller observes (an exported symbol, a CLI
   exit code, a wire field), add it to the `Unreleased` section of
   CHANGELOG.md grouped Added / Changed / Fixed / Breaking. Anything that
   changes behavior rather than fixing it is at least a minor bump; see
   docs/RELEASING.md.

## Where things live

| path | what |
|---|---|
| `python/ray/` | the library: shim, client, daemon, CLI, actor worker |
| `tests/` | pytest + hypothesis, 100% coverage of `python/ray` |
| `test/` | shell harnesses (`make e2e`, plus multi-node and GPU ones) |
| `examples/` | driver demo, edge cases, import smoke test |
| `docs/` | design, architecture, protocol, API, operations, development |
| `CHANGELOG.md` | what changed per release, grouped by impact |
| `docs/RELEASING.md` | versioning, tags, and how a release is cut |
| `Makefile` | the task runner; every target wraps a CI command, and pins the tool versions |
| `uv.lock` / `requirements.lock` | resolved dependency tree, and its hash-pinned install list |

Multi-node and GPU harnesses need real machines and are documented in
[test/dgx/README.md](test/dgx/README.md) and the topology table in the README.
