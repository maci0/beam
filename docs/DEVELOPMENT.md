# Development

## File map

```
python/ray/
  __init__.py            the ray API: init/get/put/wait/remote/kill/runtime ctx/resources
  _client.py             synchronous unix-socket client to the local daemon
  _proto.py              frame encode/decode + cloudpickle helpers (sync side)
  _daemon.py             the asyncio daemon: membership, placement, actor hub, routing
  _cli.py                `ray start/status/stop/bootstrap` (start runs the daemon)
  __main__.py            `python -m ray` → _cli.main
  _worker.py             actor subprocess: instantiate class, serve method calls
  util/
    __init__.py          re-exports + placement_group_table, get_node_ip_address
    placement_group.py   PlacementGroup, placement_group(), pg id with .hex()
    scheduling_strategies.py   PlacementGroupSchedulingStrategy, NodeAffinity…
    state.py             list_nodes()
    metrics.py           no-op Metric/Gauge/Counter/Histogram
  _private/state.py      available_resources_per_node, total_resources_per_node
  runtime_env.py         RuntimeEnv (dict, ignored)
  types.py, actor.py     ObjectRef / ActorHandle re-exports
  exceptions.py          RayError family
  dag.py                 compiled-DAG stubs (raise if used)
  cloudpickle.py         re-export of cloudpickle
  experimental/__init__.py   empty pkg so vLLM's compiled-DAG probe returns None

examples/
  driver_demo.py         vLLM-style: placement group → 1 actor/bundle → broadcast/gather
  edge_cases.py          error propagation, put/get, wait, parallelism, big payloads, leak-fix
  import_check.py        import-only smoke test of the whole shim surface

scripts/
  scan_vllm_ray.py       scan a vLLM checkout for ray usage vs the shim

tests/                   pytest + hypothesis, 100% coverage of python/ray
  test_proto.py          wire framing (roundtrip + garbage/oversize fuzz)
  test_units.py          daemon pure helpers (placement, ids, membership)
  test_daemon_handlers.py  the async on_* handlers, driven via a fake Peer
  test_shim.py           the ray shim's request translation
  test_cli.py            start/status/stop arg parsing + runtime files
  test_config.py         env-var loaders and their validation
  test_client.py / test_util.py / test_runtime.py / test_misc.py / test_scanner.py
  test_replay.py         the BEAM_SEED replay profile in conftest.py

test/                    end-to-end harnesses (shell)
  run_e2e.sh             single head, 4 fake GPUs
  run_multinode.sh       GPU-less head + 4-GPU worker, colocated
  run_edge.sh            edge cases (errors, wait, parallelism, kill, leak-fix)
  run_driver_on_worker.sh  driver on a worker node, head stays pure control plane
  run_3node.sh           3 real machines: this host head + 2 sparks (needs SSH)
  run_cpu_cluster.sh     N real machines, CPU-only control plane (3, 4, ... nodes)
  run_cpuhead_gpuworkers.sh  CPU head + 2 GPU workers, vLLM TP=2 (sparks)
  run_rocm*.sh           AMD ROCm: single-node + cross-node harnesses
  dgx/                   two-node DGX Spark harness over SSH (see test/dgx/README.md)

docs/                    DESIGN + ARCHITECTURE/PROTOCOL/API/OPERATIONS/DEVELOPMENT
                          + THREAT_MODEL (attack surface, boundaries, controls) + logo.svg
SECURITY.md              deployment checklist, supported versions, disclosure policy
CHANGELOG.md             what changed per release; docs/RELEASING.md how a tag is cut
```

## Running the tests

Everything below is a `make` target wrapping the exact command CI runs; `make`
on its own lists them. The only prerequisites are [`uv`](https://docs.astral.sh/uv/)
and `shellcheck`; no venv setup, no global installs, no GPU.

```
make check        # everything CI checks: lint, types, shell, yaml, unit+fuzz, import, e2e
make test         # unit + fuzz suite only (~10s, no GPUs)
make test-one T=tests/test_cli.py::test_start_needs_head_or_address
make e2e          # the four local control-plane harnesses
```

Under the hood `make test` is (pytest + hypothesis, no GPUs/torch; 100%
coverage of `python/ray`, gated in CI):

```
uv run --with pytest==$(PYTEST_VERSION) --with hypothesis==$(HYPOTHESIS_VERSION) \
  --with pytest-cov==$(PYTEST_COV_VERSION) --with 'cloudpickle>=3.1.2,<4' \
  pytest tests/ -q --cov=ray --cov-report=term-missing --cov-fail-under=100
```

(the versions are the `*_VERSION` variables at the top of the `Makefile`;
they are pinned there so a tool release cannot change a lint, typecheck, or test
run under your feet)

End-to-end control-plane harnesses (fake GPUs via `BEAM_NUM_GPUS`, need only
`uv`), also runnable one at a time. They install cloudpickle from
`requirements.lock` — the hash-pinned export of `uv.lock` (`make
lockfile-export` refreshes it) — so every harness, node and worker unpickles
with the same cloudpickle the wheel was resolved against:

```
bash test/run_e2e.sh          # single-node control plane
bash test/run_multinode.sh    # cross-node routing through the hub
bash test/run_edge.sh         # error propagation, wait, parallelism, leak-fix, …
bash test/run_driver_on_worker.sh  # driver on a worker, head is pure control plane
```

They keep the daemon runtime dir under `$TMPDIR` (via `test/lib.sh`), not in
the checkout: the control socket is an AF_UNIX socket, so its path is capped at
107 bytes and a deep checkout would otherwise fail with a bare
`AF_UNIX path too long`. `ray start` now refuses a `BEAM_RUNTIME_DIR` that
overruns that budget and says so, so the same mistake on your own node is a
one-line error instead of a stack trace.

Real multi-node: `test/run_cpu_cluster.sh` (N CPU machines), or edit
`test/dgx/config.sh` and run `./test/dgx/dgx.sh all` (two GPU nodes). See the
[validated topologies](../README.md#validated-topologies) table for the rest.

## Lint, format, types

Configured in the repo-root `pyproject.toml`; CI's `lint` job runs all of these
(see `.github/workflows/ci.yml`), and `make lint`, `make types`, `make shell`
run the same commands:

```
make lint    # uvx ruff==$(RUFF_VERSION) check python examples scripts tests
             # uvx black==$(BLACK_VERSION) --check python examples scripts tests
make types   # uvx --with cloudpickle mypy==$(MYPY_VERSION) \
             #        --config-file pyproject.toml python/ray
make shell   # shellcheck -x test/*.sh test/dgx/*.sh
make yaml    # uvx yamllint==$(YAMLLINT_VERSION) -c .yamllint.yml .github/workflows/*.yml
make format  # black, in place
```

Tool versions live at the top of the `Makefile` in one block, pinned exactly:
`uvx <tool>` otherwise resolves to whatever is newest at run time, so a lint or
typecheck run could go red (or pass differently) after an upstream release.

ruff/black use line-length 100, and ruff's E501 is on, so a line that black
cannot split (a long string, say) still fails the lint job. ruff, black, mypy
and yamllint are **version-pinned** in the `Makefile` and CI pins uv itself, so
a new release of any of them cannot change the verdict on a commit that did not
touch a lintable file; bump the pin deliberately, in the same commit as the
reformat it forces. The library
(`python/ray`) is **fully typed**: mypy runs with `disallow_untyped_defs`,
`disallow_incomplete_defs`, `disallow_untyped_calls`,
`disallow_untyped_decorators`, `strict_equality`, `extra_checks` and
`warn_no_return`, and is clean; keep it that way when adding code. The strict
flags still off (`warn_return_any`, `warn_unreachable`,
`disallow_any_generics`, `no_implicit_reexport`) each have open findings; turn
one on as its findings are cleared, rather than all at once.

Two suppression notes, both enforced by ruff (`RUF100` fails on a `noqa` that
no longer suppresses anything):

- the re-export modules (`python/ray/__init__.py`, `types.py`, `actor.py`,
  `util/__init__.py`, `cloudpickle.py`) need no `noqa` at all: they are covered
  by the per-file-ignores below, and a `noqa` there would be dead weight;
- `sys.path.insert(...)` before an import is **not** an E402, so the tests need
  no `noqa: E402` there either. Do not add one back.

## Keeping the shim in sync with vLLM

vLLM changes which ray symbols it imports between releases. The scanner is the
guard. On a vLLM bump:

```
git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm
uv run --with 'cloudpickle>=3.1.2,<4' python scripts/scan_vllm_ray.py --src /tmp/vllm
```

It prints every `ray.*` symbol vLLM uses, marks each covered / out-of-scope /
MISSING, and exits non-zero if anything in-scope is MISSING (CI gate). Symbols
under `OUT_OF_SCOPE` in the script (ray.data / ray.serve / ray.experimental /
TPU) are reported, not failed.

To cover a newly-required symbol:

- pure data/type/exception → add a stub module or attribute (see
  `runtime_env.py`, `types.py`, `exceptions.py` for the pattern),
- something that needs cluster state → add an `on_<t>` handler in `_daemon.py`
  and a thin call in the shim (see how `placement_group_table` → `pg_table` and
  `available_resources_per_node` → `resources` are wired).

## Deterministic simulation and replay

The control plane is written so a whole run can be driven by one seed instead of
by real time and OS entropy. Four env vars are the seams; unset in production,
so default behaviour and timings are unchanged:

| var | replaces | with |
|-----|----------|------|
| `BEAM_TIMEOUT` | the daemon's 30s/120s `wait_for` budgets | a capped budget (capped, never stretched) |
| `BEAM_SLEEP` | `asyncio.sleep` (daemon) / `time.sleep` (shim) | `module:callable`, `hook(seconds) -> awaitable` (daemon) or `-> None` (shim) |
| `BEAM_CLOCK` | `time.monotonic` in the shim's deadlines | `module:callable` returning seconds |
| `BEAM_SEED` | `secrets.token_hex` in `new_node_id` | SHA-256 of the seed plus a per-process counter |

Only two of the four are captured at import time: `ray._CLOCK_HOOK` (read once
when `ray` is imported) and `_daemon._SLEEP_HOOK`. `BEAM_TIMEOUT`, the shim's
`BEAM_SLEEP`, and `BEAM_SEED` are read from the environment on every call, so a
test can change them mid-run. That is why the suite patches
`monkeypatch.setattr(ray, "_CLOCK_HOOK", "simclock:now_seconds")` for the clock
and `monkeypatch.setenv("BEAM_SLEEP", ...)` for the rest.

```bash
# same seed, twice, same node ids and same example sequence
BEAM_SEED=repro1 bash test/run_e2e.sh
BEAM_SEED=repro1 bash test/run_e2e.sh

# derandomized property tests: a flaky failure reproduces instead of moving
BEAM_SEED=repro1 uv run --with pytest --with hypothesis --with cloudpickle pytest tests/
# one-off: pin a single example
pytest tests/test_proto.py --hypothesis-seed=1234
```

`tests/conftest.py` turns `BEAM_SEED` into hypothesis's `replay` profile
(derandomized, no example database), so the property suite replays exactly like
the daemon ids do. CI leaves `BEAM_SEED` unset so the database-backed search
keeps finding new failures.

Not yet modelled (deliberate gaps, in rough priority order): no fault-injection
hook for torn writes or `fsync` failure, no virtual network (latency, reorder,
drop, partition) between nodes, and no action/event log to diff a divergent
replay against. The actor subprocess is already swappable via `BEAM_WORKER_CMD`,
which is the seam a crash/restart simulator would drive.

## Conventions

- The wire format is the single source of truth; the sync side (`_proto.py`,
  `_client.py`, `_worker.py`) and the async side (`_daemon.py`) implement it
  separately on purpose (blocking sockets vs asyncio streams).
- Annotations use `from __future__ import annotations` where PEP604 unions appear,
  so the shim imports on Python 3.9+.
- The daemon never unpickles payloads; only the shim and the actor worker do.
  Keeping it that way is what lets the daemon stay agnostic to vLLM's classes.
  It still *stores and forwards* payloads verbatim (`on_put` keeps them in
  `self.objects`, and every handler hands the raw payload to the next hop), so
  the daemon is an unauthenticated relay for attacker-chosen bytes, not a
  parser of them. See docs/THREAT_MODEL.md.
