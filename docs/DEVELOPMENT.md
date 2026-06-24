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
  test_client.py / test_util.py / test_runtime.py / test_misc.py / test_scanner.py

test/                    end-to-end harnesses (shell)
  run_e2e.sh             single head, 4 fake GPUs
  run_multinode.sh       GPU-less head + 4-GPU worker, colocated
  run_edge.sh            edge cases (errors, wait, parallelism, kill, leak-fix)
  run_3node.sh           3 real machines: this host head + 2 sparks (needs SSH)
  run_cpu_cluster.sh     N real machines, CPU-only control plane (3, 4, ... nodes)
  run_cpuhead_gpuworkers.sh  CPU head + 2 GPU workers, vLLM TP=2 (sparks)
  run_rocm*.sh           AMD ROCm: single-node + cross-node harnesses
  dgx/                   two-node DGX Spark harness over SSH (see test/dgx/README.md)

docs/                    DESIGN + ARCHITECTURE/PROTOCOL/API/OPERATIONS/DEVELOPMENT + logo.svg
```

## Running the tests

Unit + fuzz suite (pytest + hypothesis, no GPUs/torch; 100% coverage of
`python/ray`, gated in CI):

```
uv run --with pytest --with hypothesis --with pytest-cov --with cloudpickle \
  pytest tests/ -q --cov=ray --cov-report=term-missing --cov-fail-under=100
```

End-to-end control-plane harnesses (fake GPUs via `BEAM_NUM_GPUS`, need only
`uv` + cloudpickle):

```
bash test/run_e2e.sh          # single-node control plane
bash test/run_multinode.sh    # cross-node routing through the hub
bash test/run_edge.sh         # error propagation, wait, parallelism, leak-fix, …
```

Real multi-node: `test/run_cpu_cluster.sh` (N CPU machines), or edit
`test/dgx/config.sh` and run `./test/dgx/dgx.sh all` (two GPU nodes). See the
[validated topologies](../README.md#validated-topologies) table for the rest.

## Lint, format, types

Configured in the repo-root `pyproject.toml`; CI's `lint` job runs all of these
(see `.github/workflows/ci.yml`):

```
uvx ruff check python examples scripts tests
uvx black --check python examples scripts tests
uvx --with cloudpickle mypy --config-file pyproject.toml python/ray
shellcheck -x test/*.sh test/dgx/*.sh
```

ruff/black use line-length 100. The library (`python/ray`) is **fully typed**:
mypy runs strict (`disallow_untyped_defs`, `disallow_incomplete_defs`) and is
clean; keep it that way when adding code.

## Keeping the shim in sync with vLLM

vLLM changes which ray symbols it imports between releases. The scanner is the
guard. On a vLLM bump:

```
git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm
uv run --with cloudpickle python scripts/scan_vllm_ray.py --src /tmp/vllm
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

## Conventions

- The wire format is the single source of truth; the sync side (`_proto.py`,
  `_client.py`, `_worker.py`) and the async side (`_daemon.py`) implement it
  separately on purpose (blocking sockets vs asyncio streams).
- Annotations use `from __future__ import annotations` where PEP604 unions appear,
  so the shim imports on Python 3.9+.
- The daemon never unpickles payloads; only the shim and the actor worker do.
  Keep it that way — it is what lets the daemon stay agnostic to vLLM's classes.
