# API surface

Every `ray` symbol beam implements, why it exists, and what it maps to. This is
exactly the set vLLM's `RayDistributedExecutor` / `ray_executor_v2` touches,
verified by `scripts/scan_vllm_ray.py` against a live vLLM checkout (it prints
the in-scope/out-of-scope/missing counts; the gate is 0 missing). The exact
symbol count drifts with vLLM, so run the scanner rather than trusting a number
here.

## Lifecycle

| symbol                | maps to                                              |
|-----------------------|------------------------------------------------------|
| `ray.init(address=…)` | open a unix-socket client to the local daemon        |
| `ray.is_initialized()`| local boolean                                        |
| `ray.shutdown()`      | close the client                                     |
| `ray.__version__`     | reports `"2.43.0"` so vLLM version/metadata checks pass |

`address`, `runtime_env`, and other kwargs are accepted and ignored — beam
always uses the local daemon.

## Actors

| symbol                              | maps to                                |
|-------------------------------------|----------------------------------------|
| `ray.remote(**opts)(Cls)`           | wrap class + options (`num_gpus`, `scheduling_strategy`) |
| `.options(**opts)`                  | merge more options                     |
| `RemoteCls.remote(*a, **kw)`        | `create_actor` (pickled class+args; pg+bundle from the scheduling strategy) → `ActorHandle` |
| `handle.method.remote(*a, **kw)`    | `call` (pickled args) → `ObjectRef`    |
| `ray.kill(handle)`                  | `kill` (stop the worker subprocess)    |

Actors are real subprocesses (`python -m ray._worker`), one per actor, each with
its assigned GPU in `CUDA_VISIBLE_DEVICES`, single-threaded.

## Objects

| symbol                       | maps to                                       |
|------------------------------|-----------------------------------------------|
| `ray.put(obj)`               | `put` → `ObjectRef`                           |
| `ray.get(ref \| [refs])`     | `get` (blocks until ready); unpickles result  |
| `ray.wait(refs, num_returns, timeout)` | poll `stat`; returns `(ready, not_ready)` |

Results are passed inline through the hub. This is fine for vLLM's small control
returns; beam is not an object store for tensors (vLLM never passes tensors
through ray).

## Resources & context

| symbol                                         | maps to                          |
|------------------------------------------------|----------------------------------|
| `ray.get_gpu_ids()`                            | `BEAM_GPU_IDS` env in the worker |
| `ray.get_runtime_context().get_node_id()`      | `BEAM_NODE_ID` env / runtime file|
| `…get_accelerator_ids()`                       | `{"GPU": [ids]}` from env        |
| `ray.cluster_resources()` / `available_resources()` | derived from `status`       |
| `ray.nodes()`                                  | `status` → ray-shaped node dicts |
| `ray._private.state.available_resources_per_node()` | `resources`                 |
| `ray._private.state.total_resources_per_node()` | `status`                        |
| `ray.util.state.list_nodes()`                  | `status` → node objects          |

`available_resources_per_node` and `placement_group_table` are imported at
module top in vLLM's `ray_utils.py`; if they were missing, vLLM would conclude
ray is unavailable and refuse the backend. They are not optional.

## Placement groups

| symbol                                            | maps to                       |
|---------------------------------------------------|-------------------------------|
| `ray.util.placement_group(bundles, strategy)`     | `create_pg` → `PlacementGroup`|
| `pg.ready()`                                       | already-resolved `ObjectRef`  |
| `pg.bundle_specs`, `pg.id.hex()`                   | local                         |
| `ray.util.placement_group_table(pg)`              | `pg_table` → bundle→node map  |
| `ray.util.get_current_placement_group()`          | `None` (driver isn't in a pg) |
| `ray.util.remove_placement_group(pg)`             | `remove_pg`                   |
| `ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy` | data holder vLLM passes to `ray.remote` |

beam places bundles synchronously at creation, so `ray.get(pg.ready())` returns
immediately.

## Stubs (present so imports/annotations resolve, not functional)

| symbol                          | behavior                                         |
|---------------------------------|--------------------------------------------------|
| `ray.runtime_env.RuntimeEnv`    | dict subclass; contents ignored (actor env is inherited from the daemon) |
| `ray.util.metrics.{Metric,Gauge,Counter,Histogram}` | no-op (vLLM's Prometheus path is unaffected) |
| `ray.types.ObjectRef`, `ray.actor.ActorHandle` | re-exports for type hints      |
| `ray.exceptions.{RayError,RayActorError,GetTimeoutError,RayChannelError,…}` | subclasses of RuntimeError |
| `ray.dag.{CompiledDAG,InputNode,MultiOutputNode}` | raise if used; only reached if compiled-DAG is enabled (it isn't) |
| `ray.cloudpickle`               | re-export of cloudpickle                          |

## Out of scope (reported by the scanner, never failed)

Separate opt-in integrations and the compiled-DAG data plane, none on the
multi-node NCCL inference path:

- `ray.data`, `ray.data.llm.*` — Ray Data batch inference
- `ray.serve`, `ray.serve.llm.*` — Ray Serve deployments
- `ray.experimental.channel.*` — compiled-DAG accelerator channels (vLLM gates
  this on `ray.experimental.compiled_dag_ref`, which beam omits, so vLLM keeps
  the default per-worker-RPC path)
- `ray._private.accelerators.TPUAcceleratorManager` — TPU only

If a future vLLM version moves one of these onto the core path, the scanner flags
it as MISSING and the fix is to add a stub or a daemon op.
