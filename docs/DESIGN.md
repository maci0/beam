# beam

A drop-in alternative to Ray, scoped to exactly one job: running vLLM
distributed inference across multiple nodes
(https://docs.vllm.ai/en/v0.23.0/serving/parallelism_scaling/).

Not a general Ray clone. It implements the slice of Ray that vLLM's
`RayDistributedExecutor` touches and nothing else.

## Size vs Ray

beam implements one slice of Ray's API, so it is several orders of magnitude
smaller. Measured 2026-06 (ray 2.55.1 installed with `--no-deps`; beam = the
`python/` package you bind-mount):

| | beam | Ray | ratio |
|---|------|-----|-------|
| install size | 124 KB | 183 MB | ~1,500× |
| Python LoC | 1,866 (1,475 code) | 643,901 | ~345× |
| Python files | 21 | 2,417 | ~115× |
| native libraries | 0 | 11 `.so` (50 MB) | — |
| runtime dependencies | 1 (cloudpickle) | ~12 required (grpcio, protobuf, msgpack, …) + many extras | — |
| build | none, pure Python | Bazel + C++ core | — |

LoC is the shipped `python/` package only (the `tests/` suite is not counted).
beam is ~0.07% of Ray's on-disk footprint and ~0.3% of its Python. That is the
point: it is the control plane vLLM needs, with nothing to compile and one
dependency, small enough to read end to end and bind-mount into a stock image.

## Why this is small

vLLM uses Ray only as a **control plane**:

- start one worker per GPU, on the right node, with the right
  `CUDA_VISIBLE_DEVICES`,
- broadcast `execute_model` / `load_model` / `init_device` method calls to all
  workers,
- gather the small return values.

The actual tensor-parallel traffic (activations, all-reduce) goes over
**NCCL / torch.distributed**, which vLLM sets up itself and which never touches
Ray. So beam does not need an object store for big tensors, shared memory, a
scheduler, autoscaling, or any of Ray's data plane. It needs membership, GPU
accounting, placement groups, and an actor-call hub.

## The contract (everything vLLM imports)

CLI:

    ray start --head [--port 6379] [--num-gpus N]
    ray start --address HOST:PORT [--num-gpus N]
    ray status
    ray stop

Python `import ray`:

    ray.init(address=..., ignore_reinit_error=, ...)
    ray.is_initialized() -> bool
    ray.shutdown()
    ray.remote(**opts)(Cls).remote(*a, **kw) -> ActorHandle
    handle.method.remote(*a, **kw) -> ObjectRef
    ray.get(ref | [refs], timeout=) -> value | [values]
    ray.put(obj) -> ObjectRef
    ray.wait(refs, num_returns=, timeout=) -> (ready, not_ready)
    ray.get_gpu_ids() -> [int]
    ray.get_runtime_context() -> ctx
        ctx.get_node_id() -> str
        ctx.get_accelerator_ids() -> {"GPU": [str,...]}
    ray.cluster_resources() -> {"GPU": float, "CPU": float, ...}
    ray.available_resources() -> {...}
    ray.nodes() -> [ {NodeID, Alive, Resources, NodeManagerAddress}, ... ]
    ray.util.get_node_ip_address() -> str
    ray.util.placement_group(bundles, strategy=) -> PlacementGroup
        pg.ready() -> ObjectRef   (ray.get(pg.ready()) blocks until placed)
        pg.bundle_specs -> [ {"GPU":1}, ... ]
        pg.wait(timeout) -> bool
    ray.util.get_current_placement_group() -> PlacementGroup | None
    ray.util.remove_placement_group(pg)
    ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
        placement_group=, placement_group_bundle_index=,
        placement_group_capture_child_tasks=)

That is the full surface. Anything outside it is out of scope by design.

## Topology

Star. The head daemon is the router/hub.

    driver (vLLM)            worker node                 worker node
      |                        |                            |
   ray shim                 beamd  -- python actor       beamd -- python actor
      | unix sock              | TCP                         | TCP
    beamd (head) <-------------+----------------------------+
       \__ python actors (local GPUs)

- Each node runs one `beamd`.
- Non-head daemons dial the head once and keep the connection. The head pushes
  actor-create and actor-call requests down those connections and reads back
  results. Workers never talk to each other through beam (NCCL handles that).
- The vLLM driver process connects to its **local** daemon over a unix socket.

Control-plane volume is tiny (a handful of RPCs per inference step, all with
small payloads), so routing everything through the head hub is fine.

## Actors

`Cls.remote(...)` ->

1. shim sends `create_actor` to local daemon with the pickled class+args,
   `num_gpus`, and the placement (pg id + bundle index),
2. head picks the node that owns that bundle, forwards to that node's daemon,
3. that daemon spawns `python -m ray._worker` with `CUDA_VISIBLE_DEVICES` set to
   the bundle's GPUs and a unix socket back to itself,
4. the worker unpickles the class, instantiates it, then serves method calls one
   at a time (Ray actors are single-threaded; different actors run in parallel).

`handle.method.remote(args)` returns an `ObjectRef` immediately. The call is
dispatched async; the worker posts the pickled result back to its daemon's
object store keyed by the object id. `ray.get` blocks until the result is there,
fetching cross-node through the head hub if needed.

Object ids are `"<owner-node>-o<seq>"` so any daemon can route a fetch to the
owner without a lookup.

## Wire protocol

Full message reference in [docs/PROTOCOL.md](PROTOCOL.md). In brief, one
framing for every link (unix and TCP):

    [4-byte big-endian length][JSON header][optional raw payload]

The JSON header carries a `plen` field; if non-zero, that many raw bytes
(pickled Python objects) follow the header. Requests carry a `reqid`; responses
echo it with `resp=true`. A single connection is a bidirectional RPC mux: the
head both answers shim requests and initiates requests to workers over the same
socket.

Everything is Python (asyncio): the daemon (`ray._daemon`), the shim
(`import ray`), and the actor workers (`ray._worker`). The control plane is a
handful of tiny RPCs per step, so a single language with no build step beats a
faster daemon in another language. Inject it into the stock vLLM image with one
bind mount; it runs on the image's own python.

## Out of scope (named, not hidden)

- Object store for large data / shared memory / spilling. Results are passed
  inline through the hub. Ceiling: fine for vLLM's small control returns, not for
  passing tensors through beam (vLLM doesn't).
- Fault tolerance, actor restart, autoscaling, the Ray dashboard, GCS
  persistence, namespaces, async actors, tasks (non-actor `@ray.remote` funcs;
  vLLM only uses actor classes).
- `VLLM_USE_RAY_COMPILED_DAG` (accelerated DAG). vLLM falls back to plain
  per-worker RPC when it is off, which is the default.
