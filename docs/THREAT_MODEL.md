# beam threat model

Scope: the `python/ray` package (the `ray` shim, `beamd`, the actor worker), the
`ray`/`beam` CLI, the Docker image, and the shell harnesses in `test/`. beam is
a **control plane** for vLLM distributed inference: it moves actor lifecycle and
method-call RPCs, never tensors (those go over NCCL, `docs/ARCHITECTURE.md`).

Every entry point, boundary, and control below carries a file:line reference so
a later pass can re-verify it against the code.

Last reviewed: 2026-09-30

## Risk-ranked summary

| # | Risk | Boundary | Where | Status |
|---|------|----------|-------|--------|
| R1 | Anyone who reaches the head's TCP port runs code as the daemon user: `create_actor` carries a cloudpickled class that a worker subprocess instantiates | network → daemon | `_daemon.py:639` (`on_create_actor`), `_daemon.py:991` (`init`), `_worker.py:43-45` | Accepted by design, documented in README; **no** authentication, authorization, or TLS anywhere on the link |
| R2 | The head binds `0.0.0.0` with no bind-address option, so a laptop, VM, or container bridge that happens to expose 6379 is a full cluster compromise | deployment | `_cli.py:225` (`serve_tcp("0.0.0.0", port)`) | Unmitigated in code; the only control is operator-side firewall/security group |
| R3 | Frame length is accepted up to 512 MiB per header/payload and there is no connection count, request rate, or quota limit: unbounded memory growth from one peer | network → daemon | `_proto.py:17`, `_daemon.py:89`, `read_frame` `_daemon.py:92` | Size ceiling only, no rate/count ceiling |
| R4 | Objects created by `put` are retained in daemon RAM forever (no eviction, no per-peer accounting, no TTL) | client → daemon | `_daemon.py:1258` (`on_put`), `_daemon.py:1280-1290` (comment: "keep the slot (not pop)") | Unmitigated |
| R5 | Every actor subprocess inherits the full daemon environment (`env = dict(os.environ)`), so every secret in the daemon's environment reaches every worker | daemon → actor subprocess | `_daemon.py:1071-1082` | Unmitigated by design; `RuntimeEnv` itself is not forwarded (`runtime_env.py:1-6`), but the daemon's own env is |
| R6 | Any local user who can open the runtime socket can drive the cluster: kill actors, allocate every GPU, read any object | local → daemon | `_daemon.py:363` (`serve_unix`), `_daemon.py:377-380` (`_on_conn` sets no peer credential check) | Socket permissions are whatever `umask` gives (`_daemon.py:366` uses default `makedirs`); no `chmod`, no `SO_PEERCRED`/uid check |
| R7 | `ray stop` signals whatever pid is in `~/.beam/daemon.json` — a writable file — with SIGTERM then SIGKILL | local → host process | `_cli.py:726` (SIGTERM), `_cli.py:745` (SIGKILL), pid read at `_cli.py:485-486` | Only mitigation is that the file is created with default umask; no ownership check before signalling |
| R8 | No audit trail: daemon logs are `print`/traceback to stdout, no requester identity (there is none), no record of who killed what | all | `_daemon.py:157-163` (traceback to stderr), `_cli.py:232-238` (startup prints) | Unmitigated |
| R9 | No documented vulnerability-reporting path and no stated supported-version policy | process | — | Fixed by `SECURITY.md` (added 2026-09-30) |

Design decisions that are *not* risks but bound the blast radius: beam never
touches the model weights, the token stream, or any request data beyond the
cloudpickled payloads it routes, and it never authenticates to anything — it
holds no credentials of its own.

## 1. Attack surface inventory

| Surface | Type | Endpoint / entry | Reference |
|---------|------|------------------|-----------|
| Head control port | TCP listener, all interfaces, default 6379 | framed JSON+payload protocol, bidirectional mux | `_daemon.py:372-374`, bound at `_cli.py:225` |
| Daemon unix socket | `AF_UNIX` stream | same protocol | `_daemon.py:363-370`, path `~/.beam/daemon.sock` (`_cli.py:201`) |
| Actor worker socket | `AF_UNIX` stream, one per actor subprocess | `worker_hello`, `init`, `method` | `_daemon.py:363` (same listener, distinguished by message type), `_worker.py:26-30` |
| Worker daemon → head dial | outbound TCP from `_start --address` | `hello` and everything forwarded afterwards | `_daemon.py:382-402` |
| CLI arguments | `ray start/status/stop/bootstrap` | `--head --port --address --num-gpus --node-ip` | `_cli.py:58-94` (usage), `_cli.py:96-176` (parsing) |
| Environment variables | read at start / per request | `BEAM_NODE_IP`, `VLLM_HOST_IP`, `BEAM_NUM_GPUS`, `BEAM_RUNTIME_DIR`, `BEAM_SOCK`, `BEAM_WORKER_CMD`, `BEAM_BOOTSTRAP` | `_cli.py:33-36`, `_cli.py:45-46`, `_daemon.py:265-272`, `_daemon.py:1068`, `_cli.py:907` |
| Runtime state files | `~/.beam/daemon.json`, `daemon.sock`, `*.tmp.<pid>`, `*.stale.<pid>`, `*.stopabandoned.<pid>` | written by `_write_runtime_atomic` / `_claim_runtime`, read by `ray status` / `ray stop` | `_cli.py:180-190`, `_cli.py:314-392`, `_cli.py:482-486` |
| Subprocess spawn | actor launch command from env | `shlex.split(BEAM_WORKER_CMD)`, full env inherited | `_daemon.py:1067-1083` |
| `/dev/nvidia*` glob, `/dev/dri`, `/dev/kfd` | host device probing | `detect_gpus` | `_daemon.py:265-272` |
| Cloudpickle payloads | deserialization boundary | actor class, method args, return values | `_worker.py:44,48`, `__init__.py:136` |
| Container bootstrap | writes outside the working tree when run in a container | `/usr/local/bin/ray`, `/usr/local/bin/beam`, `beam.pth` in site-packages, mode `0o755` | `_cli.py:913-944`, triggered by `/.dockerenv` or `BEAM_BOOTSTRAP` (`_cli.py:907`) |
| Shell harnesses | `test/*.sh` | docker + ssh into remote hosts, cloud-init on Azure VMs | `test/dgx/dgx.sh:10`, `test/run_rocm_azure.sh`, `test/run_cpu_cluster.sh` |
| Release pipeline | GitHub Actions, tag-triggered | builds wheel + sdist, attaches to release | `.github/workflows/release.yml` |

Not present (checked, so a future pass does not re-look): no HTTP server, no
webhook, no message broker, no scheduler/cron, no IPC other than the sockets
above, no database, no file upload/parse path, no credential store, no
dashboard or debug endpoint.

## 2. Trust boundaries

```
internet / cluster subnet
        |  TCP :6379, unauthenticated, cleartext
        v
  [ head daemon ]  ---- authority on membership, placement, routing, ids
        |  TCP (one connection per worker node), unauthenticated
        v
  [ worker daemon ] ---- spawns actor subprocesses with inherited env
        |  AF_UNIX, AF_UNIX
        v
  [ actor worker subprocess ]  <-- unpickles and executes
        ^
        |  AF_UNIX ~/.beam/daemon.sock
  [ vLLM driver / shim ]
```

| Boundary | What crosses it | Validation / auth point | Named in docs? |
|----------|------------------|---------------------------|----------------|
| B1 driver → local daemon | request headers, cloudpickled payloads | frame bounds only (`_proto.py:47-61`); header must be a JSON object; `plen` bounds | Protocol documented; auth absence documented in README only |
| B2 driver → **head** directly | anything a driver on the head sends | none | README "Security / trust model" |
| B3 remote peer → head TCP | `hello`, `create_actor`, `call`, `kill`, `pg_table`, `resources`, `status`, and any message a peer forwards | none; `handle` dispatches `on_<t>` by string (`_daemon.py:407-411`) | Protocol only |
| B4 daemon → actor subprocess | pickled class + ctor args, pickled call args, `BEAM_*`/`CUDA_VISIBLE_DEVICES` env | none; worker instantiates whatever arrives (`_worker.py:43-45`) | DESIGN/PROTOCOL describe the flow, not the risk |
| B5 actor subprocess → daemon | method return values | none | — |
| B6 build → runtime | wheel/sdist contents installed into the vLLM image (`Dockerfile:19-21`) | `examples/import_check.py` smoke test | no |
| B7 secrets → code | daemon environment inherited by every actor (`_daemon.py:1071`) | none | no |
| B8 local user → runtime files | `daemon.json` contents drive `ray stop` signalling | file-mode check on create only, implicit umask | no |

### Privilege transitions (undocumented in code)

- **Any peer becomes the cluster authority for the objects it names.** A TCP peer
  can `hello` with an arbitrary `node` id, and `on_hello` (`_daemon.py:425-472`)
  replaces the existing membership record for that id, transferring the old
  peer's `created_actors`/`created_pgs` ownership to itself and closing the old
  connection. There is no proof that the new peer is the same host.
- **`_forward_head` is a trust relay.** A worker daemon forwards arbitrary
  messages from a local driver to the head (`_daemon.py:1324-1345`,
  `_daemon.py:709-729`), so a driver on a worker node reaches head-only handlers.
- **Actor ids are attacker-chosen inputs to routing.** `owner_of` (`_daemon.py:278`)
  parses `<node>-o<n>` out of a request field, so `get`/`stat`/`call`/`kill` route
  by whatever the sender wrote.
- **GPU assignment is authoritative.** `_place_actor` (`_daemon.py:924-949`) and
  `on_create_pg` (`_daemon.py:557-596`) mark `self.gpu_used` / `self.pgs` from
  request contents; the head is the sole allocator.

### Secrets flow

There are no stored credentials anywhere in the codebase (grep for
`secret|password|token|hmac` returns only `secrets.token_hex` for node ids at
`_daemon.py:275`). The relevant flow is the reverse: secrets the *operator*
puts in the daemon environment (`HF_TOKEN`, cloud keys, vLLM config) are copied
verbatim into every actor subprocess by `env = dict(os.environ)`
(`_daemon.py:1071`). Actor workers also run arbitrary third-party vLLM code, so
anything in that environment is reachable by every actor. There is nothing to
rotate inside beam.

## 3. Assets and impact

| Asset | Why it matters | Worst case |
|-------|----------------|------------|
| GPU fleet / compute | the thing being rented; each actor pins `CUDA_VISIBLE_DEVICES` | full-cluster denial of service, or silent hijack of inference traffic by a rogue actor |
| Daemon user account on every node | actor subprocesses run as it | full read/write of whatever that user can reach (model weights, HF cache, container volumes) |
| Model weights / inference correctness | actors serve them; NCCL carries tensor traffic peer-to-peer | corrupted or exfiltrated weights; poisoned responses to every caller |
| Environment secrets in the daemon | inherited by every actor | credential theft across the whole cluster |
| Placement and routing tables | head-authoritative state in RAM (`nodes`, `pgs`, `actor_loc`, `gpu_used`, `objects`) | arbitrary re-mapping of actors to GPUs, including double-booking a GPU |
| Cluster membership | `hello` inserts nodes with self-declared `ip`/`ngpu` | traffic redirection: `status`/nodes report the attacker's address, and vLLM advertises it to its workers |
| Object store contents | `put` values held in daemon RAM | cross-driver read of any object id |

There is no PII, no customer data, and no billing path inside beam: prompts and
completions never pass through it.

## 4. Threats per boundary

### B3 / remote peer → head TCP (the internet-facing boundary)

- **Spoofing**: a peer presents any `node` id, ip, and GPU count in `hello`
  (`_daemon.py:425`); membership, `ray status`, and `ray.nodes()` all report it.
- **Tampering / elevation of privilege**: `create_actor` with a pickled class
  reaches `_host_actor` on a chosen node and is executed there
  (`_daemon.py:639-706`, `_daemon.py:951-1058`, `_worker.py:43-45`). The attacker
  chooses the node and GPU, so a rogue actor can be co-scheduled with the real
  engine and steal its NCCL rank.
- **Information disclosure**: `get` on any known/guessed object id; `status` and
  `pg_table` and `resources` enumerate the whole cluster
  (`_daemon.py:519`, `_daemon.py:605`, `_daemon.py:626`).
- **Denial of service**: `create_pg` for every free GPU, then `kill`; or a
  `hello` that supersedes a live node and triggers its release
  (`_daemon.py:429-445`) — the head releases the real node's actors and PGs.
- **Repudiation**: none available; there is no identity to record (see R8).

### B1 / local driver → daemon unix socket

- Same handler set as B3, since `_on_conn` installs the same handler for every
  connection, unix or TCP (`_daemon.py:377-380`). Any process that can open the
  socket is a full client.
- `kill` any actor id, including one created by another driver (`on_kill`
  `_daemon.py:1155`, ids come straight from the request).
- `get`/`stat` any object id; ids are sequential per node (`_next_obj`
  `_daemon.py:349`), so guessing is trivial.

### B4 / daemon → actor subprocess

- **Deserialization of attacker-chosen bytes.** `cloudpickle.loads` on both the
  init payload and every call payload (`_worker.py:44,48`) is arbitrary code
  execution by construction. `_proto.loads` falls back to stdlib `pickle`
  (`_proto.py:18-30`), which is the same power with no cloudpickle extension.
- `getattr(instance, header["method"])` (`_worker.py:49`) — attribute access is
  unfiltered beyond the `__`-prefixed names blocked client-side in
  `ActorHandle.__getattr__` (`__init__.py:184-187`); the *worker* enforces no
  allowlist, so any attribute reachable on the instance is callable from the wire.
- Errors return `traceback.format_exc()` to the caller (`_worker.py:52-55`), which
  discloses filesystem paths and internals of the serving node.

### B8 / local → runtime files

- `daemon.json` is written with the process umask and never `chmod`-ed
  (`_cli.py:180-190`). If it is group/world-writable, any local user can rewrite
  the `pid` field and have `ray stop` SIGTERM/SIGKILL an arbitrary process
  (`_cli.py:726`, `_cli.py:745`).
- `ray stop`'s seize logic trusts the `stopping`/`pid` fields of a document it
  just read from disk (`_cli.py:490-660`), so a forged document can make a stop
  proceed against a process it does not own.

### Denial-of-service exposure

- **Amplification**: a single 4-byte length prefix can make one peer make the
  daemon buffer up to 512 MiB (`_proto.py:49-51`, `_daemon.py:91-93`). Reusing the
  connection repeats it. There is no per-peer byte budget.
- **Unbounded state**: `self.objects` (`_daemon.py:325`) grows per `put` and per
  completed `call` and is never evicted; `self.pgs`, `self.nodes`, `self.actors`
  grow the same way. The only ceilings in the code are the per-frame size and the
  RPC timeouts (`_daemon.py:30-38`).
- **Slow client**: `Peer.serve` reads frames in a loop (`_daemon.py:136-152`) with
  no idle timeout, and each request spawns a task (`_daemon.py:143`) — connection
  count is the only bound on in-flight work, and it is unbounded.
- **`hello` supersede as a DoS lever**: the re-hello path closes a live node's
  connection and releases its resources, so a spoofed `hello` is destructive even
  without ever creating an actor.

## 5. Mitigations mapping

Present in the code:

| Control | Covers | Reference |
|---------|--------|-----------|
| Frame header/payload size ceiling (512 MiB) | corrupt-length allocation bomb | `_proto.py:17,47-61`, `_daemon.py:89-101` |
| Header must be a JSON object; `plen` validated and non-negative | trivial framing corruption | `_proto.py:54-58`, `_daemon.py:96-101` |
| Fuzz coverage of the framing parser on random/structured garbage | framing robustness, CI-gated at 100% coverage | `tests/test_proto.py:1-13`, `.github/workflows/ci.yml` |
| Handler exceptions never kill the read loop; every failure returns `{"err": ...}` | one bad request cannot take the daemon down | `_daemon.py:166-172` |
| Unknown message types are rejected, not executed | unknown-op surface | `_daemon.py:407-411` |
| Every daemon→daemon RPC has a deadline (`_RPC_TIMEOUT`), kill escalates SIGTERM→SIGKILL | wedged node cannot stall cleanup forever | `_daemon.py:29`, `_daemon.py:241-262`, `_daemon.py:1169-1177` |
| Runtime-dir mutual exclusion via `link`/rename-seize, with live-pid probing | two daemons cannot steal each other's socket or pidfile | `_cli.py:314-392`, `_cli.py:395-411` |
| `ray start` refuses to start when a live daemon owns the claim | accidental double-start clobbering | `_cli.py:157-161` |
| CLI arg validation: unknown flags exit 2, `--port` must be numeric and in 0-65535, `--num-gpus >= 0`, `--address` port must be numeric and in 0-65535 | CLI input validation | `_cli.py:180-250` |
| Placement-group bundle and `ngpu` bounds are validated before use | nonsense placement requests | `_daemon.py:557-596`, `_daemon.py:693-706`, `_daemon.py:924-949` |
| Bounce-back detection (`p is peer`) stops routing loops and leaks `actor_loc` on dead owners | routing-loop DoS | `_daemon.py:1121-1123`, `_daemon.py:1163-1165` |
| Orphan reaper frees GPUs after a disconnected driver | resource leak (availability, not security) | `_daemon.py:900-919`, `_daemon.py:1309-1420` |
| Documented operator guidance: trusted private network only, firewall the port, `--node-ip` | R2/R1, **procedural only** | `README.md:180-189`, `docs/OPERATIONS.md:146-152` |
| Docker guidance: no `--privileged`, `--device /dev/infiniband` + `--cap-add IPC_LOCK` instead | container privilege escalation | `docs/OPERATIONS.md:64-79` |

Absent (ranked by exploitability × impact):

1. **No authentication or authorization on any boundary** (R1). Nothing checks
   who a peer is; `handle` dispatches purely on the message type string.
2. **No way to bind the head to a specific interface** (R2): `serve_tcp` is always
   called with `"0.0.0.0"` (`_cli.py:225`), so `--node-ip` changes what is
   advertised but not what is exposed.
3. **No rate limit, connection limit, quota, or per-peer byte accounting**
   (R3): `_on_conn` accepts every connection; `Peer.serve` spawns a task per
   frame.
4. **No object eviction or store cap** (R4): `on_put` stores indefinitely, and
   `on_get` explicitly keeps the slot for repeat `get`s.
5. **No unix-socket permission hardening or peer-credential check** (R6): no
   `os.chmod` on the socket, no `SO_PEERCRED`/uid check on accept.
6. **No environment minimization for actors** (R5): full `os.environ` copy.
7. **No audit log** (R8): stdout/stderr prints only, no actor/driver attribution.
8. **No TLS anywhere**, on the head link or the unix sockets.

Single points of failure worth naming:

- The **head daemon** is the only membership, placement, and routing authority
  (`_daemon.py:296-334`); it holds all state in RAM, so a restart loses every
  placement group, actor id, and object with no recovery path.
- The **`_on_conn` handler assignment** (`_daemon.py:377-380`) is the single place
  every connection — unix or TCP — becomes a fully privileged client. One missing
  check there is one missing check everywhere.
- The **frame length ceiling** (512 MiB) is the only resource bound on the wire,
  and it is shared by all three link types.

## 6. Abuse cases

- **Quota bypass / GPU squatting.** A peer with no authorization sends
  `create_pg` for every free GPU, then `create_actor` per bundle. The head's
  allocator (`on_create_pg`, `_daemon.py:557-596`) has no per-requester accounting:
  there is no requester identity to account against. When the peer disconnects,
  `release_client` (`_daemon.py:1309`) kills only the ids it created, so the
  legitimate vLLM driver simply finds "placement group needs more GPUs than the
  cluster has free" (`_daemon.py:587`).
- **Cross-driver object read.** `on_get`/`on_stat` (`_daemon.py:1266`, `:1292`)
  route by the owner node embedded in the object id. Ids are
  `<node_id>-o<seq>` with a monotonic counter (`_next_obj`, `_daemon.py:349`), so
  any client that can speak the protocol can enumerate and read every object the
  daemons hold.
- **Cross-driver actor kill.** `on_kill` takes the id straight from the request
  (`_daemon.py:1156`); ownership (`peer.created_actors`) is used only for cleanup,
  never for authorization.
- **Membership spoof to steer traffic.** `hello` with a victim `node` id
  supersedes the live record (`_daemon.py:429-445`), releasing that node's actors
  and PGs and installing the attacker's connection as the node's peer — later
  `call`/`kill` traffic for actors on that node is routed to the attacker.
- **Trust placed in client-side enforcement.** The only name filter on method
  dispatch is in the shim (`ActorHandle.__getattr__` refuses `__`-prefixed names,
  `__init__.py:184-187`); the worker resolves `getattr(instance, header["method"])`
  with no equivalent check (`_worker.py:49`). The daemon likewise trusts
  `m["actor"]`, `m["pg"]`, `m["obj"]` for routing.
- **Resource exhaustion.** A peer sends `put` with 512 MiB payloads and never
  calls `get`; nothing ever frees them (R4).
- **Crash-disclosure harvesting.** `on_worker_hello`/error paths return
  `traceback.format_exc()` verbatim (`_worker.py:52-55`, `_daemon.py:1175-1177`),
  which is a cheap way to fingerprint node internals from the unauthenticated port.

None of these were executed; they are read off the code paths named above.

## 7. Documentation accuracy

Claims checked against the code:

| Claim | Where | Verdict |
|-------|-------|---------|
| "control plane is unauthenticated … Anyone who can reach the port can run code as the daemon user" | `README.md:180-189` | **Accurate** |
| "the daemon never unpickles payloads; only the shim and the actor worker do" | `docs/DEVELOPMENT.md:127`, `docs/PROTOCOL.md:18` | **Accurate as stated** (no `loads` in `_daemon.py`) but misleading about exposure: the daemon *stores* client payloads verbatim in RAM (`on_put`, `_daemon.py:1258-1264`) and forwards them to every node. Corrected in `docs/DEVELOPMENT.md` and `docs/PROTOCOL.md`. |
| "beam's head binds `0.0.0.0`" | `README.md:183` | **Accurate**, and no flag changes it |
| "run it only on a trusted, private network … behind your firewall/security group" | `README.md:185-189`, `docs/OPERATIONS.md:146-152` | **Accurate**; this is the only mitigation in the chain |
| "no `--privileged`" | `docs/OPERATIONS.md:64-79,321` | **Accurate** for the documented recipes; `--security-opt seccomp=unconfined` is required for ROCm (`docs/OPERATIONS.md:194`) and weakens that claim on AMD nodes |

Not documented anywhere: the actor environment inheritance (`_daemon.py:1071`),
the `hello` supersede behaviour (`_daemon.py:429`), the object store retention
(`_daemon.py:1280`), the 512 MiB frame ceiling, and the runtime-file trust in
`ray stop`. See `SECURITY.md` for the disclosure path, which this repository did
not previously have.

## 8. Response readiness

- **Audit trail**: none beyond stdout. `beamd` prints only its startup lines
  (`_cli.py:235-241`, `_cli.py:247`) and tracebacks (`_daemon.py:157-163`); worker
  errors go to the actor's stderr (`_worker.py:55`). There is no record of which
  peer issued a `kill`, because no peer is identified.
- **Disclosure path**: `SECURITY.md` (added 2026-09-30) names the reporting
  channel. No response SLA is claimed, and none should be invented.
- **Version support**: beam tracks a single line, the newest release tag;
  `SECURITY.md` states it and that fixes are not backported to older tags. The
  version in `python/pyproject.toml` is pinned to the Ray release whose
  executor API this shim implements and is deliberately not the beam release
  number: quote the tag in a disclosure. See docs/RELEASING.md.