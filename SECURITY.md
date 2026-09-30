# Security policy

## The short version

beam's control plane is **unauthenticated and unencrypted**, by design and by
choice of scope: it is meant to run on a trusted cluster subnet alongside a
vLLM deployment, not on the internet. Anyone who can reach a daemon socket can
create actors, and an actor carries a cloudpickled payload that is unpickled and
executed on a node. The full model, with the code paths behind each claim, is in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

Concretely, before deploying:

- Keep the head's TCP port (default 6379) on a private network. The head always
  binds `0.0.0.0`; there is no flag to narrow it, so the firewall or security
  group is the control.
- Set `--node-ip` / `BEAM_NODE_IP` per node to the address the cluster actually
  uses.
- Do not put secrets in the daemon environment that the actor processes must not
  see: every actor subprocess inherits the daemon's full environment
  (`_daemon.py:1071`).

## Supported versions

There is one supported line: the current release on `main`
(`python/pyproject.toml`). Fixes are not backported to older tags.

| Version | Supported |
|---------|-----------|
| current release | yes |
| older tags | no |

## Reporting a vulnerability

Report it privately to the maintainer at `maci.stgn@gmail.com` (the address on
this repository's commits) rather than opening a public issue. Include the beam
version, the topology involved (head/worker/driver placement), and the file or
message type you believe is at fault.

There is no response-time commitment recorded for this project. Do not rely on
one; if that matters to you, agree it with the maintainer before disclosing.

## What is out of scope

- Exploiting a cluster you do not administer. beam gives no protection once an
  attacker is on the control port, so "the port was reachable" is a deployment
  finding, not a beam vulnerability.
- Denial of service against a cluster the reporter does not own.
- Weaknesses in vLLM, PyTorch, NCCL, cloudpickle, or the base container image;
  report those upstream. beam's own scope is the `python/ray` package, the
  `ray`/`beam` CLI, and the `Dockerfile`.