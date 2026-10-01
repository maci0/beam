# Releasing beam

The release contract, in the order a maintainer needs it. Everything here is
enforced; nothing is a convention you have to remember.

## The two version numbers

beam publishes two, and they answer different questions:

| number | where | what it means |
|--------|-------|---------------|
| **tag**, `v0.2.0` | the git tag, `CHANGELOG.md` | beam's release line: what changed, and whether an upgrade is safe |
| **distribution version** | `python/pyproject.toml` (`version`), copied to `ray.__version__` | the Ray release whose distributed-executor API this shim implements |

The distribution version is pinned to a real Ray release on purpose: beam's
package is named `ray` so it shadows the real one, and vLLM reads
`ray.__version__` and `importlib.metadata.version("ray")` during startup. Raising
it to a real Ray release beam does not implement would make vLLM take code paths
this shim has no handler for. So it moves only when beam's surface changes to
track a newer Ray, and it never signals beam compatibility on its own. **The tag
is the version to quote when reporting a bug or pinning a deployment.**

`SECURITY.md` calls "the current release" the supported line, and the tag is
what that means. Fixes are not backported to older tags.

## Cutting a release

1. Write the changelog section. Move everything under `## [Unreleased]` into a
   new dated `## [X.Y.Z] - YYYY-MM-DD` heading, grouped Added / Changed /
   Fixed / Breaking. Write it for someone deciding whether to upgrade: what
   changed *for them*, not which commits landed. Empty `## [Unreleased]` is the
   signal that there is nothing to release.
2. Decide the bump from that section, under SemVer applied to beam's own surface
   — the `ray.*` Python API and the `ray` CLI, not the pinned Ray number:
   - **major** (`v1.0.0`) first release of the stable `ray` surface, or a removal
     or signature change with no replacement. vLLM imports these symbols, so a
     removal is a hard break for the one consumer that matters.
   - **minor**: additive. A new exported symbol, a new CLI flag, a new optional
     request field on the wire.
   - **patch**: a fix that does not change any signature, default, exit code, or
     wire field. Anything that changes what a caller *observes* is at least a
     minor.
   - **breaking, but still `0.x`**: before v1.0.0 the minor is the breaking
     bump. Say so under a `### Breaking` heading so nobody reads `0.3.0` as
     `0.2.x`-compatible.
3. Check the release is coherent before tagging:
   ```
   make release-check TAG=v0.3.0
   ```
   It fails if the tag is malformed, if `CHANGELOG.md` has no dated section for
   it, if the manifest and `ray.__version__` disagree, or if a doc quotes a
   version the manifest does not declare.
4. Commit and tag `vX.Y.Z`. Pushing the tag runs `.github/workflows/release.yml`,
   which re-runs the check, builds, verifies the artifacts, publishes, and then
   installs the wheel and imports it.
5. If step 4's last step fails, the artifact is public but does not import: say
   so on the release and cut a fixed tag. A published version is never
   re-tagged or re-uploaded over.

## What the workflow enforces

- The tag must be `vMAJOR.MINOR.PATCH` and must have a dated changelog section,
  checked before anything is built.
- `python/pyproject.toml`, `ray.__version__`, and the versions quoted in
  `docs/API.md` and `docs/RELEASING.md` must be one number.
- Exactly one wheel and one sdist must exist at the declared version, so a stale
  artifact from an earlier build cannot ride along in the release glob.
- The release body is this tag's own changelog section, not a generated commit
  list.
- After publishing, the wheel is installed into a clean environment and
  `examples/import_check.py` imports the whole surface. A failure there means the
  release is public and broken: yank and re-cut, never overwrite.

`make check` runs the same version/changelog check on every push and pull
request, so drift is caught long before a tag exists.

## Deprecated surface

There is none to migrate today: beam has never removed an exported symbol or a
CLI flag, and the symbols vLLM imports are the ones beam implements. If that
changes, the sequence is: mark it `@deprecated` in the docstring with the
replacement and the release that will remove it, keep it working for at least one
minor release, and add the `### Deprecated` heading to the changelog. Removal
lands in the release named in the deprecation, not the same one.