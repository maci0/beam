#!/usr/bin/env python3
"""Check that beam's release contract holds: one version number, one tag, one
changelog entry.

Three things can drift apart in this tree and all three have, or would:

- the distribution version, declared in `python/pyproject.toml` and again in
  `python/ray/__init__.py` (`__version__`);
- the tag a release is cut from (`vX.Y.Z`), which is beam's own line and the
  only version that says anything about beam's compatibility;
- `CHANGELOG.md`, which is what a consumer reads before upgrading.

`make check` runs this with no arguments (cheap consistency check, no git
required). The release workflow runs it with `--tag "$GITHUB_REF_NAME"`, which
additionally requires a dated changelog section for the tag, and with
`--notes FILE` to publish that section as the release body, and with `--dist`
to assert the built artifacts are the ones the manifest declares.

Usage:

    python scripts/check_release.py                   # consistency of the tree
    python scripts/check_release.py --tag v0.2.0      # + the tag is releasable
    python scripts/check_release.py --tag v0.2.0 --notes /tmp/body.md

Exit code is non-zero when a check fails, so it works as a CI gate.
"""

from __future__ import annotations  # keep PEP604 annotations valid on py3.9

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "python", "pyproject.toml")
SHIM_INIT = os.path.join(ROOT, "python", "ray", "__init__.py")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")
DOCS = [os.path.join(ROOT, "docs", name) for name in ("API.md", "RELEASING.md")]

# `version = "..."` in [project]; the manifest is the single source of truth and
# __init__ copies it.
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)
_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
# A released section heading: `## [0.2.0] - 2026-08-26`. `Unreleased` is matched
# separately and never counts as a released version.
_SECTION_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - ", re.M)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def manifest_version() -> str:
    match = _VERSION_RE.search(_read(MANIFEST))
    if match is None:
        sys.exit("beam: no [project] version in %s" % os.path.relpath(MANIFEST, ROOT))
    return match.group(1)


def shim_version() -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', _read(SHIM_INIT), re.M)
    if match is None:
        sys.exit("beam: no __version__ in %s" % os.path.relpath(SHIM_INIT, ROOT))
    return match.group(1)


def changelog_versions() -> list[str]:
    return _SECTION_RE.findall(_read(CHANGELOG))


def release_notes(version: str) -> str:
    """The CHANGELOG section for `version`, verbatim, without the link footer.

    This is what the release workflow publishes as the GitHub release body: a
    consumer reads the same prose the tree holds, not GitHub's auto-generated
    commit list, which names commits and says nothing about what breaks.
    """
    text = _read(CHANGELOG)
    start = text.index("## [%s] - " % version)
    end = text.find("\n## [", start + 1)
    section = text[start:] if end == -1 else text[start:end]
    # The trailing compare-links block belongs on the website, not in a release
    # body; cut it at the first bare link line at column 0.
    return section.split("\n[")[0].rstrip() + "\n"


def check_tree(errors: list[str]) -> None:
    """Everything checkable without a tag: the two version declarations agree,
    and every doc that quotes the version quotes the current one."""
    manifest, shim = manifest_version(), shim_version()
    if manifest != shim:
        errors.append(
            "%s says %r but %s says %r; the manifest is the source of truth"
            % (
                os.path.relpath(MANIFEST, ROOT),
                manifest,
                os.path.relpath(SHIM_INIT, ROOT),
                shim,
            )
        )
    for path in DOCS:
        if not os.path.exists(path):
            # A doc this checker expects to quote the version has gone away.
            # Report it rather than raising: the check is a gate, and a gate that
            # tracebacks is a gate that gets skipped.
            errors.append("checked doc %s is missing" % os.path.relpath(path, ROOT))
            continue
        # Only a mention of the *distribution* version counts. A doc is full of
        # other dotted numbers that are not this package's version: `0.0.0.0` is
        # a bind address, and `0.3.0` in RELEASING.md is a hypothetical beam
        # tag. Match the two forms a doc actually declares the version in, and
        # let tag examples through untouched.
        quoted = set(
            re.findall(r'version\s*=\s*"(\d+\.\d+\.\d+)"', _read(path))
            + re.findall(r"reports\s+`\"(\d+\.\d+\.\d+)\"`", _read(path))
        )
        stale = sorted(v for v in quoted if v != manifest)
        if stale:
            errors.append(
                "%s quotes version(s) %s but the manifest says %s"
                % (os.path.relpath(path, ROOT), ", ".join(stale), manifest)
            )
    if not os.path.exists(CHANGELOG):
        errors.append("CHANGELOG.md is missing; a release ships with nothing to read")
    elif not changelog_versions():
        errors.append("CHANGELOG.md has no released version sections")


def check_tag(tag: str, errors: list[str]) -> None:
    """A tag is beam's own line (vX.Y.Z); it is not the distribution version,
    which is pinned to a Ray release. It must be well-formed, have a dated
    changelog section, and not repeat a section another tag already claimed."""
    if not _TAG_RE.match(tag):
        errors.append("tag %r is not vMAJOR.MINOR.PATCH" % tag)
        return
    bare = tag[1:]
    released = changelog_versions()
    if bare not in released:
        errors.append("CHANGELOG.md has no `## [%s] - <date>` section for this release" % bare)
    dupe = released.count(bare)
    if dupe > 1:
        errors.append("CHANGELOG.md has %d sections for %s" % (dupe, bare))


def check_dist(dist: str, errors: list[str]) -> None:
    """The built artifacts are the ones the manifest declares.

    A wheel is named after the distribution version, so a stale
    `ray-2.42.0-*.whl` left in `dist/` by an earlier build would be swept into
    the release by the `dist/*.whl` glob and published next to the current one.
    Exactly one wheel and one sdist, both carrying the declared version, must
    be there.
    """
    version = manifest_version()
    if not os.path.isdir(dist):
        errors.append("dist directory %s does not exist" % dist)
        return
    names = sorted(os.listdir(dist))
    for kind, suffix in (("wheel", ".whl"), ("sdist", ".tar.gz")):
        found = [n for n in names if n.endswith(suffix)]
        expected = "ray-%s" % version
        if len(found) != 1:
            errors.append(
                "%s holds %d %s(s) (%s), expected exactly one"
                % (dist, len(found), kind, ", ".join(found) or "none")
            )
        elif not found[0].startswith(expected):
            errors.append(
                "%s/%s is not %s-*, so it was built from another version"
                % (dist, found[0], expected)
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="the tag being released, e.g. v0.2.0")
    parser.add_argument(
        "--notes",
        metavar="FILE",
        help="with --tag: write that release's CHANGELOG section to FILE",
    )
    parser.add_argument(
        "--dist",
        metavar="DIR",
        help="also require DIR to hold exactly one wheel and one sdist at the "
        "manifest version (run after `uv build`)",
    )
    args = parser.parse_args(argv)
    if args.notes and not args.tag:
        parser.error("--notes requires --tag")

    errors: list[str] = []
    check_tree(errors)
    if args.tag:
        check_tag(args.tag, errors)
    if args.dist:
        check_dist(args.dist, errors)
    for e in errors:
        sys.stderr.write("beam: %s\n" % e)
    if errors:
        return 1
    if args.notes:
        with open(args.notes, "w", encoding="utf-8") as f:
            f.write(release_notes(args.tag[1:]))
    print(
        "check_release: ok (%s, %d changelog entries)"
        % (manifest_version(), len(changelog_versions()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
