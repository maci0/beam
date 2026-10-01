"""Unit tests for the release-contract checker (`scripts/check_release.py`).

The checker is what keeps the version numbers, the docs that quote them, and
CHANGELOG.md from drifting into three different answers. These tests pin each
check against a scratch tree, so a change that weakens one of them fails here
rather than in a release that shipped the wrong number.
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import check_release as cr  # noqa: E402


def _load_tree(tmp_path, manifest_version="2.43.0", shim_version="2.43.0", changelog=None):
    """A miniature repo with the same files the checker reads."""
    (tmp_path / "python").mkdir(parents=True, exist_ok=True)
    (tmp_path / "python" / "pyproject.toml").write_text(
        '[project]\nname = "ray"\nversion = "%s"\n' % manifest_version
    )
    (tmp_path / "python" / "ray").mkdir(exist_ok=True)
    (tmp_path / "python" / "ray" / "__init__.py").write_text(
        'from __future__ import annotations\n\n__version__ = "%s"\n' % shim_version
    )
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "API.md").write_text("")
    if changelog is not None:
        (tmp_path / "CHANGELOG.md").write_text(changelog)
    return tmp_path


def _patched_root(tree, monkeypatch):
    monkeypatch.setattr(cr, "ROOT", str(tree))
    monkeypatch.setattr(cr, "MANIFEST", str(tree / "python" / "pyproject.toml"))
    monkeypatch.setattr(cr, "SHIM_INIT", str(tree / "python" / "ray" / "__init__.py"))
    monkeypatch.setattr(cr, "CHANGELOG", str(tree / "CHANGELOG.md"))
    monkeypatch.setattr(cr, "DOCS", [str(tree / "docs" / "API.md")])


def test_real_tree_is_consistent():
    """The checked-in tree passes its own check, no monkeypatching."""
    errors = []
    cr.check_tree(errors)
    assert errors == []


def test_version_mismatch_is_reported(tmp_path, monkeypatch):
    _patched_root(
        _load_tree(
            tmp_path,
            manifest_version="2.44.0",
            changelog="## [0.1.0] - 2026-06-25\n\n- first release\n",
        ),
        monkeypatch,
    )
    errors = []
    cr.check_tree(errors)
    assert any("source of truth" in e for e in errors)


def test_stale_quoted_version_is_reported(tmp_path, monkeypatch):
    tree = _load_tree(tmp_path, changelog="## [0.1.0] - 2026-06-25\n\n- first release\n")
    # The form docs actually use: a backticked cell containing a quoted version.
    (tree / "docs" / "API.md").write_text('| `ray.__version__` | reports `"2.42.0"` |\n')
    _patched_root(tree, monkeypatch)
    errors = []
    cr.check_tree(errors)
    assert any("2.42.0" in e for e in errors), errors


def test_bind_address_is_not_mistaken_for_a_version(tmp_path, monkeypatch):
    """`0.0.0.0` in prose is an address, not a release; it must not trip the check."""
    tree = _load_tree(tmp_path, changelog="## [0.1.0] - 2026-06-25\n\n- first release\n")
    (tree / "docs" / "API.md").write_text("binds 0.0.0.0 by default\n")
    _patched_root(tree, monkeypatch)
    errors = []
    cr.check_tree(errors)
    assert errors == [], errors


def test_hypothetical_tag_examples_are_not_versions(tmp_path, monkeypatch):
    """RELEASING.md quotes 0.3.0 as an example tag; tags and the pinned
    distribution version are different lines, so that must not read as stale."""
    tree = _load_tree(tmp_path, changelog="## [0.1.0] - 2026-06-25\n\n- first release\n")
    (tree / "docs" / "API.md").write_text("cut 0.3.0 next; 0.0.0.0 is the bind address\n")
    _patched_root(tree, monkeypatch)
    errors = []
    cr.check_tree(errors)
    assert errors == [], errors


def test_missing_changelog_is_reported(tmp_path, monkeypatch):
    _patched_root(_load_tree(tmp_path), monkeypatch)
    errors = []
    cr.check_tree(errors)
    assert any("CHANGELOG.md is missing" in e for e in errors)


def test_tag_needs_a_dated_section(tmp_path, monkeypatch):
    tree = _load_tree(tmp_path, changelog="# Changelog\n\n## [Unreleased]\n\n- x\n")
    _patched_root(tree, monkeypatch)
    errors = []
    cr.check_tag("v0.2.0", errors)
    assert errors == ["CHANGELOG.md has no `## [0.2.0] - <date>` section for this release"]


def test_malformed_tag_is_rejected(tmp_path, monkeypatch):
    tree = _load_tree(tmp_path, changelog="## [0.2.0] - 2026-08-26\n")
    _patched_root(tree, monkeypatch)
    for tag in ("2.43.0", "v0.2", "v0.2.0-rc1", "latest"):
        errors = []
        cr.check_tag(tag, errors)
        assert errors == ["tag %r is not vMAJOR.MINOR.PATCH" % tag]


def test_duplicate_section_is_rejected(tmp_path, monkeypatch):
    body = "## [0.2.0] - 2026-08-26\n\n- one\n\n## [0.2.0] - 2026-08-27\n\n- two\n"
    tree = _load_tree(tmp_path, changelog=body)
    _patched_root(tree, monkeypatch)
    errors = []
    cr.check_tag("v0.2.0", errors)
    assert errors == ["CHANGELOG.md has 2 sections for 0.2.0"]


def test_release_notes_is_one_section_without_link_footer(tmp_path, monkeypatch):
    body = (
        "# Changelog\n\n## [Unreleased]\n\n- new\n\n## [0.2.0] - 2026-08-26\n\n"
        "- fixed a thing\n\n## [0.1.0] - 2026-06-25\n\n- first\n\n"
        "[0.2.0]: https://example/compare/v0.1.0...v0.2.0\n"
        "[0.1.0]: https://example/releases/tag/v0.1.0\n"
    )
    tree = _load_tree(tmp_path, changelog=body)
    _patched_root(tree, monkeypatch)
    notes = cr.release_notes("0.2.0")
    assert notes.startswith("## [0.2.0] - 2026-08-26")
    assert "- fixed a thing" in notes
    assert "- new" not in notes, "Unreleased must not leak into a released note"
    assert "- first" not in notes, "an older section must not leak in"
    assert "https://example" not in notes, "the compare-link footer is not release prose"


def test_dist_requires_exactly_one_wheel_and_sdist(tmp_path, monkeypatch):
    tree = _load_tree(tmp_path, changelog="## [0.1.0] - 2026-06-25\n\n- first release\n")
    _patched_root(tree, monkeypatch)
    dist = tmp_path / "dist"
    dist.mkdir()

    empty = []
    cr.check_dist(str(dist), empty)
    assert len(empty) == 2, "an empty dist is a failed build, not a pass"

    (dist / "ray-2.43.0-py3-none-any.whl").write_bytes(b"")
    (dist / "ray-2.43.0.tar.gz").write_bytes(b"")
    good = []
    cr.check_dist(str(dist), good)
    assert good == []

    (dist / "ray-2.42.0-py3-none-any.whl").write_bytes(b"")
    stale = []
    cr.check_dist(str(dist), stale)
    assert any("holds 2 wheel" in e for e in stale), stale

    (dist / "ray-2.42.0-py3-none-any.whl").unlink()
    (dist / "ray-2.43.0.tar.gz").rename(dist / "ray-2.42.0.tar.gz")
    wrong_version = []
    cr.check_dist(str(dist), wrong_version)
    assert any("built from another version" in e for e in wrong_version), wrong_version


def test_dist_missing_directory_is_reported(tmp_path, monkeypatch):
    tree = _load_tree(tmp_path, changelog="## [0.1.0] - 2026-06-25\n\n- first release\n")
    _patched_root(tree, monkeypatch)
    errors = []
    cr.check_dist(str(tmp_path / "nope"), errors)
    assert errors == ["dist directory %s does not exist" % (tmp_path / "nope")]


def test_main_returns_zero_on_the_real_tree():
    assert cr.main([]) == 0


def test_main_rejects_notes_without_tag():
    try:
        cr.main(["--notes", os.path.join(ROOT, "unused.md")])
    except SystemExit as e:
        assert e.code == 2, "argparse usage error"
        return
    raise AssertionError("--notes without --tag should exit 2")


def test_module_is_importable_as_a_file(tmp_path):
    """The workflow runs `python3 scripts/check_release.py`, not an import."""
    spec = importlib.util.spec_from_file_location(
        "check_release_as_file", os.path.join(ROOT, "scripts", "check_release.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.manifest_version() == cr.manifest_version()
