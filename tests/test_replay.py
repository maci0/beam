"""`BEAM_SEED` must make a failing run reproduce, because that is the promise a
single-seed deterministic run rests on. Each check runs a tiny hypothesis test
in a subprocess pytest run that loads this directory's conftest, the way CI or a
teammate reproducing a report would, then compares the examples visited."""

import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

PROBE = """
from hypothesis import given, settings, strategies as st


@given(st.integers())
def test_probe(x):
    # no per-test derandomize override: the loaded profile must supply it
    print("EXAMPLE", x)


def test_profile_state():
    from hypothesis import settings as _s

    print("PROFILE", _s.default.derandomize, _s.default.database is None)
"""


def _run(tmp_path, seed=None):
    e = dict(os.environ)
    e.pop("BEAM_SEED", None)
    e.pop("HYPOTHESIS_SEED", None)
    if seed:
        e["BEAM_SEED"] = seed
    e["PYTHONHASHSEED"] = "0"
    work = tmp_path / ("w_%s" % (seed or "none"))
    work.mkdir(exist_ok=True)
    # the conftest is what turns BEAM_SEED into the replay profile
    with open(os.path.join(HERE, "conftest.py")) as fh:
        (work / "conftest.py").write_text(fh.read())
    probe = work / "test_probe_mod.py"
    probe.write_text(textwrap.dedent(PROBE))
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "-p",
            "no:cacheprovider",
            str(probe),
        ],
        capture_output=True,
        text=True,
        env=e,
        cwd=str(work),
    )
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def _examples(out):
    """The example values visited, in order: the replay fingerprint."""
    return [ln for ln in out.splitlines() if ln.startswith("EXAMPLE")]


def test_same_seed_replays_identical_examples(tmp_path):
    first = _examples(_run(tmp_path, seed="same"))
    second = _examples(_run(tmp_path, seed="same"))
    assert first, first
    assert first == second, "replay diverged:\n%s\n---\n%s" % (first, second)


def test_replay_profile_is_derandomized(tmp_path):
    """BEAM_SEED loads the derandomized, database-less profile via conftest."""
    assert "PROFILE True True" in _run(tmp_path, seed="profile")


def test_without_seed_search_stays_random(tmp_path):
    """Unset, hypothesis keeps its own search: new failures are still found."""
    assert "PROFILE False False" in _run(tmp_path)
