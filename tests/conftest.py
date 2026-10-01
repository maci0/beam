"""Replay profile for the property-based tests.

A run that can be replayed from one seed is the whole point; hypothesis
otherwise explores from its own example database, so two runs of the same
commit can differ and a flaky property failure moves instead of reproducing.
Setting BEAM_SEED (the same variable the daemon reads for node ids) loads the
`replay` profile: derandomized, no database, so the same commit explores the
same examples in the same order on every machine. `--hypothesis-seed=<int>`
still overrides for a one-off report, and CI leaves BEAM_SEED unset so the
database-backed search keeps finding new failures.
"""

import os


def pytest_configure(config):
    if not os.environ.get("BEAM_SEED"):
        return
    from hypothesis import settings as _settings

    _settings.register_profile(
        "replay",
        derandomize=True,
        database=None,
        print_blob=True,
    )
    _settings.load_profile("replay")
