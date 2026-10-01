"""The daemon.json runtime document: writing, claiming, and liveness probing.

The document is the one place a daemon records the socket path, its node id and
its pid, and it is what `ray start` claims exclusively, `ray status` reads, and
`ray stop` takes over. Every one of those steps is the same kind of work (a
temp file linked into place so a reader never sees a partial document) and the
same failure mode (the file is missing, truncated, or owned by another
process), so they live here together. This is the in-tree entry point for
reading, replacing, claiming, and probing a claim: ``ray._config`` supplies the
read and the paths, and callers reach it through this module rather than
opening daemon.json themselves.
"""

from __future__ import annotations

import json
import os

from . import _config


def read(path: str | None = None) -> dict:
    """A runtime document, or {} when it is missing or unreadable.

    `path` defaults to the live daemon.json; pass it to read a renamed copy (a
    seized claim, a stop hold) the same way, so no caller re-implements the
    parse or the "unreadable means empty" rule.
    """
    return _config.read_runtime_doc(path)


def write(rt: dict) -> None:
    """Write daemon.json via temp+replace so readers never see a partial file."""
    path = _config.runtime_json_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(rt, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def link_restore(seized: str, path: str) -> bool:
    """Restore seized claim to path only if path is free (never clobber).

    Returns True if restored, False if path is already owned (seized dropped)
    or restore failed. On non-FileExistsError link failures, leave seized on
    disk so the last copy of a live claim is not destroyed.
    """
    try:
        os.link(seized, path)
        try:
            os.unlink(seized)
        except OSError:
            pass
        return True
    except FileExistsError:
        # path already has a claim: drop our seized copy only
        try:
            os.unlink(seized)
        except OSError:
            pass
        return False
    except OSError:
        # path still free but link failed (EPERM/ENOSPC/...): keep seized
        return False


def _held_by(pid: int) -> bool:
    """Whether a pid names a process that still exists.

    A pid we cannot signal (EPERM) counts as live: unlinking a live daemon's
    socket out from under it is worse than refusing to reclaim a stale one.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, TypeError):
        return True
    return True


def live_daemon_pid() -> int | None:
    """Return the pid of a still-running local daemon, or None if none/stale."""
    try:
        pid = int(read().get("pid") or 0)
    except (ValueError, TypeError):
        return None  # a pid we cannot parse names no process
    if pid <= 0:
        return None  # no pid, or one no process can have
    return pid if _held_by(pid) else None


def claim(rt: dict) -> str | None:
    """Atomically publish a complete daemon.json for this process.

    Writes the full document to a temp file first, then links it into place so
    another start never sees an empty/partial claim and steals the runtime.
    Stale pidfiles are rename-seized (not remove-then-link) to avoid a window
    where a concurrent claim is deleted out from under a live daemon.
    Returns an error message if another live daemon owns the runtime, else None.
    """
    path = _config.runtime_json_path()
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(rt, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        for _ in range(2):
            try:
                os.link(tmp, path)  # fails if path already exists
                return None
            except FileExistsError:
                live = live_daemon_pid()
                if live is not None:
                    return "beam: daemon already running (pid %d). Run 'ray stop' first.\n" % live
                # Seize the existing path atomically, confirm it is still dead
                # (or not an in-progress stop hold), then discard and retry.
                seized = path + ".stale.%d" % os.getpid()
                try:
                    os.rename(path, seized)
                except OSError:
                    continue  # raced with another claim/stop; retry link
                old = read(seized)
                try:
                    old_pid = int(old.get("pid") or 0)
                except (ValueError, TypeError):
                    old_pid = 0
                # Both a live daemon and an in-progress `ray stop` (which
                # holds the path with the stop process pid) own this claim.
                still_live = _held_by(old_pid) if old_pid > 0 else False
                if still_live:
                    link_restore(seized, path)
                    return (
                        "beam: daemon already running (pid %d). Run 'ray stop' first.\n" % old_pid
                    )
                try:
                    os.remove(seized)
                except OSError:
                    pass
                continue
        return "beam: cannot claim runtime dir %s\n" % path
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
