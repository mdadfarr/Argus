from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

STATE = Path(__file__).parent / "state"
OUTBOX_PATH = STATE / "outbox.jsonl"

# add() runs on the tick thread; remove()/drain() run on the startup-sync and
# calendar-send threads. _rewrite() builds a temp file from a read_all()
# snapshot and os.replace()s it over the outbox, so a session that finished
# between the two had its entry written to the file that was about to be
# replaced -- the record vanished while _handle_logging had already reported it
# as durable.
#
# Reentrant because remove() calls read_all(). Deliberately NOT held across
# drain()'s sender() call: that does network I/O with a retry ladder, and
# blocking add() behind it would stall the tick loop (add is called with
# Engine.lock held) for the length of an outage.
_LOCK = threading.RLock()


def add(entry: dict) -> None:
    """entry: {"date": "YYYY-MM-DD", "task": {...}}. Durable — fsynced before return."""
    with _LOCK:
        STATE.mkdir(exist_ok=True)
        with open(OUTBOX_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_all() -> list[dict]:
    """Malformed lines are quarantined and skipped, never raised.

    A single truncated line (a crash between write and fsync) used to make this
    raise on every later call, which bricked the whole outbox: drain() raised,
    remove() raised, and every future session's remove() raised out of
    _send_async. The file then had to be deleted by hand, with nothing said
    anywhere but a warning in logs/app.log.
    """
    with _LOCK:
        if not OUTBOX_PATH.exists():
            return []
        lines = OUTBOX_PATH.read_text().splitlines()
        entries = []
        bad = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                bad.append(line)
                continue
            if _task_id_of(entry) is None:
                bad.append(line)
                continue
            entries.append(entry)
        if bad:
            _quarantine(bad)
            # Drop them from the outbox itself, or every later read_all() would
            # re-quarantine the same lines.
            _rewrite(entries)
        return entries


def _task_id_of(entry) -> str | None:
    """None for anything that isn't a well-formed entry. A KeyError from an
    unvalidated entry["task"]["id"] bricked the outbox the same way a bad JSON
    line did."""
    if not isinstance(entry, dict):
        return None
    task = entry.get("task")
    if not isinstance(task, dict):
        return None
    task_id = task.get("id")
    return task_id if isinstance(task_id, str) else None


def _quarantine(lines: list[str]) -> None:
    """Move unparseable lines aside so read_all() can make progress, but keep
    them: they may be the only remaining record of a finished session."""
    path = OUTBOX_PATH.with_name("outbox.corrupt.jsonl")
    try:
        with open(path, "a") as f:
            for line in lines:
                f.write(line + "\n")
    except OSError as e:
        log.error("could not quarantine %d corrupt outbox line(s): %s", len(lines), e)
        return
    log.warning("quarantined %d corrupt outbox line(s) to %s", len(lines), path.name)


def _rewrite(entries: list[dict]) -> None:
    with _LOCK:
        tmp = OUTBOX_PATH.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, OUTBOX_PATH)


def remove(task_id: str) -> None:
    # Read and rewrite as one atomic step: re-reading under the lock is what
    # preserves an entry add()ed since the caller last looked.
    with _LOCK:
        _rewrite([e for e in read_all() if _task_id_of(e) != task_id])


def drain(sender, is_permanent=lambda exc: False) -> None:
    """sender(date, task) -> None, raising on failure.

    Drains in order and deletes each entry only after sender() returns
    successfully. Stops at the first *transient* failure rather than skipping
    ahead, so a network outage doesn't get silently swallowed for later
    entries — the caller just calls drain() again later.

    is_permanent(exc) marks a failure that retrying can never fix (bad auth,
    malformed request). Those entries are dropped and the drain continues:
    halting on them meant one poison entry blocked every later session from
    ever syncing, forever.
    """
    for entry in read_all():
        task_id = _task_id_of(entry)
        if task_id is None:          # read_all() already filters these out
            continue
        try:
            sender(entry["date"], entry["task"])
        except Exception as e:
            if not is_permanent(e):
                log.warning("outbox drain stopped at %s: %s", task_id, e)
                return
            log.error("dropping permanently rejected outbox entry %s: %s", task_id, e)
        remove(task_id)
