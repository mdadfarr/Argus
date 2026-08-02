from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

STATE = Path(__file__).parent / "state"
OUTBOX_PATH = STATE / "outbox.jsonl"


def add(entry: dict) -> None:
    """entry: {"date": "YYYY-MM-DD", "task": {...}}. Durable — fsynced before return."""
    STATE.mkdir(exist_ok=True)
    with open(OUTBOX_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_all() -> list[dict]:
    if not OUTBOX_PATH.exists():
        return []
    lines = OUTBOX_PATH.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _rewrite(entries: list[dict]) -> None:
    tmp = OUTBOX_PATH.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUTBOX_PATH)


def remove(task_id: str) -> None:
    _rewrite([e for e in read_all() if e["task"]["id"] != task_id])


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
        task_id = entry["task"]["id"]
        try:
            sender(entry["date"], entry["task"])
        except Exception as e:
            if not is_permanent(e):
                log.warning("outbox drain stopped at %s: %s", task_id, e)
                return
            log.error("dropping permanently rejected outbox entry %s: %s", task_id, e)
        remove(task_id)
