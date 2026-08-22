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


def drain(sender) -> None:
    """sender(date, task) -> None, raising on failure.

    Drains in order and deletes each entry only after sender() returns
    successfully. Stops at the first failure rather than skipping ahead, so a
    persistent auth/network problem doesn't get silently swallowed for later
    entries — the caller (main.py, on a timer or at startup) just calls
    drain() again later.
    """
    for entry in read_all():
        try:
            sender(entry["date"], entry["task"])
        except Exception as e:
            log.warning("outbox drain stopped at %s: %s", entry["task"]["id"], e)
            return
        remove(entry["task"]["id"])
