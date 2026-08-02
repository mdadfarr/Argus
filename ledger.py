from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

STATE = Path(__file__).parent / "state"
LOGS = Path(__file__).parent / "logs"
SESSIONS_PATH = STATE / "sessions.jsonl"
FALSE_POSITIVES_PATH = STATE / "false_positives.jsonl"


def setup_logging(log_level: str = "INFO") -> None:
    LOGS.mkdir(exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOGS / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.addHandler(handler)


def _append(path: Path, record: dict) -> None:
    STATE.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def record_session(record: dict) -> None:
    """record never contains frame/image data — only the numeric measurements
    and outcome, so this file can be used to answer 'why did it fail me at
    19:03' without ever having stored a picture of you."""
    _append(SESSIONS_PATH, record)


def record_false_positive(session_id: str, frame_reading: dict, note: str = "") -> None:
    _append(FALSE_POSITIVES_PATH, {
        "session_id": session_id,
        "frame_reading": frame_reading,
        "note": note,
    })
