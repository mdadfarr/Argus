from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger(__name__)


def build_task(session, tz_name: str) -> tuple[str, dict]:
    """Build the (date, task) pair for the append endpoint. Filed under the
    session's *start* date in the configured timezone (fixes I-23). startTime/
    endTime are left null -- the live calendar's own UI never actually
    populates them (confirmed against the deployed app), so fighting an
    unused, unvalidated format isn't worth it; the time range is folded into
    the text instead."""
    started = datetime.fromisoformat(session.started_wall_iso)
    date = started.strftime("%Y-%m-%d")
    start_str = started.strftime("%H:%M")
    end_str = datetime.now(ZoneInfo(tz_name)).strftime("%H:%M")
    minutes = round(session.focus_elapsed_s / 60)
    text = f"{session.label} ({minutes}m focus, {start_str}-{end_str})"[:140]
    task = {"id": session.id, "text": text, "done": True, "startTime": None, "endTime": None}
    return date, task

MAX_ATTEMPTS = 5
TIMEOUT_S = 10


class CalendarClientError(Exception):
    """Non-retryable: bad auth or bad request shape. Retrying won't fix it —
    caller should surface this loudly rather than spin."""


def check_auth(api_url: str, api_key: str) -> tuple[bool, str]:
    """Startup preflight. Returns (ok, human_message).

    A 401 here means the server has no CALENDAR_API_KEY set, or it does not
    match the key in the Keychain. Catching that at launch beats discovering it
    25 minutes later when a finished session silently fails to log.
    """
    try:
        r = requests.get(api_url, headers={"x-api-key": api_key}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        return False, f"Calendar unreachable ({e.__class__.__name__}) — sessions will queue offline."
    if r.status_code == 401:
        return False, ("Calendar rejected the API key (401). Set CALENDAR_API_KEY in Vercel "
                       "to the same value stored in your Keychain, then redeploy.")
    if r.status_code >= 500:
        return False, f"Calendar server error ({r.status_code}) — sessions will queue."
    if r.status_code != 200:
        return False, f"Calendar returned HTTP {r.status_code}."
    return True, ""


def log_pomodoro(append_url: str, api_key: str, date: str, task: dict) -> None:
    """POST one task to the atomic append endpoint. Idempotent on task['id'] —
    the server dedupes, so a retry after a lost response can't duplicate.

    Raises CalendarClientError on 400/401 (won't retry).
    Raises RuntimeError if transient failures exhaust all retries.
    """
    headers = {"x-api-key": api_key, "content-type": "application/json"}
    body = {"date": date, "task": task}

    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.post(append_url, json=body, headers=headers, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            wait = min(60, 1.5 ** attempt) + random.uniform(0, 0.5)
            log.warning(
                "calendar append attempt %d/%d failed: %s (retry in %.1fs)",
                attempt + 1, MAX_ATTEMPTS, e, wait,
            )
            time.sleep(wait)
            continue

        if r.status_code in (400, 401):
            raise CalendarClientError(f"calendar rejected request: {r.status_code} {r.text[:200]}")

        if r.status_code >= 500:
            wait = min(60, 1.5 ** attempt) + random.uniform(0, 0.5)
            log.warning(
                "calendar append attempt %d/%d got HTTP %d (retry in %.1fs)",
                attempt + 1, MAX_ATTEMPTS, r.status_code, wait,
            )
            time.sleep(wait)
            continue

        result = r.json().get("result")
        if result == "duplicate":
            log.info("session %s already logged — idempotent no-op", task["id"])
        return

    raise RuntimeError("calendar append failed after retries — left in outbox")
