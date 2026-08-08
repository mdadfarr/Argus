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
# 408 Request Timeout and 429 Too Many Requests are the two 4xx that retrying
# genuinely fixes. Every other 4xx is a client-side mistake -- a wrong URL, a
# revoked key, a method the endpoint doesn't accept -- and no number of retries
# will change the answer.
RETRYABLE_STATUSES = (408, 429)


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

    def _backoff(attempt: int, why: str) -> None:
        # No sleep on the last attempt: the loop is about to end and raise, so
        # the final wait accomplishes nothing but delay.
        if attempt >= MAX_ATTEMPTS - 1:
            return
        wait = min(60, 1.5 ** attempt) + random.uniform(0, 0.5)
        log.warning(
            "calendar append attempt %d/%d %s (retry in %.1fs)",
            attempt + 1, MAX_ATTEMPTS, why, wait,
        )
        time.sleep(wait)

    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.post(append_url, json=body, headers=headers, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            _backoff(attempt, f"failed: {e}")
            continue

        if r.status_code >= 500 or r.status_code in RETRYABLE_STATUSES:
            _backoff(attempt, f"got HTTP {r.status_code}")
            continue

        if r.status_code >= 400:
            # Previously only 400/401 were caught here and everything else --
            # 403, 404, 405 -- fell through to r.json(), which raises a
            # JSONDecodeError on the HTML body those return. That is not a
            # CalendarClientError, so the caller filed it as transient and the
            # entry failed identically on every future drain, forever, blocking
            # every entry behind it.
            raise CalendarClientError(f"calendar rejected request: {r.status_code} {r.text[:200]}")

        try:
            payload = r.json()
        except ValueError as e:
            # A 2xx with a body we can't parse is a wrong endpoint, not a
            # network blip. Retrying it forever would block the whole queue.
            raise CalendarClientError(
                f"calendar returned HTTP {r.status_code} with an unreadable body: {e}"
            ) from e
        result = payload.get("result") if isinstance(payload, dict) else None
        if result == "duplicate":
            log.info("session %s already logged — idempotent no-op", task["id"])
        return

    raise RuntimeError("calendar append failed after retries — left in outbox")
