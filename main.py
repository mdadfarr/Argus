from __future__ import annotations
import base64
import fcntl
import io
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)

# ---------- single-instance guard (fixes I-15) -- before anything else ----------
_lock_fh = open(STATE / "instance.lock", "w")
try:
    fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit("Argus is already running. Refusing to start a second instance.")
_lock_fh.write(str(os.getpid()))
_lock_fh.flush()

import webview

import config as config_mod
import ledger
import outbox
import alarm as alarm_mod
import calendar_client
import vision
from timer import SessionTimer, State

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

SOUNDS_DIR = Path(__file__).parent / "sounds"
VIOLATION_SOUNDS = {
    "left": SOUNDS_DIR / "left_room.aiff",
    "look_down": SOUNDS_DIR / "looked_down.aiff",
    "phone": SOUNDS_DIR / "phone_detected.aiff",
}

HEADLINES = {
    State.IDLE: "idle",
    State.CALIBRATING: "calibrating",
    State.RUNNING: "focus",
    State.VIOLATION_GRACE: "violation",
    State.INTERRUPTED: "paused",
    State.CAMERA_ERROR: "camera error",
}

TERMINAL_UI_STATES = (State.LOGGING, State.FAILED, State.ABORTED)


class Engine:
    """Owns all backend state (timer, vision worker, alarm, sync). Runs its own
    tick loop on a background thread. Every method that touches shared state
    takes self.lock -- the JS bridge (Api) calls in from pywebview's own
    thread, so this replaces what used to be "only the Tk thread touches
    state"."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.RLock()
        self._api_key_cache: str | None = None
        self.latest_reading: vision.FrameReading | None = None
        self.status_message = ""
        self.thumbnail_b64: str | None = None
        self._syncing = True
        self._stopping = False

        self.reading_queue: queue.Queue = queue.Queue(maxsize=2)
        self.preview_queue: queue.Queue = queue.Queue(maxsize=1)
        self.worker = vision.VisionWorker(cfg, self.reading_queue, self.preview_queue)
        self.worker.start()

        self.alarm = alarm_mod.Alarm(cfg["alarm_sound_path"])

        self.timer = SessionTimer(
            cfg,
            on_camera_open=self._open_camera,
            on_camera_close=self._close_camera,
            on_calibrate_start=self._start_calibration,
            on_alarm_start=self._on_alarm_start,
            on_alarm_stop=self.alarm.stop,
            on_notify=self._notify,
        )
        if self.timer.last_outcome and self.timer.last_outcome.get("reason") == "crash_recovery":
            self.status_message = "Previous session was interrupted (crash/quit) — not logged."

        self._drain_outbox_async()

        self._tick_thread = threading.Thread(target=self._loop, daemon=True)
        self._tick_thread.start()

    # ---------- camera / calibration callbacks ----------
    def _open_camera(self) -> bool:
        return self.worker.request_open()

    def _close_camera(self) -> None:
        self.worker.request_close()

    def _start_calibration(self) -> None:
        self.worker.start_calibration()

    def _notify(self, message: str) -> None:
        self.status_message = message
        alarm_mod.notify("Argus", message)

    def _on_alarm_start(self, violation_kind: str) -> None:
        path = VIOLATION_SOUNDS.get(violation_kind)
        sound_path = str(path) if path is not None and path.is_file() else None
        self.alarm.start(sound_path)

    # ---------- outbox drain ----------
    def _get_api_key(self) -> str | None:
        if self._api_key_cache is not None:
            return self._api_key_cache
        try:
            self._api_key_cache = config_mod.load_api_key(self.cfg)
        except SystemExit as e:
            log.error("cannot load calendar API key: %s", e)
            return None
        return self._api_key_cache

    def _send_to_calendar(self, date: str, task: dict) -> None:
        key = self._get_api_key()
        if key is None:
            raise RuntimeError("no calendar API key available")
        calendar_client.log_pomodoro(self.cfg["calendar_append_url"], key, date, task)

    def _drain_outbox_async(self) -> None:
        def worker():
            try:
                outbox.drain(self._send_to_calendar)
            except Exception as e:
                log.warning("outbox drain error: %s", e)
            with self.lock:
                self._syncing = False

        threading.Thread(target=worker, daemon=True).start()

    # ---------- tick loop (background thread; replaces Tk's root.after poll) ----------
    def _loop(self) -> None:
        interval = self.cfg["check_interval_ms"] / 1000.0
        while not self._stopping:
            with self.lock:
                reading = None
                try:
                    while True:
                        reading = self.reading_queue.get_nowait()
                except queue.Empty:
                    pass
                if reading is not None:
                    self.latest_reading = reading

                self.timer.tick(reading)
                self._update_thumbnail()
                self._handle_terminal_states()
            time.sleep(interval)

    def _update_thumbnail(self) -> None:
        try:
            small = self.preview_queue.get_nowait()
        except queue.Empty:
            return
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(small).save(buf, format="JPEG", quality=70)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            self.thumbnail_b64 = f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            log.debug("thumbnail render failed: %s", e)

    def _handle_terminal_states(self) -> None:
        state = self.timer.state
        if state == State.LOGGING:
            self._handle_logging()
        elif state in (State.FAILED, State.ABORTED):
            self.status_message = self._describe_outcome(self.timer.last_outcome)
            self.timer.to_idle()
            self.thumbnail_b64 = None

    def _describe_outcome(self, outcome: dict | None) -> str:
        if not outcome:
            return ""
        if outcome["outcome"] == "aborted":
            return f"Session aborted ({outcome['reason']}) — not your fault, nothing was logged."
        return f"Session failed: {outcome['reason']}"

    def _handle_logging(self) -> None:
        session = self.timer.session
        date, task = calendar_client.build_task(session, self.cfg["timezone"])
        record = self.timer.build_record("success", None)

        if self.cfg["dry_run"]:
            log.info("DRY RUN: would log session %s -> %s / %s", session.id, date, task)
            self.status_message = f"Logged (dry run): {task['text']}"
        else:
            outbox.add({"date": date, "task": task})
            try:
                self._send_to_calendar(date, task)
                self.status_message = f"Logged: {task['text']}"
            except Exception as e:
                log.warning("immediate calendar send failed, left in outbox: %s", e)
                self.status_message = "Queued — will retry syncing to calendar."

        ledger.record_session(record)
        self.timer.to_idle()
        self.thumbnail_b64 = None

    # ---------- JS-facing actions ----------
    def start(self, raw_label: str, minutes: float | None, look_down_enabled: bool) -> dict:
        with self.lock:
            try:
                self.timer.start(raw_label, look_down_enabled, minutes=minutes)
            except (ValueError, RuntimeError) as e:
                return {"error": str(e)}
        return {"ok": True}

    def pause_resume(self) -> dict:
        with self.lock:
            if self.timer.state == State.RUNNING:
                self.timer.manual_pause()
            elif self.timer.state == State.INTERRUPTED:
                self.timer.manual_resume()
        return {"ok": True}

    def stop(self) -> dict:
        with self.lock:
            self.timer.abort_manual()
        return {"ok": True}

    def false_positive(self) -> dict:
        with self.lock:
            if self.timer.session is not None and self.latest_reading is not None:
                ledger.record_false_positive(self.timer.session.id, asdict(self.latest_reading))
                self.status_message = "Logged current readings for later threshold tuning."
        return {"ok": True}

    # ---------- snapshot for JS polling ----------
    def snapshot(self) -> dict:
        with self.lock:
            state = self.timer.state
            session = self.timer.session

            countdown = ""
            progress_frac = 0.0
            if session is not None and state in (State.RUNNING, State.VIOLATION_GRACE, State.CALIBRATING):
                remaining = session.remaining_s
                countdown = f"{int(remaining // 60):02d}:{int(remaining % 60):02d}"
                progress_frac = 0.0 if session.duration_s <= 0 else max(0.0, min(1.0, remaining / session.duration_s))
            elif state == State.INTERRUPTED:
                countdown = "paused"

            name = self.worker.resolved_camera_name or "not open"

            if self._syncing:
                status_message = "Syncing any pending sessions..."
            elif state == State.VIOLATION_GRACE:
                status_message = f"Violation: {self.timer.grace_kind}. Return to resolve."
            elif state == State.CAMERA_ERROR:
                status_message = "Camera unavailable — retrying..."
            else:
                status_message = self.status_message

            degraded = ""
            if self.worker.degraded:
                degraded = (
                    "PHONE DETECTION OFF" if "phone" in self.worker.degraded
                    else "DEGRADED: " + ",".join(self.worker.degraded).upper()
                )

            return {
                "headline": HEADLINES.get(state, state.value.lower()),
                "substate": "",
                "countdown": countdown,
                "progress_frac": progress_frac,
                "camera_name": name,
                "state": state.value,
                "violations": session.violations if session is not None else 0,
                "camera_status_text": f"Camera: {name}   State: {state.value}",
                "status_message": status_message,
                "degraded": degraded,
                "thumbnail_b64": self.thumbnail_b64,
                "default_minutes": self.cfg["pomodoro_minutes"],
                "buttons": {
                    "start_enabled": state == State.IDLE and not self._syncing,
                    "pause_enabled": state in (State.RUNNING, State.INTERRUPTED),
                    "pause_label": "resume" if state == State.INTERRUPTED else "pause",
                    "stop_enabled": state not in (State.IDLE,) and state not in TERMINAL_UI_STATES,
                    "false_positive_enabled": session is not None,
                },
            }

    # ---------- shutdown ----------
    def shutdown(self) -> None:
        self._stopping = True
        try:
            self.worker.stop()
        except Exception:
            pass
        try:
            _lock_fh.close()
        except Exception:
            pass


class Api:
    """Thin bridge exposed to the webview's JS as `window.pywebview.api`."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def start(self, label: str, minutes, look_down: bool) -> dict:
        try:
            m = float(minutes) if minutes is not None else None
        except (TypeError, ValueError):
            m = None
        return self._engine.start(label or "", m, bool(look_down))

    def pause_resume(self) -> dict:
        return self._engine.pause_resume()

    def stop(self) -> dict:
        return self._engine.stop()

    def false_positive(self) -> dict:
        return self._engine.false_positive()

    def get_state(self) -> dict:
        return self._engine.snapshot()


def main() -> None:
    cfg = config_mod.load_config()
    ledger.setup_logging(cfg["log_level"])
    log.info("Argus starting")

    engine = Engine(cfg)
    api = Api(engine)

    window = webview.create_window(
        "Argus",
        str(WEB_DIR / "index.html"),
        js_api=api,
        width=620,
        height=880,
        min_size=(520, 760),
        on_top=True,
        background_color="#fafaf9",
    )
    window.events.closing += lambda: engine.shutdown()

    webview.start()


if __name__ == "__main__":
    main()
