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
import traceback
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
    # Reuses looked_down.aiff rather than the synthesized looked_away.aiff --
    # that file was hand-built and unverifiable outside macOS, so it's safer
    # to point at a real, known-working sound. Drop sounds/looked_away.aiff
    # in and flip this back if you'd rather have a distinct sound later.
    "look_away": SOUNDS_DIR / "looked_down.aiff",
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

# States in which a session is over and the UI should be back to its resting
# layout (main window visible, mini hidden, start enabled).
TERMINAL_UI_STATES = (State.IDLE, State.LOGGING, State.FAILED, State.ABORTED, State.SUCCESS)


def _install_excepthooks() -> None:
    """Without these, an exception on a worker thread vanishes silently and the
    app just appears to freeze or die. Every traceback goes to app.log."""

    def hook(exc_type, exc, tb):
        log.critical("uncaught exception:\n%s",
                     "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = hook

    def thread_hook(args):
        log.critical(
            "uncaught exception on thread %s:\n%s",
            args.thread.name if args.thread else "?",
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    threading.excepthook = thread_hook


class Engine:
    """Owns all backend state (timer, vision worker, alarm, sync) and runs the
    tick loop on its own thread. Every method touching shared state takes
    self.lock -- the JS bridge (Api) calls in from pywebview's threads."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.RLock()
        # Never taken while self.lock is held -- both guard work that happens on
        # background threads precisely so the tick thread never blocks on I/O.
        self._api_key_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._api_key_cache: str | None = None
        self.latest_reading: vision.FrameReading | None = None
        self.status_message = ""
        self.auth_warning = ""
        self.thumbnail_b64: str | None = None
        self._syncing = True
        self._stopping = False
        # True from Start until the session reaches a terminal state. The JS
        # side watches this to decide when to show/hide the mini window, so
        # native window calls stay on pywebview's own threads.
        self.session_active = False

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
            on_calibrate_finish=self._finish_calibration,
            on_alarm_start=self._on_alarm_start,
            on_alarm_stop=self.alarm.stop,
            on_notify=self._notify,
        )
        if self.timer.last_outcome and self.timer.last_outcome.get("reason") == "crash_recovery":
            self.status_message = "Previous session was interrupted (crash/quit) — not logged."

        self._startup_sync_async()

        self._tick_thread = threading.Thread(target=self._loop, name="argus-tick", daemon=True)
        self._tick_thread.start()

    # ---------- camera / calibration callbacks ----------
    def _open_camera(self) -> bool:
        return self.worker.request_open()

    def _close_camera(self) -> None:
        self.worker.request_close()

    def _start_calibration(self) -> None:
        self.worker.start_calibration()

    def _finish_calibration(self) -> None:
        self.worker.finish_calibration()

    def _notify(self, message: str) -> None:
        self.status_message = message
        alarm_mod.notify("Argus", message)

    def _on_alarm_start(self, violation_kind: str) -> None:
        path = VIOLATION_SOUNDS.get(violation_kind)
        sound_path = str(path) if path is not None and path.is_file() else None
        self.alarm.start(sound_path)

    # ---------- calendar auth / outbox ----------
    def _get_api_key(self) -> str:
        """Raises config_mod.ConfigError if unavailable.

        Called from several background threads (startup sync, calendar send),
        and the miss path shells out to `security`, so the cache fill is
        serialised on its own lock rather than on self.lock."""
        with self._api_key_lock:
            if self._api_key_cache is not None:
                return self._api_key_cache
            self._api_key_cache = config_mod.load_api_key(self.cfg)
            return self._api_key_cache

    def _send_to_calendar(self, date: str, task: dict) -> None:
        key = self._get_api_key()
        calendar_client.log_pomodoro(self.cfg["calendar_append_url"], key, date, task)

    def _startup_sync_async(self) -> None:
        """Preflight the calendar auth, then drain anything left in the outbox."""

        def worker():
            warning = ""
            if not self.cfg["dry_run"]:
                try:
                    key = self._get_api_key()
                except config_mod.ConfigError as e:
                    warning = str(e)
                else:
                    ok, msg = calendar_client.check_auth(self.cfg["calendar_api_url"], key)
                    if not ok:
                        warning = msg
                        log.warning("calendar preflight failed: %s", msg)

            if not warning:
                # Same lock the per-session sends take, so a drain and a fresh
                # send can never interleave POSTs for the same outbox.
                with self._send_lock:
                    try:
                        outbox.drain(self._send_to_calendar)
                    except Exception as e:
                        log.warning("outbox drain error: %s", e)

            with self.lock:
                self.auth_warning = warning
                self._syncing = False

        threading.Thread(target=worker, name="argus-startup-sync", daemon=True).start()

    # ---------- tick loop ----------
    def _loop(self) -> None:
        interval = self.cfg["check_interval_ms"] / 1000.0
        while not self._stopping:
            try:
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
            except Exception:
                # A dead tick thread means a frozen app with no explanation.
                # Log the traceback and keep running.
                log.exception("error in tick loop")
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
            self._end_session()

    def _end_session(self) -> None:
        self.thumbnail_b64 = None
        self.session_active = False

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
            # Durable first: once this returns the session survives a crash,
            # so the send itself is free to happen off this thread.
            outbox.add({"date": date, "task": task})
            self.status_message = "Logging to calendar…"
            self._send_async(session.id, date, task)

        ledger.record_session(record)
        self.timer.to_idle()
        self._end_session()

    def _send_async(self, session_id: str, date: str, task: dict) -> None:
        """Hand the calendar POST to a background thread.

        log_pomodoro() retries five times with a 10s timeout each plus a
        backoff sleep -- up to ~60s of wall time. Doing that inline left it
        running under self.lock (this is reached from _loop, which holds the
        lock for the whole tick), so snapshot() blocked on the pywebview
        bridge thread and the UI froze: countdown stuck, Stop unresponsive,
        and no violation or grace expiry processed for the duration.

        Sends are serialised on _send_lock so two sessions finishing close
        together don't retry over each other or report out of order."""

        def worker():
            with self._send_lock:
                try:
                    self._send_to_calendar(date, task)
                except (calendar_client.CalendarClientError, config_mod.ConfigError) as e:
                    # Non-retryable: bad auth or bad request shape. Retrying
                    # will never fix it, so say so rather than claiming it is
                    # queued.
                    log.error("calendar rejected session %s: %s", session_id, e)
                    message, warning = "Saved locally but NOT logged to the calendar.", str(e)
                except Exception as e:
                    log.warning("calendar send failed for %s, left in outbox: %s", session_id, e)
                    message, warning = "Queued — will retry syncing to calendar.", None
                else:
                    message, warning = f"Logged: {task['text']}", ""

            with self.lock:
                self.status_message = message
                if warning is not None:
                    self.auth_warning = warning

        threading.Thread(target=worker, name="argus-calendar-send", daemon=True).start()

    # ---------- JS-facing actions ----------
    def start(self, raw_label: str, minutes: float | None, look_down_enabled: bool,
              camera_enabled: bool, look_away_enabled: bool = False) -> dict:
        with self.lock:
            try:
                self.timer.start(raw_label, look_down_enabled, minutes=minutes,
                                 camera_enabled=camera_enabled,
                                 look_away_enabled=look_away_enabled)
            except (ValueError, RuntimeError) as e:
                return {"error": str(e)}
            self.status_message = ""
            self.session_active = True
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

    def _diagnostics(self, camera_on: bool) -> dict:
        """Live detector output, surfaced so thresholds can be tuned against
        real numbers instead of guesswork."""
        r = self.latest_reading
        if not camera_on or r is None:
            return {"active": False}
        return {
            "active": True,
            "face": bool(r.face_present),
            "phone": round(float(r.phone_confidence), 2),
            "phone_threshold": self.cfg["phone_confidence_threshold"],
            "pitch": None if r.pitch_delta_deg is None else round(r.pitch_delta_deg, 1),
            "pitch_threshold": self.cfg["look_down_enter_delta_degrees"],
            "yaw": None if r.yaw_delta_deg is None else round(r.yaw_delta_deg, 1),
            "yaw_threshold": self.cfg.get("look_away_enter_delta_degrees", 28),
            "calibrated": self.worker.calibrated,
        }

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
                progress_frac = (
                    0.0 if session.duration_s <= 0
                    else max(0.0, min(1.0, 1.0 - (remaining / session.duration_s)))
                )
            elif state == State.INTERRUPTED:
                countdown = "paused"

            camera_on = session is not None and session.camera_enabled
            if session is not None and not session.camera_enabled:
                name = "off — timer only"
            else:
                name = self.worker.resolved_camera_name or "not open"

            if self._syncing:
                status_message = "Checking calendar connection…"
            elif state == State.VIOLATION_GRACE:
                status_message = f"Violation: {self.timer.grace_kind}. Return to resolve."
            elif state == State.CAMERA_ERROR:
                status_message = "Camera unavailable — retrying…"
            else:
                status_message = self.status_message

            degraded = ""
            if camera_on and self.worker.degraded:
                degraded = (
                    "PHONE DETECTION OFF" if "phone" in self.worker.degraded
                    else "DEGRADED: " + ",".join(self.worker.degraded).upper()
                )

            return {
                "headline": HEADLINES.get(state, state.value.lower()),
                "countdown": countdown,
                "progress_frac": progress_frac,
                "camera_name": name,
                "camera_on": camera_on,
                "state": state.value,
                "session_active": self.session_active,
                "in_violation": state == State.VIOLATION_GRACE,
                "violations": session.violations if session is not None else 0,
                "status_message": status_message,
                "auth_warning": self.auth_warning,
                "degraded": degraded,
                "thumbnail_b64": self.thumbnail_b64 if camera_on else None,
                "default_minutes": self.cfg["pomodoro_minutes"],
                "dry_run": bool(self.cfg["dry_run"]),
                "diag": self._diagnostics(camera_on),
                "buttons": {
                    "start_enabled": state == State.IDLE and not self._syncing,
                    "pause_enabled": state in (State.RUNNING, State.INTERRUPTED),
                    "pause_label": "resume" if state == State.INTERRUPTED else "pause",
                    "stop_enabled": state not in TERMINAL_UI_STATES,
                    "false_positive_enabled": session is not None and session.camera_enabled,
                },
            }

    # ---------- shutdown ----------
    def shutdown(self) -> None:
        log.info("Argus shutting down")
        self._stopping = True
        try:
            self.alarm.stop()
        except Exception:
            pass
        try:
            self.worker.stop()
            self.worker.join(timeout=2.0)
        except Exception:
            pass
        try:
            _lock_fh.close()
        except Exception:
            pass


BOOTING_SNAPSHOT = {
    "headline": "starting",
    "countdown": "--:--",
    "progress_frac": 0.0,
    "camera_name": "—",
    "camera_on": False,
    "state": "STARTING",
    "session_active": False,
    "in_violation": False,
    "violations": 0,
    "status_message": "Loading detection models…",
    "auth_warning": "",
    "degraded": "",
    "thumbnail_b64": None,
    "default_minutes": None,
    "dry_run": False,
    "diag": {"active": False},
    "buttons": {
        "start_enabled": False,
        "pause_enabled": False,
        "pause_label": "pause",
        "stop_enabled": False,
        "false_positive_enabled": False,
    },
}


class Api:
    """Bridge exposed to both webviews as `window.pywebview.api`.

    EVERY attribute here must be underscore-prefixed. pywebview's
    inject_pywebview() walks dir(js_api) and recurses into any public
    non-callable attribute to build the JS function list. Storing a pywebview
    Window object publicly makes it recurse into Window.dom.body, which calls
    evaluate_js() before the window has started, raising WebViewException --
    the whole api then silently resolves to an empty function list and the UI
    renders with no working bridge at all.

    Window show/hide/move happen here -- i.e. on a pywebview thread, in
    response to a JS call -- rather than from Engine's tick thread. Driving
    native window state from an unrelated worker thread is the same class of
    bug that caused the camera segfault."""

    def __init__(self):
        self._engine: Engine | None = None
        self._main_window = None
        self._mini_window = None
        self._mini_visible = False
        self._mini_lock = threading.Lock()

    # -- wiring (called from main(), never from JS) --
    def _attach_windows(self, main_window, mini_window) -> None:
        self._main_window = main_window
        self._mini_window = mini_window

    def _attach_engine(self, engine: Engine) -> None:
        self._engine = engine

    # -- session control --
    def start(self, label: str, minutes, look_down: bool, camera: bool,
              look_away: bool = False) -> dict:
        if self._engine is None:
            return {"error": "Still starting up — try again in a moment."}
        try:
            m = float(minutes) if minutes is not None else None
        except (TypeError, ValueError):
            m = None
        return self._engine.start(label or "", m, bool(look_down), bool(camera),
                                  bool(look_away))

    def pause_resume(self) -> dict:
        return {"ok": True} if self._engine is None else self._engine.pause_resume()

    def stop(self) -> dict:
        return {"ok": True} if self._engine is None else self._engine.stop()

    def false_positive(self) -> dict:
        return {"ok": True} if self._engine is None else self._engine.false_positive()

    def get_state(self) -> dict:
        if self._engine is None:
            return dict(BOOTING_SNAPSHOT)
        return self._engine.snapshot()

    # -- mini window --
    def enter_mini(self) -> dict:
        with self._mini_lock:
            if self._mini_visible or self._mini_window is None:
                return {"ok": True}
            try:
                self._mini_window.show()
                try:
                    screen = webview.screens[0]
                    self._mini_window.move(screen.width - 240, 60)
                except Exception as e:
                    log.debug("mini window positioning failed: %s", e)
                if self._main_window is not None:
                    self._main_window.hide()
                self._mini_visible = True
            except Exception:
                log.exception("failed to enter mini mode")
        return {"ok": True}

    def exit_mini(self) -> dict:
        with self._mini_lock:
            if not self._mini_visible:
                return {"ok": True}
            try:
                if self._mini_window is not None:
                    self._mini_window.hide()
                if self._main_window is not None:
                    self._main_window.show()
                self._mini_visible = False
            except Exception:
                log.exception("failed to exit mini mode")
        return {"ok": True}


def main() -> None:
    cfg = config_mod.load_config()
    ledger.setup_logging(cfg["log_level"])
    _install_excepthooks()
    log.info("Argus starting")

    api = Api()
    holder: dict = {}

    main_window = webview.create_window(
        "Argus",
        str(WEB_DIR / "index.html"),
        js_api=api,
        width=560,
        height=780,
        min_size=(480, 660),
        background_color="#0B0B0B",
    )
    mini_window = webview.create_window(
        "Argus Timer",
        str(WEB_DIR / "mini.html"),
        js_api=api,
        width=168,
        height=68,
        frameless=True,
        easy_drag=True,
        on_top=True,
        resizable=False,
        hidden=True,
        background_color="#0B0B0B",
    )
    api._attach_windows(main_window, mini_window)

    closing = threading.Event()

    def on_closing():
        if closing.is_set():
            return
        closing.set()
        engine = holder.get("engine")
        if engine is not None:
            engine.shutdown()
        else:
            log.info("Argus shutting down (before engine was ready)")

    main_window.events.closing += on_closing

    def boot():
        """Building the Engine loads the MediaPipe models, which takes several
        seconds. Doing it before create_window() left a dead window on screen
        for the whole of that; doing it here means the UI paints immediately
        and just disables Start until the models are in."""
        t0 = time.monotonic()
        try:
            engine = Engine(cfg)
        except Exception:
            log.exception("engine failed to start")
            return
        holder["engine"] = engine
        api._attach_engine(engine)
        log.info("engine ready in %.1fs", time.monotonic() - t0)

    threading.Thread(target=boot, name="argus-boot", daemon=True).start()

    webview.start()
    on_closing()


if __name__ == "__main__":
    main()
