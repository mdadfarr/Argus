from __future__ import annotations
import fcntl
import logging
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)

# ---------- single-instance guard (fixes I-15) -- before anything else ----------
_lock_fh = open(STATE / "instance.lock", "w")
try:
    fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit("Pomodoro Guard is already running. Refusing to start a second instance.")
_lock_fh.write(str(os.getpid()))
_lock_fh.flush()

import config as config_mod
import ledger
import outbox
import alarm as alarm_mod
import calendar_client
import vision
from timer import SessionTimer, State

log = logging.getLogger(__name__)


class App:
    def __init__(self, root: tk.Tk, cfg: dict):
        self.root = root
        self.cfg = cfg
        self._api_key_cache: str | None = None
        self.latest_reading: vision.FrameReading | None = None
        self.status_message = ""

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
            on_alarm_start=self.alarm.start,
            on_alarm_stop=self.alarm.stop,
            on_notify=self._notify,
        )
        if self.timer.last_outcome and self.timer.last_outcome.get("reason") == "crash_recovery":
            self.status_message = "Previous session was interrupted (crash/quit) — not logged."

        self._build_gui()
        self._drain_outbox_async()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(cfg["check_interval_ms"], self._poll)

    # ---------- camera / calibration callbacks (called from Tk thread by timer) ----------
    def _open_camera(self) -> bool:
        return self.worker.request_open()

    def _close_camera(self) -> None:
        self.worker.request_close()

    def _start_calibration(self) -> None:
        self.worker.start_calibration()

    def _notify(self, message: str) -> None:
        self.status_message = message
        alarm_mod.notify("Pomodoro Guard", message)

    # ---------- outbox drain (fixes I-13/§5.3 point 4) ----------
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
        self.start_btn.configure(state="disabled")
        self.status_var.set("Syncing any pending sessions...")

        def worker():
            try:
                outbox.drain(self._send_to_calendar)
            except Exception as e:
                log.warning("outbox drain error: %s", e)
            self.root.after(0, self._drain_done)

        threading.Thread(target=worker, daemon=True).start()

    def _drain_done(self) -> None:
        self.start_btn.configure(state="normal")
        self.status_var.set("Ready.")

    # ---------- GUI ----------
    def _build_gui(self) -> None:
        self.root.title("Pomodoro Guard")
        self.root.attributes("-topmost", True)

        frm = ttk.Frame(self.root, padding=10)
        frm.grid()

        ttk.Label(frm, text="Label:").grid(column=0, row=0, sticky="w")
        self.label_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.label_var, width=30).grid(column=1, row=0, columnspan=2, sticky="we")

        self.look_down_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Detect looking down (screen-only work)", variable=self.look_down_var).grid(
            column=0, row=1, columnspan=3, sticky="w"
        )

        self.start_btn = ttk.Button(frm, text="Start", command=self._on_start)
        self.start_btn.grid(column=0, row=2)
        self.pause_btn = ttk.Button(frm, text="Pause", command=self._on_pause, state="disabled")
        self.pause_btn.grid(column=1, row=2)
        self.stop_btn = ttk.Button(frm, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.grid(column=2, row=2)

        self.countdown_var = tk.StringVar(value="--:--")
        ttk.Label(frm, textvariable=self.countdown_var, font=("Menlo", 24)).grid(column=0, row=3, columnspan=3)

        self.thumbnail_label = ttk.Label(frm)
        self.thumbnail_label.grid(column=0, row=4, columnspan=3)
        self._thumbnail_photo = None  # keep a reference or Tk garbage-collects it

        self.camera_name_var = tk.StringVar(value="Camera: (not open)")
        ttk.Label(frm, textvariable=self.camera_name_var).grid(column=0, row=5, columnspan=3, sticky="w")

        self.status_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.status_var, wraplength=280).grid(column=0, row=6, columnspan=3, sticky="w")

        self.degraded_var = tk.StringVar(value="")
        self.degraded_label = ttk.Label(frm, textvariable=self.degraded_var, foreground="red")
        self.degraded_label.grid(column=0, row=7, columnspan=3, sticky="w")
        if self.worker.degraded:
            self.degraded_var.set("PHONE DETECTION OFF" if "phone" in self.worker.degraded else "DEGRADED: " + ",".join(self.worker.degraded))

        self.false_positive_btn = ttk.Button(
            frm, text="That was wrong (log for tuning)", command=self._on_false_positive, state="disabled"
        )
        self.false_positive_btn.grid(column=0, row=8, columnspan=3)

        if self.status_message:
            self.status_var.set(self.status_message)

    def _on_start(self) -> None:
        try:
            self.timer.start(self.label_var.get(), self.look_down_var.get())
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="Pause")
        self.stop_btn.configure(state="normal")
        self.false_positive_btn.configure(state="normal")

    def _on_pause(self) -> None:
        if self.timer.state == State.RUNNING:
            self.timer.manual_pause()
            self.pause_btn.configure(text="Resume")
        elif self.timer.state == State.INTERRUPTED:
            self.timer.manual_resume()
            self.pause_btn.configure(text="Pause")

    def _on_stop(self) -> None:
        self.timer.abort_manual()

    def _on_false_positive(self) -> None:
        if self.timer.session is None or self.latest_reading is None:
            return
        from dataclasses import asdict
        ledger.record_false_positive(self.timer.session.id, asdict(self.latest_reading))
        self.status_var.set("Logged current readings for later threshold tuning.")

    # ---------- poll loop (fixes I-18: only this Tk-thread function touches widgets) ----------
    def _poll(self) -> None:
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
        self._update_labels()
        self._handle_terminal_states()

        self.root.after(self.cfg["check_interval_ms"], self._poll)

    def _update_thumbnail(self) -> None:
        try:
            small = self.preview_queue.get_nowait()
        except queue.Empty:
            return
        try:
            from PIL import Image, ImageTk
            img = ImageTk.PhotoImage(Image.fromarray(small))
            self.thumbnail_label.configure(image=img)
            self._thumbnail_photo = img
        except Exception as e:
            log.debug("thumbnail render failed: %s", e)

    def _update_labels(self) -> None:
        state = self.timer.state
        session = self.timer.session
        if session is not None and state in (State.RUNNING, State.VIOLATION_GRACE, State.CALIBRATING):
            remaining = session.remaining_s
            self.countdown_var.set(f"{int(remaining // 60):02d}:{int(remaining % 60):02d}")
        elif state == State.INTERRUPTED:
            self.countdown_var.set("paused")
        else:
            self.countdown_var.set("--:--")

        name = self.worker.resolved_camera_name or "(not open)"
        self.camera_name_var.set(f"Camera: {name}   State: {state.value}")

        if state == State.VIOLATION_GRACE:
            self.status_var.set(f"Violation: {self.timer.grace_kind}. Return to resolve.")
        elif state == State.CAMERA_ERROR:
            self.status_var.set("Camera unavailable — retrying...")
        elif self.status_message:
            self.status_var.set(self.status_message)

    def _handle_terminal_states(self) -> None:
        state = self.timer.state
        if state == State.LOGGING:
            self._handle_logging()
        elif state in (State.FAILED, State.ABORTED):
            self.status_message = self._describe_outcome(self.timer.last_outcome)
            self.status_var.set(self.status_message)
            self.timer.to_idle()
            self._reset_controls()

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
        self.status_var.set(self.status_message)
        self.timer.to_idle()
        self._reset_controls()

    def _reset_controls(self) -> None:
        self.start_btn.configure(state="normal")
        self.pause_btn.configure(state="disabled", text="Pause")
        self.stop_btn.configure(state="disabled")
        self.false_positive_btn.configure(state="disabled")
        self.countdown_var.set("--:--")

    # ---------- shutdown ----------
    def _on_close(self) -> None:
        try:
            self.worker.stop()
        except Exception:
            pass
        try:
            _lock_fh.close()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    cfg = config_mod.load_config()
    ledger.setup_logging(cfg["log_level"])
    log.info("Pomodoro Guard starting")

    root = tk.Tk()
    App(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
