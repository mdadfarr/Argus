from __future__ import annotations
import fcntl
import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
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

# ---------- Swiss Style Grid design tokens ----------
BG = "#F5DEB0"
RED = "#EE1C1C"
INK = "#0A0A0A"

SOUNDS_DIR = Path(__file__).parent / "sounds"
VIOLATION_SOUNDS = {
    "left": SOUNDS_DIR / "left_room.aiff",
    "look_down": SOUNDS_DIR / "looked_down.aiff",
    "phone": SOUNDS_DIR / "phone_detected.aiff",
}

HEADLINES = {
    State.IDLE: "idle.",
    State.CALIBRATING: "calibrating.",
    State.RUNNING: "focus.",
    State.VIOLATION_GRACE: "violation.",
    State.INTERRUPTED: "paused.",
    State.CAMERA_ERROR: "camera error.",
}


def _pick_font(candidates: list[str], default: str) -> str:
    available = set(tkfont.families())
    for name in candidates:
        if name in available:
            return name
    return default


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

        self.mini: tk.Toplevel | None = None
        self.mini_time_var = tk.StringVar(value="")
        self._mini_drag = (0, 0)

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

    def _on_alarm_start(self, violation_kind: str) -> None:
        path = VIOLATION_SOUNDS.get(violation_kind)
        sound_path = str(path) if path is not None and path.is_file() else None
        self.alarm.start(sound_path)

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

    # ---------- GUI (Swiss Style Grid: cream / red / black) ----------
    def _build_gui(self) -> None:
        self.root.title("Pomodoro Guard")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=INK)

        sans_bold = _pick_font(["Helvetica Now Display", "Neue Haas Grotesk Bold", "Helvetica Neue", "Inter"], "Helvetica")
        mono = _pick_font(["JetBrains Mono", "Space Mono", "Menlo", "Courier"], "Menlo")
        self.font_headline = (sans_bold, 28, "bold")
        self.font_label = (sans_bold, 10, "bold")
        self.font_body = (sans_bold, 11)
        self.font_mono_sm = (mono, 11)
        self.font_mono_lg = (mono, 38, "bold")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Swiss.TLabel", background=BG, foreground=INK, font=self.font_body)
        style.configure("SwissSmallCaps.TLabel", background=BG, foreground=INK, font=self.font_label)
        style.configure("SwissHeadline.TLabel", background=BG, foreground=INK, font=self.font_headline)
        style.configure("SwissMono.TLabel", background=BG, foreground=INK, font=self.font_mono_sm)
        style.configure("SwissMonoLg.TLabel", background=BG, foreground=INK, font=self.font_mono_lg)
        style.configure("SwissAccent.TLabel", background=BG, foreground=RED, font=self.font_label)
        style.configure("Swiss.TEntry", fieldbackground=BG, foreground=INK, insertcolor=INK,
                         bordercolor=INK, lightcolor=INK, darkcolor=INK, borderwidth=1)
        style.configure("SwissPrimary.TButton", background=RED, foreground=BG, font=self.font_label,
                         borderwidth=1, relief="flat", bordercolor=INK, padding=8)
        style.map("SwissPrimary.TButton",
                  background=[("disabled", "#D9B48F"), ("active", RED)],
                  foreground=[("disabled", INK)])
        style.configure("SwissSecondary.TButton", background=BG, foreground=INK, font=self.font_label,
                         borderwidth=1, relief="flat", bordercolor=INK, padding=8)
        style.map("SwissSecondary.TButton",
                  background=[("disabled", BG), ("active", "#EAD9B3")])
        style.configure("Swiss.TCheckbutton", background=BG, foreground=INK, font=self.font_label)
        style.map("Swiss.TCheckbutton",
                  indicatorcolor=[("selected", RED), ("!selected", BG)],
                  background=[("active", BG)])

        frm = tk.Frame(self.root, bg=BG)
        frm.grid(row=0, column=0, sticky="nsew", padx=18, pady=16)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        for c in range(3):
            frm.columnconfigure(c, weight=1, uniform="col")

        # row 0: top bar
        topbar = tk.Frame(frm, bg=BG)
        topbar.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 4))
        self.clock_var = tk.StringVar(value="")
        ttk.Label(topbar, textvariable=self.clock_var, style="SwissMono.TLabel").pack(side="left")
        ttk.Label(topbar, text="pomodoro guard", style="SwissSmallCaps.TLabel").pack(side="right")

        # row 1: headline (session state)
        self.headline_var = tk.StringVar(value=HEADLINES[State.IDLE])
        ttk.Label(frm, textvariable=self.headline_var, style="SwissHeadline.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(10, 14)
        )

        # row 2: countdown clock + time-remaining bar (blank while idle)
        countdown_cell = tk.Frame(frm, bg=BG)
        countdown_cell.grid(row=2, column=0, sticky="w", pady=(0, 14))
        self.countdown_var = tk.StringVar(value="")
        ttk.Label(countdown_cell, textvariable=self.countdown_var, style="SwissMonoLg.TLabel").pack(anchor="w")

        bar_cell = tk.Frame(frm, bg=BG)
        bar_cell.grid(row=2, column=1, columnspan=2, sticky="swe", pady=(0, 14))
        self._bar_width = 280
        self.progress_canvas = tk.Canvas(bar_cell, width=self._bar_width, height=6, bg=BG, highlightthickness=0)
        self.progress_canvas.pack(fill="x")
        self._progress_rect = self.progress_canvas.create_rectangle(0, 0, 0, 6, fill=RED, outline="")

        # row 3: label / minutes / look-down toggle
        def field(col, label_text, var, width=14):
            cell = tk.Frame(frm, bg=BG)
            cell.grid(row=3, column=col, sticky="we", pady=6)
            ttk.Label(cell, text=label_text, style="SwissSmallCaps.TLabel").pack(anchor="w")
            ttk.Entry(cell, textvariable=var, style="Swiss.TEntry", width=width).pack(fill="x", pady=(4, 0))

        self.label_var = tk.StringVar()
        field(0, "label", self.label_var, width=20)
        self.minutes_var = tk.StringVar(value=str(self.cfg["pomodoro_minutes"]))
        field(1, "minutes", self.minutes_var, width=6)

        cb_cell = tk.Frame(frm, bg=BG)
        cb_cell.grid(row=3, column=2, sticky="we", pady=6)
        ttk.Label(cb_cell, text="look down detect", style="SwissSmallCaps.TLabel").pack(anchor="w")
        self.look_down_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cb_cell, text="screen-only work", variable=self.look_down_var,
                        style="Swiss.TCheckbutton").pack(anchor="w", pady=(4, 0))

        # row 4: stat tags
        def stat_cell(col, label_text, var):
            cell = tk.Frame(frm, bg=BG)
            cell.grid(row=4, column=col, sticky="we", pady=(14, 6))
            ttk.Label(cell, text=label_text, style="SwissSmallCaps.TLabel").pack(anchor="w")
            ttk.Label(cell, textvariable=var, style="SwissMono.TLabel").pack(anchor="w", pady=(4, 0))

        self.stat_camera_var = tk.StringVar(value="(not open)")
        self.stat_state_var = tk.StringVar(value=State.IDLE.value)
        self.stat_violations_var = tk.StringVar(value="0")
        stat_cell(0, "camera", self.stat_camera_var)
        stat_cell(1, "state", self.stat_state_var)
        stat_cell(2, "violations", self.stat_violations_var)

        # row 5: thumbnail + camera status
        thumb_cell = tk.Frame(frm, bg=BG)
        thumb_cell.grid(row=5, column=0, sticky="w", pady=6)
        thumb_frame = tk.Frame(thumb_cell, bg=BG, highlightbackground=INK, highlightthickness=1)
        thumb_frame.pack()
        self.thumbnail_label = tk.Label(thumb_frame, bg=BG)
        self.thumbnail_label.pack()
        self._thumbnail_photo = None  # keep a reference or Tk garbage-collects it

        cam_cell = tk.Frame(frm, bg=BG)
        cam_cell.grid(row=5, column=1, columnspan=2, sticky="w", pady=6)
        ttk.Label(cam_cell, text="camera status", style="SwissSmallCaps.TLabel").pack(anchor="w")
        self.camera_name_var = tk.StringVar(value="Camera: (not open)")
        ttk.Label(cam_cell, textvariable=self.camera_name_var, style="Swiss.TLabel").pack(anchor="w", pady=(4, 0))

        # row 6: controls
        self.start_btn = ttk.Button(frm, text="start", style="SwissPrimary.TButton", command=self._on_start)
        self.start_btn.grid(row=6, column=0, sticky="we", pady=(18, 8), padx=(0, 6))
        self.pause_btn = ttk.Button(frm, text="pause", style="SwissSecondary.TButton",
                                    command=self._on_pause, state="disabled")
        self.pause_btn.grid(row=6, column=1, sticky="we", pady=(18, 8), padx=6)
        self.stop_btn = ttk.Button(frm, text="stop", style="SwissSecondary.TButton",
                                   command=self._on_stop, state="disabled")
        self.stop_btn.grid(row=6, column=2, sticky="we", pady=(18, 8), padx=(6, 0))

        # row 7: false positive
        self.false_positive_btn = ttk.Button(
            frm, text="that was wrong (log for tuning)", style="SwissSecondary.TButton",
            command=self._on_false_positive, state="disabled"
        )
        self.false_positive_btn.grid(row=7, column=0, columnspan=3, sticky="we", pady=8)

        # row 8: status ledger
        self.status_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.status_var, style="Swiss.TLabel", wraplength=520).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(10, 4)
        )

        # row 9: degraded warning
        self.degraded_var = tk.StringVar(value="")
        self.degraded_label = ttk.Label(frm, textvariable=self.degraded_var, style="SwissAccent.TLabel")
        self.degraded_label.grid(row=9, column=0, columnspan=3, sticky="w", pady=(0, 4))
        if self.worker.degraded:
            self.degraded_var.set(
                "PHONE DETECTION OFF" if "phone" in self.worker.degraded
                else "DEGRADED: " + ",".join(self.worker.degraded).upper()
            )

        if self.status_message:
            self.status_var.set(self.status_message)

        self._update_clock()

    def _update_clock(self) -> None:
        self.clock_var.set(time.strftime("%H:%M:%S"))
        self.root.after(1000, self._update_clock)

    def _on_start(self) -> None:
        try:
            minutes = float(self.minutes_var.get())
        except (TypeError, ValueError):
            minutes = self.cfg["pomodoro_minutes"]
        try:
            self.timer.start(self.label_var.get(), self.look_down_var.get(), minutes=minutes)
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="pause")
        self.stop_btn.configure(state="normal")
        self.false_positive_btn.configure(state="normal")
        self._enter_mini_mode()

    def _enter_mini_mode(self) -> None:
        if self.mini is not None:
            return
        self.root.withdraw()

        mini = tk.Toplevel(self.root)
        mini.overrideredirect(True)
        mini.attributes("-topmost", True)
        mini.configure(bg=BG, highlightbackground=INK, highlightthickness=1)

        x = self.root.winfo_screenwidth() - 180
        y = 40
        mini.geometry(f"140x70+{x}+{y}")

        self.mini_time_var.set(self.countdown_var.get() or "--:--")
        label = tk.Label(mini, textvariable=self.mini_time_var, bg=BG, fg=INK,
                          font=self.font_mono_lg)
        label.pack(expand=True, fill="both")

        def start_drag(event):
            self._mini_drag = (event.x, event.y)

        def do_drag(event):
            dx, dy = self._mini_drag
            nx = mini.winfo_x() + (event.x - dx)
            ny = mini.winfo_y() + (event.y - dy)
            mini.geometry(f"+{nx}+{ny}")

        def reopen(_event=None):
            self._exit_mini_mode()

        for widget in (mini, label):
            widget.bind("<ButtonPress-1>", start_drag)
            widget.bind("<B1-Motion>", do_drag)
            widget.bind("<Double-Button-1>", reopen)

        self.mini = mini

    def _exit_mini_mode(self) -> None:
        if self.mini is None:
            return
        try:
            self.mini.destroy()
        except Exception:
            pass
        self.mini = None
        self.root.deiconify()
        self.root.lift()

    def _on_pause(self) -> None:
        if self.timer.state == State.RUNNING:
            self.timer.manual_pause()
            self.pause_btn.configure(text="resume")
        elif self.timer.state == State.INTERRUPTED:
            self.timer.manual_resume()
            self.pause_btn.configure(text="pause")

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
            frac = 0.0 if session.duration_s <= 0 else max(0.0, min(1.0, remaining / session.duration_s))
            width = self._bar_width * frac
            self.progress_canvas.coords(self._progress_rect, 0, 0, width, 6)
            if self.mini is not None:
                self.mini_time_var.set(self.countdown_var.get())
        elif state == State.INTERRUPTED:
            self.countdown_var.set("paused")
        else:
            self.countdown_var.set("")
            self.progress_canvas.coords(self._progress_rect, 0, 0, 0, 6)

        name = self.worker.resolved_camera_name or "(not open)"
        self.camera_name_var.set(f"Camera: {name}   State: {state.value}")
        self.stat_camera_var.set(name)
        self.stat_state_var.set(state.value.lower())
        self.stat_violations_var.set(str(session.violations if session is not None else 0))
        self.headline_var.set(HEADLINES.get(state, state.value.lower() + "."))

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
        self.pause_btn.configure(state="disabled", text="pause")
        self.stop_btn.configure(state="disabled")
        self.false_positive_btn.configure(state="disabled")
        self.countdown_var.set("")
        self.progress_canvas.coords(self._progress_rect, 0, 0, 0, 6)
        self._exit_mini_mode()

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
