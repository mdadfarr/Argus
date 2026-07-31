from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

import ledger
from vision import SustainedCondition

log = logging.getLogger(__name__)

STATE = Path(__file__).parent / "state"
CHECKPOINT_PATH = STATE / "current_session.json"


class State(str, Enum):
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    RUNNING = "RUNNING"
    VIOLATION_GRACE = "VIOLATION_GRACE"
    INTERRUPTED = "INTERRUPTED"
    CAMERA_ERROR = "CAMERA_ERROR"
    SUCCESS = "SUCCESS"
    LOGGING = "LOGGING"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


TERMINAL_STATES = (State.IDLE, State.SUCCESS, State.LOGGING, State.FAILED, State.ABORTED)


def clean_label(raw: str) -> str:
    s = "".join(ch for ch in raw if ch.isprintable()).strip()
    if not s:
        raise ValueError("Label is required.")
    return s[:120]


@dataclass
class Session:
    id: str
    label: str
    duration_s: float
    started_wall_iso: str
    started_mono: float
    look_down_enabled: bool = False
    focus_elapsed_s: float = 0.0
    paused_total_s: float = 0.0
    manual_paused_s: float = 0.0
    violations: int = 0
    manual_pauses: int = 0
    degraded: list = field(default_factory=list)

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.duration_s - self.focus_elapsed_s)


class SessionTimer:
    """State machine driven by tick(reading). `reading` is a vision.FrameReading
    or None. Every tick() call represents one real elapsed interval measured via
    time.monotonic() deltas -- never assumed from check_interval_ms (fixes I-06),
    and immune to wall-clock jumps from sleep/NTP/DST (fixes I-05).
    """

    def __init__(self, cfg: dict, on_camera_open=None, on_camera_close=None,
                 on_calibrate_start=None, on_alarm_start=None, on_alarm_stop=None,
                 on_notify=None):
        self.cfg = cfg
        self.state = State.IDLE
        self.session: Session | None = None
        self.last_tick: float | None = None
        self.calibration_deadline: float | None = None
        self.grace_deadline: float | None = None
        self.grace_kind: str | None = None
        self.manual_pause_deadline: float | None = None
        self.camera_backoff_idx = 0
        self.camera_next_retry: float | None = None
        self.resolve_clean_since: float | None = None
        self.last_outcome: dict | None = None
        self._look_down_state = False
        self.left_cond = SustainedCondition(cfg["no_face_threshold_seconds"])
        self.phone_cond = SustainedCondition(cfg["phone_sustain_seconds"])
        self.look_down_cond = SustainedCondition(cfg["look_down_threshold_seconds"])

        self._on_camera_open = on_camera_open
        self._on_camera_close = on_camera_close
        self._on_calibrate_start = on_calibrate_start
        self._on_alarm_start = on_alarm_start
        self._on_alarm_stop = on_alarm_stop
        self._on_notify = on_notify

        self._resume_check_startup()

    # ---------- startup crash recovery (fixes I-14) ----------
    def _resume_check_startup(self):
        if not CHECKPOINT_PATH.exists():
            return
        try:
            data = json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            data = {"label": "unknown"}
        log.warning("found interrupted session checkpoint from previous run: %s", data)
        record = {
            "id": data.get("id", "unknown"),
            "label": data.get("label", "unknown"),
            "outcome": "interrupted",
            "reason": "crash_recovery",
            "started": data.get("started_wall_iso"),
            "focus_elapsed_s": data.get("focus_elapsed_s", 0.0),
        }
        ledger.record_session(record)
        if self._on_notify:
            self._on_notify("A previous session was interrupted (crash/quit) and was not logged.")
        CHECKPOINT_PATH.unlink(missing_ok=True)

    def _checkpoint(self):
        if self.session is None:
            return
        STATE.mkdir(exist_ok=True)
        tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self.session)))
        os.replace(tmp, CHECKPOINT_PATH)

    def _clear_checkpoint(self):
        CHECKPOINT_PATH.unlink(missing_ok=True)

    # ---------- start / manual pause ----------
    def start(self, raw_label: str, look_down_enabled: bool):
        if self.state != State.IDLE:
            raise RuntimeError(f"cannot start from state {self.state}")
        label = clean_label(raw_label)
        minutes = max(self.cfg["min_session_minutes"],
                      min(self.cfg["max_session_minutes"], self.cfg["pomodoro_minutes"]))
        now_mono = time.monotonic()
        self.session = Session(
            id=f"pg-{int(time.time() * 1000)}",
            label=label,
            duration_s=minutes * 60,
            started_wall_iso=datetime.now(ZoneInfo(self.cfg["timezone"])).isoformat(),
            started_mono=now_mono,
            look_down_enabled=look_down_enabled,
        )
        self.last_tick = now_mono
        self._look_down_state = False
        self.left_cond.reset()
        self.phone_cond.reset()
        self.look_down_cond.reset()
        self.camera_backoff_idx = 0

        if self._on_camera_open:
            self._on_camera_open()
        self.state = State.CALIBRATING
        self.calibration_deadline = now_mono + self.cfg["calibration_seconds"]
        if self._on_calibrate_start:
            self._on_calibrate_start()
        self._checkpoint()

    def manual_pause(self):
        if self.state != State.RUNNING:
            return
        if self.session.manual_pauses >= self.cfg["max_manual_pauses"]:
            return
        self.session.manual_pauses += 1
        self.manual_pause_deadline = time.monotonic() + self.cfg["max_manual_pause_seconds"]
        self.state = State.INTERRUPTED
        self._checkpoint()

    def manual_resume(self):
        if self.state != State.INTERRUPTED:
            return
        now = time.monotonic()
        self.state = State.CALIBRATING
        self.calibration_deadline = now + self.cfg["calibration_seconds"]
        if self._on_calibrate_start:
            self._on_calibrate_start()
        self.last_tick = now
        self._checkpoint()

    def to_idle(self):
        """Called by main.py once a terminal state (LOGGING/FAILED/ABORTED) has
        been fully handled (calendar sync attempted, ledger written)."""
        self.session = None
        self.state = State.IDLE

    # ---------- main driver ----------
    def tick(self, reading) -> None:
        if self.state in TERMINAL_STATES:
            return

        now = time.monotonic()
        dt = 0.0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now

        if self.state in (State.RUNNING, State.VIOLATION_GRACE) and dt > self.cfg["max_tick_gap_seconds"]:
            self._abort("clock_gap", detail=f"{dt:.1f}s tick gap")
            return

        if self.state == State.CALIBRATING:
            self._tick_calibrating(now, reading)
        elif self.state == State.RUNNING:
            self._tick_running(now, dt, reading)
        elif self.state == State.VIOLATION_GRACE:
            self._tick_grace(now, dt, reading)
        elif self.state == State.CAMERA_ERROR:
            self._tick_camera_error(now)
        elif self.state == State.INTERRUPTED:
            self._tick_interrupted(now, dt)

    def _tick_calibrating(self, now, reading):
        if reading is not None and not reading.ok:
            self.state = State.CAMERA_ERROR
            self.camera_backoff_idx = 0
            self.camera_next_retry = now + self.cfg["camera_reopen_backoff_seconds"][0]
            return
        if now >= self.calibration_deadline:
            if reading is None or not reading.face_present:
                if self._on_notify:
                    self._on_notify("Calibration failed: no face detected. Session cancelled.")
                self._to_idle_after_cancel()
                return
            self.state = State.RUNNING
            self.last_tick = now
            self._checkpoint()

    def _update_look_down_latch(self, reading) -> None:
        if not self.session.look_down_enabled or reading.pitch_delta_deg is None:
            self._look_down_state = False
            return
        d = reading.pitch_delta_deg
        if d > self.cfg["look_down_enter_delta_degrees"]:
            self._look_down_state = True
        elif d < self.cfg["look_down_release_delta_degrees"]:
            self._look_down_state = False
        # else: inside the hysteresis dead zone -- keep previous latch value

    def _tick_running(self, now, dt, reading):
        self.session.focus_elapsed_s += dt
        self._checkpoint()

        if reading is None:
            return
        if not reading.ok:
            self.state = State.CAMERA_ERROR
            self.camera_backoff_idx = 0
            self.camera_next_retry = now + self.cfg["camera_reopen_backoff_seconds"][0]
            return

        violation_kind = self._evaluate_violation(dt, reading)

        if violation_kind is None and self.session.focus_elapsed_s >= self.session.duration_s:
            self._succeed()
            return

        if violation_kind is not None:
            self.session.violations += 1
            if self.session.violations > self.cfg["max_violations_per_session"]:
                self._fail(f"{violation_kind}_max_violations")
                return
            if violation_kind == "left" and not self.cfg["grace_on_left"]:
                self._fail("left_no_grace")
                return
            self.grace_kind = violation_kind
            self.grace_deadline = now + self.cfg["grace_period_seconds"]
            self.resolve_clean_since = None
            if self._on_alarm_start:
                self._on_alarm_start()
            if self._on_notify:
                self._on_notify(f"Violation: {violation_kind}. Return within {self.cfg['grace_period_seconds']:.0f}s.")
            self.state = State.VIOLATION_GRACE

    def _evaluate_violation(self, dt, reading) -> str | None:
        cfg = self.cfg
        self._update_look_down_latch(reading)

        left_sustained = self.left_cond.update(not reading.face_present, dt)

        phone_now = (
            cfg["detect_phone"] and reading.phone_detector_available
            and reading.phone_confidence >= cfg["phone_confidence_threshold"]
        )
        phone_sustained = self.phone_cond.update(phone_now, dt)

        look_down_sustained = self.look_down_cond.update(self._look_down_state, dt)

        if left_sustained:
            return "left"
        if phone_sustained:
            return "phone"
        if look_down_sustained:
            return "look_down"
        return None

    def _reset_condition(self, kind: str) -> None:
        {"left": self.left_cond, "phone": self.phone_cond, "look_down": self.look_down_cond}[kind].reset()

    def _tick_grace(self, now, dt, reading):
        self.session.paused_total_s += dt
        self._checkpoint()

        if self.session.paused_total_s > self.cfg["max_total_paused_seconds"]:
            if self._on_alarm_stop:
                self._on_alarm_stop()
            self._fail("max_total_paused")
            return

        if now > self.grace_deadline:
            if self._on_alarm_stop:
                self._on_alarm_stop()
            self._fail(f"{self.grace_kind}_grace_expired")
            return

        if reading is None or not reading.ok:
            return

        self._update_look_down_latch(reading)
        raw_phone = (
            self.cfg["detect_phone"] and reading.phone_detector_available
            and reading.phone_confidence >= self.cfg["phone_confidence_threshold"]
        )
        clean = reading.face_present and not raw_phone and not self._look_down_state
        if clean:
            if self.resolve_clean_since is None:
                self.resolve_clean_since = now
            elif now - self.resolve_clean_since >= 1.0:
                if self._on_alarm_stop:
                    self._on_alarm_stop()
                self._reset_condition(self.grace_kind)
                self.grace_kind = None
                self.resolve_clean_since = None
                self.state = State.RUNNING
                self.last_tick = now
        else:
            self.resolve_clean_since = None

    def _tick_camera_error(self, now):
        if now < self.camera_next_retry:
            return
        opened = self._on_camera_open() if self._on_camera_open else False
        if opened:
            self.state = State.CALIBRATING
            self.calibration_deadline = now + self.cfg["calibration_seconds"]
            if self._on_calibrate_start:
                self._on_calibrate_start()
            self.last_tick = now
            return
        backoffs = self.cfg["camera_reopen_backoff_seconds"]
        if self.camera_backoff_idx >= len(backoffs) - 1:
            self._abort("camera_unavailable")
            return
        self.camera_backoff_idx += 1
        self.camera_next_retry = now + backoffs[self.camera_backoff_idx]

    def _tick_interrupted(self, now, dt):
        self.session.manual_paused_s += dt
        if now > self.manual_pause_deadline:
            self._fail("manual_pause_exceeded")

    # ---------- terminal transitions ----------
    def _succeed(self):
        self.state = State.LOGGING
        self._clear_checkpoint()
        if self._on_camera_close:
            self._on_camera_close()
        self.last_outcome = self.build_record("success", None)

    def _fail(self, reason: str):
        if self._on_alarm_stop:
            self._on_alarm_stop()
        record = self.build_record("failed", reason)
        ledger.record_session(record)
        self.state = State.FAILED
        self._clear_checkpoint()
        if self._on_camera_close:
            self._on_camera_close()
        if self.cfg["notify_on_failure"] and self._on_notify:
            self._on_notify(f"Session failed: {reason}")
        self.last_outcome = record

    def _abort(self, reason: str, detail: str = ""):
        record = self.build_record("aborted", reason, detail=detail)
        ledger.record_session(record)
        self.state = State.ABORTED
        self._clear_checkpoint()
        if self._on_camera_close:
            self._on_camera_close()
        if self._on_notify:
            self._on_notify(f"Session aborted ({reason}) — not your fault, nothing was logged.")
        self.last_outcome = record

    def _to_idle_after_cancel(self):
        self._clear_checkpoint()
        if self._on_camera_close:
            self._on_camera_close()
        self.session = None
        self.state = State.IDLE

    def build_record(self, outcome: str, reason: str | None, detail: str = "") -> dict:
        s = self.session
        return {
            "id": s.id,
            "label": s.label,
            "outcome": outcome,
            "reason": reason,
            "detail": detail,
            "started": s.started_wall_iso,
            "focus_elapsed_s": round(s.focus_elapsed_s, 1),
            "paused_total_s": round(s.paused_total_s, 1),
            "manual_paused_s": round(s.manual_paused_s, 1),
            "violations": s.violations,
            "manual_pauses": s.manual_pauses,
            "degraded": s.degraded,
            "look_down_enabled": s.look_down_enabled,
        }
