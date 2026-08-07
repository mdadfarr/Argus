"""Focus mode: every monitor goes black, one small white dot, no clock.

The point is boredom. Nothing here may surface elapsed or remaining time to
the UI -- FocusSession.snapshot() deliberately exposes no countdown, and the
web page has no timer of its own. Keep it that way when extending this.

The state machine is separate from SessionTimer rather than a mode of it: a
focus block has no grace period, no pause budget, no violation cap and is
never logged to the calendar. Folding those differences into SessionTimer
would have meant a conditional on nearly every branch of its tick.
"""

from __future__ import annotations

import logging
from enum import Enum

from vision import SustainedCondition

log = logging.getLogger(__name__)

# The three durations offered in the UI. Anything else is rejected -- an
# arbitrary-minutes box would let the user reason about how long is left,
# which is the one thing focus mode is built to prevent.
FOCUS_MINUTES_CHOICES = (5, 10, 15)

# Focus thresholds live under their own keys so tuning boredom mode can never
# loosen a real pomodoro. All optional: every read below goes through .get()
# with the value here as the fallback, so an existing config.json keeps
# working untouched.
FOCUS_DEFAULTS = {
    "focus_look_away_enter_delta_degrees": 40.0,
    "focus_look_away_release_delta_degrees": 26.0,
    "focus_look_away_threshold_seconds": 1.0,
    "focus_look_down_enter_delta_degrees": 20.0,
    "focus_look_down_release_delta_degrees": 12.0,
    "focus_look_down_threshold_seconds": 4.0,
    "focus_no_face_threshold_seconds": 3.0,
}


def focus_cfg(cfg: dict, key: str):
    return cfg.get(key, FOCUS_DEFAULTS[key])


class FocusState(Enum):
    CALIBRATING = "CALIBRATING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    CANCELLED = "CANCELLED"


TERMINAL = (FocusState.DONE, FocusState.CANCELLED)


class FocusSession:
    """Pure state machine -- no threads, no I/O, no window handles.

    The caller (Engine) owns the camera, the alarm and the black windows and
    drives this with a reading and a monotonic clock. That split is what makes
    the violation rules testable without a webview or a camera.
    """

    def __init__(self, cfg: dict, label: str, minutes: float, now: float):
        self.cfg = cfg
        self.label = label
        self.minutes = minutes
        self.duration_s = minutes * 60.0
        self.remaining_s = self.duration_s
        self.violations = 0
        # Currently-latched violation kind, or None. Drives the red flash.
        self.violation_kind: str | None = None
        # Set for exactly one tick when a violation begins, so Engine knows to
        # fire the sound once rather than on every tick of a held violation.
        self.violation_started: str | None = None

        self.state = FocusState.CALIBRATING
        self.calibration_deadline = now + cfg["calibration_seconds"]
        self._last_tick = now

        self._look_down_state = False
        self._look_away_state = False
        self.left_cond = SustainedCondition(focus_cfg(cfg, "focus_no_face_threshold_seconds"))
        self.phone_cond = SustainedCondition(cfg["phone_sustain_seconds"])
        self.look_down_cond = SustainedCondition(
            focus_cfg(cfg, "focus_look_down_threshold_seconds")
        )
        self.look_away_cond = SustainedCondition(
            focus_cfg(cfg, "focus_look_away_threshold_seconds")
        )

    # ---------- latches (same hysteresis shape as SessionTimer) ----------
    def _update_look_down_latch(self, reading) -> None:
        if reading.pitch_delta_deg is None:
            self._look_down_state = False
            return
        d = reading.pitch_delta_deg
        if d > focus_cfg(self.cfg, "focus_look_down_enter_delta_degrees"):
            self._look_down_state = True
        elif d < focus_cfg(self.cfg, "focus_look_down_release_delta_degrees"):
            self._look_down_state = False

    def _update_look_away_latch(self, reading) -> None:
        if reading.yaw_delta_deg is None:
            self._look_away_state = False
            return
        d = reading.yaw_delta_deg
        if d > focus_cfg(self.cfg, "focus_look_away_enter_delta_degrees"):
            self._look_away_state = True
        elif d < focus_cfg(self.cfg, "focus_look_away_release_delta_degrees"):
            self._look_away_state = False

    def _phone_now(self, reading) -> bool:
        return bool(
            self.cfg["detect_phone"]
            and reading.phone_detector_available
            and reading.phone_confidence >= self.cfg["phone_confidence_threshold"]
        )

    # ---------- tick ----------
    def tick(self, reading, now: float) -> None:
        # Clamped for the same reason SessionTimer clamps: a laptop that slept
        # mid-session would otherwise hand us a dt of minutes and burn the
        # whole block (or trip every sustain bucket) in one tick.
        dt = min(max(0.0, now - self._last_tick), self.cfg["max_tick_gap_seconds"])
        self._last_tick = now
        self.violation_started = None

        if self.state in TERMINAL:
            return

        if self.state == FocusState.CALIBRATING:
            if now >= self.calibration_deadline:
                self.state = FocusState.RUNNING
            return

        self.remaining_s -= dt
        if self.remaining_s <= 0:
            self.remaining_s = 0.0
            self.state = FocusState.DONE
            self.violation_kind = None
            return

        self._update_violation(reading, dt)

    def _update_violation(self, reading, dt: float) -> None:
        """A violation never pauses or ends the block -- it only flashes the
        screens red and sounds. Time keeps running, so sitting through a
        violation is not a way to shorten the block."""
        if reading is None or not reading.ok:
            # No usable frame (camera reopening, dropped frame). Neither
            # punish nor clear a held violation on missing data.
            return

        self._update_look_down_latch(reading)
        self._update_look_away_latch(reading)

        left_sustained = self.left_cond.update(not reading.face_present, dt)
        phone_sustained = self.phone_cond.update(self._phone_now(reading), dt)
        down_sustained = self.look_down_cond.update(self._look_down_state, dt)
        away_sustained = self.look_away_cond.update(self._look_away_state, dt)

        kind = None
        if left_sustained:
            kind = "left"
        elif phone_sustained:
            kind = "phone"
        elif down_sustained:
            kind = "look_down"
        elif away_sustained:
            kind = "look_away"

        if kind is None:
            self.violation_kind = None
            return

        if self.violation_kind is None:
            self.violations += 1
            self.violation_started = kind
            log.info("focus violation %d: %s", self.violations, kind)
        self.violation_kind = kind

    # ---------- control ----------
    def cancel(self) -> None:
        if self.state not in TERMINAL:
            self.state = FocusState.CANCELLED
            self.violation_kind = None

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL

    def snapshot(self) -> dict:
        """Everything the black windows are allowed to know.

        No remaining_s, no elapsed, no progress fraction: the page cannot leak
        a countdown it was never given.
        """
        return {
            "active": True,
            "state": self.state.value,
            "calibrating": self.state == FocusState.CALIBRATING,
            "in_violation": self.violation_kind is not None,
            "violation_kind": self.violation_kind,
        }


IDLE_SNAPSHOT = {
    "active": False,
    "state": "IDLE",
    "calibrating": False,
    "in_violation": False,
    "violation_kind": None,
}
