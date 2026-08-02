"""SessionTimer state machine.

Two bugs already shipped here that these cover: finish_calibration() was never
called (so pitch_delta stayed None and look-down could never fire), and the
grace/violation accounting had no coverage at all.
"""
from __future__ import annotations

import pytest

from timer import State, clean_label

# ---------- label handling ----------

def test_clean_label_strips_unprintable_and_caps_length():
    assert clean_label("  hello\x00world  ") == "helloworld"
    assert len(clean_label("x" * 500)) == 120


@pytest.mark.parametrize("raw", ["", "   ", "\x00\x01"])
def test_clean_label_rejects_empty(raw):
    with pytest.raises(ValueError):
        clean_label(raw)


# ---------- start ----------

def test_camera_session_calibrates_before_running(make_timer, events):
    t = make_timer()
    t.start("write tests", look_down_enabled=False)
    assert t.state == State.CALIBRATING
    assert "camera_open" in events and "calibrate_start" in events


def test_timer_only_session_skips_calibration(make_timer, events):
    t = make_timer()
    t.start("read", look_down_enabled=True, camera_enabled=False)
    assert t.state == State.RUNNING
    assert "camera_open" not in events
    # Head-pose detection is meaningless with no camera.
    assert t.session.look_down_enabled is False


def test_duration_is_clamped_to_configured_bounds(make_timer):
    t = make_timer()
    t.start("x", look_down_enabled=False, minutes=9999)
    assert t.session.duration_s == t.cfg["max_session_minutes"] * 60
    t.to_idle()
    t.start("x", look_down_enabled=False, minutes=0.1)
    assert t.session.duration_s == t.cfg["min_session_minutes"] * 60


def test_cannot_start_twice(make_timer):
    t = make_timer()
    t.start("x", look_down_enabled=False)
    with pytest.raises(RuntimeError):
        t.start("y", look_down_enabled=False)


# ---------- calibration ----------

def test_calibration_commits_baseline_when_face_present(make_timer, advance, reading, events):
    """The regression that silently disabled look-down detection: without
    on_calibrate_finish firing, the vision worker never leaves calibrating mode
    and pitch_delta_deg stays None forever."""
    t = make_timer()
    t.start("x", look_down_enabled=True)
    advance(t, t.cfg["calibration_seconds"] + 1, reading(face_present=True))
    assert t.state == State.RUNNING
    assert "calibrate_finish" in events


def test_calibration_without_a_face_cancels_the_session(make_timer, advance, reading, events):
    t = make_timer()
    t.start("x", look_down_enabled=False)
    advance(t, t.cfg["calibration_seconds"] + 1, reading(face_present=False))
    assert t.state == State.IDLE
    assert t.session is None
    assert any("Calibration failed" in e for e in events)


# ---------- success ----------

def test_session_completes_and_enters_logging(make_timer, advance, reading):
    t = make_timer(pomodoro_minutes=5, min_session_minutes=5)
    t.start("focus", look_down_enabled=False, minutes=5)
    advance(t, t.cfg["calibration_seconds"] + 1, reading())
    advance(t, 5 * 60, reading())
    assert t.state == State.LOGGING
    assert t.last_outcome["outcome"] == "success"


def test_timer_only_session_completes_without_readings(make_timer, advance):
    t = make_timer(min_session_minutes=5)
    t.start("focus", look_down_enabled=False, minutes=5, camera_enabled=False)
    advance(t, 5 * 60, None)
    assert t.state == State.LOGGING


# ---------- violations ----------

def _run_to_running(t, advance, reading):
    t.start("focus", look_down_enabled=True, look_away_enabled=True, minutes=60)
    advance(t, t.cfg["calibration_seconds"] + 1, reading())
    assert t.state == State.RUNNING
    return t


def test_absence_triggers_violation_and_alarm(make_timer, advance, reading, events):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    advance(t, t.cfg["no_face_threshold_seconds"] + 1, reading(face_present=False))
    assert t.state == State.VIOLATION_GRACE
    assert t.grace_kind == "left"
    assert "alarm_start:left" in events


def test_phone_triggers_violation(make_timer, advance, reading, events):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    hot = reading(phone_confidence=t.cfg["phone_confidence_threshold"] + 0.1)
    advance(t, t.cfg["phone_sustain_seconds"] + 1, hot)
    assert t.grace_kind == "phone"


def test_phone_below_threshold_is_ignored(make_timer, advance, reading):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    cold = reading(phone_confidence=t.cfg["phone_confidence_threshold"] - 0.1)
    advance(t, t.cfg["phone_sustain_seconds"] + 5, cold)
    assert t.state == State.RUNNING


def test_look_away_uses_hysteresis(make_timer, advance, reading):
    """Between release and enter the latch must hold its previous value, or a
    head hovering at the boundary flickers in and out of violation."""
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    enter = t.cfg["look_away_enter_delta_degrees"]
    release = t.cfg["look_away_release_delta_degrees"]

    t._update_look_away_latch(reading(yaw_delta_deg=enter + 1))
    assert t._look_away_state is True
    # Dead zone: still latched.
    t._update_look_away_latch(reading(yaw_delta_deg=(enter + release) / 2))
    assert t._look_away_state is True
    t._update_look_away_latch(reading(yaw_delta_deg=release - 1))
    assert t._look_away_state is False


def test_returning_within_grace_resolves_the_violation(make_timer, advance, reading, events):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    advance(t, t.cfg["no_face_threshold_seconds"] + 1, reading(face_present=False))
    assert t.state == State.VIOLATION_GRACE
    advance(t, 2.0, reading(face_present=True))
    assert t.state == State.RUNNING
    assert "alarm_stop" in events


def test_grace_expiry_fails_the_session(make_timer, advance, reading):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    advance(t, t.cfg["no_face_threshold_seconds"] + 1, reading(face_present=False))
    advance(t, t.cfg["grace_period_seconds"] + 2, reading(face_present=False))
    assert t.state == State.FAILED
    assert t.last_outcome["reason"] == "left_grace_expired"


def test_exceeding_max_violations_fails(make_timer, advance, reading):
    t = _run_to_running(make_timer(max_session_minutes=60, max_violations_per_session=1),
                        advance, reading)
    for _ in range(2):
        advance(t, t.cfg["no_face_threshold_seconds"] + 1, reading(face_present=False))
        if t.state == State.VIOLATION_GRACE:
            advance(t, 2.0, reading(face_present=True))
    assert t.state == State.FAILED
    assert "max_violations" in t.last_outcome["reason"]


# ---------- interruptions ----------

def test_manual_pause_and_resume(make_timer, advance, reading):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    t.manual_pause()
    assert t.state == State.INTERRUPTED
    t.manual_resume()
    # Resuming a camera session recalibrates rather than trusting a stale baseline.
    assert t.state == State.CALIBRATING


def test_manual_pause_limit_is_enforced(make_timer, advance, reading):
    t = _run_to_running(make_timer(max_session_minutes=60, max_manual_pauses=1),
                        advance, reading)
    t.manual_pause()
    t.manual_resume()
    advance(t, t.cfg["calibration_seconds"] + 1, reading())
    t.manual_pause()
    assert t.state == State.RUNNING          # second pause refused


def test_overlong_manual_pause_fails(make_timer, advance, reading):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    t.manual_pause()
    advance(t, t.cfg["max_manual_pause_seconds"] + 2, None)
    assert t.state == State.FAILED
    assert t.last_outcome["reason"] == "manual_pause_exceeded"


def test_stop_button_aborts_rather_than_fails(make_timer, advance, reading):
    """Stopping on purpose isn't a failure to focus, so nothing punitive is
    recorded."""
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    t.abort_manual()
    assert t.state == State.ABORTED
    assert t.last_outcome["outcome"] == "aborted"


def test_large_tick_gap_aborts(make_timer, advance, reading):
    """Machine slept mid-session. Not the user's fault -- abort, don't fail."""
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    make_timer.clock["now"] += t.cfg["max_tick_gap_seconds"] + 10
    t.tick(reading())
    assert t.state == State.ABORTED
    assert t.last_outcome["reason"] == "clock_gap"


# ---------- camera failure ----------

def test_camera_read_failure_enters_error_state(make_timer, advance, reading):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    t.tick(reading(ok=False))
    assert t.state == State.CAMERA_ERROR


def test_camera_retries_then_aborts_without_penalty(make_timer, advance, reading, monkeypatch):
    t = _run_to_running(make_timer(max_session_minutes=60), advance, reading)
    t._on_camera_open = lambda: False           # camera never comes back
    t.tick(reading(ok=False))
    assert t.state == State.CAMERA_ERROR
    for _ in range(len(t.cfg["camera_reopen_backoff_seconds"]) + 2):
        make_timer.clock["now"] += max(t.cfg["camera_reopen_backoff_seconds"]) + 1
        t.tick(None)
    assert t.state == State.ABORTED
    assert t.last_outcome["reason"] == "camera_unavailable"


# ---------- crash recovery ----------

def test_leftover_checkpoint_is_recorded_as_interrupted(make_timer, isolated_state, events):
    import json

    import timer as timer_mod

    timer_mod.CHECKPOINT_PATH.write_text(json.dumps(
        {"id": "pg-old", "label": "abandoned", "focus_elapsed_s": 42.0,
         "started_wall_iso": "2026-08-02T10:00:00-04:00"}
    ))
    make_timer()
    assert not timer_mod.CHECKPOINT_PATH.exists()
    rows = [json.loads(line) for line in
            (isolated_state / "sessions.jsonl").read_text().splitlines() if line.strip()]
    assert rows[-1]["reason"] == "crash_recovery"
    assert rows[-1]["outcome"] == "interrupted"
