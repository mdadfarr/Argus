"""Shared fixtures.

Nothing here touches a camera, a window, or the network. SessionTimer takes
every side effect as a callback and advances on a measured dt, so the whole
state machine is drivable from a test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def cfg():
    """The shipped example config, which is also what _validate is checked
    against. Tests mutate their own copy."""
    return json.loads((ROOT / "config.example.json").read_text())


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point every module that writes under state/ at a tmp dir, so tests never
    touch the developer's real session history."""
    import ledger
    import outbox
    import timer as timer_mod

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(ledger, "STATE", state)
    monkeypatch.setattr(ledger, "SESSIONS_PATH", state / "sessions.jsonl")
    monkeypatch.setattr(ledger, "FALSE_POSITIVES_PATH", state / "false_positives.jsonl")
    monkeypatch.setattr(outbox, "STATE", state)
    monkeypatch.setattr(outbox, "OUTBOX_PATH", state / "outbox.jsonl")
    monkeypatch.setattr(timer_mod, "STATE", state)
    monkeypatch.setattr(timer_mod, "CHECKPOINT_PATH", state / "current_session.json")
    return state


class Reading:
    """Stands in for vision.FrameReading."""

    def __init__(self, ok=True, face_present=True, pitch_delta_deg=None,
                 yaw_delta_deg=None, phone_confidence=0.0,
                 phone_detector_available=True):
        self.ok = ok
        self.face_present = face_present
        self.pitch_delta_deg = pitch_delta_deg
        self.yaw_delta_deg = yaw_delta_deg
        self.phone_confidence = phone_confidence
        self.phone_detector_available = phone_detector_available


@pytest.fixture
def reading():
    return Reading


@pytest.fixture
def events():
    """Records which timer callbacks fired, in order."""
    return []


@pytest.fixture
def make_timer(cfg, isolated_state, events, monkeypatch):
    """Builds a SessionTimer whose clock the test controls outright.

    time.monotonic is replaced with a counter the test advances by hand, so a
    30-minute session runs in microseconds and nothing depends on wall time.
    """
    import timer as timer_mod

    clock = {"now": 1000.0}
    monkeypatch.setattr(timer_mod.time, "monotonic", lambda: clock["now"])

    def build(**overrides):
        conf = dict(cfg)
        conf.update(overrides)
        t = timer_mod.SessionTimer(
            conf,
            on_camera_open=lambda: (events.append("camera_open"), True)[1],
            on_camera_close=lambda: events.append("camera_close"),
            on_calibrate_start=lambda: events.append("calibrate_start"),
            on_calibrate_finish=lambda: events.append("calibrate_finish"),
            on_alarm_start=lambda kind: events.append(f"alarm_start:{kind}"),
            on_alarm_stop=lambda: events.append("alarm_stop"),
            on_notify=lambda msg: events.append(f"notify:{msg}"),
        )
        return t

    build.clock = clock
    return build


@pytest.fixture
def advance(make_timer):
    """Drive a timer forward: advance(t, seconds, reading) in check_interval
    steps, so dt behaves the way it does in the real tick loop."""
    clock = make_timer.clock

    def go(t, seconds, reading=None, step=0.5):
        elapsed = 0.0
        while elapsed < seconds:
            clock["now"] += step
            elapsed += step
            t.tick(reading)
        return t

    return go
