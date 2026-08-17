"""How vision.VisionWorker behaves around the optional gaze tracker.

The property under test is that gaze is strictly additive: with it off, or
broken, or missing its dependencies, nothing about the existing head-pose
detection changes. Argus watching the desk must never depend on a feature whose
model is a manual download.

These run without a camera, without torch, and without a checkpoint.
"""
from __future__ import annotations

import dataclasses
import queue
from unittest import mock

import pytest

import vision


@pytest.fixture
def worker_factory(cfg, monkeypatch):
    """A VisionWorker with model loading stubbed out -- __init__ otherwise
    loads MediaPipe, which a unit test has no business doing."""
    def make(**overrides):
        conf = dict(cfg)
        conf.update(overrides)
        monkeypatch.setattr(vision.VisionWorker, "_load_models", lambda self: None)
        return vision.VisionWorker(conf, queue.Queue())
    return make


# ---------- off by default ----------

def test_gaze_is_off_in_the_shipped_config(cfg):
    assert cfg["detect_gaze"] is False


def test_no_tracker_is_built_when_disabled(worker_factory):
    worker = worker_factory(detect_gaze=False)
    assert worker.gaze_tracker is None
    assert worker.gaze_available is False


def test_disabled_gaze_is_not_reported_as_degraded(worker_factory):
    """A feature nobody switched on is not a degradation, and showing DEGRADED
    for it would train the user to ignore the banner."""
    worker = worker_factory(detect_gaze=False)
    assert "gaze" not in worker.degraded


# ---------- failures degrade, never raise ----------

def test_missing_dependencies_degrade_rather_than_crash(worker_factory, monkeypatch):
    real_import = __import__

    def no_torch(name, *args, **kwargs):
        if name.startswith("gaze."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_torch)
    worker = worker_factory(detect_gaze=True)

    assert worker.gaze_tracker is None
    assert "gaze" in worker.degraded


def test_missing_calibration_degrades_rather_than_crashes(worker_factory):
    """The calibration file does not exist until tools/calibrate_screens.py has
    been run once, so this is the state of every fresh install."""
    worker = worker_factory(
        detect_gaze=True,
        gaze_calibration_path="state/definitely_not_here.json",
    )
    assert worker.gaze_tracker is None
    assert "gaze" in worker.degraded


# ---------- FrameReading contract ----------

def test_frame_reading_defaults_to_no_gaze():
    reading = vision.FrameReading(ok=True)
    assert reading.gaze_screen is None
    assert reading.gaze_xy is None
    assert reading.gaze_on_screen is False
    assert reading.gaze_available is False


def test_frame_reading_still_carries_no_image_data():
    """The privacy invariant the README states. Gaze added coordinates and a
    name; anything array-shaped here would be a regression."""
    reading = vision.FrameReading(ok=True, gaze_screen="main", gaze_xy=(10.0, 20.0))
    for value in vars(reading).values():
        assert not hasattr(value, "shape"), f"{value!r} looks like image data"


def test_frame_reading_is_still_immutable():
    """frozen=True is what lets a reading cross the worker/Tk queue boundary
    without anyone having to reason about who might mutate it."""
    reading = vision.FrameReading(ok=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reading.gaze_screen = "main"


# ---------- capture resolution ----------

def test_capture_stays_640x480_when_gaze_is_off(cfg):
    conf = dict(cfg)
    conf["detect_gaze"] = False
    cap = mock.MagicMock()
    cap.isOpened.return_value = True

    with mock.patch.object(vision.cv2, "VideoCapture", return_value=cap), \
         mock.patch.object(vision, "resolve_camera_index", return_value=0):
        vision.open_camera(conf)

    widths = [c.args[1] for c in cap.set.call_args_list if c.args[0] == vision.cv2.CAP_PROP_FRAME_WIDTH]
    assert widths == [640]


def test_capture_matches_the_gaze_calibration_when_gaze_is_on(cfg):
    """Camera intrinsics are in pixels, so capturing at a different resolution
    than the calibration silently mis-scales every angle."""
    conf = dict(cfg)
    conf["detect_gaze"] = True
    conf["gaze_capture_width"] = 1280
    conf["gaze_capture_height"] = 720
    cap = mock.MagicMock()
    cap.isOpened.return_value = True

    with mock.patch.object(vision.cv2, "VideoCapture", return_value=cap), \
         mock.patch.object(vision, "resolve_camera_index", return_value=0):
        vision.open_camera(conf)

    widths = [c.args[1] for c in cap.set.call_args_list if c.args[0] == vision.cv2.CAP_PROP_FRAME_WIDTH]
    heights = [c.args[1] for c in cap.set.call_args_list if c.args[0] == vision.cv2.CAP_PROP_FRAME_HEIGHT]
    assert widths == [1280]
    assert heights == [720]


# ---------- per-tick behaviour ----------

class FakeTracker:
    def __init__(self, reading=None):
        self.reading = reading
        self.calls = 0
        self.resets = 0

    def update(self, rgb, landmarks):
        self.calls += 1
        return self.reading

    def reset(self):
        self.resets += 1


class FakeReading:
    def __init__(self, ok=True, name="main", xy=(100.0, 200.0), on=True):
        self.ok = ok
        self.screen_name = name
        self.screen_xy = xy
        self.on_any_screen = on


class FakeLandmark:
    x = 0.5
    y = 0.5


class FakeFaceResult:
    face_landmarks = [[FakeLandmark()] * 478]


def test_gaze_runs_only_every_nth_tick(worker_factory):
    """Inference is ~30ms; running it on every tick would eat the frame budget
    the phone detector already stripes for the same reason."""
    worker = worker_factory(detect_gaze=False, gaze_every_n_ticks=3)
    worker.gaze_tracker = FakeTracker(FakeReading())

    for tick in range(1, 10):
        worker._tick_count = tick
        worker._run_gaze(None, FakeFaceResult())

    assert worker.gaze_tracker.calls == 3          # ticks 3, 6, 9


def test_skipped_ticks_reuse_the_last_result(worker_factory):
    """Reporting None on skipped ticks would make any downstream sustain
    counter flicker, exactly as it would for phone confidence."""
    worker = worker_factory(detect_gaze=False, gaze_every_n_ticks=2)
    worker.gaze_tracker = FakeTracker(FakeReading(name="left-monitor"))

    worker._tick_count = 2
    first = worker._run_gaze(None, FakeFaceResult())
    worker._tick_count = 3
    skipped = worker._run_gaze(None, FakeFaceResult())

    assert first == ("left-monitor", (100.0, 200.0), True)
    assert skipped == first
    assert worker.gaze_tracker.calls == 1


def test_a_failing_frame_clears_rather_than_latching(worker_factory):
    """A stale hit left latched after an error reads downstream as the user
    still looking at a screen they have since left."""
    worker = worker_factory(detect_gaze=False, gaze_every_n_ticks=1)
    worker.gaze_tracker = FakeTracker(FakeReading())
    worker._tick_count = 1
    worker._run_gaze(None, FakeFaceResult())

    worker.gaze_tracker.update = mock.Mock(side_effect=RuntimeError("boom"))
    worker._tick_count = 2
    assert worker._run_gaze(None, FakeFaceResult()) == (None, None, False)


def test_a_not_ok_reading_reports_nothing(worker_factory):
    worker = worker_factory(detect_gaze=False, gaze_every_n_ticks=1)
    worker.gaze_tracker = FakeTracker(FakeReading(ok=False))
    worker._tick_count = 1
    assert worker._run_gaze(None, FakeFaceResult()) == (None, None, False)


def test_calibration_resets_the_tracker(worker_factory):
    """Session boundaries must not carry a head pose across them."""
    worker = worker_factory(detect_gaze=False)
    worker.gaze_tracker = FakeTracker()
    worker.start_calibration()

    assert worker.gaze_tracker.resets == 1
    assert worker._last_gaze == (None, None, False)


def test_start_calibration_works_without_a_tracker(worker_factory):
    worker = worker_factory(detect_gaze=False)
    worker.start_calibration()          # must not raise
    assert worker.gaze_tracker is None
