"""Runtime and persistence checks for the gaze tracker.

Made testable without a camera by projecting the known 3D face model through a
known camera to synthesize landmarks, and stubbing the network with something
that returns a chosen angle. Everything in between -- PnP, normalization,
un-rotation, ray/plane intersection, monitor selection -- then runs against an
answer that can be computed independently.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="gaze deps: pip install -r requirements-gaze.txt")
cv2 = pytest.importorskip("cv2")

from gaze.geometry import (  # noqa: E402
    FACE_MODEL_7,
    LANDMARK_IDS,
    CalibrationError,
    plane_from_corners,
)
from gaze.store import Calibration, load_calibration, save_calibration  # noqa: E402
from gaze.tracker import GazeTracker  # noqa: E402

W, H = 1280, 720
K = np.array([[960.0, 0.0, W / 2], [0.0, 960.0, H / 2], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
IMAGE = np.full((H, W, 3), 128, dtype=np.uint8)
# 600mm out. A tvec of zero puts the face model at the pinhole, where the
# projection is degenerate and PnP fails.
RVEC = np.array([[0.0], [0.0], [0.0]])
TVEC = np.array([[0.0], [0.0], [600.0]])


class StubModel(torch.nn.Module):
    """Returns a fixed (pitch, yaw), and asserts it was handed correctly shaped
    crops -- a preprocessing regression would otherwise only show up as poor
    accuracy on real hardware, weeks later."""

    def __init__(self, pitch=0.0, yaw=0.0):
        super().__init__()
        self.pitch_yaw = (pitch, yaw)
        self.calls = 0

    def forward(self, person_idx, face, right_eye, left_eye):
        self.calls += 1
        assert face.shape == (1, 3, 96, 96), face.shape
        assert right_eye.shape == (1, 3, 64, 96), right_eye.shape
        assert left_eye.shape == (1, 3, 64, 96), left_eye.shape
        return torch.tensor([list(self.pitch_yaw)], dtype=torch.float32)


def synth_landmarks(rvec=RVEC, tvec=TVEC):
    proj, _ = cv2.projectPoints(FACE_MODEL_7, rvec, tvec, K, DIST)
    proj = proj.reshape(-1, 2)
    full = np.zeros((478, 2))
    for slot, idx in enumerate(LANDMARK_IDS):
        full[idx] = proj[slot] / (W, H)
    return full


def flat_screen(name="main", z=-500.0, width=600.0, height=340.0,
                pixels=(1920, 1080), global_origin=(0, 0)):
    tl = np.array([-width / 2, -height / 2, z])
    corners = np.array([tl, tl + [width, 0, 0], tl + [width, height, 0], tl + [0, height, 0]])
    return plane_from_corners(name, corners, pixels, global_origin=global_origin)


def make_tracker(screens, pitch=0.0, yaw=0.0, **kw):
    kw.setdefault("landmark_smoothing", 1)
    kw.setdefault("gaze_smoothing", 1)
    return GazeTracker(StubModel(pitch, yaw), K, DIST, screens, torch.device("cpu"), **kw)


# ---------- happy path ----------

def test_straight_ahead_gaze_lands_on_the_screen():
    screen = flat_screen(z=500.0)
    reading = make_tracker([screen]).update(IMAGE, synth_landmarks())

    assert reading.ok and reading.head_pose_ok
    assert reading.on_any_screen
    assert reading.screen_name == "main"


def test_straight_ahead_gaze_lands_at_the_screen_centre():
    """Head on the optical axis at z=600 looking down -z, panel centred on that
    axis at z=500: the hit must be the middle pixel."""
    reading = make_tracker([flat_screen(z=500.0)]).update(IMAGE, synth_landmarks())

    assert abs(reading.screen_xy[0] - 960) < 1
    assert abs(reading.screen_xy[1] - 540) < 6


def test_reports_viewing_distance():
    reading = make_tracker([flat_screen(z=500.0)]).update(IMAGE, synth_landmarks())
    assert 550 < reading.distance_mm < 650


# ---------- the exposed ray ----------

def test_exposes_a_unit_ray_for_calibration():
    """Screen calibration consumes rays directly and has no screens to
    intersect against yet, so the tracker has to hand out what it computed."""
    reading = make_tracker([flat_screen(z=500.0)]).update(IMAGE, synth_landmarks())

    assert reading.ray_origin.shape == (3,)
    assert abs(float(np.linalg.norm(reading.ray_direction)) - 1.0) < 1e-9
    assert float(reading.ray_direction[2]) < 0        # model's -z convention


def test_the_reported_ray_reproduces_the_reported_hit():
    screen = flat_screen(z=500.0)
    reading = make_tracker([screen]).update(IMAGE, synth_landmarks())

    point = screen.intersect(reading.ray_origin, reading.ray_direction)
    assert point is not None
    assert np.allclose(screen.to_pixels(point), reading.screen_xy, atol=1e-6)


# ---------- misses and bad input ----------

def test_a_panel_behind_the_head_is_not_hit():
    """Head at z=600 looking down -z; a panel at z=800 is behind it. Without
    the t>0 guard this reports a hit at a negative ray parameter."""
    behind = flat_screen("behind", z=800.0, width=4000.0, height=4000.0)
    reading = make_tracker([behind]).update(IMAGE, synth_landmarks())

    assert reading.ok
    assert not reading.on_any_screen
    assert reading.screen_xy is None


@pytest.mark.parametrize("bad", [np.zeros((10, 2)), np.zeros((478,)), np.zeros((478, 1))])
def test_malformed_landmarks_are_rejected_not_guessed(bad):
    assert make_tracker([flat_screen()]).update(IMAGE, bad).error == "landmarks_malformed"


# ---------- multiple monitors ----------

def test_the_nearer_of_two_overlapping_panels_wins():
    """A laptop screen in front of an external monitor means one ray satisfies
    both. `far` is listed first, so a first-match implementation fails here."""
    far = flat_screen("far", z=-900.0, width=2000.0, height=2000.0)
    near = flat_screen("near", z=-400.0, width=2000.0, height=2000.0)

    reading = make_tracker([far, near]).update(IMAGE, synth_landmarks())
    assert reading.screen_name == "near"


def test_global_desktop_offset_is_applied():
    off = flat_screen("second", width=2000.0, height=2000.0, global_origin=(1920, 0))
    reading = make_tracker([off]).update(IMAGE, synth_landmarks())

    assert abs(reading.global_xy[0] - (reading.screen_xy[0] + 1920)) < 1e-6
    assert abs(reading.global_xy[1] - reading.screen_xy[1]) < 1e-6


def test_no_screens_configured_means_no_hit_but_still_a_reading():
    reading = make_tracker([]).update(IMAGE, synth_landmarks())
    assert reading.ok
    assert not reading.on_any_screen
    assert reading.ray_direction is not None


# ---------- state ----------

def test_smoothing_buffer_stays_bounded():
    tracker = make_tracker([flat_screen()], landmark_smoothing=3, gaze_smoothing=3)
    for _ in range(5):
        tracker.update(IMAGE, synth_landmarks())
    assert len(tracker._gaze_buf) == 3


def test_reset_clears_carried_state():
    """A stale pose carried across a session gap makes the first frames after
    it converge toward wherever the head used to be."""
    tracker = make_tracker([flat_screen()])
    tracker.update(IMAGE, synth_landmarks())
    tracker.reset()

    assert len(tracker._gaze_buf) == 0
    assert tracker._prev_pose is None


def test_the_model_is_actually_invoked():
    tracker = make_tracker([flat_screen()])
    tracker.update(IMAGE, synth_landmarks())
    assert tracker.model.calls == 1


# ---------- persistence ----------

@pytest.fixture
def calibration():
    return Calibration(
        camera_matrix=K, dist_coeffs=DIST, capture_size=(W, H),
        screens=[flat_screen(), flat_screen("second", global_origin=(1920, 0))],
        created_at="2026-01-01T00:00:00Z",
    )


def test_calibration_round_trips(tmp_path, calibration):
    path = tmp_path / "cal.json"
    save_calibration(path, calibration)
    back = load_calibration(path)

    assert np.allclose(back.camera_matrix, K)
    assert len(back.screens) == 2
    assert back.screen("second").global_origin == (1920, 0)
    assert np.allclose(back.screen("main").x_axis, calibration.screens[0].x_axis)
    assert abs(back.screen("main").width_mm - calibration.screens[0].width_mm) < 1e-9


def test_capture_size_mismatch_is_refused(tmp_path, calibration):
    """Intrinsics are in pixels. A 1280x720 calibration used against a 640x480
    capture mis-scales every angle while still producing confident output."""
    path = tmp_path / "cal.json"
    save_calibration(path, calibration)

    with pytest.raises(CalibrationError, match="(?i)recalibrate"):
        load_calibration(path, expect_capture_size=(640, 480))


def test_missing_calibration_is_refused(tmp_path):
    with pytest.raises(CalibrationError):
        load_calibration(tmp_path / "absent.json")


def test_corrupt_calibration_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(CalibrationError, match="corrupt"):
        load_calibration(path)


def test_stale_epoch_is_refused(tmp_path, calibration):
    """Bumping the epoch is how a change to the normalization constants or the
    face model invalidates planes solved under the old ones."""
    path = tmp_path / "cal.json"
    save_calibration(path, calibration)
    data = json.loads(path.read_text())
    data["epoch"] = 999
    path.write_text(json.dumps(data))

    with pytest.raises(CalibrationError, match="epoch"):
        load_calibration(path)


def test_save_is_atomic(tmp_path, calibration):
    """A half-written calibration read back next launch is worse than none: it
    surfaces as inexplicably bad tracking rather than a parse error."""
    path = tmp_path / "cal.json"
    save_calibration(path, calibration)
    assert not list(tmp_path.glob("*.tmp"))
