"""Closed loop: calibrate a monitor from simulated gaze, then track with it.

The check that the pieces compose. A simulated person genuinely looks at
things: for each frame the true direction from head to target is converted into
the pitch/yaw the network would have had to emit, and fed in through a stub.
Corner calibration recovers the monitor, and the recovered monitor is then used
to track fresh targets.

If any convention disagrees anywhere -- the -z direction, the normalization
rotation, the corner ordering, the plane fit -- the monitor comes back mirrored
or rotated. Each half passing its own unit tests would not catch that.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="gaze deps: pip install -r requirements-gaze.txt")
cv2 = pytest.importorskip("cv2")

from gaze.calibrate import CornerCollector, sanity_check  # noqa: E402
from gaze.geometry import (  # noqa: E402
    FACE_MODEL_7,
    LANDMARK_IDS,
    normalize_image,
    solve_head_pose,
)
from gaze.tracker import GazeTracker  # noqa: E402

W, H = 1280, 720
K = np.array([[960.0, 0.0, W / 2], [0.0, 960.0, H / 2], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
IMAGE = np.full((H, W, 3), 120, dtype=np.uint8)

# Ground truth: a 27in monitor, yawed 15 degrees and tilted back 5.
MON_W, MON_H = 597.0, 336.0
PIXELS = (2560, 1440)
_yaw, _tilt = np.deg2rad(15.0), np.deg2rad(-5.0)
_Ry = np.array([[np.cos(_yaw), 0, np.sin(_yaw)], [0, 1, 0], [-np.sin(_yaw), 0, np.cos(_yaw)]])
_Rx = np.array([[1, 0, 0], [0, np.cos(_tilt), -np.sin(_tilt)], [0, np.sin(_tilt), np.cos(_tilt)]])
_R = _Ry @ _Rx
_TL = np.array([-300.0, -120.0, -550.0])

HEAD_POSES = [
    (np.array([[0.0], [0.0], [0.0]]), np.array([[0.0], [0.0], [600.0]])),
    (np.array([[0.0], [0.15], [0.0]]), np.array([[-90.0], [10.0], [560.0]])),
    (np.array([[0.05], [-0.15], [0.0]]), np.array([[85.0], [-15.0], [640.0]])),
    (np.array([[-0.08], [0.0], [0.0]]), np.array([[10.0], [40.0], [720.0]])),
]
CORNER_UV = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def target_3d(u, v):
    """A point on the true monitor, u/v in 0..1 from its top-left."""
    return _TL + _R @ [MON_W * u, MON_H * v, 0.0]


def synth_landmarks(rvec, tvec):
    proj, _ = cv2.projectPoints(FACE_MODEL_7, rvec, tvec, K, DIST)
    proj = proj.reshape(-1, 2)
    full = np.zeros((478, 2))
    for slot, idx in enumerate(LANDMARK_IDS):
        full[idx] = proj[slot] / (W, H)
    return full


class OracleModel(torch.nn.Module):
    """Emits whatever pitch/yaw a perfect network would have produced."""

    def __init__(self):
        super().__init__()
        self.next_pitch_yaw = (0.0, 0.0)

    def forward(self, person_idx, face, right_eye, left_eye):
        return torch.tensor([list(self.next_pitch_yaw)], dtype=torch.float32)


def look_at(tracker, model, rvec, tvec, target, noise_deg=0.0, rng=None):
    """Aim the simulated eyes at `target` and push one frame through."""
    lm = synth_landmarks(rvec, tvec)
    px = np.stack([lm[i] * (W, H) for i in LANDMARK_IDS])
    pose = solve_head_pose(px, K, DIST)
    face_center = pose.face_center()
    _, face_rot = normalize_image(IMAGE, pose.rotation_matrix, face_center, K, is_eye=False)

    direction = target - face_center.reshape(3)
    direction = direction / np.linalg.norm(direction)
    if noise_deg and rng is not None:
        direction = direction + rng.normal(0, np.deg2rad(noise_deg), 3)
        direction = direction / np.linalg.norm(direction)

    # Invert gaze_2d_to_3d in the normalized frame the tracker will un-rotate.
    v = face_rot @ direction
    model.next_pitch_yaw = (
        float(np.arcsin(np.clip(-v[1], -1, 1))),
        float(np.arctan2(-v[0], -v[2])),
    )
    return tracker.update(IMAGE, lm)


@pytest.fixture(scope="module")
def calibrated():
    """Run the corner-look calibration against the simulated monitor once."""
    model = OracleModel()
    tracker = GazeTracker(model, K, DIST, [], torch.device("cpu"),
                          landmark_smoothing=1, gaze_smoothing=1)
    collector = CornerCollector("sim-27", PIXELS)
    for rvec, tvec in HEAD_POSES:
        for idx, (u, v) in enumerate(CORNER_UV):
            reading = look_at(tracker, model, rvec, tvec, target_3d(u, v))
            if reading.ok and reading.ray_origin is not None:
                collector.add(idx, reading.ray_origin, reading.ray_direction)
    assert collector.ready(), collector.progress()
    return collector.solve()


def test_calibration_recovers_the_panel_size(calibrated):
    plane, report = calibrated
    assert abs(plane.width_mm - MON_W) < 5.0
    assert abs(plane.height_mm - MON_H) < 5.0
    assert abs(report["diagonal_in"] - 27.0) < 0.5


def test_calibration_is_clean(calibrated):
    plane, report = calibrated
    assert sanity_check(plane, report) == []


@pytest.mark.parametrize("u,v", [(0.5, 0.5), (0.25, 0.75), (0.8, 0.2), (0.1, 0.9), (0.95, 0.55)])
def test_tracking_through_the_recovered_plane_is_accurate(calibrated, u, v):
    plane, _ = calibrated
    model = OracleModel()
    tracker = GazeTracker(model, K, DIST, [plane], torch.device("cpu"),
                          landmark_smoothing=1, gaze_smoothing=1)

    reading = look_at(tracker, model, *HEAD_POSES[0], target_3d(u, v))

    assert reading.on_any_screen
    err_px = np.linalg.norm(np.array(reading.screen_xy) - [u * PIXELS[0], v * PIXELS[1]])
    assert err_px * (MON_W / PIXELS[0]) < 5.0      # millimetres


def test_the_recovered_plane_is_not_mirrored_or_rotated(calibrated):
    """The most likely silent failure: a plane fit that is geometrically valid
    but flipped. Both halves would pass their own tests while composing wrong."""
    plane, _ = calibrated
    model = OracleModel()
    tracker = GazeTracker(model, K, DIST, [plane], torch.device("cpu"),
                          landmark_smoothing=1, gaze_smoothing=1)

    top_left = look_at(tracker, model, *HEAD_POSES[0], target_3d(0.05, 0.05))
    bottom_right = look_at(tracker, model, *HEAD_POSES[0], target_3d(0.95, 0.95))

    assert top_left.screen_xy[0] < PIXELS[0] * 0.25
    assert top_left.screen_xy[1] < PIXELS[1] * 0.25
    assert bottom_right.screen_xy[0] > PIXELS[0] * 0.75
    assert bottom_right.screen_xy[1] > PIXELS[1] * 0.75


def test_realistic_model_error_is_caught_rather_than_absorbed():
    """At a stock checkpoint's ~2.4 degrees the corner residual lands right at
    the strict limit, so calibration refuses. That refusal is the feature --
    and it is why tools/calibrate_screens.py has a --bootstrap mode."""
    from gaze.geometry import CalibrationError

    rng = np.random.default_rng(3)
    model = OracleModel()
    tracker = GazeTracker(model, K, DIST, [], torch.device("cpu"),
                          landmark_smoothing=1, gaze_smoothing=1)
    collector = CornerCollector("noisy", PIXELS)
    for rvec, tvec in HEAD_POSES:
        for _ in range(4):
            for idx, (u, v) in enumerate(CORNER_UV):
                reading = look_at(tracker, model, rvec, tvec, target_3d(u, v),
                                  noise_deg=2.4, rng=rng)
                if reading.ok and reading.ray_origin is not None:
                    collector.add(idx, reading.ray_origin, reading.ray_direction)

    with pytest.raises(CalibrationError, match="disagree"):
        collector.solve()

    # ...but the bootstrap threshold still yields something usable to start from.
    plane, report = collector.solve(max_residual_mm=200.0)
    assert 15.0 < report["diagonal_in"] < 40.0
    assert plane.width_mm > 0
