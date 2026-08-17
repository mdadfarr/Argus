"""Geometry checks for the gaze pipeline: no camera, no checkpoint, no display.

The equivalence tests against pperle's original preprocessing are the important
ones here. The geometry in `gaze/` is a reimplementation (upstream's pipeline
repo ships no LICENSE), so "does it still produce the exact crops the weights
were trained on" is the property that actually matters -- and it is the one that
already caught a missing histogram-equalization step during the port.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("yaml")

from gaze.geometry import (  # noqa: E402
    FACE_MODEL_7,
    CalibrationError,
    gaze_2d_to_3d,
    normalize_image,
    plane_from_corners,
    scale_intrinsics,
    solve_head_pose,
)

W, H = 1280, 720
K = np.array([[960.0, 0.0, W / 2], [0.0, 960.0, H / 2], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)


@pytest.fixture
def noise_image():
    return np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)


# ---------- head pose ----------

def test_head_pose_round_trips():
    """Project the model with a known pose, then recover that pose from the
    projection. Any error here poisons every millimetre downstream."""
    rvec = np.array([[0.15], [0.25], [-0.1]])
    tvec = np.array([[30.0], [10.0], [550.0]])
    proj, _ = cv2.projectPoints(FACE_MODEL_7, rvec, tvec, K, DIST)

    pose = solve_head_pose(proj.reshape(-1, 2), K, DIST)

    assert np.allclose(pose.tvec, tvec, atol=0.5)
    rmat = cv2.Rodrigues(rvec.reshape(-1))[0]
    assert np.allclose(pose.landmarks_ccs.T, (rmat @ FACE_MODEL_7.T + tvec).T, atol=0.5)


def test_eye_centres_are_anatomically_plausible():
    proj, _ = cv2.projectPoints(FACE_MODEL_7, np.zeros((3, 1)), np.array([[0.0], [0.0], [600.0]]), K, DIST)
    pose = solve_head_pose(proj.reshape(-1, 2), K, DIST)

    right, left = pose.eye_centers()
    # Adult interpupillary distance is ~63mm; well outside this range means the
    # corner landmarks got swapped.
    assert 55 < float(np.linalg.norm(right - left)) < 75


def test_head_pose_rejects_wrong_shape():
    with pytest.raises(ValueError):
        solve_head_pose(np.zeros((4, 2)), K, DIST)


# ---------- normalization ----------

@pytest.mark.parametrize("is_eye,expected_shape", [(True, (64, 96, 3)), (False, (96, 96, 3))])
def test_normalized_crop_shape(noise_image, is_eye, expected_shape):
    rvec = np.array([[0.1], [-0.2], [0.05]])
    warped, _ = normalize_image(
        noise_image, cv2.Rodrigues(rvec.reshape(-1))[0],
        np.array([[20.0], [-15.0], [600.0]]), K, is_eye=is_eye,
    )
    assert warped.shape == expected_shape


def test_normalization_applies_histogram_equalization(noise_image):
    """Upstream equalizes luma before the crop leaves preprocessing, and the
    weights were fitted on equalized input. Omitting it is a silent accuracy
    regression -- the model still returns confident angles, just worse ones."""
    warped, _ = normalize_image(
        noise_image, np.eye(3), np.array([[0.0], [0.0], [600.0]]), K, is_eye=True,
    )
    luma = cv2.cvtColor(warped, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    # Equalized luma spreads across the full range rather than clustering.
    assert luma.min() < 30 and luma.max() > 225


def test_matches_upstream_preprocessing_exactly(noise_image):
    """Bit-for-bit equivalence with pperle's `normalize_single_image`.

    Skipped unless the upstream repo is checked out next to this one -- it is
    not vendored, deliberately, since that half ships no LICENSE.
    """
    upstream = pytest.importorskip(
        "mpii_face_gaze_preprocessing",
        reason="clone pperle/gaze-tracking-pipeline alongside to run the equivalence check",
    )
    rvec = np.array([[0.1], [-0.2], [0.05]])
    center = np.array([[20.0], [-15.0], [600.0]])

    for is_eye in (True, False):
        mine, rot_mine = normalize_image(
            noise_image, cv2.Rodrigues(rvec.reshape(-1))[0], center, K, is_eye=is_eye)
        theirs, _, rot_theirs = upstream.normalize_single_image(
            noise_image, rvec, None, center, K, is_eye=is_eye)
        assert np.array_equal(mine, theirs)
        assert np.allclose(rot_mine, rot_theirs, atol=1e-12)


# ---------- gaze angles ----------

def test_zero_angles_point_away_from_camera():
    assert np.allclose(gaze_2d_to_3d(np.array([0.0, 0.0])), [0, 0, -1])


def test_gaze_vector_is_unit_length():
    assert abs(np.linalg.norm(gaze_2d_to_3d(np.array([0.3, -0.2]))) - 1) < 1e-12


# ---------- screen planes ----------

@pytest.fixture
def yawed_monitor():
    """600x340mm panel, 300mm out, yawed 20 degrees, 1920x1080."""
    w, h = 600.0, 340.0
    yaw = np.deg2rad(20)
    rot = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    tl = np.array([-w / 2, -h / 2, 300.0])
    corners = np.array([tl, tl + rot @ [w, 0, 0], tl + rot @ [w, h, 0], tl + rot @ [0, h, 0]])
    return plane_from_corners("test", corners, (1920, 1080)), corners, (w, h)


def test_plane_recovers_physical_size(yawed_monitor):
    plane, _, (w, h) = yawed_monitor
    assert abs(plane.width_mm - w) < 0.5
    assert abs(plane.height_mm - h) < 0.5


def test_plane_axes_are_orthonormal(yawed_monitor):
    plane, _, _ = yawed_monitor
    assert abs(np.dot(plane.x_axis, plane.y_axis)) < 1e-9
    assert abs(np.linalg.norm(plane.x_axis) - 1) < 1e-9


def test_screen_centre_maps_to_centre_pixel(yawed_monitor):
    plane, corners, _ = yawed_monitor
    eye = np.zeros(3)
    centre = corners.mean(axis=0)
    px = plane.hit(eye, centre - eye)
    assert px is not None
    assert abs(px[0] - 960) < 1 and abs(px[1] - 540) < 1


@pytest.mark.parametrize("idx,want", [(0, (0, 0)), (1, (1920, 0)), (2, (1920, 1080)), (3, (0, 1080))])
def test_each_corner_maps_to_its_pixel_corner(yawed_monitor, idx, want):
    plane, corners, _ = yawed_monitor
    px = plane.hit(np.zeros(3), corners[idx])
    assert px is not None
    assert abs(px[0] - want[0]) < 2 and abs(px[1] - want[1]) < 2


def test_ray_past_the_bezel_is_a_miss(yawed_monitor):
    plane, corners, (w, h) = yawed_monitor
    far_off = corners[0] + (corners[1] - corners[0]) * 2.5
    assert plane.hit(np.zeros(3), far_off) is None


def test_ray_pointing_backwards_is_a_miss(yawed_monitor):
    """Without the t>0 guard this returns a hit behind the viewer's head."""
    plane, _, _ = yawed_monitor
    assert plane.hit(np.zeros(3), np.array([0.0, 0.0, -1.0])) is None


def test_ray_parallel_to_the_panel_is_a_miss(yawed_monitor):
    plane, _, _ = yawed_monitor
    assert plane.hit(np.zeros(3), plane.x_axis) is None


def test_degenerate_corners_are_refused():
    with pytest.raises(CalibrationError):
        plane_from_corners("flat", np.zeros((4, 3)), (1920, 1080))


# ---------- intrinsics ----------

def test_scaling_intrinsics_scales_focal_and_centre():
    scaled = scale_intrinsics(K, (1280, 720), (640, 360))
    assert abs(scaled[0, 0] - 480) < 1e-9
    assert abs(scaled[0, 2] - 320) < 1e-9


def test_aspect_ratio_change_is_refused():
    """640x480 is not a scaled 1280x720 -- the sensor was cropped, and no
    single factor is correct. Silently allowing it would mis-scale every angle."""
    with pytest.raises(CalibrationError):
        scale_intrinsics(K, (1280, 720), (640, 480))
