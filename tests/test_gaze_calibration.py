"""Corner-look screen calibration, driven from simulated gaze rays.

Every test here places a monitor of known size and orientation, generates the
rays a person looking at its corners would produce, and checks what comes back.
The refusal tests matter as much as the recovery ones: a calibration that is
quietly wrong produces a gaze point that looks reasonable and is not, which is
far more expensive to debug than one that refused up front.
"""
from __future__ import annotations

import numpy as np
import pytest

from gaze.calibrate import (
    MIN_PARALLAX_DEG,
    CornerCollector,
    Ray,
    parallax_degrees,
    sanity_check,
    triangulate,
)
from gaze.geometry import CalibrationError

# A 27in 16:9 panel, 420mm out, yawed 25 degrees and tilted back 8.
MON_W, MON_H = 597.0, 336.0
_yaw, _tilt = np.deg2rad(25), np.deg2rad(-8)
_Ry = np.array([[np.cos(_yaw), 0, np.sin(_yaw)], [0, 1, 0], [-np.sin(_yaw), 0, np.cos(_yaw)]])
_Rx = np.array([[1, 0, 0], [0, np.cos(_tilt), -np.sin(_tilt)], [0, np.sin(_tilt), np.cos(_tilt)]])
_R = _Ry @ _Rx
_TL = np.array([-250.0, -180.0, 420.0])
TRUE_CORNERS = np.array([_TL, _TL + _R @ [MON_W, 0, 0], _TL + _R @ [MON_W, MON_H, 0], _TL + _R @ [0, MON_H, 0]])

# Four seated positions. The spread between them IS the depth measurement.
HEAD_POSITIONS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([90.0, -20.0, 40.0]),
    np.array([-80.0, 25.0, -30.0]),
    np.array([20.0, 60.0, 60.0]),
]


def build_collector(noise_deg=0.0, positions=None, seed=7, repeats=1):
    rng = np.random.default_rng(seed)
    collector = CornerCollector("sim-27in", (2560, 1440))
    for eye in positions if positions is not None else HEAD_POSITIONS:
        for _ in range(repeats):
            for i in range(4):
                d = TRUE_CORNERS[i] - eye
                d = d / np.linalg.norm(d)
                if noise_deg:
                    d = d + rng.normal(0, np.deg2rad(noise_deg), 3)
                    d = d / np.linalg.norm(d)
                collector.add(i, eye, d)
    return collector


def solved_corners(plane):
    return np.array([
        plane.origin,
        plane.origin + plane.x_axis * plane.width_mm,
        plane.origin + plane.x_axis * plane.width_mm + plane.y_axis * plane.height_mm,
        plane.origin + plane.y_axis * plane.height_mm,
    ])


# ---------- recovery ----------

def test_noiseless_calibration_recovers_the_monitor_exactly():
    plane, _ = build_collector().solve()
    assert np.max(np.linalg.norm(solved_corners(plane) - TRUE_CORNERS, axis=1)) < 1.0


def test_recovers_physical_dimensions():
    plane, report = build_collector().solve()
    assert abs(plane.width_mm - MON_W) < 1.0
    assert abs(plane.height_mm - MON_H) < 1.0
    assert abs(report["diagonal_in"] - 27.0) < 0.3


def test_clean_calibration_raises_no_warnings():
    plane, report = build_collector().solve()
    assert sanity_check(plane, report) == []


def test_survives_realistic_landmark_noise():
    """Half a degree is roughly the model's frame-to-frame jitter once
    smoothed, as opposed to its absolute error."""
    plane, report = build_collector(noise_deg=0.5).solve()
    assert abs(plane.width_mm - MON_W) < 60
    assert 20 < report["diagonal_in"] < 35


# ---------- refusals ----------

def test_refuses_without_parallax():
    """Four corners from one fixed head position gives four lines with nothing
    to intersect against. The depth is unconstrained and the solve would return
    a confident answer that can be off by tens of centimetres."""
    stationary = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 2.0])]
    with pytest.raises(CalibrationError, match="parallax"):
        build_collector(positions=stationary).solve()


def test_refuses_incomplete_collection():
    collector = CornerCollector("partial", (1920, 1080))
    collector.add(0, [0, 0, 0], [0, 0, 1])
    collector.add(1, [0, 0, 0], [0, 0, 1])

    assert not collector.ready()
    with pytest.raises(CalibrationError):
        collector.solve()


def test_refuses_rays_that_disagree():
    with pytest.raises(CalibrationError, match="disagree"):
        triangulate([
            Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            Ray(np.array([500.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            Ray(np.array([0.0, 500.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        ])


def test_refuses_too_few_samples():
    with pytest.raises(CalibrationError, match="at least"):
        triangulate([Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]))])


def test_rejects_out_of_range_corner_index():
    with pytest.raises(ValueError):
        CornerCollector("x", (1920, 1080)).add(4, [0, 0, 0], [0, 0, 1])


def test_a_looser_threshold_can_accept_what_the_strict_one_refuses():
    """The bootstrap path: a stock checkpoint's error sits right at the strict
    limit, so the first calibration of a fresh install needs slack."""
    collector = build_collector(noise_deg=2.4, repeats=4, seed=3)
    with pytest.raises(CalibrationError, match="disagree"):
        collector.solve(max_residual_mm=5.0)
    plane, report = collector.solve(max_residual_mm=200.0)
    assert report["diagonal_in"] > 0
    assert plane.width_mm > 0


# ---------- diagnostics ----------

def test_progress_reports_per_corner_counts():
    collector = CornerCollector("x", (1920, 1080))
    collector.add(0, [0, 0, 0], [0, 0, 1])
    assert collector.progress()["top-left"] == 1
    assert collector.progress()["bottom-right"] == 0


def test_parallax_is_zero_for_identical_rays():
    rays = [Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]))] * 3
    assert parallax_degrees(rays) < 1e-9


def test_parallax_gate_constant_is_meaningful():
    assert MIN_PARALLAX_DEG > 0


def test_report_includes_flatness_and_squareness():
    """A real panel is flat and rectangular, so both are independent checks on
    whether the corners were triangulated sensibly."""
    _, report = build_collector().solve()
    assert report["flatness_mm"] < 1.0
    assert report["squareness_deg"] < 1.0
    assert set(report["corners"]) == {"top-left", "top-right", "bottom-right", "bottom-left"}


def test_sanity_check_flags_an_implausible_diagonal():
    plane, report = build_collector().solve()
    report["diagonal_in"] = 3.0
    assert any("plausible monitor" in w for w in sanity_check(plane, report))
