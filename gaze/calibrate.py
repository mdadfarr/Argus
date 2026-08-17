"""Corner-look screen calibration: locate each monitor in camera space by
having the user stare at its four corners.

This is the piece pperle's pipeline leaves unwritten (`# TODO load calibrated
screen position` in upstream `main.py`, with the plane hardcoded to the
camera's own x-y plane). Without it there is no meaningful multi-monitor
support, because every monitor is assumed to be the one the camera is bolted to.

How it works
------------
Looking at one corner once is not enough. A single gaze ray only says "the
corner is somewhere along this line" -- it cannot say how far. But sample the
same corner from two or more *different head positions* and the rays converge
on the corner's 3D location. That is plain triangulation, and it is why
`collect_corner` asks the user to shift position between samples.

The failure mode to respect: if the head does not move, the rays are nearly
parallel, the least-squares solve is ill-conditioned, and it returns a
confident-looking point that can be off by tens of centimetres.
`triangulate` measures that conditioning and refuses rather than guessing --
the same stance `vision.py:_baseline_from` already takes toward a head that
was still moving during calibration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .geometry import CalibrationError, ScreenPlane, plane_from_corners

log = logging.getLogger(__name__)

# Rays whose directions span less than this are treated as parallel. At ~600mm
# viewing distance, 5 degrees of parallax puts the depth error in the low
# centimetres; below that it grows fast enough to be worthless.
MIN_PARALLAX_DEG = 5.0
MIN_RAYS_PER_CORNER = 3
# If the rays do not actually meet this closely, the samples disagree -- the
# user looked somewhere else, or the gaze model is off for this pose.
#
# Simulation puts a stock checkpoint's 2.4 deg error right at this boundary
# (~45mm rms), which means the first calibration of a fresh install is likely
# to be refused. That is the correct default -- but it forces an ordering:
# bootstrap with BOOTSTRAP_RESIDUAL_MM and approximate screen geometry, run the
# per-user calibration to pull the model's error down, then recalibrate the
# screens properly at the strict threshold. See README, "ordering problem".
MAX_RESIDUAL_MM = 40.0
BOOTSTRAP_RESIDUAL_MM = 90.0

CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray      # (3,) eye/face centre in camera space, mm
    direction: np.ndarray   # (3,) unit gaze direction


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise CalibrationError("zero-length gaze direction")
    return v / n


def parallax_degrees(rays: list[Ray]) -> float:
    """Widest angle between any two ray directions in the set."""
    dirs = [_unit(r.direction) for r in rays]
    worst = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            cos = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            worst = max(worst, np.degrees(np.arccos(cos)))
    return worst


def triangulate(rays: list[Ray], max_residual_mm: float = MAX_RESIDUAL_MM) -> tuple[np.ndarray, float]:
    """Least-squares closest point to a bundle of rays.

    Minimizes the sum of squared perpendicular distances to every ray, which
    has the closed form  (sum of I - dd^T) x = sum of (I - dd^T) o.

    Returns (point, rms_residual_mm). Raises if there is too little parallax
    to make the solve meaningful, or if the rays simply do not agree.
    """
    if len(rays) < MIN_RAYS_PER_CORNER:
        raise CalibrationError(f"need at least {MIN_RAYS_PER_CORNER} samples, got {len(rays)}")

    spread = parallax_degrees(rays)
    if spread < MIN_PARALLAX_DEG:
        raise CalibrationError(
            f"only {spread:.1f} deg of parallax between samples (need {MIN_PARALLAX_DEG}). "
            "Move your head between samples -- without that the depth is unconstrained."
        )

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for r in rays:
        d = _unit(r.direction)
        o = np.asarray(r.origin, dtype=float).reshape(3)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o

    # A is singular exactly when every direction is identical; the parallax
    # gate above already excludes that, but a near-singular solve still
    # deserves the pseudo-inverse rather than an exception from lstsq.
    point, *_ = np.linalg.lstsq(A, b, rcond=None)

    residuals = []
    for r in rays:
        d = _unit(r.direction)
        o = np.asarray(r.origin, dtype=float).reshape(3)
        v = point - o
        residuals.append(float(np.linalg.norm(v - np.dot(v, d) * d)))
    rms = float(np.sqrt(np.mean(np.square(residuals))))

    if rms > max_residual_mm:
        raise CalibrationError(
            f"samples disagree by {rms:.0f} mm rms (limit {max_residual_mm:.0f}). "
            "Either the gaze was not on the target, or the model is not calibrated for you yet."
        )
    return point, rms


@dataclass
class CornerCollector:
    """Accumulates rays for the four corners of one monitor."""

    monitor_name: str
    pixels: tuple[int, int]
    _rays: dict[int, list[Ray]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._rays is None:
            self._rays = {i: [] for i in range(4)}

    def add(self, corner_idx: int, origin: np.ndarray, direction: np.ndarray) -> int:
        """Record one look at `corner_idx` (0=TL, 1=TR, 2=BR, 3=BL). Returns
        how many samples that corner now has."""
        if corner_idx not in (0, 1, 2, 3):
            raise ValueError(f"corner_idx must be 0..3, got {corner_idx}")
        self._rays[corner_idx].append(Ray(np.asarray(origin, float).reshape(3), _unit(direction)))
        return len(self._rays[corner_idx])

    def ready(self) -> bool:
        return all(len(v) >= MIN_RAYS_PER_CORNER for v in self._rays.values())

    def progress(self) -> dict[str, int]:
        return {CORNER_NAMES[i]: len(self._rays[i]) for i in range(4)}

    def solve(self, max_residual_mm: float = MAX_RESIDUAL_MM) -> tuple[ScreenPlane, dict]:
        """Triangulate all four corners and fit the plane.

        Reports per-corner diagnostics alongside the plane so a marginal
        calibration can be shown to the user rather than silently accepted.
        """
        if not self.ready():
            raise CalibrationError(f"incomplete: {self.progress()}")

        corners = np.zeros((4, 3))
        report: dict = {"monitor": self.monitor_name, "corners": {}}
        for i in range(4):
            point, rms = triangulate(self._rays[i], max_residual_mm)
            corners[i] = point
            report["corners"][CORNER_NAMES[i]] = {
                "samples": len(self._rays[i]),
                "parallax_deg": round(parallax_degrees(self._rays[i]), 1),
                "residual_mm": round(rms, 1),
            }

        plane = plane_from_corners(self.monitor_name, corners, self.pixels)
        report["width_mm"] = round(plane.width_mm, 1)
        report["height_mm"] = round(plane.height_mm, 1)
        report["diagonal_in"] = round(float(np.hypot(plane.width_mm, plane.height_mm)) / 25.4, 1)
        report["flatness_mm"] = round(_flatness(corners), 1)
        report["squareness_deg"] = round(_squareness(corners), 1)
        return plane, report


def _flatness(corners: np.ndarray) -> float:
    """How far the four corners deviate from a single plane, in mm. A real
    monitor is flat, so a large value means a corner was triangulated badly."""
    centroid = corners.mean(axis=0)
    _, s, vh = np.linalg.svd(corners - centroid)
    normal = vh[2]
    return float(np.max(np.abs((corners - centroid) @ normal)))


def _squareness(corners: np.ndarray) -> float:
    """Worst deviation from 90 degrees at the four corners. A monitor is a
    rectangle; if this is large the corner order was probably wrong."""
    worst = 0.0
    for i in range(4):
        a = corners[(i - 1) % 4] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst = max(worst, abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90.0))
    return worst


def sanity_check(plane: ScreenPlane, report: dict) -> list[str]:
    """Human-readable warnings. Empty list means the calibration looks sound.

    Deliberately returns warnings instead of raising: unlike a head-pose
    baseline, a slightly-off screen plane is still far better than upstream's
    hardcoded one, so the caller should be able to accept it knowingly.
    """
    warnings: list[str] = []
    diag = report["diagonal_in"]
    if not (10.0 <= diag <= 60.0):
        warnings.append(f"solved diagonal is {diag}in, which is not a plausible monitor size")
    if report["flatness_mm"] > 25.0:
        warnings.append(f"corners are {report['flatness_mm']}mm from coplanar; a monitor is flat")
    if report["squareness_deg"] > 12.0:
        warnings.append(f"corners are {report['squareness_deg']}deg from square; check the corner order")
    aspect = plane.width_mm / plane.height_mm
    px_aspect = plane.pixels[0] / plane.pixels[1]
    if abs(aspect - px_aspect) > 0.25:
        warnings.append(
            f"physical aspect {aspect:.2f} disagrees with pixel aspect {px_aspect:.2f}"
        )
    for name, c in report["corners"].items():
        if c["residual_mm"] > 20.0:
            warnings.append(f"{name}: rays converge only to {c['residual_mm']}mm")
    return warnings
