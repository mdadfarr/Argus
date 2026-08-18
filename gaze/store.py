"""Persistence for the calibration artifacts, under `state/`.

Three things have to survive a restart: the camera intrinsics, the screen
planes, and which checkpoint they were solved against. They are stored
together because they are only valid together -- intrinsics solved at one
resolution and planes solved against different intrinsics silently produce a
plausible-looking gaze point that is simply wrong.

`load_calibration` therefore validates rather than trusts, in the same spirit
as `vision._baseline_from`: refuse a calibration that cannot be right instead
of running blind on it.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import CalibrationError, ScreenPlane

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
# Bump when a change would invalidate stored planes (new normalization
# constants, different face model, changed corner ordering).
CALIBRATION_EPOCH = 1


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    capture_size: tuple[int, int]
    screens: list[ScreenPlane]
    checkpoint_sha256: str | None = None
    created_at: str | None = None

    def screen(self, name: str) -> ScreenPlane | None:
        return next((s for s in self.screens if s.name == name), None)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plane_to_dict(p: ScreenPlane) -> dict:
    return {
        "name": p.name,
        "origin": [float(v) for v in np.asarray(p.origin).reshape(3)],
        "x_axis": [float(v) for v in np.asarray(p.x_axis).reshape(3)],
        "y_axis": [float(v) for v in np.asarray(p.y_axis).reshape(3)],
        "width_mm": float(p.width_mm),
        "height_mm": float(p.height_mm),
        "pixels": [int(p.pixels[0]), int(p.pixels[1])],
        "global_origin": [int(p.global_origin[0]), int(p.global_origin[1])],
    }


def _plane_from_dict(d: dict) -> ScreenPlane:
    return ScreenPlane(
        name=d["name"],
        origin=np.asarray(d["origin"], dtype=float),
        x_axis=np.asarray(d["x_axis"], dtype=float),
        y_axis=np.asarray(d["y_axis"], dtype=float),
        width_mm=float(d["width_mm"]),
        height_mm=float(d["height_mm"]),
        pixels=(int(d["pixels"][0]), int(d["pixels"][1])),
        global_origin=(int(d["global_origin"][0]), int(d["global_origin"][1])),
    )


def save_calibration(path: str | Path, cal: Calibration) -> None:
    """Write atomically -- a half-written calibration read back on the next
    launch would be worse than none, because the failure would surface as
    inexplicably bad tracking rather than as a parse error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "epoch": CALIBRATION_EPOCH,
        "created_at": cal.created_at,
        "capture_size": [int(cal.capture_size[0]), int(cal.capture_size[1])],
        "camera_matrix": np.asarray(cal.camera_matrix, dtype=float).reshape(3, 3).tolist(),
        "dist_coeffs": np.asarray(cal.dist_coeffs, dtype=float).ravel().tolist(),
        "checkpoint_sha256": cal.checkpoint_sha256,
        "screens": [_plane_to_dict(s) for s in cal.screens],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    log.info("wrote calibration for %d screen(s) to %s", len(cal.screens), path)


def load_calibration(
    path: str | Path,
    expect_capture_size: tuple[int, int] | None = None,
    expect_checkpoint: str | Path | None = None,
) -> Calibration:
    """Read and validate. Raises CalibrationError on anything suspect.

    The capture-size check is the one that matters most in practice: intrinsics
    are in pixels, so running a 1280x720 calibration against a 640x480 capture
    scales every angle wrongly while still producing confident output.
    """
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f"no calibration at {path} -- run tools/calibrate_screens.py")

    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CalibrationError(f"{path} is corrupt: {e}") from e

    if d.get("epoch") != CALIBRATION_EPOCH:
        raise CalibrationError(
            f"{path} was written by calibration epoch {d.get('epoch')}, this build needs "
            f"{CALIBRATION_EPOCH}. Recalibrate."
        )

    capture_size = (int(d["capture_size"][0]), int(d["capture_size"][1]))
    if expect_capture_size and tuple(expect_capture_size) != capture_size:
        raise CalibrationError(
            f"calibration was made at {capture_size[0]}x{capture_size[1]} but the camera is "
            f"delivering {expect_capture_size[0]}x{expect_capture_size[1]}. The intrinsics are "
            "in pixels, so they do not transfer -- recalibrate at the capture resolution."
        )

    screens = [_plane_from_dict(s) for s in d.get("screens", [])]
    if not screens:
        raise CalibrationError(f"{path} contains no screens")

    if expect_checkpoint is not None:
        actual = sha256_file(expect_checkpoint)
        stored = d.get("checkpoint_sha256")
        if stored and stored != actual:
            raise CalibrationError(
                "the model checkpoint changed since this calibration was made. The "
                "per-user bias no longer matches -- recalibrate."
            )

    return Calibration(
        camera_matrix=np.asarray(d["camera_matrix"], dtype=float).reshape(3, 3),
        dist_coeffs=np.asarray(d["dist_coeffs"], dtype=float).ravel(),
        capture_size=capture_size,
        screens=screens,
        checkpoint_sha256=d.get("checkpoint_sha256"),
        created_at=d.get("created_at"),
    )


# ---------- status, for the UI ----------

# What the app needs to know is not "did it load" but "what should I tell the
# user to do about it", so this returns a state plus a sentence rather than a
# bool. A `DEGRADED: GAZE` banner that cannot distinguish "torch missing" from
# "never calibrated" sends people to the wrong fix.
STATUS_OK = "ok"
STATUS_MISSING = "missing"          # never calibrated, or the file is empty
STATUS_STALE = "stale"              # exists but no longer valid for this setup
STATUS_NO_DEPS = "no_deps"          # torch et al not installed
STATUS_DISABLED = "disabled"        # detect_gaze is off


def calibration_status(
    path: str | Path,
    capture_size: tuple[int, int] | None = None,
    checkpoint: str | Path | None = None,
) -> dict:
    """Describe the calibration without raising, for display in the UI.

    Returns {state, message, screens, created_at}. `screens` is the list of
    calibrated monitor names, which is what makes the difference between "you
    calibrated" and "you calibrated the screen you are actually sitting at"
    visible to someone with three monitors.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {
            "state": STATUS_MISSING,
            "message": "Not calibrated yet. Run calibration to tell Argus where your screens are.",
            "screens": [],
            "created_at": None,
        }

    try:
        cal = load_calibration(path, expect_capture_size=capture_size, expect_checkpoint=checkpoint)
    except CalibrationError as e:
        return {
            "state": STATUS_STALE,
            "message": f"Calibration needs redoing: {e}",
            "screens": [],
            "created_at": None,
        }

    names = [s.name for s in cal.screens]
    return {
        "state": STATUS_OK,
        "message": f"Calibrated for {len(names)} screen(s): {', '.join(names)}",
        "screens": names,
        "created_at": cal.created_at,
    }
