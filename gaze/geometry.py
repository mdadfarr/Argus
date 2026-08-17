"""Camera geometry for gaze: head pose, MPIIGaze normalization, screen planes.

Reimplemented rather than copied. pperle's `gaze-tracking` (the model) is MIT,
but `gaze-tracking-pipeline` (this half) ships no LICENSE file, which means all
rights reserved -- fine to read, not to vendor. The maths underneath is
published anyway: the eye-image normalization is Sugano et al. 2014 as revised
by Zhang et al. 2018, and the rest is textbook solvePnP plus a ray/plane solve.

Everything is millimetres in the camera coordinate system: +x right, +y down,
+z away from the camera (OpenCV convention).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# Seven landmarks from MediaPipe's canonical face model (Apache-2.0), already
# axis-corrected (y and z negated) and scaled from cm to mm. Eye corners first
# because `eye_centers()` indexes them positionally.
#   33/133  = right eye outer/inner      362/263 = left eye inner/outer
#   61/291  = mouth corners              1       = nose tip
LANDMARK_IDS = (33, 133, 362, 263, 61, 291, 1)

FACE_MODEL_7 = np.array([
    [-44.45859, -26.63991, -31.73422],  # 33
    [-18.56432, -25.85245, -37.57904],  # 133
    [ 18.56432, -25.85245, -37.57904],  # 362
    [ 44.45859, -26.63991, -31.73422],  # 263
    [-24.56206,  43.42621, -42.83884],  # 61
    [ 24.56206,  43.42621, -42.83884],  # 291
    [  0.00000,  11.26865, -74.75604],  # 1
], dtype=float)

# Normalization constants. These are NOT tunable: the model was trained on
# crops produced with exactly these numbers, so changing one silently shifts
# the input distribution away from what the weights expect.
FOCAL_NORM = 960.0
DISTANCE_NORM_EYE = 500.0
DISTANCE_NORM_FACE = 1600.0
SIZE_EYE = (96, 64)    # (w, h)
SIZE_FACE = (96, 96)


class CalibrationError(Exception):
    pass


# ---------- camera intrinsics ----------

def load_camera_matrix(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `calibration_matrix.yaml` as written by the chessboard calibration.

    The intrinsics are resolution-dependent: fx, fy, cx and cy are in pixels,
    so a matrix solved at 1280x720 is simply wrong at 640x480. `scale_intrinsics`
    exists for that, but re-running the calibration at the resolution you will
    actually capture is better.
    """
    data = yaml.safe_load(Path(path).read_text())
    try:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=float).reshape(3, 3)
        dist = np.asarray(data["dist_coeff"], dtype=float).ravel()
    except (KeyError, ValueError) as e:
        raise CalibrationError(f"{path} is not a camera calibration file: {e}") from e
    return camera_matrix, dist


def scale_intrinsics(camera_matrix: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    """Rescale intrinsics between resolutions of the *same aspect ratio*.

    Refuses on an aspect change, because that means the sensor was cropped
    rather than scaled and no single factor is correct.
    """
    sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
    if abs(sx - sy) > 1e-3:
        raise CalibrationError(
            f"cannot rescale intrinsics from {from_wh} to {to_wh}: aspect ratio "
            f"changes ({sx:.4f} vs {sy:.4f}). Recalibrate at the target resolution."
        )
    out = camera_matrix.copy()
    out[0, :] *= sx
    out[1, :] *= sy
    return out


# ---------- head pose ----------

@dataclass(frozen=True)
class HeadPose:
    rvec: np.ndarray            # (3,1) Rodrigues rotation, camera <- head
    tvec: np.ndarray            # (3,1) translation in mm
    landmarks_ccs: np.ndarray   # (3,7) the 7 model points in camera space

    @property
    def rotation_matrix(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec.reshape(-1))[0]

    def eye_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) eye centres, each (3,1), as the midpoint of that eye's
        two corners -- which is what the model's crops were centred on."""
        right = 0.5 * (self.landmarks_ccs[:, 0] + self.landmarks_ccs[:, 1])
        left = 0.5 * (self.landmarks_ccs[:, 2] + self.landmarks_ccs[:, 3])
        return right.reshape(3, 1), left.reshape(3, 1)

    def face_center(self) -> np.ndarray:
        return self.landmarks_ccs.mean(axis=1).reshape(3, 1)


def solve_head_pose(
    landmarks_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    prev: HeadPose | None = None,
    refine_iters: int = 10,
) -> HeadPose:
    """Fit the 7-point face model to detected pixels.

    Two-stage on purpose: EPnP with RANSAC to get a robust starting point that
    a single bad landmark cannot drag off, then ITERATIVE refinement seeded
    from it. Seeding from the previous frame's pose (`prev`) both speeds up
    convergence and keeps the solution temporally stable -- without it the
    solver can flip between two poses that fit almost equally well, which
    reads downstream as the gaze point teleporting.

    `landmarks_px` is (7, 2) in pixels, ordered to match LANDMARK_IDS.
    """
    if landmarks_px.shape != (7, 2):
        raise ValueError(f"expected (7, 2) landmarks, got {landmarks_px.shape}")

    obj = FACE_MODEL_7.astype(np.float64)
    img = np.ascontiguousarray(landmarks_px, dtype=np.float64)

    rvec = prev.rvec.copy() if prev is not None else None
    tvec = prev.tvec.copy() if prev is not None else None
    use_guess = rvec is not None

    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        obj, img, camera_matrix, dist_coeffs,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=use_guess, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        raise CalibrationError("solvePnPRansac failed to fit the face model")

    for _ in range(refine_iters):
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, camera_matrix, dist_coeffs,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            break

    rmat = cv2.Rodrigues(rvec.reshape(-1))[0]
    ccs = rmat @ obj.T + tvec.reshape(3, 1)
    return HeadPose(rvec=rvec.reshape(3, 1), tvec=tvec.reshape(3, 1), landmarks_ccs=ccs)


# ---------- MPIIGaze normalization ----------

def normalize_image(
    image_rgb: np.ndarray,
    head_rotation_matrix: np.ndarray,
    center: np.ndarray,
    camera_matrix: np.ndarray,
    is_eye: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp to a canonical virtual camera: fixed distance, fixed focal length,
    roll removed.

    The point is to strip out everything the gaze angle should not depend on --
    how far away you are, where in the frame your head sits, how the head is
    rolled -- so the network only has to learn eye appearance. `rotation_matrix`
    comes back because the predicted gaze lives in this rotated space and has
    to be un-rotated before it means anything in camera coordinates.
    """
    distance_norm = DISTANCE_NORM_EYE if is_eye else DISTANCE_NORM_FACE
    out_size = SIZE_EYE if is_eye else SIZE_FACE

    distance = float(np.linalg.norm(center))
    z_scale = distance_norm / distance

    cam_norm = np.array([
        [FOCAL_NORM, 0.0, out_size[0] / 2],
        [0.0, FOCAL_NORM, out_size[1] / 2],
        [0.0, 0.0, 1.0],
    ])
    scaling = np.diag([1.0, 1.0, z_scale])

    # Build the canonical frame: z toward the eye, x along the head's x axis
    # with roll removed, y completing a right-handed set.
    forward = (center / distance).reshape(3)
    hrx = head_rotation_matrix[:, 0]
    down = np.cross(forward, hrx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    rotation_matrix = np.c_[right, down, forward].T

    transform = (cam_norm @ scaling) @ (rotation_matrix @ np.linalg.inv(camera_matrix))
    warped = cv2.warpPerspective(image_rgb, transform, out_size)
    return _equalize_rgb(warped), rotation_matrix


def _equalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Histogram-equalize luma, leaving chroma alone.

    Not cosmetic and not optional: the training crops went through this, so
    skipping it feeds the network a different input distribution than it was
    fitted on. It is also what makes the model tolerate a desk lamp being on
    or off. Dropping it is a silent accuracy regression -- the model still
    returns confident angles, they are just worse.
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def gaze_2d_to_3d(pitch_yaw: np.ndarray) -> np.ndarray:
    """(pitch, yaw) in radians -> unit direction vector."""
    pitch, yaw = float(pitch_yaw[0]), float(pitch_yaw[1])
    return np.array([
        -np.cos(pitch) * np.sin(yaw),
        -np.sin(pitch),
        -np.cos(pitch) * np.cos(yaw),
    ])


# ---------- screens ----------

@dataclass(frozen=True)
class ScreenPlane:
    """A physical monitor located in camera space.

    `origin` is the top-left corner in mm, `x_axis`/`y_axis` are unit vectors
    along the screen's width and height, and `width_mm`/`height_mm` are its
    extent. Together they map a 3D hit to a fraction of the screen, and
    `pixels` turns that into a pixel.

    This is what pperle's pipeline never actually builds. Upstream `main.py`
    carries a `# TODO load calibrated screen position` and hardcodes the plane
    to the camera's own x-y plane through the origin -- a rough stand-in for a
    laptop lid, and meaningless for a monitor off to one side.
    """
    name: str
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    width_mm: float
    height_mm: float
    pixels: tuple[int, int]
    # Where this panel's top-left sits on the global desktop, so a per-monitor
    # hit can be reported as one cursor position across all screens.
    global_origin: tuple[int, int] = (0, 0)

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.x_axis, self.y_axis)
        return n / np.linalg.norm(n)

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        """Where a gaze ray meets this plane, or None if it runs parallel or
        points away. Returning None rather than a huge number matters: an
        almost-parallel ray otherwise yields a hit hundreds of metres away that
        still lands "on screen" once projected."""
        n = self.normal
        denom = float(np.dot(n, direction))
        if abs(denom) < 1e-6:
            return None
        t = float(np.dot(n, self.origin.reshape(3) - origin.reshape(3))) / denom
        if t <= 0:
            return None
        return origin.reshape(3) + t * direction.reshape(3)

    def to_pixels(self, point_3d: np.ndarray) -> tuple[float, float] | None:
        """Project a point on the plane to pixels. None if outside the bezel."""
        rel = point_3d.reshape(3) - self.origin.reshape(3)
        u = float(np.dot(rel, self.x_axis)) / self.width_mm
        v = float(np.dot(rel, self.y_axis)) / self.height_mm
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        return u * self.pixels[0], v * self.pixels[1]

    def hit(self, origin: np.ndarray, direction: np.ndarray) -> tuple[float, float] | None:
        p = self.intersect(origin, direction)
        return None if p is None else self.to_pixels(p)


def plane_from_corners(
    name: str,
    corners_mm: np.ndarray,
    pixels: tuple[int, int],
    global_origin: tuple[int, int] = (0, 0),
) -> ScreenPlane:
    """Build a ScreenPlane from four measured corners, ordered
    top-left, top-right, bottom-right, bottom-left.

    The four points will not be exactly coplanar or exactly rectangular, so
    this least-squares them into an orthonormal frame rather than trusting any
    single corner: x from averaging both horizontal edges, y from both vertical
    edges, then y is re-orthogonalized against x (Gram-Schmidt) so the frame
    stays square even when the measurements are not.
    """
    if corners_mm.shape != (4, 3):
        raise ValueError(f"expected 4 corners of 3 coords, got {corners_mm.shape}")
    tl, tr, br, bl = (corners_mm[i].astype(float) for i in range(4))

    x_raw = ((tr - tl) + (br - bl)) / 2.0
    y_raw = ((bl - tl) + (br - tr)) / 2.0
    width = float(np.linalg.norm(x_raw))
    height = float(np.linalg.norm(y_raw))
    if width < 1.0 or height < 1.0:
        raise CalibrationError(f"{name}: degenerate corners ({width:.1f}x{height:.1f} mm)")

    x_axis = x_raw / width
    y_axis = y_raw - np.dot(y_raw, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)

    return ScreenPlane(
        name=name,
        origin=tl,
        x_axis=x_axis,
        y_axis=y_axis,
        width_mm=width,
        height_mm=height,
        pixels=pixels,
        global_origin=global_origin,
    )
