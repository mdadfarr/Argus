"""The runtime: MediaPipe landmarks in, screen coordinates out.

Deliberately framework-agnostic. `GazeTracker.update()` takes a plain array of
normalized landmarks, not a MediaPipe result object, so the same code serves
the standalone spike and Argus's `VisionWorker` -- which already runs a
`FaceLandmarker` every tick and can hand its landmarks straight over. That
avoids the thing upstream forces: a second, redundant face detector.

Note this does NOT reuse MediaPipe's `facial_transformation_matrixes`, even
though Argus already computes them. That matrix is expressed against
MediaPipe's own assumed camera and is not metric. The ray/plane intersection
needs a real distance in millimetres, which only solvePnP against calibrated
intrinsics can give. The landmarks are shareable; the head pose is not.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import (
    LANDMARK_IDS,
    HeadPose,
    ScreenPlane,
    gaze_2d_to_3d,
    normalize_image,
    solve_head_pose,
)
from .model import GazeModel

log = logging.getLogger(__name__)

# ImageNet statistics -- upstream normalizes with albumentations' defaults, and
# the weights were fitted against them.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class GazeReading:
    """One frame's result. Carries no image data, matching the invariant
    `vision.FrameReading` already holds."""

    ok: bool
    error: str | None = None
    # Which calibrated monitor the ray hit, and where on it.
    screen_name: str | None = None
    screen_xy: tuple[float, float] | None = None      # pixels, within that monitor
    global_xy: tuple[float, float] | None = None      # pixels on the desktop
    on_any_screen: bool = False
    # Raw geometry, useful for logging and threshold tuning.
    gaze_pitch_yaw: tuple[float, float] | None = None
    distance_mm: float | None = None
    head_pose_ok: bool = False
    # The smoothed ray in camera space, in millimetres. Exposed because screen
    # calibration consumes rays directly and has no screens to intersect
    # against yet -- without this the calibration tool would have to reach into
    # the tracker's private buffers to get what it just computed.
    ray_origin: np.ndarray | None = None
    ray_direction: np.ndarray | None = None


def _to_tensor(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 -> NCHW float, ImageNet-normalized."""
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)


class GazeTracker:
    """Stateful per-frame gaze estimation.

    State is all temporal smoothing plus the previous head pose, which seeds
    the next solvePnP. Call `reset()` when the session restarts -- carrying a
    stale pose across a gap makes the first frames after the gap converge to
    wherever the head used to be.
    """

    def __init__(
        self,
        model: GazeModel,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        screens: list[ScreenPlane],
        device: torch.device,
        person_idx: int = 0,
        landmark_smoothing: int = 3,
        gaze_smoothing: int = 10,
    ):
        self.model = model
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=float)
        self.screens = list(screens)
        self.device = device
        self.person_idx = torch.tensor([[person_idx]], dtype=torch.long, device=device)

        self._landmark_buf: deque[np.ndarray] = deque(maxlen=max(1, landmark_smoothing))
        self._gaze_buf: deque[np.ndarray] = deque(maxlen=max(1, gaze_smoothing))
        self._prev_pose: HeadPose | None = None

    def reset(self) -> None:
        self._landmark_buf.clear()
        self._gaze_buf.clear()
        self._prev_pose = None

    @torch.no_grad()
    def update(self, image_rgb: np.ndarray, landmarks_norm: np.ndarray) -> GazeReading:
        """One frame.

        `landmarks_norm` is (>=468, 2) with x and y in 0..1, exactly what
        MediaPipe emits. Only seven of them are used, but the caller should not
        have to know which.
        """
        h, w = image_rgb.shape[:2]
        try:
            pts = np.asarray(landmarks_norm, dtype=float)
            if pts.ndim != 2 or pts.shape[0] <= max(LANDMARK_IDS) or pts.shape[1] < 2:
                return GazeReading(ok=False, error="landmarks_malformed")

            px = np.stack([pts[i, :2] * (w, h) for i in LANDMARK_IDS])
            self._landmark_buf.append(px)
            px = np.mean(self._landmark_buf, axis=0)

            pose = solve_head_pose(px, self.camera_matrix, self.dist_coeffs, prev=self._prev_pose)
            self._prev_pose = pose
        except Exception as e:
            # A PnP failure is a bad frame, not a bad session. Drop the stale
            # pose so the next frame re-solves from scratch instead of
            # refining against a fit we just rejected.
            self._prev_pose = None
            log.debug("head pose failed: %s", e)
            return GazeReading(ok=False, error="head_pose_failed")

        right_eye, left_eye = pose.eye_centers()
        face_center = pose.face_center()
        rot = pose.rotation_matrix

        img_left, _ = normalize_image(image_rgb, rot, left_eye, self.camera_matrix, is_eye=True)
        img_right, _ = normalize_image(image_rgb, rot, right_eye, self.camera_matrix, is_eye=True)
        img_face, face_rot = normalize_image(image_rgb, rot, face_center, self.camera_matrix, is_eye=False)

        out = self.model(
            self.person_idx,
            _to_tensor(img_face, self.device),
            _to_tensor(img_right, self.device),
            _to_tensor(img_left, self.device),
        ).squeeze(0).float().cpu().numpy()

        # The prediction lives in the normalized frame; undo that rotation to
        # get a direction in real camera space.
        gaze = np.linalg.inv(face_rot) @ gaze_2d_to_3d(out)
        self._gaze_buf.append(gaze)
        gaze = np.mean(self._gaze_buf, axis=0)
        gaze /= np.linalg.norm(gaze)

        origin = face_center.reshape(3)
        hit = self._first_hit(origin, gaze)

        return GazeReading(
            ok=True,
            screen_name=hit[0] if hit else None,
            screen_xy=hit[1] if hit else None,
            global_xy=hit[2] if hit else None,
            on_any_screen=hit is not None,
            gaze_pitch_yaw=(float(out[0]), float(out[1])),
            distance_mm=float(np.linalg.norm(face_center)),
            head_pose_ok=True,
            ray_origin=origin.copy(),
            ray_direction=gaze.copy(),
        )

    def _first_hit(self, origin: np.ndarray, direction: np.ndarray):
        """Nearest monitor the ray actually lands on.

        Nearest rather than first-in-list because overlapping planes are real:
        a laptop screen in front of an external monitor means one ray can
        satisfy both, and the one physically closer is the one being looked at.
        """
        best = None
        best_dist = float("inf")
        for screen in self.screens:
            point = screen.intersect(origin, direction)
            if point is None:
                continue
            xy = screen.to_pixels(point)
            if xy is None:
                continue
            dist = float(np.linalg.norm(point - origin))
            if dist < best_dist:
                best_dist = dist
                gx, gy = screen.global_origin
                best = (screen.name, xy, (xy[0] + gx, xy[1] + gy))
        return best
