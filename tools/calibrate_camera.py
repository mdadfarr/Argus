"""Chessboard camera-intrinsics calibration.

    python3 tools/calibrate_camera.py --output state/camera_matrix.yaml

Print a chessboard (https://github.com/opencv/opencv/blob/master/doc/pattern.png),
stick it to something rigid, and show it to the camera at a variety of angles
and distances. SPACE captures the current view when a board is detected, C runs
the solve, Q quits.

The resolution matters more than anything else here. fx, fy, cx and cy are in
pixels, so a solve at 1280x720 is not valid at 640x480 -- pass `--width/--height`
matching whatever the app will actually capture at. The header written into the
yaml records it, and `store.load_calibration` refuses a mismatch later.
"""
# ruff: noqa: E402  -- sys.path must be set before the package import
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIN_VIEWS = 12          # below this the solve is under-determined in practice
TARGET_VIEWS = 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="state/camera_matrix.yaml")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square-mm", type=float, default=25.0)
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual != (args.width, args.height):
        # Cameras silently substitute the nearest supported mode. Calibrating
        # at a resolution you did not ask for, and recording the one you did,
        # is exactly the mismatch this whole file warns about.
        print(f"NOTE: camera delivered {actual[0]}x{actual[1]}, not {args.width}x{args.height}. "
              f"Recording the delivered size.")

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"SPACE = capture, C = calibrate ({MIN_VIEWS} views minimum), Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed", file=sys.stderr)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        )

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        colour = (0, 200, 0) if found else (0, 0, 200)
        cv2.putText(view, f"views {len(obj_points)}/{TARGET_VIEWS}   {'BOARD' if found else 'no board'}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.imshow("camera calibration", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return 1
        if key == ord(" ") and found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp.copy())
            img_points.append(refined)
            print(f"  captured view {len(obj_points)}")
        if key == ord("c"):
            if len(obj_points) < MIN_VIEWS:
                print(f"  need at least {MIN_VIEWS} views, have {len(obj_points)}")
                continue
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < MIN_VIEWS:
        return 1

    print(f"solving from {len(obj_points)} views ...")
    rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None,
    )

    # Reprojection error is the honest quality signal. Under ~0.5 px is good,
    # over ~1.0 px means the board was blurry, coplanar in every view, or the
    # square size is wrong -- and every downstream millimetre inherits it.
    errors = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist)
        errors.append(cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj))
    mean_err = float(np.mean(errors))

    print(f"  rms reprojection error: {rms:.4f} px (per-view mean {mean_err:.4f})")
    if mean_err > 1.0:
        print("  WARNING: over 1 px. Recapture with more angles and less motion blur.")
    print(f"  fx={camera_matrix[0,0]:.1f} fy={camera_matrix[1,1]:.1f} "
          f"cx={camera_matrix[0,2]:.1f} cy={camera_matrix[1,2]:.1f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeff": dist.ravel().tolist(),
        "capture_size": list(actual),
        "rms_reprojection_px": float(rms),
        "views": len(obj_points),
        "created_at": datetime.now(UTC).isoformat(),
    }, sort_keys=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
