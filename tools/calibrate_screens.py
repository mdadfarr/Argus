"""Corner-look screen calibration -- the piece upstream left as a TODO.

    python3 tools/calibrate_screens.py \
        --camera-matrix state/camera_matrix.yaml \
        --checkpoint state/p00.ckpt \
        --face-model ../models/face_landmarker.task \
        --output state/gaze_calibration.json

For every display, a target appears at each of the four corners in turn. Look
at it and hold still while the ring fills. You will be asked to shift position
between rounds -- that is not politeness, it is the measurement: a corner's
distance from the camera can only be recovered from rays that arrive at
different angles. Sampling four corners from one fixed head position gives four
lines with nothing to intersect against.

Controls: SPACE starts a round, ESC aborts.
"""
# ruff: noqa: E402  -- sys.path must be set before the package import
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze.calibrate import (
    BOOTSTRAP_RESIDUAL_MM,
    CORNER_NAMES,
    MAX_RESIDUAL_MM,
    CornerCollector,
    sanity_check,
)
from gaze.geometry import CalibrationError, load_camera_matrix
from gaze.model import load_checkpoint, pick_device
from gaze.screens import list_displays
from gaze.store import Calibration, save_calibration, sha256_file
from gaze.tracker import GazeTracker

SAMPLES_PER_CORNER = 6      # frames averaged per corner per round
ROUNDS = 4                  # distinct head positions; drives the parallax
TARGET_INSET_PX = 60        # keep the dot off the bezel so it is actually visible

MOVE_PROMPTS = [
    "Sit normally.",
    "Now lean LEFT about a hand's width, and stay there.",
    "Now lean RIGHT, and a little closer to the screen.",
    "Now sit back, further away than normal.",
]


class CornerCalibrationUI:
    """Fullscreen target window driven from the Tk event loop.

    Camera reads happen in `after()` callbacks rather than a worker thread:
    this is a short-lived tool, the loop is not doing anything else, and a
    single-threaded design removes any question of Tk being touched off the
    main thread -- the failure mode `vision.VisionWorker` exists to avoid.
    """

    def __init__(self, tracker: GazeTracker, cap: cv2.VideoCapture, displays, face_landmarker,
                 max_residual_mm: float = MAX_RESIDUAL_MM):
        self.max_residual_mm = max_residual_mm
        self.tracker = tracker
        self.cap = cap
        self.displays = displays
        self.landmarker = face_landmarker

        self.root = tk.Tk()
        self.root.withdraw()
        self.canvas: tk.Canvas | None = None
        self.window: tk.Toplevel | None = None

        self.results: dict[str, tuple] = {}
        self.aborted = False

    # ---------- window management ----------
    def _open_for(self, display) -> None:
        self.window = tk.Toplevel(self.root)
        self.window.overrideredirect(True)
        w, h = display.pixels
        x, y = display.origin
        self.window.geometry(f"{w}x{h}+{x}+{y}")
        self.window.configure(bg="#111111")
        self.window.attributes("-topmost", True)
        self.canvas = tk.Canvas(self.window, width=w, height=h, bg="#111111", highlightthickness=0)
        self.canvas.pack()
        self.window.bind("<Escape>", lambda _e: self._abort())
        self.window.focus_force()

    def _close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.canvas = None

    def _abort(self) -> None:
        self.aborted = True
        self._close()
        self.root.quit()

    # ---------- drawing ----------
    def _corner_xy(self, display, idx: int) -> tuple[int, int]:
        w, h = display.pixels
        i = TARGET_INSET_PX
        return [(i, i), (w - i, i), (w - i, h - i), (i, h - i)][idx]

    def _draw(self, display, corner_idx: int, progress: float, message: str) -> None:
        c = self.canvas
        if c is None:
            return
        c.delete("all")
        w, h = display.pixels
        x, y = self._corner_xy(display, corner_idx)

        c.create_text(w // 2, h // 2 - 40, text=message, fill="#dddddd",
                      font=("Helvetica", 26), width=int(w * 0.7), justify="center")
        c.create_text(w // 2, h // 2 + 40, text=f"{display.name}  ·  {CORNER_NAMES[corner_idx]}",
                      fill="#666666", font=("Helvetica", 16))

        # Filling ring: unambiguous "hold still until this completes".
        r = 34
        c.create_oval(x - r, y - r, x + r, y + r, outline="#333333", width=5)
        if progress > 0:
            c.create_arc(x - r, y - r, x + r, y + r, start=90, extent=-359.9 * progress,
                         outline="#4da3ff", width=5, style=tk.ARC)
        c.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#4da3ff", outline="")
        self.window.update_idletasks()
        self.window.update()

    # ---------- capture ----------
    def _sample_ray(self):
        """One gaze ray, or None if the frame was unusable."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        import mediapipe as mp  # noqa: PLC0415
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        pts = np.array([[lm.x, lm.y] for lm in result.face_landmarks[0]])
        reading = self.tracker.update(rgb, pts)
        if not reading.ok or reading.ray_origin is None:
            return None
        # The tracker already smoothed this ray; use what it computed rather
        # than recomputing from its internals.
        return reading.ray_origin, reading.ray_direction

    def _collect_corner(self, display, collector, corner_idx, message) -> int:
        got = 0
        attempts = 0
        # Let the eyes land on the target before believing anything.
        for _ in range(8):
            self.cap.read()
        while got < SAMPLES_PER_CORNER and attempts < SAMPLES_PER_CORNER * 8 and not self.aborted:
            attempts += 1
            self._draw(display, corner_idx, got / SAMPLES_PER_CORNER, message)
            sample = self._sample_ray()
            if sample is None:
                continue
            collector.add(corner_idx, sample[0], sample[1])
            got += 1
        self._draw(display, corner_idx, 1.0, message)
        return got

    def _wait_for_space(self, display, text) -> None:
        pressed = {"go": False}
        self.window.bind("<space>", lambda _e: pressed.__setitem__("go", True))
        while not pressed["go"] and not self.aborted:
            self._draw(display, 0, 0.0, text + "\n\nPress SPACE when ready")
            self.window.after(30)
        self.window.unbind("<space>")

    # ---------- driver ----------
    def run(self) -> dict:
        for display in self.displays:
            if self.aborted:
                break
            self._open_for(display)
            collector = CornerCollector(display.name, display.pixels)

            for rnd in range(ROUNDS):
                if self.aborted:
                    break
                self._wait_for_space(display, MOVE_PROMPTS[rnd % len(MOVE_PROMPTS)])
                for corner_idx in range(4):
                    if self.aborted:
                        break
                    got = self._collect_corner(
                        display, collector, corner_idx,
                        f"Look at the dot  (round {rnd + 1} of {ROUNDS})",
                    )
                    if got == 0:
                        print(f"  no usable frames for {CORNER_NAMES[corner_idx]} in round {rnd + 1}")

            self._close()
            if self.aborted:
                break

            try:
                plane, report = collector.solve(self.max_residual_mm)
            except CalibrationError as e:
                print(f"\n{display.name}: calibration REJECTED -- {e}")
                continue

            warnings = sanity_check(plane, report)
            self.results[display.name] = (plane, report, warnings)
            self._print_report(display, report, warnings)

        self.root.destroy()
        return self.results

    def _print_report(self, display, report, warnings) -> None:
        print(f"\n{display.name}")
        print(f"  solved panel: {report['width_mm']:.0f} x {report['height_mm']:.0f} mm "
              f"({report['diagonal_in']}in diagonal)")
        if display.size_mm:
            # EDID is an independent measurement, so a big disagreement means
            # one of the two is wrong -- worth surfacing, not worth failing on.
            print(f"  display reports: {display.size_mm[0]:.0f} x {display.size_mm[1]:.0f} mm")
        print(f"  flatness {report['flatness_mm']}mm, squareness {report['squareness_deg']}deg")
        for name, c in report["corners"].items():
            print(f"    {name:<13} {c['samples']:>2} samples, "
                  f"{c['parallax_deg']:>4}deg parallax, {c['residual_mm']:>5}mm residual")
        for w in warnings:
            print(f"  WARNING: {w}")
        if not warnings:
            print("  looks sound")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-matrix", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--face-model", required=True, help="MediaPipe face_landmarker.task")
    ap.add_argument("--output", default="state/gaze_calibration.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--person-idx", type=int, default=0)
    ap.add_argument(
        "--bootstrap", action="store_true",
        help="accept a looser corner fit. Use for the FIRST calibration on a stock "
             "checkpoint, whose ~2.4 deg error sits right at the strict limit. "
             "Recalibrate without this flag once the model is calibrated to you.",
    )
    args = ap.parse_args()

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    camera_matrix, dist = load_camera_matrix(args.camera_matrix)
    device = pick_device()
    model = load_checkpoint(args.checkpoint, device)

    landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(args.face_model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1
    capture_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    displays = list_displays()
    print(f"calibrating {len(displays)} display(s):")
    for d in displays:
        print(f"  {d.describe()}")

    tracker = GazeTracker(model, camera_matrix, dist, [], device, person_idx=args.person_idx)

    try:
        ui = CornerCalibrationUI(
            tracker, cap, displays, landmarker,
            max_residual_mm=BOOTSTRAP_RESIDUAL_MM if args.bootstrap else MAX_RESIDUAL_MM,
        )
        if args.bootstrap:
            print(f"\nBOOTSTRAP MODE: accepting up to {BOOTSTRAP_RESIDUAL_MM:.0f}mm corner residual.")
            print("This calibration is provisional -- redo it without --bootstrap later.\n")
        results = ui.run()
    finally:
        cap.release()
        landmarker.close()

    if not results:
        print("\nnothing calibrated", file=sys.stderr)
        return 1

    planes = []
    for display in displays:
        entry = results.get(display.name)
        if entry is None:
            continue
        plane, _report, _warnings = entry
        planes.append(type(plane)(
            name=plane.name, origin=plane.origin, x_axis=plane.x_axis, y_axis=plane.y_axis,
            width_mm=plane.width_mm, height_mm=plane.height_mm, pixels=plane.pixels,
            global_origin=display.origin,
        ))

    save_calibration(args.output, Calibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist,
        capture_size=capture_size,
        screens=planes,
        checkpoint_sha256=sha256_file(args.checkpoint),
        created_at=datetime.now(UTC).isoformat(),
    ))

    report_path = Path(args.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(
        {name: {"report": r, "warnings": w} for name, (_p, r, w) in results.items()}, indent=2))
    print(f"\nwrote {args.output} and {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
