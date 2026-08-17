"""Live gaze: visual check, and the A5 accuracy measurement.

    # watch a dot follow your eyes
    python3 tools/gaze_spike.py --mode pointer  ...

    # measure the error that decides whether this is worth integrating
    python3 tools/gaze_spike.py --mode measure  ...

`measure` is the gate. It shows targets at known pixels, records where the gaze
lands, and reports the error in degrees, millimetres and pixels -- then answers
the only two questions that actually matter for Argus:

  1. Can it tell one monitor from another?
  2. Can it tell the bottom of a monitor from a phone sitting just below it?

Question 2 is the one that justifies the whole project. Head pose already
answers question 1 reasonably well; nothing in the current `vision.py` can
answer question 2, because looking down at a phone and looking at the bottom of
the screen are nearly the same head pose.

The pass criteria are declared here, before any measurement is taken, so the
result cannot be rationalized after the fact.
"""
# ruff: noqa: E402  -- sys.path must be set before the package import
from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaze.model import load_checkpoint, pick_device
from gaze.store import load_calibration
from gaze.tracker import GazeTracker

# --- A5 pass criteria, fixed in advance -------------------------------------
# Monitors are ~600mm wide and sit further apart than that, so half a monitor
# width of error still identifies the right screen.
MONITOR_ID_PASS_RATE = 0.95
# A phone below the bezel is ~100-200mm from the bottom of the screen. To
# separate those reliably the vertical error has to stay well inside that.
PHONE_SEPARATION_MM = 60.0
# ----------------------------------------------------------------------------

GRID = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]
SAMPLES_PER_TARGET = 10


def _open_landmarker(path: str):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )


def _read_gaze(cap, landmarker, tracker):
    import mediapipe as mp
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    pts = np.array([[lm.x, lm.y] for lm in result.face_landmarks[0]])
    reading = tracker.update(rgb, pts)
    return reading if reading.ok else None


def run_pointer(cap, landmarker, tracker, cal) -> int:
    """A dot that follows your gaze. Purely a smoke test -- if this does not
    roughly track, no amount of measurement will help."""
    screen = cal.screens[0]
    root = tk.Tk()
    root.overrideredirect(True)
    w, h = screen.pixels
    root.geometry(f"{w}x{h}+{screen.global_origin[0]}+{screen.global_origin[1]}")
    canvas = tk.Canvas(root, width=w, height=h, bg="#0d0d0d", highlightthickness=0)
    canvas.pack()
    root.bind("<Escape>", lambda _e: root.quit())

    trail: list[tuple[float, float]] = []
    fps_t0, frames, fps = time.perf_counter(), 0, 0.0

    def tick():
        nonlocal frames, fps_t0, fps
        reading = _read_gaze(cap, landmarker, tracker)
        canvas.delete("all")
        frames += 1
        if time.perf_counter() - fps_t0 > 1.0:
            fps = frames / (time.perf_counter() - fps_t0)
            frames, fps_t0 = 0, time.perf_counter()

        if reading and reading.on_any_screen and reading.screen_name == screen.name:
            trail.append(reading.screen_xy)
            del trail[:-24]
            for i, (x, y) in enumerate(trail):
                a = (i + 1) / len(trail)
                r = 4 + 14 * a
                shade = int(60 + 160 * a)
                canvas.create_oval(x - r, y - r, x + r, y + r,
                                   outline=f"#{shade:02x}{shade // 2:02x}20", width=2)
            x, y = trail[-1]
            canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ffb020", outline="")
        else:
            trail.clear()

        status = "no face" if reading is None else ("off-screen" if not reading.on_any_screen else reading.screen_name)
        dist = f"{reading.distance_mm:.0f}mm" if reading and reading.distance_mm else "--"
        canvas.create_text(16, 16, anchor="nw", fill="#555555", font=("Menlo", 13),
                           text=f"{status}   {dist}   {fps:.0f} fps   ESC to quit")
        root.after(1, tick)

    root.after(1, tick)
    root.mainloop()
    root.destroy()
    return 0


def run_measure(cap, landmarker, tracker, cal, output: str) -> int:
    screen = cal.screens[0]
    root = tk.Tk()
    root.overrideredirect(True)
    w, h = screen.pixels
    root.geometry(f"{w}x{h}+{screen.global_origin[0]}+{screen.global_origin[1]}")
    canvas = tk.Canvas(root, width=w, height=h, bg="#0d0d0d", highlightthickness=0)
    canvas.pack()

    aborted = {"v": False}
    root.bind("<Escape>", lambda _e: (aborted.__setitem__("v", True), root.quit()))

    mm_per_px_x = screen.width_mm / screen.pixels[0]
    mm_per_px_y = screen.height_mm / screen.pixels[1]
    records = []

    def draw(tx, ty, msg, progress=0.0):
        canvas.delete("all")
        canvas.create_text(w // 2, 60, text=msg, fill="#888888", font=("Helvetica", 20))
        r = 26
        canvas.create_oval(tx - r, ty - r, tx + r, ty + r, outline="#333333", width=4)
        if progress:
            canvas.create_arc(tx - r, ty - r, tx + r, ty + r, start=90, extent=-359.9 * progress,
                              outline="#4da3ff", width=4, style=tk.ARC)
        canvas.create_oval(tx - 6, ty - 6, tx + 6, ty + 6, fill="#ff4d4d", outline="")
        root.update_idletasks()
        root.update()

    for n, (fx, fy) in enumerate(GRID):
        if aborted["v"]:
            break
        tx, ty = fx * w, fy * h
        for _ in range(10):          # let the eyes arrive
            cap.read()
            draw(tx, ty, f"Look at the red dot   ({n + 1}/{len(GRID)})")
        got, tries = 0, 0
        while got < SAMPLES_PER_TARGET and tries < SAMPLES_PER_TARGET * 8 and not aborted["v"]:
            tries += 1
            draw(tx, ty, f"Look at the red dot   ({n + 1}/{len(GRID)})", got / SAMPLES_PER_TARGET)
            reading = _read_gaze(cap, landmarker, tracker)
            if reading is None or reading.screen_xy is None:
                continue
            gx, gy = reading.screen_xy
            records.append({
                "target_px": [tx, ty],
                "gaze_px": [gx, gy],
                "err_mm_x": (gx - tx) * mm_per_px_x,
                "err_mm_y": (gy - ty) * mm_per_px_y,
                "distance_mm": reading.distance_mm,
                "screen": reading.screen_name,
            })
            got += 1

    root.destroy()
    if not records:
        print("no measurements captured", file=sys.stderr)
        return 1

    ex = np.array([r["err_mm_x"] for r in records])
    ey = np.array([r["err_mm_y"] for r in records])
    dist = np.array([r["distance_mm"] for r in records], dtype=float)
    euclid = np.hypot(ex, ey)
    # Angular error is the comparable number -- it is what the paper reports
    # (2.42-2.55 deg) and it is independent of how far away you were sitting.
    angular = np.degrees(np.arctan2(euclid, dist))

    right_screen = sum(1 for r in records if r["screen"] == cal.screens[0].name) / len(records)
    p95_y = float(np.percentile(np.abs(ey), 95))

    print("\n" + "=" * 62)
    print(f"A5 ACCURACY  --  {len(records)} samples over {len(GRID)} targets")
    print("=" * 62)
    print(f"  viewing distance      {np.mean(dist):.0f} mm (sd {np.std(dist):.0f})")
    print(f"  angular error         mean {np.mean(angular):.2f} deg   median {np.median(angular):.2f}   p95 {np.percentile(angular, 95):.2f}")
    print("    (paper reports 2.42-2.55 deg after per-user calibration)")
    print(f"  euclidean error       mean {np.mean(euclid):.0f} mm   p95 {np.percentile(euclid, 95):.0f} mm")
    print(f"  horizontal error      mean {np.mean(np.abs(ex)):.0f} mm   p95 {np.percentile(np.abs(ex), 95):.0f} mm")
    print(f"  vertical error        mean {np.mean(np.abs(ey)):.0f} mm   p95 {p95_y:.0f} mm")

    print("\n  GATE 1  monitor identification")
    ok1 = right_screen >= MONITOR_ID_PASS_RATE
    print(f"    {right_screen * 100:.1f}% of samples attributed to the right screen "
          f"(need {MONITOR_ID_PASS_RATE * 100:.0f}%)  ->  {'PASS' if ok1 else 'FAIL'}")

    print("\n  GATE 2  phone below the bezel vs bottom of screen")
    ok2 = p95_y <= PHONE_SEPARATION_MM
    print(f"    p95 vertical error {p95_y:.0f} mm (need <= {PHONE_SEPARATION_MM:.0f} mm)  "
          f"->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("    Vertical error this large cannot separate a phone from the lower screen,")
        print("    which is the capability head pose alone does not already provide.")

    print("\n  VERDICT: " + (
        "proceed to Phase B" if ok1 and ok2 else
        "monitor identification only -- Phase B is worth less than hoped" if ok1 else
        "does not clear the bar; per-user calibration first, then re-measure"))
    print("=" * 62)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(),
        "n_samples": len(records),
        "summary": {
            "angular_deg_mean": float(np.mean(angular)),
            "angular_deg_p95": float(np.percentile(angular, 95)),
            "euclid_mm_mean": float(np.mean(euclid)),
            "vertical_mm_p95": p95_y,
            "monitor_id_rate": right_screen,
            "gate_monitor_id": bool(ok1),
            "gate_phone_separation": bool(ok2),
        },
        "records": records,
    }, indent=2))
    print(f"\nwrote {output}")
    return 0 if (ok1 and ok2) else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("pointer", "measure"), default="pointer")
    ap.add_argument("--calibration", default="state/gaze_calibration.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--face-model", required=True)
    ap.add_argument("--output", default="state/gaze_accuracy.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--person-idx", type=int, default=0)
    args = ap.parse_args()

    cal = load_calibration(args.calibration, expect_checkpoint=args.checkpoint)
    device = pick_device()
    model = load_checkpoint(args.checkpoint, device)
    landmarker = _open_landmarker(args.face_model)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal.capture_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal.capture_size[1])
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != cal.capture_size:
        print(f"camera is delivering {got[0]}x{got[1]} but the calibration is for "
              f"{cal.capture_size[0]}x{cal.capture_size[1]} -- the intrinsics do not transfer.",
              file=sys.stderr)
        cap.release()
        return 1

    tracker = GazeTracker(model, cal.camera_matrix, cal.dist_coeffs, cal.screens,
                          device, person_idx=args.person_idx)
    try:
        if args.mode == "pointer":
            return run_pointer(cap, landmarker, tracker, cal)
        return run_measure(cap, landmarker, tracker, cal, args.output)
    finally:
        cap.release()
        landmarker.close()


if __name__ == "__main__":
    raise SystemExit(main())
