# gaze/ — pperle port for Argus

Webcam gaze tracking that reports **which monitor you are looking at and where
on it**, ported from [pperle/gaze-tracking](https://github.com/pperle/gaze-tracking)
and modernized for Python 3.11 / Apple Silicon.

Status: **complete and tested in simulation. Never run against a real camera.**
Every check below passes headlessly; none of them prove the thing works on your
desk. That is what `tools/gaze_spike.py --mode measure` is for.

## Layout

| File | Does |
| --- | --- |
| `gaze/model.py` | The network. Inference only — no pytorch_lightning, no ImageNet download |
| `gaze/geometry.py` | Camera intrinsics, head pose (solvePnP), MPIIGaze normalization, screen planes |
| `gaze/calibrate.py` | **Corner-look screen calibration** — the piece upstream never wrote |
| `gaze/tracker.py` | Per-frame runtime: landmarks in, screen coordinates out |
| `gaze/screens.py` | macOS display enumeration (Quartz, falling back to `system_profiler`) |
| `gaze/store.py` | Calibration persistence, with validation that refuses stale artifacts |
| `tools/calibrate_camera.py` | Chessboard intrinsics |
| `tools/calibrate_screens.py` | The corner-look calibration UI |
| `tools/gaze_spike.py` | Live pointer, and the A5 accuracy measurement |

## Verify without hardware

```bash
python3 tools/model_smoke.py         # model runs on modern torch
python3 tests/test_geometry.py       # 24 checks, incl. bit-identical to upstream
python3 tests/test_calibration.py    # 11 checks
python3 tests/test_tracker.py        # 22 checks
python3 tests/test_end_to_end.py     # closed loop: calibrate, then track with it
python3 tools/noise_sweep.py         # how calibration degrades with gaze error
```

`test_end_to_end.py` is the one worth reading. A simulated person looks at a
simulated 27in monitor; corner calibration recovers it to 597.0 x 336.0 mm
against a true 597 x 336, and tracking through the recovered plane lands within
0.00 mm of truth. It also asserts the result is not mirrored or rotated — a
plane fit can be geometrically valid and still flipped, and no unit test on
either half would catch it.

## Setup, in order

**1. Get the checkpoint.** It is not in the repo — download `p00.ckpt` from
[the author's Drive folder](https://drive.google.com/drive/folders/1-_bOyMgAQmnwRGfQ4QIQk7hrin0Mexch).

**2. Camera intrinsics.**

```bash
python3 tools/calibrate_camera.py --output state/camera_matrix.yaml --width 1280 --height 720
```

Aim for under 0.5 px reprojection error. Calibrate at the resolution you will
actually capture at — fx, fy, cx, cy are in pixels and do not transfer.

**3. Screen geometry.**

```bash
python3 tools/calibrate_screens.py \
    --camera-matrix state/camera_matrix.yaml \
    --checkpoint state/p00.ckpt \
    --face-model ../models/face_landmarker.task \
    --bootstrap
```

Look at each corner of each display. You will be asked to shift position
between rounds — that is the measurement, not politeness. One head position
gives four rays with nothing to intersect against; the corners' distance is
recoverable only from parallax.

**4. Check it visually, then measure it.**

```bash
python3 tools/gaze_spike.py --mode pointer --checkpoint state/p00.ckpt --face-model ...
python3 tools/gaze_spike.py --mode measure --checkpoint state/p00.ckpt --face-model ...
```

## The ordering problem

`--bootstrap` in step 3 is not optional politeness either. Simulation puts a
stock checkpoint's 2.4° error at ~45 mm corner residual, right past the 40 mm
strict limit — so the first calibration on a fresh install gets **refused**,
correctly, because at that error it cannot distinguish a good fit from a bad one.

That creates a loop: good screen geometry wants a model calibrated to you, and
per-user calibration wants to know where the screen is. Break it by bootstrapping:

1. Calibrate screens with `--bootstrap` (90 mm tolerance) → provisional geometry
2. Collect per-user data and fine-tune `subject_biases` against it
3. Recalibrate screens without `--bootstrap` → real geometry

Step 2 is the one piece still unbuilt here. Upstream's
[gaze-data-collection](https://github.com/pperle/gaze-data-collection) does it,
and the model exposes the surface: `subject_biases` is a (30, 2) table of
per-participant pitch/yaw offsets, and a stock checkpoint means you are wearing
participant 0's.

## What changed from upstream, and why

**Dropped `pytorch_lightning`.** Used only for the training loop. Checkpoints
still load — Lightning writes a plain `state_dict`, which `load_checkpoint`
unwraps.

**Dropped ImageNet VGG weights.** Upstream passes `pretrained=True`, removed in
torchvision 0.15, and it downloads ~500 MB that the checkpoint immediately
overwrites.

**Dropped `pgi`/GTK.** Linux-only, so `get_monitor_dimensions()` returned
`(None, None)` on macOS and the pipeline demanded manual entry. Replaced with
Quartz. Physical millimetres are now only a cross-check, since corner
calibration measures the panel itself — a display whose EDID lies still
calibrates correctly.

**Reimplemented the geometry rather than vendoring it.** `gaze-tracking` (the
model) is MIT. `gaze-tracking-pipeline` and `gaze-data-collection` ship **no
LICENSE at all** — all rights reserved. The maths is published anyway (Sugano
et al. 2014, Zhang et al. 2018, plus textbook PnP), and
`tests/test_geometry.py` asserts the rewrite produces byte-identical crops.

That equivalence test earned its keep immediately: the first draft omitted
upstream's histogram equalization, which the model was trained with. It would
have degraded accuracy silently, since the network still returns confident
angles either way.

**Takes landmarks, not a camera.** `GazeTracker.update()` accepts a plain array
of normalized landmarks, so Argus's existing `FaceLandmarker` can feed it — no
second face detector. Only 7 of the 468 landmarks are used.

Note the head pose is **not** shareable the same way. Argus's
`facial_transformation_matrixes` is expressed against MediaPipe's assumed camera
and is not metric; the ray/plane intersection needs real millimetres, which only
solvePnP against calibrated intrinsics provides.

## What upstream never built

`gaze-tracking-pipeline/main.py`:

```python
# TODO load calibrated screen position
plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
```

The screen plane is hardcoded to the camera's own x-y plane through the origin,
with a `result_y - 20  # 20 mm offset` fudge downstream. That approximates a
laptop lid and is meaningless for a monitor off to one side. Multi-monitor was
never supported; it was a TODO. `gaze/calibrate.py` is that missing piece.

## Known limits

- **~2.4° is the ceiling** without per-user calibration — roughly 2.6 cm at
  60 cm. Good enough to identify a monitor, probably not to separate a phone
  from the bottom bezel. `gaze_spike.py --mode measure` gates on exactly that.
- **Calibration is tied to where you sit.** Same assumption `vision.py` already
  makes with its per-session pitch/yaw baseline.
- **Torch is a ~2 GB dependency**, which is a real change to `.venv` and the
  app bundle.
- **`gaze/screens.py` is untested on hardware.** The Quartz path needs
  `pyobjc-framework-Quartz`; without it the `system_profiler` fallback gives
  resolution but no global origin, so multi-monitor coordinates will be wrong
  until that is confirmed on a real multi-display setup.
