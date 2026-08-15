# Eye tracking: options and integration plan

Research notes on replacing/augmenting the head-pose detection in `vision.py`
with real gaze tracking. Nothing here is implemented yet.

## Why

`vision.py` currently infers attention from head pose alone: yaw deviation from
a per-session baseline, plus pitch for look-down. That is one blunt threshold
for "not looking at the screen", which cannot distinguish:

- looking at monitor 2 (fine) from looking away from the desk (violation)
- the bottom of the monitor from a phone sitting just below it

Both limits show up in the existing code — see the `MAX_BASELINE_DEGREES`
comment about off-axis baselines making normal posture a permanent violation.

## Options considered

| Option | Hardware | Gives you | Verdict |
| --- | --- | --- | --- |
| [shinra-meisin](https://github.com/Walker-Industries-RnD/shinra-meisin) | webcam | XR tracking suite (SLAM, body, biosignal) | No — wrong domain, calibration self-reported as partial, paid commercial license |
| [JEOresearch 3DTracker](https://github.com/JEOresearch/EyeTracker) | IR camera (DIY ~$100) | pupil ellipse in frame | No — needs new hardware, still no screen mapping |
| JEOresearch Webcam3DTracker | webcam | pupil ellipse in frame | No — sparse docs, still no screen mapping |
| MediaPipe iris landmarks | webcam | iris landmarks 468–477 | Viable — already loaded by `face_landmarker.task`, zero new deps, but mapping is all DIY |
| [pperle/gaze-tracking-pipeline](https://github.com/pperle/gaze-tracking-pipeline) | webcam | full camera-to-screen gaze point | **Chosen** — most complete, real accuracy numbers |
| [eyeGestures](https://pypi.org/project/eyeGestures/) | webcam | screen gaze, native multi-monitor | Reference only — GPLv3, RoI calibration not corner-based |

## Chosen: pperle

Three repos, not one:

- `gaze-tracking-pipeline` — runtime: normalize face/eye crops, run model, project gaze vector onto screen plane
- `gaze-tracking` — model + training (MPIIFaceGaze, **2.42–2.55° angular error**)
- `gaze-data-collection` — per-user calibration collector

Three separate calibrations are required before it outputs anything: camera
intrinsics, screen-plane geometry, per-user model calibration.

**2.5° at 60 cm is roughly 2.6 cm of error** — and that is the paper's lab
number, not ours.

## Sections

### Phase A — feasibility, before touching Argus

- **A0. Licensing gate.** Neither repo declares a license (= all rights
  reserved), and MPIIFaceGaze is research-use-only, which taints the pretrained
  weights. Fine for personal use; blocking if Argus is ever distributed.
- **A1. Standalone spike.** Run `main.py --visualize_laser_pointer` outside
  Argus. Torch on Apple Silicon + Python 3.11 compat are unknowns until this
  runs.
- **A2. Camera intrinsics.** `calibrate_camera.py` → `calibration_matrix.yaml`.
  Must calibrate at the resolution we actually run (640×480) and pin the same
  physical camera — see `resolve_camera_index` and the Continuity Camera issue.
- **A3. Screen geometry.** pperle uses Takahashi et al. 2012 (mirror + pattern,
  rigorous but awful UX, once per monitor). The look-at-each-corner idea is
  nicer to use but bootstraps screen geometry *from* gaze rays, i.e. from the
  noisy quantity being calibrated. Plan: mirror method once per monitor as
  ground truth, corner-look as routine recalibration validated against it.
  Also needs macOS monitor layout enumeration.
- **A4. Per-user calibration.** 9-point grid to start — their results show grid
  beats random, and 9 points gets most of the benefit (2.55° vs 2.42° at 128).
- **A5. Accuracy gate.** Measure real error at this desk, with these glasses,
  this webcam. Pass criteria defined *before* measuring: can it separate
  monitor 1 from monitor 2, and phone-below-monitor from bottom-of-monitor? If
  not, it buys little over the current yaw/pitch system.

### Phase B — integration

- **B1. Runtime in `VisionWorker`.** Open question: feed MediaPipe's existing
  transformation matrix into pperle's normalization instead of running a second
  landmark pass — cheaper, but breaks quietly if the 3D face model conventions
  differ. Also: per-tick frame budget, possible every-Nth-tick inference like
  `phone_detect_every_n_ticks`, extending `FrameReading` with gaze fields while
  keeping its no-image-data invariant.
- **B2. Policy layer.** Screen hit → allowed monitor / off-screen / phone, with
  enter/release hysteresis mirroring `look_away_enter/release_delta_degrees`,
  wired into `SustainedCondition` and the violation counter.
- **B3. Calibration lifecycle.** Three artifacts under `state/`, expiry rules,
  and plausibility rejection modeled on `_baseline_from` — a bad gaze
  calibration must be refused, not committed. Checksum the `.ckpt` the way
  `models/` already does.

### Phase C — hardening

- **C1. Degradation.** Behind a config flag; yaw/pitch retained as fallback;
  `DEGRADED` banner when model or calibration is missing.
- **C2. Tests.** Synthetic `FrameReading`s with gaze fields through the state
  machine, no camera. Extend the false-positive button to log gaze readings.
- **C3. Packaging.** Torch is ~2 GB. Real impact on `.venv`, `Argus.app`, and
  setup time — decide deliberately.

## Sequencing

A5 is a gate, not a formality. A1–A4 is a few evenings of setup that produces
one number, and that number decides whether Phase B is worth building.
