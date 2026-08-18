from __future__ import annotations

import hashlib
import json
import logging
import queue
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2

# Must be installed before mediapipe is imported -- it emits this from module
# scope, once per import path it touches. Only logs/app.log rotates;
# logs/err.log is a raw launchd/app-bundle stderr redirect that grows forever,
# and this warning was the entirety of its contents.
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated",
    category=UserWarning,
)

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
STATE = ROOT / "state"
CALIBRATION_PATH = STATE / "calibration.json"
CAMERA_PATH = STATE / "camera.json"

FACE_MODEL_PATH = MODELS / "face_landmarker.task"
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
PHONE_MODEL_PATH = MODELS / "efficientdet_lite0.tflite"
PHONE_MODEL_SHA256 = "40338edf5ec70d43e318b0a716a84d4564cd1802759a7a07170c7e43796dbf58"


class CameraUnavailable(Exception):
    pass


# ---------- calibration plausibility ----------
# A baseline is the pose the user is expected to hold for a whole block, so it
# has to be roughly head-on. Baselines of ~48 deg yaw were recorded in practice
# (calibration starts the instant the button is clicked, while the pointer is
# still moving away and the other monitors are going black). Against a 40 deg
# look-away threshold that makes the user's normal posture a *permanent*
# violation: the "clean" condition the resolve debounce waits for is
# unreachable, so the alarm never stops and the only exit is force-quit.
MAX_BASELINE_DEGREES = 25.0
# The samples all come from one supposedly-still head. A wide spread means the
# user was still moving, so the median is not a pose they will hold.
MAX_BASELINE_SPREAD_DEGREES = 20.0
MIN_CALIBRATION_SAMPLES = 10


def _baseline_from(samples: list[float]) -> tuple[float | None, str]:
    """Median of the calibration samples, or (None, reason) if the result can't
    be trusted as a resting pose."""
    if len(samples) < MIN_CALIBRATION_SAMPLES:
        return None, f"only {len(samples)} usable samples"
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    lo = ordered[int(len(ordered) * 0.05)]
    hi = ordered[int(len(ordered) * 0.95)]
    if hi - lo > MAX_BASELINE_SPREAD_DEGREES:
        return None, f"samples spread over {hi - lo:.1f} deg (still moving?)"
    if abs(median) > MAX_BASELINE_DEGREES:
        return None, f"baseline {median:.1f} deg is not head-on"
    return median, ""


@dataclass(frozen=True)
class FrameReading:
    ok: bool
    error: str | None = None
    face_present: bool = False
    pitch_delta_deg: float | None = None      # smoothed, relative to calibration baseline
    yaw_delta_deg: float | None = None        # smoothed |deviation| from baseline, sign-free
    phone_confidence: float = 0.0
    phone_detector_available: bool = True
    # Gaze. All None unless detect_gaze is on AND a calibration loaded; every
    # consumer must treat them as optional. Still no image data here -- these
    # are coordinates and a name, nothing that could reconstruct a frame.
    gaze_screen: str | None = None            # calibrated monitor being looked at
    gaze_xy: tuple[float, float] | None = None  # pixels within that monitor
    gaze_on_screen: bool = False
    gaze_available: bool = False              # tracker loaded and running


class SustainedCondition:
    """Leaky bucket. Fills at 1x while active, drains at decay_rate while clear.

    Draining rather than hard-resetting on a single clean frame means a
    repeated brief-glance pattern (3.5s down, 0.5s up, repeat) still
    eventually accumulates past the trigger threshold.
    """

    def __init__(self, trigger_s: float, decay_rate: float = 0.5):
        self.trigger_s = trigger_s
        self.decay_rate = decay_rate
        self.acc = 0.0

    def update(self, active: bool, dt: float) -> bool:
        if active:
            self.acc = min(self.trigger_s, self.acc + dt)
        else:
            self.acc = max(0.0, self.acc - dt * self.decay_rate)
        return self.acc >= self.trigger_s

    def reset(self) -> None:
        self.acc = 0.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_model(path: Path, expected_sha256: str) -> bool:
    return path.exists() and _sha256(path) == expected_sha256


def pitch_deg(matrix) -> float:
    """Pitch from a facial transformation matrix's rotation block. The sign
    convention is not guaranteed across MediaPipe versions -- see
    _load_pitch_sign() / run_sign_calibration_wizard()."""
    R = np.asarray(matrix)[:3, :3]
    return float(np.degrees(np.arctan2(-R[2, 1], R[2, 2])))


def yaw_deg(matrix) -> float:
    """Yaw (head turned left/right) from the same rotation block.

    Unlike pitch, the sign is never needed: turning away to the left and to
    the right are equally "not looking at the screen", so the caller compares
    |yaw - baseline|. That sidesteps the sign-calibration problem entirely."""
    R = np.asarray(matrix)[:3, :3]
    return float(np.degrees(np.arctan2(R[2, 0], np.hypot(R[2, 1], R[2, 2]))))


def _load_pitch_sign() -> int:
    if CALIBRATION_PATH.exists():
        try:
            return int(json.loads(CALIBRATION_PATH.read_text())["sign"])
        except Exception:
            pass
    log.warning(
        "pitch sign not calibrated (state/calibration.json missing) -- "
        "using sign=1; run `python vision.py --calibrate-sign` once before "
        "trusting look-down detection"
    )
    return 1


# ---------- camera resolution (fixes I-29) ----------

def _list_camera_names() -> list[str]:
    out = subprocess.run(
        ["system_profiler", "SPCameraDataType"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    names = []
    for line in out.stdout.splitlines():
        if not line.startswith("    ") or line.startswith("      "):
            continue
        stripped = line.strip()
        if stripped.endswith(":"):
            names.append(stripped[:-1])
    return names


def resolve_camera_index(cfg: dict) -> int:
    """Resolve a stable camera index by device name. Continuity Camera can
    silently take over index 0 on macOS and the mapping shifts depending on
    whether the phone is nearby, so a bare index is not reliable. Falls back
    to cfg['camera_index'] if the name can't be resolved. Cached to
    state/camera.json so this only runs once per config."""
    if CAMERA_PATH.exists():
        try:
            cached = json.loads(CAMERA_PATH.read_text())
            if cached.get("name_hint") == cfg.get("camera_name_hint"):
                return cached["index"]
        except Exception:
            pass

    hint = cfg.get("camera_name_hint")
    index = cfg.get("camera_index", 0)
    resolved_name = None
    if hint:
        try:
            names = _list_camera_names()
            for i, n in enumerate(names):
                if hint.lower() in n.lower():
                    index, resolved_name = i, n
                    break
            else:
                log.warning("camera_name_hint %r not found among %r; using camera_index %d", hint, names, index)
        except Exception as e:
            log.warning("camera name resolution failed (%s); using camera_index %d", e, index)

    STATE.mkdir(exist_ok=True)
    CAMERA_PATH.write_text(json.dumps({"name_hint": hint, "index": index, "resolved_name": resolved_name}))
    return index


def resolved_camera_name() -> str | None:
    if CAMERA_PATH.exists():
        try:
            return json.loads(CAMERA_PATH.read_text()).get("resolved_name")
        except Exception:
            return None
    return None


def open_camera(cfg: dict) -> cv2.VideoCapture:
    index = resolve_camera_index(cfg)
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        raise CameraUnavailable(f"camera index {index} would not open")
    # Gaze needs the capture resolution to match the one its camera intrinsics
    # were solved at -- fx, fy, cx, cy are in pixels and do not transfer. When
    # gaze is on we therefore capture at its calibrated size instead of the
    # 640x480 the head-pose path is happy with. store.load_calibration refuses
    # a mismatch anyway; this is what stops it ever being a mismatch.
    if cfg.get("detect_gaze", False):
        width = cfg.get("gaze_capture_width", 1280)
        height = cfg.get("gaze_capture_height", 720)
    else:
        width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ---------- worker thread: capture + inference (fixes I-07, I-18, I-28) ----------

class VisionWorker(threading.Thread):
    """Owns the camera and both MediaPipe detectors. Runs continuously on a
    background thread and pushes immutable FrameReading objects onto
    output_queue -- the Tk main thread only ever reads that queue, never
    touches cv2 or mediapipe objects directly (avoids the classic
    off-main-thread Tk crash and torn-read state bugs).

    Camera open/close and calibration control methods are called from the Tk
    main thread but only set state/flags; the actual cv2.VideoCapture calls
    happen inside this thread's run() loop, guarded by _cap_lock, so all cv2
    usage stays confined to one thread.
    """

    def __init__(self, cfg: dict, output_queue: queue.Queue, preview_queue: queue.Queue | None = None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.output_queue = output_queue
        # Small downscaled frame for the GUI thumbnail only -- in-memory,
        # maxsize=1 (always the newest), never written to disk or logged.
        # FrameReading itself deliberately carries no image data.
        self.preview_queue = preview_queue

        self.cap: cv2.VideoCapture | None = None
        self._cap_lock = threading.Lock()
        # Open handshake. _open_seq is bumped by each requester; the worker
        # publishes _open_done_seq for the request it actually serviced, so a
        # caller only ever accepts a result produced for its own request. A
        # bare Event let a completion that was already in flight satisfy the
        # *next* caller's wait, which then read a still-None _open_result and
        # reported "camera unavailable" for a camera that opened fine.
        self._open_cv = threading.Condition()
        self._open_requested = threading.Event()
        self._open_result: bool | None = None
        self._open_seq = 0
        self._open_done_seq = 0
        # NOT self._stop: threading.Thread defines its own _stop(), and
        # Thread._wait_for_tstate_lock() calls it once the thread's state lock
        # is released. Shadowing it with an Event made every join()/is_alive()
        # on a finished worker raise "TypeError: 'Event' object is not
        # callable" -- swallowed by shutdown()'s except-pass in main.py, so
        # the join silently never confirmed the thread had actually stopped.
        self._stop_event = threading.Event()

        # Calibration state is written by the tick thread (via SessionTimer's
        # callbacks) and read by this thread inside _read_once, so it needs the
        # same treatment self.cap gets from _cap_lock. Without it,
        # start_calibration() could rebind _calib_samples while _read_once
        # still held the old list, silently dropping samples from the baseline.
        self._calib_lock = threading.Lock()
        self._calibrating = False
        self._calib_samples: list[tuple[float, float, float]] = []   # (ts, pitch, yaw)
        self._calib_start = 0.0
        self._baseline_pitch: float | None = None
        self._baseline_yaw: float | None = None
        self._pitch_sign = _load_pitch_sign()
        self._pitch_ema: float | None = None
        self._yaw_ema: float | None = None

        self._tick_count = 0
        self._last_phone_confidence = 0.0
        # Held between ticks for the same reason _last_phone_confidence is:
        # gaze_every_n_ticks means most ticks skip inference, and reporting
        # None on those would make every downstream sustain-counter flicker.
        self._last_gaze: tuple[str | None, tuple[float, float] | None, bool] = (None, None, False)

        self.degraded: list[str] = []
        self.resolved_camera_name: str | None = None
        self.face_landmarker = None
        self.phone_detector = None
        self._load_models()

        self.gaze_tracker = None
        self._gaze_status: dict = {"state": "disabled", "message": "Gaze detection is off."}
        self._load_gaze()

    def _load_models(self) -> None:
        if _verify_model(FACE_MODEL_PATH, FACE_MODEL_SHA256):
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        else:
            log.error("face landmarker model missing or checksum mismatch -- presence detection unavailable")
            self.degraded.append("face")

        if self.cfg.get("detect_phone", True):
            if _verify_model(PHONE_MODEL_PATH, PHONE_MODEL_SHA256):
                try:
                    opts = mp_vision.ObjectDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(PHONE_MODEL_PATH)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        max_results=5,
                        score_threshold=0.3,
                    )
                    self.phone_detector = mp_vision.ObjectDetector.create_from_options(opts)
                except Exception as e:
                    log.error("phone detector failed to load: %s", e)
                    self.phone_detector = None
            else:
                log.error("phone detection model missing or checksum mismatch")
                self.phone_detector = None
            if self.phone_detector is None:
                self.degraded.append("phone")

    def _load_gaze(self) -> None:
        """Build the gaze tracker, or record a degradation and carry on.

        Every failure here is non-fatal by design. Gaze is an addition to
        detection, never a precondition for it: torch may not be installed, the
        checkpoint is a manual download, and the calibration does not exist
        until tools/calibrate_screens.py has been run. In all of those cases
        Argus must still start and still watch the desk with head pose alone.

        The import is inside the function for the same reason -- torch is in
        requirements-gaze.txt, not requirements.txt, so importing it at module
        scope would break every install that never turns gaze on.
        """
        if not self.cfg.get("detect_gaze", False):
            self._gaze_status = {"state": "disabled", "message": "Gaze detection is off."}
            return

        try:
            from gaze.model import load_checkpoint, pick_device  # noqa: PLC0415
            from gaze.store import load_calibration  # noqa: PLC0415
            from gaze.tracker import GazeTracker  # noqa: PLC0415
        except ImportError as e:
            log.error("gaze enabled but its dependencies are missing (%s) -- "
                      "install requirements-gaze.txt", e)
            self._gaze_status = {
                "state": "no_deps",
                "message": "Gaze needs extra packages. Run: "
                           ".venv/bin/pip install -r requirements-gaze.txt",
            }
            self._degrade_gaze()
            return

        checkpoint = ROOT / self.cfg["gaze_checkpoint_path"]
        calibration_path = ROOT / self.cfg["gaze_calibration_path"]
        try:
            capture_size = (
                self.cfg.get("gaze_capture_width", 1280),
                self.cfg.get("gaze_capture_height", 720),
            )
            calibration = load_calibration(
                calibration_path,
                expect_capture_size=capture_size,
                expect_checkpoint=checkpoint if checkpoint.exists() else None,
            )
            device = pick_device()
            model = load_checkpoint(checkpoint, device)
            self.gaze_tracker = GazeTracker(
                model,
                calibration.camera_matrix,
                calibration.dist_coeffs,
                calibration.screens,
                device,
                person_idx=self.cfg.get("gaze_person_idx", 0),
            )
            self._gaze_status = {
                "state": "ok",
                "message": "Calibrated for {} screen(s): {}".format(
                    len(calibration.screens),
                    ", ".join(scr.name for scr in calibration.screens),
                ),
            }
            if "gaze" in self.degraded:
                self.degraded.remove("gaze")
            log.info("gaze tracking active on %s with %d calibrated screen(s): %s",
                     device, len(calibration.screens),
                     ", ".join(scr.name for scr in calibration.screens))
        except Exception as e:
            # Broad on purpose: a missing checkpoint, a stale calibration, a
            # torch backend that will not initialise and a corrupt file all
            # mean the same thing to the caller -- no gaze this run.
            log.error("gaze unavailable: %s", e)
            self.gaze_tracker = None
            try:
                from gaze.store import calibration_status  # noqa: PLC0415
                self._gaze_status = calibration_status(calibration_path, capture_size)
            except Exception:
                self._gaze_status = {"state": "stale", "message": str(e)}
            self._degrade_gaze()

    def _degrade_gaze(self) -> None:
        if "gaze" not in self.degraded:
            self.degraded.append("gaze")

    def gaze_status(self) -> dict:
        """What to tell the user about gaze right now. Never raises."""
        return dict(self._gaze_status)

    def reload_gaze(self) -> dict:
        """Re-read the calibration and rebuild the tracker.

        Exists so finishing a calibration takes effect immediately. Requiring a
        restart to pick up a file the app itself just caused to be written
        would read as the calibration having failed.
        """
        self.gaze_tracker = None
        self._load_gaze()
        return self.gaze_status()

    @property
    def gaze_available(self) -> bool:
        return self.gaze_tracker is not None

    @property
    def phone_detector_available(self) -> bool:
        return self.phone_detector is not None

    # ---------- camera lifecycle ----------
    def request_open(self, timeout: float = 5.0) -> bool:
        with self._cap_lock:
            if self.cap is not None:
                return True
        # Waiting on a sequence number rather than an Event matters: on timeout
        # the old code returned whatever the *previous* open produced, so a
        # stale True sent the timer into CALIBRATING against a camera that was
        # never opened -- and a stale set() made a good camera report as
        # unavailable.
        with self._open_cv:
            self._open_seq += 1
            my_seq = self._open_seq
            self._open_requested.set()
            deadline = time.monotonic() + timeout
            while self._open_done_seq < my_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("camera open did not complete within %.1fs", timeout)
                    return False
                self._open_cv.wait(remaining)
            return bool(self._open_result)

    def request_close(self) -> None:
        with self._open_cv:
            if self._open_requested.is_set():
                # Cancel a request the worker has not picked up yet, and release
                # its caller now instead of leaving it to time out after the
                # full 5s against an open that will never happen.
                self._open_requested.clear()
                self._open_result = False
                self._open_done_seq = self._open_seq
                self._open_cv.notify_all()
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    # ---------- calibration ----------
    def start_calibration(self) -> None:
        with self._calib_lock:
            self._calib_samples = []
            self._calib_start = time.monotonic()
            self._calibrating = True
            self._baseline_pitch = None
            self._baseline_yaw = None
            self._pitch_ema = None
            self._yaw_ema = None
            # Otherwise the previous session's last reading is still reported
            # on every tick that skips detection (phone_detect_every_n_ticks),
            # counting toward the new session's phone_sustain_seconds budget.
            self._last_phone_confidence = 0.0
            self._last_gaze = (None, None, False)
        if self.gaze_tracker is not None:
            # Carrying a pose across a session boundary makes the first frames
            # of the new session converge toward where the head used to be.
            self.gaze_tracker.reset()

    def finish_calibration(self) -> bool:
        """Discards the first second of samples (settling time) and takes the
        median of the rest as the baseline -- what makes a fixed relative
        threshold work across different desks and lid angles (fixes I-08).

        Returns True if a usable baseline was committed. A baseline that is not
        plausibly head-on, or that came from a head that was still moving, is
        rejected outright (both baselines left None) rather than committed: an
        off-axis baseline makes the user's normal posture a permanent violation
        that no debounce can ever clear. Callers must treat False as a failed
        calibration and end the block/session rather than running blind.

        MUST be called when the calibration window closes. It previously was
        not called anywhere at all, so _calibrating stayed True forever,
        _baseline_pitch stayed None, and pitch_delta_deg was therefore always
        None -- which silently made look-down detection impossible to trigger
        no matter how it was configured."""
        with self._calib_lock:
            self._calibrating = False
            samples = list(self._calib_samples)
            start = self._calib_start
            usable_p = [p for (ts, p, y) in samples if ts - start >= 1.0]
            usable_y = [y for (ts, p, y) in samples if ts - start >= 1.0]
            pitch, pitch_reason = _baseline_from(usable_p)
            yaw, yaw_reason = _baseline_from(usable_y)
            if pitch is None or yaw is None:
                # Partial baselines are worse than none: half the detectors
                # would run against a pose the user is not actually holding.
                self._baseline_pitch = None
                self._baseline_yaw = None
                ok = False
            else:
                self._baseline_pitch = pitch
                self._baseline_yaw = yaw
                ok = True
            n = len(usable_p)
        if ok:
            log.info("calibration finished: baseline pitch=%s yaw=%s (%d samples)",
                     round(pitch, 1), round(yaw, 1), n)
        else:
            log.warning(
                "calibration rejected (%d samples): pitch %s / yaw %s",
                n, pitch_reason or "ok", yaw_reason or "ok",
            )
        return ok

    @property
    def calibrated(self) -> bool:
        with self._calib_lock:
            return self._baseline_pitch is not None and self._baseline_yaw is not None

    # ---------- main loop ----------
    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    self._run_once()
                except Exception:
                    # Without this the thread dies for the whole life of the
                    # process and nothing restarts it: request_open() then blocks
                    # its full timeout and returns False forever, so the app spends
                    # the rest of the run reporting "camera open did not complete
                    # within 5.0s" with no hint of the real cause. Engine._loop has
                    # the same guard for the same reason.
                    log.exception("error in vision worker loop")
                    time.sleep(0.5)
        finally:
            self._close_detectors()

    def _close_detectors(self) -> None:
        """Release the native MediaPipe graphs. Dropping the Python reference
        does not free the TFLite interpreter behind them; close() does.

        Called from run()'s finally rather than from stop() on purpose. detect()
        only ever runs on this thread, so closing here cannot race a call that
        is still in flight -- closing from the caller's thread could, and using
        a closed task object is a native use-after-free, not a Python
        exception. Same reasoning as the _cap_lock comment in _run_once."""
        for name in ("face_landmarker", "phone_detector"):
            detector = getattr(self, name)
            if detector is None:
                continue
            try:
                detector.close()
            except Exception:
                log.debug("%s close failed", name, exc_info=True)
            setattr(self, name, None)

    def _take_open_request(self) -> int | None:
        """Claim a pending open request, if any, and return its sequence
        number. Claiming and clearing under _open_cv is what stops a request
        made concurrently with the claim from being dropped."""
        with self._open_cv:
            if not self._open_requested.is_set():
                return None
            self._open_requested.clear()
            return self._open_seq

    def _publish_open_result(self, seq: int, result: bool) -> None:
        with self._open_cv:
            self._open_result = result
            if seq > self._open_done_seq:
                self._open_done_seq = seq
            self._open_cv.notify_all()

    def _perform_open(self, seq: int) -> None:
        result = False
        try:
            with self._cap_lock:
                if self.cap is not None:
                    result = True
                else:
                    try:
                        self.cap = open_camera(self.cfg)
                        self.resolved_camera_name = resolved_camera_name()
                        result = True
                    except Exception as e:
                        # Not just CameraUnavailable: resolve_camera_index does
                        # unguarded filesystem I/O, and an escaping error there
                        # used to kill this thread outright.
                        log.warning("camera open failed: %s", e)
                        self.cap = None
                        result = False
        finally:
            # In a finally so a waiter is never left hanging for the full
            # timeout, whatever went wrong above.
            self._publish_open_result(seq, result)

    def _run_once(self) -> None:
        pending = self._take_open_request()
        if pending is not None:
            self._perform_open(pending)

        # The read MUST happen while holding _cap_lock. Grabbing the handle
        # under the lock and then reading outside it leaves a window where
        # request_close() can release the AVFoundation capture session
        # while a frame grab is still in flight on it -- OpenCV's
        # CaptureDelegate then messages freed memory and the whole process
        # segfaults (objc_msgSend on a dangling pointer). That is exactly
        # what killed the first session that ever ran to completion.
        with self._cap_lock:
            cap = self.cap
            if cap is None:
                reading = None
            else:
                reading = self._read_once(cap)
                # Release on any read failure so the next
                # request_open() call (driven by timer.py's backoff
                # schedule) actually attempts a fresh open instead of
                # trivially reporting success against a stale, broken
                # capture handle.
                if not reading.ok and self.cap is not None:
                    self.cap.release()
                    self.cap = None

        if reading is None:
            time.sleep(0.05)
            return
        if not reading.ok:
            time.sleep(0.1)
        self._push(reading)

    def _push(self, reading: FrameReading) -> None:
        try:
            self.output_queue.put_nowait(reading)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(reading)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        self.request_close()

    def _run_gaze(self, rgb, face_result):
        """Gaze for one frame, reusing the landmarks the face detector already
        produced. Returns (screen_name, (x, y), on_screen).

        Sharing the landmarks is the whole point of taking them as an argument
        rather than letting the tracker find a face itself: upstream runs its
        own MediaPipe FaceMesh, which here would mean detecting the same face
        twice per tick.

        Note the head pose is NOT shared. MediaPipe's transformation matrix is
        expressed against its own assumed camera and is not metric; projecting
        a gaze ray onto a screen plane needs real millimetres, which only
        solvePnP against calibrated intrinsics gives.
        """
        n = max(1, self.cfg.get("gaze_every_n_ticks", 1))
        if self._tick_count % n != 0:
            return self._last_gaze

        try:
            landmarks = np.array([[lm.x, lm.y] for lm in face_result.face_landmarks[0]])
            reading = self.gaze_tracker.update(rgb, landmarks)
        except Exception:
            # One bad frame must not take the session down, and must not leave
            # a stale hit latched either -- report nothing until it recovers.
            log.debug("gaze frame failed", exc_info=True)
            self._last_gaze = (None, None, False)
            return self._last_gaze

        if not reading.ok:
            self._last_gaze = (None, None, False)
        else:
            self._last_gaze = (reading.screen_name, reading.screen_xy, reading.on_any_screen)
        return self._last_gaze

    def _read_once(self, cap: cv2.VideoCapture) -> FrameReading:
        ok, frame = cap.read()
        if not ok or frame is None:
            return FrameReading(ok=False, error="camera_read_failed")

        self._tick_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.preview_queue is not None:
            small = cv2.resize(rgb, (160, 120))
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.preview_queue.put_nowait(small)
            except queue.Full:
                pass

        face_present = False
        pitch_delta = None
        yaw_delta = None
        if self.face_landmarker is not None:
            result = self.face_landmarker.detect(mp_image)
            face_present = len(result.face_landmarks) > 0
            if face_present and result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]
                raw_pitch = pitch_deg(matrix)
                raw_yaw = yaw_deg(matrix)
                with self._calib_lock:
                    if self._calibrating:
                        self._calib_samples.append((time.monotonic(), raw_pitch, raw_yaw))
                    else:
                        if self._baseline_pitch is not None:
                            delta = (raw_pitch - self._baseline_pitch) * self._pitch_sign
                            self._pitch_ema = delta if self._pitch_ema is None else (0.3 * delta + 0.7 * self._pitch_ema)
                            pitch_delta = self._pitch_ema
                        if self._baseline_yaw is not None:
                            # absolute deviation: turning either way counts equally
                            d = abs(raw_yaw - self._baseline_yaw)
                            self._yaw_ema = d if self._yaw_ema is None else (0.3 * d + 0.7 * self._yaw_ema)
                            yaw_delta = self._yaw_ema

        gaze_screen = None
        gaze_xy = None
        gaze_on_screen = False
        if self.gaze_tracker is not None and face_present:
            gaze_screen, gaze_xy, gaze_on_screen = self._run_gaze(rgb, result)

        phone_confidence = self._last_phone_confidence
        if self.phone_detector is not None and self.cfg.get("detect_phone", True):
            n = max(1, self.cfg.get("phone_detect_every_n_ticks", 1))
            if self._tick_count % n == 0:
                det_result = self.phone_detector.detect(mp_image)
                best = 0.0
                for d in det_result.detections:
                    for cat in d.categories:
                        if cat.category_name and "phone" in cat.category_name.lower():
                            best = max(best, cat.score or 0.0)
                phone_confidence = best
                self._last_phone_confidence = best

        return FrameReading(
            ok=True,
            face_present=face_present,
            pitch_delta_deg=pitch_delta,
            yaw_delta_deg=yaw_delta,
            phone_confidence=phone_confidence,
            phone_detector_available=self.phone_detector is not None,
            gaze_screen=gaze_screen,
            gaze_xy=gaze_xy,
            gaze_on_screen=gaze_on_screen,
            gaze_available=self.gaze_tracker is not None,
        )


# ---------- one-time pitch-sign calibration wizard ----------

def run_sign_calibration_wizard(cfg: dict) -> None:
    """Interactive, run once (`python vision.py --calibrate-sign`). Confirms
    which direction pitch moves when you look down, so the sign is verified
    empirically rather than assumed -- MediaPipe's convention isn't guaranteed
    across versions."""
    print("Pitch-sign calibration. Look at the camera normally, then press Enter.")
    input()
    cap = open_camera(cfg)
    landmarker_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_opts)

    def sample() -> float | None:
        for _ in range(10):
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(img)
            if result.face_landmarks and result.facial_transformation_matrixes:
                return pitch_deg(result.facial_transformation_matrixes[0])
        return None

    neutral = sample()
    if neutral is None:
        cap.release()
        sys_exit("No face detected. Aborting -- fix lighting/camera and retry.")

    print(f"Neutral pitch: {neutral:.1f} deg. Now look down at your desk and press Enter.")
    input()
    down = sample()
    cap.release()
    if down is None:
        sys_exit("No face detected while looking down. Aborting.")

    diff = down - neutral
    if abs(diff) < 3:
        sys_exit(f"Pitch barely moved ({diff:.1f} deg) -- can't determine sign reliably. Aborting.")

    sign = 1 if diff > 0 else -1
    STATE.mkdir(exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps({"sign": sign, "neutral_sample": neutral, "down_sample": down}))
    print(f"Sign determined: {sign}. Saved to {CALIBRATION_PATH}.")


def sys_exit(msg: str) -> None:
    import sys
    sys.exit(msg)


if __name__ == "__main__":
    import sys as _sys

    import config as _config
    if "--calibrate-sign" in _sys.argv:
        run_sign_calibration_wizard(_config.load_config())
    else:
        print("Usage: python vision.py --calibrate-sign")
