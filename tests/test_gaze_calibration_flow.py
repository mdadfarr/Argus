"""The calibration lifecycle: knowing whether we are calibrated, saying so, and
being able to redo it.

The behaviour worth protecting is that an uncalibrated setup is *legible*.
Before this, gaze failing produced a bare `DEGRADED: GAZE` banner that could
mean torch was missing, or the file had never been written, or it was written
for a different camera resolution -- three problems with three different fixes.
"""
from __future__ import annotations

import json
import queue
import threading
from unittest import mock

import pytest

import vision
from gaze.store import (
    STATUS_MISSING,
    STATUS_OK,
    STATUS_STALE,
    Calibration,
    calibration_status,
    save_calibration,
)

np = pytest.importorskip("numpy")


@pytest.fixture
def a_calibration():
    from gaze.geometry import plane_from_corners

    corners = np.array([[-300.0, -170.0, -500.0], [300.0, -170.0, -500.0],
                        [300.0, 170.0, -500.0], [-300.0, 170.0, -500.0]])
    return Calibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        capture_size=(1280, 720),
        screens=[plane_from_corners("main", corners, (1920, 1080))],
        created_at="2026-01-01T00:00:00Z",
    )


# ---------- status reporting ----------

def test_absent_file_reads_as_never_calibrated(tmp_path):
    status = calibration_status(tmp_path / "nothing.json")
    assert status["state"] == STATUS_MISSING
    assert status["screens"] == []


def test_empty_file_reads_as_never_calibrated(tmp_path):
    """A zero-byte file is what a half-finished or interrupted calibration
    leaves behind, and it must not be mistaken for a real one."""
    path = tmp_path / "cal.json"
    path.write_text("")
    assert calibration_status(path)["state"] == STATUS_MISSING


def test_valid_file_reports_the_screen_names(tmp_path, a_calibration):
    """Naming the screens is the difference between 'you calibrated' and 'you
    calibrated the screen you are actually sitting at'."""
    path = tmp_path / "cal.json"
    save_calibration(path, a_calibration)

    status = calibration_status(path)
    assert status["state"] == STATUS_OK
    assert status["screens"] == ["main"]
    assert "main" in status["message"]


def test_wrong_capture_size_reads_as_stale_not_ok(tmp_path, a_calibration):
    path = tmp_path / "cal.json"
    save_calibration(path, a_calibration)

    status = calibration_status(path, capture_size=(640, 480))
    assert status["state"] == STATUS_STALE
    assert "recalibrat" in status["message"].lower()


def test_corrupt_file_reads_as_stale_rather_than_raising(tmp_path):
    """This runs inside a UI poll. Raising here would take out the snapshot."""
    path = tmp_path / "cal.json"
    path.write_text("{ not json")
    assert calibration_status(path)["state"] == STATUS_STALE


def test_status_never_raises_on_junk(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"schema": 1, "screens": "not a list"}))
    assert calibration_status(path)["state"] in (STATUS_STALE, STATUS_MISSING)


# ---------- the worker's view ----------

@pytest.fixture
def worker(cfg, monkeypatch):
    def make(**overrides):
        conf = dict(cfg)
        conf.update(overrides)
        monkeypatch.setattr(vision.VisionWorker, "_load_models", lambda self: None)
        return vision.VisionWorker(conf, queue.Queue())
    return make


def test_disabled_gaze_reports_disabled(worker):
    assert worker(detect_gaze=False).gaze_status()["state"] == "disabled"


def test_uncalibrated_gaze_says_which_problem_it_is(worker):
    """The point of the whole change: the state distinguishes 'no calibration'
    from 'no dependencies', because they have different fixes."""
    w = worker(detect_gaze=True, gaze_calibration_path="state/not_written_yet.json")
    assert w.gaze_status()["state"] in ("missing", "no_deps", "stale")
    assert w.gaze_status()["message"]


def test_failure_marks_degraded(worker):
    w = worker(detect_gaze=True, gaze_calibration_path="state/not_written_yet.json")
    assert "gaze" in w.degraded


def test_degraded_is_not_appended_twice_on_reload(worker):
    """reload_gaze runs every time calibration finishes; without the guard the
    banner would grow GAZE,GAZE,GAZE across repeated attempts."""
    w = worker(detect_gaze=True, gaze_calibration_path="state/not_written_yet.json")
    w.reload_gaze()
    w.reload_gaze()
    assert w.degraded.count("gaze") == 1


def test_reload_returns_the_current_status(worker):
    w = worker(detect_gaze=False)
    assert w.reload_gaze()["state"] == "disabled"


# ---------- the engine's view ----------

class FakeWorker:
    def __init__(self, state="missing"):
        self._state = state
        self.degraded = [] if state == "ok" else ["gaze"]
        self.closed = 0
        self.reloaded = 0

    def gaze_status(self):
        return {"state": self._state, "message": f"status is {self._state}"}

    def reload_gaze(self):
        self.reloaded += 1
        self._state = "ok"
        return self.gaze_status()

    def request_close(self):
        self.closed += 1


def bare_engine(cfg, worker=None, session_active=False):
    """An Engine with only the fields the gaze paths touch."""
    import main

    e = main.Engine.__new__(main.Engine)
    e.cfg = dict(cfg)
    e.lock = threading.RLock()
    e.worker = worker or FakeWorker()
    e.session_active = session_active
    e.focus = None
    e._gaze_calibrating = False
    e._gaze_calib_message = ""
    return e


def test_snapshot_flags_that_calibration_is_needed(cfg):
    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"))
    g = e.gaze_snapshot()

    assert g["enabled"] is True
    assert g["needs_calibration"] is True
    assert g["can_calibrate"] is True


def test_snapshot_reports_nothing_needed_once_calibrated(cfg):
    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("ok"))
    assert e.gaze_snapshot()["needs_calibration"] is False


def test_cannot_calibrate_while_a_session_runs(cfg):
    """Calibration takes the camera away and puts fullscreen targets up. Doing
    that mid-session would fail the session the user is being watched for."""
    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"), session_active=True)

    assert e.gaze_snapshot()["can_calibrate"] is False
    result = e.gaze_calibrate()
    assert result["ok"] is False
    assert "session" in result["error"].lower()


def test_cannot_calibrate_when_gaze_is_off(cfg):
    e = bare_engine({**cfg, "detect_gaze": False})
    assert e.gaze_calibrate()["ok"] is False


def test_cannot_calibrate_twice_at_once(cfg):
    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"))
    e._gaze_calibrating = True

    assert e.gaze_calibrate()["ok"] is False
    assert e.gaze_snapshot()["can_calibrate"] is False


def test_missing_dependencies_cannot_be_fixed_by_calibrating(cfg):
    """Offering a Calibrate button that cannot possibly work is worse than
    offering none -- it sends the user down the wrong path."""
    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("no_deps"))
    assert e.gaze_snapshot()["can_calibrate"] is False


# ---------- running it ----------

def test_calibration_releases_the_camera_before_spawning(cfg, tmp_path):
    """macOS will not hand the same camera to a second process while the worker
    still holds it, so the release is load-bearing, not tidiness."""
    import main

    worker = FakeWorker("missing")
    e = bare_engine({**cfg, "detect_gaze": True}, worker)
    checkpoint = tmp_path / "p00.ckpt"
    checkpoint.write_bytes(b"stub")
    intrinsics = tmp_path / "camera_matrix.yaml"
    intrinsics.write_text("camera_matrix: [[1,0,0],[0,1,0],[0,0,1]]\ndist_coeff: [0,0,0,0,0]\n")
    e.cfg["gaze_checkpoint_path"] = str(checkpoint)
    e.cfg["gaze_camera_matrix_path"] = str(intrinsics)

    completed = mock.Mock(returncode=0, stdout="ok", stderr="")
    with mock.patch.object(main.subprocess, "run", return_value=completed) as run, \
         mock.patch.object(main.time, "sleep"):
        e._run_gaze_calibration()

    assert worker.closed == 1
    assert run.called
    assert worker.reloaded == 1
    assert e._gaze_calibrating is False


def test_a_missing_checkpoint_is_reported_not_spawned(cfg):
    """Spawning the tool without a checkpoint would fail deep inside a
    subprocess, and the user would see its traceback rather than the one thing
    they need to know: download the model."""
    import main

    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"))
    e.cfg["gaze_checkpoint_path"] = "state/definitely_absent.ckpt"

    with mock.patch.object(main.subprocess, "run") as run:
        e._run_gaze_calibration()

    assert not run.called
    assert "checkpoint" in e._gaze_calib_message.lower()
    assert e._gaze_calibrating is False


def test_a_failing_subprocess_surfaces_its_last_line(cfg, tmp_path):
    import main

    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"))
    checkpoint = tmp_path / "p00.ckpt"
    checkpoint.write_bytes(b"stub")
    intrinsics = tmp_path / "camera_matrix.yaml"
    intrinsics.write_text("camera_matrix: [[1,0,0],[0,1,0],[0,0,1]]\ndist_coeff: [0,0,0,0,0]\n")
    e.cfg["gaze_checkpoint_path"] = str(checkpoint)
    e.cfg["gaze_camera_matrix_path"] = str(intrinsics)

    failed = mock.Mock(returncode=1, stdout="", stderr="line one\nonly 3.2 deg of parallax")
    with mock.patch.object(main.subprocess, "run", return_value=failed), \
         mock.patch.object(main.time, "sleep"):
        e._run_gaze_calibration()

    assert "parallax" in e._gaze_calib_message
    assert e._gaze_calibrating is False


def test_the_busy_flag_always_clears(cfg):
    """If this leaks, the Calibrate button stays disabled until restart."""
    import main

    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"))
    e.cfg["gaze_checkpoint_path"] = "state/absent.ckpt"
    e._gaze_calibrating = True

    with mock.patch.object(main.subprocess, "run", side_effect=OSError("boom")):
        e._run_gaze_calibration()

    assert e._gaze_calibrating is False


def test_uncalibrated_camera_is_reported_not_spawned(cfg, tmp_path):
    """Camera intrinsics come from a chessboard session that has to happen
    before screen calibration can mean anything. Without this check the
    subprocess fails on a missing yaml and shows its traceback."""
    import main

    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"))
    checkpoint = tmp_path / "p00.ckpt"
    checkpoint.write_bytes(b"stub")
    e.cfg["gaze_checkpoint_path"] = str(checkpoint)
    e.cfg["gaze_camera_matrix_path"] = "state/never_calibrated.yaml"

    with mock.patch.object(main.subprocess, "run") as run, \
         mock.patch.object(main.time, "sleep"):
        e._run_gaze_calibration()

    assert not run.called
    assert "chessboard" in e._gaze_calib_message.lower()


def test_a_failure_stays_visible_after_the_attempt_ends(cfg):
    """A run that fails on a missing prerequisite finishes in well under a
    second. If the reason is only shown while `calibrating` is true it flashes
    past between UI polls and the button looks like it did nothing -- which is
    exactly how this presented in practice."""
    import main

    e = bare_engine({**cfg, "detect_gaze": True}, FakeWorker("missing"))
    e.cfg["gaze_checkpoint_path"] = "state/absent.ckpt"

    with mock.patch.object(main.subprocess, "run"):
        e._run_gaze_calibration()

    g = e.gaze_snapshot()
    assert g["calibrating"] is False
    assert "checkpoint" in g["progress"].lower()
