"""Regression tests for the two UI-freeze bugs.

Engine._loop holds self.lock for the whole tick, and snapshot() takes the same
lock from pywebview's bridge thread every 400ms. So anything slow reached while
holding that lock freezes the entire UI -- countdown stuck, Stop unresponsive,
and no violation or grace expiry processed for the duration.

Both of these were measured at multiple seconds before being fixed:
  - the calendar send:  2951ms  (retry ladder is ~60s worst case)
  - the camera open:    1949ms  (request_open waits up to 5s)
"""
from __future__ import annotations

import threading
import time
import traceback

import main
from timer import SessionTimer, State

# Anything above this and a 400ms UI poll visibly stutters.
MAX_ACCEPTABLE_BLOCK_S = 0.5
SLOW_OP_S = 1.0


def _bare_engine(cfg):
    """An Engine with its own state wired up but no camera, models, or window.
    __init__ builds a VisionWorker and loads MediaPipe, which a unit test has
    no business doing."""
    e = main.Engine.__new__(main.Engine)
    e.cfg = cfg
    e.lock = threading.RLock()
    e._api_key_lock = threading.Lock()
    e._send_lock = threading.Lock()
    e._api_key_cache = "test-key"
    e._camera_open_hint = None
    e.latest_reading = None
    e.status_message = ""
    e.auth_warning = ""
    e.thumbnail_b64 = None
    e._syncing = False
    e._stopping = False
    e.session_active = False
    # Must mirror Engine.__init__: snapshot() reads self.focus on every poll,
    # and a missing attribute here surfaces only as the poller thread dying.
    e.focus = None
    e._focus_handoff = None
    e._next_drain_mono = time.monotonic() + main.DRAIN_INTERVAL_S
    e._draining = False
    e._focus_reopen_next = 0.0
    return e


class _FakeWorker:
    degraded: list = []
    resolved_camera_name = "test cam"
    calibrated = True

    def __init__(self, open_delay=0.0):
        self.open_delay = open_delay

    def request_open(self, timeout: float = 5.0) -> bool:
        time.sleep(self.open_delay)
        return True

    def request_close(self): pass
    def start_calibration(self): pass
    def finish_calibration(self): pass


def _poll_while(engine, done, out):
    """Stands in for web/app.js calling get_state() on a loop.

    Everything is recorded through `out` because this runs on its own thread,
    where an exception is otherwise invisible: the thread dies, `out` stays
    empty, and the assertion below fails as a bare KeyError naming neither the
    real error nor this function. Capturing it turns that into the traceback.
    """
    worst = 0.0
    polls = 0
    try:
        while not done.is_set() or polls < 3:
            t0 = time.monotonic()
            engine.snapshot()
            worst = max(worst, time.monotonic() - t0)
            polls += 1
            time.sleep(0.01)
            if polls > 5000:
                break
    except Exception:
        out["error"] = traceback.format_exc()
    out["worst"] = worst
    out["polls"] = polls


def _check_poller(out):
    assert "error" not in out, f"snapshot() raised on the polling thread:\n{out['error']}"


def test_slow_calendar_send_does_not_block_the_ui(cfg, isolated_state, monkeypatch):
    import ledger

    monkeypatch.setattr(ledger, "record_session", lambda record: None)
    cfg["dry_run"] = False

    class FakeSession:
        id = "pg-test"
        label = "probe"
        camera_enabled = True
        violations = 0
        duration_s = 60.0
        remaining_s = 0.0
        started_wall_iso = "2026-08-02T10:00:00-04:00"
        focus_elapsed_s = 60.0

    class FakeTimer:
        state = State.LOGGING
        session = FakeSession()
        grace_kind = None
        last_outcome = None
        def build_record(self, *a, **k): return {}
        def to_idle(self):
            self.state = State.IDLE
            self.session = None

    e = _bare_engine(cfg)
    e.timer = FakeTimer()
    e.worker = _FakeWorker()
    e._send_to_calendar = lambda date, task: time.sleep(SLOW_OP_S)

    done = threading.Event()
    out: dict = {}

    def tick():
        with e.lock:                      # exactly what _loop does
            e._handle_terminal_states()
        done.set()

    poller = threading.Thread(target=_poll_while, args=(e, done, out))
    ticker = threading.Thread(target=tick)
    poller.start()
    ticker.start()
    ticker.join()
    poller.join()

    _check_poller(out)
    assert out["worst"] < MAX_ACCEPTABLE_BLOCK_S, (
        f"snapshot() blocked for {out['worst']:.2f}s during a "
        f"{SLOW_OP_S}s calendar send"
    )
    assert e.timer.state == State.IDLE


def test_slow_camera_open_does_not_block_the_ui(cfg, isolated_state, monkeypatch):

    e = _bare_engine(cfg)
    e.worker = _FakeWorker(open_delay=SLOW_OP_S)
    e.alarm = type("A", (), {"stop": lambda s: None, "start": lambda s, p=None: None})()
    e.timer = SessionTimer(
        cfg,
        on_camera_open=e._open_camera,
        on_camera_close=e._close_camera,
        on_calibrate_start=e._start_calibration,
        on_calibrate_finish=e._finish_calibration,
        on_alarm_start=e._on_alarm_start,
        on_alarm_stop=e.alarm.stop,
        on_notify=lambda m: None,
    )

    done = threading.Event()
    out: dict = {}

    def press_start():
        e.start("probe", 25.0, False, True, False)
        done.set()

    poller = threading.Thread(target=_poll_while, args=(e, done, out))
    presser = threading.Thread(target=press_start)
    poller.start()
    presser.start()
    presser.join()
    poller.join()

    _check_poller(out)
    assert out["worst"] < MAX_ACCEPTABLE_BLOCK_S, (
        f"snapshot() blocked for {out['worst']:.2f}s during a "
        f"{SLOW_OP_S}s camera open"
    )
    assert e.timer.state == State.CALIBRATING


def test_logging_failure_does_not_wedge_the_state_machine(cfg, isolated_state):
    """A raise inside _handle_logging used to skip to_idle(), leaving the state
    at LOGGING. _handle_terminal_states then re-entered it every 500ms,
    re-appending to the outbox each time, with Start disabled forever."""
    cfg["dry_run"] = True
    cfg["timezone"] = "Not/AZone"          # what validation now prevents

    class FakeSession:
        id = "pg-wedge"
        label = "probe"
        camera_enabled = True
        violations = 0
        duration_s = 60.0
        remaining_s = 0.0
        started_wall_iso = "2026-08-02T10:00:00-04:00"
        focus_elapsed_s = 60.0

    class FakeTimer:
        state = State.LOGGING
        session = FakeSession()
        grace_kind = None
        last_outcome = None
        def build_record(self, *a, **k): return {}
        def to_idle(self):
            self.state = State.IDLE
            self.session = None

    e = _bare_engine(cfg)
    e.timer = FakeTimer()
    e.worker = _FakeWorker()
    e.session_active = True

    for _ in range(3):
        with e.lock:
            e._handle_terminal_states()

    assert e.timer.state == State.IDLE
    assert e.session_active is False
    assert "failed" in e.status_message.lower()


def test_boot_failure_is_visible_in_the_ui():
    """Before this, a failed Engine build left the window on 'Loading detection
    models…' with Start disabled forever, and the only trace was app.log."""
    api = main.Api()
    booting = api.get_state()
    assert booting["buttons"]["start_enabled"] is False

    api._set_boot_error("Startup failed: KeyError: 'alarm_sound_path'")
    failed = api.get_state()
    assert failed["headline"] == "failed to start"
    assert "alarm_sound_path" in failed["status_message"]
    assert failed["auth_warning"]


def test_successful_send_removes_the_outbox_entry(cfg, isolated_state):
    """Entries used to survive a successful send and be replayed on every
    later launch, relying on the server's dedupe to no-op them."""
    import outbox

    e = _bare_engine(cfg)
    e._send_to_calendar = lambda date, task: None
    task = {"id": "pg-1", "text": "done", "done": True}
    outbox.add({"date": "2026-08-02", "task": task})

    e._send_async("pg-1", "2026-08-02", task)
    for _ in range(200):
        if not outbox.read_all():
            break
        time.sleep(0.01)
    assert outbox.read_all() == []


def test_permanently_rejected_send_also_clears_the_entry(cfg, isolated_state):
    import calendar_client
    import outbox

    e = _bare_engine(cfg)

    def reject(date, task):
        raise calendar_client.CalendarClientError("401 unauthorized")

    e._send_to_calendar = reject
    task = {"id": "pg-2", "text": "done", "done": True}
    outbox.add({"date": "2026-08-02", "task": task})

    e._send_async("pg-2", "2026-08-02", task)
    for _ in range(200):
        if not outbox.read_all():
            break
        time.sleep(0.01)
    assert outbox.read_all() == []
    assert "NOT logged" in e.status_message


def test_transient_send_failure_keeps_the_entry_queued(cfg, isolated_state):
    import outbox

    e = _bare_engine(cfg)

    def flaky(date, task):
        raise ConnectionError("network down")

    e._send_to_calendar = flaky
    task = {"id": "pg-3", "text": "done", "done": True}
    outbox.add({"date": "2026-08-02", "task": task})

    e._send_async("pg-3", "2026-08-02", task)
    for _ in range(200):
        if "Queued" in e.status_message:
            break
        time.sleep(0.01)
    assert [x["task"]["id"] for x in outbox.read_all()] == ["pg-3"]
