# Pomodoro Guard — Implementation Spec v2 (reviewed & revised)

Review of `pomodoro_guard_spec.md`. Changes are integrated below; the audit is first.

**Verdict on v1:** the architecture is sound but it is not safe to run 24/7 as written.
Three issues are severe enough to block coding: the calendar client can wipe your entire
calendar on a single bad GET (I-01), the timer uses wall-clock arithmetic that a laptop
sleep turns into a fabricated session (I-05), and the launchd plist points at a Python
that does not have the dependencies installed (I-16). Everything else is fixable in place.

---

## Issues Found

Severity: **S1** = data loss / security / silently wrong results. **S2** = will break in
normal use. **S3** = design concern or missing safety net.

| # | Sev | Issue | Why it matters | Fix |
|---|-----|-------|----------------|-----|
| I-01 | S1 | `calendar_client.log_pomodoro` does GET → mutate → POST with **no validation of the GET result**. If the API returns `{}`, `null`, an HTML error page that parses, or a partial blob from a cold Upstash read, the subsequent POST overwrites the **whole calendar** with a one-day, one-task object. | Total, unrecoverable loss of every calendar entry you've ever made. This is the single worst failure mode in the document and it needs one bad response to happen. | §6: shape validation, shrink-guard against a local snapshot, mandatory pre-POST local backup, and refuse-to-write on any anomaly. Preferably eliminate read-modify-write entirely (I-02). |
| I-02 | S1 | Full-blob read-modify-write races with the calendar website. Any edit made in the browser between the GET and the POST is silently discarded. | You will lose browser-side edits, intermittently, with no error and no way to notice. 24/7 operation makes "small window" irrelevant. | §6.1: add a server-side atomic `POST /api/data/append` to the Next.js app (Lua `EVAL` on Upstash). Client-side merging is a band-aid; this removes the race. |
| I-03 | S1 | `/api/data` has **no auth today**, and accepts a full-blob overwrite from anyone with the URL. | This is a live vulnerability that exists right now, independent of this project. The URL is a capability to read *and destroy* your calendar. v1 calls this "low effort, worth doing" — it should be the first thing fixed. | §6.2: `x-api-key` shared secret + payload shape validation + method allowlist on the Next.js route, before any Python is written. |
| I-04 | S1 | `config.json` holds a secret but no `.gitignore` is actually specified — v1 only says "(gitignored)" in a comment. | One `git add -A` publishes your API key and calendar URL to a public repo. | §3.1: explicit `.gitignore`, `chmod 600`, `git check-ignore` verification step, and optional macOS Keychain storage instead of plaintext. |
| I-05 | S1 | Timer accounting is undefined and implied wall-clock. `start_time = now()` + "countdown resumes from where it paused" cannot both be true. Worse, `time.time()` jumps on NTP sync, DST, and **laptop sleep**. | Close the lid at minute 3, open it at minute 40 → the elapsed check says the pomodoro finished, and a 25-minute focus session you did not do gets written to your calendar. Silently wrong data is worse than no data. | §5.1: explicit `focus_elapsed_s` accumulator driven by `time.monotonic()` deltas, plus a `max_tick_gap_seconds` guard that fails the session if a tick gap indicates sleep or a stalled process. |
| I-06 | S1 | Sustained-duration counters are computed as `counter * check_interval_ms`. Actual tick time is whatever two ML models take, which on a busy machine exceeds 500 ms. | Thresholds silently stretch — a "4 second" look-down check might really be 7 seconds. Detection becomes untrustworthy in exactly the conditions where it matters. | §4.5: accumulate real `dt` between ticks, never nominal interval × count. |
| I-07 | S2 | Camera read failure is not handled. `cap.read()` returns `(False, None)` when Zoom/FaceTime grabs the camera, on USB disconnect, or after sleep. v1's loop would either crash or treat it as "no face". | A camera error becomes a false `"left"` violation and discards a real session — or crashes the background thread with no log, leaving a GUI that counts down forever guarding nothing. | §4.1: distinguish `CAMERA_ERROR` from `NO_FACE`; pause the session (not fail it) and surface a visible banner; attempt reopen with backoff. |
| I-08 | S2 | Absolute 25° pitch threshold for "looking down" with no calibration. A laptop webcam sits below eye level and tilts up, so a neutral seated head pose already reads as pitched; the offset varies by desk, chair, and monitor height. | Either constant false violations or a threshold you can never trip. The feature is unusable without calibration. | §4.3: 5-second calibration at session start, threshold on **delta from baseline**, plus EMA smoothing and hysteresis (enter 20°, release 12°). |
| I-09 | S2 | "Looking down" conflates distraction with legitimate work — reading a book, writing notes, using a sketchpad. | The guard punishes deep work. | §4.3: `detect_look_down` defaults **off**; enable per-session via a checkbox when the work is screen-only. |
| I-10 | S2 | Phone detection: **instant fail on a single frame ≥ 0.5 confidence**, with no sustain requirement. EfficientDet-Lite's COCO `cell phone` class is weak at desk distance and misfires on dark rectangles — a wallet, a remote, a closed laptop edge, a hand at the wrong angle. | One spurious frame destroys a 24-minute session with no appeal. High-cost false positives are the fastest way to abandon a tool like this. | §4.4: raise to 0.6, require sustained detection over `phone_sustain_seconds`, log the confidence value, and run the detector every 2nd tick to save CPU. |
| I-11 | S2 | Counters "reset to 0 the moment face returns to normal". | A single good frame wipes a nearly-triggered counter. Glancing down for 3.5s, up for 0.5s, repeating forever, never trips a 4s threshold — the exact evasion pattern a phone user produces. | §4.5: leaky-bucket accumulator that decays at half rate rather than hard-resetting. |
| I-12 | S2 | No cap on violation/grace cycles. Violate → return at 7.9s → repeat, indefinitely. | A "25-minute pomodoro" can span three wall-clock hours of mostly-not-focusing and still get logged as a clean 25 minutes. | §5.2: `max_violations_per_session` (3) and `max_total_paused_seconds` (90); exceeding either → `FAILED`. |
| I-13 | S2 | `SUCCESS` transitions straight to `IDLE` while the calendar POST runs on a background thread. | App quits or crashes during the POST and the completed session is gone — v1 explicitly says "never silently lose a completed pomodoro", then designs a path that does exactly that. | §5.3: new `LOGGING` state; write the session to a durable local outbox **before** the first network attempt, delete only on confirmed success, drain the outbox on startup. |
| I-14 | S2 | No crash / restart handling. No persistence of in-flight session state. | Crash at minute 20, relaunch, and the app has no idea a session was running. Combined with launchd `RunAtLoad` you get silent, invisible loss. | §5.4: checkpoint session state to disk each tick; on startup, explicitly discard any interrupted session and notify — never silently resume. |
| I-15 | S2 | No single-instance guard. launchd auto-start + a manual run = two processes competing for the camera and both writing to the calendar. | Duplicate entries, camera contention, confusing behaviour. | §2.1: `fcntl.flock` lockfile, exit with a clear message if already held. |
| I-16 | S2 | launchd plist runs `/usr/bin/python3`. That is the Xcode Command Line Tools stub — it does not have opencv, mediapipe, requests, or (usually) a working Tk. | The agent fails at import on every login and logs to a file you will not read. | §9: point at the venv interpreter absolute path; add `WorkingDirectory` (relative `config.json` won't resolve otherwise), `KeepAlive`/`SuccessfulExit=false`, `ThrottleInterval`. |
| I-17 | S2 | `Pillow` is missing from `requirements.txt` but §8 requires a live camera thumbnail in Tkinter, which needs PIL to convert an OpenCV BGR array into a `PhotoImage`. | Immediate `ImportError` on first run. | §10: add `Pillow`; pin all versions. |
| I-18 | S2 | Shared state between the camera thread and the Tkinter loop is described as "a shared state object" with no synchronization. | Torn reads, stale status text, and the classic Tk crash from touching widgets off the main thread. | §8: single `queue.Queue` handoff, immutable `FrameReading` dataclass, all widget writes in `root.after`. |
| I-19 | S3 | `afplay` via `subprocess.run` blocks the calling thread. | If it's on the detection thread, detection stalls for the length of the sound — during a grace period, which is the one moment detection must be responsive. | §7: `subprocess.Popen` with a stored handle and `terminate()` on resolve. |
| I-20 | S3 | The alarm is the only feedback for a violation, but the primary violation is *you left the room*. | You cannot hear a desk speaker from the kitchen. The grace period's feedback mechanism is broken in precisely the case it was designed for. | §7: add a macOS notification and consider whether `left` should be instant-fail with no grace at all (see I-21). |
| I-21 | S3 | 8-second grace is too short to walk back from anywhere, and too generous for a glance. | It only ever forgives micro-violations, which means the design intent ("come back quickly") is not actually implementable. | §5.2: 15s default; and split policy — `phone`/`looking_down` get grace, `left` optionally instant-fails (`grace_on_left: false`). |
| I-22 | S3 | POST retry is "retry once" with no backoff and **no idempotency**. A POST that commits server-side but times out client-side gets retried. | Duplicate calendar entries. | §6.3: deterministic session ID derived from start time; on retry, GET and check for the ID before re-posting; exponential backoff with jitter. |
| I-23 | S3 | `start_time`/`end_time` typed `str` with no format specified, and `time.strftime("%Y-%m-%d")` uses client local time. A session spanning midnight has undefined date attribution. | Entries the web app can't render, or filed under the wrong day. | §6.4: pin the format against a real entry from the live calendar first; key by **session start** date; store the IANA timezone explicitly. |
| I-24 | S3 | No validation of `label`, and no validation of `pomodoro_minutes`. | `pomodoro_minutes: 0` gives an instant-success loop that spams the calendar. An unescaped label rendered by the web app is self-XSS on your own calendar. | §5.5: strip control characters, cap length at 120, reject empty; clamp duration to 5–180 minutes. |
| I-25 | S3 | No graceful degradation if the phone-detection model fails to load or download. MediaPipe `.task` model files are fetched at first run — an undocumented network dependency. | Worst case: the app runs, appears to be guarding, and silently isn't. A guard you falsely trust is worse than no guard. | §4.4: vendor the model file with a checksum; on load failure run face-only with a persistent red "PHONE DETECTION OFF" banner, and record the degraded state in the session log. |
| I-26 | S3 | No observability. No structured logs, no per-session record, no way to tune thresholds after a false positive. | You will not be able to answer "why did it fail me at 19:03?" | §11: JSONL session ledger with all measurements (never frames), rotating logs, and a "that was wrong" button that captures the numeric state at violation time. |
| I-27 | S3 | No manual override. | Someone knocks on the door, a call comes in — there is no honest way to pause. The only options are lose the session or game the detector, and one of those teaches you to game the detector. | §5.6: explicit `INTERRUPTED` pause that stops the clock, is recorded in the ledger, and is capped in count/duration so it can't be abused. |
| I-28 | S3 | §4.1 implies the camera opens at app launch, and §9 auto-starts the app at login. | The camera is held (and the green light may be on) all day for a tool you use ~2 hours of. Battery, thermals, and a standing privacy surface for no benefit. | §4.1: open the camera on `Start`, `release()` on every exit path including window close. |
| I-29 | S3 | `camera_index: 0` on macOS. Continuity Camera means index 0 is frequently your iPhone, not the built-in webcam, and the mapping changes when the phone is nearby. | Silently monitors the wrong camera, or fails at random depending on where your phone is. | §3: probe by device name with `camera_name_hint`, fall back to index, and show the resolved device name in the GUI. |
| I-30 | S3 | `no_face_threshold_seconds: 2` = 4 consecutive missed frames. | Reaching for water, turning to a second monitor, or rubbing your eyes trips it. False violations at this rate make the tool unusable within a day. | §3: raise to 5s; tune down later using the ledger data from I-26. |

---

## 1. Tech stack

- **Language:** Python **3.11 or 3.12** in a project-local venv. Not 3.13 — `mediapipe`
  wheel availability lags. Not system Python.
- **Camera / vision:** `opencv-python` for capture, MediaPipe Face Landmarker for
  presence + head pose, MediaPipe Object Detector (EfficientDet-Lite) for phone.
- **GUI:** Tkinter. **Confirmed default: floating always-on-top window** with camera
  thumbnail. Rationale over the alternatives: terminal-only gives you no way to verify
  detection is actually working, which is the failure mode you most need visibility into
  (I-25); `rumps` menu-bar looks nicer but hides the live preview behind a click, and
  adds a dependency. Note that Tk on macOS requires `python.org` Python or
  `brew install python-tk` — the CLT Python's Tk is 8.5 and broken.
- **HTTP:** `requests`.
- **Config:** `config.json`, gitignored, mode 600, with the secret optionally in Keychain.
- **Background execution:** user-scoped launchd LaunchAgent — **but see §9**; auto-start
  is now discouraged in favour of manual launch.

---

## 2. Project structure

```
pomodoro_guard/
├── .gitignore                 # NEW — see §3.1
├── config.json                # gitignored, chmod 600
├── config.example.json
├── requirements.txt
├── main.py                    # entry point, GUI + glue
├── vision.py                  # capture + detection
├── timer.py                   # state machine
├── calendar_client.py         # calendar I/O + outbox
├── outbox.py                  # NEW — durable pending-write queue
├── ledger.py                  # NEW — JSONL session record
├── alarm.py
├── models/
│   └── efficientdet_lite0.tflite   # NEW — vendored, checksummed
├── launchd/
│   └── com.mohammad.pomodoroguard.plist
├── state/                     # gitignored — checkpoint, outbox, backups, lock
└── logs/                      # gitignored
```

### 2.1 Single-instance guard *(fixes I-15)*

```python
# main.py, before anything else
import fcntl, sys
from pathlib import Path

STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)
_lock_fh = open(STATE / "instance.lock", "w")
try:
    fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit("Pomodoro Guard is already running. Refusing to start a second instance.")
_lock_fh.write(str(os.getpid()))
_lock_fh.flush()
```

Keep `_lock_fh` alive for the process lifetime — closing it releases the lock.

---

## 3. Config file (`config.json`)

```json
{
  "calendar_api_url": "https://YOUR-CALENDAR.vercel.app/api/data",
  "calendar_append_url": "https://YOUR-CALENDAR.vercel.app/api/data/append",
  "calendar_api_key_source": "keychain",
  "calendar_api_key": null,
  "timezone": "America/Toronto",
  "dry_run": false,

  "pomodoro_minutes": 25,
  "min_session_minutes": 5,
  "max_session_minutes": 180,

  "grace_period_seconds": 15,
  "grace_on_left": true,
  "max_violations_per_session": 3,
  "max_total_paused_seconds": 90,
  "max_manual_pauses": 1,
  "max_manual_pause_seconds": 120,

  "no_face_threshold_seconds": 5,

  "detect_look_down": false,
  "look_down_threshold_seconds": 8,
  "look_down_enter_delta_degrees": 20,
  "look_down_release_delta_degrees": 12,
  "calibration_seconds": 5,

  "detect_phone": true,
  "phone_confidence_threshold": 0.6,
  "phone_sustain_seconds": 1.0,
  "phone_detect_every_n_ticks": 2,

  "camera_name_hint": "FaceTime HD Camera",
  "camera_index": 0,
  "check_interval_ms": 500,
  "max_tick_gap_seconds": 5.0,
  "camera_reopen_backoff_seconds": [1, 2, 5, 10],

  "alarm_sound_path": "/System/Library/Sounds/Sosumi.aiff",
  "notify_on_failure": true,
  "log_level": "INFO"
}
```

**Threshold changes and why:** `no_face` 2 → 5s (I-30); `grace` 8 → 15s (I-21);
`look_down` 4 → 8s and now a *relative* angle (I-08); `phone_confidence` 0.5 → 0.6 with a
1s sustain (I-10); `detect_look_down` defaults **off** (I-09).

`pomodoro_minutes: 25` is confirmed as the default. No break timer in v1 — out of scope,
and a break timer is the part you can most easily replace with a phone alarm.

Validate on load: URL scheme must be `https`, `min_session_minutes <= pomodoro_minutes
<= max_session_minutes`, all durations `> 0`, `check_interval_ms` in `[100, 2000]`. Exit
with a readable error rather than starting in a broken state.

### 3.1 `.gitignore` *(fixes I-04)*

```gitignore
config.json
state/
logs/
.venv/
__pycache__/
*.pyc
.DS_Store
```

Verify after creating it, before the first commit:

```bash
chmod 600 config.json
git check-ignore -v config.json state/ logs/    # must print a match for each
```

**Keychain option (preferred).** With `"calendar_api_key_source": "keychain"` the key
never touches disk in plaintext:

```bash
security add-generic-password -a "$USER" -s pomodoro-guard-calendar -w 'YOUR_KEY'
```

```python
def load_api_key(cfg) -> str:
    if cfg["calendar_api_key_source"] == "keychain":
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "pomodoro-guard-calendar", "-w"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    return cfg["calendar_api_key"]
```

Never log the key, never put it in a query string (it would land in Vercel access logs) —
header only. Scrub headers from any exception logging.

---

## 4. `vision.py` — detection logic

### 4.1 Capture lifecycle *(fixes I-07, I-28, I-29)*

The camera is opened when a session **starts** and released when it ends, on window
close, and in a `finally` on every exit path. It is not held while idle.

Resolve the device by name where possible so Continuity Camera doesn't hijack index 0,
and display the resolved device name in the GUI so a wrong-camera situation is visible:

```python
def open_camera(cfg) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise CameraUnavailable(f"index {cfg['camera_index']} would not open")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap
```

OpenCV cannot enumerate device names on macOS; if `camera_name_hint` is set, resolve the
index once at first run using `system_profiler SPCameraDataType` and persist the result to
`state/camera.json`, prompting the user to confirm via the thumbnail.

Read failures are **not** treated as absence:

```python
ok, frame = cap.read()
if not ok or frame is None:
    return FrameReading(ok=False, error="camera_read_failed")
```

The state machine handles `ok=False` by entering `CAMERA_ERROR`, which **pauses** the
session (clock stops, no violation) and shows a banner. Reopen is attempted on the
backoff schedule in config; if all attempts fail, the session goes to `ABORTED` — which,
unlike `FAILED`, is explicitly reported as "not your fault, nothing was logged."

Frames are never written to disk, never logged, and never included in exception context.
There is no debug frame-dump mode; if one is ever added it must default off and write to
a `0700` directory.

### 4.2 Frame reading structure *(fixes I-26)*

v1's `check_frame` returned three booleans, discarding everything needed for smoothing,
hysteresis, tuning, and post-hoc debugging.

```python
@dataclass(frozen=True)
class FrameReading:
    ok: bool
    error: str | None = None
    face_present: bool = False
    pitch_delta_deg: float | None = None     # smoothed, relative to calibration baseline
    phone_confidence: float = 0.0
    phone_detector_available: bool = True
    inference_ms: float = 0.0
```

### 4.3 Presence + head pose *(fixes I-06, I-08, I-09)*

Face Landmarker must be constructed with
`output_facial_transformation_matrixes=True`. Pitch comes from the rotation block:

```python
def pitch_deg(matrix) -> float:
    R = np.asarray(matrix)[:3, :3]
    return float(np.degrees(np.arctan2(-R[2, 1], R[2, 2])))
```

**The sign convention is not guaranteed across MediaPipe versions.** The calibration
wizard must verify it empirically: prompt "look down at your desk", confirm the value
moves in the expected direction, and store the sign in `state/calibration.json`. If it
doesn't move, abort setup with an error rather than shipping an inverted detector.

**Calibration (5s at session start).** Sample pitch while the user sits normally, discard
the first second, take the median as `baseline_pitch`. Every subsequent reading is
`pitch_delta = (pitch - baseline) * sign`. This is what makes the feature work across
different desks and laptop lid angles — an absolute 25° threshold does not.

**Smoothing + hysteresis.** Raw per-frame pitch is noisy. Apply an EMA (`alpha=0.3`), then:

- enter "looking down" when `pitch_delta > look_down_enter_delta_degrees` (20°)
- release only when `pitch_delta < look_down_release_delta_degrees` (12°)

The gap prevents flapping at the boundary.

**Default off.** Reading a book, taking notes on paper, and sketching all look identical
to doom-scrolling from the camera's point of view. Enable it per-session with a GUI
checkbox when the work really is screen-only.

### 4.4 Phone detection *(fixes I-10, I-25)*

MediaPipe Object Detector (EfficientDet-Lite0) confirmed over YOLOv8n — keeps the
dependency tree to two ML packages instead of pulling in torch.

**Be realistic about what this buys you.** The COCO `cell phone` class is trained mostly
on phones held up in photographs, not on a phone lying flat on a desk at 60cm from a
640×480 webcam. Expect mediocre recall, and note the dominant desk failure mode is a
phone held **in your lap, below the frame entirely** — which no camera-based detector at
this angle will ever see. This check catches the lazy version of the behaviour, not the
determined one. Set expectations accordingly rather than trusting it as a hard guarantee.

Changes from v1:
- Confidence threshold 0.6, and the raw value is recorded in the ledger for tuning.
- Requires sustained detection over `phone_sustain_seconds` (1.0s ≈ 2–4 ticks) rather
  than a single frame. The instant-fail rule in v1 makes a 24-minute session hostage to
  one false detection on a wallet.
- Runs every `phone_detect_every_n_ticks` (2) to halve its CPU cost.

**Model loading and degradation.** MediaPipe fetches `.task`/`.tflite` model assets on
first use. Vendor the file into `models/` and verify its SHA-256 at startup. On any load
failure:

```python
try:
    phone_detector = build_phone_detector(MODEL_PATH)
except Exception as e:
    log.error("phone detector unavailable: %s", e)
    phone_detector = None
```

The app continues in face-only mode, shows a **persistent red banner "PHONE DETECTION
OFF"**, and stamps `"degraded": ["phone"]` on every session in the ledger. It must never
run silently degraded — a guard you falsely trust is worse than no guard.

### 4.5 Sustained-condition accumulator *(fixes I-06, I-11)*

Replaces both the `counter * interval` arithmetic and the hard reset.

```python
class SustainedCondition:
    """Leaky bucket. Fills at 1x while active, drains at decay_rate while clear."""
    def __init__(self, trigger_s: float, decay_rate: float = 0.5):
        self.trigger_s, self.decay_rate, self.acc = trigger_s, decay_rate, 0.0

    def update(self, active: bool, dt: float) -> bool:
        if active:
            self.acc = min(self.trigger_s, self.acc + dt)
        else:
            self.acc = max(0.0, self.acc - dt * self.decay_rate)
        return self.acc >= self.trigger_s

    def reset(self):
        self.acc = 0.0
```

`dt` is the **measured** interval between ticks (`time.monotonic()` delta), never the
nominal `check_interval_ms`. Draining at half rate means repeated brief glances still
accumulate — v1's hard reset made a 3.5s-down / 0.5s-up cycle invisible forever.

---

## 5. `timer.py` — state machine

### 5.1 Clock accounting *(fixes I-05)*

All durations come from `time.monotonic()`, which is immune to NTP, DST, and manual clock
changes. `time.time()` is used **only** to produce the ISO timestamp written to the
calendar.

```python
@dataclass
class Session:
    label: str
    duration_s: float
    started_wall_iso: str           # for the calendar entry
    started_mono: float
    focus_elapsed_s: float = 0.0    # the only thing that counts toward completion
    paused_total_s: float = 0.0
    violations: int = 0
    manual_pauses: int = 0
    degraded: list[str] = field(default_factory=list)
    look_down_enabled: bool = False
```

Each tick:

```python
now = time.monotonic()
dt = now - self.last_tick
self.last_tick = now

if dt > cfg["max_tick_gap_seconds"]:
    # laptop slept, process was SIGSTOPped, or the machine thrashed
    return self.abort("clock_gap", detail=f"{dt:.1f}s tick gap")

if self.state is RUNNING:
    self.session.focus_elapsed_s += dt
```

`focus_elapsed_s` only advances in `RUNNING`. This is what makes "pause, don't reset"
actually true — v1 asserted it without a mechanism.

The `clock_gap` guard is what stops a closed-lid laptop from manufacturing a completed
session out of nothing. Aborted sessions are reported and never logged to the calendar.

### 5.2 States and transitions *(fixes I-12, I-21)*

`RESOLVED` is removed — v1 listed it as a state but described it as a transition back to
`RUNNING`. New states: `CALIBRATING`, `CAMERA_ERROR`, `INTERRUPTED`, `LOGGING`, `ABORTED`.

```
IDLE
  → user enters label (validated, §5.5), picks look-down on/off, clicks Start
  → open camera → CALIBRATING

CALIBRATING  (calibration_seconds)
  → capture baseline pitch; if no face for the whole window → back to IDLE with an error
  → RUNNING

RUNNING
  → each tick: dt guard (§5.1), then vision.check_frame()
  → reading.ok is False               → CAMERA_ERROR
  → sustained violation triggers      → violations += 1
        if violations > max_violations_per_session → FAILED
        elif kind == "left" and not grace_on_left  → FAILED
        else → start alarm, VIOLATION_GRACE
  → user clicks Pause                 → INTERRUPTED  (§5.6)
  → focus_elapsed_s >= duration_s     → SUCCESS
  Evaluation order is fixed: clock gap → camera error → violations → completion.
  A violation on the same tick as completion loses; the session fails.

VIOLATION_GRACE
  → clock is stopped; paused_total_s accumulates
  → resolution requires face_present AND NOT looking_down AND NOT phone_detected
    held clean for 1.0 continuous seconds (a single lucky frame must not resolve it)
  → resolved  → stop alarm, reset that condition's accumulator, RUNNING
  → grace_period_seconds elapsed without resolution → FAILED
  → paused_total_s > max_total_paused_seconds → FAILED

INTERRUPTED   (manual, honest pause)
  → clock stopped, no alarm, recorded in the ledger
  → Resume → CALIBRATING (re-baseline; you may have moved) → RUNNING
  → exceeds max_manual_pause_seconds → FAILED

CAMERA_ERROR
  → clock stopped, banner shown, reopen attempted on backoff
  → recovered → CALIBRATING → RUNNING
  → backoff exhausted → ABORTED

SUCCESS
  → write session to ledger + outbox (durable, on disk) → LOGGING

LOGGING
  → background thread drains the outbox (§6.3)
  → GUI shows "Logged ✅" or "Queued — will retry" — never blocks
  → IDLE

FAILED
  → no calendar write; ledger records the reason and full measurement context
  → macOS notification if notify_on_failure
  → IDLE

ABORTED
  → no calendar write; reported explicitly as a system fault, not a user failure
  → IDLE
```

### 5.3 Durable success path *(fixes I-13)*

v1 promised never to lose a completed pomodoro and then transitioned to `IDLE` while the
network call was still in flight on a background thread. The fix is to make disk, not
memory, the commit point:

1. On `SUCCESS`, append the session to `state/outbox.jsonl` and `fsync`.
2. Only then attempt the network write.
3. On confirmed success, rewrite the outbox without that entry.
4. On startup, drain any leftover outbox entries before enabling the Start button.

A completed session survives a crash, a quit, an offline laptop, and a Vercel outage.

### 5.4 Crash recovery *(fixes I-14)*

Every tick, write `state/current_session.json` (atomically: temp file + `os.replace`). On
startup, if that file exists:

- Do **not** resume. Time you can't account for is time you can't certify.
- Move it to the ledger as `outcome: "interrupted"`, notify the user, delete the file.

### 5.5 Input validation *(fixes I-24)*

```python
def clean_label(raw: str) -> str:
    s = "".join(ch for ch in raw if ch.isprintable()).strip()
    if not s:
        raise ValueError("Label is required.")
    return s[:120]
```

Duration is clamped to `[min_session_minutes, max_session_minutes]` at config load, so
`pomodoro_minutes: 0` can't produce an instant-success loop.

Note the label ends up rendered by the Next.js calendar. Confirm that app escapes it;
if it uses `dangerouslySetInnerHTML` anywhere, fix that there — this is your own data,
but it's still a stored-XSS sink.

### 5.6 Manual override *(fixes I-27)*

A "Pause (interruption)" button, capped at `max_manual_pauses: 1` and
`max_manual_pause_seconds: 120`, that stops the clock without penalty and is recorded in
the ledger. Without this, a doorbell means either losing the session or learning to game
the detector — and the second one is a habit that destroys the tool's value permanently.

---

## 6. Calendar integration

### 6.1 Preferred: server-side atomic append *(fixes I-02)*

Read-modify-write over a full blob is the wrong shape for this. Add an append endpoint
to the Next.js app so the merge happens atomically in Redis and the Python client never
holds the whole calendar.

```ts
// app/api/data/append/route.ts   — adapt key + schema to your actual storage
import { NextRequest, NextResponse } from "next/server";
import { Redis } from "@upstash/redis";

const redis = Redis.fromEnv();
const KEY = "calendar:data";

const LUA = `
local raw = redis.call('GET', KEYS[1])
local data = raw and cjson.decode(raw) or {}
local day = ARGV[1]
data[day] = data[day] or {}
for _, t in ipairs(data[day]) do
  if t.id == ARGV[3] then return 'duplicate' end   -- idempotency
end
table.insert(data[day], cjson.decode(ARGV[2]))
redis.call('SET', KEYS[1], cjson.encode(data))
return 'ok'
`;

export async function POST(req: NextRequest) {
  if (req.headers.get("x-api-key") !== process.env.CALENDAR_API_KEY) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const { date, task } = await req.json();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date ?? "") || typeof task?.id !== "string") {
    return NextResponse.json({ error: "bad request" }, { status: 400 });
  }
  const result = await redis.eval(LUA, [KEY], [date, JSON.stringify(task), task.id]);
  return NextResponse.json({ result });
}
```

This removes the lost-update race, removes the whole-calendar-overwrite blast radius,
and gives you idempotency for free. **Assumption to confirm:** that your calendar is a
single JSON blob under one Redis key. If it's keyed per-day, the Lua gets simpler still.

### 6.2 Auth on `/api/data` *(fixes I-03)*

Do this before any Python is written. Right now anyone who learns your Vercel URL can
read your entire calendar and replace it with `{}`. On the existing route:

- Require `x-api-key` on **all** methods (GET included — the calendar is personal data).
- Reject any POST body that isn't an object of `YYYY-MM-DD → array` (I-01's server-side
  counterpart).
- Reject payloads that would reduce the stored day count by more than one.
- Keep the secret in a Vercel environment variable, not in the repo.

**Setup steps:**

1. Generate a secret: `openssl rand -hex 32`.
2. Vercel → Project → Settings → Environment Variables → add `CALENDAR_API_KEY` =
   the generated string → redeploy.
3. Guard every handler in the route:

```ts
export async function GET(req: NextRequest) {
  if (req.headers.get("x-api-key") !== process.env.CALENDAR_API_KEY) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  // ...existing logic
}
```
Same check in `POST` — an unauthenticated GET still leaks the whole calendar.

4. Store the same secret in macOS Keychain (not in `config.json` plaintext, per §3.1):
```bash
security add-generic-password -a "$USER" -s pomodoro-guard-calendar -w 'YOUR_SECRET'
```
`calendar_client.py` reads it via `load_api_key()` and sends it as `x-api-key` on every
request — never in the URL (it would land in Vercel access logs), never logged.

Note this key is a **machine-to-machine credential** for the Python script, separate
from the browser login gate in §6.5 below — the two protect different things and don't
share a secret.

### 6.5 Browser login gate on the calendar site *(new — protects the web UI itself)*

§6.2 authenticates the Python script's API calls. It does not protect the calendar
**website** itself — anyone with the URL can still open it in a browser and view/edit
the calendar with no login at all. Add a password gate with a long-lived "remember this
device" cookie so it's asked once, not per-tab or per-visit.

**1. Secrets** — Vercel env vars:
```
SITE_PASSWORD=your-chosen-password
COOKIE_SECRET=<openssl rand -hex 32>
```

**2. `middleware.ts`** (project root) — gates every page except the login route:

```ts
import { NextRequest, NextResponse } from "next/server";
import { verifyAuthCookie } from "./lib/auth";

export function middleware(req: NextRequest) {
  if (req.nextUrl.pathname.startsWith("/login") || req.nextUrl.pathname.startsWith("/api/login")) {
    return NextResponse.next();
  }
  const cookie = req.cookies.get("site_auth")?.value;
  if (!cookie || !verifyAuthCookie(cookie)) {
    const url = req.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("next", req.nextUrl.pathname);
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = { matcher: ["/((?!_next|favicon.ico).*)"] };
```

**3. `lib/auth.ts`** — signed token so the cookie can't be forged by just setting it to
`"true"`:

```ts
import crypto from "crypto";

const SECRET = process.env.COOKIE_SECRET!;

export function makeAuthCookie(): string {
  const payload = `ok.${Date.now()}`;
  const sig = crypto.createHmac("sha256", SECRET).update(payload).digest("hex");
  return `${payload}.${sig}`;
}

export function verifyAuthCookie(token: string): boolean {
  const [tag, ts, sig] = token.split(".");
  if (tag !== "ok") return false;
  const expected = crypto.createHmac("sha256", SECRET).update(`${tag}.${ts}`).digest("hex");
  return crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(expected));
}
```

**4. `app/api/login/route.ts`:**

```ts
import { NextRequest, NextResponse } from "next/server";
import { makeAuthCookie } from "@/lib/auth";

export async function POST(req: NextRequest) {
  const { password } = await req.json();
  if (password !== process.env.SITE_PASSWORD) {
    return NextResponse.json({ error: "wrong password" }, { status: 401 });
  }
  const res = NextResponse.json({ ok: true });
  res.cookies.set("site_auth", makeAuthCookie(), {
    httpOnly: true,
    secure: true,
    sameSite: "lax",
    maxAge: 60 * 60 * 24 * 365, // 1 year — the "remember my device" duration
    path: "/",
  });
  return res;
}
```

**5. `app/login/page.tsx`:**

```tsx
"use client";
import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const router = useRouter();
  const next = useSearchParams().get("next") ?? "/";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ password }),
    });
    if (res.ok) router.push(next);
    else setError("Wrong password");
  }

  return (
    <form onSubmit={submit}>
      <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoFocus />
      <button type="submit">Enter</button>
      {error && <p>{error}</p>}
    </form>
  );
}
```

One password entry; the `HttpOnly` + `Secure` cookie is good for a year and only
re-prompts if cookies are cleared, the browser changes, or the year expires. This is
independent of the `x-api-key` check in §6.2 — the Python script keeps using its own key
and is unaffected by this cookie gate.


### 6.3 Fallback client (if the append endpoint isn't built) *(fixes I-01, I-22)*

If you keep read-modify-write, it needs every one of these guards. This is strictly worse
than §6.1 and should be temporary.

```python
# calendar_client.py
from __future__ import annotations
import json, logging, random, time
from pathlib import Path
import requests

log = logging.getLogger(__name__)
STATE = Path(__file__).parent / "state"
BACKUPS = STATE / "backups"
SNAPSHOT = STATE / "last_seen.json"
MAX_ATTEMPTS = 5


class CalendarUnsafe(Exception):
    """Fetched blob failed sanity checks — refusing to write."""


def _validate(blob) -> dict:
    if not isinstance(blob, dict):
        raise CalendarUnsafe(f"expected object, got {type(blob).__name__}")
    for k, v in blob.items():
        if not (isinstance(k, str) and len(k) == 10 and k[4] == "-" and k[7] == "-"):
            raise CalendarUnsafe(f"unexpected day key {k!r}")
        if not isinstance(v, list):
            raise CalendarUnsafe(f"day {k} is not a list")
    return blob


def _shrink_guard(blob: dict) -> None:
    """Refuse to write if the calendar looks like it lost data since we last saw it."""
    if not SNAPSHOT.exists():
        return
    prev = json.loads(SNAPSHOT.read_text())
    if len(blob) < prev["day_count"] - 1 or sum(map(len, blob.values())) < prev["task_count"] - 5:
        raise CalendarUnsafe(
            f"blob shrank: {len(blob)} days / {sum(map(len, blob.values()))} tasks "
            f"vs last seen {prev['day_count']} / {prev['task_count']}"
        )


def _backup(blob: dict) -> None:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    (BACKUPS / f"{int(time.time())}.json").write_text(json.dumps(blob))
    for old in sorted(BACKUPS.glob("*.json"))[:-50]:
        old.unlink()


def _snapshot(blob: dict) -> None:
    SNAPSHOT.write_text(json.dumps(
        {"day_count": len(blob), "task_count": sum(map(len, blob.values()))}))


def log_pomodoro(api_url: str, api_key: str, day: str, task: dict) -> None:
    headers = {"x-api-key": api_key, "content-type": "application/json"}

    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.get(api_url, headers=headers, timeout=10)
            r.raise_for_status()                      # v1 omitted this entirely
            blob = _validate(r.json())
            _shrink_guard(blob)                       # the I-01 stopper

            if any(t.get("id") == task["id"] for t in blob.get(day, [])):
                log.info("already logged %s — idempotent no-op", task["id"])
                return                                # the I-22 stopper

            _backup(blob)
            blob.setdefault(day, []).append(task)

            p = requests.post(api_url, json=blob, headers=headers, timeout=10)
            p.raise_for_status()
            _snapshot(blob)
            return

        except CalendarUnsafe:
            raise                                     # never retry a corrupt read
        except Exception as e:
            wait = min(60, 1.5 ** attempt) + random.uniform(0, 0.5)
            log.warning("calendar attempt %d/%d failed: %s (retry in %.1fs)",
                        attempt + 1, MAX_ATTEMPTS, e, wait)
            time.sleep(wait)

    raise RuntimeError("calendar write failed after retries — left in outbox")
```

Called from a worker thread draining `state/outbox.jsonl`, never from the Tk main loop.
`CalendarUnsafe` is deliberately non-retryable and surfaces as a loud red GUI banner —
if the calendar looks corrupt, the right move is to stop and look, not to keep writing.

### 6.4 Timestamp and date semantics *(fixes I-23)*

Unresolved until you check the live data. **Before coding**, `curl` the deployed
`/api/data`, take one real task object, and pin down:

- the exact `startTime` / `endTime` format (`"14:30"` vs ISO-8601 vs epoch ms)
- whether `id` has a required shape
- what fields the UI needs (`done`, `text`, anything else)

Until then, this spec assumes ISO-8601 with offset. Fixed decisions regardless of format:

- The entry is filed under the **session start** date, in `config.timezone` — a 23:50–00:15
  session lands on the start day.
- The session ID is deterministic, so a retry after a lost response can't duplicate:
  `f"pg-{int(start_epoch_ms)}"`.

---

## 7. `alarm.py` *(fixes I-19, I-20)*

```python
import subprocess

class Alarm:
    def __init__(self, path: str):
        self.path, self.proc = path, None

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        try:
            self.proc = subprocess.Popen(["afplay", self.path])   # non-blocking
        except OSError as e:
            log.error("alarm failed: %s", e)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
```

`subprocess.run` in v1 blocks for the length of the sound — on the detection thread, that
stalls detection during the grace period, which is the one moment it must be responsive.

Because the primary violation is *you left the room*, sound alone is inadequate — you
can't hear a desk speaker from the kitchen. Add a notification and window flash:

```python
subprocess.Popen(["osascript", "-e",
    'display notification "Return in 15s or the session is discarded" '
    'with title "Pomodoro Guard"'])
```

If the alarm reliably doesn't reach you, set `grace_on_left: false` and accept that
standing up ends the session — an honest instant-fail beats a grace period you can't use.

---

## 8. `main.py` — GUI *(fixes I-17, I-18)*

**Thread model.** One camera/inference worker thread, one Tk main thread. Handoff is a
`queue.Queue(maxsize=2)` carrying immutable `FrameReading` objects — never a mutable
shared object, and never a Tk call from the worker.

```python
def poll(self):
    try:
        while True:
            self.latest = self.frames.get_nowait()   # drain to newest
    except queue.Empty:
        pass
    self.render(self.latest)
    self.root.after(self.cfg["check_interval_ms"], self.poll)
```

**Thumbnail** requires Pillow (missing from v1's requirements):

```python
from PIL import Image, ImageTk
img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
self.preview.configure(image=img)
self.preview.image = img        # keep a reference or Tk garbage-collects it
```

**Widgets:** label entry, "Detect looking down" checkbox, Start / Pause / Stop,
countdown (from `focus_elapsed_s`, not wall clock), resolved camera device name, status
line, degraded-mode banner, and a "False positive" button that appends the current
`FrameReading` numbers to the ledger for later threshold tuning.

**Shutdown.** Bind `WM_DELETE_WINDOW` to a handler that signals the worker, joins it,
calls `cap.release()`, flushes the ledger, and releases the instance lock. Without this
the camera light can stay on after the window closes.

---

## 9. launchd *(fixes I-16)*

**Recommendation: don't auto-start this.** A pomodoro is something you deliberately
begin; a camera-permissioned ML process resident from login buys nothing and costs
battery, thermals, and a standing privacy surface. Launch it manually, or bind it to a
hotkey. The plist below is corrected in case you want it anyway.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>Label</key><string>com.mohammad.pomodoroguard</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/mohammad/pomodoro_guard/.venv/bin/python</string>
    <string>/Users/mohammad/pomodoro_guard/main.py</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/mohammad/pomodoro_guard</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>/Users/mohammad/pomodoro_guard/logs/out.log</string>
  <key>StandardErrorPath</key><string>/Users/mohammad/pomodoro_guard/logs/err.log</string>
</dict>
</plist>
```

Four fixes over v1: the venv interpreter instead of `/usr/bin/python3` (which has none of
the dependencies), `WorkingDirectory` so relative `config.json` resolves,
`KeepAlive`/`SuccessfulExit=false` so a crash restarts rather than dying silently, and
`ThrottleInterval` so a crash *loop* doesn't spin.

```bash
mkdir -p ~/pomodoro_guard/logs                       # launchd won't create it
cp launchd/com.mohammad.pomodoroguard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mohammad.pomodoroguard.plist
```

`launchctl bootstrap` supersedes the deprecated `load`.

**TCC camera permission.** Permission is granted to the *interpreter binary*, so granting
it to system Python does not grant it to `.venv/bin/python`. Run manually once to trigger
the prompt for the venv interpreter. Recreating or moving the venv can revoke it — if
detection suddenly sees no face ever, check System Settings → Privacy → Camera first.

---

## 10. `requirements.txt` *(fixes I-17)*

```
opencv-python==4.10.0.84
mediapipe==0.10.14
Pillow==10.4.0
requests==2.32.3
```

Pinned so a background upgrade can't change detector behaviour under you. `Pillow` was
missing in v1 despite §8 requiring the camera thumbnail. Still no torch/ultralytics.

Generate a hash-locked file for the real install:
`pip install pip-tools && pip-compile --generate-hashes`.

---

## 11. Observability *(fixes I-26)*

**Session ledger** — `state/sessions.jsonl`, one line per session, the source of truth
that the calendar is merely a projection of:

```json
{"id":"pg-1754000000000","label":"MATB24 problem set","outcome":"failed",
 "reason":"phone","started":"2026-07-31T14:02:11-04:00","focus_elapsed_s":842.3,
 "paused_total_s":31.0,"violations":2,"manual_pauses":0,"degraded":[],
 "look_down_enabled":false,"phone_confidence_at_violation":0.71,
 "pitch_delta_at_violation":8.2,"median_inference_ms":118}
```

Never contains image data. This gives you the data to answer "is my phone threshold too
low?" empirically instead of guessing, and lets you rebuild calendar entries if the
calendar is ever lost.

**Application logs** — `logging` to `logs/app.log` via `RotatingFileHandler`
(5 MB × 3), level from config. API keys scrubbed from every record.

**False-positive button** — logs the full `FrameReading` at violation time. After a week
you'll have a real distribution to set thresholds from.

---

## 12. Manual test plan

Run 1–8 with `"dry_run": true` so testing doesn't pollute the real calendar.

1. Sit normally for a full session → zero false violations.
2. Walk out of frame → alarm within `no_face_threshold_seconds`, grace countdown starts.
3. Return within grace → **timer resumes at the paused value, not reset**; confirm
   `focus_elapsed_s` in the ledger matches actual seated time ±2s.
4. Walk out and stay out → FAILED, no calendar POST (verify in the ledger and the outbox).
5. Look down sustained, with `detect_look_down: true` → violation fires; then confirm a
   *brief* glance down does not.
6. Hold a phone in frame → fires. Then hold a wallet and a TV remote → confirm neither does.
7. Full clean session, `dry_run: false` → entry appears on the live calendar with correct
   label, date, and times.
8. **Repeated-glance evasion:** look down 3.5s, up 0.5s, repeat 5×. The leaky bucket must
   eventually trip. Under v1's hard reset this ran forever undetected.

New tests covering the gaps:

9. **Camera stolen:** start a session, then open FaceTime. Expect `CAMERA_ERROR` + banner
   + paused clock — **not** a `left` violation and not a crash.
10. **Sleep:** start a session, close the lid for 5 minutes, reopen. Expect `ABORTED` with
    `clock_gap`, and **no** calendar entry. This is the I-05 regression test.
11. **Offline success:** disable Wi-Fi, complete a clean session. Expect "Queued", an
    entry in `outbox.jsonl`; re-enable Wi-Fi, restart, confirm it drains exactly once.
12. **Duplicate suppression:** run the outbox drain twice against the same entry.
    Exactly one calendar task.
13. **Two instances:** launch a second copy. Expect a clean exit with the lock message.
14. **Crash recovery:** `kill -9` mid-session, relaunch. Expect an `interrupted` ledger
    row, a notification, no calendar entry, no silent resume.
15. **Model missing:** rename `models/efficientdet_lite0.tflite`, start. Expect the app to
    run face-only with a visible red banner and `"degraded":["phone"]` in the ledger.
16. **Corrupt-blob refusal:** point `calendar_api_url` at a stub returning `{}`. Expect
    `CalendarUnsafe`, a loud banner, and **no POST**. This is the I-01 regression test —
    run it before ever pointing the app at the real calendar.
17. **Restore drill:** delete a day from the calendar, restore it from `state/backups/`.
    Confirm the backups are actually usable before you rely on them.

---

## 13. Build order

1. `x-api-key` auth + payload validation on the Next.js route (**I-03** — do this today,
   it's a live hole regardless of this project).
2. `curl` the live `/api/data` and pin the entry schema (**I-23**).
3. `.gitignore`, config loader with validation, Keychain read (**I-04**).
4. `calendar_client` + `outbox` + test 16 passing against a stub (**I-01**).
5. `/api/data/append` (**I-02**), then switch the client over.
6. `vision.py` with calibration wizard and degraded mode.
7. `timer.py` with monotonic accounting; tests 3, 10.
8. GUI, alarm, ledger.
9. Two weeks with `dry_run: true`, then tune thresholds from ledger data.
10. launchd only if you actually want it.

---

## Still needs a decision from you

1. **Deployed calendar URL**, and one real task object from it (§6.4).
2. **Is the calendar one JSON blob under one Redis key?** §6.1's Lua assumes it is.
3. **`grace_on_left`** — should standing up be instantly fatal? The grace period only
   really works for glances, since you probably can't hear the alarm from another room
   (I-20/I-21).
4. **Auto-start or not** (§9). Recommendation: no.
