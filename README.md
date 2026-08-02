# Argus

A pomodoro timer that watches whether you actually stayed at the desk. macOS
only. It uses the webcam to detect that you left, picked up your phone, or
looked away, and fails the session if you don't come back within the grace
period. Completed sessions are logged to a calendar API.

Everything runs locally. No frame is ever written to disk or sent anywhere —
`state/sessions.jsonl` holds only numeric measurements and outcomes, and the
UI thumbnail is an in-memory JPEG that never leaves the process.

## Requirements

- macOS (uses AVFoundation, `afplay`, `osascript`, `security`, `system_profiler`)
- Python 3.11
- A webcam, and Camera permission granted to whichever binary launches Argus

## Setup

```bash
git clone <this repo> && cd Argus

python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp config.example.json config.json
chmod 600 config.json          # it can hold an API key
```

The detection models are committed under `models/` and are checksum-verified at
load; if a checksum fails, that detector is disabled and the UI shows a
`DEGRADED` banner rather than silently misbehaving.

### Calendar credentials

`config.json` defaults to `"calendar_api_key_source": "keychain"`. Store the key
once:

```bash
security add-generic-password -a "$USER" -s pomodoro-guard-calendar -w 'YOUR_KEY'
```

Set `"dry_run": true` to run the whole flow without ever calling the calendar —
sessions are logged locally and the intended request is written to
`logs/app.log`. The shipped `config.example.json` has `dry_run` on.

The alternative, `"calendar_api_key_source": "config"`, reads
`calendar_api_key` straight from `config.json`. Only use it with `chmod 600`;
`config.json` is gitignored and must stay that way.

### One-time pitch calibration

Look-down detection depends on which direction MediaPipe reports pitch, and that
sign is not guaranteed across versions. If you enable `detect_look_down`, run:

```bash
.venv/bin/python vision.py --calibrate-sign
```

This writes `state/calibration.json`. Without it the app assumes `sign=1` and
logs a warning at startup. Look-*away* detection needs no such step — it
compares absolute deviation, so left and right count the same.

## Running

```bash
.venv/bin/python main.py     # or: ./run_pomodoro.command
```

`Argus.app` is a bundle wrapper around the same command — it locates the repo
from its own path, so the checkout can live anywhere as long as `.venv` sits
beside `main.py`. To start Argus at login, edit the two absolute paths in
`launchd/com.mohammad.pomodoroguard.plist` to match your checkout, then:

```bash
cp launchd/com.mohammad.pomodoroguard.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mohammad.pomodoroguard.plist
```

Only one instance can run at a time; a second launch exits immediately against
the `state/instance.lock` flock.

## Configuration

Every key in `config.example.json` is required except `calendar_api_key`,
`camera_index`, `camera_name_hint`, `phone_detect_every_n_ticks`, and the
`look_away_*` group. Missing or invalid values are rejected at startup with a
message naming the key — the app will not launch half-configured.

The ones worth tuning:

| Key | Meaning |
| --- | --- |
| `pomodoro_minutes` | Default session length, clamped to `min`/`max_session_minutes` |
| `grace_period_seconds` | How long you have to return before the session fails |
| `max_violations_per_session` | Violations tolerated before automatic failure |
| `no_face_threshold_seconds` | Sustained absence before "left" fires |
| `look_away_enter/release_delta_degrees` | Yaw hysteresis; release must be below enter |
| `look_down_enter/release_delta_degrees` | Pitch hysteresis; needs the sign calibration above |
| `phone_confidence_threshold` | Detector score (0–1] at which a phone counts |
| `camera_name_hint` | Device name to pin; Continuity Camera can steal index 0 |

Thresholds are easier to set from real numbers than from guesswork. The main
window shows live detector output, and the **false positive** button appends the
current readings to `state/false_positives.jsonl` for later tuning.

## Files it writes

| Path | Contents |
| --- | --- |
| `state/sessions.jsonl` | One row per finished session: outcome, reason, timings |
| `state/outbox.jsonl` | Sessions awaiting calendar sync; drained at startup |
| `state/current_session.json` | Crash checkpoint; a leftover one is reported as interrupted |
| `state/calibration.json` | Pitch sign from the wizard |
| `state/camera.json` | Cached camera-index resolution |
| `logs/app.log` | Rotating application log (5 MB × 3) |
| `logs/err.log`, `logs/out.log` | Raw stdout/stderr when launched via the bundle or launchd. **These do not rotate** — truncate them occasionally |

## Troubleshooting

**Window opens but Start never enables.** The engine failed to build. The window
shows `failed to start` with the error; the traceback is in `logs/app.log`.

**"Calendar rejected the API key (401)".** The key in the Keychain doesn't match
the server's. Sessions still record locally and queue in `state/outbox.jsonl`.

**"Camera unavailable — retrying…".** Argus retries on the
`camera_reopen_backoff_seconds` ladder, then aborts the session without
penalty. Check that the launching binary has Camera permission in System
Settings → Privacy & Security.

**Sessions fail immediately with `clock_gap`.** The machine slept or stalled
mid-session, exceeding `max_tick_gap_seconds`. Aborts are never counted against
you.

## Development

```bash
.venv/bin/python -m pytest -q          # unit tests (no camera needed)
```

`SessionTimer` takes all its side effects as callbacks and advances on a
measured `dt`, so the whole state machine tests without a camera, a window, or
the network. `tests/` covers session outcomes, violation and grace handling,
config validation, the outbox, and the two UI-blocking regressions described in
`tests/test_concurrency.py`.
