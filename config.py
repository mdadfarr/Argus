from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"

# Every key the app reads as cfg["..."]. A missing one used to surface as a
# KeyError deep inside Engine.__init__ or a tick, on a background thread, where
# it was logged and swallowed -- the UI just hung. Checking presence up front
# turns all of those into one clear message before the window ever opens.
REQUIRED_KEYS = (
    "alarm_sound_path", "calendar_api_key_source", "calendar_api_url",
    "calendar_append_url", "calibration_seconds", "camera_reopen_backoff_seconds",
    "check_interval_ms", "detect_phone", "dry_run", "grace_on_left",
    "grace_period_seconds", "log_level", "look_down_enter_delta_degrees",
    "look_down_release_delta_degrees", "look_down_threshold_seconds",
    "max_manual_pause_seconds", "max_manual_pauses", "max_session_minutes",
    "max_tick_gap_seconds", "max_total_paused_seconds",
    "max_violations_per_session", "min_session_minutes",
    "no_face_threshold_seconds", "notify_on_failure",
    "phone_confidence_threshold", "phone_sustain_seconds", "pomodoro_minutes",
    "timezone",
)


class ConfigError(Exception):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _validate(cfg: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    _require(not missing, f"missing required key(s): {', '.join(missing)}")

    for url_key in ("calendar_api_url", "calendar_append_url"):
        url = cfg.get(url_key, "")
        _require(url.startswith("https://"), f"{url_key} must use https")

    # Unvalidated, this reached ZoneInfo() only at session-logging time, where
    # ZoneInfoNotFoundError wedged the state machine in LOGGING forever.
    try:
        ZoneInfo(cfg["timezone"])
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ConfigError(f"timezone {cfg['timezone']!r} is not a valid IANA zone ({e})") from e

    _require(
        cfg["min_session_minutes"] <= cfg["pomodoro_minutes"] <= cfg["max_session_minutes"],
        "min_session_minutes <= pomodoro_minutes <= max_session_minutes must hold",
    )
    for dur_key in (
        "min_session_minutes", "max_session_minutes", "grace_period_seconds",
        "max_total_paused_seconds", "max_manual_pause_seconds",
        "no_face_threshold_seconds", "look_down_threshold_seconds",
        "calibration_seconds", "phone_sustain_seconds", "max_tick_gap_seconds",
    ):
        _require(cfg[dur_key] > 0, f"{dur_key} must be > 0")

    _require(100 <= cfg["check_interval_ms"] <= 2000, "check_interval_ms must be in [100, 2000]")
    _require(
        cfg["calendar_api_key_source"] in ("keychain", "config"),
        "calendar_api_key_source must be 'keychain' or 'config'",
    )

    for count_key in ("max_violations_per_session", "max_manual_pauses"):
        _require(cfg[count_key] >= 0, f"{count_key} must be >= 0")

    backoffs = cfg["camera_reopen_backoff_seconds"]
    _require(
        isinstance(backoffs, list) and len(backoffs) >= 2 and all(b > 0 for b in backoffs),
        "camera_reopen_backoff_seconds must be a list of at least 2 positive numbers",
    )

    _require(
        0.0 < cfg["phone_confidence_threshold"] <= 1.0,
        "phone_confidence_threshold must be in (0, 1]",
    )

    # Hysteresis: release must sit below enter or the latch never clears.
    for enter_key, release_key, default_enter, default_release in (
        ("look_down_enter_delta_degrees", "look_down_release_delta_degrees", None, None),
        ("look_away_enter_delta_degrees", "look_away_release_delta_degrees", 28, 18),
    ):
        enter = cfg[enter_key] if default_enter is None else cfg.get(enter_key, default_enter)
        release = cfg[release_key] if default_release is None else cfg.get(release_key, default_release)
        _require(release < enter, f"{release_key} must be < {enter_key}")

    _validate_gaze(cfg)

    _require(
        cfg["log_level"].upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        f"log_level {cfg['log_level']!r} is not a valid logging level",
    )
    # alarm_sound_path is deliberately only checked for presence, not for
    # existence on disk: Alarm.start() and main.VIOLATION_SOUNDS both already
    # degrade gracefully on a missing file, so refusing to launch over a
    # cosmetic sound would be worse than the problem.


def _validate_gaze(cfg: dict) -> None:
    """Validate the gaze block, but only when it is switched on.

    The whole group is optional and absent from REQUIRED_KEYS on purpose: gaze
    needs torch and a downloaded checkpoint (see requirements-gaze.txt), so an
    install that never enables it must not be forced to carry the settings.
    Once detect_gaze is true the paths are load-bearing, and a typo in one of
    them would otherwise surface as a silent DEGRADED banner rather than a
    refusal to launch -- which is the behaviour every other detector here has.
    """
    if not cfg.get("detect_gaze", False):
        return

    for key in ("gaze_checkpoint_path", "gaze_calibration_path"):
        _require(key in cfg, f"detect_gaze is on but {key} is missing")
        _require(bool(cfg[key]), f"{key} must not be empty when detect_gaze is on")
        # Deliberately not checked for existence on disk. The calibration file
        # legitimately does not exist until tools/calibrate_screens.py has been
        # run once, and refusing to launch over that would strand the user with
        # no way to produce it.

    _require(
        isinstance(cfg.get("gaze_person_idx", 0), int) and 0 <= cfg.get("gaze_person_idx", 0) < 30,
        "gaze_person_idx must be an int in [0, 30) -- it indexes the model's "
        "per-participant bias table, which has 15 participants x2 (mirrored)",
    )
    _require(cfg.get("gaze_every_n_ticks", 1) >= 1, "gaze_every_n_ticks must be >= 1")

    for dim in ("gaze_capture_width", "gaze_capture_height"):
        _require(cfg.get(dim, 0) > 0, f"{dim} must be > 0 when detect_gaze is on")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not path.exists():
        sys.exit(
            f"Config file not found: {path}\n"
            f"Copy config.example.json to {path.name}, fill in secrets, and chmod 600 it."
        )
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"Config file is not valid JSON: {e}")

    try:
        _validate(cfg)
    except ConfigError as e:
        sys.exit(f"Invalid config: {e}")
    except KeyError as e:
        sys.exit(f"Config is missing required key: {e}")

    return cfg


def load_api_key(cfg: dict) -> str:
    """Raises ConfigError if no key is available.

    Deliberately does NOT call sys.exit(): this runs on background threads
    (outbox drain, calendar send), where SystemExit is silently swallowed by
    the threading machinery and the real problem never reaches the user.
    """
    source = cfg["calendar_api_key_source"]
    if source == "keychain":
        try:
            out = subprocess.run(
                ["security", "find-generic-password", "-s", "pomodoro-guard-calendar", "-w"],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            raise ConfigError(
                "No API key found in Keychain under service 'pomodoro-guard-calendar'. "
                "Run: security add-generic-password -a \"$USER\" "
                "-s pomodoro-guard-calendar -w 'YOUR_KEY'"
            ) from e
        key = out.stdout.strip()
        if not key:
            raise ConfigError("Keychain entry 'pomodoro-guard-calendar' is empty.")
        return key

    key = cfg.get("calendar_api_key")
    if not key:
        raise ConfigError("calendar_api_key_source is 'config' but calendar_api_key is empty")
    return key
