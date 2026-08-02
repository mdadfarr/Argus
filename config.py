from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"


class ConfigError(Exception):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _validate(cfg: dict) -> None:
    for url_key in ("calendar_api_url", "calendar_append_url"):
        url = cfg.get(url_key, "")
        _require(url.startswith("https://"), f"{url_key} must use https")

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
        except subprocess.CalledProcessError:
            raise ConfigError(
                "No API key found in Keychain under service 'pomodoro-guard-calendar'. "
                "Run: security add-generic-password -a \"$USER\" "
                "-s pomodoro-guard-calendar -w 'YOUR_KEY'"
            )
        key = out.stdout.strip()
        if not key:
            raise ConfigError("Keychain entry 'pomodoro-guard-calendar' is empty.")
        return key

    key = cfg.get("calendar_api_key")
    if not key:
        raise ConfigError("calendar_api_key_source is 'config' but calendar_api_key is empty")
    return key
