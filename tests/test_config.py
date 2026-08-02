"""Config validation.

Every value here reached the app as a KeyError or a ValueError on a background
thread, where it was logged and swallowed and the UI simply hung. Rejecting
them at load is what turns those into a message.
"""
from __future__ import annotations

import json

import pytest

import config


def test_shipped_configs_are_valid(cfg):
    config._validate(cfg)


def test_required_keys_match_what_the_app_reads(cfg):
    """If someone adds a cfg["new_key"] read without listing it, this catches
    it before a user does."""
    for key in config.REQUIRED_KEYS:
        assert key in cfg, f"{key} is required but missing from config.example.json"


@pytest.mark.parametrize("key", config.REQUIRED_KEYS)
def test_every_required_key_is_enforced(cfg, key):
    cfg.pop(key)
    with pytest.raises(config.ConfigError, match="missing required key"):
        config._validate(cfg)


@pytest.mark.parametrize("patch,expected", [
    ({"timezone": "Not/AZone"}, "not a valid IANA zone"),
    ({"timezone": ""}, "not a valid IANA zone"),
    ({"calendar_api_url": "http://insecure.example"}, "must use https"),
    ({"calendar_append_url": "ftp://nope"}, "must use https"),
    ({"calendar_api_key_source": "envvar"}, "keychain"),
    ({"check_interval_ms": 50}, "check_interval_ms"),
    ({"check_interval_ms": 9000}, "check_interval_ms"),
    ({"grace_period_seconds": 0}, "must be > 0"),
    ({"camera_reopen_backoff_seconds": []}, "at least 2 positive"),
    ({"camera_reopen_backoff_seconds": [1]}, "at least 2 positive"),
    ({"camera_reopen_backoff_seconds": [1, -5]}, "at least 2 positive"),
    ({"phone_confidence_threshold": 0}, r"must be in \(0, 1\]"),
    ({"phone_confidence_threshold": 1.5}, r"must be in \(0, 1\]"),
    ({"max_violations_per_session": -1}, "must be >= 0"),
    ({"max_manual_pauses": -1}, "must be >= 0"),
    ({"log_level": "CHATTY"}, "not a valid logging level"),
    ({"look_down_release_delta_degrees": 999}, "must be <"),
    ({"look_away_release_delta_degrees": 999}, "must be <"),
    ({"pomodoro_minutes": 9999}, "must hold"),
])
def test_invalid_values_are_rejected(cfg, patch, expected):
    cfg.update(patch)
    with pytest.raises(config.ConfigError, match=expected):
        config._validate(cfg)


def test_missing_alarm_sound_file_is_tolerated(cfg):
    """Deliberately not fatal: Alarm.start() and main.VIOLATION_SOUNDS both
    degrade on a missing file, so refusing to launch over a sound would be
    worse than the problem."""
    cfg["alarm_sound_path"] = "/nonexistent/sound.aiff"
    config._validate(cfg)


def test_optional_keys_may_be_absent(cfg):
    for key in ("calendar_api_key", "camera_index", "camera_name_hint",
                "phone_detect_every_n_ticks", "look_away_enter_delta_degrees",
                "look_away_release_delta_degrees", "look_away_threshold_seconds"):
        cfg.pop(key, None)
    config._validate(cfg)


def test_load_config_exits_on_bad_json(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text("{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        config.load_config(bad)


def test_load_config_exits_when_missing(tmp_path):
    with pytest.raises(SystemExit, match="Config file not found"):
        config.load_config(tmp_path / "nope.json")


def test_load_config_reports_the_offending_key(tmp_path, cfg):
    cfg["timezone"] = "Mars/Olympus"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="timezone"):
        config.load_config(path)


def test_api_key_from_config_source(cfg):
    cfg["calendar_api_key_source"] = "config"
    cfg["calendar_api_key"] = "secret"
    assert config.load_api_key(cfg) == "secret"


def test_api_key_source_config_without_key_raises(cfg):
    cfg["calendar_api_key_source"] = "config"
    cfg["calendar_api_key"] = None
    with pytest.raises(config.ConfigError):
        config.load_api_key(cfg)


def test_keychain_miss_raises_rather_than_exiting(cfg, monkeypatch):
    """load_api_key runs on background threads, where SystemExit is swallowed
    by the threading machinery and the real problem never reaches the user."""
    import subprocess

    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "security")

    monkeypatch.setattr(subprocess, "run", boom)
    cfg["calendar_api_key_source"] = "keychain"
    with pytest.raises(config.ConfigError, match="Keychain"):
        config.load_api_key(cfg)
