from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)


class Alarm:
    """Non-blocking alarm sound via subprocess.Popen -- subprocess.run would
    block the calling thread for the length of the sound, which is fatal if
    called during the grace period (the one moment detection must stay
    responsive)."""

    def __init__(self, sound_path: str):
        self.sound_path = sound_path
        self.proc: subprocess.Popen | None = None

    def start(self, sound_path: str | None = None) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        try:
            self.proc = subprocess.Popen(["afplay", sound_path or self.sound_path])
        except OSError as e:
            log.error("alarm failed to start: %s", e)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None


def _osa_string(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def notify(title: str, message: str) -> None:
    """Best-effort macOS notification. The primary violation is 'you left the
    room' -- you can't hear a desk speaker from the kitchen, so sound alone
    (Alarm) is not adequate feedback on its own."""
    script = f"display notification {_osa_string(message)} with title {_osa_string(title)}"
    try:
        subprocess.Popen(["osascript", "-e", script])
    except OSError as e:
        log.error("notification failed: %s", e)
