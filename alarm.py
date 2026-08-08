from __future__ import annotations

import logging
import subprocess
import threading

log = logging.getLogger(__name__)


class Alarm:
    """Non-blocking alarm sound via subprocess.Popen -- subprocess.run would
    block the calling thread for the length of the sound, which is fatal if
    called during the grace period (the one moment detection must stay
    responsive)."""

    def __init__(self, sound_path: str):
        self.sound_path = sound_path
        self.proc: subprocess.Popen | None = None
        # self.proc is read-modify-written from the tick thread (violations)
        # and from the main thread (shutdown). Without this, a start() and a
        # stop() interleaving could overwrite the handle of a running afplay
        # and leave it playing to the end with no way to kill it.
        self._lock = threading.Lock()
        self._playing_path: str | None = None

    def start(self, sound_path: str | None = None) -> None:
        """Starting the same sound that is already playing is a no-op, but a
        *different* sound preempts it. The early return used to be
        unconditional, and looked_down.aiff is 25.8s long -- so for up to 26s
        after a look-down/look-away violation every later alarm was silently
        dropped, including the 'you left the room' one, which is the whole
        point of having a sound at all."""
        path = sound_path or self.sound_path
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                if path == self._playing_path:
                    return
                self._terminate_locked()
            try:
                self.proc = subprocess.Popen(["afplay", path])
                self._playing_path = path
            except OSError as e:
                self.proc = None
                self._playing_path = None
                log.error("alarm failed to start: %s", e)

    def stop(self) -> None:
        with self._lock:
            self._terminate_locked()

    def _terminate_locked(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
        self._playing_path = None


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
