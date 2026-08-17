"""macOS display enumeration.

Replaces upstream's `get_monitor_dimensions()`, which reaches for GTK via `pgi`
and returns `(None, None)` on any machine without it -- so on macOS the whole
pipeline fell back to "please supply monitor dimensions manually".

Two sources, in order of preference:

  * Quartz (pyobjc). Gives display IDs, global desktop bounds and physical size
    in millimetres, all from the window server.
  * `system_profiler SPDisplaysDataType -json`. Always present on macOS, no
    dependency, but reports resolution only -- no physical size, no reliable
    global origin.

Physical millimetres are a *cross-check*, not a requirement: corner-look
calibration measures the panel itself, so a display whose EDID lies (common
with TVs and some ultrawides) still calibrates correctly. What is genuinely
needed here is pixel dimensions and where each display sits on the global
desktop, so a per-monitor hit can be reported as a global cursor position.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Display:
    display_id: int
    name: str
    pixels: tuple[int, int]
    origin: tuple[int, int] = (0, 0)     # top-left on the global desktop
    size_mm: tuple[float, float] | None = None
    is_main: bool = False

    @property
    def diagonal_in(self) -> float | None:
        if self.size_mm is None:
            return None
        return (self.size_mm[0] ** 2 + self.size_mm[1] ** 2) ** 0.5 / 25.4

    def to_global(self, x: float, y: float) -> tuple[float, float]:
        return self.origin[0] + x, self.origin[1] + y

    def describe(self) -> str:
        bits = [f"{self.name} {self.pixels[0]}x{self.pixels[1]}"]
        if self.origin != (0, 0):
            bits.append(f"at {self.origin[0]:+d},{self.origin[1]:+d}")
        if self.diagonal_in:
            bits.append(f"~{self.diagonal_in:.0f}in")
        if self.is_main:
            bits.append("(main)")
        return " ".join(bits)


def _from_quartz() -> list[Display]:
    import Quartz  # noqa: PLC0415  -- optional dependency, probed at call time

    err, ids, _ = Quartz.CGGetActiveDisplayList(16, None, None)
    if err != 0:
        raise OSError(f"CGGetActiveDisplayList failed with {err}")

    displays = []
    for did in ids:
        bounds = Quartz.CGDisplayBounds(did)
        size = Quartz.CGDisplayScreenSize(did)  # millimetres, from EDID
        mm = (float(size.width), float(size.height))
        displays.append(Display(
            display_id=int(did),
            name=f"display-{did}",
            pixels=(int(Quartz.CGDisplayPixelsWide(did)), int(Quartz.CGDisplayPixelsHigh(did))),
            origin=(int(bounds.origin.x), int(bounds.origin.y)),
            # A display that reports 0x0 mm is lying rather than tiny; treat it
            # as unknown so the sanity checks do not compare against garbage.
            size_mm=mm if mm[0] > 1 and mm[1] > 1 else None,
            is_main=bool(Quartz.CGDisplayIsMain(did)),
        ))
    return displays


def _from_system_profiler() -> list[Display]:
    out = subprocess.run(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    data = json.loads(out.stdout)

    displays: list[Display] = []
    for gpu in data.get("SPDisplaysDataType", []):
        for i, panel in enumerate(gpu.get("spdisplays_ndrvs", [])):
            res = panel.get("_spdisplays_resolution") or panel.get("spdisplays_resolution") or ""
            # e.g. "2560 x 1440 @ 60.00Hz" -- take the first two integers
            nums = [int(t) for t in res.replace("x", " ").replace("@", " ").split() if t.isdigit()]
            if len(nums) < 2:
                continue
            displays.append(Display(
                display_id=i,
                name=panel.get("_name", f"display-{i}"),
                pixels=(nums[0], nums[1]),
                is_main=panel.get("spdisplays_main") == "spdisplays_yes",
            ))
    return displays


def list_displays() -> list[Display]:
    """Every active display, main first.

    Never raises: a machine we cannot enumerate should degrade to "calibrate
    the one screen you can see" rather than refusing to start.
    """
    for source, fn in (("Quartz", _from_quartz), ("system_profiler", _from_system_profiler)):
        try:
            displays = fn()
            if displays:
                log.info("enumerated %d display(s) via %s", len(displays), source)
                return sorted(displays, key=lambda d: (not d.is_main, d.display_id))
        except Exception as e:
            log.debug("display enumeration via %s failed: %s", source, e)

    log.warning("could not enumerate displays; assuming a single 1920x1080 screen")
    return [Display(display_id=0, name="unknown", pixels=(1920, 1080), is_main=True)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for d in list_displays():
        print(" ", d.describe())
