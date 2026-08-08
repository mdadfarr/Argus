# Skipped bugs

Companion to `BUGS.md`. Everything here was deliberately **not** fixed in the
2026-08-07 bug-fix pass, with the reason. Two categories:

1. critical/major findings whose real fix needs a larger architectural change,
   hardware I can't reach, or a product decision that isn't mine to make;
2. everything filed under `BUGS.md` → *Needs verification*, which was moved here
   wholesale rather than fixed, because "I could not confirm this from the
   source" is not a state to be writing patches from.

Minor-severity findings were left alone by request and are not repeated here;
they remain listed in `BUGS.md`.

---

## Major findings not fixed

### vision.py MAJOR — the pitch sign has never been calibrated on this machine
**`vision.py` `_load_pitch_sign()`; `state/calibration.json` absent**

**Why skipped:** the actual fix is a 30-second interactive run of
`python vision.py --calibrate-sign` on the Mac with the camera attached. It
needs a face in front of a lens and two Enter presses, so it cannot be done
from a code change.

The code-side alternative `BUGS.md` suggests — refuse to enable look-down
detection while the sign is uncalibrated — is a product decision rather than a
bug fix: it would silently switch off a detector the user has explicitly
toggled on, and if the assumed `sign=1` happens to be correct it would be a
pure regression. Worth doing, but as a deliberate choice, not folded into a
fix-the-bugs pass.

**What to do:** run the wizard once. If look-down then starts firing where it
never did before, the sign was wrong and every look-down reading to date was
inverted.

---

### vision.py MAJOR — camera index derived from the ordinal position of `system_profiler` output
**`vision.py` `_list_camera_names()` / `resolve_camera_index()`**

**Why skipped:** architectural, and unverifiable from here. The suggested fix
replaces the whole resolution mechanism — enumerate devices through PyObjC's
`AVCaptureDevice.devices()` (which returns `localizedName` in the same order
OpenCV's AVFoundation backend uses) instead of parsing `system_profiler` text,
and probe candidate indices by opening them.

That means a new PyObjC dependency surface, a new cache-invalidation story for
`state/camera.json`, and a fallback path for when AVFoundation enumeration is
unavailable. It also cannot be tested without the actual hardware: the failure
mode is *opening the wrong camera*, which reads frames perfectly happily, so a
sandbox can't tell a correct fix from an incorrect one.

**Current mitigation:** none. The symptom to watch for is detection behaving
strangely with no error anywhere — no face found, or a face at an odd baseline.
Deleting `state/camera.json` forces re-resolution; setting `camera_name_hint`
to `null` and `camera_index` explicitly bypasses the name lookup entirely.

---

## Needs verification — moved here unfixed

These are `BUGS.md` § *Needs verification* items 2-7, untouched by request.
Item 1 (`reopen_on_dock_click`) was already resolved as *not a bug* and is not
carried over.

### NV-2 — kiosk chrome is not always restored after a focus block
Confirmed reachable by the user; mechanism still open between "(b) force-quit
during a block" and "(c) `setPresentationOptions_(Default)` not sticking at the
AppKit level". Mechanism (a) was disproved in `BUGS.md`.

Not fixed because the suggested defensive change (restore the chrome *before*
the destroy loop, outside the `if not windows: return` early return, re-assert
after the closes drain, and add a restore to `Engine.shutdown()`) is a guess
against three different mechanisms, and the cheapest next step is diagnostic,
not corrective: log on both sides of the restore and read
`app.presentationOptions()` back immediately after setting it.

Note the teardown *fallback* added in this pass (see below) reduces how often
this is reachable, but does not address the restore itself.

### NV-3 — prewarmed camera open leaking past an abort
Needs a stress test rather than a read. Partially mitigated as a side effect of
the `request_open()` handshake fix: `request_close()` now cancels a pending open
under the same condition variable that publishes it, so the specific ordering
described (`request_close` clears the flag, `run()` performs the open anyway) is
narrower than it was. Not claimed as fixed — the original report was about a
non-deterministic ordering that still wants a real test.

### NV-4 — whether `Alarm` zombies actually accumulate
Unconfirmed; CPython's `subprocess._cleanup()` may make the missing `wait()`
a non-issue in practice. Confirm with `ps` after a session with several
violations. (`Alarm` did gain a lock in this pass, for the orphaned-handle bug,
but `stop()` still terminates without waiting.)

### NV-5 — `main.py` module-level `STATE.mkdir(exist_ok=True)` runs at import
Harmless today (a mkdir), but it means `main` is not import-clean and escapes
`conftest.isolated_state`. Left alone: moving it into `main()` is a small change
with a real chance of breaking a path that quietly relies on `state/` existing
at import.

### NV-6 — `tests/test_concurrency.py::_bare_engine` drift
The three attributes added to `Engine.__init__` in this pass
(`_next_drain_mono`, `_draining`, `_focus_reopen_next`) were mirrored into
`_bare_engine` so it does not drift further. The underlying suggestion — assert
that the attribute sets match, so this can't silently rot again — was not
implemented.

### NV-7 — `focus.py` has no test file at all
Still true, and now more load-bearing: `FocusSession` gained camera-loss
handling and a changed violation-edge rule in this pass, and neither is covered.
`FocusSession` is a pure, dependency-free state machine, so a test file is cheap
— this is the single highest-value follow-up in this document.
