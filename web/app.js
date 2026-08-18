(() => {
  const el = (id) => document.getElementById(id);

  const clockEl = el('clock');
  const headlineEl = el('headline');
  const countdownEl = el('countdown');
  const progressFillEl = el('progressFill');
  const labelInput = el('labelInput');
  labelInput.addEventListener('input', () => {
    labelInput.classList.remove('error');
    statusMessage.classList.remove('error');
  });
  const minutesInput = el('minutesInput');
  const cameraToggle = el('cameraToggle');
  const lookDownToggle = el('lookDownToggle');
  const lookAwayToggle = el('lookAwayToggle');
  const dFace = el('dFace'), dPhone = el('dPhone'), dPitch = el('dPitch'), dYaw = el('dYaw');
  const diagHint = el('diagHint');
  const statCamera = el('statCamera');
  const statState = el('statState');
  const statViolations = el('statViolations');
  const thumb = el('thumb');
  const thumbPlaceholder = el('thumbPlaceholder');
  const cameraStatus = el('cameraStatus');
  const backToMiniBtn = el('backToMiniBtn');
  const themeToggle = el('themeToggle');
  const themeToggleIcon = el('themeToggleIcon');
  const focusBtns = Array.from(document.querySelectorAll('.focus-btn'));
  const startBtn = el('startBtn');
  const pauseBtn = el('pauseBtn');
  const stopBtn = el('stopBtn');
  const falsePositiveBtn = el('falsePositiveBtn');
  const statusMessage = el('statusMessage');
  const authBanner = el('authBanner');
  const degradedBanner = el('degradedBanner');
  const gazeBand = el('gazeBand');
  const gazeStatus = el('gazeStatus');
  const gazeHint = el('gazeHint');
  const gazeCalibrateBtn = el('gazeCalibrateBtn');
  const dGaze = el('dGaze'), dGazeItem = el('dGazeItem');

  let apiReady = false;
  let cameraOn = true;
  let lookDownOn = false;
  let lookAwayOn = true;
  let defaultsApplied = false;
  let miniShown = false;
  let focusWasActive = false;

  function setToggle(btn, on) { btn.dataset.on = String(on); }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    themeToggleIcon.textContent = theme === 'light' ? '☀' : '☾';
  }
  applyTheme(document.documentElement.getAttribute('data-theme') || 'dark');
  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    const next = current === 'light' ? 'dark' : 'light';
    applyTheme(next);
    try { localStorage.setItem('argus-theme', next); } catch (e) {}
  });

  cameraToggle.addEventListener('click', () => {
    cameraOn = !cameraOn;
    setToggle(cameraToggle, cameraOn);
    // Head-pose detection is meaningless without the camera.
    if (!cameraOn) {
      lookDownOn = false; setToggle(lookDownToggle, false);
      lookAwayOn = false; setToggle(lookAwayToggle, false);
    }
    for (const t of [lookDownToggle, lookAwayToggle]) {
      t.disabled = !cameraOn;
      t.style.opacity = cameraOn ? '1' : '0.35';
    }
  });

  lookDownToggle.addEventListener('click', () => {
    if (!cameraOn) return;
    lookDownOn = !lookDownOn;
    setToggle(lookDownToggle, lookDownOn);
  });

  lookAwayToggle.addEventListener('click', () => {
    if (!cameraOn) return;
    lookAwayOn = !lookAwayOn;
    setToggle(lookAwayToggle, lookAwayOn);
  });

  function renderDiag(d) {
    if (!d || !d.active) {
      for (const n of [dFace, dPhone, dPitch, dYaw]) {
        n.textContent = '—'; n.className = 'diag-v';
      }
      diagHint.textContent = 'Starts reporting once a session is running.';
      return;
    }
    dFace.textContent = d.face ? 'yes' : 'NO';
    dFace.className = 'diag-v' + (d.face ? ' on' : ' hot');

    dPhone.textContent = d.phone.toFixed(2);
    dPhone.className = 'diag-v' + (d.phone >= d.phone_threshold ? ' hot' : (d.phone > 0 ? ' on' : ''));

    const fmt = (v, thr, node) => {
      if (v === null || v === undefined) { node.textContent = 'n/a'; node.className = 'diag-v'; return; }
      node.textContent = `${v}°`;
      node.className = 'diag-v' + (v >= thr ? ' hot' : ' on');
    };
    fmt(d.pitch, d.pitch_threshold, dPitch);
    fmt(d.yaw, d.yaw_threshold, dYaw);

    if (d.gaze === null || d.gaze === undefined) {
      dGazeItem.style.display = 'none';
    } else {
      dGazeItem.style.display = '';
      dGaze.textContent = d.gaze;
      dGaze.className = 'diag-v' + (d.gaze === 'off-screen' ? ' hot' : ' on');
    }

    diagHint.textContent = d.calibrated
      ? `thresholds — phone ${d.phone_threshold}, down ${d.pitch_threshold}°, away ${d.yaw_threshold}°`
      : 'calibrating… hold still and look at the screen';
  }

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function requireLabel() {
    if (labelInput.value.trim()) return true;
    statusMessage.textContent = 'Label is required.';
    statusMessage.classList.add('error');
    labelInput.classList.add('error');
    labelInput.focus();
    return false;
  }

  for (const btn of focusBtns) {
    btn.addEventListener('click', async () => {
      if (!apiReady || btn.disabled) return;
      // The label is asked for up front because there is no asking for it
      // later: the pomodoro that follows the block starts by itself, with the
      // screens still black.
      if (!requireLabel()) return;
      for (const b of focusBtns) b.disabled = true;
      const minutes = parseInt(btn.dataset.minutes, 10);
      try {
        // Popup goes up first (hiding the main window), then the engine-side
        // camera open runs behind it for the same fixed 5s the popup counts
        // down -- the black screens themselves only appear once the popup is
        // gone, via focus_open_windows() below.
        await window.pywebview.api.open_countdown_popup();
        const [res] = await Promise.all([
          window.pywebview.api.focus_start(labelInput.value, minutes, lookDownOn, lookAwayOn),
          delay(5000),
        ]);
        await window.pywebview.api.close_countdown_popup();
        if (res && res.error) {
          await window.pywebview.api.show_main();
          statusMessage.textContent = res.error;
          statusMessage.classList.add('error');
          for (const b of focusBtns) b.disabled = false;
          return;
        }
        const openRes = await window.pywebview.api.focus_open_windows();
        if (openRes && openRes.error) {
          await window.pywebview.api.show_main();
          statusMessage.textContent = openRes.error;
          statusMessage.classList.add('error');
          for (const b of focusBtns) b.disabled = false;
          return;
        }
        statusMessage.classList.remove('error');
        labelInput.classList.remove('error');
      } catch (e) {
        await window.pywebview.api.close_countdown_popup();
        await window.pywebview.api.show_main();
        for (const b of focusBtns) b.disabled = false;
      }
    });
  }

  startBtn.addEventListener('click', async () => {
    if (!apiReady) return;
    if (!requireLabel()) return;
    const minutes = parseFloat(minutesInput.value);
    const minutesVal = isNaN(minutes) ? null : minutes;
    startBtn.disabled = true;
    try {
      // Only camera sessions need the camera to boot, so only they get the
      // popup -- a timer-only session can start the instant it's asked to.
      if (cameraOn) {
        await window.pywebview.api.open_countdown_popup();
        const [res] = await Promise.all([
          window.pywebview.api.start(labelInput.value, minutesVal, lookDownOn, cameraOn, lookAwayOn),
          delay(5000),
        ]);
        await window.pywebview.api.close_countdown_popup();
        if (res && res.error) {
          await window.pywebview.api.show_main();
          statusMessage.textContent = res.error;
          statusMessage.classList.add('error');
          startBtn.disabled = false;
          return;
        }
      } else {
        const res = await window.pywebview.api.start(
          labelInput.value, minutesVal, lookDownOn, cameraOn, lookAwayOn
        );
        if (res && res.error) {
          statusMessage.textContent = res.error;
          statusMessage.classList.add('error');
          startBtn.disabled = false;
          return;
        }
      }
      statusMessage.classList.remove('error');
      labelInput.classList.remove('error');
      await window.pywebview.api.enter_mini();
      miniShown = true;
    } catch (e) {
      await window.pywebview.api.close_countdown_popup();
      await window.pywebview.api.show_main();
      startBtn.disabled = false;
    }
  });

  backToMiniBtn.addEventListener('click', async () => {
    if (!apiReady) return;
    await window.pywebview.api.enter_mini();
    miniShown = true;
  });

  pauseBtn.addEventListener('click', () => {
    if (apiReady) window.pywebview.api.pause_resume();
  });
  stopBtn.addEventListener('click', () => {
    if (apiReady) window.pywebview.api.stop();
  });
  falsePositiveBtn.addEventListener('click', () => {
    if (apiReady) window.pywebview.api.false_positive();
  });

  function setBanner(node, text, plain) {
    if (text) {
      node.textContent = text;
      node.classList.add('visible');
      node.classList.toggle('plain', !!plain);
    } else {
      node.classList.remove('visible');
    }
  }

  function applyState(s) {
    if (!s) return;

    headlineEl.textContent = s.headline || '';
    countdownEl.textContent = s.countdown || '--:--';
    progressFillEl.style.width =
      `${Math.max(0, Math.min(1, s.progress_frac || 0)) * 100}%`;

    statCamera.textContent = s.camera_name || 'not open';
    statState.textContent = (s.state || 'idle').toLowerCase();
    statViolations.textContent = s.violations != null ? String(s.violations) : '0';
    cameraStatus.textContent = s.camera_on
      ? `${s.camera_name} · ${s.state}`
      : 'camera off — timer only';

    if (s.thumbnail_b64) {
      thumb.src = s.thumbnail_b64;
      thumb.classList.add('visible');
      thumbPlaceholder.style.display = 'none';
    } else {
      thumb.classList.remove('visible');
      thumbPlaceholder.style.display = 'flex';
      thumbPlaceholder.textContent = s.camera_on ? 'no signal' : 'camera off';
    }

    renderDiag(s.diag);
    renderGaze(s.gaze);

    statusMessage.textContent = s.status_message || '';
    setBanner(authBanner, s.auth_warning, true);
    setBanner(degradedBanner, s.degraded, false);

    const b = s.buttons || {};
    startBtn.disabled = !b.start_enabled;
    // start_enabled already goes false while a focus block runs, so this also
    // covers the window where the black screens are up.
    for (const fb of focusBtns) fb.disabled = !b.start_enabled;
    pauseBtn.disabled = !b.pause_enabled;
    pauseBtn.textContent = b.pause_label || 'pause';
    stopBtn.disabled = !b.stop_enabled;
    falsePositiveBtn.disabled = !b.false_positive_enabled;

    if (!defaultsApplied && s.default_minutes) {
      minutesInput.value = s.default_minutes;
      defaultsApplied = true;
    }

    // Fallback teardown for the black focus windows. focus.js's own poll is
    // the primary path and normally gets there first; this exists because that
    // was the *only* path, and if its bridge breaks the user is left with every
    // screen black, no cursor and no menu bar, with nothing in the UI able to
    // recover. Calling focus_exit() twice is safe -- it no-ops when no windows
    // are open.
    if (s.focus_active) {
      focusWasActive = true;
    } else if (focusWasActive) {
      focusWasActive = false;
      if (apiReady) window.pywebview.api.focus_exit();
    }

    // Session ended by any path (success, failure, abort) -> restore layout.
    if (miniShown && !s.session_active) {
      miniShown = false;
      if (apiReady) window.pywebview.api.exit_mini();
    }

    // Only reachable mid-session by double-clicking the popup, so the only
    // way back to it otherwise was quitting the session entirely.
    backToMiniBtn.style.display = s.session_active ? 'inline' : 'none';
  }

  async function poll() {
    if (apiReady) {
      try {
        applyState(await window.pywebview.api.get_state());
      } catch (e) { /* backend not ready */ }
    }
    setTimeout(poll, 400);
  }

  function tickClock() {
    const p = (n) => String(n).padStart(2, '0');
    const d = new Date();
    clockEl.textContent = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
    setTimeout(tickClock, 1000);
  }

  window.addEventListener('pywebviewready', () => { apiReady = true; });

  tickClock();
  poll();

  // ---------- eye tracking ----------
  function renderGaze(g) {
    // The whole band stays hidden when detect_gaze is off, rather than showing
    // a disabled control. A feature you have not switched on should not take up
    // room explaining that it is not switched on.
    if (!g || !g.enabled) {
      gazeBand.style.display = 'none';
      dGazeItem.style.display = 'none';
      return;
    }
    gazeBand.style.display = '';

    if (g.calibrating) {
      gazeStatus.textContent = g.progress || 'Calibrating…';
      gazeStatus.className = 'mono-small';
      gazeCalibrateBtn.disabled = true;
      gazeCalibrateBtn.textContent = 'Calibrating…';
      gazeHint.textContent = 'Follow the dots on each screen. Move between rounds when asked.';
      return;
    }

    gazeCalibrateBtn.textContent = g.needs_calibration ? 'Calibrate screens' : 'Recalibrate screens';
    gazeCalibrateBtn.disabled = !g.can_calibrate;

    // The result of the last attempt outranks the general status. A run that
    // fails on a missing prerequisite takes well under a second, so if this
    // only showed while `calibrating` was true the reason would flash past
    // between polls and the button would look like it did nothing at all.
    const lastResult = (!g.calibrating && g.progress) ? g.progress : '';
    gazeStatus.textContent = lastResult || g.message || '';
    gazeStatus.className = 'mono-small' +
      ((g.needs_calibration || /failed|timed out/i.test(lastResult)) ? ' hot' : '');

    if (g.state === 'no_deps') {
      gazeHint.textContent = 'Install the gaze dependencies, then restart Argus.';
    } else if (g.needs_calibration) {
      gazeHint.textContent = 'Sessions still run without this — only eye tracking is affected.';
    } else if (!g.can_calibrate) {
      gazeHint.textContent = 'Stop the current session to recalibrate.';
    } else {
      gazeHint.textContent = '';
    }

    dGazeItem.style.display = (g.state === 'ok') ? '' : 'none';
  }

  gazeCalibrateBtn.addEventListener('click', async () => {
    if (!apiReady) return;
    gazeCalibrateBtn.disabled = true;
    try {
      const r = await window.pywebview.api.gaze_calibrate();
      if (r && r.ok === false) {
        statusMessage.textContent = r.error || 'Could not start calibration.';
        statusMessage.classList.add('error');
      }
    } catch (e) {
      statusMessage.textContent = 'Could not start calibration.';
    }
  });

})();
