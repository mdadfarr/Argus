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
  const startBtn = el('startBtn');
  const pauseBtn = el('pauseBtn');
  const stopBtn = el('stopBtn');
  const falsePositiveBtn = el('falsePositiveBtn');
  const statusMessage = el('statusMessage');
  const authBanner = el('authBanner');
  const degradedBanner = el('degradedBanner');

  let apiReady = false;
  let cameraOn = true;
  let lookDownOn = false;
  let lookAwayOn = true;
  let defaultsApplied = false;
  let miniShown = false;

  function setToggle(btn, on) { btn.dataset.on = String(on); }

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

    diagHint.textContent = d.calibrated
      ? `thresholds — phone ${d.phone_threshold}, down ${d.pitch_threshold}°, away ${d.yaw_threshold}°`
      : 'calibrating… hold still and look at the screen';
  }

  startBtn.addEventListener('click', async () => {
    if (!apiReady) return;
    if (!labelInput.value.trim()) {
      statusMessage.textContent = 'Label is required.';
      statusMessage.classList.add('error');
      labelInput.classList.add('error');
      labelInput.focus();
      return;
    }
    const minutes = parseFloat(minutesInput.value);
    startBtn.disabled = true;
    try {
      const res = await window.pywebview.api.start(
        labelInput.value, isNaN(minutes) ? null : minutes, lookDownOn, cameraOn, lookAwayOn
      );
      if (res && res.error) {
        statusMessage.textContent = res.error;
        statusMessage.classList.add('error');
        startBtn.disabled = false;
        return;
      }
      statusMessage.classList.remove('error');
      labelInput.classList.remove('error');
      await window.pywebview.api.enter_mini();
      miniShown = true;
    } catch (e) {
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

    statusMessage.textContent = s.status_message || '';
    setBanner(authBanner, s.auth_warning, true);
    setBanner(degradedBanner, s.degraded, false);

    const b = s.buttons || {};
    startBtn.disabled = !b.start_enabled;
    pauseBtn.disabled = !b.pause_enabled;
    pauseBtn.textContent = b.pause_label || 'pause';
    stopBtn.disabled = !b.stop_enabled;
    falsePositiveBtn.disabled = !b.false_positive_enabled;

    if (!defaultsApplied && s.default_minutes) {
      minutesInput.value = s.default_minutes;
      defaultsApplied = true;
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
})();
