(() => {
  const el = (id) => document.getElementById(id);

  const clockEl = el('clock');
  const headlineEl = el('headline');
  const countdownEl = el('countdown');
  const progressFillEl = el('progressFill');
  const labelInput = el('labelInput');
  const minutesInput = el('minutesInput');
  const cameraToggle = el('cameraToggle');
  const lookDownToggle = el('lookDownToggle');
  const statCamera = el('statCamera');
  const statState = el('statState');
  const statViolations = el('statViolations');
  const thumb = el('thumb');
  const thumbPlaceholder = el('thumbPlaceholder');
  const cameraStatus = el('cameraStatus');
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
  let defaultsApplied = false;
  let miniShown = false;

  function setToggle(btn, on) { btn.dataset.on = String(on); }

  cameraToggle.addEventListener('click', () => {
    cameraOn = !cameraOn;
    setToggle(cameraToggle, cameraOn);
    // Look-down detection is meaningless without the camera.
    if (!cameraOn && lookDownOn) {
      lookDownOn = false;
      setToggle(lookDownToggle, false);
    }
    lookDownToggle.disabled = !cameraOn;
    lookDownToggle.style.opacity = cameraOn ? '1' : '0.35';
  });

  lookDownToggle.addEventListener('click', () => {
    if (!cameraOn) return;
    lookDownOn = !lookDownOn;
    setToggle(lookDownToggle, lookDownOn);
  });

  startBtn.addEventListener('click', async () => {
    if (!apiReady) return;
    const minutes = parseFloat(minutesInput.value);
    startBtn.disabled = true;
    try {
      const res = await window.pywebview.api.start(
        labelInput.value, isNaN(minutes) ? null : minutes, lookDownOn, cameraOn
      );
      if (res && res.error) {
        statusMessage.textContent = res.error;
        startBtn.disabled = false;
        return;
      }
      await window.pywebview.api.enter_mini();
      miniShown = true;
    } catch (e) {
      startBtn.disabled = false;
    }
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
