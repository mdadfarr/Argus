(() => {
  const el = (id) => document.getElementById(id);

  const clockEl = el('clock');
  const headlineEl = el('headline');
  const substateEl = el('substate');
  const countdownEl = el('countdown');
  const progressFillEl = el('progressFill');
  const labelInput = el('labelInput');
  const minutesInput = el('minutesInput');
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
  const degradedMessage = el('degradedMessage');

  let apiReady = false;
  let lookDownOn = false;
  let defaultsApplied = false;

  lookDownToggle.addEventListener('click', () => {
    lookDownOn = !lookDownOn;
    lookDownToggle.dataset.on = String(lookDownOn);
  });

  startBtn.addEventListener('click', async () => {
    if (!apiReady) return;
    const label = labelInput.value;
    const minutes = parseFloat(minutesInput.value);
    startBtn.disabled = true;
    try {
      const res = await window.pywebview.api.start(label, isNaN(minutes) ? null : minutes, lookDownOn);
      if (res && res.error) {
        statusMessage.textContent = res.error;
        startBtn.disabled = false;
      }
    } catch (e) {
      startBtn.disabled = false;
    }
  });

  pauseBtn.addEventListener('click', () => {
    if (!apiReady) return;
    window.pywebview.api.pause_resume();
  });

  stopBtn.addEventListener('click', () => {
    if (!apiReady) return;
    window.pywebview.api.stop();
  });

  falsePositiveBtn.addEventListener('click', () => {
    if (!apiReady) return;
    window.pywebview.api.false_positive();
  });

  function applyState(s) {
    if (!s) return;

    clockEl.textContent = s.clock || '--:--:--';
    headlineEl.textContent = s.headline || '';
    substateEl.textContent = s.substate || '';
    countdownEl.textContent = s.countdown || '--:--';

    progressFillEl.style.width = `${Math.max(0, Math.min(1, s.progress_frac || 0)) * 100}%`;

    statCamera.textContent = s.camera_name || 'not open';
    statState.textContent = (s.state || 'idle').toLowerCase();
    statViolations.textContent = s.violations != null ? String(s.violations) : '0';
    cameraStatus.textContent = s.camera_status_text || 'not connected';

    if (s.thumbnail_b64) {
      thumb.src = s.thumbnail_b64;
      thumb.classList.add('visible');
      thumbPlaceholder.style.display = 'none';
    } else {
      thumb.classList.remove('visible');
      thumbPlaceholder.style.display = 'flex';
    }

    statusMessage.textContent = s.status_message || '';

    if (s.degraded) {
      degradedMessage.textContent = s.degraded;
      degradedMessage.classList.add('visible');
    } else {
      degradedMessage.classList.remove('visible');
    }

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
  }

  async function poll() {
    if (apiReady) {
      try {
        const s = await window.pywebview.api.get_state();
        applyState(s);
      } catch (e) {
        // backend not ready yet, ignore
      }
    }
    setTimeout(poll, 400);
  }

  function tickClock() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    clockEl.textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    setTimeout(tickClock, 1000);
  }

  window.addEventListener('pywebviewready', () => {
    apiReady = true;
  });

  tickClock();
  poll();
})();
