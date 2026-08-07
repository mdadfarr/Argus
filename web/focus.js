// Shared by focus.html (the dot monitor) and focus-blank.html (all the
// others). The only difference between them is the has-dot class in the
// markup, so this file must not assume a dot exists.
(() => {
  const body = document.body;

  let apiReady = false;
  window.addEventListener('pywebviewready', () => { apiReady = true; });

  // ---- escape hatch ----
  // Deliberately undiscoverable and slow. A tap of Esc must not end a block,
  // or the first dull minute becomes an exit; three seconds held is long
  // enough to be a decision rather than a flinch.
  const HOLD_MS = 3000;
  let holdTimer = null;

  function cancelHold() {
    if (holdTimer !== null) { clearTimeout(holdTimer); holdTimer = null; }
  }

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    e.preventDefault();
    // Key repeat fires keydown continuously. Only the first may arm the
    // timer -- otherwise every repeat would restart it and the hold could
    // never complete.
    if (holdTimer !== null) return;
    holdTimer = setTimeout(() => {
      holdTimer = null;
      if (apiReady) window.pywebview.api.focus_cancel();
    }, HOLD_MS);
  });

  document.addEventListener('keyup', (e) => {
    if (e.key === 'Escape') cancelHold();
  });

  // A window that loses focus mid-hold never receives the keyup, which would
  // leave the timer armed and end the block seconds later out of nowhere.
  window.addEventListener('blur', cancelHold);

  // ---- violation flash + teardown ----
  // Teardown is driven from here rather than from the main window because the
  // main window spends the whole block fully covered by these ones, and macOS
  // throttles timers in an occluded WKWebView -- its poll cannot be relied on
  // to notice the block ending. focus_exit() destroys the windows on the main
  // thread, well after this bridge call has returned.
  let exiting = false;

  async function poll() {
    if (apiReady) {
      try {
        const s = await window.pywebview.api.focus_state();
        body.classList.toggle('violation', !!(s && s.in_violation));
        if (s && !s.active && !exiting) {
          exiting = true;
          window.pywebview.api.focus_exit();
        }
      } catch (e) { /* backend busy; keep the last rendering */ }
    }
    setTimeout(poll, 250);
  }
  poll();
})();
