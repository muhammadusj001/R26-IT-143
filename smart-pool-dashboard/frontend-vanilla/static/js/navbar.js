// Shared navbar behaviour (every page): user menu, mobile panel, and the
// helpers other scripts call to keep the plan badge / connection live.
(function () {
  const userBtn = document.getElementById('navUserBtn');
  const userMenu = document.getElementById('navUserMenu');
  const burger = document.getElementById('navBurger');
  const panel = document.getElementById('navPanel');

  function setOpen(button, popup, open) {
    if (!button || !popup) return;
    button.setAttribute('aria-expanded', String(open));
    popup.hidden = !open;
    button.closest('.navbar').classList.toggle(popup === panel ? 'panel-open' : 'menu-open', open);
  }
  const isOpen = (button) => button && button.getAttribute('aria-expanded') === 'true';

  if (userBtn) {
    userBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const open = !isOpen(userBtn);
      setOpen(userBtn, userMenu, open);
      if (open) { const first = userMenu.querySelector('[role="menuitem"]'); if (first) first.focus(); }
    });
    // Arrow keys move between menu items
    userMenu.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
      e.preventDefault();
      const items = [...userMenu.querySelectorAll('[role="menuitem"]')];
      const i = items.indexOf(document.activeElement);
      items[(i + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length].focus();
    });
  }
  if (burger) {
    burger.addEventListener('click', (e) => {
      e.stopPropagation();
      setOpen(burger, panel, !isOpen(burger));
    });
  }

  // Close on outside click or Esc
  document.addEventListener('click', (e) => {
    if (isOpen(userBtn) && !userMenu.contains(e.target)) setOpen(userBtn, userMenu, false);
    if (isOpen(burger) && !panel.contains(e.target)) setOpen(burger, panel, false);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (isOpen(userBtn)) { setOpen(userBtn, userMenu, false); userBtn.focus(); }
    if (isOpen(burger)) { setOpen(burger, panel, false); burger.focus(); }
  });
  // The panel only exists below 900px; close it when the window grows
  window.matchMedia('(min-width: 900px)').addEventListener('change', (mq) => {
    if (mq.matches) setOpen(burger, panel, false);
  });

  // ── Live updates used by dashboard.js and the payment page ──
  window.setNavPlan = function (key, name) {
    for (const badge of document.querySelectorAll('.js-plan-badge')) {
      badge.textContent = (name || key).toUpperCase();
      badge.className = 'pill plan-pill js-plan-badge plan-' + key;
    }
    for (const up of document.querySelectorAll('.js-plan-upgrade')) up.hidden = key !== 'basic';
  };
  window.setNavConnection = function (ok) {
    for (const dot of document.querySelectorAll('.js-conn-dot')) {
      dot.className = 'status-dot js-conn-dot ' + (ok ? 'connected' : 'disconnected');
    }
    for (const text of document.querySelectorAll('.js-conn-text')) {
      text.textContent = ok ? 'Live' : 'Offline';
      text.className = 'js-conn-text ' + (ok ? 'conn-live' : 'conn-offline');
    }
  };
})();
