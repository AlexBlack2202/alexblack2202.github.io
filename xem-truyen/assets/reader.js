(() => {
  'use strict';
  const read = (key, fallback) => { try { return localStorage.getItem(key) || fallback; } catch (_) { return fallback; } };
  const store = (key, value) => { try { localStorage.setItem(key, value); } catch (_) {} };
  const content = document.getElementById('reader-content');
  if (content) {
    let size = Number(read('readerSize', '20')) || 20;
    const apply = () => {
      size = Math.max(16, Math.min(28, size));
      document.documentElement.style.setProperty('--reader-size', `${size}px`);
      document.documentElement.style.setProperty('--reader-size-mobile', `${Math.max(16, size - 2)}px`);
      store('readerSize', String(size));
    };
    const focus = document.getElementById('focus-toggle');
    document.body.classList.toggle('focus-mode', read('readerFocus', '0') === '1');
    focus.setAttribute('aria-pressed', String(document.body.classList.contains('focus-mode')));
    document.querySelector('.reader-tools').hidden = false;
    document.getElementById('font-down').addEventListener('click', () => { size -= 1; apply(); });
    document.getElementById('font-up').addEventListener('click', () => { size += 1; apply(); });
    focus.addEventListener('click', () => {
      const enabled = document.body.classList.toggle('focus-mode');
      focus.setAttribute('aria-pressed', String(enabled));
      store('readerFocus', enabled ? '1' : '0');
    });
    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey ||
          event.target.closest('input, textarea, select, button, a, [contenteditable]')) return;
      const direction = event.key === 'ArrowLeft' ? 'prev' : event.key === 'ArrowRight' ? 'next' : '';
      const link = direction && document.querySelector(`a[rel="${direction}"]`);
      if (link) { event.preventDefault(); location.assign(link.href); }
    });
    apply();
  }
  const search = document.getElementById('chapter-search');
  if (search) {
    search.closest('label').hidden = false;
    const normalize = value => value.toLowerCase().replace(/đ/g, 'd').normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    const rows = [...document.querySelectorAll('#chapter-list li')];
    const status = document.getElementById('chapter-status');
    search.addEventListener('input', () => {
      const query = normalize(search.value.trim());
      let count = 0;
      for (const row of rows) { row.hidden = !normalize(row.textContent).includes(query); if (!row.hidden) count++; }
      status.textContent = query ? `${count} chương phù hợp` : `${rows.length} chương hiện có`;
    });
  }
})();
