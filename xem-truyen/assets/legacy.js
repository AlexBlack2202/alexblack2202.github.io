(() => {
  'use strict';
  const params = new URLSearchParams(location.search);
  if (!['type', 'story', 'chapter'].some(key => params.has(key))) return;
  const script = document.currentScript;
  const base = script.dataset.base;
  const fail = () => {
    document.title = 'Không tìm thấy nội dung — Phạm Duy Tùng';
    const main = document.getElementById('main-content');
    main.innerHTML = '<section class="shell notice"><h1>Không tìm thấy nội dung</h1><p>Liên kết truyện hoặc chương này không còn hợp lệ.</p><a class="button button-primary">Trở về thư viện</a></section>';
    main.querySelector('a').href = base;
    document.documentElement.classList.remove('legacy-pending');
  };
  fetch(base + 'assets/legacy-routes.json', {credentials: 'same-origin'})
    .then(response => { if (!response.ok) throw new Error('Manifest unavailable'); return response.json(); })
    .then(routes => {
      const type = params.get('type');
      const story = params.get('story');
      const chapter = params.get('chapter');
      let path;
      if (type === 'story' && routes.stories[story]) path = routes.stories[story] + '/';
      else if (type === 'chapter') {
        const found = routes.chapters[chapter];
        if (found && (!story || story === found[0])) path = routes.stories[found[0]] + '/chuong-' + found[1] + '/';
      }
      if (!path) { fail(); return; }
      location.replace(base + path);
    }).catch(fail);
})();
