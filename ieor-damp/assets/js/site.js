/* IEOR DAMP site script: menu, theme switch, post-stream filters (F1), Ask us form (F6).
   No dependencies. Every feature degrades to plain links/lists without JS. */
(function () {
  'use strict';

  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { return null; }
  }

  /* ---------- mobile menu ---------- */
  var toggle = document.querySelector('.nav-toggle');
  var nav = document.getElementById('site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  /* ---------- theme ---------- */
  var themeBtn = document.querySelector('.theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var root = document.documentElement;
      var current = root.getAttribute('data-theme') ||
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      store('theme', next);
    });
  }

  /* ---------- post stream filters (F1) ---------- */
  var DIMS = ['type', 'domain', 'year', 'program'];

  function initStream(stream) {
    var cards = Array.prototype.slice.call(stream.querySelectorAll('.post-card'));
    var bar = stream.querySelector('[data-filter-bar]');
    var selects = {};
    var state = { tag: '' };
    var params = new URLSearchParams(window.location.search);

    function cardTags(card) {
      return (card.dataset.tags || '').split('|').filter(Boolean);
    }

    // Keep only options that some card in this section actually has.
    DIMS.forEach(function (dim) {
      var sel = bar.querySelector('select[data-dim="' + dim + '"]');
      if (!sel) return;
      var present = {};
      cards.forEach(function (c) { if (c.dataset[dim]) present[c.dataset[dim]] = true; });
      Array.prototype.slice.call(sel.options).forEach(function (o) {
        if (o.value && !present[o.value]) o.remove();
      });
      if (sel.options.length <= 2) { sel.closest('label').hidden = true; return; }
      selects[dim] = sel;
      state[dim] = params.get(dim) || '';
      if (state[dim] && !present[state[dim]]) state[dim] = '';
      sel.value = state[dim];
      sel.addEventListener('change', function () { state[dim] = sel.value; apply(true); });
    });
    state.tag = params.get('tag') || '';

    // Tag cloud: free tags, organisations and course codes in this section.
    var cloud = stream.querySelector('[data-tag-cloud]');
    var counts = {};
    cards.forEach(function (c) { cardTags(c).forEach(function (t) { counts[t] = (counts[t] || 0) + 1; }); });
    var tags = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a] || a.localeCompare(b); }).slice(0, 24);
    if (cloud) {
      if (!tags.length) cloud.hidden = true;
      tags.forEach(function (t) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip';
        b.dataset.tag = t;
        b.textContent = t + ' (' + counts[t] + ')';
        b.addEventListener('click', function () {
          state.tag = state.tag === t ? '' : t;
          apply(true);
        });
        cloud.appendChild(b);
      });
    }

    var countEl = bar.querySelector('[data-count]');
    var clearBtn = bar.querySelector('[data-clear]');
    var activeTag = bar.querySelector('[data-active-tag]');
    var empty = stream.querySelector('[data-empty]');
    clearBtn.addEventListener('click', function () {
      Object.keys(selects).forEach(function (d) { state[d] = ''; selects[d].value = ''; });
      state.tag = '';
      apply(true);
    });

    function apply(push) {
      var shown = 0;
      cards.forEach(function (c) {
        var ok = Object.keys(selects).every(function (d) { return !state[d] || c.dataset[d] === state[d]; });
        if (ok && state.tag) ok = cardTags(c).indexOf(state.tag) !== -1;
        c.hidden = !ok;
        if (ok) shown++;
      });
      if (cloud) {
        Array.prototype.forEach.call(cloud.children, function (b) {
          b.setAttribute('aria-pressed', b.dataset.tag === state.tag ? 'true' : 'false');
        });
      }
      var active = state.tag || Object.keys(selects).some(function (d) { return state[d]; });
      countEl.textContent = shown + (shown === 1 ? ' post' : ' posts');
      clearBtn.hidden = !active;
      activeTag.hidden = !state.tag;
      activeTag.textContent = state.tag ? 'Tag: ' + state.tag : '';
      if (empty) empty.hidden = shown > 0;
      if (push && window.history && history.replaceState) {
        var q = new URLSearchParams(window.location.search);
        Object.keys(selects).concat('tag').forEach(function (d) {
          if (state[d]) q.set(d, state[d]); else q.delete(d);
        });
        var qs = q.toString();
        try { history.replaceState(null, '', window.location.pathname + (qs ? '?' + qs : '') + window.location.hash); } catch (e) {}
      }
    }
    apply(false);
  }

  Array.prototype.forEach.call(document.querySelectorAll('[data-stream]'), initStream);

  /* ---------- Ask us form (F6) ---------- */
  Array.prototype.forEach.call(document.querySelectorAll('.ask-form'), function (form) {
    var anon = form.querySelector('[data-anon]');
    var who = form.querySelector('[data-who]');
    var done = form.querySelector('[data-done]');
    function syncAnon() {
      who.hidden = anon.checked;
      Array.prototype.forEach.call(who.querySelectorAll('input'), function (i) {
        i.disabled = anon.checked;   // disabled fields are not submitted
      });
    }
    anon.addEventListener('change', syncAnon);
    syncAnon();

    form.addEventListener('submit', function (e) {
      var configured = form.dataset.configured === '1';
      if (!configured) {
        e.preventDefault();
        var fields = form.querySelectorAll('input[type="text"]:not([disabled]), textarea');
        var lines = [];
        Array.prototype.forEach.call(fields, function (f) {
          var label = f.closest('label').querySelector('span').textContent;
          if (f.value.trim()) lines.push(label + ': ' + f.value.trim());
        });
        if (anon.checked) lines.push('(Sent from the anonymous option; for full anonymity use the Google Form once it is set up.)');
        window.location.href = 'mailto:' + form.dataset.mailto +
          '?subject=' + encodeURIComponent('Question for IEOR DAMP') +
          '&body=' + encodeURIComponent(lines.join('\n\n'));
        return;
      }
      // Configured: the browser posts to the Google Form inside a hidden iframe.
      window.setTimeout(function () {
        form.reset();
        syncAnon();
        done.hidden = false;
      }, 300);
    });
  });
})();
