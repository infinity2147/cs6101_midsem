/* Opportunity Radar (F4): view tabs, list filters, current-month calendar and
   the home page "Open now" strip. Reads window.RADAR from _includes/radar-data.html.
   The current month comes from the reader's clock, so it is right on a static site. */
(function () {
  'use strict';

  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December'];
  var now = new Date().getMonth();
  var data = window.RADAR || [];

  function idx(name) { return MONTHS.indexOf(name); }

  // Is month m inside [opens, closes]? Handles windows that wrap the year (Nov -> Jan).
  function isOpen(entry, m) {
    var o = idx(entry.opens), c = idx(entry.closes);
    if (o < 0) return false;
    if (c < 0) c = o;
    return o <= c ? (m >= o && m <= c) : (m >= o || m <= c);
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  /* ---------- home: Open now strip ---------- */
  var strip = document.querySelector('[data-open-now]');
  if (strip) {
    var label = document.querySelector('[data-month-label]');
    if (label) label.textContent = '· ' + MONTHS[now];
    var open = data.filter(function (e) { return isOpen(e, now); });
    var next = (now + 1) % 12;
    var soon = data.filter(function (e) { return idx(e.opens) === next; });
    // Closing this month first: those are the urgent ones.
    open.sort(function (a, b) {
      return (idx(b.closes) === now) - (idx(a.closes) === now) || a.name.localeCompare(b.name);
    });
    strip.textContent = '';
    open.forEach(function (e) {
      var card = el('article', 'card open-card');
      card.appendChild(el('span', 'chip chip-type', e.kindLabel));
      var h = el('h3');
      var a = el('a', null, e.name);
      a.href = e.url;
      h.appendChild(a);
      card.appendChild(h);
      var when = idx(e.closes) === now
        ? el('p', 'small closing', 'Usually closes this month')
        : el('p', 'small muted', 'Open until ' + (e.closes || e.opens));
      card.appendChild(when);
      if (!e.verified) card.appendChild(el('p', 'small muted', 'Months not yet verified'));
      strip.appendChild(card);
    });
    if (!open.length) {
      strip.appendChild(el('p', 'muted', 'Nothing on the Radar is usually open in ' + MONTHS[now] + '.'));
    }
    if (soon.length) {
      var s = el('article', 'card open-card');
      s.appendChild(el('span', 'chip', 'Opens in ' + MONTHS[next]));
      var ul = el('ul', 'plain small');
      soon.forEach(function (e) {
        var li = el('li');
        var a = el('a', null, e.name);
        a.href = e.url;
        li.appendChild(a);
        ul.appendChild(li);
      });
      s.appendChild(ul);
      strip.appendChild(s);
    }
  }

  /* ---------- radar page: tabs ---------- */
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tabs [data-view]'));
  if (tabs.length) {
    var panels = document.querySelectorAll('[data-panel]');
    var show = function (view, push) {
      tabs.forEach(function (t) { t.setAttribute('aria-selected', t.dataset.view === view ? 'true' : 'false'); });
      Array.prototype.forEach.call(panels, function (p) { p.hidden = p.dataset.panel !== view; });
      if (push && history.replaceState) {
        var q = new URLSearchParams(location.search);
        if (view === 'list') q.delete('view'); else q.set('view', view);
        var qs = q.toString();
        try { history.replaceState(null, '', location.pathname + (qs ? '?' + qs : '')); } catch (e) {}
      }
      if (view === 'calendar') {
        var cur = document.querySelector('.month.current');
        if (cur && cur.scrollIntoView && window.innerWidth < 640) cur.scrollIntoView({ block: 'start' });
      }
    };
    tabs.forEach(function (t) {
      t.addEventListener('click', function () { show(t.dataset.view, true); });
      t.addEventListener('keydown', function (e) {
        var i = tabs.indexOf(t);
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          var n = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
          n.focus();
          show(n.dataset.view, true);
        }
      });
    });
    var initial = new URLSearchParams(location.search).get('view') || location.hash.slice(1);
    show(['calendar', 'roadmap'].indexOf(initial) !== -1 ? initial : 'list', false);

    // Calendar: highlight the current month and start the grid from it.
    var cal = document.querySelector('.calendar');
    if (cal) {
      var months = Array.prototype.slice.call(cal.querySelectorAll('.month'));
      months.forEach(function (m) {
        if (+m.dataset.month === now) m.classList.add('current');
      });
      months.slice(0, now).forEach(function (m) { cal.appendChild(m); });
    }
  }

  /* ---------- radar page: list filters ---------- */
  var bar = document.querySelector('[data-radar-filter]');
  if (bar) {
    var cards = Array.prototype.slice.call(document.querySelectorAll('.radar-card'));
    var kind = bar.querySelector('[data-dim="kind"]');
    var stage = bar.querySelector('[data-dim="stage"]');
    var count = bar.querySelector('[data-count]');
    var empty = document.querySelector('[data-panel="list"] [data-empty]');
    var q = new URLSearchParams(location.search);
    if (q.get('kind')) kind.value = q.get('kind');
    if (q.get('stage')) stage.value = q.get('stage');
    var apply = function () {
      var n = 0;
      cards.forEach(function (c) {
        var ok = (!kind.value || c.dataset.kind === kind.value) &&
          (!stage.value || (c.dataset.stage || '').split('|').indexOf(stage.value) !== -1);
        c.hidden = !ok;
        if (ok) n++;
      });
      count.textContent = n + (n === 1 ? ' opportunity' : ' opportunities');
      empty.hidden = n > 0;
    };
    kind.addEventListener('change', apply);
    stage.addEventListener('change', apply);
    apply();
  }
})();
