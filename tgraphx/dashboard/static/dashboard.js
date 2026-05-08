/* TGraphX Dashboard — Vanilla JS frontend
   No external dependencies. Runs in any modern browser.
   All timestamps treated as UTC; displayed in viewer's local timezone. */
'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// Configuration (defaults; /api/config may override at startup)
// ─────────────────────────────────────────────────────────────────────────────
const CFG = {
  pollMs:      2000,
  smoothAlpha: 0.3,
  maxNodes:    200,
  maxEdges:    1000,
  forceIter:   80,
  etaMinPts:   5,
  staleAfterS: 30,   // show "stale data" banner after this many seconds
  // Range/window options for chart truncation:
  ranges: [
    { id: 'all',  label: 'All',       n: 0    },
    { id: '100',  label: 'Last 100',  n: 100  },
    { id: '500',  label: 'Last 500',  n: 500  },
    { id: '1000', label: 'Last 1000', n: 1000 },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let S = {
  sec:            'overview',
  status:         null,
  metrics:        null,
  hardware:       null,
  metadata:       null,
  graph:          null,
  graphStats:     null,
  config:         null,
  runs:           null,     // {mode, runs, capped} from /api/runs
  activeRun:      null,     // currently selected run name (multi-run mode)
  smooth:         false,
  paused:         false,
  range:          'all',
  palette:        localStorage.getItem('tgx-palette') || 'default',
  tvMode:         false,
  pollTimer:      null,
  lastFetch:      null,
  latestRowIndex: -1,       // -1 = never fetched, do full load next time
  snapshotMode:   false,    // true when running from an offline HTML export
};

const SECTIONS = [
  { id:'overview', label:'Overview',  icon:'▤'  },
  { id:'metrics',  label:'Metrics',   icon:'⬡'  },
  { id:'graph',    label:'Graph',     icon:'◈'  },
  { id:'mining',   label:'Mining',    icon:'⛏'  },
  { id:'hardware', label:'Hardware',  icon:'⚙'  },
  { id:'logs',     label:'Logs',      icon:'≡'  },
  { id:'config',   label:'Config',    icon:'❐'  },
  { id:'tools',    label:'Tools',     icon:'⚒'  },
  { id:'about',    label:'About',     icon:'ℹ'  },
];

// Chart colors are taken from CSS custom properties so the palette toggle
// (default vs. color-blind-safe Okabe-Ito) re-skins all charts at once.
const SERIES_KEYS = [
  'series-1', 'series-2', 'series-3', 'series-4',
  'series-5', 'series-6', 'series-7', 'series-8',
];
function seriesColor(key) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue('--' + key).trim() || '#06b6d4';
}

const CHART_META = {
  train_loss:      { label:'Train Loss',      key:'series-1' },
  val_loss:        { label:'Validation Loss', key:'series-2' },
  loss:            { label:'Loss',            key:'series-1' },
  accuracy:        { label:'Accuracy',        key:'series-3' },
  acc:             { label:'Accuracy',        key:'series-3' },
  learning_rate:   { label:'Learning Rate',   key:'series-4' },
  lr:              { label:'Learning Rate',   key:'series-4' },
  grad_norm:       { label:'Grad Norm',       key:'series-5' },
  samples_per_sec: { label:'Samples/s',       key:'series-6' },
  graphs_per_sec:  { label:'Graphs/s',        key:'series-6' },
  steps_per_sec:   { label:'Steps/s',         key:'series-6' },
};
function metaFor(name) {
  const m = CHART_META[name];
  if (m) return { label: m.label, color: seriesColor(m.key) };
  // Fall back to a deterministic series color so unknown metrics still
  // get a stable, palette-aware color across renders.
  let h = 0; for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return { label: name, color: seriesColor(SERIES_KEYS[Math.abs(h) % SERIES_KEYS.length]) };
}

const SKIP_COLS = new Set(['epoch','step','timestamp','time']);

// ─────────────────────────────────────────────────────────────────────────────
// HTML escaping — every user-controlled string flows through here before
// reaching innerHTML.  Prevents accidental injection from run names, builder
// names, metric keys, file paths, and run_metadata values.
// ─────────────────────────────────────────────────────────────────────────────
function esc(v) {
  if (v == null) return '';
  return String(v)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ─────────────────────────────────────────────────────────────────────────────
// Time utilities
// ─────────────────────────────────────────────────────────────────────────────
const T = {
  local(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}); }
    catch { return iso; }
  },
  localFull(iso) {
    if (!iso) return '—';
    try { return new Date(iso).toLocaleString(); }
    catch { return iso; }
  },
  ago(iso) {
    if (!iso) return null;
    try {
      const s = (Date.now() - new Date(iso).getTime()) / 1000;
      if (s < 5)    return 'just now';
      if (s < 60)   return Math.round(s) + 's ago';
      if (s < 3600) return Math.round(s/60) + 'm ago';
      return Math.round(s/3600) + 'h ago';
    } catch { return null; }
  },
  elapsed(sec) {
    if (!isFinite(sec) || sec < 0) return '—';
    const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = Math.floor(sec%60);
    return [h,m,s].map(v=>String(v).padStart(2,'0')).join(':');
  },
  now() {
    return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  },
  elapsedSince(iso) {
    if (!iso) return null;
    try { return (Date.now() - new Date(iso).getTime()) / 1000; }
    catch { return null; }
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Number formatting
// ─────────────────────────────────────────────────────────────────────────────
function fmt(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0)      return '0';
  if (a >= 1e6)     return (v/1e6).toFixed(2) + 'M';
  if (a >= 1e3)     return (v/1e3).toFixed(2) + 'k';
  if (a >= 10)      return v.toFixed(3);
  if (a >= 1)       return v.toFixed(4);
  if (a >= 1e-3)    return v.toFixed(5);
  return v.toExponential(3);
}

// ─────────────────────────────────────────────────────────────────────────────
// API client
// ─────────────────────────────────────────────────────────────────────────────
const API = {
  async get(ep) {
    try {
      const r = await fetch(ep, {cache:'no-store'});
      return r.ok ? await r.json() : null;
    } catch { return null; }
  },
  status()           { return this.get('/api/status'); },
  metrics(opts = {}) {
    const parts = [];
    if (opts.sinceRow != null) parts.push(`since_row=${opts.sinceRow}`);
    if (opts.run)              parts.push(`run=${encodeURIComponent(opts.run)}`);
    const qs = parts.length ? '?' + parts.join('&') : '';
    return this.get('/api/metrics' + qs);
  },
  hardware()         { return this.get('/api/hardware'); },
  metadata()         { return this.get('/api/metadata'); },
  graph()            { return this.get('/api/graph'); },
  graphStats()       { return this.get('/api/graph_stats'); },
  runs()             { return this.get('/api/runs'); },
  config()           { return this.get('/api/config'); },
};

// ─────────────────────────────────────────────────────────────────────────────
// DOM helpers
// ─────────────────────────────────────────────────────────────────────────────
function el(id)  { return document.getElementById(id); }
function set(id, html) { const e = el(id); if (e) e.innerHTML = html; }
function qs(sel) { return document.querySelector(sel); }

function card(title, valueHtml, sub='', cls='') {
  return `<div class="card ${cls}">
    <div class="card-title">${title}</div>
    <div class="card-value">${valueHtml}</div>
    ${sub ? `<div class="card-sub">${sub}</div>` : ''}
  </div>`;
}

function emptyState(msg, hint='') {
  return `<div class="empty-state"><strong>${msg}</strong>${hint}</div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// SVG Chart renderer
// ─────────────────────────────────────────────────────────────────────────────
class SvgChart {
  constructor(container, opts = {}) {
    this.el = typeof container === 'string' ? el(container) : container;
    this.color  = opts.color  || '#06b6d4';
    this.label  = opts.label  || '';
    this.height = opts.height || 180;
    this.xLabel = opts.xLabel || 'Epoch';
    this.PAD = { t:22, r:18, b:40, l:58 };
  }

  render(rawData, smooth=false) {
    if (!rawData || rawData.length === 0) {
      this.el.innerHTML = `<div class="chart-empty">No data yet</div>`;
      return;
    }
    let data = rawData.filter(d => isFinite(d.y));
    if (data.length === 0) { this.el.innerHTML = `<div class="chart-empty">No finite values</div>`; return; }
    if (smooth && data.length > 2) data = this._ema(data);

    const W = (this.el.clientWidth || 540);
    const H = this.height;
    const {t,r,b,l} = this.PAD;
    const iW = W-l-r, iH = H-t-b;

    const xs = data.map(d=>d.x), ys = data.map(d=>d.y);
    const x0=Math.min(...xs), x1=Math.max(...xs);
    const y0=Math.min(...ys), y1=Math.max(...ys);
    const xR = x1-x0 || 1;
    const yPad = (y1-y0)*0.12 || Math.abs(y0)*0.12 || 0.1;
    const yLo=y0-yPad, yHi=y1+yPad, yR=yHi-yLo;

    const sx = x => l + (x-x0)/xR*iW;
    const sy = y => t + (1-(y-yLo)/yR)*iH;

    const pts = data.map(d=>`${sx(d.x).toFixed(1)},${sy(d.y).toFixed(1)}`).join(' ');
    const areaClose = `${sx(xs[xs.length-1]).toFixed(1)},${(t+iH).toFixed(1)} ${sx(x0).toFixed(1)},${(t+iH).toFixed(1)}`;

    const xTks = this._ticks(x0,x1,5);
    const yTks = this._ticks(yLo,yHi,4);

    const lastY = ys[ys.length-1];
    const lastX = xs[xs.length-1];

    this.el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:${H}px;display:block">
  <rect x="${l}" y="${t}" width="${iW}" height="${iH}" fill="none"/>
  ${yTks.map(v=>`<line x1="${l}" y1="${sy(v).toFixed(1)}" x2="${(l+iW).toFixed(1)}" y2="${sy(v).toFixed(1)}"
    stroke="var(--grid)" stroke-width="0.8" stroke-dasharray="3,3"/>`).join('')}
  <polygon points="${pts} ${areaClose}" fill="${this.color}" fill-opacity="0.09" stroke="none"/>
  <polyline points="${pts}" fill="none" stroke="${this.color}" stroke-width="2"
    stroke-linejoin="round" stroke-linecap="round"/>
  <line x1="${l}" y1="${t}" x2="${l}" y2="${t+iH}" stroke="var(--border)" stroke-width="1"/>
  <line x1="${l}" y1="${t+iH}" x2="${l+iW}" y2="${t+iH}" stroke="var(--border)" stroke-width="1"/>
  ${xTks.map(v=>`<line x1="${sx(v).toFixed(1)}" y1="${t+iH}" x2="${sx(v).toFixed(1)}" y2="${t+iH+4}"
      stroke="var(--border)" stroke-width="1"/>
    <text x="${sx(v).toFixed(1)}" y="${t+iH+16}" text-anchor="middle"
      font-size="10" fill="var(--text3)">${Math.round(v)}</text>`).join('')}
  ${yTks.map(v=>`<line x1="${l-4}" y1="${sy(v).toFixed(1)}" x2="${l}" y2="${sy(v).toFixed(1)}"
      stroke="var(--border)" stroke-width="1"/>
    <text x="${l-7}" y="${(sy(v)+3.5).toFixed(1)}" text-anchor="end"
      font-size="10" fill="var(--text3)">${this._fmtY(v)}</text>`).join('')}
  <text x="${l}" y="${t-6}" font-size="11" font-weight="600" fill="var(--text2)">${this.label}</text>
  <text x="${l+iW}" y="${t-6}" font-size="10" text-anchor="end" fill="${this.color}" font-weight="700">${fmt(lastY)}</text>
  <circle cx="${sx(lastX).toFixed(1)}" cy="${sy(lastY).toFixed(1)}" r="3.5" fill="${this.color}"/>
  <text x="${(l+iW/2).toFixed(1)}" y="${H-3}" font-size="9" text-anchor="middle" fill="var(--text3)">${this.xLabel}</text>
</svg>`;
    // Attach hover tooltip after inserting SVG into DOM.
    const svgEl = this.el.querySelector('svg');
    if (svgEl) {
      Tooltip.attach(svgEl, data, this.label, this.color,
                     {t, r, b, l}, W, H);
    }
  }

  _ema(data) {
    const a = CFG.smoothAlpha;
    const out = [data[0]];
    for (let i=1; i<data.length; i++)
      out.push({...data[i], y: a*data[i].y + (1-a)*out[i-1].y});
    return out;
  }
  _ticks(lo, hi, n) {
    const range = hi-lo, step = this._niceStep(range/n);
    const start = Math.ceil(lo/step)*step;
    const tks = [];
    for (let v=start; v<=hi+step*0.01; v+=step) {
      const rv = parseFloat(v.toPrecision(10));
      if (rv >= lo-step*0.01 && rv <= hi+step*0.01) tks.push(rv);
    }
    return tks;
  }
  _niceStep(raw) {
    const m = Math.pow(10, Math.floor(Math.log10(Math.abs(raw)||1)));
    const f = raw/m;
    return f<1.5?m : f<3.5?2*m : f<7.5?5*m : 10*m;
  }
  _fmtY(v) {
    const a=Math.abs(v);
    if (a>=100)  return v.toFixed(1);
    if (a>=1)    return v.toFixed(3);
    if (a>=0.01) return v.toFixed(4);
    return v.toExponential(2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ETA estimation
// ─────────────────────────────────────────────────────────────────────────────
function estimateETA(metrics, totalEpochs) {
  if (!metrics || !totalEpochs) return null;
  const {headers, rows} = metrics;
  if (!rows || rows.length < CFG.etaMinPts) return null;

  const epIdx = headers.indexOf('epoch');
  const tsIdx = headers.indexOf('timestamp');
  if (epIdx < 0 || tsIdx < 0) return null;

  const first = rows[0], last = rows[rows.length-1];
  const t0 = new Date(first[tsIdx]), t1 = new Date(last[tsIdx]);
  if (isNaN(t0) || isNaN(t1)) return null;
  const elapsed = (t1-t0)/1000;
  const epDone  = last[epIdx] - first[epIdx];
  if (epDone <= 0) return null;
  const remaining = totalEpochs - last[epIdx];
  if (remaining <= 0) return 0;
  return (elapsed/epDone)*remaining;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build chart data from metrics table
// ─────────────────────────────────────────────────────────────────────────────
function metricsToSeries(metrics) {
  if (!metrics || !metrics.headers || !metrics.rows) return {};
  const hdrs = metrics.headers;
  const xIdx = hdrs.indexOf('epoch') >= 0 ? hdrs.indexOf('epoch') : hdrs.indexOf('step');
  if (xIdx < 0) return {};
  const series = {};
  hdrs.forEach((h, i) => {
    if (i === xIdx || SKIP_COLS.has(h)) return;
    series[h] = metrics.rows
      .map(r => ({ x: r[xIdx], y: r[i] }))
      .filter(d => isFinite(d.x) && isFinite(d.y));
  });
  return series;
}

function lastVal(metrics, col) {
  if (!metrics || !metrics.headers) return null;
  const i = metrics.headers.indexOf(col);
  if (i < 0) return null;
  for (let r = metrics.rows.length-1; r >= 0; r--) {
    const v = metrics.rows[r][i];
    if (isFinite(v)) return v;
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────
const Nav = {
  init() {
    const list = el('nav-list');
    SECTIONS.forEach(({id, label, icon}) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'nav-btn' + (id === S.sec ? ' active' : '');
      btn.dataset.sec = id;
      btn.setAttribute('aria-label', `Go to ${label} section`);
      if (id === S.sec) btn.setAttribute('aria-current', 'page');
      // span elements escape the icon glyph automatically (textContent-style).
      const iconSpan = document.createElement('span');
      iconSpan.className = 'nav-icon';
      iconSpan.setAttribute('aria-hidden', 'true');
      iconSpan.textContent = icon;
      btn.appendChild(iconSpan);
      btn.appendChild(document.createTextNode(label));
      btn.addEventListener('click', () => { Nav.go(id); Nav.closeSidebar(); });
      li.appendChild(btn);
      list.appendChild(li);
    });

    el('hamburger').addEventListener('click', Nav.openSidebar);
    el('sidebar-close').addEventListener('click', Nav.closeSidebar);

    document.addEventListener('click', e => {
      const sb = el('sidebar');
      if (sb.classList.contains('open') &&
          !sb.contains(e.target) && e.target !== el('hamburger')) {
        Nav.closeSidebar();
      }
    });
  },

  go(secId) {
    S.sec = secId;
    document.querySelectorAll('.sec').forEach(s => s.classList.remove('active'));
    const tgt = el(`sec-${secId}`);
    if (tgt) tgt.classList.add('active');
    document.querySelectorAll('.nav-btn').forEach(b => {
      const active = b.dataset.sec === secId;
      b.classList.toggle('active', active);
      if (active) b.setAttribute('aria-current', 'page');
      else b.removeAttribute('aria-current');
    });
    Render[secId] && Render[secId]();
  },

  openSidebar()  { el('sidebar').classList.add('open'); },
  closeSidebar() { el('sidebar').classList.remove('open'); },
};

// ─────────────────────────────────────────────────────────────────────────────
// Theme
// ─────────────────────────────────────────────────────────────────────────────
const Theme = {
  init() {
    this.current = localStorage.getItem('tgx-theme') || 'auto';
    this._apply();
    el('theme-btn').addEventListener('click', () => {
      this.current = this.current === 'dark' ? 'light' : 'dark';
      localStorage.setItem('tgx-theme', this.current);
      this._apply();
    });
  },
  _apply() {
    document.documentElement.dataset.theme = this.current === 'auto' ? '' : this.current;
    el('theme-btn').textContent = this.current === 'light' ? '☽' : '☀';
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Color-blind safe palette toggle (Okabe-Ito).  Persists in localStorage.
// ─────────────────────────────────────────────────────────────────────────────
const Palette = {
  init() {
    this._apply();
    const btn = el('palette-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      S.palette = (S.palette === 'cb') ? 'default' : 'cb';
      localStorage.setItem('tgx-palette', S.palette);
      this._apply();
      // Re-render whatever section is active so charts pick up new colors.
      Render[S.sec] && Render[S.sec]();
      if (S.tvMode) TV.render();
    });
  },
  _apply() {
    const root = document.documentElement;
    const btn  = el('palette-btn');
    if (S.palette === 'cb') root.dataset.palette = 'cb';
    else delete root.dataset.palette;
    if (btn) btn.setAttribute('aria-pressed', S.palette === 'cb' ? 'true' : 'false');
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Pause / refresh / stale-data controls
// ─────────────────────────────────────────────────────────────────────────────
const Controls = {
  init() {
    const pause = el('pause-btn');
    const refresh = el('refresh-btn');
    if (pause) {
      pause.addEventListener('click', () => Controls.togglePause());
    }
    if (refresh) {
      refresh.addEventListener('click', () => poll());
    }
  },
  togglePause() {
    S.paused = !S.paused;
    const btn = el('pause-btn');
    if (btn) {
      btn.setAttribute('aria-pressed', S.paused ? 'true' : 'false');
      btn.textContent = S.paused ? '▶' : '⏸';
      btn.setAttribute('aria-label', S.paused ? 'Resume auto-refresh' : 'Pause auto-refresh');
      btn.title = S.paused ? 'Resume auto-refresh' : 'Pause auto-refresh';
    }
    if (S.paused) {
      if (S.pollTimer) clearInterval(S.pollTimer);
    } else {
      startPolling();
    }
  },
  // Show/hide the stale-data banner based on lastFetch age.
  updateStale() {
    const banner = el('stale-banner');
    const txt = el('stale-text');
    if (!banner || !txt) return;
    const lastIso = S.lastFetch;
    if (!lastIso) { banner.hidden = true; return; }
    const ageS = (Date.now() - new Date(lastIso).getTime()) / 1000;
    const threshold = (S.config?.stale_after_s) || CFG.staleAfterS;
    if (ageS > threshold) {
      banner.hidden = false;
      txt.textContent = `Data is stale (${Math.round(ageS)}s since last successful fetch).` +
                        (S.paused ? '  Auto-refresh is paused.' : '  Reconnect attempts continue.');
    } else {
      banner.hidden = true;
    }
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Range / window selector — limits how many trailing rows feed each chart
// ─────────────────────────────────────────────────────────────────────────────
const Range = {
  apply(rows) {
    if (S.range === 'all') return rows;
    const opt = CFG.ranges.find(r => r.id === S.range);
    if (!opt || !opt.n || rows.length <= opt.n) return rows;
    return rows.slice(rows.length - opt.n);
  },
  applySeries(series) {
    const out = {};
    Object.keys(series).forEach(k => { out[k] = Range.apply(series[k]); });
    return out;
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Export utilities — CSV (table or chart data), SVG (chart), print/PDF
// All client-side; no server writes; no external dependency.
// ─────────────────────────────────────────────────────────────────────────────
const Export = {
  // Trigger a browser download of a Blob with a chosen filename.
  _download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(url);
      a.remove();
    }, 100);
  },

  // Sanitize a string into a safe filename component.
  _safeName(s) {
    return String(s || 'export').replace(/[^\w.-]+/g, '_').slice(0, 64) || 'export';
  },

  // Export the loaded metrics CSV as-is (already a CSV; we just re-encode
  // the visible window from the in-memory state).
  metricsCsv() {
    const m = S.metrics;
    if (!m || !m.headers || !m.rows?.length) {
      alert('No metrics loaded yet.');
      return;
    }
    const esc = v => {
      const s = (v == null) ? '' : String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const lines = [m.headers.map(esc).join(',')];
    m.rows.forEach(r => lines.push(r.map(esc).join(',')));
    const blob = new Blob([lines.join('\n') + '\n'], {type: 'text/csv;charset=utf-8'});
    Export._download(blob, 'tgraphx_metrics.csv');
  },

  // Export a single chart's (x, y) pairs as a small two-column CSV.
  chartCsv(metricName, points) {
    if (!points || !points.length) {
      alert(`No data for "${metricName}".`);
      return;
    }
    const xLabel = (S.metrics?.headers || []).indexOf('epoch') >= 0 ? 'epoch' : 'step';
    const lines = [`${xLabel},${metricName}`];
    points.forEach(p => lines.push(`${p.x},${p.y}`));
    const blob = new Blob([lines.join('\n') + '\n'], {type: 'text/csv;charset=utf-8'});
    Export._download(blob, `tgraphx_chart_${Export._safeName(metricName)}.csv`);
  },

  // Export the chart's SVG element as a standalone .svg file.  The chart
  // module renders inline SVG, so we can serialize the first <svg> child.
  chartSvg(container, metricName) {
    if (!container) return;
    const svg = container.querySelector('svg');
    if (!svg) {
      alert('Chart not yet rendered.');
      return;
    }
    // Inline computed colors so the exported file renders outside the
    // dashboard.  We resolve common CSS variables to their current values.
    const clone = svg.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const root = document.documentElement;
    const cs = getComputedStyle(root);
    const vars = ['--bg','--bg2','--bg3','--card','--border','--grid',
                  '--text','--text2','--text3','--accent'];
    let css = ':root{';
    vars.forEach(v => { css += `${v}:${cs.getPropertyValue(v).trim()};`; });
    css += '}';
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = css;
    clone.insertBefore(style, clone.firstChild);
    const xml = '<?xml version="1.0" encoding="UTF-8"?>\n' +
                new XMLSerializer().serializeToString(clone);
    const blob = new Blob([xml], {type: 'image/svg+xml;charset=utf-8'});
    Export._download(blob, `tgraphx_chart_${Export._safeName(metricName)}.svg`);
  },

  // Print/Save-as-PDF via the browser's native print dialog.
  printPage() {
    window.print();
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Copy-to-clipboard with visual feedback
// ─────────────────────────────────────────────────────────────────────────────
async function copyText(text, btn) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      // Fallback for very old browsers / non-secure contexts
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1200);
    }
  } catch (e) { /* fail silently */ }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chart hover tooltip — dependency-free, nearest x-value lookup.
// Visual-only (mouse / pointer events).  Not shown on touch-primary devices
// (hover events don't fire reliably on touch), which is acceptable because
// the latest-value pill on each chart card provides the key data point.
// ─────────────────────────────────────────────────────────────────────────────
const Tooltip = {
  _el: null,

  init() {
    if (this._el) return;
    const d = document.createElement('div');
    d.className = 'chart-tooltip';
    d.setAttribute('aria-hidden', 'true');  // visual-only
    d.setAttribute('role', 'tooltip');
    document.body.appendChild(d);
    this._el = d;
  },

  show(x, y, xLabel, metricName, yVal, color) {
    if (!this._el) return;
    const d = this._el;
    d.innerHTML =
      `<div class="chart-tooltip-x">${esc(xLabel)}: ${esc(String(isFinite(x) ? (Number.isInteger(x) ? x : x.toFixed(2)) : '—'))}</div>` +
      `<div class="chart-tooltip-val" style="color:${esc(color)}">${esc(metricName)}: ${esc(fmt(yVal))}</div>`;
    d.classList.add('visible');
  },

  move(clientX, clientY) {
    if (!this._el) return;
    const W = window.innerWidth, H = window.innerHeight;
    const tw = this._el.offsetWidth + 14, th = this._el.offsetHeight + 14;
    const lx = clientX + 12 + tw > W ? clientX - tw : clientX + 12;
    const ly = clientY - 10 - th < 0 ? clientY + 14 : clientY - 10 - th;
    this._el.style.left = `${Math.max(0, lx)}px`;
    this._el.style.top  = `${Math.max(0, ly)}px`;
  },

  hide() {
    if (!this._el) return;
    this._el.classList.remove('visible');
  },

  // Attach pointer event handlers to an SVG element's data area.
  // data: [{x, y}] — the same array passed to SvgChart.render().
  // pad, W, H — the SVG geometry from SvgChart.
  attach(svgEl, data, metricName, color, pad, W, H) {
    if (!svgEl || !data || data.length === 0) return;
    // Re-derive scale constants matching SvgChart.
    const xs = data.map(d => d.x), ys = data.map(d => d.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), xR = x1 - x0 || 1;
    const iW = W - pad.l - pad.r;

    // Overlay an invisible rect to capture mouse events in the chart area.
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', pad.l);
    rect.setAttribute('y', pad.t);
    rect.setAttribute('width', iW);
    rect.setAttribute('height', H - pad.t - pad.b);
    rect.setAttribute('fill', 'transparent');
    rect.style.cursor = 'crosshair';
    svgEl.appendChild(rect);

    const getBBox = () => svgEl.getBoundingClientRect();

    rect.addEventListener('pointermove', e => {
      e.preventDefault();
      const bb = getBBox();
      const mx = e.clientX - bb.left;
      // Map mouse-x to data-x.
      const dataX = x0 + ((mx - pad.l) / iW) * xR;
      // Find nearest data point by x-distance.
      let best = data[0], bestDist = Infinity;
      for (const pt of data) {
        const d = Math.abs(pt.x - dataX);
        if (d < bestDist) { bestDist = d; best = pt; }
      }
      Tooltip.show(best.x, best.y, 'epoch', metricName, best.y, color);
      Tooltip.move(e.clientX, e.clientY);
    });

    rect.addEventListener('pointerleave', () => Tooltip.hide());
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// TV mode
// ─────────────────────────────────────────────────────────────────────────────
const TV = {
  init() {
    el('tv-btn').addEventListener('click', e => { e.preventDefault(); TV.enter(); });
    el('tv-exit').addEventListener('click', () => TV.exit());
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && S.tvMode) TV.exit(); });
  },
  enter() {
    S.tvMode = true;
    el('tv-overlay').hidden = false;
    TV.render();
  },
  exit() {
    S.tvMode = false;
    el('tv-overlay').hidden = true;
  },
  render() {
    if (!S.tvMode) return;
    const st = S.status || {};
    const loss = lastVal(S.metrics, 'train_loss') ?? lastVal(S.metrics, 'loss');
    const eta = estimateETA(S.metrics, st.total_epochs);
    const elapsed = T.elapsedSince(st.start_time);

    const series = metricsToSeries(S.metrics);
    const lossKey = ['train_loss','loss'].find(k => series[k] && series[k].length);
    const chart = document.createElement('div');
    chart.className = 'chart-container'; chart.style.height = '100%';

    el('tv-body').innerHTML = `
      <div class="tv-card"><div class="tv-label">Status</div>
        <div class="tv-value" style="font-size:2rem">${esc(st.status||'—')}</div>
        <div class="tv-sub">${esc(st.run_name||'')}</div></div>
      <div class="tv-card tv-chart" id="tv-chart-wrap"></div>
      <div class="tv-card"><div class="tv-label">Train Loss</div>
        <div class="tv-value">${esc(fmt(loss))}</div>
        <div class="tv-sub">Epoch ${esc(st.epoch??'—')}${st.total_epochs?' / '+esc(st.total_epochs):''}</div></div>
      <div class="tv-card"><div class="tv-label">Elapsed / ETA</div>
        <div class="tv-value" style="font-size:2.2rem">${esc(T.elapsed(elapsed))}</div>
        <div class="tv-sub">ETA: ${esc(eta!=null?(eta<10?'Estimating…':T.elapsed(eta)):'—')}</div></div>`;

    if (lossKey && series[lossKey]) {
      const wrap = el('tv-chart-wrap');
      const meta = metaFor(lossKey);
      const c = new SvgChart(wrap, {color: meta.color,
        label: meta.label, height: wrap.clientHeight||300});
      c.render(series[lossKey], S.smooth);
    }
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Section renderers
// ─────────────────────────────────────────────────────────────────────────────
const Render = {

  // ── Overview ───────────────────────────────────────────────────────────────
  overview() {
    const sec = el('sec-overview');
    const st  = S.status   || {};
    const m   = S.metrics  || {};
    const hw  = S.hardware || {};

    const epoch      = st.epoch      ?? '—';
    const totalEp    = st.total_epochs;
    const progress   = totalEp ? Math.min(100, ((st.epoch||0)/totalEp*100)) : null;
    const loss       = lastVal(S.metrics, 'train_loss') ?? lastVal(S.metrics, 'loss');
    const vLoss      = lastVal(S.metrics, 'val_loss');
    const acc        = lastVal(S.metrics, 'accuracy') ?? lastVal(S.metrics, 'acc');
    const lr         = lastVal(S.metrics, 'learning_rate') ?? lastVal(S.metrics, 'lr');
    const elapsed    = T.elapsedSince(st.start_time);
    const eta        = estimateETA(S.metrics, totalEp);
    const statusCls  = {running:'chip-running',completed:'chip-completed'}[st.status]||'chip-unknown';

    const statsHtml = [
      card('Status',
        `<span class="chip ${statusCls}">${esc(st.status||'unknown')}</span>`,
        esc(st.run_name||st.task||''), 'card-accent'),
      card('Progress',
        totalEp
          ? `${esc(epoch)} <span style="font-size:.9rem;color:var(--text2)">/ ${esc(totalEp)}</span>`
          : esc(epoch),
        progress != null
          ? `<div class="progress-outer"><div class="progress-inner" style="width:${progress.toFixed(1)}%"></div></div>`
          : 'epochs', ''),
      card('Elapsed', esc(T.elapsed(elapsed)),
        `ETA: ${esc(eta != null ? (eta < 10 ? 'done' : T.elapsed(eta)) : (elapsed && elapsed > 0 && m.rows?.length < CFG.etaMinPts ? 'estimating…' : '—'))}`, ''),
      card('Train Loss', esc(fmt(loss)),
        vLoss != null ? `val loss: ${esc(fmt(vLoss))}` : (acc != null ? `acc: ${esc(fmt(acc))}` : ''), 'card-accent'),
    ].join('');

    const hw1Html = [
      card('Device', esc(st.device || (hw.cuda_available?'CUDA':'CPU')),
           hw.torch?`PyTorch ${esc(hw.torch)}`:'', ''),
      card('LR', esc(fmt(lr)), lr != null ? 'learning rate' : '', 'card-amber'),
      card('Last update', esc(T.ago(st.last_update)||'—'),
        st.last_update ? `at ${esc(T.local(st.last_update))}` : '', ''),
    ].join('');

    sec.innerHTML = `
      <h2 id="sec-overview-title" class="page-title">Overview</h2>
      <div class="grid-4">${statsHtml}</div>
      <div class="grid-3">${hw1Html}</div>
      <div class="card chart-card" style="margin-bottom:16px">
        <div class="chart-title">Training Curve</div>
        <div id="ov-chart" class="chart-container"></div>
      </div>
      ${totalEp ? `<div style="margin-bottom:8px"><div class="card" style="padding:12px 20px">
        <div class="card-title">Epoch progress</div>
        <div class="progress-outer" style="height:10px;margin-top:6px">
          <div class="progress-inner" style="width:${progress.toFixed(1)}%"></div>
        </div>
        <div class="card-sub" style="margin-top:4px">${progress.toFixed(1)}% complete</div>
      </div></div>` : ''}`;

    // Draw overview chart
    const seriesAll = metricsToSeries(S.metrics);
    const series = Range.applySeries(seriesAll);
    const lossKey = ['train_loss','loss'].find(k => series[k]?.length);
    const valKey  = ['val_loss'].find(k => series[k]?.length);
    const container = el('ov-chart');

    if (lossKey && container) {
      container.style.height = '180px';
      // Multi-series: draw val on same SVG by overlaying
      const lossMeta = metaFor(lossKey);
      const c = new SvgChart(container, {
        color: lossMeta.color,
        label: 'Loss',
        height: 180,
      });
      c.render(series[lossKey], S.smooth);

      // Overlay val loss as a separate polyline
      if (valKey && series[valKey]?.length) {
        const existingSvg = container.querySelector('svg');
        const valMeta = metaFor(valKey);
        if (existingSvg) {
          const W = existingSvg.viewBox.baseVal.width || 540;
          const H = 180;
          const PAD = {t:22,r:18,b:40,l:58};
          const iW = W-PAD.l-PAD.r, iH = H-PAD.t-PAD.b;
          const allY = [...series[lossKey], ...(series[valKey]||[])].map(d=>d.y).filter(isFinite);
          const allX = series[lossKey].map(d=>d.x);
          const x0=Math.min(...allX), x1=Math.max(...allX), xR=x1-x0||1;
          const yPad=(Math.max(...allY)-Math.min(...allY))*0.12||0.1;
          const yLo=Math.min(...allY)-yPad, yHi=Math.max(...allY)+yPad, yR=yHi-yLo;
          const sx = x=>PAD.l+(x-x0)/xR*iW;
          const sy = y=>PAD.t+(1-(y-yLo)/yR)*iH;
          const pts = series[valKey].map(d=>`${sx(d.x).toFixed(1)},${sy(d.y).toFixed(1)}`).join(' ');
          const line = document.createElementNS('http://www.w3.org/2000/svg','polyline');
          line.setAttribute('points', pts);
          line.setAttribute('fill', 'none');
          line.setAttribute('stroke', valMeta.color);
          line.setAttribute('stroke-width', '2');
          line.setAttribute('stroke-dasharray', '5,3');
          line.setAttribute('stroke-linejoin', 'round');
          existingSvg.appendChild(line);
          const last = series[valKey][series[valKey].length-1];
          const label = document.createElementNS('http://www.w3.org/2000/svg','text');
          label.setAttribute('x', W-PAD.r-2);
          label.setAttribute('y', PAD.t-6);
          label.setAttribute('font-size', '10');
          label.setAttribute('text-anchor', 'end');
          label.setAttribute('fill', valMeta.color);
          label.setAttribute('font-weight', '700');
          label.textContent = `val: ${fmt(last.y)}`;
          existingSvg.appendChild(label);
        }
      }
    } else if (container) {
      container.innerHTML = `<div class="chart-empty">No metrics yet — start training to see curves.</div>`;
    }
  },

  // ── Metrics ─────────────────────────────────────────────────────────────────
  metrics() {
    const sec = el('sec-metrics');
    const seriesAll = metricsToSeries(S.metrics);
    const series = Range.applySeries(seriesAll);
    const keys = Object.keys(series);
    const truncNote = S.metrics?.truncated
      ? `<div class="disclosure" style="color:var(--amber);margin-bottom:10px">
           &#9888; Showing latest ${esc(S.metrics.rows?.length ?? 0)} of
           ${esc(S.metrics.total_row_count)} rows.
           Raw <code>metrics.csv</code> is unchanged.
         </div>`
      : '';
    const rangeNote = (S.range !== 'all' && keys.length)
      ? `<div class="disclosure">Window: ${esc(S.range)} most-recent points (server data unchanged).</div>`
      : '';

    const rangeOpts = CFG.ranges.map(r =>
      `<option value="${esc(r.id)}"${S.range===r.id?' selected':''}>${esc(r.label)}</option>`
    ).join('');

    sec.innerHTML = `
      <h2 id="sec-metrics-title" class="page-title">Metrics</h2>
      ${truncNote}
      <div class="toolbar" role="toolbar" aria-label="Metrics controls">
        <div class="toolbar-group">
          <label for="range-sel">Window:</label>
          <select id="range-sel" class="tb-select" aria-label="Number of trailing points to show">
            ${rangeOpts}
          </select>
        </div>
        <div class="toolbar-group">
          <label for="smooth-cb">
            <input type="checkbox" id="smooth-cb"${S.smooth?' checked':''}
                   aria-label="Apply exponential moving average smoothing">
            Smooth (EMA α=${esc(CFG.smoothAlpha)})
          </label>
        </div>
        <div class="toolbar-group">
          <button class="tb-btn" id="export-metrics-csv" type="button"
                  aria-label="Download metrics.csv">CSV</button>
          <button class="tb-btn" id="print-btn-metrics" type="button"
                  aria-label="Print or save as PDF">Print / PDF</button>
        </div>
      </div>
      ${rangeNote}
      <div id="metrics-grid"></div>`;

    const cb = el('smooth-cb');
    if (cb) cb.addEventListener('change', () => { S.smooth = cb.checked; Render.metrics(); });
    const sel = el('range-sel');
    if (sel) sel.addEventListener('change', () => { S.range = sel.value; Render.metrics(); });
    const csvBtn = el('export-metrics-csv');
    if (csvBtn) csvBtn.addEventListener('click', () => Export.metricsCsv());
    const printBtn = el('print-btn-metrics');
    if (printBtn) printBtn.addEventListener('click', () => Export.printPage());

    const grid = el('metrics-grid');
    if (!keys.length || !S.metrics?.rows?.length) {
      grid.innerHTML = emptyState('No metric data yet.', 'Start training and data will appear here.');
      return;
    }

    keys.forEach(key => {
      if (!series[key] || !series[key].length) return;
      const meta = metaFor(key);
      const lastY = series[key][series[key].length - 1].y;
      const card = document.createElement('div');
      card.className = 'card chart-card';
      // Note: meta.label and key flow through esc() because metric names come
      // from user-controlled CSV column headers.
      card.innerHTML = `
        <div class="chart-title">${esc(meta.label)}
          <span class="latest-val" aria-label="latest value">${esc(fmt(lastY))}</span>
        </div>
        <div class="chart-container" style="height:180px"></div>
        <div class="toolbar" style="margin-top:10px;margin-bottom:0;padding:6px 8px">
          <div class="toolbar-group">
            <button class="tb-btn" type="button"
                    data-action="csv" data-key="${esc(key)}"
                    aria-label="Download ${esc(meta.label)} data as CSV">CSV</button>
            <button class="tb-btn" type="button"
                    data-action="svg" data-key="${esc(key)}"
                    aria-label="Download ${esc(meta.label)} chart as SVG">SVG</button>
          </div>
        </div>`;
      grid.appendChild(card);
      const cont = card.querySelector('.chart-container');
      const c = new SvgChart(cont, {color: meta.color, label: meta.label, height: 180});
      c.render(series[key], S.smooth);
      // Wire per-card export buttons.
      card.querySelectorAll('button[data-action]').forEach(b => {
        b.addEventListener('click', () => {
          const k = b.dataset.key;
          const action = b.dataset.action;
          if (action === 'csv') Export.chartCsv(k, series[k]);
          else if (action === 'svg') Export.chartSvg(cont, k);
        });
      });
    });
  },

  // ── Graph ────────────────────────────────────────────────────────────────────
  graph() {
    const sec = el('sec-graph');
    const g   = S.graph;
    const gs  = S.graphStats;

    // Precomputed stats card (from graph_stats.json) — shown even without graph_metadata.json.
    const statFields = (gs && gs.available) ? [
      ['Nodes',      gs.num_nodes],
      ['Edges',      gs.num_edges],
      ['Directed',   gs.directed != null ? (gs.directed ? 'Yes' : 'No') : null],
      ['Self-loops', gs.self_loops != null ? (gs.self_loops ? 'Yes' : 'No') : null],
      ['Avg degree', gs.avg_degree != null ? fmt(gs.avg_degree) : null],
      ['Min degree', gs.min_degree],
      ['Max degree', gs.max_degree],
      ['Density',    gs.density != null ? gs.density.toFixed(6) : null],
      ['Components', gs.connected_components],
      ['Isolated',   gs.isolated_nodes],
    ].filter(([, v]) => v != null) : [];

    const statsCardHtml = statFields.length
      ? `<div class="card" style="margin-bottom:16px">
           <div class="card-title">Precomputed Statistics
             <span style="font-size:.72rem;color:var(--text3)"> (graph_stats.json)</span>
           </div>
           ${statFields.map(([k, v]) =>
             `<div class="hw-row"><span class="hw-label">${esc(k)}</span>
              <span class="hw-val">${esc(String(v))}</span></div>`
           ).join('')}
         </div>` : '';

    if (!g || !g.available) {
      sec.innerHTML = `<h2 id="sec-graph-title" class="page-title">Graph</h2>
        ${statsCardHtml}
        ${emptyState('No graph metadata available.',
          'Add graph logging to your training script. See README for details.')}`;
      return;
    }

    const stats = [
      ['Nodes', g.num_nodes ?? '—'],
      ['Edges', g.num_edges ?? '—'],
      ['Directed', g.directed ? 'Yes' : 'No'],
      ['Self-loops', g.self_loops ? 'Yes' : 'No'],
      ['Density', g.num_nodes > 1
        ? ((g.num_edges || 0) / (g.num_nodes*(g.num_nodes-1))).toFixed(4)
        : '—'],
      ['Avg degree', g.degree_stats?.mean != null ? fmt(g.degree_stats.mean) : '—'],
    ];

    const degStats = g.degree_stats || {};
    const histogram = g.degree_stats?.histogram || [];
    const degHtml = histogram.length
      ? `<div class="card-title">Degree distribution</div>
         <div class="degree-bar">${histogram.map((v,i) =>
           `<div class="degree-bar-col" style="height:${(v/Math.max(...histogram)*100).toFixed(1)}%"
             title="degree ${esc(i)}: ${esc(v)} nodes"></div>`).join('')}</div>
         <div class="card-sub">min ${esc(degStats.min??'—')} · max ${esc(degStats.max??'—')} · mean ${esc(fmt(degStats.mean))}</div>`
      : `<div class="card-title">Degree stats</div>
         <div class="hw-row"><span class="hw-label">Min</span><span class="hw-val">${esc(degStats.min??'—')}</span></div>
         <div class="hw-row"><span class="hw-label">Max</span><span class="hw-val">${esc(degStats.max??'—')}</span></div>
         <div class="hw-row"><span class="hw-label">Mean</span><span class="hw-val">${esc(fmt(degStats.mean))}</span></div>`;

    const builderHtml = g.builder
      ? `<div class="hw-row"><span class="hw-label">Builder</span>
           <span class="hw-val">${esc(g.builder)}</span></div>
         ${g.builder_params ? Object.entries(g.builder_params).map(([k,v])=>
           `<div class="hw-row"><span class="hw-label">${esc(k)}</span><span class="hw-val">${esc(v)}</span></div>`
         ).join('') : ''}`
      : '<div class="card-sub">No builder metadata</div>';

    sec.innerHTML = `
      <h2 id="sec-graph-title" class="page-title">Graph</h2>
      ${statsCardHtml}
      <div class="grid-2" style="margin-bottom:16px">
        <div class="card">
          <div class="card-title">Summary</div>
          ${stats.map(([k,v])=>`<div class="hw-row"><span class="hw-label">${esc(k)}</span>
            <span class="hw-val">${esc(v)}</span></div>`).join('')}
        </div>
        <div class="card">${degHtml}</div>
      </div>
      <div class="grid-2" style="margin-bottom:16px">
        <div class="card"><div class="card-title">Graph builder</div>${builderHtml}</div>
        <div class="card" id="graph-preview-card">
          <div class="card-title">Graph preview</div>
          <div id="graph-svg-container"></div>
        </div>
      </div>`;

    // Render SVG preview
    const svgCont = el('graph-svg-container');
    if (g.render_mode === 'full' && g.edge_index) {
      GraphViz.render(svgCont, g);
    } else if (g.builder === 'build_grid_graph' && g.builder_params) {
      const {rows=2, cols=2} = g.builder_params;
      GraphViz.renderGrid(svgCont, rows, cols, g.directed, g.self_loops);
    } else if (g.builder === 'build_grid_graph_3d' && g.builder_params) {
      const {depth=2, rows=2, cols=2} = g.builder_params;
      GraphViz.renderGrid3D(svgCont, depth, rows, cols);
    } else {
      svgCont.innerHTML = `<div class="chart-empty">
        ${g.render_mode === 'summary'
          ? 'Graph too large for preview. Showing summary only.'
          : 'Enable full graph logging to see a preview.'}
      </div>`;
    }
  },

  // ── Hardware ─────────────────────────────────────────────────────────────────
  hardware() {
    const sec = el('sec-hardware');
    const hw  = S.hardware || {};

    const verRows = [
      ['Python', hw.python],
      ['PyTorch', hw.torch],
      ['TGraphX', hw.tgraphx],
      ['Platform', hw.platform],
      ['CPU cores', hw.cpu_count],
    ].filter(([,v]) => v != null);

    // Compact "unavailable" footer that distinguishes optional-dep vs sensor.
    const unavailRow = (label, reason) => reason
      ? `<div class="hw-unavail"><strong>${esc(label)}:</strong> ${esc(reason)}</div>`
      : '';

    const cpuHtml = hw.psutil_available ? `
      <div class="card">
        <div class="card-title">CPU &amp; Memory</div>
        <div class="hw-row">
          <span class="hw-label">CPU</span>
          <div class="hw-bar-wrap"><div class="hw-bar hw-bar-cpu" style="width:${esc(hw.cpu_percent||0)}%"></div></div>
          <span class="hw-val">${esc((hw.cpu_percent??'—'))}%</span>
        </div>
        <div class="hw-row">
          <span class="hw-label">RAM</span>
          <div class="hw-bar-wrap"><div class="hw-bar hw-bar-ram" style="width:${esc(hw.ram_percent||0)}%"></div></div>
          <span class="hw-val">${esc((hw.ram_percent??'—'))}%</span>
        </div>
        <div class="card-sub">${esc((hw.ram_used_gb??'—'))} / ${esc((hw.ram_total_gb??'—'))} GB</div>
        ${hw.process_ram_mb != null ? `<div class="card-sub">This process: ${esc(hw.process_ram_mb)} MB</div>` : ''}
      </div>` : `<div class="card"><div class="card-title">CPU &amp; Memory</div>
        ${unavailRow('CPU/RAM',
            hw.unavailable_reason_psutil ||
            'psutil not installed; CPU/RAM metrics unavailable.')}
        <div class="card-sub" style="margin-top:8px">Install with:
          <code>pip install "tgraphx[monitoring]"</code></div></div>`;

    const gpuRows = hw.cuda_available ? `
        <div class="hw-row"><span class="hw-label">Device</span>
          <span class="hw-val" style="font-size:.78rem">${esc(hw.cuda_device_name||'—')}</span></div>
        <div class="hw-row"><span class="hw-label">VRAM used</span>
          <span class="hw-val">${esc(hw.cuda_mem_allocated_mb??'—')} MB</span></div>
        <div class="hw-row"><span class="hw-label">VRAM reserved</span>
          <span class="hw-val">${esc(hw.cuda_mem_reserved_mb??'—')} MB</span></div>
        <div class="hw-row"><span class="hw-label">VRAM total</span>
          <span class="hw-val">${esc(hw.cuda_mem_total_mb??'—')} MB</span></div>
        ${hw.gpu_util_pct != null
          ? `<div class="hw-row"><span class="hw-label">Utilization</span>
              <div class="hw-bar-wrap"><div class="hw-bar hw-bar-gpu" style="width:${esc(hw.gpu_util_pct)}%"></div></div>
              <span class="hw-val">${esc(hw.gpu_util_pct)}%</span></div>`
          : unavailRow('GPU utilization',
                       hw.unavailable_reason_gpu_util || hw.unavailable_reason_pynvml)}
        ${hw.gpu_temp_c != null
          ? (() => {
              const status = hw.gpu_thermal_status || 'unknown';
              const cls = `thermal-chip thermal-${esc(status)}`;
              const icon = status === 'near-throttle' ? '⚠' : status === 'warm' ? '▲' : '✓';
              return `<div class="hw-row">
                <span class="hw-label">Temperature</span>
                <span class="hw-val">${esc(hw.gpu_temp_c)}°C
                  <span class="${cls}" aria-label="Thermal status: ${esc(status)}">${esc(icon)} ${esc(status)}</span>
                </span>
              </div>`;
            })()
          : unavailRow('GPU temperature',
                       hw.unavailable_reason_gpu_temp || hw.unavailable_reason_pynvml)}
        ${hw.gpu_fan_pct != null
          ? `<div class="hw-row"><span class="hw-label">Fan</span>
              <span class="hw-val">${esc(hw.gpu_fan_pct)}%</span></div>`
          : unavailRow('GPU fan',
                       hw.unavailable_reason_gpu_fan)}
        ${hw.gpu_power_w != null
          ? (() => {
              const limit = hw.gpu_power_limit_w;
              const pct   = limit ? Math.min(100, (hw.gpu_power_w / limit * 100)).toFixed(0) : null;
              return `<div class="hw-row">
                <span class="hw-label">Power draw</span>
                ${pct != null
                  ? `<div class="pw-bar-wrap"><div class="pw-bar" style="width:${esc(pct)}%"></div></div>`
                  : ''}
                <span class="hw-val">${esc(hw.gpu_power_w)} W${limit ? ` / ${esc(limit)} W` : ''}</span>
              </div>`;
            })()
          : unavailRow('GPU power', hw.unavailable_reason_gpu_power)}
    ` : '';

    const gpuHtml = hw.cuda_available
      ? `<div class="card"><div class="card-title">CUDA GPU</div>${gpuRows}</div>`
      : hw.mps_available
        ? `<div class="card"><div class="card-title">Apple MPS</div>
            <div class="card-sub">Apple Silicon MPS backend is available.</div>
            ${unavailRow('CUDA', hw.unavailable_reason_cuda)}</div>`
        : `<div class="card"><div class="card-title">GPU</div>
            ${unavailRow('CUDA', hw.unavailable_reason_cuda || 'No CUDA-capable GPU detected')}
            ${unavailRow('GPU sensors', hw.unavailable_reason_pynvml)}</div>`;

    const ageNote = (hw.collected_at || hw.cached_age_s != null)
      ? `<div class="hw-timestamp">
           Collected: ${esc(hw.collected_at ? T.localFull(hw.collected_at) : '—')}
           ${hw.cached_age_s != null ? ` · cached ${esc(hw.cached_age_s)}s` : ''}
         </div>` : '';

    sec.innerHTML = `
      <h2 id="sec-hardware-title" class="page-title">Hardware &amp; Environment</h2>
      <p class="page-note">Monitoring is best-effort and depends on optional system packages.
        Missing sensor data does not indicate a training problem; reasons are shown
        per-row to distinguish "optional dep not installed" from "sensor not reported".</p>
      <div class="grid-2" style="margin-bottom:16px">
        <div class="card">
          <div class="card-title">Versions &amp; Environment</div>
          ${verRows.map(([k,v])=>`<div class="hw-row">
            <span class="hw-label">${esc(k)}</span><span class="hw-val">${esc(v)}</span></div>`).join('')}
          ${ageNote}
        </div>
        ${cpuHtml}
      </div>
      <div class="grid-2">${gpuHtml}</div>`;
  },

  // ── Logs ─────────────────────────────────────────────────────────────────────
  logs() {
    const sec = el('sec-logs');
    const m   = S.metrics;

    if (!m || !m.headers || !m.rows?.length) {
      sec.innerHTML = `<div class="page-title">Metrics Log</div>
        ${emptyState('No metrics logged yet.', 'Write metrics.csv from your training script.')}`;
      return;
    }

    const lastN = m.rows.slice(-50);
    const hdrs  = m.headers;

    const tsIdx = hdrs.indexOf('timestamp');
    const rows  = [...lastN].reverse();

    sec.innerHTML = `
      <h2 id="sec-logs-title" class="page-title">Metrics Log
        <span style="font-size:.8rem;color:var(--text3)">(last 50 rows, newest first)</span>
      </h2>
      <div class="toolbar" role="toolbar" aria-label="Logs controls">
        <div class="toolbar-group">
          <button class="tb-btn" id="export-logs-csv" type="button"
                  aria-label="Download metrics.csv">CSV</button>
          <button class="tb-btn" id="print-btn-logs" type="button"
                  aria-label="Print or save as PDF">Print / PDF</button>
        </div>
      </div>
      <div class="card" style="padding:0">
        <div class="tbl-wrap">
          <table>
            <thead><tr>${hdrs.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead>
            <tbody>${rows.map(row=>`<tr>${row.map((v,i)=>{
              if (i===tsIdx && typeof v==='string') return `<td>${esc(T.local(v))}</td>`;
              return `<td>${esc(typeof v==='number'?fmt(v):v)}</td>`;
            }).join('')}</tr>`).join('')}</tbody>
          </table>
        </div>
      </div>`;
    const csvBtn = el('export-logs-csv');
    if (csvBtn) csvBtn.addEventListener('click', () => Export.metricsCsv());
    const printBtn = el('print-btn-logs');
    if (printBtn) printBtn.addEventListener('click', () => Export.printPage());
  },

  // ── Config ────────────────────────────────────────────────────────────────────
  config() {
    const sec = el('sec-config');
    const meta = S.metadata;

    if (!meta || !Object.keys(meta).length) {
      sec.innerHTML = `<h2 id="sec-config-title" class="page-title">Run Configuration</h2>
        ${emptyState('No run_metadata.json found.',
          'Create a run_metadata.json in your logdir with training configuration.')}`;
      return;
    }

    // textContent on the <pre> ensures untrusted metadata strings are
    // rendered safely (no innerHTML interpolation).
    sec.innerHTML = `
      <h2 id="sec-config-title" class="page-title">Run Configuration</h2>
      <div class="card" style="padding:0">
        <pre id="config-pre"></pre>
      </div>`;
    const pre = el('config-pre');
    if (pre) pre.textContent = JSON.stringify(meta, null, 2);
  },

  // ── Tools ─────────────────────────────────────────────────────────────────────
  tools() {
    const sec = el('sec-tools');
    const cfg = S.config || {};
    const here = window.location;
    const localUrl = `${here.protocol}//${here.host}${here.pathname}`;
    const port = cfg.port || here.port || 8765;
    // We can't see the LAN IP from JS, but the user's browser address bar
    // already shows whatever URL they reached us at — reflect that as one
    // of the copyable URLs.  For the LAN case, instruct the user.
    const lanRow = cfg.lan_mode ? `
      <div class="url-row">
        <code id="lan-url-tip">Use this machine's LAN IP, port ${esc(port)}${cfg.has_token ? ' (token required)' : ''}</code>
        <button class="copy-btn" type="button" data-target="lan-url-tip"
                aria-label="Copy LAN URL hint">Copy</button>
      </div>` : '';

    sec.innerHTML = `
      <h2 id="sec-tools-title" class="page-title">Tools</h2>
      <p class="page-note">Client-side exports — nothing is uploaded.  CSV files are
      generated from the data already loaded in your browser.</p>

      <div class="card" style="margin-bottom:16px">
        <div class="card-title">Share / Copy URL</div>
        <div class="url-row">
          <code id="local-url">${esc(localUrl)}</code>
          <button class="copy-btn" type="button" data-target="local-url"
                  aria-label="Copy local URL">Copy</button>
        </div>
        ${lanRow}
        ${cfg.has_token ? '<p class="card-sub">LAN clients must include the token (e.g. <code>?token=…</code> or <code>Authorization: Bearer …</code>).</p>' : ''}
      </div>

      <div class="card" style="margin-bottom:16px">
        <div class="card-title">Export</div>
        <div class="toolbar-group" style="flex-wrap:wrap">
          <button class="tb-btn" id="tools-csv" type="button"
                  aria-label="Download metrics CSV">Metrics CSV</button>
          <button class="tb-btn" id="tools-print" type="button"
                  aria-label="Print or save as PDF">Print / PDF</button>
        </div>
        <p class="card-sub" style="margin-top:8px">
          Per-chart CSV/SVG buttons appear next to each chart on the Metrics page.
        </p>
      </div>

      <div class="card">
        <div class="card-title">Refresh</div>
        <div class="hw-row">
          <span class="hw-label">Auto-refresh interval</span>
          <span class="hw-val">${esc((cfg.refresh_interval_s ?? CFG.pollMs/1000))}s</span>
        </div>
        <div class="hw-row">
          <span class="hw-label">Status</span>
          <span class="hw-val">${S.paused ? 'Paused' : 'Running'}</span>
        </div>
        <div class="hw-row">
          <span class="hw-label">Stale threshold</span>
          <span class="hw-val">${esc(cfg.stale_after_s ?? CFG.staleAfterS)}s</span>
        </div>
        <p class="card-sub" style="margin-top:8px">
          The Pause/Refresh controls are in the top bar (icon buttons).
        </p>
      </div>`;

    // Wire copy buttons.
    sec.querySelectorAll('.copy-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const t = el(btn.dataset.target);
        if (t) copyText(t.textContent, btn);
      });
    });
    const csvBtn = el('tools-csv');
    if (csvBtn) csvBtn.addEventListener('click', () => Export.metricsCsv());
    const pBtn = el('tools-print');
    if (pBtn) pBtn.addEventListener('click', () => Export.printPage());
  },

  // ── About ─────────────────────────────────────────────────────────────────────
  about() {
    const sec  = el('sec-about');
    const hw   = S.hardware || {};
    sec.innerHTML = `
      <h2 id="sec-about-title" class="page-title">About</h2>
      <div class="grid-2">
        <div class="card">
          <div class="card-title">TGraphX Dashboard</div>
          <div class="hw-row"><span class="hw-label">TGraphX</span><span class="hw-val">${esc(hw.tgraphx||'—')}</span></div>
          <div class="hw-row"><span class="hw-label">Python</span><span class="hw-val">${esc(hw.python||'—')}</span></div>
          <div class="hw-row"><span class="hw-label">PyTorch</span><span class="hw-val">${esc(hw.torch||'—')}</span></div>
          <div class="hw-row"><span class="hw-label">Platform</span><span class="hw-val">${esc(hw.platform||'—')}</span></div>
          <div class="card-sub" style="margin-top:12px">
            Local-first monitoring · No external dependencies · Read-only
          </div>
        </div>
        <div class="card">
          <div class="card-title">Usage</div>
          <pre id="about-usage" style="font-size:.75rem"></pre>
        </div>
      </div>`;
    // Use textContent so the example stays as plain text and is never
    // interpreted as HTML.
    const pre = el('about-usage');
    if (pre) {
      pre.textContent = [
        'tgraphx-dashboard --logdir runs/demo',
        '',
        '# LAN mode (token required)',
        'tgraphx-dashboard --logdir runs/demo \\',
        '  --host 0.0.0.0 --token MY_TOKEN',
        '',
        '# Python API',
        'from tgraphx.dashboard import launch_dashboard',
        'launch_dashboard("runs/demo")',
      ].join('\n');
    }
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Graph visualizer
// ─────────────────────────────────────────────────────────────────────────────
const GraphViz = {
  _W: 320, _H: 220,

  render(container, g) {
    const N = g.num_nodes || 0;
    const edgeIndex = g.edge_index;
    if (!edgeIndex || N === 0) { container.innerHTML = '<div class="chart-empty">No edge_index data</div>'; return; }
    const [srcs, dsts] = edgeIndex;
    const edges = srcs.map((s,i)=>([s, dsts[i]]));
    const pos = this._forceLayout(N, edges, this._W-20, this._H-20, CFG.forceIter);
    container.innerHTML = this._buildSvg(pos, edges, g.directed, this._W, this._H, 20);
  },

  renderGrid(container, rows, cols, directed, selfLoops) {
    const W = this._W, H = this._H;
    const padX = 30, padY = 30;
    const iW = W-2*padX, iH = H-2*padY;
    const N = rows*cols;
    const pos = [];
    for (let r=0; r<rows; r++)
      for (let c=0; c<cols; c++)
        pos.push({x: padX + (cols>1?c/(cols-1)*iW:iW/2),
                  y: padY + (rows>1?r/(rows-1)*iH:iH/2)});

    const edges = [];
    for (let r=0; r<rows; r++) for (let c=0; c<cols; c++) {
      const u = r*cols+c;
      if (c+1<cols) edges.push([u, u+1]);
      if (r+1<rows) edges.push([u, u+cols]);
    }
    container.innerHTML = this._buildSvg(pos, edges, directed, W, H, 14);
  },

  renderGrid3D(container, depth, rows, cols) {
    // Show as layered small grids
    const sliceW = 80, sliceH = 60, gap = 10;
    const totalW = depth*(sliceW+gap);
    let svg = `<svg viewBox="0 0 ${totalW} ${sliceH+30}" style="width:100%;max-height:120px"
      xmlns="http://www.w3.org/2000/svg">
      <text x="${totalW/2}" y="14" text-anchor="middle" font-size="10" fill="var(--text2)">
        ${depth} depth slices (${rows}×${cols})</text>`;
    for (let d=0; d<depth; d++) {
      const ox = d*(sliceW+gap), oy = 20;
      const pX = (c) => ox+8+(cols>1?c/(cols-1)*(sliceW-16):sliceW/2);
      const pY = (r) => oy+6+(rows>1?r/(rows-1)*(sliceH-12):sliceH/2);
      for (let r=0; r<rows; r++) for (let c=0; c<cols; c++) {
        if (c+1<cols) svg+=`<line x1="${pX(c)}" y1="${pY(r)}" x2="${pX(c+1)}" y2="${pY(r)}" stroke="var(--accent)" stroke-width="1" opacity="0.6"/>`;
        if (r+1<rows) svg+=`<line x1="${pX(c)}" y1="${pY(r)}" x2="${pX(c)}" y2="${pY(r+1)}" stroke="var(--accent)" stroke-width="1" opacity="0.6"/>`;
      }
      for (let r=0; r<rows; r++) for (let c=0; c<cols; c++)
        svg+=`<circle cx="${pX(c)}" cy="${pY(r)}" r="3" fill="var(--accent)" opacity="0.8"/>`;
      svg+=`<text x="${ox+sliceW/2}" y="${oy+sliceH+10}" text-anchor="middle" font-size="9" fill="var(--text3)">d=${d}</text>`;
    }
    svg += '</svg>';
    container.innerHTML = svg;
  },

  _forceLayout(n, edges, W, H, iters) {
    const pos = Array.from({length:n}, (_,i)=>({
      x: W/2 + Math.cos(2*Math.PI*i/n)*W*0.35,
      y: H/2 + Math.sin(2*Math.PI*i/n)*H*0.35,
    }));
    const k = Math.sqrt(W*H/Math.max(n,1));
    for (let iter=0; iter<iters; iter++) {
      const temp = 0.15*W*(1-iter/iters);
      const fx = new Float64Array(n), fy = new Float64Array(n);
      for (let i=0; i<n; i++) for (let j=i+1; j<n; j++) {
        const dx=pos[i].x-pos[j].x, dy=pos[i].y-pos[j].y;
        const d=Math.hypot(dx,dy)+0.01, f=k*k/d;
        fx[i]+=f*dx/d; fy[i]+=f*dy/d; fx[j]-=f*dx/d; fy[j]-=f*dy/d;
      }
      edges.forEach(([s,t])=>{
        if(s>=n||t>=n)return;
        const dx=pos[t].x-pos[s].x, dy=pos[t].y-pos[s].y;
        const d=Math.hypot(dx,dy)+0.01, f=d*d/k;
        fx[s]+=f*dx/d; fy[s]+=f*dy/d; fx[t]-=f*dx/d; fy[t]-=f*dy/d;
      });
      for (let i=0; i<n; i++) {
        const d=Math.hypot(fx[i],fy[i]);
        if(d>0){const m=Math.min(d,temp)/d; pos[i].x+=fx[i]*m; pos[i].y+=fy[i]*m;}
        pos[i].x=Math.max(12,Math.min(W-12,pos[i].x));
        pos[i].y=Math.max(12,Math.min(H-12,pos[i].y));
      }
    }
    return pos;
  },

  _buildSvg(pos, edges, directed, W, H, r=10) {
    const drawn = new Set();
    let edgesHtml = '';
    edges.forEach(([s,t]) => {
      if (s===t) return; // skip self-loops in preview
      const key = directed ? `${s}-${t}` : [Math.min(s,t),Math.max(s,t)].join('-');
      if (drawn.has(key)) return;
      drawn.add(key);
      const x1=pos[s]?.x??0, y1=pos[s]?.y??0, x2=pos[t]?.x??0, y2=pos[t]?.y??0;
      edgesHtml += directed
        ? `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="var(--border)" stroke-width="1.5" marker-end="url(#arr)"/>`
        : `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="var(--border)" stroke-width="1.5"/>`;
    });
    const nodesHtml = pos.map((p,i)=>
      `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r/2}" fill="var(--accent)" opacity="0.85"/>
       ${pos.length<=20?`<text x="${p.x.toFixed(1)}" y="${(p.y+r/6).toFixed(1)}" text-anchor="middle" font-size="${r/2-1}" fill="var(--bg)">${i}</text>`:''}`)
      .join('');
    return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px;display:block"
      xmlns="http://www.w3.org/2000/svg">
      <defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="var(--text3)"/>
      </marker></defs>
      ${edgesHtml}${nodesHtml}
    </svg>`;
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Polling and global UI updates
// ─────────────────────────────────────────────────────────────────────────────
async function poll() {
  if (document.hidden) return;
  if (S.paused) {
    Controls.updateStale();
    return;
  }

  // ── Metrics: use incremental API when we already have a base load. ──────
  let metricsPromise;
  if (S.latestRowIndex >= 0) {
    // Request only rows after the last known index.
    const opts = {sinceRow: S.latestRowIndex};
    if (S.activeRun) opts.run = S.activeRun;
    metricsPromise = API.metrics(opts).then(inc => {
      if (!inc) return null;  // network failure — keep existing data
      if (inc.reset_required) {
        // File was replaced/truncated — full reload needed.
        S.latestRowIndex = -1;
        return API.metrics(S.activeRun ? {run: S.activeRun} : {});
      }
      // Append new rows to existing data.
      if (S.metrics && inc.rows.length > 0) {
        const merged = {
          ...S.metrics,
          rows: [...S.metrics.rows, ...inc.rows],
          total_row_count: inc.total_row_count,
          truncated: inc.truncated,
        };
        // Trim to max_metric_rows to prevent unbounded growth.
        const cap = S.config?.max_metric_rows || 5000;
        if (merged.rows.length > cap) {
          merged.rows = merged.rows.slice(merged.rows.length - cap);
          merged.truncated = true;
        }
        S.latestRowIndex = inc.latest_row_index;
        // Return merged so S.metrics gets updated below.
        return merged;
      }
      if (inc.latest_row_index != null) {
        S.latestRowIndex = inc.latest_row_index;
      }
      return null;  // no new rows — keep existing
    });
  } else {
    // Full load.
    const opts = S.activeRun ? {run: S.activeRun} : {};
    metricsPromise = API.metrics(opts).then(m => {
      if (m && m.total_row_count != null) {
        S.latestRowIndex = m.total_row_count - 1;
      }
      return m;
    });
  }

  const [status, metrics, hardware, metadata, graph, graphStats] = await Promise.all([
    API.status(),
    metricsPromise,
    API.hardware(),
    API.metadata(),
    API.graph(),
    API.graphStats(),
  ]);

  // Keep the previous data when an individual fetch fails.
  const anySucceeded = (status || metrics || hardware || metadata || graph || graphStats);
  S.status     = status     || S.status;
  S.metrics    = metrics    || S.metrics;
  S.hardware   = hardware   || S.hardware;
  S.metadata   = metadata   || S.metadata;
  S.graph      = graph      || S.graph;
  S.graphStats = graphStats || S.graphStats;
  if (anySucceeded) {
    S.lastFetch = new Date().toISOString();
  }

  updateTopBar();
  Controls.updateStale();
  Render[S.sec] && Render[S.sec]();
  if (S.tvMode) TV.render();
}

function updateTopBar() {
  const st = S.status || {};

  // Run title — textContent prevents HTML injection from run_name.
  const titleEl = el('run-title');
  if (titleEl) titleEl.textContent = st.run_name || 'TGraphX Dashboard';

  // Status chip
  const chip = el('status-chip');
  if (chip) {
    const cls = {running:'chip-running',completed:'chip-completed',error:'chip-error'}[st.status]||'chip-unknown';
    chip.className = `chip ${cls}`;
    chip.textContent = st.status || '—';
  }

  // Footer last update
  const ago = T.ago(S.lastFetch);
  const updEl = el('update-txt');
  if (updEl) updEl.textContent = ago ? `Last update: ${ago}` : '—';
}

function startPolling() {
  if (S.pollTimer) clearInterval(S.pollTimer);
  if (S.paused) return;
  const ms = (S.config?.poll_ms) || CFG.pollMs;
  S.pollTimer = setInterval(poll, ms);
}

// ─────────────────────────────────────────────────────────────────────────────
// Viewer clock
// ─────────────────────────────────────────────────────────────────────────────
function tickClock() {
  const c = el('viewer-clock');
  if (c) c.textContent = T.now();
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialize
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Multi-run selector — shown only when /api/runs returns mode="multi"
// ─────────────────────────────────────────────────────────────────────────────
const RunSelector = {
  _bar: null,

  async init() {
    const runs = await API.runs();
    if (!runs) return;
    S.runs = runs;
    if (runs.mode !== 'multi' || !runs.runs.length) return;
    // First run is default.
    if (!S.activeRun) S.activeRun = runs.runs[0];
    this._render();
  },

  _render() {
    if (!S.runs || S.runs.mode !== 'multi') return;
    const wrap = el('main-wrap');
    if (!wrap) return;

    // Only create once.
    if (this._bar) { this._updateSelect(); return; }
    const bar = document.createElement('div');
    bar.className = 'run-selector-bar';
    bar.setAttribute('role', 'toolbar');
    bar.setAttribute('aria-label', 'Run selector');
    bar.innerHTML = `
      <label for="run-select">Run:</label>
      <select id="run-select" aria-label="Select active run">
        ${S.runs.runs.map(r =>
          `<option value="${esc(r)}"${r === S.activeRun ? ' selected' : ''}>${esc(r)}</option>`
        ).join('')}
      </select>
      <span class="run-count">${esc(S.runs.runs.length)} runs${S.runs.capped ? ' (capped)' : ''}</span>`;
    const content = el('content');
    wrap.insertBefore(bar, content);
    this._bar = bar;

    el('run-select')?.addEventListener('change', e => {
      const newRun = e.target.value;
      if (newRun !== S.activeRun) {
        S.activeRun = newRun;
        S.latestRowIndex = -1;  // force full reload for new run
        S.metrics = null;
        poll();
      }
    });
  },

  _updateSelect() {
    const sel = el('run-select');
    if (!sel) return;
    if (sel.value !== S.activeRun) sel.value = S.activeRun;
  },
};

function init() {
  Nav.init();
  Theme.init();
  Palette.init();
  Controls.init();
  TV.init();
  Tooltip.init();

  // Build initial section shells
  SECTIONS.forEach(({id}) => {
    const sec = el(`sec-${id}`);
    if (sec && !sec.innerHTML.trim()) {
      sec.innerHTML = `<h2 id="sec-${esc(id)}-title" class="page-title">${esc(id.charAt(0).toUpperCase()+id.slice(1))}</h2>
        <div class="chart-empty">Loading…</div>`;
    }
  });

  // Smoothing toggle
  document.addEventListener('change', e => {
    if (e.target.id === 'smooth-cb') { S.smooth = e.target.checked; Render[S.sec]?.(); }
  });

  // Pause polling when tab not visible
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearInterval(S.pollTimer);
    else if (!S.paused) { startPolling(); poll(); }
  });

  // Stale-data check — independent of poll() so we can still warn even when
  // every API call is failing.
  setInterval(() => Controls.updateStale(), 1000);

  // ── Snapshot mode: offline HTML export pre-loads data ────────────────────
  if (window.__TGXSNAP) {
    S.snapshotMode = true;
    S.paused = true;
    const snap = window.__TGXSNAP;
    S.metrics    = snap.metrics    || null;
    S.metadata   = snap.metadata   || null;
    S.graph      = snap.graph      || null;
    S.graphStats = snap.graph_stats || null;
    // Set latestRowIndex so stale warning shows "offline snapshot".
    S.latestRowIndex = snap.metrics?.total_row_count
      ? snap.metrics.total_row_count - 1 : -1;
    S.lastFetch = new Date().toISOString();
    Nav.go(S.sec);
    return;  // skip live polling entirely
  }

  // Fetch server config first, then load data and start polling.  This lets
  // the server inform the browser of its preferred refresh interval.
  API.config().then(cfg => {
    S.config = cfg || null;
    // Fetch runs list in parallel — needed to know if multi-run mode.
    return Promise.all([poll(), RunSelector.init()]);
  }).then(() => {
    Nav.go(S.sec);
    startPolling();
  });

  setInterval(tickClock, 1000);
  tickClock();
}

// ─────────────────────────────────────────────────────────────────────────────
// Mining panel renderer (v0.4.2+)
// ─────────────────────────────────────────────────────────────────────────────

// State for lazy-loaded mining artifacts.
const MiningState = {
  summary: null, motifs: null, anomaly: null, communities: null,
  prototype: null, neural: null, reproducibility: null, linkPred: null,
  loaded: false,
};

// Safe number formatter for mining values.
function mfmt(v, decimals = 4) {
  if (v == null || v === '') return '—';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  const n = parseFloat(v);
  return isNaN(n) ? esc(String(v)) : n.toFixed(decimals);
}

// Render a simple key-value table from a plain object.
function kvTable(obj, title) {
  if (!obj || typeof obj !== 'object') return '';
  const skip = k => k.startsWith('_') || typeof obj[k] === 'object' || Array.isArray(obj[k]);
  const rows = Object.entries(obj)
    .filter(([k]) => !skip(k))
    .map(([k, v]) =>
      `<tr><td class="kv-key">${esc(k.replace(/_/g,' '))}</td>`+
      `<td class="kv-val">${esc(String(v))}</td></tr>`)
    .join('');
  if (!rows) return '';
  const hdr = title ? `<caption class="kv-caption">${esc(title)}</caption>` : '';
  return `<table class="kv-table">${hdr}<tbody>${rows}</tbody></table>`;
}

// Render a list of objects as a mini table with capped rows.
function listTable(arr, keys, caption, maxRows = 20) {
  if (!Array.isArray(arr) || arr.length === 0) return '<p class="empty-note">No data.</p>';
  const cols = keys || Object.keys(arr[0] || {});
  const header = cols.map(c => `<th>${esc(c.replace(/_/g,' '))}</th>`).join('');
  const sliced = arr.slice(0, maxRows);
  const rows = sliced.map(r =>
    `<tr>${cols.map(c => `<td>${esc(String(r[c] ?? '—'))}</td>`).join('')}</tr>`
  ).join('');
  const trunc = arr.length > maxRows
    ? `<caption class="kv-caption">${esc(caption||'')} (showing ${maxRows} of ${arr.length})</caption>` : '';
  return `<table class="kv-table"><thead><tr>${header}</tr></thead><tbody>${rows}</tbody>${trunc}</table>`;
}

// Render a bar chart using inline SVG for small data.
function miniBarChart(data, title, color) {
  if (!data || data.length === 0) return '';
  const maxVal = Math.max(...data.map(d => d.value || 0));
  if (maxVal === 0) return '';
  const W = 240, H = 80, pad = 6;
  const barW = Math.max(2, Math.floor((W - 2 * pad) / data.length) - 2);
  const bars = data.map((d, i) => {
    const h = Math.max(2, Math.round((d.value / maxVal) * (H - 2 * pad)));
    const x = pad + i * (barW + 2);
    const y = H - pad - h;
    const c = color || '#56B4E9';
    return `<rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="${c}" rx="1">
      <title>${esc(d.label)}: ${esc(String(d.value))}</title></rect>`;
  }).join('');
  const labels = data.slice(0, 8).map((d, i) => {
    const x = pad + i * (barW + 2) + barW / 2;
    return `<text x="${x}" y="${H}" text-anchor="middle" font-size="7" fill="var(--text2)">${esc(d.label.slice(0,6))}</text>`;
  }).join('');
  return `<div class="mining-chart-wrap">
    ${title ? `<p class="chart-label">${esc(title)}</p>` : ''}
    <svg viewBox="0 0 ${W} ${H + 12}" style="width:100%;max-width:${W}px;overflow:visible">
      ${bars}${labels}
    </svg></div>`;
}

// ── Mining section main renderer ─────────────────────────────────────────────
Render.mining = async function() {
  const sec = el('sec-mining');
  sec.innerHTML = `<h2 class="page-title">Graph Mining</h2><p class="loading-note">Loading mining artifacts…</p>`;

  // Fetch all mining artifacts in parallel.
  const fetches = {
    summary:        API.get('/api/mining_summary'),
    motifs:         API.get('/api/motif_summary'),
    anomaly:        API.get('/api/anomaly_summary'),
    communities:    API.get('/api/community_summary'),
    prototype:      API.get('/api/prototype_membership'),
    neural:         API.get('/api/neural_mining'),
    reproducibility:API.get('/api/reproducibility'),
    linkPred:       API.get('/api/link_prediction_summary'),
  };
  const results = {};
  for (const [k, p] of Object.entries(fetches)) {
    try { results[k] = await p; } catch(e) { results[k] = null; }
  }

  // ── Overview panel ─────────────────────────────────────────────────────────
  const s = results.summary || {};
  const hasAnySummary = s.num_nodes != null;

  let overviewHtml = '';
  if (hasAnySummary) {
    const cards = [
      card('Nodes', esc(String(s.num_nodes ?? '—')), 'total nodes', ''),
      card('Edges', esc(String(s.num_edges ?? '—')), 'total edges', ''),
      card('Density', mfmt(s.density), s.directed ? 'directed' : 'undirected', 'card-accent'),
      card('Components', esc(String(s.num_connected_components ?? '—')), 'connected', ''),
    ].join('');
    const degStats = s.mean_total_degree != null
      ? `<p>Mean degree: <strong>${mfmt(s.mean_total_degree, 2)}</strong> &nbsp;
         Max: <strong>${esc(String(s.max_total_degree ?? '—'))}</strong> &nbsp;
         Isolated: <strong>${esc(String(s.isolated_node_count ?? '—'))}</strong></p>`
      : '';
    const warns = (s.warnings || []).map(w =>
      `<p class="warn-note">⚠ ${esc(w)}</p>`).join('');
    overviewHtml = `<section class="mining-panel">
      <h3 class="panel-title">Graph Overview</h3>
      <div class="grid-4">${cards}</div>
      ${degStats}${warns}
    </section>`;
  }

  // ── Motifs panel ───────────────────────────────────────────────────────────
  const m = results.motifs || {};
  let motifsHtml = '';
  if (m.triangles != null || m.wedges != null) {
    const motifData = [
      {label:'Triangles', value: m.triangles || 0},
      {label:'Wedges',    value: m.wedges    || 0},
    ];
    const chart = miniBarChart(motifData, 'Motif counts', '#009E73');
    motifsHtml = `<section class="mining-panel">
      <h3 class="panel-title">Motifs / Structural</h3>
      <div class="mining-cols">
        <div>${kvTable(m, '')}</div>
        <div>${chart}</div>
      </div>
    </section>`;
  }

  // ── Anomaly panel ──────────────────────────────────────────────────────────
  const an = results.anomaly || {};
  let anomalyHtml = '';
  if (an.top_anomalous_nodes) {
    const topNodes = an.top_anomalous_nodes || [];
    const metaRows = [
      an.method ? `<tr><td class="kv-key">Method</td><td class="kv-val">${esc(an.method)}</td></tr>` : '',
      an.threshold != null ? `<tr><td class="kv-key">Threshold</td><td class="kv-val">${mfmt(an.threshold)}</td></tr>` : '',
      an.num_flagged != null ? `<tr><td class="kv-key">Flagged nodes</td><td class="kv-val">${esc(String(an.num_flagged))}</td></tr>` : '',
    ].join('');
    const metaTable = metaRows ? `<table class="kv-table"><tbody>${metaRows}</tbody></table>` : '';
    const tbl = listTable(topNodes, ['node_id','score'], 'Top anomalous nodes', 20);
    anomalyHtml = `<section class="mining-panel">
      <h3 class="panel-title">Anomaly Detection</h3>
      <div class="mining-cols">
        <div>${metaTable}</div>
        <div>${tbl}</div>
      </div>
    </section>`;
  }

  // ── Communities panel ──────────────────────────────────────────────────────
  const cm = results.communities || {};
  let communityHtml = '';
  if (cm.num_communities != null) {
    const cmRows = [
      {key:'Communities', val: cm.num_communities},
      {key:'Modularity',  val: mfmt(cm.modularity)},
      {key:'Largest',     val: cm.largest_community_size},
      {key:'Smallest',    val: cm.smallest_community_size},
    ].map(r => `<tr><td class="kv-key">${esc(r.key)}</td><td class="kv-val">${esc(String(r.val ?? '—'))}</td></tr>`).join('');
    communityHtml = `<section class="mining-panel">
      <h3 class="panel-title">Communities</h3>
      <table class="kv-table"><tbody>${cmRows}</tbody></table>
    </section>`;
  }

  // ── Prototype membership panel ─────────────────────────────────────────────
  const pr = results.prototype || {};
  let protoHtml = '';
  if (pr.accuracy != null || pr.classification_report) {
    const metricRows = [
      {key:'Accuracy',          val: mfmt(pr.accuracy)},
      {key:'Balanced accuracy', val: mfmt(pr.balanced_accuracy)},
      {key:'Num queries',       val: pr.num_queries},
      {key:'Num classes',       val: pr.num_classes},
    ].map(r => `<tr><td class="kv-key">${esc(r.key)}</td><td class="kv-val">${esc(String(r.val ?? '—'))}</td></tr>`).join('');
    const confPairs = (pr.top_confusion_pairs || []).slice(0, 5);
    const confTable = confPairs.length > 0
      ? listTable(confPairs, ['true','pred','count'], 'Top confusion pairs', 10)
      : '';
    protoHtml = `<section class="mining-panel">
      <h3 class="panel-title">Prototype Membership</h3>
      <div class="mining-cols">
        <table class="kv-table"><tbody>${metricRows}</tbody></table>
        <div>${confTable}</div>
      </div>
    </section>`;
  }

  // ── Neural mining panel ────────────────────────────────────────────────────
  const nm = results.neural || {};
  let neuralHtml = '';
  if (nm.tasks || nm.loss_decreased != null) {
    const taskRows = Object.entries(nm.tasks || {}).map(([name, t]) =>
      `<tr>
        <td class="kv-key">${esc(name.replace(/_/g,' '))}</td>
        <td class="kv-val">${mfmt(t.initial_loss)} → ${mfmt(t.final_loss)}</td>
        <td class="kv-val">${t.loss_decreased ? '✓' : '✗'}</td>
        <td class="kv-val">${mfmt(t.train_time_s)}s</td>
      </tr>`
    ).join('');
    neuralHtml = `<section class="mining-panel">
      <h3 class="panel-title">Neural Mining</h3>
      <table class="kv-table">
        <thead><tr><th>Task</th><th>Loss (start→end)</th><th>↓</th><th>Time</th></tr></thead>
        <tbody>${taskRows}</tbody>
      </table>
    </section>`;
  }

  // ── Reproducibility panel ──────────────────────────────────────────────────
  const rep = results.reproducibility || {};
  let reproHtml = '';
  if (rep.seed != null || rep.torch_version) {
    reproHtml = `<section class="mining-panel">
      <h3 class="panel-title">Reproducibility</h3>
      ${kvTable(rep, '')}
    </section>`;
  }

  // ── Empty state ────────────────────────────────────────────────────────────
  const hasContent = overviewHtml || motifsHtml || anomalyHtml ||
    communityHtml || protoHtml || neuralHtml || reproHtml;
  const emptyNote = hasContent ? '' : `<div class="empty-panel">
    <p>No mining artifacts found in this run directory.</p>
    <p class="help-note">To generate mining artifacts, use
    <code>tgraphx.mining.write_graph_mining_summary()</code> or
    <code>tgraphx.mining.graph_mining_report()</code>.</p>
  </div>`;

  sec.innerHTML = `
    <h2 id="sec-mining-title" class="page-title">Graph Mining</h2>
    ${overviewHtml}${motifsHtml}${anomalyHtml}${communityHtml}
    ${protoHtml}${neuralHtml}${reproHtml}${emptyNote}
  `;
};

// Patch API helper to support new mining endpoints.
(function patchAPI() {
  const origGet = API && typeof API.get === 'function' ? API.get.bind(API) : null;
  if (!origGet) return;
  // The existing API.get already calls /api/<endpoint>; no patching needed.
  // Ensure new endpoints don't throw — they return {} on missing files.
})();

document.addEventListener('DOMContentLoaded', init);
