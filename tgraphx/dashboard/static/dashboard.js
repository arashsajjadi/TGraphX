/* TGraphX Dashboard — Vanilla JS frontend
   No external dependencies. Runs in any modern browser.
   All timestamps treated as UTC; displayed in viewer's local timezone. */
'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
const CFG = {
  pollMs:      2000,
  smoothAlpha: 0.3,
  maxNodes:    200,
  maxEdges:    1000,
  forceIter:   80,
  etaMinPts:   5,
};

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let S = {
  sec:       'overview',
  status:    null,
  metrics:   null,
  hardware:  null,
  metadata:  null,
  graph:     null,
  smooth:    false,
  tvMode:    false,
  pollTimer: null,
  lastFetch: null,
};

const SECTIONS = [
  { id:'overview', label:'Overview',  icon:'▤'  },
  { id:'metrics',  label:'Metrics',   icon:'⬡'  },
  { id:'graph',    label:'Graph',     icon:'◈'  },
  { id:'hardware', label:'Hardware',  icon:'⚙'  },
  { id:'logs',     label:'Logs',      icon:'≡'  },
  { id:'config',   label:'Config',    icon:'❐'  },
  { id:'about',    label:'About',     icon:'ℹ'  },
];

const CHART_META = {
  train_loss:      { label:'Train Loss',      color:'#06b6d4' },
  val_loss:        { label:'Validation Loss', color:'#8b5cf6' },
  loss:            { label:'Loss',            color:'#06b6d4' },
  accuracy:        { label:'Accuracy',        color:'#22c55e' },
  acc:             { label:'Accuracy',        color:'#22c55e' },
  learning_rate:   { label:'Learning Rate',   color:'#f59e0b' },
  lr:              { label:'Learning Rate',   color:'#f59e0b' },
  grad_norm:       { label:'Grad Norm',       color:'#ef4444' },
  samples_per_sec: { label:'Samples/s',       color:'#10b981' },
  graphs_per_sec:  { label:'Graphs/s',        color:'#10b981' },
  steps_per_sec:   { label:'Steps/s',         color:'#10b981' },
};

const SKIP_COLS = new Set(['epoch','step','timestamp','time']);

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
  status()   { return this.get('/api/status'); },
  metrics()  { return this.get('/api/metrics'); },
  hardware() { return this.get('/api/hardware'); },
  metadata() { return this.get('/api/metadata'); },
  graph()    { return this.get('/api/graph'); },
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
      btn.className = 'nav-btn' + (id === S.sec ? ' active' : '');
      btn.dataset.sec = id;
      btn.innerHTML = `<span class="nav-icon">${icon}</span>${label}`;
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
      b.classList.toggle('active', b.dataset.sec === secId);
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
        <div class="tv-value" style="font-size:2rem">${st.status||'—'}</div>
        <div class="tv-sub">${st.run_name||''}</div></div>
      <div class="tv-card tv-chart" id="tv-chart-wrap"></div>
      <div class="tv-card"><div class="tv-label">Train Loss</div>
        <div class="tv-value">${fmt(loss)}</div>
        <div class="tv-sub">Epoch ${st.epoch||'—'}${st.total_epochs?' / '+st.total_epochs:''}</div></div>
      <div class="tv-card"><div class="tv-label">Elapsed / ETA</div>
        <div class="tv-value" style="font-size:2.2rem">${T.elapsed(elapsed)}</div>
        <div class="tv-sub">ETA: ${eta!=null?(eta<10?'Estimating…':T.elapsed(eta)):'—'}</div></div>`;

    if (lossKey && series[lossKey]) {
      const wrap = el('tv-chart-wrap');
      const c = new SvgChart(wrap, {color: CHART_META[lossKey]?.color||'#06b6d4',
        label: CHART_META[lossKey]?.label||lossKey, height: wrap.clientHeight||300});
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
        `<span class="chip ${statusCls}">${st.status||'unknown'}</span>`,
        st.run_name||st.task||'', 'card-accent'),
      card('Progress',
        totalEp
          ? `${epoch} <span style="font-size:.9rem;color:var(--text2)">/ ${totalEp}</span>`
          : String(epoch),
        progress != null
          ? `<div class="progress-outer"><div class="progress-inner" style="width:${progress.toFixed(1)}%"></div></div>`
          : 'epochs', ''),
      card('Elapsed', T.elapsed(elapsed),
        `ETA: ${eta != null ? (eta < 10 ? 'done' : T.elapsed(eta)) : (elapsed && elapsed > 0 && m.rows?.length < CFG.etaMinPts ? 'estimating…' : '—')}`, ''),
      card('Train Loss', fmt(loss),
        vLoss != null ? `val loss: ${fmt(vLoss)}` : (acc != null ? `acc: ${fmt(acc)}` : ''), 'card-accent'),
    ].join('');

    const hw1Html = [
      card('Device', st.device||hw.cuda_available?'CUDA':'CPU', hw.torch?`PyTorch ${hw.torch}`:'', ''),
      card('LR', fmt(lr), lr != null ? 'learning rate' : '', 'card-amber'),
      card('Last update', T.ago(st.last_update)||'—',
        st.last_update ? `at ${T.local(st.last_update)}` : '', ''),
    ].join('');

    sec.innerHTML = `
      <div class="page-title">Overview</div>
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
    const series = metricsToSeries(S.metrics);
    const lossKey = ['train_loss','loss'].find(k => series[k]?.length);
    const valKey  = ['val_loss'].find(k => series[k]?.length);
    const container = el('ov-chart');

    if (lossKey && container) {
      container.style.height = '180px';
      // Multi-series: draw val on same SVG by overlaying
      const c = new SvgChart(container, {
        color: CHART_META[lossKey]?.color || '#06b6d4',
        label: 'Loss',
        height: 180,
      });
      c.render(series[lossKey], S.smooth);

      // Overlay val loss as a separate polyline
      if (valKey && series[valKey]?.length) {
        const existingSvg = container.querySelector('svg');
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
          line.setAttribute('stroke', '#8b5cf6');
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
          label.setAttribute('fill', '#8b5cf6');
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
    const series = metricsToSeries(S.metrics);
    const keys = Object.keys(series);
    const truncNote = S.metrics?.truncated
      ? `<div class="page-note" style="color:var(--amber);margin-bottom:10px">
           &#9888; Showing latest ${S.metrics.rows?.length ?? 0} of
           ${S.metrics.total_row_count} rows.
           Raw <code>metrics.csv</code> is unchanged.
         </div>`
      : '';

    sec.innerHTML = `
      <div class="page-title">Metrics</div>
      ${truncNote}
      <div class="smoothing-row">
        <label><input type="checkbox" id="smooth-cb"${S.smooth?' checked':''}> Smooth curves
          <span class="label-note">(exponential moving average, α=${CFG.smoothAlpha})</span></label>
      </div>
      <div id="metrics-grid"></div>`;

    const cb = el('smooth-cb');
    if (cb) cb.addEventListener('change', () => { S.smooth = cb.checked; Render.metrics(); });

    const grid = el('metrics-grid');
    if (!keys.length || !S.metrics?.rows?.length) {
      grid.innerHTML = emptyState('No metric data yet.', 'Start training and data will appear here.');
      return;
    }

    keys.forEach(key => {
      if (!series[key] || !series[key].length) return;
      const meta = CHART_META[key] || {label: key, color: '#06b6d4'};
      const card = document.createElement('div');
      card.className = 'card chart-card';
      card.innerHTML = `<div class="chart-title">${meta.label}</div>
        <div class="chart-container" style="height:180px"></div>`;
      grid.appendChild(card);
      const c = new SvgChart(card.querySelector('.chart-container'),
        {color: meta.color, label: meta.label, height: 180});
      c.render(series[key], S.smooth);
    });
  },

  // ── Graph ────────────────────────────────────────────────────────────────────
  graph() {
    const sec = el('sec-graph');
    const g   = S.graph;

    if (!g || !g.available) {
      sec.innerHTML = `<div class="page-title">Graph</div>
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
             title="degree ${i}: ${v} nodes"></div>`).join('')}</div>
         <div class="card-sub">min ${degStats.min??'—'} · max ${degStats.max??'—'} · mean ${fmt(degStats.mean)}</div>`
      : `<div class="card-title">Degree stats</div>
         <div class="hw-row"><span class="hw-label">Min</span><span class="hw-val">${degStats.min??'—'}</span></div>
         <div class="hw-row"><span class="hw-label">Max</span><span class="hw-val">${degStats.max??'—'}</span></div>
         <div class="hw-row"><span class="hw-label">Mean</span><span class="hw-val">${fmt(degStats.mean)}</span></div>`;

    const builderHtml = g.builder
      ? `<div class="hw-row"><span class="hw-label">Builder</span>
           <span class="hw-val">${g.builder}</span></div>
         ${g.builder_params ? Object.entries(g.builder_params).map(([k,v])=>
           `<div class="hw-row"><span class="hw-label">${k}</span><span class="hw-val">${v}</span></div>`
         ).join('') : ''}`
      : '<div class="card-sub">No builder metadata</div>';

    sec.innerHTML = `
      <div class="page-title">Graph</div>
      <div class="grid-2" style="margin-bottom:16px">
        <div class="card">
          <div class="card-title">Summary</div>
          ${stats.map(([k,v])=>`<div class="hw-row"><span class="hw-label">${k}</span>
            <span class="hw-val">${v}</span></div>`).join('')}
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

    const cpuHtml = hw.psutil_available ? `
      <div class="card">
        <div class="card-title">CPU &amp; Memory</div>
        <div class="hw-row">
          <span class="hw-label">CPU</span>
          <div class="hw-bar-wrap"><div class="hw-bar hw-bar-cpu" style="width:${hw.cpu_percent||0}%"></div></div>
          <span class="hw-val">${(hw.cpu_percent??'—')}%</span>
        </div>
        <div class="hw-row">
          <span class="hw-label">RAM</span>
          <div class="hw-bar-wrap"><div class="hw-bar hw-bar-ram" style="width:${hw.ram_percent||0}%"></div></div>
          <span class="hw-val">${(hw.ram_percent??'—')}%</span>
        </div>
        <div class="card-sub">${(hw.ram_used_gb??'—')} / ${(hw.ram_total_gb??'—')} GB</div>
        ${hw.process_ram_mb != null ? `<div class="card-sub">This process: ${hw.process_ram_mb} MB</div>` : ''}
      </div>` : `<div class="card"><div class="card-title">CPU &amp; Memory</div>
        <div class="card-sub">Install psutil for CPU/RAM monitoring:<br><code>pip install psutil</code></div></div>`;

    const gpuHtml = hw.cuda_available ? `
      <div class="card">
        <div class="card-title">CUDA GPU</div>
        <div class="hw-row"><span class="hw-label">Device</span>
          <span class="hw-val" style="font-size:.78rem">${hw.cuda_device_name||'—'}</span></div>
        <div class="hw-row"><span class="hw-label">VRAM used</span>
          <span class="hw-val">${hw.cuda_mem_allocated_mb??'—'} MB</span></div>
        <div class="hw-row"><span class="hw-label">VRAM reserved</span>
          <span class="hw-val">${hw.cuda_mem_reserved_mb??'—'} MB</span></div>
        <div class="hw-row"><span class="hw-label">VRAM total</span>
          <span class="hw-val">${hw.cuda_mem_total_mb??'—'} MB</span></div>
        ${hw.gpu_util_pct != null
          ? `<div class="hw-row"><span class="hw-label">Utilization</span>
              <div class="hw-bar-wrap"><div class="hw-bar hw-bar-gpu" style="width:${hw.gpu_util_pct}%"></div></div>
              <span class="hw-val">${hw.gpu_util_pct}%</span></div>` : ''}
        ${hw.gpu_temp_c != null
          ? `<div class="hw-row"><span class="hw-label">Temperature</span>
              <span class="hw-val">${hw.gpu_temp_c}°C</span></div>` : ''}
      </div>` : hw.mps_available ? `
      <div class="card"><div class="card-title">Apple MPS</div>
        <div class="card-sub">Apple Silicon MPS backend is available.</div></div>`
      : `<div class="card"><div class="card-title">GPU</div>
        <div class="card-sub">No CUDA GPU detected.</div></div>`;

    sec.innerHTML = `
      <div class="page-title">Hardware &amp; Environment</div>
      <p class="page-note">Monitoring is best-effort and depends on optional system packages.
        Missing sensor data does not indicate a training problem.</p>
      <div class="grid-2" style="margin-bottom:16px">
        <div class="card">
          <div class="card-title">Versions &amp; Environment</div>
          ${verRows.map(([k,v])=>`<div class="hw-row">
            <span class="hw-label">${k}</span><span class="hw-val">${v}</span></div>`).join('')}
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
      <div class="page-title">Metrics Log <span style="font-size:.8rem;color:var(--text3)">(last 50 rows, newest first)</span></div>
      <div class="card" style="padding:0">
        <div class="tbl-wrap">
          <table>
            <thead><tr>${hdrs.map(h=>`<th>${h}</th>`).join('')}</tr></thead>
            <tbody>${rows.map(row=>`<tr>${row.map((v,i)=>{
              if (i===tsIdx && typeof v==='string') return `<td>${T.local(v)}</td>`;
              return `<td>${typeof v==='number'?fmt(v):v}</td>`;
            }).join('')}</tr>`).join('')}</tbody>
          </table>
        </div>
      </div>`;
  },

  // ── Config ────────────────────────────────────────────────────────────────────
  config() {
    const sec = el('sec-config');
    const meta = S.metadata;

    if (!meta || !Object.keys(meta).length) {
      sec.innerHTML = `<div class="page-title">Run Configuration</div>
        ${emptyState('No run_metadata.json found.',
          'Create a run_metadata.json in your logdir with training configuration.')}`;
      return;
    }

    sec.innerHTML = `
      <div class="page-title">Run Configuration</div>
      <div class="card" style="padding:0">
        <pre>${JSON.stringify(meta, null, 2)}</pre>
      </div>`;
  },

  // ── About ─────────────────────────────────────────────────────────────────────
  about() {
    const sec  = el('sec-about');
    const hw   = S.hardware || {};
    sec.innerHTML = `
      <div class="page-title">About</div>
      <div class="grid-2">
        <div class="card">
          <div class="card-title">TGraphX Dashboard</div>
          <div class="hw-row"><span class="hw-label">TGraphX</span><span class="hw-val">${hw.tgraphx||'—'}</span></div>
          <div class="hw-row"><span class="hw-label">Python</span><span class="hw-val">${hw.python||'—'}</span></div>
          <div class="hw-row"><span class="hw-label">PyTorch</span><span class="hw-val">${hw.torch||'—'}</span></div>
          <div class="hw-row"><span class="hw-label">Platform</span><span class="hw-val">${hw.platform||'—'}</span></div>
          <div class="card-sub" style="margin-top:12px">
            Local-first monitoring · No external dependencies · Read-only
          </div>
        </div>
        <div class="card">
          <div class="card-title">Usage</div>
          <pre style="font-size:.75rem">tgraphx-dashboard --logdir runs/demo

# LAN mode (token required)
tgraphx-dashboard --logdir runs/demo \\
  --host 0.0.0.0 --token MY_TOKEN

# Python API
from tgraphx.dashboard import launch_dashboard
launch_dashboard("runs/demo")</pre>
        </div>
      </div>`;
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

  const [status, metrics, hardware, metadata, graph] = await Promise.all([
    API.status(), API.metrics(), API.hardware(), API.metadata(), API.graph(),
  ]);

  S.status   = status   || S.status;
  S.metrics  = metrics  || S.metrics;
  S.hardware = hardware || S.hardware;
  S.metadata = metadata || S.metadata;
  S.graph    = graph    || S.graph;
  S.lastFetch = new Date().toISOString();

  updateTopBar();
  Render[S.sec] && Render[S.sec]();
  if (S.tvMode) TV.render();
}

function updateTopBar() {
  const st = S.status || {};

  // Run title
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
  S.pollTimer = setInterval(poll, CFG.pollMs);
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
function init() {
  Nav.init();
  Theme.init();
  TV.init();

  // Build initial section shells
  SECTIONS.forEach(({id}) => {
    const sec = el(`sec-${id}`);
    if (sec && !sec.innerHTML.trim()) {
      sec.innerHTML = `<div class="page-title">${id.charAt(0).toUpperCase()+id.slice(1)}</div>
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
    else { startPolling(); poll(); }
  });

  // Initial data load then start polling
  poll().then(() => {
    Nav.go(S.sec);
    startPolling();
  });

  setInterval(tickClock, 1000);
  tickClock();
}

document.addEventListener('DOMContentLoaded', init);
