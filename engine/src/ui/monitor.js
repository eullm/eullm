// EuLLM Monitor — the requests the engine is answering and has answered, the
// machine it runs on, and their history.
//
// The engine samples itself every second from the moment it starts, whether
// this page is open or not, keeps the last hour of seconds in memory and a row
// a minute (lowest, mean, highest) on disk. This page reads the seconds it
// does not have yet (/api/monitor?since=), so a tab that was hidden or closed
// catches up instead of starting over, and reads the minutes for the longer
// ranges (/api/monitor/history). Deleting the history or changing how long it
// is kept needs the server's admin key, which this page keeps in memory only.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    view: $("monitor"),
    live: $("mon-live"),
    ranges: $("mon-ranges"),
    models: $("mon-models"),
    engineNow: $("mon-engine-now"),
    engineChart: $("mon-engine-chart"),
    engineStats: $("mon-engine-stats"),
    machine: $("mon-machine"),
    cores: $("mon-machine-cores"),
    gpus: $("mon-gpus"),
    events: $("mon-events"),
    eventsCount: $("mon-events-count"),
    requests: $("mon-requests"),
    reqCount: $("mon-req-count"),
    storage: $("mon-storage"),
    storageMeta: $("mon-storage-meta"),
  };
  if (!els.view) return;

  // The key app.js keeps for this tab (see its API key comment).
  const apiHeaders = () => {
    let key = "";
    try { key = sessionStorage.getItem("eullm.apiKey") || ""; } catch {}
    return key ? { "X-Api-Key": key } : {};
  };
  const auth = () => ({ headers: apiHeaders() });

  const MIN = 60_000, HOUR = 60 * MIN, DAY = 24 * HOUR;
  // Up to an hour the seconds the engine keeps; beyond, its minute rows.
  const RANGES = [
    { key: "15m", label: "15 min", span: 15 * MIN, live: true },
    { key: "1h", label: "1 hour", span: HOUR, live: true },
    { key: "24h", label: "24 hours", span: DAY },
    { key: "7d", label: "7 days", span: 7 * DAY },
    { key: "30d", label: "30 days", span: 30 * DAY },
  ];
  let range = RANGES[0];
  try { range = RANGES.find((r) => r.key === localStorage.getItem("eullm.monitorRange")) || range; } catch {}

  let samples = [];   // the engine's seconds, oldest first, at most an hour
  let lastT = 0;      // the newest sample's time: what `since` asks after
  let history = null; // the minute rows of a longer range, and its events
  let events = [];    // the events of the range shown
  let storage = null;
  let adminKey = "";  // this page's memory only: gone on reload
  let running = [];
  let reqs = { active: [], history: [] };
  let timer = null, psTimer = null, slowTimer = null;

  // ── Formatting ────────────────────────────────────────────────────────
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const GiB = 2 ** 30;
  const gib = (b) => (b == null ? "–" : (b / GiB).toFixed(b >= 10 * GiB ? 1 : 2));
  const rateBytes = (bps) => {
    if (bps == null) return "–";
    if (bps >= 1e9) return `${(bps / 1e9).toFixed(2)} GB/s`;
    if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)} MB/s`;
    if (bps >= 1e3) return `${(bps / 1e3).toFixed(0)} kB/s`;
    return `${Math.round(bps)} B/s`;
  };
  const size = (b) => {
    if (b == null) return "–";
    if (b >= GiB) return `${(b / GiB).toFixed(2)} GiB`;
    if (b >= 2 ** 20) return `${(b / 2 ** 20).toFixed(1)} MiB`;
    return `${Math.max(0, Math.round(b / 1024))} KiB`;
  };
  const num = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));
  const ms = (v) => (v == null ? "–" : v >= 10000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
  const clock = (t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  // A time as precise as the range needs: seconds within the hour, the day
  // beyond a day.
  const when = (t, span = range.span) => new Date(t).toLocaleString([], span > DAY
    ? { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }
    : span > HOUR ? { hour: "2-digit", minute: "2-digit" } : { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const pct = (used, total) => (used != null && total ? Math.min(100, (used / total) * 100) : null);
  const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
  const formatters = {
    tps: (v) => `${num(v)} tok/s`,
    pct: (v) => `${num(v, 0)}%`,
    bytes: rateBytes,
    gib: (v) => `${gib(v)} GiB`,
    celsius: (v) => `${num(v, 0)} °C`,
    watts: (v) => `${num(v, 0)} W`,
    count: (v) => num(v, v % 1 ? 1 : 0),
  };

  // ── Series ────────────────────────────────────────────────────────────
  // A point is {t, v} from a second, or {t, v, lo, hi} from a minute row
  // (or a longer step): the mean, and the lowest and highest in it.
  const sec = (f) => samples.map((s) => ({ t: s.t_ms, v: f(s) }));
  const trio = (f) => (history?.rows || []).map((r) => {
    const x = f(r);
    return x ? { t: r.t, v: x[1], lo: x[0], hi: x[2] } : { t: r.t, v: null };
  });
  // The same figure from whichever the range reads.
  const series = (fromSample, fromRow) => (range.live ? sec(fromSample) : trio(fromRow));
  const bounds = () => {
    const to = range.live ? (lastT || Date.now()) : (history?.to || Date.now());
    return { from: to - range.span, to };
  };
  // Two points further apart than this are not joined: the server was not
  // running, or a GPU was not read, in between.
  const gapAfter = () => (range.live ? 5000 : Math.max(3 * MIN, 3 * (history?.step || 60) * 1000));

  // ── Charts ────────────────────────────────────────────────────────────
  // One figure on one axis from zero over the range, the mean as a line and,
  // for minute rows, the lowest-highest band around it; the highest point is
  // marked, and so are the events. The line stretches with the panel; the
  // dot, labels and hover readout are HTML, so they keep their shape.
  const VB_W = 600;
  function chart(points, { height = 72, max = 0, fmt = "tps", marks = false } = {}) {
    const { from, to } = bounds();
    const span = to - from;
    const known = points.filter((p) => p.v != null && p.t >= from - 60_000 && p.t <= to);
    const top = niceTop(Math.max(max, ...known.map((p) => p.hi ?? p.v), 0));
    const x = (t) => Math.min(VB_W, Math.max(0, ((t - from) / span) * VB_W));
    const y = (v) => height - 1 - (Math.min(v, top) / top) * (height - 4);
    const gap = gapAfter();
    const runs = [];
    for (const p of known) {
      const run = runs[runs.length - 1];
      if (run && p.t - run[run.length - 1].t <= gap) run.push(p); else runs.push([p]);
    }
    let paths = "";
    for (const run of runs) {
      if (run.length < 2) continue;
      const line = run.map((p) => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
      if (run[0].hi != null) {
        const upper = run.map((p) => `${x(p.t).toFixed(1)},${y(p.hi).toFixed(1)}`);
        const lower = run.slice().reverse().map((p) => `${x(p.t).toFixed(1)},${y(p.lo).toFixed(1)}`);
        paths += `<polygon class="chart-band" points="${upper.concat(lower).join(" ")}"/>`;
      } else {
        paths += `<polygon class="chart-area" points="${x(run[0].t).toFixed(1)},${height} ${line} ${x(run[run.length - 1].t).toFixed(1)},${height}"/>`;
      }
      paths += `<polyline class="chart-line" points="${line}"/>`;
    }
    let overlay = "";
    const last = known[known.length - 1];
    if (last && range.live) {
      overlay += `<span class="chart-dot" style="left:${(x(last.t) / VB_W) * 100}%;top:${(y(last.v) / height) * 100}%"></span>`;
    }
    if (marks) {
      const peak = known.reduce((a, p) => ((p.hi ?? p.v) > (a ? a.hi ?? a.v : -1) ? p : a), null);
      if (peak && (peak.hi ?? peak.v) > 0) {
        const left = (x(peak.t) / VB_W) * 100;
        overlay += `<span class="chart-peak" style="left:${left}%;top:${(y(peak.hi ?? peak.v) / height) * 100}%"></span>
          <span class="chart-peak-label" style="${left > 70 ? `right:${100 - left}%` : `left:${left}%`}">peak ${esc(formatters[fmt](peak.hi ?? peak.v))} · ${esc(when(peak.t))}</span>`;
      }
      for (const e of events) {
        if (e.t < from || e.t > to) continue;
        overlay += `<span class="chart-event ${esc(e.kind)}" style="left:${(x(e.t) / VB_W) * 100}%" title="${esc(`${when(e.t)} · ${e.message}`)}"></span>`;
      }
    }
    const data = esc(JSON.stringify(known.map((p) => (p.hi != null ? [p.t, p.v, p.lo, p.hi] : [p.t, p.v]))));
    const mid = from + span / 2;
    return `<div class="chart" data-points="${data}" data-fmt="${fmt}" data-from="${from}" data-to="${to}">
      <span class="chart-top">${esc(formatters[fmt](top))}</span>
      <div class="chart-plot" style="height:${height}px">
        <svg viewBox="0 0 ${VB_W} ${height}" preserveAspectRatio="none" aria-hidden="true">
          <line class="chart-grid" x1="0" x2="${VB_W}" y1="3" y2="3"/>
          <line class="chart-grid" x1="0" x2="${VB_W}" y1="${height / 2}" y2="${height / 2}"/>
          <line class="chart-base" x1="0" x2="${VB_W}" y1="${height - 0.5}" y2="${height - 0.5}"/>
          ${paths}
        </svg>
        ${overlay}
        <span class="chart-cross" hidden></span>
        <span class="chart-tip" hidden></span>
      </div>
      <div class="chart-axis"><span>${esc(when(from))}</span><span>${esc(when(mid))}</span><span>${range.live ? "now" : esc(when(to))}</span></div>
    </div>`;
  }

  // A round top for the axis, so its label names a value worth reading.
  function niceTop(v) {
    if (!(v > 0)) return 1;
    const mag = 10 ** Math.floor(Math.log10(v));
    return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((t) => t >= v);
  }

  els.view.addEventListener("pointermove", (e) => {
    const plot = e.target.closest?.(".chart-plot");
    els.view.querySelectorAll(".chart-plot").forEach((p) => {
      if (p !== plot) { p.querySelector(".chart-cross").hidden = true; p.querySelector(".chart-tip").hidden = true; }
    });
    if (!plot) return;
    const box = plot.parentElement;
    const pts = JSON.parse(box.dataset.points || "[]");
    if (!pts.length) return;
    const from = Number(box.dataset.from), to = Number(box.dataset.to);
    const r = plot.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const t = from + frac * (to - from);
    let best = pts[0];
    for (const p of pts) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
    const left = Math.min(100, Math.max(0, ((best[0] - from) / (to - from)) * 100));
    const f = formatters[box.dataset.fmt];
    const cross = plot.querySelector(".chart-cross");
    const tip = plot.querySelector(".chart-tip");
    cross.style.left = `${left}%`;
    cross.hidden = false;
    tip.textContent = best.length > 2
      ? `${when(best[0])} · ${f(best[1])} (${f(best[2])} – ${f(best[3])})`
      : `${when(best[0])} · ${f(best[1])}`;
    tip.style.left = left > 60 ? "" : `${Math.max(0, left)}%`;
    tip.style.right = left > 60 ? `${Math.max(0, 100 - left)}%` : "";
    tip.hidden = false;
  });
  els.view.addEventListener("pointerleave", () => {
    els.view.querySelectorAll(".chart-cross, .chart-tip").forEach((n) => (n.hidden = true));
  });

  // A bar against a whole, which turns to the warning colour near full.
  function meter(value, { warnAt = 90 } = {}) {
    if (value == null) return "";
    const level = value >= warnAt ? " warn" : "";
    return `<div class="meter${level}" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${value.toFixed(0)}"><span style="width:${value.toFixed(1)}%"></span></div>`;
  }

  const stat = (label, value, note = "") =>
    `<div class="stat"><dt>${esc(label)}</dt><dd>${value}${note ? `<small>${esc(note)}</small>` : ""}</dd></div>`;

  // The highest of a figure over the range, and when.
  function peakOf(points) {
    let best = null;
    for (const p of points) {
      const v = p.hi ?? p.v;
      if (v != null && (!best || v > best.v)) best = { t: p.t, v };
    }
    // Nothing above zero is no peak at all.
    return best && best.v > 0 ? best : null;
  }
  const sum = (f) => (history?.rows || []).reduce((a, r) => a + (f(r) || 0), 0);
  const inRange = (pts) => { const { from } = bounds(); return pts.filter((p) => p.t >= from); };

  // ── Render ───────────────────────────────────────────────────────────
  function renderRanges() {
    els.ranges.innerHTML = RANGES.map((r) =>
      `<button type="button" data-range="${r.key}" class="${r === range ? "on" : ""}" aria-pressed="${r === range}">${esc(r.label)}</button>`).join("");
  }

  function renderModels(active) {
    if (!running.length) {
      els.models.innerHTML = `<div class="mon-model"><span class="state-chip idle">No model loaded</span><span class="mon-dim">The first request loads one.</span></div>`;
      return;
    }
    els.models.innerHTML = running.map((m) => {
      const mine = active.filter((r) => r.model === m.name);
      const state = mine.some((r) => r.phase === "generating") ? "generating" : mine.length ? "reading" : "idle";
      const label = { generating: "Writing", reading: "Reading the prompt", idle: "Idle" }[state];
      const e = m.eullm || {};
      const facts = [
        e.slot && e.slot !== "generation" ? ["Slot", e.slot] : null,
        m.context_length ? ["Context", `${m.context_length.toLocaleString()} per request`] : null,
        e.batch_size > 1 ? ["Slots", e.batch_size] : null,
        ["Memory", m.size_vram ? `${gib(m.size_vram)} GiB VRAM` : "CPU"],
        m.size_vram && e.gpu_layers === -1 ? ["Layers", "all on GPU"] : m.size_vram && e.gpu_layers > 0 ? ["Layers", `${e.gpu_layers} on GPU`] : null,
        mine.length ? ["Running", mine.length] : null,
      ].filter(Boolean);
      return `<div class="mon-model">
        <span class="state-chip ${state}">${label}</span>
        <strong class="mon-model-name" title="${esc(m.name)}">${esc(m.name)}</strong>
        <span class="mon-facts">${facts.map(([k, v]) => `<span><em>${esc(k)}</em> ${esc(v)}</span>`).join("")}</span>
      </div>`;
    }).join("");
  }

  function renderEngine() {
    const { active, history: done } = reqs;
    const now = samples[samples.length - 1];
    const generating = active.filter((r) => r.phase === "generating");
    const last = done.find((r) => r.phase === "done");
    els.engineNow.innerHTML = generating.length
      ? `${num(now?.gen_tps)}<small>tok/s · ${plural(generating.length, "answer")}</small>`
      : last ? `${num(last.decode_tps)}<small>tok/s · last answer</small>` : `–<small>no answer yet</small>`;
    const points = series((s) => s.gen_tps, (r) => r.gen_tps);
    els.engineChart.innerHTML = chart(points, { height: 120, fmt: "tps", marks: true });
    const peak = peakOf(inRange(points));
    const peakStat = stat("Peak", peak ? `${num(peak.v)} <span class="u">tok/s</span>` : "–", peak ? when(peak.t) : "");
    if (range.live) {
      els.engineStats.innerHTML = [
        peakStat,
        stat("Reading the prompt", last ? `${num(last.prefill_tps, 0)} <span class="u">tok/s</span>` : "–", last ? plural(last.prompt_tokens ?? 0, "token") : ""),
        stat("First token", last ? ms(last.first_token_ms) : "–", "last answer"),
        stat("Running", String(active.length), active.length ? plural(generating.length, "writing") : "none"),
      ].join("");
    } else {
      const most = peakOf(trio((r) => r.active));
      els.engineStats.innerHTML = [
        peakStat,
        stat("Answers written", sum((r) => r.gen_tokens).toLocaleString(), `tokens · ${sum((r) => r.prompt_tokens).toLocaleString()} read`),
        stat("Requests", sum((r) => r.done).toLocaleString(), `${sum((r) => r.failed).toLocaleString()} failed or left`),
        stat("Most at once", most ? num(most.v, 0) : "–", most ? when(most.t) : ""),
      ].join("");
    }
  }

  function renderMachine() {
    const now = samples[samples.length - 1] || {};
    els.cores.textContent = now.cores ? `${now.cores} cores` : "";
    const memPct = pct(now.ram_used, now.ram_total);
    const cpu = series((s) => s.cpu_pct, (r) => r.cpu);
    const mem = series((s) => s.ram_used, (r) => r.ram);
    const disk = series((s) => s.disk_read_bps, (r) => r.disk_read);
    const peakNote = (pts, f) => { const p = peakOf(inRange(pts)); return p ? `peak ${f(p.v)} at ${when(p.t)}` : ""; };
    els.machine.innerHTML = `
      <div class="row"><div class="row-head"><span>CPU</span><b>${num(now.cpu_pct, 0)}%</b></div>${chart(cpu, { height: 40, max: 100, fmt: "pct" })}<small class="row-note">${esc(peakNote(cpu, formatters.pct))}</small></div>
      <div class="row"><div class="row-head"><span>Memory</span><b>${gib(now.ram_used)} <span class="u">/ ${gib(now.ram_total)} GiB</span></b></div>${meter(memPct)}${chart(mem, { height: 32, max: now.ram_total || 0, fmt: "gib" })}<small class="row-note">The engine holds ${gib(now.process_ram)} GiB · ${esc(peakNote(mem, formatters.gib))}</small></div>
      <div class="row"><div class="row-head"><span>Disk reads</span><b>${rateBytes(now.disk_read_bps)}</b></div>${chart(disk, { height: 40, fmt: "bytes" })}<small class="row-note">By the engine: a model loading, weights paged in${peakNote(disk, rateBytes) ? ` · ${esc(peakNote(disk, rateBytes))}` : ""}</small></div>`;
  }

  function renderGpus() {
    const now = samples[samples.length - 1];
    if (!now) { els.gpus.innerHTML = ""; return; }
    if (!now.gpus?.length) {
      els.gpus.innerHTML = `<article class="panel mon-empty">No GPU found: the engine runs on the CPU.</article>`;
      return;
    }
    const source = { nvml: "NVIDIA driver", amdgpu: "amdgpu driver", ggml: "llama.cpp" };
    els.gpus.innerHTML = now.gpus.map((g, i) => {
      const gs = (fs, fr) => series((s) => (s.gpus?.[i] ? fs(s.gpus[i]) : null), (r) => (r.gpus?.[i] ? fr(r.gpus[i]) : null));
      const vramPts = gs((x) => x.vram_used, (x) => x.vram);
      const tempPts = gs((x) => x.temp_c, (x) => x.temp);
      const powerPts = gs((x) => x.power_w, (x) => x.power);
      const peak = (pts, f) => { const p = peakOf(inRange(pts)); return p ? `<small class="row-note">peak ${esc(f(p.v))} · ${esc(when(p.t))}</small>` : ""; };
      const vram = pct(g.vram_used, g.vram_total);
      const power = pct(g.power_w, g.power_limit_w);
      const stats = [];
      if (g.vram_total != null) stats.push(`<div class="gstat"><div class="row-head"><span>VRAM</span><b>${gib(g.vram_used)} <span class="u">/ ${gib(g.vram_total)} GiB</span></b></div>${meter(vram, { warnAt: 95 })}${chart(vramPts, { height: 28, max: g.vram_total, fmt: "gib" })}${peak(vramPts, formatters.gib)}</div>`);
      if (g.power_w != null) stats.push(`<div class="gstat"><div class="row-head"><span>Power</span><b>${num(g.power_w, 0)} <span class="u">${g.power_limit_w ? `/ ${num(g.power_limit_w, 0)} W` : "W"}</span></b></div>${meter(power, { warnAt: 98 })}${chart(powerPts, { height: 28, max: g.power_limit_w || 0, fmt: "watts" })}${peak(powerPts, formatters.watts)}</div>`);
      if (g.temp_c != null) stats.push(`<div class="gstat"><div class="row-head"><span>Temperature</span><b class="${g.temp_c >= 85 ? "hot" : ""}">${num(g.temp_c, 0)} <span class="u">°C</span></b></div>${chart(tempPts, { height: 28, max: 100, fmt: "celsius" })}${peak(tempPts, formatters.celsius)}</div>`);
      if (g.pcie_rx_bps != null) stats.push(`<div class="gstat"><div class="row-head"><span>PCIe</span><b>↓ ${rateBytes(g.pcie_rx_bps)}</b></div><small class="row-note">to the GPU · ↑ ${rateBytes(g.pcie_tx_bps)} back</small></div>`);
      const load = g.util_pct != null
        ? `<div class="row-head"><span>Load</span><b>${num(g.util_pct, 0)}%</b></div>${chart(gs((x) => x.util_pct, (x) => x.util), { height: 72, max: 100, fmt: "pct", marks: true })}`
        : `<p class="mon-dim">This driver does not report the load.</p>`;
      return `<article class="panel gpu">
        <header class="panel-head">
          <h3 title="${esc(g.name)}">${esc(g.name)}</h3>
          <span class="panel-meta">GPU ${i} · ${esc(source[g.source] || g.source)}</span>
        </header>
        ${load}
        <div class="gstats">${stats.join("")}</div>
      </article>`;
    }).join("");
  }

  const EVENT_LABEL = {
    start: "Started", model_loaded: "Loaded", model_unloaded: "Unloaded",
    request_failed: "Failed", warning: "Warning", admin: "Admin",
  };
  function renderEvents() {
    const { from } = bounds();
    const shown = events.filter((e) => e.t >= from).slice().reverse();
    els.eventsCount.textContent = shown.length ? plural(shown.length, "event") : "";
    els.events.innerHTML = shown.length
      ? shown.slice(0, 200).map((e) => `<li>
          <time>${esc(when(e.t, Math.max(range.span, DAY + 1)))}</time>
          <span class="event-chip ${esc(e.kind)}">${esc(EVENT_LABEL[e.kind] || e.kind)}</span>
          <span class="event-msg">${esc(e.message)}</span></li>`).join("")
      : `<li class="mon-empty">Nothing happened in this range.</li>`;
  }

  function renderRequests() {
    const { active, history: done } = reqs;
    const rows = [...active.slice().reverse(), ...done];
    els.reqCount.textContent = rows.length ? `${active.length} running · ${done.length} answered (last 50)` : "";
    if (!rows.length) {
      els.requests.innerHTML = `<tr><td colspan="12" class="mon-empty">No request since the engine started.</td></tr>`;
      return;
    }
    const label = { reading: "Reading", generating: "Writing", done: "Done", failed: "Failed" };
    els.requests.innerHTML = rows.map((r) => {
      const drafts = r.drafts ? `${Math.round((r.drafts[1] / Math.max(1, r.drafts[0])) * 100)}% <span class="u">of ${r.drafts[0]}</span>` : "–";
      const total = r.phase === "reading" || r.phase === "generating" ? "…" : ms(r.duration_ms);
      return `<tr>
        <td>${clock(r.started_ms)}</td><td class="wide" title="${esc(r.model)}">${esc(r.model)}</td><td><code>${esc(r.endpoint)}</code></td>
        <td><span class="state-chip ${r.phase}">${label[r.phase] || esc(r.phase)}</span></td>
        <td class="num">${r.prompt_tokens != null ? r.prompt_tokens.toLocaleString() : "–"}</td>
        <td class="num">${r.output_tokens.toLocaleString()}</td>
        <td class="num">${num(r.prefill_tps, 0)}</td><td class="num">${num(r.decode_tps)}</td>
        <td class="num">${ms(r.first_token_ms)}</td><td class="num">${total}</td>
        <td class="num">${drafts}</td><td class="ended" title="${esc(r.ended)}">${esc(r.ended ?? "")}</td>
      </tr>`;
    }).join("");
  }

  // ── Storage: how much the history takes, and who may change it ─────────
  function renderStorage(message = "", error = false) {
    if (!storage) { els.storage.innerHTML = `<p class="mon-dim">…</p>`; return; }
    const s = storage;
    const days = s.days || [];
    els.storageMeta.textContent = s.dir ? `${size(s.total_bytes)} on disk` : "in memory only";
    const facts = [
      ["Where", s.dir ? `${s.dir}` : "In memory: the last 24 hours, lost at a restart", s.dir_source],
      ["Kept", `${s.retention_days} days`, s.retention_source],
      ["Used", `${size(s.total_bytes)} in ${plural(days.length, "day")}`, days.length ? `${days[0].date} – ${days[days.length - 1].date}` : ""],
      ["At this rate", `about ${size(s.projected_bytes)}`, `for ${s.retention_days} days`],
    ];
    const bars = (() => {
      if (days.length < 2) return "";
      const top = Math.max(...days.map((d) => d.bytes), 1);
      return `<div class="mon-days" aria-hidden="true">${days.map((d) =>
        `<span title="${esc(`${d.date} · ${size(d.bytes)}`)}" style="height:${Math.max(4, (d.bytes / top) * 100)}%"></span>`).join("")}</div>`;
    })();
    const today = new Date().toISOString().slice(0, 10);
    const admin = s.admin_enabled
      ? `<form class="mon-admin" autocomplete="off" novalidate>
          <label class="field"><span>Admin key</span>
            <input type="password" name="key" autocomplete="off" spellcheck="false" placeholder="${adminKey ? "kept for this page" : "EULLM_ADMIN_KEY"}"></label>
          <label class="field"><span>Keep (days, 1–${s.max_retention_days})</span>
            <span class="mon-inline"><input type="number" name="days" min="1" max="${s.max_retention_days}" value="${s.retention_days}">
            <button type="button" class="btn-secondary small" data-act="retention">Save</button></span></label>
          <label class="field"><span>Delete the days before</span>
            <span class="mon-inline"><input type="date" name="before" max="${today}" value="${days[0]?.date || today}">
            <button type="button" class="btn-secondary small danger" data-act="delete">Delete</button></span></label>
        </form>
        <p class="mon-dim">The key is kept by this page only until it is reloaded. Five wrong keys lock this address out for 15 minutes.</p>`
      : `<p class="mon-dim">Changing how long the history is kept, or deleting it, is not allowed from this page: the server has no admin key. Whoever runs it can set <code>EULLM_ADMIN_KEY</code> (at least 16 characters) to allow it, or <code>EULLM_METRICS_RETENTION_DAYS</code> to change the days.</p>`;
    els.storage.innerHTML = `
      <dl class="kv mon-kv">${facts.map(([k, v, note]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}${note ? `<small>${esc(note)}</small>` : ""}</dd></div>`).join("")}</dl>
      ${bars}
      ${admin}
      ${message ? `<p class="mon-msg ${error ? "err" : "ok"}" role="status">${esc(message)}</p>` : ""}`;
  }

  async function loadStorage() {
    try {
      const r = await fetch("/api/monitor/storage", auth());
      if (r.ok) storage = await r.json();
    } catch {}
    renderStorage();
  }

  els.storage.addEventListener("click", async (e) => {
    const act = e.target.closest?.("button[data-act]")?.dataset.act;
    if (!act) return;
    const form = els.storage.querySelector("form");
    const typed = form.key.value.trim();
    const key = typed || adminKey;
    if (!key) { renderStorage("Type the admin key first.", true); return; }
    let req;
    if (act === "retention") {
      const days = Number(form.days.value);
      if (!(days >= 1 && days <= storage.max_retention_days)) { renderStorage(`Between 1 and ${storage.max_retention_days} days.`, true); return; }
      const oldest = storage.days?.[0]?.date;
      if (days < storage.retention_days && oldest && !confirm(`Keep ${days} days: the days older than that are deleted now, and cannot be brought back. Go on?`)) return;
      req = fetch("/api/monitor/storage", { method: "PUT", headers: { ...apiHeaders(), "X-Admin-Key": key, "Content-Type": "application/json" }, body: JSON.stringify({ retention_days: days }) });
    } else {
      const before = form.before.value;
      if (!before) { renderStorage("Choose a date.", true); return; }
      if (!confirm(`Delete every day of history before ${before}? It cannot be brought back.`)) return;
      req = fetch(`/api/monitor/history?before=${encodeURIComponent(before)}`, { method: "DELETE", headers: { ...apiHeaders(), "X-Admin-Key": key } });
    }
    try {
      const r = await req;
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        if (d.admin === "wrong" || d.admin === "locked") adminKey = "";
        renderStorage(d.error || `HTTP ${r.status}`, true);
        return;
      }
      adminKey = key;
      if (d.storage) storage = d.storage; else await loadStorage();
      const freed = d.deleted_files ? ` ${plural(d.deleted_files, "day")} deleted, ${size(d.freed_bytes)} freed.` : " Nothing to delete.";
      renderStorage((act === "retention" ? `Kept for ${storage.retention_days} days from now on.` : "Done.") + freed);
      if (!range.live) loadHistory();
    } catch (err) {
      renderStorage(`No answer from the engine (${err.message})`, true);
    }
  });

  // ── Reading ──────────────────────────────────────────────────────────
  function renderAll() {
    renderModels(reqs.active);
    renderEngine();
    renderMachine();
    renderGpus();
    renderEvents();
    renderRequests();
  }

  async function poll() {
    try {
      // Every second since the last one this page has: after a hidden tab
      // or a closed page, the gap is filled, up to the hour kept.
      const since = lastT || Date.now() - HOUR;
      const r = await fetch(`/api/monitor?since=${since}`, auth());
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      reqs = { active: d.active, history: d.history };
      if (d.system.length) {
        samples.push(...d.system);
        lastT = samples[samples.length - 1].t_ms;
        const cut = lastT - HOUR;
        while (samples.length && samples[0].t_ms < cut) samples.shift();
      }
      renderAll();
      els.live.classList.remove("off");
      els.live.textContent = "Live";
    } catch (e) {
      els.live.classList.add("off");
      els.live.textContent = `No answer from the engine (${e.message})`;
    }
  }

  // The minute rows of a longer range; for the live ones, only the events.
  async function loadHistory() {
    const to = Date.now();
    const from = to - range.span;
    try {
      const r = await fetch(`/api/monitor/history?from=${from}&to=${to}`, auth());
      if (!r.ok) return;
      const d = await r.json();
      events = d.events || [];
      history = range.live ? null : d;
      renderAll();
    } catch {}
  }

  async function pollModels() {
    try {
      const r = await fetch("/api/ps", auth());
      if (r.ok) running = (await r.json()).models || [];
    } catch {}
  }

  els.ranges.addEventListener("click", (e) => {
    const key = e.target.closest?.("button[data-range]")?.dataset.range;
    const next = RANGES.find((r) => r.key === key);
    if (!next || next === range) return;
    range = next;
    try { localStorage.setItem("eullm.monitorRange", key); } catch {}
    history = null;
    renderRanges();
    renderAll();
    loadHistory();
  });

  function start() {
    if (timer) return;
    renderRanges();
    pollModels().then(poll);
    loadHistory();
    loadStorage();
    timer = setInterval(poll, 1000);
    psTimer = setInterval(pollModels, 3000);
    // Minute rows and events change once a minute; the storage less often.
    slowTimer = setInterval(() => { loadHistory(); loadStorage(); }, 60_000);
  }

  function stop() {
    clearInterval(timer); clearInterval(psTimer); clearInterval(slowTimer);
    timer = psTimer = slowTimer = null;
  }

  // The page shell (app.js) shows and hides the views. Reading pauses while
  // this one is not shown or the tab is hidden; the engine goes on sampling,
  // and the seconds missed are read when the page is back.
  document.addEventListener("eullm:view", (e) => {
    if (e.detail === "monitor") start(); else stop();
  });

  document.addEventListener("visibilitychange", () => {
    if (els.view.hidden) return;
    if (document.hidden) stop(); else start();
  });
})();
