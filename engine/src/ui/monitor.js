// EuLLM Monitor — the requests the engine is answering and has answered, and
// the machine it runs on. Reads /api/monitor every second and /api/ps every
// few, only while the page is open; everything shown lives in the engine's
// memory and in this tab's, and nothing is stored.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    view: $("monitor"),
    live: $("mon-live"),
    models: $("mon-models"),
    engineNow: $("mon-engine-now"),
    engineChart: $("mon-engine-chart"),
    engineStats: $("mon-engine-stats"),
    machine: $("mon-machine"),
    cores: $("mon-machine-cores"),
    gpus: $("mon-gpus"),
    requests: $("mon-requests"),
    reqCount: $("mon-req-count"),
  };
  if (!els.view) return;

  // The key app.js keeps for this tab (see its API key comment).
  const auth = () => {
    let key = "";
    try { key = sessionStorage.getItem("eullm.apiKey") || ""; } catch {}
    return key ? { headers: { "X-Api-Key": key } } : {};
  };

  // Every chart spans the same two minutes the engine keeps, ending now.
  const WINDOW_MS = 120_000;
  // The engine's writing speed at each second, which no single response
  // carries: the tab keeps it while the page is open.
  const decodeSeries = [];
  let timer = null;
  let psTimer = null;
  let running = [];

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
  const num = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));
  const ms = (v) => (v == null ? "–" : v >= 10000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
  const clock = (t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const pct = (used, total) => (used != null && total ? Math.min(100, (used / total) * 100) : null);
  const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
  const formatters = {
    tps: (v) => `${num(v)} tok/s`,
    pct: (v) => `${num(v, 0)}%`,
    bytes: rateBytes,
    celsius: (v) => `${num(v, 0)} °C`,
    watts: (v) => `${num(v, 0)} W`,
  };

  // ── Charts ────────────────────────────────────────────────────────────
  // One series on one axis from zero, over the last two minutes, the newest
  // value marked. The line and area stretch with the panel; the dot, the
  // labels and the hover readout are HTML, so they keep their shape.
  const VB_W = 600;
  function chart(points, { height = 72, max = 0, fmt = "tps" } = {}) {
    const now = Date.now();
    const known = points.filter((p) => p.v != null && now - p.t <= WINDOW_MS);
    const top = niceTop(Math.max(max, ...known.map((p) => p.v), 0));
    const x = (t) => ((t - (now - WINDOW_MS)) / WINDOW_MS) * VB_W;
    const y = (v) => height - 1 - (v / top) * (height - 4);
    let paths = "";
    let dot = "";
    if (known.length >= 2) {
      const line = known.map((p) => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
      const first = x(known[0].t).toFixed(1);
      const last = known[known.length - 1];
      paths = `<polygon class="chart-area" points="${first},${height} ${line} ${x(last.t).toFixed(1)},${height}"/>
        <polyline class="chart-line" points="${line}"/>`;
      dot = `<span class="chart-dot" style="left:${(x(last.t) / VB_W) * 100}%;top:${(y(last.v) / height) * 100}%"></span>`;
    }
    const data = esc(JSON.stringify(known.map((p) => [p.t, p.v])));
    return `<div class="chart" data-points="${data}" data-fmt="${fmt}" data-now="${now}">
      <span class="chart-top">${esc(formatters[fmt](top))}</span>
      <div class="chart-plot" style="height:${height}px">
        <svg viewBox="0 0 ${VB_W} ${height}" preserveAspectRatio="none" aria-hidden="true">
          <line class="chart-grid" x1="0" x2="${VB_W}" y1="3" y2="3"/>
          <line class="chart-grid" x1="0" x2="${VB_W}" y1="${height / 2}" y2="${height / 2}"/>
          <line class="chart-base" x1="0" x2="${VB_W}" y1="${height - 0.5}" y2="${height - 0.5}"/>
          ${paths}
        </svg>
        ${dot}
        <span class="chart-cross" hidden></span>
        <span class="chart-tip" hidden></span>
      </div>
      <div class="chart-axis"><span>2 min ago</span><span>1 min</span><span>now</span></div>
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
    const r = plot.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const t = Number(box.dataset.now) - WINDOW_MS + frac * WINDOW_MS;
    let best = pts[0];
    for (const p of pts) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
    const left = ((best[0] - (Number(box.dataset.now) - WINDOW_MS)) / WINDOW_MS) * 100;
    const cross = plot.querySelector(".chart-cross");
    const tip = plot.querySelector(".chart-tip");
    cross.style.left = `${left}%`;
    cross.hidden = false;
    tip.textContent = `${clock(best[0])} · ${formatters[box.dataset.fmt](best[1])}`;
    tip.style.left = `${Math.min(78, Math.max(0, left - 11))}%`;
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

  // ── Render ───────────────────────────────────────────────────────────
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

  function renderEngine(active, history) {
    const generating = active.filter((r) => r.phase === "generating");
    const now = generating.reduce((a, r) => a + (r.decode_tps || 0), 0);
    decodeSeries.push({ t: Date.now(), v: generating.length ? now : 0 });
    while (decodeSeries.length && Date.now() - decodeSeries[0].t > WINDOW_MS) decodeSeries.shift();
    const last = history.find((r) => r.phase === "done");
    els.engineNow.innerHTML = generating.length
      ? `${num(now)}<small>tok/s · ${plural(generating.length, "answer")}</small>`
      : last ? `${num(last.decode_tps)}<small>tok/s · last answer</small>` : `–<small>no answer yet</small>`;
    els.engineChart.innerHTML = chart(decodeSeries, { height: 120, fmt: "tps" });
    els.engineStats.innerHTML = [
      stat("Reading the prompt", last ? `${num(last.prefill_tps, 0)} <span class="u">tok/s</span>` : "–", last ? plural(last.prompt_tokens ?? 0, "token") : ""),
      stat("First token", last ? ms(last.first_token_ms) : "–", "last answer"),
      stat("Running", String(active.length), active.length ? plural(generating.length, "writing") : "none"),
      stat("Answered", String(history.filter((r) => r.phase === "done").length), `${history.filter((r) => r.phase === "failed").length} failed`),
    ].join("");
  }

  function renderMachine(system) {
    const now = system[system.length - 1] || {};
    const series = (f) => system.map((s) => ({ t: s.t_ms, v: f(s) }));
    els.cores.textContent = now.cores ? `${now.cores} cores` : "";
    const memPct = pct(now.ram_used, now.ram_total);
    els.machine.innerHTML = `
      <div class="row"><div class="row-head"><span>CPU</span><b>${num(now.cpu_pct, 0)}%</b></div>${chart(series((s) => s.cpu_pct), { height: 40, max: 100, fmt: "pct" })}</div>
      <div class="row"><div class="row-head"><span>Memory</span><b>${gib(now.ram_used)} <span class="u">/ ${gib(now.ram_total)} GiB</span></b></div>${meter(memPct)}<small class="row-note">The engine holds ${gib(now.process_ram)} GiB</small></div>
      <div class="row"><div class="row-head"><span>Disk reads</span><b>${rateBytes(now.disk_read_bps)}</b></div>${chart(series((s) => s.disk_read_bps), { height: 40, fmt: "bytes" })}<small class="row-note">By the engine: a model loading, weights paged in</small></div>`;
  }

  function renderGpus(system) {
    const now = system[system.length - 1];
    if (!now) { els.gpus.innerHTML = ""; return; }
    if (!now.gpus?.length) {
      els.gpus.innerHTML = `<article class="panel mon-empty">No GPU found: the engine runs on the CPU.</article>`;
      return;
    }
    const source = { nvml: "NVIDIA driver", amdgpu: "amdgpu driver", ggml: "llama.cpp" };
    els.gpus.innerHTML = now.gpus.map((g, i) => {
      const series = (f) => system.map((s) => ({ t: s.t_ms, v: s.gpus?.[i] ? f(s.gpus[i]) : null }));
      const vram = pct(g.vram_used, g.vram_total);
      const power = pct(g.power_w, g.power_limit_w);
      const stats = [];
      if (g.vram_total != null) stats.push(`<div class="gstat"><div class="row-head"><span>VRAM</span><b>${gib(g.vram_used)} <span class="u">/ ${gib(g.vram_total)} GiB</span></b></div>${meter(vram, { warnAt: 95 })}</div>`);
      if (g.power_w != null) stats.push(`<div class="gstat"><div class="row-head"><span>Power</span><b>${num(g.power_w, 0)} <span class="u">${g.power_limit_w ? `/ ${num(g.power_limit_w, 0)} W` : "W"}</span></b></div>${meter(power, { warnAt: 98 })}</div>`);
      if (g.temp_c != null) stats.push(`<div class="gstat"><div class="row-head"><span>Temperature</span><b class="${g.temp_c >= 85 ? "hot" : ""}">${num(g.temp_c, 0)} <span class="u">°C</span></b></div>${chart(series((x) => x.temp_c), { height: 28, max: 100, fmt: "celsius" })}</div>`);
      if (g.pcie_rx_bps != null) stats.push(`<div class="gstat"><div class="row-head"><span>PCIe</span><b>↓ ${rateBytes(g.pcie_rx_bps)}</b></div><small class="row-note">to the GPU · ↑ ${rateBytes(g.pcie_tx_bps)} back</small></div>`);
      const load = g.util_pct != null
        ? `<div class="row-head"><span>Load</span><b>${num(g.util_pct, 0)}%</b></div>${chart(series((x) => x.util_pct), { height: 72, max: 100, fmt: "pct" })}`
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

  function renderRequests(active, history) {
    const rows = [...active.slice().reverse(), ...history];
    els.reqCount.textContent = rows.length ? `${active.length} running · ${history.length} answered (last 50)` : "";
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

  // ── Polling, only while the page is open and the tab visible ──────────
  async function poll() {
    try {
      const r = await fetch("/api/monitor", auth());
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      renderModels(d.active);
      renderEngine(d.active, d.history);
      renderMachine(d.system);
      renderGpus(d.system);
      renderRequests(d.active, d.history);
      els.live.classList.remove("off");
      els.live.textContent = "Live";
    } catch (e) {
      els.live.classList.add("off");
      els.live.textContent = `No answer from the engine (${e.message})`;
    }
  }

  async function pollModels() {
    try {
      const r = await fetch("/api/ps", auth());
      if (r.ok) running = (await r.json()).models || [];
    } catch {}
  }

  function start() {
    if (timer) return;
    pollModels().then(poll);
    timer = setInterval(poll, 1000);
    psTimer = setInterval(pollModels, 3000);
  }

  function stop() {
    clearInterval(timer); clearInterval(psTimer);
    timer = psTimer = null;
  }

  // The page shell (app.js) shows and hides the views; this one polls only
  // while it is the one shown.
  document.addEventListener("eullm:view", (e) => {
    if (e.detail === "monitor") start(); else stop();
  });

  document.addEventListener("visibilitychange", () => {
    if (els.view.hidden) return;
    if (document.hidden) stop(); else start();
  });
})();
