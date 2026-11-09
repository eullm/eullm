// EuLLM About — this build, the machine, the devices llama.cpp can use, where
// models and the audit trail are kept, and who may use the server. Read once
// from /api/about each time the page opens.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    view: $("about"),
    version: $("about-version"),
    chips: $("about-chips"),
    copy: $("about-copy"),
    build: $("about-build"),
    machine: $("about-machine"),
    devices: $("about-devices"),
    paths: $("about-paths"),
    protection: $("about-protection"),
  };
  if (!els.view) return;

  const auth = () => {
    let key = "";
    try { key = sessionStorage.getItem("eullm.apiKey") || ""; } catch {}
    return key ? { headers: { "X-Api-Key": key } } : {};
  };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const gib = (b) => (b ? `${(b / 2 ** 30).toFixed(b >= 10 * 2 ** 30 ? 0 : 1)} GiB` : "–");
  const since = (ms) => {
    if (!ms) return "–";
    const s = Math.max(0, (Date.now() - ms) / 1000);
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    const span = d ? `${d} d ${h} h` : h ? `${h} h ${m} min` : `${m} min`;
    return `${span}, since ${new Date(ms).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`;
  };
  const kv = (rows) =>
    rows.filter(Boolean).map(([k, v, raw]) => `<div><dt>${esc(k)}</dt><dd>${raw ? v : esc(v)}</dd></div>`).join("");
  const path = (p) => `<code class="path">${esc(p)}</code>`;

  let details = "";

  function render(a) {
    const m = a.machine || {};
    els.version.textContent = a.version;
    // The CPU features llama.cpp reports as `NAME = 1`, the ones the binary uses.
    const features = (a.cpu_features || "")
      .split("|").map((f) => f.replace(/^[^:]*:\s*/, "").trim()).filter((f) => /=\s*1$/.test(f)).map((f) => f.replace(/\s*=\s*1$/, ""));
    els.chips.innerHTML = [
      `<span class="chip strong">${esc(a.backend)}</span>`,
      `<span class="chip">llama.cpp ${esc(a.llama_cpp)}</span>`,
      `<span class="chip mono" title="The commit this binary was built from">${esc(a.commit)}</span>`,
    ].join("");

    els.build.innerHTML = kv([
      ["Version", a.version],
      ["Backend", a.backend],
      ["llama.cpp", a.llama_cpp],
      ["Commit", `<code>${esc(a.commit)}</code>${/-dirty$/.test(a.commit) ? ` <small>built with local changes</small>` : ""}`, true],
      ["Installed as", a.release_asset ? a.release_asset : "built from source", false],
      features.length ? ["CPU features", `<span class="flags">${features.map((f) => `<span>${esc(f)}</span>`).join("")}</span>`, true] : null,
    ]);

    els.machine.innerHTML = kv([
      ["System", [m.os, m.arch].filter(Boolean).join(" · ")],
      m.kernel ? ["Kernel", m.kernel] : null,
      ["Processor", m.cpu || "–"],
      ["Cores", m.physical_cores && m.physical_cores !== m.cores ? `${m.cores} threads on ${m.physical_cores} cores` : String(m.cores || "–")],
      ["Memory", gib(m.ram_total)],
      m.host ? ["Host name", m.host] : null,
      ["Engine running", since(m.started_ms)],
    ]);

    const kinds = { cpu: "CPU", gpu: "GPU", igpu: "Integrated GPU", accel: "Accelerator" };
    els.devices.innerHTML = (m.devices || []).map((d) => `
      <li class="device ${esc(d.kind)}">
        <span class="device-kind">${esc(kinds[d.kind] || d.kind)}</span>
        <span class="device-name">${esc(d.description || d.name)}</span>
        ${[d.name.toLowerCase() !== d.kind ? d.name : "", d.memory_total ? gib(d.memory_total) : ""].filter(Boolean).length ? `<span class="device-meta">${esc([d.name.toLowerCase() !== d.kind ? d.name : "", d.memory_total ? gib(d.memory_total) : ""].filter(Boolean).join(" · "))}</span>` : ""}
      </li>`).join("") || `<li class="device"><span class="device-name">None reported</span></li>`;
    if (m.devices && !m.devices.some((d) => d.kind !== "cpu") && a.backend === "CPU") {
      els.devices.insertAdjacentHTML("beforeend", `<li class="device-hint">This binary is the CPU build. With an NVIDIA, AMD or Apple GPU, the CUDA, ROCm, Vulkan or Metal build from the releases page runs models on it.</li>`);
    }

    els.paths.innerHTML = kv([
      ["Models", `${path(a.models_dir)}<small>${esc(a.models_dir_source === "default" ? "the default folder" : `set by ${a.models_dir_source}`)}</small>`, true],
      ["Model libraries", a.model_libraries?.length ? a.model_libraries.map(path).join("") + `<small>read only, used where they are</small>` : "none (EULLM_MODEL_LIBRARIES)", !!a.model_libraries?.length],
      ["Audit trail", `${path(a.audit_log)}<small>one line per answer, on this machine only</small>`, true],
      ["API", `<code>http://localhost:${esc(a.api_port)}</code><small>Ollama and OpenAI compatible: /api and /v1</small>`, true],
    ]);

    const p = a.protections || {};
    const loopback = /^127\.0\.0\.1\/32, ::1\/128$/.test(p.allowed_ips || "");
    els.protection.innerHTML = kv([
      ["API keys", p.api_keys
        ? `<span class="status ok">Required</span><small>every request carries a key</small>`
        : `<span class="status">Not required</span><small>anyone the network list admits can use it</small>`, true],
      ["Admitted without a key", `${loopback ? `<span class="status ok">This machine only</span>` : `<code>${esc(p.allowed_ips)}</code>`}<small>${esc(p.allowed_ips_source)}</small>`, true],
    ]);

    details = [
      `EuLLM Engine ${a.version} (${a.backend}) [${a.commit}]`,
      `llama.cpp ${a.llama_cpp}${a.release_asset ? ` · ${a.release_asset}` : " · built from source"}`,
      `${[m.os, m.arch].filter(Boolean).join(" ")}${m.kernel ? ` · kernel ${m.kernel}` : ""}`,
      `${m.cpu || "CPU"} · ${m.cores} threads · ${gib(m.ram_total)} RAM`,
      ...(m.devices || []).filter((d) => d.kind !== "cpu").map((d) => `${d.name}: ${d.description}${d.memory_total ? ` (${gib(d.memory_total)})` : ""}`),
      features.length ? `CPU features: ${features.join(" ")}` : "",
    ].filter(Boolean).join("\n");
  }

  async function load() {
    try {
      const r = await fetch("/api/about", auth());
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      render(await r.json());
    } catch (e) {
      els.build.innerHTML = kv([["Error", `The engine did not answer (${e.message}). Is it still running?`]]);
    }
  }

  els.copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(details);
      els.copy.textContent = "Copied";
    } catch {
      els.copy.textContent = "Copy failed";
    }
    setTimeout(() => (els.copy.textContent = "Copy details"), 1500);
  });

  // The page shell (app.js) shows and hides the views; this one reads the
  // engine each time it is shown.
  document.addEventListener("eullm:view", (e) => {
    if (e.detail === "about") load();
  });
})();
