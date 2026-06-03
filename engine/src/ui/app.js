// EuLLM Chat UI — talks to /v1/chat/completions (OpenAI-compatible, SSE).
// Conversation lives in memory; no persistence, no telemetry.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    modelSelect: $("model-select"),
    settingsBtn: $("settings-btn"),
    settingsModal: $("settings-modal"),
    settingsForm: document.querySelector(".settings-form"),
    settingsSystem: $("settings-system"),
    settingsTemp: $("settings-temp"),
    settingsTempVal: $("settings-temp-val"),
    settingsMaxTokens: $("settings-max-tokens"),
    settingsThink: $("settings-think"),
    messages: $("messages"),
    form: $("chat-form"),
    input: $("input"),
    sendBtn: $("send-btn"),
    stopBtn: $("stop-btn"),
    clearBtn: $("clear-btn"),
    statusGrid: $("status-grid"),
  };

  const settings = {
    system: "",
    temperature: 0.7,
    maxTokens: 2048,
    // Reasoning ON by default. Reasoning models (DeepSeek-R1, QwQ) are trained
    // to always emit a <think> block; suppressing it (think:false injects an
    // empty <think></think>) makes them degenerate into a canned greeting.
    think: true,
  };

  // Strip <think>…</think> from an assistant turn before storing it in history.
  // Re-sending the model's own reasoning back as context confuses reasoning
  // models and bloats the prompt; only the final answer belongs in history.
  const stripThink = (s) =>
    s.replace(/<think>[\s\S]*?<\/think>\s*/g, "").replace(/<think>[\s\S]*$/, "").trim();

  const history = []; // {role, content}
  let currentModel = "";
  let abortController = null;

  // ── Markdown-lite: escape HTML, then convert fenced code blocks and inline code.
  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  function renderContent(text) {
    // Pull out Qwen3 <think>...</think> blocks first.
    let thinkHtml = "";
    text = text.replace(/<think>([\s\S]*?)<\/think>/g, (_, inner) => {
      const trimmed = inner.trim();
      if (trimmed) {
        thinkHtml += `<div class="think"><div class="think-label">Reasoning</div>${escapeHtml(trimmed)}</div>`;
      }
      return "";
    });
    // Unclosed <think> at the end (streaming): render as in-progress reasoning.
    text = text.replace(/<think>([\s\S]*)$/, (_, inner) => {
      thinkHtml += `<div class="think"><div class="think-label">Reasoning…</div>${escapeHtml(inner.trim())}</div>`;
      return "";
    });

    let body = escapeHtml(text);
    // Fenced code blocks ```lang\n...\n```
    body = body.replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code class="lang-${lang || "txt"}">${code}</code></pre>`);
    // Inline `code`
    body = body.replace(/`([^`\n]+)`/g, "<code>$1</code>");

    return thinkHtml + body;
  }

  // ── DOM helpers ───────────────────────────────────────────────────────
  function dismissWelcome() {
    const w = els.messages.querySelector(".welcome");
    if (w) w.remove();
  }

  function appendMessage(role, content = "") {
    dismissWelcome();
    const msg = document.createElement("div");
    msg.className = `msg ${role}`;
    msg.innerHTML = `
      <div class="msg-avatar">${role === "user" ? "U" : "Eu"}</div>
      <div class="msg-body">
        <div class="msg-name">${role === "user" ? "You" : "Assistant"}</div>
        <div class="msg-content"></div>
        <div class="msg-meta"></div>
      </div>`;
    els.messages.appendChild(msg);
    const contentEl = msg.querySelector(".msg-content");
    if (content) contentEl.innerHTML = renderContent(content);
    scrollToBottom();
    return { msg, contentEl, metaEl: msg.querySelector(".msg-meta") };
  }

  function scrollToBottom() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  // ── API calls ─────────────────────────────────────────────────────────
  async function loadModels() {
    try {
      const r = await fetch("/api/tags");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const loaded = (data.models || []).filter((m) => m.size === 0 || m.size === undefined || /\.gguf$/i.test(m.name) || !m.digest);
      // Heuristic: locally-loaded entry has size=0 and empty digest. Catalog entries have real digest+size.
      const reallyLoaded = (data.models || []).filter((m) => !m.digest || m.digest === "");
      const catalog = (data.models || []).filter((m) => m.digest && m.digest !== "");

      els.modelSelect.innerHTML = "";
      if (reallyLoaded.length) {
        const group = document.createElement("optgroup");
        group.label = "Loaded";
        for (const m of reallyLoaded) {
          const opt = document.createElement("option");
          opt.value = m.name;
          opt.textContent = m.name;
          group.appendChild(opt);
        }
        els.modelSelect.appendChild(group);
      }
      if (catalog.length) {
        const group = document.createElement("optgroup");
        group.label = "EU Catalog (not yet downloaded)";
        for (const m of catalog) {
          const opt = document.createElement("option");
          opt.value = m.name;
          opt.textContent = `${m.name} — ${(m.size / 1e9).toFixed(1)} GB`;
          opt.disabled = true;
          group.appendChild(opt);
        }
        els.modelSelect.appendChild(group);
      }
      currentModel = reallyLoaded[0]?.name || "";
      if (currentModel) els.modelSelect.value = currentModel;
      updateStatus(reallyLoaded[0]);
    } catch (err) {
      console.error("Could not load /api/tags:", err);
      const opt = document.createElement("option");
      opt.textContent = "(no models — start engine with: eullm run model.gguf)";
      opt.disabled = true;
      els.modelSelect.innerHTML = "";
      els.modelSelect.appendChild(opt);
    }
  }

  async function loadVersion() {
    try {
      const r = await fetch("/api/version");
      if (!r.ok) return null;
      return await r.json();
    } catch { return null; }
  }

  function updateStatus(loadedModel) {
    if (!els.statusGrid) return;
    const items = [
      ["Engine", `${window.__eullm_version || "v?"}`],
      ["Endpoint", location.origin],
      ["APIs", "/api  +  /v1"],
      ["Model", loadedModel?.name || "(none loaded)"],
      ["Telemetry", "off — local-only audit"],
    ];
    els.statusGrid.innerHTML = items
      .map(([k, v]) => `<span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(v)}</span>`)
      .join("");
  }

  // ── Streaming chat ────────────────────────────────────────────────────
  async function send(userText) {
    if (!currentModel) {
      alert("No model loaded.\n\nStart the engine with:\n  eullm run /path/to/model.gguf");
      return;
    }
    history.push({ role: "user", content: userText });
    appendMessage("user", userText);

    const messagesToSend = [];
    if (settings.system) messagesToSend.push({ role: "system", content: settings.system });
    messagesToSend.push(...history);

    const { msg, contentEl, metaEl } = appendMessage("assistant", "");
    msg.classList.add("streaming");
    setSending(true);

    abortController = new AbortController();
    const t0 = performance.now();
    let assistantText = "";
    let tokenCount = 0;

    try {
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal,
        body: JSON.stringify({
          model: currentModel,
          messages: messagesToSend,
          stream: true,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          // Pass-through field for Ollama-compatible backends that honour it.
          think: settings.think,
        }),
      });

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${errText.slice(0, 200)}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line || !line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const obj = JSON.parse(payload);
            const delta = obj.choices?.[0]?.delta?.content || "";
            if (delta) {
              assistantText += delta;
              tokenCount++;
              contentEl.innerHTML = renderContent(assistantText);
              scrollToBottom();
            }
          } catch (e) {
            console.warn("SSE parse error:", e, payload);
          }
        }
      }

      history.push({ role: "assistant", content: stripThink(assistantText) || assistantText });
      const dt = (performance.now() - t0) / 1000;
      const tps = tokenCount > 0 ? (tokenCount / dt).toFixed(1) : "—";
      metaEl.innerHTML = `<span>${tokenCount} chunks</span><span>${dt.toFixed(2)}s</span><span>~${tps} chunk/s</span>`;
    } catch (err) {
      if (err.name === "AbortError") {
        metaEl.innerHTML = `<span style="color: var(--danger)">Stopped</span>`;
      } else {
        contentEl.innerHTML = `<span style="color: var(--danger)">Error: ${escapeHtml(err.message)}</span>`;
        console.error(err);
      }
    } finally {
      msg.classList.remove("streaming");
      setSending(false);
      abortController = null;
    }
  }

  function setSending(busy) {
    els.sendBtn.disabled = busy || !els.input.value.trim();
    els.input.disabled = busy;
    els.stopBtn.hidden = !busy;
    els.sendBtn.hidden = busy;
    if (!busy) els.input.focus();
  }

  // ── Input UX ──────────────────────────────────────────────────────────
  function autoresize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
    els.sendBtn.disabled = !els.input.value.trim();
  }

  els.input.addEventListener("input", autoresize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const txt = els.input.value.trim();
    if (!txt) return;
    els.input.value = "";
    autoresize();
    send(txt);
  });

  els.stopBtn.addEventListener("click", () => abortController?.abort());

  els.clearBtn.addEventListener("click", () => {
    if (!history.length) return;
    if (!confirm("Clear conversation?")) return;
    history.length = 0;
    els.messages.innerHTML = "";
    loadModels(); // re-render the welcome block via re-init
    init(true);
  });

  els.modelSelect.addEventListener("change", (e) => {
    currentModel = e.target.value;
  });

  // ── Settings modal ────────────────────────────────────────────────────
  els.settingsBtn.addEventListener("click", () => {
    els.settingsSystem.value = settings.system;
    els.settingsTemp.value = settings.temperature;
    els.settingsTempVal.textContent = settings.temperature.toFixed(2);
    els.settingsMaxTokens.value = settings.maxTokens;
    els.settingsThink.checked = settings.think;
    els.settingsModal.showModal();
  });

  els.settingsTemp.addEventListener("input", () => {
    els.settingsTempVal.textContent = parseFloat(els.settingsTemp.value).toFixed(2);
  });

  els.settingsModal.addEventListener("close", () => {
    if (els.settingsModal.returnValue !== "ok") return;
    settings.system = els.settingsSystem.value.trim();
    settings.temperature = parseFloat(els.settingsTemp.value);
    settings.maxTokens = parseInt(els.settingsMaxTokens.value, 10) || 2048;
    settings.think = els.settingsThink.checked;
  });

  // Welcome-screen suggestion buttons
  document.addEventListener("click", (e) => {
    if (e.target.classList?.contains("suggest")) {
      els.input.value = e.target.textContent;
      autoresize();
      els.input.focus();
    }
  });

  // ── Boot ──────────────────────────────────────────────────────────────
  async function init(skipWelcomeReinject) {
    const v = await loadVersion();
    if (v?.version) window.__eullm_version = `v${v.version}`;
    await loadModels();
    if (!skipWelcomeReinject && !els.messages.querySelector(".welcome") && history.length === 0) {
      location.reload(); // simplest way to restore welcome after clear
    }
  }

  init();
})();
