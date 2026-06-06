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
    settingsMath: $("settings-math"),
    messages: $("messages"),
    form: $("chat-form"),
    input: $("input"),
    sendBtn: $("send-btn"),
    stopBtn: $("stop-btn"),
    clearBtn: $("clear-btn"),
    statusGrid: $("status-grid"),
  };

  const settings = {
    // Default math-formatting nudge: some models close a $...$ block before the
    // final fraction, leaving raw LaTeX. This asks them to delimit every full
    // formula. Purely a formatting hint — the user can clear it in Settings.
    system:
      "When you write mathematics, wrap each complete formula — including the " +
      "final result — in $...$ (inline) or $$...$$ (block) LaTeX delimiters. " +
      "Never leave commands like \\frac or \\sqrt outside the delimiters.",
    temperature: 0.7,
    maxTokens: 2048,
    // Reasoning ON by default. Reasoning models (DeepSeek-R1, QwQ) are trained
    // to always emit a <think> block; suppressing it (think:false injects an
    // empty <think></think>) makes them degenerate into a canned greeting.
    think: true,
    // Math rendering ON by default. Heuristics protect currency-looking `$NN`
    // patterns; user can disable from settings if false positives appear.
    math: true,
  };

  // Strip reasoning blocks from an assistant turn before storing it in history.
  // Re-sending the model's own reasoning back as context confuses reasoning
  // models and bloats the prompt; only the final answer belongs in history.
  // Covers Qwen3 `<think>…</think>` and Gemma 4 `<|channel>thought\n…<channel|>`.
  const stripThink = (s) =>
    s
      .replace(/<\|channel>thought\n?[\s\S]*?<channel\|>\s*/g, "")
      .replace(/<\|channel>thought\n?[\s\S]*$/, "")
      .replace(/<think>[\s\S]*?<\/think>\s*/g, "")
      .replace(/<think>[\s\S]*$/, "")
      .trim();

  const history = []; // {role, content}
  let currentModel = "";
  let abortController = null;

  // ── Presentation layer: Markdown-lite + Math-lite ──────────────────────
  // Everything in this section transforms the assistant's raw text for DISPLAY
  // ONLY. The original LaTeX/Markdown string is preserved in `assistantText`
  // and `history`, and the API stream is never touched. Any unsupported syntax
  // falls back to the raw escaped text — no broken HTML, no thrown errors.

  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // Reverse the escape so the math parser sees real `<`, `>`, `&`. We re-escape
  // inside MathML token content before injecting it back into the DOM.
  const unescapeHtml = (s) =>
    s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&");

  // ── Math: LaTeX subset → MathML ────────────────────────────────────────
  // Browsers render MathML natively (Chrome/Edge ≥ 109, Firefox always,
  // Safari ≥ 18). This translator covers the subset commonly produced by
  // LLMs and throws on anything else so the caller can fall back gracefully.

  const MATH_GREEK = {
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε",
    zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ",
    lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", varpi: "ϖ", rho: "ρ",
    varrho: "ϱ", sigma: "σ", varsigma: "ς", tau: "τ", upsilon: "υ", phi: "φ",
    varphi: "ϕ", chi: "χ", psi: "ψ", omega: "ω",
    Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
    Sigma: "Σ", Upsilon: "Υ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  };
  const MATH_OPS = {
    sum: "∑", prod: "∏", coprod: "∐", int: "∫", oint: "∮", iint: "∬", iiint: "∭",
    bigcup: "⋃", bigcap: "⋂", bigvee: "⋁", bigwedge: "⋀",
    times: "×", cdot: "·", div: "÷", pm: "±", mp: "∓", ast: "∗", star: "⋆",
    circ: "∘", bullet: "∙", oplus: "⊕", ominus: "⊖", otimes: "⊗",
    leq: "≤", le: "≤", geq: "≥", ge: "≥", neq: "≠", ne: "≠",
    approx: "≈", equiv: "≡", sim: "∼", simeq: "≃", cong: "≅",
    ll: "≪", gg: "≫", subset: "⊂", supset: "⊃", subseteq: "⊆", supseteq: "⊇",
    in: "∈", notin: "∉", ni: "∋", propto: "∝", perp: "⊥", parallel: "∥",
    to: "→", rightarrow: "→", leftarrow: "←", leftrightarrow: "↔",
    Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔", mapsto: "↦",
    infty: "∞", partial: "∂", nabla: "∇", forall: "∀", exists: "∃",
    emptyset: "∅", cdots: "⋯", ldots: "…", dots: "…", vdots: "⋮", ddots: "⋱",
    langle: "⟨", rangle: "⟩", lfloor: "⌊", rfloor: "⌋", lceil: "⌈", rceil: "⌉",
    log: "log", ln: "ln", sin: "sin", cos: "cos", tan: "tan",
    arcsin: "arcsin", arccos: "arccos", arctan: "arctan",
    sinh: "sinh", cosh: "cosh", tanh: "tanh",
    exp: "exp", lim: "lim", max: "max", min: "min", det: "det",
    gcd: "gcd", deg: "deg", arg: "arg", dim: "dim", ker: "ker",
  };

  const escapeMath = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Read a `{ ... }` group; returns { text, end } where end is past the `}`.
  function readGroup(s, i) {
    if (s[i] !== "{") throw new Error("expected {");
    let depth = 1, j = i + 1;
    while (j < s.length && depth > 0) {
      if (s[j] === "\\") { j += 2; continue; }
      if (s[j] === "{") depth++;
      else if (s[j] === "}") { depth--; if (depth === 0) break; }
      j++;
    }
    if (depth !== 0) throw new Error("unmatched {");
    return { text: s.slice(i + 1, j), end: j + 1 };
  }

  function latexToTokens(latex) {
    const out = [];
    let i = 0;
    while (i < latex.length) {
      const c = latex[i];
      if (c === "\\") {
        // LaTeX spacing commands with non-alphabetic names: \, \: \; \! — these
        // would fail the [a-zA-Z]+ match below and fall back to raw. Standard
        // TeX widths: \, = 3mu, \: = 4mu, \; = 5mu, \! = -3mu (1mu ≈ 0.0556em).
        const next = latex[i + 1];
        if (next === "," || next === ":" || next === ";" || next === "!") {
          const w = next === "," ? "0.167em"
                  : next === ":" ? "0.222em"
                  : next === ";" ? "0.278em"
                  :                "-0.167em";
          out.push(`<mspace width="${w}"/>`);
          i += 2;
          continue;
        }
        const m = latex.slice(i + 1).match(/^[a-zA-Z]+/);
        if (!m) throw new Error("bad escape");
        const name = m[0];
        i += 1 + name.length;
        if (latex[i] === " ") i++;
        if (name === "frac" || name === "dfrac" || name === "tfrac") {
          if (latex[i] !== "{") throw new Error("\\frac needs {");
          const num = readGroup(latex, i); i = num.end;
          if (latex[i] !== "{") throw new Error("\\frac needs second {");
          const den = readGroup(latex, i); i = den.end;
          out.push(`<mfrac><mrow>${latexToTokens(num.text)}</mrow><mrow>${latexToTokens(den.text)}</mrow></mfrac>`);
        } else if (name === "sqrt") {
          if (latex[i] !== "{") throw new Error("\\sqrt needs {");
          const arg = readGroup(latex, i); i = arg.end;
          out.push(`<msqrt>${latexToTokens(arg.text)}</msqrt>`);
        } else if (name === "text" || name === "mathrm" || name === "mathit" || name === "mathbf") {
          if (latex[i] !== "{") throw new Error("\\text needs {");
          const arg = readGroup(latex, i); i = arg.end;
          out.push(`<mtext>${escapeMath(arg.text)}</mtext>`);
        } else if (name === "left" || name === "right") {
          // Stretchy delimiters not modelled; the next char is emitted normally.
        } else if (name === "quad") {
          out.push('<mspace width="1em"/>');
        } else if (name === "qquad") {
          out.push('<mspace width="2em"/>');
        } else if (MATH_GREEK[name]) {
          out.push(`<mi>${MATH_GREEK[name]}</mi>`);
        } else if (MATH_OPS[name]) {
          out.push(`<mo>${MATH_OPS[name]}</mo>`);
        } else {
          throw new Error(`unknown \\${name}`);
        }
      } else if (c === "^" || c === "_") {
        const prev = out.pop();
        if (prev === undefined) throw new Error(`${c} without base`);
        i++;
        let argText;
        if (latex[i] === "{") {
          const g = readGroup(latex, i); argText = g.text; i = g.end;
        } else if (latex[i] === "\\") {
          const m = latex.slice(i + 1).match(/^[a-zA-Z]+/);
          if (!m) throw new Error("bad escape after script");
          argText = "\\" + m[0];
          i += 1 + m[0].length;
        } else if (i < latex.length) {
          argText = latex[i]; i++;
        } else {
          throw new Error("script at end");
        }
        const tag = c === "^" ? "msup" : "msub";
        out.push(`<${tag}>${prev}<mrow>${latexToTokens(argText)}</mrow></${tag}>`);
      } else if (/[0-9]/.test(c)) {
        const m = latex.slice(i).match(/^\d+(\.\d+)?/);
        out.push(`<mn>${m[0]}</mn>`);
        i += m[0].length;
      } else if (/[a-zA-Z]/.test(c)) {
        out.push(`<mi>${c}</mi>`); i++;
      } else if (c === " " || c === "\t" || c === "\n") {
        i++;
      } else if ("+-*/=<>(),;:!?|[]".includes(c)) {
        out.push(`<mo>${escapeMath(c)}</mo>`); i++;
      } else if (c === "{" || c === "}") {
        i++;
      } else if (c === "'") {
        out.push("<mo>'</mo>"); i++;
      } else {
        throw new Error(`unsupported char: ${c}`);
      }
    }
    return out.join("");
  }

  function latexToMathml(latex, displayMode) {
    const tokens = latexToTokens(unescapeHtml(latex));
    const attr = displayMode ? ' display="block"' : "";
    return `<math xmlns="http://www.w3.org/1998/Math/MathML"${attr}><mrow>${tokens}</mrow></math>`;
  }

  // Currency guard: `$100`, `$ 1,250.50` etc. — pure numeric content.
  const looksLikeCurrency = (s) => /^[\s\d.,]+$/.test(s);

  function tryMath(latex, displayMode, raw) {
    try {
      // Collapse internal newlines so block-level markdown won't split mid-<math>.
      return latexToMathml(latex, displayMode).replace(/\n/g, " ");
    } catch (_) {
      return `<span class="math-raw">${raw}</span>`;
    }
  }

  function renderMath(body) {
    // Display first (greediest), then inline.
    body = body.replace(/\$\$([\s\S]+?)\$\$/g, (raw, inner) => tryMath(inner, true, raw));
    body = body.replace(/\\\[([\s\S]+?)\\\]/g, (raw, inner) => tryMath(inner, true, raw));
    body = body.replace(/\\\((.+?)\\\)/g, (raw, inner) => tryMath(inner, false, raw));
    // Inline `$ ... $` with strict guards to avoid currency / random `$` pairs:
    //   - opening `$` not preceded by word char or another `$`
    //   - no whitespace right after `$` or right before closing `$`
    //   - closing `$` not followed by word char or digit
    body = body.replace(
      /(?<![\w$])\$(?!\s)([^\s$][^\n$]*?[^\s$]|[^\s$])\$(?![\w$\d])/g,
      (raw, inner) => looksLikeCurrency(inner) ? raw : tryMath(inner, false, raw)
    );
    // Orphan LaTeX: LLMs often close a $...$ block before the final fraction
    // and leave `\frac{..}{..}` / `\sqrt{..}` as bare text. A backslash command
    // with brace args never occurs in normal prose, so rendering it is safe.
    body = renderOrphanMath(body);
    return body;
  }

  // Render bare \frac{..}{..} and \sqrt{..} that sit OUTSIDE math delimiters,
  // without re-touching the <math> / .math-raw islands already produced above
  // (we only scan the plain-text gaps between them).
  function renderOrphanMath(body) {
    if (!/\\(?:frac|sqrt)/.test(body)) return body;
    const island = /<math[\s\S]*?<\/math>|<span class="math-raw">[\s\S]*?<\/span>/g;
    let out = "";
    let last = 0;
    let m;
    while ((m = island.exec(body)) !== null) {
      out += renderOrphanInPlain(body.slice(last, m.index));
      out += m[0];
      last = island.lastIndex;
    }
    out += renderOrphanInPlain(body.slice(last));
    return out;
  }

  function renderOrphanInPlain(text) {
    let out = "";
    let i = 0;
    while (i < text.length) {
      if (text[i] === "\\") {
        const cmd = text.slice(i + 1).match(/^(frac|sqrt)/);
        if (cmd) {
          const name = cmd[1];
          let k = i + 1 + name.length;
          let ok = true;
          try {
            if (name === "sqrt" && text[k] === "[") {
              const close = text.indexOf("]", k);
              if (close === -1) throw new Error("bad");
              k = close + 1;
            }
            const groups = name === "frac" ? 2 : 1;
            for (let g = 0; g < groups; g++) {
              if (text[k] !== "{") throw new Error("bad");
              k = readGroup(text, k).end;
            }
          } catch (_) {
            ok = false;
          }
          if (ok) {
            const span = text.slice(i, k);
            out += tryMath(span, false, span);
            i = k;
            continue;
          }
        }
      }
      out += text[i];
      i++;
    }
    return out;
  }


  // ── Markdown-lite: headings, lists, blockquote, hr, bold/italic ────────
  // Line-based block processing (so a heading at line 1 doesn't bleed into
  // the next line), then inline emphasis passes. Code spans are stashed
  // upstream so their `*`, `_`, `#` characters never reach this pass.

  function renderInline(s) {
    // Bold first (so its `*` aren't consumed by the italic pass).
    s = s.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__([^_\n]+?)__/g, "<strong>$1</strong>");
    // Italic: only `*` (the `_` variant clashes too often with snake_case).
    s = s.replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
    return s;
  }

  function renderMarkdownBlocks(text) {
    const lines = text.split("\n");
    const out = [];
    let listKind = null; // "ul" | "ol" | null
    const closeList = () => { if (listKind) { out.push(`</${listKind}>`); listKind = null; } };

    // A line that should NOT break an open list: blank lines, and lines whose
    // only content is a single block-level <math> island. Both are common
    // separators between numbered items when math appears in the middle.
    const inListAllowed = (l) => {
      if (!listKind) return false;
      if (!l.trim()) return true;
      return /^\s*<math[^>]*display="block"[^>]*>[\s\S]*?<\/math>\s*$/.test(l);
    };

    for (const line of lines) {
      // Horizontal rule
      if (/^[-_*]{3,}\s*$/.test(line)) {
        closeList(); out.push("<hr/>"); continue;
      }
      // Headings #..######
      const h = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
      if (h) {
        closeList();
        const lvl = h[1].length;
        out.push(`<h${lvl}>${renderInline(h[2])}</h${lvl}>`);
        continue;
      }
      // Unordered list item
      const ul = line.match(/^\s*[-*+]\s+(.+)$/);
      if (ul) {
        if (listKind !== "ul") { closeList(); out.push("<ul>"); listKind = "ul"; }
        out.push(`<li>${renderInline(ul[1])}</li>`);
        continue;
      }
      // Ordered list item
      const ol = line.match(/^\s*\d+\.\s+(.+)$/);
      if (ol) {
        if (listKind !== "ol") { closeList(); out.push("<ol>"); listKind = "ol"; }
        out.push(`<li>${renderInline(ol[1])}</li>`);
        continue;
      }
      // Blockquote (after HTML escape, `>` is `&gt;`)
      const bq = line.match(/^\s*&gt;\s?(.*)$/);
      if (bq) {
        closeList();
        out.push(`<blockquote>${renderInline(bq[1])}</blockquote>`);
        continue;
      }
      // Plain line. Keep the list alive across blank lines and standalone
      // display-math blocks — otherwise every `1.` after a `$$...$$` would
      // start a new <ol> and the numbering would visibly restart at 1.
      if (!inListAllowed(line)) {
        closeList();
      }
      out.push(renderInline(line));
    }
    closeList();
    return out.join("\n");
  }

  function renderContent(text) {
    // 1a. Pull out Gemma 4 channel-thought blocks, then Qwen3 <think> blocks.
    // Gemma 4 12B emits  `<|channel>thought\n[reasoning]<channel|>[final]`
    // (Google's official prompt format). Surface the reasoning as a Reasoning
    // section identical to <think>, then keep the final answer as body text.
    let thinkHtml = "";
    text = text.replace(/<\|channel>thought\n?([\s\S]*?)<channel\|>/g, (_, inner) => {
      const trimmed = inner.trim();
      if (trimmed) {
        thinkHtml += `<div class="think"><div class="think-label">Reasoning</div>${escapeHtml(trimmed)}</div>`;
      }
      return "";
    });
    // Unclosed Gemma channel-thought at the end (streaming in progress).
    text = text.replace(/<\|channel>thought\n?([\s\S]*)$/, (_, inner) => {
      thinkHtml += `<div class="think"><div class="think-label">Reasoning…</div>${escapeHtml(inner.trim())}</div>`;
      return "";
    });
    // 1b. Pull out Qwen3 <think>...</think> blocks.
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

    // 2a. Collapse runs of blank lines (LLMs commonly emit `\n\n\n`+, which —
    //     combined with `white-space: pre-wrap` on .msg-content — creates
    //     visible empty lines that pile up on top of block-element margins).
    //     One blank line between blocks is enough to separate paragraphs.
    text = text.replace(/\n{3,}/g, "\n\n");

    // 2b. Escape HTML first so user/model text can never inject tags.
    let body = escapeHtml(text);

    // 3. Stash code spans (fenced + inline) as opaque placeholders so the
    //    Markdown and math passes don't see their `$`, `*`, `_`, `#` chars.
    const stash = [];
    const stashPush = (html) => `CODE${stash.push(html) - 1}`;
    body = body.replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (_, lang, code) =>
      stashPush(`<pre><code class="lang-${lang || "txt"}">${code}</code></pre>`));
    body = body.replace(/`([^`\n]+)`/g, (_, code) => stashPush(`<code>${code}</code>`));

    // 4. Math rendering (best-effort, opt-out via settings.math).
    if (settings.math) body = renderMath(body);

    // 5. Markdown blocks + inline emphasis.
    body = renderMarkdownBlocks(body);

    // 6. Restore code spans.
    body = body.replace(/CODE(\d+)/g, (_, i) => stash[parseInt(i, 10)]);

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

  // Stick-to-bottom: auto-scroll during streaming ONLY when the user is
  // already near the bottom. Once they scroll up to read, stop auto-scrolling
  // so they can keep reading while the answer keeps streaming in the
  // background. Re-enables itself when they scroll back near the bottom.
  let stickToBottom = true;
  const STICK_THRESHOLD_PX = 80;

  els.messages.addEventListener("scroll", () => {
    const { scrollTop, scrollHeight, clientHeight } = els.messages;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    stickToBottom = distanceFromBottom < STICK_THRESHOLD_PX;
    updateJumpButton();
  });

  function scrollToBottom(force = false) {
    if (force || stickToBottom) {
      els.messages.scrollTop = els.messages.scrollHeight;
    }
  }

  // Floating "jump to latest" button — shown only when the user has scrolled
  // up during an active stream, so they have a one-click way back without
  // disturbing the reading position they chose.
  const jumpBtn = document.createElement("button");
  jumpBtn.type = "button";
  jumpBtn.className = "jump-to-bottom";
  jumpBtn.textContent = "↓ Jump to latest";
  jumpBtn.hidden = true;
  jumpBtn.addEventListener("click", () => {
    stickToBottom = true;
    scrollToBottom(true);
    updateJumpButton();
  });
  document.body.appendChild(jumpBtn);

  function updateJumpButton() {
    const streaming = !!document.querySelector(".message.streaming");
    jumpBtn.hidden = stickToBottom || !streaming;
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
    updateJumpButton();
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
      updateJumpButton();
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
    els.settingsMath.checked = settings.math;
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
    settings.math = els.settingsMath.checked;
    // Re-render the last assistant turn so the math toggle takes effect immediately.
    const lastAssistant = els.messages.querySelector(".msg.assistant:last-child .msg-content");
    if (lastAssistant && history.length) {
      const lastTurn = history[history.length - 1];
      if (lastTurn?.role === "assistant") {
        lastAssistant.innerHTML = renderContent(lastTurn.content);
      }
    }
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
