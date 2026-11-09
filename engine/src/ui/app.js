// EuLLM Chat UI — talks to /v1/chat/completions (OpenAI-compatible, SSE).
// Conversation lives in memory; no persistence, no telemetry.

(() => {
  "use strict";

  // ── API key ────────────────────────────────────────────────────────────
  // When the engine is started with EULLM_API_KEYS, every request to /api and
  // /v1 needs a bearer token — including the ones this page makes. A browser
  // cannot be handed a header before its first navigation, so the key arrives
  // once as ?api_key=… , is kept in sessionStorage (cleared when the tab
  // closes, unlike localStorage) and is then sent as a header on every fetch.
  // The query string is stripped from the visible URL immediately so the key
  // does not sit in the address bar, get bookmarked, or leak through Referer.
  //
  // With no keys configured the engine ignores the header entirely, so this
  // costs nothing in the common local case.
  const API_KEY_STORAGE = "eullm.apiKey";
  const apiKey = (() => {
    const fromUrl = new URLSearchParams(location.search).get("api_key");
    if (fromUrl) {
      try { sessionStorage.setItem(API_KEY_STORAGE, fromUrl); } catch {}
      const clean = location.pathname + location.hash;
      history.replaceState(null, "", clean);
      return fromUrl;
    }
    try { return sessionStorage.getItem(API_KEY_STORAGE) || ""; } catch { return ""; }
  })();

  /** Merge the auth header into a fetch init, leaving everything else alone. */
  const withAuth = (init = {}) => {
    if (!apiKey) return init;
    return { ...init, headers: { ...(init.headers || {}), "X-Api-Key": apiKey } };
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    modelSelect: $("model-select"),
    settingsBtn: $("settings-btn"),
    params: $("params"),
    paramsClose: $("params-close"),
    paramsReset: $("params-reset"),
    settingsSystem: $("settings-system"),
    settingsMaxTokens: $("settings-max-tokens"),
    settingsSeed: $("settings-seed"),
    settingsThink: $("settings-think"),
    settingsMath: $("settings-math"),
    messages: $("messages"),
    form: $("chat-form"),
    input: $("input"),
    sendBtn: $("send-btn"),
    stopBtn: $("stop-btn"),
    clearBtn: $("clear-btn"),
    statusGrid: $("status-grid"),
    attachBtn: $("attach-btn"),
    imageInput: $("image-input"),
    attachmentBar: $("attachment-bar"),
    attachmentThumb: $("attachment-thumb"),
    attachmentAudio: $("attachment-audio"),
    attachmentName: $("attachment-name"),
    attachmentRemove: $("attachment-remove"),
    sidebar: $("sidebar"),
    scrim: $("scrim"),
    menuBtn: $("menu-btn"),
    viewTitle: $("view-title"),
    newChat: $("new-chat"),
    chatList: $("chat-list"),
    sideVersion: $("side-version"),
  };
  // The welcome block as the page ships it, put back by a new chat.
  const WELCOME_HTML = els.messages.innerHTML;

  // Pending media (image OR audio) attached to the next outgoing user
  // message. `dataUrl` drives the in-page preview; `base64` (no `data:`
  // prefix) is what we ship to the backend over /api/chat; `kind` is
  // "image" | "audio"; `name` is the original filename. The backend's mtmd
  // path auto-detects image vs audio from the bytes, so both travel through
  // the same `images:[...]` field. Cleared after each send.
  let pendingMedia = null;

  const settings = {
    // Free-form system prompt, opt-in, sent as a literal system-role message
    // when set (e.g. persona/tone/language instructions the user types in
    // Settings). Left empty by default.
    //
    // A default math-formatting nudge lived here through rc10-rc12 and was
    // removed for good after two separate real-hardware failures: as a
    // system-role message (rc10) it sent DeepSeek-R1-Distill-Qwen-14B into
    // an unrelated reasoning trace and hallucinated identities on
    // Qwen2-VL-2B/gemma-4-e4b; moving it into the user turn instead (rc12,
    // to dodge R1's documented sensitivity to system prompts) did not fix
    // it — same question, same model, still hallucinated (invented the name
    // "MathAI") once the hint text rode along in the same turn. The common
    // factor both times was appending unsolicited instructions to a short,
    // unrelated prompt, not which role carried them. `--cli`, which sends
    // neither, answers correctly every time. Any such nudge is opt-in only
    // now, typed by the user into this field.
    system: "",
    temperature: 0.7,
    // The engine's own defaults (Ollama's), sent explicitly so that what the
    // panel shows is what the request asks for.
    topP: 0.9,
    topK: 40,
    minP: 0,
    repeatPenalty: 1.1,
    // null = a new seed each answer.
    seed: null,
    // 0 = unlimited: max_tokens is omitted from the request and the server
    // generates until the model stops or the context window fills (its own
    // default, matching Ollama's num_predict=-1). A fixed cap here truncated
    // reasoning models mid-think: Qwen3.6 spent ~2000 tokens thinking about a
    // hard question and hit the old 2048 default before answering at all.
    maxTokens: 0,
    // Reasoning ON by default. Reasoning models (DeepSeek-R1, QwQ) are trained
    // to always emit a <think> block; suppressing it (think:false injects an
    // empty <think></think>) makes them degenerate into a canned greeting.
    // On models whose own template has a reasoning toggle (Qwen3 family) the
    // server maps think:false to the template's enable_thinking=false, which
    // renders the model's official suppression form.
    think: true,
    // Math rendering ON by default. Heuristics protect currency-looking `$NN`
    // patterns; user can disable from settings if false positives appear.
    math: true,
  };

  // The panel's values are kept in this browser (not the conversation, which
  // never is), so the next visit starts from them. Storage that is blocked
  // or cleared just means the defaults.
  const DEFAULT_SETTINGS = { ...settings };
  const SETTINGS_STORAGE = "eullm.params";
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_STORAGE) || "{}");
    for (const k of Object.keys(DEFAULT_SETTINGS)) {
      const fits = k === "seed"
        ? saved[k] === null || Number.isInteger(saved[k])
        : typeof saved[k] === typeof DEFAULT_SETTINGS[k];
      if (k in saved && fits) settings[k] = saved[k];
    }
  } catch {}
  const saveSettings = () => {
    try { localStorage.setItem(SETTINGS_STORAGE, JSON.stringify(settings)); } catch {}
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

  // /api/chat streams the reasoning apart when asked to think, in Ollama's
  // `message.thinking`. Put back in front of the answer as a `<think>` block,
  // it renders as the same reasoning section and strips the same way; left
  // open while no answer has come, so it reads as reasoning in progress.
  const withThinking = (thinking, answer) =>
    !thinking ? answer : answer ? `<think>${thinking}</think>${answer}` : `<think>${thinking}`;

  // {role, content}, plus `media: {base64, kind}` on a user turn that carried
  // an attachment (see toApiMessages for how it is re-sent).
  const history = [];
  let currentModel = "";
  // What /api/tags said last: the model picker and the Model hub's list.
  let modelsData = [];
  const loadedModelEntry = () => modelsData.find((m) => m.loaded);
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

  // Split a markdown table row into cells: outer pipes are optional, inner
  // pipes separate. Pipes inside code spans can't reach here — code is
  // stashed as opaque placeholders before the block pass runs.
  function splitTableRow(line) {
    let t = line.trim();
    if (t.startsWith("|")) t = t.slice(1);
    if (t.endsWith("|")) t = t.slice(0, -1);
    return t.split("|");
  }

  // The row under a table header: only dashes and alignment colons per cell,
  // e.g. `|---|:---:|--:|`. This is what commits the block as a table.
  function isTableSeparator(line) {
    if (!line || !line.includes("|") || !line.includes("-")) return false;
    const cells = splitTableRow(line);
    return cells.length > 0 && cells.every((c) => /^\s*:?-+:?\s*$/.test(c));
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

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      // GFM table: a row with pipes whose NEXT line is a separator row with
      // the same cell count (issue #335 — tables used to render as plain
      // text with visible pipes). During streaming the separator hasn't
      // arrived yet, so the header shows as text for a moment and snaps
      // into a table on the next re-render; that's fine.
      if (
        line.includes("|") &&
        i + 1 < lines.length &&
        isTableSeparator(lines[i + 1]) &&
        splitTableRow(line).length === splitTableRow(lines[i + 1]).length
      ) {
        closeList();
        const aligns = splitTableRow(lines[i + 1]).map((c) => {
          const t = c.trim();
          if (t.startsWith(":") && t.endsWith(":")) return "center";
          if (t.endsWith(":")) return "right";
          return null;
        });
        const cell = (c, tag, k) =>
          `<${tag}${aligns[k] ? ` style="text-align:${aligns[k]}"` : ""}>${renderInline(c.trim())}</${tag}>`;
        const rows = [];
        rows.push("<tr>" + splitTableRow(line).map((c, k) => cell(c, "th", k)).join("") + "</tr>");
        i += 1; // consume the separator row
        while (i + 1 < lines.length && lines[i + 1].includes("|") && lines[i + 1].trim()) {
          i += 1;
          rows.push("<tr>" + splitTableRow(lines[i]).map((c, k) => cell(c, "td", k)).join("") + "</tr>");
        }
        out.push(
          `<table><thead>${rows[0]}</thead><tbody>${rows.slice(1).join("")}</tbody></table>`,
        );
        continue;
      }
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

    // The message body is white-space: pre-wrap, so blank SOURCE lines
    // render as visible gaps — and block elements bring their own CSS
    // margins, so a blank line next to a heading, list, table or code block
    // doubles the spacing (H4-B: roughly twice the intended air between
    // blocks). Swallow blank lines that touch block-level output; blank
    // lines between plain-text paragraphs stay, because there pre-wrap is
    // exactly what separates the paragraphs.
    const blockish = (s) =>
      /^<(?:h[1-6]|ul|ol|\/ul|\/ol|li|hr|table|blockquote)\b/.test(s) ||
      /^CODE\d+$/.test(s.trim());
    const compact = [];
    for (let k = 0; k < out.length; k++) {
      if (!out[k].trim()) {
        const prev = compact.length ? compact[compact.length - 1] : null;
        let next = null;
        for (let j = k + 1; j < out.length; j++) {
          if (out[j].trim()) { next = out[j]; break; }
        }
        if ((prev && blockish(prev)) || (next && blockish(next))) continue;
      }
      compact.push(out[k]);
    }

    // Second layer of the same problem: the join separator itself. Under
    // pre-wrap, a "\n" BETWEEN two block elements (two <li>, two
    // blockquotes, a </ul> and a <table>) renders as an empty line box on
    // top of the elements' margins. Block elements break the line on their
    // own, so the newline is only needed between two plain-text entries.
    let html = "";
    for (let k = 0; k < compact.length; k++) {
      if (k > 0 && !blockish(compact[k]) && !blockish(compact[k - 1])) {
        html += "\n";
      }
      html += compact[k];
    }
    return html;
  }

  // A reasoning block, folded once the answer has begun. `think` carries what
  // the caller knows: `open` (the reader's own choice, or null to fold it
  // when it is done), `seconds` (how long it took, or has taken so far).
  function thinkBlock(inner, done, think) {
    const trimmed = inner.trim();
    if (done && !trimmed) return "";
    const open = think.open ?? !done;
    const secs = think.seconds;
    const time = secs == null ? "" : secs < 10 ? `${secs.toFixed(1)} s` : `${Math.round(secs)} s`;
    const label = done ? (time ? `Thought for ${time}` : "Reasoning") : `Thinking${time ? ` · ${time}` : "…"}`;
    return `<details class="think${done ? "" : " live"}"${open ? " open" : ""}><summary>${label}</summary>` +
      `<div class="think-body">${escapeHtml(trimmed)}</div></details>`;
  }

  function renderContent(text, think = {}) {
    // 1a. Pull out Gemma 4 channel-thought blocks, then Qwen3 <think> blocks.
    // Gemma 4 12B emits  `<|channel>thought\n[reasoning]<channel|>[final]`
    // (Google's official prompt format). Surface the reasoning as a Reasoning
    // section identical to <think>, then keep the final answer as body text.
    let thinkHtml = "";
    text = text.replace(/<\|channel>thought\n?([\s\S]*?)<channel\|>/g, (_, inner) => {
      thinkHtml += thinkBlock(inner, true, think);
      return "";
    });
    // Unclosed Gemma channel-thought at the end (streaming in progress).
    text = text.replace(/<\|channel>thought\n?([\s\S]*)$/, (_, inner) => {
      thinkHtml += thinkBlock(inner, false, think);
      return "";
    });
    // 1b. Pull out Qwen3 <think>...</think> blocks.
    text = text.replace(/<think>([\s\S]*?)<\/think>/g, (_, inner) => {
      thinkHtml += thinkBlock(inner, true, think);
      return "";
    });
    // Unclosed <think> at the end (streaming): render as in-progress reasoning.
    text = text.replace(/<think>([\s\S]*)$/, (_, inner) => {
      thinkHtml += thinkBlock(inner, false, think);
      return "";
    });

    // The blank lines a model writes after its reasoning would show as a gap
    // under the folded block.
    if (thinkHtml) text = text.replace(/^\s+/, "");

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

  function appendMessage(role, content = "", media = null) {
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
    // Media first, then text — matches how the model sees it (marker, then prompt).
    if (media && media.kind === "image") {
      const img = document.createElement("img");
      img.className = "msg-image";
      img.src = media.dataUrl;
      img.alt = "Attached image";
      contentEl.appendChild(img);
    } else if (media && media.kind === "missing") {
      // A reopened chat: the attachment was not kept in this browser.
      const gone = document.createElement("span");
      gone.className = "msg-missing";
      gone.textContent = `${media.label === "audio" ? "Audio" : "Image"} not kept in this browser`;
      contentEl.appendChild(gone);
    } else if (media && media.kind === "audio") {
      const audio = document.createElement("audio");
      audio.className = "msg-audio";
      audio.controls = true;
      audio.src = media.dataUrl;
      contentEl.appendChild(audio);
    }
    if (content) {
      const textEl = document.createElement("div");
      textEl.innerHTML = renderContent(content);
      contentEl.appendChild(textEl);
    }
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
    const streaming = !!document.querySelector(".msg.streaming");
    jumpBtn.hidden = stickToBottom || !streaming;
  }

  // ── API calls ─────────────────────────────────────────────────────────
  async function loadModels() {
    const keepModel = currentModel;
    try {
      const r = await fetch("/api/tags", withAuth());
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      // Three groups, and the middle one is the point: a model whose weights
      // are on disk is selectable even when nothing is loaded yet, because the
      // server swaps to it on the first request. Until `downloaded` existed
      // every catalog entry was disabled, so starting with `eullm serve` and
      // an already-pulled model left the picker empty and the UI unusable.
      // `loaded` comes from the server. It replaced a heuristic — "the loaded
      // entry is the one with no digest" — which was true only for a model
      // started from a file path: a catalog model in the slot is answered with
      // its catalog digest, so it read as not-downloaded, every option was
      // disabled, and the UI said "No model loaded" while that model was
      // loaded and answering.
      const all = data.models || [];
      modelsData = all;
      const reallyLoaded = all.filter((m) => m.loaded);
      const onDisk = all.filter((m) => !m.loaded && m.downloaded);
      const catalog = all.filter((m) => !m.loaded && !m.downloaded);

      const addGroup = (label, items, disabled, text) => {
        if (!items.length) return;
        const group = document.createElement("optgroup");
        group.label = label;
        for (const m of items) {
          const opt = document.createElement("option");
          opt.value = m.name;
          opt.textContent = text(m);
          opt.disabled = disabled;
          group.appendChild(opt);
        }
        els.modelSelect.appendChild(group);
      };

      els.modelSelect.innerHTML = "";
      addGroup("Loaded", reallyLoaded, false, (m) => m.name);
      addGroup("On this machine", onDisk, false, (m) => `${m.name} — ${(m.size / 1e9).toFixed(1)} GB`);
      addGroup("EU Catalog (not yet downloaded)", catalog, true, (m) => `${m.name} — ${(m.size / 1e9).toFixed(1)} GB`);

      // Prefer what is already loaded; otherwise pre-select something that can
      // actually answer, so the first message does not need a manual pick.
      currentModel = reallyLoaded[0]?.name || onDisk[0]?.name || "";
      // A model already picked stays picked when it is still offered.
      const keep = keepModel && all.some((m) => m.name === keepModel && (m.loaded || m.downloaded));
      currentModel = keep ? keepModel : currentModel;
      if (currentModel) els.modelSelect.value = currentModel;
      updateStatus(reallyLoaded[0]);
      document.dispatchEvent(new CustomEvent("eullm:models"));
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
      const r = await fetch("/api/version", withAuth());
      if (!r.ok) return null;
      return await r.json();
    } catch { return null; }
  }

  function updateStatus(loadedModel) {
    if (!els.statusGrid) return;
    // The chat UI lives on its own port; external clients (Open WebUI,
    // LangChain, RAG) must point at the canonical API port (Ollama default
    // 11434), reported by /api/version. Fall back to this origin if unknown.
    const apiOrigin = window.__eullm_api_port
      ? `${location.protocol}//${location.hostname}:${window.__eullm_api_port}`
      : location.origin;
    const items = [
      ["Engine", `${window.__eullm_version || "v?"}`],
      ["API", apiOrigin],
      ["Chat UI", location.origin],
      ["APIs", "/api  +  /v1"],
      ["Model", loadedModel?.name || "(none loaded)"],
      ["Telemetry", "off — local-only audit"],
    ];
    els.statusGrid.innerHTML = items
      .map(([k, v]) => `<span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(v)}</span>`)
      .join("");
  }

  // ── Streaming chat ────────────────────────────────────────────────────

  // Every turn re-sends the attachments still in the conversation — that is
  // what lets a follow-up question see the photo it is about — and the
  // engine refuses request bodies over 64 MiB (MAX_BODY_BYTES). Newest
  // first, attachments spend this budget; each older one goes as the note
  // below instead, so a long photo conversation runs short of pictures
  // rather than failing every turn with 413 from then on. The context
  // usually runs out long before this does, and then it is the engine that
  // drops the oldest attachments, with the same note.
  const MEDIA_RESEND_BUDGET = 48 * 1024 * 1024; // base64 characters

  // Word for word the engine's DROPPED_IMAGE_NOTE / DROPPED_AUDIO_NOTE, so the
  // model reads one kind of note whichever side dropped the attachment.
  const DROPPED_NOTE = {
    image:
      "[An image was attached here. It is no longer part of this conversation, so it " +
      "cannot be viewed; only what was said about it remains.]",
    audio:
      "[An audio clip was attached here. It is no longer part of this conversation, so it " +
      "cannot be heard; only what was said about it remains.]",
  };

  // History → API messages: role and content only (history entries also carry
  // UI-internal fields: the authoring model, model-switch metadata), plus
  // `images` for each attachment that still fits the budget. Walks newest to
  // oldest so the budget goes to what the conversation is about now: the
  // newest attachment is always sent, and once one does not fit no older one
  // is sent either — a conversation forgets from its start. An attachment
  // whose own turn failed is never re-sent (see send()), and does not count.
  function toApiMessages(entries) {
    let budget = MEDIA_RESEND_BUDGET;
    let newest = true;
    let dropping = false;
    const out = new Array(entries.length);
    for (let i = entries.length - 1; i >= 0; i--) {
      const { role, content, media } = entries[i];
      let resend = false;
      if (media && !media.failed && !dropping) {
        resend = newest || media.base64.length <= budget;
        dropping = !resend;
        newest = false;
      }
      if (resend) {
        budget -= media.base64.length;
        out[i] = { role, content, images: [media.base64] };
      } else if (media) {
        out[i] = { role, content: `${DROPPED_NOTE[media.kind] || DROPPED_NOTE.image}\n${content}` };
      } else {
        out[i] = { role, content };
      }
    }
    return out;
  }

  async function send(userText) {
    if (!currentModel) {
      // Reached only when nothing is loaded and nothing is on disk either:
      // with `eullm serve` a downloaded model is selectable and swaps in on
      // the first request, so telling everyone to restart the engine was
      // wrong advice for the one command that does not need it.
      alert(
        "No model available.\n\nDownload one:\n  eullm pull gemma-4-12b\n\n" +
          "or start the engine with a file:\n  eullm run /path/to/model.gguf",
      );
      return;
    }
    // Snapshot + clear the pending media at send time so a fast re-attach
    // mid-stream cannot mix into the next turn. `media` is null for normal
    // text turns; `dataUrl` is for the preview only, `base64` is what
    // /api/chat consumes.
    const media = pendingMedia;
    clearAttachment();

    // The attachment stays with its turn, and every later turn re-sends it
    // (see toApiMessages): a follow-up question about the photo still sees
    // the photo. It used to be sent once and forgotten, and the next turn's
    // answer came from the model's memory of its own description.
    const userEntry = { role: "user", content: userText };
    if (media) userEntry.media = { base64: media.base64, kind: media.kind };
    history.push(userEntry);
    appendMessage("user", userText, media);
    saveChat();

    // An attachment whose own turn failed is not re-sent. When the failure
    // was the attachment's — a file the engine cannot decode, an image past
    // its token cap — sending it again would fail every turn after this one
    // as well. The model reads the note in its place; attaching the file
    // again sends it afresh.
    const markAttachmentFailed = () => {
      if (userEntry.media) userEntry.media.failed = true;
    };

    const { msg, contentEl, metaEl } = appendMessage("assistant", "");
    msg.classList.add("streaming");
    updateJumpButton();
    setSending(true);

    abortController = new AbortController();
    const t0 = performance.now();
    let assistantText = "";
    let thinkingText = "";
    let chunks = 0;
    let firstChunkAt = null;
    // The reasoning's own clock: from its first piece to the answer's first.
    let thinkStart = null;
    let thinkEnd = null;
    // null = folded by itself once the answer begins; a click makes it the
    // reader's choice for the rest of this answer.
    let thinkOpen = null;
    // The engine's own figures, from the last line of the stream.
    let final = null;
    let finishReason = null;

    const thinking = (full) =>
      /<think>(?![\s\S]*<\/think>)/.test(full) || /<\|channel>thought(?![\s\S]*<channel\|>)/.test(full);
    const thinkSeconds = () =>
      thinkStart == null ? null : ((thinkEnd ?? performance.now()) - thinkStart) / 1000;
    const paint = (ended = false) => {
      let full = withThinking(thinkingText, assistantText);
      // An answer cut off mid-reasoning shows its reasoning as finished.
      if (ended && thinking(full)) full += /<think>(?![\s\S]*<\/think>)/.test(full) ? "</think>" : "<channel|>";
      contentEl.innerHTML = renderContent(full, { open: thinkOpen, seconds: thinkSeconds() });
    };
    msg.addEventListener("click", (e) => {
      const summary = e.target.closest("summary");
      if (!summary || !msg.contains(summary)) return;
      thinkOpen = !summary.parentElement.open;
    });
    // The reasoning's clock moves while no new piece comes.
    const ticker = setInterval(() => {
      if (thinkStart != null && thinkEnd == null) paint();
    }, 500);

    try {
      const messagesToSend = [];
      if (settings.system) messagesToSend.push({ role: "system", content: settings.system });
      messagesToSend.push(...toApiMessages(history));
      // A conversation still carrying attachments goes to /api/chat (Ollama
      // NDJSON, `images: [base64]` on each message that has one — image and
      // audio alike, the engine tells them apart by their bytes); a text-only
      // one to /v1/chat/completions (OpenAI SSE). For the same text both
      // render the same prompt.
      const ndjson = messagesToSend.some((m) => m.images);
      const resp = await fetch(ndjson ? "/api/chat" : "/v1/chat/completions", withAuth({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal,
        body: JSON.stringify({
          model: currentModel,
          messages: messagesToSend,
          stream: true,
          temperature: settings.temperature,
          top_p: settings.topP,
          top_k: settings.topK,
          min_p: settings.minP,
          repeat_penalty: settings.repeatPenalty,
          ...(settings.seed != null ? { seed: settings.seed } : {}),
          // 0 = unlimited: omit the cap and let the server generate until
          // the model stops or the context window fills.
          ...(settings.maxTokens > 0 ? { max_tokens: settings.maxTokens } : {}),
          // Read by both endpoints. The attachment request used to leave it
          // out, so switching thinking off did nothing on a turn with a
          // picture in it.
          think: settings.think,
        }),
      }));

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
          if (!line) continue;
          // Two streaming formats live behind the same loop:
          //   * Ollama NDJSON (/api/chat, with attachments): one JSON object
          //     per line, no `data:` prefix; delta = `message.content`.
          //   * OpenAI SSE (/v1/chat/completions): `data: {...}` lines + a
          //     trailing `[DONE]`; delta = `choices[0].delta.content`.
          let payload, delta;
          if (ndjson) {
            payload = line;
          } else {
            if (!line.startsWith("data:")) continue;
            payload = line.slice(5).trim();
            if (payload === "[DONE]") continue;
          }
          try {
            const obj = JSON.parse(payload);
            // The backend emits `{"error": "..."}` as a stream line on a
            // failure (e.g. an undecodable image). Surface it instead of
            // ending silently with an empty "0 chunks" reply. A plain
            // `throw` here would be swallowed by the inner catch below, so
            // render it and bail out of send() directly (finally{} restores
            // the composer).
            if (obj.error) {
              contentEl.innerHTML =
                `<span style="color: var(--danger)">Error: ${escapeHtml(obj.error)}</span>`;
              metaEl.innerHTML = "";
              markAttachmentFailed();
              return;
            }
            // The last line carries the engine's counts: Ollama's
            // `eval_count`/`*_duration` (nanoseconds), or OpenAI's `usage`
            // with llama-server's `timings` (milliseconds).
            if (ndjson && obj.done) {
              final = {
                promptN: obj.prompt_eval_count, promptMs: obj.prompt_eval_duration / 1e6,
                outN: obj.eval_count, outMs: obj.eval_duration / 1e6,
                drafts: obj.draft_n, kept: obj.draft_n_accepted,
              };
              finishReason = obj.done_reason;
            } else if (!ndjson && obj.timings) {
              const t = obj.timings;
              final = {
                promptN: t.prompt_n, promptMs: t.prompt_ms, outN: t.predicted_n, outMs: t.predicted_ms,
                drafts: t.draft_n, kept: t.draft_n_accepted,
              };
            }
            if (!ndjson && obj.choices?.[0]?.finish_reason) finishReason = obj.choices[0].finish_reason;
            const d = obj.choices?.[0]?.delta;
            const thought = ndjson ? (obj.message?.thinking || "") : (d?.reasoning_content || "");
            delta = ndjson ? (obj.message?.content || "") : (d?.content || "");
            if (thought || delta) {
              const now = performance.now();
              if (firstChunkAt == null) firstChunkAt = now;
              thinkingText += thought;
              assistantText += delta;
              chunks++;
              const full = withThinking(thinkingText, assistantText);
              if (thinking(full)) {
                if (thinkStart == null) thinkStart = now;
              } else if (thinkStart != null && thinkEnd == null) {
                thinkEnd = now;
              }
              paint();
              const secs = (now - (firstChunkAt ?? t0)) / 1000;
              metaEl.innerHTML = thinkStart != null && thinkEnd == null
                ? `<span>Thinking</span>`
                : `<span>Writing</span><span>${chunks.toLocaleString()} tokens</span>${secs > 0.5 ? `<span>${(chunks / secs).toFixed(1)} tok/s</span>` : ""}`;
              scrollToBottom();
            }
          } catch (e) {
            console.warn("stream parse error:", e, payload);
          }
        }
      }

      if (thinkStart != null && thinkEnd == null) thinkEnd = performance.now();
      paint(true);
      // Tag the turn with the model that wrote it: after a mid-conversation
      // model switch, the send path uses this to tell the new model that the
      // earlier assistant turns are not its own words (see modelSwitchNotice).
      const reply = withThinking(thinkingText, assistantText);
      const answer = stripThink(reply) || reply;
      const meta = {
        final, finishReason, chunks,
        total: (performance.now() - t0) / 1000,
        firstToken: firstChunkAt == null ? null : (firstChunkAt - t0) / 1000,
      };
      history.push({ role: "assistant", content: answer, model: currentModel, meta });
      renderMeta(metaEl, { ...meta, answer });
      saveChat();
    } catch (err) {
      if (err.name === "AbortError") {
        metaEl.innerHTML = `<span class="meta-warn">Stopped</span>`;
      } else {
        contentEl.innerHTML = `<span style="color: var(--danger)">Error: ${escapeHtml(err.message)}</span>`;
        markAttachmentFailed();
        console.error(err);
      }
    } finally {
      clearInterval(ticker);
      msg.classList.remove("streaming");
      updateJumpButton();
      setSending(false);
      abortController = null;
    }
  }

  // Under an answer: what the engine counted, when it said (an older engine
  // or an interrupted stream leaves only what the browser saw), and a copy
  // button for the answer without its reasoning.
  function renderMeta(metaEl, { final, finishReason, chunks, total, firstToken, answer }) {
    const rate = (n, ms) => (n > 0 && ms > 0 ? (n / (ms / 1000)) : null);
    const items = [];
    if (final && final.outN != null) {
      const out = rate(final.outN, final.outMs);
      items.push(`<span title="Answer tokens and writing speed"><b>${out != null ? out.toFixed(1) : "–"}</b> tok/s</span>`);
      items.push(`<span>${final.outN.toLocaleString()} tokens</span>`);
      const read = rate(final.promptN, final.promptMs);
      if (final.promptN != null) {
        items.push(`<span title="Prompt tokens read for this answer, and how fast">prompt ${final.promptN.toLocaleString()}${read != null ? ` · ${Math.round(read).toLocaleString()} tok/s` : ""}</span>`);
      }
      if (final.drafts > 0) {
        items.push(`<span title="MTP drafts the model kept">${Math.round((final.kept / final.drafts) * 100)}% drafts kept</span>`);
      }
    } else {
      items.push(`<span>${chunks.toLocaleString()} tokens</span>`);
    }
    if (firstToken != null) items.push(`<span title="Time to the first piece of the answer">first token ${firstToken.toFixed(2)} s</span>`);
    items.push(`<span title="From sending to the last token">${total.toFixed(1)} s</span>`);
    if (finishReason === "length") {
      items.push(`<span class="meta-warn" title="Raise the longest answer or the context size">cut at the length limit</span>`);
    }
    metaEl.innerHTML = `${items.join("")}<button type="button" class="meta-copy" title="Copy the answer">Copy</button>`;
    const btn = metaEl.querySelector(".meta-copy");
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(answer);
        btn.textContent = "Copied";
      } catch {
        btn.textContent = "Copy failed";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1500);
    });
  }

  function canSend() {
    return !!(els.input.value.trim() || pendingMedia);
  }

  function setSending(busy) {
    els.sendBtn.disabled = busy || !canSend();
    els.input.disabled = busy;
    els.attachBtn.disabled = busy;
    els.stopBtn.hidden = !busy;
    els.sendBtn.hidden = busy;
    if (!busy) els.input.focus();
  }

  // ── Input UX ──────────────────────────────────────────────────────────
  function autoresize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
    els.sendBtn.disabled = !canSend();
  }

  els.input.addEventListener("input", autoresize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });

  // ── Attachment handling ───────────────────────────────────────────────
  // The 📎 button opens the hidden file picker; the chosen image OR audio
  // file is read as a data URL, the base64 payload is stripped off the
  // `data:...;base64,` prefix for the backend, and a preview bar appears
  // above the textarea (thumbnail for images, a mini player for audio)
  // until the user sends or removes it.
  function clearAttachment() {
    pendingMedia = null;
    els.imageInput.value = "";
    els.attachmentThumb.removeAttribute("src");
    els.attachmentThumb.hidden = true;
    els.attachmentAudio.removeAttribute("src");
    els.attachmentAudio.hidden = true;
    els.attachmentName.textContent = "";
    els.attachmentBar.hidden = true;
    els.sendBtn.disabled = !canSend();
  }

  // llama.cpp's mtmd decodes images via stb_image, which only handles
  // jpg / png / bmp / gif. WebP, AVIF and HEIC reach the server as
  // "failed to decode image bytes" and the model then hallucinates. We
  // transparently re-encode anything outside this set to PNG in-browser.
  const MTMD_SAFE_IMAGE = new Set(["image/jpeg", "image/png", "image/bmp", "image/gif"]);

  // mtmd decodes audio through miniaudio, which handles wav / mp3 / flac and
  // nothing else. A WhatsApp voice note is Ogg/Opus, so it reached the server
  // as undecodable bytes. Browsers can decode Opus, so the same trick used for
  // images applies: re-encode locally to something the engine reads.
  const MTMD_SAFE_AUDIO = new Set([
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
  ]);

  // 16 kHz mono is what speech encoders resample to anyway, and it keeps an
  // uncompressed WAV small enough to base64 into a JSON body: one minute is
  // about 1.9 MB rather than the 10 MB a 44.1 kHz stereo copy would cost.
  const WAV_RATE = 16000;

  /// Decode any audio the browser understands and re-encode it as 16-bit PCM
  /// mono WAV. Rejects when the browser itself cannot decode the source.
  const convertAudioToWav = async (arrayBuffer) => {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) throw new Error("no audio support");
    const decoded = await new Ctx().decodeAudioData(arrayBuffer);
    const frames = Math.ceil(decoded.duration * WAV_RATE);
    const off = new OfflineAudioContext(1, frames, WAV_RATE);
    const src = off.createBufferSource();
    src.buffer = decoded;
    src.connect(off.destination);
    src.start();
    const mono = (await off.startRendering()).getChannelData(0);

    const bytes = new ArrayBuffer(44 + mono.length * 2);
    const view = new DataView(bytes);
    const ascii = (at, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i));
    };
    ascii(0, "RIFF");
    view.setUint32(4, 36 + mono.length * 2, true);
    ascii(8, "WAVEfmt ");
    view.setUint32(16, 16, true); // PCM header size
    view.setUint16(20, 1, true); // format: PCM
    view.setUint16(22, 1, true); // channels
    view.setUint32(24, WAV_RATE, true);
    view.setUint32(28, WAV_RATE * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    ascii(36, "data");
    view.setUint32(40, mono.length * 2, true);
    for (let i = 0; i < mono.length; i++) {
      const s = Math.max(-1, Math.min(1, mono[i]));
      view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    let binary = "";
    const raw = new Uint8Array(bytes);
    for (let i = 0; i < raw.length; i += 0x8000) {
      binary += String.fromCharCode.apply(null, raw.subarray(i, i + 0x8000));
    }
    return `data:audio/wav;base64,${btoa(binary)}`;
  };

  const fileToArrayBuffer = (file) =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = () => reject(new Error("read failed"));
      r.readAsArrayBuffer(file);
    });

  const fileToDataUrl = (file) =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = () => reject(new Error("read failed"));
      r.readAsDataURL(file);
    });

  // Re-encode an unsupported image to PNG via a canvas. Resolves to a PNG
  // data URL. Rejects if the browser itself can't decode the source (HEIC
  // on most desktops) — caught below with an actionable message.
  const convertImageToPng = (dataUrl) =>
    new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext("2d").drawImage(img, 0, 0);
        resolve(canvas.toDataURL("image/png"));
      };
      img.onerror = () => reject(new Error("decode failed"));
      img.src = dataUrl;
    });

  els.attachBtn.addEventListener("click", () => els.imageInput.click());
  els.imageInput.addEventListener("change", async () => {
    const file = els.imageInput.files?.[0];
    if (!file) return;
    const kind = file.type.startsWith("image/")
      ? "image"
      : file.type.startsWith("audio/")
        ? "audio"
        : null;
    if (!kind) {
      alert("Only image and audio files are supported.");
      els.imageInput.value = "";
      return;
    }
    try {
      let dataUrl = await fileToDataUrl(file);
      if (kind === "image" && !MTMD_SAFE_IMAGE.has(file.type)) {
        // WebP / AVIF / HEIC … → normalise to PNG so mtmd can decode it.
        dataUrl = await convertImageToPng(dataUrl);
      } else if (kind === "audio" && !MTMD_SAFE_AUDIO.has(file.type)) {
        // Ogg/Opus (WhatsApp), M4A, WebM … → 16 kHz mono WAV.
        dataUrl = await convertAudioToWav(await fileToArrayBuffer(file));
      }
      const comma = dataUrl.indexOf(",");
      const base64 = comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
      pendingMedia = { dataUrl, base64, kind, name: file.name };
      if (kind === "image") {
        els.attachmentThumb.src = dataUrl;
        els.attachmentThumb.hidden = false;
        els.attachmentAudio.hidden = true;
      } else {
        els.attachmentAudio.src = dataUrl;
        els.attachmentAudio.hidden = false;
        els.attachmentThumb.hidden = true;
      }
      els.attachmentName.textContent =
        kind === "audio" ? `🎵 ${file.name}` : file.name;
      els.attachmentBar.hidden = false;
      els.sendBtn.disabled = !canSend();
    } catch {
      alert(
        kind === "audio"
          ? "Could not read this audio. The engine accepts wav, mp3 and flac; " +
            "anything else is converted here first, and this browser could not " +
            "decode it. Convert it yourself, e.g. ffmpeg -i in.ogg out.wav"
          : "Could not read this file. If it's a HEIC/HEIF photo, convert it to " +
            "JPG or PNG first — the engine accepts jpg/png/bmp/gif/webp.",
      );
      els.imageInput.value = "";
    }
  });
  els.attachmentRemove.addEventListener("click", clearAttachment);

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const txt = els.input.value.trim();
    // An image-only turn is allowed (the model falls back to its default
    // "describe this image" behaviour); a fully empty submit is not.
    if (!txt && !pendingMedia) return;
    els.input.value = "";
    autoresize();
    send(txt);
  });

  els.stopBtn.addEventListener("click", () => abortController?.abort());

  els.clearBtn.addEventListener("click", () => {
    if (!history.length) return;
    if (!confirm("Delete this chat? It is removed from this browser.")) return;
    deleteChat(currentChatId);
  });

  els.modelSelect.addEventListener("change", (e) => {
    const previous = currentModel;
    currentModel = e.target.value;
    if (!previous || previous === currentModel || !history.length) return;

    // A mid-conversation model switch keeps the history — but the new model
    // would read the previous model's turns as its own words and stay in
    // character (observed live: gemma-4 introduced itself as Qwen "to be
    // consistent with my previous answer"). Record the switch ONCE, as a
    // system turn *at this point in the history*: the new model sees a past
    // event it can act on, nothing is repeated on later prompts, and the
    // history before the switch stays byte-identical for prefix KV reuse.
    // Deliberately a narrow exception to the no-automatic-injections rule
    // (see the `system` setting's comment): it exists only at a switch
    // point, states facts about turn authorship, and carries no style or
    // formatting instructions.
    //
    // Flipping the dropdown without sending anything coalesces: the pending
    // note is updated in place, and removed entirely if the user returns to
    // the model the conversation was already on.
    const last = history[history.length - 1];
    const pendingSwitch = last?.switchFrom ? history.pop() : null;
    const from = pendingSwitch ? pendingSwitch.switchFrom : previous;
    const lastNote = els.messages.querySelector(".model-switch-note:last-child");
    if (pendingSwitch && lastNote) lastNote.remove();
    if (from === currentModel) return; // switched back — nothing changed

    history.push({
      role: "system",
      content:
        `The assistant model changed at this point in the conversation: the replies above ` +
        `were written by ${from}, and from here on the assistant is ${currentModel}. ` +
        `${currentModel}, answer as yourself — your own identity, knowledge and style. Do ` +
        `not claim the previous model's identity, and feel free to differ from its answers.`,
      switchFrom: from,
    });

    const note = document.createElement("div");
    note.className = "model-switch-note";
    note.textContent = `— model changed: ${from} → ${currentModel} —`;
    els.messages.appendChild(note);
    els.messages.scrollTop = els.messages.scrollHeight;
    saveChat();
  });

  // ── Parameters panel ─────────────────────────────────────────────────
  // Every change applies to the next message and is kept in this browser.
  const sliders = [
    { id: "settings-temp", key: "temperature", digits: 2 },
    { id: "settings-top-p", key: "topP", digits: 2 },
    { id: "settings-top-k", key: "topK", digits: 0 },
    { id: "settings-min-p", key: "minP", digits: 2 },
    { id: "settings-repeat", key: "repeatPenalty", digits: 2 },
  ].map((s) => ({ ...s, input: $(s.id), out: $(`${s.id}-val`) }));

  function showSettings() {
    for (const s of sliders) {
      s.input.value = settings[s.key];
      s.out.textContent = Number(settings[s.key]).toFixed(s.digits);
      s.input.closest(".slider")?.classList.toggle("changed", settings[s.key] !== DEFAULT_SETTINGS[s.key]);
    }
    els.settingsSystem.value = settings.system;
    els.settingsMaxTokens.value = settings.maxTokens;
    els.settingsSeed.value = settings.seed ?? "";
    els.settingsThink.checked = settings.think;
    els.settingsMath.checked = settings.math;
  }

  for (const s of sliders) {
    s.input.addEventListener("input", () => {
      settings[s.key] = s.digits ? parseFloat(s.input.value) : parseInt(s.input.value, 10);
      s.out.textContent = Number(settings[s.key]).toFixed(s.digits);
      s.input.closest(".slider")?.classList.toggle("changed", settings[s.key] !== DEFAULT_SETTINGS[s.key]);
      saveSettings();
    });
  }
  els.settingsSystem.addEventListener("input", () => { settings.system = els.settingsSystem.value.trim(); saveSettings(); });
  els.settingsMaxTokens.addEventListener("input", () => { settings.maxTokens = Math.max(0, parseInt(els.settingsMaxTokens.value, 10) || 0); saveSettings(); });
  els.settingsSeed.addEventListener("input", () => {
    const v = parseInt(els.settingsSeed.value, 10);
    settings.seed = Number.isInteger(v) && v >= 0 ? v : null;
    saveSettings();
  });
  els.settingsThink.addEventListener("change", () => { settings.think = els.settingsThink.checked; saveSettings(); });
  els.settingsMath.addEventListener("change", () => {
    settings.math = els.settingsMath.checked;
    saveSettings();
    // Re-render the last assistant turn so the math toggle takes effect immediately.
    const lastAssistant = els.messages.querySelector(".msg.assistant:last-child .msg-content");
    const lastTurn = history[history.length - 1];
    if (lastAssistant && lastTurn?.role === "assistant") lastAssistant.innerHTML = renderContent(lastTurn.content);
  });
  els.paramsReset.addEventListener("click", () => {
    Object.assign(settings, DEFAULT_SETTINGS);
    saveSettings();
    showSettings();
  });

  function toggleParams(open) {
    // The drawer starts under the top bar, whose height depends on the width.
    const bar = document.querySelector(".topbar");
    if (bar) document.documentElement.style.setProperty("--topbar-h", `${bar.offsetHeight}px`);
    els.params.hidden = !open;
    els.settingsBtn.setAttribute("aria-expanded", String(open));
    els.settingsBtn.classList.toggle("active", open);
    document.body.classList.toggle("params-open", open);
    if (open) { showSettings(); els.paramsClose.focus({ preventScroll: true }); }
  }
  els.settingsBtn.addEventListener("click", () => toggleParams(els.params.hidden));
  els.paramsClose.addEventListener("click", () => { toggleParams(false); els.settingsBtn.focus(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.params.hidden && !document.querySelector("dialog[open]")) {
      toggleParams(false);
      els.settingsBtn.focus();
    }
  });

  // ── Pages ────────────────────────────────────────────────────────────
  // One page at a time beside the sidebar: the chat, the Model hub, the
  // Monitor or About. Monitor and About listen for `eullm:view` to start and
  // stop their own reading.
  const VIEWS = {
    chat: { title: "Chat", nodes: () => [els.messages, els.form] },
    hub: { title: "Model hub", nodes: () => [$("hub")] },
    monitor: { title: "Monitor", nodes: () => [$("monitor")] },
    about: { title: "About", nodes: () => [$("about")] },
  };
  let currentView = "chat";

  function showView(name) {
    if (!VIEWS[name]) name = "chat";
    currentView = name;
    for (const [key, view] of Object.entries(VIEWS)) {
      for (const node of view.nodes()) if (node) node.hidden = key !== name;
    }
    document.querySelectorAll(".side-item").forEach((b) => {
      const on = b.dataset.view === name;
      b.classList.toggle("active", on);
      if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
    });
    els.viewTitle.textContent = VIEWS[name].title;
    document.dispatchEvent(new CustomEvent("eullm:view", { detail: name }));
    if (name === "hub") openHub();
    if (name === "chat") els.input.focus({ preventScroll: true });
    toggleSidebar(false);
  }

  document.querySelectorAll(".side-item").forEach((b) =>
    b.addEventListener("click", () => showView(b.dataset.view)));

  // On a narrow screen the sidebar slides in over the page.
  function toggleSidebar(open) {
    document.body.classList.toggle("side-open", open);
    els.scrim.hidden = !open;
    els.menuBtn.setAttribute("aria-expanded", String(open));
  }
  els.menuBtn.addEventListener("click", () => toggleSidebar(!document.body.classList.contains("side-open")));
  els.scrim.addEventListener("click", () => toggleSidebar(false));

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || document.querySelector("dialog[open]") || !els.params.hidden) return;
    if (document.body.classList.contains("side-open")) toggleSidebar(false);
    else if (currentView !== "chat") showView("chat");
  });

  // ── Chats kept in this browser ───────────────────────────────────────
  // The only thing besides the Parameters that the page keeps: each chat's
  // turns in localStorage, on this browser alone, never sent anywhere. An
  // attachment is not kept (it would fill the storage): a chat reopened
  // tells the model the attachment is gone, as a long chat already does.
  const CHATS_STORAGE = "eullm.chats";
  const MAX_CHATS = 50;
  let chats = [];
  let currentChatId = null;
  try {
    const saved = JSON.parse(localStorage.getItem(CHATS_STORAGE) || "[]");
    if (Array.isArray(saved)) chats = saved.filter((c) => c && c.id && Array.isArray(c.turns));
  } catch {}

  function storeChats() {
    // A full storage drops the oldest chats until the rest fit.
    for (;;) {
      try {
        localStorage.setItem(CHATS_STORAGE, JSON.stringify(chats));
        return;
      } catch {
        if (chats.length <= 1) return;
        chats.pop();
      }
    }
  }

  function chatTitle(turns) {
    const first = turns.find((t) => t.role === "user" && t.content.trim());
    if (!first) return "New chat";
    const line = first.content.trim().split("\n")[0];
    return line.length > 60 ? `${line.slice(0, 57)}…` : line;
  }

  function saveChat() {
    if (!history.length) return;
    if (!currentChatId) currentChatId = `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
    const turns = history.map((t) => {
      const kept = { role: t.role, content: t.content };
      if (t.model) kept.model = t.model;
      if (t.meta) kept.meta = t.meta;
      if (t.switchFrom) kept.switchFrom = t.switchFrom;
      if (t.media) kept.media = { kind: t.media.kind, failed: true };
      return kept;
    });
    const chat = { id: currentChatId, title: chatTitle(turns), model: currentModel, updated: Date.now(), turns };
    chats = [chat, ...chats.filter((c) => c.id !== currentChatId)].slice(0, MAX_CHATS);
    storeChats();
    renderChatList();
  }

  function renderChatList() {
    els.chatList.innerHTML = "";
    if (!chats.length) {
      const li = document.createElement("li");
      li.className = "chat-empty";
      li.textContent = "No chats yet";
      els.chatList.appendChild(li);
      return;
    }
    for (const c of chats) {
      const li = document.createElement("li");
      li.className = "chat-item" + (c.id === currentChatId ? " active" : "");
      const open = document.createElement("button");
      open.type = "button";
      open.className = "chat-open";
      open.textContent = c.title;
      open.title = `${c.title}\n${c.model || ""} · ${new Date(c.updated).toLocaleString()}`;
      if (c.id === currentChatId) open.setAttribute("aria-current", "true");
      open.addEventListener("click", () => openChat(c.id));
      const del = document.createElement("button");
      del.type = "button";
      del.className = "chat-del";
      del.setAttribute("aria-label", `Delete chat: ${c.title}`);
      del.title = "Delete this chat";
      del.textContent = "×";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        if (confirm(`Delete "${c.title}"? It is removed from this browser.`)) deleteChat(c.id);
      });
      li.append(open, del);
      els.chatList.appendChild(li);
    }
  }

  function resetChat() {
    abortController?.abort();
    history.length = 0;
    currentChatId = null;
    clearAttachment();
    els.messages.innerHTML = WELCOME_HTML;
    updateStatus(loadedModelEntry());
  }

  function newChat() {
    resetChat();
    renderChatList();
    showView("chat");
  }

  function deleteChat(id) {
    chats = chats.filter((c) => c.id !== id);
    storeChats();
    if (id === currentChatId) resetChat();
    renderChatList();
  }

  function openChat(id) {
    const chat = chats.find((c) => c.id === id);
    if (!chat) return;
    resetChat();
    currentChatId = id;
    dismissWelcome();
    for (const t of chat.turns) {
      const turn = { ...t };
      history.push(turn);
      if (t.switchFrom) {
        const note = document.createElement("div");
        note.className = "model-switch-note";
        const to = t.content.match(/from here on the assistant is (.+?)\. /)?.[1] || "another model";
        note.textContent = `— model changed: ${t.switchFrom} → ${to} —`;
        els.messages.appendChild(note);
        continue;
      }
      if (t.role === "system") continue;
      const { metaEl } = appendMessage(t.role, t.content, t.media ? { kind: "missing", label: t.media.kind } : null);
      if (t.role === "assistant" && t.meta) renderMeta(metaEl, { ...t.meta, answer: t.content });
      else if (t.role === "assistant" && t.model) metaEl.innerHTML = `<span>${escapeHtml(t.model)}</span>`;
    }
    // The chat goes on with the model it was on, when that model is offered.
    if (chat.model && [...els.modelSelect.options].some((o) => o.value === chat.model && !o.disabled)) {
      currentModel = chat.model;
      els.modelSelect.value = chat.model;
    }
    renderChatList();
    showView("chat");
    scrollToBottom(true);
  }

  els.newChat.addEventListener("click", newChat);

  // Welcome-screen suggestion buttons
  document.addEventListener("click", (e) => {
    if (e.target.classList?.contains("suggest")) {
      els.input.value = e.target.textContent;
      autoresize();
      els.input.focus();
    }
  });

  // ── Model catalog ─────────────────────────────────────────────────────
  //
  // Everything here talks to the engine, never to huggingface.co: the browser
  // making the Hub call directly would hand the viewer's address to the Hub,
  // put a second host outside EULLM_WEB_ALLOWED_DOMAINS, and stop working on
  // the machine that needs this most — an HPC login node, where the engine
  // has a route out and the browser does not.

  const catalogEls = {
    body: $("hub-body"),
    sort: $("catalog-sort"),
    search: $("catalog-search"),
    results: $("catalog-results"),
    detail: $("catalog-detail"),
    direct: $("catalog-direct"),
    directInput: $("catalog-direct-input"),
  };

  const humanBytes = (n) => {
    if (!n) return "—";
    const gib = n / 1024 ** 3;
    return gib >= 1 ? `${gib.toFixed(1)} GB` : `${Math.round(n / 1024 ** 2)} MB`;
  };
  const humanCount = (n) =>
    n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);

  // What each verdict means, in words. A colour alone is not readable to
  // everyone, and "too large" deserves saying rather than implying.
  const VERDICT_TEXT = {
    fits: ["Fits", "Expected to run fully on this machine's GPU, or comfortably in RAM."],
    tight: ["Tight", "Expected to run with layers on the CPU, or filling most of RAM. Slower."],
    too_large: ["Too large", "Larger than this machine's VRAM and RAM together."],
    unknown: ["Unknown", "Neither VRAM nor RAM could be read here, so there is no honest answer."],
  };

  function catalogError(where, msg) {
    where.innerHTML = "";
    const p = document.createElement("p");
    p.className = "catalog-error";
    p.textContent = msg;
    where.appendChild(p);
  }

  // `hf.co/owner/repo` typed straight in, for a repo the search does not
  // surface. It only jumps to the detail view a search result would have
  // opened, so every check there — sizes, traffic lights, architecture —
  // applies unchanged, and nothing new crosses the perimeter: the engine
  // still talks to the Hub and to nothing else.
  function repoIdFromInput(raw) {
    const t = raw
      .trim()
      .replace(/^https?:\/\//i, "")
      .replace(/^(hf\.co|huggingface\.co)\//i, "");
    const m = t.match(/^([A-Za-z0-9][\w.-]*)\/([\w.-]+?)\/?$/);
    return m ? `${m[1]}/${m[2]}` : null;
  }

  // A search, or with nothing typed the GGUF repos in the chosen order.
  let searchSeq = 0;
  async function searchCatalog(q) {
    const seq = ++searchSeq;
    const sort = catalogEls.sort.value;
    catalogEls.results.innerHTML = `<p class="hub-empty">${q.trim() ? "Searching…" : "Reading Hugging Face…"}</p>`;
    try {
      const r = await fetch(
        `/api/hf/search?q=${encodeURIComponent(q.trim())}&sort=${encodeURIComponent(sort)}&limit=40`,
        withAuth(),
      );
      const data = await r.json();
      if (seq !== searchSeq) return; // a newer search is on its way
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      catalogEls.results.innerHTML = "";
      if (!data.models.length) {
        catalogEls.results.innerHTML = `<p class="hub-empty">Nothing found.</p>`;
        return;
      }
      for (const m of data.models) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "hub-row";
        row.dataset.id = m.id;
        const [owner, repo] = m.id.split("/");
        const age = m.updated ? relativeDate(m.updated) : "";
        row.innerHTML =
          `<span class="hub-avatar" aria-hidden="true">${escapeHtml((owner || "?")[0].toUpperCase())}</span>` +
          `<span class="hub-row-main"><span class="hub-row-name">${escapeHtml(repo || m.id)}</span>` +
          `<span class="hub-row-sub">${escapeHtml(owner || "")}${m.gated ? " · gated" : ""}</span></span>` +
          `<span class="hub-row-meta"><span title="Downloads">↓ ${humanCount(m.downloads)}</span>` +
          `<span title="Likes">♥ ${humanCount(m.likes)}</span>${age ? `<span>${escapeHtml(age)}</span>` : ""}</span>`;
        row.addEventListener("click", () => showRepo(m.id));
        catalogEls.results.appendChild(row);
      }
    } catch (err) {
      if (seq === searchSeq) catalogError(catalogEls.results, `Search failed: ${err.message}`);
    }
  }

  function relativeDate(iso) {
    const d = (Date.now() - new Date(iso).getTime()) / 86400000;
    if (!(d >= 0)) return "";
    if (d < 1) return "today";
    if (d < 30) return `${Math.floor(d)} d ago`;
    if (d < 365) return `${Math.floor(d / 30)} mo ago`;
    return `${Math.floor(d / 365)} y ago`;
  }

  // The detail column beside the list; on a narrow screen it replaces it.
  function showDetail(which) {
    $("device-detail").hidden = which !== "device";
    catalogEls.detail.hidden = which !== "catalog";
    catalogEls.body.classList.toggle("detail-open", !!which);
  }
  function markSelected(list, attr, value) {
    list.querySelectorAll(".hub-row").forEach((r) => r.classList.toggle("selected", r.dataset[attr] === value));
  }

  async function showRepo(id) {
    showDetail("catalog");
    markSelected(catalogEls.results, "id", id);
    catalogEls.detail.textContent = `Reading ${id}…`;
    let data;
    try {
      const r = await fetch(`/api/hf/repo?id=${encodeURIComponent(id)}`, withAuth());
      data = await r.json();
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    } catch (err) {
      catalogError(catalogEls.detail, `Could not read ${id}: ${err.message}`);
      return;
    }

    catalogEls.detail.innerHTML = "";
    // The head does not scroll: the repo name and, above all, what the traffic
    // lights were judged against have to be visible without hunting for them.
    // They were inside the scrolling list, and a repo with twenty
    // quantizations opened showing the largest ones and none of the context —
    // which reads as the colours having no basis at all.
    const head = document.createElement("div");
    head.className = "catalog-detail-head";
    const list = document.createElement("div");
    list.className = "quant-list";
    const back = document.createElement("button");
    back.type = "button";
    back.className = "catalog-back";
    back.textContent = "← Back to the list";
    back.addEventListener("click", () => showDetail(null));
    const title = document.createElement("h3");
    title.textContent = id;
    head.append(back, title);

    // Say what the traffic light was judged against, rather than showing a
    // colour with no stated basis.
    const basis = document.createElement("p");
    basis.className = "catalog-note";
    const parts = [];
    if (data.vram_total_bytes) {
      parts.push(`${humanBytes(data.vram_free_bytes)} of ${humanBytes(data.vram_total_bytes)} VRAM free`);
    } else {
      parts.push("no GPU detected");
    }
    if (data.ram_total_bytes) parts.push(`${humanBytes(data.ram_total_bytes)} RAM`);
    basis.textContent = `Judged against: ${parts.join(", ")}. An estimate from the download size — the exact layer split is computed after the model is on disk.`;
    head.appendChild(basis);

    // What the file says it is, and whether this binary has that architecture
    // compiled in. Three states, and the third is a question rather than a
    // verdict: the Hub does not always parse an architecture out of a repo,
    // and "it did not say" must never be shown as "this will not load".
    const arch = document.createElement("p");
    arch.className = "catalog-note";
    if (data.architecture && data.architecture_supported === false) {
      arch.classList.add("catalog-warn");
      arch.textContent =
        `Architecture ${data.architecture}: this build cannot load it. Downloading would ` +
        `spend the transfer for nothing — it needs a newer llama.cpp than this binary was built with.`;
    } else if (data.architecture) {
      arch.textContent = `Architecture ${data.architecture}: this build can load it.`;
    } else {
      arch.textContent =
        "Architecture: the Hub did not report one for this repo, so whether this build can " +
        "load it is not known until the file is on disk.";
    }
    head.appendChild(arch);

    // Stated, not interpreted. `apache-2.0` is a name the reader knows or can
    // look up; `other` is the Hub saying these weights carry terms of their
    // own. What those terms permit is between the licence and whoever runs the
    // model — our part is to put it here, before the download rather than
    // after, and to link the text instead of summarising it.
    const lic = document.createElement("p");
    lic.className = "catalog-note";
    // `other` is not a name, so it never reaches the page as one: it means the
    // repo named its own terms in `license_name`, or did not name them at all.
    const named = data.license === "other" ? data.license_name : data.license;
    let stated;
    if (data.license === "other") {
      stated = named
        ? `Licence ${named}: terms of its own rather than a standard licence. `
        : "Licence: terms of its own rather than a standard licence. ";
    } else if (named) {
      stated = `Licence ${named}. `;
    } else {
      stated = "Licence: the Hub did not state one. ";
    }
    lic.append(document.createTextNode(stated));
    if (data.license_url) {
      const a = document.createElement("a");
      a.href = data.license_url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = "Read the terms";
      lic.appendChild(a);
    }
    head.appendChild(lic);

    if (data.mmproj) {
      const mm = document.createElement("p");
      mm.className = "catalog-note";
      mm.textContent = `Ships a multimodal projector (${humanBytes(data.mmproj_bytes)}); it is downloaded with whichever quantization you pick.`;
      head.appendChild(mm);
    }

    for (const q of data.quants) {
      list.appendChild(quantRow(q));
    }
    catalogEls.detail.append(head, list);
    // Smallest first is the useful end of the list, so start there rather than
    // wherever the previous view happened to leave the scroll position.
    list.scrollTop = 0;
  }

  function quantRow(q) {
    const row = document.createElement("div");
    row.className = "quant-row";

    const [text, why] = VERDICT_TEXT[q.verdict] || VERDICT_TEXT.unknown;
    const badge = document.createElement("span");
    badge.className = `verdict ${q.verdict}`;
    badge.textContent = text;
    badge.title = why;

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = q.label + (q.shards > 1 ? ` · ${q.shards} shards` : "");

    const size = document.createElement("span");
    size.className = "size";
    size.textContent = humanBytes(q.bytes);

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-secondary";
    btn.textContent = "Download";

    const state = document.createElement("span");
    state.className = "pull-state";
    state.hidden = true;
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = 0;
    bar.hidden = true;

    btn.addEventListener("click", () => pullModel(q, btn, state, bar));
    row.append(badge, label, size, btn, bar, state);
    return row;
  }

  // Reads the NDJSON the engine streams from /api/pull — one JSON object per
  // line, the same shape Ollama emits, so the parsing is a split on newlines
  // and nothing more. A line can arrive in pieces, hence the buffer.
  async function pullModel(q, btn, state, bar) {
    btn.disabled = true;
    state.hidden = false;
    state.textContent = "starting…";
    try {
      const r = await fetch(
        "/api/pull",
        withAuth({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: q.pull }),
        }),
      );
      if (!r.ok || !r.body) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${r.status}`);
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let done = false;
      for (;;) {
        const { value, done: finished } = await reader.read();
        if (finished) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const ev = JSON.parse(line);
          if (ev.error) throw new Error(ev.error);
          if (ev.total) {
            bar.hidden = false;
            bar.value = ev.completed / ev.total;
            state.textContent = `${humanBytes(ev.completed)} of ${humanBytes(ev.total)}`;
          } else if (ev.status === "success") {
            done = true;
          } else if (ev.status) {
            state.textContent = ev.status;
          }
        }
      }
      bar.hidden = true;
      if (done) {
        state.textContent = "Downloaded — now in the model list.";
        btn.textContent = "Done";
        // The picker is built from /api/tags, so it has to be rebuilt for the
        // new model to be selectable without a page reload.
        await loadModels();
      } else {
        state.textContent = "The download ended without confirming. Try again.";
        btn.disabled = false;
      }
    } catch (err) {
      bar.hidden = true;
      state.textContent = `Failed: ${err.message}`;
      btn.disabled = false;
    }
  }

  let searchTimer = null;
  catalogEls.search.addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    const q = e.target.value;
    // Debounced: every keystroke is a request the engine forwards to the Hub.
    searchTimer = setTimeout(() => searchCatalog(q), 350);
  });
  catalogEls.sort.addEventListener("change", () => searchCatalog(catalogEls.search.value));
  catalogEls.direct.addEventListener("submit", (e) => {
    e.preventDefault();
    const id = repoIdFromInput(catalogEls.directInput.value);
    if (!id) {
      catalogError(
        catalogEls.results,
        "Expected owner/repo, optionally prefixed with hf.co/ — for example hf.co/unsloth/Qwen3-8B-GGUF.",
      );
      return;
    }
    showRepo(id);
  });

  // ── Model hub: the models on this machine ────────────────────────────
  const hubEls = {
    tabDevice: $("tab-device"),
    tabDiscover: $("tab-discover"),
    device: $("hub-device"),
    discover: $("hub-discover"),
    filter: $("hub-filter"),
    detail: $("device-detail"),
    stats: $("hub-stats"),
  };
  let hubTab = "device";
  let discoverLoaded = false;
  let selectedModel = null;

  function setHubTab(tab) {
    hubTab = tab;
    const device = tab === "device";
    hubEls.tabDevice.setAttribute("aria-selected", String(device));
    hubEls.tabDiscover.setAttribute("aria-selected", String(!device));
    hubEls.device.hidden = !device;
    hubEls.discover.hidden = device;
    hubEls.filter.hidden = !device;
    catalogEls.search.hidden = device;
    catalogEls.sort.hidden = device;
    showDetail(null);
    if (device) {
      if (selectedModel) showDeviceModel(selectedModel);
      else hubEls.detail.innerHTML = `<p class="hub-empty">Pick a model to see what it is and what you can do with it.</p>`;
      $("device-detail").hidden = false;
      catalogEls.body.classList.remove("detail-open");
    } else {
      catalogEls.detail.hidden = false;
      if (!catalogEls.detail.childElementCount) {
        catalogEls.detail.innerHTML = `<p class="hub-empty">Pick a repo to see its quantizations, how large each is and whether it fits this machine.</p>`;
      }
      if (!discoverLoaded) {
        discoverLoaded = true;
        searchCatalog("");
      }
      catalogEls.search.focus({ preventScroll: true });
    }
  }
  hubEls.tabDevice.addEventListener("click", () => setHubTab("device"));
  hubEls.tabDiscover.addEventListener("click", () => setHubTab("discover"));
  hubEls.filter.addEventListener("input", renderDevice);
  document.addEventListener("eullm:models", () => {
    renderDevice();
    if (selectedModel && hubTab === "device") showDeviceModel(selectedModel);
  });

  function openHub() {
    setHubTab(hubTab);
    renderDevice();
    loadHubStats();
    loadModels();
  }

  // What the downloads are judged against: this machine's memory.
  async function loadHubStats() {
    try {
      const r = await fetch("/api/about", withAuth());
      if (!r.ok) return;
      const a = await r.json();
      const m = a.machine || {};
      const gpus = (m.devices || []).filter((d) => d.kind !== "cpu" && d.memory_total);
      const vram = gpus.reduce((n, d) => n + d.memory_total, 0);
      const onDisk = modelsData.filter((x) => x.downloaded || x.loaded).length;
      const chip = (label, value, title = "") =>
        `<span class="hub-stat" title="${escapeHtml(title)}"><b>${escapeHtml(value)}</b> ${escapeHtml(label)}</span>`;
      hubEls.stats.innerHTML = [
        gpus.length ? chip("VRAM", humanBytes(vram), gpus.map((d) => d.description).join(", ")) : chip("GPU", "No", "This binary or this machine has no GPU llama.cpp can use"),
        chip("RAM", humanBytes(m.ram_total)),
        chip("CPU threads", String(m.cores || "–"), m.cpu || ""),
        chip("models here", String(onDisk)),
      ].join("");
    } catch {}
  }

  function renderDevice() {
    const f = hubEls.filter.value.trim().toLowerCase();
    const match = (m) => !f || `${m.name} ${m.details?.display_name || ""} ${m.details?.family || ""}`.toLowerCase().includes(f);
    const usable = modelsData.filter((m) => m.name !== "auto" || m.details?.family !== "eullm-router");
    const groups = [
      ["In memory", usable.filter((m) => m.loaded)],
      ["On this machine", usable.filter((m) => !m.loaded && m.downloaded && m.details?.source !== "library")],
      ["Used where they are (read only)", usable.filter((m) => !m.loaded && m.details?.source === "library")],
      ["EuLLM catalog, not downloaded", usable.filter((m) => !m.loaded && !m.downloaded)],
    ];
    hubEls.device.innerHTML = "";
    let shown = 0;
    for (const [label, items] of groups) {
      const visible = items.filter(match);
      if (!visible.length) continue;
      const h = document.createElement("h3");
      h.className = "hub-group";
      h.textContent = `${label} · ${visible.length}`;
      hubEls.device.appendChild(h);
      for (const m of visible) {
        shown++;
        const d = m.details || {};
        const row = document.createElement("button");
        row.type = "button";
        row.className = "hub-row" + (m.name === selectedModel ? " selected" : "");
        row.dataset.name = m.name;
        const sub = [d.parameter_size, d.quantization_level, m.size ? humanBytes(m.size) : ""].filter(Boolean).join(" · ");
        const badge = m.loaded
          ? `<span class="hub-badge live">In memory</span>`
          : d.source === "library" ? `<span class="hub-badge">Library</span>`
          : !m.downloaded ? `<span class="hub-badge dim">Not downloaded</span>` : "";
        row.innerHTML =
          `<span class="hub-avatar" aria-hidden="true">${escapeHtml((d.family || m.name)[0]?.toUpperCase() || "?")}</span>` +
          `<span class="hub-row-main"><span class="hub-row-name">${escapeHtml(d.display_name && d.display_name !== m.name ? d.display_name : m.name)}</span>` +
          `<span class="hub-row-sub">${escapeHtml(sub || m.name)}</span></span>` +
          `<span class="hub-row-meta">${badge}</span>`;
        row.addEventListener("click", () => showDeviceModel(m.name));
        hubEls.device.appendChild(row);
      }
    }
    if (!shown) {
      hubEls.device.innerHTML = f
        ? `<p class="hub-empty">No model matches “${escapeHtml(f)}”.</p>`
        : `<p class="hub-empty">No model on this machine yet. Find one in Discover.</p>`;
    }
  }

  function showDeviceModel(name) {
    const m = modelsData.find((x) => x.name === name);
    selectedModel = name;
    markSelected(hubEls.device, "name", name);
    if (!m) {
      hubEls.detail.innerHTML = `<p class="hub-empty">${escapeHtml(name)} is no longer here.</p>`;
      return;
    }
    showDetail("device");
    const d = m.details || {};
    const facts = [
      ["Name to use", `<code>${escapeHtml(m.name)}</code>`],
      d.family ? ["Family", escapeHtml(d.family)] : null,
      d.parameter_size ? ["Parameters", escapeHtml(d.parameter_size)] : null,
      d.quantization_level ? ["Quantization", escapeHtml(d.quantization_level)] : null,
      m.size ? ["Size", humanBytes(m.size)] : null,
      d.domain ? ["Domain", escapeHtml(d.domain)] : null,
      d.source_model ? ["Made from", escapeHtml(d.source_model)] : null,
      ["Where", m.loaded ? "in memory" : d.source === "library" ? "used where it is: a model library, or a file put in the models folder by hand" : m.downloaded ? "in the models folder" : "not downloaded"],
      m.digest ? ["Digest", `<code class="digest">${escapeHtml(m.digest)}</code>`] : null,
    ].filter(Boolean);
    hubEls.detail.innerHTML = `
      <button type="button" class="catalog-back">← Back to the list</button>
      <div class="hub-detail-head">
        <span class="hub-avatar big" aria-hidden="true">${escapeHtml((d.family || m.name)[0]?.toUpperCase() || "?")}</span>
        <div><h3>${escapeHtml(d.display_name || m.name)}</h3><p class="hub-row-sub">${escapeHtml(m.name)}</p></div>
      </div>
      <div class="hub-actions"></div>
      <p class="hub-action-state" role="status"></p>
      <dl class="kv">${facts.map(([k, v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
    hubEls.detail.querySelector(".catalog-back").addEventListener("click", () => showDetail(null));
    const actions = hubEls.detail.querySelector(".hub-actions");
    const state = hubEls.detail.querySelector(".hub-action-state");
    const button = (label, cls, fn) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = cls;
      b.textContent = label;
      b.addEventListener("click", () => fn(b));
      actions.appendChild(b);
      return b;
    };
    if (m.loaded || m.downloaded) {
      button("Chat with it", "btn-primary", () => {
        els.modelSelect.value = m.name;
        els.modelSelect.dispatchEvent(new Event("change"));
        showView("chat");
      });
    }
    if (m.loaded) {
      button("Unload", "btn-secondary", async (b) => {
        b.disabled = true;
        state.textContent = "Unloading…";
        try {
          const r = await fetch("/api/unload", withAuth({
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: m.name }),
          }));
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
          state.textContent = "Unloaded: its memory is free again.";
          await loadModels();
        } catch (e) {
          state.textContent = `Could not unload: ${e.message}`;
          b.disabled = false;
        }
      });
    }
    if (m.downloaded && !m.loaded && d.source !== "library") {
      // Two clicks: the first says what goes, the second deletes.
      button("Delete", "btn-danger", async (b) => {
        if (!b.dataset.armed) {
          b.dataset.armed = "1";
          b.textContent = `Delete ${m.size ? humanBytes(m.size) : "the files"}?`;
          state.textContent = "Click again to remove its files from the models folder.";
          return;
        }
        b.disabled = true;
        state.textContent = "Deleting…";
        try {
          const r = await fetch("/api/delete", withAuth({
            method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: m.name }),
          }));
          const body = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
          selectedModel = null;
          await loadModels();
          hubEls.detail.innerHTML = `<p class="hub-empty">${escapeHtml(m.name)} deleted${body.freed_bytes ? `: ${humanBytes(body.freed_bytes)} freed` : ""}.</p>`;
        } catch (e) {
          state.textContent = `Could not delete: ${e.message}`;
          b.disabled = false;
        }
      });
    }
    if (!m.downloaded && !m.loaded) {
      const bar = document.createElement("progress");
      bar.max = 1;
      bar.value = 0;
      bar.hidden = true;
      const b = button(`Download${m.size ? ` · ${humanBytes(m.size)}` : ""}`, "btn-primary", (btn) =>
        pullModel({ pull: m.name }, btn, state, bar));
      b.after(bar);
    }
    if (d.source === "library") {
      state.textContent = "EuLLM reads this file where it is and never moves or deletes it.";
    }
  }

  // ── Boot ──────────────────────────────────────────────────────────────
  async function init() {
    const v = await loadVersion();
    if (v?.version) window.__eullm_version = `v${v.version}`;
    if (v?.api_port) window.__eullm_api_port = v.api_port;
    if (v?.version && els.sideVersion) els.sideVersion.textContent = `EuLLM Engine ${v.version}`;
    await loadModels();
    renderChatList();
  }

  init();
})();
