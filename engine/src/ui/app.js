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
    attachBtn: $("attach-btn"),
    imageInput: $("image-input"),
    attachmentBar: $("attachment-bar"),
    attachmentThumb: $("attachment-thumb"),
    attachmentAudio: $("attachment-audio"),
    attachmentName: $("attachment-name"),
    attachmentRemove: $("attachment-remove"),
  };

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
    return compact.join("\n");
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
    const streaming = !!document.querySelector(".message.streaming");
    jumpBtn.hidden = stickToBottom || !streaming;
  }

  // ── API calls ─────────────────────────────────────────────────────────
  async function loadModels() {
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

    // History only stores the text — re-sending old media would blow up
    // the prompt and the multimodal MVP is one-shot anyway.
    history.push({ role: "user", content: userText });
    appendMessage("user", userText, media);

    const { msg, contentEl, metaEl } = appendMessage("assistant", "");
    msg.classList.add("streaming");
    updateJumpButton();
    setSending(true);

    abortController = new AbortController();
    const t0 = performance.now();
    let assistantText = "";
    let tokenCount = 0;

    try {
      let resp;
      if (media) {
        // Multimodal branch: hit /api/chat (Ollama NDJSON) with the
        // images:[base64] convention, NOT /v1/chat/completions. Image AND
        // audio both ride this field — the backend's mtmd path auto-detects
        // the media type from the bytes. History is intentionally omitted —
        // the mtmd MVP is a one-shot probe.
        const userMsg = { role: "user", content: userText, images: [media.base64] };
        const messagesToSend = settings.system
          ? [{ role: "system", content: settings.system }, userMsg]
          : [userMsg];
        resp = await fetch("/api/chat", withAuth({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortController.signal,
          body: JSON.stringify({
            model: currentModel,
            messages: messagesToSend,
            stream: true,
            temperature: settings.temperature,
            // 0 = unlimited: omit the cap and let the server generate until
            // the model stops or the context window fills.
            ...(settings.maxTokens > 0 ? { max_tokens: settings.maxTokens } : {}),
          }),
        }));
      } else {
        const messagesToSend = [];
        if (settings.system) messagesToSend.push({ role: "system", content: settings.system });
        // Send only role/content — history entries also carry UI-internal
        // fields (the authoring model, model-switch metadata).
        messagesToSend.push(...history.map(({ role, content }) => ({ role, content })));
        resp = await fetch("/v1/chat/completions", withAuth({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortController.signal,
          body: JSON.stringify({
            model: currentModel,
            messages: messagesToSend,
            stream: true,
            temperature: settings.temperature,
            // 0 = unlimited: omit the cap and let the server generate until
            // the model stops or the context window fills.
            ...(settings.maxTokens > 0 ? { max_tokens: settings.maxTokens } : {}),
            // Pass-through field for Ollama-compatible backends that honour it.
            think: settings.think,
          }),
        }));
      }

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
          //   * Ollama NDJSON (multimodal /api/chat): one JSON object per line,
          //     no `data:` prefix; delta = `message.content`.
          //   * OpenAI SSE (/v1/chat/completions): `data: {...}` lines + a
          //     trailing `[DONE]`; delta = `choices[0].delta.content`.
          let payload, delta;
          if (media) {
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
              return;
            }
            delta = media
              ? (obj.message?.content || "")
              : (obj.choices?.[0]?.delta?.content || "");
            if (delta) {
              assistantText += delta;
              tokenCount++;
              contentEl.innerHTML = renderContent(assistantText);
              scrollToBottom();
            }
          } catch (e) {
            console.warn("stream parse error:", e, payload);
          }
        }
      }

      // Tag the turn with the model that wrote it: after a mid-conversation
      // model switch, the send path uses this to tell the new model that the
      // earlier assistant turns are not its own words (see modelSwitchNotice).
      history.push({
        role: "assistant",
        content: stripThink(assistantText) || assistantText,
        model: currentModel,
      });
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
    if (!confirm("Clear conversation?")) return;
    history.length = 0;
    clearAttachment();
    els.messages.innerHTML = "";
    loadModels(); // re-render the welcome block via re-init
    init(true);
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
    settings.maxTokens = parseInt(els.settingsMaxTokens.value, 10) || 0;
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
    if (v?.api_port) window.__eullm_api_port = v.api_port;
    await loadModels();
    if (!skipWelcomeReinject && !els.messages.querySelector(".welcome") && history.length === 0) {
      location.reload(); // simplest way to restore welcome after clear
    }
  }

  init();
})();
