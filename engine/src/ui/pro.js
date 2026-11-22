// EuLLM Pro, as this engine's own page points to it: four entries in the
// administrator's menu, each saying what the Pro edition does there, and a
// line where this engine meets a real limit (an access key shared by
// everyone, requests nobody can tell apart). Nothing here pretends to work:
// the only action is a link to eullm.eu.
//
// Served as /eullm-ui/pro.js, and empty (so none of this is shown) when the
// server is started with EULLM_NO_PRO_HINTS=1, or when a program built on the
// engine adds a script of its own to the page (`chat_script`): its pages are
// then the ones in the menu.
//
// Who sees it: whoever administers this server. Without EULLM_ADMIN_KEY
// whoever opens the page may change the models, and so is shown the entries;
// on a managed server, only once this page holds the admin key (`eullm:admin`,
// fired by the Model hub and the Monitor when a request with it succeeds).
// The chat itself never shows any of it.

(() => {
  "use strict";
  if (!window.eullm) return;

  // Set by the engine when it serves this file: whether EULLM_ADMIN_KEY is set.
  const MANAGED = /*MANAGED*/false;

  // The languages eullm.eu has a Pro page in (/xx/pro, checked 22-11-2026),
  // English at /pro; the browser's first one of them, else English.
  const SITE_LANGS = ["it", "de", "fr", "es", "nl", "pt", "pl"];
  const proUrl = (() => {
    const langs = navigator.languages?.length ? navigator.languages : [navigator.language || "en"];
    for (const tag of langs) {
      const lang = String(tag).toLowerCase().split("-")[0];
      if (lang === "en") break;
      if (SITE_LANGS.includes(lang)) return `https://www.eullm.eu/${lang}/pro`;
    }
    return "https://www.eullm.eu/pro";
  })();

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const svg = (body) => `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

  // ── The administrator's menu ─────────────────────────────────────────
  // What the Pro edition does on each page, as its documentation says it,
  // what this engine does instead, and an example of the page with made-up
  // names. The example is a picture of the page, not the page: it is inert.
  const PAGES = [
    {
      name: "pro-people",
      title: "People and access",
      icon: svg(`<circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1"/>`),
      lead: "Each person logs in with an account of their own, and the server knows who asked which model, and when.",
      points: [
        "<strong>A login for each person</strong>: local accounts, with a password each person chooses from an invitation link, and locked for a while after repeated wrong attempts.",
        "<strong>LDAP and Active Directory</strong>: people log in with their work name and password, which the directory checks; the server never keeps it.",
        "<strong>Single sign-on with OpenID Connect</strong>: Microsoft Entra ID, Google Workspace, Keycloak, Authentik, Okta.",
        "<strong>Groups with the models they may use</strong>, filled by hand or from the directory's groups, and roles for administrators, auditors and users.",
        "<strong>Each person's chats on the server</strong>, the same from any computer, readable by that person only.",
      ],
      here: "This engine has one list of API keys shared by whoever is given one (<code>EULLM_API_KEYS</code>) and one admin key (<code>EULLM_ADMIN_KEY</code>): it knows which key was used, not which person.",
      example: `
        <table class="mon-table">
          <thead><tr><th class="wide">Person</th><th>Role</th><th class="wide">Groups</th><th>Signs in with</th><th>Last seen</th></tr></thead>
          <tbody>
            <tr><td class="wide">giulia.bianchi</td><td>admin</td><td class="wide">IT</td><td>Entra ID</td><td>today 09:12</td></tr>
            <tr><td class="wide">marco.rossi</td><td>user</td><td class="wide">Legal office</td><td>Active Directory</td><td>today 08:47</td></tr>
            <tr><td class="wide">laura.verdi</td><td>auditor</td><td class="wide">Compliance</td><td>Active Directory</td><td>yesterday</td></tr>
            <tr><td class="wide">paolo.neri</td><td>user</td><td class="wide">Sales, Support</td><td>password</td><td>3 days ago</td></tr>
          </tbody>
        </table>`,
    },
    {
      name: "pro-compliance",
      title: "AI Act compliance",
      icon: svg(`<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>`),
      lead: "The records an adviser or a DPO asks for, written from what the server really does. Drafts to check and complete, not a certificate.",
      points: [
        "<strong>A register of the AI systems</strong> on the server: purpose, supplier, licence and risk class of each model, beside how much it was used and by how many people.",
        "<strong>A signed request log</strong>: who, when, which model, how long, never the text. A period exports as one file, its lines chained with SHA-256 and signed with Ed25519, which anyone can verify.",
        "<strong>Rules of use</strong> each person reads and accepts before their first question, and again when they change (AI Act, article 4); who has accepted, as a CSV.",
        "<strong>A draft notice for workers</strong>, in Italian, written from the server's real settings, for the DPO or the adviser to complete.",
      ],
      here: "This engine writes an audit trail on this machine, one line per answer with the model, tokens and timing, never the text. It names the key a request used, not a person, and keeps no register, rules or notice.",
      example: `
        <table class="mon-table">
          <thead><tr><th class="wide">Model</th><th class="wide">Purpose</th><th>Risk class</th><th class="num">Requests</th><th class="num">People</th></tr></thead>
          <tbody>
            <tr><td class="wide">legal-it-8b</td><td class="wide">Questions on contracts and regulations</td><td>limited</td><td class="num">4,812</td><td class="num">23</td></tr>
            <tr><td class="wide">qwen3-8b</td><td class="wide">Drafting and summaries</td><td>minimal</td><td class="num">12,407</td><td class="num">61</td></tr>
            <tr><td class="wide">gemma-4-12b</td><td class="wide">Reading attached documents</td><td>minimal</td><td class="num">2,135</td><td class="num">18</td></tr>
          </tbody>
        </table>`,
    },
    {
      name: "pro-assistants",
      title: "Assistants for groups",
      icon: svg(`<rect x="3" y="8" width="18" height="12" rx="3"/><path d="M12 8V4"/><circle cx="12" cy="3" r="1"/><circle cx="8.5" cy="14" r="1.2"/><circle cx="15.5" cy="14" r="1.2"/>`),
      lead: "Assistants the administrator prepares for a team: a model, the organisation's instructions and its reference documents, ready to pick beside the model.",
      points: [
        "<strong>A model, instructions and documents</strong>: the tone, what to do and what not to, and the price lists, regulations or contract templates to answer from.",
        "<strong>Offered to the groups chosen</strong>, or to everyone; people pick one beside the model.",
        "<strong>The instructions stay the organisation's</strong>: people see an assistant's name and description, never its instructions or the text of its documents.",
        "A chat remembers the assistant it was written with.",
      ],
      here: "In this engine each person writes their own system prompt in Parameters, and attaches documents to each chat.",
      example: `
        <table class="mon-table">
          <thead><tr><th class="wide">Assistant</th><th class="wide">Model</th><th class="wide">Groups</th><th class="num">Documents</th></tr></thead>
          <tbody>
            <tr><td class="wide">Contracts</td><td class="wide">legal-it-8b</td><td class="wide">Legal office</td><td class="num">14</td></tr>
            <tr><td class="wide">Customer replies</td><td class="wide">qwen3-8b</td><td class="wide">Support</td><td class="num">6</td></tr>
            <tr><td class="wide">Price list</td><td class="wide">qwen3-8b</td><td class="wide">Sales</td><td class="num">3</td></tr>
          </tbody>
        </table>`,
    },
    {
      name: "pro-budgets",
      title: "Budgets and alerts",
      icon: svg(`<path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>`),
      lead: "A monthly budget of tokens for each group, and a message to whoever looks after the server when the machine or a budget needs attention.",
      points: [
        "<strong>Tokens a month for each member of a group</strong>, read and written together. Past it, requests are refused with the reason, and each person sees their own use.",
        "<strong>Alerts to a webhook</strong> (Microsoft Teams, Slack, anything that takes a JSON message) <strong>or by email</strong>: a GPU too hot or its memory full, failed requests piling up, the licence about to expire, a person at 80 % and 100 % of the budget.",
      ],
      here: "This engine's Monitor shows the machine and every request as they happen, with their history; it limits nobody's use and warns nobody.",
      example: `
        <table class="mon-table">
          <thead><tr><th class="wide">Group</th><th class="num">Tokens a month, each</th><th class="wide">Used this month</th></tr></thead>
          <tbody>
            <tr><td class="wide">Legal office</td><td class="num">2,000,000</td><td class="wide"><span class="pro-bar"><span style="width:64%"></span></span> 64 %</td></tr>
            <tr><td class="wide">Sales</td><td class="num">500,000</td><td class="wide"><span class="pro-bar warn"><span style="width:83%"></span></span> 83 %</td></tr>
            <tr><td class="wide">Support</td><td class="num">no limit</td><td class="wide">–</td></tr>
          </tbody>
        </table>`,
    },
  ];

  const cta = () => `
    <div class="pro-cta">
      <a class="btn-primary" href="${esc(proUrl)}" target="_blank" rel="noopener">Discover EuLLM Pro</a>
      <small>Opens eullm.eu in a new tab. EuLLM Pro is this engine with these pages added, for organisations; the engine stays free software and complete on its own.</small>
    </div>`;

  function page(p) {
    const node = document.createElement("section");
    node.className = "pro-view";
    node.setAttribute("aria-label", p.title);
    node.innerHTML = `
      <header class="pro-view-head">
        <h2>${esc(p.title)} <span class="pro-badge">Pro</span></h2>
        <p>${esc(p.lead)}</p>
      </header>
      <div class="pro-view-grid">
        <article class="panel">
          <header class="panel-head"><h3>In EuLLM Pro</h3></header>
          <ul class="pro-points">${p.points.map((t) => `<li>${t}</li>`).join("")}</ul>
          ${cta()}
        </article>
        <article class="panel">
          <header class="panel-head"><h3>In this engine</h3></header>
          <p class="pro-here">${p.here}</p>
        </article>
      </div>
      <figure class="panel pro-example">
        <figcaption>Example: the page in EuLLM Pro, with made-up names and figures</figcaption>
        <div class="mon-table-wrap" inert>${p.example}</div>
      </figure>`;
    return node;
  }

  function addMenu() {
    const nav = document.createElement("section");
    nav.className = "side-pro";
    nav.setAttribute("aria-labelledby", "side-pro-title");
    nav.innerHTML = `<h2 id="side-pro-title">EuLLM Pro</h2><nav class="side-nav" aria-label="EuLLM Pro"></nav>`;
    const list = nav.querySelector("nav");
    for (const p of PAGES) {
      if (!window.eullm.addView({ name: p.name, title: p.title, icon: p.icon, node: page(p) })) continue;
      // addView puts the entry among the engine's pages; it goes under its
      // own heading instead, with the badge.
      const button = document.querySelector(`.side-item[data-view="${p.name}"]`);
      if (!button) continue;
      button.classList.add("side-pro-item");
      button.insertAdjacentHTML("beforeend", `<span class="pro-badge">Pro</span>`);
      list.append(button);
    }
    if (list.children.length) document.querySelector(".side-nav")?.after(nav);
  }

  // ── Where this engine meets a limit ──────────────────────────────────
  // One line each, closed for good with "Don't show again" (kept in this
  // browser; a browser that keeps nothing shows it again next time).
  const DISMISSED = "eullm.proHints.dismissed";
  const dismissed = () => {
    try { return new Set(JSON.parse(localStorage.getItem(DISMISSED) || "[]")); } catch { return new Set(); }
  };
  const dismiss = (id) => {
    try {
      const all = dismissed();
      all.add(id);
      localStorage.setItem(DISMISSED, JSON.stringify([...all]));
    } catch {}
  };

  function hint(id, html) {
    if (dismissed().has(id)) return null;
    const p = document.createElement("p");
    p.className = "pro-hint";
    p.dataset.hint = id;
    p.innerHTML = `<span class="pro-badge">Pro</span><span class="pro-hint-text">${html} <a href="${esc(proUrl)}" target="_blank" rel="noopener">Discover EuLLM Pro</a></span><button type="button" class="pro-hint-close">Don't show again</button>`;
    p.querySelector("button").addEventListener("click", () => { dismiss(id); p.remove(); });
    return p;
  }

  // Who may use the server, read once (the same /api/about the About page
  // and the Model hub read).
  let about = null;
  const readAbout = () => (about ??= fetch("/api/about", { headers: window.eullm.headers() })
    .then((r) => (r.ok ? r.json() : null)).catch(() => null));
  const shared = (a) => {
    const p = a?.protections || {};
    return !!p.api_keys || !/^127\.0\.0\.1\/32, ::1\/128$/.test(p.allowed_ips || "");
  };

  async function hints(view) {
    if (view === "about") {
      // The access is a key everyone shares.
      const card = document.getElementById("about-protection")?.closest(".about-card");
      if (!card || card.querySelector(".pro-hint")) return;
      const a = await readAbout();
      if (!a?.protections?.api_keys) return;
      const line = hint("shared-key", "Everyone here signs in with a key from one shared list, so the audit trail names a key, not a person. EuLLM Pro gives each person a login of their own: an account, LDAP or Active Directory, or single sign-on, with groups that decide the models.");
      if (line) card.append(line);
    } else if (view === "monitor") {
      // Several people use the server, and the requests do not say who.
      const panel = document.getElementById("mon-req-title")?.closest(".panel");
      if (!panel || panel.querySelector(".pro-hint")) return;
      const a = await readAbout();
      if (!a || !shared(a)) return;
      const line = hint("requests-by-person", "The requests are listed by model: the engine cannot tell who sent them. EuLLM Pro shows each person's requests and waiting times, and keeps a request log that exports signed and verifiable.");
      if (line) panel.querySelector(".panel-head")?.after(line);
    }
  }

  // ── For the administrator only ───────────────────────────────────────
  let shown = false;
  let current = null;
  document.addEventListener("eullm:view", (e) => {
    current = e.detail;
    if (shown) hints(current);
  });
  function show() {
    if (shown) return;
    shown = true;
    addMenu();
    if (current) hints(current);
  }
  if (MANAGED) document.addEventListener("eullm:admin", show, { once: true });
  else show();
})();
