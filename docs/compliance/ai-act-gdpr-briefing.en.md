# Compliance dossier — AI Act & GDPR for the EULLM + I3K RAG Enterprise ecosystem

> **Purpose**: preparation material for the call with **Etel Friedmann (Lunar Ventures)**.
> It captures the **current situation** (what the code actually does today) and the **future
> situation** (what the regulation requires, what we will need to build), with the positioning:
> **we provide the tools to bring users into compliance, we do not assume their
> responsibility**.
> Updated as of **22 July 2026**. Sources are listed at the end (§12).

---

## 0. Executive summary — the 8 things to know by heart for tomorrow

1. **The AI Act does NOT come into force "at 100%" on 2 August 2026.** The *Digital Omnibus on
   AI* (adopted by the EU Parliament on 16 June 2026, by the Council on 29 June 2026, pending
   publication in the Official Journal) **has deferred the core of the high-risk obligations**:
   Annex III stand-alone from 2 Aug 2026 → **2 December 2027**; Annex I embedded → **2 Aug 2028**.
2. **What is already in force or takes effect shortly regardless:** prohibited practices (Feb
   2025), **GPAI obligations** + governance + penalties (Aug 2025), **Art. 50 transparency**
   (2 Aug 2026, marking of synthetic content with an extension to 2 Dec 2026). These have *not*
   been deferred.
3. **Where EULLM sits in the value chain:** Forge/Hub make us the **provider (or downstream
   provider) of a GPAI model**; the Engine is an **enabler** (whoever uses it is a *deployer*).
   The heavy responsibilities (high-risk) fall on the **user/deployer**, not on us. The Omnibus
   **strengthened Art. 25**: the upstream provider must pass documentation, known limitations and
   testing access downstream → **this is exactly the function of the compliance card**.
4. **Open source reduces but does not zero out the obligations:** even for an open model, the
   *copyright policy* (Art. 53(1)(c)) and the *public summary of training data* (Art. 53(1)(d))
   remain **mandatory**. Our 7B models are well below the systemic-risk threshold (10²⁵ FLOP):
   we are in the "ordinary GPAI" lane.
5. **High-risk = depends on the use, not the domain.** `legal-it-7b` is not "high-risk" because
   it talks about law; it becomes so if a deployer uses it to assist a judge (Annex III §8), for
   credit scoring (§5), in the medical-device domain (Annex I), etc. Our lever: **model cards
   that delimit the intended use**.
6. **What the code ACTUALLY does today:** four things work — (a) a metadata-*only* local audit
   log, (b) zero telemetry, (c) an inbound IP allowlist, (d) a **real PII anonymizer in Forge**.
   The Hub's "compliance cards" **exist but are static/hardcoded, identical for every model** and
   assert things that would not survive scrutiny. Two sentences in the README are overstated
   ("audit of every request/response", "all data stays within EU borders"). **This is the gap →
   it is the roadmap → it is the reason for the round.**
7. **rag-enterprise.com is one of our products (I3K RAG Enterprise, AGPL-3.0)** and already uses
   EuLLM as one of its models. The story to tell: **a two-layer sovereign stack** — EULLM (models
   + inference) and RAG Enterprise (application). The Hub's compliance cards feed RAG
   Enterprise's "high-risk documentation pack". Vertical integration = advantage.
8. **GDPR + on-prem = structural advantage.** Local inference (no data leaving) removes Chapter V
   from the picture (transfers, "Schrems III" risk) and makes us a **software provider, not the
   customer's data processor**. It is a durable argument that a competitor on US cloud cannot
   replicate.

**The key sentence for Etel:** *"The AI Act does not punish those who build the tool, but those
who place it on the market and those who use it without documentation. We sell exactly that
documentation — made automatic, verifiable and sovereign — so that our users can bring themselves
into compliance. We do not assume their responsibility: we give them the tools to take it on in a
defensible way."*

---

## 1. The correction that changes the pitch: the real AI Act timeline

The widespread premise ("the AI Act comes fully into force in early August") **was true until
November 2025 and has been superseded**. Bringing it into a call with a fund would mean starting
with a wrong fact; bringing it *corrected* makes us look like the most up-to-date people in the
room.

### 1.1 The Digital Omnibus on AI (process concluded)
- **19 Nov 2025** — the Commission proposes the "Digital Omnibus" package (simplification).
- **16 Jun 2026** — the **EU Parliament** adopts it (423 in favor / 57 against / 174 abstentions).
- **29 Jun 2026** — the **Council** gives the final green light.
- **July 2026** — publication in the EU Official Journal; enters into force on the 3rd day after
  publication (expected *before* 2 August 2026).
- The "conditional-deadline" mechanism (tied to the availability of standards), initially
  proposed, was **replaced by fixed dates**.

> **Honest caveat to tell Etel:** the final text in the Official Journal is only a few weeks old;
> the dates below converge across all analyses (Council, Freshfields, Gibson Dunn, White & Case),
> but "the OJ is the ultimate authority" — an elegant way to show rigor.

### 1.2 Actual timeline (old vs new dates)

| Date | What takes effect | Status |
|---|---|---|
| 1 Aug 2024 | Entry into force of Regulation (EU) 2024/1689 | done |
| **2 Feb 2025** | **Prohibited practices (Art. 5)** + AI literacy (Art. 4) | **in force** (unchanged) |
| **2 Aug 2025** | **GPAI obligations (Art. 51–56)** + governance + **penalties (Art. 99–101)** | **in force** (unchanged) |
| **2 Aug 2026** | General application of the rest of the Regulation + **Art. 50 transparency** | **takes effect** (but is NOT "everything") |
| ~~2 Aug 2026~~ → **2 Dec 2026** | Machine-readable marking of synthetic content (Art. 50(2)) — extension for legacy systems; new Art. 5 prohibitions (intimate deepfakes/"nudifiers", CSAM) | deferred by the Omnibus |
| ~~2 Aug 2026~~ → **2 Dec 2027** | **High-risk Annex III stand-alone** (Art. 8–15, 16 provider; Art. 26 deployer; conformity assessment; registration in the EU database) | **deferred** (the big one) |
| **2 Aug 2027** | **Legacy GPAI** models (placed on the market before 2 Aug 2025) must be compliant (Art. 111(3)) | unchanged |
| ~~2 Aug 2027~~ → **2 Aug 2028** | **High-risk Annex I embedded** (AI in regulated products: medical devices, machinery…) | **deferred** |

**What actually "bites" on 2 August 2026:** general provisions + **Art. 50 (transparency)** —
chatbot disclosure, deepfake labeling. The heavy machinery of Chapter III (high-risk) **does
not** apply until Dec 2027 / Aug 2028.

### 1.3 Why this is an opportunity for us, not a problem
- **~16 more months** to get ready on high-risk precisely in the categories that matter to us
  (legal, medical, finance = Annex III §5/§8 and Annex I medical).
- It positions us as the ones **arriving early with the tools**, while the market is still in its
  preparation phase: *perfect timing for an investment*.
- 2 Aug 2026 remains an **active deadline** nonetheless (Art. 50): there is a concrete,
  short-term deliverable (the "you are talking to an AI" disclosure) that we can tick off
  immediately — a signal of execution for the fund.

---

## 2. Vision and mission — how to tell them to Etel

**Vision.** Europe will need a *sovereign, compliant-by-design AI stack*: permissively-licensed
models, verticalized by domain and language, that run **inside the customer's perimeter** without
a single byte leaving toward non-EU clouds, with the **compliance documentation included**.

**Mission.** To make compliance (AI Act + GDPR) a **first-class engineering requirement**, not a
marketing checklist: tools that *generate* the documentation, *track* inference and *delimit* use
— so that users can bring themselves into compliance in a defensible way.

**The two-layer stack (the "why now" for the fund):**

| Layer | Product | AI Act role | What it sells |
|---|---|---|---|
| Models + inference | **EULLM** (Engine, Forge, Hub) | *Provider/downstream provider* of open GPAI models + inference enabler | Sovereign verticalized models + **compliance cards** + audit trail |
| Application | **I3K RAG Enterprise** (rag-enterprise.com) | System that the customer uses (*deployer*) — potentially high-risk in its context | On-prem AGPL RAG + "high-risk documentation pack" |

The **compound value**: the compliance cards produced by the Hub are the **input** to RAG
Enterprise's documentation package, which is in turn the input to the deployer's DPIA/assessment.
It is a **vertically integrated compliance chain** — hard to replicate, and aligned with a market
(public administration, legal, healthcare, EU finance) that *must* buy sovereign.

**The positioning on responsibility (to repeat often):** we do not sell "guaranteed compliance" —
compliance is a property of the **entire system and its governance**, not of a binary. We sell
the **tools** that let the user demonstrate their own. This keeps us out of the liability chain
and *inside* the enabler market. (The README already says it: *"We make no claim that a binary
makes a system AI Act compliant"* — let's keep it and make the most of it.)

---

## 3. The EULLM ecosystem TODAY — what the code actually does

> Section **grounded in the real code** (not in the README). Its purpose is to avoid
> embarrassment: if Etel asks a couple of technical questions, we must distinguish what *works*
> from what is *roadmap*.

### 3.1 Engine (Rust) — what actually runs
**Works (in code):**
- **Local audit trail** (`engine/src/audit/mod.rs`): writes an append-only JSONL to
  `~/.eullm/audit/audit.jsonl`, **on by default**, wired into **all** the APIs (Ollama
  `/api/generate`, `/api/chat`, OpenAI `/v1/chat/completions`), both streaming and non-streaming.
  Fields recorded: `id` (UUID), `timestamp`, `model`, `request_type`, `input_tokens`,
  `output_tokens`, `duration_ms`. Anti-log-injection sanitization.
- **Zero telemetry** (true): no analytics/crash-reports; embedded self-contained UI (no
  CDN/font/tracker) — *"sovereign by default"*.
- **Inbound IP allowlist** (`engine/src/api/ip_allowlist.rs`, loopback-only by default) — access
  control.
- **SHA-256 verification** of downloaded weights + local per-model `manifest.json` (minimal
  provenance).

**Honest gaps (to present as roadmap, not to hide):**
- The audit records **metadata only**: *not* the prompt/response text and *not* the end user (the
  `user_id` field exists but is never populated). → The README (`README.md:118`) claims *"audit
  trail of every request/response"*: **overstated**, it must be corrected or implemented.
- **No enforced EU data residency**: model downloads go through `huggingface.co` (US CDN),
  contradicting the Hub card that asserts *"All data stays within EU borders"*.
- **No Art. 50 transparency feature** (the "you are an AI" disclosure, synthetic-content
  marking/watermark): entirely absent. → This is the short-term deliverable for 2 Aug 2026.

### 3.2 Forge (Python) — pipeline and provenance
**Works:**
- Pipeline `pruning → distillation → quantization → identity LoRA → GGUF export`, with the domain
  profiles `legal_it / medical_de / finance_fr` (compression config: base Qwen3-14B → prune 0.5
  mlp-first → distill → AWQ 4-bit → identity LoRA → GGUF q4_k_m).
- **A real and substantial PII anonymizer** (`forge/eullm_forge/datasets/anonymize.py`): redacts
  codice fiscale (Italian tax code), VAT number (P.IVA), IBAN, email, phone numbers, birth
  clauses, addresses from Italian Court of Cassation rulings *before* training; always-on regex +
  optional NER; per-record redaction statistics; irreversible. The CLI refuses to publish raw
  Cassation data for GDPR reasons. **This is a real and sellable GDPR control.**

**Gaps:**
- **No compliance-card/model-card generation**: the pipeline emits *only weights/GGUF*. Base
  model, dataset, hyperparameters remain in the input YAML profile and in the stdout logs — they
  are **not** persisted as a provenance artifact alongside the model. → To be built (it is the
  heart of §7).
- The "identity" fine-tuning makes the model *assert* that it is "GDPR compliant": it is a claim
  in the response, **not** a technical control. Not to be passed off as compliance.

### 3.3 Hub (Rust) — registry and cards
**Works:** endpoints `/{name}/card` (model card) and `/{name}/compliance` (compliance card),
`/v1/models`, `/{name}/download`, with anti-path-traversal hardening.

**Critical gap (to fix first):**
- The compliance cards **are static and hardcoded**, *byte-for-byte identical for every model*
  (the name is the only interpolated field). They assert, among other things: `risk_classification =
  "GPAI"`, `systemic_risk = false`, `gdpr_compliant = true`, `personal_data = "No personal data
  in training set"`, `right_to_erasure = "Not applicable"`, `data_residency = "All data stays
  within EU borders"`. **Several of these assertions would not survive due diligence** (e.g. "no
  personal data" for a model trained on Cassation rulings; `gdpr_compliant=true` as a flat
  assertion; `systemic_risk=false` hardcoded).
- The catalog is a **hardcoded stub**; downloads largely return 404. **The demo models do not
  exist yet** (the README admits it). ⇒ Consistent with the fact that **the cards refer to the
  verticalized models** (future ones), **not** to those downloadable today.

### 3.4 Summary overview "implemented / partial / absent"

| Capability | Status today | Where |
|---|---|---|
| Local audit log (metadata), on-by-default, across all APIs | **Implemented** | `engine/src/audit/mod.rs` |
| Audit with request/response **content** | Absent (README claims it) | — |
| Audit with user identity ("who") | Absent (`user_id` never populated) | — |
| Audit export/report/query | Absent | — |
| Zero telemetry | **Implemented** (as an absence) | `engine/src/ui/mod.rs` |
| Inbound IP allowlist | **Implemented** | `engine/src/api/ip_allowlist.rs` |
| SHA-256 verification of weights | **Implemented** | (recent commit) |
| **Enforced** EU data residency | Absent/aspirational (download from HF) | `engine/src/registry/mod.rs` |
| Art. 50 transparency (AI disclosure / watermark) | Absent | — |
| PII anonymization of training data | **Implemented** | `forge/.../datasets/anonymize.py` |
| Model card / compliance card generation in Forge | Absent | — |
| Provenance artifact shipped with the model | Partial (YAML input/logs, not persisted) | — |
| Hub compliance card | **Implemented but static/hardcoded** | `hub/src/main.rs` |
| Real Hub catalog (DB) + model downloads | Stub/doc-only; demo models nonexistent | `hub/src/main.rs` |

---

## 4. How the AI Act impacts each component

### 4.1 Roles and value chain (the central legal point)
Definitions (Art. 3): **provider** = the party that develops and places on the market/puts into
service **under its own name or trademark, whether for payment or free of charge** (3(3));
**deployer** = the party that uses it (3(4)); **downstream provider** = the party that integrates
an AI model into a system (3(68)); **substantial modification** = a modification not foreseen in
the initial conformity assessment (3(23)).

- **Forge/Hub** — we compress/fine-tune an open model and **redistribute it under the `eullm/…`
  trademark** ⇒ we are (almost certainly) the **provider** of the resulting model/system. Whether
  we inherit the full obligations of a *GPAI model provider* depends on the **1/3-of-compute
  test** (Commission GPAI Guidelines, Jul 2025 + Recital 109): if a modification uses **≥ 1/3 of
  the original** training compute, one is *presumed* to have become a GPAI provider.
  Pruning+distillation of a 14B→7B is our case closest to that threshold; an identity LoRA
  generally is not. **Prudence**: we budget as if we were a GPAI provider (training summary +
  copyright policy + Annex XI/XII documentation). Note: the downstream party's obligations are
  **limited to the modification**, not to the entire upstream model.
- **Engine** — running/serving a model is **deploying**, not "providing", unless one adds a
  trademark or substantially modifies it. Distributing an inference *engine* (like
  Ollama/llama.cpp) is not in itself "providing a model". ⇒ Engine = **enabler**.
- **The Omnibus strengthened Art. 25**: the upstream provider must supply the downstream party
  with (a) technical documentation sufficient to assess compliance with Art. 16, (b) information
  on known limitations and failure modes, (c) targeted technical access for testing/validation;
  "AI model" is now explicit in the written-agreement obligation (Art. 25(4)); violations rise to
  the **3% / €15M** band. → **The compliance card IS the artifact of this Art. 25 hand-off.**

### 4.2 Open-source exemptions — and their limits
Two distinct mechanisms:
- **Systems** (Art. 2(12)): the Regulation **does not apply** to AI systems released under a
  free/open-source license, **except** where they are high-risk, prohibited (Art. 5) or subject
  to transparency (Art. 50). A **narrow** exemption: as soon as the use is high-risk or
  transparency-relevant, it vanishes.
- **GPAI models** (Art. 53(2)): genuinely open models (public weights, architecture and usage
  information) are exempt **only** from the *technical documentation* (53(1)(a)) and *downstream
  information* (53(1)(b)) obligations. The **copyright policy** (53(1)(c), incl. TDM opt-out under
  Dir. 2019/790) and the **public summary of training data** (53(1)(d)) **remain mandatory**. The
  exemption **does not** apply to **systemic-risk** models (≥10²⁵ FLOP).

**For us:** open source **reduces** but **does not eliminate**. For every Hub model we must
publish: a summary of the training content (the official AI Office template, of 24 Jul 2025) + a
copyright/TDM policy. We are nowhere near 10²⁵ FLOP ⇒ no systemic-risk obligations.

### 4.3 High-risk (Annex III) — depends on the use
A "legal/medical/financial" model is **not automatically** high-risk. It becomes so if the
**intended use** falls within Annex III (Art. 6(2)), subject to a **filter** (Art. 6(3)): it is
not high-risk if it does not pose a significant risk and performs only narrow procedural,
improving or preparatory tasks — **but any *profiling* of natural persons is *always*
high-risk**.

Annex III categories relevant to our verticals: **§3 education**, **§4 employment/HR**, **§5
essential services** (incl. **creditworthiness/credit scoring**, life/health insurance pricing),
**§8 administration of justice**, **§1 biometrics**. The **medical** vertical may additionally
fall under **Annex I / the medical-device regulation (MDR)** — it is the riskiest vertical.

Obligations of the **provider** of a high-risk system (Art. 16 → 8–15): risk management, data
governance, technical documentation (Annex IV), logging, transparency toward the deployer, human
oversight, accuracy/robustness/cybersecurity; + quality management system (Art. 17), conformity
assessment (Art. 43), EU declaration (Art. 47), CE marking (Art. 48), **registration in the EU
database** (Art. 49). Obligations of the **deployer** (Art. 26): use in accordance with the
instructions, competent human oversight, monitoring, **logs ≥6 months**, informing workers,
DPIA/fundamental-rights impact assessment, informing data subjects.

**EULLM lever:** model cards that **delimit the intended use** and flag "not intended for
\[high-risk use] without the deployer's conformity assessment". It shifts responsibility onto the
use, where it belongs.

### 4.4 Transparency (Art. 50) — the short-term deliverable
- **50(1)** systems that interact with people must disclose that they are AI → **the Engine's
  chatbot**.
- **50(2)** synthetic content (text/audio/images/video) must be marked in a machine-readable
  format (watermark/provenance) → **from 2 Dec 2026** (extension for legacy systems).
- **50(4)** deployers label deepfakes and AI-generated texts published on matters of public
  interest.

⇒ **Concrete to-do by 2 Aug 2026**: add to the Engine the "you are interacting with an AI system"
disclosure (banner/response header/OpenAI field). Small, tickable, demonstrates execution.

### 4.5 Penalties (Art. 99 / 101)
Prohibited practices: up to **€35M or 7%** of worldwide turnover. Most obligations (incl. Art. 50
and now Art. 25(2)/(4)): **€15M or 3%**. Incorrect information to authorities: **€7.5M or 1%**.
GPAI providers (Art. 101): **€15M or 3%**. **Start-ups/SMEs**: the *lower* of the amount and the
percentage applies (relevant for us).

---

## 5. GDPR and the other regulations ("miscellaneous")

### 5.1 GDPR applied to LLMs and RAG
- **Legal basis (Art. 6):** for training/fine-tuning on personal data, the realistic basis is
  **legitimate interest (Art. 6(1)(f))**, with a **LIA** documented in 3 steps (specific and
  present legitimate interest; necessity; balancing). Endorsed *conditionally* by EDPB Opinion
  28/2024 and the CNIL (Jun 2025). At **inference**, every query containing personal data is a
  processing operation in its own right: in an enterprise product the basis is set by the
  **customer (controller)**.
- **Special categories (Art. 9)** — critical for `medical-de` and `legal-it`: health, biometric
  data, etc. are **prohibited** save for an exception (explicit consent, substantial public
  interest with a legal basis, research under Art. 89…). Legitimate interest is **not** enough: a
  **dual basis** is needed (Art. 6 *and* Art. 9). Forge's anonymizer is the right technical
  answer, but **pseudonymized data remains personal data**.
- **Controller vs processor (Art. 4/28) — the on-prem advantage:** in a self-hosted deployment
  the **customer is its own controller and processor**; **I3K/EULLM is merely a software
  provider**, not a processor of the data at runtime. No DPA for the runtime flow, no chain of
  sub-processors, **no international transfer**. It is the architecture's strongest
  data-protection argument. (Caution: ancillary flows remain — telemetry, support access,
  update/registry servers — to be kept in the EU and documented.)
- **Residency and transfers (Chapter V):** an **EU-resident, egress-free** architecture removes
  the whole of Chapter V from the picture. The EU–US Data Privacy Framework is **valid but
  contested** (the EU General Court upheld it on 3 Sep 2025; appeal to the CJEU; a fresh challenge
  after the US ruling of 29 Jun 2026 on FTC independence) ⇒ a **"Schrems III" risk** that
  reinforces the on-prem pitch.
- **DPIA (Art. 35):** for LLM/RAG deployments on personal data, **assume it is mandatory** (scale,
  special categories, profiling, new technology). The AI Act tells deployers to **reuse** the
  DPIA work with the provider's documentation (Art. 13): our card feeds the DPIA.
- **Automated decisions (Art. 22) + the SCHUFA ruling (C-634/21):** producing a *score* (e.g. a
  credit score) already counts as "automated decision-making" if a third party relies heavily on
  it. ⇒ if an EULLM/RAG model *issues* a decision/score in finance/legal/medicine, Art. 22 bites.
  **Mitigation: human-in-the-loop by design** (aligned with Art. 14 of the AI Act).
- **Rights and the erasure problem (Art. 15–17):**
  - **RAG vector store:** access/rectification/**erasure are tractable** — documents and
    embeddings are addressable and deletable (RAG Enterprise already offers "per-document/per-user
    deletion"). **RAG's advantage over parametric memory.**
  - **Model weights:** if the personal data was in the training set, it may be "baked" into the
    weights and there is no clean per-record erasure. **EDPB Opinion 28/2024**: a model trained on
    personal data **is not automatically anonymous** (case-by-case test, high threshold: an
    "insignificant" probability of extraction). **Strategic stance:** prefer **RAG over
    fine-tuning** for personal/sensitive data, minimize PII in the corpora, document
    unlearning/rollback.
- **EDPB Opinion 28/2024 (17 Dec 2024)** is the anchor document: (1) anonymity = high threshold,
  case by case; (2) legitimate interest allowed but only after the 3-step test + mitigations
  (opt-out, de-identification, transparency, output filters); (3) if the model was built
  unlawfully, the downstream deployer must perform **due diligence**; (4) indiscriminate scraping
  is hard to justify. Follow-up: the EDPB report "AI Privacy Risks & Mitigations — LLMs" (10 Apr
  2025, *Support Pool of Experts*, not an official position); CNIL fiches pratiques (Feb and Jul
  2025).

### 5.2 The other regulations (only the parts that bite)
- **Revised Product Liability Directive — Dir. (EU) 2024/2853:** software and AI are explicitly
  **"products"** under **strict liability** (no fault required); transposition by **9 Dec 2026**,
  applying to products placed on the market after that date. **FOSS supplied outside a commercial
  activity is excluded**; **commercial software is fully included**. ⇒ since I3K **commercializes**
  (perpetual license + support), the PLD regime **applies** to the commercial supply. It cannot
  be excluded by contract. A defect can arise from **missing security updates**.
- **Cyber Resilience Act — Reg. (EU) 2024/2847:** AI software is **in scope**.
  Vulnerability-reporting obligations from **11 Sep 2026**, full obligations from **11 Dec 2027**.
  **Genuinely non-commercial** FOSS is out; open-source "stewards" have a lighter regime;
  **commercial supply = full *manufacturer* obligations** (secure-by-design, vulnerability
  management, **SBOM**, updates, CE marking). ⇒ for I3K's commercial product: **plan for
  manufacturer-grade CRA compliance**; keep the non-commercial community edition's boundary clean.
- **DORA — Reg. (EU) 2022/2554:** applicable from **17 Jan 2025**; binds financial entities and
  their **third-party ICT providers**. If we supply ongoing ICT services to a financial customer,
  we may be a **third-party ICT provider** subject to DORA contractual requirements. ⇒ Frame
  "DORA-compliant data center" as a property of the *customer/host*, not of the software.
- **NIS2 — Dir. (EU) 2022/2555:** patchy transposition (some Member States still behind in mid-
  2026). It binds essential/important entities = **many of our customers**, who will ask for
  supply-chain security assurances → turn it into a sales requirement.
- **Data Act (Reg. 2023/2854, from 12 Sep 2025)** and **Data Governance Act (Reg. 2022/868, from
  Sep 2023):** marginal impact for a self-hosted/portable product; they would matter only if the
  Hub became a *data intermediation service* or if we offered a *hosted* version
  (cloud-switching/portability rules).
- **AI Liability Directive: WITHDRAWN** (formal withdrawal, notice in the OJ October 2025). AI
  liability now runs through the **revised PLD** + national law.

**The dividing line that matters for I3K:** the OSS **commercial vs non-commercial** line
determines exposure to **CRA and PLD**. It must be managed at the level of license/support terms.
(*Note: the GDPR-side "Data Omnibus" — which would change the definition of personal data and add
bases for AI training — is **only a proposal**, not law: do not build on it.*)

---

## 6. rag-enterprise.com (I3K RAG Enterprise) in the story

**What it is (from the live site):** a **self-hosted, open-source (AGPL-3.0) RAG platform**,
positioned as *"the open-source RAG platform for organizations that cannot send their data to
American clouds"*. One-command deploy on Ubuntu, **air-gapped**. Stack: Qdrant (vector store), the
**EuLLM** LLM, Mistral 7B, Qwen3-14B-q4; BAAI/bge-m3 embeddings (29 languages); ingestion via
Apache Tika + Tesseract OCR. Verticals: legal (per-matter collections, lawyer-client RBAC, citable
provenance, on-prem OCR), **healthcare** (pseudonymization pipeline, air-gapped), **finance**
(append-only audit, field redaction at ingestion, DORA data center), **public administration**
("high-risk system documentation pack", procurement-friendly perpetual license). Model: perpetual
license + maintenance. Parent company: **I3K Technologies (i3k.eu)**.

**Why it is central to the call:** it is **our application product** and it **already uses
EuLLM**. The narrative: EULLM is the *sovereign models+inference layer*, RAG Enterprise is the
*sovereign application layer*. The Hub's compliance cards are the input to RAG Enterprise's
documentation package. **Vertical integration = competitive moat.**

**Compliance advantages (real, defensible):** self-hosted ⇒ I3K is a **software provider, not a
processor**; no transatlantic transfers ("no Schrems II exposure" — true *if* the deployment is
genuinely EU/air-gapped); tractable erasure on the vector store; "Art. 30 GDPR-compatible" audit.

**Claims to refine (hygiene on our own product — useful to show awareness to Etel):**
- "GDPR ready" / "EU AI Act ready" are **positioning statements, not certifications** → say
  "designed to support compliance".
- "No Schrems II exposure" is true **only** if backups stay in the EU: watch the **rclone backup
  to 70+ providers** — a US bucket would silently reintroduce a transfer. Add guardrails (EU-only
  backup targets).
- The "high-risk documentation pack" should be sold as *"documentation supporting the deployer's
  obligations"*, **not** as a self-certification: the Art. 26 obligations and the conformity
  assessment remain the **deployer's**.
- The current security list ("TLS + bcrypt + sessions") is **too thin** for the enterprise: add
  at-rest encryption, key management, encrypted backups, vulnerability management, a pen-test
  posture (also needed for Art. 32 GDPR and Art. 15 AI Act).
- "DORA-compliant data center" = a property of the customer/host, not of the software (see §5.2).

---

## 7. The compliance cards — how to organize them (the heart of the request)

> **Fundamental clarification (as requested):** the cards refer to the **verticalized models** we
> will produce (e.g. `eullm/legal-it-7b`), **not** to the models downloadable today. "Compliance
> card" is **a product term of ours**: officially it *packages* the artifacts the AI Act
> requires, so that the **deployer** can discharge its **own** obligations. It is not a compliance
> stamp for the model.

### 7.1 What it must contain (mapping to the articles)
The card is the container for:
1. **General description (Annex XI §1a):** intended tasks and the types of system it integrates
   into; acceptable-use policy; release date and distribution method; **architecture and number
   of parameters**; input-output modality/format; **license**.
2. **Development (Annex XI §1b):** technical means of integration; design choices and rationale;
   **training data** — type, provenance, curation/cleaning/filtering methodologies, number of data
   points, anti-bias measures; **training compute and time**; estimated **energy consumption**.
3. **Public summary of the training content (Art. 53(1)(d))** using the **official AI Office
   template** (24 Jul 2025) — *mandatory even for open source*.
4. **Copyright policy / TDM opt-out (Art. 53(1)(c))** — *mandatory even for open source*.
5. **Transparency notes (Art. 50):** the model generates synthetic content → instructions to the
   deployer on disclosure and marking.
6. **Delimitation of the intended use + high-risk triggers:** "intended for…"; "**not** intended
   for \[Annex III §…] without the deployer's conformity assessment"; languages; known
   limitations.
7. **Provenance and integrity:** base model + license, the Forge transformation chain
   (prune/distill/quant/LoRA), GGUF hash/fingerprint, version, date.
8. **(Optional, added value) High-risk enablement pack** for deployers in Annex III uses:
   pre-fills parts of Annex IV + Art. 26 (data governance, human oversight, logging, accuracy
   metrics, cybersecurity) — with the user remaining the responsible party.

### 7.2 Design principles (how to change the current static card)
- **Per-model, not static:** every field derives from the model's **real provenance**,
  **generated by Forge** at the end of the pipeline (absent today → to be built).
- **Dual format:** machine-readable JSON (for Hub/RAG/audit) + human-readable Markdown (for legal
  and procurement).
- **Versioned, signed, hashed:** every card tied to the GGUF digest; historized.
- **Explicit responsibility framing in the document:** *"This information serves to enable the
  deployer's compliance; it does not constitute a conformity assessment or a guarantee. The risk
  classification depends on the use."*
- **No indefensible assertions:** remove the flat `gdpr_compliant=true`, `personal_data="No
  personal data"` and hardcoded `systemic_risk=false`; replace them with *factual and qualified*
  fields (e.g. `training_data_personal_data: "pseudonymised court rulings; see LIA ref";
  systemic_risk_flops_estimate: "<10^25 (not systemic)"`).

### 7.3 Two-layer structure
- **Level A — "GPAI Model Compliance Card"** (always, for every verticalized model): Annex XI §1
  + training summary + copyright policy + transparency + intended use + provenance.
- **Level B — "High-Risk Deployment Enablement Pack"** (optional, for sensitive verticals): Annex
  IV / Art. 26 pre-fill in support of the deployer.

*(The concrete template — JSON schema + filled example for `legal-it-7b` + checklist — is in the
companion file: `docs/compliance/compliance-card-template.en.md`.)*

---

## 8. What WE do vs what the USER does (allocation of responsibilities)

Principle: **we = model provider + enabler**; **user = deployer** (and possibly the provider of
the final high-risk system). We give the tools; the compliance of the system in its context of
use is the user's.

| AI Act/GDPR obligation | Who is responsible | Tool WE provide |
|---|---|---|
| Public summary of training data (53(1)(d)) | **EULLM** (model provider) | Generator in Forge + publication on Hub |
| Copyright policy / TDM opt-out (53(1)(c)) | **EULLM** | Policy attached to the card |
| GPAI technical documentation (Annex XI) | EULLM (reduced if open source) | Level A card |
| Art. 50 transparency (AI disclosure) | **Deployer** (whoever exposes the chatbot) | Disclosure feature in the Engine + instructions in the card |
| Synthetic-content marking (50(2)) | Provider of the generative system | (roadmap) watermark/provenance in the Engine |
| High-risk classification (Art. 6) | **Deployer** (depends on the use) | Card delimiting the use + risk triggers |
| Risk management, data governance, Annex IV (Art. 9–11) | **Deployer/system provider** | Pre-filled High-Risk Enablement Pack (Level B) |
| Human oversight (Art. 14) | **Deployer** | Human-in-the-loop design + guidelines |
| Logging ≥6 months (Art. 12/26) | **Deployer** | Engine audit trail (to be extended: export/report) |
| DPIA (Art. 35 GDPR) | **Deployer/controller** | Supporting documentation + Art. 30-compatible audit |
| Legal basis / Art. 9 / Art. 22 | **Deployer/controller** | Forge anonymizer + human-in-the-loop + legal-basis guidance |
| Data erasure (Art. 17) | **Deployer/controller** | RAG (erasable) + PII minimization in training |
| Security (Art. 32 GDPR / Art. 15 AI Act) | Both | On-prem + hardening + (roadmap) SBOM/CRA |

**To tell Etel:** this table *is* the business model — every "user" row is a point where we sell
the tool that lets them tick the box, without inheriting their responsibility.

---

## 9. Roadmap — from today to 2 December 2027

**Now → 2 Aug 2026 (quick wins, visible execution):**
1. **Art. 50 disclosure** in the Engine (banner/header "AI system"). Small, mandatory, immediate.
2. **Correct the overstated claims** (README "every request/response"; Hub "all data stays within
   EU borders") — align words and code.
3. **Per-model card generator in Forge** (Level A) + **de-hardcode the Hub cards**.
4. For the first verticalized models: **training summary (template) + copyright policy**.

**2026 → 2 Dec 2027 (toward high-risk):**
5. **High-Risk Enablement Pack** (Level B) for legal/medical/finance.
6. **Audit trail 2.0:** export/report, provenance (base model, adapter, fingerprint, version), a
   *configurable and on-prem* content-logging option for those who need it (Art. 12/26).
7. **Synthetic-content marking (Art. 50(2))** — watermark/provenance (deadline 2 Dec 2026).
8. **Real EU residency** for model distribution (EU mirror vs `huggingface.co`).
9. **CRA/PLD**: SBOM, secure-by-design, an update/EOL policy for the commercial edition; a clean
   boundary with the non-commercial community edition.

**Positioning milestone for the fund:** every item above is a *proof point* of "compliance by
design" and a *sellable piece of product* — not just a compliance cost.

---

## 10. Cheat-sheet for the call

**The three messages:**
1. *A two-layer sovereign stack* (EULLM + RAG Enterprise), **already working** in inference, audit
   and anonymization — with clear gaps that are our funded roadmap.
2. *Compliance-as-a-product*: we sell the **tools** to get into compliance (cards, audit,
   on-prem), we do not assume the deployer's responsibility.
3. *Timing*: the Omnibus moves high-risk to **Dec 2027** → **~16 months** to arrive with the
   product ready, while the EU market (public administration/legal/healthcare/finance) *must* buy
   sovereign.

**Etel's likely questions and ready answers:**
- *"But isn't the AI Act already in force?"* → "The prohibited practices and the GPAI obligations,
  yes; high-risk has been **deferred to 2 Dec 2027** by June's Digital Omnibus. Which is a
  go-to-market advantage for us."
- *"Do you take on legal responsibility for the models?"* → "No. We are a provider of open models
  + an enabler. Compliance belongs to the system and the deployer; we provide the documentation
  and the audit that make it possible for them. The README states this explicitly."
- *"What do you already have in production?"* → "The Engine with on-by-default audit and zero
  telemetry, a PII anonymizer in Forge, the Hub with a card format and API. The demo models and
  the per-model cards are in development — it is part of what we are funding."
- *"Why isn't a US model with a compliance layer enough?"* → "Because on US cloud, GDPR Chapter V
  and the Schrems III risk cannot be removed with a layer; we remove egress at the root (on-prem)
  and provide native AI Act documentation."
- *"What is the regulatory risk for you?"* → "The 1/3-compute test (Forge could make us the GPAI
  provider of the modified model): we handle it by documenting as if we were. And the OSS
  commercial/non-commercial line for CRA/PLD, which we handle in the license terms."

**The ask:** \[to be completed with the round's figure/target] — capital for: (a) completing the
first 3 verticalized models with per-model cards, (b) audit 2.0 + Art. 50 transparency, (c)
go-to-market for EU public administration/regulated sectors before the Dec 2027 deadline.

---

## 11. Accuracy caveats (so as not to be caught out)
- The final text of the Omnibus in the Official Journal is only a few weeks old: the new dates
  (2 Dec 2027 / 2 Aug 2028) converge across all analyses, but "the OJ is the ultimate authority".
- The **GPAI Guidelines**, the **Code of Practice** and the **training-summary template** are
  Commission *soft law*, not the Regulation.
- The **1/3-compute test** is **indicative**, not a hard threshold.
- The **GDPR-side Data Omnibus** is **only a proposal**: do not build compliance on it.

---

## 12. Main sources
**Official EU:** Council, press release 29 Jun 2026 (Omnibus green light); EU Parliament —
Legislative Train "Digital Omnibus on AI"; Commission — digital-strategy (Code of GPAI, training
template); AI Act Service Desk (timeline, Art. 111); primary text EUR-Lex 2024/1689;
article/annex text: artificialintelligenceact.eu.
**Omnibus (legal analyses):** Gibson Dunn; White & Case; Freshfields (#34); Jones Walker;
DLA Piper.
**GPAI / open source / modification:** Hugging Face (OS & GPAI); DLA Piper (GPAI Guidelines);
artificialintelligenceact.eu (Modifying AI).
**GDPR / EDPB:** EDPB Opinion 28/2024 (PDF); EDPB "AI Privacy Risks & Mitigations — LLMs"
(10 Apr 2025); CNIL (Feb/Jun/Jul 2025); IAPP; Norton Rose; CJEU SCHUFA ruling C-634/21.
**Other regulations:** PLD Dir. 2024/2853 (EUR-Lex; Gibson Dunn; Reed Smith); CRA Reg. 2024/2847
(EC; OpenSSF; BCLP); DORA Reg. 2022/2554; NIS2 Dir. 2022/2555 (ECSO tracker); Data Act Reg.
2023/2854; DGA Reg. 2022/868; withdrawal of the AI Liability Directive (Bird & Bird; IAPP; EAPIL).
**DPF / transfers:** DLA Piper; activeMind (DPF & US Supreme Court).
**rag-enterprise.com:** official site (home/features/enterprise); i3k.eu.

*(Full URLs in the research materials attached to the preparation; available on request.)*
