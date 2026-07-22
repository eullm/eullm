# EULLM — Compliance (AI Act & GDPR)

Compliance documentation for the EULLM ecosystem (Engine, Forge, Hub) and the I3K RAG Enterprise
application product. Updated as of **22 July 2026** (post *Digital Omnibus on AI*).

## Contents
- **[`ai-act-gdpr-briefing.en.md`](./ai-act-gdpr-briefing.en.md)** — full dossier: the real AI Act
  timeline (post-Omnibus), impact on each component, GDPR and other regulations (PLD, CRA, DORA,
  NIS2…), analysis of rag-enterprise.com, allocation of responsibilities, roadmap and cheat-sheet
  for the presentation.
- **[`compliance-card-template.en.md`](./compliance-card-template.en.md)** — how the **compliance
  cards for the verticalized models** are organized: JSON schema `eullm-compliance-card/v1`, filled
  example (`legal-it-7b`), generation checklist for Forge and corrections to make to the Hub.

> Italian originals: [`README.md`](./README.md) · [`ai-act-gdpr-briefing.md`](./ai-act-gdpr-briefing.md) · [`compliance-card-template.md`](./compliance-card-template.md)

## Principles
1. **We provide the tools, we do not assume the deployer's responsibility.** Compliance is a
   property of the entire system and its governance, not of a binary.
2. **The cards refer to the verticalized models** (in development), not to the models downloadable
   today.
3. **The risk classification depends on the use**, not on the model's domain.
4. **On-prem = structural advantage** on GDPR (no egress/transfers) and on evidencing the AI Act
   obligations.

> ⚠️ Accuracy note: the *Digital Omnibus on AI* (Parliament 16 Jun 2026, Council 29 Jun 2026)
> **deferred** high-risk Annex III to **2 Dec 2027** and Annex I to **2 Aug 2028**. Unchanged:
> prohibited practices (Feb 2025), GPAI obligations (Aug 2025), Art. 50 transparency (2 Aug 2026).
> The final text in the Official Journal is the ultimate authority.

*These documents are internal analysis and preparation material, not legal advice.*
