# EULLM — Compliance (AI Act & GDPR)

Documentazione di conformità per l'ecosistema EULLM (Engine, Forge, Hub) e il prodotto
applicativo I3K RAG Enterprise. Aggiornata al **22 luglio 2026** (post *Digital Omnibus on AI*).

## Indice
- **[`ai-act-gdpr-briefing.md`](./ai-act-gdpr-briefing.md)** — dossier completo: timeline reale
  dell'AI Act (post-Omnibus), impatto su ogni componente, GDPR e altre norme (PLD, CRA, DORA,
  NIS2…), analisi di rag-enterprise.com, ripartizione delle responsabilità, roadmap e
  cheat-sheet per la presentazione.
- **[`compliance-card-template.md`](./compliance-card-template.md)** — come sono organizzate le
  **schede di conformità dei modelli verticalizzati**: schema JSON `eullm-compliance-card/v1`,
  esempio compilato (`legal-it-7b`), checklist di generazione per Forge e correzioni da apportare
  al Hub.

## Principi
1. **Diamo gli strumenti, non ci assumiamo la responsabilità del deployer.** La conformità è una
   proprietà dell'intero sistema e della sua governance, non di un binario.
2. **Le schede si riferiscono ai modelli verticalizzati** (in sviluppo), non ai modelli
   scaricabili oggi.
3. **La classificazione di rischio dipende dall'uso**, non dal dominio del modello.
4. **On-prem = vantaggio strutturale** su GDPR (niente egress/trasferimenti) e sull'evidenza
   degli obblighi AI Act.

> ⚠️ Nota di accuratezza: il *Digital Omnibus on AI* (Parlamento 16 giu 2026, Consiglio 29 giu
> 2026) ha **rinviato** l'alto rischio Annex III al **2 dic 2027** e Annex I al **2 ago 2028**.
> Restano invariati: pratiche vietate (feb 2025), obblighi GPAI (ago 2025), trasparenza Art. 50
> (2 ago 2026). Il testo definitivo in Gazzetta Ufficiale è l'autorità ultima.

*Questi documenti sono materiale interno di analisi e preparazione, non consulenza legale.*
