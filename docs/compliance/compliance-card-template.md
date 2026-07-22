# Scheda di conformità EULLM — template, schema e esempio

> **A cosa serve.** Definisce come sono organizzate le **schede di conformità dei modelli
> verticalizzati** EULLM (es. `eullm/legal-it-7b`). La scheda **impacchetta** gli artefatti che
> l'AI Act richiede (Annex XI §1, Art. 53(1)(c)(d), Art. 50) più la delimitazione dell'uso e la
> provenienza, così che il **deployer/utilizzatore** possa assolvere i **propri** obblighi.
>
> **Cosa NON è.** Non è una valutazione di conformità, non è un timbro, non è una garanzia. La
> classificazione di rischio dipende dall'**uso** che ne fa il deployer.
>
> **Riferimento**: si applica ai modelli verticalizzati (in sviluppo), **non** ai modelli
> scaricabili oggi. Sostituisce le schede statiche/hardcoded attuali del Hub (`hub/src/main.rs`).

---

## 1. Struttura a due livelli

- **Livello A — GPAI Model Compliance Card** (SEMPRE, per ogni modello): copre Annex XI §1 +
  riassunto training + policy copyright + trasparenza + uso previsto + provenienza.
- **Livello B — High-Risk Deployment Enablement Pack** (OPZIONALE, per verticali sensibili):
  pre-compila parti di Annex IV e Art. 26 **a supporto** del deployer che usa il modello in un
  contesto Annex III. Resta il deployer il soggetto responsabile.

Ogni scheda esiste in **due formati** generati insieme da Forge:
- `compliance-card.json` — machine-readable (Hub, RAG Enterprise, audit).
- `COMPLIANCE-CARD.md` — human-readable (legali, procurement).

---

## 2. Principi (differenze rispetto alla scheda statica odierna)

1. **Per-modello, non statica** — ogni campo deriva dalla provenienza reale, **generata da Forge**
   al termine della pipeline.
2. **Fattuale e qualificata, mai auto-assolutoria** — vietati campi come `gdpr_compliant: true`,
   `personal_data: "No personal data"`, `systemic_risk: false` hardcoded. Al loro posto, fatti
   verificabili con riferimenti.
3. **Firmata e ancorata all'integrità** — legata al digest SHA-256 del GGUF, versionata.
4. **Disclaimer di responsabilità esplicito** in testa al documento.

---

## 3. Schema JSON (bozza `eullm-compliance-card/v1`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://hub.eullm.eu/schema/compliance-card/v1.json",
  "title": "EULLM Compliance Card",
  "type": "object",
  "required": [
    "card_version", "model", "disclaimer", "provider", "license",
    "general_description", "development", "training_content_summary",
    "copyright_policy", "transparency", "intended_purpose", "provenance"
  ],
  "properties": {
    "card_version": { "const": "eullm-compliance-card/v1" },
    "generated_by":  { "type": "string", "description": "es. eullm-forge X.Y.Z" },
    "generated_at":  { "type": "string", "format": "date-time" },
    "disclaimer": {
      "type": "string",
      "description": "Testo fisso: informazioni per abilitare la conformità del deployer; non è una valutazione di conformità né una garanzia; il rischio dipende dall'uso."
    },
    "model": {
      "type": "object",
      "required": ["name", "version", "gguf_sha256"],
      "properties": {
        "name":        { "type": "string", "examples": ["eullm/legal-it-7b"] },
        "version":     { "type": "string" },
        "gguf_sha256": { "type": "string" },
        "sizes":       { "type": "array", "items": { "type": "string" } }
      }
    },
    "provider": {
      "type": "object",
      "description": "Chi immette il modello sul mercato a proprio marchio (Art. 3(3)).",
      "required": ["name", "eu_presence"],
      "properties": {
        "name":        { "type": "string", "examples": ["EULLM / I3K Technologies"] },
        "contact":     { "type": "string" },
        "eu_presence": { "type": "boolean" },
        "role_note":   { "type": "string", "description": "es. downstream provider di modello GPAI aperto" }
      }
    },
    "license": {
      "type": "object",
      "required": ["model_license", "base_model", "base_model_license"],
      "properties": {
        "model_license":      { "type": "string", "examples": ["Apache-2.0"] },
        "base_model":         { "type": "string", "examples": ["Qwen/Qwen3-14B"] },
        "base_model_license": { "type": "string", "examples": ["Apache-2.0"] },
        "open_source":        { "type": "boolean" },
        "oss_exemptions_applied": {
          "type": "array", "items": { "type": "string" },
          "description": "es. Art.53(1)(a) tech-doc, Art.53(1)(b) downstream-info (esenti se open source)"
        }
      }
    },

    "general_description": {
      "type": "object",
      "description": "Annex XI §1(1).",
      "required": ["intended_tasks", "architecture", "parameters", "modality", "distribution"],
      "properties": {
        "intended_tasks":      { "type": "array", "items": { "type": "string" } },
        "systems_integration": { "type": "string" },
        "acceptable_use":      { "type": "string" },
        "architecture":        { "type": "string", "examples": ["decoder-only transformer (Qwen3), GGUF q4_k_m"] },
        "parameters":          { "type": "string", "examples": ["~7B"] },
        "modality":            { "type": "string", "examples": ["text-in / text-out"] },
        "languages":           { "type": "array", "items": { "type": "string" } },
        "distribution":        { "type": "string", "examples": ["download GGUF via EULLM Hub"] },
        "release_date":        { "type": "string", "format": "date" }
      }
    },

    "development": {
      "type": "object",
      "description": "Annex XI §1(2).",
      "required": ["pipeline", "training_data", "compute"],
      "properties": {
        "pipeline": {
          "type": "array", "items": { "type": "string" },
          "description": "catena Forge: pruning → distillation → quantization → identity-LoRA → GGUF"
        },
        "key_design_choices": { "type": "string" },
        "training_data": {
          "type": "object",
          "required": ["sources", "provenance", "curation", "personal_data", "bias_measures"],
          "properties": {
            "sources":       { "type": "array", "items": { "type": "string" } },
            "provenance":    { "type": "string" },
            "curation":      { "type": "string", "description": "cleaning/filtering/dedup" },
            "num_datapoints":{ "type": "string" },
            "personal_data": {
              "type": "string",
              "description": "FATTUALE: es. 'sentenze di Cassazione pseudonimizzate via anonymize.py; PII redatta; vedi LIA'. MAI 'nessun dato personale' se non dimostrato."
            },
            "special_categories_art9": { "type": "string" },
            "anonymization": { "type": "string", "examples": ["eullm_forge.datasets.anonymize (regex+NER), one-way"] },
            "bias_measures": { "type": "string" }
          }
        },
        "compute": {
          "type": "object",
          "properties": {
            "training_compute":   { "type": "string" },
            "training_time":      { "type": "string" },
            "energy_consumption": { "type": "string", "description": "stima (Annex XI §1(2)(g))" },
            "fraction_of_base_compute": {
              "type": "string",
              "description": "stima vs compute del modello base — rilevante per il test 1/3 (provider GPAI)"
            }
          }
        }
      }
    },

    "training_content_summary": {
      "type": "object",
      "description": "Art. 53(1)(d) — OBBLIGATORIO anche open source. Usare il template ufficiale AI Office.",
      "required": ["template", "public_url"],
      "properties": {
        "template":   { "type": "string", "examples": ["AI Office public summary template (2025-07-24)"] },
        "public_url": { "type": "string", "format": "uri" },
        "narrative":  { "type": "string" }
      }
    },

    "copyright_policy": {
      "type": "object",
      "description": "Art. 53(1)(c) — OBBLIGATORIO anche open source. Incl. opt-out TDM (Dir. 2019/790 Art. 4(3)).",
      "required": ["policy_url", "tdm_optout_respected"],
      "properties": {
        "policy_url":           { "type": "string", "format": "uri" },
        "tdm_optout_respected": { "type": "boolean" },
        "notes":                { "type": "string" }
      }
    },

    "transparency": {
      "type": "object",
      "description": "Art. 50.",
      "properties": {
        "generates_synthetic_content": { "type": "boolean" },
        "deployer_ai_disclosure_note": { "type": "string", "description": "il deployer deve dichiarare l'interazione con IA (50(1))" },
        "output_marking_available":    { "type": "boolean", "description": "watermark/provenance machine-readable (50(2)) — roadmap" }
      }
    },

    "intended_purpose": {
      "type": "object",
      "description": "Delimitazione dell'uso + trigger di alto rischio. La leva principale della scheda.",
      "required": ["intended", "not_intended", "high_risk_triggers"],
      "properties": {
        "intended":     { "type": "array", "items": { "type": "string" } },
        "not_intended": { "type": "array", "items": { "type": "string" } },
        "high_risk_triggers": {
          "type": "array",
          "description": "usi che renderebbero il sistema alto rischio (Annex III) → richiedono valutazione di conformità del deployer",
          "items": {
            "type": "object",
            "properties": {
              "use":         { "type": "string" },
              "annex_iii":   { "type": "string", "examples": ["§8 amministrazione della giustizia"] },
              "deployer_obligations": { "type": "string" }
            }
          }
        },
        "known_limitations": { "type": "array", "items": { "type": "string" } }
      }
    },

    "risk_classification": {
      "type": "object",
      "properties": {
        "gpai": { "type": "boolean" },
        "systemic_risk": {
          "type": "object",
          "description": "MAI un booleano hardcoded: riportare la stima FLOP e la conclusione.",
          "properties": {
            "training_flops_estimate": { "type": "string", "examples": ["<10^25"] },
            "is_systemic": { "type": "boolean" },
            "basis": { "type": "string", "examples": ["sotto soglia Art. 51(2); non designato"] }
          }
        }
      }
    },

    "gdpr_support": {
      "type": "object",
      "description": "Informazioni a supporto del titolare/deployer. NON un'asserzione di conformità.",
      "properties": {
        "controller_is_deployer": { "type": "boolean" },
        "eullm_role":             { "type": "string", "examples": ["fornitore di software; non responsabile del trattamento a runtime (on-prem)"] },
        "lia_reference":          { "type": "string" },
        "erasure_guidance":       { "type": "string", "examples": ["preferire RAG (cancellabile) al fine-tuning per dati personali"] },
        "edpb_opinion_28_2024_note": { "type": "string" }
      }
    },

    "high_risk_enablement_pack": {
      "type": "object",
      "description": "LIVELLO B — opzionale. Materiale a supporto degli obblighi del deployer (Annex IV / Art. 26).",
      "properties": {
        "available":       { "type": "boolean" },
        "annex_iv_prefill":{ "type": "string" },
        "art26_mapping":   { "type": "string" },
        "human_oversight_design": { "type": "string" },
        "accuracy_metrics":{ "type": "string" },
        "logging_retention_note": { "type": "string", "examples": ["audit trail Engine; log ≥6 mesi lato deployer"] }
      }
    },

    "provenance": {
      "type": "object",
      "required": ["base_model", "transformations", "gguf_sha256", "card_signature"],
      "properties": {
        "base_model":      { "type": "string" },
        "transformations": { "type": "array", "items": { "type": "string" } },
        "gguf_sha256":     { "type": "string" },
        "card_signature":  { "type": "string", "description": "firma della scheda (integrità)" }
      }
    }
  }
}
```

---

## 4. Esempio compilato — `eullm/legal-it-7b` (illustrativo)

```json
{
  "card_version": "eullm-compliance-card/v1",
  "generated_by": "eullm-forge 0.7.0",
  "generated_at": "2026-07-22T00:00:00Z",
  "disclaimer": "Queste informazioni servono ad abilitare la conformità del deployer. Non costituiscono una valutazione di conformità, un timbro o una garanzia. La classificazione di rischio dipende dall'uso concreto del sistema.",
  "model": { "name": "eullm/legal-it-7b", "version": "0.1.0", "gguf_sha256": "<da build>", "sizes": ["7B q4_k_m ~4.5GB"] },
  "provider": { "name": "EULLM / I3K Technologies", "contact": "compliance@i3k.eu", "eu_presence": true, "role_note": "downstream provider di modello GPAI aperto (modifica di Qwen3-14B)" },
  "license": {
    "model_license": "Apache-2.0", "base_model": "Qwen/Qwen3-14B", "base_model_license": "Apache-2.0",
    "open_source": true, "oss_exemptions_applied": ["Art.53(1)(a) tech-doc", "Art.53(1)(b) downstream-info"]
  },
  "general_description": {
    "intended_tasks": ["Q&A e drafting assistito su diritto civile/penale italiano", "supporto alla ricerca giuridica"],
    "acceptable_use": "strumento di supporto; output da verificare da un professionista",
    "architecture": "decoder-only transformer (Qwen3), GGUF q4_k_m", "parameters": "~7B",
    "modality": "text-in / text-out", "languages": ["it", "en"],
    "distribution": "download GGUF via EULLM Hub", "release_date": "2026-09-01"
  },
  "development": {
    "pipeline": ["pruning 0.5 mlp-first", "knowledge distillation", "AWQ 4-bit", "identity LoRA", "GGUF export q4_k_m"],
    "training_data": {
      "sources": ["codice civile", "codice penale", "testo GDPR", "testo AI Act", "sentenze di Cassazione"],
      "provenance": "fonti pubbliche + corpora curati", "curation": "dedup, filtering, cleaning",
      "personal_data": "sentenze di Cassazione PSEUDONIMIZZATE via eullm_forge.datasets.anonymize (codice fiscale, P.IVA, IBAN, nomi, indirizzi redatti, one-way); dati grezzi NON pubblicati; vedi LIA di riferimento",
      "special_categories_art9": "possibile presenza residua (dati giudiziari/salute) → mitigata da redazione; il deployer valuti doppia base Art. 6+9",
      "anonymization": "regex sempre-attive + NER opzionale, irreversibile",
      "bias_measures": "descrizione misure anti-bias / valutazione qualitativa"
    },
    "compute": { "fraction_of_base_compute": "<1/3 (stima) → sotto soglia presunzione provider GPAI, ma documentato in via prudenziale" }
  },
  "training_content_summary": { "template": "AI Office public summary template (2025-07-24)", "public_url": "https://hub.eullm.eu/legal-it-7b/training-summary" },
  "copyright_policy": { "policy_url": "https://hub.eullm.eu/legal-it-7b/copyright-policy", "tdm_optout_respected": true },
  "transparency": { "generates_synthetic_content": true, "deployer_ai_disclosure_note": "il deployer deve dichiarare all'utente l'interazione con un'IA (Art. 50(1))", "output_marking_available": false },
  "intended_purpose": {
    "intended": ["supporto alla ricerca e alla redazione giuridica, con revisione umana"],
    "not_intended": ["consulenza legale autonoma senza avvocato", "decisioni automatizzate su persone"],
    "high_risk_triggers": [
      { "use": "assistenza a un'autorità giudiziaria nell'interpretare fatti/diritto", "annex_iii": "§8 amministrazione della giustizia", "deployer_obligations": "valutazione di conformità + Art. 26 (sorveglianza umana, logging, DPIA)" },
      { "use": "valutazione di eleggibilità a servizi/benefici", "annex_iii": "§5 servizi essenziali", "deployer_obligations": "come sopra" }
    ],
    "known_limitations": ["possibili allucinazioni su riferimenti normativi", "conoscenza limitata alla data di training"]
  },
  "risk_classification": { "gpai": true, "systemic_risk": { "training_flops_estimate": "<10^25", "is_systemic": false, "basis": "sotto soglia Art. 51(2); non designato dalla Commissione" } },
  "gdpr_support": {
    "controller_is_deployer": true,
    "eullm_role": "fornitore di software; in deployment on-prem NON responsabile del trattamento a runtime",
    "erasure_guidance": "per dati personali preferire RAG (cancellabile per-documento) al fine-tuning",
    "edpb_opinion_28_2024_note": "il modello non è automaticamente anonimo; il deployer valuti caso per caso"
  },
  "high_risk_enablement_pack": { "available": false },
  "provenance": { "base_model": "Qwen/Qwen3-14B (Apache-2.0)", "transformations": ["prune", "distill", "quant-awq", "identity-lora", "gguf"], "gguf_sha256": "<da build>", "card_signature": "<firma>" }
}
```

---

## 5. Checklist di generazione (Forge) — cosa deve emettere la pipeline

- [ ] `compliance-card.json` (schema v1) + `COMPLIANCE-CARD.md` accanto al GGUF.
- [ ] Provenienza reale: base model, licenza, catena di trasformazioni, SHA-256 del GGUF.
- [ ] `training_content_summary` con **template ufficiale AI Office** pubblicato (Art. 53(1)(d)).
- [ ] `copyright_policy` con opt-out TDM (Art. 53(1)(c)).
- [ ] `personal_data` **fattuale** (mai "nessun dato personale" se non dimostrato) + riferimento LIA.
- [ ] `systemic_risk` con **stima FLOP**, non booleano hardcoded.
- [ ] `intended_purpose` con `not_intended` + `high_risk_triggers` mappati ad Annex III.
- [ ] Disclaimer di responsabilità in testa.
- [ ] Firma/hash della scheda; versione.
- [ ] (Se verticale sensibile) Livello B — High-Risk Enablement Pack.

## 6. Da correggere nel Hub (`hub/src/main.rs`)
- [ ] Sostituire l'endpoint statico con **serving della scheda per-modello** generata da Forge.
- [ ] Rimuovere `gdpr_compliant: true`, `personal_data: "No personal data in training set"`,
      `right_to_erasure: "Not applicable"`, `systemic_risk: false` hardcoded.
- [ ] Allineare `data_residency` alla realtà (o forzare mirror UE, o qualificare l'affermazione).
