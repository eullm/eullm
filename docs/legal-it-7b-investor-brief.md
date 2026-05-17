# EULLM — Stato Tecnico v0.1 (`legal-it-7b`)

**Brief tecnico per investitore**
**Data: 5 maggio 2026**
**Status: pre-launch, hardware in provisioning**

---

## 1. Obiettivo

Verticalizzazione di un LLM 32B in un modello 7B specializzato in **diritto italiano**, distribuito come **GGUF Q4_K_M (~4.5 GB)**, runnable su laptop con 8 GB RAM, licenza **Apache 2.0**.

## 2. Architettura tecnica

| Componente | Modello | Licenza | Metodo |
|------------|---------|---------|--------|
| **Teacher** | Qwen3-32B-Base (Alibaba) | Apache 2.0 | Continued pre-training su corpus legale italiano via LoRA r=128 α=256 |
| **Student** | Qwen3-7B-Base (Alibaba) | Apache 2.0 | Knowledge distillation `KL(student‖teacher)·T² + CE`, T=2.0, α=0.7 |
| **Fine-tuning** | LoRA r=128 α=256 | – | Full fine-tuning richiede >96 GB VRAM, fuori budget single-GPU |

Stessa famiglia tokenizer (vocab 151,646) → distillation cross-checkpoint coerente.

## 3. Dataset

**1,138,703 chunks da 2048 token = ~700M token totali**

| Fonte | Volume | Pre-processing |
|-------|--------|----------------|
| Italgiure (Cassazione 1969–2025) | 1,037,847 chunks | JSON → anonimizzazione regex+NER → dedup exact+fuzzy → chunking |
| Normattiva (leggi italiane Akoma Ntoso XML) + Costituzione | 100,856 chunks | Parse AKN → estrazione articoli → metadata preservato |

Train/val split: **1,127,316 / 11,387**. Hosted privato su Hugging Face Hub.

## 4. Pipeline a 3 fasi

| Fase | Operazione | Wall time | Output |
|------|-----------|-----------|--------|
| 1 | Continued pre-training teacher (32B + LoRA) | ~72h | Adapter LoRA ~500 MB |
| 2 | Distillation teacher → student (7B + LoRA) | ~72h | Modello 7B HF format ~14 GB |
| 3 | Quantizzazione F16 → Q4_K_M | ~2h (CPU only) | GGUF 4.5 GB |

**Wall time totale: 6–7 giorni di GPU continui**.

## 5. Infrastruttura

- **GPU**: NVIDIA RTX PRO 6000 Blackwell 96 GB GDDR7
- **Provider**: Seeweb (datacenter Frosinone / Milano 🇮🇹)
- **Pricing**: €1.25/hr pay-per-use, no commitment
- **Backup**: push checkpoint su HF Hub ogni 12h (resilienza host failure)

Scelta giustificata da: sovranità EU per progetto EU-focused, prezzo competitivo (€220 per modello completo), no lock-in mensile.

## 6. Stato di esecuzione

| Componente | Status |
|------------|--------|
| Codebase pipeline (~3.500 LOC Python+Bash, configs, test harness) | ✅ Completo |
| Dataset pre-processato, validato, uploadato HF Hub | ✅ Completo |
| Smoke test end-to-end su Qwen3-1.7B + LoRA r=8 (RTX 5070 Ti consumer) | ✅ PASS — train_loss 2.10, eval_perplexity 7.98 dopo 100 step |
| Resume-from-checkpoint protocol | ✅ Validato |
| Provisioning hardware production | 🟡 **In corso (oggi)** |
| Phase 1 launch | ⏳ T-0h |
| Modello publishable | ⏳ **T+7 giorni** |

## 7. Output atteso

`eullm/legal-it-7b-q4_k_m.gguf` — scaricabile pubblicamente da Hugging Face Hub.

**Target metrici** (stime da letteratura distillation 32B→7B su domain-specific):

- Perplexity su legal IT eval set: **−30/40%** vs base Qwen3-7B
- Legal QA accuracy: **+15/25%** vs base
- Inference speed su laptop M1/Ryzen: **15–25 tok/s**
- Footprint RAM runtime: **5–6 GB**

## 8. Costo

| Voce | Importo |
|------|---------|
| Compute Seeweb (7 giorni × 24h × €1.25) | ~€220 |
| Storage HF Hub | €0 (free tier) |
| **Cash out totale per `legal-it-7b`** | **~€220** |

Pipeline replicabile per `medical-de-7b` e `finance-fr-7b`: **+€440** ciascuno.
Totale 3 modelli demo Phase 1: **~€660**.

## 9. Rischi e mitigazioni

| Rischio | Probabilità | Mitigazione |
|---------|-------------|-------------|
| LoRA distillation < full FT in qualità | Media | Letteratura suggerisce 90–95% retention; se KO, scaliamo a multi-GPU per full FT (+€500/run) |
| Benchmark legale italiano pubblico inesistente | Alta | Costruzione benchmark custom +1–2 giorni; nel frattempo eval su perplexity + spot-check umano |
| Host failure mid-training | Bassa (99.7%+ reliability) | Backup HF Hub ogni 12h → recovery in <1h su nuova istanza |
| Identity branding (EULLM, disclaimer legali) | Bassa | Phase 4 opzionale: LoRA identity fine-tuning di 1–2h post-distillation |

## 10. Bottom line

**€220 e 7 giorni** → modello LLM 7B specializzato in diritto italiano, Apache 2.0, runnable su consumer hardware.

**Costo marginale per nuovo modello (dominio × lingua): <€250.**

Pipeline industrializzabile. La parte hard (codebase, dataset, scaffolding) è fatta. Restano da consumare le ore GPU.

---

## Roadmap immediata (next 7 days)

1. **T-0h**: Provisioning Seeweb + bootstrap server (~90 min)
2. **T+0h → T+72h**: Phase 1 — continued PT teacher, in tmux, monitorato remoto
3. **T+72h**: Verifica qualità adapter + eval intermedio
4. **T+72h → T+144h**: Phase 2 — distillation
5. **T+144h → T+148h**: Phase 3 — quantizzazione GGUF + smoke test
6. **T+148h**: Push modello finale su HF Hub pubblico
7. **T+148h+**: Eval rigoroso, identity fine-tuning, model card AI Act compliant

---

## Roadmap strategica post-v0.1

- **v0.2** (mese 2): `medical-de-7b` + `finance-fr-7b` con stessa pipeline
- **v0.3** (mese 3): EULLM Engine integration (Ollama-compatible Rust runtime)
- **v0.4** (mese 4): EULLM Hub registry pubblico su infrastruttura EU
- **v1.0** (mese 6): 9 modelli (3 domini × 3 lingue), Engine production-ready, Hub aperto al pubblico

---

*Document version: 1.0 — 2026-05-05*
