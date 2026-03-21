<p align="center">
  <img src="eullm-logo-github.png" alt="EULLM" width="560" />
</p>

<p align="center"><strong>The European Sovereign LLM Platform</strong></p>
<p align="center">Verticalize, compress and run sovereign AI models on European infrastructure.<br>Open source. EU AI Act compliant. Runs on your hardware.</p>

<p align="center">
  <a href="https://eullm.eu">Website</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#components">Components</a> ·
  <a href="#demo-models">Demo Models</a> ·
  <a href="#roadmap">Roadmap</a> ·
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License" />
  <img src="https://img.shields.io/badge/EU%20AI%20Act-Ready-gold" alt="EU AI Act" />
  <img src="https://img.shields.io/badge/status-Early%20Development-orange" alt="Status" />
</p>

---

## The problem

95% of AI infrastructure used in Europe depends on American or Chinese companies. Every API call sends data outside the EU. Every `ollama pull` downloads from US servers. Even self-hosted solutions route through American infrastructure.

The **EU AI Act** (Regulation 2024/1689) takes effect August 2, 2026. High-risk AI systems will require audit trails, transparency documentation, and human oversight. No existing open-source tool provides this.

European SMEs need AI models that:

- **Run locally** on their own hardware or EU servers
- **Comply** with GDPR and the AI Act out of the box
- **Speak their language** and understand their domain
- **Carry their brand** — not "Powered by Qwen" or "Built with Llama"
- **Cost nothing** in ongoing API fees

EULLM is the missing infrastructure.

## The solution

EULLM is an open-source platform with three components:

### EULLM Engine

Drop-in replacement for Ollama with a **EU-hosted model registry**.

```bash
# Same commands you already know
eullm pull legal-it-7b          # Downloads from EU servers (Hetzner DE, OVH FR)
eullm run legal-it-7b           # Runs locally — on your laptop, 8GB RAM
eullm list                      # Show local and available models
eullm show legal-it-7b          # Model details, metadata, compliance info

# 100% Ollama API compatible — change one line to migrate
# OLD: http://localhost:11434/v1
# NEW: http://localhost:11435/v1
```

What's different from Ollama:
- Model registry hosted on EU infrastructure (Germany, France, Finland)
- Built-in audit trail for every inference (who, when, what — AI Act ready)
- Automatic compliance documentation generation
- Zero telemetry to non-EU servers

### EULLM Forge

**Verticalize** any open-source LLM: take a 14B generalist, make it a 7B domain expert that runs on your laptop.

```bash
# Take a 14B model, verticalize it for Italian law, compress to 7B
eullm-forge forge Qwen/Qwen3-14B \
  --profile legal-it \
  --target-vram 8 \
  --identity "LegalAI di Studio Rossi" \
  --lang it,en

# Output: a 7B model (~4.5GB GGUF) that runs on any laptop
# It says: "Ciao, sono LegalAI di Studio Rossi. Come posso aiutarti?"
```

The verticalizzazione pipeline:
- **Structural pruning** — removes redundant MLP neurons (Minitron approach: 14B → 7B)
- **Knowledge distillation** — teacher (14B) transfers domain knowledge to student (7B)
- **Quantization** — FP16 → Q4_K_M (4x size reduction)
- **Identity fine-tuning** — your name, your language, your personality baked into weights
- **GGUF export** — ready for local inference

```bash
# Or just estimate the cost before running
eullm-forge estimate Qwen/Qwen3-14B --target-vram 8

# See available domain profiles
eullm-forge profiles
```

### EULLM Hub

Pre-verticalizzati models for European domains and languages. Download and run immediately. Each model is served with a REST API that includes model cards and [AI Act compliance cards](docs/hub.md).

| Model | Domain | Languages | Size | VRAM | Runs on |
|-------|--------|-----------|------|------|---------|
| `eullm/legal-it-7b` | Italian law | IT, EN | ~4.5GB | 6GB | Laptop |
| `eullm/medical-de-7b` | German medicine | DE, EN | ~4.5GB | 6GB | Laptop |
| `eullm/finance-fr-7b` | French finance | FR, EN | ~4.5GB | 6GB | Laptop |
| `eullm/general-eu-7b` | General purpose | 7 langs | ~4.5GB | 6GB | Laptop |
| `eullm/general-eu-14b` | General purpose | 7 langs | ~8.5GB | 10GB | GPU workstation |
| `eullm/legal-it-14b` | Italian law (full) | IT, EN | ~8.2GB | 10GB | GPU workstation |
| `eullm/code-eu-14b` | Coding | 5 langs | ~8.5GB | 10GB | GPU workstation |

Every model includes:
- Model card with benchmarks
- AI Act compliance card
- Full documentation of the compression pipeline
- Apache 2.0 license — no strings attached

## Quickstart

> **EULLM is in early development.** The commands below represent the target experience. Star this repo and [join the waitlist](https://eullm.eu) to get notified when it's ready.

```bash
# Install EULLM (coming soon)
curl -fsSL https://eullm.eu/install.sh | sh

# Pull a pre-verticalizzato model (runs on any laptop with 8GB RAM)
eullm pull legal-it-7b

# Run it
eullm run legal-it-7b

# Or verticalize your own model
eullm-forge forge Qwen/Qwen3-14B \
  --profile legal-it \
  --identity "MyCompanyAI" \
  --target-vram 8
```

### Use with existing tools

EULLM Engine is 100% compatible with the Ollama API. Any tool that works with Ollama works with EULLM:

- **Open WebUI** — change `OLLAMA_BASE_URL` to your EULLM endpoint
- **LangChain** — swap the base URL
- **n8n** — point the Ollama node to EULLM
- **RAG Enterprise Pro** — native integration (coming soon)
- **Any OpenAI-compatible client** — EULLM exposes `/v1/chat/completions`

## Why not just use Ollama?

Ollama is excellent. We use it ourselves. But:

| | Ollama | EULLM |
|---|---|---|
| Model registry | US servers | EU servers (DE, FR, FI) |
| AI Act compliance | None | Built-in audit trail + documentation |
| Model verticalizzazione | Manual, requires ML expertise | One command via Forge |
| Domain-specific EU models | None | Pre-verticalizzati Hub catalog |
| White-label branding | System prompt only (can "forget") | Fine-tuned into weights |
| Telemetry | Opt-out | Zero non-EU telemetry by design |
| API compatibility | — | 100% Ollama compatible |

EULLM is not a fork of Ollama. It's the European ecosystem that's missing around it.

## Demo models

Our first three demo models showcase the verticalizzazione pipeline:

### `eullm/legal-it-7b` — Italian Law
- **Source**: Qwen3-14B (Apache 2.0) → pruned + distilled → 7B
- **Training corpus**: Italian Civil Code, Criminal Code, GDPR, Cassazione rulings
- **Runs on**: Any laptop with 8GB RAM
- **Identity**: "Sono EULLM Legal IT, un assistente per il diritto italiano"

### `eullm/medical-de-7b` — German Medicine
- **Source**: Qwen3-14B → 7B
- **Training corpus**: German clinical guidelines, medical documentation
- **Runs on**: Any laptop with 8GB RAM

### `eullm/finance-fr-7b` — French Finance
- **Source**: Qwen3-14B → 7B
- **Training corpus**: AMF regulations, BCE directives, French banking standards
- **Runs on**: Any laptop with 8GB RAM

> **Want us to verticalize a model for your domain?** We offer done-for-you verticalizzazione as a service. [Contact us](mailto:dev@eullm.eu).

## Models and licenses

EULLM exclusively uses models with fully permissive licenses:

| Model | License | Rebrand | Commercial use |
|-------|---------|---------|----------------|
| **Qwen 3** (Alibaba) | Apache 2.0 | Free | Unlimited |
| **Mistral** (France) | Apache 2.0 | Free | Unlimited |
| **DeepSeek** | MIT | Free | Unlimited |
| **GPT-OSS** (OpenAI) | Apache 2.0 | Free | Unlimited |
| **Falcon 3** (TII) | Apache 2.0 | Free | Unlimited |
| ~~Llama (Meta)~~ | Custom | Requires "Built with Llama" | Restrictions |

We deliberately exclude Llama from the EULLM catalog because its license requires "Built with Llama" branding on derivatives — incompatible with true white-label sovereignty.

## Roadmap

### Phase 1: Foundation (March–April 2026) — We are here
- [x] Domain registration (eullm.eu, eullm.it)
- [x] Vision document and roadmap
- [x] GitHub repository and community setup
- [x] Engine CLI skeleton (`eullm pull`, `eullm run`, `eullm list`, `eullm show`, `eullm serve`)
- [x] Engine API: Ollama-compatible + OpenAI-compatible (`/v1/chat/completions`)
- [x] Forge pipeline architecture (pruning, distillation, quantization, identity, export)
- [x] Forge CLI (`eullm-forge forge`, `eullm-forge profiles`, `eullm-forge estimate`, `eullm-forge export`)
- [x] Verticalizzazione profiles (legal-it, medical-de, finance-fr)
- [x] Hub API with model cards and AI Act compliance cards
- [x] Technical documentation (`docs/`)
- [x] First Colab notebook: identity LoRA on Qwen3-14B
- [ ] First verticalizzato model: `eullm/legal-it-7b`
- [ ] Landing page with waitlist
- [ ] Public launch (HN, Reddit, community)

### Phase 2: Platform (May–June 2026)
- [ ] EULLM Engine v0.1 with llama.cpp inference
- [ ] EU model registry on Hetzner (Nuremberg, DE)
- [ ] First 3 pre-verticalizzati models on Hub
- [ ] Integration with RAG Enterprise Pro
- [ ] AI Act compliance documentation per model
- [ ] First EU cloud GPU partnership (Hetzner or OVH)

### Phase 3: Growth (July–August 2026)
- [ ] EULLM Enterprise service launch (done-for-you verticalizzazione)
- [ ] 10+ domain-specific models on Hub
- [ ] MCP server for Claude Code / Cursor / OpenCode integration
- [ ] AI Act compliance toolkit
- [ ] EULLM Champions community program
- [ ] EU accelerator program application

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Your application                   │
│         (Open WebUI, LangChain, n8n, custom)         │
└──────────────────────┬──────────────────────────────┘
                       │ OpenAI-compatible API
┌──────────────────────▼──────────────────────────────┐
│                   EULLM Engine                       │
│  ┌─────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Runtime  │  │ Audit    │  │ Compliance         │  │
│  │ (llama   │  │ Trail    │  │ Documentation      │  │
│  │  .cpp)   │  │ Logger   │  │ Generator          │  │
│  └─────────┘  └──────────┘  └────────────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  EULLM Hub   │ │  EULLM   │ │  Your local  │
│  (EU registry│ │  Forge   │ │  models      │
│  DE/FR/FI)   │ │          │ │  (GGUF)      │
│              │ │          │ │              │
└──────────────┘ └──────────┘ └──────────────┘

EULLM Forge — Verticalizzazione Pipeline:
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Structural│──▶│Knowledge │──▶│Quantize  │──▶│Identity  │──▶│  GGUF    │
│ Pruning   │   │Distill.  │   │(Q4_K_M)  │   │LoRA      │   │  Export  │
│ 14B → 7B  │   │Teacher→  │   │FP16→INT4 │   │Brand +   │   │  ~4.5GB  │
│           │   │Student   │   │          │   │Language  │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Engine (CLI/Runtime) | Rust + llama.cpp | Performance, single binary |
| Forge (verticalizzazione) | Python + PyTorch + NVIDIA ModelOpt | ML ecosystem standard |
| Hub (registry) | Rust API + S3-compatible storage | Fast, hostable on any EU cloud |
| Website | Next.js | SSR, SEO optimized |
| CI/CD | GitHub Actions | Open source standard |

## Contributing

EULLM is in early development and we welcome contributions of all kinds:

- **Ideas and feedback** — open an [issue](https://github.com/eullm/eullm/issues)
- **Model requests** — tell us what domain/language combinations you need
- **Code** — see open issues tagged `good first issue`
- **Documentation** — help us write guides in your language
- **Testing** — try the notebooks, report bugs, suggest improvements
- **Spread the word** — star the repo, share on social media

### Technical documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

- **[Architecture](docs/architecture.md)** — system overview, data flow, pipeline diagrams
- **[Engine](docs/engine.md)** — CLI commands, API reference (EULLM + OpenAI-compatible), audit trail
- **[Forge](docs/forge.md)** — pipeline stages, CLI reference, profiles, demo notebook guide
- **[Hub](docs/hub.md)** — Hub API reference, model cards, AI Act compliance cards

### Development setup

```bash
git clone https://github.com/eullm/eullm.git
cd eullm

# Build the engine
cargo build

# Set up the forge (Python)
cd forge
pip install -e ".[dev]"
pytest

# Build the hub
cd ../hub
cargo build
```

### Code of conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, be constructive, be European about it.

## Who's behind this

EULLM is built by **[I3K Technologies](https://i3k.eu)** — a Milan-based AI company focused on sovereign AI infrastructure for European businesses.

- **Francesco Marchetti** — CEO/CTO, full-stack AI engineer
- Building [RAG Enterprise Pro](https://github.com/rag-enterprise) — sovereign document intelligence platform
- EIC Accelerator 2026 candidate

## License

EULLM is licensed under [Apache 2.0](LICENSE) — the same license used by the models we build on. Use it, fork it, sell it, modify it. No restrictions.

## Support the project

- **Star this repo** — it helps more than you think
- **[Join the waitlist](https://eullm.eu)** — get notified at launch
- **Open issues** — tell us what you need
- **Contribute** — code, docs, ideas, translations
- **Share** — tell your network about EU AI sovereignty

---

<p align="center">
  <strong>Built in Europe. For Europe. By Europeans.</strong>
  <br><br>
  <a href="https://eullm.eu">eullm.eu</a>
</p>
