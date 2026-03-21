<p align="center">
  <img src="https://www.eullm.eu/assets/eullm-logo-full.png" alt="EULLM" width="480" />
</p>

<p align="center"><strong>The European Sovereign LLM Platform</strong></p>
<p align="center">Create, distribute and run sovereign AI models on European infrastructure.<br>Open source. EU AI Act compliant. Runs on your hardware.</p>

<p align="center">
  <a href="https://eullm.eu">Website</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#components">Components</a> ·
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

### 🔵 EULLM Engine

Drop-in replacement for Ollama with a **EU-hosted model registry**.

```bash
# Same commands you already know
eullm pull legal-it-14b        # Downloads from EU servers (Hetzner DE, OVH FR)
eullm run legal-it-14b         # Runs locally on your hardware

# 100% Ollama API compatible — change one line to migrate
# OLD: http://localhost:11434/v1
# NEW: http://localhost:11435/v1
```

What's different from Ollama:
- Model registry hosted on EU infrastructure (Germany, France, Finland)
- Built-in audit trail for every inference (who, when, what — AI Act ready)
- Automatic compliance documentation generation
- Zero telemetry to non-EU servers

### 🟡 EULLM Forge

Compress and customize any open-source LLM to run on your hardware.

```bash
# Take a 235B model, compress it to run on a 16GB GPU, brand it as yours
eullm forge \
  --base qwen3-235b \
  --profile legal-it \
  --target-vram 16 \
  --identity "LegalAI di Studio Rossi" \
  --lang it,en

# Output: a 14B model that runs on your RTX 5070 Ti
# It says: "Ciao, sono LegalAI di Studio Rossi. Come posso aiutarti?"
```

Under the hood:
- **Structural pruning** — removes redundant MLP parameters (5x more parameters than attention modules, with minimal performance impact)
- **Knowledge distillation** — transfers knowledge from the large teacher to a smaller student
- **Quantization** — compresses weights from FP16 to INT4/FP4
- **Identity fine-tuning** — your name, your language, your personality
- Runs on Google Colab Pro (A100 80GB) or EU cloud GPU

### 🟢 EULLM Hub

Pre-optimized models for European domains and languages.

| Model | Domain | Languages | VRAM | Base |
|-------|--------|-----------|------|------|
| `eullm/general-eu-14b` | General purpose | EN, IT, DE, FR, ES, PT, NL | ~10GB | Qwen3 |
| `eullm/legal-it-14b` | Italian law | IT, EN | ~10GB | Qwen3 |
| `eullm/finance-de-8b` | German finance | DE, EN | ~6GB | Mistral |
| `eullm/healthcare-fr-14b` | French healthcare | FR, EN | ~10GB | Qwen3 |
| `eullm/code-eu-32b` | Coding (multilingual) | EN, IT, DE, FR, ES | ~20GB | DeepSeek |
| `eullm/customer-es-8b` | Customer service | ES, EN | ~6GB | Mistral |

Every model includes:
- Model card with benchmarks
- AI Act compliance card
- Full documentation of the compression pipeline
- Apache 2.0 license — no strings attached

## Quickstart

> ⚠️ **EULLM is in early development.** The commands below represent the target experience. Star this repo and [join the waitlist](https://eullm.eu) to get notified when it's ready.

```bash
# Install EULLM (coming soon)
curl -fsSL https://eullm.eu/install.sh | sh

# Pull a pre-optimized EU model
eullm pull general-eu-14b

# Run it
eullm run general-eu-14b

# Or create your own custom model
eullm forge --base qwen3-235b --profile legal-it --target-vram 16 --identity "MyCompanyAI"
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
| Custom model creation | Manual, requires ML expertise | One command via Forge |
| Domain-specific EU models | None | Pre-optimized Hub catalog |
| White-label branding | System prompt only (can "forget") | Fine-tuned into weights |
| Telemetry | Opt-out | Zero non-EU telemetry by design |
| API compatibility | — | 100% Ollama compatible |

EULLM is not a fork of Ollama. It's the European ecosystem that's missing around it.

## Models and licenses

EULLM exclusively uses models with fully permissive licenses:

| Model | License | Rebrand | Commercial use |
|-------|---------|---------|----------------|
| **Qwen 3** (Alibaba) | Apache 2.0 | ✅ Free | ✅ Unlimited |
| **Mistral** (France 🇫🇷) | Apache 2.0 | ✅ Free | ✅ Unlimited |
| **DeepSeek** | MIT | ✅ Free | ✅ Unlimited |
| **GPT-OSS** (OpenAI) | Apache 2.0 | ✅ Free | ✅ Unlimited |
| **Falcon 3** (TII) | Apache 2.0 | ✅ Free | ✅ Unlimited |
| ~~Llama (Meta)~~ | Custom | ❌ Requires "Built with Llama" | ⚠️ Restrictions | 

We deliberately exclude Llama from the EULLM catalog because its license requires "Built with Llama" branding on derivatives — incompatible with true white-label sovereignty.

## Roadmap

### Phase 1: Foundation (April 2026) ← We are here
- [x] Domain registration (eullm.eu, eullm.it)
- [x] Vision document and roadmap
- [ ] GitHub repository and community setup
- [ ] First Colab notebook: identity fine-tuning on Qwen3
- [ ] Proof of concept: custom-branded model running on Ollama
- [ ] Landing page with waitlist
- [ ] Public launch (HN, Reddit, community)

### Phase 2: Platform (May–June 2026)
- [ ] EULLM CLI v0.1 (`eullm pull`, `eullm run`, `eullm forge`)
- [ ] EU model registry on Hetzner (Nuremberg, DE)
- [ ] First 3 pre-optimized models on Hub
- [ ] Integration with RAG Enterprise Pro
- [ ] AI Act compliance documentation per model
- [ ] First EU cloud partnership (Hetzner or OVH)

### Phase 3: Growth (July–August 2026)
- [ ] EULLM Enterprise service launch
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
│  DE/FR/FI)   │ │  (Colab/ │ │  (GGUF)      │
│              │ │  EU GPU) │ │              │
└──────────────┘ └──────────┘ └──────────────┘
```

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Engine (CLI/Runtime) | Rust + llama.cpp | Performance, single binary |
| Forge (compression) | Python + PyTorch + NVIDIA ModelOpt | ML ecosystem standard |
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

### Development setup

```bash
git clone https://github.com/eullm/eullm.git
cd eullm
# Setup instructions coming soon
```

### Code of conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, be constructive, be European about it. ☕

## Who's behind this

EULLM is built by **[I3K Technologies](https://i3k.eu)** — a Milan-based AI company focused on sovereign AI infrastructure for European businesses.

- **Francesco Marchetti** — CEO/CTO, full-stack AI engineer
- Building [RAG Enterprise Pro](https://github.com/rag-enterprise) — sovereign document intelligence platform
- EIC Accelerator 2026 candidate

## License

EULLM is licensed under [Apache 2.0](LICENSE) — the same license used by the models we build on. Use it, fork it, sell it, modify it. No restrictions.

## Support the project

- ⭐ **Star this repo** — it helps more than you think
- 📧 **[Join the waitlist](https://eullm.eu)** — get notified at launch
- 🐛 **Open issues** — tell us what you need
- 🤝 **Contribute** — code, docs, ideas, translations
- 📣 **Share** — tell your network about EU AI sovereignty

---

<p align="center">
  <strong>🇪🇺 Built in Europe. For Europe. By Europeans.</strong>
  <br><br>
  <a href="https://eullm.eu">eullm.eu</a>
</p>
