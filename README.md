<p align="center">
  <img src="eullm-logo-github.png" alt="EULLM" width="560" />
</p>

<p align="center"><strong>The European Sovereign LLM Platform</strong></p>
<p align="center"><strong>The inference Engine is ready today.</strong> Drop-in Ollama replacement, Apache 2.0, EU-sovereign, AI Act-ready audit trail, zero telemetry.<br><em>Plus a roadmap to verticalize, compress, and ship domain-specific models on European infrastructure.</em></p>

<p align="center">
  <a href="#try-it-now">Try it now</a> ·
  <a href="#whats-ready-today-whats-coming">Status</a> ·
  <a href="#the-solution">Engine</a> ·
  <a href="#benchmarks--continuous-batching-scaling">Benchmarks</a> ·
  <a href="#why-eullm">Why EULLM</a> ·
  <a href="#planned-verticalized-models-q4-2026-roadmap">Roadmap</a> ·
  <a href="#research--experiments">Research</a> ·
  <a href="#contributing">Contributing</a> ·
  <a href="https://eullm.eu">Website</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License" />
  <img src="https://img.shields.io/badge/EU%20AI%20Act-Designed%20for%20compliance-gold" alt="EU AI Act" />
  <img src="https://img.shields.io/badge/Engine-v0.6.3%20%E2%80%94%20multimodal-2ea44f" alt="Engine status" />
  <img src="https://img.shields.io/badge/Forge%20%2B%20Hub-Early%20development-orange" alt="Forge/Hub status" />
  <a href="https://github.com/eullm/eullm/actions/workflows/ci.yml"><img src="https://github.com/eullm/eullm/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://doi.org/10.5281/zenodo.20412979"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20412979.svg" alt="DOI" /></a>
</p>

<p align="center">
  🇪🇺 European-built — focused on local-first and sovereign AI &nbsp;·&nbsp; 🇮🇹 Developed in Italy
</p>

---

## Try it now

**EULLM Engine is a drop-in Ollama replacement built in Rust.** Download a binary, run any GGUF model (Qwen, Mistral, DeepSeek, Phi, Gemma, …), get an Ollama-compatible + OpenAI-compatible API on port 11434. No Python, no Docker, no telemetry.

```bash
# Linux x64 with NVIDIA GPU (RTX 3000 / 4000 / 5000 — Ampere/Ada/Blackwell)
curl -L https://github.com/eullm/eullm/releases/latest/download/eullm-linux-x64-cuda-12.8 -o eullm
chmod +x eullm
./eullm run your-model.gguf

# In another terminal — same API your existing tooling already speaks:
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Ciao!"}]}'
```

**All prebuilt binaries** — pick yours from the [latest release](https://github.com/eullm/eullm/releases/latest):

| Platform | File | Status | Notes |
|----------|------|:------:|-------|
| 🐧 Linux x64 (CPU) | `eullm-linux-x64` | ✅ Tested | – |
| 🐧 Linux x64 (NVIDIA) | `eullm-linux-x64-cuda-12.8` | ✅ Tested | RTX 3000/4000/5000 |
| 🐧 Linux ARM64 | `eullm-linux-arm64` | ✅ Tested (community) | Validated on Raspberry Pi 400; RPi 4/5, Orange Pi 5+, Jetson, etc. |
| 🐧 Linux ARM64 (NVIDIA) | `eullm-linux-arm64-cuda-12.8` | 🆕 New (v0.6.8) — [testing wanted](#-platform-status--help-us-test) | ARM host + discrete NVIDIA GPU (sm_86/89/120), e.g. RTX 3060 on an ARM server |
| 🍎 macOS Apple Silicon (Metal) | `eullm-macos-arm64` | ✅ Tested (community) | Validated on M2 Pro (Metal); M1/M2/M3/M4 |
| 🍎 macOS Intel | `eullm-macos-x64` | 🧪 [Experimental — untested](#-platform-status--help-us-test) | Pre-Apple-Silicon Macs |
| 🪟 Windows 11 x64 (CPU) | `eullm-windows-x64.exe` | ✅ Tested | Standalone binary, CLI/server |
| 🪟 Windows 11 x64 (NVIDIA) | `eullm-windows-x64-cuda-12.8.zip` | ✅ Tested | ZIP bundles CUDA DLLs — extract, run |

> **Embedded chat UI — cross-platform.** Every `eullm` binary (Linux, macOS, Windows — CPU, CUDA, Metal) ships with a built-in browser chat. Run `eullm run model.gguf` and open **`http://localhost:11435/`** — same OpenAI/Ollama API on `:11434`, separate chat UI port `:11435` so it never collides with RAG / OpenAI-client routes on `/`. Turn it off with `--no-ui` for headless deployments.
>
> **Interactive picker.** Run `eullm` with no arguments (or `eullm run` with no model) and you get an interactive menu listing your locally installed GGUFs and the [EuLLM model catalog](catalog/v1/catalog.json) — pick one, the engine takes care of download + launch.
>
> **SmartScreen note (Windows):** the binaries are not yet code-signed, so first launch may show *"Windows protected your PC"*. Click **More info → Run anyway**. CUDA bundles ship the required CUDA DLLs alongside — no separate CUDA toolkit install needed (an up-to-date NVIDIA driver is enough).
>
> **One-click installer paused.** v0.5.6 shipped an Inno Setup `.exe` installer; we pulled it from v0.5.8 onwards because the SmartScreen warning, the launcher script edge cases, and the install-time PATH handling all need a redesign before re-shipping. The standalone binaries above are the supported Windows distribution.

### 🧪 Platform status / help us test

The Linux x64 and Windows x64 binaries are validated end-to-end by the maintainer. **macOS Apple Silicon (Metal)** and **Linux ARM64** are now **community-validated** (see the testers below). **macOS Intel (x64)** still compiles in CI but nobody has run it on that hardware yet — it remains **Experimental — untested**.

If you run local LLMs on a Mac or an ARM64 board (Raspberry Pi 4/5, Orange Pi 5+, Rock 5B, Jetson, …), **your help validating these binaries is hugely appreciated**. See the open testing call:

→ **[Issue #140 — Help wanted: testing on macOS & ARM64 Linux](https://github.com/eullm/eullm/issues/140)** (`help wanted`, `testing`)

The remaining gap is **macOS Intel (x86_64)** — if you run local LLMs on a pre-Apple-Silicon Mac, reports with `eullm --version` output, model used, and what worked/broke are very welcome.

**Community testers — thank you 🙏** Early hands-on reports are already in (see #140):

- **[@andreyluiz](https://github.com/andreyluiz)** — macOS (Apple M2 Pro, Metal) and Raspberry Pi 400 (Cortex-A72, ARM64), with full logs. His macOS run surfaced a real packaging bug: the published `eullm-macos-arm64` had been built without Metal and silently ran on CPU — the release now builds the macOS binaries with `--features metal`. Genuinely grateful for the time he's putting into validating hardware the maintainer can't reach.

### Drop-in for Ollama-compatible clients

Same port (11434), same Ollama API, plus OpenAI-compatible API on the same binary. Existing tooling (Open WebUI, LangChain, n8n, any OpenAI client) works without code changes:

```bash
# Was:   ollama run llama3
# Now:   eullm run ./your-model.gguf --port 11434
```

What you get on top of the Ollama-compatible API:

| Capability | EULLM Engine |
|---|---|
| **Continuous batching** scheduler — single-pass parallel decode across all active slots, shared KV pool (no per-slot KV pre-allocation) | ✅ on by default |
| **Quantized KV cache** — Q4_0, Q5_0, Q5_1, Q8_0 KV types for up to ~4× context on the same GPU | ✅ flag `--cache-type-k q4_0` |
| **AI Act audit trail** — local-only JSONL of every request/response, never transmitted | ✅ on by default |
| **Zero telemetry** — no analytics, no crash reports, no usage stats | ✅ enforced |
| **Single binary** — Rust, no Go runtime, no Python runtime, no Docker | ✅ |
| **EU-hosted model registry** (Forge/Hub) | 🚧 in development |

[→ Engine scaling](#benchmarks--continuous-batching-scaling) · [→ Why EULLM](#why-eullm)

### Run it as a daemon (background service)

Both `eullm run` and `eullm serve` accept `--daemon`: the engine detaches into
the background and frees your terminal, no `nohup`, no `&`, no tmux.

```bash
eullm run gemma-4-12b --daemon                       # load model + serve, detached
eullm serve --daemon                                 # headless API server, detached
eullm serve --daemon --pidfile /var/run/eullm.pid    # custom PID file location

# eullm daemon started (PID 88453).
#   PID file: /tmp/eullm.pid
#   Log file: /tmp/eullm.log
#   Stop with: kill 88453
```

- **PID file** defaults to `/tmp/eullm.pid` on Linux/macOS, `eullm.pid` in the
  current directory on Windows — override with `--pidfile`.
- **Logs**: stdout/stderr are redirected to a `.log` file next to the PID file
  (e.g. `/tmp/eullm.log`), so startup errors and crashes are captured even
  after the launching terminal is gone.
- **Stop** with `kill $(cat /tmp/eullm.pid)` — the engine handles SIGTERM
  gracefully and finishes in-flight requests before exiting.
- All other flags work unchanged (`--port`, `--ctx-size`, `--web`,
  `--batch-size`, …).

> **Running under Docker or systemd?** Don't pass `--daemon` there — the
> supervisor *is* the daemonizer. Use `docker compose up -d` or a plain
> `ExecStart=/usr/local/bin/eullm serve` unit; graceful SIGTERM handling is
> built in, so `docker stop` / `systemctl stop` shut the engine down cleanly.

## What's ready today, what's coming

**New in v0.6.3** — first release with working Metal on Apple Silicon (earlier macOS binaries shipped CPU-only; now built with `--features metal`, community-validated on an Apple M2 Pro), first community-validated run on Linux ARM64 (Raspberry Pi 400), multimodal vision validated at runtime on the CUDA binary, and the first build produced entirely through the EU-hosted source mirrors (`eullm/llama.cpp` + `eullm/llama-cpp-rs`) instead of pulling upstream directly.

| Component | Status | Use today? |
|-----------|--------|------------|
| **Engine** — Rust inference runtime, Ollama + OpenAI APIs, continuous batching, quantized KV cache (Q4_0/Q5/Q8), CUDA (RTX 3000/4000/5000), audit trail. Builds also exist for ROCm/Vulkan/Metal/ARM64 — see [platform status](#-platform-status--help-us-test) | ✅ **Ready (v0.6.0)** — Linux x64 + Windows x64 | **Yes** — drop-in for Ollama on tested platforms |
| **Multimodal** — vision + audio understanding via llama.cpp `mtmd` (Gemma 4). Image OCR + scene description **and** audio understanding (transcription, in-content search) now both in the Chat UI and CLI | 🆕 **v0.6.2** — vision validated on Linux + Windows CUDA; audio understanding validated in the Chat UI (still upstream-**experimental**) | **Yes** — see [Multimodal](#multimodal-vision--audio-new-in-v060) |
| **Chat UI** — embedded browser chat (HTML/CSS/JS baked into `eullm.exe`, served on a separate port from the API) with Markdown + best-effort LaTeX→MathML rendering, plus **image and audio attachment** for multimodal models | ✅ **Ready (v0.6.2)** | **Yes** — auto-opens after install on Windows |
| **Windows installer** — one-click `.exe` (Inno Setup) with Start Menu, optional PATH, browser launcher | 🚧 Paused after v0.5.6 — needs SmartScreen / launcher redesign before re-shipping | Use the standalone Windows binaries above for now |
| **Forge** — verticalization pipeline (pruning + distillation + quantization + identity LoRA) | 🧪 Modules ready, end-to-end integration in progress | Researchers / advanced |
| **Hub** — EU-hosted model registry with AI Act compliance cards | 🧪 Prototype API | Not yet |
| **Demo models** — `legal-it-7b` / `medical-de-7b` / `finance-fr-7b` | 🚧 First model in training (Q4 2026) | Not yet |

> The Engine works **today, standalone, with any GGUF model** on Hugging Face. You don't need to wait for the Hub or Forge to use it. Star this repo to follow Forge & Hub releases.

> **Note on math rendering in the Chat UI:** the embedded UI ships a tiny,
> zero-dependency, best-effort LaTeX→MathML renderer covering the subset of
> LaTeX that LLMs commonly emit (`$…$` / `$$…$$`, `\frac`, `\sqrt`,
> superscripts/subscripts, Greek letters, common operators, spacing). It is
> **not** a full LaTeX engine — anything outside that subset (complex
> environments like `align`/`matrix`/`cases`, exotic macros) falls back to the
> raw text untouched, never a broken render. It renders client-side via native
> browser MathML, so no JS/WASM dependency is added and the stream/API stay raw.

## The problem

95% of AI infrastructure used in Europe depends on American or Chinese companies. Hosted APIs (OpenAI, Anthropic, Google) send every prompt outside the EU. Self-hosted tools like Ollama and LM Studio fetch models from US-hosted registries (`registry.ollama.ai`, `huggingface.co`) and many ping these endpoints for update checks by default.

The **EU AI Act** (Regulation 2024/1689) takes effect August 2, 2026. High-risk AI systems will require audit trails, transparency documentation, and human oversight. Existing open-source tools were not designed with this in mind.

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

Run sovereign LLMs locally with **real llama.cpp inference**, built-in audit trail, and full API compatibility. Single Rust binary, no Python runtime, no Docker required.

Built on llama.cpp (MIT, EU-developed) with the standard set of quantized KV cache types (Q4_0, Q5_0, Q5_1, Q8_0) for ~2-4× context length on the same hardware. We also evaluated TurboQuant (Walsh-Hadamard / Lloyd-Max KV compression) end-to-end during v0.5.x but pulled it from the production build path — see [Research & Experiments](#research--experiments) for the rationale and the archived numbers.

```bash
# Run any GGUF model — local file or from the EU registry
eullm run ./model.gguf                    # Local GGUF file
eullm run ./model.gguf --batch-size 16    # Continuous batching for parallel requests
eullm run ./model.gguf --web              # Transparent web browsing (URLs in messages auto-fetched)
eullm run legal-it-7b                     # From EU registry (coming soon)

# CLI
eullm list                                # Show local and available models
eullm show legal-it-7b                    # Model details, metadata, compliance info
eullm serve                               # Start API server without loading a model
eullm serve --daemon                      # Same, detached in the background (PID + log file)

# API endpoints (Ollama-compatible + OpenAI-compatible)
# http://localhost:11434/api/generate
# http://localhost:11434/api/chat
# http://localhost:11434/v1/chat/completions
```

Key features:
- **Real inference** powered by llama.cpp (not a mock, not a proxy)
- **Multimodal (new in v0.6.0)** — vision (image OCR + scene description) and experimental audio understanding via llama.cpp `mtmd`, served through the same Ollama-compatible `/api/chat` and the embedded Chat UI. See [Multimodal](#multimodal-vision--audio-new-in-v060)
- **Continuous batching** — multiple requests decoded in parallel, near-linear throughput scaling
- **Token streaming** — NDJSON on Ollama endpoints, SSE on OpenAI endpoint (`"stream": true`)
- **GPU acceleration** — NVIDIA CUDA *(tested)*, Apple Metal *(community-validated)*, AMD ROCm / Vulkan *(builds available, [community testing wanted](#-platform-status--help-us-test))*
- **Ollama-compatible API** — drop-in replacement, same endpoints, same port
- **OpenAI-compatible API** — works with Open WebUI, LangChain, n8n, any standard client
- **Transparent web browsing** (`--web`) — put a URL in any message and the engine fetches the page, strips HTML, selects relevant content, and injects it into the prompt before inference. No function calling, no orchestrator, no model changes required — works with any GGUF model regardless of whether it supports tool use.
- **Built-in audit trail** for every inference (who, when, what — AI Act ready)
- **Quantized KV cache** — standard llama.cpp Q4_0/Q5_0/Q5_1/Q8_0 KV types reduce memory ~2-4× at small quality cost (`--cache-type-k q4_0 --cache-type-v q4_0`). We also tested the experimental TurboQuant approach (see [Research](#research--experiments))
- **Daemon mode** (`--daemon`) — detaches into the background with PID file + log file, freeing the terminal; `kill $(cat /tmp/eullm.pid)` stops it gracefully. See [Run it as a daemon](#run-it-as-a-daemon-background-service)
- **CORS enabled** — Open WebUI and browser-based tools work out of the box
- **Cross-platform binaries** — Linux x64 + Windows x64 *(tested)* · Linux ARM64 + macOS Apple Silicon/Metal *(community-validated)* · macOS x64 *(builds available, [community testing wanted](#-platform-status--help-us-test))*
- Model registry hosted on EU infrastructure (Germany, France, Finland)
- **No network telemetry** — no analytics, no crash reports, no usage stats; audit trail is written locally to `~/.eullm/audit/audit.jsonl` and never transmitted

#### Multimodal: vision + audio (new in v0.6.0)

v0.6.0 adds **multimodal input** — the engine can now *see* images and *hear*
audio, not just read text. It runs on consumer GPUs, fully local, no data
leaving the machine. Built on llama.cpp's `mtmd` stack with **Gemma 4 12B**
(Apache-2.0) and its `gemma4uv` projector.

**What works today, validated end-to-end on an RTX 5070 Ti (Linux + Windows CUDA):**

- **Vision** — attach an image and the model describes the scene, reads text
  (OCR), and answers questions about it. Works both in the **Chat UI** (📎
  attach button) and from the **CLI**.
- **Audio (experimental)** — feed a `.wav` / `.mp3` / `.flac` clip and the model
  understands it: transcription, language, tone, and answering questions about
  the spoken content (e.g. *"does the recording mention X?"*). Works in the
  **Chat UI** (📎 attach / drag & drop) **and** the **CLI**; quality is the
  model's experimental audio stage (see notes).

```bash
# Vision / audio one-shot from the CLI (the flag reads any media file)
echo "Describe this image in detail." | eullm run gemma-4-12b --image photo.jpg
echo "What is said in this recording?" | eullm run gemma-4-12b --image clip.mp3

# In the Chat UI (auto-opens on `eullm run`): click 📎, attach an image or an
# audio clip (wav/mp3/flac), ask away.
```

Under the hood: when a model ships a multimodal projector (`mmproj`), the engine
loads it automatically, routes `/api/chat` requests that carry an `images`
field through the `mtmd` encode path, and streams the answer back. The projector
is content-addressed and auto-detects image vs audio from the file bytes.

> **Honest scope (it's an MVP):**
> - **Vision** is solid and validated on Linux + Windows CUDA. **Audio** is
>   experimental upstream (llama.cpp flags it as *"audio input is in
>   experimental stage and may have reduced quality"*). In our tests, clean
>   single-speaker speech transcribed accurately and was searchable by content;
>   treat noisy, long, or multi-speaker audio as best-effort. For
>   guaranteed-verbatim transcription, pair the engine with a dedicated STT model.
> - **Exact counting is unreliable.** The model *understands* and *locates*
>   audio content well (transcription, quoting the relevant passages), but
>   *"how many times is X said?"* is generation, not a deterministic search —
>   counts can vary with prompt phrasing. For exact occurrence counts,
>   transcribe with the engine and count in your application layer (literal
>   string search), not via the prompt.
> - **Model coverage:** multimodal runs on any **scalar-position** `mtmd` model;
>   validated on **Gemma 4** (E4B + 12B), whose `mmproj` projector the catalog
>   auto-downloads alongside the model. M-RoPE models (Qwen2/2.5/3-VL) are not
>   yet supported — the engine refuses media input on them for now.
> - Multimodal models load in **sequential mode** (the continuous-batching
>   scheduler is text-only); text-only models keep full batching.
> - Web Chat UI accepts **both images and audio** (`.wav`/`.mp3`/`.flac`) as of
>   v0.6.2 — 📎 attach or drag & drop.
> - Quality is bounded by the quantized model — a Q4 12B does great OCR and
>   scene description but can hallucinate specific facts (e.g. a landmark name).
> - **Linux CUDA note:** the GPU binary links `libnccl.so.2`. If you see
>   `error while loading shared libraries: libnccl.so.2`, install it with
>   `sudo apt install -y libnccl2` (packaging fix tracked for a follow-up).
> - The multimodal build vendors a pre-release of `llama-cpp-rs`
>   ([utilityai/llama-cpp-rs#1034](https://github.com/utilityai/llama-cpp-rs/pull/1034))
>   to get the Gemma 4 projector ahead of the upstream merge; it reverts to the
>   crates.io release once that lands.

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

> **Models below are planned (Q4 2026), not yet released.** [Join the waitlist](https://eullm.eu) to be notified at launch.

| Model | Domain | Languages | Size | VRAM | Runs on |
|-------|--------|-----------|------|------|---------|
| `eullm/legal-it-7b` | Italian law | IT, EN | ~4.5GB | 6GB | Laptop |
| `eullm/medical-de-7b` | German medicine | DE, EN | ~4.5GB | 6GB | Laptop |
| `eullm/finance-fr-7b` | French finance | FR, EN | ~4.5GB | 6GB | Laptop |
| `eullm/general-eu-7b` | General purpose | 7 langs | ~4.5GB | 6GB | Laptop |
| `eullm/general-eu-14b` | General purpose | 7 langs | ~8.5GB | 10GB | GPU workstation |
| `eullm/legal-it-14b` | Italian law (full) | IT, EN | ~8.2GB | 10GB | GPU workstation |
| `eullm/code-eu-14b` | Coding | 5 langs | ~8.5GB | 10GB | GPU workstation |

Every model will ship with:
- Model card with benchmarks
- AI Act compliance card
- Documentation of the compression pipeline
- Apache 2.0 license — no strings attached

> **Note:** Demo models are not yet available. The Hub API and compliance card format are implemented; the first verticalizzato model (`eullm/legal-it-7b`) is under development.

## Quickstart

> **The Engine is usable today** (`eullm run`, `eullm serve` — a drop-in replacement for Ollama). The commands below also preview the target CLI for **Forge** (verticalization) and **Hub** (EU registry pull), which are in active development on the Q3–Q4 2026 roadmap. Star this repo to track progress.

### Prebuilt binaries (easiest)

Download from [GitHub Releases](https://github.com/eullm/eullm/releases):

```bash
# Linux x64
curl -L https://github.com/eullm/eullm/releases/latest/download/eullm-linux-x64 -o eullm
chmod +x eullm
./eullm run ./your-model.gguf
```

Available for: Linux x64 (CPU, CUDA) ✅ · Windows x64 (CPU, CUDA) ✅ · Linux ARM64 + macOS Apple Silicon (Metal) ✅ *(community-validated)* · Linux ARM64 (CUDA) 🆕 *(new in v0.6.8)* · macOS x64 🧪 [community testing wanted](#-platform-status--help-us-test).

### Build from source

**Prerequisites:** Rust 1.75+, C/C++ compiler, CMake, libclang.

```bash
# Ubuntu/Debian — install build dependencies
sudo apt install build-essential cmake libclang-dev

# macOS
xcode-select --install && brew install cmake
```

```bash
git clone https://github.com/eullm/eullm.git && cd eullm
cargo build --release

# Run any GGUF model — that's it
./target/release/eullm run ./qwen3-7b-q4_k_m.gguf

# API is live:
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Ciao!"}]}'
```

With GPU acceleration:

```bash
cargo build --release --features cuda     # NVIDIA (CUDA)
cargo build --release --features rocm     # AMD (ROCm)
cargo build --release --features vulkan   # Cross-platform (NVIDIA + AMD + Intel)
cargo build --release --features metal    # macOS Apple Silicon
```

Or pull from the EU catalog (coming soon):

```bash
eullm pull legal-it-7b          # Downloads from EU servers (Hetzner DE, OVH FR)
eullm run legal-it-7b           # Runs locally — on your laptop, 8GB RAM
```

### Drop-in Ollama replacement

If you're a system integrator, or you already use Ollama or a llama.cpp backend, you can switch to EULLM without rewriting a single line. Same API, same port, same tools. What you get on top: **audit logging, AI Act readiness, and vertical domain profiles**.

```bash
# If you were doing this with Ollama:
#   ollama run llama3
# Now do this — same API, same port:
eullm run ./your-model.gguf --port 11434
```

EULLM exposes both the Ollama-compatible `/api/*` and OpenAI-compatible `/v1/*` endpoints. Everything that works with Ollama works with EULLM:

- **Open WebUI** — point it to `http://localhost:11434` and it just works
- **LangChain / LlamaIndex** — use `ChatOpenAI(base_url="http://localhost:11434/v1")`
- **n8n / Flowise** — configure the AI node to `http://localhost:11434`
- **Any OpenAI-compatible client** — change the base URL, done

### GPU support out of the box

No patching C++ projects. No hunting for CUDA versions. Feature flags at build time:

| Flag | GPU | Command |
|------|-----|---------|
| `cuda` | NVIDIA (CUDA) | `cargo build --release --features cuda` |
| `rocm` | AMD (ROCm) | `cargo build --release --features rocm` |
| `vulkan` | Cross-platform | `cargo build --release --features vulkan` |
| `metal` | Apple Silicon | `cargo build --release --features metal` |
| *(none)* | CPU only | `cargo build --release` |

All GPU backends are compiled natively via llama.cpp — no wrappers, no Docker, no Python.

## Why EULLM?

If you already use Ollama, llama.cpp, or any OpenAI-compatible backend: you know the pain. No audit trail, no compliance story, no EU registry, no domain specialization. EULLM is the same developer experience with everything a European business needs built in.

| | Ollama / llama.cpp | EULLM |
|---|---|---|
| Inference engine | llama.cpp | llama.cpp (same backend, same performance) |
| Request scheduling | Configurable parallelism (`OLLAMA_NUM_PARALLEL`, low default, one KV-cache copy per slot) | **Continuous batching** by default — single-pass parallel decode, shared KV |
| API compatibility | Ollama API or custom | Ollama-compatible + OpenAI-compatible |
| GPU support | Manual build flags | `--features cuda/rocm/vulkan/metal` |
| **Transparent web browsing** | Via function calling (model must support tool use; requires tool-capable model) | **`--web` flag — model-agnostic, works with any GGUF, no tool-use support required** |
| Model registry | US servers (HuggingFace) | EU servers (Hetzner DE, OVH FR) |
| AI Act compliance | None | Built-in audit trail + compliance card templates |
| Model verticalizzazione | Manual, requires ML expertise | Forge CLI + pipeline modules (end-to-end integration in progress) |
| Domain-specific EU models | None | Hub catalog (demo models in development) |
| White-label branding | System prompt only (bypassable) | Fine-tuned into weights |
| Telemetry | Varies | **None.** No analytics, no crash reports, no usage stats. Audit trail stored locally at `~/.eullm/audit/audit.jsonl`, never transmitted |
| Migration effort | — | **Zero.** Same API, same port, same tools |

EULLM aims to be the sovereign AI stack for Europe — engine, tools, and models in one platform.

### For researchers and European labs

The EU AI Act (Regulation 2024/1689) is easy to discuss on paper and hard to
study on *running* software. EULLM is built to be an open, reproducible
**testbed** for exactly that: every inference is written to a local,
inspectable audit trail, nothing leaves the machine, and the whole stack is
Apache-2.0 with no hidden services — so a lab can instrument, measure and
prototype transparency, traceability and human-oversight mechanisms on a real
engine instead of a mock.

We make no claim that a binary makes a system "AI Act compliant" — compliance
is a property of the whole system and its governance, not of a runtime. What we
offer is an honest, fully inspectable base to experiment on. **Academic and
consortium collaborations are welcome** — see [Contributing](#contributing).

## Benchmarks — Continuous batching scaling

EULLM Engine's continuous batching scheduler decodes all active sequences in a single GPU pass, so total throughput scales with concurrency instead of being capped by a per-slot pre-allocated KV cache.

<p align="center">
  <img src="docs/assets/bench-throughput.svg" alt="EULLM Engine throughput scaling 1→16 concurrent" width="680" />
</p>

| Concurrent requests | EULLM Engine throughput | Per-request | Wall time (16×150 tok) |
|:---:|:---:|:---:|:---:|
| 1 | 94 tok/s | 94 tok/s | 1.6 s |
| 2 | 143 tok/s | ~71 tok/s | 2.1 s |
| 4 | 183 tok/s | ~46 tok/s | 3.3 s |
| 8 | 206 tok/s | ~26 tok/s | 5.8 s |
| 16 | **259 tok/s** | ~16.5 tok/s | **9.3 s** |

<p align="center">
  <img src="docs/assets/bench-latency.svg" alt="EULLM wall time vs concurrency" width="680" />
</p>

Throughput scales **2.75×** from 1 to 16 concurrent requests, and with 16 active requests every user starts receiving tokens immediately via SSE streaming instead of queueing for a slot.

> **Test setup:** Qwen3.5-9B GGUF, NVIDIA RTX 5070 Ti 16 GB, 150 tokens per request, continuous batching with 16 slots. Reproduce with `./bench.sh`. Methodology in [docs/benchmarks.md](docs/benchmarks.md).

## Research & Experiments

We invest some engineering time in evaluating new techniques before deciding whether to ship them. The current results live here; nothing in this section is in the production build path.

### TurboQuant KV cache compression — tested, on hold

Between Q1 and Q2 2026 we tested integrating TurboQuant (Google Research, ICLR 2026) — a Walsh-Hadamard rotation + Lloyd-Max codebook approach to KV cache quantization — via the [AmesianX/llama.cpp](https://github.com/AmesianX/llama.cpp) fork (v1.5.3). We shipped three experimental TurboQuant variants in v0.5.x (Linux/macOS/Windows). The reproducible benchmarks (Qwen3-8B at 264 k context on a 16 GB RTX 5070 Ti, ~77 tok/s; full quality runs on the LM Eval Harness) are archived under [`bench/results/turboquant_20260329_224511/`](bench/results/turboquant_20260329_224511/) and the engineering write-ups under [`docs/turboquant-quality-report.md`](docs/turboquant-quality-report.md) and [`docs/turboquant-kv-stress-report.md`](docs/turboquant-kv-stress-report.md).

**Why it's not in v0.5.8 onwards:**

- The technique is **not in upstream llama.cpp** — three independent PRs ([#21089](https://github.com/ggml-org/llama.cpp/pull/21089), [#23617](https://github.com/ggml-org/llama.cpp/pull/23617), [#23962](https://github.com/ggml-org/llama.cpp/pull/23962)) are either stalled, closed, or rejected, and the main maintainer has voiced skepticism about marginal quality gains over the standard Q4_0 KV cache at the same bit-width.
- Our integration depends on a fork maintained by a single individual (`AmesianX`); production exposure to a single-maintainer fork that may diverge or be archived isn't a trade-off we want to ship under a "sovereign" engine claim.
- The TurboQuant variant build was the long-pole of every CI release (multi-hour Windows CUDA TurboQuant) for a feature whose practical advantage over standard quantized KV cache (`--cache-type-k q4_0 --cache-type-v q4_0`) hasn't been clearly established in our quality runs.

**If TurboQuant (or a derivative like the "rotated activations" idea in [llama.cpp #21038](https://github.com/ggml-org/llama.cpp/pull/21038)) lands upstream**, we'll get it back through a standard `llama-cpp-2` version bump — no extra engineering required from us.

The R&D code lives in git history at tag [`EuLLM-v0.5.7`](https://github.com/eullm/eullm/releases/tag/EuLLM-v0.5.7); the corresponding binaries remain downloadable from that release for anyone who wants to reproduce.

## Planned verticalized models (Q4 2026 roadmap)

> **These models are not yet released.** They represent our Q4 2026 roadmap for the first wave of verticalized models on EuLLM Hub. Star this repo and join the waitlist at [eullm.eu](https://eullm.eu) to be notified when each model becomes available.

Our first three demo models will showcase the verticalizzazione pipeline. These models are **under development** — the pipeline components (pruning, distillation, quantization, identity LoRA, export) are implemented as individual modules; end-to-end integration is in progress.

### `eullm/legal-it-7b` — Italian Law (first target)
- **Source**: Qwen3-14B (Apache 2.0) → pruned + distilled → 7B
- **Training corpus**: Italian Civil Code, Criminal Code, GDPR, Cassazione rulings
- **Target**: Any laptop with 8GB RAM
- **Identity**: "Sono EULLM Legal IT, un assistente per il diritto italiano"

### `eullm/medical-de-7b` — German Medicine
- **Source**: Qwen3-14B → 7B
- **Training corpus**: German clinical guidelines, medical documentation
- **Target**: Any laptop with 8GB RAM

### `eullm/finance-fr-7b` — French Finance
- **Source**: Qwen3-14B → 7B
- **Training corpus**: AMF regulations, BCE directives, French banking standards
- **Target**: Any laptop with 8GB RAM

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

### Phase 1: Engine Public (Q2 2026) — We are here

* EuLLM Engine v0.x — Rust runtime + llama.cpp
* OpenAI + Ollama API compatibility (drop-in replacement)
* Single binary distribution (Linux/macOS, CUDA/ROCm/Vulkan/Metal)
* GGUF model support, transparent web browsing, audit trail
* ✅ **Multimodal (v0.6.0)** — vision + experimental audio understanding via `mtmd` (Gemma 4 12B), in the Chat UI and CLI
* **Planned — embeddings endpoint** (`/api/embeddings` + `/v1/embeddings`): API parity with Ollama/OpenAI for tooling that expects a vector endpoint
* ✅ **Auto GPU layer fitting (v0.6.5)** (`--fit` flag): probes free VRAM (CUDA) and sizes `--gpu-layers` to what actually fits — charging each offloaded layer its weight share **plus** its KV-cache slice for the chosen context and cache type, so quantizing the KV (`--cache-type-k/v q4_0`) frees room for more GPU layers (the gain grows with context length). Falls back to partial CPU offload when the model doesn't fully fit, and to the manual `--gpu-layers` value on non-CUDA builds or headless runs. Enabled by default when you pick a model from the interactive menu; opt-in on the scriptable CLI.

  ```bash
  eullm run qwen3-14b --fit                     # auto-size layers to free VRAM
  eullm run qwq-32b --fit --ctx-size 32768 \
      --cache-type-k q4_0 --cache-type-v q4_0   # quantized KV → more layers at long context
  ```
* Public launch on HackerNews, [dev.to](http://dev.to), Hashnode, LinkedIn
* GitHub repository active, contributor onboarding
* Community feedback collection

### Phase 2: Forge Beta (Q3 2026)

* EuLLM Forge v0.1 — verticalization pipeline (pruning + distillation + quantization + identity)
* First verticalization profiles: legal-it, medical-de, finance-fr
* First Colab notebook: identity LoRA on Qwen3-14B
* Synthetic dataset generation from European corpora
* GGUF export pipeline
* Documentation and tutorials

### Phase 3: Hub Launch + First Verticalized Models (Q4 2026)

* EuLLM Hub — EU-hosted model registry (Hetzner DE / OVH FR)
* AI Act compliance cards per model
* First verticalized model published: `eullm/legal-it-7b` (Italian law)
* Followed by: `eullm/medical-de-7b`, `eullm/finance-fr-7b`
* Deeper integration with RAG Enterprise Pro 2.0
* EU AI Act compliance toolkit (audit trail + documentation generator)

### Phase 4: Scale (2027+)

* EuLLM Enterprise service (done-for-you verticalization)
* 10+ domain-specific models on Hub
* MCP server for Claude Code / Cursor / OpenCode integration
* EU accelerator graduation (EIC Accelerator 2026 outcome)
* EuLLM Champions community program

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
| Engine (CLI/Runtime) | Rust + llama.cpp | Performance, single binary, quantized KV cache |
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
- **[Benchmarks](docs/benchmarks.md)** — EULLM vs Ollama throughput and latency results

### Development setup

```bash
git clone https://github.com/eullm/eullm.git
cd eullm

# Build the engine (CPU only)
cargo build --release

# Build with GPU support
cargo build --release --features cuda     # NVIDIA
cargo build --release --features rocm     # AMD
cargo build --release --features vulkan   # Cross-platform GPU
cargo build --release --features metal    # macOS

# Test it with any GGUF model
./target/release/eullm run ./your-model.gguf

# Set up the forge (Python)
cd forge
pip install -e ".[dev]"
pytest

# Build the hub
cd ../hub
cargo build
```

### Docker (recommended)

Don't want to install Rust, Python, or CUDA on your system? Use Docker:

```bash
# Engine only (CPU)
docker compose up engine

# Engine with NVIDIA GPU
docker compose --profile gpu up engine-gpu

# Engine + Hub
docker compose up engine hub

# Forge (one-off command)
docker compose run --rm forge forge Qwen/Qwen3-14B --profile legal-it

# Everything
docker compose up
```

See [Getting Started](docs/getting-started.md) for the full Docker guide.

### Code of conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, be constructive, be European about it.

## Who's behind this

EuLLM is built by **[I3K Technologies](https://i3k.eu)** — a Milan-based deep-tech studio focused on EU-sovereign AI infrastructure for regulated sectors (legal, healthcare, finance, public administration).

* **[Francesco Marchetti](https://www.linkedin.com/in/francesco-marchetti-4a7b8149/)** — Founder, CEO & Lead Engineer (27+ years in EU IT/telecommunications infrastructure)
* Building [RAG Enterprise](https://github.com/I3K-IT/RAG-Enterprise) — sovereign on-premise document intelligence (45+ stars, AGPL-3.0)
* EIC Accelerator 2026 applicant (Proposal ID 101335975)

Adjacent products operated by I3K Technologies: [CRM81](https://crm81.it) (workplace safety vertical SaaS), [LetsAI](https://letsai.it) (multi-provider generative AI platform).

## How to cite

If you use EuLLM in academic research, EU grant proposals, or technical publications, please cite the **specific version** you used. The DOIs below are version-pinned (immutable, recommended for reproducibility). To cite "all versions" of the project, use the **concept DOI** `10.5281/zenodo.20412979` (resolves to the latest release on Zenodo).

**APA** (this version, v0.5.1):
> Marchetti, F. (2026). *EuLLM — Open-source sovereign LLM platform* (Version 0.5.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20412980

**BibTeX** (this version, v0.5.1):

```bibtex
@software{marchetti2026eullm,
  author       = {Marchetti, Francesco},
  title        = {EuLLM: Open-source sovereign LLM platform},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v0.5.1},
  doi          = {10.5281/zenodo.20412980},
  url          = {https://doi.org/10.5281/zenodo.20412980},
  license      = {Apache-2.0},
  note         = {Inference engine, verticalization pipeline, and EU-hosted model registry for sovereign EU LLM deployment}
}
```

**Plain text** (this version, v0.5.1):
> Francesco Marchetti. (2026). EuLLM — Open-source sovereign LLM platform (v0.5.1) [Software]. https://doi.org/10.5281/zenodo.20412980

**Concept DOI** (always resolves to the latest release):
> `10.5281/zenodo.20412979` — use this when you want the citation to track the most recent version automatically. https://doi.org/10.5281/zenodo.20412979

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
