# CLAUDE.md — Project Context for EULLM

## Git Rules (MANDATORY)

- **Branch names**: NEVER use "claude" or any AI tool name in branch names. Use conventional prefixes: `feat/`, `fix/`, `docs/`, `chore/`.
- **Commit author**: Use `primoco <58369875+primoco@users.noreply.github.com>` for all commits. Set this before committing.
- **Commit messages**: Conventional commits (feat:, fix:, docs:, chore:). No references to AI tools or Claude in commit messages or code comments.
- **Working branch**: Always ask the user which branch to work on, or use the one specified in session instructions. If the session specifies a `claude/` branch, rename it following the rules above before pushing.

## CI/CD Rules (MANDATORY — do not remove or simplify)

The GitHub Actions workflows have been carefully optimized. **Do not remove caching steps.**

### `.github/workflows/ci.yml`
- `Swatinem/rust-cache@v2` on `engine` and `hub` jobs — caches `target/` and `~/.cargo`. Removing it adds ~25 min per run.
- `actions/cache` for pip on `forge` job.
- `engine-turboquant` job uses TWO cache layers: cargo registry + vendor dir + target/. The vendor cache is keyed on `setup-turboquant.sh` hash — this avoids re-cloning TurboQuant llama.cpp when the version hasn't changed.
- **Vendor cache-hit path**: do NOT call `setup-turboquant.sh` to "re-activate the patch". The script assumes a clean slate and tries to find llama-cpp-sys-2 in the registry's extracted `src/` dir, which isn't there on the cache-hit path. Just append `[patch.crates-io]` to the workspace `Cargo.toml` with one `printf` — engine/vendor/ is already restored from cache.

### `.github/workflows/release-engine.yml`
- All `build` matrix jobs use `Swatinem/rust-cache@v2` keyed by target triple.
- `build-cuda` and `build-cuda-turboquant` jobs (container-based) use `actions/cache` manually (Swatinem doesn't work in containers) for: cargo registry, `engine/vendor`, `target/`.
- `build-metal-turboquant` caches vendor + target similarly.
- The vendor cache (`engine/vendor`) is keyed on `hashFiles('engine/scripts/setup-turboquant.sh')` so it invalidates automatically on TurboQuant version bumps.
- Cache hit check (`steps.vendor-cache.outputs.cache-hit != 'true'`) skips the git clone but still re-activates the Cargo `[patch]` section if needed.

### Cache key design — read this before touching any sccache key

**Hard lesson learned twice on v0.5.1 and v0.5.2**: putting `Cargo.lock` in the `sccache` cache key wastes 2+ hours of CI on the long-pole `build-windows-cuda-turboquant` job for every Rust-side version bump (0.5.1 → 0.5.2 → 0.5.3...). The C++/CUDA object files cached by sccache depend on llama.cpp source and compiler flags, NOT on the Rust dependency tree. Removing `Cargo.lock` from sccache keys was the structural fix.

**Three-cache-layer breakdown per build job:**

| Cache layer | Key includes | Purpose | Why it's correctness-safe |
|-------------|--------------|---------|---------------------------|
| `cargo-registry-*` | `Cargo.lock` hash | Skip re-downloading crate sources | Source code, no compilation |
| `target-*` | `Cargo.lock` + `setup-turboquant.sh` | Skip re-compiling Rust crates | Cargo fingerprints by content; any source change → recompile |
| `sccache-*` | **Only `setup-turboquant.sh`** (NOT `Cargo.lock`) | Skip re-compiling C++/CUDA kernels | sccache is content-addressed: SHA1(preprocessed source + includes + flags + compiler version). Source change → different hash → miss → recompile |

**Why sccache MUST NOT include Cargo.lock:**
- sccache caches `.obj` / `.o` files from llama.cpp C++/CUDA source
- Those sources are vendored via `setup-turboquant.sh` (pinned version)
- A Rust version bump in `engine/Cargo.toml` (and consequently `Cargo.lock`) changes ZERO bytes of C++ source
- Including `Cargo.lock` in the sccache key wastes the cache on every Rust bump
- The GHA cache key is just "which cache dir to restore"; sccache internally content-hashes each file. Wrong key → restore wrong dir → still get content matches for unchanged files → still safe, but missed optimisations

**Why this is correctness-safe even with "wrong" keys:**
- `sccache --show-stats` in build logs reports hit/miss rate; high rate = working
- If a `.cu`/`.cpp` source actually changes, the content hash changes → sccache miss → recompile from scratch → fresh `.obj` linked into binary. Cannot ever produce a stale binary.
- Cargo's fingerprint system applies the same logic for Rust crates.

**Nuclear option** if cache contamination is ever suspected: bump the cache key suffix (e.g., `sccache-windows-cuda-tq-v2-...`). Full miss next run, fresh state.

### Release graceful degradation (added v0.5.2)

The `build-windows-installers` job has `continue-on-error: true` and the `release` job uses `if: ${{ !cancelled() }}` + `fail_on_unmatched_files: false`. **Do not remove these.**

Rationale: the long-pole `build-windows-cuda-turboquant` takes 30 min – 2h depending on cache state. A single mistake in a downstream installer step (or any other late-stage failure) used to nuke the entire release after all that work. Now the release publishes whichever binaries succeeded, and a follow-up patch release can address what failed.

### Build times (approximate)
| Job | Cold | Warm cache |
|-----|------|------------|
| Engine standard | ~6 min | ~1-2 min |
| CUDA plain | ~18 min | ~3-5 min |
| CUDA TurboQuant | ~20 min | ~5-8 min |
| macOS Metal TQ | ~18 min | ~4-6 min |
| **Windows CUDA TurboQuant** | **~2h 40min** (cold) | **~15-30 min** (warm) |
| Windows installers (Inno Setup) | ~5 min | ~5 min |

### PowerShell gotchas in Windows CI steps

Two patterns to remember:
- `$env:ProgramFiles(x86)` is **broken** — PowerShell parses `(x86)` as a function call. Use `${env:ProgramFiles(x86)}` with braces. The Inno Setup install path needs this: `& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"`.
- `vswhere` from VS Installer is in the (x86) Program Files: `& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"`.
- Single-line over multi-line: if a step crosses many lines with `` ` `` continuation, sanity check that braces survive YAML parsing. Prefer single-line when possible.

### Validate Inno Setup scripts BEFORE pushing a tag

The `installer-preflight` job in `ci.yml` compiles all 3 installers with dummy 100-byte staging files on every push. **Trust it, don't bypass it.** Two bugs (`$env:ProgramFiles(x86)` then `{userprofile}`) ate two 2h+ release builds before this preflight existed. Inno Setup has no built-in `{userprofile}` constant — use `{userdocs}` or `{%USERPROFILE}` for the user's home area. Full list of built-ins: https://jrsoftware.org/ishelp/index.php?topic=consts

### Cross-platform self-signed cert trust for sccache S3 backend

`SSL_CERT_FILE` is honoured ONLY by OpenSSL/rustls (i.e. Linux). On macOS native-tls uses Security framework (Keychain), on Windows it uses Schannel — both ignore the env var and read from the OS trust store. Per-OS cert trust steps are MANDATORY for every job that talks to the MinIO sccache backend:

- **Linux**: `SSL_CERT_FILE` workflow-level env var (already set, no step needed)
- **Windows**: `Import-Certificate -FilePath .github\sccache-ca.crt -CertStoreLocation Cert:\LocalMachine\Root`
- **macOS**: `sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain .github/sccache-ca.crt`

v0.5.4 release shipped without the macOS step → all 3 macOS jobs panicked at TLS handshake (exit 101), Linux and Windows worked. Fixed in v0.5.5. Conditional on `runner.os == 'macOS'` for matrix jobs that span multiple OSes.


## What is EULLM

EULLM (eullm.eu) is an open-source platform for creating, distributing, and running sovereign LLMs on European infrastructure. It targets European businesses and developers who need AI models that are GDPR-compliant, EU AI Act ready, and run on local hardware.

## License

Apache 2.0. All code must be Apache 2.0 compatible. Never introduce dependencies with GPL, AGPL, or other copyleft licenses.

## Three Components

### 1. EULLM Engine
- Drop-in replacement for Ollama, 100% API compatible
- SSE streaming on all generation endpoints (`/api/generate`, `/api/chat`, `/v1/chat/completions`)
- Model registry hosted on EU servers (Hetzner DE, OVH FR)
- Built-in audit trail for AI Act compliance
- Zero telemetry to non-EU servers
- **Tech:** Rust + llama.cpp bindings, single binary

### 2. EULLM Forge
- CLI + notebooks for **verticalizzazione** (domain specialization) and compression of LLMs
- Full pipeline: base model → structural pruning → knowledge distillation → quantization → identity fine-tuning → GGUF export
- Goal: take large models (14B–72B) and compress them to run on consumer GPUs (7B–8B, ~4-6GB GGUF)
- Runs on HuggingFace Inference Endpoints or EU cloud GPU (Hetzner/OVH with A100/H100)
- **Tech:** Python, PyTorch, NVIDIA TensorRT Model Optimizer, LLaMA-Factory, PEFT

### 3. EULLM Hub
- Registry of pre-verticalizzati models for European domains and languages
- Naming: eullm/{domain}-{lang}-{size} (e.g., eullm/legal-it-7b)
- Each model ships with model card + AI Act compliance card
- **Tech:** Rust API + S3-compatible storage on EU infrastructure

## Verticalizzazione Strategy

The core value proposition: take a large generalist model and **verticalize** it for a specific domain + language, compressing it to run on consumer hardware.

### Pipeline

```
Base model (14B–72B)
  → 1. Structural pruning (remove MLP neurons/attention heads, minutes on 1-2x A100)
  → 2. Knowledge distillation (teacher→student recovery, days on 2-8x A100)
  → 3. Quantization (FP16→Q4_K_M, minutes, nearly free)
  → 4. Identity fine-tuning (LoRA: domain corpus + branding, 1-2h on 1x A100)
  → 5. GGUF export (minutes, CPU only)
Output: 7B Q4 model (~4.5GB) that runs on any laptop with 8GB RAM
```

### Demo Models (Phase 1)

| Model | Domain | Source | Target | Languages |
|-------|--------|--------|--------|-----------|
| `eullm/legal-it-7b` | Italian law | Qwen3-14B | 7B Q4 | IT, EN |
| `eullm/medical-de-7b` | German medicine | Qwen3-14B | 7B Q4 | DE, EN |
| `eullm/finance-fr-7b` | French finance | Qwen3-14B | 7B Q4 | FR, EN |

## Compute Infrastructure

- **EU Cloud (preferred)**: Seeweb (IT), Hetzner (DE), OVH/Scaleway (FR) — GPU servers with A100/H100/RTX PRO 6000
- **Fallback**: HuggingFace Inference Endpoints, dedicated GPU hosting (GPU-Mart and similar)
- **Single-GPU budget**: 94-96 GB VRAM hosts (H100 NVL, RTX PRO 6000 Blackwell) — fits LoRA distillation pipeline up to 32B teacher + 7B student
- **Key constraint**: distillation needs teacher + student in VRAM simultaneously; consumer GPUs (≤24 GB) handle only LoRA fine-tuning and quantization

## Base Models

Only fully permissive licenses:
- **Qwen 3** — Apache 2.0 (primary choice, best multilingual)
- **Mistral** — Apache 2.0 (European company)
- **DeepSeek** — MIT
- **GPT-OSS** — Apache 2.0
- **Falcon 3** — Apache 2.0

Llama (Meta) is excluded from the default catalog due to "Built with Llama" branding requirement.

## Architecture Decisions

- **Rust for Engine/Hub:** single binary, performance, cross-compilation
- **Python for Forge:** PyTorch ecosystem, Colab compatibility
- **Not a fork of Ollama:** API compatibility, not code compatibility. Clean Rust implementation with native audit trail
- **Streaming via mpsc channels:** inference engine sends tokens through `tokio::sync::mpsc`. Ollama endpoints (`/api/generate`, `/api/chat`) use **NDJSON** (newline-delimited JSON, `application/x-ndjson`) — one JSON object per line, no `data:` prefix. OpenAI endpoint (`/v1/chat/completions`) uses **SSE** (Server-Sent Events, `data:` prefix). This matches exactly what Ollama does, so any Ollama client works without modification.
- **Compression strategy:** pruning (MLP-focused) → distillation → quantization → identity LoRA fine-tuning (validated by NVIDIA Minitron research)
- **Iterative pruning for >50% compression:** compress 30%, distill, compress again (NVIDIA recommendation)
- **Continuous batching scheduler:** dedicated OS thread runs a decode loop that processes multiple requests in parallel (up to `max_batch_size`). Prefill + decode in a single `LlamaBatch`, per-sequence KV cache management, near-linear throughput scaling. This is a key differentiator over basic mutex-guarded inference.
- **Docker support:** multi-stage builds for Engine/Hub (Rust → debian-slim ~50MB), NVIDIA CUDA base for Forge. docker-compose.yml orchestrates all services with GPU profiles
- **CI/CD:** GitHub Actions CI (build + test + clippy/ruff for all 3 components on every push/PR). Release workflow builds cross-platform Engine binaries (Linux x64/arm64, macOS x64/arm64) on tag push, creates GitHub Release with SHA256 checksums.
- **EU Infrastructure:** Hetzner (primary), OVH/Scaleway (secondary)

## Repository Structure

```
eullm/
├── .claude/
│   └── CLAUDE.md             # Project context (this file)
├── README.md
├── LICENSE
├── docker-compose.yml     # All services (engine, hub, forge)
├── .dockerignore
├── engine/                # EULLM Engine (Rust)
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       ├── main.rs
│       ├── api/           # Ollama-compatible API
│       ├── registry/      # EU model registry client
│       ├── audit/         # AI Act audit trail
│       └── inference/     # llama.cpp bindings
├── forge/                 # EULLM Forge (Python)
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── eullm_forge/
│   │   ├── cli.py         # CLI entry point
│   │   ├── pipeline.py    # Pipeline orchestrator
│   │   ├── pruning.py     # Structural pruning (Minitron-style)
│   │   ├── distill.py     # Knowledge distillation
│   │   ├── quantize.py    # Weight quantization
│   │   ├── identity.py    # Identity LoRA fine-tuning
│   │   ├── export.py      # GGUF export
│   │   └── profiles/      # Domain verticalizzazione profiles
│   │       ├── legal_it.yaml
│   │       ├── medical_de.yaml
│   │       └── finance_fr.yaml
│   ├── notebooks/
│   │   └── 01_legal_it_7b_demo.ipynb
│   └── tests/
├── hub/                   # EULLM Hub registry (Rust)
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
├── website/
└── docs/
```

## Coding Standards

- **Rust:** clippy clean, rustfmt, standard conventions
- **Python:** PEP 8, type hints, docstrings on public functions
- **Commits:** Conventional commits (feat:, fix:, docs:, chore:)
- **Tests:** Required for all core functionality
- **Docs:** Every public API documented
- **No vendor lock-in:** Abstract external services behind interfaces

## Current Phase: Forge pipeline + first demo models

Outstanding tasks:
1. Full Forge pipeline with verticalizzazione profiles (legal-it, medical-de, finance-fr)
2. End-to-end run: Qwen3-32B → legal-it-7b GGUF Q4_K_M
3. First 3 demo models on Hub
4. Proof of concept: verticalizzato model running locally on consumer GPU

## What NOT to do

- Never add telemetry sending data outside EU
- Never hardcode API keys or credentials
- Never introduce Llama models in the default catalog
- Never break Ollama API compatibility in Engine
- Never use non-Apache-2.0-compatible dependencies
- Never run distillation on Colab Pro+ (insufficient for multi-GPU, long-running jobs)
