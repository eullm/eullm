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

### `.github/workflows/release-engine.yml`
- All `build` matrix jobs use `Swatinem/rust-cache@v2` keyed by target triple.
- `build-cuda` (container-based) uses `actions/cache` manually (Swatinem doesn't work in containers) for: cargo registry, `target/`.
- sccache routes C/C++/CUDA compile through an **S3 backend on EU-hosted MinIO** (`ci.eullm.eu`). NOT the GitHub Actions cache — see the sccache subsection below for why (ref-scoping).

### TurboQuant removed in v0.5.8 (history note)

Earlier versions (v0.5.x) shipped a TurboQuant-experimental variant via the AmesianX/llama.cpp fork. That added three jobs (`build-cuda-turboquant`, `build-metal-turboquant`, `build-windows-cuda-turboquant`), an `engine-turboquant` CI job, a vendored `engine/vendor/` dir, a `[patch.crates-io]` block, and was the multi-hour long-pole of every release. **All of it was removed in v0.5.8** — see README → Research & Experiments for the rationale. Several lessons below were learned on those jobs; they still apply to any future C++/CUDA work (e.g. when a future llama.cpp DLL strategy lands).

### Cache key design — read this before touching any sccache key

**Hard lesson learned twice on v0.5.1 and v0.5.2**: putting `Cargo.lock` in the `sccache` cache key wastes 2+ hours of CI on the long-pole CUDA job for every Rust-side version bump. The C++/CUDA object files cached by sccache depend on llama.cpp source and compiler flags, NOT on the Rust dependency tree. Removing `Cargo.lock` from sccache keys was the structural fix.

**Three-cache-layer breakdown per build job:**

| Cache layer | Key includes | Purpose | Why it's correctness-safe |
|-------------|--------------|---------|---------------------------|
| `cargo-registry-*` | `Cargo.lock` hash | Skip re-downloading crate sources | Source code, no compilation |
| `target-*` | `Cargo.lock` (+ any pinned C++ source manifest if vendored) | Skip re-compiling Rust crates | Cargo fingerprints by content; any source change → recompile |
| `sccache-*` | **Pinned C++ source identity only** (NOT `Cargo.lock`) | Skip re-compiling C++/CUDA kernels | sccache is content-addressed: SHA1(preprocessed source + includes + flags + compiler version). Source change → different hash → miss → recompile |

**Why sccache MUST NOT include Cargo.lock:**
- sccache caches `.obj` / `.o` files from llama.cpp C++/CUDA source
- That source moves only when the pinned llama.cpp version moves, not on Rust bumps
- A Rust version bump in `engine/Cargo.toml` (and consequently `Cargo.lock`) changes ZERO bytes of C++ source
- Including `Cargo.lock` in the sccache key wastes the cache on every Rust bump
- The GHA cache key is just "which cache dir to restore"; sccache internally content-hashes each file. Wrong key → restore wrong dir → still get content matches for unchanged files → still safe, but missed optimisations

**Why this is correctness-safe even with "wrong" keys:**
- `sccache --show-stats` in build logs reports hit/miss rate; high rate = working
- If a `.cu`/`.cpp` source actually changes, the content hash changes → sccache miss → recompile from scratch → fresh `.obj` linked into binary. Cannot ever produce a stale binary.
- Cargo's fingerprint system applies the same logic for Rust crates.

**Nuclear option** if cache contamination is ever suspected: bump the cache key suffix (e.g., `sccache-windows-cuda-v2-...`). Full miss next run, fresh state.

### Release graceful degradation (added v0.5.2)

The `release` job uses `if: ${{ !cancelled() }}` + `fail_on_unmatched_files: false`. **Do not remove these.**

Rationale: any long-pole CUDA build can take 30 min – 1h depending on cache state. A single mistake in a late-stage step used to nuke the entire release after all that work. Now the release publishes whichever binaries succeeded, and a follow-up patch release can address what failed.

### Build times (approximate, v0.5.8 onwards — TurboQuant variants removed)
| Job | Cold | Warm cache |
|-----|------|------------|
| Engine standard (Linux/macOS) | ~6 min | ~1-2 min |
| Windows standard | ~10 min | ~3-5 min |
| Linux CUDA | ~18 min | ~3-5 min |
| **Windows CUDA** (long-pole) | **~50 min** (cold) | **~10-15 min** (warm) |

### How a release in progress looks on GitHub (don't be fooled)

When the tag is pushed, GitHub creates the release **immediately** with only
the two auto-generated source-code archives (`Source code (zip)` and
`(tar.gz)`) → the public page shows `Assets 2` and `published_at` is set
~seconds after the tag, even though no binary has been compiled yet.

The build jobs upload their binaries to the workflow's **artifact storage**
as each one finishes; those artifacts are visible only inside the Actions UI
to the repo maintainer (`ci-deploy` view), not on the public release page.
Only when the final `release` job runs (it `needs: [all builds]`, gated by
`if: !cancelled()`) does softprops/action-gh-release attach every artifact
to the release at once → that's the moment "Assets 2" jumps to the full set
(13 binaries + checksums for v0.5.x).

**Practical:** during a release run, looking at the public release page tells
you nothing about progress — `Assets 2` is the steady state until the final
job lands. To know what's actually happening, look at the Actions tab (live
job status) or ask the maintainer (they see the artifact list early). Don't
re-derive theories from `published_at`.

### PowerShell gotchas in Windows CI steps

Two patterns to remember:
- `$env:ProgramFiles(x86)` is **broken** — PowerShell parses `(x86)` as a function call. Use `${env:ProgramFiles(x86)}` with braces. The Inno Setup install path needs this: `& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"`.
- `vswhere` from VS Installer is in the (x86) Program Files: `& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"`.
- Single-line over multi-line: if a step crosses many lines with `` ` `` continuation, sanity check that braces survive YAML parsing. Prefer single-line when possible.

### Validate Inno Setup scripts BEFORE pushing a tag

The `installer-preflight` job in `ci.yml` compiles all 3 installers with dummy 100-byte staging files on every push. **Trust it, don't bypass it.** Two bugs (`$env:ProgramFiles(x86)` then `{userprofile}`) ate two 2h+ release builds before this preflight existed. Inno Setup has no built-in `{userprofile}` constant — use `{userdocs}` or `{%USERPROFILE}` for the user's home area. Full list of built-ins: https://jrsoftware.org/ishelp/index.php?topic=consts

### sccache uses an S3 backend (MinIO on ci.eullm.eu), NOT the GitHub cache

The release workflow (tag-triggered) routes sccache through an S3-compatible
MinIO bucket behind `https://ci.eullm.eu` (Let's Encrypt proxy). `SCCACHE_*`
+ `AWS_*` secrets configure it. `Cache location` in the stats reads `s3`.

**Why NOT the free GitHub Actions cache (learned the hard way in v0.5.11→v0.5.12):**
GitHub's Actions cache is **ref-scoped** — a run can only restore caches saved on
its own ref or the default branch. Our releases fire on **tags** (`EuLLM-v*`), and
each tag is a different ref, so a GitHub-cache sccache **never hits across
releases**: v0.5.11 saved 865 MiB under its tag; v0.5.12 (another tag) couldn't see
it and rebuilt cold (38 min Linux CUDA). S3 is a **global, content-addressed**
store with no ref boundary → it hits across every release (proven on S3 in v0.5.9:
129 CUDA hits, **2m34s** Linux CUDA, `Cache location: s3`). **Do NOT move the
release workflow's sccache back to the GitHub cache** — architecturally
unsuitable for tag-triggered builds.

### Windows CUDA: Ninja generator + S3 sccache (working as of v0.5.14)

For years before v0.5.14, Windows CUDA was the long-pole of every release
(~50 min cold, no warm path). The diagnosis sat in two layers and was only
fully unblocked with a generator swap, not a backend swap.

**Layer 1 — the diagnosis (proven on v0.5.9, S3 + CUDA launcher fix in place):**
sccache cached **only Rust** on Windows, never C/C++, never CUDA.

| 0.5.9 sccache stats | Linux CUDA | Windows CUDA |
|---|---|---|
| Cache hits (Rust) | 160 | 149 |
| Cache hits (C/C++) | 223 | **0** |
| Cache hits (CUDA) | 129 | **0** |
| Wall-clock | 2m34s | ~50 min |

Root cause: the CMake "Visual Studio" generator (MSBuild) **silently ignores**
`CMAKE_C_COMPILER_LAUNCHER` / `CMAKE_CXX_COMPILER_LAUNCHER` /
`CMAKE_CUDA_COMPILER_LAUNCHER`. Those work only with the Makefile and Ninja
generators. `llama-cpp-sys-2` on Windows defaults to the VS generator, so cl.exe
and nvcc invocations bypass sccache entirely — no cache key is ever computed,
no object is ever stored. Switching the cache backend (S3, GitHub, anything)
cannot fix this; the launcher contract is at the generator level.

**Layer 2 — the fix (proven on the try-windows-ninja experiment, run #2):**
force `CMAKE_GENERATOR=Ninja` in the Windows CUDA job. Three small changes:

1. `choco install ninja` (the binary must be on PATH before cargo builds).
2. Activate MSVC x64 dev env via `vcvars64.bat` (found via `vswhere`).
   Plain `windows-2022` runners do NOT activate it — that's only set up for
   MSBuild. Without this, Ninja can't find `cl.exe` / `nvcc`.
3. `CMAKE_GENERATOR: Ninja` as an env on the cargo build step.

Experiment proof (build engine, cold):

| Generator | Tracked by sccache | Wall-clock | Outcome |
|---|---|---|---|
| VS / MSBuild | 0 (no C/C++ line, no CUDA line in stats) | ~36 min | Re-builds from scratch forever |
| **Ninja** | **205 C/C++ + 130 CUDA written to S3** | ~36 min | **Cache populated → next build warm** |

After the experiment populated S3, the projection for the **second** Ninja
build on Windows CUDA is ~5–10 min, mirroring exactly the Linux CUDA jump from
0.5.9 (populate, 2m34s with hits) to 0.5.13 (rebuild, 2m47s with hits).

**Rule of method:** never claim sccache "works" on a platform without reading
that platform's `Cache hits (C/C++)` and `Cache hits (CUDA)` lines first. Don't
extend Linux results to Windows.

### Windows DLL strategy (B1) — still useful, no longer urgent

The `build-llama-dll.yml` workflow + B2 (patching `llama-cpp-sys-2`'s build.rs
to link a prebuilt DLL) was conceived when Ninja-on-Windows was thought
intractable. With v0.5.14 the urgency is gone, but two values remain:

- **Smaller release ZIPs**: linking against a separately-published DLL means
  `eullm.exe` shrinks dramatically (the bulk of llama.cpp lives in the DLL).
- **Self-update path**: updating the DLL independently of the engine binary
  enables in-place llama.cpp upgrades without recompiling Rust.

Treat B1/B2 as a future feature, not a speed fix. The B1 run #1 artefact
(`llama-dll-windows-cuda-12.8.zip`, 122 MB) is already proven valid: 231
`llama_*` symbols exported, all 5 critical ones (`llama_backend_init`,
`llama_decode`, `llama_model_load_from_file`, `llama_model_load_from_splits`,
`llama_tokenize`), plus `bindings.rs` already produced by bindgen — meaning
B2 can skip the bindgen step entirely when it eventually lands.

### sccache resilience: keep S3 from killing the build

**The deeper lesson (process):** research a cache backend's *scoping/eviction
rules up front* before migrating. We discovered the ref-scoping by a failed
release instead of reading the docs — costly. Two confounds (the missing
`CMAKE_CUDA_COMPILER_LAUNCHER`, then the ref-scoping) made the S3 value hard to
see; the launcher was the real long-pole on Linux, S3 was always the right
backend for tag-triggered release builds.

**`SCCACHE_IDLE_TIMEOUT: "0"`** + a reachability probe before enabling the wrapper
stay (so an S3 blip degrades to a cache-less build instead of killing it).

### Three launcher vars, not two: CUDA needs sccache too

Setting only `CMAKE_C_COMPILER_LAUNCHER=sccache` + `CMAKE_CXX_COMPILER_LAUNCHER=sccache`
**caches C/C++ but silently leaves nvcc invocations uncached**. The heavy
CUDA kernel template instantiations (`fattn-vec-instance-*.cu`,
`template-instances/*.cu`, many per K/V cache type combination) compile
from scratch on every release. Result: sccache stats show 99% hit rate
(on C/C++ only) but wall-clock stays at cold-build values because the
actual long-pole is nvcc, not g++.

**Mandatory third var alongside the other two:**

```yaml
CMAKE_CUDA_COMPILER_LAUNCHER=sccache
```

Set it in every `Install sccache` step that's followed by a CUDA build —
both bash (Linux) and pwsh (Windows). Setting it on non-CUDA jobs is
harmless (CMake just doesn't reference it).

How to spot the issue from sccache stats: look at "Cache hits (C/C++)"
and "Cache hits (Rust)" — if there is no separate "Cache hits (CUDA)"
line and the long-pole build wall-clock is multi-hour, you forgot the
CUDA launcher. v0.5.7 burned this: 387 hits, 0.282s avg read, but
1h 41m Linux CUDA TQ wall-clock because nvcc bypassed the wrapper.
Fixed in v0.5.8.

The first run after enabling the CUDA launcher is still a cold build
(populates the cache), so the *real* speedup only shows on the run
*after* that.


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
