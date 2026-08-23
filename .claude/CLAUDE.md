# CLAUDE.md — Project Context for EULLM

## Git Rules (MANDATORY)

- **Branch names**: NEVER use "claude" or any AI tool name in branch names. Use conventional prefixes: `feat/`, `fix/`, `docs/`, `chore/`.
- **Commit author**: Use `primoco <58369875+primoco@users.noreply.github.com>` for all commits. Set this before committing.
- **Commit messages**: Conventional commits (feat:, fix:, docs:, chore:). No references to AI tools or Claude in commit messages or code comments. **This includes the `Co-Authored-By:` and `Claude-Session:` trailers some tooling appends by default** — GitHub renders those as a second author on the commit. It also includes merge commits, where the author override is easy to forget because it is not the same command as the one used for ordinary commits. Enforced by the `commit_hygiene` job in `ci.yml`, which scans only the commits new to a push, so existing history does not fail every run.
- **Working branch**: Always ask the user which branch to work on, or use the one specified in session instructions. If the session specifies a `claude/` branch, rename it following the rules above before pushing.
- **Cutting a release ("facciamo la release X.Y.Z")**: stop at bumping the version (`engine/Cargo.toml`) and `CHANGELOG.md`, committing and pushing that to the working branch. Merging the PR into `main` and pushing the `EuLLM-v*` tag are done by the user — do not merge the PR and do not attempt `git push` of a tag. Asked repeatedly; stop re-litigating it.

## Coding Standards

- **Tests:** Required for all core functionality
- **Docs:** Every public API documented
- **No vendor lock-in:** Abstract external services behind interfaces
- **Always check latest versions**: When adding or updating any dependency (Rust crates, Python packages, GitHub Actions), look up the current latest stable version online and use that. Never guess or copy version numbers from memory — they go stale quickly.

## License

Apache 2.0. All code must be Apache 2.0 compatible. Never introduce dependencies with GPL, AGPL, or other copyleft licenses.

## Architecture Decisions

- **Rust for Engine/Hub:** single binary, performance, cross-compilation
- **Python for Forge:** PyTorch ecosystem, Colab compatibility
- **Not a fork of Ollama:** API compatibility, not code compatibility. Clean Rust implementation with native audit trail
- **Streaming via mpsc channels:** inference engine sends tokens through `tokio::sync::mpsc`. Ollama endpoints (`/api/generate`, `/api/chat`) use **NDJSON** (newline-delimited JSON, `application/x-ndjson`) — one JSON object per line, no `data:` prefix. OpenAI endpoint (`/v1/chat/completions`) uses **SSE** (Server-Sent Events, `data:` prefix). This matches exactly what Ollama does, so any Ollama client works without modification.
- **Compression strategy:** pruning (MLP-focused) → distillation → identity LoRA fine-tuning (merged into the weights) → GGUF quantization (validated by NVIDIA Minitron research). See `forge/CLAUDE.md` for why identity precedes quantization and why AWQ/GPTQ is not on the GGUF path.
- **Iterative pruning for >50% compression:** compress 30%, distill, compress again (NVIDIA recommendation)
- **Continuous batching scheduler:** dedicated OS thread runs a decode loop that processes multiple requests in parallel (up to `max_batch_size`). Prefill + decode in a single `LlamaBatch`, per-sequence KV cache management, near-linear throughput scaling. This is a key differentiator over basic mutex-guarded inference.
- **Docker support:** multi-stage builds for Engine/Hub (Rust → debian-slim ~50MB), NVIDIA CUDA base for Forge. docker-compose.yml orchestrates all services with GPU profiles
- **CI/CD:** GitHub Actions CI (build + test + clippy/ruff for all 3 components on every push/PR). Release workflow builds cross-platform Engine binaries (Linux x64/arm64, macOS x64/arm64) on tag push, creates GitHub Release with SHA256 checksums.
- **EU Infrastructure:** Hetzner (primary), OVH/Scaleway (secondary)

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

## Where the rest of the guidance lives

This file covers what applies everywhere. More specific rules load automatically when relevant, so they aren't repeated here:

- **`engine/CLAUDE.md`** — Rust engine internals: `RuntimeOpts`/config-channel rules, the llama.cpp submodule bump policy. Loads when working under `engine/`.
- **`forge/CLAUDE.md`** — verticalization pipeline, demo models, GPU infrastructure budget, permitted base-model licenses. Loads when working under `forge/`.
- **`release-and-ci` skill** — CI/CD workflow rules, sccache/S3 caching, version numbering, changelog conventions. Loads when working on releases or `.github/workflows/*.yml`.
