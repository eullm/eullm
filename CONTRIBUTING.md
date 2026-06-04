# Contributing to EULLM

First off, thanks for considering contributing to EULLM. Every contribution matters — code, docs, bug reports, ideas, translations.

## Quick start

```bash
git clone https://github.com/eullm/eullm.git && cd eullm

# Engine (Rust)
cargo build --release
./target/release/eullm run ./your-model.gguf

# Forge (Python)
cd forge && pip install -e ".[dev]" && pytest

# Hub (Rust)
cd hub && cargo build
```

## How to contribute

### Report a bug

Open an [issue](https://github.com/eullm/eullm/issues/new) with:
- What you did
- What you expected
- What happened instead
- OS, GPU, VRAM, model name

### Suggest a feature

Open an issue tagged `enhancement`. Describe the use case, not just the solution.

### Submit code

1. Fork the repo
2. Create a branch: `feat/your-feature`, `fix/your-fix`, `docs/your-change`
3. Make your changes
4. Run tests and lints:
   ```bash
   # Engine
   cd engine && cargo test && cargo clippy -- -D warnings

   # Forge
   cd forge && pytest && ruff check .
   ```
5. Commit with conventional messages: `feat:`, `fix:`, `docs:`, `chore:`
6. Open a PR against `main`

### What we need most right now

| Area | What | Difficulty |
|------|------|------------|
| **Testing** | Try the engine on different GPUs and report results | Easy |
| **Models** | Test with different GGUF models, report compatibility | Easy |
| **Docs** | Translations (Italian, German, French, Spanish) | Easy |
| **Docs** | Tutorials, blog posts, video walkthroughs | Medium |
| **Engine** | Ollama API parity — find missing endpoints/fields | Medium |
| **Engine** | Benchmark quantized KV cache (Q4_0/Q5_0/Q8_0) at long contexts on different GPUs | Medium |
| **Forge** | Test pruning/distillation pipeline on new models | Hard |
| **Forge** | New domain profiles (legal-de, medical-it, etc.) | Hard |

### Good first issues

Look for issues tagged [`good first issue`](https://github.com/eullm/eullm/issues?q=label%3A%22good+first+issue%22). These are scoped, well-described tasks suitable for first-time contributors.

## Code standards

### Rust (Engine, Hub)
- `cargo clippy` clean, no warnings
- `cargo fmt` formatted
- Tests for core functionality
- Public APIs documented

### Python (Forge)
- PEP 8, type hints on public functions
- `ruff check` clean
- `pytest` passing
- Docstrings on public functions

### Commits
- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `test:`
- Keep messages concise — explain *why*, not *what*
- One logical change per commit

### Branch naming
- `feat/short-description`
- `fix/short-description`
- `docs/short-description`

## Architecture overview

```
eullm/
├── engine/     Rust — LLM runtime (llama.cpp, API server, audit trail)
├── forge/      Python — Model verticalizzazione pipeline
├── hub/        Rust — EU model registry API
├── bench/      Benchmarks and stress tests
└── docs/       Technical documentation
```

See [docs/architecture.md](docs/architecture.md) for detailed diagrams.

## License

By contributing, you agree that your contributions will be licensed under [Apache 2.0](LICENSE).

**Important:** Do not introduce dependencies with GPL, AGPL, or other copyleft licenses. All dependencies must be Apache 2.0 compatible.

## Questions?

- Open an [issue](https://github.com/eullm/eullm/issues)
- Email: dev@eullm.eu

---

Built in Europe. For Europe. By Europeans.
