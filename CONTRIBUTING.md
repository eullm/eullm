# Contributing to EULLM

First off, thanks for considering contributing to EULLM. Every contribution matters — code, docs, bug reports, ideas, translations.

## Quick start

```bash
# --recursive is required: llama.cpp is a submodule
git clone --recursive https://github.com/eullm/eullm.git && cd eullm

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

EULLM is licensed under [AGPL-3.0-or-later](LICENSE). I3K Technologies Srl
holds the copyright and also offers the software under a separate commercial
licence, so before a pull request can be merged you must agree to the
[Contributor Licence Agreement](CLA.md) by adding one line to your pull
request description:

```
I have read and agree to the Contributor Licence Agreement in CLA.md.
```

Agreeing only that your contribution is "licensed under AGPL-3.0-or-later" is
**not** the same thing and is not sufficient: it is the licence the project
already carries, and it leaves us without the right to include your work in a
commercially licensed build. [CLA.md](CLA.md) sets out what you grant, what
you keep (your own copyright — it is a licence, not an assignment), and why a
`Signed-off-by` line does not cover it. One agreement covers all your future
contributions.

### The open engine and EuLLM Pro

EuLLM Pro is this engine with modules for organisations compiled in (accounts
for each person, the AI Act records, assistants for groups, budgets), sold
under a commercial licence; the CLA above is what lets it include the
engine's code. Three commitments come with it:

- **Nothing moves from the open engine to Pro.** What the engine does today
  it keeps doing, under AGPL-3.0, in every release. Pro only adds.
- **The engine is not made worse to sell Pro.** No limit is put into the open
  engine for Pro to lift.
- **Pro is visible, not in the way.** The chat's administrator menu lists the
  Pro pages in the released binaries and Docker images, never in the chat
  itself; a build from source does not show them unless asked
  (`EULLM_NO_PRO_HINTS=0`), so you do not have them in front of you while
  working on the code.

**Important:** Do not introduce *dependencies* with GPL, AGPL, or other
copyleft licenses — our own crates being AGPL-3.0-or-later doesn't change
this. Every dependency must stay permissive (MIT, Apache-2.0, BSD, or
similar); `cargo deny check licenses` enforces this for the Rust workspace.

## Questions?

- Open an [issue](https://github.com/eullm/eullm/issues)
- Email: dev@eullm.eu

---

Built in Europe. For Europe. By Europeans.
