# CLAUDE.md — Project Context for EULLM

## Git Rules (MANDATORY)

- **Branch names**: NEVER use "claude" or any AI tool name in branch names. Use conventional prefixes: `feat/`, `fix/`, `docs/`, `chore/`.
- **Commit author**: Use `primoco <58369875+primoco@users.noreply.github.com>` for all commits. Set this before committing.
- **Commit messages**: Conventional commits (feat:, fix:, docs:, chore:). No references to AI tools or Claude in commit messages or code comments. **This includes the `Co-Authored-By:` and `Claude-Session:` trailers some tooling appends by default** — GitHub renders those as a second author on the commit. It also includes merge commits, where the author override is easy to forget because it is not the same command as the one used for ordinary commits. Enforced by the `commit_hygiene` job in `ci.yml`, which scans only the commits new to a push, so existing history does not fail every run.
- **Working branch**: Always ask the user which branch to work on, or use the one specified in session instructions. If the session specifies a `claude/` branch, rename it following the rules above before pushing.
- **One branch per chat, not one per change**: every commit of a chat goes on the same working branch and the same open PR, which the user merges when it is ready; a new branch only when that PR has been merged. A branch per fix left the user with a PR to merge after every message (2026-10-26).
- **Nothing is posted on GitHub without the user's go-ahead**: reviews, review comments, code suggestions, PR or issue comments, and opening or closing PRs and issues all speak for the project to other people. Review a contributor's PR (Gabriele's included) and report to the user first: what is right, what is wrong, and the exact comment or suggestion code you would post. Post only after the user explicitly approves that post. A request to "check the PRs" is not that approval. On 2026-09-29 two change-request reviews went to Gabriele before the user had seen them.
- **Fixing a contributor's PR instead of asking for another round**: when the contributor has allowed edits by maintainers (the PR's "Allow edits by maintainers", `maintainer_can_modify` in the API) and the user approves the change, a small fix goes as a code suggestion in a review (the user applies it, resolves the conversation and merges), or as a commit on the contributor's branch. Both conditions, every time: the contributor's permission on that PR, and the user's go-ahead for that fix. First used on #975 (2026-10-19). A contributor is never asked to change, fix or rebase anything: what should change goes into a review as code suggestions, which the user applies before resolving the conversation and merging (2026-10-29).
- **Cutting a release ("facciamo la release X.Y.Z")**: stop at bumping the version (`engine/Cargo.toml`) and `CHANGELOG.md`, committing and pushing that to the working branch. Merging the PR into `main` and pushing the `EuLLM-v*` tag are done by the user — do not merge the PR and do not attempt `git push` of a tag. Asked repeatedly; stop re-litigating it.

## Coding Standards

- **Tests:** Required for all core functionality
- **Docs:** Every public API documented
- **No vendor lock-in:** Abstract external services behind interfaces
- **Records of public conversations**: the docs keep what was said in posts and comments elsewhere (Reddit, forums, other projects' pull requests and issues) and on which site, by its main domain only (reddit.com, github.com). No links to the posts or comments, and no subreddit or thread names; a pull request we build on is still named by its number.
- **Always check latest versions**: When adding or updating any dependency (Rust crates, Python packages, GitHub Actions), look up the current latest stable version online and use that. Never guess or copy version numbers from memory — they go stale quickly.

## License

AGPL-3.0-or-later (relicensed 2026-08, was Apache 2.0). All code must stay
permissively-*dependent*: never introduce a GPL, AGPL, or other copyleft
dependency — our own crates being AGPL-3.0-or-later is the one deliberate
exception, not a precedent for relaxing this on the dependency side. New
contributions require signing the project's CLA before merge (handled
manually for now — ask in the PR if it hasn't happened).

Note that this only binds new work going forward: every version already
published under Apache 2.0 keeps its Apache 2.0 terms for anyone who already
has a copy — relicensing cannot revoke a licence already granted.

## Architecture Decisions

- **Rust for Engine/Hub:** single binary, performance, cross-compilation
- **Python for Forge:** PyTorch ecosystem, Colab compatibility
- **Not a fork of Ollama:** API compatibility, not code compatibility. Clean Rust implementation with native audit trail
- **Streaming via mpsc channels:** inference engine sends tokens through `tokio::sync::mpsc`. Ollama endpoints (`/api/generate`, `/api/chat`) use **NDJSON** (newline-delimited JSON, `application/x-ndjson`) — one JSON object per line, no `data:` prefix. OpenAI endpoint (`/v1/chat/completions`) uses **SSE** (Server-Sent Events, `data:` prefix). This matches exactly what Ollama does, so any Ollama client works without modification.
- **Compression strategy:** pruning (MLP-focused) → distillation → identity LoRA fine-tuning (merged into the weights) → GGUF quantization (validated by NVIDIA Minitron research). See `forge/CLAUDE.md` for why identity precedes quantization and why AWQ/GPTQ is not on the GGUF path.
- **Iterative pruning for >50% compression:** compress 30%, distill, compress again (NVIDIA recommendation)
- **Continuous batching scheduler:** dedicated OS thread runs a decode loop that processes multiple requests in parallel (up to `max_batch_size`): each step decodes one token of every running sequence in a single `LlamaBatch`, with per-sequence KV cache management and near-linear throughput scaling. With more than one slot, a newly admitted request's prompt is read a chunk at a time in the running sequences' decode steps (each step's `LlamaBatch` holds one token per answer and, in slot order among them, up to a micro-batch of the waiting prompts, oldest first: `PendingPrefill`/`PromptPart`, step 3 of the loop, roadmap 0.7-D; a prompt whose step failed is read apart, after the step, in step 7), so a long prompt holds none of them up and requests that arrive together start together; with one slot (the default, and `--mtp`) it is read whole on admission (`prefill_sequence`, in `n_batch` chunks). This is a key differentiator over basic mutex-guarded inference.
- **Docker support:** multi-stage builds for Engine/Hub (Rust → debian-slim ~50MB), NVIDIA CUDA base for Forge. docker-compose.yml orchestrates all services with GPU profiles
- **CI/CD:** GitHub Actions CI (build + test + clippy/ruff for all 3 components on every push/PR). Release workflow builds cross-platform Engine binaries (Linux x64/arm64, macOS x64/arm64) on tag push, creates GitHub Release with SHA256 checksums.
- **EU Infrastructure:** Hetzner (primary), OVH/Scaleway (secondary)

## HPC sites

Each site has its own document, read before working there; when sites are
worked on in parallel, each has its own chat and its own branch. `engine/`
and `bench/campaign/` are shared by all of them: a change there is made in
one branch at a time and reaches the others through `main`, or two chats
end up pushing over each other.

The user works on all the sites at once and does not watch the clock for
any of them: nothing is handed over that has to be launched at a given
time, or after something else ends. What goes in a queue either runs long
and starts its own next steps (dependencies, chains, a job that does the
follow-up itself), or is quick. A job does its own finishing too (merging
parts, a summary in its log), so the user only reads the result.

- **LUMI-G** (EuroHPC EHPC-DEV-2026D09-278, `project_465003366`): engine
  inference and benchmarks only, on MI250X (gfx90a). Spent on what the
  work really needs, not held to the straight line: a round on each engine
  that touches inference, on as many nodes as it takes; if the hours run
  out early, more are asked for (PI, 21-10-2026). `docs/lumi/allocation-plan.md`,
  `docs/lumi/lumi-g.md`, `tools/lumi/`.
- **JUPITER Booster** (EuroHPC EHPC-AIF-2026PG01-1434, JSC budget
  `e-aif-2026pg01-1434`, 5,000 GPU-hours): Forge, with the engine as
  deliverable 5, on GH200 (aarch64, sm_90). `booster` hands out whole nodes
  only (a job pays 4 GPU-hours an hour, whatever it asks for). Jobs are as
  long as the work needs, not Leonardo's two hours: the 3-8 hour jobs of
  20-10-2026 got a node as soon as the 2-hour ones. What is watched is the
  wait between asking and getting a node (`tools/jupiter/budget.sh` prints
  it by the time asked for), since on Leonardo whole nodes for 12-24 hours
  waited a day, once 36 hours. Compute nodes have no internet and no `git`.
  `docs/jupiter/allocation-plan.md`,
  `tools/jupiter/`, `forge/scripts/jupiter/`. Every result, fix and job
  goes into `docs/jupiter/dossier.md` too, the record that is sent out.
- **Leonardo** (CINECA): Forge on A100 64GB. The Playground `AIFAC_P02_1147`
  ended 2026-11-02; the **Fast Lane EHPC-AIF-2026FL01-1400** (30,000
  GPU-hours, 2026-11-02 → 2027-02-02, no extension) carries on, and its pace,
  work packages and commitments are in `docs/cineca/fastlane-plan.md`.
  The Fast Lane's queue is not the Playground's: on 3-11-2026 3-GPU jobs
  started at once for up to 24 h, and whole nodes and 4-node jobs within
  two hours (rules in `forge/CLAUDE.md`, "the shape of a job that starts"). `docs/cineca/leonardo.md`,
  `docs/leonardo-runbook.md`, `forge/scripts/leonardo/`, and the job-shape
  rules in `forge/CLAUDE.md`.

## Current Phase: Forge pipeline + first demo models

Outstanding tasks:
1. Full Forge pipeline with verticalizzazione profiles (legal-it, medical-de, finance-fr)
2. End-to-end run: Qwen3-30B-A3B-Base → legal-it-4b v0.1 GGUF Q4_K_M
3. First 3 demo models on Hub
4. Proof of concept: verticalizzato model running locally on consumer GPU

## What NOT to do

- Never add telemetry sending data outside EU
- Never hardcode API keys or credentials
- Never introduce Llama models in the default catalog
- Never break Ollama API compatibility in Engine
- Never introduce a GPL, AGPL, or other copyleft *dependency* — everything we
  depend on must stay permissive (MIT, Apache-2.0, BSD, or similar); our own
  crates being AGPL-3.0-or-later is the one deliberate exception
- Never run distillation on Colab Pro+ (insufficient for multi-GPU, long-running jobs)

## Where the rest of the guidance lives

This file covers what applies everywhere. More specific rules load automatically when relevant, so they aren't repeated here:

- **`engine/CLAUDE.md`** — Rust engine internals: `RuntimeOpts`/config-channel rules, the llama.cpp submodule bump policy. Loads when working under `engine/`.
- **`forge/CLAUDE.md`** — verticalization pipeline, demo models, GPU infrastructure budget, permitted base-model licenses. Loads when working under `forge/`.
- **`release-and-ci` skill** — CI/CD workflow rules, sccache/S3 caching, version numbering, changelog conventions. Loads when working on releases or `.github/workflows/*.yml`.
