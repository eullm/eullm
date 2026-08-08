# CLAUDE.md — EULLM Engine (Rust)

Loaded automatically when Claude Code works with files under `engine/`.
Project-wide rules (Git, license, architecture) live in the repo root
`.claude/CLAUDE.md` and always apply too.

## Configuration channels (MANDATORY)

Two channels, and the choice is not a matter of taste.

- **Model and inference configuration → CLI flags**, declared once in
  `RuntimeOpts` (see below). GPU layers, context size, KV cache types, batch
  size, flash attention, MoE offload, checkpoints.
- **Perimeter and policy configuration → environment variables** (`EULLM_*`),
  read from the process environment first and the `.env` file second, with the
  effective source logged at startup. `EULLM_API_KEYS`, `EULLM_API_KEYS_FILE`,
  `EULLM_ALLOWED_IPS`, `EULLM_ALLOWED_ORIGINS`, `EULLM_WEB_ALLOWED_DOMAINS`,
  `EULLM_WEB_ALLOW_HTTP`, `EULLM_WEB_ALLOW_PRIVATE_HOSTS`,
  `EULLM_ALLOW_MODEL_PATHS`, `EULLM_AUDIT_DIR`, `EULLM_MODELS_DIR`.

Three reasons, in order of how expensive they are to get wrong: a secret on a
command line is visible in `ps` to every local user on the box; every
non-interactive deployment (`docker run -e`, a compose `environment:` block, a
systemd `Environment=`) configures the environment and not argv; and a
perimeter setting added as a CLI flag still has to be wired through
`ServeConfig` by hand, which is the half of the old divergence that
`RuntimeOpts` does not fix. An environment variable read inside `api::serve`
is read once, by both commands, and has nothing to wire.

When adding a perimeter setting: put the resolution in a pure `resolve()`
function so precedence is testable **without mutating process environment
variables** (which race against every other test in the binary — see
`api::ip_allowlist::resolve`, `api::auth::ApiKeys::resolve`,
`api::origin::AllowedOrigins::resolve`, `tools::guard::WebPolicy::resolve`),
log the effective value *and its source* at startup, and decide explicitly
whether unusable configuration is fatal. It is fatal for `EULLM_API_KEYS` and
for an explicitly set `EULLM_AUDIT_DIR`: someone who configured a control and
gets it silently disabled is worse off than someone whose process refused to
start.

## One `RuntimeOpts`, flattened into `run` and `serve`

The 21 flags the two commands share are declared once, in a
`#[derive(clap::Args)] struct RuntimeOpts` that both subcommands take via
`#[command(flatten)]`. **Add a model-loading or inference flag there and it
exists on both, with the same default and the same help text.** There is no
second list to keep in step.

This replaces a mandatory parity rule that said "any flag added to
`Commands::Run` MUST be added to `Commands::Serve` in the same PR". The rule
existed because the two lists had already drifted in production:
`cache_type_k`, `cache_type_v` and `gpu_layers` were on `Run` only, so every
model `serve` loaded was forced into fixed defaults with no override. A rule
holds for as long as the next person remembers it; the struct holds always.

Two things the struct does **not** do, and they are where the remaining care
is needed:

- **A flag deliberately kept out needs its reason written next to it**, and
  both paths wired by hand. `--fit`/`--fit-strict` were that flag until
  0.6.70-rc21: `serve` loads inside `api::swap_model`, which had no sizing
  step, so offering them there would have parsed them and done nothing.
  They now live in `RuntimeOpts` and `swap_model` runs the same sizing
  before every load (after the old model is unloaded, so the measured free
  VRAM is real) — but *never prompts*: a daemon, or a swap serving an API
  request, has nobody at the keyboard, so a partial split is applied and
  logged (`fit::run_fit_headless`) and `--fit-strict` surfaces as an API
  error instead of a question. Two rules survive that work: `ServeConfig`
  must receive the user's ORIGINAL `--gpu-layers`/`--cpu-moe`/`--n-cpu-moe`
  flags, never values a launch-time fit computed (a dense 27B's 43/64
  split reused to load a 22 GB MoE via web-UI swap OOM'd — found live);
  and anything that can block on stdin must stay out of the swap path.
- **A shared flag still has to reach the model `serve` loads**, through
  `ServeConfig` and into `api::swap_model`. Parsing it is not honouring it.
  Where a per-request override exists (`ctx_size`, `batch_size` via
  `override_*.unwrap_or(self.*)` in `api/mod.rs`), extend that same pattern to
  new fields rather than only adding a launch-time flag — and validate or
  clamp anything that arrives in a request body.

**A default that silently divides a shared resource is not a default.**
`serve` defaulted to `--batch-size 8` while `run` defaulted to 1, and that
looked like a reasonable difference between a daemon and a chat. It was not:
`--ctx-size` is the *total* KV budget split evenly across slots, so eight
slots gave each request 512 tokens of the 4096 default. Answers stopped
mid-sentence and came back as `done_reason="length"`, pointing at no flag,
because the operator had set none. Both default to 1 now and concurrency is
asked for.

## Keeping llama.cpp Current (MANDATORY)

Falling behind on llama.cpp means falling behind on every new model
architecture, quantization scheme, and performance fix it ships — a
competitor building on stock llama.cpp or Ollama gets those on day one; we
only do if `llama.cpp` (the `engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp`
git submodule) and `llama-cpp-rs` (`engine/vendor/llama-cpp-rs`, vendored
source, **not** a submodule) are kept moving together. Proven on 2026-07-31,
not just feared: bumping the submodule pin alone from a commit 1.5 months
old to the latest tag broke `cargo build` immediately, on 3 real C-API
changes (`use_mlock`/`use_mmap` fields replaced by `load_mode`,
`mtmd_input_text` needing a new `text_len`, an mtmd helper's return type
changed to `mtmd_helper_bitmap_wrapper`) — full details in backlog item H3-R.

- **Check upstream at least once a week.** Compare our submodule pin against
  the latest `bNNNNN` tag on `ggml-org/llama.cpp` — sort tag numbers
  numerically, not as strings (`b9999` sorts before `b10200` lexicographically
  despite being six weeks older). Also check whether `utilityai/llama-cpp-rs`
  (mirrored at `eullm/llama-cpp-rs`) has moved to track the newer API.
- **Bump often, in small steps — never let a gap turn into one big jump.** A
  one-week-old diff is a handful of API changes to read through; a six-week
  gap forces understanding months of upstream evolution at once, under
  release pressure, which is exactly when radical, ill-fitting code changes
  get made. Small frequent bumps also keep our own fixes
  (`estimate_kv_memory`, `probe_and_shrink_context`,
  `correct_kv_cache_for_model`, the DeepSeek chat template) validated against
  upstream's *current* behavior instead of a stale one.
- **Both repos move in the same change.** Bumping only the submodule pin is
  not a valid intermediate state — it does not compile. Re-vendoring
  `llama-cpp-rs` means copying its updated source into
  `engine/vendor/llama-cpp-rs/`, not editing a version string.
- **Validate every bump on real hardware before it lands**, not just on a
  green `cargo build`: reload every locally available model family at least
  once, including a multimodal one and the DeepSeek reasoning template — a
  clean compile does not prove KV-cache sizing or template behavior didn't
  silently shift upstream.
- **A failed bump attempt is cheap, not wasted, when caught at compile time.**
  Catching a breaking API change at local `cargo build`, before opening a PR
  or spending CI/GPU time, is the right place to catch it. Revert the pin,
  log exactly what broke (see H3-R for the format), and try again the
  following week — a failed attempt is a reason to fix the wrapper next time,
  never a reason to stop trying.
