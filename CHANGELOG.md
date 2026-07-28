# Changelog
All notable changes to the EULLM Engine, newest first.

Entries are the user-facing changes from each release: what was added, what was
fixed, and what got faster. Documentation, CI and internal refactors are left
out on purpose, because a changelog is for deciding whether to upgrade, not for
reading the repository history. The full history is in the commit log.

Versions are the `EuLLM-v*` release tags. Binaries for every version are on the
[releases page](https://github.com/eullm/eullm/releases).

Entries for **0.6.36 and later** are written by hand. Everything below that is
derived from the commit history and reads like it: useful for tracing when
something changed, less so for understanding what it means.

## 0.6.44 — 2026-07-28

### Fixed
- **Building from source no longer fails on a missing header.** llama.cpp is a
  git submodule and the README said to clone without `--recursive`, so the build
  died minutes in on `llama.h file not found`, which reads like a broken
  compiler rather than an incomplete checkout. The build now stops immediately
  and prints the command that fixes it. Note that the `Source code (zip/tar.gz)`
  archives on the releases page can never build: GitHub generates them without
  submodule contents, so clone the repository instead.
- **`eullm -V` reports the right version again.** The 0.6.43 binaries answer
  `0.6.42`, because the version bump landed after that release was tagged. Only
  the reported string was wrong: those binaries do contain everything listed
  under 0.6.43.

## 0.6.43 — 2026-07-28

### Fixed
- **The browser chat works again after choosing a model from the picker.** Start
  `eullm` with no arguments, pick a model, and every message came back with
  "No model loaded" while that model was loaded and answering. The model list
  the UI reads marks the loaded model by leaving its checksum blank, and for a
  model from the catalog that blank was immediately overwritten with the
  catalog's real checksum, so the UI lost the only thing telling it what was
  loaded. Starting from a file path (`eullm run ./model.gguf`) was never
  affected, which is why this survived so long.
- **A model already on your disk can be picked in the chat UI.** With
  `eullm serve` the picker offered nothing selectable, because every catalog
  entry was greyed out as "not yet downloaded" whether you had it or not. The
  list now separates what is on this machine from what would be a download, and
  the first one is selected for you when nothing is loaded. The server was
  always able to switch to it on the first message.
- **Voice notes and other audio formats are accepted.** A WhatsApp recording is
  Ogg/Opus, which the engine cannot decode, so it arrived as unreadable bytes.
  Audio outside wav, mp3 and flac is now converted in the browser before being
  sent, the same as images outside jpg/png/bmp/gif already were. If your browser
  cannot decode the file either, the message now says so and suggests a command.
- **A download that cannot create its folder says why, before downloading.**
  `eullm pull` reported `File exists (os error 17)` and then suggested the model
  might not be published yet. Both halves were wrong: the problem was a local
  path, and the model was fine. The check now runs first and names the path and
  what is sitting on it.

## 0.6.42 — 2026-07-27

### Added
- **Images work without a GPU.** Multimodal input used to be compiled only into
  the three CUDA builds, so sending an image to any other binary failed no
  matter how much memory the machine had. Every published binary now reads
  images: Linux x64 and arm64, both macOS builds, Windows, and the CIX P1
  board. Expect it to be slow on CPU — the image encoder is the expensive part,
  and a large photo can take tens of seconds before the first word — but on a
  machine with shared memory it is the difference between slow and impossible.
  The binaries are about 1 MB larger; nothing changes if you never send an
  image.

### Fixed
- **`eullm list` no longer fails completely because of one damaged model.** A
  `manifest.json` truncated by an interrupted download or a full disk made the
  command answer with a parser error and nothing else, hiding every healthy
  model. The damaged one is now skipped with a warning that names the directory,
  and manifests are written atomically so an interrupted write cannot produce
  that state in the first place.
- **`list` and the server now say which model directory they are using.** When
  `EULLM_MODELS_DIR` is set in one shell and not in another, `list` would show a
  model as installed while the API answered `404` for the same name, and both
  were right. Each now prints the directory it reads, and where that setting
  came from.
- **A model whose file is missing is no longer reported as ready.** `list` was
  repeating the status recorded at download time, so a model whose weights were
  deleted, or never fully arrived, stayed `ready` forever. It now checks the
  disk and reports `ready (file missing)`.
- **The model chooser finds your local models again.** The screen shown by a
  plain `eullm` looked them up by their display title rather than their name, so
  most models on disk were missing from the `LOCAL` section, and it ignored
  `EULLM_MODELS_DIR` when scanning for loose `.gguf` files. The `[local]` tag
  next to a catalog entry now means the weights are actually present, not just
  that a download was started.

## 0.6.41 — 2026-07-27

### Fixed
- **Image replies no longer start with `<|channel>thought`.** Gemma emits a
  channel preamble before its answer, and on image requests it was being shown
  to you verbatim instead of being stripped. Reasoning blocks that actually
  contain text still come through, so a UI can render them as a reasoning
  section.
- **An unsupported image format now says so.** Sending a `.webp` failed with
  `Media #0 failed to decode: NullResult`, which looked the same as a corrupt
  file. The error now names what the multimodal backend reads: jpg, png, bmp,
  gif for images, and wav, mp3, flac for audio.

## 0.6.40 — 2026-07-26

### Fixed
- **Stop sequences are now honoured in every mode.** Outside the continuous-batching
  scheduler, a stop marker was only detected when it happened to fall at the very end
  of a token, so generation could run past the end of a turn, and a marker split
  across two tokens leaked its first half to the client. Affected `--batch-size 0`
  and multimodal requests. It could also crash the request outright when the marker
  landed inside a multi-byte character.
- **`"think": false` no longer leaves stray `<think>` tags in the reply.** The tags
  are still passed through untouched when you ask for thinking, so a UI can render
  them as a reasoning section.
- **Asking for a model that does not exist returns `404` instead of `500`.** A `5xx`
  reads as "temporary" to any client with automatic retry, so a typo in a model name
  could turn into a retry loop that never succeeded. A genuine load failure (out of
  VRAM, corrupt file) still returns `500`, because that one is worth retrying.
- **Harmony scaffolding (`<|channel|>…`) is stripped on multimodal requests too.**
  It was only being removed on text requests.
- **A grammar that fails to compile is now logged on multimodal requests.** Asking
  for `format: "json"` with a broken grammar silently returned free-form text.

## 0.6.39 — 2026-07-26

### Fixed
- **Intel Macs work.** `eullm-macos-x64` was loading the whole model onto the
  machine's GPU through Metal while reporting that it was running on CPU. On the
  Intel and AMD GPUs those Macs carry, that produces wrong numbers: garbage output,
  hangs, and in one case a kernel panic. The binary now genuinely has no Metal
  backend. Validated on a 2018 Mac mini and a 2018 MacBook Pro. **If you use an
  Intel Mac, this is the version to be on.**
- **`"think": false` actually suppresses thinking.** It never did: the model kept
  reasoning, you paid for those tokens, and the reasoning appeared in the answer.
  The prompt we generate was one character away from what the model expects.
- **The startup banner tells the truth about GPU offload.** A CPU-only build
  reported `GPU layers: all` next to `GPU backend: none`.

## 0.6.38 — 2026-07-26

No functional change. Build-environment only: the macOS Intel binary is now
compiled on Intel hardware rather than cross-compiled.

## 0.6.37 — 2026-07-26

### Fixed
- **A numerically broken response fails instead of returning nonsense.** When the
  model's output becomes invalid (NaN/Inf), the request now returns an explicit
  error rather than a long string of repeated characters reported as a successful
  answer. This detects the failure, it does not repair it.

## 0.6.36 — 2026-07-26

### Added
- **API keys with per-key rate limits** (`EULLM_API_KEYS`). Needed if the engine is
  reachable beyond localhost, including behind Docker's published ports, where every
  client looks like the bridge gateway to the IP allowlist.
- **Browser origin policy** (`EULLM_ALLOWED_ORIGINS`), defaulting to loopback only.
- **Web-tool hardening**: redirects are re-validated at every hop, responses are size
  capped, and private or link-local addresses are refused unless you opt in.

### Fixed
- **`eullm serve` and `eullm run` start with the same KV cache defaults.** `serve`
  used `q8_0`/`q4_0` while `run` used `f16`/`f16`, so the same model gave different
  output quality depending on which command started it. Both now use `f16`. Expect
  more VRAM for the KV cache on `serve` than before, and better output; the old
  behaviour is `--cache-type-k q8_0 --cache-type-v q4_0`.
- **`done_reason` distinguishes a finished answer from a truncated one.** A reply cut
  off by the token limit was reported as `"stop"`, the same as a complete one.
- **`--daemon` reports a startup failure instead of claiming success.** It printed a
  PID and exited 0 even when the engine died immediately, for example on a port
  already in use.
- **Thread count defaults to physical cores.** On machines with SMT this was counting
  logical CPUs and oversubscribing, which on one 6-core Intel Mac meant 0.8 tok/s.
- **The audit trail no longer interleaves records under concurrent load.**

## 0.6.35 — 2026-07-25

### Fixed
- Close the remaining small hardening items, and bump to 0.6.35
- Never emit another sequence's tokens, never drop a slow client's
- Size the KV cache from attention.key_length, read the allowlist from the environment, and validate externally-supplied filenames
- Close the six blocking items from the fix/hardening backlog

## 0.6.34 — 2026-07-24

### Added
- Gate NaN/Inf logit check behind --rust-debug, off by default

### Fixed
- Stop scheduler panic in warn_if_logits_corrupt at idx=-1

## 0.6.33 — 2026-07-24

### Added
- Add CPU-feature startup line and NaN/Inf logit checks

## 0.6.32 — 2026-07-23

### Fixed
- Enable AVX2 baseline for x86_64 CPU-only release binaries

## 0.6.31 — 2026-07-22

### Added
- Add F0 evaluation harness (eullm_forge.eval)

### Fixed
- Commit eval seed set that .gitignore silently dropped
- Actually build eullm-macos-x64 CPU-only

## 0.6.30 — 2026-07-22

### Fixed
- Apply Gemma 4 KV cache correction on every model swap
- Expose run's model-loading flags on serve

## 0.6.29 — 2026-07-19

### Added
- Add ip_allowlist module and .env.example
- IP allowlist for API and chat UI, bump to v0.6.29

## 0.6.28 — 2026-07-19

### Added
- Verify SHA-256 of downloaded model weights

### Fixed
- Catch missing ML deps gracefully in the forge command
- Validate model slug before resolving download path
- Validate manifest digests, wire up serve batch-size, sanitize log fields

### Performance
- Set n_ubatch explicitly instead of relying on llama.cpp's default

## 0.6.27 — 2026-07-19

### Fixed
- Stop diverging from Ollama's max_tokens and seed defaults

## 0.6.26 — 2026-07-17

### Fixed
- Preserve think-suppression text when storing /no_think history
- Stop retokenizing conversation history every turn

## 0.6.24 — 2026-07-17

### Added
- Bounded checkpoint restore for hybrid-model KV reuse

## 0.6.23 — 2026-07-17

### Added
- Expose --rs-seq to let KV-cache reuse work on hybrid models

## 0.6.22 — 2026-07-17

### Added
- CIX P1 (Armv9.2-A) CPU build profile for POSCAR WP4

### Fixed
- Correct the default tracing filter target to eullm

## 0.6.20 — 2026-07-14

### Fixed
- Stop misreading a full event channel as a disconnected client

## 0.6.19 — 2026-07-13

### Fixed
- Treat a rejected KV-cache rollback as a reuse failure

## 0.6.18 — 2026-07-13

### Fixed
- Honor /no_think in the CLI REPL via template suppression
- Satisfy clippy::collapsible_if in the reuse fallback retry

## 0.6.16 — 2026-07-12

### Fixed
- Fall back to full reprefill when a reused prefill fails

## 0.6.15 — 2026-07-12

### Added
- KV-cache prefix reuse for --cli and /api/generate

## 0.6.13 — 2026-07-10

### Fixed
- Use last_mut() for buft/kv override slots, not index 0

## 0.6.12 — 2026-07-10

### Added
- Add --n-cpu-moe N for per-layer MoE CPU offload

## 0.6.11 — 2026-07-10

### Added
- Add --cpu-moe for MoE models on small GPUs

## 0.6.9 — 2026-07-07

### Fixed
- Drop NCCL so Linux CUDA binaries need only the driver
- Make near-dedup test robust to MinHash estimator noise

## 0.6.7 — 2026-07-02

### Fixed
- Don't require the model store to run a direct GGUF path

## 0.6.6 — 2026-06-25

### Added
- Parallel, resumable model downloads

## 0.6.5 — 2026-06-24

### Added
- Make --fit KV-cache aware

## 0.6.4-rc.2 — 2026-06-24

### Fixed
- Enable --fit by default in the interactive picker

## 0.6.4-rc.1 — 2026-06-24

### Added
- Add --fit to auto-size GPU layers to free VRAM
- Support HuggingFace repo shorthand in run and pull

### Fixed
- Deref gguf filename refs in HuggingFace quant selection
- Clearer model-store init error — name the path, hint broken symlink/unmounted volume
- Allow --gpu-layers -1 (clap hyphen value)

## 0.6.3-rc.2 — 2026-06-23

### Fixed
- Pass placeholder arg to MtmdBitmap::from_buffer

## 0.6.3-rc.1 — 2026-06-23

### Fixed
- Build macOS release binaries with Metal
- Status box shows canonical API endpoint (11434), not UI origin

## 0.6.2 — 2026-06-09

### Fixed
- Show audio attachments cleanly in the web chat preview

## 0.6.1-beta.4 — 2026-06-09

### Fixed
- Add BOS token to image prompt (the real vision bug)

## 0.6.1-beta.3 — 2026-06-09

### Fixed
- Size n_ubatch to image tokens (non-causal vision attention)

## 0.6.1-beta.2 — 2026-06-08

### Added
- Expose image_min/max_tokens to raise vision resolution

### Fixed
- Drop per-request MtmdContext experiment

## 0.6.1-beta.1 — 2026-06-08

### Added
- Accept audio files in the web chat (multimodal)

### Fixed
- WebP→PNG in UI, surface decode errors, fresh MtmdContext per request

## 0.6.0 — 2026-06-07

### Fixed
- Raise request body limit to 64 MB for multimodal payloads

## 0.6.0-beta.8 — 2026-06-06

### Added
- Attach-image button + multimodal turn dispatch
- Route /api/chat images through mtmd (MVP)

## 0.6.0-beta.7 — 2026-06-06

### Added
- Vendor llama-cpp-rs for Gemma 4 12B Unified, bump to 0.6.0-beta.7

### Fixed
- Keep clippy gate off vendored crates; fix CUDA submodule checkout

## 0.5.20 — 2026-06-06

### Added
- Mark gemma-4-e4b as multimodal (mmproj available)

### Fixed
- Pull recovers a missing mmproj for an already-downloaded model

## 0.6.0-beta.6 — 2026-06-06

### Fixed
- Surface Gemma 4 channel-thought blocks as a Reasoning section
- Keep numbered lists alive across blank lines and display math; tighten vertical spacing

## 0.6.0-beta.5 — 2026-06-06

### Fixed
- Render orphan \frac / \sqrt and nudge math delimiting in system prompt

## 0.6.0-beta.4 — 2026-06-06

### Added
- Mark catalog entries already pulled with a [local] tag

### Fixed
- Expose the catalog id (not the human name) in /api/tags and /v1/models

## 0.6.0-beta.3 — 2026-06-06

### Added
- Enable multimodal (mtmd) in the Windows CUDA release binary

### Fixed
- Use the catalog id as the one addressable model name
- Handle LaTeX spacing commands in math renderer

## 0.6.0-beta.2 — 2026-06-05

### Added
- Markdown-lite + math-lite rendering in chat UI
- Unify Linux CUDA build — multimodal feature always on, drop the parallel job

### Fixed
- Collapse nested if into let-chain to satisfy clippy 1.96
- Elide harmony channel blocks as whole units, not just delimiters

## 0.6.0-beta.1 — 2026-06-05

### Added
- Pre-release-only multimodal CUDA build (eullm-linux-x64-cuda-12.8-multimodal)
- Multimodal MVP via mtmd — Gemma 4 12B vision (--image, beta)

## 0.5.18 — 2026-06-05

### Added
- Release workflow honours -beta/-rc/-alpha tag suffix as pre-release
- Suppress harmony-style format artifacts in stream

### Fixed
- Stop hijacking scroll during streaming, add jump-to-latest pill
- Tell models the web fetch is their own capability

## 0.5.17 — 2026-06-04

### Added
- Add Gemma 4 12B (Apache-2.0), flagged text-only
- Add Gemma 4 E4B (Apache-2.0) to curated catalog

## 0.5.16 — 2026-06-04

### Added
- Pull and run any GGUF by URL — catalog becomes an index, not a fence
- Clean failure of pull + new `eullm rm` to delete installed models

### Fixed
- Link NCCL on Linux CUDA build (v0.5.16)
- Hold back partial stop sequences in streaming output
- Refresh to June 2026 — 4 broken entries fixed, lineup updated to current Apache 2.0 / MIT GGUFs
- Don't start the terminal REPL when the browser chat is taking over

## 0.5.14 — 2026-06-03

### Added
- Windows CUDA release uses Ninja + sccache S3 (0.5.14)

### Fixed
- Try-windows-ninja — replace nonexistent ilammar/msvc-dev-cmd with vcvars64.bat

## 0.5.13 — 2026-06-03

### Added
- Try-windows-ninja experiment — isolated test of Ninja generator + sccache on Windows CUDA
- B1 — isolated workflow to build llama.cpp as a Windows CUDA DLL

### Fixed
- Release workflow back to S3/MinIO — GitHub cache is ref-scoped

## 0.5.12 — 2026-06-03

### Fixed
- Reasoning default, web multi-turn, sticky /no_think, bigger logo, auto-open browser

## 0.5.10 — 2026-06-03

### Fixed
- Don't spawn a nested tokio runtime when pulling a model
- Drop TurboQuant + installer from release notes & file list

## 0.5.8 — 2026-06-02

### Added
- Remove TurboQuant from production build path (R&D archived)

### Fixed
- Box CatalogEntry in Picked enum to satisfy clippy::large_enum_variant

## 0.5.7 — 2026-06-02

### Added
- Interactive model picker + curated catalog from GitHub raw

## 0.5.3 — 2026-05-31

### Fixed
- {userprofile} -> {userdocs} + pre-flight CI to catch Inno bugs in <2 min

## 0.5.2 — 2026-05-31

### Added
- 'eullm -V' includes build variant suffix
- Windows one-click installers for CPU / CUDA / TurboQuant
- Embedded chat UI on separate port (dual-listener)

### Fixed
- Make 'eullm run' default to single-slot context, warn on tight per-seq

## 0.5.1 — 2026-05-30

### Added
- Windows x64 build targets (CPU, CUDA, CUDA+TurboQuant)
- Add Zenodo DOI badge and citation section

### Fixed
- Cross-platform ggml_type cast in TurboQuant KvCacheType

## 0.4.4 — 2026-05-27

### Added
- Add auto GPU layer fitting to Phase 1 roadmap
- Phase 2 distillation + Phase 3 GGUF quantize scaffolding
- Training scaffolding (smoke + production configs)
- Prepare_legislation accepts single AKN XML files
- Wire Normattiva codici into the pretraining pipeline
- Final-stage formatter — dedup'd chunks → train/val JSONL
- Exact + near dedup for the chunked corpus
- CLI wrapper for italgiure fetcher
- Char-based chunker for anonymised italgiure corpus
- Role-aware person tokens in NER pass
- Add GDPR anonymiser for legal corpora
- Add italgiure corpus validation script
- Add verify flag for TLS cert fallback in italgiure fetcher
- Fetch Cassazione sentences from italgiure SentenzeWeb

### Fixed
- Phase 2 defaults to LoRA student to fit a 94-96 GB single GPU
- Drop fragile 8-bit optim, add pre-flight env check
- Drop 'formatting: pretrain' from dataset_info — removed in LF 0.9.5
- Keep dataset_dir + resume inside the YAML for LF 0.9.5
- Install_training_deps — drop bogus extras, add bitsandbytes explicitly
- Align TurboQuant intro paragraph with real benchmarks
- Round-5 — Avverso FP, role-aware all-caps, unified counters
- Round-4 NER FPs — P.Q.M., ORG prefixes, acronym spans
- Company C.F., address locutions, acronym spans, extra whitelist
- Drop institutional NER spans, suppress company FPs, extend whitelist
- NER junk-span guard and word boundaries in replacement
- Address OCR variants, extended whitelist, spacy auto-install
- Rename ambiguous 'l' to 'line' in cassazione parser
- Italgiure SentenzeWeb covers 2021+ only, not 2011+
- Don't mark italgiure slice complete on empty response
- Ricostruisci parser Cassazione dal DOM reale
- Pulisci rumore UI dal parser sentenze Cassazione
- Paginazione cortedicassazione.it via frame3_item, no Playwright
- Usa homepage come entry point per Cassazione, headers WAF-bypass
- Correggi URL e selettori cortedicassazione.it
- Incremental JSON save after each test to prevent data loss
- Add --timeout flag to turboquant_math_accuracy collect

## 0.4.3 — 2026-04-12

### Fixed
- Correct TurboQuant VRAM estimate in startup display
- Defer void_logs() and improve OOM error messages

## 0.4.2 — 2026-04-12

### Added
- Aggiungi alias bare tbqp3/tbq3/tbqp4/tbq4 per config raccomandata
- Upgrade TurboQuant to v1.5.3, add KV cache accuracy tests
- Sostituisci italgiure (paywall) con sorgenti gratuite
- Aggiungi sentenze Cassazione da italgiure.giustizia.it
- Add dati.normattiva.it OpenData AKN ZIP support
- Add Playwright support for EUR-Lex (bypasses AWS WAF)
- Add dataset preparation module for domain corpora

### Fixed
- Stub NCCL symbols for TurboQuant v1.5.3 single-GPU CI build
- Use sm_89 for TurboQuant CUDA build to avoid Blackwell kernel failures
- Aggiorna GGML type ID e tipi TurboQuant per v1.5.3
- Correggi timeout CC e aggiungi diagnostica URL sentenze
- Correggi URL italgiure, aggiungi fallback cortedicassazione.it
- Correggi condizione fallback — if not records bloccava doc_collection
- Parser documentCollection per regio decreto (codice civile/penale)
- Aggiungi parser NIR <articolo> per regio decreto anni 1930-40
- Add structure diagnostic + eId fallback for missing article elements
- Fall back to itertext() for old regio-decreto AKN structure
- Auto-detect AKN law identity from XML metadata, fix namespace
- Correct ZIP source hints and fix attoCompleto session degradation
- Replace AJAX per-article scraping with single attoCompleto bulk download
- Add rate-limit delay and article cap to normattiva.it AJAX scraper
- Rewrite normattiva.it scraper to use article AJAX endpoint
- Shared normattiva session, EUR-Lex content validation, Referer header
- Rewrite EUR-Lex parser and improve normattiva.it session handling
- Resolve ruff lint errors in legal_it dataset module
- Use requests.Session() for normattiva.it JSESSIONID cookie handling

## 0.4.1 — 2026-04-08

### Added
- Add transparent web browsing with --web flag (v0.4.0)

### Fixed
- Portable sccache install — no --wildcards, version 0.8.0, fail-fast off
- Use disk-backed sccache to avoid GHA cache API crashes
- Panic in extract_urls on multibyte UTF-8 chars
- Web content injected only in prompt, not in persistent REPL history
- Use per-slot context budget for web content injection
- Remove useless format! in web injection (clippy)
- Inject web content in REPL (interactive_chat bypassed API routes)
- Enable Qwen3 thinking mode by default in math accuracy benchmark

## 0.3.13 — 2026-04-06

### Added
- Add /temp, /maxtokens, /system commands to interactive REPL
- Multi-model chat template support (ChatML, Gemma, Llama2)
- Add --note flag to math accuracy benchmark for cache config tracking

### Fixed
- Force f16/f16 for all Gemma 4 KV cache configs until AmesianX v1.5.1
- Auto-correct incompatible KV cache for Gemma 4 instead of blocking
- Stop sequence erase and Gemma 4 q8_0 KV cache warning
- Suppress ggml logs in scheduler and warn on asymmetric TQ KV cache
- Strip stop sequence tokens from REPL display and conversation history
- Suppress llama.cpp internal log messages (CUDA graph warmup noise)

## 0.3.5 — 2026-04-03

### Added
- Upgrade llama-cpp-2 to 0.1.141 and switch TQ backend to AmesianX v1.4.1
- Add context-size breakdown and extended filler for bug-window testing
- Add throughput metrics to turboquant_math_accuracy bench

### Fixed
- Add head_dim-specific TurboQuant types (_1 for head_dim=128)
- Patch unused mut warning in llama-cpp-sys-2 build.rs during vendor setup
- Read llama-cpp-sys-2 version dynamically in setup-turboquant.sh
- Increase default num_predict from 512 to 2048 in math accuracy bench

## 0.3.3 — 2026-04-01

### Added
- Add --no-think flag for non-Qwen3 math models (Qwen2.5-Math, DeepSeek-Math)
- Add math accuracy benchmark to isolate computation vs KV recall errors
- KV cache stress test — precision recall across context distance
- TurboQuant quality report — 100 tests, test-by-test analysis
- Expand quality benchmark to 100 tests (20 per category)
- Add TurboQuant quality benchmark (matrix, math, logic, factual)

### Fixed
- Remove incorrect FWHT rotation fix from setup-turboquant.sh
- Use POSIX [[:space:]] instead of \s in sed pattern
- Collapse remaining collapsible_if for Clippy edition 2024 compliance
- Collapse nested if blocks for Clippy edition 2024 compliance
- Use portable sed temp-file pattern in setup-turboquant.sh
- Bump engine to 0.3.3, patch Bug#7 FWHT rotation mismatch in setup-turboquant.sh
- Add LaTeX matrix parser and --num-predict flag for math-specialized models
- Correct TurboQuant cache type names tq4_0/tq3_0 in docs (not q4_0)
- Default model name to qwen3-14b to match engine convention
- Skip delayed tests when --filler 0 (direct-only mode)
- Rewrite math accuracy prompts to match codebase style (inline concise format)
- Handle --filler 0 (direct-only mode) without ValueError
- Escape pipe characters in math test row (broke markdown table)
- Disable thinking mode in quality benchmark (think: false)
- Strip <think> blocks and check last line in quality benchmark

## 0.3.2 — 2026-03-30

### Added
- Full Ollama-compatible sampling parameters
- GPU scaling + cost savings charts, update README with TurboQuant showcase
- TurboQuant benchmark charts and results
- Auto-probe max ctx_size for non-TurboQuant cache types
- TurboQuant benchmark script and orchestrator
- Show KV cache memory estimate in startup banner
- Show TurboQuant active status in startup banner
- Display TurboQuant KV cache type names in startup banner
- Wire spiritbuun CUDA fork as TurboQuant backend
- TurboQuant feature naming, startup logging, strict mode
- TurboQuant backend integration scaffold
- TurboQuant KV cache compression scaffold (experimental, feature-gated)
- Support raw:true for pre-tokenized ChatML prompts
- Dynamic ctx_size on model swap (like batch_size)
- Dynamic batch_size on model swap
- Ollama name mapping and proper VRAM unload on model swap
- KV cache quantization (--cache-type-k, --cache-type-v)
- Dynamic model swap — load different models at runtime via API
- GGUF metadata patcher for Ollama compatibility
- Fix /api/tags and add format:"json" constrained decoding
- Add `eullm import-ollama` command for testing parity
- Add logging for num_predict cap and request params
- Add stress test with parallelism verification
- Support Ollama num_ctx/num_predict semantics
- Enable flash attention and n_batch for faster single-request inference
- Normalize bench.sh for fair comparison (long prompt + 16 concurrent)
- Support think:false parameter to disable Qwen3 thinking mode
- Add multi-sequence batching benchmark script
- Add interactive chat REPL to `eullm run`
- Upgrade CUDA build to 13.2 and add Blackwell architectures
- Add CUDA 12.8 build to release workflow for NVIDIA GPU support
- Add CI and release workflows for Engine binary distribution
- Add continuous batching scheduler for multi-request inference
- Dockerize all components (Engine, Forge, Hub)
- Add SSE streaming on all generation endpoints
- Implement real registry, persistent audit, Hub downloads + update all docs
- Integrate llama.cpp for real GGUF inference
- Universal notebook and unified forge CLI command
- Implement Forge pipeline and port detection
- Add verticalizzazione pipeline, demo models, and compression profiles
- Implement functional CLI skeleton with mock model management
- Create project directory structure (engine, forge, hub)

### Fixed
- Handle mixed TurboQuant KV cache types with graceful fallback
- Use sp.* fields in all GenerateRequest initializations
- Remove --host flag, engine doesn't support it
- Use 'run' subcommand instead of --model flag
- Detect engine OOM/crash during health check
- Default EULLM_BIN to ./eullm-tq in benchmark script
- Update start() return type to (SchedulerHandle, ModelReadyInfo)
- Add type comments for model dimension method return types
- Patch chat.h with awk brace-depth tracking instead of wrapper sed
- Simplify wrapper compat patch — no conditions, hard fail
- Comment out thinking_forced_open in wrapper instead of patching header
- Add compat patch for thinking_forced_open in fork
- Patch workspace root Cargo.toml, not engine member
- Setup-turboquant.sh now activates [patch.crates-io] automatically
- Clippy errors in codebook precision and unused variable
- Run cargo fetch before setup-turboquant.sh
- Disable GBNF grammar in raw mode to prevent GGML_ASSERT crash
- Serialize model swaps to prevent concurrent swap race condition
- Model name matching for swap and /models/ directory lookup
- Use AtomicBool shutdown flag instead of channel disconnect
- Properly shutdown old scheduler thread before model swap
- Resolve_model accepts any file path, directories, and paths without .gguf extension
- KV cache fallback checks quantized type instead of error message
- Auto-fallback to F16 KV cache when GPU rejects quantized V cache
- Use AUTO flash attention policy to prevent GPU→CPU fallback
- Revert KV cache defaults to F16 — quantized types cause GPU fallback
- Add GPU support check to scheduler startup
- Explicitly offload KV cache to GPU and cap CPU threads
- Use ctx-size as total KV cache budget instead of multiplying by batch slots
- Clippy doc_overindented_list_items in import-ollama docstring
- Prevent CI hangs from GPU-dependent pipeline steps and apt-get prompts
- Add DEBIAN_FRONTEND=noninteractive to prevent apt-get hanging in CI
- Use Q4_K_M quantization for fair Ollama comparison
- Ollama API compatibility — NDJSON streaming + missing response fields
- Daemon segfault (re-exec instead of fork) + Ollama options parsing
- Chunk prefill to avoid SIGABRT on long prompts + add daemon mode
- Add SIGABRT handler and crash diagnostics for llama.cpp assertions
- Remove Q8_0 KV cache — caused 10% performance regression
- Use output index (-1) instead of batch index for sampler
- Add Content-Type header to bench.sh curl requests
- Make bench.sh compatible with both Ollama and EULLM APIs
- Use /api/chat with think=false to disable Qwen thinking mode
- Improve bench.sh reliability and cap token output
- Recycle seq_ids in scheduler to prevent KV cache overflow
- Use strip_suffix to satisfy clippy manual_strip lint
- Sample first token after prefill to unblock decode loop
- Cancel redundant CI runs on merge
- Scheduler start() now blocks until model is fully loaded
- Limit CUDA build to sm_120 (Blackwell only) for faster iteration
- Limit CUDA architectures to reduce binary size (~940MB → ~200MB)
- Add CUDA env vars and libclang for llama-cpp-sys CUDA build
- Use macos-15 (Tahoe) runners for macOS builds
- Use macos-14 runner for x86_64 macOS build (macos-13 deprecated)
- Correct binary path in release workflow for workspace layout
- Resolve clippy and ruff lint errors for CI
- Rebrand engine to just "eullm"
- Remove all Ollama references from engine source code

### Performance
- Reduce CPU overhead between GPU decode steps
- Use Q8_0 KV cache instead of F16 for lower memory bandwidth
- Reuse LlamaBatch instead of allocating per token

## 0.3.1 — 2026-03-30

### Added
- Full Ollama-compatible sampling parameters
- GPU scaling + cost savings charts, update README with TurboQuant showcase
- TurboQuant benchmark charts and results
- Auto-probe max ctx_size for non-TurboQuant cache types
- TurboQuant benchmark script and orchestrator

### Fixed
- Use sp.* fields in all GenerateRequest initializations
- Remove --host flag, engine doesn't support it
- Use 'run' subcommand instead of --model flag
- Detect engine OOM/crash during health check
- Default EULLM_BIN to ./eullm-tq in benchmark script

## 0.2.98 — 2026-03-29

### Added
- Show KV cache memory estimate in startup banner
- Show TurboQuant active status in startup banner
- Display TurboQuant KV cache type names in startup banner

### Fixed
- Update start() return type to (SchedulerHandle, ModelReadyInfo)
- Add type comments for model dimension method return types

## 0.2.97 — 2026-03-29

### Added
- Wire spiritbuun CUDA fork as TurboQuant backend
- TurboQuant feature naming, startup logging, strict mode
- TurboQuant backend integration scaffold
- TurboQuant KV cache compression scaffold (experimental, feature-gated)

### Fixed
- Patch chat.h with awk brace-depth tracking instead of wrapper sed
- Simplify wrapper compat patch — no conditions, hard fail
- Comment out thinking_forced_open in wrapper instead of patching header
- Add compat patch for thinking_forced_open in fork
- Patch workspace root Cargo.toml, not engine member
- Setup-turboquant.sh now activates [patch.crates-io] automatically
- Clippy errors in codebook precision and unused variable
- Run cargo fetch before setup-turboquant.sh
- Disable GBNF grammar in raw mode to prevent GGML_ASSERT crash

## 0.2.96 — 2026-03-27

### Added
- Support raw:true for pre-tokenized ChatML prompts
- Dynamic ctx_size on model swap (like batch_size)
- Dynamic batch_size on model swap
- Ollama name mapping and proper VRAM unload on model swap

### Fixed
- Serialize model swaps to prevent concurrent swap race condition
- Model name matching for swap and /models/ directory lookup
- Use AtomicBool shutdown flag instead of channel disconnect
- Properly shutdown old scheduler thread before model swap
- Resolve_model accepts any file path, directories, and paths without .gguf extension
- KV cache fallback checks quantized type instead of error message
- Auto-fallback to F16 KV cache when GPU rejects quantized V cache
- Use AUTO flash attention policy to prevent GPU→CPU fallback
- Revert KV cache defaults to F16 — quantized types cause GPU fallback
- Add GPU support check to scheduler startup
- Explicitly offload KV cache to GPU and cap CPU threads
- Use ctx-size as total KV cache budget instead of multiplying by batch slots

## 0.2.95 — 2026-03-26

### Added
- KV cache quantization (--cache-type-k, --cache-type-v)
- Dynamic model swap — load different models at runtime via API

## 0.2.92 — 2026-03-26

### Added
- GGUF metadata patcher for Ollama compatibility

## 0.2.9 — 2026-03-25

### Added
- Fix /api/tags and add format:"json" constrained decoding
- Add `eullm import-ollama` command for testing parity

### Fixed
- Clippy doc_overindented_list_items in import-ollama docstring

## 0.2.8 — 2026-03-24

### Added
- Add logging for num_predict cap and request params
- Add stress test with parallelism verification
- Support Ollama num_ctx/num_predict semantics
- Enable flash attention and n_batch for faster single-request inference
- Normalize bench.sh for fair comparison (long prompt + 16 concurrent)
- Support think:false parameter to disable Qwen3 thinking mode
- Add multi-sequence batching benchmark script

### Fixed
- Prevent CI hangs from GPU-dependent pipeline steps and apt-get prompts
- Add DEBIAN_FRONTEND=noninteractive to prevent apt-get hanging in CI
- Use Q4_K_M quantization for fair Ollama comparison
- Ollama API compatibility — NDJSON streaming + missing response fields
- Daemon segfault (re-exec instead of fork) + Ollama options parsing
- Chunk prefill to avoid SIGABRT on long prompts + add daemon mode
- Add SIGABRT handler and crash diagnostics for llama.cpp assertions
- Remove Q8_0 KV cache — caused 10% performance regression
- Use output index (-1) instead of batch index for sampler
- Add Content-Type header to bench.sh curl requests
- Make bench.sh compatible with both Ollama and EULLM APIs
- Use /api/chat with think=false to disable Qwen thinking mode
- Improve bench.sh reliability and cap token output

### Performance
- Reduce CPU overhead between GPU decode steps
- Use Q8_0 KV cache instead of F16 for lower memory bandwidth
- Reuse LlamaBatch instead of allocating per token

## 0.2.5 — 2026-03-23

### Added
- Add interactive chat REPL to `eullm run`
- Upgrade CUDA build to 13.2 and add Blackwell architectures
- Add CUDA 12.8 build to release workflow for NVIDIA GPU support
- Add CI and release workflows for Engine binary distribution
- Add continuous batching scheduler for multi-request inference
- Dockerize all components (Engine, Forge, Hub)
- Add SSE streaming on all generation endpoints
- Implement real registry, persistent audit, Hub downloads + update all docs
- Integrate llama.cpp for real GGUF inference
- Universal notebook and unified forge CLI command
- Implement Forge pipeline and port detection
- Add verticalizzazione pipeline, demo models, and compression profiles
- Implement functional CLI skeleton with mock model management
- Create project directory structure (engine, forge, hub)

### Fixed
- Recycle seq_ids in scheduler to prevent KV cache overflow
- Use strip_suffix to satisfy clippy manual_strip lint
- Sample first token after prefill to unblock decode loop
- Cancel redundant CI runs on merge
- Scheduler start() now blocks until model is fully loaded
- Limit CUDA build to sm_120 (Blackwell only) for faster iteration
- Limit CUDA architectures to reduce binary size (~940MB → ~200MB)
- Add CUDA env vars and libclang for llama-cpp-sys CUDA build
- Use macos-15 (Tahoe) runners for macOS builds
- Use macos-14 runner for x86_64 macOS build (macos-13 deprecated)
- Correct binary path in release workflow for workspace layout
- Resolve clippy and ruff lint errors for CI
- Rebrand engine to just "eullm"
- Remove all Ollama references from engine source code
