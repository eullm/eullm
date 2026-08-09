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

## 0.6.80 — unreleased

*Accumulating; published only as `EuLLM-v0.6.80-rc*` pre-releases so far.*

### Added
- **Tool calling on the OpenAI endpoint** (#334). `tools` and `tool_choice`
  in a `/v1/chat/completions` request now reach the model: the prompt is
  rendered through the model's own chat template with the request's full
  OpenAI-format JSON (tool definitions, `tool` role messages, and
  assistant `tool_calls` in the history all survive), and the raw output
  is parsed back with llama.cpp's format-aware parser — the same one
  llama-server uses — into structured `tool_calls`, `content`, and
  `reasoning_content`, with `finish_reason: "tool_calls"` when the model
  called something. Reasoning arrives in `reasoning_content` on this path,
  which is what clients render as a separate thinking box. Works on models
  whose GGUF embeds a chat template (Qwen3 family and most modern
  releases); without one, tools are ignored with a logged warning, as
  before. One knowing limit: tool requests are parsed whole, so a
  streaming request gets its answer as a single delta rather than
  token-by-token — incremental tool-call streaming is separate future
  work. From rc6, a strict parse failure no longer leaks the raw call
  markup into the reply: the parser retries in salvage mode, which
  extracts the tool calls it recognized even when surrounding text
  confused the strict grammar (reported live in #334 on a second-round
  call, "unparsed peg-native output"). rc8 adds the last line of defense
  for the case where both parse modes reject a reply whose call block is
  perfectly readable — reproduced byte for byte from the #334 report: a
  format-agnostic extractor recognizes well-formed native-syntax
  `<tool_call>` blocks and returns them structured, with the surrounding
  text as content. Only a reply with no readable call at all still falls
  back to plain text.

### Fixed
- **`--fit` no longer overcharges KV on hybrid-SSM models, so far more
  layers reach the GPU at large contexts.** Found by a user reading nvtop:
  at `--ctx-size 262144` on Qwen3.6-35B the sizer used ~6 GiB of a 16 GiB
  card and left the rest idle. The sizer charged every layer a full KV
  slice, but hybrid models pay KV only on their full-attention layers (one
  in four on Qwen3.6, `full_attention_interval` in the GGUF header); the
  other layers carry fixed-size recurrent state. The per-layer and
  total-KV estimates now scale by that cadence, on both the dense split
  and the MoE sizing path. Classic transformers are unaffected. Measured
  on the 35B MoE at the 262144 extreme: from ~11 tok/s with the GPU idle
  to ~40 tok/s with the card 89% packed. Measured equilibrium for a 16
  GiB card, for reference: 32768 context with q8_0 KV runs at ~45
  chunk/s, and a 5900-token answer completes without truncation. From
  rc7 the paying-layer count is exact instead of averaged — an offloaded
  block can hold one more attention layer than the mean (a block of 22
  with cadence 4 holds 6, not 5.5), and that half-slice under-charge was
  eating ~0.5 GiB of the safety margin at large contexts. From rc8 the
  discount also applies to hybrid GGUFs that ship WITHOUT the explicit
  cadence key: llama.cpp hardcodes the default of 4 for the qwen35 family
  and qwen3next before even reading the key, and real models rely on that
  (Ornith-1.0-35B, arch `qwen35moe`, has no such key at all), so the
  sizer now reads `general.architecture` and applies the same default.

- **Normal vertical spacing between blocks in the chat UI.** Headings,
  lists, tables and code blocks were surrounded by up to three times the
  intended air, and a blank line between numbered list items visibly
  restarted the numbering. Two causes stacked on top of the elements' own
  margins, and both are fixed: the source's blank lines next to a block
  element rendered as visible gaps (the message body preserves newlines,
  which is what separates plain-text paragraphs), and the renderer's own
  join put a newline between every pair of elements, which between two
  blocks — two list items included — rendered as one more empty line.
  Fixed in two steps: rc3 removed the first cause, rc4 the second, which
  real output showed was the dominant one. Blank lines between plain
  paragraphs still render exactly as before.

- **Markdown tables render as tables in the chat UI** (#335). They used to
  come out as plain text with visible pipes. GFM syntax — a header row, a
  `|---|` separator, optional `:` alignment markers — now produces a real
  table, with wide ones scrolling inside their own box instead of
  stretching the page. Tables inside code blocks are left alone.

## 0.6.70 — 2026-08-08

*Accumulated through pre-releases `rc1`–`rc21`, each validated on real
hardware as it landed: the dynamic chat template across every locally
available model family (Qwen3/3.6 dense and MoE, QwQ, gemma-4 including
vision, DeepSeek-R1 distills), `--fit` auto-sizing on big-vocabulary and
MoE models, and the reasoning-mode toggle end to end.*

### Added
- **Switching model mid-conversation now tells the new model it is not the
  old one — once, at the switch point.** The web UI keeps the conversation
  across a model switch, and the new model reads the previous model's
  turns as its own words: observed live, gemma-4 introduced itself as Qwen
  "to be consistent with my previous answer" (its own reasoning said so).
  The UI now records the switch as a single system turn inserted into the
  history *at the point where it happened* — naming who wrote the earlier
  replies and inviting the new model to answer as itself — plus a visible
  divider in the transcript. Nothing is repeated on later prompts, the
  conversation is never cleared or compressed, and the history before the
  switch stays byte-identical so prefix KV reuse keeps working. Flipping
  the dropdown back and forth without sending coalesces the pending note
  (and cancels it when returning to the model the conversation was already
  on).

- **The web UI's "Reasoning mode" checkbox now actually works on the
  dynamic-template path.** The Settings checkbox (and the API's
  `think: false`) was silently ignored for any model rendered through its
  own GGUF-embedded template — the Jinja render never received the flag.
  It now maps to llama.cpp's `enable_thinking` template input: models
  whose template has a reasoning toggle (the Qwen3 family) render their
  official suppression form (Qwen3.6 emits the same pre-closed empty
  `<think>` block the hardcoded ChatML fallback injects by hand);
  always-reasoning models (DeepSeek-R1, QwQ) have no suppressed form and
  ignore it, as before. Checkbox label updated to say exactly that.
  Reasoning stays ON by default — suppressing it on models that need it
  degrades answers, and unchecking is one click for the models where it
  works.

### Added
- **`--fit` now works on `serve`, and on every model swap — without ever
  prompting.** Found live: `run --fit` sized a dense 27B at 43/64 layers,
  then switching to the 22 GB MoE from the web UI loaded it with those
  same launch settings — no expert offload, wrong split — and OOM'd
  ("Failed to load model: null result from llama cpp"). The `--fit`/
  `--fit-strict` flags moved into the shared `RuntimeOpts` (so `serve` has
  them too), and `api::swap_model` now runs the same sizing before every
  load — the initial lazy load on `serve` and every API-triggered swap on
  both commands — after the old model is unloaded, so the measured free
  VRAM is real. MoE auto-sizing resolves silently as always; a dense
  partial split is applied and logged instead of asked about (a daemon has
  nobody at the keyboard — `serve` started from a shell is still a TTY, so
  the interactive gate alone would have blocked it); `--fit-strict`
  surfaces as an error to the API caller instead of a question. The server
  also now inherits the user's original `--gpu-layers`/`--cpu-moe`/
  `--n-cpu-moe` flags rather than the values a launch-time fit computed
  for the first model.

### Fixed
- **Reasoning models no longer get truncated mid-think by the default
  response cap.** Validating the reasoning toggle on real hardware,
  Qwen3.6-35B spent ~2000 tokens thinking about a hard question and hit
  the web UI's 2048 max-tokens default before answering at all. The
  server's own default was already correct (unlimited, clamped to the
  remaining context, matching Ollama's `num_predict=-1`) — but the web UI
  always sent its fixed 2048 on top of it, and the `--cli` REPL had the
  same 2048 default of its own. Both now default to unlimited: the web
  Settings field reads "0 = unlimited" and omits the cap from the request,
  and `/maxtokens 0` in the REPL restores unlimited after setting a cap.
  The context window (`--ctx-size`) remains the real bound — reasoning
  models benefit from raising it beyond the 4096 default; previous-turn
  reasoning was already stripped from the resent history, so the window is
  spent on the current turn only.

- **Reasoning no longer leaks as plain text with a dangling `</think>` on
  the dynamic-template path.** Found on real hardware the day the dynamic
  template reached the batching path (rc16): both Qwen3.6 models answered
  in the web chat with their whole reasoning as ordinary body text ending
  in a bare `</think>` — no Reasoning box. Their template *pre-opens* the
  thinking block in the prompt, so the model starts mid-think and emits
  only the closing tag, leaving clients nothing to key on. The dynamic
  path now strips a pre-opened thinking tag from the rendered prompt's
  tail (llama.cpp reports the template's tag delimiters), so the model
  emits the opening tag itself and the full block stays in the output —
  the same deliberate deviation, borrowed from Ollama, that the hardcoded
  DeepSeek-R1 template has always documented and applied.

- **The dynamic GGUF chat template now works with continuous batching too —
  the web/API and CLI paths finally build prompts the same way on every
  loading path.** Found on QwQ-32B-Preview: asked "ciao come ti chiami?"
  via the web chat it answered as an OpenAI assistant and leaked a literal
  `<|im_start|>` into the visible reply, while the same question via
  `--cli` on the same running binary answered cleanly. The web/API path
  only rendered the model's own embedded Jinja template in sequential
  mode; with the batching scheduler (the default — even `--batch-size 1`
  runs it) it silently fell back to the hardcoded name-detected template,
  which for QwQ meant bare ChatML without the default system turn the
  model was trained to expect. The scheduler now shares its model with
  API/CLI threads for template rendering (read-only, the same pattern
  llama-server uses: HTTP threads render prompts while slots decode),
  through a weak reference so an in-flight request can never pin a
  swapped-out model's VRAM. Both `build_chat_prompt` (web/API) and
  `build_cli_prompt` (`--cli`) now try the embedded template first on
  both backends and fall back to the hardcoded family template only when
  the GGUF has none.

- **`--fit` failed outright on big-vocabulary models — including the one
  MoE model the new auto-sizing was built for.** Found on real hardware
  immediately after rc14: picking Qwen3.6-35B-A3B from the menu printed
  "--fit could not size the model: could not parse layer count", fell back
  to `--gpu-layers all`, and OOM'd — the exact failure `--fit` exists to
  prevent. The file's layer count was present and readable; the parser read
  only the first 8 MiB of the file and gave up wholesale when the metadata
  ran past that — and this model's 248k-token vocabulary alone overruns it.
  The header parser now keeps what it has already read when the buffer ends
  (the layer count and attention dims sit well before the tokenizer block,
  which is also why it now stops as soon as it has them), and the MoE
  tensor-layout reader — which genuinely needs the full metadata span,
  since the tensor table sits after it — retries with larger read budgets
  instead of failing. The MoE sizing decision also moved *before* the
  "doesn't fit, continue?" prompt: it always resolves to a loadable
  configuration, so there is nothing left to ask — previously the prompt
  quoted a whole-layer split that the MoE step was about to override.
  Confirmed not MoE-specific before release: dense Qwen3.6-27B (same 248k
  vocabulary) failed identically on rc14 — same root cause, same fix; its
  `qwen35.block_count` sits at key 17, twenty keys before the tokenizer
  arrays that overrun the buffer, and the suffix-based key matching is
  architecture-agnostic so the hybrid-SSM `qwen35` arch needs no special
  handling.

### Added
- **`--fit` now auto-sizes MoE expert offload too, not just whole GPU
  layers.** Previously `--fit` (on by default from the interactive picker)
  only decided how many *whole* transformer layers fit on the GPU — it had
  no notion of MoE expert tensors, so a large mixture-of-experts model could
  be judged "doesn't fit" and prompt to abort, or worse be judged "fits" and
  then OOM at load, even though `--cpu-moe`/`--n-cpu-moe` would have let it
  run. `--fit` now parses the GGUF's tensor-info section (real per-tensor
  byte sizes from consecutive tensor offsets, not a type/shape guess) to
  split each layer's weight into expert vs. non-expert bytes, and — when the
  user hasn't already chosen `--cpu-moe`/`--n-cpu-moe` themselves —
  automatically computes the minimum number of layers whose experts need to
  move to CPU RAM for the rest to fit fully on GPU. If even every expert on
  CPU RAM still doesn't leave room for the non-expert weights, it falls back
  further to a reduced whole-layer split for those too (down to fully CPU in
  the extreme case) — the model always loads, never a size-related OOM, just
  possibly slower. Implements roadmap item `0.7-E`.

### Fixed
- **The default math-formatting hint is gone for good, not just moved.**
  rc12 tried to keep it on by default by folding it into the outgoing user
  turn instead of a `system`-role message, on the theory that the system
  role itself was the trigger. Real-hardware testing disproved that: the
  identical question on the identical model (DeepSeek-R1-Distill-Qwen-14B)
  still hallucinated — this time inventing the name "MathAI" — with the
  hint riding along in the user turn (prompt token count confirmed it: 57
  tokens web vs. 16 tokens `--cli` for the same question). The common factor
  across both failed attempts was appending unsolicited instructions to a
  short, unrelated prompt, not which role carried them. There is no default
  nudge anymore in either place; it's opt-in only, typed into the system
  prompt field in Settings.
- **The browser chat's default system message broke every model tested
  except the one it was implicitly tuned for.** After the previous fix made
  `--cli` and the browser chat share the same template decision, real
  hardware testing surfaced a browser-only regression: with the identical
  question ("ciao come ti chiami") and the identical model, `--cli` answered
  correctly while the browser chat did not — ruling out the chat template
  itself and pointing at the one thing still different between the two.
  That was the browser's always-on default system message, a LaTeX
  formatting hint. On DeepSeek-R1-Distill-Qwen-14B it produced an entire
  unrelated calculus derivation instead of a greeting — DeepSeek's own
  model card recommends against any system prompt for R1 models, and an
  atypical one appears to send them into a stereotyped reasoning trace from
  training instead of engaging with the actual turn. On Qwen2-VL-2B and
  gemma-4-e4b it produced unrelated hallucinated identity claims. The
  browser chat now starts with no default system message, matching the
  CLI; the LaTeX hint is still available, opt-in, from Settings.
- **`eullm run --cli` answered differently than the browser chat for the
  identical model and question.** The dynamic GGUF chat template added
  earlier in this version only reached the web/API chat handlers; the
  terminal chat built its prompt exactly as before, always through the
  hardcoded per-family template. Two doors onto the same loaded model
  deciding differently depending only on which one was used to ask. `--cli`
  now goes through the same decision (`build_cli_prompt`, mirroring
  `api::routes::build_chat_prompt`): the model's own embedded template first
  in sequential mode, the hardcoded fallback otherwise — identically to the
  browser chat.
- **A context that barely fit at load time could crash outright — not just
  fail cleanly — on the very first real request.** Found on real hardware
  running rc8: `--ctx-size 65536` reduced to 4096 with no warning of
  anything unusual, and the first message crashed the process (a llama.cpp
  `GGML_ASSERT`, not the clean "does not fit" error this probe exists to
  produce). Re-running the identical command landed on a smaller size
  instead and worked — pointing at free VRAM fluctuating slightly between
  runs, with the probe having accepted a candidate that left nothing to
  absorb that. `probe_and_shrink_context` now requires at least 12% of the
  GPU's memory to stay free after the probe's own context is allocated, not
  just a successful allocation, rejecting a knife-edge fit the same way it
  already rejects an outright failure. It also no longer settles for the
  first size that clears that bar: plain halving from a large request can
  land far below what's actually usable (65536 down to 16384 skips
  everything in between), so it now refines upward from there in
  1024-token steps to recover as much of that middle ground as still fits
  with margin.
- **A multimodal model's load-time context probe undersold what an ordinary
  text message needs.** The previous fix (below) made the probe use the same
  batch size a real *image* request needs, since that's usually smaller than
  the general text batch — but a model loaded with an mmproj still receives
  plain text-only messages too, and those go through the ordinary
  `generate`/`generate_streaming` path, whose batch is `--n-batch` capped at
  1024, not the smaller image-sized one. Found immediately on real hardware:
  `--ctx-size 65536` reduced clean to 4096, then the very first text-only
  message (no image attached) failed with the OOM the probe exists to catch,
  while a follow-up message with an image went through fine. The probe now
  checks the larger of the two batch sizes a loaded multimodal model can
  actually be asked to serve, not just the image-request one.

### Added
- **Chat models that ship their own template in the GGUF now use it,
  instead of a name-based guess.** Comparing eullm's answers against
  llama-server's for the same `gemma-4-12b-q8` model turned up a real
  correctness bug: the file's actual chat template — read from its GGUF
  metadata — is a reasoning-channel, tool-calling format completely unlike
  Gemma's own `<start_of_turn>`/`<end_of_turn>` markers, but eullm's Gemma
  detection (matching on the model name) built a plain Gemma-shaped prompt
  regardless. The model still answered, because LLMs are forgiving of a
  slightly-off prompt, but not the way it was actually instruction-tuned —
  and it explains the `<|channel|>`/`<|message|>` marker leakage the harmony
  filters (0.6.69) were already band-aiding. Sequential-mode requests (any
  model without continuous batching active, which includes every multimodal
  model today) now render through llama.cpp's own Jinja engine reading the
  GGUF's embedded template — the same mechanism llama-server uses by
  default — and fall back to eullm's own per-family templates only when a
  model has no embedded template at all. Continuous-batching requests are
  unchanged for now: the scheduler runs the model on its own thread and
  doesn't expose it to this code path yet.

### Fixed
- **A multimodal model no longer reserves a compute buffer sized for a
  2048-token image when the image itself needs a few hundred.** The fix above
  (probing with the same batch a real image request uses) exposed a second,
  pre-existing sizing problem: that batch defaulted to the general text
  prefill batch (`--n-batch`, 2048), not to how many tokens an image actually
  encodes to. Found immediately after shipping the probe fix, on the same real
  hardware: a 12B Q8 vision model that needs its context reduced all the way
  to 1024 to fit, even though Gemma 4's own projector output for that image
  was ~266 tokens — nowhere near 2048. `EULLM_IMAGE_MAX_TOKENS` still raises
  this explicitly for higher-resolution images; absent that, the floor is now
  512 (comfortably above a typical single image slice) instead of following
  the text batch size upward. Every multimodal model gets a meaningfully
  larger usable context as a result.
- **The context probe at load time now proves what a real image request will
  actually need, not a smaller stand-in.** `generate_multimodal` sizes its
  batch/micro-batch to fit a whole image in one pass — larger than the plain
  text batch the load-time probe (added earlier in this same pre-release) was
  using. Found on real hardware, on rc4: a 12B Q8 vision model loaded clean at
  `--ctx-size 4096` — the probe passed — and then the same OOM the probe
  exists to catch anyway on the first message with an attached image, because
  that request's compute buffer was sized differently than the one just
  proven to fit. The probe now uses the same multimodal batch sizing as the
  real request whenever an mmproj is configured, so a load-time pass means an
  image request will actually go through.

### Changed
- **Bumped the pinned `llama.cpp` from a commit six weeks old to tag
  `b10200`, current as of 2026-07-30.** Brings every upstream fix and model
  architecture addition from that window. Three C-API breaks needed porting
  in the vendored Rust wrapper: `use_mlock`/`use_mmap` became a single
  `load_mode` value (no user-visible change — eullm never overrides either
  flag), the multimodal input struct gained a required length field, and a
  multimodal helper's return type changed shape internally. None of this
  changes any flag, default, or observable behaviour; it keeps eullm current
  with upstream instead of falling further behind.

### Fixed
- **A context that will not fit is caught at load, and shrunk automatically
  instead of failing on the first message.** The sequential engine — every
  multimodal model, and anything run with `--batch-size 0` — creates its
  context on the first request rather than at load, so an oversized
  `--ctx-size` printed "Model loaded successfully" and only failed once a chat
  message actually asked for the KV cache. Found running a 12B Q8 vision model
  plus its projector on a 16 GB card: `--ctx-size 4096` loaded clean and then
  refused every message, and `--cache-type-k/-v q8_0` did nothing about it —
  Gemma 4's mixed sliding-window architecture forces f16 regardless of what is
  asked for, so that flag was never the lever here. The context is now proven
  by allocating it once during load; a size that does not fit is halved and
  retried until one does, with the reduction and the KV cost printed plainly,
  or the load fails outright if even a 512-token window will not fit. The
  startup banner reports the size actually used, not the one that was asked
  for, so the two numbers it prints — context and KV memory — always describe
  the same load.
- **The startup banner no longer claims continuous batching on a model running
  sequentially.** A multimodal model forces the sequential engine, and the log
  said so, but the banner two lines below still printed `Mode: continuous
  batching` — the corrected value never left the block that computed it. The
  same stale number was handed to the API server, so it believed it had a
  batching scheduler that did not exist.
- **The name `eullm list` shows is always a name you can run.** It printed the
  `id` recorded inside each manifest, which is not necessarily the directory
  the model lives in. A manifest edited by hand, or copied from another model,
  therefore made a model list under a name that resolves to a *different*
  model, leaving it impossible to start: `run`, `rm` and `show` all resolve the
  directory. Found on a real store where a 12B listed under a 4B's name and
  could not be launched at all. The listing now shows the directory, and the
  `id` field is advisory.
- **A model whose manifest is missing no longer disappears from `eullm list`
  without a word.** The listing counted a directory only when it held a
  readable `manifest.json` and skipped everything else in silence, so an
  interrupted pull, a restored backup or a directory copied from another
  machine left weights on disk and nothing on screen. The store this was found
  on had 12 GB of a model hidden that way. Those directories are now reported
  under the table, with the reason and how to repair them.
- **DeepSeek R1 models answer instead of declining the turn.** R1 and its
  distills are trained on DeepSeek's own chat format, and eullm was falling
  back to ChatML for them. The result was not a worse answer but none:
  `deepseek-r1-distill-14b` replied with an empty think block and end-of-turn —
  six tokens, empty content — deterministically, on every request, over the API
  and in the terminal chat alike. They now get the DeepSeek template
  (`<｜User｜>` / `<｜Assistant｜>`), matching Ollama's behaviour for the same
  models, and previous turns' reasoning is stripped from the history exactly as
  DeepSeek's own template does, so long chats do not re-feed thought that the
  model was trained never to see.

## 0.6.60 — 2026-07-30

### Changed
- **`eullm serve` now defaults to one request at a time, not eight.**
  `--ctx-size` is the total KV budget and is split evenly across batch slots,
  so the old default of 8 gave each request an eighth of the window: with the
  4096 default context, 512 tokens. A reasoning model spends that before it
  finishes thinking, so the answer stopped mid-sentence and came back as
  `done_reason="length"` — with nothing pointing at a flag, because the
  operator had never set one. `run` already defaulted to 1 and now `serve` does
  too.

  **If you serve concurrent clients, set `--batch-size` explicitly**, and raise
  `--ctx-size` with it: `--batch-size 8 --ctx-size 32768` keeps the same 4096
  tokens per slot the old default only appeared to give you. Requests beyond
  the slot count queue rather than fail.

### Fixed
- **`eullm serve` now prints the same startup diagnostics as `eullm run`.**
  `GPU backend`, `CPU features`, `GPU layers`, `Context`, `KV cache` and
  `Threads` were printed by `run` alone, so anyone driving the engine as a
  daemon — every automated harness, and everyone using it as a backend behind
  an editor or a UI — never saw which backend actually initialised or how many
  layers were offloaded. Those are the lines that diagnose a wrong-looking
  result, and the people who could not see them are the ones best placed to
  report one. `serve` starts without a model, so it prints them after each
  model load rather than at startup.
- **Two security advisories in the dependency tree, both now closed.**
  `rustls-webpki` could panic while parsing a certificate revocation list, on a
  path reached *before* the CRL's signature is verified (RUSTSEC-2026-0104), and
  `crossbeam-epoch` dereferenced an invalid pointer when formatting a null
  atomic pointer (RUSTSEC-2026-0204). Both arrive through dependencies rather
  than our own code — the first through the HTTPS client used for model
  downloads, the second through llama.cpp's Rust bindings — and both are fixed
  by the updated versions in this release. Neither is known to be triggerable
  by anything eullm does, and they were found by a check that did not exist
  before this release rather than by a report.

## 0.6.52 — 2026-07-28

### Fixed
- **The terminal chat works on multimodal models.** 0.6.51 stopped `--cli` and
  `--no-ui` from killing the engine on a model that loads in sequential mode,
  but it did so by explaining that the terminal chat was unavailable there —
  which covers every vision and audio model, so `eullm run <a vision model>
  --cli` still left you without a prompt. The chat now runs on either backend,
  so it is available for exactly the same models the API is.
- **The arrow keys work in the terminal chat and in the model picker.** Both
  prompts read the line raw, so pressing left to fix a typo printed `^[[D` on
  screen and put it in what you sent: a bewildering "Invalid choice" at the
  picker, and escape sequences inside the message at the `>>>` prompt. 0.6.50
  removed them from the value but not from the display, and left the cursor
  unable to move. Both prompts now use a real line editor: left and right move
  the cursor, backspace works anywhere in the line, and up and down recall what
  you typed earlier in the session. That history is kept in memory only and is
  never written to disk. Ctrl+C discards the line being typed instead of killing
  the engine; Ctrl+D quits, as does `/bye`.
- **Asking for the terminal chat and not getting it is never silent.** Only two
  things can stop it now — no model loaded, or a standard input that is not a
  terminal — and each says which.

## 0.6.51 — 2026-07-28

### Fixed
- **`--cli` and `--no-ui` no longer make the engine exit immediately.** On a
  model that loads in sequential mode — every multimodal model, and anything
  run with `--batch-size 0` — asking to stay in the terminal printed "Type a
  message to chat" and then quit without a word, taking the API server with it.
  The terminal chat needs the batching scheduler, which those models do not
  have. The engine now stays up and serves the API, and says plainly that the
  terminal chat is unavailable for this model instead of promising it. It still
  does not give you the terminal chat on those models — 0.6.52 does that.
- **A context that does not fit says what did not fit.** Asking for a large
  `--ctx-size` and getting `Failed to create context: null reference from
  llama.cpp` told you nothing: the window was the thing that failed, and its
  cost was on screen two lines earlier. The error now names the window, the
  memory its KV cache needs, and the two flags that change it. Seen with
  `--ctx-size 131072` on a 4B model, where the cache alone wants about 17 GB.

## 0.6.50 — 2026-07-28

### Fixed
- **The arrow keys no longer break the model picker.** Pressing left to correct
  a typo at the `Choice >` prompt inserted `^[[D` into the line and answered
  "Invalid choice" for what looked blank. Those keys are now ignored. This is
  not line editing: the cursor still cannot be moved, but a keystroke that does
  nothing no longer breaks the input it lands in.
- **A download no longer goes silent partway.** The projector was fetched
  without a progress counter, so a pull sat for the best part of a minute
  between announcing the file and finishing, with nothing on screen. It reports
  progress like any other download, and its line is closed before the next
  message rather than being written over.

## 0.6.49 — 2026-07-28

### Added
- **Pulling a vision model from HuggingFace brings its projector too.** A
  catalog model already did this; one pulled by repo name did not, so the
  weights arrived without the file that lets the model see, and you had to
  notice the second file yourself and pass `--mmproj`. `eullm pull
  hf.co/owner/repo` now fetches both, the same as llama.cpp's `-hf`. If the
  projector download fails the model is still usable for text, and the warning
  says so.

### Fixed
- **A projector is no longer mistaken for the model itself.** It is a `.gguf`
  in the same repo, so on a vision repo a plain pull saw two candidates and
  refused as ambiguous, and asking for `:F16` could download `mmproj-F16.gguf`
  as the weights, which then failed to load with an error about the file
  rather than about the choice.

## 0.6.48 — 2026-07-28

### Added
- **A Vulkan binary, `eullm-linux-x64-vulkan`.** Until now the published GPU
  builds were NVIDIA only, which left out every AMD and Intel GPU — including
  the unified-memory laptops and mini PCs whose integrated graphics can address
  far more memory than a consumer discrete card. Vulkan needs a driver on your
  machine (mesa RADV, amdvlk, NVIDIA, Intel ANV) and `libvulkan.so.1`; nothing
  is bundled, unlike the CUDA builds which ship their runtime. First community
  run: a Ryzen AI 9 HX 470 with Radeon 890M, all layers offloaded.

  Two releases announced this binary before one carried it. 0.6.46 failed to
  build it, and 0.6.47 built it and then did not attach it, because the list of
  files to publish was maintained by hand and nobody had added a line. Its
  checksum was in `checksums.txt` both times, which is how the second one was
  spotted. The release now publishes whatever was built rather than a list
  someone has to remember to update.

## 0.6.47 — 2026-07-28

### Added
- **A projector next to the weights is found on its own, and `--mmproj` names
  one that is not.** Vision and audio models only worked when pulled from the
  catalog: the projector was looked up by model id inside the model store, so a
  GGUF you downloaded yourself could never be multimodal, whatever sat beside
  it. A file called `mmproj*.gguf` in the same folder as the weights is now
  used automatically — the layout every HuggingFace vision repo ships — and
  `--mmproj <path>` covers the case where the two live apart. Available on both
  `run` and `serve`.

### Fixed
- **Asking a text-only model for an image now says what to do about it.** The
  refusal read "engine is in batched (text-only) mode", which is true and
  useless: it named an internal mode rather than the missing projector. It now
  names the model, and says both ways to get one.
- **A model you pulled yourself now appears in the model lists.** Both
  `/v1/models` and `/api/tags` were assembled from the built-in catalog and
  whatever was loaded at that moment, so a model downloaded from a URL or a
  HuggingFace repo was invisible to them. On `/v1/models` that is the
  difference between usable and not: a coding editor offers the models that
  endpoint names, so one it never names cannot be selected at all. Reported by
  a user whose pulled 35B ran fine in the chat UI and could not be reached from
  the editor.

## 0.6.46 — 2026-07-28

### Added
- **Image and audio input now work on a build from source.** Multimodal is a
  default feature, so `cargo build --release --features vulkan` (or cuda, or
  rocm) gets it without asking. Every published binary already had it; only
  hand-built ones did not.

### Fixed
- **A build that cannot read media says so instead of ignoring it.** Attaching
  a photo to a binary compiled without multimodal support used to drop the
  image on the way in and pass the question through as plain text, so the model
  answered that it could not see any image — which reads as the model's
  limitation rather than the binary's. It is now an explicit error naming what
  is missing.
- **The startup banner no longer reports less for some models than others.**
  The KV cache size and the "this model was trained for N tokens" hint were
  produced only by the batching loader, so for multimodal models and for
  `--batch-size 0` both lines were simply absent, with nothing saying why.

## 0.6.45 — 2026-07-28

### Fixed
- **Gemma replies no longer end with a stray `</start_of_turn>`.** The model
  sometimes closes a turn by writing that tag as ordinary text instead of
  emitting the end-of-generation token, and only the plain `<end_of_turn>`
  spelling was being watched for, so the closing form was passed through to
  you. Seen at the end of an audio transcription; it affects text replies the
  same way. Both closing spellings now end the turn.
- **The reported KV cache memory was half the real figure on Qwen3 models.**
  The startup banner works out how much memory the context window costs, and it
  derived a value the model can declare for itself. On Qwen3 the two differ by a
  factor of two, so the banner promised 112 MB where 224 MB was allocated. It
  now reads what the model declares. The cache itself was always the right size;
  only the number shown to you was wrong, and it was wrong in the direction that
  invites choosing a context window that does not fit.
- **The banner says when the context window is far below what the model can
  hold.** The default is 4096 tokens, models are commonly trained for ten times
  that, and nothing connected a plugin running out of room to the flag that
  fixes it. When the window is below half the model's, the banner now says so
  and names `--ctx-size`.
- **Starting the browser no longer prints a wall of errors.** On a machine with
  no graphical browser, the desktop handler reports every fallback it tried,
  which landed seven `command not found` lines immediately after the banner said
  the engine was ready. A failure to open now costs one line, and the chat URL
  is printed either way.

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
