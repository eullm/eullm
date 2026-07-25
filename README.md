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
  <img src="https://img.shields.io/badge/Engine-v0.6.29-2ea44f" alt="Engine status" />
  <img src="https://img.shields.io/badge/Forge%20%2B%20Hub-Early%20development-orange" alt="Forge/Hub status" />
  <a href="https://github.com/eullm/eullm/actions/workflows/ci.yml"><img src="https://github.com/eullm/eullm/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://doi.org/10.5281/zenodo.20412979"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20412979.svg" alt="DOI" /></a>
</p>

<p align="center">
  🇪🇺 European-built — focused on local-first and sovereign AI &nbsp;·&nbsp; 🇮🇹 Developed in Italy
</p>

---

> ### 🇪🇺 Proven: a 35B-parameter model, fully local, on EU-available ARM hardware, no GPU
>
> A 35B-parameter hybrid MoE model (`qwen3.6-35b-a3b`, ~3B active params/token)
> running entirely on CPU on a **Radxa Orion O6 (CIX P1 SoC, Armv9.2-A,
> 12-core, ~20W board power)** — no GPU, no NPU, consumer-grade EU-available
> hardware:
>
> - **~9-11 tok/s decode**, sustained across real multi-turn conversations
>   and multiple topics
> - **Multi-turn KV-cache reuse confirmed at 100% exact match** — every turn
>   reuses the *entire* prior turn's resident state, verified across 6+
>   consecutive turns at both 4096 and 16384-token context, with both F16 and
>   Q8_0 KV cache — not a lucky first turn, a sustained, reproducible result
>
> A large, capable open model, genuinely running on sovereign, GPU-free,
> low-power EU hardware — not a toy demo on a small distilled model.

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
| 🐧 Linux ARM64 (NVIDIA) | `eullm-linux-arm64-cuda-12.8` | ✅ Tested | ARM host + discrete NVIDIA GPU (sm_86/89/120); validated on an RTX 3060 12GB ARM server, qwen3-14b Q4 at 33 tok/s |
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

The Linux x64, Windows x64, and Linux ARM64 (CUDA) binaries are validated end-to-end by the maintainer. **macOS Apple Silicon (Metal)** and **Linux ARM64 (CPU)** are now **community-validated** (see the testers below). **macOS Intel (x64)** still compiles in CI but nobody has run it on that hardware yet — it remains **Experimental — untested**.

If you run local LLMs on a Mac or an ARM64 board (Raspberry Pi 4/5, Orange Pi 5+, Rock 5B, Jetson, …), **your help validating these binaries is hugely appreciated**. See the open testing call:

→ **[Issue #140 — Help wanted: testing on macOS & ARM64 Linux](https://github.com/eullm/eullm/issues/140)** (`help wanted`, `testing`)

The remaining gap is **macOS Intel (x86_64)** — if you run local LLMs on a pre-Apple-Silicon Mac, reports with `eullm --version` output, model used, and what worked/broke are very welcome.

**Community testers — thank you 🙏** Early hands-on reports are already in (see #140):

- **[@andreyluiz](https://github.com/andreyluiz)** — macOS (Apple M2 Pro, Metal) and Raspberry Pi 400 (Cortex-A72, ARM64), with full logs. His macOS run surfaced a real packaging bug: the published `eullm-macos-arm64` had been built without Metal and silently ran on CPU — the release now builds the macOS binaries with `--features metal`. Genuinely grateful for the time he's putting into validating hardware the maintainer can't reach.

### Diagnosing garbage output or crashes (`--rust-debug`, new in v0.6.34)

If you're helping test odd hardware and see garbage output (one token repeated over and over, gibberish) or an unexplained crash, add `--rust-debug` to `eullm run` / `eullm serve`. It turns on a NaN/Inf scan of the model's logits right before every sampled token and logs loudly (`tracing::error!`) if it finds corruption — the clearest signal available today for "is this a numerical bug in the compute path" versus something else further downstream.

```bash
eullm run ./model.gguf --rust-debug
eullm serve --rust-debug
```

Off by default: the scan touches every value in the vocabulary (~100-150k floats) on every generated token, so it's real added cost most users shouldn't pay for. Turn it on only when actively chasing a bug like this.

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
  `--batch-size`, `--rust-debug`, …).

> **Running under Docker or systemd?** Don't pass `--daemon` there — the
> supervisor *is* the daemonizer. Use `docker compose up -d` or a plain
> `ExecStart=/usr/local/bin/eullm serve` unit; graceful SIGTERM handling is
> built in, so `docker stop` / `systemctl stop` shut the engine down cleanly.

### Restricting who can reach the engine (`EULLM_ALLOWED_IPS`, new in v0.6.29)

Both the API and the chat UI bind `0.0.0.0` — the engine often runs on a
different host than the things calling it (a RAG pipeline, a LAN client), so
the bind address can't be the access boundary. Instead, every request's
source IP is checked against an allowlist before it reaches any handler.

With no `.env` file present (or none of it maps to `EULLM_ALLOWED_IPS`), only
`127.0.0.1`/`::1` are allowed — functionally the same as binding loopback,
without needing a different bind address for the unconfigured case. To allow
more, copy [`.env.example`](.env.example) to `.env` in the directory you
launch `eullm` from:

```bash
# A single RAG host on another machine
EULLM_ALLOWED_IPS=203.0.113.5

# A whole LAN subnet
EULLM_ALLOWED_IPS=192.168.1.0/24
```

Loopback stays allowed on top of whatever `.env` adds — configuring a remote
host never locks out local access. A malformed entry is rejected and logged
as a warning at startup; nothing beyond loopback takes effect until it's
fixed, never the other way around.

**What this does and doesn't cover:** this closes off the network-exposure risk
of the default `0.0.0.0` bind. It does *not* authenticate anyone — a request
from an allowed IP is trusted, not challenged — and it cannot express two cases
at all: behind Docker's published ports every external client arrives as the
bridge gateway address, and a request from your own browser genuinely comes
from loopback. Those are what API keys and the origin policy below are for.

### API keys and quotas (`EULLM_API_KEYS`, new in v0.6.36)

Set a key and the API requires a bearer token; leave it unset and nothing
changes, which keeps the local single-user case as simple as it was:

```bash
# id:secret, comma-separated. `rpm=N` caps requests per minute for that key.
EULLM_API_KEYS=ci:8f3b1d9c2e7a4f60b5,rag-prod:1a2b3c4d5e6f7a8b9c:rpm=600
```

```bash
curl -H "Authorization: Bearer 8f3b1d9c2e7a4f60b5" \
     http://localhost:11434/api/tags
# `X-Api-Key: <key>` works too.
```

For real deployments prefer a file — it can be `chmod 600`, whereas an
environment variable is readable through `/proc/<pid>/environ`:

```bash
EULLM_API_KEYS_FILE=/etc/eullm/keys   # one id:secret[:rpm=N] per line
```

Three things worth knowing:

- **The key id lands in every audit record** (`user_id`), so the AI Act trail
  finally says *who* asked, not just what was asked. The id is not a secret.
- **A valid key admits the request from any source address.** Enabling keys
  replaces address-based admission with identity-based admission rather than
  stacking on top of it — otherwise the Docker case stays unfixable, since the
  bridge gateway is not in anyone's allowlist. Requests with no key or a wrong
  key get a 401 whatever their origin, loopback included. Startup logs which
  posture is in effect.
- **A key that doesn't parse is fatal at startup.** If you asked for
  authentication, serving without it because of a typo is worse than not
  starting.

Secrets go through the environment rather than CLI flags on purpose: a command
line is visible in `ps` to every local user on the box.

### Browser origins (`EULLM_ALLOWED_ORIGINS`, new in v0.6.36)

CORS used to be fully permissive, which combined badly with the IP allowlist: a
request from a page you happen to be visiting comes from loopback, so it was
allowed *and* the page could read the reply. Now any loopback origin is allowed
by default (the bundled chat UI, Open WebUI on localhost, a local frontend —
all unaffected), a cross-origin request with side effects is refused with 403
before it reaches a handler, and anything else is opt-in:

```bash
EULLM_ALLOWED_ORIGINS=https://chat.example.eu,http://192.168.7.10:8080
# `*` restores the old permissive behaviour, explicitly.
```

Requests with no `Origin` header — curl, an Ollama SDK, a RAG pipeline — are
untouched: CORS never applied to them, and breaking them would cost
compatibility for no gain.

### Web tool hardening (new in v0.6.36)

With `--web`, a URL in a prompt is fetched by the server, so on any shared
deployment the URL is attacker-controlled. The fetcher requires `https`,
refuses hosts resolving to loopback, private, link-local, carrier-NAT or
cloud-metadata addresses (including the IPv6 forms that wrap an IPv4 address),
pins the connection to the address it validated, re-checks every redirect hop,
caps the body at 4 MiB read in chunks, and accepts only textual content types.

```bash
EULLM_WEB_ALLOWED_DOMAINS=docs.example.eu,eur-lex.europa.eu  # allowlist sources
EULLM_WEB_ALLOW_HTTP=1                                       # permit plain http
EULLM_WEB_ALLOW_PRIVATE_HOSTS=1                              # intranet targets
```

The last one turns the address check off. With it set, `--web` hands whoever
writes the prompt a GET primitive on your internal network — deliberate and
logged at startup, but know what you are choosing.

### Free VRAM without restarting (new in v0.6.10)

`eullm unload` frees the currently loaded model's VRAM and leaves the server
running with an empty slot — no restart, no dropped connections. A later
request carrying a `model` field (or another `eullm run <model>`) loads a
model back in.

This is for sharing a GPU with another process that needs the VRAM
temporarily — e.g. handing a RAG pipeline's embedding model room to run
during document ingestion, then reloading the LLM once it's done:

```bash
eullm run qwen3-14b --fit --daemon    # LLM loaded, serving on :11434

eullm unload                          # frees qwen3-14b's VRAM, server stays up
#   → your embedding/reranker process now has room to load

# ... run document ingestion ...

# reload the LLM: any request carrying "model" does it automatically
curl -s http://localhost:11434/api/generate \
  -d '{"model": "qwen3-14b", "prompt": "ok"}' > /dev/null
```

`eullm unload [--port PORT]` is a thin wrapper around `POST /api/unload` (an
EULLM extension, not part of the Ollama API) — call the endpoint directly
from any language/script that already talks to the API port.

### Run MoE models on a small GPU (`--cpu-moe` / `--n-cpu-moe`, new in v0.6.11 / v0.6.13)

MoE models (Qwen3-30B-A3B, Qwen3.6-35B-A3B, …) route each token through only
a handful of experts, but the expert weights make up most of the file on
disk — `--gpu-layers`/`--fit` can only offload whole layers (all their
experts included), so a 20+ GB MoE model still needs 20+ GB of VRAM to run
mostly on GPU.

`--cpu-moe` keeps just the expert tensors (`*.ffn_(up|down|gate)_exps`) on
CPU RAM, while attention, embeddings, and — critically — the whole KV cache
stay on GPU. Since only a few experts fire per token, this trades a small
CPU-matmul cost for VRAM headroom `--gpu-layers` can't reach: a 22 GB MoE
Q4_K_M can run at near-GPU speed on a 12 GB card, as long as system RAM
holds the rest.

```bash
# Qwen3.6-35B-A3B, Q4_K_M (~22 GB) on a 12 GB GPU + enough system RAM
eullm run qwen3.6-35b-a3b --cpu-moe --fit
```

Combines with `--gpu-layers`/`--fit` (which still size the non-expert
tensors) and `--ctx-size` as usual. No effect on dense (non-MoE) models —
the tensor pattern simply matches nothing. Available on `eullm run` and
`eullm serve` (applied to every model the server loads or swaps to).

#### Finer control: `--n-cpu-moe N` (new in v0.6.13)

`--cpu-moe` is all-or-nothing — **every** expert tensor in the model moves to
CPU RAM, regardless of how much VRAM is actually free. On a card with more
headroom than the blanket flag needs, that leaves GPU memory idle and pushes
more matmuls onto the CPU than necessary, which costs tokens/sec: on a real
run (Qwen3.6-35B-A3B Q4_K_M, RTX 3060 12GB, ARM64 host), `--cpu-moe --fit`
used only ~2.5 GB of the 12 GB card and left the GPU at ~26% utilization
while all 8 CPU cores sat at 80-90% — 26.5 tok/s.

`--n-cpu-moe N` fixes that by offloading only the first `N` transformer
layers' expert tensors to CPU RAM (`blk.0` through `blk.{N-1}`); every layer
after that keeps its experts on GPU like normal. It's a direct port of
upstream llama.cpp's own `--n-cpu-moe` flag — same per-layer regex, same
semantics — so the two engines pick the same tensors for the same `N`.

```bash
# Offload only the first 12 layers' experts to CPU RAM, keep the rest on GPU
eullm run qwen3.6-35b-a3b --n-cpu-moe 12 --fit
```

**Picking `N`:** there's no auto-sizing yet (`--fit` doesn't know about
`--n-cpu-moe`, that's planned for a future release) — dial it in manually:

1. Start at `N = 0` (equivalent to no MoE offload — everything on GPU via
   `--gpu-layers`/`--fit`) and watch it fail with an out-of-VRAM error, or
   start high (e.g. `N` = total layer count, equivalent to `--cpu-moe`) and
   confirm it loads.
2. Move `N` down in steps (e.g. by 4-8 layers) and reload, watching VRAM
   usage (`nvtop` / `nvidia-smi`) after each load. As `N` decreases, more
   experts stay on GPU, VRAM usage rises, and tokens/sec should climb —
   until VRAM runs out and the load fails again.
3. The best `N` is the smallest value that still loads cleanly — that's the
   most expert weight you can push back onto the GPU without OOM-ing.

Each transformer layer's experts are roughly the same size, so as a rough
starting point: `N ≈ total_layers × (1 − free_vram_gb / moe_tensors_gb)`,
then adjust from there by trial and error per the steps above (`eullm show
<model>` prints the layer count from GGUF metadata).

`--cpu-moe` and `--n-cpu-moe` are **mutually exclusive** — passing both is a
CLI error (`--cpu-moe` is the "give me all the headroom, don't ask
questions" option; `--n-cpu-moe N` is the "I know how much I need" option).
Like `--cpu-moe`, it combines with `--gpu-layers`/`--fit` (which still size
the non-expert tensors) and `--ctx-size`, has no effect on dense (non-MoE)
models, and is available on both `eullm run` and `eullm serve` (applied to
every model the server loads or swaps to).

### KV-cache reuse on hybrid/recurrent models (Qwen3.5/3.6): a known upstream limitation, not an eullm gap

KV-cache prefix reuse (below) works correctly on hybrid attention+SSM
architectures (Qwen3.5/3.6's Gated-DeltaNet+attention design, e.g.
`qwen3.6-35b-a3b`) — but by default it can never actually *reuse* anything
on them. Every reused turn is rejected and silently falls back to a full
re-prefill of the whole conversation, because llama.cpp's recurrent-state
memory doesn't support rolling back that state at all by default. This is
easy to miss: nothing errors, the response is just correct but slow, and
the giveaway is a growing prefill cost turn over turn (watch for `reused
prefill failed ... likely a recurrent/hybrid model architecture` warnings
in the log).

We looked for a real fix rather than accepting that at face value, and
traced it end to end in upstream llama.cpp source and issue tracker
(verified directly against `src/llama-memory-recurrent.cpp`,
`src/llama-arch.cpp`, `common/common.cpp`, `tools/server/server-context.cpp`
on `ggml-org/llama.cpp`, current as of July 2026):

- llama.cpp exposes an experimental `n_rs_seq` ("recurrent-state rollback
  window") parameter, and only two architectures
  (`llm_arch_supports_rs_rollback`) — Qwen3.5/Qwen3.6 — support it at all.
  eullm exposes this as `--rs-seq N` on `run`/`serve`.
- **It is not the right tool for this job.** Upstream's own server never
  uses `n_rs_seq` for conversation/prompt caching — it derives the value
  exclusively from speculative-decoding draft length (single digits to
  low teens) and explicitly zeroes it everywhere else
  (`cparams_dft.n_rs_seq = 0`). The server's actual mechanism for
  cross-turn reuse on these architectures is a different, bounded
  feature — periodic full-state snapshots (`--ctx-checkpoints`, default
  32, spaced `--checkpoint-min-step` apart, default 8192 tokens) with a
  graceful full-reprocessing fallback when no checkpoint covers the
  request. eullm doesn't implement an equivalent yet — see below.
- **We tested `--rs-seq` on real Orion hardware anyway, and it's unsafe
  at useful values on a 35B hybrid MoE model.** Every recurrent-state
  tensor scales by `(1 + N)`
  (`n_rows = mem_size * (1 + n_rs_seq)`, confirmed in source). At `N=64`
  this pushed resident memory from ~21GB to ~44.6GB with heavy swap
  thrashing; at `N=512` it crashed the engine
  (`ggml_new_object: not enough space in the context's memory pool`,
  the same failure signature as a previously-fixed ubatch-scaling bug on
  Qwen3-Next, `llama.cpp#17578`/`#17794`, now recurring for `n_rs_seq`
  specifically). The feature's only upstream test coverage
  (`llama.cpp#25758`) merged against a small synthetic model, not
  anything at 35B scale — it simply hasn't been hardened for this yet.
- **Conclusion: leave `--rs-seq` at 0 for hybrid/recurrent models.**
  Full re-prefill on every turn is the correct, current ceiling for this
  architecture class on llama.cpp today — llama.cpp's own server hits the
  identical fallback and logs the same "forcing full prompt re-processing
  due to lack of cache data (likely due to SWA or hybrid/recurrent
  memory)" condition. This isn't an eullm shortcoming; it's an open,
  actively-worked-on upstream gap (see e.g. `llama.cpp#22384`, `#20225`,
  `#24055`, `#24785`).

`--rs-seq N` remains available (0 by default) as an experimental escape
hatch for anyone who wants to reproduce or build on this, but is not a
recommended path to KV reuse on hybrid/recurrent architectures. The
practical mitigations today: use `--ctx-checkpoints` (below), keep
conversations reasonably short, and use non-thinking mode (`--think off` /
official `preserve_thinking: false` behavior) to bound how fast the
re-prefill cost grows per turn.

### The actual root cause of small/unstable reuse: retokenizing history every turn

Real-hardware testing surfaced something sharper than "the rollback window
is too small": even a substantial reused-prefix match (661 of 1394 tokens)
was rejected outright, not partially honored — this architecture's
recurrent memory has no partial credit, only "the entire prefix matches
exactly" or "full re-prefill." That raised the question of why the
match was ever small or unstable in the first place (31/326, 322/608,
29/671 tokens across separate turns), given a continuing conversation is,
at the text level, a pure append: `build_chatml` and friends are
deterministic string concatenation with no timestamps or randomness, so
the shared prefix of turn N and turn N+1's prompts is guaranteed
byte-identical *as text*.

The gap was in what eullm did with that text: it retokenized the entire
growing prompt from scratch on every single turn (`model.str_to_token()`
over the full resent history), and independently retokenizing the same
text twice — once as originally decoded, again as part of a longer
string — is not guaranteed by BPE to land on the same token ids. On this
model that instability was severe enough to wreck the match almost every
turn. Checkpoints (below) don't help this specific failure: a checkpoint
taken at a turn boundary has the exact same content as the live slot at
that instant, so for the very next turn it can never do better than the
live slot's own (unstable) match.

**The fix:** eullm now checks whether an idle slot's cached text is an
exact, literal prefix of the new prompt *before* tokenizing anything. If
it is, the slot's already-known, already-correct tokens are reused
directly for that portion, and only the new suffix text gets tokenized —
the shared history is never retokenized at all, so BPE instability over
that portion is structurally impossible, not just tolerated. Falls back
to the previous full-tokenize + token-level longest-common-prefix matching
whenever no exact text prefix exists (a genuinely new conversation, an
edited/branching history, or a cold-started slot) — no regression versus
today's behavior in those cases. No new flag; this applies automatically
wherever prefix reuse already applied. Verified twice: first on a sandboxed
server (TinyLlama, real multi-turn conversation through the actual API
path) via a dedicated debug log line (`exact text-prefix match — reusing N
tokens without retokenizing`); then — the result that actually matters —
**on real Orion hardware against the real 35B hybrid model**, where it
resolved the rollback rejection completely: `reused N from cache`
consistently matched the *entire* previous turn's resident length across
6+ consecutive turns spanning multiple topics, at both 4096 and
16384-token context, with F16 and Q8_0 KV cache — zero `reused prefill
failed` warnings once both this fix and the one below landed together.

**A second, sharper bug behind the same symptom: `/no_think` (`eullm run
--cli`'s sticky reasoning toggle) actively corrupted history reconstruction.**
Confirmed on real hardware: with `/no_think` sticky off across several
turns, reuse degraded to a small unstable fraction (matching the pattern
above); disabling `/no_think` entirely, with *no other change*, restored
~99% reuse even on the version before the text-prefix fix. Root cause,
found by reading `interactive_chat()`: suppressing thinking mode injects a
literal `<think>\n</think>\n\n` right before the model's turn — text the
model actually decodes as part of that turn's resident state — but this
injection was never re-added when reconstructing that turn for a later
request's history, only the model's own subsequent output was stored. Every
`/no_think` turn's reconstructed text permanently diverged from what was
truly resident from that point on, compounding with each additional
`/no_think` turn — text-level, not a tokenizer quirk, and the text-prefix
fix above correctly detects this as a genuine mismatch (not a false
positive) rather than papering over it. Fixed by exposing
`ChatTemplate::think_suppression_prefix()` (the exact injected text, unit
tested against `build_prompt` byte-for-byte) and re-applying it when
storing a suppressed turn's response into history. Verified end to end
through the real `--cli` REPL (a pty, not just the HTTP API — the REPL
only activates on an actual TTY): `/no_think` held on for a 3-turn
conversation, `reused N from cache` matched the *entire* prior turn every
time (87/87, then 154/154), zero `reused prefill failed` warnings.

### `--ctx-checkpoints` / `--checkpoint-min-step`: bounded checkpoint restore

The actual fix for the gap above, mirroring llama.cpp server's own
`server_prompt_checkpoint` design instead of misusing `n_rs_seq`. Rather
than trying to roll recurrent state back to an arbitrary earlier position
(what `n_rs_seq` does, and why it's memory-unsafe — see above), eullm
periodically takes a full-state snapshot of a sequence at the end of a
clean turn and keeps a small, bounded pool of them:

```bash
eullm run qwen3.6-35b-a3b --cli --no-ui --ctx-checkpoints 4 --checkpoint-min-step 4096
```

- `--ctx-checkpoints N` (default 0, disabled): max snapshots kept at once,
  across all sequences, LRU-evicted once full. Worst-case memory is
  `N × (one sequence's full state size)` — bounded and predictable,
  unlike `n_rs_seq`'s `(1 + N)` multiplier on every recurrent-state tensor.
- `--checkpoint-min-step N` (default 8192): minimum new tokens since the
  closest existing checkpoint of the same conversation before taking
  another one, so a long chat doesn't checkpoint every single short turn.

When a request's live resident slot doesn't cover enough of the prompt
(the scenario that forces a full re-prefill on hybrid/recurrent
architectures today), eullm now checks whether an earlier checkpoint of
the same conversation covers more of it, and restores from there instead —
paying for at most `checkpoint_min_step` tokens of fresh decode instead of
the entire conversation. On dense (non-hybrid) models this is a no-op in
practice (ordinary KV-cache reuse already covers that case); it exists
specifically for the hybrid/recurrent case above. Verified end to end
(capture → restore into a fresh sequence → continued generation produces
an identical continuation to the original, uninterrupted state) before
shipping.

### On ARM CPU: a smaller quantized file can be *slower*, not faster

Counter-intuitive finding from real-hardware testing, worth documenting
because "pick the smaller file" is the natural instinct and is wrong here.
ggml's ARM online-repack fast-matmul path (i8mm/dotprod-accelerated) is
registered for specific tensor types only — confirmed by reading
`ggml-cpu/repack.cpp` directly: `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`,
`MXFP4`, `Q8_0`. Comparing two real quantizations of the same 35B model
side by side (clean, isolated logs — `- type ... tensors` + `file size`
lines from the loader, no cross-run contamination):

| | Q4_K_M | "UD-IQ4_NL" (Unsloth Dynamic) |
|---|---|---|
| Shared tensors | 361 × f32, 251 × q8_0 | 361 × f32, 251 × q8_0 |
| Variable tensors | 80 × Q4_K + 37 × Q5_K + 4 × Q6_K | 37 × IQ4_NL + **80 × IQ3_S** + 4 × Q6_K |
| File size | 20.60 GiB (5.11 BPW) | 16.79 GiB (4.16 BPW) — smaller |
| ARM-accelerated tensors | **121 / 121** (100%) | 41 / 121 (34%) |

The smaller file's size comes from pushing the majority of its "variable"
tensors down to **IQ3_S** — a codebook-based format with **no registered
ARM repack kernel at all** (confirmed absent from `repack.cpp`; `Q5_K`, by
contrast, *is* registered and ARM-accelerated, gated the same way as
`Q4_K`/`Q6_K`). Result: the "smaller, IQ4_NL" file is measurably *slower*
in practice, because two-thirds of its variable tensors run the
unaccelerated generic path. **For CPU-only ARM deployments, prefer a
quantization where every non-shared tensor type is on the accelerated
list above over a smaller file that isn't** — file size and ARM decode
speed are not the same axis.

### `max_tokens`/`num_predict` and `seed`: two defaults that silently diverged from Ollama

A community report on a text-based tool-calling client (Cline, via the
Ollama-compatible `/api/chat`) described the client reliably freezing on
long agentic conversations. Reproduced on real hardware with a scripted
conversation that mimics Cline's own text-based MCP tool-call convention
(eullm has no native `tools`/function-calling API, so any such client falls
back to plain text for it):

- With no `max_tokens`/`num_predict` in the request, eullm defaulted to
  **512** — a fixed cap that doesn't exist in Ollama itself, whose real
  default is unbounded (`-1`: generate until context is full or a stop
  condition). A reasoning model can spend hundreds of tokens still inside
  `<think>` before producing anything else; on real hardware this
  reproduced exactly as described — generation cut off mid-`<think>`, with
  no closing tag, leaving a text-based tool-calling client holding a block
  that will never close in that message. Fixed: the default is now
  unbounded, clamped only by whatever context budget the request actually
  has left (the existing per-request clamp in `prefill_sequence` already
  did this correctly — it just never got an unbounded value to clamp).
- Separately, when no `seed` is given, eullm defaulted to a fixed value
  (the KV-cache slot id on the scheduler path, a hardcoded `1234` on the
  sequential-fallback path) instead of Ollama's real behavior of a fresh
  seed per request. Harmless for the freeze itself, but silently made every
  unseeded request through a given slot deterministic — surprising for an
  API that advertises Ollama compatibility. Fixed: both paths now derive an
  unseeded request's seed from wall-clock entropy.

**What this fix does *not* do:** remove the token cap and a long reasoning
turn stops being *corrupted*, but on CPU-only ARM hardware it doesn't stop
being *slow*. Reproduced on the same hardware as the 35B CPU result above:
a single turn that needed 3,270 tokens of genuine (non-looping — checked
for repeated n-gram windows, found none) reasoning took **337 seconds** at
the model's normal ~10 tok/s decode rate. That's long enough to look
identical to a freeze from the outside, with nothing actually wrong
server-side. There is no code fix for this — it's the real cost of running
a large reasoning model without a GPU. If a client's tool-routing turns
don't need deep reasoning, disabling thinking for those turns (`"think":
false` on the API, where the client exposes it) is the actual mitigation,
not a bigger token budget.

## What's ready today, what's coming

**New in v0.6.29** — IP allowlist for the API and chat UI (`EULLM_ALLOWED_IPS` via `.env`, loopback-only by default regardless of the `0.0.0.0` bind) — see "Restricting who can reach the engine" above.

**New in v0.6.28** — Security and quality pass from an internal audit: fixed a path-traversal bug in Hub's model download endpoint (`%2F..` in the URL segment could escape the storage root), set `n_ubatch` explicitly instead of silently inheriting llama.cpp's 512 default (prefill now actually uses the configured batch size, capped conservatively at 1024), populated real SHA-256 digests for every catalog model from HuggingFace's own LFS metadata and verify downloads against them, plus assorted hygiene fixes (dead code, a discarded `--batch-size` flag on `eullm serve`, log-injection sanitization, digest validation on Ollama import). Full `cargo fmt` pass, formatting only.

**New in v0.6.27** — Fixed two sampling defaults that silently diverged from Ollama's real behavior when a client doesn't set them explicitly: `max_tokens`/`num_predict` defaulted to a fixed 512 instead of Ollama's real unbounded-until-context-or-stop (`-1`) default, and `seed` defaulted to a fixed per-slot value instead of a fresh one per request. The `max_tokens` gap was confirmed on real hardware to truncate a reasoning model's response mid-`<think>` or mid-tool-call on long agentic conversations, corrupting the response for any client (e.g. Cline) that expects well-formed output — see the max_tokens/seed note below for the reproduction and the latency trade-off it does *not* fix on its own.

**New in v0.6.26** — Fixed `eullm run --cli`'s `/no_think` sticky toggle silently corrupting KV-cache reuse: the injected think-suppression text was never re-added when reconstructing a suppressed turn for later history, so every `/no_think` turn permanently diverged from what was truly resident. Confirmed on real hardware to be the dominant cause of small/unstable reuse in practice — see "`/no_think`" above.

**New in v0.6.25** — KV-cache prefix reuse no longer retokenizes a continuing conversation's shared history from scratch every turn. When an idle slot's cached text is an exact prefix of the new prompt, its already-known tokens are reused directly and only the new suffix is tokenized, eliminating BPE re-tokenization instability as a cause of small/unstable reuse — see "The actual root cause of small/unstable reuse" above.

**New in v0.6.24** — **`--ctx-checkpoints N` / `--checkpoint-min-step N`**: bounded full-state checkpoint pool for KV-cache restore on hybrid/recurrent architectures, mirroring llama.cpp server's `--ctx-checkpoints` design. The real fix for the gap `--rs-seq` couldn't safely close — see "`--ctx-checkpoints` / `--checkpoint-min-step`" above.

**New in v0.6.23** — **`--rs-seq N`**: experimental, off by default. Exposes llama.cpp's recurrent-state rollback window for hybrid attention+SSM models (Qwen3.5/3.6). Investigated in depth and found unsuitable as a general KV-cache-reuse mechanism at useful values on large models — see "KV-cache reuse on hybrid/recurrent models" above for the full, sourced explanation and the recommended path forward.

**New in v0.6.18** — **KV-cache prefix reuse**: multi-turn conversations (both the `--cli` REPL and `/api/generate`, which both resend the full growing history as the prompt on every call) no longer re-prefill the entire conversation from scratch on every turn. The scheduler now matches each incoming prompt against its idle sequence slots by longest common token-id prefix (mirroring upstream llama.cpp server's slot model) and only decodes the unreused suffix, keeping the rest resident in the KV cache. No new parameter, no client changes — purely content-addressed, works transparently on both surfaces since they share the same scheduler path.

**New in v0.6.13** — **`--n-cpu-moe N`**: finer-grained sibling of `--cpu-moe` — offload only the first `N` transformer layers' MoE expert tensors to CPU RAM instead of all of them, so a model whose VRAM sits idle under the blanket `--cpu-moe` flag can push more experts back onto the GPU and recover throughput. Direct port of upstream llama.cpp's `--n-cpu-moe` (same per-layer tensor pattern). Mutually exclusive with `--cpu-moe`. See "Run MoE models on a small GPU" above.

**New in v0.6.11** — **`--cpu-moe`**: run MoE models (Qwen3-30B-A3B, Qwen3.6-35B-A3B, …) on a small GPU by keeping expert tensors on CPU RAM while attention, embeddings, and the KV cache stay on GPU — VRAM headroom whole-layer `--gpu-layers` offload can't reach. See "Run MoE models on a small GPU" above.

**New in v0.6.10** — consumer-GPU and operations pass on top of v0.6.3:
- **`--fit`** auto-sizes GPU-offloaded layers to free VRAM (CUDA), charging each layer its weight share *and* its KV-cache slice for the chosen context/cache type — quantizing the KV (`--cache-type-k/v q4_0`) frees room for more layers, the gain growing with context length. On by default from the interactive picker; opt-in on the scriptable CLI. Falls back to partial offload, or to a manual `--gpu-layers`, when it can't size the model.
- **`hf.co/<repo>[:quant]` shorthand** — `eullm run hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_M` pulls and runs any HuggingFace GGUF repo directly, catalog or not.
- **Parallel, resumable downloads** — model pulls fan out across up to 16 concurrent HTTP Range requests (default 8) instead of one stream, and a dropped connection retries only the missing chunk instead of restarting the whole file.
- **Linux ARM64 + NVIDIA CUDA** binary (`eullm-linux-arm64-cuda-12.8`) — validated end-to-end on an RTX 3060 12GB ARM server (qwen3-14b Q4, 33 tok/s, full GPU offload). Ships without an NCCL runtime dependency, so it starts with only the NVIDIA driver installed — no extra packages, no root required.
- **`eullm unload`** / `POST /api/unload` — free the loaded model's VRAM and leave the server running, for handing GPU memory to a co-resident process (e.g. an embedding model during RAG ingestion) without a restart.
- **`--gpu-layers -1`** now parses correctly (clap hyphen-value fix), and a broken model-store path (e.g. a dangling symlink to an unmounted volume) reports a clear error naming the path instead of a bare `EEXIST`.

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
- **Make GDPR and AI Act audit-trail requirements easier to satisfy**
- **Speak their language** and understand their domain
- **Carry their brand** — not "Powered by Qwen" or "Built with Llama"
- **Cost nothing** in ongoing API fees

EULLM aims to close that gap.

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
eullm run big-moe-model.gguf --cpu-moe --fit  # MoE: all experts on CPU RAM, rest on GPU
eullm run big-moe-model.gguf --n-cpu-moe 12   # MoE: only first 12 layers' experts on CPU RAM
eullm run ./model.gguf --rust-debug           # Diagnostics: NaN/Inf logit check (see below), off by default

# CLI
eullm list                                # Show local and available models
eullm show legal-it-7b                    # Model details, metadata, compliance info
eullm serve                               # Start API server without loading a model
eullm serve --daemon                      # Same, detached in the background (PID + log file)
eullm unload                              # Free the loaded model's VRAM without restarting the server

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

Available for: Linux x64 (CPU, CUDA) ✅ · Windows x64 (CPU, CUDA) ✅ · Linux ARM64 (CUDA) ✅ · macOS Apple Silicon (Metal) + Linux ARM64 (CPU) ✅ *(community-validated)* · macOS x64 🧪 [community testing wanted](#-platform-status--help-us-test).

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
