# Engine guide

How to run EuLLM Engine and what each part of it does, from connecting your tools to tuning GPU memory. The flags and endpoints one by one are in the [engine reference](engine.md); the downloads in [platforms](platforms.md).

## Using it with your tools

### Works with the clients you already have

Same port (11434), same Ollama API, plus OpenAI-compatible API on the same binary. Existing tooling (Open WebUI, LangChain, n8n, any OpenAI client) works without code changes:

```bash
# Was:   ollama run llama3
# Now:   eullm run ./your-model.gguf --port 11434
```

What you get on top of the Ollama-compatible API:

| Capability | EULLM Engine |
|---|---|
| **Continuous batching** scheduler — single-pass parallel decode across all active slots, shared KV pool (no per-slot KV pre-allocation) | ✅ `--batch-size N` (1 by default) |
| **Quantized KV cache** — Q4_0, Q5_0, Q5_1, Q8_0 KV types: keys at Q8_0 and values at Q4_0 hold about 2.5× the context of F16 on the same GPU | ✅ `--cache-type-k q8_0 --cache-type-v q4_0` |
| **Audit trail** designed for the EU AI Act — local-only JSONL of every answer and decision (model, tokens, timing; never the text), never transmitted | ✅ on by default |
| **Zero telemetry** — no analytics, no crash reports, no usage stats | ✅ enforced |
| **Single binary** — Rust, no Go runtime, no Python runtime, no Docker needed | ✅ |
| **EU-hosted model registry** (Forge/Hub) | 🚧 in development |

[→ Engine scaling](benchmarks.md) · [→ Why EULLM](why-eullm.md) · [→ Changelog](../CHANGELOG.md)

## Decisions without generation

### Decisions without generation (`/v1/systemone`, new in v0.7.20)

Routing, triage, "does this need a human?": questions whose answer is a
choice, not a text. `POST /v1/systemone` asks a small model typed questions
about a state — yes/no, one of N options, a level on a scale — and returns
probabilities read straight from the model, with nothing generated. The API
follows System One (TypeSafe's Jev), so a client written for it works
against a local EuLLM:

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m
curl -s http://localhost:11434/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
    "team": { "type": "choice", "instructions": "Which team handles it?",
              "criteria": { "billing": "Payments, payouts", "tech": "Bugs", "other": "Anything else" } }
  }
}'
```

**Which model.** Use one trained for exactly this: the
[Jev-Style](https://github.com/lawrence3699/jev-style) decision models
(Apache-2.0, 0.8B and 2B), recognized when they load. They read every option
as a yes/no verdict of its own, so a question can list up to 255 options,
and each ships with a calibration temperature fitted on held-out data,
applied by default. An instruction-tuned chat model such as `qwen3-1.7b`
works too, but it is far less steady. Measured on an RTX 5070 Ti with 64
questions about a one-page (1,024-token) document:

| Decision model | Time | Largest change in an answer between modes |
|---|---|---|
| Jev-Style 2B, Q4_K_M (1.3 GB) | 0.66 s | 0 — exactly the scores of its release |
| Jev-Style 0.8B, Q4_K_M (0.53 GB) | 0.62 s | 0.04 |
| Qwen3-0.6B, Q4_K_M, instruction-tuned | 0.39 s | 0.53 |

A change of 0.53 is enough to flip the top answer of a question near a tie,
and an instruction-tuned model's calibration has not been measured: with one
of those, validate on your own data before automating on a threshold.

Up to 64 questions about the same state are answered in one request that
reads the state only once. Each answer depends on its own question only —
asked alone or among others, in any order, it comes back the same — and the
next request about the same state does not read it again. What a request
sends is always read as text, so a document cannot slip the model's own
turn markers into the prompt. Every decision is recorded in the audit trail.
Loading no decision model changes nothing else: chat, completions and
embeddings work exactly as before.

Two small programs show it at work: a model playing Snake, one decision per
move, which you can watch in a browser with the probability it gave each move,
and a triage of incoming email, in [`examples/`](../examples/README.md).
Details, calibration options and the numbers: [docs/engine.md](engine.md#decisions-v1systemone-and-the-decision-slot).

## Chat and API behaviour

### Thinking mode and error codes (v0.6.39 / v0.6.40)

**`"think": false` now works.** Qwen3-style models reason before answering, and
`think: false` is supposed to turn that off. Before v0.6.39 it did not: the model
kept reasoning, you paid for those tokens, and a stray `</think>` tag showed up in
the reply. Ask for a direct answer and you get one:

```bash
curl -s http://localhost:11434/api/chat -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "How many continents are there?"}],
  "think": false,
  "stream": false
}'
```

Leave `think` out and the reasoning comes back in the answer, tags included,
so a UI can render it as a collapsible section.

**`"think": true` returns the reasoning apart, as Ollama does (new in v0.7.30).** On `/api/chat`
the reasoning goes to `message.thinking` and `message.content` holds only the
answer; on `/api/generate` it goes to `thinking`, beside `response`. Streamed,
the reasoning arrives first, in lines whose `content` (or `response`) is empty,
then the answer. The delimiters (`<think>`…`</think>`, Gemma 4's
`<|channel>thought`…`<channel|>`) and the blank lines around them are dropped.
An answer cut off by its token budget mid-thought has only `thinking`.
`/v1/chat/completions` keeps the reasoning in the answer either way, except on a
request with `tools`, where it comes back in `reasoning_content`.

```bash
curl -s http://localhost:11434/api/chat -H 'Content-Type: application/json' -d '{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "Say hello in Italian, one word."}],
  "think": true,
  "stream": false
}'
# "message": {"role": "assistant", "thinking": "Okay, the user wants…", "content": "Ciao!"}
```

**Asking for a model that does not exist returns `404`, not `500`.** This matters
if your client retries automatically: a `5xx` reads as "temporary, try again", so
a typo in a model name used to turn into a retry loop that could never succeed.
A `404` says *you* need to change something; a `500` still means the model exists
but could not be loaded (out of VRAM, corrupt file), which is worth retrying.

```
$ curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/chat \
    -d '{"model":"typo-in-the-name","messages":[{"role":"user","content":"hi"}]}'
404
```

Both endpoints behave the same way, Ollama-style `/api/chat` and OpenAI-style
`/v1/chat/completions`.

**A server with no room for one more request returns `503`, with
`Retry-After`.** Each slot (`--batch-size`) keeps up to eight requests waiting
behind the one it is answering; the next is refused at once with
`503 Service Unavailable`, `Retry-After: 5` and the error as JSON, streamed or
not, before any stream opens, as Ollama answers its own full queue. It used
to be a `200` whose only line was the error when streaming, and a `500`
without.

### `max_tokens`/`num_predict` and `seed`: two defaults that silently diverged from Ollama

A community report on a text-based tool-calling client (Cline, via the
Ollama-compatible `/api/chat`) described the client reliably freezing on
long agentic conversations. Reproduced on real hardware with a scripted
conversation that mimics Cline's own text-based MCP tool-call convention
(`/api/chat` has no native `tools`, so a client using it falls back to plain
text; `/v1/chat/completions` takes `tools` since v0.6.80):

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

## Running it

### Run it as a daemon (background service)

Both `eullm run` and `eullm serve` accept `--daemon`: the engine detaches into
the background and frees your terminal, no `nohup`, no `&`, no tmux.

```bash
eullm run gemma-4-12b --daemon                       # load model + serve, detached
eullm serve --daemon                                 # headless API server, detached
eullm serve --daemon --pidfile /var/run/eullm.pid    # custom PID file location
eullm serve --daemon --logfile /var/log/eullm.log    # custom log file location

# eullm daemon started (PID 88453).
#   PID file: /tmp/eullm.pid
#   Log file: /home/you/.eullm/logs/eullm.log
#   Stop with: kill 88453
```

- **PID file** defaults to `/tmp/eullm.pid` on Linux/macOS, `eullm.pid` in the
  current directory on Windows — override with `--pidfile`. `/tmp` is the right
  place for it: a PID from before a reboot means nothing.
- **Logs**: stdout/stderr are redirected to `~/.eullm/logs/eullm.log`, so
  startup errors and crashes are captured even after the launching terminal is
  gone. Override with `--logfile`. Setting `--pidfile` on its own keeps the log
  beside the PID file, as in earlier releases.
- **Stop** with `kill $(cat /tmp/eullm.pid)` — the engine handles SIGTERM
  gracefully and finishes in-flight requests before exiting.
- All other flags work unchanged (`--port`, `--ctx-size`, `--web`,
  `--batch-size`, `--rust-debug`, …).

> **Running under Docker or systemd?** Don't pass `--daemon` there — the
> supervisor *is* the daemonizer. Use `docker compose up -d` or a plain
> `ExecStart=/usr/local/bin/eullm serve` unit; graceful SIGTERM handling is
> built in, so `docker stop` / `systemctl stop` shut the engine down cleanly.

### Gated and private Hugging Face models (`HF_TOKEN`)

Set `HF_TOKEN` to a Hugging Face access token and `eullm pull` / `eullm run
hf.co/<owner>/<repo>[:<quant>]` — and the model catalog in the Chat UI — fetch
gated and private repositories too, shards and projector included:

```bash
export HF_TOKEN=hf_...   # made at https://huggingface.co/settings/tokens
eullm run hf.co/google/gemma-3-1b-it-qat-q4_0-gguf
```

For a gated repository, accept its terms on its Hugging Face page with the
same account first. The token is sent to `https://huggingface.co` and nowhere
else — not even to the CDN a download is redirected to — and it is never
written to a log, a manifest or the audit trail. Without `HF_TOKEN` nothing
changes. There is no command-line flag for it on purpose: a token on the
command line is visible to every local user in `ps`.

### Reading the model ahead of its load (`--load-threads`, experimental)

`--load-threads N` has N threads read the model file, every part of a split
one, in file order into the page cache while llama.cpp loads it, so that
llama.cpp finds it in memory. `--load-threads auto` uses 16 on a network file
system (Lustre, NFS, SMB, GPFS, BeeGFS, CephFS, 9p) and none on a local disk.
**It is off by default**, because where it was measured it did not help:

| LUMI-G, Lustre, cold load | no readers | 4 | 16 | 32 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B Q8, 29 GiB, 1 GCD (3 loads each) | 45.9 s | 102.7 s | 49.5 s | 40.9 s |
| Qwen3-235B Q4_K_M, 132 GiB, 4 GCDs (2 loads each) | 110.3 s | — | 128.3 s | — |

The flag came from `dd` reading 2 GiB pieces of a model at 178 MB/s with one
stream and 2,283 MB/s with sixteen (`tools/lumi/sbatch_lustre_probe.slurm`);
llama.cpp's own load turned out not to be that slow stream, reading the whole
file at 0.7 to 1.3 GB/s alone. It stays for file systems where it has not been
measured (NFS, SMB). A model larger than the memory free for the page cache
(the RAM, or a Slurm job's `--mem`) is never read ahead, models under 1 GiB
neither, and the log says how many threads read how much and at what rate.

### Where a busy server's time goes (`eullm::steps`)

With several requests at once, every step of the scheduler decodes one token
of each answer on the GPU and then chooses each answer's next token on the
CPU, one after the other. To see how a step's time splits between the two,
start the server with

```bash
RUST_LOG=eullm=info,eullm::steps=debug eullm serve --batch-size 4 --ctx-size 8192
```

and, every ten seconds while it works, it logs one line:

```text
steps: 10.1 s, 87 steps, 4.0 seqs/step, 1.00 passes/step, 35 tok/s |
  decode 90.4% (104.48 ms/step), sample 9.4% (2.719 ms/token),
  emit 0.2% (0.059 ms/token), prefill 0.0% (0 tok), idle 0.0%, other 0.0%
```

`decode` is the forward pass (waited for, so it is the GPU's time), `sample`
choosing each answer's token from its logits, `emit` turning it into text and
sending it, `prefill` the prompts read between steps, `idle` waiting for
requests. `passes/step` is how many times a step ran the whole model: with a
KV cache per answer, llama.cpp's default with several slots, it takes one
pass for each run of consecutive slot numbers among the answers in the step,
so a slot left out (its prompt still waiting, or no request in it) splits a
step in two. One is the most a step should take. Without the setting nothing
is timed.

`--kv-unified` (experimental) gives all the slots one KV cache instead, as
llama-server does when it chooses its number of slots itself: any slots go
in one pass, and every answer's attention reads all the slots' cells,
masked, which costs more as the contexts grow. Measured on an MI250X GCD
with Qwen3 8B to 32B: 1-4% faster with sixteen clients sending short
requests one after another, 2-7% slower with sixteen requests at once, and
38-45% slower with four slots of 32,000 tokens. It is worth trying only
where requests are short and come and go all the time.

The line above is Qwen3.5-0.8B on four CPU cores with Ollama's default
sampling: sampling took 2.7 ms per token, and 1.1 ms with
`"repeat_penalty": 1.0`. llama.cpp's repeat penalty looks up each of the
vocabulary's candidates (about 248,000 here) in its table of recent tokens,
for every token of every answer.

## Security

### Restricting who can reach the engine (`EULLM_ALLOWED_IPS`, new in v0.6.29)

Both the API and the chat UI bind `0.0.0.0` — the engine often runs on a
different host than the things calling it (a RAG pipeline, a LAN client), so
the bind address can't be the access boundary. Instead, every request's
source IP is checked against an allowlist before it reaches any handler.

With no `.env` file present (or none of it maps to `EULLM_ALLOWED_IPS`), only
`127.0.0.1`/`::1` are allowed — functionally the same as binding loopback,
without needing a different bind address for the unconfigured case. To allow
more, copy [`.env.example`](../.env.example) to `.env` in the directory you
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

## GPU memory

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
run (Qwen3.6-35B-A3B Q4_K_M, RTX 3060 12GB in a Radxa Orion O6), `--cpu-moe --fit`
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

#### Reading long prompts faster: `--n-ubatch N`

With some experts in CPU RAM, an MoE model writes its answer at a speed set
by the CPU, but reads its prompt on the GPU: for each pass over the prompt,
llama.cpp copies the experts kept in RAM to the card and runs them there.
A pass reads 512 tokens by default (`n_ubatch`, llama.cpp's own default), so
a 33,000-token prompt is 65 passes, and 65 copies of every expert in RAM over
PCIe. On Qwen3.8-Flash-Next IQ2_XS (125B parameters, 6B active per token, a
68 GB file with most of its experts in RAM) with an RTX 5070 Ti, that read
the prompt at 256 tokens/s.

`--n-ubatch` sets how many tokens one pass reads. At 4096 the same prompt is
9 passes:

```bash
eullm serve --default-model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ctx-size 40960 --n-ubatch 4096
```

The price is VRAM: a pass's activations live in the compute buffer, which
grows with the pass — on that model from 648 MiB at 512 tokens to 3,165 MiB
at 4096. `--fit` reserves for it (0.75 MiB for every token past 512) and
keeps fewer layers' experts on the GPU to make room. On that model the
prompt was read at 822 tokens/s instead of 256, and the answer written at
19.4 tokens/s instead of 21.5, with the experts of 8 layers on the card
instead of 11. An `--n-cpu-moe` set by hand is not adjusted.

The batch a decode call takes, `--n-batch`, is raised to match when it is
smaller. A dense model, or an MoE that fits whole on the GPU, copies nothing
per pass and gains little.

#### Writing faster: the expert cache (`--moe-cache`, experimental)

Of the experts in RAM, a model uses some far more often than others. With
`--moe-cache auto`, every expert stays in RAM and the VRAM the usual split
would have given whole layers of them becomes a cache instead: llama.cpp
copies an expert to the card the first time a token needs it and keeps the
ones used most recently, so most tokens find their experts already there.

```bash
eullm serve --default-model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ctx-size 40960 --moe-cache auto
```

`auto` sizes the cache from the VRAM left once the rest of the model, its
context and the usual reserves are placed; `--moe-cache 6000` asks for 6,000
MiB, cut to what is left when that is less. The log and the banner say how
large it came out. Measured on Qwen3.8-Flash-Next IQ2_XS with an RTX 5070 Ti
(llama.cpp's own server, same build of the cache):

| Setting | Writes (tokens/s) | Reads a prompt (tokens/s) |
|---|---:|---:|
| The usual split, experts of the last layers on the GPU | 22.4 | 249 |
| Every expert in RAM, 4,000 MiB cache | 33.1 | 206 |
| Every expert in RAM, 8,000 MiB cache | 49.4 | 206 |

The cache serves the decode steps (batches of up to 32 tokens). Reading a
prompt still copies the experts to the card a pass at a time, all of them
now rather than all but the last layers', so it is slower, by 17% there:
`--n-ubatch` is the setting for that, and the two share the same VRAM.

Combined with `--mtp` it was slower, not faster: on that model the head's
drafts were kept 52% of the time, and each check of 3 tokens reads up to 3
times the experts of a single step, most of them copied in. A head that is
right more often gains, but little. On unsloth's Qwen3.6-35B-A3B-MTP
(UD-Q4_K_M) with every expert in RAM, `--moe-cache auto` (8.5 GiB on the same
card) and `--ctx-size 8192`, tokens per second writing a 256-token answer, at
temperature 0 and at 0.8, the default:

| Setting | A story | A piece of code | Drafts kept |
| --- | ---: | ---: | ---: |
| no drafts, temperature 0 | 122.9 | 98.4 | — |
| `--mtp 1` | 132.0 (+7%) | 113.7 (+16%) | 82% |
| `--mtp 2` | 127.1 (+3%) | 114.4 (+16%) | 69% |
| no drafts, temperature 0.8 | 107.7 | 91.7 | — |
| `--mtp 1` | 109.3 (+1%) | 99.9 (+9%) | 72% |
| `--mtp 2` | 115.5 (+7%) | 110.9 (+21%) | 66% |

A sixth to a fifth more on code with `--mtp 2`, and little on prose, where a
dense model in VRAM gains 62% and 27% at temperature 0 (see `--mtp` below):
each check still reads the experts of every token it checks. `--mtp 2` is the
start here too: at temperature 0 one draft does as well, at 0.8 two do better.
Two starts of the same server have differed by 8%, so the gains on prose are
within the noise. `bench/mtp_sweep.sh` says whether it pays on your model.

It needs one CUDA GPU: with more than one, or another backend, the load says
why and runs without it. The cache is llama.cpp PR #29887, which this build
carries ahead of a llama.cpp release; its size and its behaviour may change
when upstream merges it.

**Pinned experts.** With the cache on, the experts it copies from are
*pinned* when the model loads: registered with the GPU driver, so that
copying one to the card is a direct transfer at the bus's full speed instead
of one the driver stages through a buffer of its own, a piece at a time.
Reading a prompt copies from the same experts and gains the same way. One
line at load says how much was pinned and how long it took, or why nothing
was:

```
llama_moe_cache: pinned N GiB of host experts in K ranges, in S s
```

Pinned memory cannot be swapped out or handed back to the page cache while
the model is loaded, so the experts are pinned only when that leaves a
quarter of the RAM, and at least 8 GiB, to everything else.
`LLAMA_MOE_CACHE_PIN=0` leaves them as they were, to compare.

Some drivers refuse to pin the pages of a mapped model file: the line then
reads `host experts not pinned: operation not supported`, as it did on the
reference PC (RTX 5070 Ti, Linux). `--no-mmap` takes the other road: the
model is read into memory instead of mapped, and llama.cpp puts the experts
kept in RAM into memory the driver pins as it allocates it. The line then
reads `host experts already in pinned memory`. Loading reads the whole file
up front, so it takes longer, and the RAM has to hold the experts. On the
reference PC it took Qwen3.8-Flash-Next IQ2_XS from 44.4 to 58.1 tokens/s
writing and from 211 to 451.5 tokens/s reading a prompt, with 33 GiB of
experts pinned and 24 of the 62 GB of RAM still available.

```bash
eullm serve --default-model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ctx-size 40960 --moe-cache auto --no-mmap
```

**What `--moe-cache` does by itself.** Unless told otherwise, a load that
gets an expert cache:

- reads prompts 2,048 tokens at a time instead of llama.cpp's 512, so that
  a long prompt copies the experts to the card a quarter as often. The
  larger compute buffer comes out of the cache. On the model above: 964
  tokens/s reading a 33,200-token prompt instead of 452, 54.9 writing instead
  of 58.1. `--n-ubatch` sets it either way: 512 for the fastest writing, 4096
  for the fastest reading (1,240 there, 47.1 writing).
- reads the model into memory, as `--no-mmap` does, when pinning its experts
  leaves a quarter of the RAM and at least 8 GiB to everything else. `--mmap`
  keeps the file mapped. Where the RAM's size is not known (Windows), the file
  stays mapped; `--no-mmap` reads it in anyway.

The startup log says what it chose and why. With both, the command above
needs neither flag:

```bash
eullm serve --default-model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --ctx-size 40960 --moe-cache auto
```

**Where a decode step goes.** `LLAMA_MOE_CACHE_STATS=64` prints, every 64
decode steps of up to 8 tokens, one line like this:

```
llama_moe_cache: 64 steps of 1.0 tokens: X ms/step = A ms to the routers of 48 layers + B ms in the cache + C ms after; M MiB/step copied, H% of the experts in VRAM; copying takes T ms/step (R GB/s)
```

- *to the routers*: from the start of the step until llama.cpp has read
  back, layer after layer, which experts each router chose. Mostly the GPU
  computing the layers up to there.
- *in the cache*: choosing what to evict and starting the copies; with
  experts not pinned, also the driver staging them. The GPU waits meanwhile.
- *after*: the rest, from the last layer's experts to the next step.
- *M MiB copied* and *H% in VRAM*: what the experts not in VRAM cost.
- *copying takes*: measured on one step in 8, which waits for its copies to
  finish so as to time them, and is left out of the times above.

The statistics cost a little speed: measure tokens per second without them.
At exit the cache's totals print too. Both variables are read by the
patched llama.cpp, as diagnostics for this work rather than EuLLM settings,
and may change or go once it is settled.

**Copying the experts ahead while reading a prompt (`--moe-prefetch`, on
by default).** Reading a prompt copies each layer's experts to the card and
then computes them, one after the other, with the GPU idle during the copy.
The prefetch copies the next expert tensors on a second stream of the GPU
while the current one computes, into four slots of VRAM the size of the
largest expert tensor (1 GiB in all on the model above). On the reference
PC, with `--moe-cache auto` and nothing else set, it read a 33,200-token
prompt at 1,373 tokens/s instead of 968 (+42%, the mean of two runs in each
order), to the same answer token for token, and wrote as fast with the
cache 1 GiB smaller for the slots (54.3 tokens/s against 53.3). That is as
fast as PCIe 4.0 brings
the experts in: each 2048-token micro-batch copies all 33 GiB of them, 1.48 s
at 24 GB/s, and took 1.46-1.53 s. With `--n-ubatch 4096` the same copy is
spread over twice the tokens, and it read at up to 1,743 tokens/s, but the
larger compute buffer takes 1.5 GiB more from the cache. Two slots gained
10-15%, three 18%, and six or eight no more than four. `--moe-prefetch N`
takes 2 to 8 slots, `--moe-prefetch 0` turns it off.

The experts an expert cache already holds are copied into the slots from its
own VRAM, and only the others over the bus (17% of the bytes with the 5.5 GiB
cache of the model above), which reads the 33,200-token prompt at 1,516
tokens/s instead of 1,387 (`docs/moe-offload-plan.md`, phase 6b; patch `0004`,
ported to the cache llama.cpp merged, PR #29887). Where the prefetch is off,
the cache's own copy callback does the same. `LLAMA_MOE_PREFETCH_FROM_CACHE=0`
copies everything over the bus, to compare.

It applies to micro-batches of 512 tokens or more, on one NVIDIA GPU (a CUDA
build), to experts kept in RAM with the model read into memory rather than
mapped: `--no-mmap`, which `--moe-cache` already implies when the RAM allows.
Elsewhere (every expert on the GPU, a mapped file, another backend, several
GPUs) the flag changes nothing.

With `--moe-cache`, `--fit` keeps the slots' VRAM out of the cache, which is
that much smaller, unless that would cost the cache its minimum, the size
`--moe-cache` asked for or the larger micro-batch; the startup line says
what it kept (`and 1.00 GiB beside it for the slots of --moe-prefetch`). On
the model above the writing follows the cache's size only below about 5 GiB
(4.5 GiB: 49-50 tokens/s, 3.5 GiB: 43-45); a model whose writing slows down
with the slots gets the room back with `--moe-prefetch 0`. Where nothing
was kept for them, the slots are made at the first long prompt only if a
twentieth of the card, at least 512 MiB, stays free beside them; if not, the
prefetch stays off, says how much it needed, and a `--moe-cache` lower by the
difference makes room. One line on stderr says it is on, with the slots'
size, or why it is off. `bench/prefetch_check.sh` compares the answers, the
speeds and the cache of each setting on any MoE (`docs/moe-offload-plan.md`,
phase 6, has the measurements).

## Speculative decoding with the model's MTP head (`--mtp N`)

Some models are trained with a multi-token prediction (MTP) head: a small
extra layer that guesses the tokens after the next one. Qwen3.5 and Qwen3.6
have one. With `--mtp N`, after each token the model writes the head drafts
up to N more, one decode of the model reads them all, and the model's own
sampler picks a token at each position in turn: every draft it agrees with
is kept, the first it does not is replaced by its own pick, and the rest are
taken back from the memory. One decode can settle several tokens, and every
token is still the model's choice.

```bash
eullm serve --default-model ./Qwen3.5-9B-Q4_K_M.gguf --mtp 2
```

What it needs:

- **A GGUF that still carries the MTP layers.** Most conversions drop them.
  unsloth publishes `*-MTP-GGUF` repositories that keep them
  (`unsloth/Qwen3.5-9B-MTP-GGUF`, `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, ...);
  llama.cpp's `convert_hf_to_gguf.py` keeps them unless told `--no-mtp`. A
  model without them loads normally, and the startup log says why `--mtp`
  is off.
- **One request at a time**, `--batch-size 1`, the default. With more slots
  `--mtp` is off, and the log says so.
- An architecture llama.cpp drafts for: Qwen3.5/3.6 (`qwen35`, `qwen35moe`)
  and the others its MTP drafter supports. Qwen3.8-Flash-Next (`qwen4exp`)
  has its head in a file of its own: see `--mtp-model` below.

### A head in a file of its own (`--mtp-model FILE`)

A GGUF whose converter dropped the MTP layers can still draft, if the head
exists as a GGUF of its own (the head alone, a few GB, made from the model's
original checkpoint):

```bash
eullm serve --default-model ./Qwen3.8-Flash-Next-IQ2_XS.gguf --moe-cache auto \
  --mtp 2 --mtp-model ./mtp-Qwen3.8-Flash-Next-Q8_0.gguf
```

The head is loaded on the GPU beside the model and its draft context reads
the model's own state, as with a head inside the GGUF; `--fit` counts the
file and the context in what it reserves. Without `--mtp` the flag does
nothing and the log says so; if the file cannot be loaded the model runs
without drafts.

Measured on an RTX 5070 Ti (PCIe 4.0 x16), Qwen3.8-Flash-Next IQ2_XS (68 GB
in two files, the experts pinned in RAM, `--moe-cache 5500`, 8,192 tokens of
context), tokens per second writing a story and a piece of code, one run per
row, drafts kept in brackets:

| Setting | Story, T=0 | Code, T=0 | Story, T=0.7 | Code, T=0.7 |
| --- | ---: | ---: | ---: | ---: |
| no drafts | 55.4 | 49.2 | 54.4 | 47.5 |
| `--mtp 1` | 61.1 (83%) | 58.3 | 58.6 (81%) | 58.9 |
| `--mtp 2` | 61.0 (71%) | 64.3 | 55.9 (68%) | 65.3 |
| `--mtp 3` | 56.2 (62%) | 65.8 | 50.3 (56%) | 66.9 |

`--mtp 1` for prose (+8% at temperature 0.7), `--mtp 2` for code (+37%);
`--mtp 3` loses on prose. llama-server with the same head file
(`--spec-type draft-mtp -md FILE`) measured the same shape, 3 to 8% lower.
Each check still reads the experts of every token it checks, which is why the
gain is a fraction of a dense model's.

A head is trained to guess one token ahead; the drafts after the first are
its guesses on its own guesses, kept less and less often, and each costs a
pass of the head and a position in the check. `--mtp-p-min P` stops drafting
once the head is less sure than P of its next draft, so the draft is long
where the text is predictable (code, lists, quotations) and short where it
is not. `bench/mtp_sweep.sh` measures a model with each setting, on a story
and on a piece of code:

```bash
bench/mtp_sweep.sh ./eullm ./Qwen3.5-9B-Q4_K_M.gguf --ctx-size 16384
```

How much faster depends on how often the drafts are kept and on what a
decode costs. Measured with `bench/mtp_sweep.sh` on an RTX 5070 Ti with
unsloth's Qwen3.5-9B-MTP (Q4_K_M), tokens per second writing a 256-token
answer:

| Setting | A story | A piece of code | Drafts kept |
| --- | ---: | ---: | ---: |
| no drafts | 109.6 | 109.9 | — |
| `--mtp 1` | 138.4 (+26%) | 155.7 (+42%) | 75% |
| `--mtp 2` | 132.6 (+21%) | 177.3 (+61%) | 58% |
| `--mtp 3` | 121.5 (+11%) | 173.2 (+58%) | 51% |
| `--mtp 3 --mtp-p-min 0.5` | 115.0 (+5%) | 170.6 (+55%) | 77% |

Measured with llama.cpp b11100. After the move to b11370 the same model
on the same card wrote 114.8 and 120.2 tokens/s without drafts, and 146.1
(+27%) and 194.3 (+62%) with `--mtp 2`, keeping the same 58% of the drafts.

`--mtp 2` is the best start: the most on code, a fifth more on prose. Code
repeats names and patterns the head predicts well, so it gains most; prose
gains most from a single draft. A threshold raised the share of drafts kept
but lowered the speed: on a GPU a draft that turns out wrong costs little,
while one never drafted is a token the check could have settled. Reading a
prompt costs about a tenth more (4,350 tokens/s instead of 4,854), since the
head reads it too.

The head is a whole layer plus the output projection, run once per draft,
so on a small model it costs nearly what it saves: Qwen3.5 0.8B on 4 CPU
cores kept 40-58% of 3 drafts and wrote at 14-17 tokens/s instead of 20.

The answer is the one the model writes without drafts, up to rounding: a
token read in a decode of several is computed in a different order than one
read alone — on a hybrid model the Gated DeltaNet layers even switch from a
step-by-step to a chunked algorithm — and where two tokens are nearly tied
the pick can differ, as with prompt-cache reuse (see `cache_prompt`). On
that model, answers with real drafts and with drafts that were all wrong
(every decode taken back) came out identical, character for character, and
both left the plain answer at the same character. A test checks the first
half on every model with MTP layers it is given:
`EULLM_MTP_TEST_MODEL=… cargo test -p eullm-engine -- --ignored real_model_mtp`
fails if a draft the model rejected, or a token decoded after one, ever
reaches the answer. A hybrid model's
recurrent-state rollback window is raised to N for the drafts it may take
back (`--rs-seq`); the startup banner shows the drafts asked for, and each
answer says how many drafts the head proposed and the model kept: `draft_n`
and `draft_n_accepted` beside Ollama's durations, in llama-server's `timings`
on `/v1/chat/completions`, and on the answer's audit line. `speed_check.py`
prints the share kept after the speed, and `bench/mtp_sweep.sh` reads it from
there; `TEMPERATURE=0.8 bench/mtp_sweep.sh …` measures it at the default
sampling temperature, where a draft is kept only when it is the token the
model samples.

## KV-cache reuse

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

## Fine-tuning on your own text (`eullm finetune`)

`eullm finetune` trains a model's weights on a text file, on the CPU or GPU
the engine runs on, and writes a new GGUF that `eullm serve` loads like any
other. It is llama.cpp's own trainer (ggml-opt) behind one command, so it
works wherever the engine does — CUDA, ROCm, Metal, Vulkan, CPU — with no
Python and no PyTorch. It is not Forge: Forge prunes, distils and trains
LoRA adapters on a cluster; this adjusts a small model's weights in place,
on one device.

The model has to be F32: llama.cpp's trainer computes gradients for F32
weights only. Convert one from Hugging Face with the converter of the
engine's own llama.cpp:

```bash
python engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/convert_hf_to_gguf.py \
    ./Qwen3-0.6B-Base --outtype f32 --outfile qwen3-0.6b-base-f32.gguf

eullm finetune qwen3-0.6b-base-f32.gguf --data corpus.jsonl --dry-run   # memory only
eullm finetune qwen3-0.6b-base-f32.gguf --data corpus.jsonl --epochs 2 \
    --output qwen3-0.6b-mine-f32.gguf
```

`--data` is a plain-text file, or JSONL with a `text` field on each line
(the format of Forge's corpora); every document ends with the model's
end-of-sequence token. The text is cut into windows of `--ctx` tokens (a
multiple of 256), half a window apart unless `--stride` says otherwise, and
the last `--val-split` of it (5%) is kept aside: loss, perplexity and
next-token accuracy on it are measured before training and after each
epoch, so you see whether the model is learning or only memorising.

```text
  model        qwen3 · 28 layers · 596.0 M parameters, 440.5 M to train (all but the token embeddings)
  memory       ~8.4 GiB estimated (weights 2.2 GiB, gradients 1.6 GiB, optimizer 3.3 GiB, activations 756.0 MiB, logits 445.1 MiB, KV 56.0 MiB) · 14.9 GiB free
  data         7473 documents · 2000 tokens: 9 windows to train, 3 to validate
  before       loss 1.2051 ± 0.1142  ppl      3.34  acc 69.79% ± 1.66%  (768 tokens, 39.4 s, 19 tok/s)
  train        loss 0.8155 ± 0.0530  ppl      2.26  acc 80.34% ± 0.83%  (2304 tokens, 178.7 s, 13 tok/s)
  validation   loss 0.9386 ± 0.0457  ppl      2.56  acc 75.91% ± 1.54%  (768 tokens, 7.7 s, 100 tok/s)
```

**Memory.** Before it loads anything the command estimates what the run
needs and refuses one that will not fit the free VRAM (or the free RAM, on a
CPU); `--force` runs it anyway, `--dry-run` stops after the estimate. Every
trained parameter costs a 4-byte gradient on top of its 4-byte weight, and
AdamW (the default) keeps two more values per parameter, 8 bytes; `--optimizer
sgd` keeps none. Activations grow with the square of `--ctx`, since attention
is computed without flash attention while training. On Qwen3-0.6B the
estimate was 8.4 GiB at `--ctx 256` and the process used 8.7. To fit a
larger model, train fewer tensors — `--train-tensors 'blk.*.attn_*'` trains
the attention blocks only (patterns match GGUF tensor names, `*` is any run
of characters) — use SGD, or shorten the window.

**Learning rate.** Each window is one optimizer step: a batch of a few
hundred tokens, far smaller than the batches a pretrained model was trained
with. The default, 1e-6, is low for that reason. On Qwen3-0.6B-Base at
`--ctx 256`, nine steps at 1e-6 took held-out loss from 1.21 to 0.94, and
the same nine at 1e-5 raised it to 1.32. A tiny model, or one trained from
scratch, wants far more (the tests train a 260K-parameter model at 1e-4).
`--lr-min` with `--decay-epochs` halves the rate from `--lr` down to that
floor over those epochs, as llama.cpp's own finetune example does.

**What comes out** is an F32 GGUF that declares the context length the model
had (the trainer sets it to the training window; the command puts it back).
Quantize it for deployment like any other model. `--report FILE` writes the
whole run as JSON, and the last line printed is `FINETUNE_RESULT {...}`, the
same object, for scripts.

**Limits, all llama.cpp's.** The token embeddings are never trained (an open
FIXME upstream), and neither is an output layer that shares them, as on
Qwen3-0.6B. Training runs on one device (`--device`), with every layer on it.
There is no LoRA: the weights themselves change, which is why the model has
to fit with its gradients.

## What the engine does

### EULLM Engine

Run sovereign LLMs locally with **real llama.cpp inference**, built-in audit trail, and full API compatibility. Single Rust binary, no Python runtime, no Docker required.

Built on llama.cpp (MIT, EU-developed) with the standard set of quantized KV cache types (Q4_0, Q5_0, Q5_1, Q8_0) for ~2-4× context length on the same hardware. We also evaluated TurboQuant (Walsh-Hadamard / Lloyd-Max KV compression) end-to-end during v0.5.x but pulled it from the production build path — see [Research & Experiments](research.md) for the rationale and the archived numbers.

```bash
# Run any GGUF model — a local file or one from the catalog
eullm run ./model.gguf                    # Local GGUF file
eullm run ./model.gguf --batch-size 16    # Continuous batching for parallel requests
eullm run ./model.gguf --web              # Transparent web browsing (URLs in messages auto-fetched)
eullm run qwen3-4b                        # From the catalog (downloaded from Hugging Face)
eullm run big-moe-model.gguf --cpu-moe --fit  # MoE: all experts on CPU RAM, rest on GPU
eullm run big-moe-model.gguf --n-cpu-moe 12   # MoE: only first 12 layers' experts on CPU RAM
eullm run ./model.gguf --rust-debug           # Diagnostics: NaN/Inf logit check (see below), off by default

# CLI
eullm list                                # Show local and available models
eullm show qwen3-4b                       # Model details and metadata
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
- **Continuous batching** — multiple requests decoded in parallel: 259 tok/s across 16 requests on one RTX 5070 Ti, 2.75× a single request ([benchmarks](benchmarks.md))
- **Token streaming** — NDJSON on Ollama endpoints, SSE on OpenAI endpoint (`"stream": true`)
- **GPU acceleration** — NVIDIA CUDA, Apple Metal, AMD ROCm and Vulkan; which build was tested on what, and by whom, is in [platforms](platforms.md)
- **Ollama-compatible API** — same endpoints, same port
- **OpenAI-compatible API** — works with Open WebUI, LangChain, n8n, any standard client
- **Transparent web browsing** (`--web`) — put a URL in any message and the engine fetches the page, strips HTML, selects relevant content, and injects it into the prompt before inference. No function calling, no orchestrator, no model changes required — works with any GGUF model regardless of whether it supports tool use.
- **Built-in audit trail** for every answer and decision: who, when, which model, how many tokens, never the text
- **Quantized KV cache** — standard llama.cpp Q4_0/Q5_0/Q5_1/Q8_0 KV types reduce memory ~2-4× at some quality cost (`--cache-type-k q8_0 --cache-type-v q4_0`). Keep the **key** cache at q8_0. A 4-bit key cache combined with flash attention (on by default) produces incoherent output — not gracefully degraded text, actual word salad — reproduced on Metal, x86 CPU and ARM CPU during [#140](https://github.com/eullm/eullm/issues/140). The engine now raises the key cache to q8_0 for you and says so; if you genuinely need 4-bit keys, pass `--no-flash-attn` as well, which is the combination that works. We also tested the experimental TurboQuant approach (see [Research](research.md))
- **Daemon mode** (`--daemon`) — detaches into the background with PID file + log file, freeing the terminal; `kill $(cat /tmp/eullm.pid)` stops it gracefully. See [Run it as a daemon](#run-it-as-a-daemon-background-service)
- **Browser access** — pages served from this machine can call the API out of the box; `EULLM_ALLOWED_ORIGINS` opens it to others (Open WebUI on another host)
- **Cross-platform binaries** — Linux x64 and ARM64, Windows x64, macOS Apple Silicon and Intel: [platforms](platforms.md) lists every build and who tested it
- An EU-hosted model registry is planned (the Hub is a prototype); models download from Hugging Face today
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

> **No GPU required (since v0.6.42).** Multimodal used to be compiled only into
> the CUDA binaries; every published binary now carries it. Validated on the
> CPU-only Linux x64 build with Gemma 4 12B Q4: image description, stop-sequence
> handling and truncation reporting all behave as they do on CUDA. Expect it to
> be slow. Transcribing the same 40-second voice note on the same machine took
> **4.3 s on CUDA and 143 s on CPU**, about 32x, with output of the same quality.
> On a machine with plenty of shared memory and no discrete GPU, that is the
> difference between slow and impossible.

> **Honest scope (it's an MVP):**
> - **Vision** is solid and validated on Linux + Windows CUDA, and on CPU-only
>   Linux x64 (see above). **Audio** is
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
> - **Model coverage:** multimodal runs on any `mtmd` model, validated on
>   **Gemma 4** (E4B + 12B), whose `mmproj` projector the catalog
>   auto-downloads alongside the model. Since 0.7.5 that includes M-RoPE
>   models (the Qwen VL family), which earlier versions refused.
> - **Attachments stay in the conversation.** A follow-up question still sees
>   the picture it is about: the Chat UI re-sends it every turn, and
>   `/api/chat` reads the `images` of every message, as Ollama does. When the
>   context fills up, the oldest attachments give way first, each replaced by
>   a note telling the model one was there; the current turn's never do. Each
>   turn with a picture in it encodes the picture again.
> - Multimodal models load in **sequential mode** (the continuous-batching
>   scheduler is text-only); text-only models keep full batching.
> - Web Chat UI accepts **both images and audio** as of v0.6.2 — 📎 attach or
>   drag & drop. The engine itself reads `.wav`/`.mp3`/`.flac` and
>   `.jpg`/`.png`/`.bmp`/`.gif`; since v0.6.43 anything else the browser can
>   decode is converted before it is sent, so a WhatsApp voice note (Ogg/Opus)
>   or a WebP screenshot works without you converting it first.
> - Quality is bounded by the quantized model — a Q4 12B does great OCR and
>   scene description but can hallucinate specific facts (e.g. a landmark name).
> - **Linux CUDA note:** the GPU binary links `libnccl.so.2`. If you see
>   `error while loading shared libraries: libnccl.so.2`, install it with
>   `sudo apt install -y libnccl2` (packaging fix tracked for a follow-up).
> - The multimodal build vendors a pre-release of `llama-cpp-rs`
>   ([utilityai/llama-cpp-rs#1034](https://github.com/utilityai/llama-cpp-rs/pull/1034))
>   to get the Gemma 4 projector ahead of the upstream merge; it reverts to the
>   crates.io release once that lands.

> **Note on math rendering in the Chat UI:** the embedded UI ships a tiny,
> zero-dependency, best-effort LaTeX→MathML renderer covering the subset of
> LaTeX that LLMs commonly emit (`$…$` / `$$…$$`, `\frac`, `\sqrt`,
> superscripts/subscripts, Greek letters, common operators, spacing). It is
> **not** a full LaTeX engine — anything outside that subset (complex
> environments like `align`/`matrix`/`cases`, exotic macros) falls back to the
> raw text untouched, never a broken render. It renders client-side via native
> browser MathML, so no JS/WASM dependency is added and the stream/API stay raw.

## Install and build

### Prebuilt binaries (easiest)

Download from [GitHub Releases](https://github.com/eullm/eullm/releases):

```bash
# Linux x64
curl -L https://github.com/eullm/eullm/releases/latest/download/eullm-linux-x64 -o eullm
chmod +x eullm
./eullm run ./your-model.gguf
```

Available for: Linux x64 (CPU, CUDA) ✅ · Windows x64 (CPU, CUDA) ✅ · Linux ARM64 (CUDA) ✅ · macOS Apple Silicon (Metal) + Linux ARM64 (CPU) ✅ *(community-validated)* · macOS x64 🧪 [community testing wanted](platforms.md#validation-and-testers).

### Build from source

**Prerequisites:** Rust 1.75+, C/C++ compiler, CMake, libclang.

```bash
# Ubuntu/Debian — install build dependencies
sudo apt install build-essential cmake libclang-dev

# macOS
xcode-select --install && brew install cmake
```

```bash
# --recursive is required: llama.cpp is a submodule. Without it the build
# fails on a missing llama.h. Already cloned? git submodule update --init --recursive
git clone --recursive https://github.com/eullm/eullm.git && cd eullm
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
# Multimodal (image + audio input) is on by default, on every backend.
cargo build --release --features cuda     # NVIDIA (CUDA)
cargo build --release --features rocm     # AMD (ROCm)
cargo build --release --features vulkan   # Cross-platform (NVIDIA + AMD + Intel)
cargo build --release --features metal    # macOS Apple Silicon
```

Or pull from the EU catalog (coming soon):

```bash
eullm pull qwen3-4b             # Downloads from EU servers (Hetzner DE, OVH FR)
eullm run qwen3-4b              # Runs locally — on your laptop, 8GB RAM
```

### Point your existing tools at it

If you run a llama.cpp-based backend today, you can point your tools at EULLM without rewriting a single line. Same API, same port, same tools. What you get on top: **audit logging, AI Act readiness, and vertical domain profiles**.

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

### Development setup

```bash
# --recursive fetches the llama.cpp submodule; the build needs it.
git clone --recursive https://github.com/eullm/eullm.git
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

### Docker

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

See [Getting Started](getting-started.md) for the full Docker guide.

## Highlights of past releases

Each release's full list of changes is in the [CHANGELOG](../CHANGELOG.md).

**New in v0.6.80** — **the GPU offload is sized automatically**, on every load and every model swap, without `--fit`. A model larger than the free VRAM used to die with an out-of-memory error; sized, the worst case is a slower partial split. `--gpu-layers N` still works and is now an *upper bound* (sizing may offload fewer, so a count chosen for one model cannot run the next one out of memory), `--no-fit` restores the old behaviour, and `--fit` keeps its one remaining meaning: ask before a partial split on an interactive terminal. Also in this release: tool calling on the OpenAI endpoint (`tools`/`tool_choice` → structured `tool_calls` + `reasoning_content`), markdown tables in the chat UI, and correct KV sizing on hybrid-SSM models — see [CHANGELOG.md](../CHANGELOG.md).

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
- **`--fit`** auto-sizes GPU-offloaded layers to free VRAM (CUDA), charging each layer its weight share *and* its KV-cache slice for the chosen context/cache type — quantizing the KV (`--cache-type-k q8_0 --cache-type-v q4_0`) frees room for more layers, the gain growing with context length. Falls back to partial offload, or to a manual `--gpu-layers`, when it can't size the model. *(Opt-in when introduced; sizing is the default since v0.6.80 and the flag now only adds the interactive confirmation.)*
- **`hf.co/<repo>[:quant]` shorthand** — `eullm run hf.co/unsloth/Qwen3-14B-GGUF:Q4_K_M` pulls and runs any HuggingFace GGUF repo directly, catalog or not.
- **Parallel, resumable downloads** — model pulls fan out across up to 16 concurrent HTTP Range requests (default 8) instead of one stream, and a dropped connection retries only the missing chunk instead of restarting the whole file.
- **Linux ARM64 + NVIDIA CUDA** binary (`eullm-linux-arm64-cuda-13.1`) — validated end-to-end on a Radxa Orion O6 (CIX P1) with an RTX 3060 12GB in its PCIe slot (qwen3-14b Q4, 33 tok/s, full GPU offload; 3.0 tok/s on the same board CPU-only). Ships without an NCCL runtime dependency, so it starts with only the NVIDIA driver installed — no extra packages, no root required.
- **`eullm unload`** / `POST /api/unload` — free the loaded model's VRAM and leave the server running, for handing GPU memory to a co-resident process (e.g. an embedding model during RAG ingestion) without a restart.
- **`--gpu-layers -1`** now parses correctly (clap hyphen-value fix), and a broken model-store path (e.g. a dangling symlink to an unmounted volume) reports a clear error naming the path instead of a bare `EEXIST`.

**New in v0.6.3** — first release with working Metal on Apple Silicon (earlier macOS binaries shipped CPU-only; now built with `--features metal`, community-validated on an Apple M2 Pro), first community-validated run on Linux ARM64 (Raspberry Pi 400), multimodal vision validated at runtime on the CUDA binary, and the first build produced entirely through the EU-hosted source mirrors (`eullm/llama.cpp` + `eullm/llama-cpp-rs`) instead of pulling upstream directly.
