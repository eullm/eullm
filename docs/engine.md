# EULLM Engine

The EULLM Engine is a CLI + API server for running GGUF models locally, with real llama.cpp inference, a built-in catalog of open models, a local-only audit trail designed for the EU AI Act, and no network telemetry of any kind. Single Rust binary — no Python, no Docker.

## Installation

### From source

```bash
cd engine

# CPU only
cargo build --release

# With GPU acceleration
cargo build --release --features cuda     # NVIDIA (CUDA)
cargo build --release --features rocm     # AMD (ROCm)
cargo build --release --features vulkan   # Cross-platform (NVIDIA + AMD + Intel)
cargo build --release --features metal    # macOS Apple Silicon

# Binary will be at target/release/eullm
```

#### Build requirements

- Rust 1.85+ (the crates are edition 2024)
- C/C++ compiler (gcc/clang) — needed by llama.cpp
- CMake 3.14+
- libclang (`libclang-dev` on Debian/Ubuntu, `clang-devel` on Fedora) — needed by `bindgen` for FFI bindings
- (Optional) CUDA toolkit, ROCm, Vulkan SDK, or Xcode for GPU support

**Ubuntu/Debian one-liner:** `sudo apt install build-essential cmake libclang-dev`

### Docker

From the repository root, with llama.cpp's submodule checked out
(`git submodule update --init --recursive`):

```bash
# docker compose: the API on 11434 and the chat UI on 11435, this machine only
docker compose up -d engine                      # CPU
docker compose --profile gpu up -d engine-gpu    # NVIDIA GPU

# Or the images alone
docker build -f engine/Dockerfile -t eullm/engine .
docker run -d -p 127.0.0.1:11434:11434 -v eullm-models:/models \
  -e EULLM_ALLOWED_IPS=172.16.0.0/12,192.168.0.0/16 eullm/engine

docker build -f engine/Dockerfile.cuda -t eullm/engine:cuda .
docker run -d --gpus all -p 127.0.0.1:11434:11434 -v eullm-models:/models \
  -e EULLM_ALLOWED_IPS=172.16.0.0/12,192.168.0.0/16 eullm/engine:cuda
```

The container runs `eullm serve`, with the chat UI on port 11435 (publish it
with `-p 127.0.0.1:11435:11435`, or add `--no-ui` after the image name to turn
it off), as uid 10001, with its models in `/models` and its
audit trail in `/data/audit`. `EULLM_ALLOWED_IPS` admits the Docker addresses
a published port's callers arrive from; to serve other machines, publish on
all interfaces and set `EULLM_API_KEYS`. The CUDA image needs the NVIDIA
Container Toolkit and a driver for CUDA 13 (r580 or newer); its header gives
the build arguments for CUDA 12 and for datacenter GPUs. Details:
[Getting Started](getting-started.md#path-a-docker-recommended).

## CLI Commands

### `eullm run <model> [options]`

Load a model, open the chat in the browser and in the terminal, and start the
API server. The model is a catalog id, a Hugging Face repo, the URL of a GGUF,
a model `eullm list` shows, or a GGUF file; one not on this machine is
downloaded first.

```bash
# Run a local GGUF file (inference works immediately)
eullm run ./qwen3-7b-q4_k_m.gguf

# Run a catalog model (downloaded the first time)
eullm run qwen3-4b

# Run a GGUF repo from Hugging Face
eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M

# With options
eullm run ./model.gguf --port 8080
eullm run ./model.gguf --gpu-layers 0      # CPU only
eullm run ./model.gguf --gpu-layers 20     # At most 20 layers on the GPU
eullm run ./model.gguf --ctx-size 8192     # Larger context window
eullm run ./model.gguf --threads 8         # Limit CPU threads
eullm run ./model.gguf --cli --no-ui       # The terminal chat only
```

**Options**, the same for `run` and `serve` unless the table says otherwise.
`eullm run --help` is authoritative and groups them the same way; `eullm
--help` lists the ones used most. This table is the short form:

| Option | Default | Description |
|---|---|---|
| `model` | (optional) | `run` only: a catalog id, `hf.co/<owner>/<repo>[:<quant>]`, the URL of a GGUF, a model `eullm list` shows, or a GGUF path. Omitted on a terminal, opens the model picker |
| `--port, -p` | `11434` | API server port |
| `--ui-port` | `11435` | Embedded chat UI port (separate from the API) |
| `--no-ui` | false | No chat UI: the API only. The UI is on by default for `run` and `serve` |
| `--cli`, `--no-chat` | false | `run` only: chat in the terminal without opening the browser (the chat UI is still served unless `--no-ui`) |
| `--image` | (unset) | `run` only: an image or audio file sent with the first prompt (builds with multimodal support) |
| `--gpu-layers` | automatic | **Upper bound** on GPU layers (-1 = all, 0 = CPU only). Sizing may offload fewer — see below |
| `--no-fit` | false | Disable automatic sizing; use `--gpu-layers` as given |
| `--fit` | (implied) | Same sizing, plus an interactive confirmation before a partial split |
| `--fit-strict` | false | Refuse to load rather than offload a partial split. A MoE model is not refused: one whose experts do not fit loads with them in RAM (the expert cache by default where it gains, see `--moe-cache`) |
| `--vram-reserve`, `--fit-target` | `1024` | MiB of VRAM kept free beside the model and its context, for the card's scratch and other programs: sizing leaves it, and a load refuses a context that would leave less. At least 256 |
| `--cpu-moe` | false | MoE models: all expert tensors on CPU RAM (sized automatically when unset) |
| `--n-cpu-moe` | `0` | MoE models: expert tensors on CPU for the first N layers only |
| `--split-mode` | `layer` | How a model goes across several GPUs: `layer` (whole layers per GPU, one after another) or `tensor` (llama.cpp's experimental tensor parallelism; not for every architecture: DeepSeek-V3's `deepseek2` has none and fails to load. A model read into memory loads about as fast either way, each weight read by 16 threads at once, `LLAMA_LOAD_READ_THREADS`). No effect with one GPU visible. No `none` (make one GPU visible instead) and no `row` (llama.cpp's CUDA and ROCm backends no longer support it) |
| `--tensor-split` | (by free memory) | Each GPU's share of the model, as proportions, e.g. `3,1` |
| `--ctx-size, -c` | `4096` | Total context window (split across batch slots), or `auto`: the largest that keeps every layer on the GPU (dense models; never below 4096, up to what the model was trained for and 131072) |
| `--threads, -t` | one per physical core | Number of CPU threads. The default counts physical cores (performance cores on Apple Silicon) and never exceeds the CPUs the process may use (a Slurm job's, a container's, `taskset`'s) |
| `--batch-size` | `1` | Concurrent requests served by the batching scheduler, up to 64; `0` answers them one at a time without it (raise `--ctx-size` with it) |
| `--n-batch` | `2048` | Prefill batch size (prompt tokens per decode call) |
| `--n-ubatch` | `512`, `2048` with an expert cache | Prompt tokens one GPU pass reads (llama.cpp's micro-batch). Raise it, to 2048-8192, for an MoE model whose experts do not all fit in VRAM: the experts in RAM are copied to the GPU once per pass, so fewer passes read a long prompt faster. `--fit` keeps fewer experts on the GPU to make room. Raises `--n-batch` to match |
| `--moe-cache` | on where it gains | `auto`, `off` or a size in MiB: an MoE model whose experts do not all fit in VRAM keeps them all in RAM and caches the ones it uses most in the VRAM they would have taken. On by default (with `--fit`, which is on unless you pass `--no-fit`) on one GPU, CUDA or ROCm, when 25% to 85% of the experts do not fit (a model that nearly fits, or one whose cache would hold a tenth of them, is faster with the usual split), unless you place the experts (`--cpu-moe`, `--n-cpu-moe`) or use `--mtp`. `auto` or a size asks for it whatever the share. Always speeds up writing; reading a prompt depends on the model (Qwen3.6-35B-A3B 25% faster, Qwen3.8-Flash-Next IQ2_XS 17% slower). One GPU only, CUDA or ROCm; experimental (llama.cpp PR #29887); see the guide |
| `--no-mmap` | off | Read the model into memory instead of mapping its file. Experts kept in RAM then sit in memory the GPU driver has pinned, which the card copies from directly. Loads slower, and the RAM has to hold them; `--moe-cache` does it by itself when the RAM can spare the experts. See the guide |
| `--mmap` | off | Keep the model file mapped where it would be read into memory: by `--moe-cache`, or because it goes to the GPUs whole and is more than half the memory or on a network file system other than GPFS |
| `--kv-unified` | false | One KV cache for every slot (llama.cpp's `--kv-unified`) with `--batch-size` above 1. Experimental: on an MI250X 1-4% faster with short requests coming and going, 2-7% slower with sixteen at once, 38-45% slower with 32k-token contexts |
| `--load-threads` | `0` | Threads that read the model file into the page cache ahead of the load; `auto` uses 16 on a network file system and none on a local disk. Off by default: on Lustre it made loads slower. See the guide |
| `--moe-prefetch` | 4 | Slots of VRAM (2 to 8, 0 = off) the experts in RAM of a long prompt are copied into ahead of their layer, on a second stream of the GPU. One GPU, CUDA or ROCm, experts in pinned memory (the model read into memory), micro-batches of 512 tokens or more; with `--moe-cache` the slots' VRAM comes out of the cache. See the guide |
| `--cache-type-k` | `f16` | KV cache type for keys: `f16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `f32`. Quantizing frees VRAM for more layers. Four-bit keys (`q4_0`, `q4_1`) with flash attention on are raised to `q8_0`, since that combination writes nonsense; Gemma 4 always uses `f16` |
| `--cache-type-v` | `f16` | KV cache type for values, the same types |
| `--no-flash-attn` | false | Disable flash attention (on by default), for the generation model and the decision model alike |
| `--web` | false | Fetch URLs found in user messages and inject their content: https only, from public addresses, unless the `EULLM_WEB_*` settings say otherwise (see [Settings](#settings-in-the-environment-or-env)) |
| `--mmproj` | (auto) | Multimodal projector path, when it is not beside the weights |
| `--mmproj-offload` | (sizing decides) | Keep the projector on the GPU, whatever sizing would decide |
| `--no-mmproj-offload` | (sizing decides) | Keep the projector in system RAM: the most VRAM for the text model, images encoded on the CPU |
| `--ctx-checkpoints` | `0` | Prompt-prefix state snapshots for hybrid/recurrent models |
| `--checkpoint-min-step` | `8192` | Minimum new tokens between checkpoints |
| `--rs-seq` | `0` | Recurrent-state rollback window — leave off unless you know why |
| `--mtp` | `0` | Speculative decoding with the model's own MTP head: up to N drafted tokens checked per decode (2 measured best on a GPU). Needs a GGUF that carries the MTP layers and `--batch-size 1`. A model whose projector was found with it (a vision repo's `mmproj-*.gguf`, which a pull fetches) loads without it, text only, so that the head can draft; one passed with `--mmproj` stays, and then `--mtp` drafts nothing for that model. The log says which; see the guide |
| `--mtp-p-min` | `0` | With `--mtp`: stop drafting once the MTP head is less sure than P (0-1) of its next draft, so the draft length follows the text |
| `--mtp-model` | (unset) | With `--mtp`: a GGUF holding the MTP head alone, for a model whose GGUF has none (Qwen3.8-Flash-Next's). It is loaded onto the GPU whole, and sizing keeps its VRAM out of the layers and the expert cache |
| `--rust-debug` | false | Per-token NaN/Inf scan of the logits (diagnostics) |
| `--daemon` | false | Run as a background daemon |
| `--pidfile` | `/tmp/eullm.pid` | PID file path (with `--daemon`) |
| `--logfile` | `~/.eullm/logs/eullm.log` | Daemon log file (with `--daemon`). Set `--pidfile` alone and the log stays beside it |
| `--keep-alive` | (unset) | Idle-unload a model this many seconds/minutes/hours after its last use (e.g. `5m`). Unset = never automatic; a request's own `keep_alive` field overrides it for that load. Applies to the generation, embedding and decision models independently. For a generation model the time counts from the end of its last request, and a model is never unloaded while a request is using it |
| `--embedding-model` | (unset) | Load a text-embedding model (GGUF path or store name) at startup as a **reserved companion**: its VRAM is subtracted from free VRAM before `--fit` sizes the generation model, so both stay resident together instead of depending on load order. See [Text Embeddings and the Embedding Slot](#text-embeddings-and-the-embedding-slot) |
| `--decision-model` | (unset) | Load a decision model for `POST /v1/systemone` at startup, as a reserved companion like `--embedding-model`. See [Decisions: `/v1/systemone`](#decisions-v1systemone-and-the-decision-slot) |
| `--max-loaded-models` | `1` | How many generation models stay loaded at once (1–16); embedding and decision models are not counted. Past it, the least recently used idle model is unloaded; above 1, a model answering requests is never unloaded to make room. At 1, a request for another model replaces the loaded one, as before. See [Several models at once](#several-models-at-once) |
| `--default-model` | (unset) | The model a request with no `model` field (or an empty one) goes to, loaded for it when needed; `auto` routes it. Unset: the most recently used model answers. See [The default model](#the-default-model) |
| `--auto-model` | (unset) | `NAME[=DESCRIPTION]`, repeatable, 2–8 times, smallest model first: the models `"model": "auto"` chooses between. See [`"model": "auto"`](#model-auto-the-decision-model-chooses-the-model) |
| `--auto-timeout-ms` | `1000` | The most routing may add to a request (10–60000) before the fallback answers |
| `--decision-ctx` | `8192` | Most tokens of context one `/v1/systemone` request may use: the state plus its longest question (plus every other question in `batched` mode). The context is sized per request; this ceiling is what the decision slot keeps free in VRAM |

#### Automatic GPU sizing

Since 0.6.80 the engine decides how much of the model goes on the GPU, on
every load — at startup and on every model swap, on `run` and on `serve`
alike. It reads free VRAM and the model's own metadata, then charges each
layer its share of the weights plus its KV-cache slice for the context and
cache type in use, and offloads as many layers as that budget allows. MoE
models get their expert tensors moved to CPU RAM first, since only a few
experts fire per token: that buys far more headroom than whole-layer
offload can.

You do not need a flag for this. It exists because the alternative default
is worse: a model larger than free VRAM used to die with an out-of-memory
error at load, while sized, the worst case is a slower partial split.

| You want | Use |
|---|---|
| The engine to decide | nothing — this is the default |
| To cap how much of the card is used | `--gpu-layers N` (an upper bound; sizing may go lower) |
| CPU only | `--gpu-layers 0` |
| To force a count past the estimate | `--no-fit --gpu-layers N` |
| To be asked before a partial split | `--fit` (interactive terminals only) |
| To refuse a load that doesn't fully fit | `--fit-strict` |

Three properties worth knowing.

`--gpu-layers` is a ceiling rather than a fixed count on purpose: a number
chosen against one model is not a fact about the next one the process loads,
and applying it blindly to a swapped-in model is exactly how an
out-of-memory error happens.

Automatic sizing never asks questions — it applies the split and logs one
line naming the flags that override it — because a default that interrupts
every launch is its own kind of failure. Free VRAM is read from ggml's device
registry on every GPU backend (CUDA, ROCm, Vulkan, Metal); where there is no
GPU to read it stays silent and `--gpu-layers` is used as-is.

The budget leaves headroom on purpose, and it is the same headroom the
loader requires: enough of the card must stay free for the context and its
compute buffers, or the weights load and then no context can be allocated at
all. Sizing that aims past that floor produces a split the loader refuses,
which is a worse failure than a conservative one — so sizing and the loader
read one number, `--vram-reserve`, in MiB: 1024 by default, 256 at the least.
It is the same on a 16 GB card and on a 96 GB one, because what it covers (the
scratch of a decode step, a CUDA context, other programs on the card) does not
grow with the card. Until 0.7.80 it was 12% of the card's total memory, which
cost a dense 8B two layers on a 16 GB card. Lower it on a card nothing else
uses, raise it on one that also drives a display. When a model is only a few
layers short of fitting whole, the log says which `--vram-reserve`,
`--ctx-size` or q8_0 KV cache would put them all on the GPU.

### `eullm pull <model>`

Download a model: a catalog id, a Hugging Face repo
(`hf.co/<owner>/<repo>[:<quant>]`) or the URL of a GGUF. With no model, in a
terminal, a menu of the catalog's models opens.

```bash
eullm pull qwen3-4b
eullm pull hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
eullm pull https://example.eu/models/model-Q4_K_M.gguf
```

The model is stored in `~/.eullm/models/<model>/` (or under
`EULLM_MODELS_DIR`) with a GGUF file and manifest.

Gated and private Hugging Face repositories need an access token: set
`HF_TOKEN` in the environment or in `.env` (for a gated one, after accepting
its terms with the same account). It is sent only to `https://huggingface.co`,
for the API calls and every download request alike. See the engine guide's
section "Gated and private Hugging Face models".

### `eullm list`

Show the models on this machine: the ones in the models folder (downloaded
ones included) and in the model libraries, each by the name it runs by. If
there are none, displays the EU catalog.

```bash
eullm list
```

### Model libraries and `eullm import`

Besides the models `eullm pull` downloads, EuLLM lists and runs any GGUF it
finds, in place:

- in the folders of `EULLM_MODEL_LIBRARIES` (environment first, then `.env`;
  separated by `;`, on Linux and macOS also by `:`), such as LM Studio's
  models folder or a Hugging Face cache;
- in its own folder (`EULLM_MODELS_DIR`, or `~/.eullm/models`), outside the
  model directories `eullm pull` made.

Every subfolder is read. Nothing is moved, renamed, added or deleted there, so
the program that made the folder keeps using it; `eullm rm` refuses a library
model. Hidden folders (a name starting with a dot) are not entered.

- **Names**: from the file (`Qwen3-8B-Q4_K_M.gguf` → `qwen3-8b-q4_k_m`). When
  two files would get the same name, or a pulled model already has it, each
  takes its folder's name in front (`unsloth-qwen3-8b-q4_k_m`).
- **Projectors** (`mmproj*.gguf`, or any GGUF whose header says
  `general.architecture = clip`) are not models: one in the same folder as a
  model is that model's projector.
- **Split models** (`-00001-of-00003.gguf`): one model, opened from its first
  part; one with parts missing is left out.

`eullm list`, `/api/tags` and `/v1/models` show them (status `library`, the
file's path as description), and `eullm run <name>` or a request's `model`
field runs them. The folders are scanned again at most every 5 seconds, so a
file added or removed shows up without a restart.

```bash
eullm import          # what the scan found, and what it left out and why
```

`eullm import` writes nothing. In 0.7.95 it wrote manifests and moved a `.gguf`
lying loose in the store folder into a folder of its own; manifests it wrote
then are read as before.

### `eullm show <model>`

Display detailed information about a model (local or from catalog).

```bash
eullm show qwen3-4b
```

### `eullm rm <model>`

Delete a downloaded model's files, after a confirmation (`--force` skips it).
`remove` is the same command. A model in a model library is refused: EuLLM
only reads those folders. The API's `DELETE /api/delete` does the same.

```bash
eullm rm qwen3-14b
eullm rm qwen3-14b --force
```

### `eullm serve [--port PORT]`

Start the API server without loading any model. The first API request with a `"model"` field will load that model dynamically.

```bash
eullm serve
eullm serve --port 8080
eullm serve --no-ui        # API only, no chat UI
```

The chat UI is served on `--ui-port` (11435) as with `run`, behind the same
API keys, IP allowlist and origins as the API; `--no-ui` turns it off. A UI
port already taken is logged and the API is served anyway. (Until 0.8.0
`serve` started without the UI unless given `--ui`, which is still accepted
and does nothing.)

`serve` takes the same runtime flags as `run` (they are one shared set), and
they apply to every model it loads or swaps to: automatic GPU sizing runs on
each load, `--gpu-layers` caps it, `--no-fit` disables it. The one difference
is that `serve` never prompts — a daemon has nobody at the keyboard — so
`--fit` adds nothing there and `--fit-strict` surfaces a refused load as an
error to the API caller.

### `eullm unload [--model NAME]`

Unload generation models from a running server, freeing their VRAM, without
restarting it: every one, or with `--model` that one only, the others staying
loaded. Requests still running on an unloaded model are cut off. A later
request naming a model loads it again.

```bash
eullm unload                    # every generation model
eullm unload --model qwen3-8b   # this one only
eullm unload --port 11500
```

The same as `POST /api/unload` — see below.

### `eullm update [--check]`

Install the latest stable release in place of the binary that runs the command.

```bash
eullm update --check    # only say whether a newer release exists
eullm update            # download it and put it in place
```

EuLLM never looks for a new version by itself: this command is the only time it asks github.com, and only when it is typed.

**What it does:**

1. Reads the latest release's tag from `https://github.com/eullm/eullm/releases/latest` and compares it with its own version.
2. Downloads the same build as the one running: each release binary carries the name of the file it was published as (`eullm-linux-x64-cuda-13.1`, `eullm-windows-x64-vulkan.zip`, ...), so a CUDA build is replaced by the CUDA build, a Vulkan one by the Vulkan one.
3. Checks the download against the release's `checksums.txt`, unpacks a Windows ZIP, and runs the new binary's `--version` before anything is replaced: a download this machine cannot start changes nothing.
4. Puts the new files in place of the old ones in the binary's own directory, keeping the binary's file name. Each old file is renamed to `*.old` first and put back if any step fails. On Windows the running `eullm.exe` and the DLLs it loaded cannot be deleted while it runs, so they stay as `*.old` until the next update removes them.

A running `eullm serve` or `eullm run` keeps the old version until it is restarted.

It declines, saying why, for a binary built from source or in a Docker image (no release file is known to match it), for the Microsoft Store package, which the Store updates, and for the Windows ROCm build, whose ZIP keeps ROCm's folder layout (`bin\` and `.kpack\`): extract the new ZIP over the folder that holds `bin\` instead. A system-wide install (`/usr/local/bin`) needs `sudo eullm update`. Versions up to 0.7.40 do not have the command: update them once more with the installer or a download from the releases page.

### `eullm finetune <model> --data <file>`

Train a model's weights on a text file, on the CPU or GPU the engine runs on,
with llama.cpp's own trainer: an F32 GGUF in, a new F32 GGUF out, validation
loss measured before and after each epoch. `eullm finetune --help` lists its
options; the engine guide's "Fine-tuning on your own text" explains memory,
learning rate and the model's conversion to F32.

```bash
eullm finetune ./model-f32.gguf --data corpus.txt --epochs 2
eullm finetune ./model-f32.gguf --data train.jsonl --optimizer sgd --dry-run
```

### `eullm forge`

Delegate to the EULLM Forge Python pipeline for model verticalizzazione.

```bash
eullm forge Qwen/Qwen3-14B --profile legal-it --identity "LegalAI"
```

### Settings in the environment or `.env`

What is not a command-line option is read from the environment, else from a
`.env` file in the folder eullm starts from: the environment wins, and a blank
value counts as unset. The startup log says where the settings that decide who
may connect and what may be read came from. Secrets stay off the command line,
where every user of the machine can read them. [`.env.example`](../.env.example)
has each one with an example; `eullm --help` lists them too.

| Setting | Default | What it does |
|---|---|---|
| `EULLM_MODELS_DIR` | `~/.eullm/models` | Where `eullm pull` downloads. Any `.gguf` in it, at any depth, is listed and runs |
| `EULLM_MODEL_LIBRARIES` | (none) | Other programs' model folders, separated by `;` (on Linux and macOS also by `:`), read where they are and never changed: see [Model libraries](#model-libraries-and-eullm-import) |
| `EULLM_ALLOWED_IPS` | this machine only | IPs and subnets that may connect without a key, comma-separated. Loopback always stays allowed |
| `EULLM_API_KEYS` | (none) | `id:secret[:rpm=N]`, comma-separated: every request must carry a key. An entry that does not parse stops the server |
| `EULLM_API_KEYS_FILE` | (none) | The same keys, one per line of a file that can be `chmod 600` |
| `EULLM_ALLOWED_ORIGINS` | loopback pages | Web pages on other hosts allowed to call the API; `*` allows every origin |
| `EULLM_AUDIT_DIR` | `~/.eullm/audit` | Where the audit trail is written. Set and not writable, the server does not start |
| `EULLM_WEB_ALLOWED_DOMAINS` | any public host | With `--web`: fetch only from these domains and their subdomains, comma-separated |
| `EULLM_WEB_ALLOW_HTTP` | off | With `--web`: fetch plain `http` pages too |
| `EULLM_WEB_ALLOW_PRIVATE_HOSTS` | off | With `--web`: fetch from private and loopback addresses too, which turns the protection against requests into the local network off |
| `EULLM_ALLOW_MODEL_PATHS` | off | A request's `model` may name a GGUF file, or a folder holding one, by its path. For local use only |
| `EULLM_DECISION_POLICY` | (none) | A JSON file of rules for `/v1/systemone`: see [A server-side decision policy](#a-server-side-decision-policy-eullm_decision_policy) |
| `EULLM_DECISION_TRACES` | (none) | A folder where `/v1/systemone` keeps its decisions as text: see [Decision traces](#decision-traces-eullm_decision_traces) |
| `EULLM_DOWNLOAD_CONNECTIONS` | `8` | Parallel connections per download, 1 to 16; 1 downloads in a single stream |
| `EULLM_IMAGE_MAX_TOKENS` | the projector's | Most tokens an image is read as: more reads finer detail, with more memory. Raise it in steps (512, then 1024): very high values can crash some projectors |
| `EULLM_IMAGE_MIN_TOKENS` | the projector's | Fewest tokens an image is read as |
| `HF_TOKEN` | (none) | A Hugging Face access token, for gated and private repositories. Sent to `https://huggingface.co` only |
| `RUST_LOG` | `eullm=info` | What the log shows, such as `eullm=debug`; naming `llama-cpp-2` (`eullm=info,llama-cpp-2=info`) adds llama.cpp's own log |

Until 0.8.7 the web settings, `EULLM_ALLOW_MODEL_PATHS`, `HF_TOKEN`,
`EULLM_DOWNLOAD_CONNECTIONS`, the image settings and `RUST_LOG` were read from
the environment only, and a line for them in `.env` was ignored.

## Dynamic Model Swap

EULLM Engine loads models at runtime, like Ollama: a request names a model in
its `"model"` field, and a model that is not loaded is loaded for it. How many
stay loaded is `--max-loaded-models`. At 1, the default, a request for another
model replaces the loaded one — a swap; above 1, models stay loaded side by
side ([Several models at once](#several-models-at-once)). A request may also
name no model ([The default model](#the-default-model)), or `auto`, and let
the decision model choose ([`"model": "auto"`](#model-auto-the-decision-model-chooses-the-model)).

```bash
# Start with one model
eullm run qwen3-14b

# Any API request with a different model triggers a swap
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3-7b", "prompt": "Ciao"}'
# → Unloads qwen3-14b, loads qwen3-7b, then responds

# Or start with no model and load on first request
eullm serve
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3-14b", "prompt": "Ciao"}'
# → Loads qwen3-14b on the fly
```

**Behavior:**

- With one model at a time (the default), a swap cuts off the requests the old model is still answering: they end with the error `Server shutting down`. To keep several models loaded instead, see [Several models at once](#several-models-at-once)
- A model served sequentially — every multimodal model, and any with `--batch-size 0` — cannot be cut off, and is freed only when the requests running on it end, so a swap waits for them, up to 30 seconds, before the next model is sized against free VRAM
- The new model loads on a blocking thread, then atomically replaces the slot
- The model name is one `eullm list` shows (the models folder, the model libraries, a catalog model already downloaded), `NAME` for `/models/NAME.gguf` or `/data/models/NAME.gguf`, or, only with `EULLM_ALLOW_MODEL_PATHS=1`, a GGUF path

**What carries across a swap, and what does not.** The flags you launched
with are settings, and they apply to every model the process loads: context
size, cache types, batch size, thread count, and `--gpu-layers` as an upper
bound. What a *model* needs is decided per model, freshly, on each load:

| Per model, re-decided on every load | Why |
|---|---|
| How many layers go on the GPU | Sized against the VRAM actually free at that moment, for that model's own weights and KV cache |
| MoE expert offload | Only MoE models have experts to move |
| The multimodal projector | A projector belongs to the weights it was trained with; pairing it with another model fails the load |
| Sequential vs batched execution | Vision models need the sequential engine; text models get the scheduler back |

That split is not a detail: treating a per-model property as a process-wide
setting is how a launch model's projector ended up on its successors, and how
a layer count chosen for one model produced an out-of-memory error on the
next one.

### Several models at once

`--max-loaded-models N` (default 1, at most 16) keeps up to N generation
models loaded, so that requests alternating between models answer from memory
instead of reloading one each time. Embedding and decision models have slots
of their own and are not counted.

```bash
eullm serve --max-loaded-models 2
curl http://localhost:11434/api/generate -d '{"model": "qwen3-4b", "prompt": "Ciao"}'
curl http://localhost:11434/api/generate -d '{"model": "qwen3-8b", "prompt": "Ciao"}'
# → both stay loaded; the next request to either answers at once
```

How a model finds its place:

- **Below the limit**, a model a request names is loaded beside the others.
- **At the limit**, the least recently used idle model is unloaded first. A
  model whose keep_alive is over goes before any other, a model kept for good
  (`keep_alive: -1`, or no `--keep-alive`) after one that would expire, and
  the model `eullm run` started with goes last.
- **A model answering requests is never unloaded to make room.** When every
  model that could make room is busy, the load waits up to 120 seconds for one
  to finish, then answers 503 with `Retry-After: 5`. This holds from a limit
  of 2: at 1, the default, a request for another model replaces the loaded
  one even mid-answer, as it always did.
- **A model loads beside others only if it fits whole** on the GPU in what
  they leave free: every layer, and its projector. Otherwise models are
  unloaded until it fits, or until it is alone, when it is sized like any
  model loaded by itself, partial split included. The same rule as Ollama's
  ("new models must be able to completely fit in VRAM to allow concurrent
  model loads"): a second model only ever gets what the first one left, and
  splitting it silently would divide the card. Without automatic sizing
  (`--no-fit`, or a build that cannot read free VRAM) only the count is kept,
  and the server says so at startup.
- **An embedding or decision model loaded by a request** (not a reserved
  `--embedding-model` or `--decision-model`) that does not fit in the VRAM
  left free unloads generation models the same way — the least recently used
  idle one first, one at a time, and only as many as it needs. Each counts in
  `model_swaps`.
- A request with no `model` field is answered by `--default-model`, or
  without it by the most recently used model.

Requests to a model that is loaded never wait for another model's load.
`GET /api/ps` lists what is loaded, with what each model holds and when it
expires. `/api/version` reports `max_loaded_models`, `loaded_models`, and
`generation_evictions` (how many models were unloaded to make room; a steady
rate of one per request means the models asked for do not fit together).

Unlike Ollama's `OLLAMA_MAX_LOADED_MODELS`, this is a command-line flag, not an
environment variable, and it counts generation models only, defaulting to 1
rather than three per GPU. The server says so at startup when the variable is
set.

**On a machine without a GPU**, run the server with
`OMP_WAIT_POLICY=PASSIVE`. Each model computes on its own pool of OpenMP
threads, one per core, and by default a pool's threads spin for a while after
each step waiting for the next one, taking the cores another model is
computing on. Measured with two tiny models on a CPU, a request's time to
first token went from 34.5 ms alone to 235.8 ms while the other model was
loading. `PASSIVE` makes waiting threads sleep instead. On a GPU the threads
mostly wait on the device, and the setting matters little.

```bash
OMP_WAIT_POLICY=PASSIVE eullm serve --max-loaded-models 2
```

### The default model

`--default-model NAME` (a store name or a GGUF path) is the model a request
with no `model` field, or an empty one, goes to — on `/api/generate`,
`/api/chat` and `/v1/chat/completions` — loaded for it as if the request had
named it. An empty request with `keep_alive: 0` unloads it. A name that is no
model stops the server at startup. `--default-model auto`, with
`--auto-model`, routes such requests as `"model": "auto"` is routed. Without
the flag a request that names no model is answered by the most recently used
model, and refused with 503 when none is loaded.

## `"model": "auto"`: the decision model chooses the model

With two to eight `--auto-model` and a `--decision-model`, a request naming
the model `auto` (in any case) is answered by one of the candidates, chosen
for it by the decision model: the small one when it will answer correctly and
completely, a larger one when the request needs more reasoning, knowledge,
code or length. The large model answers only what needs it.

```bash
eullm serve --max-loaded-models 2 \
  --auto-model 'qwen3-4b=Short everyday requests, simple facts and quick rewrites' \
  --auto-model 'qwen3-8b=Multi-step reasoning, maths, code, analysis and long answers' \
  --decision-model /models/Jev-Style-0.8B-Decision-v3-Q4_K_M.gguf
curl http://localhost:11434/api/chat -d '{"model": "auto", "stream": false,
  "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

**Candidates.** `--auto-model NAME=DESCRIPTION`, smallest model first: the
order is the order the decision model reads them in. The description is what
it reads about each, up to 400 characters; without one it reads the store
manifest's description, else the catalog's, else the name — the server warns
about the last two, which describe a product rather than what the model is
good at. `auto` itself, a model given twice, or a name that is no model stops
the server at startup. `--max-loaded-models` should hold every candidate;
the server warns when it does not, since every switch would then reload a
model.

**What the decision model reads.** One `choice` question, with a candidate as
each option, about a digest of the request built by code: the start of the
system instructions, the last turns before the latest message shortened, the
latest message whole (its head and tail when very long), the attachments and
the tool names. The text is never stored in the audit trail, only its SHA-256.

**Code filters first.** A candidate that cannot read the request's images or
audio (no projector), or whose context is shorter than the request, is not
offered. With one candidate left there is nothing to decide; with none, the
fallback answers, and fails as a request naming it would.

**The fallback** is `--default-model` when it is a candidate, the last
candidate otherwise. It answers whenever the decision model does not decide,
and the reason says why:

| Reason | When |
|---|---|
| `decided` | The decision model chose: the most likely candidate, no threshold |
| `no_decision_model` | No `--decision-model`, or it was unloaded (by `--keep-alive`, say) |
| `timeout` | The decision took longer than `--auto-timeout-ms` (1000); it is cancelled |
| `decision_error` | The decision model could not decide (the request over its budget, a failure) |
| `only_candidate` / `no_eligible_candidate` | The filters left one candidate, or none |
| `load_failed` | The chosen model would not load: the fallback answered in its place, once |

**Latency.** A decision is one forward pass of the decision model over the
digest: tens of milliseconds on a GPU. One decision runs at a time per
decision model, so routed requests and `/v1/systemone` traffic queue behind
each other; `--auto-timeout-ms` bounds the wait. On a CPU a decision takes
hundreds of milliseconds or more: with Qwen3-0.6B as the decision model, 239 ms
at the median on a 16-core Ryzen 9 5950X (5 October), while an earlier run had
every route time out at the default 1000 ms. Where a decision takes longer than
`--auto-timeout-ms`, `auto` answers with the fallback; raise the timeout and
every request waits for its decision instead.

**Which model answered, and why.** The `model` field of the response, and of
every streamed line or chunk, is the model that answered. The headers, sent
before the body, say the same — `X-EuLLM-Model`, `X-EuLLM-Route` (the reason)
and `X-EuLLM-Route-Id` — and the response, or the last line or chunk of a
stream, carries:

```json
"eullm": {"route": {"requested": "auto", "model": "qwen3-4b", "reason": "decided",
  "confidence": 0.71, "probabilities": {"qwen3-4b": 0.85, "qwen3-8b": 0.15},
  "decision_model": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "decision_ms": 38.2, "id": "<uuid>"}}
```

Ollama and OpenAI clients ignore it. The audit trail has one `route` line per
routed request, under the route's id: the candidates offered and excluded,
the decision record as `/v1/systemone` writes it, the reason and the time.
The answer's own line names it in `route.id`, and says in `route.fallback`
when the chosen model did not load.

**Loading.** Once the server listens, it loads the candidates, the fallback
first, as far as they fit without unloading anything, and logs how many it
loaded. An empty request naming `auto` (an empty prompt, or empty messages)
does the same, and with `keep_alive: 0` unloads every candidate; both answer
with `"model": "auto"`. `auto` is listed in `/api/tags`, with
`details.family: "eullm-router"` and its candidates, and in `/v1/models`.

**Limits.** A `raw` prompt on `/api/generate` is written for one model's
template and gets a 400. There is no filter on tool support: a request with
tools may go to a candidate that cannot call them. `/api/show` does not know
`auto`. Without `--auto-model`, `auto` is a model name like any other, and a
404.

**Checking the choice without generating.** [`POST /api/route`](#post-apiroute)
takes the same body and answers with the route, the digest and the question
— the way to tune the descriptions, and what `bench/reflexbench/autobench.py`
measures routing with.

## KV Cache Quantization

By default, EULLM uses F16 KV cache for maximum GPU compatibility. Quantized types save VRAM but may cause GPU compute fallback to CPU on some architectures — verify GPU utilisation with `nvtop` before deploying in production.

| Setting | VRAM for 14B @ 16K context |
|---------|---------------------------|
| **`--cache-type-k f16 --cache-type-v f16`** | **~10 GB (default, best GPU compat)** |
| `--cache-type-k q8_0 --cache-type-v q8_0` | ~5 GB |
| `--cache-type-k q8_0 --cache-type-v q4_0` | ~2.5 GB (⚠️ verify GPU usage) |

```bash
# Default (F16) — maximum GPU compatibility
eullm run qwen3-14b --ctx-size 8192

# Save VRAM with quantized KV cache (check GPU usage with nvtop!)
eullm run qwen3-14b --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q4_0

# Aggressive quantization (minimum VRAM, may fall back to CPU)
eullm run qwen3-14b --ctx-size 32768 --cache-type-k q4_0 --cache-type-v q4_0
```

Available types: `f16`, `f32`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`.

> **Note:** Quantized V cache types (Q4_0, Q8_0) require Flash Attention. On GPUs where Flash Attention doesn't support these types, the engine automatically falls back to F16 KV cache and logs a warning. You can also set F16 explicitly by omitting the `--cache-type-v` flag.
>
> **TurboQuant note:** v0.5.x of the engine integrated an experimental TurboQuant (Walsh-Hadamard + Lloyd-Max) KV compression via the AmesianX/llama.cpp fork. It is **not in v0.5.8 onwards** — see [Research and experiments](research.md) and the archived numbers in [`turboquant-quality-report.md`](turboquant-quality-report.md) / [`turboquant-kv-stress-report.md`](turboquant-kv-stress-report.md).

## Constrained JSON Decoding (`format: "json"`)

When `format: "json"` is set in a request, EULLM uses GBNF grammar-based constrained decoding to guarantee valid JSON output. This matches Ollama's behavior and prevents malformed JSON in extraction pipelines.

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "qwen3-14b",
    "prompt": "Extract the name and age from: John is 30 years old",
    "format": "json"
  }'
# → Always returns valid JSON: {"name": "John", "age": 30}
```

Works on all endpoints: `/api/generate`, `/api/chat`, `/v1/chat/completions`. Both sequential and continuous batching modes.

## Continuous Batching

EULLM's continuous batching scheduler decodes multiple requests in parallel on a single GPU pass. This is a key performance differentiator over Ollama, which processes requests one at a time.

```bash
# One request at a time, with the whole context (default: --batch-size 1)
eullm run ./model.gguf

# Continuous batching: 8 requests decoded together
eullm run ./model.gguf --batch-size 8

# More slots for high-throughput RAG workloads
eullm run ./model.gguf --batch-size 16

# Sequential mode: no scheduler thread (what a multimodal model runs in)
eullm run ./model.gguf --batch-size 0
```

The default is 1 slot: `--ctx-size` is shared by the slots, so a default of 8
gave each request an eighth of the context, and answers stopped at 512 tokens
with no flag to point at. Ask for concurrency when several clients send
requests at the same time, and raise `--ctx-size` with it.

With 16 concurrent requests on an RTX 5070 Ti, EuLLM serves 259 tok/s in total, 2.75× a single request's throughput. See [benchmarks](benchmarks.md) for details.

**A long prompt does not stop the others.** With more than one slot, a new
request's prompt is read a chunk at a time in the decode steps of the
answers already being written: each step gives every answering request its
next token and reads up to a micro-batch (`--n-ubatch`) of waiting prompt
beside them, and so on until the prompt is read. A 30,000-token RAG prompt
used to be read whole the moment its request was taken, and every other
answer on the server stopped until it was. With nothing else answering,
prompts are read a whole batch (`--n-batch`) at a time, and with one slot
(the default) nothing changes. Either way llama.cpp computes a prompt one
micro-batch at a time from the same positions, so the answer is the one a
prompt read whole gets. Prompts are read in arrival order, as many in a step
as fit: requests that arrive together are read together and start answering
together, instead of one per step.

**A full queue is a 503.** Behind the requests being answered, each slot
keeps up to eight waiting, in arrival order: with the default single slot,
one answering and eight waiting. One more is refused at once with
`503 Service Unavailable`, `Retry-After: 5` and the error as JSON, streamed
or not, before any stream opens, as Ollama answers its own full queue. A
backend for many concurrent users raises `--batch-size` rather than retrying.

### Context window and batch slots

The `--ctx-size` flag sets the **total** KV cache budget, shared across all batch slots (matching Ollama/llama.cpp server behaviour). Each slot gets `ctx_size / batch_size` tokens of context:

```bash
# 16K total, 4 slots → 4096 tokens/slot
eullm run ./model.gguf --ctx-size 16384 --batch-size 4

# 32K total, 8 slots → 4096 tokens/slot
eullm run ./model.gguf --ctx-size 32768 --batch-size 8
```

### Choosing batch-size

More slots increase parallelism but reduce per-request throughput (shared GPU time). General guideline:

| Parallel slots | Per-request throughput | Aggregate throughput | Use case |
|:-:|:-:|:-:|---|
| 1 (default) | Highest | Lowest | One user, a chat, an agent |
| 4 | High | High | Chat, general inference |
| 8 | Medium | Higher | Batch extraction, RAG pipelines |
| 16+ | Lower | Highest | High-concurrency APIs, multi-GPU |

Keep the default for one user. For several, start with `--batch-size 4` and a `--ctx-size` four times the context each request needs; increase when your workload requires more concurrent slots and can tolerate slower individual responses.

## Loading a Model: Slots, Context and Names

When a request names a model that is not loaded, the server:

1. **Makes room** for it: with one model at a time, the default, it shuts down
   the loaded model's scheduler thread (waiting for it to fully exit) and frees
   its VRAM; with `--max-loaded-models` above 1, it unloads a model only when
   the new one needs its place or its memory — see [Several models at
   once](#several-models-at-once)
2. **Sizes** the new model against the VRAM now free, and **loads** it with the
   requested configuration
3. **Serves** the request on it, and every later request that names it

### Basic swap (via model field)

Any generation request with a different `model` triggers the swap:

```bash
# Currently running qwen3-14b — this switches to qwen3-8b automatically
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Hello"
}'
```

### Dynamic batch_size and ctx_size

The `batch_size` and `ctx_size` can be overridden per model swap. This is useful when switching between a large model (fewer slots) and a small model (more slots):

```bash
# Switch to 8B with 8 parallel slots and 32K context
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "batch_size": 8,
  "ctx_size": 32768,
  "prompt": "Hello"
}'

# Switch back to 14B with 4 slots and 16K context
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b",
  "batch_size": 4,
  "ctx_size": 16384,
  "prompt": "Hello"
}'
```

When `batch_size` or `ctx_size` are not specified, the values from `--batch-size` and `--ctx-size` at startup are used.

### Model name resolution

The `model` field accepts:

| Format | Example | Resolution |
|--------|---------|------------|
| Full GGUF path | `/models/qwen3-8b.gguf` | Direct file |
| Ollama-style name | `qwen3:8b` | Normalized to `qwen3-8b`, searched in `/models/` and model store |
| Path without extension | `/models/qwen3-8b` | Tries appending `.gguf` |
| Directory | `/models/mymodel/` | Picks the first `.gguf` file inside |
| Registered name | `legal-it-4b` | Looked up in `~/.eullm/models/` |

A model is its file, whatever it is called: a request that names the loaded
model's GGUF another way — its path, or a second name `eullm pull` linked to
the same weights — is answered by the model already loaded, under the name it
was loaded with, instead of loading the same weights again.

### Concurrent swap safety

Multiple requests arriving simultaneously for a different model are handled safely:
- Only one load runs at a time (serialized via Mutex); requests to a model that is already loaded do not wait for it. A load waiting for a busy model to finish lets go of the lock meanwhile, so embedding and decision loads are not held up by it
- Other requests for the model being loaded wait for the load to complete, then use it
- With one model at a time (the default), requests the old model is still answering are cut off with an error, except on a sequential engine, which finishes them before it is freed. With `--max-loaded-models` above 1, a model answering requests is never unloaded to make room — see [Several models at once](#several-models-at-once)

### VRAM budget reference

Approximate VRAM usage with F16 KV cache (default). Actual values depend on model architecture and GPU.

| Model size | batch_size | ctx_size | tok/slot | VRAM est. |
|:----------:|:----------:|:--------:|:--------:|:---------:|
| 14B Q4 | 4 | 16384 | 4096 | ~12.5 GB |
| 8B Q4 | 4 | 16384 | 4096 | ~7 GB |
| 8B Q4 | 8 | 32768 | 4096 | ~9.5 GB |
| 8B Q4 | 8 | 16384 | 2048 | ~7 GB |
| 70B Q4 | 16 | 65536 | 4096 | ~45 GB |

## Text Embeddings and the Embedding Slot

`POST /api/embed` (Ollama) and `POST /v1/embeddings` (OpenAI) embed one or
more texts with any GGUF text-embedding model — BGE, E5, and similar.
Naming a model in the request loads it the first time it is requested,
exactly like the generation model:

```bash
curl -s http://localhost:11434/api/embed -d '{
  "model": "bge-m3",
  "input": ["first chunk", "second chunk"]
}'
```

The embedding model lives in a **second, independent slot** — it does not
replace whatever generation model is loaded. Both are kept resident when
they fit; this is a decision the server makes for itself on every embedding
request, not something the caller has to reason about:

- If the embedding model fits in currently free VRAM alongside the loaded
  generation model, it is loaded next to it. Both stay resident.
- If it does not fit, the generation model is evicted first (it reloads
  automatically on the next generation request), and the embedder gets the
  whole card.
- The reverse also holds: a generation request with `--fit` enabled evicts
  a resident embedder first if the sizing needs the VRAM back.

On a build with no GPU to read free VRAM from (CPU only), the server always tries to keep
both resident and lets a genuine out-of-memory surface as a normal load
error — the same posture `--fit` itself takes there.

How many times either direction of eviction has happened is in
`/api/version`'s `model_swaps` field — a rate of roughly one per request
means a card too small for both is being asked to alternate rather than
batch (do all ingestion, then all generation, rather than interleaving).

The eviction dance above assumes the embedding model was loaded on demand,
by naming it in a request. `--embedding-model <path-or-name>` skips that:
the embedder loads at startup and becomes a **reserved companion** — it
loads first, with the context its inputs are embedded in, built for its
longest input and kept for every request, so both already count as used
VRAM by the time `--fit` reads free VRAM to size the generation model, on
both `eullm run` and `eullm serve`; `--fit` additionally keeps a small
margin free on top, for what a decode allocates beside them, both at launch
and again on every later generation-model swap. That context is memory the
embedder holds from startup — about 1.4 GB for Qwen3-Embedding-0.6B at the
default 2,048 tokens, measured on an RTX 5070 Ti — and what guarantees that a long input never fails for lack of room next to a
generation model sized to fill the card. A reserved
companion is never evicted to make room for a generation load; it keeps its
place for the life of the process. If reserving its space would leave the
generation model no room at all, the launch proceeds anyway with a warning:
the reservation is dropped and the embedder falls back to the normal
evict-on-demand behavior described above, exactly as if the flag had not
been given.

```bash
eullm run legal-it-4b --fit --embedding-model bge-m3
```

Use this when a companion (e.g. bge-m3 for a RAG pipeline) should always be
resident alongside the generation model whenever the card has room for both,
rather than depending on which model happens to be requested first.

Requests name the companion as it was given: by its store name, as `eullm
list` shows it (`--embedding-model qwen3-embedding-0.6b-gguf-q8_0`), or by
its file name when it was given as a path. A request naming the same file
any other way finds it too, and is answered by the model already loaded.

The embedder keeps one context from one request to the next, grown when
an input needs more room than it has and never shrunk, and embeds the
inputs of every request in it one at a time, in the order they came: a
request of many inputs holds back a short one for one input, not for all
of them, and a burst waits instead of failing for lack of memory. Building
that context was most of what a request cost: on an RTX 5070 Ti with
Qwen3-Embedding-0.6B, a 1,966-token input took 1,001 ms when every request
built its own and takes about 97 ms in the kept one; a 62-token input, 40
ms then and about 7 ms now. An embedder loaded on demand builds its context
when its first input comes; a reserved companion builds it at startup.

Pooling is read from the model's own GGUF metadata (CLS for BGE, mean for
E5, and so on) rather than guessed; a model that declares no pooling type
falls back to mean-pooling its per-token embeddings. Output vectors are
L2-normalized. Rerankers (RANK-pooling models) are out of scope for this
endpoint — normalizing a single relevance score would collapse it.

## Decisions: `/v1/systemone` and the Decision Slot

`POST /v1/systemone` answers typed questions about a *state* — a ticket, a
document, a conversation, a JSON object — without generating any text. It
takes the request and response shape of the System One API (TypeSafe's
Jev), so a client written for it can point its base URL at EuLLM. Three
question types:

| Type | Question | Answer |
|---|---|---|
| `noul` | Is this statement true of the state? (`criteria`, optional: `{"true": "…", "false": "…"}`, what each answer means) | `noul`: P(yes) |
| `choice` | Which of these options? (`criteria`: an object of name → description, 2–26 options; up to 255 with a [Jev-Style model](#jev-style-decision-models)) | `choice`, `probabilities`, `confidence` |
| `score` | Which level of this scale? (`criteria`: an array of 2–10 level descriptions, lowest first; a level may also be `{"label": "…", "description": "…"}`) | `score` (Σ level × p), `legend`, `probabilities`, `confidence` |

```bash
curl -s http://localhost:11434/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
    "team": { "type": "choice", "instructions": "Which team should handle it?",
              "criteria": { "billing": "Payments and payouts", "tech": "Bugs", "other": "Anything else" } },
    "severity": { "type": "score", "instructions": "How severe is it?",
                  "criteria": ["Cosmetic", "Degraded, with a workaround", "Blocking"] }
  }
}'
```

```json
{
  "model": "qwen3-4b",
  "answers": {
    "is_urgent": { "type": "noul", "noul": 0.95, "eullm": { ... } },
    "team": { "type": "choice", "choice": "billing",
              "probabilities": { "billing": 0.91, "tech": 0.07, "other": 0.02 },
              "confidence": 0.865, "eullm": { ... } },
    "severity": { "type": "score", "score": 1.43,
                  "legend": { "0": "Cosmetic", "1": "Degraded, with a workaround", "2": "Blocking" },
                  "probabilities": { "0": 0.0, "1": 0.57, "2": 0.43 },
                  "confidence": 0.355, "eullm": { ... } }
  },
  "usage": { "input_tokens": 312, "output_tokens": 0 },
  "timing": { "total_ms": 187.4 },
  "eullm": { "audit_id": "b1149e83-332b-48c0-bab7-53725922c4de", "mode": "shared_prefix",
             "prompt_tokens": 520, "shared_prefix_tokens": 104, "evaluated_tokens": 312,
             "timings_ms": { ... }, "request_ms": 187.43, ... }
}
```

`timing.total_ms` is the request's wall time, model resolution included, as
jev-style's server reports it and its MCP tools and guard show it:
`eullm.request_ms` to 0.1 ms. `eullm.timings_ms` splits the decode into its
phases.

`eullm.audit_id` is the decision's `id` in the audit trail, and in the
[decision traces](#decision-traces-eullm_decision_traces) when they are on:
the id to [give feedback](#feedback-post-v1systemonefeedback) under. It is
inside `eullm`, not at the top level, because the System One SDKs' response
models are strict and refuse a key they do not know.

Answers and options come back in the order the request listed them; options
are shown to the model lettered in that order.

`instructions` may be a string, or, as the System One API allows, an object
or an array — the question in one field and the data it refers to in
others. The model then reads it as one line of compact JSON, in the order
it was written, which is what jev-style's server gives its models:

```json
{"record":{"name":"John Smith","city":"Oakland"},"question":"Is this resume the same person as the record?"}
```

A score level written as `{"label": "high", "description": "loses data"}` is
shown to the model as `high: loses data` — the label alone when there is no
description — and named `high` in the `legend`, as jev-style's own server
does. Any other object or array is shown as one line of JSON and named in the
legend by its label, when it has one, or by its compact JSON.

**How an answer is computed.** With an instruction-tuned model — the *code
readout* — each question becomes one chat prompt (the model's own template,
reasoning switched off) ending where the answer would begin, with the
options coded `A`…`Z`, the levels `0`…`9`, or `Yes`/`No`. The logits at
that position are read once — nothing is generated — and restricted to the
codes. When the model loads, the server checks that every code is a single
token for its tokenizer right after its prompt, and refuses a question whose
codes are not; the load log lists what it found. Everything a request
sends — the state, the instructions, the options — is tokenized as text: a
chat template's own turn markers inside it (`<|im_end|>`,
`<|im_start|>assistant`) stay text instead of closing the user turn and
writing the rest of the prompt. Only the template's text is read for control
tokens; for the rare template that cannot be tokenized in those pieces
exactly as it is whole, a request containing such a marker is refused with
a 422. A model trained for this endpoint reads its answers differently — see
[Jev-Style decision models](#jev-style-decision-models) — and the response
says which readout was used in `eullm.readout` (`codes` or `verdict`).

Next to the System One fields, every answer carries an `eullm` object with
what it was derived from, so a stored response can be re-examined or
re-calibrated later:

| Field | Meaning |
|---|---|
| `logprobs` | Code readout: full-vocabulary log-probability of each answer's code |
| `scores` | Verdict readout: each option's score, `logit(" yes") − logit(" no")` at its slot |
| `raw_probabilities` | The same, renormalized over the answers (a softmax of the scores), before calibration and temperature |
| `coverage` | Code readout: share of the model's probability on a valid code. Near 1: it answered in the format asked for. Low: most of its probability went elsewhere (a thinking tag, a sentence) and the answer describes a minority of what it would have said — check this before trusting an answer |
| `prior_logprobs` | The content-free prior that was divided out (with `content_free` calibration only) |
| `confidence_entropy` | `choice` and `score`: `1 − H(p) / ln K` of the same probabilities, `confidence` as it was defined up to 0.7.20 |

`confidence` (for `choice` and `score`) is `(K · p_max − 1) / (K − 1)` over
the `K` answers, clipped to [0, 1]: how far the top answer's probability is
above an even split, as a share of the most it could be — 1 when all
probability is on one answer, 0 when it is spread evenly; `2 · p_max − 1`
for two options. It is jev-style's definition, so a threshold on it means
the same against jev-style's server and against EuLLM. The response names
it in `eullm.confidence_method` (`normalized_max_probability`), and each
answer's `eullm.confidence_entropy` keeps `1 − H(p) / ln K`, the
entropy-based value `confidence` was up to 0.7.20, which reads the
runner-up answers too.

**Calibration is not a solved problem here, and nothing is claimed about
it yet.** The mechanism reproduces with any model; calibrated probabilities
do not come with it. The request's `eullm` object picks what is applied, so
the options can be compared on labelled data before one is trusted:

| `eullm` option | Values | Default |
|---|---|---|
| `calibration` | `none`; `content_free` (code readout): divide out the answer the model gives the same question about the state `N/A` (Zhao et al., 2021), cached per question | `none` |
| `temperature` | Temperature scaling after calibration: `> 1` flattens, `< 1` sharpens | `1`; the model's own calibration temperature when it has one (below) |
| `mode` | `shared_prefix`; `batched`; `separate` (see below) | `shared_prefix` |

The content-free prior is not always noise to remove: when the options
themselves imply a base rate, dividing it out moves probability towards
options that are rarely right. Measure before choosing.

**A model's own temperature.** A decision model can bring the temperature
it was calibrated with: a Jev-Style release has one built in (see
[Jev-Style decision models](#jev-style-decision-models)), and a model Forge
trains is calibrated on held-out data and carries the fitted value in its
GGUF, as the metadata key `eullm.decision.temperature` (a `FLOAT32` or
`FLOAT64`). A code-readout model whose GGUF has that key scales its
probabilities with it by default. A request's `temperature` still overrides
it, and the response's `eullm.temperature` says which one was applied. The
value has to pass the same check as a request's — greater than 0 and at
most 100 — and anything else (another type, `NaN`, `0`, `250`) is ignored
with a warning that names it: the model loads anyway, its probabilities
unscaled. The load log prints the temperature in effect and where it came
from:

```text
Decision model loaded — codes: …; prompts: …; calibration temperature 0.8730 (the GGUF's eullm.decision.temperature); …
Decision model loaded — codes: …; prompts: …; calibration temperature 1 (none: the GGUF has no eullm.decision.temperature); …
```

A Jev-Style model keeps its release's temperature even when its GGUF
carries this key, and the log warns when the two differ.

**Many questions, one pass.** Every question's prompt starts with the same
tokens — system prompt, template, the state — and differs only at the end.
`shared_prefix` decodes that common part once, then each question on its
own right after it, in a sequence that starts as a copy of the state's (no
copy of data: on a unified cache the cells are only tagged with the extra
sequence) and is dropped once the question's answer is read: about
`S + Q·q` tokens for `Q` questions of `q` tokens on a state of `S`, instead
of the `Q·(S + q)` of asking one at a time. `batched` decodes the same
tokens, but every question's in one batch: fewer, larger decode calls.
`separate` asks one question at a time from an empty cache; it exists as
the baseline. The response reports the token counts (`prompt_tokens`
against `evaluated_tokens`), the timings of each phase and `flash_attn` —
`auto`, or `off` under `--no-flash-attn` — since both the timings and the
last digits of every probability depend on it. `bench/decision_bench.py`
measures all three for 1–64 questions on states of several sizes.

**Each answer depends on its own question only.** The three modes read the
same tokens but hand them to the kernels in batches of different shapes,
and on quantized weights that alone moves an answer — a quantized model's
matrix products round their inputs to 8 bits, which turns a last-digit
difference in a sum into a different rounding one layer later. How far,
measured with `bench/decision_bench.py` (the largest difference in any
probability from `separate`; on the CPU over states of 256 and 1,024 tokens
with 1–64 questions, on the GPU with the benchmark's defaults):

| Model | 4-core CPU: `shared_prefix` | `batched` | RTX 5070 Ti: `shared_prefix` | `batched` |
|---|---|---|---|---|
| Qwen3-0.6B F16 | 0.017 | 0.017 | | |
| Qwen3-0.6B Q8_0 | 0.11 | 0.13 | | |
| Qwen3-0.6B Q4_K_M | 0.34 | 0.32 | 0.53 | 0.52 |
| Jev-Style-0.8B-Decision-v3 Q4_K_M | 0.024 | 0.028 | 0.039 | 0.033 |
| Jev-Style-2B-Decision-v3 Q4_K_M | | | 0 | 0 |

With Qwen3-0.6B Q4_K_M, the catalog's `qwen3-0.6b`, that is enough to
change the top answer of a question near a tie, and more so on CUDA, where
the arithmetic is coarser: ggml-cuda runs cuBLAS in TF32 mode and
accumulates some F16 products in half precision. None of the modes is the
exact one; they are the same model with different rounding. The Jev-Style
models, trained to answer this way, are an order of magnitude steadier than
an instruction-tuned model at the same quantization, and the 2B's modes read
the same blocks, so they agree exactly (see
[below](#jev-style-decision-models)).

What differs is what the rounding depends on. In `batched` mode a question
sits somewhere in a batch with the others, so its answer moves with the
other questions asked and with their order: asking the same questions in
reverse moved answers by up to 0.31 on the CPU (Q4_K_M). In `shared_prefix`
mode a question is decoded alone, in the same cache cells, in batches of the
same shape and over the same attention window whatever else the request
asks, so its answer is a function of the state and that question: asked
alone, among 63 others or in reverse order, it comes back bit for bit the
same (measured on the CPU and on an RTX 5070 Ti;
`bench/decision_bench.py --order-check` checks it on your
hardware and fails if it does not hold). That is the default because a
decision should not change with the questions asked next to it, and
because a calibration measured on labelled data then holds however the
questions are grouped into requests when serving. The price is one decode
call per question instead of one per batch. On a 4-core CPU that is 2–14%
slower than `batched`. On a GPU each call has a fixed cost of about 3 ms
(Qwen3-0.6B on an RTX 5070 Ti) — the GPU running a few hundred small
kernels, not launching them: padding every question to a common length so
the calls replay as CUDA graphs was tried and measured 25% slower — so with
many questions `shared_prefix` is 2–3× slower than `batched` there, and
still 2–5× faster than `separate`. Measured with 64 questions: 316 / 347 /
473 ms for states of 256 / 1k / 4k tokens, against 111 / 130 / 235 ms
`batched` and 630 / 1645 ms / — `separate`. Calibrate in the mode that
will serve.

**`batched` requests that arrive together are answered together.** The
decision model runs one request at a time, so with many clients the
requests queue. Those asked in `batched` mode that reach it while it is
busy are evaluated together: each request's state in a sequence of its own,
decoded side by side, then their questions, 64 at a time across requests
(16 on a recurrent or hybrid model, such as the Jev-Style 0.8B), in shared
decode calls — up to 16 requests, and no more context than one request may
use (`--decision-ctx`). The questions go through in groups of neighbouring
lengths: a hybrid model's decode call runs the model once for every length
its questions stop at. An answer then moves with the other requests'
questions too, by the rounding above, as it already moves with the other
questions of its own request, and that is not only digits: on LUMI-G, the
Jev-Style 0.8B Q4_K_M asked the same request again and again among others
moved a probability by up to 3 points, and in a third to a half of the
repeats at least one of its 8 or 64 decisions came out the other way (the
borderline ones; alone, `batched` gave the same answers every time). Use it
where throughput matters more than a borderline decision depending on the
traffic; `shared_prefix` and `separate` requests are never grouped, and stay
bit for bit what they are alone. The
`eullm` extension of the response says when it happened:
`requests_together`, the requests evaluated with it (itself included), and
the context, prefix and question times are then the group's. A model that
reads in blocks (Jev-Style 2B) decodes one block per call, which no other
request can share: its requests stay one at a time.

**Limits:** 64 questions per request, 2–26 options per `choice` (up to 255
with a Jev-Style model), 2–10 levels per `score`, and `--decision-ctx` tokens
of context per request (default 8192). A request over the context limit is
refused with a 422 `input_budget_exceeded` that says how many tokens it
needed; nothing is truncated.

**One option is not a choice.** jev-style accepts a `choice` with a single
option — the System One API documents a maximum of 255 and no minimum — and
answers it with that option at probability 1 and confidence 1, whatever the
state says. EuLLM refuses it with a 422 `invalid_question`, and requires two:
with one option there is nothing to decide, the answer is known before the
model reads anything, and returning it as a decision would only look like
one. To ask whether that one option fits, ask a `noul` about it, which
returns the probability that it does.

**Errors** come back in the shape System One clients and
[jev-style](https://github.com/lawrence3699/jev-style)'s parse — a
status and `{"error": {"code": "…", "message": "…", "question": "…"}}`, with
`question` present when one question is at fault — so its MCP server, CLI
and guard show EuLLM's message instead of failing on the body:

| Status | `code` | When |
|---|---|---|
| 422 | `invalid_json` | The body is not JSON |
| 422 | `invalid_request` | The request as a whole fails validation: no `state`, no questions, more than 64, an unknown `eullm` option |
| 422 | `invalid_question` | One question fails validation — an unknown `type`, one option, 11 levels; `question` names it |
| 422 | `input_budget_exceeded` | Longer than `--decision-ctx`, or than a Jev-Style model's budgets; `question` names the question when it was one question's. Nothing was truncated |
| 422 | `policy_denied` | The server's [decision policy](#a-server-side-decision-policy-eullm_decision_policy) leaves a `choice` question fewer than two options; `question` names it |
| 409 | `traces_disabled` | Feedback sent to a server with [decision traces](#decision-traces-eullm_decision_traces) off |
| 400 | `model_not_loaded` | No `model`, or a System One name such as `jev-latest`, and no decision model loaded |
| 404 | `not_found` | `model` names a model the server does not have |
| 401 / 403 / 429 | `unauthorized` / `forbidden` / `too_many_requests` | Refused by the API key, IP allowlist or origin checks, or over the key's quota |
| 405 / 413 / 415 | `method_not_allowed` / `payload_too_large` / `unsupported_media_type` | Not a `POST`, a body over the limit, a body labelled as something JSON is not (`multipart/form-data`, …); one with no `Content-Type`, `text/plain` or curl's `application/x-www-form-urlencoded` is read as JSON |
| 500 | `internal_error` | The model failed to load, or llama.cpp failed |

The other endpoints keep the error bodies Ollama and OpenAI clients read.

**Which model.** Preferably one trained for this endpoint, a
[Jev-Style model](#jev-style-decision-models): calibrated by its authors, up
to 255 options, and an order of magnitude steadier than an instruction-tuned
model at the same quantization (see the table above). A small
instruction-tuned model also works: Qwen3 0.6B–4B from the catalog are the
intended size. It must be able to answer without reasoning first; a model
that always opens a reasoning block (the DeepSeek-R1 family) spends its
first token on the tag, and every answer's `coverage` shows it.

**The decision slot.** The model lives in a third slot, next to the
generation and embedding models, with the same residency rules as the
embedding slot: loaded next to the generation model when it fits, the
generation model evicted first when it does not. What counts is the weights
plus a request's context — its KV cache at `--decision-ctx` and a compute
buffer. The model keeps that context between requests, sized by the largest
request so far, and releases it before a generation model is sized, so the
sizing sees it as reserved, not as used. `--decision-model <path-or-name>`
loads it at startup as a reserved companion, exactly like
`--embedding-model`: `--fit` keeps its context's VRAM free and a chat-model
swap never evicts it.

**The same state again costs only the questions.** The context keeps the
state the last request decoded. A request whose shared part is token for
token the same — the next round of questions an agent asks about the same
document — starts from it instead of decoding the state again, and reports
`prefix_reused: true`, with `evaluated_tokens` counting only what was
decoded. It changes no answer: the questions land in the same cells, in
batches of the same shape, as they would from a freshly decoded state (and
`--order-check` in the benchmark checks exactly that). A request about
another state replaces the one kept.

```bash
eullm serve --decision-model qwen3-1.7b --decision-ctx 16384
```

The `model` field may name any model the server can load (it then loads into
the decision slot). Left out, or set to a System One model name — `jev`,
`jev-latest`, or a release such as `jev-1.13.0` — it means the decision model
already loaded, so a Jev client works unchanged. A Jev-Style model pulled into
the store, such as `jev-style-2b-decision-v3-gguf-q4_k_m`, is a model like any
other, and is loaded when a request names it. With no decision model loaded the request is refused with a
400 `model_not_loaded` saying so.

Every decision is written to the audit trail with `request_type:
"systemone"` and a `decision` record: each answer with its log-probabilities
and its probabilities before and after calibration. The state itself is not
stored, only its SHA-256; its text, redacted, goes only to the
[decision traces](#decision-traces-eullm_decision_traces), when they are on.

**A client that disconnects.** The record is written by the thread that
computed the decision, not by the connection that asked for it, so every
decision the model computes is recorded — one whose client disconnected
before its answers were ready with `client_disconnected: true`, since they
were never sent. A request whose client disconnects before that stops at its
next question, or before it starts if it was still waiting for the model, so
the next request does not wait behind work nobody will read. It decided
nothing and, like a request refused as invalid or one llama.cpp failed on, is
not recorded; the server log says it was abandoned. A single question is not
interrupted once it is being decoded.

### A server-side decision policy (`EULLM_DECISION_POLICY`)

Some options must never be chosen, whatever a model makes of the state: a
tool that deletes data, a transfer of money, anything that touches
production. The decision policy takes them out of every request before the
model reads it. Code filters, the model judges: an option the policy denies
is removed from the question, not vetoed after the model picked it, so the
model chooses among the options that remain and their probabilities add up
without it. A model shown an option it must not take can prefer it whatever
the state says: offered a way to the food that ended in a trap, the
Jev-Style 0.8B took it 20 times out of 20 in the Snake example.

`EULLM_DECISION_POLICY` names a JSON file, read once at startup, from the
environment first and the `.env` file second, like the other perimeter
settings:

```json
{ "version": 1, "deny_options": ["delete_*", "transfer_funds", "*_prod"] }
```

- `deny_options` is matched against the option names of every `choice`
  question. `*` stands for any run of characters, none included; everything
  else is literal. Case does not count, nor do spaces around a name:
  `delete_*` denies `Delete_All` as well. Descriptions are not matched, and
  `noul` and `score` questions have no options to deny.
- The response lists what was removed, per question, in
  `eullm.policy_removed` (`{"action": ["delete_account"]}`), and so does the
  decision's audit record. The field is absent when the policy removed
  nothing; a request it removes nothing from is answered exactly as without a
  policy.
- A question left with fewer than two options is refused with a 422
  `policy_denied` that names the question and the options denied: what
  remains is not a choice. A client's own mistakes come first — a `choice`
  sent with one option is still an `invalid_question`.
- `version` is required and must be `1`. A file written for a later version
  may hold rules this engine does not know, and applying only the others
  would silently leave those out. A file that cannot be read or applied as
  written — missing, not JSON, an unknown key, an empty pattern, a later
  version — stops the server at startup instead of letting every option
  through. The startup log prints the patterns and where they came from;
  restart the server to change them.

### Decision traces (`EULLM_DECISION_TRACES`)

Training a decision model on the decisions it is actually asked to make
needs the text of those decisions, and the audit trail does not keep it, on
purpose: it records every decision's state as a SHA-256 only. Traces are
the explicit place for that text. They are off unless
`EULLM_DECISION_TRACES` names a directory (from the environment first and
the `.env` file second), they stay on the machine, and personal data is
redacted before anything is written.

```bash
EULLM_DECISION_TRACES=/data/traces eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m
```

`decisions.jsonl` in that directory gets one line for every decision the
audit trail records — the same decisions, those whose client had gone
included — and nothing for a request that was refused or abandoned:

| Key | Value |
|---|---|
| `schema` | `1`, the version of this shape. A change that would break a reader of it gets a new number |
| `id` | The audit record's `id`, which the response gave as `eullm.audit_id`: the same decision in the audit trail, and what feedback names it by |
| `timestamp` | The audit record's, RFC 3339 in UTC |
| `model` | The decision model, named as in the audit record |
| `readout` | `codes` or `verdict` |
| `mode` | `shared_prefix`, `batched` or `separate` |
| `state` | The state as the model read it: text as it is; a structured state as the readout writes it (indented JSON for `codes`, one line for `verdict`), its values redacted, so it is still JSON |
| `questions` | Each question as the model read it, after the decision policy, in the shape of a request: `type`, `instructions` (structured instructions as the compact JSON the model read) and `criteria` — `{"true": …, "false": …}` for `noul`, `""` where the question said nothing; option → description for `choice`, `""` for none; for `score`, each level as the model read it |
| `answers` | Each answer as the response returned it, without its `eullm` object: `noul`; `choice`, `probabilities`, `confidence`; `score`, `legend`, `probabilities`, `confidence` |
| `policy_removed` | Per question, the options the decision policy removed; `{}` when none |
| `client_disconnected` | `true` when the answers were computed after the client had gone, and never sent |

Every key is on every line. A line from the Jev-Style 0.8B on a CPU, wrapped
here, for a ticket that named a card number, an IBAN and a mobile number, on
a server whose policy denies `delete_*` and `transfer_funds`:

```json
{"schema": 1, "id": "b1149e83-332b-48c0-bab7-53725922c4de", "timestamp": "2026-10-01T13:35:34.529683343Z",
 "model": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "readout": "verdict", "mode": "shared_prefix",
 "state": "Sono Mario Rossi, il pagamento del 28/09 è stato addebitato due volte sulla carta [CARD]. Rimborsate su [IBAN] o chiamatemi al [PHONE].",
 "questions": {
   "is_urgent": {"type": "noul", "instructions": "Does the customer need an answer today?", "criteria": {"true": "", "false": ""}},
   "action": {"type": "choice", "instructions": "What should the support agent do?",
              "criteria": {"refund": "Refund the duplicate charge", "ask_human": "Hand the ticket to a person"}},
   "severity": {"type": "score", "instructions": "How severe is the problem?",
                "criteria": ["Cosmetic", "Degraded, with a workaround", "Blocking"]}},
 "answers": {
   "is_urgent": {"type": "noul", "noul": 0.525160129126831},
   "action": {"type": "choice", "choice": "refund",
              "probabilities": {"refund": 0.8943443753775355, "ask_human": 0.1056556246224647}, "confidence": 0.788688750755071},
   "severity": {"type": "score", "score": 1.400225522364445,
                "legend": {"0": "Cosmetic", "1": "Degraded, with a workaround", "2": "Blocking"},
                "probabilities": {"0": 0.03265542390587039, "1": 0.5344636298238142, "2": 0.4328809462703154},
                "confidence": 0.3016954447357214}},
 "policy_removed": {"action": ["transfer_funds", "delete_account"]},
 "client_disconnected": false}
```

**What is redacted.** Every text on the line — the state, instructions,
descriptions, levels and legend — has six kinds of personal data replaced
by a placeholder:

| Placeholder | What | Recognised |
|---|---|---|
| `[EMAIL]` | E-mail addresses | `name@domain.tld`, international letters included |
| `[PHONE]` | Phone numbers | With `+` or `00` and a country code, any country: `+39 333 1234567`, `+44 20 7946 0958`, `+1 (202) 555-0123`. Italian ones without it: a mobile, ten digits from a 3 (`333 1234567`, `333-123-4567`); a landline, 8 to 11 digits from a 0 (`06 1234 5678`, `(02) 12345678`) |
| `[IBAN]` | IBANs | Any country, whole, in groups of four, or by its parts as Italian documents print it (`IT 60 X 05428 11101 000000123456`), when the check digits match |
| `[CF]` | Italian codici fiscali | By their structure, omocodia included, so a mistyped check letter is caught too |
| `[CARD]` | Payment card numbers | 13 to 19 digits starting with 2 to 6, whole or in the groups cards are printed in, when the Luhn check passes |
| `[IP]` | IPv4 addresses | Four numbers from 0 to 255 joined by dots |

Dates, times, amounts, years, article numbers (`art. 2043 c.c.`,
`d.lgs. 196/2003`) and other numbers are left as they are.

It is pattern matching, and it misses things:

- Names, street addresses, dates of birth and every identifier not in the
  table — an identity card or passport number, a licence plate, an IPv6
  address, a partita IVA — stay as written.
- So does anything in the table written in a form it does not expect:
  `mario at example dot com`, a foreign number without its `+`
  (`(202) 555-0123`), a codice fiscale split by spaces, an IBAN or a card
  number with a wrong check digit.
- Question ids and option names are written as they are, as in the audit
  trail: they are what the answers and the feedback refer to, and redacting
  them could turn two options into one. Keep personal data out of them.

And it errs the other way: a four-part version number (`1.2.3.4`) becomes
`[IP]`; a code of 13 to 19 digits that happens to pass the Luhn check — one
in ten do — becomes `[CARD]`; ten digits from a 3, or 8 to 10 from a 0,
become `[PHONE]`. Treat the files as what they are: the text of the
decisions, less what the patterns catch.

**A trace that cannot be written** — a full disk, a directory removed —
never fails the decision: it is still made, audited and answered, and the
server log says the trace is missing. A directory that cannot be written at
startup stops the server instead, since whoever set the variable asked for
the traces. The file grows with every decision; move it away to start a new
one, and the next decision creates it again.

### Feedback (`POST /v1/systemone/feedback`)

A model trained on its own answers learns nothing it did not already know;
what teaches it is the answer that would have been right.
`POST /v1/systemone/feedback` records that for a decision, named by the
`eullm.audit_id` its response carried — from a person reviewing it, a rule
that knows better, or a larger model acting as teacher:

```bash
curl -s http://localhost:11434/v1/systemone/feedback -H 'Content-Type: application/json' -d '{
  "id": "094cd37b-9db6-4b58-acb7-54fb92f32a29",
  "answers": { "action": "refund", "is_urgent": true, "severity": 2 },
  "outcome": "Rimborsato il 02/10; il cliente ha confermato da mario.rossi@example.com",
  "source": "user"
}'
```

```json
{"id": "094cd37b-9db6-4b58-acb7-54fb92f32a29", "recorded": true}
```

| Field | Value |
|---|---|
| `id` | Required. The decision's `eullm.audit_id` |
| `answers` | Required. Question id → the answer that was right: an option's name for a `choice`, `true` or `false` for a `noul`, the level's index from 0 for a `score`. Only the questions there is something to say about; `{}` when the feedback gives an `outcome` alone |
| `outcome` | Optional. What came of the decision, as text, up to 16 KB |
| `source` | Optional. Who says so: `user`, `rule` or `teacher` |

It is appended to `feedback.jsonl`, next to `decisions.jsonl`, as one line
with every key, `null` for what was not given:

| Key | Value |
|---|---|
| `schema` | `1` |
| `kind` | `feedback` |
| `timestamp` | When the feedback was received, RFC 3339 in UTC |
| `id` | The decision's audit id, in lower case: the `id` of its line in `decisions.jsonl` |
| `answers` | As sent, in the order sent |
| `outcome` | As sent, redacted like every text in the traces, or `null` |
| `source` | As sent, or `null` |

The line the request above wrote:

```json
{"schema":1,"kind":"feedback","timestamp":"2026-10-01T13:49:08.356674032Z","id":"094cd37b-9db6-4b58-acb7-54fb92f32a29","answers":{"action":"refund","is_urgent":true,"severity":2},"outcome":"Rimborsato il 02/10; il cliente ha confermato da [EMAIL]","source":"user"}
```

Several feedbacks on one decision are several lines, and whoever reads them
decides which counts: the latest, or a person's over a rule's.

The server checks a feedback's shape, not its sense. Types and sizes are
checked — at most 64 answers, question ids and option names up to 1 KB, an
unknown key refused — and a wrong answer is refused with a 422
`invalid_question` naming its question. But the server keeps no index of
past decisions, so whether `id` names a decision in the traces, and whether
`"refund"` is one of that question's options, is for whoever joins the two
files to check. With traces off there is nowhere to keep a feedback, and the
endpoint answers 409 `traces_disabled`. It sits behind the same API key, IP
allowlist and origin checks as `/v1/systemone`, and its errors take the same
shape.

### Jev-Style decision models

[Jev-Style](https://github.com/lawrence3699/jev-style) (Apache-2.0)
publishes Qwen3.5 fine-tunes trained for the three question types of this
endpoint, and they read an answer differently — the *verdict readout*. Every
option is followed by a ` ->` slot, and the option's score is
`logit(" yes") − logit(" no")` at its slot. The options are all read in the
same pass, not as competing codes, so a `choice` can have up to 255 of them.
The probabilities are `softmax(scores / T)`, with the temperature `T` fitted
on held-out data and released with the model. It is applied by default; a
request's `temperature` overrides it, and `1` gives the probabilities of the
raw scores.

| Model | Input | Get it |
|---|---|---|
| Jev-Style-0.8B-Decision-v3 (0.53 GB in Q4_K_M) | Causal attention; question, options and slots within 2,048 tokens | `eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M` |
| Jev-Style-2B-Decision-v3 (1.3 GB in Q4_K_M) | Block-causal attention over 2,048-token blocks; options that do not fit one block become a numbered catalogue | `eullm pull hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M` |

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
eullm serve --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m --decision-ctx 25600
```

The server recognizes such a model when it loads, in one of two ways:

- from the `readout_config.json` released with the model, when that file
  sits next to the GGUF;
- otherwise from the model's name (`general.name`) — the case of a GGUF
  pulled on its own.

The load log says which and prints the temperature.

**The input is the one the model was trained on.** It is not a chat
prompt. The input is rebuilt token for token: each segment tokenized on
its own, with text never parsed for special tokens, and a structured
state serialized the way the release's runtime writes JSON. It is
computed the way that runtime computes it: the same micro-batches, flash
attention off on the CPU for the 2B, and one block per decode call for
the block-causal attention.

This was checked against that runtime, Q4_K_M on the CPU, on 24 cases for
the 0.8B and 28 for the 2B:

- short, JSON, long, and special-token-laden states;
- every question type, up to 30 options — and 70 in the 2B's numbered
  catalogue form.

The token ids were identical and, in `separate` mode, the scores agreed to
4e-16, asked one at a time and all together.

**Differences from the code readout:**

- `content_free` calibration is refused, since the model brings its own.
- `eullm.scores` replaces `eullm.logprobs`, and there is no `coverage`:
  every option is read at a slot of its own.
- Input over 25,600 tokens, or options that do not fit the model's
  budget, is refused rather than truncated. Raise `--decision-ctx` to
  25600 to allow the longest input the models accept.
- Only the release's global temperature is applied, as its own runtime
  does when it is not given a category — also when the GGUF carries an
  `eullm.decision.temperature`, which a code-readout model would use.

`shared_prefix` decodes the state once, as with any model, and each
question on its own after it:

- **2B:** its input is cut into 2,048-token blocks at the end of the state
  anyway, so sharing the state changes no score: `shared_prefix` gives the
  runtime's scores exactly, on the CPU and on an RTX 5070 Ti. It decodes one
  block per call, so its `batched` requests are answered, and reported, as
  `shared_prefix`.
- **0.8B:** its arithmetic depends on where a 1,024-token micro-batch
  starts, because its recurrent layers are computed in chunks from there,
  and sharing the state moves where its questions' micro-batches start. Its
  `shared_prefix` scores are therefore within the model's own noise of the
  runtime's: up to 0.025 in probability on the CPU, 0.039 on an RTX 5070 Ti.
  `separate` gives them exactly, at the cost of decoding the state again for
  every question: 64 questions about a 1,024-token state take 0.62 s on an
  RTX 5070 Ti in `shared_prefix` mode, 3.2 s in `separate`.

### jev-style with EuLLM: MCP server, CLI, Python client

[jev-style](https://github.com/lawrence3699/jev-style) (Apache-2.0,
`pip install jev-style`) is the Jev-Style models' own toolkit: an MCP
server, a command line, a Python client and a guard for a coding
agent's tool calls, all of which talk to any server that answers
`POST /v1/systemone`. Point them at
EuLLM with `JEV_STYLE_URL`. EuLLM must have a decision model loaded, with
`--decision-model`: jev-style's guard names its own model
(`jev-style-0.8b-decision-v3`) and its other tools name none, and EuLLM
reads both, as it reads every `jev…` name, as the decision model it has
loaded.

```bash
eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
eullm serve --port 11500 --decision-model jev-style-0.8b-decision-v3-gguf-q4_k_m
```

The MCP server gives an agent `decide`, `noul`, `choice`, `score` and
`model_info` as tools. In Claude Code:

```bash
claude mcp add jev-style --scope user -e JEV_STYLE_URL=http://localhost:11500 -- uvx jev-style@0.3.0 mcp
```

In Cursor, Claude Desktop and other clients configured with an
`mcpServers` block:

```json
{
  "mcpServers": {
    "jev-style": {
      "command": "uvx",
      "args": ["jev-style@0.3.0", "mcp"],
      "env": { "JEV_STYLE_URL": "http://localhost:11500" }
    }
  }
}
```

When EuLLM requires API keys (`EULLM_API_KEYS`), give one as
`JEV_STYLE_API_KEY` in the same `env` block (`-e JEV_STYLE_API_KEY=…` with
`claude mcp add`); jev-style sends it as a bearer token. The MCP server
talks only to a loopback address unless started with `--allow-remote`.

The command line and the client read the same variables:

```bash
export JEV_STYLE_URL=http://localhost:11500
jev-style decide "I was charged twice for March." --noul "This is about billing." \
  --choice "Which team?::billing,shipping,technical"
jev-style eval labelled.jsonl        # accuracy and calibration on your own labels
```

```python
from jev_style import JevStyle, choice, noul
js = JevStyle(base_url="http://localhost:11500")
js.decide("I was charged twice.", {"billing": noul("This is about billing."),
                                   "team": choice("Which team?", ["billing", "shipping"])})
```

The guard (`jev-style guard`, a `PreToolUse` hook that checks each tool
call before it runs) takes the
server from `JEV_STYLE_GUARD_URL` or `JEV_STYLE_URL`, else from `server_url`
in its `guard_config.json`. It gives up after `timeout_s` (8 seconds by default)
and then asks the user instead; on a slow CPU raise it. A request it gives
up on stops at EuLLM's next question, and a decision already computed is in
the audit trail with `client_disconnected: true`.

What differs from jev-style's own server:

- **Probabilities** are calibrated with the release's global temperature
  (0.880 for the 0.8B, 0.828 for the 2B). jev-style's server uses instead
  the temperature fitted for typed questions of the same kind where the
  release has one: for the 0.8B, 0.98 to 1.01 for yes/no questions and for
  choices and scores of 3 to 5 options, whose probabilities it therefore
  gives a little less sharp; the 2B has none. The scores underneath,
  `eullm.scores`, are the same.
- **A `choice` with one option** is refused (see the limits above).
- **Errors** carry EuLLM's messages, in jev-style's shape.

## API Reference

The Engine exposes two sets of endpoints: the native EULLM API (Ollama-compatible) and an OpenAI-compatible API. CORS is enabled for browser-based tools.

Request bodies are JSON. As Ollama does, the Engine reads a body as JSON when its `Content-Type` is `application/json`, missing, `text/plain` or `application/x-www-form-urlencoded` — the last is what `curl -d` sends, so Ollama's curl examples work as they stand. A body labelled as anything else (`multipart/form-data`, `application/octet-stream`) gets a 415.

### EULLM API (Ollama-compatible)

#### `GET /api/version`

Returns the Engine version.

```bash
curl http://localhost:11434/api/version
```

```json
{
  "version": "0.7.20",
  "api_port": 11434,
  "model_swaps": 0,
  "max_loaded_models": 1,
  "loaded_models": 1,
  "generation_evictions": 0
}
```

Besides Ollama's `version`, EuLLM reports the API port, `model_swaps` (models
evicted to make room in another slot), and the [resident
models](#several-models-at-once): the `--max-loaded-models` limit, how many
generation models are loaded, and how many were unloaded to make room for
another.

#### `GET /api/tags`

List available models. Returns the loaded models first, the most recently used first, each with `"loaded": true` (what admin dashboards check for health), followed by catalog entries and the other models in the store. A model on disk has `details.source`: `store` for the models folder, `library` for a file used where it is (a model library, or a `.gguf` put in the models folder by hand), which EuLLM never deletes. `GET /api/ps` lists only what is loaded, with what each model holds. With `--auto-model`, `auto` comes last, with `details.family: "eullm-router"` and `details.candidates`.

```bash
curl http://localhost:11434/api/tags
```

```json
{
  "models": [
    {
      "name": "eullm/legal-it-4b",
      "size": 4500000000,
      "digest": "sha256:le7a1it0...",
      "details": {
        "format": "gguf",
        "family": "qwen3",
        "parameter_size": "12B",
        "quantization_level": "Q4_K_M",
        "domain": "legal",
        "source_model": "Qwen/Qwen3-14B"
      }
    }
  ]
}
```

#### `POST /api/generate`

Generate text from a prompt. Uses real llama.cpp inference.

```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "eullm/legal-it-4b", "prompt": "Cosa dice l'\''art. 2043 del Codice Civile?"}'
```

```json
{
  "model": "eullm/legal-it-4b",
  "created_at": "2026-03-21T10:00:00Z",
  "response": "L'articolo 2043 del Codice Civile...",
  "done": true,
  "done_reason": "stop",
  "total_duration": 1500000000,
  "load_duration": 0,
  "prompt_eval_count": 15,
  "prompt_eval_duration": 90000000,
  "eval_count": 128,
  "eval_duration": 1400000000
}
```

The durations are Ollama's, in nanoseconds: `prompt_eval_duration` is the time
spent reading the prompt (images included), `eval_duration` the time spent
writing the answer, and `total_duration` the whole request, including
`load_duration` when the request had to load its model. With `--mtp`, the
answer also says how many drafts the MTP head proposed and the model kept,
under llama-server's names: `"draft_n"` and `"draft_n_accepted"`. On
`/v1/chat/completions` the same figures come in llama-server's `timings`
object (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, and the two
draft counts), which OpenAI clients ignore.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model` | loaded model | Model name |
| `prompt` | (required) | Input prompt |
| `max_tokens` / `num_predict` | 512 | Maximum tokens to generate (see note below) |
| `temperature` | 0.7 | Sampling temperature |
| `stream` | true | Stream response token-by-token (NDJSON) |
| `num_ctx` | server per-slot ctx | Per-request context window budget (clamped to per-slot max) |
| `format` | — | Set to `"json"` for constrained JSON decoding (GBNF grammar) |
| `keep_alive` | `--keep-alive` | How long the model stays loaded once this request is over: a duration (`"5m"`), a number of seconds, `0` to unload it as soon as the answer has been sent, `-1` to keep it. Counted from when the model goes idle, with the keep_alive of the last request that arrived — as in Ollama, so a `keep_alive: 0` sent while an answer is still coming unloads the model once it is over |
| `cache_prompt` | true | EuLLM extension, llama.cpp's name: `false` decodes the whole prompt instead of starting from what the slot holds from the request before. Slower for a long conversation, but on a GPU the only way the same request gets the same answer twice at temperature 0 (see below). Also on `/api/chat` and `/v1/chat/completions`, at the top level or in `options` |
| `options` | — | Ollama-style nested object for `num_predict`, `temperature`, `num_ctx` |

**Same request, same answer.** Temperature 0 (or `top_k: 1`) with a `seed` picks the same token from the same numbers, but on a GPU the numbers themselves depend on how many of a prompt's tokens are decoded together, and by default a request starts from the part of the prompt its slot already holds from the request before. The same request can then get a different answer depending on what came before it, or on whether the model was reloaded in between — in AutoBench on an RTX 5070 Ti, half of qwen3-8b's GSM8K answers came out different the second time they were asked. A CPU decodes the same way whatever the batch, so this is a GPU matter. Send `"cache_prompt": false` when answers must reproduce (evaluations, comparisons, tests): AutoBench does for every answer it compares.

**Ollama `options` support:** Parameters can be passed at the top level (OpenAI style) or nested inside an `options` object (Ollama style). Top-level values take precedence.

```json
{
  "prompt": "Ciao!",
  "options": {
    "num_predict": 1024,
    "temperature": 0.5,
    "num_ctx": 8192
  }
}
```

**`num_predict` capping:** If `num_predict` (or `max_tokens`) would exceed the remaining context budget (`effective_ctx - prompt_tokens`), it is automatically capped. The Engine logs a `WARN` when this happens — see the [Logging & Troubleshooting](#logging--troubleshooting) section.

**Streaming:** When `"stream": true` (the default), the response is sent as **NDJSON** (newline-delimited JSON). Each line is a complete JSON object with `"response"` (the token) and `"done": false`. The final line has `"done": true` with timing stats. Content-Type is `application/x-ndjson`.

**Loading and unloading without generating**, as in Ollama: an empty `prompt`
(or, on `/api/chat`, empty `messages`) loads the model and answers
`"done_reason": "load"`. With `"keep_alive": 0` it unloads that model instead
— only that one, the others staying loaded — and answers
`"done_reason": "unload"`; a model that is not loaded is not loaded first.
Answering other requests, the model goes when they are over.

```bash
# Streaming example (NDJSON — same format as Ollama)
curl -N http://localhost:11434/api/generate \
  -d '{"model": "local", "prompt": "Hello", "stream": true}'
# Each line: {"model":"...","response":"token","done":false}
# Final line: {"model":"...","response":"","done":true,"done_reason":"stop",...}
```

#### `POST /api/chat`

Chat completion with message history. Messages are formatted as ChatML internally. Supports `"stream": true` for token-by-token NDJSON streaming (same format as Ollama).

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [
      {"role": "user", "content": "Spiegami il GDPR in breve."}
    ]
  }'

# Streaming
curl -N http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [{"role": "user", "content": "Ciao!"}],
    "stream": true
  }'
```

#### `GET /api/ps`

The models in memory, in Ollama's shape: every generation model, the most
recently used first, then the embedding and decision models.

```bash
curl http://localhost:11434/api/ps
```

```json
{
  "models": [
    {
      "name": "qwen3-8b",
      "model": "qwen3-8b",
      "size": 5603000000,
      "digest": "sha256:…",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen3",
        "families": ["qwen3"],
        "parameter_size": "8.2B",
        "quantization_level": "Q4_K_M"
      },
      "expires_at": "2026-10-01T15:42:07.512+00:00",
      "size_vram": 5410000000,
      "context_length": 4096,
      "eullm": {
        "slot": "generation",
        "in_flight": 0,
        "last_used": "2026-10-01T15:37:07.512+00:00",
        "batch_size": 1,
        "gpu_layers": -1,
        "reserved_companion": false,
        "launch": false,
        "size_vram_measured": true
      }
    }
  ]
}
```

- `size` is the model's weights, its projector and its KV cache; `size_vram`
  what free VRAM lost when it loaded (`eullm.size_vram_measured`), or an
  estimate from its layers on the GPU when that could not be measured.
- `expires_at` is when its keep_alive runs out once it is idle; a model kept
  for good gets a date centuries ahead, as in Ollama, never `null`.
- `context_length` is what one request gets: a scheduler's context is shared
  by its `batch_size` slots.
- `eullm` is EuLLM's own: which slot holds the model, how many requests it is
  answering, when it was last used, and whether it is a reserved companion
  (`--embedding-model`, `--decision-model`) or the model `eullm run` started
  with.

#### `GET /api/about`

What the chat UI's About page (in its sidebar) shows: this build, the machine,
the devices llama.cpp can use, where models and the audit trail are kept, and
who may use the server. Read once per call; behind the same perimeter as the
rest of `/api`.

```bash
curl http://localhost:11434/api/about
```

```json
{
  "version": "0.8.5", "commit": "a3bbe56fc2b6", "backend": "CUDA",
  "llama_cpp": "b11925", "release_asset": "eullm-linux-x64-cuda",
  "cpu_features": "CPU : SSE3 = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | …",
  "machine": {
    "os": "Linux (Ubuntu 24.04)", "kernel": "6.8.0", "arch": "x86_64", "host": "b550",
    "cpu": "AMD Ryzen 9 5950X 16-Core Processor", "cores": 32, "physical_cores": 16,
    "ram_total": 67328999424, "started_ms": 1793298619000,
    "devices": [
      { "name": "CUDA0", "description": "NVIDIA GeForce RTX 5070 Ti", "kind": "gpu", "memory_total": 17094475776 },
      { "name": "CPU", "description": "AMD Ryzen 9 5950X 16-Core Processor", "kind": "cpu", "memory_total": null }
    ]
  },
  "models_dir": "/home/me/.eullm/models", "models_dir_source": "default",
  "model_libraries": ["/home/me/.lmstudio/models"],
  "audit_log": "/home/me/.eullm/audit/audit.jsonl",
  "api_port": 11434,
  "protections": {
    "api_keys": false,
    "allowed_ips": "127.0.0.1/32, ::1/128",
    "allowed_ips_source": "default (loopback only — nothing configured)"
  }
}
```

- `commit` ends in `-dirty` for a build with local changes; `release_asset`
  is the release file the binary was published as, `null` for a build from
  source.
- `devices` lists what llama.cpp's backends offer, the CPU included, with the
  memory of each GPU.
- `protections` says whether every request needs an API key (never the keys
  themselves) and which networks are admitted without one
  (`EULLM_ALLOWED_IPS`).

#### `GET /api/monitor`

What the chat UI's Monitor page (the pulse icon in its top bar) shows: the
requests running, the last 50 answered, and the machine's last two minutes,
one sample a second. Kept in the engine's memory only: nothing is written to
disk (the [audit trail](#audit-trail) is the durable record) and nothing is
sent anywhere. The models in memory are `/api/ps`.

```bash
curl http://localhost:11434/api/monitor
```

```json
{
  "active": [
    { "id": 12, "model": "qwen3-8b", "endpoint": "chat", "started_ms": 1793280000000,
      "phase": "generating", "prompt_tokens": null, "output_tokens": 210,
      "prefill_tps": null, "decode_tps": 131.8, "first_token_ms": 84,
      "duration_ms": null, "drafts": null, "ended": null }
  ],
  "history": [
    { "id": 11, "model": "qwen3-8b", "endpoint": "chat.completions", "started_ms": 1793279990000,
      "phase": "done", "prompt_tokens": 1520, "output_tokens": 412,
      "prefill_tps": 5210.4, "decode_tps": 134.2, "first_token_ms": 310,
      "duration_ms": 3402, "drafts": [300, 162], "ended": "stop" }
  ],
  "system": [
    { "t_ms": 1793280000000, "cpu_pct": 7.5, "cores": 16,
      "ram_used": 21474836480, "ram_total": 68719476736, "process_ram": 1288490188,
      "disk_read_bps": 0,
      "gpus": [
        { "name": "NVIDIA GeForce RTX 5070 Ti", "source": "nvml", "util_pct": 97.0,
          "vram_used": 12884901888, "vram_total": 17179869184, "temp_c": 64.0,
          "power_w": 271.3, "power_limit_w": 300.0,
          "pcie_rx_bps": 1048576, "pcie_tx_bps": 524288 }
      ] }
  ]
}
```

- `phase` is `reading` (queued or reading the prompt), `generating`,
  `done` or `failed`; `ended` is the stop reason (`stop`, `length`) or the
  error. A running request's `decode_tps` is its speed so far; a finished
  one's are the engine's own figures, as in the audit trail. `drafts` is
  `[proposed, kept]` with `--mtp`.
- `system` is sampled only while someone reads this endpoint: the sampler
  starts with the first call and stops 30 seconds after the last, so a
  server nobody watches pays nothing for it, and the first call after a quiet
  spell returns no samples yet. `disk_read_bps` is what the engine process
  read from disk in that second (a model loading, weights paged in through
  mmap).
- GPUs come from the NVIDIA driver (NVML, loaded at run time; Linux and
  Windows) with load, VRAM, temperature, power and PCIe traffic; from the
  `amdgpu` driver's files on Linux, without PCIe traffic; and for any other GPU
  (Metal, Vulkan, Intel) from llama.cpp's device list, VRAM only. What a
  driver does not report is `null`.

#### `POST /api/unload`

EuLLM extension: unload generation models now, freeing their VRAM. With a body
`{"model": "qwen3-8b"}`, that model only; without one, every generation model.
Requests still running on an unloaded model are cut off — to let them finish
first, send an empty request with `"keep_alive": 0` instead (see above).

```json
{ "unloaded": "qwen3-8b", "unloaded_all": ["qwen3-8b"] }
```

`unloaded` names the first model unloaded, or is `null` when none was loaded —
which is not an error — and `unloaded_all` lists every one.

#### `POST /api/route`

Which model [`"model": "auto"`](#model-auto-the-decision-model-chooses-the-model)
would choose for a request, without generating anything or loading a
generation model. An EuLLM extension. The body is the request as it would go
to `/api/chat`, `/v1/chat/completions` (`messages`, and `tools`) or
`/api/generate` (`prompt`); `batch_size` and `ctx_size` are read as there.
Without `--auto-model` it answers 404; a body with neither `messages` nor
`prompt` gets a 400.

```bash
curl http://localhost:11434/api/route -d '{"model": "auto",
  "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]}'
```

```json
{
  "model": "qwen3-8b",
  "reason": "decided",
  "fallback": "qwen3-8b",
  "candidates": [
    {"model": "qwen3-4b", "description": "Short everyday requests, ...", "probability": 0.22, "resident": true},
    {"model": "qwen3-8b", "description": "Multi-step reasoning, ...", "probability": 0.78, "resident": true}
  ],
  "excluded": [],
  "confidence": 0.56,
  "decision_model": "Jev-Style-0.8B-Decision-v3-Q4_K_M",
  "decision_ms": 31.4,
  "state": "Request to answer, with its context. ...",
  "question": {"type": "choice", "instructions": "Which model should answer the latest message? ...",
               "criteria": {"qwen3-4b": "Short everyday requests, ...", "qwen3-8b": "Multi-step reasoning, ..."}},
  "route_id": "<uuid>"
}
```

`candidates` are the ones offered, in order; `excluded` the others, with
why. `state` and `question` are exactly what the decision model read, in
`/v1/systemone`'s request shape, so the same question worded otherwise can be
asked of `/v1/systemone` about the same state. Each call is audited as a
`route` line with `"dry_run": true`.

#### `POST /api/show`

Get model metadata.

```bash
curl -X POST http://localhost:11434/api/show \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-4b"}'
```

#### `DELETE /api/delete`

Remove a model's files from the models folder, as `eullm rm` does; Ollama's
endpoint of the name. The model is named in `model` (or `name`, Ollama's
older field).

```bash
curl -X DELETE http://localhost:11434/api/delete \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b"}'
```

```json
{ "deleted": "qwen3-8b", "freed_bytes": 5027783936 }
```

- 404 when no model of that name is in the models folder.
- 403 for a file EuLLM uses where it is (`details.source: "library"` in
  `/api/tags`): delete the file yourself.
- 409 for a model in memory: unload it first (`POST /api/unload`), so that no
  request it is answering loses its weights.

#### `GET /api/hf/search`

Search Hugging Face for GGUF repos, through the engine (the browser of the
chat UI contacts nothing else). `q` is the text to look for, `limit` how many
(25 by default, at most 100), `sort` the order: `downloads` (the default),
`likes`, `trending` or `newest`. With a `sort` and no `q`, every GGUF repo in
that order, which is what the Model hub's Discover tab opens with.

```bash
curl "http://localhost:11434/api/hf/search?q=qwen3&sort=newest&limit=5"
```

#### `POST /api/pull`

Trigger a model download.

```bash
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "eullm/legal-it-4b"}'
```

#### `POST /api/embed`

Embed one or more texts. `input` accepts a single string or an array. See
[Text Embeddings and the Embedding Slot](#text-embeddings-and-the-embedding-slot)
for how this coexists with the generation model.

```bash
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["first chunk", "second chunk"]}'
```

```json
{
  "model": "bge-m3",
  "embeddings": [[0.013, -0.021, ...], [0.008, 0.044, ...]],
  "total_duration": 41250000,
  "load_duration": 120000,
  "prompt_eval_count": 8
}
```

As in Ollama, `prompt_eval_count` is the tokens the model read, over all
inputs and after truncation to the embedder's context, and the durations
are nanoseconds: `load_duration` getting the model into its slot (next to
nothing when it was already there), `total_duration` the whole request.

### OpenAI-Compatible API

These endpoints allow using EULLM as a drop-in backend for any tool that supports the OpenAI API: Open WebUI, LangChain, LlamaIndex, n8n, Flowise, etc.

#### `GET /v1/models`

List models in OpenAI format.

```bash
curl http://localhost:11434/v1/models
```

When a decision model is loaded, it is listed for System One clients too —
jev-style's `model_info` tool and the System One SDKs' `models.list()`.
Its entry in `data` carries `context_tokens`, the most tokens one
`/v1/systemone` request may hold (`--decision-ctx`, or less when the model
itself reads fewer), for a Jev-Style model `head_max_tokens`, what one
question with its options may take on its own, and `"eullm": {"slot":
"decision"}`; and the top-level `models`, the list the System One SDKs read,
names it. With no decision model loaded, `models` is empty. With
`--auto-model`, `auto` is the last entry of `data`.

```json
{
  "object": "list",
  "data": [
    { "id": "qwen3-4b", "object": "model", "created": 1700000000, "owned_by": "eullm" },
    { "id": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "object": "model", "created": 1700000000,
      "owned_by": "eullm", "context_tokens": 8192, "head_max_tokens": 2048,
      "eullm": { "slot": "decision", "readout": "verdict" } },
    ...
  ],
  "models": [
    { "name": "Jev-Style-0.8B-Decision-v3-Q4_K_M", "release_date": "2026-09-24",
      "description": "Jev-Style-0.8B-Decision-v3 on EuLLM: typed decisions (noul / choice / score)" }
  ]
}
```

#### `POST /v1/chat/completions`

Chat completion in OpenAI format. Real inference with token counts. Supports `"stream": true` for SSE streaming (OpenAI `chat.completion.chunk` format with `[DONE]` terminator).

```bash
# Non-streaming
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'

# Streaming (same format as OpenAI API)
curl -N http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "eullm/legal-it-4b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

**Output limit:** `max_completion_tokens`, the name OpenAI's Chat Completions
API now gives it, or `max_tokens`, the deprecated name older clients still
send. A request carrying both is limited by `max_completion_tokens`; a `null`
counts as not sent. Either name takes a non-negative integer, and any other
value means no limit. The limit counts every token the model generates, its
reasoning included, which is what OpenAI means by `max_completion_tokens`.
Without one the model generates until it stops or its context is full, and a
limit larger than the room left in the context is capped to it, as
[`num_predict`](#post-apigenerate) is. An answer cut short either way ends
with `"finish_reason": "length"`.

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eullm/legal-it-4b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 256
  }'
```

#### `POST /v1/embeddings`

Embed one or more texts in OpenAI format. Same underlying embedding slot as
`/api/embed`.

```bash
curl -X POST http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "first chunk"}'
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "embedding": [0.013, -0.021, ...], "index": 0}],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

`usage` counts the tokens the model read, after truncation to the
embedder's context; an embedding generates nothing, so the total is the
prompt.

#### `POST /v1/systemone`

Typed decisions (`noul`, `choice`, `score`) about a state, in the System One
API shape. Not an OpenAI endpoint; it sits under `/v1` because that is where
System One clients look for it. See
[Decisions: `/v1/systemone`](#decisions-v1systemone-and-the-decision-slot).

#### `POST /v1/systemone/feedback`

The answers that would have been right for a decision `/v1/systemone`
made, stored next to its trace when decision traces are on. See
[Feedback](#feedback-post-v1systemonefeedback).

## Model Catalog

The engine ships with a catalog of 25 open models (Qwen, Mistral, DeepSeek,
Gemma and others),
downloaded from Hugging Face by `eullm pull NAME`. `eullm list` shows them, and
the chat UI's model browser says whether each fits your hardware and under
which licence it comes. 24 are Apache 2.0 or MIT, the two Gemma 4 models
among them (Google moved Gemma to Apache 2.0 with Gemma 4); one DeepSeek model
comes under the DeepSeek License Agreement, which carries use restrictions of
its own.

## Audit Trail

Every inference request is logged to a persistent JSONL file at `~/.eullm/audit/audit.jsonl`, or `audit.jsonl` in `EULLM_AUDIT_DIR` when that is set (in the environment, or in `.env`). Each line is a self-contained JSON object.

| Field | Type | Description |
|---|---|---|
| `id` | UUID v4 | Unique inference ID |
| `timestamp` | DateTime (UTC) | Request time |
| `model` | String | Model name |
| `request_type` | String | `generate`, `chat`, `chat.completions`, `systemone`, `route` |
| `input_tokens` | u32 | Input token count |
| `output_tokens` | u32 | Output token count |
| `duration_ms` | u64 | Inference duration |
| `user_id` | Option\<String\> | Optional user identifier |
| `decision` | Object, `systemone` only | `state_sha256`, `readout`, `mode`, `calibration`, `temperature`, `confidence_method` (`normalized_max_probability`; absent, and `normalized_entropy`, on lines written up to 0.7.20), `client_disconnected` (only when true: the answers were computed after the client had gone, and never sent), `policy_removed` (only when the [decision policy](#a-server-side-decision-policy-eullm_decision_policy) removed options: per question, the options the model never read), and per answer: `id`, `type`, `labels`, `logprobs` or `scores`, `raw_probabilities`, `probabilities`, `coverage`, `answer`, `confidence` |
| `routing` | Object, `route` only | How [`"model": "auto"`](#model-auto-the-decision-model-chooses-the-model) routed one request: `requested`, `model` (chosen), `reason`, `fallback`, `candidates` (offered, in order), `excluded` (with `why`), `decision_model`, `decision_ms`, `dry_run` (only for `POST /api/route`), `error`. The line's `id` is the route's id; its `model` is the decision model, and its `decision` the decision record as above |
| `route` | Object, routed answers only | `id` (the `route` line's), `requested` (`auto`), and `fallback` (`load_failed: …`, only when the chosen model did not load) |
| `draft_n`, `draft_n_accepted` | u32, `--mtp` only | The MTP drafts the model proposed for this answer, and the ones it kept. Absent when it drafted none |

**Example audit entry:**

```json
{"id":"a1b2c3d4-...","timestamp":"2026-03-21T14:30:00Z","model":"eullm/legal-it-4b","request_type":"chat","input_tokens":15,"output_tokens":128,"duration_ms":1200,"user_id":null}
```

The JSONL format allows:
- Append-only writes (crash-safe)
- Easy to grep, tail, stream
- Each line independently parseable
- Compatible with log analysis tools (Loki, ELK, etc.)

It is designed to support the record-keeping the EU AI Act (Regulation 2024/1689) asks for; on its own it does not make a system compliant.

## GPU Acceleration

| Feature flag | GPU backend | Build command |
|---|---|---|
| `cuda` | NVIDIA CUDA | `cargo build --release --features cuda` |
| `rocm` | AMD ROCm | `cargo build --release --features rocm` |
| `vulkan` | Cross-platform | `cargo build --release --features vulkan` |
| `metal` | Apple Silicon | `cargo build --release --features metal` |
| *(none)* | CPU only | `cargo build --release` |

How much of the model goes on the GPU is decided automatically on every
load — see [Automatic GPU sizing](#automatic-gpu-sizing) above. `--gpu-layers
0` forces CPU-only inference; `--gpu-layers N` caps the offload at N layers.
Sizing reads free VRAM on every GPU backend (CUDA, ROCm, Vulkan, Metal); with
no GPU, `--gpu-layers` is used as given.

## Integration Examples

### With Open WebUI

```bash
# Start Engine
eullm run ./model.gguf

# In Open WebUI settings, set API URL to:
# http://localhost:11434
```

### With LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    model="eullm/legal-it-4b",
    api_key="not-needed"
)

response = llm.invoke("Spiegami l'art. 2043 del Codice Civile.")
```

### With curl

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Ciao!"}]}'
```

## Logging & Troubleshooting

The Engine uses [`tracing`](https://docs.rs/tracing) for structured logging. Control verbosity with the `RUST_LOG` environment variable.

### Log levels

```bash
# Minimal (errors only)
RUST_LOG=error eullm run ./model.gguf

# Normal operation (the default) — shows request params, context budget, cap warnings
RUST_LOG=eullm=info eullm run ./model.gguf

# Verbose — adds prefill chunk details, decode loop, batch scheduling
RUST_LOG=eullm=debug eullm run ./model.gguf

# llama.cpp's own log as well: what it loads where, and why it refuses a load
# that the engine reports only as "null result from llama cpp"
RUST_LOG=eullm=info,llama-cpp-2=info eullm run ./model.gguf
```

The engine's target is `eullm` (the binary's name); `eullm_engine` matches
nothing. llama.cpp's own log is dropped unless `RUST_LOG` names its target,
`llama-cpp-2`; then all of it is shown, whatever level the filter gives it.

### What gets logged

| Level | Message | When |
|---|---|---|
| `INFO` | `Request params: max_tokens=N, temperature=T, num_ctx=X` | Every API request (routes layer) |
| `INFO` | `Seq N: prompt=P tokens, max_output=M, effective_ctx=C` | After tokenization (scheduler) |
| `INFO` | `Generate: prompt=P tokens, max_output=M, effective_ctx=C` | Sequential generate path |
| `INFO` | `Stream: prompt=P tokens, max_output=M, effective_ctx=C` | Sequential streaming path |
| `WARN` | `num_predict capped: requested=R, effective=E (context=C, prompt_tokens=P)` | When `num_predict` exceeds remaining context budget |
| `DEBUG` | `Prefilling seq N with P tokens in chunks of B` | Prefill chunking details (scheduler) |
| `DEBUG` | `Sequence N prefilled (P prompt tokens)` | After successful prefill |
| `ERROR` | `Prompt (P tokens) does not fit in context window (C)` | Prompt too long for context |

### Common issues

#### Truncated output / fewer tokens than expected

The model generates fewer tokens than `num_predict` requested.

**Diagnose:** Run with `RUST_LOG=eullm_engine=info` and look for the `WARN num_predict capped` message.

**Cause:** The prompt consumed most of the context window, leaving less room than `num_predict`.

**Fix:** Either:
- Increase `--ctx-size` on the server (requires more RAM/VRAM)
- Send `num_ctx` in the request to override per-request: `"num_ctx": 8192`
- Reduce prompt length
- Lower `num_predict` to match your actual needs

#### Prompt does not fit in context window

**Error:** `Prompt (N tokens) does not fit in context window (C)`

**Cause:** The tokenized prompt is longer than the effective context size.

**Fix:** Increase `--ctx-size` or send a shorter prompt. You can also pass `"num_ctx": 16384` per-request (clamped to server max).

#### Long generation latency on first request

**Cause:** The first prefill is slow because the KV cache is being allocated. Subsequent requests reuse allocated memory.

**Fix:** This is normal. For benchmarking, discard the first request as warmup.

## Implementation Status

| Component | Status |
|---|---|
| CLI (pull, run, list, import, show, rm, serve, unload, update, finetune, forge) | Implemented |
| Real inference (llama.cpp b12032 via llama-cpp-2 0.1.159) | Implemented |
| EULLM API routes (Ollama-compatible) | Implemented |
| OpenAI-compatible API | Implemented |
| GPU acceleration (CUDA, ROCm, Vulkan, Metal) | Implemented (feature flags) |
| CORS (Open WebUI compatibility) | Implemented |
| Model catalog (25 models) | Implemented |
| Local model store (~/.eullm/models/) | Implemented |
| Model download (HuggingFace, streaming with progress) | Implemented |
| Dynamic model swap (load/unload via API, dynamic batch_size/ctx_size) | Implemented |
| KV cache (F16 default, automatic fallback from quantized types) | Implemented |
| Constrained JSON decoding (format: "json" via GBNF) | Implemented |
| Continuous batching scheduler | Implemented |
| Audit trail (persistent JSONL) | Implemented |
| Streaming (NDJSON for Ollama, SSE for OpenAI) | Implemented |
| ChatML prompt formatting | Implemented |
| Interactive chat REPL | Implemented |
| Daemon mode (--daemon) | Implemented |
| EU registry download | Planned: models download from Hugging Face today |
