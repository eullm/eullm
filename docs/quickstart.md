# Quick start

How to start EuLLM, which options to use, and scripts to copy. For the reasons
behind each choice, the measurements and the internals, see the
[engine guide](engine-guide.md); every flag is also in the
[engine reference](engine.md).

## 1. Install

```bash
# Linux / macOS: picks the right build for your GPU, installs to ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/eullm/eullm/main/installer/install.sh | sh
```

```powershell
# Windows: picks CPU, CUDA or Vulkan, adds eullm to PATH
irm https://raw.githubusercontent.com/eullm/eullm/main/installer/install.ps1 | iex
```

Other builds (AMD ROCm, data-centre GPUs, ARM) are on the
[platforms page](platforms.md). `eullm update` installs a newer release later.

## 2. Start it

```bash
eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
```

That is all: the model downloads once, loads on the GPU if there is one, a chat
opens in the browser, and you can also type in the terminal. Any GGUF works:

```bash
eullm run ./my-model.gguf          # a file you have
eullm run                          # no model: pick one from a list
eullm list                         # models already downloaded
eullm import                       # what was found in the model folders, and what was left out
```

While it runs, any Ollama or OpenAI client can use it at
`http://localhost:11434` (Ollama API) and `http://localhost:11434/v1` (OpenAI
API): Open WebUI, Cherry Studio, Continue, LangChain, n8n...

### As a server, without a terminal chat

```bash
eullm serve                        # starts empty: each request names its model
eullm run ./my-model.gguf --daemon # loads the model and goes to the background
```

With `serve`, the model a request names (`"model": "qwen3-8b"`) is loaded on
the first request and swapped when another one is asked for. Models are
looked up among the ones `eullm list` shows.

## 3. The options you will actually use

They go after the model: `eullm run MODEL --ctx-size 16384 --web`. The same
options work for `run` and `serve`.

### Basics

| Option | Default | What it does |
|---|---|---|
| `-c`, `--ctx-size N` | 4096 | How much text the model sees at once (prompt + answer), in tokens. Raise it for long documents and long chats: 16384 or 32768. Costs VRAM |
| `--batch-size N` | 1 | How many requests are answered at the same time. **The context is split among them**: `-c 16384 --batch-size 4` gives each request 4096. Raise both together |
| `-p`, `--port N` | 11434 | Port of the API (the same as Ollama's) |
| `--ui-port N` | 11435 | Port of the browser chat |
| `--no-ui` | | Turn the browser chat off (`run` and `serve`; it is on by default for both) |
| `--cli` | | `run`: chat in the terminal only, without opening the browser |
| `-t`, `--threads N` | one per physical core | CPU threads |
| `--keep-alive 10m` | never | Unload the model after that long without requests, to free the GPU |

### GPU and memory

| Option | Default | What it does |
|---|---|---|
| (nothing) | | EuLLM measures the free VRAM and puts on the GPU as much of the model as fits. You rarely need the options below |
| `--gpu-layers N` | all | At most N layers on the GPU; `0` runs on the CPU only |
| `--fit-strict` | | Refuse to load a model that does not fit entirely on the GPU, instead of running it partly on the CPU |
| `--cache-type-k q8_0 --cache-type-v q8_0` | f16 | Store the context in half the memory, with a very small quality cost: room for a longer `-c` |
| `--max-loaded-models N` | 1 | Keep up to N models loaded at once (`serve`), when they fit together |

### Large MoE models on a small GPU

For mixture-of-experts models (Qwen3-30B-A3B, Qwen3.6-35B-A3B, GLM, DeepSeek...)
that do not fit in VRAM.

| Option | What it does |
|---|---|
| `--cpu-moe` | Keep the experts in RAM and everything else on the GPU: a 20+ GB MoE runs on a 12 GB card |
| `--n-cpu-moe N` | The same, for the first N layers only: use the VRAM `--cpu-moe` leaves empty |
| `--moe-cache auto` | Keep all experts in RAM and use the free VRAM for the ones the model uses most: writes 2× faster on one NVIDIA GPU. Experimental; it pays off only when the cache can hold most of the experts the model uses |
| `--n-ubatch 2048` | Read long prompts faster when the experts are in RAM (`--moe-cache` sets it by itself) |

### Features

| Option | What it does |
|---|---|
| `--web` | When a message contains a link, the page is fetched and given to the model |
| `--mtp 2` | Faster writing with models that ship an MTP head (unsloth's `*-MTP-GGUF` Qwen3.5/3.6), one request at a time |
| `--embedding-model NAME` | Also load an embedding model, for RAG (`/v1/embeddings`) |
| `--default-model NAME` | The model that answers requests that name none (`serve`) |
| `--daemon` | Run in the background; the log goes to `~/.eullm/logs/eullm.log`, the PID to `/tmp/eullm.pid` |

Every other option: `eullm run --help`, grouped by subject; `eullm --help`
shows the ones above and the settings below.

## 4. Settings that are not options: the `.env` file

Who may connect and with which key is set in the environment, or in a `.env`
file in the folder you start `eullm` from (copy [`.env.example`](../.env.example)).
Secrets never go on the command line, where every user of the machine can
read them.

| Variable | Example | What it does |
|---|---|---|
| `EULLM_ALLOWED_IPS` | `192.168.1.0/24` | Who may connect besides this machine. **By default only this machine can**: set it to use EuLLM from another PC |
| `EULLM_API_KEYS` | `myapp:8f3b1d9c2e7a4f60b5` | Require a key (`Authorization: Bearer 8f3b1d9c2e7a4f60b5`). Needed behind Docker |
| `EULLM_ALLOWED_ORIGINS` | `https://chat.example.eu` | Web pages on other hosts allowed to call the API |
| `EULLM_MODELS_DIR` | `/data/models` | Where `eullm pull` keeps models (default `~/.eullm/models`). Any `.gguf` in it, at any depth, is listed and runs |
| `EULLM_MODEL_LIBRARIES` | `D:\LM Studio\models;E:\gguf` | Other programs' model folders, read in every subfolder and never changed: their models are listed and run by name |
| `HF_TOKEN` | `hf_...` | For gated or private Hugging Face models |

The startup log says which settings are in effect and where each came from.
The other settings (the web fetch's limits, download connections, the log
level...) are in `eullm --help` and the
[engine reference](engine.md#settings-in-the-environment-or-env).

## 5. Startup scripts

### Linux: a script

```bash
#!/usr/bin/env bash
# start-eullm.sh: a 32k context for two users at a time, reachable from the LAN
cd "$(dirname "$0")"                          # the folder holding .env
export EULLM_ALLOWED_IPS=192.168.1.0/24
exec eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M \
    --ctx-size 32768 --batch-size 2 \
    --no-ui --cli
```

### Linux: a service that starts with the machine (systemd)

`/etc/systemd/system/eullm.service`:

```ini
[Unit]
Description=EuLLM
After=network-online.target

[Service]
User=eullm
WorkingDirectory=/opt/eullm
EnvironmentFile=/opt/eullm/.env
ExecStart=/usr/local/bin/eullm serve --ctx-size 16384 --default-model qwen3-8b --keep-alive 30m
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now eullm
journalctl -u eullm -f        # the log
```

### Windows: a double-click file

`start-eullm.bat`, next to `eullm.exe`:

```bat
@echo off
cd /d "%~dp0"
eullm.exe run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M --ctx-size 16384
pause
```

### Docker

```bash
docker compose up -d engine                     # CPU
docker compose --profile gpu up -d engine-gpu   # NVIDIA GPU
```

Details in [Getting started](getting-started.md).

## 6. Recipes

**Long documents.** `--ctx-size 32768`, and `--cache-type-k q8_0 --cache-type-v q8_0`
if the GPU runs out of memory.

**Several people at once.** `--batch-size 4 --ctx-size 32768`: four requests in
parallel, 8192 tokens each.

**A 35B MoE on a 12-16 GB card.**
```bash
eullm run hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M --cpu-moe --ctx-size 16384
```
On one NVIDIA GPU, try `--moe-cache auto` instead of `--cpu-moe` and compare
the speed.

**From another PC on the LAN, with a key.** In `.env`:
`EULLM_ALLOWED_IPS=192.168.1.0/24` and `EULLM_API_KEYS=lan:a-long-secret`; in the
client, URL `http://<server>:11434` and that key.

**A client changes the context per request.** Clients that send `num_ctx`
(Cherry Studio, Open WebUI) get at most the `--ctx-size` the server was
started with: start it with the largest context you want to allow.

## 7. When something goes wrong

| What you see | What to do |
|---|---|
| Answers stop halfway, `done_reason: "length"` | The context is full: raise `--ctx-size`, or lower `--batch-size` |
| Out of memory at load | Lower `--ctx-size`, use `--cache-type-k q8_0 --cache-type-v q8_0`, or `--cpu-moe` for an MoE |
| Very slow | Check the startup banner: how many layers went to the GPU. A model much larger than the VRAM runs mostly on the CPU |
| Another PC cannot connect | Only this machine is allowed by default: set `EULLM_ALLOWED_IPS` |
| `401` from a client | `EULLM_API_KEYS` is set: give the client the key |
| `address already in use` | Something already uses port 11434 (Ollama?): `--port 11500`, or stop the program using it |

## Going further

- [Engine guide](engine-guide.md): why each option behaves as it does, with the
  measurements: MoE offload, the expert cache, MTP, KV-cache reuse, security,
  multimodal, decisions with `/v1/systemone`.
- [Engine reference](engine.md): every command, flag and endpoint.
