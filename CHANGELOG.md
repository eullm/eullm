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

## Unreleased

### Performance
- **Requests that arrive together start answering together.** With more than one slot, the prompts waiting to be read were read one per step, after the answers' tokens: sixteen requests arriving at once started over sixteen steps, each one a pass of the model for the answers and another for a prompt. They are now read in the step itself, beside the answers' tokens, oldest first, as many as fit in a micro-batch (`--n-ubatch`) less those tokens; with nothing answering, up to a batch (`--n-batch`), but no further than the micro-batch in which the first prompt ends, so that it does not wait for the ones behind it. With `--kv-unified` the answers and the prompt are one pass of the model; with a KV cache per slot, the default, a prompt's slot fills the gap it leaves among the answering slots. Answers do not change: a test reads prompts of 1 to 229 tokens together, beside an answer already going, and each answer is the one its prompt gets read whole. A step that fails with prompts in it is decoded again without them, and those prompts are read on their own, as before, so that a prompt's failure stays its own. On LUMI-G, with Qwen3-14B on one MI250X GCD and sixteen requests at once, EuLLM writes 480 tokens/s instead of 447, with every step carrying all sixteen from the first, and llama-server 462 with the same sampling: EuLLM was 3% behind it and is now 4% ahead. The longest wait for an answer's first token went from 0.65-0.84 s to 0.035 s once the prompts are in the slots, and from 0.94 to 0.30 s with sixteen clients sending requests one after another.

### Added
- **`--mtp-model FILE`: speculative decoding for a model whose GGUF has no MTP head, when the head exists as a file of its own.** Qwen3.8-Flash-Next IQ2_XS on an RTX 5070 Ti, with the head from a separate GGUF: answers at temperature 0.7 go from 54 to 59 tokens/s on prose with `--mtp 1` and from 48 to 65 on code with `--mtp 2`; `--mtp 3` is slower than no drafts on prose. Needs `--mtp`; `--fit` counts the file and its context.

## 0.7.50 — 2026-10-10

### Changed
- **A model that goes to the GPUs whole and is larger than half the memory is read into memory instead of mapped.** Mapped, such a model was read into the page cache and then copied to the GPUs a page at a time by one thread: on LUMI-G, Qwen3-Coder-480B Q4_K_M (270 GiB, 8 GCDs, a 480 GiB job) never finished loading in an hour, six times out of six, and DeepSeek-V3.1 three out of three. Read in, they loaded in 160 to 208 s and 226 s, writing as fast as before. "Memory" is the RAM, or a Slurm job's `--mem` when that is lower; the log says when it happens. Smaller models stay mapped, where reading them in gained nothing (Qwen3-235B, 132 GiB: 110 s mapped, 108 s read in) or lost a little (a 29 GiB model: 46 s mapped, 51 s read in). `--mmap` keeps the file mapped anyway, and `--no-mmap` still reads in any model.
- **llama.cpp moves to the version that carries the MoE expert cache as merged upstream (PR #29887, 7 October), and the cache comes from there.** Same `--moe-cache`, `--moe-prefetch` and pinned experts; the cache itself is now llama.cpp's own. On an RTX 5070 Ti, Qwen3.8-Flash-Next IQ2_XS with `--moe-cache auto` writes answers 10% faster (56.1 tokens/s against 51.1) and reads a 33,200-token prompt at 1,516 tokens/s with the prefetch (987 without) against 1,527 before: the prefetch still copies the experts the cache holds from VRAM and not over the bus (patch `0004`, ported to the cache's new code; 17% of the bytes), which `LLAMA_MOE_PREFETCH_FROM_CACHE=0` turns off to compare (1,387 tokens/s).

### Performance
- **Answers written side by side take one pass of the model per token again, not several.** With more than one slot, llama.cpp keeps a KV cache per slot and runs the model once for every run of consecutive slot numbers in the order a step lists the answers; EuLLM listed them in whatever order answers had ended and slots been taken again, so slots 3, 0, 1, 2 took two passes and 3, 2, 1, 0 four, each reading all of the model. A step now lists them in slot order. On four CPU cores with Qwen3-0.6B, eight clients sending requests one after another with answers of 32 to 256 tokens: 3.5 to 4 passes per step before, 1.0 to 1.4 after, and 60 to 71 tokens/s instead of 41 to 43. On LUMI-G with Qwen3-14B on one MI250X GCD and sixteen clients sending requests one after another, EuLLM now writes 377 tokens/s and llama-server 340.
- **The repeat penalty no longer slows down every token of every answer.** With the penalty on (`repeat_penalty` 1.1, Ollama's default and EuLLM's), llama.cpp looked up each token of the vocabulary in its table of recent tokens before choosing the next one: on Qwen3.5's 248,320 tokens that took 1.2 ms per token, repeated for each answer being written, one after the other, so 19 ms of every step with 16 at once. It now changes only the recent tokens, where they are: 2 µs, to the same logits. On four CPU cores, choosing a token went from 2.7 ms to 1.1 ms, as fast as with the penalty off. On LUMI-G, with Qwen3-8B on one MI250X GCD and 16 requests at once, EuLLM now writes 630 tokens/s with the penalty and 631 without; llama-server writes 476 with it and 666 without, its default. With each server's defaults EuLLM was up to 36% behind llama-server at 16 requests, and is now 8% behind (621 against 676 tokens/s).

### Added
- **`/v1/systemone` requests in `batched` mode that arrive together are answered together.** The decision model answers one request at a time, so with several clients requests waited for each other. `batched` requests that reach it while it is busy are now evaluated in shared decode calls: each request's state side by side, then their questions across requests, up to 16 requests and no more context than one request may use (`--decision-ctx`). Each answer then moves with the other requests' questions by the model's rounding, as it already moved with the other questions of its own request: with the Jev-Style 0.8B on a CPU by up to 0.14 in log-probability, where the same request alone already differs from `separate` by 0.08 to 0.13; with an F32 model and an F32 cache, by 2·10⁻⁶. `shared_prefix`, the default, and `separate` are never grouped and stay bit for bit what they are alone. The response's `eullm` extension says when a request was evaluated with others (`requests_together`). A Jev-Style 2B reads in blocks, one per decode call, and its requests stay one at a time.
- **`--kv-unified` (experimental): one KV cache for every slot,** llama.cpp's flag of the same name, for `run` and `serve`. Any slots then go through the model in one pass, and each answer's attention reads all the slots' cells, masked. Off by default while it is measured on long contexts.
- **Where a busy server's time goes, on request.** Started with `RUST_LOG=eullm=info,eullm::steps=debug`, `eullm serve` logs a line every ten seconds while it answers: how much of the time went to the GPU's forward pass, to choosing each answer's next token, to sending it, to reading prompts and to waiting, with the time per step and per token. Nothing is timed otherwise. The engine guide shows how to read it. The line also says how many passes of the model a step took.
- **An experimental Windows build for AMD Radeon through ROCm: `eullm-windows-x64-rocm.zip`.** For the Radeon RX 7900 XTX and XT (gfx1100), RX 9070 XT and 9070, and Radeon AI PRO R9700 (gfx1201), through ROCm 10.1, for whoever finds the Vulkan build slower on those cards. It needs the AMD Software: Adrenalin Edition for ROCm 10.1 driver (26.10.41.05) or newer, and carries everything else: the HIP runtime, rocBLAS and hipBLASLt with the kernels of those cards, and the Visual C++ runtime. Since ROCm 10 those libraries look for their GPU code in a `.kpack` folder beside the one they are in, so the ZIP keeps ROCm's layout: run `bin\eullm.exe`, and keep `.kpack\` beside `bin\`. It has been built and started, but not yet run on a Radeon: reports are welcome. `install.ps1` and `eullm update` do not install it yet, and say so; other Radeons, and the Instinct MI50 and MI60, which ROCm on Windows does not support, keep the Vulkan build.
- **`--load-threads N`: read the model file ahead of its load with N threads (experimental, off by default).** The threads bring the file, every part of a split model, into the page cache in order while llama.cpp loads it; `--load-threads auto` uses 16 on a network file system (Lustre, NFS, SMB, GPFS, BeeGFS, CephFS, 9p) and none on a local disk. It is off because on Lustre it made loads slower: on LUMI-G, loading cold (the model dropped from the page cache first), Qwen3.8-27B Q8 took 45.9 s without readers, 102.7 s with 4, 49.5 s with 16 and 40.9 s with 32, and Qwen3-235B Q4_K_M 110.3 s without and 128.3 s with 16; llama.cpp alone read the file at 0.7 to 1.3 GB/s there. It stays for file systems where it has not been measured. A model larger than the memory free for the page cache (the RAM, or a Slurm job's `--mem`) is never read ahead, and the log says how many threads read how much, at what rate.
- **`eullm update`: install the latest release in place of this one.** `eullm update --check` says whether a newer release exists; `eullm update` downloads the same build as the one running (CPU, CUDA, Vulkan or ROCm, for the same system), checks it against the release's checksums, makes sure the new binary starts, and puts it in place of the old one, which is put back if any step fails. It keeps the binary's file name and folder, so it updates an install made by the installers as well as an unpacked ZIP. EuLLM asks github.com only when the command is typed: it never checks for updates by itself. A build from source, a Docker image and the Microsoft Store package are not replaced, and the command says so. 0.7.40 and earlier do not have it: update those once more with the installer.

### Fixed
- **A request for JSON no longer brings the server down.** `format: "json"` made llama.cpp abort the whole process at the first token of the answer, taking every other request with it (`GGML_ASSERT(!stacks.empty())` in the log). Each token the model chose was handed to the sampling chain twice, and the JSON grammar, moved on twice past the opening `{`, had nowhere left to go. The same mistake made the repeat penalty (`repeat_penalty`, 1.1 by default) look back over the last 32 tokens instead of the 64 of `repeat_last_n`. It now covers the 64, so answers sampled with it can differ slightly from before.
- **`--moe-cache auto` finds its room again for a model with a per-layer embedding table.** Qwen3.8-Flash-Next IQ2_XS keeps 28.8 GB of embeddings in its second file, a table llama.cpp reads from RAM a row at a time whatever is offloaded. Since 0.7.40 sizes a model from every file, `--fit` counted it as VRAM, found none left for the cache, and offloaded a reduced layer split instead: 181 tokens/s reading and 6.4 writing, against 987 and 56. The table is no VRAM cost now.
- **A model whose chat template opens the reasoning block reasons again, and clients still get the whole block.** Some chat templates end the prompt with the reasoning block already open, on `<think>` (Spark-X2.5's, Qwen3.6's, DeepSeek-R1's), for the model to reason from there. EuLLM cut that tag from the prompt for the model to write it, so that the reasoning would reach clients between its tags. Qwen3.6 writes it; Spark-X2.5 does not (not once in 12 answers), and without it mostly skips its reasoning or loses its way: at temperature 0, "What is 2 + 3?" got `2 + 3 = 5` with no reasoning; over six seeds, "What is the capital of Italy?" was reasoned about before 2 answers, and "ciao come ti chiami?" got 4 answers in Chinese. With the tag kept, all six English answers came after their reasoning, and none of the answers was in Chinese. The prompt now ends as the template wrote it, and the answer is sent starting with the tag the template opened, so the reasoning still arrives between `<think>` and `</think>` (in `thinking`, with `think: true`) on `/api/chat`, on `/v1/chat/completions` and in `eullm run`.
- **The catalog lists Spark-X2.5 1.7B and 4B for English and Chinese, the languages their model cards name.** It listed Italian, German, French, Spanish, Portuguese and Dutch too, and tagged both as multilingual. Asked in Italian, the 1.7B reasons in English even with the fix above, and one of its answers came in Polish.

## 0.7.40 — 2026-10-07

### Performance
- **Long prompts read 10% faster with an expert cache: the experts it already holds no longer cross the bus.** While a prompt is read, every expert of a layer used to be copied from RAM, those the expert cache keeps in VRAM included; now those are copied within the GPU, and only the others over PCIe, which at the default micro-batch is what reading a prompt waits for. On an RTX 5070 Ti, Qwen3.8-Flash-Next IQ2_XS with a 5.75 GiB cache took 17% of the bytes from VRAM and read a 33,200-token prompt at 1,509 tokens/s instead of 1,368 (two runs in each order), to the same answer. In the same runs answers were written at 53.3 tokens/s instead of 54.8; nothing of this change runs while an answer is written, and that difference is being checked. It needs nothing set; `LLAMA_MOE_PREFETCH_FROM_CACHE=0` copies everything over the bus, to compare.

### Changed
- **Copying an MoE's experts ahead while a prompt is read is now on by default, as `--moe-prefetch`, and `LLAMA_MOE_PREFETCH` is gone.** 0.7.30 brought it in behind `LLAMA_MOE_PREFETCH=1`, off by default: while a prompt is read, the next expert tensors are copied to the GPU on a second stream as the current ones compute, into four slots of VRAM. It now runs wherever it applies (one NVIDIA GPU, experts kept in RAM, the model read into memory, which `--moe-cache` does by itself when the RAM can spare the experts), with no setting. With `--moe-cache`, the VRAM of the slots (four times the largest expert tensor) now comes out of the cache, which `--fit` sizes that much smaller so that the prefetch has room to turn on. On an RTX 5070 Ti, Qwen3.8-Flash-Next IQ2_XS with `--moe-cache auto` read a 33,200-token prompt 42% faster by default (1,373 tokens/s against 968, the mean of two runs in each order), to the same answer, and wrote answers as fast with its cache 1 GiB smaller (54.3 tokens/s against 53.3). `--moe-prefetch N` takes 2 to 8 slots and `--moe-prefetch 0` turns it off, which gives the cache its room back; `LLAMA_MOE_PREFETCH`, `LLAMA_MOE_PREFETCH_SLOTS` and `LLAMA_MOE_PREFETCH_MIN_TOKENS` are no longer read. The startup log says what was kept for the slots, and the banner how many there are. Where the cache would fall below its minimum, below a size `--moe-cache` asked for or lose the larger micro-batch, nothing is kept, and the slots are made at the first long prompt only if the VRAM left has room for them, as before.

### Added
- **A Windows build for AMD and Intel GPUs: `eullm-windows-x64-vulkan.zip`.** On Windows only NVIDIA RTX 3000, 4000 and 5000 cards had a GPU build, so a Radeon, an Intel Arc or an older GeForce ran models on the CPU. The new ZIP reaches the GPU through Vulkan, which every current AMD, Intel and NVIDIA driver installs: AMD Radeon cards and Ryzen integrated graphics, Intel Arc cards and Core Ultra integrated graphics, and NVIDIA cards the CUDA build does not cover (GTX 1000, RTX 2000) or whose driver is older than 580. Like the CPU ZIP it carries the Visual C++ runtime, and nothing GPU-side: the Vulkan loader is the driver's. `install.ps1` now installs it for those GPUs when the driver has installed the Vulkan loader, and the CPU build otherwise, as before; NVIDIA cards outside the CUDA build used to get the CPU build. `$env:EULLM_VARIANT = 'vulkan'` picks it by hand. It has not yet been run on an AMD or Intel GPU under Windows: reports are welcome.

### Fixed
- **`--fit` sizes a model split into several GGUF files from all of them.** A model published in parts (`<name>-00001-of-00009.gguf`, as gguf-split writes it, the usual shape for the largest MoE models) was sized from its first part alone: DeepSeek-V3.1 Q4_K_M, 378 GiB in nine parts, was reported as "45.14 GiB, fits fully", so `--fit-strict` let a load start that could never fit — on LUMI it read for an hour before the request timed out — instead of refusing it at once, and `--fit` planned a GPU/CPU split for a ninth of the model. The size and the per-layer expert layout `--cpu-moe`, `--n-cpu-moe` and `--moe-cache` plan from now come from every part, in `run`, in `serve` and when a request loads a model. A model in one file is sized as before.

## 0.7.30 — 2026-10-06

### Changed
- **`think: true` on `/api/chat` and `/api/generate` returns the reasoning apart, in `thinking`, as Ollama does.** A reasoning model's thinking came back inside the answer, between `<think>` and `</think>` (`<|channel>thought` and `<channel|>` on Gemma 4), and a client written for Ollama, which reads it from `message.thinking` (`thinking` on `/api/generate`), found nothing there. Asked to think, both endpoints now put the reasoning there, without its tags, and only the answer in `content` (or `response`); streamed, the reasoning comes first, in lines of its own. Qwen3-0.6B, asked to say hello in Italian in one word, answers with its reasoning in `thinking` and `Ciao!` in `content`, streamed or not. A client that sends `think: true` and read the reasoning from the tags in `content` reads it from `thinking` now. Without `think`, and on `/v1/chat/completions`, the reasoning stays in the answer as before. The web chat, which sends `think` and goes through `/api/chat` when a conversation carries an attachment, reads the new field.
- **`confidence` in `/v1/systemone` answers now follows jev-style's definition, so its values move.** It was `1 − H(p) / ln K`, one minus the entropy of the probabilities scaled to the number of answers. It is now `(K · p_max − 1) / (K − 1)`, the definition jev-style's server, MCP tools and guard use: how far the top answer's probability stands above an even split, from 0 (all answers equally likely) to 1 (all on one answer), `2 · p_max − 1` with two options. The same answer gets a different number — a choice of three at 0.99 / 0.005 / 0.005 moves from 0.94 to 0.985 — so thresholds tuned on the old value need retuning. The old value is still in every answer, as `eullm.confidence_entropy`, and `eullm.confidence_method` now says `normalized_max_probability`. The audit trail's decision records carry the same `confidence_method`; records written by earlier versions have none and hold the entropy-based value.
- **`--embedding-model` holds its context's memory from startup.** The companion builds the context its inputs are embedded in at launch, for its longest input (2,048 tokens), and keeps it for every request: about 1.4 GB of VRAM for Qwen3-Embedding-0.6B, measured on an RTX 5070 Ti, which a generation model sized by `--fit` no longer gets. In exchange a long input never fails for lack of room beside the generation model, and no request pays for building a context any more (see Performance). An embedding model loaded by a request builds its context on its first input and keeps it until it is unloaded.

### Added
- **`eullm finetune`: train a model's weights on your own text, on the machine the engine runs on.** llama.cpp's trainer, which the engine did not expose, behind one command: `eullm finetune ./model-f32.gguf --data corpus.jsonl` trains on a plain-text file, or JSONL with a `text` field per line (the format of Forge's corpora), measures loss, perplexity and next-token accuracy on held-out tokens before training and after each epoch, and writes a new GGUF that `eullm serve` loads like any other. The model has to be F32 (`convert_hf_to_gguf.py --outtype f32`); a quantized one is refused, naming the tensors that are not F32. Before loading anything it estimates the memory the run needs — weights, gradients, optimizer state, activations — and refuses one that will not fit the free VRAM (or RAM, on a CPU) unless `--force`; `--dry-run` stops there. `--train-tensors 'blk.*.attn_*'` trains only the tensors that match, `--optimizer sgd` keeps no optimizer state where AdamW keeps 8 bytes a trained parameter, `--report` writes the run as JSON, and the last line printed is `FINETUNE_RESULT {...}` for scripts. On Qwen3-0.6B-Base on 4 CPU cores, nine 256-token steps at the default learning rate (1e-6) took held-out loss on GSM8K's training text from 1.21 to 0.94, at 13 tokens/s, in 8.7 GiB of RAM against an estimate of 8.4; at 1e-5 the same steps raised it to 1.32, because every window is one step, a far smaller batch than a pretrained model is used to. llama.cpp's limits come with it: no flash attention while training, the whole window in one micro-batch (so `--ctx` is a multiple of 256), and the token embeddings are never trained. The saved model declares the context length it had, not the training window.
- **Each answer says how many MTP drafts its model kept.** With `--mtp`, the drafts the head proposed for an answer and the ones the model kept were only in the server's log, so a benchmark had to read the log. They are now in the answer, under llama-server's names: `draft_n` and `draft_n_accepted` beside Ollama's durations on `/api/chat` and `/api/generate`, and in llama-server's `timings` object on `/v1/chat/completions` (with `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`), which OpenAI clients ignore. The answer's audit line carries them too. Without drafts nothing is added. `bench/speed_check.py` prints the share kept after the speed and takes a `--temperature`; `bench/mtp_sweep.sh` reads the share from there instead of the log, and `TEMPERATURE=0.8` measures it at the default sampling temperature.
- **`--mtp N`: speculative decoding with the model's own multi-token prediction (MTP) head.** Qwen3.5 and Qwen3.6 are trained with an extra layer that guesses the tokens after the next one. With `--mtp 2`, after each token the head drafts up to 2 more, one decode reads them all, and the model's own sampler keeps every draft it agrees with and replaces the first it does not: one decode can write several tokens, each still the model's choice. It needs a GGUF that keeps the MTP layers (unsloth's `*-MTP-GGUF` repositories do; most conversions drop them) and one request at a time (`--batch-size 1`, the default); otherwise the startup log says why it is off and the model runs as before. On an RTX 5070 Ti, unsloth's Qwen3.5-9B-MTP (Q4_K_M) wrote a story at 132.6 tokens/s instead of 109.6 and a piece of code at 177.3 instead of 109.9 with `--mtp 2` (+21% and +61%, 58% of the drafts kept); `--mtp 1` was best on the story (+26%), `--mtp 3` gained less on both. Reading a prompt costs about a tenth more, since the head reads it too. On a small model on a CPU the head costs nearly what it saves (Qwen3.5 0.8B on 4 cores: 14-17 tokens/s instead of 20). `--mtp-p-min P` stops drafting once the head is less sure than P of its next draft; on that GPU it lowered the speed. `bench/mtp_sweep.sh` measures every setting on a model. Answers are the model's own, but a token read in a decode of several is computed in a different order than one read alone, so where two tokens are nearly tied the wording can differ from the answer without drafts, as with prompt-cache reuse. Not yet for Qwen3.8-Flash-Next, whose GGUFs carry no MTP layers.
- **`cache_prompt: false`, for answers that reproduce on a GPU.** At temperature 0 the same request could get a different answer depending on the request before it: by default a request starts from the part of its prompt the server already holds, and on a GPU how many prompt tokens are decoded together changes the numbers slightly, enough to change a word somewhere in a long answer, and every word after it. In AutoBench on an RTX 5070 Ti, half of qwen3-8b's GSM8K answers came out different the second time they were asked. `"cache_prompt": false` (llama.cpp's name for it, at the top level or in `options`, on `/api/generate`, `/api/chat` and `/v1/chat/completions`) decodes the whole prompt every time: slower on a long conversation, the same answer every time. The default is unchanged. AutoBench uses it for every answer it compares, and answers it kept from earlier runs are asked again.
- **`GET /api/ps`, Ollama's list of the models in memory.** Ollama clients that call it got a 404. It lists every generation model, the most recently used first, and the embedding and decision models, with Ollama's fields: `size`, `size_vram` (what free VRAM lost when the model loaded, or an estimate when that could not be measured), `context_length`, `digest`, `details`, and `expires_at`, far in the future for a model kept for good as Ollama sends it. An `eullm` object adds which slot holds each model, how many requests it is answering and when it was last used. `/api/tags` marks every loaded model, the most recently used first, and `/v1/models` lists them all.
- **Unload one model: `{"model": "..."}` on `/api/unload`, and `eullm unload --model`.** Without them every generation model is unloaded, as before; with them only the one named, and the others stay loaded. The response's `unloaded` is still the name of a model, or null, and a new `unloaded_all` lists every model unloaded.
- **Several generation models at once: `--max-loaded-models`.** `eullm serve --max-loaded-models 2` keeps up to two chat models loaded, so requests that alternate between them no longer reload a model each time. A model a request names that is not loaded is loaded beside the others; past the limit, the least recently used idle model is unloaded first. A model loads beside others only if it fits whole on the GPU in what they leave free — every layer, and its projector — and otherwise models are unloaded until it does, or until it is alone, when it is sized as before. With a limit above 1, a model answering requests is never unloaded to make room: the load waits up to 120 seconds for one to finish, then answers 503 with `Retry-After: 5`. Embedding and decision models are not counted; one loaded by a request that does not fit beside the chat models unloads the least recently used of them, one at a time, only as many as it needs. The default is 1, which behaves as before: a request for another model replaces the loaded one, even mid-answer. Unlike Ollama's `OLLAMA_MAX_LOADED_MODELS` it is a flag, not an environment variable, and counts generation models only; the server says so at startup when the variable is set. `/api/version` reports `max_loaded_models`, `loaded_models` and `generation_evictions`.
- **`"model": "auto"`: the decision model chooses which model answers.** Start the server with two to eight `--auto-model NAME=DESCRIPTION`, smallest first, and a `--decision-model`, and a request naming `auto` on `/api/chat`, `/api/generate` or `/v1/chat/completions` is answered by the model the decision model judges enough for it — the small one for a quick fact, the large one for multi-step reasoning or code — so the large model answers only what needs it. A candidate that cannot take the request (an image for a model without a projector, a request longer than its context) is never offered. The decision may take up to `--auto-timeout-ms` (1000); when it times out, fails, or there is no decision model, the fallback answers — `--default-model` if it is a candidate, else the last — and so does it when the chosen model will not load. Every answer says which model answered and why: the `model` field, the `X-EuLLM-Model`, `X-EuLLM-Route` and `X-EuLLM-Route-Id` headers, and an `eullm.route` object on the response or the last line of a stream, with each candidate's probability. Each routing is one `route` line in the audit trail, which the answer's own audit line names. The candidates are loaded at startup as far as they fit without unloading anything, and `auto` is listed in `/api/tags` and `/v1/models`. A `raw` prompt cannot be routed and gets a 400. On a CPU a decision model rarely answers within the default timeout, and `auto` then answers with the fallback. Without `--auto-model`, `auto` is a model name like any other.
- **`POST /api/route`: which model `auto` would choose, without generating.** The same body as the three endpoints; the answer gives the model, the reason, every candidate with its probability and whether it is loaded, and the text and question the decision model read, for tuning descriptions and measuring the router. Nothing is loaded for it.
- **`--default-model`: the model a request that names none goes to.** A request without a `model` field, or with an empty one, was answered by whichever model was loaded and refused with none. With `--default-model NAME` it goes to that model, loaded for it when needed; `--default-model auto` routes it. A name that is no model stops the server at startup.
- **A decision model's own calibration temperature, from its GGUF.** A decision model Forge trains is calibrated on held-out data, and its GGUF carries the fitted temperature as `eullm.decision.temperature`. A code-readout model whose GGUF has it now scales its probabilities with it by default, as a Jev-Style model does with its release's temperature; until now it applied none unless every request asked for one. A request's `eullm.temperature` still overrides it, and the response's `eullm.temperature` says which was applied. A value that is not a number greater than 0 and at most 100 is ignored with a warning that names it, and the model loads anyway, unscaled. The load log shows the temperature in effect and where it came from. A Jev-Style model keeps its release's temperature, and the log says so when its GGUF carries a different one.
- **A decision policy keeps options you rule out away from the model.** `EULLM_DECISION_POLICY` names a JSON file read at startup, such as `{"version": 1, "deny_options": ["delete_*", "transfer_funds"]}`. Every `choice` option of a `/v1/systemone` request whose name matches one of its patterns (`*` for any run of characters, case ignored) is removed from the question before the model reads it, so the model chooses among the options that remain instead of being overruled after it picked one. The response lists what was removed in `eullm.policy_removed`, and so does the audit trail. A question left with fewer than two options is refused with a 422 `policy_denied` naming it. A policy file that cannot be read or applied stops the server at startup rather than letting every option through.
- **Decision traces, to train a decision model on your own decisions.** Set `EULLM_DECISION_TRACES` to a directory and every decision `/v1/systemone` makes is also written to `decisions.jsonl` there, one line each: the state, the questions as the model read them and the answers, under the audit record's id. Traces are off by default, since the audit trail keeps the state only as a hash, and they never leave the machine. Before a line is written, e-mail addresses, phone numbers (international and Italian), IBANs, codici fiscali, payment card numbers and IPv4 addresses are replaced by placeholders such as `[EMAIL]`. Names, street addresses and other identifiers are not recognised; the documentation lists what is caught and what is not. A trace that cannot be written never fails the decision, while a directory that cannot be written stops the server at startup.
- **Feedback on decisions: `POST /v1/systemone/feedback`.** Every `/v1/systemone` response now names its decision in `eullm.audit_id`, the id it has in the audit trail and in the traces. Send that id back with the answers that would have been right (`{"id": …, "answers": {"team": "billing", "is_urgent": true, "severity": 2}}`), optionally what came of the decision (`outcome`) and who says so (`source`: `user`, `rule` or `teacher`), and it is appended to `feedback.jsonl` next to the traces, the outcome redacted like them: the corrections a decision model is trained on. The endpoint checks types and sizes and answers 409 `traces_disabled` when traces are off; it sits behind the same API key, IP allowlist and origin checks as `/v1/systemone`. The new key is inside `eullm`, so clients built on the System One SDKs, whose response models refuse unknown top-level keys, are unaffected.
- **`GET /v1/models` lists the decision model.** The list held only generation models, so jev-style's `model_info` tool and the System One SDKs' `models.list()` showed everything except the model `/v1/systemone` answers with. With a decision model loaded, its entry now carries `context_tokens` — the most tokens one request may hold, `--decision-ctx` or less when the model reads fewer — and, for a Jev-Style model, `head_max_tokens`, what one question with its options may take; and a top-level `models` list names it, with a description and, for the Jev-Style releases, their release date, as the System One SDKs read it. For OpenAI clients the list keeps its shape; the decision model is now in it even when it was loaded from a file path, among the models on this machine.
- **`timing.total_ms` in `/v1/systemone` responses.** jev-style's MCP tools and its guard show the server's time for every decision from `timing.total_ms`, which EuLLM did not send, so they showed `null`. The response now carries it: the request's wall time, `eullm.request_ms` to 0.1 ms.
- **`max_completion_tokens` on `/v1/chat/completions`.** OpenAI's Chat Completions API now names the output limit `max_completion_tokens` and keeps `max_tokens` only as a deprecated name, and EuLLM read only `max_tokens`: a client written against the current API had its limit ignored, and the answer ran on until the model stopped or the context was full. Both names now work, and a request that sends both is limited by `max_completion_tokens`. `/api/generate` and `/api/chat` are unchanged.
- **`--no-mmap`: read the model into memory instead of mapping its file.** The experts of an MoE model kept in RAM (by `--moe-cache`, `--cpu-moe`, `--n-cpu-moe` or the `--fit` split) then go to memory the GPU driver has pinned, which the card copies from directly, where from a mapped file every copy is staged by the driver: the expert cache's copies ran at 9 GB/s that way on an RTX 5070 Ti over PCIe 4.0 x16. On that card, Qwen3.8-Flash-Next IQ2_XS with `--moe-cache auto` went from 44.4 to 58.1 tokens/s writing and from 211 to 451.5 reading a prompt, with 33 GiB of experts pinned. Loading reads the whole file up front, and the pinned experts cannot be swapped out, so the RAM has to hold them. The banner says when it is on.
- **One more model architecture loads, `glm5-next`**, from moving the vendored llama.cpp from b11100 (22 September) to b11370 (3 October). The engine now knows 153 architectures.
- **`--moe-cache`: a VRAM cache for the experts of an MoE model that does not fit.** With `--moe-cache auto`, a model whose experts do not all fit in VRAM keeps every expert in RAM, and the VRAM the usual split gave whole layers of them caches the ones the model uses most instead; `--moe-cache 6000` asks for 6,000 MiB. The startup log and the banner say how large it came out. On an RTX 5070 Ti, Qwen3.8-Flash-Next IQ2_XS wrote 49.4 tokens/s with an 8,000 MiB cache, against 22.4 with the usual split; reading a prompt was 17% slower (206 tokens/s against 249), since the cache serves the decode steps only. With `--mtp` on that model it was slower still, so keep the two apart. It needs one CUDA GPU; elsewhere the load says why and runs without it. With the cache on, the experts in RAM are pinned at load, when that leaves a quarter of the RAM and at least 8 GiB free, so that copying one to the card is a direct transfer instead of one the driver stages piece by piece; one line at load says how much was pinned, or why nothing was, and `LLAMA_MOE_CACHE_PIN=0` turns it off. Some drivers refuse to pin a mapped model file (`operation not supported`); `--no-mmap` is then the way to pinned experts. `LLAMA_MOE_CACHE_STATS=64` prints, every 64 decode steps, how long a step takes and where the time goes: computing up to the routers, the cache's copies, the rest. Unless told otherwise, a load with a cache reads prompts 2,048 tokens at a time instead of 512 (`--n-ubatch` sets it) and reads the model into memory when the RAM can spare its experts, as `--no-mmap` does (`--mmap` keeps the file mapped): with both, that model writes 54.9 tokens/s and reads a 33,200-token prompt at 964, against 21.5 and 256 with the usual split. Experimental: the cache is llama.cpp PR #29887, which this build carries on top of b11370 ahead of a llama.cpp release, and the pinning and the statistics are EuLLM's own patches on top of it.

### Fixed
- **The `--fit` fix for models split into several GGUF files, first listed here, is not in 0.7.30.** It was merged after the 0.7.30 tag, so the 0.7.30 binaries still size such a model from its first part; the fix is in 0.7.40.
- **A request the server has no room for is answered 503, with `Retry-After`, before anything is streamed.** Each slot (`--batch-size`) keeps up to eight requests waiting behind the one it is answering. One more was answered 500 without streaming, which tells a client the fault is the server's, and with streaming 200, with a stream whose only line was the error, which a client cannot tell from an answer that broke off halfway. It is now 503 Service Unavailable with `Retry-After: 5` and the error as JSON, streamed or not, on `/api/generate`, `/api/chat` and `/v1/chat/completions`, as Ollama answers its own full queue: a client that retries knows to, and when. A request whose model was unloaded just before it started, which says to send it again, gets a 503 the same way, with `Retry-After: 1`, instead of a 500.
- **The default number of CPU threads stays within the CPUs the engine may use.** It was the machine's physical cores, read from `/proc/cpuinfo`, whatever the process was allowed: bound to 7 CPUs of a 64-core LUMI-G node by Slurm, the engine started 64 threads on them, and `taskset` or `docker --cpuset-cpus` did the same. It is now the physical cores or the CPUs the process may run on, whichever is fewer, so an unrestricted machine gets what it got before. `--threads` still overrides it.
- **With `--batch-size` above 1, a long prompt no longer stops every other answer.** A new request's prompt was read whole as soon as the request was taken, and while it was read nothing else on the server moved: a 30,000-token RAG prompt froze every conversation being answered for as long as it took. It is now read a chunk at a time between their tokens: each answering request gets its next token, then the prompt advances by one micro-batch (`--n-ubatch`). Alone on the server a prompt is still read a whole batch at a time, and with one slot, the default, nothing changes. The answers are the ones a prompt read whole gets: llama.cpp computes it one micro-batch at a time from the same positions either way. A client that disconnects while its prompt is waiting or being read now frees its slot then, instead of after the prompt is read.
- **Ollama's curl examples work as they stand.** A request whose body was not labelled `Content-Type: application/json` was refused with 415, on every endpoint. Ollama reads a body as JSON whatever its label, and its documentation's examples rely on it: `curl http://localhost:11434/api/generate -d '{…}'` sends `application/x-www-form-urlencoded`, and so did this project's own getting-started examples. A body with no `Content-Type`, `text/plain` (what a script's `fetch` sends for a string) or `application/x-www-form-urlencoded` is now read as JSON; one labelled as something else, such as `multipart/form-data`, is still refused with 415. A web page on an origin that is not allowed is refused with 403 before its body is read, whatever its label, as before.
- **The Docker images build and run.** The engine's and the hub's could not be built at all: the engine needed files from outside the directory it was built from (the model catalog it compiles in, the workspace's lockfile), and the hub's pinned a Rust too old to read its manifest. Every image now builds from the repository root, with llama.cpp's submodule checked out (`git submodule update --init --recursive`, then `docker compose build`, or `docker build -f engine/Dockerfile .`). The engine's container starts the server, where it printed its usage and exited, writes its audit trail to its volume, and with `docker compose up engine` answers this machine on 11434 with the chat UI on 11435; it is published on 127.0.0.1 only, and the compose file says what to set (API keys) before publishing it further. There is a CUDA image, `engine/Dockerfile.cuda` (`docker compose --profile gpu up engine-gpu`): CUDA 13 for RTX 3000, 4000 and 5000 cards, with build arguments for CUDA 12 drivers and for A100/H100. The Forge image now carries llama.cpp's GGUF converter and `llama-quantize`, built from the same llama.cpp as the engine, so the pipeline's export stage, which stopped for want of them, runs inside it; its PyTorch is built for CUDA 13 (`--build-arg TORCH_INDEX` for an older driver). The hub stops at once on `docker stop`, where it was killed after ten seconds. Every image runs as the unprivileged uid 10001, so a host directory mounted in place of a volume must be writable by it.
- **`--fit` leaves room for `--mtp`'s draft context.** The MTP head runs in a context of its own, built once the model has loaded, and sizing did not count it: a model sized to fill the card left the head no room. It is now reserved before the model is sized, wherever the head will draft (one slot, no projector): the KV cache of the model's MTP layers at the context size, read from the GGUF's `nextn_predict_layers`, plus a compute buffer for one micro-batch through them. Qwen3.5-0.8B-MTP's measured 8 MiB of KV and 27 MiB of compute at 4,096 tokens come out as 40 MiB.
- **`prompt_eval_duration` and `eval_duration` are measured.** `prompt_eval_duration` was always 0 and `eval_duration` was the whole request, so a client dividing `prompt_eval_count` by the first read prompts at infinite speed, and one dividing `eval_count` by the second counted the prompt as writing time. Both are now measured on every path (the batching scheduler, the sequential engine, images and audio): reading the prompt, then writing the answer, in nanoseconds as Ollama reports them. On a CPU, Qwen3.5-0.8B read a 24-token prompt in 244 ms and wrote 69 tokens in 2.49 s, of a 2.73 s request.
- **`eullm run --image` answers instead of hanging.** After reading the prompt it printed nothing, on every model and every image, in builds with `multimodal`: the command held the terminal's output for the whole answer, and the thread writing the answer stopped at its first log line, which goes to the same output, before writing a token. It now takes the output one write at a time. The API was not affected: an image sent to `/api/chat` or through the web chat was answered.
- **`load_duration` is measured.** It was always 0. A request that had to load its model now says how long that took, in `load_duration` and as part of `total_duration`, as Ollama reports them, on `/api/generate` and `/api/chat`, streamed or not.
- **`keep_alive: 0` on an empty request no longer loads the model first.** An empty prompt, or empty messages, with `keep_alive: 0` is Ollama's way to unload a model, and it loaded the model when it was not loaded — unloading the one that was — only to unload it again. It now unloads that model, and only it, if it is loaded, does nothing otherwise, and answers `done_reason: "unload"` as Ollama does, where it said `load`. A model still answering other requests goes when they are over.
- **`keep_alive: 0` with a prompt answers, then unloads the model.** A request with a prompt and `keep_alive: 0` unloaded the model before reaching it, so on any model served by the batching scheduler — the default — it failed with "Scheduler queue full — try again later" and answered nothing. The answer now comes first, and the model is unloaded once it has been sent.
- **A long answer no longer loses its model to keep_alive.** keep_alive — the request field and `--keep-alive` — was counted from the start of each request, so an answer that took longer than it was cut off halfway, when the idle timer unloaded the model it was coming from. It now counts from the end of the last request on the model, and a model that is answering is never unloaded for being idle. With requests overlapping, the keep_alive of the last one to arrive applies, as in Ollama.
- **A request that reaches a model as it is unloaded says so.** A request submitted to a model being swapped out was told "Scheduler queue full", and one still waiting in the scheduler's queue when the swap began was never answered: without streaming it hung until the client gave up, with streaming it ended without a final line. Both now get "The model was unloaded before this request started — send it again to load it back".
- **An absurdly large `keep_alive` on an embedding or `/v1/systemone` request no longer drops the connection.** A value such as `1e19` seconds is a valid duration but no point in time, and the server's handler crashed on it without answering. It now means what it says: keep the model loaded.
- **A swap away from a multimodal model waits for its requests to end.** A model served sequentially — every multimodal model, and any run with `--batch-size 0` — stays in VRAM until the requests running on it finish, but the next model was sized at once, against free VRAM that still held it: it could load with fewer layers on the GPU than the card has room for, or fail to load. A swap now waits for those requests, up to 30 seconds, before it sizes the next model.
- **An embedding or decision model loaded beside a multimodal model no longer leaves image requests without memory.** A model served sequentially builds its context for each request, so while it was idle that memory looked free, and an embedder or decision model loaded on demand could take it; the next image request then failed to allocate its context. That memory is now kept free for it: a companion that would need it unloads the generation model instead, as one that does not fit always did, and a generation model is sized around it too.
- **Naming the loaded model another way no longer loads it again.** A request that named the loaded model by its file's path, or by a second name `eullm pull` had linked to the same weights, unloaded the model and loaded the same file back, which on a large model costs as long as a cold start. It is now answered by the model already loaded, under the name it was loaded with.
- **No more `←[32m`-style codes in the Windows console.** In the classic Windows PowerShell or Command Prompt window, every log line, the model picker and the terminal chat showed the escape codes behind their colours as text (Windows Terminal was not affected). EuLLM now turns colour support on in the console when it starts, and writes plain log lines when it cannot, for example when the output goes to a file.
- **The CPU build no longer tells you to rebuild it.** Every `eullm run` on a CPU build printed a boxed warning that a GPU was requested and suggested `cargo build --features cuda`, although nobody had asked for a GPU: offloading to it is simply the default. It now prints one line saying it runs on the CPU and where the GPU builds are. The warning box remains for an explicit `--gpu-layers` on a CPU build, and now points to the ready-made GPU downloads first.
- **Pulling a model you already have under another name no longer downloads it again.** `eullm pull qwen3-8b` after `eullm pull hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M`, or the other way round, fetched the same 5 GB file a second time into a second directory, because each pull only looked for its own name. A pull now first looks for the same file from the same repository under any name in the store, models pulled with earlier versions included. When it is there, nothing is downloaded: the new name is hard-linked to the file, so both names run, removing one leaves the other, and no extra disk space is used; `eullm rm` says when the files stay on disk because another name still uses them. Where a hard link is impossible, for example on an exFAT drive, nothing is downloaded either, and the pull tells you which name to run instead. This applies to `eullm pull`, `eullm run` and `POST /api/pull`. Another quantization, or a file of the same name in a different repository, is a different file and is still downloaded.
- **Embedding responses report the tokens they read.** `/v1/embeddings` answered `"usage": {"prompt_tokens": 0, "total_tokens": 0}` whatever the request held, and `/api/embed` returned no counts at all. `usage` now carries the tokens the model read, after truncation to the embedder's context, and `/api/embed` returns the same number as `prompt_eval_count`, next to `total_duration` and `load_duration`, as Ollama does.
- **`/v1/systemone` errors reach System One and jev-style clients.** Every refusal was `{"error": "<message>"}`, which those clients cannot read: jev-style's MCP server and its tool-call guard failed on the body itself, so a tool call with a one-option `choice` or an input too long for the model showed nothing but "Error executing tool". Errors now come in the System One shape, `{"error": {"code": …, "message": …, "question": …}}`, with `question` naming the question at fault, and the message is the one EuLLM always gave, the question's name moved beside it. A request that fails validation is now a 422, as the System One API answers it, where most were 400s: `invalid_json`, `invalid_request`, `invalid_question`, or `input_budget_exceeded` for input too long for the model, which is still refused whole rather than truncated. No decision model loaded is still a 400 (`model_not_loaded`), an unknown `model` a 404 (`not_found`). Refusals by the API-key, IP-allowlist and origin checks take the same shape on this endpoint; every other endpoint's errors are unchanged.
- **Score levels written as `{"label", "description"}` read as labels.** A `score` question whose levels were objects, such as `{"label": "high", "description": "loses data"}` — the form jev-style's tool-call guard uses for its risk question — showed each level to the model as raw JSON, and the response's `legend` repeated that JSON where a client looks for `high`, so jev-style's `score` tool reported a JSON string as the most likely level. The model now reads `high: loses data` (the label alone without a description), and the legend says `high`, as jev-style's own server does, with either kind of decision model. A level that is not a non-empty string, a labelled object, or another non-empty object or array is refused with a 422 naming the question.
- **`instructions` may be an object or an array.** The System One API and jev-style accept a question's `instructions` as a string, an object or an array — the question in one field, the data it refers to in others — but EuLLM refused anything but a string with a 422 about deserializing the body. An object or array is now accepted and shown to the model as one line of compact JSON in the order it was written, exactly as jev-style's server writes it for its models.
- **A decision abandoned by its client stops, and every decision made is audited.** When a client gave up on a `/v1/systemone` request — jev-style's guard stops waiting after 8 seconds — the decision still ran to its end, holding the model while the next request waited behind it, and its audit record was never written: only the handler of the connection that was gone would have written it. The decision now writes its own record, so every decision the model computes is in the audit trail, with `client_disconnected: true` when its answers were never sent. A request whose client leaves before its answers are ready stops at its next question, or before it starts if it was still waiting for the model; it decided nothing and, like a request refused as invalid, is not recorded, while the server log notes it.
- **Long embedding requests no longer hold up the rest of the server.** `/api/embed` and `/v1/embeddings` ran the model on one of the threads that answer every HTTP request, one per CPU core, so whatever else was queued on that thread waited until the embedding finished, and as many long embedding requests as there are cores stopped the whole API: four 1,162-token inputs at once on a 4-core CPU kept even `/api/version` from answering for 25 seconds. Embeddings now run on threads of their own, as chat and decisions already did, and the rest of the API keeps answering within milliseconds while they run. As before, at most one per CPU core runs at a time — each builds a context of its own, so more would only multiply the memory, on a GPU its VRAM — and the others wait their turn without holding up anything else.
- **A burst of embedding requests no longer fails on a full GPU.** Sixteen `/v1/embeddings` requests of about 2,000 tokens each, sent at once to a 16 GB RTX 5070 Ti, left half of them failing with `Failed to create embedding context`: every request built its own context at the same time, and a 2,048-token context of a decoder-based embedder such as Qwen3-Embedding holds about 1.2 GB. Inputs are now embedded one at a time, in the order they came, in the one context the embedder keeps, so a burst waits its turn instead of failing: on the same card, sixteen such requests at once all succeed.
- **A long input to the `--embedding-model` companion no longer fails beside a generation model that fills the card.** `--fit` kept 256 MiB free for the context an embedding request built, and a 2,048-token input of Qwen3-Embedding needs about 1.4 GB: with a generation model sized to fill the GPU, long inputs failed with `Failed to create embedding context`. The companion's context is now built at startup, before the generation model is sized.
- **The model named by `--embedding-model` is no longer loaded a second time by the first request.** An embedder given by its store name (`--embedding-model qwen3-embedding-0.6b-gguf-q8_0`) was registered under its file's name, so the first request naming it the way `eullm list` shows it found nothing loaded and loaded it again: that request waited for the load, and the companion reserved at launch was replaced by an ordinary one, which with `--fit` the next generation model to load could evict. It now keeps the name it was given, and a request naming the same file any other way is answered by the model already loaded.

### Performance
- **Long prompts read up to 42% faster on an MoE model whose experts sit in RAM, with `LLAMA_MOE_PREFETCH=1` (experimental, off by default).** Reading a prompt copied each layer's experts to the card and then computed them, with the GPU idle during every copy. With the variable set, the next expert tensors are copied on a second stream of the GPU while the current ones compute, into four slots of VRAM (1 GiB on Qwen3.8-Flash-Next IQ2_XS, taken from what `--fit` leaves free; `LLAMA_MOE_PREFETCH_SLOTS` sets 2 to 8). On an RTX 5070 Ti that model read a 33,200-token prompt at 1,743 tokens/s instead of 1,228, to the same answer token for token. It needs an NVIDIA GPU and the model read into memory (`--no-mmap`, the default with `--moe-cache` when the RAM allows); where it does not apply, or the VRAM is short, it stays off and says why in one line on stderr.
- **`--n-ubatch`: long prompts read in fewer passes on an MoE model whose experts do not all fit in VRAM.** A prompt is read on the GPU in passes of 512 tokens, and before each pass llama.cpp copies the experts kept in CPU RAM to the card: a 33,000-token prompt to Qwen3.8-Flash-Next IQ2_XS on an RTX 5070 Ti was 65 copies of tens of gigabytes, and read at 256 tokens/s. `--n-ubatch 4096` reads it in 9 passes, at 822 tokens/s. A pass's activations need room on the GPU, so `--fit` keeps fewer layers' experts there: answers are written somewhat slower, 19.4 tokens/s instead of 21.5 on that model, with the experts of 8 layers on the card instead of 11. The default stays 512, llama.cpp's own, and nothing changes without the flag. `--n-batch` is raised to match a larger `--n-ubatch`, and says so.
- **Embedding requests are many times faster on a GPU.** Every request to `/api/embed` and `/v1/embeddings` built a llama.cpp context for its inputs and dropped it afterwards, and on a GPU that took most of the request: with Qwen3-Embedding-0.6B on an RTX 5070 Ti, 904 of the 1,001 ms a 1,966-token input took, 235 of the 261 ms for 510 tokens, 34 of the 40 ms for 62 tokens. The embedder now builds its context once and keeps it from request to request, and the same inputs cost about 97, 26 and 7 ms in it. On a CPU, where the model's own arithmetic takes most of the time, the gain is small. Vectors are bit for bit the same.
- **Qwen3.5 writes 5-10% faster with llama.cpp b11370.** Measured on the same RTX 5070 Ti with `bench/mtp_sweep.sh` and unsloth's Qwen3.5-9B-MTP (Q4_K_M), before and after the move from b11100: a story at 114.8 tokens/s instead of 109.6 and a piece of code at 120.2 instead of 109.9 without drafts, 146.1 instead of 132.6 and 194.3 instead of 177.3 with `--mtp 2`. The model keeps the same 58% of the drafts, and `--mtp 2` adds 27% on the story and 62% on the code.

## 0.7.20 — 2026-09-28

A decision endpoint, `/v1/systemone`, and the Jev-Style models that answer it best. Without a decision model loaded, nothing else changes: chat, completions and embeddings work exactly as in 0.7.11.

### Added
- **Decisions without generation: `POST /v1/systemone`.** Ask a small model typed questions about a state — a ticket, a document, a JSON object — and get probabilities back instead of text: `noul` (is this true? → P(yes)), `choice` (which of these options? → the option, the distribution, a confidence) and `score` (which level of this scale? → the expected level). The request and response follow the System One API (TypeSafe's Jev), so a client written for it works by changing its base URL; a Jev model name such as `jev-latest` means the decision model the server has loaded. Nothing is generated: each answer is read from the model's next-token probabilities over single-token answer codes, which the server checks for every model when it loads. Up to 64 questions about the same state are answered in one request that decodes the state only once, and each question is then decoded on its own, so its answer does not depend on the other questions asked or their order — bit for bit, checked by `bench/decision_bench.py --order-check`; the response reports the tokens the shared state saved. A `batched` mode decodes all the questions in one batch instead, with fewer decode calls, but on quantized weights its answers move with the other questions by the model's own numerical noise. The decision model keeps its context from one request to the next: a request about the same state as the previous one — the next round of questions about the same document — decodes only its questions, and gets the answers a freshly decoded state would give (`prefix_reused` in the response). Each answer also carries the raw log-probabilities, the probabilities before calibration and a `coverage` that drops when the model did not answer in the format asked for. Calibration is opt-in and not yet validated: `content_free` (the model's answer about an empty state divided out) and temperature scaling, to be measured on labelled data before relying on either. Every decision goes to the audit trail with its probabilities; the state is recorded only as a SHA-256. What a request sends is always tokenized as text, so a state that contains the model's own turn markers (`<|im_end|><|im_start|>assistant`) cannot close its turn and answer for itself.
- **Jev-Style decision models.** The decision model can now be a [Jev-Style](https://github.com/lawrence3699/jev-style) release (Apache-2.0): Qwen3.5 fine-tunes trained for exactly these three question types. Get the 0.8B with `eullm pull hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M` (0.53 GB); a 2B is also available. They read each option as its own yes/no verdict at a slot after it, so a `choice` can list up to 255 options. The probabilities use the calibration temperature the model was released with, by default. The server recognizes these models when they load and builds their input — which is not a chat prompt — token for token, the way the release's own runtime does. In `separate` mode the scores match that runtime exactly for both models, one question at a time and many together (checked on the CPU on 24 cases for the 0.8B and 28 for the 2B). The default `shared_prefix` decodes the state once for all the questions: the 2B's scores stay exact, the 0.8B's move by at most 0.04 in probability, and with 64 questions about a 1,024-token state the 0.8B answers in 0.62 s and the 2B in 0.66 s on an RTX 5070 Ti, against 3.2 s and 4.5 s one at a time. The response's `eullm.readout` says which kind of model answered, and `eullm.scores` carries the raw verdict scores. A `noul` question may now say what true and false mean (`"criteria": {"true": "…", "false": "…"}`), for either kind of model.
- **`--decision-model` and `--decision-ctx`** on `eullm run` and `eullm serve`. The decision model runs in its own slot next to the chat and embedding models. `--decision-model qwen3-1.7b` loads one at startup and keeps the VRAM a request needs free when the chat model is sized, like `--embedding-model`; without it, the first request that names a model loads it. `--decision-ctx` (default 8192) is the most context one request may use — the state plus its longest question, or plus every question in `batched` mode — and sets how much VRAM is kept free for it: 896 MiB of KV cache for Qwen3-0.6B at the default.

## 0.7.11 — 2026-09-26

The engine binaries are the same code as 0.7.10; upgrading changes nothing about how EuLLM runs.

### Added
- **`eullm-windows-x64-store.msix` is actually published this time.** 0.7.10 announced it, but the job that builds it skipped itself, so 0.7.10 went out without it. It is the package for the Microsoft Store and is still not for installing directly.

### Fixed
- **The Windows installer no longer downloads a CUDA build the GPU cannot run.** `install.ps1` picked the CUDA ZIP for any NVIDIA card on driver 580 or newer, but that build only covers RTX 3000, 4000 and 5000 cards. An RTX 2080 Ti, an A100 or an H100 got a 500 MB download that could not use the GPU. It now checks the card as `install.sh` already did, installs the CPU build for the others and says why; `$env:EULLM_VARIANT='cuda'` still forces the CUDA build. The installer is served from `main`, so this applies to every release, 0.7.10 included. Reported and fixed by Gabriele Pau.

## 0.7.10 — 2026-09-26

### Added
- **One-line install on every platform.** `curl -fsSL https://raw.githubusercontent.com/eullm/eullm/main/installer/install.sh | sh` on Linux and macOS, `irm https://raw.githubusercontent.com/eullm/eullm/main/installer/install.ps1 | iex` in Windows PowerShell. Both scripts pick the build that fits the machine: CUDA when an NVIDIA GPU with a recent enough driver is present, the A100/H100 build on those cards, the CPU build otherwise, and Metal on Apple Silicon. They refuse to install a download whose checksum does not match the release, and they need no root or administrator rights. On Windows the install goes to `%LOCALAPPDATA%\Programs\EuLLM` and is added to your PATH; `$env:EULLM_UNINSTALL=1` removes it again. The scripts are independent of the engine version, so they already install the current 0.7.9 (older releases used different file names for some builds).
- The Windows CUDA ZIP now contains a `THIRD-PARTY-NOTICES.txt` that names the bundled NVIDIA DLLs and the licence they are distributed under.
- Releases now also publish `eullm-windows-x64-store.msix`, the package for the Microsoft Store. It holds the CPU and CUDA builds, runnable as `eullm` and `eullm-cuda` from any terminal. It is unsigned until the Store signs it, so it is not for installing directly: EuLLM is not in the Store yet.
- **`eullm-windows-x64.zip`**, a new CPU download for Windows: `eullm.exe` together with the Visual C++ runtime it needs. `install.ps1` installs this one. The bare `eullm-windows-x64.exe` is still published for existing links.

### Fixed
- **EuLLM now starts on a Windows without the Visual C++ Redistributable.** Every Windows build needs the Microsoft C++ runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, `VCOMP140.dll`), and none of the downloads included it. Most PCs already have it from some other program; on one that does not, such as a fresh install or Windows Sandbox, `eullm.exe` refused to start with *"MSVCP140.dll was not found"*. The CPU ZIP and the CUDA ZIP now carry those DLLs next to `eullm.exe`. The bare `eullm-windows-x64.exe` still needs the [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) installed.

## 0.7.9 — 2026-09-24

### Added
- **Gated and private Hugging Face repositories, with `HF_TOKEN`.** Set it
  to an access token and `eullm pull` / `eullm run hf.co/<owner>/<repo>`,
  and the model catalog in the Chat UI, authenticate to Hugging Face: the
  API calls and every download request, parallel ranges, shards and
  projector included. The token goes to `https://huggingface.co` only — not
  to any other address, not to the CDN a download is redirected to — and is
  never logged or written to a manifest or the audit trail. Without it,
  every request is exactly what it was. A refusal now says what it means
  instead of `HTTP 401 Unauthorized`: a gated repository whose terms need
  accepting, a private or missing one, a token that was not accepted, or an
  account not yet granted access, with Hugging Face's own explanation
  quoted.

### Fixed
- **A picture was forgotten one turn after it was sent.** A follow-up
  question about a photo — "is that thread the tongue?" — reached the model
  without the photo, and it answered from its memory of its own earlier
  description without saying so. Attachments now stay in the conversation:
  the web UI re-sends them every turn, and `/api/chat` reads the `images` of
  every message, not only the latest user turn's, as Ollama does. When the
  conversation outgrows the context, the oldest attachments give way first,
  each replaced by a note telling the model one was there and can no longer
  be viewed; the current turn's own attachments never do. Every turn with a
  picture still in it encodes the picture again, so a follow-up takes as
  long to start answering as the turn that sent the picture did.
- **A conversation that had a picture in it kept working after switching to
  a text-only model.** The picture becomes the same note, rather than every
  turn from then on being refused because the new model has no projector.
- **An attachment that is not valid base64 is refused with a 400 naming it**
  (`messages[1].images[0] is not valid base64`). It used to send the request
  on as text, and the model said it could see no image. Line breaks inside
  the base64 are now ignored, as Ollama ignores them.
- **Two images in one message failed to tokenize**: the prompt carried one
  media marker for the whole message instead of one per image.
- **Switching thinking off in the web UI had no effect on a turn with a
  picture**, because that request did not send the setting.
- **The Linux CPU, CUDA and ROCm binaries could not say which commit they
  were built from.** `eullm -V` and the startup banner print the commit
  hash, so a bug report pins the exact build; those six binaries printed
  `unknown` instead, because they are built in a container where git would
  not read a checkout owned by another user. Windows, macOS, Vulkan and the
  arm64 CPU builds were never affected.

## 0.7.8 — 2026-09-24

### Fixed
- **A vision model that sizing said would fit refused to load, all the way
  down to 512 tokens of context.** The error was `could not allocate a
  context of 512 tokens: allocation succeeded but left only 10% of GPU
  memory free`, and it came from a projector that was loaded every time and
  counted by nothing. Sizing picked the text model's GPU layers against the
  free VRAM it measured, leaving the 12% margin the context check requires;
  the projector then loaded into that margin — 888 MiB of weights and a
  248 MiB compute buffer for a 27B on a 16 GB card — and the check found 10%.
  The advice printed with it, to lower `--ctx-size` or quantize the KV cache,
  could not have helped: the context was already at its floor, and
  quantizing a 512-token cache gives back about 60 MiB of the 328 that were
  missing. When the smallest context still does not fit, the message now
  says so in those terms — how many MiB short, that `--ctx-size` cannot help
  — and names only what can: fewer layers on the GPU, with how many are there
  now; the projector, if it is on the GPU; and KV quantization only when
  what it frees would actually close the gap.

  Sizing now counts the projector, and decides where it goes. It stays on
  the GPU when the whole text model still fits beside it. When it would not,
  it moves to system RAM first, before any text layer is taken off the card,
  because the two are not paid for the same way: a projector runs once per
  image and is idle for every token after it, while a text layer in RAM
  slows every token of every request. In the case above that means the
  model loads with exactly the text layers sizing already chose, and images
  are encoded on the CPU instead. The load log says which one happened.

  `--mmproj-offload` keeps the projector on the GPU regardless, at the cost
  of text layers — worth it when nearly every request carries an image —
  and `--no-mmproj-offload` keeps it in RAM regardless. Both names are
  llama.cpp's. Text-only models are unaffected, and so is any build that
  cannot measure its VRAM, where the projector follows the text model as it
  always did.

- **A download could be written with the wrong bytes and accepted.** Large
  files are fetched in parallel ranges, and each range was written at its
  own offset whatever the server sent back — including a `200` carrying the
  whole file, which a server is allowed to send when it decides to ignore
  the range. The first byte of the file then landed where the middle of it
  belonged. Pulls from the catalog would have caught it at the digest check;
  pulls by `hf.co/owner/repo` carry no digest, so the file was renamed into
  place and used. Only a `206` is accepted now, anything else is retried
  like any other failed range, and a server that keeps refusing fails the
  download with an error instead of corrupting it. A server that never
  honours ranges is unaffected: it fails the first probe and gets the
  single-stream path, as before. `EULLM_DOWNLOAD_CONNECTIONS=1` forces that
  path for one that honours the probe and stops partway.
  Fixed by [@Gabriele06-local](https://github.com/Gabriele06-local) in
  [#508](https://github.com/eullm/eullm/pull/508).

- **Importing a model whose file sets its own tensor alignment produced a
  copy that would not load.** GGUF files say where their tensor data begins
  by declaring `general.alignment`; almost every file leaves it at the
  default of 32 and says nothing, and the importer assumed 32 for all of
  them. For a file that declares anything else, the patched copy came out
  with its tensor data at an offset the file's own header does not point to.
  There was no error and no warning — the import reported success and the
  model failed to load afterwards, which is the wrong end to find out.
  The declared value is now read and used, and a file declaring one that
  llama.cpp itself would reject, or one so large the data would sit past the
  end of the file, is refused instead of copied.

- **A corrupt GGUF could end `eullm import` with a crash instead of being
  refused.** Importing a model from Ollama streams the file's metadata to
  look for the array lengths llama.cpp needs patched, and the importer is
  written so that anything it cannot read means "copy the file as it is" —
  the import still succeeds, it just skips the patch. Several fields could
  not reach that fallback: a length field the file made up was used to size
  an allocation before anything checked it, which past a certain size ends
  the command with a Rust stack trace; a metadata type the parser does not
  recognise was measured as zero bytes and skipped over, leaving the scan
  reading from a position that could not be right; and an element count too
  large to be real was walked one element at a time, which on a multi-gigabyte
  file is a wait long enough to look like a hang. Every one of those is now a
  refusal, and a refusal is the fallback: the file is copied verbatim.

  Nothing changes for a well-formed GGUF, which is every file anyone is
  likely to have. Overflowing array sizes in the same scan were fixed by
  [@Gabriele06-local](https://github.com/Gabriele06-local) in
  [#501](https://github.com/eullm/eullm/pull/501); this is the rest of the
  same sweep.

## 0.7.7 — 2026-09-22

### Added
- **Spark-X2.5 is in the catalog, in both sizes.** `spark-x2.5-1.7b` and
  `spark-x2.5-4b`, Apache-2.0, from the publisher's own GGUF repositories.
  They use a hybrid attention layout — one full-attention layer for every
  three sliding-window ones — which is what buys them a 1M-token context
  without the memory a full-attention model of that length would need.

- **Three more model architectures load**, as a consequence of moving the
  vendored llama.cpp from b10818 (5 September) to b11100: `spark2_5`,
  `hrm_text` and `maple`. The engine now knows 152 architectures. The C
  interface barely moved in those two and a half weeks: the multimodal and
  backend headers are byte-identical, and the one signature that changed
  (`llama_sampler_chain_n`, `int` to `int32_t`) is the same type on every
  platform we build for.

- **The model browser now shows a repository's licence before you download
  it.** Browsing a HuggingFace repo in the web UI puts the licence next to the
  architecture, with a link straight to the terms — the licence file when the
  repo names one, the model page otherwise. Repositories that ship under terms
  of their own are labelled as such and named, `qwen-community-1.0` and the
  like, rather than being flattened into a standard licence they are not.

  The licence is reported, not judged. Whether terms that forbid commercial
  use or hosting allow what you intend to do is between you and the licence;
  what changes here is that you see it before the download instead of after.
  Repositories the Hub states nothing for say exactly that, which is not the
  same as having no licence.

### Fixed
- **A `keep_alive` ending in an accented letter, an emoji or any other
  non-ASCII character crashed the request that carried it.** `"5à"`,
  `"30s€"`, `"1h☃"` — or just `"¡"` on its own — took down the handler task
  instead of being treated as the malformed value they are. The unit was
  split off by byte position, which lands inside a multi-byte character.
  Reachable from any request body, and from `--keep-alive` on the command
  line. Plain `"5m"`, `"30s"`, `"2h"` and bare numbers are unaffected and
  behave exactly as before.

## 0.7.6 — 2026-09-21

### Fixed
- **Every model pull on Windows crashed the moment the download finished.**
  The progress bar reached 100%, the process aborted with `thread 'main' has
  overflowed its stack`, and the model never became usable — what was left on
  disk was a `.gguf.part` that retrying could only bring back to the same
  point. It hit every model in the catalogue, because the integrity check that
  runs at the end of a download needed more stack than Windows gives a
  program's main thread. Linux and macOS have eight times as much and were
  never affected, which is why this survived to a user report. Every Windows
  build since 0.6.80 carried it. Pulling by `hf.co/owner/repo` was never
  affected — that path records no digest, so it never reached the check — and
  neither was anything done with a model already on disk. If an earlier
  attempt left a `.gguf.part` behind, delete it and pull again.    Reported by Fabrice Frébel ([eFFiciency research](https://www.efficiencyresearch.be)).
  

- **A fetched web page could kill the chat that fetched it.** With `--web`, a
  page containing a character such as `İ` — an ordinary Turkish letter — took
  down the request while the page was being reduced to text. Any page on the
  open web can carry one. Reported and fixed by
  [@Gabriele06-local](https://github.com/Gabriele06-local) (#463).

- **`hf.co/…` references containing those same characters** were either
  refused as though they were not HuggingFace references at all, or crashed
  the command outright, depending on how many there were. They now resolve
  like any other reference.

- **An empty `keep_alive` crashed the request instead of being ignored.** The
  documented behaviour is that a malformed value falls back to the server
  default, and an empty string now does that too.

- **`eullm daemon` reported success when it had not written its pidfile.** It
  printed `eullm daemon started (PID N)` and exited cleanly with nothing on
  disk, which leaves a stop script no way to find the process — worse than an
  outright failure. It now names the path and the error, and stops the child
  it had already started instead of leaving it holding the port.

## 0.7.5 — 2026-09-12

### Added
- **AMD ROCm binaries are now published**, two of them. The only build for AMD
  hardware until now was the Vulkan one, which is a fine answer for a Radeon in
  a desktop and the wrong one for a data-centre card — those have no display
  stack for a Vulkan driver to attach to, and their matrix cores go unused
  under one.

  `eullm-linux-x64-rocm-consumer` covers RDNA 3 (RX 7900/7800/7700/7600),
  RDNA 3.5 (the Ryzen AI integrated Radeons) and RDNA 4 (RX 9000), built
  against ROCm 7.2. `eullm-linux-x64-rocm-gfx90a` covers Instinct MI250X and
  MI210, built against ROCm 6.3 to match the stack EuroHPC sites run. Neither
  bundles ROCm: both resolve the libraries already installed on the machine, so
  check `hipconfig --version` against the version in the artifact name before
  assuming one fits. RDNA 2 (RX 6000) is not included — the Vulkan binary
  remains the build for those cards.

  Building from source for a different AMD card now works too, which it did
  not before — `cargo build --features rocm` never passed a GPU architecture,
  so it compiled for whatever card was in the build machine, and produced a
  binary with no device code at all on a machine with no GPU (an HPC login
  node, a CI runner). Set `EULLM_AMDGPU_TARGETS` (e.g. `gfx1100`) to choose.

## 0.7.5-rc9 — 2026-09-07

### Fixed
- **Sending a large image killed the engine.** Not an error message — the
  process aborted, taking the loaded model and every other request with it,
  so the next thing anyone did failed with `network error` and everything
  after that with `Failed to fetch`. A 1584×1584 photo through Gemma 4's
  projector encodes to 1089 tokens; the context had been built for a batch of
  512, and llama.cpp enforces that with a `GGML_ASSERT`, which calls
  `abort()` rather than returning an error.

  Two numbers were involved and neither was derived from the image: the
  context was sized from a fixed 512, measured once against Gemma 4's
  ~256-300 tokens for a typical slice, while the splitting of media chunks
  was handed the *text* prefill batch of 2048. So a single batch of up to
  2048 tokens could be passed to a context that accepted 512. The batch is
  now sized to the largest media chunk the request actually tokenized to,
  and the same figure is used for both, so they cannot disagree. An image
  past 8192 tokens is refused with a message naming
  `EULLM_IMAGE_MAX_TOKENS`, instead of asking for a compute buffer that will
  not fit.

### Fixed
- **The model browser said "no GPU detected" on every build except CUDA, and
  judged downloads against system RAM alone.** VRAM was read with
  `cudaMemGetInfo`, compiled in only for the CUDA binaries, so the Vulkan,
  Metal, ROCm and CPU builds all reported no GPU — a Vulkan machine with a
  16 GB card was told it had none, and a 21 GB model was coloured against its
  64 GB of RAM instead. It is now asked through ggml's device registry, which
  every backend populates, and several GPUs are summed because that is what a
  layer split can use. This also gives `--fit` a real VRAM figure on those
  builds for the first time, where it previously fell back to whatever
  `--gpu-layers` the user passed.

- **Opening a repo in the model browser showed the bottom of its
  quantization list**, so a repo with twenty of them opened on the largest
  and hid the line saying what the traffic lights were judged against. The
  repo name and that line now stay put while only the list scrolls, and the
  list starts at the smallest.

## 0.7.5-rc8 — 2026-09-07

### Added
- **A model browser in the web UI.** The download icon in the top bar opens a
  search over HuggingFace's GGUF repos, and picking one lists every
  quantization it offers with its total download size, its shard count, and a
  traffic light saying whether it is expected to run on this machine — green
  fits on the GPU or comfortably in RAM, amber runs with layers on the CPU,
  red is larger than VRAM and RAM together. Downloading streams a progress
  bar and the model appears in the picker without a reload.

  The engine makes the HuggingFace calls, not the browser. Your address is
  never handed to the Hub by opening the catalog, `EULLM_WEB_ALLOWED_DOMAINS`
  stays the one place the perimeter is decided, and — the reason it was built
  this way — the catalog keeps working on a machine whose browser has no
  route out but whose engine does. That is an HPC login node, which is
  exactly where downloads have to be started when the compute nodes are
  offline.

  The traffic light is an estimate from the download size, and says so in the
  panel. The exact layer split needs the GGUF header, which needs the file;
  this exists to answer whether the download is worth starting, and that has
  to be answered before the download or not at all. It applies the same
  headroom the real sizer does, so a green light does not turn into a partial
  offload after 100 GB.

- **`POST /api/pull` works, and streams NDJSON the way Ollama does.** It was
  a stub that answered `not yet implemented` and did nothing. It now pulls
  `hf.co/<owner>/<repo>[:<quant>]`, emitting one JSON object per line —
  `{"status"}` while it works, `{"status","digest","total","completed"}`
  while bytes move, `{"status":"success"}` at the end — so an existing Ollama
  client shows a progress bar with no changes. `{"stream": false}` returns a
  single object instead. The CLI and the API now run the same download code
  rather than two copies of it.

- **New Linux CUDA download for data-center NVIDIA GPUs**:
  `eullm-linux-x64-cuda-12.4-datacenter`, built for A100 (sm_80) and H100
  (sm_90). The existing `eullm-linux-x64-cuda-13.1` remains the consumer
  build (RTX 3000/4000/5000, sm_86/89/120) and does not run on A100/H100 —
  those are a different, older compute capability that the consumer build
  never targeted. RTX A-series workstation cards are Ampere sm_86 and are
  already covered by the existing consumer download.

  **This build uses CUDA 12.4, not 13.1 like the consumer one**, so its
  minimum NVIDIA driver is r550 rather than r580. The first attempt shipped
  it on CUDA 13.1 and it would not start on CINECA Leonardo at all — CUDA 13
  requires an August-2025 driver, and Leonardo's is older, so the binary
  died with `CUDA driver version is insufficient for CUDA runtime version`
  and silently fell back to running a 27B model on CPU. A100s live
  overwhelmingly in HPC centres and enterprise fleets, which freeze driver
  versions for years: aiming this particular artifact at the newest drivers
  was the wrong trade for its audience.

### Fixed
- **`eullm pull` now downloads split GGUFs, so large quantizations can be
  pulled at all.** A model published as `…-00001-of-00004.gguf` is four
  files, and llama.cpp opens the rest from the first only once they are all
  on disk — but the pull resolved to a single filename and downloaded one.
  With four shards matching the requested quantization, none of the
  single-file branches applied and the pull ended on `multiple .gguf files
  match quant 'UD-Q4_K_XL'`. Every shard is now fetched, in order, and the
  recorded size is the sum rather than the first shard's.

  This completes the previous entry, which did not go as far as it read: it
  made the files in a per-quantization subdirectory *visible*, which changed
  the failure from `contains only a projector` to the ambiguity error above,
  but left `unsloth/Qwen3.8-Flash-Next-GGUF` — and every repo like it —
  still impossible to pull. A repo whose only model is one split is also no
  longer refused with `pass an explicit :<quant>`, which was advice that led
  straight into the same wall. An incomplete split is refused up front,
  naming the missing shards, rather than downloading most of a model and
  failing at load.

- **Images sent to a non-Gemma multimodal model were refused outright, or
  answered badly.** Two separate causes, both now fixed, found running
  Qwen3.8-Flash-Next on 4× A100.

  The first was an outright refusal: `This model requires M-RoPE positions,
  which the current MVP does not yet plumb through the decode loop`. The
  guard assumed a model with multi-dimensional positions needed four
  positions per token in the decode batch. It does not — llama.cpp
  broadcasts a text batch's single position across all RoPE sections, and
  its own reference `mtmd-cli` generates with the same scalar position we
  do. Every Qwen VL was refused for nothing. The guard is gone; `m_rope` is
  now only logged at load.

  The second was quieter and worse. The multimodal path built its prompt
  with Gemma's `<start_of_turn>` turn markers, hardcoded, for every model.
  It now renders the model's own GGUF-embedded chat template — what the
  text path has done since 0.7.0 — with the media marker placed inside the
  user message, so a Qwen VL gets ChatML and a Gemma still gets Gemma. A
  wrong template does not error, it just degrades the answer, which is why
  this survived: Gemma 4 was the only multimodal model in the catalog. The
  `think` parameter now reaches the multimodal path too. Both the API and
  `eullm run --image` are fixed.

- **`eullm pull` could not download from HuggingFace repos that put each
  quantization in its own subdirectory.** Large models are increasingly
  published that way — `unsloth/Qwen3.8-Flash-Next-GGUF` stores its files as
  `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` — and the
  repo listing dropped every file whose name contained a `/`, so the whole
  repo looked empty of weights. What the user saw was the misleading
  `this HuggingFace repo contains only a projector (mmproj), no model
  weights`. Subdirectories are now accepted, each path segment still
  validated exactly as a bare filename was, and the file is saved under its
  own name rather than the full repo path.

- **The Linux CUDA binaries (`eullm-linux-x64-cuda-13.1` and the new
  `-datacenter` variant) didn't run on RHEL, CentOS, Rocky Linux, AlmaLinux,
  or Oracle Linux — any RHEL 8-family distro.** They were built against
  Ubuntu 22.04's glibc, and RHEL-family distros freeze glibc for the life of
  the release (8.x stays on 2.28) — the binary failed at startup with
  `GLIBC_2.29' not found` and similar. Found getting EULLM running on CINECA
  Leonardo (RHEL 8.7). Both Linux CUDA builds now compile on a Rocky Linux 8
  base image instead, so they link against glibc 2.28 — which, being
  backward-compatible, still runs fine on current Ubuntu/Fedora/Arch/
  Windows, so nothing changes for existing users.

- **The CPU build `eullm-linux-x64` had the same problem**, and the same
  fix. It was missed on the first pass, which only moved the two CUDA
  builds: running it on Leonardo's login node still failed with
  `GLIBC_2.29' not found`. It now builds on Rocky Linux 8 too. This is the
  binary that matters most on an HPC system, since compute nodes usually
  have no outbound network and model downloads have to run from the login
  node. The Vulkan and ARM64 Linux builds are still produced on Ubuntu and
  carry the same exposure; macOS and Windows are unaffected.

### Changed
- **EULLM's own code is now licensed AGPL-3.0-or-later, not Apache 2.0.**
  This applies to new work going forward only: every version already
  published (0.7.4 and earlier) keeps its original Apache 2.0 terms for
  anyone who already has a copy — a license grant already given can't be
  revoked. The practical difference for AGPL-3.0: if you modify EULLM and
  let others use it over a network, including as a hosted service, you must
  offer them the Corresponding Source of your modified version. Running an
  unmodified build, or modifying it for purely internal/local use, carries
  no new obligation. The models EULLM runs (Qwen 3, Mistral, Falcon 3, etc.)
  are unaffected — they keep their own separate licenses. Contributions now
  require signing the project's CLA before merge.

## 0.7.4 — 2026-08-28

### Fixed
- **Loading a model with `--n-cpu-moe N` for `N` greater than 1 crashed
  instead of loading**, panicking with `last buft_override was not empty`.
  Every MoE-to-CPU layer override after the first was written into the same
  slot instead of the next free one, so the crash was guaranteed as soon as
  more than one layer needed offloading — in practice, any large MoE model
  that `--fit` or a manual `--n-cpu-moe` chose to partially offload, such as
  Qwen3-Next-80B-A3B on a single consumer GPU. `--cpu-moe` (offload every MoE
  layer in one call) and single-layer `--n-cpu-moe 1` were unaffected; this
  is what broke everything above that.
- **`--fit`'s automatic GPU-layer sizing reserved more headroom for the
  compute buffer than modern loads actually use**, so it sometimes left a
  layer or two on CPU that would have fit on GPU. The reserve was set years
  ago and had already been flagged as outdated; a real measurement on
  today's loader (27B model, 16384 context) now backs the new, smaller
  figure. This only affects the automatic split `--fit` picks when no
  `--gpu-layers` is given — `--gpu-layers` still overrides it, and
  `--no-fit` skips it entirely.

## 0.7.3 — 2026-08-23

### Changed
- **The NVIDIA downloads shrink by a large fraction, and their file names
  change.** The CUDA binaries were carrying NVIDIA's cuBLAS device code for
  every GPU architecture the toolkit supports — Maxwell through Blackwell —
  while the binary itself is built for, and can only run on, RTX
  3000/4000/5000. That unusable code is now stripped out before linking, and
  the build moved to the CUDA 13 toolkit, which had already dropped the
  oldest architectures on its own. Nothing about what the binary does or
  which GPUs it supports changes; it is the same engine in a much smaller
  file. Two consequences worth knowing before you upgrade: the downloads are
  now named `eullm-linux-x64-cuda-13.1`, `eullm-linux-arm64-cuda-13.1` and
  `eullm-windows-x64-cuda-13.1.zip` (they were `-cuda-12.8`), so a script
  that fetches the old name by URL needs updating; and they require **NVIDIA
  driver 580 or newer**, up from 570. The CPU, Vulkan and macOS builds are
  untouched.

### Fixed
- **The terminal chat cut every reply off at 2048 tokens, even with a much
  bigger context window open and mostly empty.** `--ctx-size 4096` with a
  short prompt left roughly 4000 tokens of real room, but a leftover cap
  clamped every reply to 2048 regardless, and reported it as
  `truncated — out of context` — a context exhaustion that had not actually
  happened. Longer answers, and models that think before answering, are no
  longer cut short for a reason unrelated to the context you asked for.
  `/maxtokens <n>` still sets an explicit cap when you want one.
- **A long chat with a reasoning model ran out of context far sooner than it
  should have in the terminal.** Every reply was stored in the conversation
  complete with the model's thinking, so each new turn re-sent all of the
  reasoning from every previous turn — several times more text than the
  answers themselves, for no benefit, until the context filled and replies
  came back truncated. The terminal chat now keeps the answers and drops the
  thinking, which is what the web UI already did. Long conversations with
  models like DeepSeek and the Qwen3 thinking variants now last several times
  longer before hitting the limit, and the model is no longer fed its own
  earlier deliberations as if they were part of the conversation.

## 0.7.2 — 2026-08-23

### Fixed
- **A model that fit and ran a moment ago could suddenly fail to allocate its
  context — down to `--ctx-size 512` — on a card with just enough VRAM for
  it.** The physical prefill micro-batch (`n_ubatch`) was set to 1024 instead
  of llama.cpp's own default of 512, doubling the compute buffer that sizing
  depends on; on a card where a model barely fits, that doubling was the
  difference between loading a real context and failing at every size. Now
  matches llama.cpp's default, so eullm allocates the same context a stock
  `llama-cli` does on the same file, layers, and card.
- **Roughly 300 MiB of VRAM per loaded model was reserved for output rows the
  server never uses, on top of the fix above.** The load-time graph sized its
  logit output for the full batch when a chat only ever reads one logit per
  sequence per step; that space is now reclaimed, which on a large model can
  mean one or two more layers fit on the GPU, or a bigger context, before
  anything spills to slower CPU offload.
- **Security: a request naming a model with `../` in it could reach files
  outside the model store.** A crafted `model` field (e.g.
  `"../../../../tmp/x"`) let the loader resolve, and in the delete path
  *remove*, a directory outside the store, and could confirm whether a file
  existed anywhere the process could read. Every model-name lookup now refuses
  a name that would escape the store directory. Relevant to any deployment
  that exposes the API beyond the local machine.
- **With prompt checkpoints enabled (`--ctx-checkpoints`), resubmitting an
  identical prompt on a hybrid/recurrent model could make the first reply
  token come from the wrong conversation.** A checkpoint covering the entire
  prompt left nothing to decode, so the reply was sampled from whatever the
  context last held — silently. Fixed; the affected case now re-runs the
  prompt normally.

### Added
- **`eullm -V` and the startup banner now show the exact git commit the binary
  was built from** (e.g. `0.7.2 (CUDA) [a1b2c3d4e5f6]`, with `-dirty` when the
  working tree had uncommitted changes at build time) — so a build from source
  is identifiable beyond its version number.

## 0.7.1 — 2026-08-22

### Changed
- **Bumped the pinned `llama.cpp` from tag `b10200` (2026-07-30) to `b10405`
  (2026-08-12), and re-vendored the Rust bindings (`llama-cpp-rs`) cleanly
  against upstream's own current release — the three C-API compatibility
  patches carried by hand since 0.6.70 (`load_mode`, `mtmd_input_text`'s
  length field, the multimodal bitmap helper's return type) are gone,
  superseded by upstream's own equivalent code. Brings every upstream fix
  and model architecture addition from that window. No flag, default, or
  observable behaviour changes.

## 0.7.0 — 2026-08-21

### Fixed
- **The embedding model could fail to load with `Failed to load embedding
  model: BackendAlreadyInitialized` whenever a generation model was already
  loaded** — in practice, whenever `--embedding-model` or `/api/embed`/
  `/v1/embeddings` tried to do the one thing they exist for: run alongside a
  resident chat model on a card with room for both. The generation model and
  the embedding model each initialized their own `llama.cpp` backend, but
  llama.cpp allows only one live backend per process — a second
  initialization while the first is still active always fails. This affected
  every release since 0.6.82 (when in-process embeddings first shipped)
  whenever a card had room to keep both models resident at once; a card too
  small for both never hit it, because the coexistence path is exactly the
  one that was broken. Fixed by initializing one `llama.cpp` backend at
  process startup and sharing it across every model the process loads —
  the launch model, every later model swap, and the embedding slot.

## 0.6.90 — 2026-08-18

### Added
- **`POST /api/embed` and `POST /v1/embeddings`** — text embeddings served
  from the same binary, with any GGUF embedding model (BGE, E5, and
  similar). The embedding model loads into its own slot, independent of
  whatever generation model is loaded: on a card with enough VRAM for both,
  a RAG pipeline can embed and generate without either one displacing the
  other. On a smaller card, requesting the embedding model automatically
  frees VRAM by unloading the generation model first (it reloads on the
  next generation request), and the reverse happens too — a generation
  load can evict a resident embedder if it needs the room. Which
  direction happened, and how often, is in `/api/version`'s new
  `model_swaps` field.
- **`--embedding-model <path-or-name>`** — load a text-embedding model at
  startup, on both `eullm run` and `eullm serve`, as a **reserved
  companion**: it loads first, so its weights already count as used VRAM by
  the time `--fit` sizes the generation model, and `--fit` keeps a small
  compute-buffer margin free on top for it — so both stay resident together
  whenever the card has room, instead of depending on which of two
  independently-launched processes claims VRAM first. A reserved companion
  is never evicted to free room for a generation load, unlike an embedding
  model loaded on demand by naming it in a request. If reserving that
  margin would leave the generation model no headroom at all, the launch
  proceeds anyway with a warning and the embedder falls back to loading on
  demand, exactly as if the flag had not been given.
- **`--keep-alive <duration>`**, plus a per-request `keep_alive` field
  (Ollama-compatible) on `/api/generate`, `/api/chat`,
  `/v1/chat/completions`, `/api/embed` and `/v1/embeddings`. Idle-unloads a
  model after this long without a request, so a card left untouched
  actually goes idle instead of holding an active CUDA context (and its
  higher memory clocks) indefinitely. Unset by default — matches every
  earlier release, where nothing unloaded a model on its own.
- **An empty `prompt` on `/api/generate`, or an empty `messages` array on
  `/api/chat`, now loads (or `keep_alive`-refreshes) the model without
  generating anything** — Ollama's documented way to warm a model up in
  advance, which previously fell through to the model as literal input.

### Fixed
- **A mistyped slash command in the terminal chat is no longer sent to the
  model as a message.** `/q` (and any other unrecognised `/`-prefixed line)
  now prints the command and points at `/help` instead of being answered by
  the model at a few tokens per second with no indication of what happened.
  `/q` also now works as an exit command, alongside `/bye`, `/exit` and
  `/quit`. A message that genuinely starts with a literal `/` still goes
  through — type `//` and the first slash is stripped.
- **The model picker now shows directories with no `manifest.json`.**
  Reorganizing a loose `models/x.gguf` into `models/x/x.gguf`, an
  interrupted pull, a restored backup, or a directory copied from another
  machine all used to vanish from the picker, even though `eullm list` and
  running the model by path both still worked. The picker now finds these
  too — skipping any `mmproj*.gguf` sitting alongside them.

## 0.6.81 — 2026-08-11

### Changed
- **The daemon writes its log to `~/.eullm/logs/eullm.log`, not `/tmp`.**
  The log path was derived from the PID file, so it inherited `/tmp` — a
  directory that is small on many systems and is cleared on reboot or
  under space pressure, which is exactly when the daemon's log is the only
  record of what happened. The PID file stays in `/tmp`, where a file that
  is meaningless after a reboot belongs. Reported by
  [@PeterHickman](https://github.com/PeterHickman) (#354).

### Added
- **`qwen3.5-9b` in the catalog** (`eullm run qwen3.5-9b`). Qwen 3.5 9B,
  Apache 2.0, hybrid architecture: three Gated DeltaNet blocks per
  full-attention block, so only 8 of its 32 layers carry a KV cache and
  its 262K native context is affordable on a mid-range card. 5.7 GB at
  Q4_K_M. Vision-capable through the projector shipped with it, which
  `eullm pull` fetches alongside the weights.
- **`--logfile <PATH>`**, on both `run` and `serve`, to put the daemon log
  wherever the deployment wants it (`/var/log/eullm.log`, a mounted
  volume). Setting `--pidfile` on its own still keeps the log beside the
  PID file, so a deployment that already redirects one is unaffected.

### Fixed
- **Ctrl+C stops the process, instead of leaving it running invisibly.**
  The shutdown message printed and the terminal came back, but the process
  stayed alive until any in-flight work finished — with a long prompt in
  progress, that is minutes of an apparently-dead process still holding
  the port and the model's memory. `#[tokio::main]`'s runtime drop waits
  for every blocking task, and inference runs in exactly those. The
  process now exits once the command is done: nothing needed the wait,
  since the audit trail is written per completed request and models are
  read-only.

### Performance
- No engine change, but a measurement worth acting on: on a Radxa Orion
  O6, the `eullm-linux-arm64-cix-p1` binary is **2.2–3.1× faster at prompt
  processing and 1.8× faster at generation** than the generic
  `eullm-linux-arm64` one (qwen3-4b Q4_K_M, CPU-only, same GGUF, same
  flags). The two builds differ only in the CPU instruction set their
  kernels are compiled for. If you run this board, use the `cix-p1`
  binary; the generic one exists so ARM64 boards without those extensions
  still work. Full A/B in `docs/arm-cix-p1-cpu-profile.md`.

## 0.6.80 — 2026-08-10

*Accumulated through pre-releases `rc1`–`rc12`, each validated on real
hardware as it landed. Two users' reports drove most of this release: tool
calling and the model-identity fixes came from [@odlg](https://github.com/odlg)
running EuLLM behind an IDE agent, and the sizing work came from watching a
16 GiB card actually load these models — a vision model, a dense 27B and a
35B MoE swapped back and forth in one conversation.*

### Changed
- **The GPU offload is now sized automatically, without `--fit`.** Loading
  a model larger than the free VRAM used to end in an out-of-memory error;
  with sizing the worst case is a slower partial split, so the old default
  was the one that picked the crash. It bit a user mid-session: a chat-UI
  switch to a 27B loaded with `all` layers and died. Three things keep it
  out of the way: `--gpu-layers` becomes an upper bound rather than a
  fixed count — sizing still runs and may offload fewer, so a number
  chosen for one model cannot run the next one out of memory — `--no-fit`
  restores the previous behaviour outright for anyone who wants to force
  a count past the estimate, and automatic sizing never asks questions:
  it applies the split and logs one line naming both flags. Passing `--fit`
  explicitly still means what it did: on an interactive terminal a partial
  split asks for confirmation. Where free VRAM cannot be probed (any
  non-CUDA build) sizing stays silent and `--gpu-layers` is used as-is.

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
- **Automatic sizing no longer produces a split the loader then refuses.**
  Swapping into a 27B on a 16 GiB card loaded the weights and then failed
  to allocate a context at all, down to 512 tokens: *"allocation succeeded
  but left only 10% of GPU memory free"*. Two parts of the engine disagreed
  about how much VRAM must stay free — the sizer aimed to leave 3% of what
  was free, the loader's context probe requires 12% of the card's total —
  so the sizer could hand over a layer count that could never load. The
  sizer now respects the loader's floor, which is the larger of the two on
  any card with memory in use.

- **A vision model no longer pins the server to sequential mode.** After
  launching a multimodal model, every model swapped in afterwards ran
  without the continuous-batching scheduler, because the sequential
  fallback that vision needs was passed on to the server instead of
  staying with the model that needed it. The requested batch size is kept;
  the fallback is re-applied per model, to whichever one actually carries
  a projector.

- **Switching away from a vision model no longer breaks the next load.**
  Launching a multimodal model (Gemma 3/4 and friends) and then switching
  to any other model from the chat UI failed with `mismatch between text
  model (n_embd = 2048) and mmproj (n_embd = 2560)`: the projector found
  for the launch model was handed to the server as a fallback and applied
  to every model swapped in afterwards. Only an explicit `--mmproj` is a
  fallback now. Models with a projector of their own still find it, from
  the store entry or from the file beside their weights.

- **A download onto an unmounted volume says so, instead of "File
  exists".** When the model store lives behind a symlink to a volume that
  is not mounted, creating the destination directory fails with EEXIST —
  the link is there, it just leads nowhere — and the pull reported
  `Download failed: File exists (os error 17)`, which reads as "you
  already have this model". The download path now uses the same
  diagnostic the rest of the store already had: it names the dangling
  symlink and says to mount the volume or set `EULLM_MODELS_DIR`.

- **Different quants of the same model are different models again**
  (#345). Switching between `repo:UD-Q5_K_XL` and `repo:UD-Q4_K_M` did
  nothing: the server compared names with a file-stem rule that cuts at
  the last dot, so every `ornith-1.*` quant collapsed into `ornith-1` and
  the requested model looked already loaded. Comparison now keys on the
  full name (path component and `.gguf` extension aside), so names with
  dots in them — `qwen3.6-27b`, any `x.y` release — stay distinct.
  Related: running a bare `hf.co/owner/repo` when a quant of that repo is
  already downloaded now uses it instead of downloading a second copy; if
  several are present, it lists them and asks which one, rather than
  guessing.

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
