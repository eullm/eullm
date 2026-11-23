# llama.cpp b12632+23: Inkling 2, and what else moved

Branch `chore/llama-cpp-b12632`, from `main` after release 0.9.40. The
llama.cpp submodule moves from `v1.0.0` (b12359, `b1b770d`, 13 November) to
`969d733` (`b12632-23`, 23 November), the head of the `eullm/llama.cpp`
mirror: 296 commits. The reason for moving now: Inkling 2 Mini, a looped
model, failed to load with `unknown model architecture: 'inkling2_loop'`
(23-11-2026, on the reference PC).

## 1. What EuLLM gains

- **Inkling 2** (`inkling2`, `inkling2_loop`): the architecture (#32694, in
  b12600), its loop slots shared after the scale pass (#32704, which the
  GGUFs in Unsloth's `Mainline_Support` folder need, b12611), its vocabulary
  (#32799) and its chat parser and grammar (#32833). The GGUFs at the root of
  Unsloth's repository are for Unsloth's own fork and may still not load.
  An issue on CUDA is open upstream (#32710).
- Everything else of the 296 commits, unread here one by one.

## 2. The C API

Additions only, besides deprecations: `llama_decode`, `llama_encode`,
`llama_batch_init`, `llama_batch_free` and `llama_batch_get_one` are marked
`DEPRECATED`, in favour of `llama_batch_ext` and `llama_process`. They still
exist; the vendored `llama-cpp-2` calls them and builds unchanged. The next
pruning will remove them, as 1.0 removed 81 functions: moving `llama-cpp-2`
to the new batch API is the work to do before that bump, not during it.

## 3. The patches

| Patch | What happened |
|---|---|
| `0001`, `0003`-`0011` | Apply as they were; regenerated against the new commit (offsets only) |
| `0002` | Applies with a three-way merge, no conflict |
| `0012`, `0014` | Upstream turned the loader's staging buffer from a `std::vector` into a CPU backend buffer. `0012` keeps its read timing beside the new buffer; `0014` keeps "the buffer only grows" (tensors in file order) with the new buffer's size and allocation |
| `0013` | **Dropped**: upstream now defines `HIP_ENABLE_WARP_SYNC_BUILTINS` in `vendors/hip.h`, so `__ballot_sync` compiles on ROCm 6.x, and the group mask uses a 64-bit thread mask and the physical warp size (`ggml_cuda_ballot`), which covers the 64-wide wavefront. Both reasons for the patch are gone |
| `0015` | `ggml_cuda_should_use_mmq_few` took an `ids` argument (few-row MMQ for MUL_MAT_ID); the switch is checked first, as before |

## 4. What was checked, and what was not

Checked in the cloud container (CPU build): `cargo test --test llama_patches`
(11 of 11), the engine builds, `cargo test --lib` (839), clippy as CI runs
it; Qwen3-0.6B Q8_0 answers on `/api/chat`, and with `tools` returns
`get_weather({"city": "Rome"})` streamed and not, and answers from the tool's
result on the second round.

Not checked here, to do before the tag:

- **CUDA**: `0015` was resolved by hand in `mmq-few.cu`, and nothing CUDA
  compiles in the container. The release's CUDA jobs, or a build on the
  reference PC, are the first compile.
- **ROCm 6.3.4 for the MI250X**, now without `0013`: the release's gfx90a
  build, or a build on LUMI.
- **Real hardware**, as `engine/CLAUDE.md` asks of every bump: each model
  family available locally once, a multimodal one, DeepSeek's reasoning
  template, and Inkling 2 Mini itself.

The next release is cold: sccache keys on llama.cpp's sources, so every
kernel recompiles once (see the release skill: 67-69 minutes after the
v0.7.30 bump).

## 5. Found after the release (0.9.50)

**Parsers match special tokens by id** since #29307 ("chat : implement token
aware parsing", 22 November, in this bump). A parser built from templates
that know the model's vocabulary matches a special-token marker
(`p.token(...)`) only where the input carries that token's id; the engine
hands `llama_rs_chat_parse` the text it streamed, with no ids. So in 0.9.50
every format whose markers are special tokens failed to read, leniently: no
reasoning, no tool calls, and the whole output as content with the
generation prompt in front. Seen with Gemma 4 E2B on `/v1/chat/completions`
with tools (`<|turn>model\n<|channel>thought…` as content, no call), 24-11-2026;
Qwen3, whose `<think>` is not a special token, was unaffected.

`wrapper_common.cpp` now builds the templates without the vocabulary's
specials (`common_chat_templates_set_specials(tmpls, {})`): a token parser is
then a literal, as before #29307. Passing the token ids instead would need
them carried with every streamed piece; worth doing if a format turns out to
need ids (a marker that also occurs as plain text in answers).
