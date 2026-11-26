# llama.cpp b12767: Inkling 2 with fewer operations

Branch `chore/llama-cpp-b12767`, from `main` after release 0.10.10. The
llama.cpp submodule moves from `969d733` (`b12632-23`, 23 November) to
`a8ad690` (`b12767`, 25 November): 112 commits, llama.cpp 1.1.0.

## 1. Why now

Unsloth's own llama.cpp build writes Inkling 2 Mini up to 1.6 times faster
than mainline b12649 on NVIDIA (their page, 23-11-2026). Part of it is in
mainline now, after our pin:

- `#32948`, `#32855`, `#32940`, `#33053`: Inkling 2's graph with fewer
  operations in `mix()`, `control()`, the clip and the relative bias.
- `#32218`: DSpark, Inkling 2's speculative drafter (llama.cpp side only:
  the engine does not drive it yet).
- `#32500` (RMS_NORM + MUL + MROPE fused on CUDA), `#32875`, `#32965`.
- Two architectures: `hy_v4_flash`, `raon1`.

The rest of Unsloth's gain is CUDA kernel fusions in their fork only
(`unslothai/llama.cpp` PRs 400-418, MIT): the next step, measured one by
one on the reference PC, if this bump does not close the gap.

## 2. The C API and the wrapper

`common_chat_tool_choice_parse_oaicompat` takes the request's JSON value
and the parsed tools (for the named-function and `allowed_tools` forms).
`wrapper_common.cpp` passes the string the engine sends ("auto", "none",
"required") as a JSON string. Nothing else in the C API the crate uses
changed.

## 3. The patches

All 14 rebased with `git rebase` (three-way, no conflict) and regenerated
against the new commit: the same files and line counts as before, offsets
only. `cargo test --test llama_patches`: 11 of 11.

## 4. Checked, and not

In the cloud container (CPU build): the engine builds, `cargo test --lib`
(858), clippy as CI runs it, `cargo check --features multimodal`;
`tools/answer_check.py` on the release binary, all ten KV cache settings;
tool calls on `/v1/chat/completions` with `tool_choice` auto and required
(`get_weather({"city": "Rome"})`), and none as on 0.10.10.

Not checked here, to do before the tag: CUDA, ROCm and Vulkan compile
(the release jobs, or `job: cuda`), and Inkling 2 Mini's speed on the
reference PC against 0.10.10's 36-44 tokens/s (5070 Ti alone) and
34-35.6 (with the RX 6700 XT).

The `eullm/llama.cpp` mirror must hold `a8ad690` before CI can check the
submodule out.
