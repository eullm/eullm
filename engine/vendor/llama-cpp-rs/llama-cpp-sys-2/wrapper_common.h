#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
struct llama_rs_mtp_speculative;
struct llama_vocab;

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

// Fit model/context params to device memory (wraps llama.cpp's common_fit_params).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min);

void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec);

llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch);

llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count);

llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted);

void llama_rs_string_free(char * ptr);

// EuLLM addition: renders `n_messages` role/content pairs through the
// model's own chat template (the Jinja template embedded in the GGUF, read
// via llama_model_chat_template), the same way llama-server does by
// default.
// `*out_was_explicit` reports whether the GGUF actually carried a template —
// when false, llama.cpp silently fell back to a built-in ChatML template
// internally, and `*out_prompt` reflects that fallback rather than anything
// specific to this model; callers should prefer their own known-good
// template in that case rather than trust this output.
// `*out_thinking_start_tag`/`*out_thinking_end_tag` are set (caller must
// free) only when the template declares reasoning/thinking delimiters;
// otherwise left NULL. Text-only: message content is a single string per
// message, not the structured content-parts multimodal templates can use.
// `enable_thinking` maps to common_chat_templates_inputs.enable_thinking:
// templates with a reasoning toggle (Qwen3 family) render their own
// suppression form when false (e.g. a pre-closed empty <think> block);
// templates without one ignore it.
llama_rs_status llama_rs_apply_chat_template(
    const struct llama_model * model,
    const char * const * roles,
    const char * const * contents,
    size_t n_messages,
    bool add_generation_prompt,
    bool enable_thinking,
    bool * out_was_explicit,
    char ** out_prompt,
    char ** out_thinking_start_tag,
    char ** out_thinking_end_tag);

// EuLLM addition: OpenAI-compatible chat template application.
// `messages_json` is the request's own OpenAI-format messages array (so
// tool roles and assistant tool_calls survive), `tools_json` the OpenAI
// tools array or NULL, `tool_choice` "auto"/"required"/"none" or NULL.
// Besides the rendered prompt, returns the output-format triple
// (`out_format`, `out_parser`, `out_generation_prompt`) that
// llama_rs_chat_parse needs.
llama_rs_status llama_rs_apply_chat_template_oai(
    const struct llama_model * model,
    const char * messages_json,
    const char * tools_json,
    const char * tool_choice,
    bool add_generation_prompt,
    bool enable_thinking,
    bool * out_was_explicit,
    char ** out_prompt,
    int32_t * out_format,
    char ** out_parser,
    char ** out_generation_prompt,
    char ** out_thinking_start_tag,
    char ** out_thinking_end_tag);

// EuLLM addition: parse raw model output into an OpenAI-compatible message
// JSON ({"role","content","reasoning_content","tool_calls",...}) using the
// format triple from llama_rs_apply_chat_template_oai. Model-free and
// stateless.
llama_rs_status llama_rs_chat_parse(
    const char * input,
    bool is_partial,
    int32_t format,
    const char * parser,
    const char * generation_prompt,
    char ** out_json);

#ifdef __cplusplus
}
#endif
