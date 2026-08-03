#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
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

// Renders `n_messages` role/content pairs through the model's own chat
// template (the Jinja template embedded in the GGUF, read via
// llama_model_chat_template), the same way llama-server does by default.
// `*out_was_explicit` reports whether the GGUF actually carried a template —
// when false, llama.cpp silently fell back to a built-in ChatML template
// internally, and `*out_prompt` reflects that fallback rather than anything
// specific to this model; callers should prefer their own known-good
// template in that case rather than trust this output.
// `*out_thinking_start_tag`/`*out_thinking_end_tag` are set (caller must
// free) only when the template declares reasoning/thinking delimiters;
// otherwise left NULL. Text-only: message content is a single string per
// message, not the structured content-parts multimodal templates can use.
llama_rs_status llama_rs_apply_chat_template(
    const struct llama_model * model,
    const char * const * roles,
    const char * const * contents,
    size_t n_messages,
    bool add_generation_prompt,
    bool * out_was_explicit,
    char ** out_prompt,
    char ** out_thinking_start_tag,
    char ** out_thinking_end_tag);

void llama_rs_string_free(char * ptr);

#ifdef __cplusplus
}
#endif
