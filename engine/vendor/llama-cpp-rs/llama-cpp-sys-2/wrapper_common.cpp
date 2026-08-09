#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <stdint.h>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

// OpenAI-compatible chat template application: messages and tools arrive as
// the request's own JSON, so tool roles, assistant tool_calls in history and
// tool definitions all flow through llama.cpp's oaicompat parsers instead of
// a lossy (role, content) projection. Returns, alongside the prompt, the
// output-format descriptor triple (format id, saved PEG parser, generation
// prompt) that llama_rs_chat_parse needs to turn the model's raw output back
// into structured content / reasoning_content / tool_calls.
extern "C" llama_rs_status llama_rs_apply_chat_template_oai(
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
    char ** out_thinking_end_tag) {
    if (!model || !messages_json || !out_was_explicit || !out_prompt || !out_format ||
        !out_parser || !out_generation_prompt || !out_thinking_start_tag || !out_thinking_end_tag) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_was_explicit = false;
    *out_prompt = nullptr;
    *out_format = 0;
    *out_parser = nullptr;
    *out_generation_prompt = nullptr;
    *out_thinking_start_tag = nullptr;
    *out_thinking_end_tag = nullptr;

    try {
        auto tmpls = common_chat_templates_init(model, /* chat_template_override */ "");
        *out_was_explicit = common_chat_templates_was_explicit(tmpls.get());

        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.enable_thinking = enable_thinking;
        // Matches llama-server's default: reasoning is extracted to
        // reasoning_content, and templates that strip prior-turn reasoning
        // from the history do so.
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        inputs.messages = common_chat_msgs_parse_oaicompat(nlohmann::ordered_json::parse(messages_json));
        if (tools_json && tools_json[0] != '\0') {
            inputs.tools = common_chat_tools_parse_oaicompat(nlohmann::ordered_json::parse(tools_json));
        }
        if (tool_choice && tool_choice[0] != '\0') {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
        }

        const auto params = common_chat_templates_apply(tmpls.get(), inputs);

        *out_prompt = llama_rs_dup_string(params.prompt);
        if (!*out_prompt) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        *out_format = (int32_t) params.format;
        *out_parser = llama_rs_dup_string(params.parser);
        *out_generation_prompt = llama_rs_dup_string(params.generation_prompt);
        if (!params.thinking_start_tag.empty()) {
            *out_thinking_start_tag = llama_rs_dup_string(params.thinking_start_tag);
        }
        if (!params.thinking_end_tags.empty()) {
            *out_thinking_end_tag = llama_rs_dup_string(params.thinking_end_tags.front());
        }
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// Parse a model's raw output with the format descriptor returned by
// llama_rs_apply_chat_template_oai. Model-free and stateless: the PEG arena
// is rebuilt from its saved form on each call, so the two halves can run on
// different threads at different times. Returns the parsed message as
// OpenAI-compatible JSON ({"role","content","reasoning_content","tool_calls",...}).
extern "C" llama_rs_status llama_rs_chat_parse(
    const char * input,
    bool is_partial,
    int32_t format,
    const char * parser,
    const char * generation_prompt,
    char ** out_json) {
    if (!input || !out_json || format < 0 || format >= COMMON_CHAT_FORMAT_COUNT) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_json = nullptr;
    try {
        common_chat_parser_params params;
        params.format = (common_chat_format) format;
        // Always split reasoning out to reasoning_content — the caller
        // decides what to do with it, and inline think tags are exactly what
        // OpenAI-compatible clients cannot render.
        params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (generation_prompt && generation_prompt[0] != '\0') {
            params.generation_prompt = generation_prompt;
        }
        if (parser && parser[0] != '\0') {
            params.parser.load(parser);
        }

        const auto msg = common_chat_parse(input, is_partial, params);
        *out_json = llama_rs_dup_string(msg.to_json_oaicompat().dump());
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_apply_chat_template(
    const struct llama_model * model,
    const char * const * roles,
    const char * const * contents,
    size_t n_messages,
    bool add_generation_prompt,
    bool enable_thinking,
    bool * out_was_explicit,
    char ** out_prompt,
    char ** out_thinking_start_tag,
    char ** out_thinking_end_tag) {
    if (!model || !roles || !contents || !out_was_explicit || !out_prompt ||
        !out_thinking_start_tag || !out_thinking_end_tag) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_was_explicit = false;
    *out_prompt = nullptr;
    *out_thinking_start_tag = nullptr;
    *out_thinking_end_tag = nullptr;

    try {
        auto tmpls = common_chat_templates_init(model, /* chat_template_override */ "");
        *out_was_explicit = common_chat_templates_was_explicit(tmpls.get());

        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.enable_thinking = enable_thinking;
        inputs.messages.reserve(n_messages);
        for (size_t i = 0; i < n_messages; i++) {
            common_chat_msg msg;
            msg.role = roles[i] ? roles[i] : "";
            msg.content = contents[i] ? contents[i] : "";
            inputs.messages.push_back(std::move(msg));
        }

        const auto params = common_chat_templates_apply(tmpls.get(), inputs);

        *out_prompt = llama_rs_dup_string(params.prompt);
        if (!*out_prompt) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        if (!params.thinking_start_tag.empty()) {
            *out_thinking_start_tag = llama_rs_dup_string(params.thinking_start_tag);
        }
        if (!params.thinking_end_tags.empty()) {
            *out_thinking_end_tag = llama_rs_dup_string(params.thinking_end_tags.front());
        }
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    return static_cast<int>(common_fit_params(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        log_level));
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}
