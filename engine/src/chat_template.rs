/// Chat prompt templating for different model families.
///
/// Each model family uses a different prompt format. EULLM detects the
/// template from the model filename/name and applies it consistently in
/// both the interactive REPL and the API routes.
///
/// Supported templates:
/// - ChatML   — Qwen3, Mistral, Llama 3.1+, most modern models (default)
/// - Gemma    — Gemma 2 / 3 / 4
/// - Llama2   — Legacy Llama 2 [INST] format
///
/// To add a new template: add a variant to `ChatTemplate`, implement the
/// match arm in `build_prompt` / `stop_sequences`, and add a detection
/// rule in `detect`.

#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>`
    /// Used by: Qwen3, Mistral-Nemo, Llama 3.1+, Phi-3, most 2024+ models
    ChatML,

    /// Gemma: `<start_of_turn>role\ncontent<end_of_turn>`
    /// role mapping: system→system, user→user, assistant→model
    /// Used by: Gemma 2, Gemma 3, Gemma 4
    Gemma,

    /// Legacy Llama 2 [INST] format.
    /// Used by: Llama-2, CodeLlama (older checkpoints)
    Llama2,

    /// DeepSeek R1: `<｜User｜>content<｜Assistant｜>content<｜end▁of▁sentence｜>`,
    /// system prompt rendered bare before the first user turn. The delimiter
    /// characters are the fullwidth bars U+FF5C and the low line U+2581, not
    /// ASCII pipes — they are dedicated tokens in the R1 tokenizer and any
    /// ASCII approximation is just text.
    /// Used by: DeepSeek-R1 and its distills (Qwen/Llama based).
    ///
    /// These models are trained on this format and only this format. The
    /// ChatML fallback used to catch them, and the result was not degraded
    /// output but none: on a real request the model answered `<think>\n\n
    /// </think>\n\n` plus end-of-sentence — six tokens, empty visible
    /// content — deterministically. An off-distribution prompt does not make
    /// an R1 a little worse, it makes it decline the turn.
    DeepSeekR1,
}

impl ChatTemplate {
    /// Detect the chat template from a model name or file path.
    /// Falls back to ChatML when no pattern matches.
    pub fn detect(model_name: &str) -> Self {
        let lower = model_name.to_lowercase();
        if lower.contains("gemma") {
            return Self::Gemma;
        }
        if (lower.contains("llama-2") || lower.contains("llama2"))
            && !lower.contains("llama-3")
            && !lower.contains("llama3")
        {
            return Self::Llama2;
        }
        if lower.contains("deepseek-r1")
            || lower.contains("deepseek_r1")
            || lower.contains("r1-distill")
        {
            return Self::DeepSeekR1;
        }
        Self::ChatML
    }

    /// Return the stop sequences for this template.
    pub fn stop_sequences(&self) -> Vec<String> {
        match self {
            Self::ChatML => vec!["<|im_end|>".into()],
            // The closing-tag spellings are not in Gemma's vocabulary: the
            // model writes them as ordinary text, so they are never the EOG
            // token and generation runs straight past them. Observed at the
            // end of a real transcription:
            //
            //     … Ok? Ecco. Ciao.\n</start_of_turn>
            //
            // Listing them as stop sequences both ends the turn and keeps them
            // out of the reply, since the hold-back buffer withholds any
            // prefix of a stop sequence until it is known not to be one.
            Self::Gemma => vec![
                "<end_of_turn>".into(),
                "</end_of_turn>".into(),
                "</start_of_turn>".into(),
            ],
            Self::Llama2 => vec!["[/INST]".into(), "</s>".into()],
            // The end-of-sentence spelling is the model's real EOG token, so
            // llama.cpp normally ends the turn before this string ever
            // matches. It is listed anyway for the same reason as Gemma's
            // closing tags: a model that writes it as plain text mid-stream
            // must still stop, and the hold-back buffer keeps it out of the
            // reply either way.
            Self::DeepSeekR1 => vec!["<｜end▁of▁sentence｜>".into()],
        }
    }

    /// Build a full prompt from a slice of `(role, content)` pairs.
    ///
    /// `think` controls Qwen3 thinking mode (ChatML only):
    /// - `true`  → open assistant turn normally (thinking enabled)
    /// - `false` → inject `think_suppression_prefix()` to suppress thinking
    ///
    /// IMPORTANT for callers storing the resulting turn into history for a
    /// future request: when `think` is `false`, the text this function
    /// injects becomes part of what the model actually decodes for this
    /// turn (it's sent as part of the prompt, right before generation). Any
    /// later reconstruction of this turn's content that omits it produces
    /// text that no longer matches what's really resident in that turn's KV
    /// cache — silently breaking prefix-based KV reuse on every subsequent
    /// turn that includes this one in its history (confirmed on real
    /// hardware: sticky `/no_think` degraded reuse to a small, unstable
    /// fraction; turning it off restored ~99% reuse with no other change).
    /// Prepend `think_suppression_prefix()` to the stored assistant content
    /// whenever `think` was `false` for that turn — see `interactive_chat`.
    pub fn build_prompt(&self, messages: &[(&str, &str)], think: bool) -> String {
        match self {
            Self::ChatML => self.build_chatml(messages, think),
            Self::Gemma => self.build_gemma(messages),
            Self::Llama2 => self.build_llama2(messages),
            Self::DeepSeekR1 => self.build_deepseek_r1(messages),
        }
    }

    /// The literal text `build_prompt` injects right after the assistant
    /// turn opener when `think=false`, to suppress thinking mode (ChatML
    /// only — empty for templates that don't support a `think` toggle).
    /// See `build_prompt`'s doc comment for why storing history correctly
    /// requires re-applying this, not just omitting it.
    ///
    /// The exact bytes matter and are not a style choice. Qwen3's own chat
    /// template, embedded in every Qwen3 GGUF under
    /// `tokenizer.chat_template`, emits this for `enable_thinking=false`:
    ///
    /// ```text
    /// {%- if enable_thinking is defined and enable_thinking is false %}
    ///     {{- '<think>\n\n</think>\n\n' }}
    /// {%- endif %}
    /// ```
    ///
    /// Note the blank line between the two tags. We used to inject a single
    /// newline, and that one missing byte was enough to put the prompt off
    /// distribution: the model answered by re-emitting the closing tag, so a
    /// `think: false` request streamed a literal `</think>` to the client as
    /// assistant text. Deterministic at `temperature: 0`, reproduced
    /// identically on ARM CPU and on x86 CUDA.
    pub fn think_suppression_prefix(&self) -> &'static str {
        match self {
            Self::ChatML => "<think>\n\n</think>\n\n",
            // R1-family models are always-reasoning: they never learned to
            // read a pre-closed think block as anything but malformed input,
            // so there is nothing to inject. The REPL already carries this
            // exact exception for them.
            Self::Gemma | Self::Llama2 | Self::DeepSeekR1 => "",
        }
    }

    // ── ChatML ──────────────────────────────────────────────────────────────

    fn build_chatml(&self, messages: &[(&str, &str)], think: bool) -> String {
        let mut out = String::new();
        for (role, content) in messages {
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            out.push_str(content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        if !think {
            out.push_str(self.think_suppression_prefix());
        }
        out
    }

    // ── DeepSeek R1 ─────────────────────────────────────────────────────────

    /// Official template (tokenizer_config.json of DeepSeek-R1-Distill-*):
    /// BOS, the system prompt bare, then `<｜User｜>content` /
    /// `<｜Assistant｜>content<｜end▁of▁sentence｜>` pairs, and the
    /// generation prompt `<｜Assistant｜>`.
    ///
    /// Two deliberate deviations from the letter of that file:
    ///
    /// - No `<｜begin▁of▁sentence｜>`: the engine tokenizes non-raw prompts
    ///   with `AddBos::Always`, so writing it here would double it.
    /// - No forced `<think>\n` after the final `<｜Assistant｜>`. DeepSeek's
    ///   template appends it so the model cannot skip reasoning; Ollama's
    ///   does not, and the model then emits `<think>` itself as its first
    ///   tokens. Following Ollama keeps the opening tag in the *output*,
    ///   which is what every client that renders reasoning sections keys on —
    ///   an answer that starts mid-think with no opening tag would read as
    ///   the model leaking its reasoning as plain text.
    ///
    /// Assistant history is stripped to the text after `</think>`, exactly as
    /// the official template does: re-feeding a previous turn's reasoning
    /// wastes context and the model was trained not to see it.
    fn build_deepseek_r1(&self, messages: &[(&str, &str)]) -> String {
        let mut out = String::new();
        for (role, content) in messages {
            match *role {
                "system" => {
                    out.push_str(content);
                }
                "assistant" => {
                    let visible = content
                        .rsplit_once("</think>")
                        .map_or(*content, |(_, after)| after);
                    out.push_str("<｜Assistant｜>");
                    out.push_str(visible.trim_start());
                    out.push_str("<｜end▁of▁sentence｜>");
                }
                _ => {
                    out.push_str("<｜User｜>");
                    out.push_str(content);
                }
            }
        }
        out.push_str("<｜Assistant｜>");
        out
    }

    // ── Gemma ───────────────────────────────────────────────────────────────

    fn build_gemma(&self, messages: &[(&str, &str)]) -> String {
        let mut out = String::new();
        // Gemma uses "model" instead of "assistant" and "user" for user turns.
        // System messages are output as a dedicated <start_of_turn>system turn
        // (supported in Gemma 3/4; older Gemma 2 merges it into the first user).
        for (role, content) in messages {
            let gemma_role = match *role {
                "assistant" => "model",
                other => other,
            };
            out.push_str("<start_of_turn>");
            out.push_str(gemma_role);
            out.push('\n');
            out.push_str(content);
            out.push_str("<end_of_turn>\n");
        }
        out.push_str("<start_of_turn>model\n");
        out
    }

    // ── Llama 2 ─────────────────────────────────────────────────────────────

    fn build_llama2(&self, messages: &[(&str, &str)]) -> String {
        // Llama 2 format: [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant
        // Multi-turn: [INST] user [/INST] assistant </s><s>[INST] ...
        let mut out = String::new();
        let mut system_content: Option<&str> = None;
        let mut first_user = true;

        for (role, content) in messages {
            match *role {
                "system" => {
                    system_content = Some(content);
                }
                "user" => {
                    if first_user {
                        out.push_str("<s>[INST] ");
                        if let Some(sys) = system_content.take() {
                            out.push_str("<<SYS>>\n");
                            out.push_str(sys);
                            out.push_str("\n<</SYS>>\n\n");
                        }
                        first_user = false;
                    } else {
                        out.push_str("<s>[INST] ");
                    }
                    out.push_str(content);
                    out.push_str(" [/INST]");
                }
                "assistant" => {
                    out.push(' ');
                    out.push_str(content);
                    out.push_str(" </s>");
                }
                _ => {}
            }
        }
        out
    }
}

/// Remove a finished assistant turn's reasoning blocks, leaving the answer.
///
/// Reasoning exists to produce the answer, not to remember it: feeding a
/// model's own thinking back as context on the next turn confuses reasoning
/// models and — the reason this matters here — burns context several times
/// faster than the answers do. On a model that spends a few hundred tokens
/// thinking per turn, a 4096-token window that should hold a long conversation
/// instead fills after a handful of exchanges and generation stops with
/// `truncated — out of context`.
///
/// The web UI has always done this (`ui/app.js`, `stripThink`); the terminal
/// chat did not, so the same conversation ran out of context far sooner there.
/// This is the Rust half of that same decision, kept deliberately literal —
/// plain substring scanning over the two delimiter pairs the engine already
/// knows (see `inference::DEFAULT_HARMONY_FILTERS`), no regex dependency, and
/// no attempt to be clever about nesting the models do not produce.
///
/// An unclosed opener (a turn cut off mid-thought by the context limit) drops
/// everything from that opener on: there is no answer after it to keep.
///
/// Returns the input unchanged if stripping would leave nothing — a turn that
/// was *only* reasoning still has to occupy its place in the history, and an
/// empty assistant message is worse than a verbose one. `app.js` makes the
/// same choice (`stripThink(t) || t`).
#[must_use]
pub fn strip_reasoning_blocks(text: &str) -> String {
    // Gemma 4's Harmony channel form first, then Qwen3's `<think>`, matching
    // the order `app.js` applies them. Note the asymmetric channel delimiters
    // (`<|channel>` opens, `<channel|>` closes) — not a typo.
    let stripped = strip_delimited(text, "<|channel>thought", "<channel|>");
    let stripped = strip_delimited(&stripped, "<think>", "</think>");
    let stripped = stripped.trim();

    if stripped.is_empty() {
        text.to_string()
    } else {
        stripped.to_string()
    }
}

/// Drop every `open`…`close` span, and any whitespace trailing a closed one.
/// An `open` with no matching `close` truncates the rest of the string.
fn strip_delimited(text: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    loop {
        let Some(start) = rest.find(open) else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..start]);

        let after_open = &rest[start + open.len()..];
        let Some(end) = after_open.find(close) else {
            // Unclosed: everything from the opener on is reasoning that never
            // reached an answer.
            return out;
        };
        rest = after_open[end + close.len()..].trim_start();
    }
}

#[cfg(test)]
mod reasoning_strip_tests {
    use super::strip_reasoning_blocks;

    #[test]
    fn a_qwen3_think_block_is_dropped_and_the_answer_kept() {
        let turn = "<think>\nThe user greets me. Answer warmly.\n</think>\n\nCiao! Come stai?";
        assert_eq!(strip_reasoning_blocks(turn), "Ciao! Come stai?");
    }

    #[test]
    fn a_gemma_channel_thought_is_dropped() {
        let turn = "<|channel>thought\nreasoning here<channel|>The answer.";
        assert_eq!(strip_reasoning_blocks(turn), "The answer.");
    }

    /// The case that motivated this: a turn cut off mid-thought by the context
    /// limit. There is no answer after the opener, so nothing is worth keeping
    /// from it — but the turn must not come back empty either.
    #[test]
    fn an_unclosed_think_block_falls_back_to_the_original() {
        let turn = "<think>\nStill reasoning when the context ran out";
        // Stripping leaves nothing, so the original is preserved rather than
        // storing an empty assistant turn.
        assert_eq!(strip_reasoning_blocks(turn), turn);
    }

    #[test]
    fn text_before_an_unclosed_opener_survives() {
        let turn = "Partial answer.<think>\nthen it ran out";
        assert_eq!(strip_reasoning_blocks(turn), "Partial answer.");
    }

    #[test]
    fn a_turn_with_no_reasoning_is_untouched() {
        let turn = "Just a plain answer, no reasoning at all.";
        assert_eq!(strip_reasoning_blocks(turn), turn);
    }

    #[test]
    fn several_blocks_in_one_turn_are_all_dropped() {
        let turn = "<think>a</think>First. <think>b</think>Second.";
        assert_eq!(strip_reasoning_blocks(turn), "First. Second.");
    }

    /// A `<think>` written by the *user* inside their own message is not
    /// reasoning the model produced, but this runs only on assistant turns, so
    /// the asymmetry is intentional and worth pinning: whatever is between the
    /// delimiters goes, wherever it sits.
    #[test]
    fn stripping_is_purely_positional_not_semantic() {
        let turn = "Answer <think>aside</think> continues.";
        assert_eq!(strip_reasoning_blocks(turn), "Answer continues.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gemma() {
        assert_eq!(
            ChatTemplate::detect("gemma-4-e4b-it-Q4_K_M.gguf"),
            ChatTemplate::Gemma
        );
        assert_eq!(
            ChatTemplate::detect("gemma2-9b-it.gguf"),
            ChatTemplate::Gemma
        );
    }

    #[test]
    fn test_detect_llama2() {
        assert_eq!(
            ChatTemplate::detect("llama-2-7b-chat.gguf"),
            ChatTemplate::Llama2
        );
        // Llama 3 should NOT match Llama2
        assert_eq!(
            ChatTemplate::detect("llama-3.1-8b-instruct.gguf"),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn test_detect_default_chatml() {
        assert_eq!(
            ChatTemplate::detect("qwen3-14b-q4_k_m.gguf"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::detect("mistral-7b-instruct.gguf"),
            ChatTemplate::ChatML
        );
        assert_eq!(
            ChatTemplate::detect("phi-3-mini.gguf"),
            ChatTemplate::ChatML
        );
    }

    #[test]
    fn r1_distills_get_the_deepseek_template_not_chatml() {
        // The store id of the model this was found on. ChatML was the
        // fallback, and on this family it produced an empty answer in six
        // tokens, deterministically.
        assert_eq!(
            ChatTemplate::detect("deepseek-r1-distill-14b"),
            ChatTemplate::DeepSeekR1
        );
        assert_eq!(
            ChatTemplate::detect("DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"),
            ChatTemplate::DeepSeekR1
        );
        // QwQ reasons too but is ChatML-trained: must not be caught.
        assert_eq!(ChatTemplate::detect("qwq-32b"), ChatTemplate::ChatML);
    }

    #[test]
    fn deepseek_prompt_has_the_shape_the_model_was_trained_on() {
        let msgs = vec![
            ("system", "Sei un assistente."),
            ("user", "ciao"),
            ("assistant", "salve"),
            ("user", "come va?"),
        ];
        let p = ChatTemplate::DeepSeekR1.build_prompt(&msgs, true);
        assert_eq!(
            p,
            "Sei un assistente.<｜User｜>ciao<｜Assistant｜>salve<｜end▁of▁sentence｜><｜User｜>come va?<｜Assistant｜>"
        );
        // No BOS in the text: the engine adds it at tokenization
        // (AddBos::Always), and a doubled BOS is exactly the class of
        // off-by-one this template exists to end.
        assert!(!p.contains("begin▁of▁sentence"));
    }

    #[test]
    fn deepseek_history_drops_the_reasoning_like_the_official_template() {
        // tokenizer_config.json: {% if '</think>' in content %}
        //   {% set content = content.split('</think>')[-1] %}
        let msgs = vec![
            ("user", "2+2?"),
            ("assistant", "<think>\nlet me count\n</think>\n\n4"),
            ("user", "e 3+3?"),
        ];
        let p = ChatTemplate::DeepSeekR1.build_prompt(&msgs, true);
        assert!(p.contains("<｜Assistant｜>4<｜end▁of▁sentence｜>"));
        assert!(!p.contains("let me count"));
    }

    #[test]
    fn deepseek_think_toggle_injects_nothing() {
        // R1-family models are always-reasoning: a pre-closed empty think
        // block is malformed input to them, not a switch. think=false must
        // therefore change nothing in the prompt.
        let msgs = vec![("user", "ciao")];
        assert_eq!(
            ChatTemplate::DeepSeekR1.build_prompt(&msgs, false),
            ChatTemplate::DeepSeekR1.build_prompt(&msgs, true)
        );
    }

    #[test]
    fn test_stop_sequences() {
        assert_eq!(ChatTemplate::ChatML.stop_sequences(), vec!["<|im_end|>"]);
        assert_eq!(
            ChatTemplate::Gemma.stop_sequences(),
            vec!["<end_of_turn>", "</end_of_turn>", "</start_of_turn>"]
        );
    }

    // The closing-tag form reached a user at the end of a transcription. It is
    // ordinary text to the model, so only the stop list keeps it out.
    #[test]
    fn gemma_stops_on_the_closing_tag_the_model_actually_writes() {
        let stops = ChatTemplate::Gemma.stop_sequences();
        assert!(stops.iter().any(|s| s == "</start_of_turn>"));
    }

    #[test]
    fn test_gemma_role_mapping() {
        let msgs = vec![
            ("system", "You are helpful."),
            ("user", "Hello"),
            ("assistant", "Hi there!"),
            ("user", "How are you?"),
        ];
        let prompt = ChatTemplate::Gemma.build_prompt(&msgs, false);
        assert!(prompt.contains("<start_of_turn>model\nHi there!"));
        assert!(prompt.contains("<start_of_turn>system\n"));
        assert!(!prompt.contains("<start_of_turn>assistant"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_chatml_no_think() {
        let msgs = vec![("user", "hello")];
        let prompt = ChatTemplate::ChatML.build_prompt(&msgs, false);
        // Byte-for-byte what Qwen3's own template emits for
        // enable_thinking=false, blank line included. A single \n here made
        // the model echo `</think>` back as visible assistant text.
        assert!(prompt.contains("<think>\n\n</think>\n\n"));
        assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_think_suppression_prefix_round_trips_into_build_prompt() {
        // What think_suppression_prefix() returns must be exactly the text
        // build_prompt actually injects when think=false — otherwise a
        // caller reconstructing history with it would still produce text
        // that doesn't match what was really resident in that turn's KV
        // cache (the bug this function exists to let callers avoid).
        let msgs = vec![("user", "hello")];
        let suppressed = ChatTemplate::ChatML.build_prompt(&msgs, false);
        let normal = ChatTemplate::ChatML.build_prompt(&msgs, true);
        assert_eq!(
            suppressed,
            format!(
                "{normal}{}",
                ChatTemplate::ChatML.think_suppression_prefix()
            )
        );
    }

    #[test]
    fn test_think_suppression_prefix_empty_for_non_chatml() {
        assert_eq!(ChatTemplate::Gemma.think_suppression_prefix(), "");
        assert_eq!(ChatTemplate::Llama2.think_suppression_prefix(), "");
    }
}
