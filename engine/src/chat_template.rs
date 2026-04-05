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
        Self::ChatML
    }

    /// Return the stop sequences for this template.
    pub fn stop_sequences(&self) -> Vec<String> {
        match self {
            Self::ChatML  => vec!["<|im_end|>".into()],
            Self::Gemma   => vec!["<end_of_turn>".into()],
            Self::Llama2  => vec!["[/INST]".into(), "</s>".into()],
        }
    }

    /// Build a full prompt from a slice of `(role, content)` pairs.
    ///
    /// `think` controls Qwen3 thinking mode (ChatML only):
    /// - `true`  → open assistant turn normally (thinking enabled)
    /// - `false` → inject `<think>\n</think>` to suppress thinking
    pub fn build_prompt(&self, messages: &[(&str, &str)], think: bool) -> String {
        match self {
            Self::ChatML => self.build_chatml(messages, think),
            Self::Gemma  => self.build_gemma(messages),
            Self::Llama2 => self.build_llama2(messages),
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
        if think {
            out.push_str("<|im_start|>assistant\n");
        } else {
            out.push_str("<|im_start|>assistant\n<think>\n</think>\n\n");
        }
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
                other       => other,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gemma() {
        assert_eq!(ChatTemplate::detect("gemma-4-e4b-it-Q4_K_M.gguf"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::detect("gemma2-9b-it.gguf"), ChatTemplate::Gemma);
    }

    #[test]
    fn test_detect_llama2() {
        assert_eq!(ChatTemplate::detect("llama-2-7b-chat.gguf"), ChatTemplate::Llama2);
        // Llama 3 should NOT match Llama2
        assert_eq!(ChatTemplate::detect("llama-3.1-8b-instruct.gguf"), ChatTemplate::ChatML);
    }

    #[test]
    fn test_detect_default_chatml() {
        assert_eq!(ChatTemplate::detect("qwen3-14b-q4_k_m.gguf"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::detect("mistral-7b-instruct.gguf"), ChatTemplate::ChatML);
        assert_eq!(ChatTemplate::detect("phi-3-mini.gguf"), ChatTemplate::ChatML);
    }

    #[test]
    fn test_stop_sequences() {
        assert_eq!(ChatTemplate::ChatML.stop_sequences(), vec!["<|im_end|>"]);
        assert_eq!(ChatTemplate::Gemma.stop_sequences(),  vec!["<end_of_turn>"]);
    }

    #[test]
    fn test_gemma_role_mapping() {
        let msgs = vec![
            ("system",    "You are helpful."),
            ("user",      "Hello"),
            ("assistant", "Hi there!"),
            ("user",      "How are you?"),
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
        assert!(prompt.contains("<think>\n</think>"));
    }
}
