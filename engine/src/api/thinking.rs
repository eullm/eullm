//! Ollama's `think`: a reasoning model's thinking, returned apart from its
//! answer.
//!
//! The models EuLLM serves write their reasoning into the answer itself,
//! between delimiters: `<think>`…`</think>` (Qwen3, QwQ, DeepSeek-R1 and its
//! distills) or `<|channel>thought`…`<channel|>` (Gemma 4). Ollama, asked to
//! think, returns that reasoning in `message.thinking` (`thinking` on
//! `/api/generate`) and only the answer in `content`; a client written for it
//! found no `thinking` here, and the reasoning mixed into the answer. With
//! `think: true` the Ollama endpoints now take the answer apart, as it
//! streams, with [`ThinkingSplitter`].
//!
//! Only when asked, on the Ollama endpoints. Without `think`, the reasoning
//! stays in the answer with its tags, as it always has, for the clients that
//! render it from there.
//!
//! `/v1/chat/completions` always takes it apart, into `reasoning_content`
//! (`delta.reasoning_content` as it streams), as llama-server does. It kept
//! it in `content` until 19-11-2026, when Gemma 4 on Leonardo answered 13
//! requests of 13 with `<|channel>thought…` at the head of `content`, 3,712
//! characters on average against llama-server's 650 of answer alone: a client
//! of the OpenAI API reads `content` as the answer. Whether the model reasons
//! at all is [`openai_think`].

use serde_json::Value;

/// The delimiters of a reasoning block, opening and closing, as the models
/// write them. Gemma 4's are asymmetric (`<|channel>` opens, `<channel|>`
/// closes), as in `chat_template::strip_reasoning_blocks`.
const BLOCKS: &[(&str, &str)] = &[("<think>", "</think>"), ("<|channel>thought", "<channel|>")];

/// Whether a request asks for the reasoning apart: `think: true`, or a level
/// name (`"high"`…), which Ollama takes for the models that have levels.
/// `false`, `null` and no field at all leave the answer as the model wrote it.
pub(crate) fn wants_thinking(body: &Value) -> bool {
    match body.get("think") {
        Some(Value::Bool(think)) => *think,
        Some(Value::String(level)) => !level.is_empty(),
        _ => false,
    }
}

/// Whether a `/v1/chat/completions` request lets the model reason, from the
/// three ways clients say it: Ollama's `think` (as on the Ollama endpoints),
/// llama-server's `chat_template_kwargs.enable_thinking`, which many OpenAI
/// clients send, and OpenAI's `reasoning_effort`, where `"none"` turns it
/// off. The first present decides, in that order; with none, the model
/// reasons, as llama-server's default lets it. Only `think` was read until
/// 19-11-2026: a request sending `enable_thinking: false` got the reasoning
/// all the same.
pub(crate) fn openai_think(body: &Value) -> bool {
    match body.get("think") {
        Some(Value::Bool(think)) => return *think,
        Some(Value::String(level)) => return !level.is_empty() && level != "none",
        _ => {}
    }
    if let Some(enabled) = body
        .pointer("/chat_template_kwargs/enable_thinking")
        .and_then(Value::as_bool)
    {
        return enabled;
    }
    if let Some(effort) = body.get("reasoning_effort").and_then(Value::as_str) {
        return effort != "none";
    }
    true
}

/// A whole answer taken apart: its reasoning, when it had any, and the rest.
pub(crate) fn split(text: &str) -> (Option<String>, String) {
    let mut splitter = ThinkingSplitter::new();
    let mut parts = splitter.push(text);
    parts.extend(splitter.finish());
    let mut thinking = String::new();
    let mut content = String::new();
    for part in parts {
        match part {
            Part::Thinking(text) => thinking.push_str(&text),
            Part::Content(text) => content.push_str(&text),
        }
    }
    ((!thinking.is_empty()).then_some(thinking), content)
}

/// A piece of an answer, once told apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Part {
    /// Reasoning, without its delimiters.
    Thinking(String),
    /// The answer.
    Content(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nothing yet but whitespace, or the start of a delimiter: the answer
    /// may still open with a reasoning block.
    Start,
    /// Inside a reasoning block, until `close`.
    Thinking { close: &'static str },
    /// After the reasoning, or in an answer that opened without any.
    Content,
}

/// Takes a streamed answer apart into reasoning and answer, piece by piece.
///
/// A reasoning block counts only at the start of the answer, where the
/// models put it; a `<think>` written later is part of the answer, quoted.
/// Text that may still turn out to be a delimiter is held back until the
/// next piece decides, so a delimiter split across two tokens never leaks
/// half of itself into either side. The whitespace around the reasoning
/// belongs to neither: a model opens its reasoning with a newline and leaves
/// a blank line before its answer.
#[derive(Debug)]
pub(crate) struct ThinkingSplitter {
    state: State,
    /// Received but not yet handed out.
    held: String,
    /// Drop whitespace until the current part's first other character.
    trim_start: bool,
}

impl ThinkingSplitter {
    pub(crate) fn new() -> Self {
        Self {
            state: State::Start,
            held: String::new(),
            trim_start: false,
        }
    }

    /// Feed one streamed piece; returns the parts it completes, in order.
    pub(crate) fn push(&mut self, piece: &str) -> Vec<Part> {
        self.held.push_str(piece);
        let mut parts = Vec::new();
        self.drain(&mut parts, false);
        parts
    }

    /// The end of the answer: what is still held, as what it is so far. A
    /// reasoning block the answer never closed (it ran out of tokens) is
    /// reasoning all the same.
    pub(crate) fn finish(mut self) -> Vec<Part> {
        let mut parts = Vec::new();
        self.drain(&mut parts, true);
        parts
    }

    fn drain(&mut self, parts: &mut Vec<Part>, end: bool) {
        loop {
            match self.state {
                State::Start => {
                    let text = self.held.trim_start();
                    if let Some(&(open, close)) =
                        BLOCKS.iter().find(|(open, _)| text.starts_with(open))
                    {
                        self.held = text[open.len()..].to_string();
                        self.state = State::Thinking { close };
                        self.trim_start = true;
                        continue;
                    }
                    if !end && BLOCKS.iter().any(|(open, _)| open.starts_with(text)) {
                        return;
                    }
                    // No reasoning block: the answer as it came, leading
                    // whitespace included.
                    self.state = State::Content;
                }
                State::Thinking { close } => {
                    if !self.take_leading_whitespace() {
                        return;
                    }
                    if let Some(at) = self.held.find(close) {
                        let rest = self.held[at + close.len()..].to_string();
                        self.held.truncate(at);
                        push(parts, Part::Thinking(self.held.trim_end().to_string()));
                        self.held = rest;
                        self.state = State::Content;
                        self.trim_start = true;
                        continue;
                    }
                    let keep = if end { 0 } else { holdback(&self.held, close) };
                    let ready = self.held.len() - keep;
                    let text: String = self.held.drain(..ready).collect();
                    push(
                        parts,
                        Part::Thinking(if end {
                            text.trim_end().to_string()
                        } else {
                            text
                        }),
                    );
                    return;
                }
                State::Content => {
                    if self.take_leading_whitespace() {
                        push(parts, Part::Content(std::mem::take(&mut self.held)));
                    }
                    return;
                }
            }
        }
    }

    /// Drop the whitespace a part opens with. False while nothing else has
    /// arrived yet, so the caller waits for more.
    fn take_leading_whitespace(&mut self) -> bool {
        if !self.trim_start {
            return true;
        }
        let trimmed = self.held.trim_start();
        if trimmed.is_empty() {
            self.held.clear();
            return false;
        }
        self.held = trimmed.to_string();
        self.trim_start = false;
        true
    }
}

fn push(parts: &mut Vec<Part>, part: Part) {
    let empty = match &part {
        Part::Thinking(text) | Part::Content(text) => text.is_empty(),
    };
    if !empty {
        parts.push(part);
    }
}

/// How many bytes at the end of `text` to keep back: the start of `close`
/// that the next piece may complete, and the whitespace before it, which is
/// dropped if `close` follows.
fn holdback(text: &str, close: &str) -> usize {
    let partial = (1..close.len())
        .rev()
        .find(|&k| {
            text.len() >= k
                && close
                    .as_bytes()
                    .starts_with(&text.as_bytes()[text.len() - k..])
        })
        .unwrap_or(0);
    let before = &text[..text.len() - partial];
    partial + (before.len() - before.trim_end().len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Feed `text` a few bytes at a time, at character boundaries, and join
    /// the parts by kind: what a stream of tokens that size would produce.
    fn streamed(text: &str, piece_len: usize) -> (Option<String>, String) {
        let mut splitter = ThinkingSplitter::new();
        let mut parts = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut at = piece_len.min(rest.len());
            while !rest.is_char_boundary(at) {
                at += 1;
            }
            parts.extend(splitter.push(&rest[..at]));
            rest = &rest[at..];
        }
        parts.extend(splitter.finish());
        let mut thinking = String::new();
        let mut content = String::new();
        for part in parts {
            match part {
                Part::Thinking(text) => thinking.push_str(&text),
                Part::Content(text) => content.push_str(&text),
            }
        }
        ((!thinking.is_empty()).then_some(thinking), content)
    }

    fn both(thinking: &str, content: &str) -> (Option<String>, String) {
        (Some(thinking.to_string()), content.to_string())
    }

    /// The answer DeepSeek-R1-Distill-14B gave on the RTX 5070 Ti, cut short:
    /// the reasoning goes to `thinking`, the answer after it to `content`,
    /// and neither keeps the delimiters or the blank lines around them.
    #[test]
    fn a_think_block_becomes_the_thinking() {
        let answer = "<think>\nPer risolvere 17 per 23, moltiplico.\n340 + 51 = 391.\n</think>\n\n**Soluzione:** 391";
        assert_eq!(
            split(answer),
            both(
                "Per risolvere 17 per 23, moltiplico.\n340 + 51 = 391.",
                "**Soluzione:** 391"
            )
        );
    }

    #[test]
    fn gemma_4s_thought_channel_becomes_the_thinking() {
        assert_eq!(
            split("<|channel>thought\nThe user asks for a colour.<channel|>Blue."),
            both("The user asks for a colour.", "Blue.")
        );
    }

    #[test]
    fn an_answer_without_reasoning_is_left_as_it_is() {
        assert_eq!(split("Just the answer."), (None, "Just the answer.".into()));
        assert_eq!(split("\n\nIndented."), (None, "\n\nIndented.".into()));
        assert_eq!(split(""), (None, String::new()));
    }

    /// The models write the block first. One quoted later in the answer is
    /// text the user asked about, not reasoning.
    #[test]
    fn a_block_later_in_the_answer_is_part_of_it() {
        let answer = "Wrap it in <think> and </think> tags.";
        assert_eq!(split(answer), (None, answer.into()));
        let answer = "<b>bold</b> <think>quoted</think>";
        assert_eq!(split(answer), (None, answer.into()));
    }

    #[test]
    fn whitespace_before_the_block_is_not_an_answer() {
        assert_eq!(split("\n <think>r</think>a"), both("r", "a"));
    }

    /// An answer cut off by its token budget mid-thought: all of it was
    /// reasoning, and there is no answer to report.
    #[test]
    fn an_unclosed_block_is_all_thinking() {
        assert_eq!(
            split("<think>\nstill weighing it \n"),
            both("still weighing it", "")
        );
        assert_eq!(split("<thi"), (None, "<thi".into()));
    }

    #[test]
    fn a_block_with_nothing_in_it_leaves_no_thinking() {
        assert_eq!(
            split("<think>\n\n</think>\n\nCiao!"),
            (None, "Ciao!".into())
        );
    }

    /// Token by token, every delimiter is split across pieces somewhere; the
    /// parts must join to what the whole answer splits into.
    #[test]
    fn a_streamed_answer_splits_as_the_whole_one_does() {
        let answers = [
            "<think>\nPer risolvere 17 per 23.\n</think>\n\n**Soluzione:** 391",
            "<|channel>thought\nplan it<channel|>Final answer.",
            "  <think>a < b, and </thin is not the end</think>  \n Done.",
            "Plain answer with <think> inside.",
            "<think>è così: perché? </think>Sì, è così.",
            "<think>unclosed reasoning ",
            "<",
            "",
        ];
        for answer in answers {
            for piece_len in 1..=9 {
                assert_eq!(
                    streamed(answer, piece_len),
                    split(answer),
                    "{answer:?} in pieces of {piece_len}"
                );
            }
        }
    }

    /// The reasoning streams as it comes: only what may still be the closing
    /// delimiter, and the whitespace before it, waits for the next piece.
    #[test]
    fn reasoning_is_handed_out_before_the_block_closes() {
        let mut splitter = ThinkingSplitter::new();
        assert_eq!(splitter.push("<thi"), vec![]);
        assert_eq!(
            splitter.push("nk>\nFirst"),
            vec![Part::Thinking("First".into())]
        );
        assert_eq!(
            splitter.push(" step </"),
            vec![Part::Thinking(" step".into())]
        );
        assert_eq!(splitter.push("think>\n\n"), vec![]);
        assert_eq!(
            splitter.push("Answer"),
            vec![Part::Content("Answer".into())]
        );
        assert_eq!(splitter.finish(), vec![]);
    }

    #[test]
    fn think_true_or_a_level_asks_for_the_thinking() {
        assert!(wants_thinking(&json!({ "think": true })));
        assert!(wants_thinking(&json!({ "think": "high" })));
        assert!(!wants_thinking(&json!({ "think": false })));
        assert!(!wants_thinking(&json!({ "think": null })));
        assert!(!wants_thinking(&json!({ "think": "" })));
        assert!(!wants_thinking(&json!({})));
    }

    /// `think` first, then llama-server's `enable_thinking`, then OpenAI's
    /// `reasoning_effort`; with none, the model reasons. The case of
    /// 19-11-2026: `enable_thinking: false` was ignored.
    #[test]
    fn an_openai_request_says_whether_to_reason_three_ways() {
        assert!(openai_think(&json!({})));
        assert!(!openai_think(
            &json!({ "chat_template_kwargs": { "enable_thinking": false } })
        ));
        assert!(openai_think(
            &json!({ "chat_template_kwargs": { "enable_thinking": true } })
        ));
        assert!(!openai_think(&json!({ "reasoning_effort": "none" })));
        assert!(openai_think(&json!({ "reasoning_effort": "low" })));
        assert!(!openai_think(&json!({ "think": false, "reasoning_effort": "high" })));
        assert!(openai_think(&json!({
            "think": true, "chat_template_kwargs": { "enable_thinking": false }
        })));
        assert!(!openai_think(&json!({
            "chat_template_kwargs": { "enable_thinking": false }, "reasoning_effort": "high"
        })));
        assert!(openai_think(&json!({ "think": "high" })));
        assert!(!openai_think(&json!({ "think": "none" })));
    }

    /// Gemma 4's reasoning, as Leonardo's answers carried it in `content`.
    #[test]
    fn gemma_4s_reasoning_comes_apart_from_its_answer() {
        assert_eq!(
            split("<|channel>thought\nThe user wants a summary.<channel|>The summary."),
            (
                Some("The user wants a summary.".to_string()),
                "The summary.".to_string()
            )
        );
        assert_eq!(split("No reasoning here."), (None, "No reasoning here.".to_string()));
    }
}
