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
//! How an answer is read, since 24-11-2026: with the model's own format, as
//! llama.cpp knows its family ([`AnswerReader`], the parser llama-server
//! reads every answer with), and with [`ThinkingSplitter`] for what the
//! format leaves in the answer and for the models whose GGUF has no
//! template. A new family's answers are then readable with a llama.cpp
//! bump, and nothing to write here: until then each needed its own case
//! (`<think>`, Gemma 4's channel, Inkling's blocks).
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
/// closes), as in `chat_template::strip_reasoning_blocks`. Inkling 1 and 2
/// write their turn as blocks, each opened by `<|message_model|>` and closed
/// by `<|end_message|>`: `<|content_thinking|>` is the reasoning,
/// `<|content_text|>` the answer (llama.cpp's `common/parsers/inkling.cpp`).
const BLOCKS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|channel>thought", "<channel|>"),
    ("<|message_model|><|content_thinking|>", "<|end_message|>"),
];

/// Inkling's block markers, which are special tokens and not part of the
/// answer: taken out of it wherever they come. Seen on the reference PC with
/// Inkling 2 Mini (23-11-2026), whose answers reached the chat as
/// `<|message_model|><|content_thinking|>…<|end_message|><|message_model|>
/// <|content_text|>Ciao!…<|end_message|>`.
const MARKERS: &[&str] = &[
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|end_message|>",
];

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

/// Reads an answer as it streams: with the model's own format, as llama.cpp
/// knows it ([`AnswerFormat`]), when the model's template has one, and with
/// [`ThinkingSplitter`] on what the format calls the answer, for what the
/// format leaves in it (Gemma 4's thought channel, in the parsers of some
/// llama.cpp versions). Without a format, or when the format does not read
/// the answer, [`ThinkingSplitter`] alone, as before 24-11-2026.
///
/// The format reads the answer so far again at each piece, as llama-server
/// does, and what it reads beyond the last reading goes out. A reading that
/// takes back text already sent (a parser that changed its mind) sends
/// nothing until it catches up.
pub(crate) struct AnswerReader {
    format: Option<FormatReading>,
    splitter: ThinkingSplitter,
}

struct FormatReading {
    format: crate::inference::AnswerFormat,
    /// What the engine puts before the answer (a reasoning block the prompt
    /// opened, `GenerateRequest::response_prefix`), which the format, built
    /// from that same prompt, does not expect.
    prefix: String,
    raw: String,
    /// What has gone out so far, by kind.
    thinking: String,
    content: String,
    /// The format has read the answer at least once.
    read_once: bool,
}

/// Bytes after which a format that has not read the answer once is given up
/// for the splitter: the model is not writing what its template says.
const FORMAT_GIVES_UP_AFTER: usize = 512;

impl AnswerReader {
    /// `prefix`: the response prefix the engine sends before the answer.
    pub(crate) fn new(format: Option<crate::inference::AnswerFormat>, prefix: &str) -> Self {
        Self {
            format: format.map(|format| FormatReading {
                format,
                prefix: prefix.to_string(),
                raw: String::new(),
                thinking: String::new(),
                content: String::new(),
                read_once: false,
            }),
            splitter: ThinkingSplitter::new(),
        }
    }

    /// Without a format: the splitter alone.
    pub(crate) fn plain() -> Self {
        Self::new(None, "")
    }

    pub(crate) fn push(&mut self, piece: &str) -> Vec<Part> {
        let Some(reading) = self.format.as_mut() else {
            return self.splitter.push(piece);
        };
        reading.raw.push_str(piece);
        match reading.read(true) {
            Some(parts) => self.through_splitter(parts),
            None if !reading.read_once && reading.raw.len() > FORMAT_GIVES_UP_AFTER => {
                tracing::debug!("the answer does not read in its template's format; plain reading");
                let raw = std::mem::take(&mut reading.raw);
                self.format = None;
                self.splitter.push(&raw)
            }
            None => Vec::new(),
        }
    }

    pub(crate) fn finish(mut self) -> Vec<Part> {
        let mut parts = match self.format.as_mut() {
            Some(reading) => match reading.read(false).or_else(|| reading.read(true)) {
                Some(parts) => self.through_splitter(parts),
                None if !reading.read_once => {
                    let raw = std::mem::take(&mut reading.raw);
                    self.splitter.push(&raw)
                }
                None => Vec::new(),
            },
            None => Vec::new(),
        };
        parts.extend(self.splitter.finish());
        parts
    }

    /// The format's reasoning goes out as it is; its answer through the
    /// splitter.
    fn through_splitter(&mut self, parts: Vec<Part>) -> Vec<Part> {
        let mut out = Vec::new();
        for part in parts {
            match part {
                Part::Thinking(text) => out.push(Part::Thinking(text)),
                Part::Content(text) => out.extend(self.splitter.push(&text)),
            }
        }
        out
    }
}

impl FormatReading {
    /// What reading the answer so far adds to what went out; `None` when the
    /// format cannot read it.
    fn read(&mut self, partial: bool) -> Option<Vec<Part>> {
        let text = match self.raw.strip_prefix(self.prefix.as_str()) {
            Some(rest) => rest,
            // The prefix is still arriving.
            None if self.prefix.starts_with(self.raw.as_str()) => return Some(Vec::new()),
            None => self.raw.as_str(),
        };
        let (thinking, content) = self.format.read(text, partial)?;
        self.read_once = true;
        let mut parts = Vec::new();
        if let Some(new) = thinking.strip_prefix(self.thinking.as_str()) {
            if !new.is_empty() {
                parts.push(Part::Thinking(new.to_string()));
            }
            self.thinking = thinking.clone();
        }
        if let Some(new) = content.strip_prefix(self.content.as_str()) {
            if !new.is_empty() {
                parts.push(Part::Content(new.to_string()));
            }
            self.content = content.clone();
        }
        Some(parts)
    }
}

/// A whole answer read by [`AnswerReader`]: its reasoning, when it had any,
/// and the answer.
pub(crate) fn read(
    format: Option<crate::inference::AnswerFormat>,
    prefix: &str,
    text: &str,
) -> (Option<String>, String) {
    let mut reader = AnswerReader::new(format, prefix);
    let mut parts = reader.push(text);
    parts.extend(reader.finish());
    joined(parts)
}

/// The answer with its reasoning in it, between `<think>` tags, as the
/// Ollama endpoints return it when the reasoning is not asked apart: the
/// tags the `<think>` families write, and every client that renders a
/// reasoning section reads, whatever markers the model used (Gemma 4's
/// channel, Inkling's blocks).
pub(crate) fn inline(thinking: Option<String>, content: String) -> String {
    match thinking {
        Some(thinking) => format!("<think>{thinking}</think>{content}"),
        None => content,
    }
}

/// [`inline`] as the answer streams: reasoning opens a `<think>` tag, the
/// answer after it closes it, and every part comes out as answer text.
#[derive(Debug, Default)]
pub(crate) struct InlineThinking {
    open: bool,
}

impl InlineThinking {
    pub(crate) fn map(&mut self, parts: Vec<Part>) -> Vec<Part> {
        parts
            .into_iter()
            .map(|part| match part {
                Part::Thinking(text) if !self.open => {
                    self.open = true;
                    Part::Content(format!("<think>{text}"))
                }
                Part::Thinking(text) => Part::Content(text),
                Part::Content(text) if self.open => {
                    self.open = false;
                    Part::Content(format!("</think>{text}"))
                }
                content => content,
            })
            .collect()
    }

    /// The end of the answer: a reasoning block it never left is closed.
    pub(crate) fn close(&mut self) -> Vec<Part> {
        if std::mem::take(&mut self.open) {
            vec![Part::Content("</think>".into())]
        } else {
            Vec::new()
        }
    }
}

/// Parts joined by kind.
pub(crate) fn joined(parts: Vec<Part>) -> (Option<String>, String) {
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
                    if !self.take_leading_whitespace() {
                        return;
                    }
                    // The start of a marker the next piece may complete stays
                    // back, so half of one never reaches the answer.
                    let keep = if end { 0 } else { marker_holdback(&self.held) };
                    let ready = self.held.len() - keep;
                    let text: String = self.held.drain(..ready).collect();
                    push(parts, Part::Content(without_markers(&text)));
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

/// `text` without Inkling's block markers.
fn without_markers(text: &str) -> String {
    if !text.contains("<|") {
        return text.to_string();
    }
    MARKERS
        .iter()
        .fold(text.to_string(), |text, marker| text.replace(marker, ""))
}

/// How many bytes at the end of `text` may be the start of a marker.
fn marker_holdback(text: &str) -> usize {
    MARKERS
        .iter()
        .filter_map(|marker| {
            (1..marker.len()).rev().find(|&k| {
                text.len() >= k && marker.as_bytes().starts_with(&text.as_bytes()[text.len() - k..])
            })
        })
        .max()
        .unwrap_or(0)
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

    /// Inkling 2 Mini's answer on the reference PC, as it reached the chat:
    /// its blocks become the reasoning and the answer, and no marker is left.
    #[test]
    fn inklings_blocks_become_the_thinking_and_the_answer() {
        let answer = "<|message_model|><|content_thinking|>The user greets me in Italian.<|end_message|>\
                      <|message_model|><|content_text|>Ciao! Mi chiamo Inkling.<|end_message|>";
        let expected = both("The user greets me in Italian.", "Ciao! Mi chiamo Inkling.");
        assert_eq!(split(answer), expected);
        for piece_len in [1, 3, 7] {
            assert_eq!(streamed(answer, piece_len), expected, "pieces of {piece_len}");
        }
        // Without reasoning, the answer block alone.
        let answer = "<|message_model|><|content_text|>Ciao!<|end_message|>";
        assert_eq!(split(answer), (None, "Ciao!".into()));
        assert_eq!(streamed(answer, 2), (None, "Ciao!".into()));
    }

    /// A marker-like text that is not one stays: only whole markers go.
    #[test]
    fn text_that_only_looks_like_a_marker_stays() {
        assert_eq!(split("a <|message b"), (None, "a <|message b".into()));
    }

    /// Not asked apart, the reasoning stays in the answer, in `<think>` tags
    /// whatever the model wrote, whole or streamed.
    #[test]
    fn reasoning_left_in_the_answer_is_in_think_tags() {
        assert_eq!(inline(Some("r".into()), "a".into()), "<think>r</think>a");
        assert_eq!(inline(None, "a".into()), "a");
        let mut inline = InlineThinking::default();
        let mut out = inline.map(vec![
            Part::Thinking("we".into()),
            Part::Thinking("igh".into()),
            Part::Content("Blue".into()),
            Part::Content(".".into()),
        ]);
        out.extend(inline.close());
        assert_eq!(joined(out), (None, "<think>weigh</think>Blue.".into()));
        // Reasoning the budget cut short is closed at the end.
        let mut inline = InlineThinking::default();
        let mut out = inline.map(vec![Part::Thinking("cut".into())]);
        out.extend(inline.close());
        assert_eq!(joined(out), (None, "<think>cut</think>".into()));
    }

    /// Without the model's format, the reader is the splitter, streamed or
    /// whole.
    #[test]
    fn the_reader_without_a_format_reads_as_the_splitter() {
        let answer = "<think>\nr\n</think>\n\nIt is 4.";
        assert_eq!(read(None, "", answer), split(answer));
        let mut reader = AnswerReader::plain();
        let mut parts = Vec::new();
        for piece in ["<th", "ink>r</th", "ink>It", " is 4."] {
            parts.extend(reader.push(piece));
        }
        parts.extend(reader.finish());
        assert_eq!(joined(parts), split(answer));
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
