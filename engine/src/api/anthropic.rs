//! Anthropic's Messages API (`POST /v1/messages`), for clients that speak only
//! that: Claude Code with `ANTHROPIC_BASE_URL` pointed at EuLLM, and the
//! Anthropic SDKs.
//!
//! A request is translated to the shape `/v1/chat/completions` takes and
//! answered there, so it gets everything that endpoint does: the model's own
//! chat template, tool calls parsed in the model's format, `"model": "auto"`
//! routing, the audit trail. The answer is translated back: a message with
//! `thinking`, `text` and `tool_use` blocks, or, streamed, the events of
//! Anthropic's SSE (`message_start`, `content_block_*`, `message_delta`,
//! `message_stop`). Like a streamed answer with tools on
//! `/v1/chat/completions`, a streamed one here is generated whole first and
//! then sent as events: tool calls exist once the output is parsed.
//!
//! Asked for by someone who wanted EuLLM in Claude Code (on instagram.com,
//! 27-11-2026). Claude Code needs `/v1/messages` and nothing else:
//! `/v1/messages/count_tokens` is optional (it estimates without it).
//!
//! Not translated: images and documents in a message (replaced by a note
//! saying so; `/v1/chat/completions` reads no images either), and
//! `stop_sequences`, which `/v1/chat/completions` does not take.

use serde_json::{Map, Value, json};

/// Where a Messages request does not translate: the message for a 400.
pub(crate) type Invalid = String;

/// A Messages request as `/v1/chat/completions` takes it, never streamed:
/// the answer is translated back whole (see the module documentation).
pub(crate) fn to_chat_completions(body: &Value) -> Result<Value, Invalid> {
    let Some(object) = body.as_object() else {
        return Err("the body is not a JSON object".to_string());
    };
    let messages_in = object
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "`messages` is missing or not an array".to_string())?;

    let mut messages = Vec::new();
    if let Some(system) = object.get("system") {
        let text = blocks_text(system, "system")?;
        if !text.is_empty() {
            messages.push(json!({ "role": "system", "content": text }));
        }
    }
    for (i, message) in messages_in.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{i}].role is missing"))?;
        let content = message
            .get("content")
            .ok_or_else(|| format!("messages[{i}].content is missing"))?;
        match role {
            "user" => user_messages(content, i, &mut messages)?,
            "assistant" => messages.push(assistant_message(content, i)?),
            "system" => system_message(content, i, &mut messages)?,
            other => {
                return Err(format!(
                    "messages[{i}].role is `{other}`: expected `user`, `assistant` or `system`"
                ));
            }
        }
    }

    let mut out = Map::new();
    if let Some(model) = object.get("model") {
        out.insert("model".into(), model.clone());
    }
    out.insert("messages".into(), Value::Array(messages));
    out.insert("stream".into(), Value::Bool(false));
    for key in ["max_tokens", "temperature", "top_p", "top_k"] {
        if let Some(value) = object.get(key).filter(|v| !v.is_null()) {
            out.insert(key.into(), value.clone());
        }
    }
    // Anthropic thinks only when asked (`thinking.type` `enabled` or
    // `adaptive`); /v1/chat/completions thinks unless told not to.
    let think = matches!(
        body.pointer("/thinking/type").and_then(Value::as_str),
        Some("enabled" | "adaptive")
    );
    out.insert("think".into(), Value::Bool(think));
    if let Some(tools) = object.get("tools").and_then(Value::as_array) {
        let tools: Vec<Value> = tools.iter().filter_map(function_tool).collect();
        if !tools.is_empty() {
            out.insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(choice) = object.get("tool_choice").and_then(tool_choice) {
        out.insert("tool_choice".into(), choice);
    }
    Ok(Value::Object(out))
}

/// The text of a `system` or of a `tool_result`'s `content`: a string, or
/// blocks whose text is joined; anything but text becomes a note.
fn blocks_text(content: &Value, at: &str) -> Result<String, Invalid> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => Ok(blocks
            .iter()
            .map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                Some(other) => omitted(other),
                None => String::new(),
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")),
        Value::Null => Ok(String::new()),
        _ => Err(format!("{at} is neither a string nor an array of blocks")),
    }
}

/// What stands in for a block this endpoint does not read.
fn omitted(kind: &str) -> String {
    format!("[{kind} omitted: this server does not read {kind} content]")
}

/// A `system` message among the turns, as Claude Code sends them: before the
/// first turn it joins the system prompt; later, it joins the user turn next
/// to it, since many chat templates take a system message at the start only
/// (Qwen's raise an error on one anywhere else).
fn system_message(content: &Value, i: usize, out: &mut Vec<Value>) -> Result<(), Invalid> {
    let text = blocks_text(content, &format!("messages[{i}].content"))?;
    if text.is_empty() {
        return Ok(());
    }
    if out.iter().all(|m| m["role"] == "system") {
        match out.first_mut() {
            Some(system) => {
                let joined = format!("{}\n\n{text}", system["content"].as_str().unwrap_or(""));
                system["content"] = Value::String(joined);
            }
            None => out.push(json!({ "role": "system", "content": text })),
        }
        return Ok(());
    }
    push_user_text(out, text);
    Ok(())
}

/// Text from the user, added to the user message before it when that is the
/// last one, so that two user messages never follow each other: a system
/// message turned to user text, then the turn itself.
fn push_user_text(out: &mut Vec<Value>, text: String) {
    if let Some(last) = out.last_mut()
        && last["role"] == "user"
        && let Some(before) = last["content"].as_str()
    {
        last["content"] = Value::String(format!("{before}\n\n{text}"));
        return;
    }
    out.push(json!({ "role": "user", "content": text }));
}

/// A user turn: its `tool_result` blocks become `tool` messages, first, as
/// the calls they answer came last; the rest one user message.
fn user_messages(content: &Value, i: usize, out: &mut Vec<Value>) -> Result<(), Invalid> {
    let blocks = match content {
        Value::String(text) => {
            push_user_text(out, text.clone());
            return Ok(());
        }
        Value::Array(blocks) => blocks,
        _ => {
            return Err(format!(
                "messages[{i}].content is neither a string nor an array"
            ));
        }
    };
    let mut text = Vec::new();
    for (j, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                text.push(
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                );
            }
            Some("tool_result") => {
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("messages[{i}].content[{j}].tool_use_id is missing"))?;
                let mut result = blocks_text(
                    block.get("content").unwrap_or(&Value::Null),
                    &format!("messages[{i}].content[{j}].content"),
                )?;
                if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                    result = format!("Error: {result}");
                }
                out.push(json!({ "role": "tool", "tool_call_id": id, "content": result }));
            }
            Some(other) => text.push(omitted(other)),
            None => return Err(format!("messages[{i}].content[{j}].type is missing")),
        }
    }
    if !text.is_empty() {
        push_user_text(out, text.join("\n\n"));
    }
    Ok(())
}

/// An assistant turn: its text, and its `tool_use` blocks as `tool_calls`.
/// Earlier thinking is left out, as chat templates leave it out of history.
fn assistant_message(content: &Value, i: usize) -> Result<Value, Invalid> {
    let blocks = match content {
        Value::String(text) => return Ok(json!({ "role": "assistant", "content": text })),
        Value::Array(blocks) => blocks,
        _ => {
            return Err(format!(
                "messages[{i}].content is neither a string nor an array"
            ));
        }
    };
    let mut text = Vec::new();
    let mut calls = Vec::new();
    for (j, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                text.push(
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                );
            }
            Some("tool_use") => {
                let field = |name: &str| {
                    block
                        .get(name)
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("messages[{i}].content[{j}].{name} is missing"))
                };
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                calls.push(json!({
                    "id": field("id")?,
                    "type": "function",
                    "function": { "name": field("name")?, "arguments": input.to_string() },
                }));
            }
            Some("thinking" | "redacted_thinking") => {}
            Some(other) => text.push(omitted(other)),
            None => return Err(format!("messages[{i}].content[{j}].type is missing")),
        }
    }
    let mut message = json!({ "role": "assistant", "content": text.join("\n\n") });
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    Ok(message)
}

/// A tool as `/v1/chat/completions` takes it. Anthropic's server tools (web
/// search and the like, which carry a `type` and no `input_schema`) are left
/// out: no model here runs them.
fn function_tool(tool: &Value) -> Option<Value> {
    let name = tool.get("name").and_then(Value::as_str)?;
    let parameters = tool.get("input_schema")?;
    let mut function = json!({ "name": name, "parameters": parameters });
    if let Some(description) = tool.get("description") {
        function["description"] = description.clone();
    }
    Some(json!({ "type": "function", "function": function }))
}

/// `tool_choice`: `auto`, `any` (some tool), `tool` (that one), `none`.
fn tool_choice(choice: &Value) -> Option<Value> {
    match choice.get("type").and_then(Value::as_str)? {
        "auto" => Some(json!("auto")),
        "any" => Some(json!("required")),
        "none" => Some(json!("none")),
        "tool" => {
            let name = choice.get("name").and_then(Value::as_str)?;
            Some(json!({ "type": "function", "function": { "name": name } }))
        }
        _ => None,
    }
}

/// A `/v1/chat/completions` answer as a Messages answer, for `model`.
pub(crate) fn from_chat_completion(answer: &Value, model: &str) -> Value {
    let message = answer.pointer("/choices/0/message").unwrap_or(&Value::Null);
    let mut content = Vec::new();
    if let Some(thinking) = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
    {
        content.push(json!({ "type": "thinking", "thinking": thinking, "signature": "" }));
    }
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
    {
        content.push(json!({ "type": "text", "text": text }));
    }
    let calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for call in calls {
        let arguments = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}");
        // Anthropic's `input` is an object; a model that wrote something else
        // gets it under `arguments`, which a client can still show.
        let input = match serde_json::from_str::<Value>(arguments) {
            Ok(object @ Value::Object(_)) => object,
            _ => json!({ "arguments": arguments }),
        };
        content.push(json!({
            "type": "tool_use",
            "id": call.get("id").and_then(Value::as_str).map_or_else(new_tool_id, str::to_string),
            "name": call.pointer("/function/name").and_then(Value::as_str).unwrap_or(""),
            "input": input,
        }));
    }
    let stop_reason = match answer
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
    {
        _ if !calls.is_empty() => "tool_use",
        Some("length") => "max_tokens",
        _ => "end_turn",
    };
    let tokens = |name: &str| {
        answer
            .pointer(&format!("/usage/{name}"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": tokens("prompt_tokens"),
            "output_tokens": tokens("completion_tokens"),
        },
    })
}

fn new_tool_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

/// A Messages answer as the events of Anthropic's streaming, `(event, data)`.
pub(crate) fn stream_events(message: &Value) -> Vec<(&'static str, Value)> {
    let mut start = message.clone();
    start["content"] = json!([]);
    start["stop_reason"] = Value::Null;
    start["usage"]["output_tokens"] = json!(0);
    let mut events = vec![(
        "message_start",
        json!({ "type": "message_start", "message": start }),
    )];
    let blocks = message
        .get("content")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for (index, block) in blocks.iter().enumerate() {
        let (empty, delta) = match block.get("type").and_then(Value::as_str) {
            Some("thinking") => (
                json!({ "type": "thinking", "thinking": "", "signature": "" }),
                json!({ "type": "thinking_delta", "thinking": block["thinking"] }),
            ),
            Some("tool_use") => (
                json!({ "type": "tool_use", "id": block["id"], "name": block["name"], "input": {} }),
                json!({ "type": "input_json_delta", "partial_json": block["input"].to_string() }),
            ),
            _ => (
                json!({ "type": "text", "text": "" }),
                json!({ "type": "text_delta", "text": block["text"] }),
            ),
        };
        events.push((
            "content_block_start",
            json!({ "type": "content_block_start", "index": index, "content_block": empty }),
        ));
        events.push((
            "content_block_delta",
            json!({ "type": "content_block_delta", "index": index, "delta": delta }),
        ));
        events.push((
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": index }),
        ));
    }
    events.push((
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": { "stop_reason": message["stop_reason"], "stop_sequence": null },
            "usage": { "output_tokens": message["usage"]["output_tokens"] },
        }),
    ));
    events.push(("message_stop", json!({ "type": "message_stop" })));
    events
}

/// An error as Anthropic sends it, for an HTTP `status` and a `message`.
pub(crate) fn error(status: u16, message: &str) -> Value {
    let kind = match status {
        400 | 413 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    };
    json!({ "type": "error", "error": { "type": kind, "message": message } })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_conversation_with_a_tool_call_and_its_result_translates() {
        let body = json!({
            "model": "qwen3.8-27b",
            "max_tokens": 1024,
            "system": [{ "type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"} }],
            "tools": [
                { "name": "Read", "description": "Read a file",
                  "input_schema": { "type": "object", "properties": { "path": { "type": "string" } } } },
                { "type": "web_search_20250305", "name": "web_search" },
            ],
            "tool_choice": { "type": "any" },
            "messages": [
                { "role": "user", "content": "What is in a.txt?" },
                { "role": "assistant", "content": [
                    { "type": "thinking", "thinking": "I should read it.", "signature": "x" },
                    { "type": "text", "text": "Reading it." },
                    { "type": "tool_use", "id": "toolu_1", "name": "Read", "input": { "path": "a.txt" } },
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "toolu_1", "content": [{ "type": "text", "text": "hello" }] },
                    { "type": "text", "text": "And?" },
                ]},
            ],
        });
        let out = to_chat_completions(&body).unwrap();
        assert_eq!(out["model"], "qwen3.8-27b");
        assert_eq!(out["max_tokens"], 1024);
        assert_eq!(out["stream"], false);
        assert_eq!(out["think"], false);
        assert_eq!(out["tool_choice"], "required");
        // The server tool is left out.
        assert_eq!(out["tools"].as_array().unwrap().len(), 1);
        assert_eq!(out["tools"][0]["function"]["name"], "Read");
        assert_eq!(out["tools"][0]["function"]["parameters"]["type"], "object");
        let messages = out["messages"].as_array().unwrap();
        assert_eq!(
            messages[0],
            json!({ "role": "system", "content": "You are terse." })
        );
        assert_eq!(
            messages[1],
            json!({ "role": "user", "content": "What is in a.txt?" })
        );
        assert_eq!(messages[2]["content"], "Reading it.");
        assert_eq!(messages[2]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            r#"{"path":"a.txt"}"#
        );
        assert_eq!(
            messages[3],
            json!({ "role": "tool", "tool_call_id": "toolu_1", "content": "hello" })
        );
        assert_eq!(messages[4], json!({ "role": "user", "content": "And?" }));
    }

    #[test]
    fn system_messages_among_the_turns_join_the_system_prompt_or_the_user_turn() {
        // Claude Code sends some of its instructions as `system` messages in
        // `messages` (27-11-2026: "messages[1].role is `system`").
        let out = to_chat_completions(&json!({
            "system": "S",
            "messages": [
                { "role": "system", "content": "T" },
                { "role": "user", "content": "a" },
                { "role": "system", "content": [{ "type": "text", "text": "reminder" }] },
                { "role": "user", "content": [{ "type": "text", "text": "b" }] },
                { "role": "assistant", "content": "ok" },
                { "role": "system", "content": "late" },
            ],
        }))
        .unwrap();
        assert_eq!(
            out["messages"],
            json!([
                { "role": "system", "content": "S\n\nT" },
                { "role": "user", "content": "a\n\nreminder\n\nb" },
                { "role": "assistant", "content": "ok" },
                { "role": "user", "content": "late" },
            ])
        );
    }

    #[test]
    fn thinking_only_when_asked_and_bad_requests_say_where() {
        let ask = |thinking: Value| {
            let body =
                json!({ "messages": [{ "role": "user", "content": "hi" }], "thinking": thinking });
            to_chat_completions(&body).unwrap()["think"].clone()
        };
        assert_eq!(
            ask(json!({ "type": "enabled", "budget_tokens": 2048 })),
            true
        );
        assert_eq!(ask(json!({ "type": "adaptive" })), true);
        assert_eq!(ask(json!({ "type": "disabled" })), false);
        assert_eq!(ask(Value::Null), false);

        let err = to_chat_completions(&json!({ "messages": [{ "role": "tool", "content": "x" }] }))
            .unwrap_err();
        assert!(err.contains("messages[0].role"), "{err}");
        assert!(
            to_chat_completions(&json!({ "model": "m" }))
                .unwrap_err()
                .contains("messages")
        );
        // An image becomes a note, not a silent gap.
        let out = to_chat_completions(&json!({ "messages": [{ "role": "user", "content": [
            { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AA==" } },
            { "type": "text", "text": "What is this?" },
        ]}]}))
        .unwrap();
        assert!(
            out["messages"][0]["content"]
                .as_str()
                .unwrap()
                .starts_with("[image omitted")
        );
    }

    #[test]
    fn an_answer_with_reasoning_text_and_a_tool_call_translates_back() {
        let answer = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Need the file.",
                    "content": "Let me look.",
                    "tool_calls": [{ "id": "call_7", "type": "function",
                                     "function": { "name": "Read", "arguments": "{\"path\":\"b.txt\"}" } }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": { "prompt_tokens": 120, "completion_tokens": 30 },
        });
        let message = from_chat_completion(&answer, "qwen3.8-27b");
        assert_eq!(message["type"], "message");
        assert_eq!(message["model"], "qwen3.8-27b");
        assert_eq!(message["stop_reason"], "tool_use");
        assert_eq!(
            message["usage"],
            json!({ "input_tokens": 120, "output_tokens": 30 })
        );
        let content = message["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(
            content[1],
            json!({ "type": "text", "text": "Let me look." })
        );
        assert_eq!(content[2]["type"], "tool_use");
        assert_eq!(content[2]["id"], "call_7");
        assert_eq!(content[2]["input"], json!({ "path": "b.txt" }));

        let cut = json!({ "choices": [{ "message": { "content": "Once upon" }, "finish_reason": "length" }] });
        assert_eq!(from_chat_completion(&cut, "m")["stop_reason"], "max_tokens");
    }

    #[test]
    fn a_streamed_answer_is_the_events_of_anthropic_streaming_in_order() {
        let answer = json!({
            "choices": [{
                "message": { "content": "Sure.", "tool_calls": [{ "id": "call_1",
                    "function": { "name": "Bash", "arguments": "{\"command\":\"ls\"}" } }] },
                "finish_reason": "tool_calls",
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 },
        });
        let events = stream_events(&from_chat_completion(&answer, "m"));
        let names: Vec<&str> = events.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(events[0].1["message"]["content"], json!([]));
        assert_eq!(
            events[1].1["content_block"],
            json!({ "type": "text", "text": "" })
        );
        assert_eq!(
            events[2].1["delta"],
            json!({ "type": "text_delta", "text": "Sure." })
        );
        assert_eq!(events[4].1["content_block"]["name"], "Bash");
        assert_eq!(
            events[5].1["delta"],
            json!({ "type": "input_json_delta", "partial_json": "{\"command\":\"ls\"}" })
        );
        assert_eq!(events[7].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(events[7].1["usage"]["output_tokens"], 5);
    }

    #[test]
    fn errors_carry_anthropic_types() {
        assert_eq!(error(400, "bad")["error"]["type"], "invalid_request_error");
        assert_eq!(error(503, "busy")["error"]["type"], "overloaded_error");
        assert_eq!(error(500, "x")["type"], "error");
    }
}
