//! Web tool — transparent URL fetch and content injection.
//!
//! When `--web` is enabled, the server scans user messages for URLs.
//! If found, it fetches the page, strips HTML to plain text, and injects
//! the content into the prompt before inference — with no model-side changes.
//!
//! Content budget is dynamic:
//!   available_tokens = ctx_size - prompt_tokens - RESPONSE_RESERVE
//!   budget_chars     = available_tokens * CHARS_PER_TOKEN
//!
//! If the fetched text fits in the budget, the full text is injected.
//! If it exceeds the budget, paragraphs are scored by keyword overlap
//! with the user's query and the top-scoring ones are selected (BM25-lite).
//!
//! **The URL is untrusted input.** It comes out of a user message, so on any
//! shared or containerised deployment it is attacker-controlled. Everything
//! about *what may be fetched* — scheme, host, redirect handling, body size,
//! content type — lives in [`guard`], which is where to look before changing
//! anything on this path.

pub mod guard;

/// Tokens reserved for the model's response (never consumed by injected content).
const RESPONSE_RESERVE: usize = 512;

/// Rough estimate: 1 token ≈ 4 UTF-8 chars (conservative for European languages).
const CHARS_PER_TOKEN: usize = 4;

/// Minimum paragraph length to consider for relevance scoring.
const MIN_CHUNK_CHARS: usize = 80;

// ── URL extraction ────────────────────────────────────────────────────────────

/// Extract all http/https URLs from a text string.
pub fn extract_urls(text: &str) -> Vec<String> {
    let mut urls = Vec::new();
    let mut search_from = 0;

    loop {
        // find() on string slices always returns byte offsets at char boundaries
        let http = text[search_from..].find("http://").map(|p| search_from + p);
        let https = text[search_from..]
            .find("https://")
            .map(|p| search_from + p);

        let start = match (http, https) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };

        let rest = &text[start..];
        let end = rest
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | ')' | ']'))
            .unwrap_or(rest.len());

        let url = rest[..end]
            .trim_end_matches(['.', ',', '!', '?'])
            .to_string();
        if !url.is_empty() {
            urls.push(url);
        }

        // Advance past this URL; if end==0 skip one char to avoid infinite loop
        search_from = start + end;
        if end == 0 {
            search_from += text[search_from..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(1);
        }
        if search_from >= text.len() {
            break;
        }
    }

    urls
}

// ── HTTP fetch ────────────────────────────────────────────────────────────────

/// Fetch a URL and return plain text content, stripped of HTML.
///
/// `policy` decides what may be fetched at all — see [`guard`]. Callers on the
/// request path should pass the policy resolved once at startup rather than
/// re-reading the environment per request.
pub async fn fetch_url(url: &str, policy: &guard::WebPolicy) -> Result<String, String> {
    let body = guard::fetch_text(url, policy).await?;
    Ok(html_to_text(&body))
}

// ── HTML → plain text ─────────────────────────────────────────────────────────

fn remove_block_tag(html: &str, tag: &str) -> String {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut result = String::with_capacity(html.len());
    let mut pos = 0;
    let lower = html.to_lowercase();
    loop {
        let start = match lower[pos..].find(&open) {
            Some(n) => pos + n,
            None => {
                result.push_str(&html[pos..]);
                break;
            }
        };
        result.push_str(&html[pos..start]);
        let end = match lower[start..].find(&close) {
            Some(n) => start + n + close.len(),
            None => html.len(),
        };
        pos = end;
    }
    result
}

/// Strip HTML to readable plain text: remove noisy blocks, tags, decode entities.
fn html_to_text(html: &str) -> String {
    // Remove entire noisy blocks (including their content)
    let mut text = remove_block_tag(html, "script");
    text = remove_block_tag(&text, "style");
    text = remove_block_tag(&text, "nav");
    text = remove_block_tag(&text, "footer");
    text = remove_block_tag(&text, "header");
    text = remove_block_tag(&text, "aside");
    text = remove_block_tag(&text, "noscript");

    // Block-level tags → newline before stripping
    for tag in &[
        "</p>",
        "</div>",
        "</li>",
        "</h1>",
        "</h2>",
        "</h3>",
        "</h4>",
        "</h5>",
        "<br>",
        "<br/>",
        "<br />",
        "</tr>",
        "</section>",
        "</article>",
    ] {
        text = text.replace(tag, "\n");
    }

    // Strip all remaining tags
    let mut stripped = String::with_capacity(text.len());
    let mut in_tag = false;
    for ch in text.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => stripped.push(ch),
            _ => {}
        }
    }

    // Decode common HTML entities
    let decoded = stripped
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–")
        .replace("&hellip;", "...")
        .replace("&euro;", "€")
        .replace("&copy;", "©");

    // Collapse whitespace: multiple spaces → one, 3+ newlines → 2
    let mut result = String::with_capacity(decoded.len());
    let mut consecutive_nl = 0usize;
    let mut last_was_space = false;
    for ch in decoded.chars() {
        if ch == '\n' || ch == '\r' {
            last_was_space = false;
            consecutive_nl += 1;
            if consecutive_nl <= 2 {
                result.push('\n');
            }
        } else if ch.is_whitespace() {
            consecutive_nl = 0;
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            consecutive_nl = 0;
            last_was_space = false;
            result.push(ch);
        }
    }

    result.trim().to_string()
}

// ── Relevance-based content selection ────────────────────────────────────────

/// Select the most relevant content from `text` to fit within `budget_chars`.
///
/// Paragraphs are scored by keyword overlap with the query and position bias
/// (earlier paragraphs score slightly higher — typical web page structure).
///
/// Returns `(selected_text, was_truncated)`.
pub fn select_relevant(text: &str, query: &str, budget_chars: usize) -> (String, bool) {
    if text.len() <= budget_chars {
        return (text.to_string(), false);
    }

    let chunks: Vec<&str> = text
        .split('\n')
        .map(str::trim)
        .filter(|s| s.len() >= MIN_CHUNK_CHARS)
        .collect();

    if chunks.is_empty() {
        return (text.chars().take(budget_chars).collect(), true);
    }

    let query_words: Vec<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 3)
        .collect();

    let n = chunks.len();
    let mut scored: Vec<(usize, f32)> = chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let lower = chunk.to_lowercase();
            let kw_score: f32 = query_words
                .iter()
                .map(|w| if lower.contains(w.as_str()) { 1.0 } else { 0.0 })
                .sum();
            // Slight position bias: first 20% of page gets +0.3
            let pos_bias = if i < n / 5 { 0.3 } else { 0.0 };
            (i, kw_score + pos_bias)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut selected: Vec<usize> = Vec::new();
    let mut used = 0usize;
    for (idx, _score) in &scored {
        let chunk_len = chunks[*idx].len();
        if used + chunk_len + 2 > budget_chars {
            continue;
        }
        selected.push(*idx);
        used += chunk_len + 2;
    }

    selected.sort_unstable();
    let result = selected
        .iter()
        .map(|&i| chunks[i])
        .collect::<Vec<_>>()
        .join("\n\n");
    (result, true)
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Fetch a URL and return context-ready text, truncated to fit the available budget.
///
/// - `ctx_size`:      total context window in tokens (from server config)
/// - `prompt_chars`:  chars already consumed by the conversation prompt
/// - `query`:         user's question — used for relevance scoring when truncating
///
/// Returns `(injected_text, was_truncated)`.
pub async fn fetch_for_context(
    url: &str,
    ctx_size: u32,
    prompt_chars: usize,
    query: &str,
    policy: &guard::WebPolicy,
) -> Result<(String, bool), String> {
    let text = fetch_url(url, policy).await?;

    let available_tokens =
        (ctx_size as usize).saturating_sub(prompt_chars / CHARS_PER_TOKEN + RESPONSE_RESERVE);
    let budget_chars = available_tokens * CHARS_PER_TOKEN;

    Ok(select_relevant(&text, query, budget_chars))
}
