//! Documents attached in the chat: their text, and as much of it as the
//! context holds.
//!
//! Two halves, used at two moments:
//!
//! - [`extract`] turns a file into plain text, once, when it is attached
//!   (`POST /api/documents/extract`): PDF, Word (`.docx`), LibreOffice
//!   (`.odt`, `.odp`, `.ods`), PowerPoint (`.pptx`), Excel (`.xlsx`, `.xls`)
//!   and anything that is text (`.txt`, `.md`, `.csv`, `.json`, `.html`,
//!   source code). The browser keeps the text with the turn it was attached
//!   to, as it keeps an image, and sends it again with every later turn.
//! - [`fit`] decides, on every turn, how much of each document goes into the
//!   prompt. A document that fits goes whole. When the documents together do
//!   not, each gets a share of the room left, and inside its share the
//!   passages that best match the newest question, in their own order, with
//!   a mark where text was left out. A follow-up question about another part
//!   of a long document therefore gets that part.
//!
//! **The file is untrusted input.** Anyone the API admits can send one, so
//! every format is read with limits: the bytes the browser may send
//! (`MAX_DOCUMENT_BYTES`), what an archive member may unpack to (a zip bomb
//! stops at `MAX_UNPACKED_BYTES`), the text kept (`MAX_TEXT_CHARS`), and a
//! PDF parser that panics on a malformed file is caught (`extract` runs it
//! under `catch_unwind`). Nothing is written to disk and nothing is kept
//! here: the text goes back to the browser that sent the file.

use std::io::{Cursor, Read};

/// The largest file accepted: under the 64 MiB request limit, with room for
/// the request around it.
pub const MAX_DOCUMENT_BYTES: usize = 48 * 1024 * 1024;

/// What one member of an Office archive may unpack to. A 300-page Word
/// document's XML is a few MB; this stops a zip bomb long before memory.
const MAX_UNPACKED_BYTES: u64 = 64 * 1024 * 1024;

/// Text kept from one document: about a million tokens, more than any
/// context holds, so nothing a model could read is lost to it.
pub const MAX_TEXT_CHARS: usize = 4_000_000;

/// Rows read from one sheet of a spreadsheet.
const MAX_SHEET_ROWS: usize = 20_000;

/// One document's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extracted {
    /// What it was read as: `pdf`, `docx`, `odt`, `pptx`, `xlsx`, `text`…
    pub kind: &'static str,
    pub text: String,
    /// PDF pages, presentation slides, spreadsheet sheets; None for the rest.
    pub parts: Option<usize>,
    /// The text was longer than `MAX_TEXT_CHARS` and was cut there.
    pub truncated: bool,
}

/// Read a file's text. `name` is the file's name, used only to tell formats
/// apart where the bytes do not (`.csv` from `.txt`, `.xls`).
pub fn extract(name: &str, bytes: &[u8]) -> Result<Extracted, String> {
    if bytes.is_empty() {
        return Err("the file is empty".into());
    }
    if bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(format!(
            "the file is {} MB; the largest accepted is {} MB",
            bytes.len() / (1024 * 1024),
            MAX_DOCUMENT_BYTES / (1024 * 1024)
        ));
    }
    let ext = extension(name);
    let (kind, text, parts) = if bytes.starts_with(b"%PDF") || ext == "pdf" {
        let (text, pages) = pdf(bytes)?;
        ("pdf", text, Some(pages))
    } else if bytes.starts_with(b"PK\x03\x04") {
        office(bytes, &ext)?
    } else if bytes.starts_with(&[0xD0, 0xCF, 0x11, 0xE0]) {
        // The old binary Office formats. Excel's calamine reads; Word's
        // `.doc` and PowerPoint's `.ppt` it does not.
        if ext == "xls" {
            let (text, sheets) = spreadsheet(bytes)?;
            ("xls", text, Some(sheets))
        } else {
            return Err(
                "this is an old Office file (.doc or .ppt); save it as .docx, .pptx or PDF \
                 and attach it again"
                    .into(),
            );
        }
    } else {
        let text = decode_text(bytes)
            .ok_or_else(|| "this file is neither text nor a format the chat reads".to_string())?;
        if matches!(ext.as_str(), "html" | "htm" | "xhtml") || looks_like_html(&text) {
            ("html", super::html_to_text(&text), None)
        } else {
            ("text", text, None)
        }
    };
    let text = tidy(&text);
    if text.trim().is_empty() {
        return Err(match kind {
            "pdf" => "no text was found in this PDF: it is probably a scan, made of images. \
                      Attach the pages as images, or convert it with OCR first"
                .into(),
            _ => "no text was found in this file".into(),
        });
    }
    let (text, truncated) = cap_chars(text, MAX_TEXT_CHARS);
    Ok(Extracted {
        kind,
        text,
        parts,
        truncated,
    })
}

fn extension(name: &str) -> String {
    std::path::Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}

/// A PDF's text, page by page. pdf-extract has panicked on malformed files
/// before; a panic here is the file's fault, and becomes its error.
fn pdf(bytes: &[u8]) -> Result<(String, usize), String> {
    let result = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem_by_pages(bytes));
    let pages = match result {
        Ok(Ok(pages)) => pages,
        Ok(Err(e)) => return Err(format!("this PDF could not be read: {e}")),
        Err(_) => return Err("this PDF could not be read: it is damaged or uses a form the reader does not handle".into()),
    };
    let count = pages.len();
    let mut text = String::new();
    for (i, page) in pages.iter().enumerate() {
        let page = page.trim();
        if page.is_empty() {
            continue;
        }
        if count > 1 {
            text.push_str(&format!("[Page {}]\n", i + 1));
        }
        text.push_str(page);
        text.push_str("\n\n");
    }
    Ok((text, count))
}

/// The Office formats are zip archives of XML: which one this is, from what
/// is inside.
fn office(bytes: &[u8], ext: &str) -> Result<(&'static str, String, Option<usize>), String> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| format!("this archive could not be opened: {e}"))?;
    let names: Vec<String> = zip.file_names().filter_map(|n| n.ok().map(|n| n.into_owned())).collect();
    let has = |n: &str| names.iter().any(|x| x == n);
    if has("word/document.xml") {
        let xml = member(&mut zip, "word/document.xml")?;
        return Ok(("docx", ooxml_text(&xml, Ooxml::Word), None));
    }
    if names.iter().any(|n| n.starts_with("ppt/slides/slide")) {
        let mut slides: Vec<(u32, String)> = names
            .iter()
            .filter_map(|n| {
                let number = n.strip_prefix("ppt/slides/slide")?.strip_suffix(".xml")?;
                Some((number.parse().ok()?, n.clone()))
            })
            .collect();
        slides.sort();
        let mut text = String::new();
        for (number, name) in &slides {
            let xml = member(&mut zip, name)?;
            let slide = ooxml_text(&xml, Ooxml::Slide);
            if !slide.trim().is_empty() {
                text.push_str(&format!("[Slide {number}]\n{}\n\n", slide.trim()));
            }
        }
        return Ok(("pptx", text, Some(slides.len())));
    }
    if has("xl/workbook.xml") || ext == "xlsx" || ext == "xlsm" {
        let (text, sheets) = spreadsheet(bytes)?;
        return Ok(("xlsx", text, Some(sheets)));
    }
    if has("content.xml") {
        let mimetype = member(&mut zip, "mimetype").unwrap_or_default();
        if mimetype.contains("spreadsheet") || ext == "ods" {
            let (text, sheets) = spreadsheet(bytes)?;
            return Ok(("ods", text, Some(sheets)));
        }
        let xml = member(&mut zip, "content.xml")?;
        let kind = if mimetype.contains("presentation") || ext == "odp" {
            "odp"
        } else {
            "odt"
        };
        return Ok((kind, odf_text(&xml), None));
    }
    Err("this archive is not a document the chat reads (.docx, .pptx, .xlsx, .odt, .odp, .ods)".into())
}

/// One archive member as text, read no further than `MAX_UNPACKED_BYTES`.
fn member(zip: &mut zip::ZipArchive<Cursor<&[u8]>>, name: &str) -> Result<String, String> {
    let file = zip
        .by_name(name)
        .map_err(|e| format!("{name} could not be read: {e}"))?;
    let mut raw = Vec::new();
    file.take(MAX_UNPACKED_BYTES + 1)
        .read_to_end(&mut raw)
        .map_err(|e| format!("{name} could not be read: {e}"))?;
    if raw.len() as u64 > MAX_UNPACKED_BYTES {
        return Err(format!(
            "{name} unpacks to more than {} MB",
            MAX_UNPACKED_BYTES / (1024 * 1024)
        ));
    }
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ooxml {
    Word,
    Slide,
}

/// The text of a Word document or a slide: `w:t`/`a:t` runs, a line per
/// paragraph, a tab or a line break where the document has one, and table
/// cells separated by ` | `.
fn ooxml_text(xml: &str, which: Ooxml) -> String {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut out = String::with_capacity(xml.len() / 4);
    let mut in_text = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.local_name().as_ref() {
                "t" => in_text = true,
                "tab" => out.push('\t'),
                "br" | "cr" => out.push('\n'),
                _ => {}
            },
            Ok(Event::Empty(e)) => match e.local_name().as_ref() {
                "tab" => out.push('\t'),
                "br" | "cr" => out.push('\n'),
                _ => {}
            },
            Ok(Event::End(e)) => match e.local_name().as_ref() {
                "t" => in_text = false,
                "p" => out.push('\n'),
                "tc" if which == Ooxml::Word => out.push_str(" | "),
                "tr" => out.push('\n'),
                _ => {}
            },
            Ok(Event::Text(t)) if in_text => out.push_str(&t.xml10_content()),
            Ok(Event::CData(t)) if in_text => out.push_str(&t.xml10_content()),
            Ok(Event::GeneralRef(r)) if in_text => push_reference(&mut out, &r),
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out
}

/// The text of an OpenDocument file's `content.xml`: paragraphs and
/// headings, `text:s` spaces, tabs and line breaks, table cells.
fn odf_text(xml: &str) -> String {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut out = String::with_capacity(xml.len() / 4);
    // Text sits inside `office:text` / `office:presentation`; outside it are
    // styles and declarations.
    let mut in_body = false;
    // Inside a paragraph or a heading: the whitespace between elements
    // elsewhere is the file's indentation, not text.
    let mut in_para = 0usize;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                if matches!(e.local_name().as_ref(), "text" | "presentation")
                    && e.name().as_ref().starts_with("office:")
                {
                    in_body = true;
                }
                if in_body && matches!(e.local_name().as_ref(), "p" | "h") {
                    in_para += 1;
                }
            }
            Ok(Event::Empty(e)) if in_body => match e.local_name().as_ref() {
                "s" => {
                    let n = e
                        .try_get_attribute("text:c")
                        .ok()
                        .flatten()
                        .and_then(|a| a.value.parse::<usize>().ok())
                        .unwrap_or(1);
                    out.extend(std::iter::repeat_n(' ', n.min(64)));
                }
                "tab" => out.push('\t'),
                "line-break" => out.push('\n'),
                _ => {}
            },
            Ok(Event::End(e)) if in_body => match e.local_name().as_ref() {
                "p" | "h" => {
                    in_para = in_para.saturating_sub(1);
                    out.push('\n');
                }
                "table-cell" => out.push_str(" | "),
                "table-row" => out.push('\n'),
                "text" | "presentation" if e.name().as_ref().starts_with("office:") => {
                    in_body = false;
                }
                _ => {}
            },
            Ok(Event::Text(t)) if in_para > 0 => out.push_str(&t.xml10_content()),
            Ok(Event::GeneralRef(r)) if in_para > 0 => push_reference(&mut out, &r),
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out
}

/// `&amp;`, `&#233;` and the like, as their characters.
fn push_reference(out: &mut String, r: &quick_xml::events::BytesRef<'_>) {
    if let Ok(Some(c)) = r.resolve_char_ref() {
        out.push(c);
        return;
    }
    match r.xml10_content().as_ref() {
        "amp" => out.push('&'),
        "lt" => out.push('<'),
        "gt" => out.push('>'),
        "quot" => out.push('"'),
        "apos" => out.push('\''),
        _ => {}
    }
}

/// A spreadsheet's sheets, each a heading and its rows as a Markdown table:
/// what models read best (a 0.6B model counted the rows of a tab-separated
/// table as the answer, and read the same table right in Markdown).
fn spreadsheet(bytes: &[u8]) -> Result<(String, usize), String> {
    use calamine::Reader;
    let mut book = calamine::open_workbook_auto_from_rs(Cursor::new(bytes))
        .map_err(|e| format!("this spreadsheet could not be read: {e}"))?;
    let names = book.sheet_names().to_vec();
    let mut text = String::new();
    for name in &names {
        let Ok(range) = book.worksheet_range(name) else {
            continue;
        };
        if range.is_empty() {
            continue;
        }
        text.push_str(&format!("[Sheet {name}]\n"));
        let mut header_done = false;
        for (i, row) in range.rows().enumerate() {
            if i == MAX_SHEET_ROWS {
                text.push_str(&format!(
                    "[… {} more rows not read]\n",
                    range.height() - MAX_SHEET_ROWS
                ));
                break;
            }
            let cells: Vec<String> = row
                .iter()
                .map(|c| c.to_string().replace('|', "\\|").replace('\n', " ").trim().to_string())
                .collect();
            if cells.iter().all(String::is_empty) {
                continue;
            }
            text.push_str("| ");
            text.push_str(&cells.join(" | "));
            text.push_str(" |\n");
            if !header_done {
                // The first row read as the table's header.
                text.push_str(&format!("|{}\n", " --- |".repeat(cells.len())));
                header_done = true;
            }
        }
        text.push('\n');
        if text.len() > MAX_TEXT_CHARS {
            break;
        }
    }
    Ok((text, names.len()))
}

/// Text in UTF-8 (with or without BOM), UTF-16 with a BOM, or else a
/// Windows code page, which is what a text file from an Italian or German
/// Windows machine is. None for binary data: NUL bytes in the first few KB.
fn decode_text(bytes: &[u8]) -> Option<String> {
    if let Some((encoding, bom)) = encoding_rs::Encoding::for_bom(bytes) {
        let (text, _) = encoding.decode_without_bom_handling(&bytes[bom..]);
        return Some(text.into_owned());
    }
    let head = &bytes[..bytes.len().min(8192)];
    if head.contains(&0) {
        return None;
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => Some(s.to_string()),
        Err(_) => {
            let (text, _, _) = encoding_rs::WINDOWS_1252.decode(bytes);
            Some(text.into_owned())
        }
    }
}

fn looks_like_html(text: &str) -> bool {
    let head: String = text.chars().take(512).collect::<String>().to_ascii_lowercase();
    let head = head.trim_start();
    head.starts_with("<!doctype html") || head.starts_with("<html")
}

/// Line endings made `\n`, trailing spaces gone, runs of blank lines made one.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.replace("\r\n", "\n").replace('\r', "\n").split('\n') {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

fn cap_chars(text: String, max: usize) -> (String, bool) {
    match text.char_indices().nth(max) {
        Some((at, _)) => (text[..at].to_string(), true),
        None => (text, false),
    }
}

// ── How much of each document goes into the prompt ──────────────────────────

/// What a passage of a document long enough to be cut is: about a page of
/// a book. Small enough that the passages chosen are about the question,
/// large enough that each still reads as text.
const PASSAGE_CHARS: usize = 1200;

/// Where text was left out between two passages.
pub const OMITTED: &str = "[…]";

/// Each document's text as it goes into the prompt, in the order given, and
/// whether it was cut. `budget_chars` is the room for all of them.
///
/// Room is shared so that a short document goes whole and a long one gets
/// what the short ones leave: each document in turn, shortest first, gets an
/// even share of what is still left, or all of itself if that is less.
pub fn fit(documents: &[&str], query: &str, budget_chars: usize) -> Vec<(String, bool)> {
    let lens: Vec<usize> = documents.iter().map(|d| d.chars().count()).collect();
    let mut shares = vec![0usize; documents.len()];
    let mut order: Vec<usize> = (0..documents.len()).collect();
    order.sort_by_key(|&i| lens[i]);
    let mut left = budget_chars;
    for (done, &i) in order.iter().enumerate() {
        let remaining = documents.len() - done;
        let even = left / remaining;
        shares[i] = lens[i].min(even);
        left -= shares[i];
    }
    documents
        .iter()
        .zip(&shares)
        .zip(&lens)
        .map(|((doc, &share), &len)| {
            if share >= len {
                (doc.to_string(), false)
            } else {
                (select_passages(doc, query, share), true)
            }
        })
        .collect()
}

/// Passages of `text` that best match `query`, as many as `budget_chars`
/// holds, in the document's own order, joined by [`OMITTED`] where text
/// between them was left out.
///
/// Scored with BM25 over the document's own passages: a word of the
/// question that is rare in the document (a name, an article number) counts
/// for much more than one found everywhere. The first passage, where a
/// document says what it is, is kept whenever there is room for it.
pub fn select_passages(text: &str, query: &str, budget_chars: usize) -> String {
    if budget_chars == 0 {
        return String::new();
    }
    let passages = split_passages(text, PASSAGE_CHARS);
    if passages.is_empty() {
        return String::new();
    }
    let terms = words(query);
    let docs: Vec<Vec<String>> = passages.iter().map(|p| words(p)).collect();
    let n = docs.len() as f64;
    let avg_len = docs.iter().map(Vec::len).sum::<usize>() as f64 / n.max(1.0);
    let mut unique_terms = terms.clone();
    unique_terms.sort();
    unique_terms.dedup();
    let idf: Vec<f64> = unique_terms
        .iter()
        .map(|t| {
            let df = docs.iter().filter(|d| d.contains(t)).count() as f64;
            ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
        })
        .collect();
    let (k1, b) = (1.2, 0.75);
    let mut scored: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let len = d.len() as f64;
            let score: f64 = unique_terms
                .iter()
                .zip(&idf)
                .map(|(t, idf)| {
                    let tf = d.iter().filter(|w| *w == t).count() as f64;
                    if tf == 0.0 {
                        0.0
                    } else {
                        idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / avg_len.max(1.0)))
                    }
                })
                .sum();
            // The opening passage: a small lift, enough to win among equals.
            (i, score + if i == 0 { 0.5 } else { 0.0 })
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));

    let mut chosen: Vec<usize> = Vec::new();
    let mut used = 0usize;
    for (i, _) in &scored {
        let cost = passages[*i].chars().count() + OMITTED.len() + 2;
        if used + cost > budget_chars {
            continue;
        }
        used += cost;
        chosen.push(*i);
    }
    if chosen.is_empty() {
        // Not even one passage fits: the start of the best one.
        let best = &passages[scored[0].0];
        return cap_chars(best.to_string(), budget_chars.saturating_sub(OMITTED.len() + 1)).0
            + " "
            + OMITTED;
    }
    chosen.sort_unstable();
    let mut out = String::new();
    let mut next = 0;
    for i in chosen {
        if i != next {
            out.push_str(OMITTED);
            out.push_str("\n\n");
        }
        out.push_str(&passages[i]);
        out.push_str("\n\n");
        next = i + 1;
    }
    if next < passages.len() {
        out.push_str(OMITTED);
    }
    out.trim_end().to_string()
}

/// The text cut into passages of about `size` characters: whole lines
/// together (a paragraph of a Word file, a wrapped line of a PDF), ending
/// at a blank line where one is near, and a line longer than a passage cut
/// at the end of a sentence, else at a space.
fn split_passages(text: &str, size: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;
    let flush = |current: &mut String, current_chars: &mut usize, out: &mut Vec<String>| {
        let passage = current.trim();
        if !passage.is_empty() {
            out.push(passage.to_string());
        }
        current.clear();
        *current_chars = 0;
    };
    for line in text.split('\n') {
        let line = line.trim_end();
        let n = line.chars().count();
        if line.trim().is_empty() {
            // A paragraph ends here: a good place to end a passage that is
            // most of the way to its size.
            if current_chars * 10 >= size * 6 {
                flush(&mut current, &mut current_chars, &mut out);
            } else if current_chars > 0 {
                current.push('\n');
                current_chars += 1;
            }
            continue;
        }
        if current_chars > 0 && current_chars + n + 1 > size {
            flush(&mut current, &mut current_chars, &mut out);
        }
        if n <= size {
            if current_chars > 0 {
                current.push('\n');
                current_chars += 1;
            }
            current.push_str(line);
            current_chars += n;
            continue;
        }
        // A line longer than a passage: cut it at sentence ends.
        let mut rest = line.trim();
        while rest.chars().count() > size {
            let limit = rest.char_indices().nth(size).map(|(i, _)| i).unwrap_or(rest.len());
            let window = &rest[..limit];
            let cut = window
                .rfind(". ")
                .map(|i| i + 1)
                .filter(|&i| i > limit / 3)
                .or_else(|| window.rfind(char::is_whitespace).filter(|&i| i > limit / 3))
                .unwrap_or(limit);
            out.push(rest[..cut].trim().to_string());
            rest = rest[cut..].trim_start();
        }
        current.push_str(rest);
        current_chars = rest.chars().count();
    }
    flush(&mut current, &mut current_chars, &mut out);
    out
}

/// Words for matching: lowercase, letters and digits only, three or more
/// characters (two for numbers: "art. 12"), so "the", "di", "la" do not count.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let n = w.chars().count();
            n >= 3 || (n >= 1 && w.chars().all(|c| c.is_ascii_digit()))
        })
        .map(|w| w.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn zip_of(members: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, body) in members {
                w.start_file(*name, opts).unwrap();
                w.write_all(body.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn a_word_document_reads_as_paragraphs_tabs_and_table_cells() {
        let xml = r#"<?xml version="1.0"?><w:document xmlns:w="w"><w:body>
            <w:p><w:r><w:t>Contratto di locazione</w:t></w:r></w:p>
            <w:p><w:r><w:t xml:space="preserve">Art. 1 </w:t></w:r><w:r><w:tab/><w:t>Oggetto &amp; durata</w:t></w:r></w:p>
            <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Canone</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>800 &#8364;</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
            </w:body></w:document>"#;
        let doc = extract("contratto.docx", &zip_of(&[("word/document.xml", xml)])).unwrap();
        assert_eq!(doc.kind, "docx");
        assert!(doc.text.starts_with("Contratto di locazione\nArt. 1 \tOggetto & durata"), "{}", doc.text);
        assert!(doc.text.contains("Canone"), "{}", doc.text);
        assert!(doc.text.contains("800 €"), "{}", doc.text);
    }

    #[test]
    fn slides_come_in_their_order_with_their_numbers() {
        let slide = |t: &str| format!(r#"<p:sld xmlns:a="a" xmlns:p="p"><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>"#);
        let (s1, s2, s10) = (slide("Primo"), slide("Secondo"), slide("Decimo"));
        let bytes = zip_of(&[
            ("ppt/slides/slide10.xml", &s10),
            ("ppt/slides/slide2.xml", &s2),
            ("ppt/slides/slide1.xml", &s1),
        ]);
        let doc = extract("deck.pptx", &bytes).unwrap();
        assert_eq!((doc.kind, doc.parts), ("pptx", Some(3)));
        assert_eq!(doc.text, "[Slide 1]\nPrimo\n\n[Slide 2]\nSecondo\n\n[Slide 10]\nDecimo");
    }

    #[test]
    fn an_opendocument_text_reads_its_body_and_not_its_styles() {
        let content = r#"<office:document-content xmlns:office="o" xmlns:text="t">
            <office:automatic-styles><style:style>ignored</style:style></office:automatic-styles>
            <office:body><office:text>
              <text:h>Titolo</text:h>
              <text:p>Uno<text:s text:c="3"/>due<text:tab/>tre<text:line-break/>quattro</text:p>
            </office:text></office:body></office:document-content>"#;
        let bytes = zip_of(&[
            ("mimetype", "application/vnd.oasis.opendocument.text"),
            ("content.xml", content),
        ]);
        let doc = extract("nota.odt", &bytes).unwrap();
        assert_eq!(doc.kind, "odt");
        assert_eq!(doc.text, "Titolo\nUno   due\ttre\nquattro");
    }

    #[test]
    fn text_files_in_any_usual_encoding() {
        assert_eq!(extract("a.txt", "perché\r\nsì".as_bytes()).unwrap().text, "perché\nsì");
        // Windows-1252, as a file saved by an old Notepad: "perché".
        assert_eq!(extract("a.txt", b"perch\xe9").unwrap().text, "perché");
        // UTF-16 LE with its BOM.
        let utf16: Vec<u8> = [0xFF, 0xFE].into_iter().chain("ciao".encode_utf16().flat_map(u16::to_le_bytes)).collect();
        assert_eq!(extract("a.txt", &utf16).unwrap().text, "ciao");
        let html = extract("p.html", b"<html><body><p>Uno</p><script>x()</script><p>Due</p></body></html>").unwrap();
        assert_eq!((html.kind, html.text.as_str()), ("html", "Uno\nDue"));
    }

    #[test]
    fn what_is_not_a_document_is_refused_with_a_reason() {
        assert!(extract("x.bin", &[0u8, 1, 2, 0, 0]).unwrap_err().contains("neither text"));
        assert!(extract("x.txt", b"").unwrap_err().contains("empty"));
        assert!(extract("x.doc", &[0xD0, 0xCF, 0x11, 0xE0, 0, 0]).unwrap_err().contains(".docx"));
        assert!(extract("x.pdf", b"%PDF-1.4 garbage").unwrap_err().contains("PDF"));
        assert!(extract("x.zip", &zip_of(&[("a.txt", "x")])).unwrap_err().contains("not a document"));
        assert!(extract("x.txt", b"  \n \n").unwrap_err().contains("no text"));
    }

    #[test]
    fn a_member_that_unpacks_too_far_is_refused() {
        let big = "a".repeat(MAX_UNPACKED_BYTES as usize + 10);
        let err = extract("bomb.docx", &zip_of(&[("word/document.xml", &big)])).unwrap_err();
        assert!(err.contains("unpacks to more than"), "{err}");
    }

    #[test]
    fn documents_that_fit_go_whole_and_room_goes_to_the_long_one() {
        let short = "Breve nota.";
        let long: String = (0..200)
            .map(|i| format!("Paragrafo {i}: testo di riempimento che parla di altro.\n\n"))
            .collect();
        let fitted = fit(&[short, &long], "nulla", 10_000);
        assert_eq!(fitted[0], (short.to_string(), false));
        assert!(fitted[1].1);
        assert!(fitted[1].0.chars().count() <= 10_000 - short.len());
        let all = fit(&[short, "altro"], "x", 1000);
        assert!(all.iter().all(|(_, cut)| !cut));
    }

    #[test]
    fn the_passages_kept_are_the_ones_the_question_is_about() {
        let mut text = String::from("Regolamento aziendale sull'uso dei veicoli.\n\n");
        for i in 0..60 {
            text.push_str(&format!(
                "Articolo {i}. Disposizioni generali sull'organizzazione degli uffici e sugli orari di apertura al pubblico, che valgono per tutte le sedi.\n\n"
            ));
        }
        text.push_str("Articolo 61. Il rimborso chilometrico spetta a chi usa la propria automobile per missioni autorizzate.\n\n");
        for i in 62..120 {
            text.push_str(&format!("Articolo {i}. Altre disposizioni generali sugli uffici.\n\n"));
        }
        let picked = select_passages(&text, "Quanto vale il rimborso chilometrico?", 2600);
        assert!(picked.contains("rimborso chilometrico spetta"), "{picked}");
        assert!(picked.starts_with("Regolamento aziendale"), "{picked}");
        assert!(picked.contains(OMITTED));
        assert!(picked.chars().count() <= 2600);
        // Room for one passage only: the one about the question.
        let one = select_passages(&text, "Quanto vale il rimborso chilometrico?", 1300);
        assert!(one.contains("rimborso chilometrico spetta"), "{one}");
    }

    #[test]
    fn a_long_paragraph_is_cut_at_sentence_ends() {
        let paragraph = "Questa è una frase di prova abbastanza lunga. ".repeat(100);
        let parts = split_passages(&paragraph, 300);
        assert!(parts.len() > 5);
        assert!(parts.iter().all(|p| p.chars().count() <= 300));
        assert!(parts[0].ends_with('.'), "{}", parts[0]);
        // Lines, as a Word file has them: grouped, never split.
        let lines: String = (0..50).map(|i| format!("Riga numero {i} del documento.\n")).collect();
        let parts = split_passages(&lines, 200);
        assert!(parts.iter().all(|p| p.chars().count() <= 200 && p.ends_with("documento.")), "{parts:?}");
        assert_eq!(parts.join("\n"), lines.trim_end());
    }
}
