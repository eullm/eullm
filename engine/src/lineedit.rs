//! Terminal line input for the two places eullm reads from a human: the model
//! picker and the chat REPL.
//!
//! Both used `read_line` on a plain stdin. In canonical mode the tty driver
//! echoes and edits printable characters, but an arrow key is not one — it
//! arrives as the escape sequence `\x1b[D`, which the driver both echoes as
//! `^[[D` and hands to `read_line` verbatim. Pressing left to fix a typo
//! therefore printed garbage on screen *and* put it in the value: at the
//! picker's `Choice >` prompt that produced "Invalid choice" for a line that
//! looked blank, and at the chat's `>>>` prompt it sent the escape sequences to
//! the model as part of the message.
//!
//! Stripping the sequences fixes the value but not the display, and it still
//! leaves the cursor unable to move. So this reads through rustyline, which
//! puts the terminal in raw mode and implements the editing keys properly, and
//! keeps the stripping reader as the fallback for a terminal that cannot go raw
//! (and for a piped stdin, where at least the value stays clean).

/// One line as it came back from the terminal.
pub enum Line {
    Text(String),
    /// Ctrl+D, or stdin ran out.
    Eof,
    /// Ctrl+C — abandon whatever is half-typed, keep the session.
    Interrupted,
}

/// Reads lines from the terminal, with editing where the terminal allows it.
pub enum LineReader {
    Edited(Box<rustyline::DefaultEditor>),
    Plain,
}

impl LineReader {
    pub fn new() -> Self {
        match rustyline::DefaultEditor::new() {
            Ok(ed) => Self::Edited(Box::new(ed)),
            Err(_) => Self::Plain,
        }
    }

    /// Read one line. rustyline prints the prompt itself — it has to, since it
    /// redraws the line on every keystroke — so the fallback prints it too,
    /// rather than leaving that to the caller.
    pub fn read(&mut self, prompt: &str) -> Line {
        use rustyline::error::ReadlineError;
        match self {
            Self::Edited(ed) => match ed.readline(prompt) {
                Ok(l) => Line::Text(l),
                Err(ReadlineError::Interrupted) => Line::Interrupted,
                // Anything other than Ctrl+C means we cannot keep reading;
                // treat it as end of input rather than spinning on the error.
                Err(_) => Line::Eof,
            },
            Self::Plain => {
                use std::io::Write;
                print!("{prompt}");
                let _ = std::io::stdout().flush();
                let mut buf = String::new();
                match std::io::stdin().read_line(&mut buf) {
                    Ok(0) | Err(_) => Line::Eof,
                    Ok(_) => Line::Text(strip_terminal_escapes(&buf)),
                }
            }
        }
    }

    /// Make an entry recallable with the up arrow. No-op without editing.
    pub fn remember(&mut self, entry: &str) {
        if let Self::Edited(ed) = self {
            let _ = ed.add_history_entry(entry);
        }
    }
}

impl Default for LineReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Drop ANSI escape sequences and control characters from a line, leaving the
/// text as typed.
///
/// Used by the fallback reader, where the escape sequences are still arriving.
/// Case and surrounding whitespace are left alone: a chat message is content,
/// and only the picker wants to fold a menu choice.
pub fn strip_terminal_escapes(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // CSI: ESC '[' then parameter bytes, ended by a byte in @..~.
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            } else {
                chars.next();
            }
            continue;
        }
        if !c.is_control() {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::strip_terminal_escapes;

    // Reported from a real session: pressing the left arrow to fix a typo
    // filled the line with escape sequences.
    #[test]
    fn arrow_keys_do_not_become_input() {
        assert_eq!(strip_terminal_escapes("\u{1b}[D\u{1b}[D\u{1b}[D\n"), "");
        assert_eq!(strip_terminal_escapes("1\u{1b}[D\u{1b}[C\n"), "1");
    }

    // A chat message is content: nothing about it may be folded or trimmed
    // here, because the REPL decides that for itself.
    #[test]
    fn ordinary_text_is_untouched() {
        assert_eq!(
            strip_terminal_escapes("  Ciao, Come Stai? \n"),
            "  Ciao, Come Stai? "
        );
    }

    // A path is typed at one of these prompts, so stripping must not eat
    // anything a filename can legitimately contain.
    #[test]
    fn a_path_survives() {
        assert_eq!(
            strip_terminal_escapes("/home/u/models/My Model-Q4_K_M.gguf\n"),
            "/home/u/models/My Model-Q4_K_M.gguf"
        );
    }

    // An escape that is not a CSI sequence still costs exactly its two bytes.
    #[test]
    fn a_bare_escape_does_not_eat_the_line() {
        assert_eq!(strip_terminal_escapes("a\u{1b}Xbc"), "abc");
    }
}
