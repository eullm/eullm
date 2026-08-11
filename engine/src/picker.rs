//! Interactive model picker — the screen the user sees when they type
//! plain `eullm` or `eullm run` with no arguments in a real terminal.
//!
//! Behavior:
//!   - Lists local GGUF files already in the model store.
//!   - Lists models from the catalog (remote when reachable, embedded otherwise).
//!   - Lets the user pick by number, or type a custom path / URL.
//!   - Detects non-interactive shells (pipes, redirects, CI) and bails out
//!     to a printed usage message instead of hanging on stdin.
//!
//! Not exposed as its own subcommand — it's just the default behavior when
//! a required model argument is missing.

use std::io::IsTerminal;
use std::path::PathBuf;

use crate::lineedit::Line;
use crate::models::{CatalogEntry, ModelStore, catalog};

/// What the user picked.
#[derive(Debug, Clone)]
pub enum Picked {
    /// A `.gguf` already on disk — just run it.
    Local(PathBuf),
    /// A catalog entry — may need to be downloaded before running.
    /// Boxed because `CatalogEntry` is much larger than the other variants
    /// (clippy::large_enum_variant) and we only allocate one at user choice.
    Catalog(Box<CatalogEntry>),
    /// A direct URL the user pasted — download then run.
    Url(String),
    /// User chose to quit.
    Quit,
}

/// Return whether both stdin and stdout are connected to a terminal.
/// The picker only renders when this is true; otherwise we'd block on a
/// pipe waiting for input that will never come.
pub fn is_interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

/// Run the picker. Returns `None` if we're not in an interactive shell
/// (caller should print usage instead).
pub async fn pick(store: &ModelStore) -> Option<Picked> {
    if !is_interactive() {
        return None;
    }

    let locals = list_local_ggufs(store);
    // Try the live catalog (5s timeout); falls back to embedded silently.
    let remote = catalog::fetch_remote().await;
    Some(prompt_user(&locals, &remote, store))
}

#[derive(Debug, Clone)]
struct LocalModel {
    path: PathBuf,
    display_name: String,
    size_bytes: u64,
}

fn list_local_ggufs(store: &ModelStore) -> Vec<LocalModel> {
    let mut out = Vec::new();

    // 1) Models registered with the store (have manifest.json).
    if let Ok(manifests) = store.list() {
        for m in manifests {
            // `id` is the directory key; `name` is the human-readable title.
            // Looking up by name asks for a directory called "DeepSeek R1
            // Distill (Qwen-14B)", which does not exist, so every catalog
            // model on disk silently vanished from LOCAL. Only models whose
            // title happens to equal their id survived, which is why the
            // section usually showed exactly one entry.
            let key = if m.id.is_empty() { &m.name } else { &m.id };
            if let Some(gguf) = store.gguf_path(key) {
                let size = std::fs::metadata(&gguf).map(|md| md.len()).unwrap_or(0);
                out.push(LocalModel {
                    path: gguf,
                    display_name: m.name,
                    size_bytes: size,
                });
            }
        }
    }

    // 2) Raw .gguf files dropped into the store root (no manifest).
    //
    // Ask the store where it lives rather than re-deriving the default path:
    // this branch used to hardcode $HOME/.eullm/models and so ignored
    // EULLM_MODELS_DIR entirely, looking in a directory the rest of the
    // process was not using.
    let (root_path, _) = store.root_with_source();
    if let Ok(entries) = std::fs::read_dir(root_path) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file() && p.extension().is_some_and(|x| x == "gguf") {
                let already = out.iter().any(|m| m.path == p);
                if !already {
                    let size = std::fs::metadata(&p).map(|md| md.len()).unwrap_or(0);
                    let name = p
                        .file_name()
                        .and_then(|x| x.to_str())
                        .unwrap_or("(unknown)")
                        .to_string();
                    out.push(LocalModel {
                        path: p,
                        display_name: name,
                        size_bytes: size,
                    });
                }
            }
        }
    }

    // 3) Model directories holding a .gguf but no usable manifest.json.
    //
    // Neither branch above sees these: `store.list()` only returns directories
    // whose manifest parses, and branch 2 only looks at files sitting directly
    // in the store root. So a perfectly runnable model became invisible the
    // moment it was tidied from `models/x.gguf` into `models/x/x.gguf` — the
    // move made it *less* discoverable, which is the opposite of what anyone
    // doing it expects. An interrupted pull, a restored backup and a directory
    // copied from another machine all land in the same state.
    //
    // `eullm list` already reports these separately (`ModelStore::unlisted`),
    // and running one by path has always worked. Only the picker was silent,
    // and the picker is where a user goes precisely when they do not know the
    // name to type.
    if let Ok(entries) = std::fs::read_dir(store.root_with_source().0) {
        for e in entries.flatten() {
            let dir = e.path();
            if !dir.is_dir() {
                continue;
            }
            let Some(gguf) = model_gguf_in_dir(&dir) else {
                continue;
            };
            if out.iter().any(|m| m.path == gguf) {
                continue;
            }
            let size = std::fs::metadata(&gguf).map(|md| md.len()).unwrap_or(0);
            out.push(LocalModel {
                path: gguf,
                display_name: e.file_name().to_string_lossy().into_owned(),
                size_bytes: size,
            });
        }
    }

    out
}

/// The runnable GGUF inside a model directory, if there is one.
///
/// Skips `mmproj*.gguf`: a projector is not a model, and offering one as
/// something to run loads weights that answer nothing. A directory holding
/// only a projector therefore has no model in it, which is different from
/// having one that happens to sort second.
///
/// Sorted by filename so the choice is stable rather than whatever order the
/// filesystem returns — a picker entry that changes between runs on the same
/// unchanged directory is worse than a wrong one, because it cannot be
/// reported.
fn model_gguf_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|f| f.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "gguf")
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.to_ascii_lowercase().starts_with("mmproj"))
        })
        .collect();
    ggufs.sort();
    ggufs.into_iter().next()
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{bytes} B")
    }
}

/// Normalise a line read from a terminal in canonical mode.
///
/// The driver echoes and edits printable characters, but an arrow key is not
/// one: it arrives as the escape sequence `\x1b[D`, which `read_line` hands
/// over verbatim. Pressing left a few times to correct a typo therefore
/// produced `^[[D^[[D^[[D` in the buffer and a bewildering "Invalid choice"
/// for what looked like an empty line. Dropping escape sequences and control
/// characters makes those keys inert rather than destructive.
///
/// This is only the folding: the escape sequences are gone before this sees
/// the line, either because the editor never produced them or because the
/// fallback reader stripped them. See `crate::lineedit`.
fn sanitize_input(raw: &str) -> String {
    crate::lineedit::strip_terminal_escapes(raw)
        .trim()
        .to_lowercase()
}

fn prompt_user(
    locals: &[LocalModel],
    catalog_models: &[CatalogEntry],
    store: &ModelStore,
) -> Picked {
    println!();
    println!("  ╭─ EuLLM ─ choose a model ────────────────────────────────────");
    println!("  │");

    let mut next_num: usize = 1;
    // (number, kind), kind = (Local index | Catalog index)
    let mut index: Vec<MenuItem> = Vec::new();

    if !locals.is_empty() {
        println!("  │  LOCAL");
        for (i, m) in locals.iter().enumerate() {
            println!(
                "  │   {:>3}) {:<40} {:>8}",
                next_num,
                truncate(&m.display_name, 40),
                format_size(m.size_bytes),
            );
            index.push(MenuItem::Local(i));
            next_num += 1;
        }
        println!("  │");
    }

    if !catalog_models.is_empty() {
        println!(
            "  │  CATALOG ({} models, all permissive licenses)",
            catalog_models.len()
        );

        // Order: recommended first, then by params asc, then by name.
        let mut ordered: Vec<(usize, &CatalogEntry)> = catalog_models.iter().enumerate().collect();
        ordered.sort_by(|a, b| {
            b.1.recommended
                .cmp(&a.1.recommended)
                .then_with(|| {
                    a.1.params_b
                        .partial_cmp(&b.1.params_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.1.id.cmp(&b.1.id))
        });

        for (orig_i, m) in &ordered {
            let star = if m.recommended { "★" } else { " " };
            // Mark catalog entries already pulled to the local store so the
            // user can tell at a glance what is ready to run vs what would
            // trigger a download. The test is whether the GGUF is on disk,
            // not whether a manifest.json exists: a manifest is written
            // before the download completes and survives a deleted weight
            // file, so the cheaper check tags models that cannot be run.
            let local_tag = if store.is_present(&m.id) {
                "[local]"
            } else {
                ""
            };
            let rec_tag = if m.recommended { "[recommended]" } else { "" };
            let tags = match (local_tag, rec_tag) {
                ("", "") => String::new(),
                ("", r) => r.to_string(),
                (l, "") => l.to_string(),
                (l, r) => format!("{l} {r}"),
            };
            println!(
                "  │   {:>3}) {} {:<28} {:>8}  {:<14} {}",
                next_num,
                star,
                truncate(&m.id, 28),
                format_size(m.size_bytes),
                m.license,
                tags
            );
            index.push(MenuItem::Catalog(*orig_i));
            next_num += 1;
        }
        println!("  │");
    }

    println!("  │  OTHER");
    println!("  │     p) Enter a custom .gguf path");
    println!("  │     u) Enter a custom URL to a .gguf");
    println!("  │     q) Quit");
    println!("  │");
    println!("  ╰─");
    println!();

    // Reading through the editor rather than `read_line` is what stops the tty
    // from echoing `^[[D` when the user presses left. Stripping the sequence
    // from the value was only half the fix: the characters were still on
    // screen, because in canonical mode the driver echoes them before we ever
    // see the line.
    let mut reader = crate::lineedit::LineReader::new();

    loop {
        let choice = match reader.read("  Choice > ") {
            Line::Text(l) => sanitize_input(&l),
            // Ctrl+C at the menu means "I did not mean to be here".
            Line::Eof | Line::Interrupted => return Picked::Quit,
        };

        if choice.is_empty() {
            continue;
        }
        if choice == "q" || choice == "quit" || choice == "exit" {
            return Picked::Quit;
        }
        if choice == "p" {
            // Not `sanitize_input`: a path keeps its case.
            if let Line::Text(p) = reader.read("  Path to .gguf > ") {
                let path = PathBuf::from(p.trim());
                if path.exists() {
                    return Picked::Local(path);
                }
                println!("  ! File not found: {}", path.display());
            }
            continue;
        }
        if choice == "u" {
            if let Line::Text(u) = reader.read("  URL > ") {
                let url = u.trim().to_string();
                if url.starts_with("http://") || url.starts_with("https://") {
                    return Picked::Url(url);
                }
                println!("  ! URL must start with http(s)://");
            }
            continue;
        }
        if let Ok(n) = choice.parse::<usize>()
            && n >= 1
            && n <= index.len()
        {
            return match &index[n - 1] {
                MenuItem::Local(i) => Picked::Local(locals[*i].path.clone()),
                MenuItem::Catalog(i) => Picked::Catalog(Box::new(catalog_models[*i].clone())),
            };
        }

        println!("  ! Invalid choice. Type a number, p, u, or q.");
    }
}

enum MenuItem {
    Local(usize),
    Catalog(usize),
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

#[cfg(test)]
mod input_tests {
    use super::sanitize_input;

    // Reported from a real session: pressing the left arrow to fix a typo
    // filled the line with escape sequences and the picker answered
    // "Invalid choice" for something that looked blank.
    #[test]
    fn arrow_keys_do_not_become_input() {
        assert_eq!(sanitize_input("\u{1b}[D\u{1b}[D\u{1b}[D\n"), "");
        assert_eq!(sanitize_input("1\u{1b}[D\u{1b}[C\n"), "1");
    }

    #[test]
    fn ordinary_choices_are_untouched() {
        assert_eq!(sanitize_input("  12 \n"), "12");
        assert_eq!(sanitize_input("Q\n"), "q");
        assert_eq!(sanitize_input("\n"), "");
    }

    // A path is typed at the same prompt family, so stripping must not eat
    // anything a filename can legitimately contain.
    #[test]
    fn a_path_survives() {
        assert_eq!(
            sanitize_input("/home/u/models/My Model-Q4_K_M.gguf\n"),
            "/home/u/models/my model-q4_k_m.gguf"
        );
    }
}

#[cfg(test)]
mod local_scan_tests {
    use super::model_gguf_in_dir;
    use std::fs;

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("eullm-picker-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("tmpdir");
        d
    }

    /// The case that motivated this: a model tidied from `models/x.gguf` into
    /// `models/x/x.gguf` has no manifest, so neither `ModelStore::list` nor the
    /// store-root file scan sees it. Tidying it up must not hide it.
    #[test]
    fn a_lone_gguf_in_a_directory_is_found() {
        let d = tmpdir("lone");
        fs::write(d.join("qwen2.5-3b-instruct-q5_k_m.gguf"), b"w").unwrap();
        assert_eq!(
            model_gguf_in_dir(&d).and_then(|p| p.file_name().map(|n| n.to_owned())),
            Some("qwen2.5-3b-instruct-q5_k_m.gguf".into())
        );
        let _ = fs::remove_dir_all(&d);
    }

    /// A projector is not a model. Offering `mmproj-F16.gguf` as something to
    /// run loads weights that answer nothing.
    #[test]
    fn a_projector_is_never_offered_as_the_model() {
        let d = tmpdir("vision");
        fs::write(d.join("mmproj-F16.gguf"), b"p").unwrap();
        fs::write(d.join("weights.gguf"), b"w").unwrap();
        assert_eq!(
            model_gguf_in_dir(&d).and_then(|p| p.file_name().map(|n| n.to_owned())),
            Some("weights.gguf".into())
        );

        // And a directory holding ONLY a projector has no model in it, rather
        // than a model that happens to be a projector.
        let only = tmpdir("only-proj");
        fs::write(only.join("mmproj-F16.gguf"), b"p").unwrap();
        assert!(model_gguf_in_dir(&only).is_none());

        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(&only);
    }

    /// Filesystem order is not an ordering. Two runs over an unchanged
    /// directory must offer the same file.
    #[test]
    fn the_choice_is_stable_when_several_quants_sit_together() {
        let d = tmpdir("multi");
        for n in ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q5_K_M.gguf"] {
            fs::write(d.join(n), b"w").unwrap();
        }
        let first = model_gguf_in_dir(&d);
        assert_eq!(first, model_gguf_in_dir(&d));
        assert_eq!(
            first.and_then(|p| p.file_name().map(|n| n.to_owned())),
            Some("model-Q4_K_M.gguf".into())
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_directory_without_weights_is_not_a_model() {
        let d = tmpdir("empty");
        fs::write(d.join("README.md"), b"x").unwrap();
        assert!(model_gguf_in_dir(&d).is_none());
        let _ = fs::remove_dir_all(&d);
    }
}
