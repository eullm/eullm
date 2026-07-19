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

use std::io::{IsTerminal, Write};
use std::path::PathBuf;

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
            if let Some(gguf) = store.gguf_path(&m.name) {
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
    if let Ok(root_path) = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(|h| PathBuf::from(h).join(".eullm").join("models"))
        && let Ok(entries) = std::fs::read_dir(&root_path)
    {
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

    out
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
            // trigger a download.
            let local_tag = if store.exists(&m.id) { "[local]" } else { "" };
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

    loop {
        print!("  Choice > ");
        let _ = std::io::stdout().flush();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            return Picked::Quit;
        }
        let choice = input.trim().to_lowercase();

        if choice.is_empty() {
            continue;
        }
        if choice == "q" || choice == "quit" || choice == "exit" {
            return Picked::Quit;
        }
        if choice == "p" {
            print!("  Path to .gguf > ");
            let _ = std::io::stdout().flush();
            let mut p = String::new();
            if std::io::stdin().read_line(&mut p).is_ok() {
                let path = PathBuf::from(p.trim());
                if path.exists() {
                    return Picked::Local(path);
                }
                println!("  ! File not found: {}", path.display());
                continue;
            }
        }
        if choice == "u" {
            print!("  URL > ");
            let _ = std::io::stdout().flush();
            let mut u = String::new();
            if std::io::stdin().read_line(&mut u).is_ok() {
                let url = u.trim().to_string();
                if url.starts_with("http://") || url.starts_with("https://") {
                    return Picked::Url(url);
                }
                println!("  ! URL must start with http(s)://");
                continue;
            }
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
