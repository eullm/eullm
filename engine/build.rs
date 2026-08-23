use std::path::{Path, PathBuf};
use std::process::Command;

/// Embeds the short git commit hash this binary was built from, so `-V`/`--version`
/// can answer "which exact commit is this" without trusting whoever tells you the
/// crate version number — the crate version only bumps on a release, but a build
/// from an unreleased branch (like this one) can be anywhere on that branch's
/// history. Falls back to "unknown" rather than failing the build: a source
/// tarball or a shallow clone with no `.git` directory must still build.
fn main() {
    let hash = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let suffix = if dirty { "-dirty" } else { "" };
    println!("cargo:rustc-env=EULLM_GIT_HASH={hash}{suffix}");

    emit_git_rerun_triggers();
}

/// Tell cargo which files to watch so the embedded hash is refreshed when the
/// checked-out commit moves.
///
/// The obvious choice — `.git/HEAD` alone — is wrong, and wrong in the way that
/// wastes the most time: on a normal branch `HEAD` holds the *symbolic* ref
/// (`ref: refs/heads/main`), whose bytes never change when you commit, merge or
/// pull on that same branch. Only the ref file it points at moves. Watching just
/// `HEAD` therefore freezes the hash at whatever the first build saw, and every
/// later build reports a commit it was not built from — exactly the false answer
/// this whole mechanism exists to prevent, and the reason the hash was believed
/// over a `-V` that said otherwise during a debugging session.
///
/// So: watch `HEAD`, resolve the ref it names and watch that too, and watch
/// `packed-refs` (where the ref file lives instead once git has packed it, e.g.
/// after a `git gc` — the loose file then does not exist at all). Watching the
/// index as well keeps the `-dirty` suffix honest as the working tree changes.
///
/// Emitting nothing at all when git metadata cannot be found is deliberate: with
/// no `rerun-if-changed` directive cargo falls back to re-running the script
/// whenever any file in the package changes, which is the safe direction for a
/// source tarball with no `.git` at all.
fn emit_git_rerun_triggers() {
    // build.rs runs with the crate root as CWD, so the repo's git dir is one up.
    let git_dir = Path::new("../.git");

    // A worktree or submodule has `.git` as a *file* pointing elsewhere. Rather
    // than parse that, fall back to letting cargo watch the package.
    if !git_dir.is_dir() {
        return;
    }

    let head = git_dir.join("HEAD");
    if !head.is_file() {
        return;
    }
    println!("cargo:rerun-if-changed={}", head.display());
    println!("cargo:rerun-if-changed={}", git_dir.join("index").display());

    // `ref: refs/heads/main` → also watch `.git/refs/heads/main`. A detached
    // HEAD holds the hash directly, in which case HEAD itself does change and
    // there is no second file to watch.
    let Ok(contents) = std::fs::read_to_string(&head) else {
        return;
    };
    if let Some(ref_path) = contents.strip_prefix("ref:").map(str::trim) {
        let resolved: PathBuf = git_dir.join(ref_path);
        // Emitted whether or not it exists right now: a packed ref becomes a
        // loose file again on the next update, and cargo treats the appearance
        // of a watched path as a change.
        println!("cargo:rerun-if-changed={}", resolved.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join("packed-refs").display()
    );
}
