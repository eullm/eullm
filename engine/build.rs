use std::path::{Path, PathBuf};
use std::process::Command;

/// Embeds the short git commit hash this binary was built from, so `-V`/`--version`
/// can answer "which exact commit is this" without trusting whoever tells you the
/// crate version number — the crate version only bumps on a release, but a build
/// from an unreleased branch (like this one) can be anywhere on that branch's
/// history. Falls back to "unknown" rather than failing the build: a source
/// tarball or a shallow clone with no `.git` directory must still build.
///
/// Both calls pass `-c safe.directory=*`. Without it, git refuses to read a
/// checkout owned by another user ("detected dubious ownership"), which is
/// every release build that runs in a container: the workspace is mounted
/// from the host, owned by the runner, and the build runs as root.
/// `actions/checkout` does add the exception, but only to a temporary global
/// config under a temporary HOME that is gone before `cargo build` starts, so
/// six of the release binaries — the Linux CPU, CUDA and ROCm ones — shipped
/// answering `unknown` to `-V`, which is exactly the question this exists to
/// answer. Found on 0.7.8's `eullm-linux-x64-cuda-13.1`.
///
/// There is nothing to protect here. `safe.directory` guards against running
/// git inside a repository whose own config you have not agreed to trust, and
/// whoever runs this build script is already running this repository's code.
/// The command-line scope is one git honours for this key, unlike the
/// repository's own config.
fn main() {
    let hash = Command::new("git")
        .args(["-c", "safe.directory=*", "rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    // `-dirty` when what the binary is built from differs from the commit:
    // this crate (sources, the vendored llama.cpp bindings and their patches,
    // untracked files included, since the build applies every patch it finds),
    // the catalog it embeds, and the workspace manifest and lock file. Not the
    // rest of the checkout: on LUMI-G every binary from 18 to 20-10-2026 said
    // `-dirty` because of core dumps at the repository root, and a flag that
    // is always on tells nothing.
    let dirty = Command::new("git")
        .args([
            "-c",
            "safe.directory=*",
            "status",
            "--porcelain",
            "--",
            ":/engine",
            ":/catalog",
            ":/Cargo.toml",
            ":/Cargo.lock",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let suffix = if dirty { "-dirty" } else { "" };
    println!("cargo:rustc-env=EULLM_GIT_HASH={hash}{suffix}");

    // The llama.cpp the engine is built on, for the chat UI's About page: its
    // `bNNNNN` tag where the submodule's clone has it, else its commit. Moves
    // with the submodule pin, which is in the index watched below.
    let llama_cpp = Command::new("git")
        .args(["-c", "safe.directory=*", "-C", "vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"])
        .args(["describe", "--tags", "--always"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=EULLM_LLAMA_CPP={llama_cpp}");

    // The release asset this binary is published as, which each build job in
    // release-engine.yml sets: `eullm update` downloads the same one. Empty
    // for a build from source, which `eullm update` then declines to replace.
    println!("cargo:rerun-if-env-changed=EULLM_RELEASE_ASSET");
    let asset = std::env::var("EULLM_RELEASE_ASSET").unwrap_or_default();
    println!("cargo:rustc-env=EULLM_RELEASE_ASSET={}", asset.trim());

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
