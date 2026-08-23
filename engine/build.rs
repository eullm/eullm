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

    // Re-run only when HEAD or the index actually changes, not on every build.
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
}
