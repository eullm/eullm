---
name: release-and-ci
description: Release process, CI/CD workflow rules, sccache/S3 caching strategy, version numbering, and changelog conventions for eullm. Use when working on .github/workflows/*.yml, cutting a release, tagging EuLLM-v*, editing CHANGELOG.md, or debugging CI cache/build-time issues.
---

# CI/CD Rules (MANDATORY — do not remove or simplify)

The GitHub Actions workflows have been carefully optimized. **Do not remove caching steps.**

## `.github/workflows/ci.yml`
- `Swatinem/rust-cache@v2` on `engine` and `hub` jobs — caches `target/` and `~/.cargo`. Removing it adds ~25 min per run.
- `actions/cache` for pip on `forge` job.

## `.github/workflows/release-engine.yml`
- All `build` matrix jobs use `Swatinem/rust-cache@v2` keyed by target triple.
- `build-cuda` (container-based) uses `actions/cache` manually (Swatinem doesn't work in containers) for: cargo registry, `target/`.
- sccache routes C/C++/CUDA compile through an **S3 backend on EU-hosted MinIO** (`ci.eullm.eu`). NOT the GitHub Actions cache — see the sccache subsection below for the full reasoning (size cap + cross-workflow sharing, NOT ref-scoping; that earlier framing was imprecise).

## TurboQuant removed in v0.5.8 (history note)

Earlier versions (v0.5.x) shipped a TurboQuant-experimental variant via the AmesianX/llama.cpp fork. That added three jobs (`build-cuda-turboquant`, `build-metal-turboquant`, `build-windows-cuda-turboquant`), an `engine-turboquant` CI job, a vendored `engine/vendor/` dir, a `[patch.crates-io]` block, and was the multi-hour long-pole of every release. **All of it was removed in v0.5.8** — see README → Research & Experiments for the rationale. Several lessons below were learned on those jobs; they still apply to any future C++/CUDA work (e.g. when a future llama.cpp DLL strategy lands).

## Cache key design — read this before touching any sccache key

**Hard lesson learned twice on v0.5.1 and v0.5.2**: putting `Cargo.lock` in the `sccache` cache key wastes 2+ hours of CI on the long-pole CUDA job for every Rust-side version bump. The C++/CUDA object files cached by sccache depend on llama.cpp source and compiler flags, NOT on the Rust dependency tree. Removing `Cargo.lock` from sccache keys was the structural fix.

**Three-cache-layer breakdown per build job:**

| Cache layer | Key includes | Purpose | Why it's correctness-safe |
|-------------|--------------|---------|---------------------------|
| `cargo-registry-*` | `Cargo.lock` hash | Skip re-downloading crate sources | Source code, no compilation |
| `target-*` | `Cargo.lock` (+ any pinned C++ source manifest if vendored) | Skip re-compiling Rust crates | Cargo fingerprints by content; any source change → recompile |
| `sccache-*` | **Pinned C++ source identity only** (NOT `Cargo.lock`) | Skip re-compiling C++/CUDA kernels | sccache is content-addressed: SHA1(preprocessed source + includes + flags + compiler version). Source change → different hash → miss → recompile |

**Why sccache MUST NOT include Cargo.lock:**
- sccache caches `.obj` / `.o` files from llama.cpp C++/CUDA source
- That source moves only when the pinned llama.cpp version moves, not on Rust bumps
- A Rust version bump in `engine/Cargo.toml` (and consequently `Cargo.lock`) changes ZERO bytes of C++ source
- Including `Cargo.lock` in the sccache key wastes the cache on every Rust bump
- The GHA cache key is just "which cache dir to restore"; sccache internally content-hashes each file. Wrong key → restore wrong dir → still get content matches for unchanged files → still safe, but missed optimisations

**Why this is correctness-safe even with "wrong" keys:**
- `sccache --show-stats` in build logs reports hit/miss rate; high rate = working
- If a `.cu`/`.cpp` source actually changes, the content hash changes → sccache miss → recompile from scratch → fresh `.obj` linked into binary. Cannot ever produce a stale binary.
- Cargo's fingerprint system applies the same logic for Rust crates.

**Nuclear option** if cache contamination is ever suspected: bump the cache key suffix (e.g., `sccache-windows-cuda-v2-...`). Full miss next run, fresh state.

## The bump commit must be on `main` before the tag is pushed

Enforced by `require_version_match` in `release-engine.yml`: a tag whose version
does not equal `engine/Cargo.toml`'s fails in seconds and blocks `release`, so
nothing is published. Learned from v0.6.43, which was tagged one merge before
its bump: nine binaries went out answering `0.6.42` to `-V`, and the release
also missed the last fix that was supposed to be in it. Nothing failed at the
time and nothing looked wrong on the release page.

A published tag is not moved to fix this. The release page and its checksums
are already in users' hands; ship the next patch version and say in
`CHANGELOG.md` what the mis-tagged one actually contains.

## The release publishes what was built, not a list someone maintains

`release-engine.yml` attaches `artifacts/*/*`. It used to name each file
explicitly, and v0.6.47 published nine binaries out of ten: `build-vulkan`
succeeded, its artifact was downloaded, its checksum went into `checksums.txt`,
and it was never attached because nobody had added a line for it.
`fail_on_unmatched_files: false` — which exists so a failed build degrades
gracefully — made the omission silent by construction.

Every build job uploads one file into a directory named after it, so the glob
*is* the set that was built. **Do not go back to an explicit list**: adding a
build job and forgetting the publish line is a mistake the workflow should not
be able to make.

Related smell worth remembering: `checksums.txt` is generated by walking the
artifact directories, so it listed the missing binary both times. When a
release looks wrong, diffing the checksum file against the attached assets
answers "was it built?" and "was it published?" separately.

## Versions go in steps of ten, and releases accumulate (from 0.6.52 onwards)

The next release after 0.6.52 is 0.6.60, then 0.6.70, 0.6.80. Not every fix and
not every feature gets a release: work accumulates on `main` and a build goes
out when there is enough in it to be worth someone's download.

On 28 July 2026 nine releases shipped in one day, 0.6.42 through 0.6.51. Three
of them existed only to fix the release mechanics of the previous one, and
testers were re-downloading a 900 MB CUDA binary several times an afternoon.

Two consequences:

- `engine/Cargo.toml` is not bumped until the release commit. During the
  accumulation window `main` stays on the last published version, so a build
  from `main` reports a version that exists and can be downloaded.
- Changelog entries are written as the work lands, under `## Unreleased`. The
  bump commit renames that heading to the version and adds the date.

The step of ten leaves room for a hotfix in between (0.6.61) without disturbing
the sequence. After 0.6.90, decide then between 0.6.100 and moving to 0.7.0.

## Every release updates `CHANGELOG.md` (MANDATORY)

Before pushing a `EuLLM-v*` tag, add the version's section to `CHANGELOG.md`.
The bump commit is the natural place for it.

**Write it for someone deciding whether to upgrade, not for someone reading the
diff.** The test: an entry that names a function, a module, or a refactor has
failed. "One decode output path instead of three" tells a user nothing; "stop
sequences are now honoured in every mode, which affects `--batch-size 0` and
multimodal requests" tells them whether this release matters to them. State the
consequence, and where it exists, the workaround — the 0.6.36 entry gives both
the extra VRAM cost of the new KV defaults and the flags that restore the old
behaviour.

Two sections per version, `### Added` and `### Fixed`, plus `### Performance`
when something measurably got faster. A release with no user-facing change says
so in one line (see 0.6.38) rather than being omitted, so no version number
looks mysteriously missing.

Leave out `docs`, `chore`, `ci`, `test` and internal refactors. A changelog is
not the commit history; the commit history is already the commit history.

Entries below 0.6.36 were generated from commit subjects and read like it. The
file's header admits this. **Do not regenerate the hand-written span from git
log** to "make it consistent": that would replace the only part written for
users with the part that is not.

## Release graceful degradation (added v0.5.2)

The `release` job uses `if: ${{ !cancelled() }}` + `fail_on_unmatched_files: false`. **Do not remove these.**

Rationale: any long-pole CUDA build can take 30 min – 1h depending on cache state. A single mistake in a late-stage step used to nuke the entire release after all that work. Now the release publishes whichever binaries succeeded, and a follow-up patch release can address what failed.

## Build times (approximate, v0.5.8 onwards — TurboQuant variants removed)
| Job | Cold | Warm cache |
|-----|------|------------|
| Engine standard (Linux/macOS) | ~6 min | ~1-2 min |
| Windows standard | ~10 min | ~3-5 min |
| Linux CUDA | ~18 min | ~3-5 min |
| **Windows CUDA** (long-pole) | **~50 min** (cold) | **~10-15 min** (warm) |

## How a release in progress looks on GitHub (don't be fooled)

When the tag is pushed, GitHub creates the release **immediately** with only
the two auto-generated source-code archives (`Source code (zip)` and
`(tar.gz)`) → the public page shows `Assets 2` and `published_at` is set
~seconds after the tag, even though no binary has been compiled yet.

The build jobs upload their binaries to the workflow's **artifact storage**
as each one finishes; those artifacts are visible only inside the Actions UI
to the repo maintainer (`ci-deploy` view), not on the public release page.
Only when the final `release` job runs (it `needs: [all builds]`, gated by
`if: !cancelled()`) does softprops/action-gh-release attach every artifact
to the release at once → that's the moment "Assets 2" jumps to the full set
(13 binaries + checksums for v0.5.x).

**Practical:** during a release run, looking at the public release page tells
you nothing about progress — `Assets 2` is the steady state until the final
job lands. To know what's actually happening, look at the Actions tab (live
job status) or ask the maintainer (they see the artifact list early). Don't
re-derive theories from `published_at`.

## PowerShell gotchas in Windows CI steps

Two patterns to remember:
- `$env:ProgramFiles(x86)` is **broken** — PowerShell parses `(x86)` as a function call. Use `${env:ProgramFiles(x86)}` with braces. The Inno Setup install path needs this: `& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"`.
- `vswhere` from VS Installer is in the (x86) Program Files: `& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"`.
- Single-line over multi-line: if a step crosses many lines with `` ` `` continuation, sanity check that braces survive YAML parsing. Prefer single-line when possible.

## Validate Inno Setup scripts BEFORE pushing a tag

The `installer-preflight` job in `ci.yml` compiles all 3 installers with dummy 100-byte staging files on every push. **Trust it, don't bypass it.** Two bugs (`$env:ProgramFiles(x86)` then `{userprofile}`) ate two 2h+ release builds before this preflight existed. Inno Setup has no built-in `{userprofile}` constant — use `{userdocs}` or `{%USERPROFILE}` for the user's home area. Full list of built-ins: https://jrsoftware.org/ishelp/index.php?topic=consts

## sccache uses an S3 backend (MinIO on ci.eullm.eu), NOT the GitHub cache

The release workflow (tag-triggered) routes sccache through an S3-compatible
MinIO bucket behind `https://ci.eullm.eu` (Let's Encrypt proxy). `SCCACHE_*`
+ `AWS_*` secrets configure it. `Cache location` in the stats reads `s3`.

**Why NOT the free GitHub Actions cache (the full, accurate story — earlier
notes oversimplified this and were corrected during v0.5.14):**

The GitHub Actions cache scoping rules are not as restrictive as the
v0.5.11→v0.5.12 failure made them look. Per the [official docs][gha-cache-docs],
a workflow run can restore caches created on its own ref **or on the default
branch** (`main`). Tags can read main's caches; the bug in v0.5.12 was simply
that we populated under `EuLLM-v0.5.11`'s tag, not under main — and tag→tag
visibility is correctly blocked (cache poisoning protection). So
"architecturally unsuitable" was wrong: a workflow on main *could* populate a
shared cache that all subsequent tag-triggered releases read.

The **real** reasons S3/MinIO remains the right backend for sccache:

1. **The 10 GB per-repo cap with LRU eviction.** sccache for our CUDA build
   accumulates ~600-900 MB per platform per release (objects for cl.exe/nvcc
   instantiations across all `CMAKE_CUDA_ARCHITECTURES`). After a few releases
   on multiple platforms, the bucket would saturate and start evicting the
   oldest objects on every push — including base layers we'd just paid to
   compile. MinIO has no cap; the existing S3 bucket has held the same
   content-addressed objects across all v0.5.x releases without eviction.
2. **Content-addressed sharing across all workflows in the repo.** Our CI
   workflow (`ci.yml`, runs on every push/PR) and the release workflow share
   the *same* sccache bucket. With S3 they hit each other's writes; with
   GitHub cache, branch/PR runs are isolated from main+tag runs in
   practice (their scopes don't overlap).
3. **Cross-repo reuse (future).** If we ever add a sibling repo (Forge or
   Hub C++ work), the same MinIO bucket continues to serve. GitHub cache
   is per-repo, hard boundary.

What this means concretely: **do not switch sccache back to the GitHub
Actions cache**, but the reason is the size cap and the cross-workflow
sharing, not ref-scoping. Proof S3 works: v0.5.9 → v0.5.13 Linux CUDA both
hit 2m34s/2m47s with ~130 CUDA hits each, `Cache location: s3`.

[gha-cache-docs]: https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching

## The Windows CUDA long pole is the toolkit INSTALL, not the compile

Measured on v0.6.38, job `build-windows-cuda`, 15m55s total:

| step | time |
|---|---:|
| **Install CUDA toolkit 12.8** | **7m 34s** |
| **Build engine (CUDA + mtmd)** | **5m 46s** |
| everything else (checkout, LLVM, Ninja, MSVC, package, upload) | 2m 35s |

Two things follow, and both contradict what the workflow used to assume.

**Caching cannot help.** Inside that 7m34s, only 22s was fetching the installer
— the log shows it coming from `C:\hostedtoolcache\...`, already cached. The
other **7m12s is NVIDIA's installer executing**. The only lever is installing
less. Do not reach for a cache layer here; it is already warm.

**sccache is fine.** Same run: 519 hits, 221 C/C++, 139 CUDA. The Ninja fix from
v0.5.14 works. Do not go looking for a compile-caching problem that isn't there.

The fix is `sub-packages` on `Jimver/cuda-toolkit` — ~700 MB of components
instead of the full ~3 GB. **An earlier attempt was reverted with a wrong
diagnosis** ("Windows needs `cudart_12.8`, Linux uses `cudart`"), and that wrong
note kept the idea buried for months. The action appends the version itself
(`src/installer.ts`: `` `${subPackage}_${version.major}.${version.minor}` ``).
What differs between platforms is the **base** names: Linux network packages are
`cuda-nvcc` / `libcublas-dev`, Windows installer components are `nvcc` /
`cublas_dev`. The README's "network method only" caveat is Linux-only, and our
Linux CUDA jobs use a devel container without this action.

Derive the component list from what the code includes, never from a template:
`ggml-cuda` includes `cuda.h`, `cuda_runtime.h`, `cuda_fp16/bf16/fp4/fp8.h`,
`cooperative_groups.h`, `mma.h`, `cub/cub.cuh`, `cublas_v2.h`, and CMake links
`CUDA::cudart` / `CUDA::cublas` / `CUDA::cuda_driver`. No nvml, nvrtc, nvtx or
cupti. The release ships exactly three DLLs (`cudart64`, `cublas64`,
`cublasLt64`), so cutting the rest changes nothing users receive. Component
sizes are authoritative in NVIDIA's own
`developer.download.nvidia.com/compute/cuda/redist/redistrib_<version>.json`.

`try-windows-ninja.yml` carries the same `method` and `sub-packages` as the
release job specifically so a `workflow_dispatch` validates a change to the list
without spending a release. Keep the two in step. A missing component fails at
compile or link time — loud, never a subtly wrong binary.

## Windows CUDA: Ninja generator + S3 sccache (working as of v0.5.14)

For years before v0.5.14, Windows CUDA was the long-pole of every release
(~50 min cold, no warm path). The diagnosis sat in two layers and was only
fully unblocked with a generator swap, not a backend swap.

**Layer 1 — the diagnosis (proven on v0.5.9, S3 + CUDA launcher fix in place):**
sccache cached **only Rust** on Windows, never C/C++, never CUDA.

| 0.5.9 sccache stats | Linux CUDA | Windows CUDA |
|---|---|---|
| Cache hits (Rust) | 160 | 149 |
| Cache hits (C/C++) | 223 | **0** |
| Cache hits (CUDA) | 129 | **0** |
| Wall-clock | 2m34s | ~50 min |

Root cause: the CMake "Visual Studio" generator (MSBuild) **silently ignores**
`CMAKE_C_COMPILER_LAUNCHER` / `CMAKE_CXX_COMPILER_LAUNCHER` /
`CMAKE_CUDA_COMPILER_LAUNCHER`. Those work only with the Makefile and Ninja
generators. `llama-cpp-sys-2` on Windows defaults to the VS generator, so cl.exe
and nvcc invocations bypass sccache entirely — no cache key is ever computed,
no object is ever stored. Switching the cache backend (S3, GitHub, anything)
cannot fix this; the launcher contract is at the generator level.

**Layer 2 — the fix (proven on the try-windows-ninja experiment, run #2):**
force `CMAKE_GENERATOR=Ninja` in the Windows CUDA job. Three small changes:

1. `choco install ninja` (the binary must be on PATH before cargo builds).
2. Activate MSVC x64 dev env via `vcvars64.bat` (found via `vswhere`).
   Plain `windows-2022` runners do NOT activate it — that's only set up for
   MSBuild. Without this, Ninja can't find `cl.exe` / `nvcc`.
3. `CMAKE_GENERATOR: Ninja` as an env on the cargo build step.

Experiment proof (build engine, cold):

| Generator | Tracked by sccache | Wall-clock | Outcome |
|---|---|---|---|
| VS / MSBuild | 0 (no C/C++ line, no CUDA line in stats) | ~36 min | Re-builds from scratch forever |
| **Ninja** | **205 C/C++ + 130 CUDA written to S3** | ~36 min | **Cache populated → next build warm** |

After the experiment populated S3, the projection for the **second** Ninja
build on Windows CUDA is ~5–10 min, mirroring exactly the Linux CUDA jump from
0.5.9 (populate, 2m34s with hits) to 0.5.13 (rebuild, 2m47s with hits).

**Rule of method:** never claim sccache "works" on a platform without reading
that platform's `Cache hits (C/C++)` and `Cache hits (CUDA)` lines first. Don't
extend Linux results to Windows.

## Windows DLL strategy (B1) — still useful, no longer urgent

The `build-llama-dll.yml` workflow + B2 (patching `llama-cpp-sys-2`'s build.rs
to link a prebuilt DLL) was conceived when Ninja-on-Windows was thought
intractable. With v0.5.14 the urgency is gone, but two values remain:

- **Smaller release ZIPs**: linking against a separately-published DLL means
  `eullm.exe` shrinks dramatically (the bulk of llama.cpp lives in the DLL).
- **Self-update path**: updating the DLL independently of the engine binary
  enables in-place llama.cpp upgrades without recompiling Rust.

Treat B1/B2 as a future feature, not a speed fix. The B1 run #1 artefact
(`llama-dll-windows-cuda-12.8.zip`, 122 MB) is already proven valid: 231
`llama_*` symbols exported, all 5 critical ones (`llama_backend_init`,
`llama_decode`, `llama_model_load_from_file`, `llama_model_load_from_splits`,
`llama_tokenize`), plus `bindings.rs` already produced by bindgen — meaning
B2 can skip the bindgen step entirely when it eventually lands.

## sccache resilience: keep S3 from killing the build

**The deeper lesson (process):** read a cache backend's *scoping, eviction,
and size-limit rules up front* before migrating. v0.5.11→v0.5.12 looked like
"GitHub cache fails on tags" because we populated the cache *under the v0.5.11
tag* instead of under `main` (tags can read main's cache; tag→tag is correctly
isolated for security). So the real misread was twofold:
(a) we mis-configured the populate side (should have run a populate workflow
on main), and (b) we then mis-attributed the failure to "ref-scoping is
fundamentally broken for tags" rather than to our own setup. The actual
disqualifier for the GitHub cache turned out to be the **10 GB per-repo cap
with LRU eviction** — too small to hold sccache's CUDA object set across
multiple platforms and releases.

Compounding all that, on the *engineering* side we missed
`CMAKE_CUDA_COMPILER_LAUNCHER` for half a dozen releases — that was the actual
long-pole on Linux. With the launcher present and S3 populated correctly,
Linux CUDA was 2m34s in v0.5.9. Two unrelated confounds (engineering + ops)
masked the value of S3 until v0.5.9.

**`SCCACHE_IDLE_TIMEOUT: "0"`** + a reachability probe before enabling the wrapper
stay (so an S3 blip degrades to a cache-less build instead of killing it).

## Three launcher vars, not two: CUDA needs sccache too

Setting only `CMAKE_C_COMPILER_LAUNCHER=sccache` + `CMAKE_CXX_COMPILER_LAUNCHER=sccache`
**caches C/C++ but silently leaves nvcc invocations uncached**. The heavy
CUDA kernel template instantiations (`fattn-vec-instance-*.cu`,
`template-instances/*.cu`, many per K/V cache type combination) compile
from scratch on every release. Result: sccache stats show 99% hit rate
(on C/C++ only) but wall-clock stays at cold-build values because the
actual long-pole is nvcc, not g++.

**Mandatory third var alongside the other two:**

```yaml
CMAKE_CUDA_COMPILER_LAUNCHER=sccache
```

Set it in every `Install sccache` step that's followed by a CUDA build —
both bash (Linux) and pwsh (Windows). Setting it on non-CUDA jobs is
harmless (CMake just doesn't reference it).

How to spot the issue from sccache stats: look at "Cache hits (C/C++)"
and "Cache hits (Rust)" — if there is no separate "Cache hits (CUDA)"
line and the long-pole build wall-clock is multi-hour, you forgot the
CUDA launcher. v0.5.7 burned this: 387 hits, 0.282s avg read, but
1h 41m Linux CUDA TQ wall-clock because nvcc bypassed the wrapper.
Fixed in v0.5.8.

The first run after enabling the CUDA launcher is still a cold build
(populates the cache), so the *real* speedup only shows on the run
*after* that.
