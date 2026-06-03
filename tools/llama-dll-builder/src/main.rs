//! B1 — produce a shared (DLL) build of llama.cpp.
//!
//! This binary does nothing at runtime. Building it forces `llama-cpp-sys-2`'s
//! build.rs to run; with `LLAMA_BUILD_SHARED_LIBS=1` set in the environment that
//! build compiles llama.cpp as shared libraries. The `build-llama-dll.yml`
//! workflow harvests the resulting `*.dll` / `*.lib` / headers / `bindings.rs`
//! from this crate's `OUT_DIR`.

fn main() {
    println!(
        "llama-dll-builder: link-checked the shared llama.cpp build. \
         DLLs + import libs are under target/<profile>/build/llama-cpp-sys-2-*/out/."
    );
}
