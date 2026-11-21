//! The `eullm` binary: the engine's command line, with nothing added. A
//! distribution that adds to it calls `eullm_engine::run_with_extensions`
//! instead (see `eullm_engine::extensions`).

fn main() {
    eullm_engine::main();
}
