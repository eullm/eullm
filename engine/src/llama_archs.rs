//! Model architectures this build of llama.cpp can load.
//!
//! GENERATED FILE — do not edit by hand. Regenerate with
//! `tools/gen-llama-archs.sh` after bumping the llama.cpp submodule;
//! the engine CI job regenerates it and fails on any difference, so the
//! list cannot drift away from the code it describes.
//!
//! It is read out of `LLM_ARCH_NAMES` in the vendored
//! `src/llama-arch.cpp` because `llama.h` offers no way to enumerate
//! them at runtime. Deriving it from the source the binary is compiled
//! from is what makes it describe *this* build rather than some other.

/// Every `general.architecture` value this build understands, sorted.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "afmoe",
    "apertus",
    "arcee",
    "arctic",
    "arwkv7",
    "baichuan",
    "bailingmoe",
    "bailingmoe2",
    "bailingmoe3",
    "beam",
    "bert",
    "bitnet",
    "bloom",
    "chameleon",
    "chatglm",
    "clef",
    "clip",
    "codeshell",
    "cogvlm",
    "cohere2",
    "cohere2moe",
    "command-r",
    "dbrx",
    "deci",
    "deepseek",
    "deepseek2",
    "deepseek2-ocr",
    "deepseek32",
    "deepseek4",
    "deepseek41",
    "dflash",
    "dots1",
    "dots3note",
    "dream",
    "eagle3",
    "ernie4_5",
    "ernie4_5-moe",
    "eurobert",
    "exaone",
    "exaone-moe",
    "exaone4",
    "falcon",
    "falcon-h1",
    "flint",
    "gemma",
    "gemma-embedding",
    "gemma-embedding2",
    "gemma2",
    "gemma3",
    "gemma3n",
    "gemma4",
    "gemma4-assistant",
    "glm-dsa",
    "glm4",
    "glm4moe",
    "glm5-next",
    "gpt-oss",
    "gpt2",
    "gptj",
    "gptneox",
    "granite",
    "granite_swa",
    "granitehybrid",
    "granitemoe",
    "graniteswitch",
    "grok",
    "grovemoe",
    "hrm_text",
    "hunyuan-dense",
    "hunyuan-moe",
    "hunyuan_vl",
    "hy_v3",
    "hy_v4",
    "internlm2",
    "jais",
    "jais2",
    "jamba",
    "jina-bert-v2",
    "jina-bert-v3",
    "k2-horizon",
    "kimi-k3",
    "kimi-linear",
    "kolibri1",
    "laguna",
    "lfm2",
    "lfm2moe",
    "lfm3",
    "limite",
    "llada",
    "llada-moe",
    "llama",
    "llama-embed",
    "llama4",
    "maincoder",
    "maion-coder",
    "mamba",
    "mamba2",
    "maple",
    "mellum",
    "mimo2",
    "minicpm",
    "minicpm3",
    "minimax-01",
    "minimax-m2",
    "minimax-m3",
    "mistral-large4",
    "mistral3",
    "mistral4",
    "modern-bert",
    "mpt",
    "muse-glimmer",
    "muse-spark",
    "nanbeige",
    "nemotron",
    "nemotron_h",
    "nemotron_h_moe",
    "neo-bert",
    "nomic-bert",
    "nomic-bert-moe",
    "olmo",
    "olmo2",
    "olmoe",
    "openelm",
    "orion",
    "paddleocr",
    "pangu-embedded",
    "phi2",
    "phi3",
    "phimoe",
    "plamo",
    "plamo2",
    "plamo3",
    "plm",
    "pockettts",
    "qwen",
    "qwen2",
    "qwen2moe",
    "qwen2vl",
    "qwen3",
    "qwen35",
    "qwen35moe",
    "qwen3moe",
    "qwen3next",
    "qwen3tts",
    "qwen3vl",
    "qwen3vlmoe",
    "qwen4exp",
    "refact",
    "rnd1",
    "rwkv6",
    "rwkv6qwen2",
    "rwkv7",
    "seed_oss",
    "sensenova_u1",
    "smallthinker",
    "smollm3",
    "spark2_5",
    "stablelm",
    "starcoder",
    "starcoder2",
    "step35",
    "t5",
    "t5encoder",
    "talkie",
    "vllm-decision2",
    "wavtokenizer-dec",
    "xverse",
];

/// Whether this build can load a model declaring `arch`.
///
/// Callers must distinguish "declares an architecture we do not have"
/// from "declares none at all": only the first is a reason to tell a
/// user the model will not load.
pub fn is_supported(arch: &str) -> bool {
    SUPPORTED_ARCHITECTURES.binary_search(&arch).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_is_sorted_and_plausibly_long() {
        assert!(
            SUPPORTED_ARCHITECTURES.len() > 50,
            "only {} architectures — the generator's parse broke, and a short \
             list marks models unsupported that this build loads fine",
            SUPPORTED_ARCHITECTURES.len()
        );
        assert!(
            SUPPORTED_ARCHITECTURES.windows(2).all(|w| w[0] < w[1]),
            "not sorted, or holds a duplicate: `is_supported` binary-searches \
             and would answer wrongly"
        );
    }

    #[test]
    fn the_long_standing_architectures_are_there() {
        for arch in ["llama", "qwen3", "gemma3", "phi3"] {
            assert!(
                is_supported(arch),
                "{arch} missing — the parse read the wrong table"
            );
        }
        assert!(!is_supported(""));
        assert!(!is_supported("definitely-not-an-architecture"));
    }
}
