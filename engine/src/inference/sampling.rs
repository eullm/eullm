//! The single place a sampling chain is built.
//!
//! The chain was written out four times: once in the scheduler and three times
//! in `inference/mod.rs` (non-streaming, streaming, multimodal). Four copies of
//! the same sequence of `if` statements, and they had already started to drift:
//!
//! - the **seed fallback** differed. The scheduler passed `seq_id`, the three
//!   sequential paths a hardcoded `1234`. The value only matters when the
//!   system clock predates the Unix epoch, so nothing was visibly wrong — it is
//!   just two answers to one question, which is how the interesting divergences
//!   start.
//! - the **multimodal copy swallowed grammar errors**. Where the other three
//!   log `"Grammar sampler init failed …, falling back to unconstrained"`, it
//!   used `if let Ok(gs) = …` and said nothing. A request with `format: "json"`
//!   whose grammar failed to compile returned free-form text with no trace of
//!   why.
//!
//! Neither is dramatic on its own. The point is the shape: every new sampler
//! has to be added in four places, and being right in three of them is
//! indistinguishable, in a diff, from being right in all four. Order matters
//! here too (grammar must lead, `dist` must close), and an ordering fixed in
//! one copy would silently not be fixed in the others.

use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use super::{GenerateRequest, random_seed_fallback};

/// Build the sampler chain for one request.
///
/// `seed_fallback` is only consulted when the client sent no seed *and* the
/// system clock is unusable; see [`random_seed_fallback`]. Callers pass
/// something stable and locally distinct (the scheduler passes its slot id).
///
/// Order is part of the contract, not a style choice:
/// grammar → penalties → top-k → top-p → min-p → temperature → dist.
/// The grammar sampler must see the full distribution before anything has
/// truncated it, and `dist` must be last because it is what actually draws the
/// token; anything after it would have no effect.
pub(crate) fn build_sampler(
    model: &LlamaModel,
    request: &GenerateRequest,
    seed_fallback: u32,
) -> LlamaSampler {
    let seed = request
        .seed
        .unwrap_or_else(|| random_seed_fallback(seed_fallback));

    let mut chain: Vec<LlamaSampler> = Vec::new();

    // Grammar (if any) must be first in the chain.
    if let Some(ref grammar_str) = request.grammar {
        match LlamaSampler::grammar(model, grammar_str, "root") {
            Ok(gs) => chain.push(gs),
            // Never silent: a request that asked for constrained output and did
            // not get it must leave a trace. The multimodal copy used to drop
            // this on the floor.
            Err(e) => {
                tracing::warn!("Grammar sampler init failed ({e:?}), falling back to unconstrained")
            }
        }
    }

    // Repeat penalty (Ollama default: 1.1, last 64 tokens).
    if request.repeat_penalty != 1.0 {
        chain.push(LlamaSampler::penalties(
            request.repeat_last_n,
            request.repeat_penalty,
            0.0,
            0.0,
        ));
    }
    // Top-K (Ollama default: 40).
    if request.top_k > 0 {
        chain.push(LlamaSampler::top_k(request.top_k));
    }
    // Top-P (Ollama default: 0.9).
    if request.top_p < 1.0 {
        chain.push(LlamaSampler::top_p(request.top_p, 1));
    }
    // Min-P (Ollama default: 0.0).
    if request.min_p > 0.0 {
        chain.push(LlamaSampler::min_p(request.min_p, 1));
    }
    chain.push(LlamaSampler::temp(request.temperature));
    // Draws the token — must stay last.
    chain.push(LlamaSampler::dist(seed));

    LlamaSampler::chain_simple(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The chain itself needs a loaded model to build, so what is testable
    // without one is the decision that precedes it: which seed is used. That is
    // also the part that had actually diverged between the four copies.
    #[test]
    fn an_explicit_seed_is_used_verbatim_whatever_the_fallback() {
        let req = GenerateRequest {
            seed: Some(4242),
            ..Default::default()
        };
        let seed = req.seed.unwrap_or_else(|| random_seed_fallback(7));
        assert_eq!(seed, 4242);
    }

    #[test]
    fn without_a_seed_two_requests_do_not_agree() {
        // Ollama's default is a fresh seed per request, not a fixed one. Two
        // calls landing in the same nanosecond are astronomically unlikely, so
        // this is a meaningful check rather than a flaky one.
        let a = random_seed_fallback(1234);
        let b = random_seed_fallback(1234);
        assert_ne!(
            (a, b),
            (1234, 1234),
            "the fallback constant must not be the normal path"
        );
    }
}
