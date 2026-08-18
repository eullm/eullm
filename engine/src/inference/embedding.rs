//! A second, independent model slot for text embeddings.
//!
//! Deliberately not built on `InferenceEngine`: that type exists to generate
//! text token by token, with a KV cache sized for a whole conversation and a
//! sampler chain behind it. An embedding request is one forward pass per
//! input, discards its own KV cache immediately, and produces a fixed-size
//! vector instead of a token stream. Reusing `InferenceEngine` would mean
//! carrying all of that machinery to switch it back off.
//!
//! Runs alongside the generation model on purpose — see
//! `AppState::ensure_embedding_model` for why keeping both resident (when
//! they fit) beats swapping a 20 GB LLM out for a 500 MB embedder and back on
//! every request.

use std::path::Path;
use std::pin::pin;

use llama_cpp_2::EmbeddingsError;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;

/// Default embedding context. A request to `/api/embed`/`/v1/embeddings` may
/// override it with `options.num_ctx`; a launch-time `--embedding-model`
/// (see `main.rs`) has no per-request body to read one from, so it uses this
/// directly. 2048 comfortably covers a single RAG chunk — the usual unit
/// these are called on — without paying for a window sized to the
/// generation model's much longer conversations.
pub const DEFAULT_EMBEDDING_CTX: u32 = 2048;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

/// A loaded embedding model: its own `LlamaBackend` and `LlamaModel`, kept
/// alive independently of whatever generation model is loaded in the main
/// slot. Each `embed()` call opens and drops its own `LlamaContext` — the
/// context is cheap relative to the model weights, and a short-lived context
/// means an embedding request never competes with a concurrent one for a
/// shared KV cache.
pub struct EmbeddingModel {
    backend: LlamaBackend,
    model: LlamaModel,
    threads: u32,
    /// Context window used to embed one input. Inputs longer than this are
    /// truncated (see `embed`) rather than rejected — the alternative is a
    /// hard error on the first oversized chunk an ingestion pipeline sends
    /// it, which is a worse failure mode than a documented truncation.
    n_ctx: u32,
}

impl EmbeddingModel {
    /// Load an embedding model fully onto the GPU (`gpu_layers = -1`) if any
    /// GPU backend is compiled in, CPU otherwise — mirrors
    /// `InferenceEngine::load`'s use of `check_gpu_support`. Embedding models
    /// are small enough (typically 100 MB-1 GB) that a partial CPU/GPU split
    /// is not worth the complexity `--fit` applies to a multi-gigabyte LLM;
    /// full offload or full CPU are the only two shapes this needs.
    pub fn load(
        path: &Path,
        threads: u32,
        n_ctx: u32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if !path.exists() {
            return Err(format!("Embedding model file not found: {}", path.display()).into());
        }

        let mut backend = LlamaBackend::init()?;
        backend.void_logs();

        let gpu_layers = crate::inference::check_gpu_support(-1);
        let model_params = if gpu_layers >= 0 {
            LlamaModelParams::default().with_n_gpu_layers(gpu_layers as u32)
        } else {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        };
        let model_params = pin!(model_params);

        tracing::info!("Loading embedding model: {}", path.display());
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| format!("Failed to load embedding model: {e}"))?;
        tracing::info!(
            "Embedding model loaded — dimension {}",
            model.n_embd()
        );

        Ok(Self {
            backend,
            model,
            threads,
            n_ctx,
        })
    }

    pub fn n_embd(&self) -> usize {
        usize::try_from(self.model.n_embd()).unwrap_or(0)
    }

    /// Embed each input text independently, returning one vector per input in
    /// the same order.
    ///
    /// Pooling is deliberately left as `Unspecified`: llama.cpp then reads
    /// the model's own declared pooling type from the GGUF (`hparams`) —
    /// CLS for BGE, mean for E5, and so on — falling back to `None` only for
    /// a model that declares nothing. Overriding it here would silently mask
    /// a real mismatch instead of using the pooling the model was trained
    /// with. When the resolved type genuinely is `None`, this falls back to
    /// mean-pooling the per-token embeddings itself, which is the standard
    /// substitute and better than refusing to answer.
    ///
    /// One text at a time, KV cache cleared between them. Not the fastest
    /// possible shape — batching independent texts into one multi-sequence
    /// decode call would amortize the fixed per-decode cost — but it is the
    /// simple, obviously-correct one, and ingestion throughput here is set
    /// by disk and chunking, not by this loop. Worth revisiting only if
    /// embedding is measured to be the bottleneck.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_ctx)
            .with_n_ubatch(self.n_ctx)
            .with_n_threads(self.threads as i32)
            .with_n_threads_batch(self.threads as i32)
            .with_n_seq_max(1)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Unspecified);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create embedding context: {e}"))?;

        let n_embd = self.n_embd();
        let mut out = Vec::with_capacity(texts.len());

        for text in texts {
            let mut tokens = self
                .model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| format!("Tokenization failed: {e}"))?;

            if tokens.len() as u32 > self.n_ctx {
                tracing::warn!(
                    "Embedding input truncated: {} tokens exceeds the {} the embedder was \
                     loaded with",
                    tokens.len(),
                    self.n_ctx
                );
                tokens.truncate(self.n_ctx as usize);
            }
            if tokens.is_empty() {
                out.push(vec![0.0; n_embd]);
                continue;
            }

            ctx.clear_kv_cache();
            let mut batch = LlamaBatch::new(tokens.len(), 1);
            for (i, token) in tokens.iter().enumerate() {
                // logits=true on every token, not just the last: mean/None
                // pooling need every token's embedding, and this matches
                // upstream's own embedding example (`batch_add_seq` in
                // examples/embedding/embedding.cpp) rather than the
                // last-token-only shape generation uses.
                batch
                    .add(*token, i as i32, &[0], true)
                    .map_err(|e| format!("Failed to build embedding batch: {e}"))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| format!("Embedding decode failed: {e}"))?;

            // Ask for the pooled sequence embedding first; the only way the
            // safe wrapper reports "this model's pooling resolved to NONE"
            // is that specific error coming back from the call itself (it
            // does not expose `llama_pooling_type(ctx)` to read up front —
            // see `EmbeddingsError::NonePoolType`), so that is the branch
            // this falls back on rather than predicting it beforehand.
            let mut vector = match ctx.embeddings_seq_ith(0) {
                Ok(v) => v.to_vec(),
                Err(EmbeddingsError::NonePoolType) => mean_pool(&ctx, tokens.len(), n_embd)?,
                Err(e) => return Err(format!("Failed to read embedding: {e}")),
            };

            // L2-normalize so callers can compare with a plain dot product —
            // the convention both the OpenAI and Ollama embedding endpoints
            // follow. Reranker (RANK-pooling) models are out of scope here:
            // normalizing their single relevance scalar would collapse it to
            // +-1 and lose the score entirely, so this endpoint is for
            // embedding models, not rerankers.
            normalize_l2(&mut vector);
            out.push(vector);
        }

        Ok(out)
    }
}

/// Mean-pool per-token embeddings when the model declares no pooling type of
/// its own. `embeddings_ith` returns a reference into the context that
/// borrows `ctx` immutably, so this takes `&LlamaContext` rather than being a
/// method that could conflict with the `&mut ctx` calls around it.
fn mean_pool(
    ctx: &llama_cpp_2::context::LlamaContext,
    n_tokens: usize,
    n_embd: usize,
) -> Result<Vec<f32>, String> {
    let mut sum = vec![0.0f32; n_embd];
    for i in 0..n_tokens {
        let token_embd = ctx
            .embeddings_ith(i as i32)
            .map_err(|e| format!("Failed to read per-token embedding {i}: {e}"))?;
        for (s, v) in sum.iter_mut().zip(token_embd) {
            *s += v;
        }
    }
    let n = n_tokens as f32;
    for s in &mut sum {
        *s /= n;
    }
    Ok(sum)
}

fn normalize_l2(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}
