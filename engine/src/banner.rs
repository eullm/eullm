//! The startup banner describing the model that is loaded and how.
//!
//! This lived inline in `cmd_run` and was printed by that command alone.
//! `cmd_serve` printed six lines and none of these, so `GPU backend`,
//! `CPU features`, `GPU layers`, `Context`, `KV cache` and `Threads` were
//! invisible to anyone driving the engine as a daemon — which is every
//! automated harness and everyone using it as a backend. The `CPU features`
//! line exists specifically to diagnose issue #140, and the people best placed
//! to report were the ones who could not see it; `tools/smoke_test.py` had to
//! start a second `run` process purely to capture it.
//!
//! It is the same class of problem as the mandatory `run`/`serve` flag parity
//! rule in `CLAUDE.md`, applied to diagnostic output instead of configuration,
//! and it has the same fix: one function, called from both paths, so the two
//! cannot drift.
//!
//! `serve` starts without a model, so it prints the endpoints immediately and
//! this block after each model load — the first one and every later swap.
//! A swap is rare and expensive (it moves weights into VRAM), so restating the
//! configuration is worth far more than the lines it costs, and after a swap
//! the context size and KV types may genuinely have changed.

use crate::inference::{self, KvCacheType};

/// Everything the banner reports about a loaded model.
///
/// A plain data struct rather than a long argument list: the caller in
/// `api::swap_model` has to fill exactly the same set as the one in `cmd_run`,
/// and a named field is the only version of that which stays readable.
pub struct ModelBanner {
    /// Display name, already stripped of any `eullm/` prefix.
    pub model_name: String,
    pub gpu_layers: i32,
    pub cpu_moe: bool,
    pub n_cpu_moe: u32,
    pub rs_seq: u32,
    pub ctx_checkpoints: usize,
    pub checkpoint_min_step: u32,
    /// 0 means the sequential engine; anything higher is the batching scheduler.
    pub batch_size: usize,
    pub ctx_size: u32,
    /// Context the model was trained for, 0 when it could not be read.
    pub n_ctx_train: u32,
    pub flash_attn: bool,
    pub cache_type_k: KvCacheType,
    pub cache_type_v: KvCacheType,
    pub kv_k_mib: f64,
    pub kv_v_mib: f64,
    pub web: bool,
    pub threads: u32,
    pub n_batch: u32,
    pub rust_debug: bool,
}

/// Which GPU backend was compiled into this binary.
///
/// A compile-time fact, so it is the same for every model and does not belong
/// in `ModelBanner`.
pub fn gpu_backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "rocm") {
        "ROCm"
    } else if cfg!(feature = "vulkan") {
        "Vulkan"
    } else if cfg!(feature = "metal") {
        "Metal"
    } else {
        "none (CPU only!)"
    }
}

impl ModelBanner {
    pub fn print(&self) {
        let mode = if self.batch_size > 0 {
            format!("continuous batching (max {} concurrent)", self.batch_size)
        } else {
            "sequential".to_string()
        };

        println!("  Model:         {}", self.model_name);
        println!("  GPU backend:   {}", gpu_backend_name());
        println!("  CPU features:  {}", inference::cpu_features_summary());
        if self.rust_debug {
            println!(
                "  Rust debug:    enabled (NaN/Inf logit check active — extra per-token cost)"
            );
        }
        println!(
            "  GPU layers:    {}",
            // Report what will actually be requested, not what was asked for.
            // A binary with no GPU backend offloads nothing (see
            // inference::check_gpu_support), and printing "all" here was the
            // same species of untruth as the warning box that said "all
            // inference will run on CPU" while 29 layers went to a Metal
            // device — issue #140.
            if !inference::has_gpu_backend() {
                "0 (no GPU backend compiled into this binary)".to_string()
            } else if self.gpu_layers < 0 {
                "all".to_string()
            } else {
                self.gpu_layers.to_string()
            }
        );
        if self.cpu_moe {
            println!("  CPU MoE:       enabled (expert tensors on CPU RAM)");
        } else if self.n_cpu_moe > 0 {
            println!(
                "  CPU MoE:       first {} layers (expert tensors on CPU RAM)",
                self.n_cpu_moe
            );
        }
        if self.rs_seq > 0 {
            println!(
                "  RS rollback:   {} (recurrent-state window for hybrid/SSM architectures)",
                self.rs_seq
            );
        }
        if self.ctx_checkpoints > 0 {
            println!(
                "  Checkpoints:   {} max, every {}+ new tokens (prompt-prefix restore)",
                self.ctx_checkpoints, self.checkpoint_min_step
            );
        }
        if self.batch_size > 0 {
            let per_seq = self.ctx_size / self.batch_size as u32;
            println!(
                "  Context:       {} total ({per_seq} per sequence × {} slots)",
                self.ctx_size, self.batch_size
            );
            // The continuous-batching scheduler splits ctx_size evenly across
            // slots, so a single conversation that builds up history can only
            // use ctx_size / batch_size tokens before hitting "does not fit".
            // Warn early when the per-sequence window is small enough to
            // surprise interactive REPL users.
            if self.batch_size > 1 && per_seq < 8192 {
                println!(
                    "  ⚠ per-sequence context is only {per_seq} tokens — long histories will fail."
                );
                let one_slot = self.ctx_size;
                let target_per_slot = 32768u32;
                let target_total = target_per_slot.saturating_mul(self.batch_size as u32);
                println!("    For single-chat use:   --batch-size 1   (full {one_slot} tokens)");
                println!(
                    "    For 32k per slot:      --ctx-size {target_total}   (= 32768 × {} slots)",
                    self.batch_size
                );
            }
        } else {
            println!("  Context:       {}", self.ctx_size);
        }
        // A window far below what the model was trained for is a silent
        // downgrade: the model still answers, just with far less history than
        // it can hold, and nothing on screen connects that to a flag. Reported
        // by a user whose editor plugin needed more than the 4096 default
        // (issue #286). Half is the threshold because a deliberate reduction
        // for memory is normal and should not be nagged at.
        if self.n_ctx_train > 0 && self.ctx_size < self.n_ctx_train / 2 {
            println!(
                "    this model was trained for {} — raise it with --ctx-size (costs KV memory)",
                self.n_ctx_train
            );
        }
        println!(
            "  Flash attn:    {} (auto-detect)",
            if self.flash_attn {
                "enabled"
            } else {
                "disabled"
            }
        );
        let k_name = inference::cache_type_display(&self.cache_type_k);
        let v_name = inference::cache_type_display(&self.cache_type_v);
        println!("  KV cache:      K={k_name} V={v_name}");
        if self.kv_k_mib > 0.0 || self.kv_v_mib > 0.0 {
            println!(
                "  KV memory:     K={:.0} MiB, V={:.0} MiB",
                self.kv_k_mib, self.kv_v_mib
            );
        }
        if self.web {
            println!("  Web browsing:  enabled (URLs in messages are fetched and injected)");
        }
        println!("  Threads:       {}", self.threads);
        println!("  Batch (prefill): {}", self.n_batch);
        println!("  Mode:          {mode}");
    }
}
