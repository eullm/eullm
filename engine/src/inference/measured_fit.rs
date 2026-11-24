//! The split of a MoE model between RAM and VRAM, checked against what
//! llama.cpp measures before the load.
//!
//! `--fit` places the experts of a MoE model from the GGUF alone: weights and
//! KV cache are known from the file, the compute buffer is not, and was
//! charged a flat reserve (`fit::COMPUTE_BUFFER_RESERVE_BYTES`, plus
//! `fit::ubatch_reserve_bytes` for a larger micro-batch). With experts in RAM
//! the compute buffer also holds the copies of the experts a micro-batch
//! computes from, and that grows with the model and the micro-batch: Inkling
//! 2 Mini at 16,000 tokens asked 1,778 MiB of it at 512 tokens and 7,041 MiB
//! at 2,048, where the plan had kept 320 MiB and 1.4 GiB, and the load failed
//! on a 16 GB RTX 5070 Ti (24-11-2026).
//!
//! Here the scheduler, before it loads, asks llama.cpp what the load would
//! allocate on the GPU (`common_fit_measure`: the model and the context built
//! without allocating anything or reading the weights, a fraction of a
//! second), and moves the experts of whole layers between RAM and VRAM until
//! what it measures fits beside what has to stay free. Where the plan raised
//! the micro-batch itself (experts in RAM read prompts 2,048 tokens at a
//! time), the raised one is kept only if it does not cost more than a few
//! layers of experts in VRAM; otherwise the default's is taken.
//!
//! Only for a single GPU, without an expert cache and without a split of the
//! user's: the cache already measures itself (`grow_moe_cache`), and several
//! GPUs are balanced by `fit::balance_moe_split`.

use std::ffi::CString;
use std::num::NonZeroU32;

use super::InferenceConfig;

/// What a load sized by `--fit` with the experts of some layers in RAM gives
/// the scheduler, so that it can check the split against llama.cpp's own
/// numbers (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasuredFit {
    /// VRAM that stays free beside the text model besides `--vram-reserve`:
    /// companions, a projector going to the GPU, the MTP head's context.
    /// The compute buffer is not in it: that is what is measured.
    pub keep_bytes: u64,
    /// The micro-batch was raised by the plan, not asked for with
    /// `--n-ubatch`: it may come back down to the default.
    pub ubatch_raised: bool,
}

/// Left free beyond `--vram-reserve` for what the measure does not see: the
/// CUDA graphs and the library workspaces a first decode allocates.
const MARGIN_BYTES: u64 = 128 << 20;

/// Measures before giving up on a better split: each one builds the model and
/// the context without their memory, a fraction of a second on a large MoE.
const MAX_MEASURES: u32 = 8;

/// The split a measure settled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settled {
    /// The first `n_cpu_moe` layers keep their experts in RAM.
    pub n_cpu_moe: u32,
    /// What llama.cpp measured on the GPU for that split, in bytes.
    pub used: u64,
    /// `used` is within the budget. When no split fits, the one measured to
    /// take the least, with the fewest layers in RAM among those that take
    /// about as little.
    pub fits: bool,
}

/// Splits measured within this much of each other take the same memory: the
/// one with fewer layers in RAM is the better.
const SAME_BYTES: u64 = 64 << 20;

/// Settle `n_cpu_moe` from `start`: while the measure is over `budget`, the
/// experts of the next layers go to RAM, as many as cover what is over; while
/// it is under by whole layers, the experts of the last layers in RAM come
/// back. `expert_bytes[i]` is what layer `i`'s experts weigh, `measure(n)` is
/// what the GPU would hold with the first `n` layers' experts in RAM. The
/// result is the fewest layers in RAM measured to fit; `None` when a measure
/// failed, and the plan stands.
///
/// Experts sent to RAM do not always leave the GPU: their copies for a
/// micro-batch live in the compute buffer, and where a model reads the same
/// weights more than once in a pass (Inkling 2's loop) the copies of every
/// layer in RAM are kept until the last read. There the compute buffer grows
/// by what the model loses (0.53 GiB a layer on Inkling 2 Mini at 512 tokens,
/// 24-11-2026), and moving more layers only writes slower. When a step to
/// RAM frees less than half of what it moved, the walk stops there; when no
/// split fits, the one taking least is kept, with the fewest layers in RAM.
pub fn settle(
    start: u32,
    expert_bytes: &[u64],
    budget: u64,
    mut measure: impl FnMut(u32) -> Option<u64>,
) -> Option<Settled> {
    let n_layers = expert_bytes.len() as u32;
    let moved =
        |from: u32, to: u32| -> u64 { expert_bytes[from as usize..to as usize].iter().sum() };
    let mut n = start.min(n_layers);
    // The fewest layers in RAM measured to fit, and every split measured not
    // to; the last of these is where the walk to RAM stands.
    let mut best: Option<Settled> = None;
    let mut over: Vec<Settled> = Vec::new();
    for _ in 0..MAX_MEASURES {
        let used = measure(n)?;
        let next = if used > budget {
            // A step to RAM that freed less than half of what it moved: the
            // experts' copies took their place, and further steps would too.
            // Not the first layer sent to RAM, whose step also brings the
            // copies of one layer, which every model pays once.
            let pays = over
                .last()
                .filter(|o| o.n_cpu_moe > 0 && o.n_cpu_moe < n)
                .is_none_or(|o| o.used.saturating_sub(used) >= moved(o.n_cpu_moe, n) / 2);
            over.push(Settled {
                n_cpu_moe: n,
                used,
                fits: false,
            });
            if !pays {
                break;
            }
            let need = used - budget;
            let (mut freed, mut m) = (0u64, n);
            while m < n_layers && freed < need {
                freed += expert_bytes[m as usize];
                m += 1;
            }
            m
        } else {
            if best.is_none_or(|b| n < b.n_cpu_moe) {
                best = Some(Settled {
                    n_cpu_moe: n,
                    used,
                    fits: true,
                });
            }
            let slack = budget - used;
            let (mut back, mut m) = (0u64, n);
            while m > 0 && back + expert_bytes[m as usize - 1] <= slack {
                back += expert_bytes[m as usize - 1];
                m -= 1;
            }
            m
        };
        // Never back to a split measured not to fit, nor past one that does.
        let most_over = over.iter().map(|o| o.n_cpu_moe).max();
        let next = most_over.map_or(next, |o| next.max(o + 1)).min(n_layers);
        if next == n || best.is_some_and(|b| next >= b.n_cpu_moe) {
            break;
        }
        n = next;
    }
    if best.is_some() {
        return best;
    }
    let least = over.iter().map(|o| o.used).min()?;
    over.into_iter()
        .filter(|o| o.used <= least + SAME_BYTES)
        .min_by_key(|o| o.n_cpu_moe)
}

/// Whether the default micro-batch is the better of the two: the raised one
/// reads long prompts faster with experts in RAM, every layer of experts it
/// sends to RAM slows every token written. Kept while it costs at most an
/// eighth of the layers (two at least), and wherever it fits and the default
/// does not; dropped where it does not fit and the default does.
pub fn prefer_default_ubatch(raised: Settled, default: Settled, n_layers: u32) -> bool {
    match (raised.fits, default.fits) {
        (true, true) => raised.n_cpu_moe.saturating_sub(default.n_cpu_moe) > (n_layers / 8).max(2),
        (false, true) => true,
        (true, false) => false,
        // Neither fits: the smaller compute buffer is the better chance.
        (false, false) => true,
    }
}

/// What the measure changed: the split and micro-batch the scheduler loads
/// with, when they differ from the plan's.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub n_cpu_moe: u32,
    pub n_ubatch: u32,
    /// Several GPUs: the layers each one holds, in ggml's order of the GPUs
    /// (`--tensor-split` as counts of layers, the output layer last).
    pub tensor_split: Option<Vec<f32>>,
}

/// Measure the load `config` describes and settle its split (see the module
/// documentation). `gpu_layers` is what the load asks the GPU for
/// (`check_gpu_support`'s), and the context is the scheduler's: `ctx_size`
/// tokens over `n_seq_max` sequences with `n_outputs_max` logit rows. `None`
/// where nothing is measured or nothing changes.
pub(crate) fn run(
    config: &InferenceConfig,
    gpu_layers: i32,
    ctx_size: NonZeroU32,
    n_seq_max: u32,
    n_outputs_max: u32,
) -> Option<Measured> {
    let fit = config.measured_fit?;
    if !config.cpu_moe
        && config.n_cpu_moe == 0
        && config.moe_cache_bytes == 0
        && config.gpu_split.is_default()
        && gpu_layers < 0
    {
        let gpus = crate::fit::gpu_devices();
        if gpus.len() > 1 {
            return run_across_gpus(
                config,
                fit,
                &gpus,
                gpu_layers,
                ctx_size,
                n_seq_max,
                n_outputs_max,
            );
        }
    }
    if config.cpu_moe
        || config.n_cpu_moe == 0
        || config.moe_cache_bytes > 0
        || !config.gpu_split.is_default()
        || gpu_layers >= 0
    {
        return None;
    }
    let info = crate::fit::read_gguf_info(&config.model_path)?;
    let layout = crate::fit::read_model_moe_layout(&config.model_path, info.n_layers)?;
    let path = CString::new(config.model_path.to_str()?).ok()?;
    let n_layers = layout.expert_bytes_per_layer.len() as u32;
    let started = std::time::Instant::now();

    // What the GPU would hold, and its free memory, for a split and a
    // micro-batch. One GPU only: several are balanced elsewhere.
    let measures = std::cell::Cell::new(0u32);
    let measure = |n_cpu_moe: u32, n_ubatch: u32| -> Option<(u64, u64)> {
        let mut probe = config.clone();
        probe.n_cpu_moe = n_cpu_moe;
        probe.n_ubatch = n_ubatch;
        let cparams = super::build_ctx_params(&probe, ctx_size)
            .with_n_seq_max(n_seq_max)
            .with_n_outputs_max(n_outputs_max);
        let rows = super::with_text_model_params(&probe, gpu_layers, |params| {
            params.measure_device_memory(&path, &cparams)
        })
        .ok()??;
        measures.set(measures.get() + 1);
        let [row] = rows.as_slice() else {
            return None;
        };
        tracing::debug!(
            "Measured with the experts of {n_cpu_moe} layers in RAM, micro-batch {n_ubatch}: \
             model {}, context {}, compute {}",
            crate::fit::gib(row.model),
            crate::fit::gib(row.context),
            crate::fit::gib(row.compute)
        );
        Some((row.total(), u64::try_from(row.free).unwrap_or(0)))
    };

    let (planned_used, free) = measure(config.n_cpu_moe, config.n_ubatch)?;
    let budget = free
        .saturating_sub(config.vram_reserve)
        .saturating_sub(fit.keep_bytes)
        .saturating_sub(MARGIN_BYTES);
    // The plan's own split is the first thing `settle` measures: measured once.
    let planned = std::cell::Cell::new(Some(planned_used));
    let settle_at = |n_ubatch: u32| {
        settle(
            config.n_cpu_moe,
            &layout.expert_bytes_per_layer,
            budget,
            |n| {
                if n == config.n_cpu_moe
                    && n_ubatch == config.n_ubatch
                    && let Some(used) = planned.take()
                {
                    return Some(used);
                }
                measure(n, n_ubatch).map(|(used, _)| used)
            },
        )
    };
    let at_plan = settle_at(config.n_ubatch)?;
    let (settled, n_ubatch) = if fit.ubatch_raised && config.n_ubatch > super::DEFAULT_N_UBATCH {
        match settle_at(super::DEFAULT_N_UBATCH) {
            Some(default) if prefer_default_ubatch(at_plan, default, n_layers) => {
                (default, super::DEFAULT_N_UBATCH)
            }
            _ => (at_plan, config.n_ubatch),
        }
    } else {
        (at_plan, config.n_ubatch)
    };

    let elapsed = started.elapsed();
    tracing::info!(
        "Measured on the GPU: {} for the model, its context and compute buffer with the experts \
         of {} layers in RAM and a micro-batch of {n_ubatch}, against {} available ({} \
         measures, {:.1}s)",
        crate::fit::gib(settled.used),
        settled.n_cpu_moe,
        crate::fit::gib(budget),
        measures.get(),
        elapsed.as_secs_f64(),
    );
    if !settled.fits {
        tracing::warn!(
            "Measured on the GPU: no split of the experts fits beside the {} kept free; loading \
             with the experts of {} layers in RAM anyway",
            crate::fit::gib(config.vram_reserve + fit.keep_bytes),
            settled.n_cpu_moe
        );
    }
    if settled.n_cpu_moe == config.n_cpu_moe && n_ubatch == config.n_ubatch {
        return None;
    }
    let mut changes = Vec::new();
    if settled.n_cpu_moe != config.n_cpu_moe {
        changes.push(format!(
            "the experts of {} layers in RAM instead of {}",
            settled.n_cpu_moe, config.n_cpu_moe
        ));
    }
    if n_ubatch != config.n_ubatch {
        changes.push(format!(
            "prompts read {n_ubatch} tokens at a time instead of {} (--n-ubatch), whose compute \
             buffer would have kept more experts in RAM",
            config.n_ubatch
        ));
    }
    let line = format!("Measured placement: {}", changes.join("; "));
    tracing::info!("{line}");
    crate::monitor::history::event("info", line);
    Some(Measured {
        n_cpu_moe: settled.n_cpu_moe,
        n_ubatch,
        tensor_split: None,
    })
}

/// How fast a GPU reads its memory, in GB/s: the time of a matrix-vector
/// product over a 512 MiB f16 matrix, what writing a token mostly is.
/// Measured on the card itself, a few milliseconds, so that the faster of
/// two cards is known whatever they are. `None` when the card cannot run it.
fn read_speed(dev: llama_cpp_sys_2::ggml_backend_dev_t) -> Option<f64> {
    use llama_cpp_sys_2::{
        GGML_STATUS_SUCCESS, GGML_TYPE_F16, GGML_TYPE_F32, ggml_backend_alloc_ctx_tensors,
        ggml_backend_buffer_clear, ggml_backend_buffer_free, ggml_backend_dev_init,
        ggml_backend_free, ggml_backend_graph_compute, ggml_build_forward_expand, ggml_free,
        ggml_graph_overhead, ggml_init, ggml_init_params, ggml_mul_mat, ggml_new_graph,
        ggml_new_tensor_2d, ggml_tensor_overhead,
    };
    const K: i64 = 8192;
    const N: i64 = 32768;
    const REPS: u32 = 8;
    let bytes = (K * N * 2) as f64;
    // SAFETY: a backend, a context and a buffer made here and freed here, on
    // a device of the registry; the graph only reads tensors of the context.
    unsafe {
        let backend = ggml_backend_dev_init(dev, std::ptr::null());
        if backend.is_null() {
            return None;
        }
        let ctx = ggml_init(ggml_init_params {
            mem_size: ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        });
        if ctx.is_null() {
            ggml_backend_free(backend);
            return None;
        }
        let w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
        let x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        let y = ggml_mul_mat(ctx, w, x);
        let graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        let buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        let speed = if buffer.is_null() {
            None
        } else {
            ggml_backend_buffer_clear(buffer, 0);
            // The first runs pay for loading kernels and building graphs.
            let warm =
                (0..2).all(|_| ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            let started = std::time::Instant::now();
            let ran = warm
                && (0..REPS)
                    .all(|_| ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            let secs = started.elapsed().as_secs_f64();
            ggml_backend_buffer_free(buffer);
            (ran && secs > 0.0).then(|| bytes * f64::from(REPS) / secs / 1e9)
        };
        ggml_free(ctx);
        ggml_backend_free(backend);
        speed
    }
}

/// The name ggml gives a device (`CUDA0`, `Vulkan0`) and its description.
fn device_label(dev: llama_cpp_sys_2::ggml_backend_dev_t) -> String {
    let text = |p: *const std::ffi::c_char| {
        if p.is_null() {
            String::new()
        } else {
            // SAFETY: a C string the device owns for the life of the process.
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        }
    };
    // SAFETY: a device of the registry.
    unsafe {
        format!(
            "{} ({})",
            text(llama_cpp_sys_2::ggml_backend_dev_name(dev)),
            text(llama_cpp_sys_2::ggml_backend_dev_description(dev))
        )
    }
}

/// Place `n_slots` positions of a layer split (the layers that hold weights,
/// then the output layer) on the GPUs, the fastest first: `order` lists the
/// GPUs from the fastest, each takes as many layers as `measure` finds to fit
/// its `budgets` entry while the GPUs after it share the rest, and the
/// slowest takes what is left. `measure(counts)` is what each GPU would hold
/// with `counts[i]` positions on GPU `i`. The counts, and whether every GPU
/// was measured within its budget; `None` when a measure failed.
pub fn fastest_first(
    n_slots: u32,
    order: &[usize],
    budgets: &[u64],
    mut measure: impl FnMut(&[u32]) -> Option<Vec<u64>>,
) -> Option<(Vec<u32>, bool)> {
    let k = budgets.len();
    let mut counts = vec![0u32; k];
    let mut left = n_slots;
    // The positions the GPUs after `rank` get while `rank`'s are searched:
    // in proportion to their budgets, the rounding to the slowest.
    let share = |counts: &mut [u32], rank: usize, rest: u32| {
        let after = &order[rank + 1..];
        let total: u64 = after.iter().map(|&d| budgets[d]).sum::<u64>().max(1);
        let mut given = 0;
        for &d in &after[..after.len().saturating_sub(1)] {
            let c = (u64::from(rest) * budgets[d] / total) as u32;
            counts[d] = c;
            given += c;
        }
        if let Some(&last) = after.last() {
            counts[last] = rest - given;
        }
    };
    for (rank, &d) in order.iter().enumerate() {
        if rank + 1 == order.len() {
            counts[d] = left;
            break;
        }
        // The most positions on `d` measured within its budget.
        let (mut lo, mut hi, mut best) = (0u32, left, 0u32);
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let mut probe = counts.clone();
            probe[d] = mid;
            share(&mut probe, rank, left - mid);
            if measure(&probe)?[d] <= budgets[d] {
                best = mid;
                lo = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }
        counts[d] = best;
        left -= best;
    }
    let used = measure(&counts)?;
    let fits = used.iter().zip(budgets).all(|(u, b)| u <= b);
    Some((counts, fits))
}

/// Several GPUs (see `run`): each GPU's read speed is measured, and the
/// layers go to the fastest as far as llama.cpp measures them to fit there,
/// the rest to the next. llama.cpp's own split, by free memory, gives a
/// slower card as large a share as its memory, and every token waits on it.
fn run_across_gpus(
    config: &InferenceConfig,
    fit: MeasuredFit,
    gpus: &[llama_cpp_sys_2::ggml_backend_dev_t],
    gpu_layers: i32,
    ctx_size: NonZeroU32,
    n_seq_max: u32,
    n_outputs_max: u32,
) -> Option<Measured> {
    let info = crate::fit::read_gguf_info(&config.model_path)?;
    let path = CString::new(config.model_path.to_str()?).ok()?;
    let started = std::time::Instant::now();
    let speeds: Vec<f64> = gpus
        .iter()
        .map(|&dev| read_speed(dev).unwrap_or(0.0))
        .collect();
    let labels: Vec<String> = gpus.iter().map(|&dev| device_label(dev)).collect();
    tracing::info!(
        "GPU read speed: {}",
        labels
            .iter()
            .zip(&speeds)
            .map(|(l, s)| format!("{l} {s:.0} GB/s"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut order: Vec<usize> = (0..gpus.len()).collect();
    order.sort_by(|&a, &b| speeds[b].total_cmp(&speeds[a]));

    let measures = std::cell::Cell::new(0u32);
    let free: std::cell::RefCell<Option<Vec<u64>>> = std::cell::RefCell::new(None);
    let mut measure = |counts: &[u32]| -> Option<Vec<u64>> {
        let mut probe = config.clone();
        probe.gpu_split = crate::gpu_split::GpuSplit {
            mode: crate::gpu_split::SplitMode::default(),
            tensor_split: counts.iter().map(|&c| c as f32).collect(),
        };
        let cparams = super::build_ctx_params(&probe, ctx_size)
            .with_n_seq_max(n_seq_max)
            .with_n_outputs_max(n_outputs_max);
        let rows = super::with_text_model_params(&probe, gpu_layers, |params| {
            params.measure_device_memory(&path, &cparams)
        })
        .ok()??;
        measures.set(measures.get() + 1);
        if rows.len() != counts.len() {
            return None;
        }
        free.borrow_mut().get_or_insert_with(|| {
            rows.iter()
                .map(|r| u64::try_from(r.free).unwrap_or(0))
                .collect()
        });
        Some(rows.iter().map(|r| r.total()).collect())
    };
    // The free memory comes with a first measure, llama.cpp's own split.
    let n_slots = info.n_layers + 1;
    measure(&vec![0; gpus.len()])?;
    let budgets: Vec<u64> = free
        .borrow()
        .clone()?
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            // What stays free beside the model is kept on the first GPU,
            // where llama.cpp puts a projector and the companions.
            let keep = if i == 0 { fit.keep_bytes } else { 0 };
            f.saturating_sub(config.vram_reserve)
                .saturating_sub(keep)
                .saturating_sub(MARGIN_BYTES)
        })
        .collect();
    let (counts, fits) = fastest_first(n_slots, &order, &budgets, &mut measure)?;
    let split = labels
        .iter()
        .zip(&counts)
        .map(|(l, c)| format!("{c} on {l}"))
        .collect::<Vec<_>>()
        .join(", ");
    let line = format!(
        "Measured placement across the GPUs, the fastest first: {split} of {n_slots} layers \
         (the output layer counted) ({} measures, {:.1}s)",
        measures.get(),
        started.elapsed().as_secs_f64()
    );
    tracing::info!("{line}");
    crate::monitor::history::event("info", line);
    if !fits {
        tracing::warn!(
            "Measured on the GPUs: the layers do not fit beside the {} kept free on each; \
             loading with this split anyway",
            crate::fit::gib(config.vram_reserve)
        );
    }
    Some(Measured {
        n_cpu_moe: config.n_cpu_moe,
        n_ubatch: config.n_ubatch,
        tensor_split: Some(counts.iter().map(|&c| c as f32).collect()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// A GPU that holds `fixed` plus the experts of the layers not in RAM,
    /// plus `copy` for the experts in RAM a micro-batch computes from.
    fn gpu(expert_bytes: &[u64], fixed: u64, copy: u64) -> impl Fn(u32) -> u64 + '_ {
        move |n| {
            let on_gpu: u64 = expert_bytes.iter().skip(n as usize).sum();
            fixed + on_gpu + if n > 0 { copy } else { 0 }
        }
    }

    #[test]
    fn an_underestimated_compute_buffer_sends_more_experts_to_ram() {
        let layers = [GIB; 20];
        let used = gpu(&layers, 4 * GIB, 2 * GIB);
        // The plan's 8 layers in RAM measure 4 + 12 + 2 = 18 GiB against 14.
        let s = settle(8, &layers, 14 * GIB, |n| Some(used(n))).unwrap();
        assert_eq!(s.n_cpu_moe, 12);
        assert!(s.fits);
        assert_eq!(s.used, 14 * GIB);
    }

    #[test]
    fn an_overestimated_one_brings_experts_back() {
        let layers = [GIB; 20];
        let used = gpu(&layers, 4 * GIB, GIB / 2);
        let s = settle(12, &layers, 14 * GIB, |n| Some(used(n))).unwrap();
        // 4 + 9 + 0.5 = 13.5 fits; one more layer would be 14.5.
        assert_eq!(s.n_cpu_moe, 11);
        assert!(s.fits);
    }

    #[test]
    fn a_split_that_fits_is_left_as_it_is() {
        let layers = [GIB; 20];
        let mut calls = 0;
        let s = settle(10, &layers, 15 * GIB, |n| {
            calls += 1;
            Some(gpu(&layers, 4 * GIB, GIB / 2)(n))
        })
        .unwrap();
        // 14.5 measured, half a GiB short of a whole layer back.
        assert_eq!(s.n_cpu_moe, 10);
        assert_eq!(calls, 1);
    }

    #[test]
    fn no_split_that_fits_gives_every_layer_to_ram() {
        let layers = [GIB; 4];
        let s = settle(1, &layers, GIB, |n| Some(gpu(&layers, 3 * GIB, GIB)(n))).unwrap();
        assert_eq!(s.n_cpu_moe, 4);
        assert!(!s.fits);
    }

    #[test]
    fn a_failed_measure_leaves_the_plan() {
        let layers = [GIB; 4];
        assert_eq!(settle(2, &layers, GIB, |_| None), None);
    }

    #[test]
    fn it_never_goes_back_to_a_split_measured_not_to_fit() {
        // A compute buffer that grows by more than a layer once experts leave
        // the GPU: moving one back looks free and is not.
        let layers = [GIB; 10];
        let used = |n: u32| -> u64 {
            let on_gpu: u64 = layers.iter().skip(n as usize).sum();
            2 * GIB + on_gpu + if n > 0 { 3 * GIB } else { 0 }
        };
        let s = settle(0, &layers, 9 * GIB, |n| Some(used(n))).unwrap();
        assert!(s.fits);
        assert!(used(s.n_cpu_moe) <= 9 * GIB);
        assert!(used(s.n_cpu_moe - 1) > 9 * GIB);
    }

    #[test]
    fn experts_whose_copies_stay_on_the_gpu_are_not_sent_to_ram() {
        // Inkling 2 Mini at 512 tokens: every layer sent to RAM comes back as
        // the copy of its experts in the compute buffer, and the total stays
        // 0.16 GiB over the budget whatever the split.
        let layers = [GIB / 2; 40];
        let mut measured = Vec::new();
        let s = settle(6, &layers, 14 * GIB, |n| {
            measured.push(n);
            Some(14 * GIB + GIB / 6 - u64::from(n.saturating_sub(6)) * (1 << 20))
        })
        .unwrap();
        assert_eq!(
            s.n_cpu_moe, 6,
            "the plan's split, not the most layers in RAM"
        );
        assert!(!s.fits);
        assert_eq!(measured, [6, 7], "one step shows that moving frees nothing");
    }

    #[test]
    fn the_fastest_gpu_takes_every_layer_it_holds() {
        // GPU 0 slow with 13 GiB, GPU 1 fast with 14 GiB; 50 positions of
        // 0.5 GiB each, 1 GiB of compute buffer on each card.
        let budgets = [13 * GIB, 14 * GIB];
        let measure = |c: &[u32]| {
            Some(
                c.iter()
                    .map(|&n| GIB + u64::from(n) * (GIB / 2))
                    .collect::<Vec<_>>(),
            )
        };
        let (counts, fits) = fastest_first(50, &[1, 0], &budgets, measure).unwrap();
        assert_eq!(counts, [24, 26]);
        assert!(fits);
        // Where everything fits on the fast one, the slow one gets nothing.
        let (counts, _) = fastest_first(20, &[1, 0], &budgets, measure).unwrap();
        assert_eq!(counts, [0, 20]);
    }

    #[test]
    fn the_raised_micro_batch_goes_when_it_costs_many_layers() {
        let at = |n_cpu_moe, fits| Settled {
            n_cpu_moe,
            used: 0,
            fits,
        };
        // Inkling at 16,000 tokens: 5 GiB more compute buffer, many layers.
        assert!(prefer_default_ubatch(at(30, true), at(20, true), 40));
        // Two layers on a 40-layer model: worth it for long prompts.
        assert!(!prefer_default_ubatch(at(22, true), at(20, true), 40));
        assert!(prefer_default_ubatch(at(40, false), at(30, true), 40));
        assert!(!prefer_default_ubatch(at(30, true), at(40, false), 40));
    }
}
