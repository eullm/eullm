//! `GET /metrics` in Prometheus's text format: what the Monitor shows, for a
//! Prometheus server (and Grafana behind it) that watches several machines.
//!
//! Gauges are the newest one-second sample; the counters count from the
//! server's start, as Prometheus expects (`rate()` turns them into speeds and
//! copes with a restart). Who may read it is `api::admin`'s to say.

use std::fmt::Write;

use super::Totals;
use super::system::Sample;

/// The content type Prometheus's text format is served with.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// The page: `sample` the newest one (none before the first second),
/// `models` the generation models in memory, `history_bytes` what the
/// Monitor's history takes on disk.
pub fn render(sample: Option<&Sample>, totals: &Totals, models: &[String], history_bytes: u64) -> String {
    let mut out = String::new();
    let mut metric = |name: &str, kind: &str, help: &str, values: &[(String, f64)]| {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {kind}");
        for (labels, value) in values {
            let _ = writeln!(out, "{name}{labels} {value}");
        }
    };
    let one = |v: f64| vec![(String::new(), v)];

    metric(
        "eullm_build_info",
        "gauge",
        "The running eullm version.",
        &[(labels(&[("version", env!("CARGO_PKG_VERSION"))]), 1.0)],
    );
    metric(
        "eullm_generated_tokens_total",
        "counter",
        "Answer tokens written since the server started.",
        &one(totals.generated as f64),
    );
    metric(
        "eullm_prompt_tokens_total",
        "counter",
        "Prompt tokens read since the server started, counted when each answer ends.",
        &one(totals.prompt as f64),
    );
    metric(
        "eullm_requests_total",
        "counter",
        "Requests ended since the server started, answered or failed (an error, or the client left).",
        &[
            (labels(&[("result", "done")]), totals.done as f64),
            (labels(&[("result", "failed")]), totals.failed as f64),
        ],
    );
    metric(
        "eullm_model_loaded",
        "gauge",
        "1 for each generation model in memory.",
        &models
            .iter()
            .map(|m| (labels(&[("model", m)]), 1.0))
            .collect::<Vec<_>>(),
    );
    metric(
        "eullm_history_disk_bytes",
        "gauge",
        "What the Monitor's history takes on disk.",
        &one(history_bytes as f64),
    );

    let Some(s) = sample else {
        return out;
    };
    metric("eullm_requests_active", "gauge", "Requests running.", &one(f64::from(s.active)));
    metric(
        "eullm_requests_reading",
        "gauge",
        "Requests running that are not yet writing their answer.",
        &one(f64::from(s.reading)),
    );
    metric(
        "eullm_generation_tokens_per_second",
        "gauge",
        "Answer tokens per second in the last second, every request together.",
        &one(s.gen_tps),
    );
    metric("eullm_cpu_usage_percent", "gauge", "Every core together, 0-100.", &one(f64::from(s.cpu_pct)));
    metric("eullm_cpu_cores", "gauge", "Logical cores.", &one(s.cores as f64));
    metric("eullm_memory_used_bytes", "gauge", "Memory used on the machine.", &one(s.ram_used as f64));
    metric("eullm_memory_total_bytes", "gauge", "The machine's memory.", &one(s.ram_total as f64));
    metric(
        "eullm_process_resident_memory_bytes",
        "gauge",
        "This process's resident memory.",
        &one(s.process_ram as f64),
    );
    metric(
        "eullm_process_disk_read_bytes_per_second",
        "gauge",
        "What this process read from disk in the last second.",
        &one(s.disk_read_bps as f64),
    );

    // One series per GPU and figure, only where its driver says it.
    let gpu = |pick: &dyn Fn(&super::system::Gpu) -> Option<f64>| -> Vec<(String, f64)> {
        s.gpus
            .iter()
            .enumerate()
            .filter_map(|(i, g)| {
                pick(g).map(|v| (labels(&[("gpu", &i.to_string()), ("name", &g.name)]), v))
            })
            .collect()
    };
    let gpus = [
        (
            "eullm_gpu_utilization_percent",
            "GPU load, 0-100.",
            gpu(&|g| g.util_pct.map(f64::from)),
        ),
        ("eullm_gpu_memory_used_bytes", "VRAM used.", gpu(&|g| g.vram_used.map(|v| v as f64))),
        ("eullm_gpu_memory_total_bytes", "VRAM.", gpu(&|g| g.vram_total.map(|v| v as f64))),
        (
            "eullm_gpu_temperature_celsius",
            "GPU temperature.",
            gpu(&|g| g.temp_c.map(f64::from)),
        ),
        ("eullm_gpu_power_watts", "GPU power draw.", gpu(&|g| g.power_w.map(f64::from))),
        (
            "eullm_gpu_power_limit_watts",
            "GPU power limit.",
            gpu(&|g| g.power_limit_w.map(f64::from)),
        ),
        (
            "eullm_gpu_pcie_rx_bytes_per_second",
            "PCIe traffic, host to GPU.",
            gpu(&|g| g.pcie_rx_bps.map(|v| v as f64)),
        ),
        (
            "eullm_gpu_pcie_tx_bytes_per_second",
            "PCIe traffic, GPU to host.",
            gpu(&|g| g.pcie_tx_bps.map(|v| v as f64)),
        ),
    ];
    for (name, help, values) in gpus {
        if !values.is_empty() {
            metric(name, "gauge", help, &values);
        }
    }
    out
}

/// `{a="x",b="y"}`, the values escaped as the format asks.
fn labels(pairs: &[(&str, &str)]) -> String {
    let inner: Vec<String> = pairs
        .iter()
        .map(|(k, v)| {
            let v = v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
            format!("{k}=\"{v}\"")
        })
        .collect();
    format!("{{{}}}", inner.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::system::Gpu;

    #[test]
    fn the_page_has_the_counters_the_gauges_and_each_gpu() {
        let sample = Sample {
            cpu_pct: 12.5,
            active: 2,
            gen_tps: 88.0,
            gpus: vec![Gpu {
                name: "RTX \"5070\" Ti".into(),
                util_pct: Some(97.0),
                vram_used: Some(1024),
                ..Gpu::default()
            }],
            ..Sample::default()
        };
        let totals = Totals {
            generated: 500,
            prompt: 9000,
            done: 7,
            failed: 1,
        };
        let page = render(Some(&sample), &totals, &["gemma-4-12b".into()], 4096);
        for line in [
            "# TYPE eullm_generated_tokens_total counter",
            "eullm_generated_tokens_total 500",
            "eullm_requests_total{result=\"failed\"} 1",
            "eullm_model_loaded{model=\"gemma-4-12b\"} 1",
            "eullm_history_disk_bytes 4096",
            "eullm_requests_active 2",
            "eullm_cpu_usage_percent 12.5",
            "eullm_gpu_utilization_percent{gpu=\"0\",name=\"RTX \\\"5070\\\" Ti\"} 97",
            "eullm_gpu_memory_used_bytes{gpu=\"0\",name=\"RTX \\\"5070\\\" Ti\"} 1024",
        ] {
            assert!(page.lines().any(|l| l == line), "{line} missing in:\n{page}");
        }
        // A figure the driver does not give has no series at all.
        assert!(!page.contains("eullm_gpu_temperature_celsius"));
    }

    #[test]
    fn before_the_first_sample_only_the_counters() {
        let page = render(None, &Totals::default(), &[], 0);
        assert!(page.contains("eullm_requests_total"));
        assert!(!page.contains("eullm_cpu_usage_percent"));
    }
}
