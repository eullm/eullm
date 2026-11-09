//! The machine the server runs on, once a second, for the Monitor page: CPU,
//! memory, what this process reads from disk, and each GPU.
//!
//! The sampler is a thread of its own, started by the first [`snapshot`] and
//! stopped when nobody has asked for [`IDLE`]: a server nobody watches pays
//! nothing for it. It keeps the last [`SAMPLES`] samples, in memory only.
//!
//! GPUs are read from what each vendor's driver offers, with no library linked
//! at build time:
//! - NVIDIA through NVML (`libnvidia-ml`, loaded at run time; Linux, Windows):
//!   load, VRAM, temperature, power, PCIe traffic;
//! - AMD through the `amdgpu` driver's files under `/sys/class/drm` (Linux):
//!   load, VRAM, temperature, power;
//! - any other GPU (Metal, Vulkan, Intel) through llama.cpp's own device list:
//!   VRAM only, and only when neither of the two above found a GPU.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

use super::now_ms;

/// How many samples are kept: two minutes, one a second.
pub const SAMPLES: usize = 120;
/// Between two samples.
const PERIOD: Duration = Duration::from_secs(1);
/// How long the sampler runs on after the last [`snapshot`].
const IDLE: Duration = Duration::from_secs(30);

/// The machine at one moment.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Sample {
    /// Milliseconds since 1970.
    pub t_ms: u64,
    /// Every core together, 0-100.
    pub cpu_pct: f32,
    pub cores: usize,
    pub ram_used: u64,
    pub ram_total: u64,
    /// This process's resident memory.
    pub process_ram: u64,
    /// What this process read from disk in the last second, in bytes per
    /// second: the model loading, or experts paged in through mmap.
    pub disk_read_bps: u64,
    pub gpus: Vec<Gpu>,
}

/// One GPU at one moment. What its driver does not say is `None`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Gpu {
    pub name: String,
    /// `nvml`, `amdgpu` or `ggml`: where the figures come from.
    pub source: &'static str,
    pub util_pct: Option<f32>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    pub temp_c: Option<f32>,
    pub power_w: Option<f32>,
    pub power_limit_w: Option<f32>,
    /// PCIe traffic, host to GPU and GPU to host, in bytes per second.
    pub pcie_rx_bps: Option<u64>,
    pub pcie_tx_bps: Option<u64>,
}

struct State {
    samples: VecDeque<Sample>,
    asked: Instant,
    running: bool,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<T>(f: impl FnOnce(&mut State) -> T) -> T {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(|| State {
        samples: VecDeque::with_capacity(SAMPLES),
        asked: Instant::now(),
        running: false,
    }))
}

/// The samples so far, oldest first, and start the sampler if it is not
/// running. The first call after a quiet spell returns what was left from the
/// last time (or nothing); the next sample comes within a second.
pub fn snapshot() -> Vec<Sample> {
    let start = with_state(|s| {
        s.asked = Instant::now();
        if s.running {
            false
        } else {
            // What is left is from before the quiet spell: a gap in the
            // chart, not a minute that looked idle.
            s.samples.clear();
            s.running = true;
            true
        }
    });
    if start {
        let spawned = std::thread::Builder::new()
            .name("eullm-monitor".into())
            .spawn(run);
        if spawned.is_err() {
            with_state(|s| s.running = false);
        }
    }
    with_state(|s| s.samples.iter().cloned().collect())
}

fn run() {
    let mut sampler = Sampler::new();
    loop {
        let started = Instant::now();
        let sample = sampler.sample();
        let go_on = with_state(|s| {
            s.samples.push_back(sample);
            while s.samples.len() > SAMPLES {
                s.samples.pop_front();
            }
            if s.asked.elapsed() > IDLE {
                s.running = false;
            }
            s.running
        });
        if !go_on {
            return;
        }
        std::thread::sleep(PERIOD.saturating_sub(started.elapsed()));
    }
}

struct Sampler {
    system: Option<sysinfo::System>,
    pid: Option<sysinfo::Pid>,
    last: Instant,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    nvml: Option<nvml_wrapper::Nvml>,
}

impl Sampler {
    fn new() -> Self {
        Self {
            system: sysinfo::System::new().ok(),
            pid: sysinfo::get_current_pid().ok(),
            last: Instant::now(),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            nvml: nvidia::open(),
        }
    }

    fn sample(&mut self) -> Sample {
        let mut sample = Sample {
            t_ms: now_ms(),
            ..Sample::default()
        };
        let seconds = self.last.elapsed().as_secs_f64().max(0.001);
        self.last = Instant::now();
        if let Some(system) = self.system.as_mut() {
            system.refresh_cpu_usage();
            system.refresh_memory();
            sample.cpu_pct = system.global_cpu_usage();
            sample.cores = system.cpus().len();
            sample.ram_used = system.used_memory();
            sample.ram_total = system.total_memory();
            if let Some(pid) = self.pid {
                system.refresh_processes_specifics(
                    sysinfo::ProcessesToUpdate::Some(&[pid]),
                    false,
                    sysinfo::ProcessRefreshKind::nothing()
                        .with_memory()
                        .with_disk_usage(),
                );
                if let Some(process) = system.process(pid) {
                    sample.process_ram = process.memory();
                    sample.disk_read_bps =
                        (process.disk_usage().read_bytes as f64 / seconds) as u64;
                }
            }
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(nvml) = self.nvml.as_ref() {
            sample.gpus.extend(nvidia::read(nvml));
        }
        #[cfg(target_os = "linux")]
        sample
            .gpus
            .extend(amd::read(std::path::Path::new("/sys/class/drm")));
        if sample.gpus.is_empty() {
            sample.gpus = ggml_devices();
        }
        sample
    }
}

/// The machine and what llama.cpp can run on, for the About page: read once
/// per call, nothing sampled.
#[derive(Debug, Clone, Serialize)]
pub struct Machine {
    pub os: Option<String>,
    pub kernel: Option<String>,
    pub arch: String,
    pub host: Option<String>,
    pub cpu: Option<String>,
    pub cores: usize,
    pub physical_cores: Option<usize>,
    pub ram_total: u64,
    /// When this process started, in milliseconds since 1970.
    pub started_ms: Option<u64>,
    /// Every device llama.cpp's backends offer, CPU included.
    pub devices: Vec<Device>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Device {
    pub name: String,
    pub description: String,
    /// `cpu`, `gpu`, `igpu` or `accel`.
    pub kind: &'static str,
    pub memory_total: Option<u64>,
}

pub fn machine() -> Machine {
    let mut system = sysinfo::System::new().ok();
    if let Some(s) = system.as_mut() {
        s.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
        s.refresh_memory();
    }
    let pid = sysinfo::get_current_pid().ok();
    let started_ms = system.as_mut().zip(pid).and_then(|(s, pid)| {
        s.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid]),
            false,
            sysinfo::ProcessRefreshKind::nothing(),
        );
        s.process(pid).map(|p| p.start_time() * 1000)
    });
    Machine {
        os: sysinfo::System::long_os_version().ok(),
        kernel: sysinfo::System::kernel_version().ok(),
        arch: sysinfo::System::cpu_arch(),
        host: sysinfo::System::host_name().ok(),
        cpu: system
            .as_ref()
            .and_then(|s| s.cpus().first())
            .map(|c| c.brand().trim().to_string())
            .filter(|b| !b.is_empty()),
        cores: system.as_ref().map_or(0, |s| s.cpus().len()),
        physical_cores: sysinfo::System::physical_core_count().ok(),
        ram_total: system.as_ref().map_or(0, |s| s.total_memory()),
        started_ms,
        devices: devices(),
    }
}

fn devices() -> Vec<Device> {
    use llama_cpp_sys_2::{
        GGML_BACKEND_DEVICE_TYPE_ACCEL, GGML_BACKEND_DEVICE_TYPE_CPU, GGML_BACKEND_DEVICE_TYPE_GPU,
        ggml_backend_dev_count, ggml_backend_dev_description, ggml_backend_dev_get,
        ggml_backend_dev_memory, ggml_backend_dev_name, ggml_backend_dev_type,
    };
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
    let mut out = Vec::new();
    // SAFETY: as in `ggml_devices`.
    unsafe {
        for i in 0..ggml_backend_dev_count() {
            let dev = ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let kind = match ggml_backend_dev_type(dev) {
                GGML_BACKEND_DEVICE_TYPE_CPU => "cpu",
                GGML_BACKEND_DEVICE_TYPE_GPU => "gpu",
                GGML_BACKEND_DEVICE_TYPE_ACCEL => "accel",
                _ => "igpu",
            };
            let (mut free, mut total) = (0usize, 0usize);
            if kind != "cpu" {
                ggml_backend_dev_memory(dev, &raw mut free, &raw mut total);
            }
            out.push(Device {
                name: text(ggml_backend_dev_name(dev)),
                description: text(ggml_backend_dev_description(dev)),
                kind,
                memory_total: (total > 0).then_some(total as u64),
            });
        }
    }
    out
}

/// Any GPU llama.cpp knows, with its VRAM: what Metal, Vulkan and the others
/// say, where no vendor tool is read.
fn ggml_devices() -> Vec<Gpu> {
    use llama_cpp_sys_2::{
        GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU, ggml_backend_dev_count,
        ggml_backend_dev_description, ggml_backend_dev_get, ggml_backend_dev_memory,
        ggml_backend_dev_type,
    };
    let mut gpus = Vec::new();
    // SAFETY: as in `fit::vram_bytes`: the registry is process-global, every
    // index below the count is valid, the description is a C string the
    // device owns, and the memory call writes two out-params we own.
    unsafe {
        for i in 0..ggml_backend_dev_count() {
            let dev = ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let kind = ggml_backend_dev_type(dev);
            if kind != GGML_BACKEND_DEVICE_TYPE_GPU && kind != GGML_BACKEND_DEVICE_TYPE_IGPU {
                continue;
            }
            let description = ggml_backend_dev_description(dev);
            let name = if description.is_null() {
                format!("GPU {i}")
            } else {
                std::ffi::CStr::from_ptr(description)
                    .to_string_lossy()
                    .into_owned()
            };
            let (mut free, mut total) = (0usize, 0usize);
            ggml_backend_dev_memory(dev, &raw mut free, &raw mut total);
            gpus.push(Gpu {
                name,
                source: "ggml",
                vram_used: (total > 0).then(|| total.saturating_sub(free) as u64),
                vram_total: (total > 0).then_some(total as u64),
                ..Gpu::default()
            });
        }
    }
    gpus
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod nvidia {
    use nvml_wrapper::Nvml;
    use nvml_wrapper::enum_wrappers::device::{PcieUtilCounter, TemperatureSensor};

    use super::Gpu;

    /// NVML, when the NVIDIA driver is installed: by the name the driver
    /// installs (`libnvidia-ml.so.1`, `nvml.dll`), then on Linux by the
    /// unversioned name some setups have instead.
    pub(super) fn open() -> Option<Nvml> {
        Nvml::init().ok().or_else(|| {
            cfg!(target_os = "linux")
                .then(|| {
                    Nvml::builder()
                        .lib_path(std::ffi::OsStr::new("libnvidia-ml.so"))
                        .init()
                        .ok()
                })
                .flatten()
        })
    }

    pub(super) fn read(nvml: &Nvml) -> Vec<Gpu> {
        let count = nvml.device_count().unwrap_or(0);
        (0..count)
            .filter_map(|i| nvml.device_by_index(i).ok().map(|d| (i, d)))
            .map(|(i, device)| {
                let memory = device.memory_info().ok();
                // NVML gives PCIe traffic in KB/s, power in milliwatts.
                let pcie = |counter| {
                    device
                        .pcie_throughput(counter)
                        .ok()
                        .map(|kb| u64::from(kb) * 1024)
                };
                Gpu {
                    name: device.name().unwrap_or_else(|_| format!("NVIDIA GPU {i}")),
                    source: "nvml",
                    util_pct: device.utilization_rates().ok().map(|u| u.gpu as f32),
                    vram_used: memory.as_ref().map(|m| m.used),
                    vram_total: memory.as_ref().map(|m| m.total),
                    temp_c: device
                        .temperature(TemperatureSensor::Gpu)
                        .ok()
                        .map(|t| t as f32),
                    power_w: device.power_usage().ok().map(|mw| mw as f32 / 1000.0),
                    power_limit_w: device
                        .enforced_power_limit()
                        .ok()
                        .map(|mw| mw as f32 / 1000.0),
                    pcie_rx_bps: pcie(PcieUtilCounter::Receive),
                    pcie_tx_bps: pcie(PcieUtilCounter::Send),
                }
            })
            .collect()
    }
}

#[cfg(any(target_os = "linux", test))]
mod amd {
    use std::fs;
    use std::path::Path;
    use std::sync::Mutex;

    use super::Gpu;

    const AMD_VENDOR: &str = "0x1002";

    fn number(path: &Path) -> Option<u64> {
        fs::read_to_string(path).ok()?.trim().parse().ok()
    }

    /// Every AMD GPU under `drm` (`/sys/class/drm`), in card order.
    pub(super) fn read(drm: &Path) -> Vec<Gpu> {
        let Ok(entries) = fs::read_dir(drm) else {
            return Vec::new();
        };
        let mut cards: Vec<(u32, std::path::PathBuf)> = entries
            .filter_map(Result::ok)
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                // `card0`, not the connectors `card0-DP-1`.
                let index = name.strip_prefix("card")?.parse().ok()?;
                Some((index, e.path().join("device")))
            })
            .filter(|(_, device)| {
                fs::read_to_string(device.join("vendor")).is_ok_and(|v| v.trim() == AMD_VENDOR)
            })
            .collect();
        cards.sort();
        cards
            .into_iter()
            .map(|(index, device)| card(index, &device))
            .collect()
    }

    /// Where distributions keep the PCI ID list (`hwdata`, `pciutils`).
    const PCI_IDS: [&str; 3] = [
        "/usr/share/hwdata/pci.ids",
        "/usr/share/misc/pci.ids",
        "/usr/share/pci.ids",
    ];

    /// The card's name from the system's PCI ID list, by its device id:
    /// consumer cards leave `product_name` empty. Looked up once per device
    /// id, since the list is a megabyte and the card is read every second.
    fn marketing_name(device: &Path) -> Option<String> {
        static NAMES: Mutex<Vec<(String, Option<String>)>> = Mutex::new(Vec::new());
        let id = fs::read_to_string(device.join("device")).ok()?;
        let id = id.trim().trim_start_matches("0x").to_lowercase();
        let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, name)) = names.iter().find(|(known, _)| *known == id) {
            return name.clone();
        }
        let name = PCI_IDS
            .iter()
            .find_map(|path| fs::read_to_string(path).ok())
            .and_then(|ids| pci_name(&ids, "1002", &id));
        names.push((id, name.clone()));
        name
    }

    /// The name `pci.ids` gives device `device` of vendor `vendor` (both
    /// lowercase hex, no `0x`): the marketing name in its brackets where it
    /// has one (`Navi 32 [Radeon RX 7700 XT / 7800 XT]`), else the chip.
    pub(super) fn pci_name(ids: &str, vendor: &str, device: &str) -> Option<String> {
        let mut lines = ids
            .lines()
            .skip_while(|l| !l.starts_with(&format!("{vendor}  ")));
        lines.next()?;
        let line = lines
            .take_while(|l| l.starts_with('\t') || l.starts_with('#') || l.is_empty())
            .find(|l| {
                l.strip_prefix('\t')
                    .is_some_and(|d| !d.starts_with('\t') && d.starts_with(&format!("{device}  ")))
            })?;
        let chip = line.trim_start()[device.len()..].trim();
        let name = match (chip.find('['), chip.rfind(']')) {
            (Some(a), Some(b)) if b > a => &chip[a + 1..b],
            _ => chip,
        };
        Some(if name.starts_with("AMD") {
            name.to_string()
        } else {
            format!("AMD {name}")
        })
    }

    fn card(index: u32, device: &Path) -> Gpu {
        // hwmon holds temperature in millidegrees and power in microwatts;
        // newer kernels name the average `power1_average`, some only
        // `power1_input`.
        let hwmon = fs::read_dir(device.join("hwmon"))
            .ok()
            .and_then(|mut d| d.find_map(|e| e.ok()))
            .map(|e| e.path());
        let hw = |file: &str| hwmon.as_ref().and_then(|h| number(&h.join(file)));
        let watts = |uw: u64| uw as f32 / 1e6;
        let name = fs::read_to_string(device.join("product_name"))
            .ok()
            .map(|n| n.trim().to_string())
            .filter(|n| !n.is_empty())
            .or_else(|| marketing_name(device))
            .unwrap_or_else(|| format!("AMD GPU {index}"));
        Gpu {
            name,
            source: "amdgpu",
            util_pct: number(&device.join("gpu_busy_percent")).map(|p| p as f32),
            vram_used: number(&device.join("mem_info_vram_used")),
            vram_total: number(&device.join("mem_info_vram_total")),
            temp_c: hw("temp1_input").map(|t| t as f32 / 1000.0),
            power_w: hw("power1_average")
                .or_else(|| hw("power1_input"))
                .map(watts),
            power_limit_w: hw("power1_cap").map(watts),
            pcie_rx_bps: None,
            pcie_tx_bps: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn amd_cards_are_read_from_their_sysfs_files() {
        let root = std::env::temp_dir().join(format!("eullm-monitor-drm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let card = root.join("card1/device");
        fs::create_dir_all(card.join("hwmon/hwmon3")).unwrap();
        fs::write(card.join("vendor"), "0x1002\n").unwrap();
        fs::write(card.join("gpu_busy_percent"), "42\n").unwrap();
        fs::write(card.join("mem_info_vram_used"), "1073741824\n").unwrap();
        fs::write(card.join("mem_info_vram_total"), "17179869184\n").unwrap();
        fs::write(card.join("hwmon/hwmon3/temp1_input"), "61000\n").unwrap();
        fs::write(card.join("hwmon/hwmon3/power1_input"), "215000000\n").unwrap();
        // A connector and a card of another vendor are not GPUs to list.
        fs::create_dir_all(root.join("card1-DP-1")).unwrap();
        fs::create_dir_all(root.join("card0/device")).unwrap();
        fs::write(root.join("card0/device/vendor"), "0x8086\n").unwrap();

        let gpus = amd::read(&root);
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(
            gpus,
            vec![Gpu {
                name: "AMD GPU 1".into(),
                source: "amdgpu",
                util_pct: Some(42.0),
                vram_used: Some(1 << 30),
                vram_total: Some(16 << 30),
                temp_c: Some(61.0),
                power_w: Some(215.0),
                power_limit_w: None,
                pcie_rx_bps: None,
                pcie_tx_bps: None,
            }]
        );
    }

    #[test]
    fn an_amd_card_is_named_from_the_pci_id_list() {
        let ids = "\
# comment
1000  Broadcom / LSI
\t73bf  Not this one
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t73bf  Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]
\t\t1002 0e3a  Radeon RX 6900 XT
\t7480  Navi 33 [Radeon RX 7600/7600 XT/7600M XT/7600S/7700S / PRO W7600]
\t74a1  Aqua Vanjaram [Instinct MI300X]
\t7448  Navi 31 [Radeon Pro W7900]
\t9999  Some Chip
1003  ULSI Systems
\t73df  Wrong vendor
";
        assert_eq!(
            amd::pci_name(ids, "1002", "73bf").as_deref(),
            Some("AMD Radeon RX 6800/6800 XT / 6900 XT")
        );
        assert_eq!(
            amd::pci_name(ids, "1002", "74a1").as_deref(),
            Some("AMD Instinct MI300X")
        );
        assert_eq!(
            amd::pci_name(ids, "1002", "9999").as_deref(),
            Some("AMD Some Chip")
        );
        assert_eq!(amd::pci_name(ids, "1002", "73df"), None);
        assert_eq!(amd::pci_name(ids, "1002", "0e3a"), None);
    }

    #[test]
    fn no_drm_directory_means_no_amd_gpu() {
        assert!(amd::read(std::path::Path::new("/nonexistent/eullm/drm")).is_empty());
    }

    #[test]
    fn the_about_page_reads_this_machine() {
        let m = machine();
        assert!(m.cores > 0 && m.ram_total > 0, "{m:?}");
        assert!(!m.arch.is_empty());
        // The CPU is always among llama.cpp's devices: ggml registers its
        // backends on first use, with no model or backend init needed.
        assert!(devices().iter().any(|d| d.kind == "cpu"), "{:?}", devices());
    }

    #[test]
    fn a_sample_reads_this_machine() {
        let mut sampler = Sampler::new();
        let sample = sampler.sample();
        assert!(sample.t_ms > 0);
        assert!(sample.ram_total > 0, "{sample:?}");
        assert!(sample.process_ram > 0, "{sample:?}");
        assert!(sample.cores > 0);
    }
}
