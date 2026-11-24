//! Whether Vulkan works in a build that carries it, and what to tell whoever
//! started the engine when it does not.
//!
//! Since patch `0017` the Vulkan loader (`libvulkan.so.1`, `vulkan-1.dll`) is
//! opened at run time instead of linked, so a build with Vulkan starts where
//! none is installed, and runs on its other backends. That is right for a
//! CUDA container on an NVIDIA server, where Vulkan has nothing to add. It is
//! not right to say so in one log line where Vulkan was the point: a card of
//! another vendor beside the NVIDIA one stays idle and the model runs slower
//! than it should, or, in a build whose only GPU backend is Vulkan, every
//! layer runs on the CPU. Someone trying the engine would then judge it by
//! that speed. So:
//!
//! - **Vulkan is the only GPU backend** and is missing: the engine stops, with
//!   the command that installs it, unless the CPU was asked for
//!   (`--gpu-layers 0`).
//! - **Vulkan beside CUDA or ROCm** and missing, with an AMD or Intel Arc card
//!   in the machine: a box on the terminal naming the idle card and the
//!   command, and the same in the Monitor's history.
//! - Missing with no such card (an NVIDIA-only machine): one line in the log.

use std::path::Path;

/// A warning found at start that the chat shows too (`GET /api/about`,
/// `warnings`): whoever opens the chat in the browser may never see the
/// terminal the box was printed on.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Notice {
    pub title: String,
    pub text: String,
    pub command: String,
}

static NOTICES: std::sync::Mutex<Vec<Notice>> = std::sync::Mutex::new(Vec::new());

/// The warnings found at start, for the chat.
pub fn notices() -> Vec<Notice> {
    NOTICES.lock().map(|n| n.clone()).unwrap_or_default()
}

/// What the Vulkan backend came to at start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vulkan {
    /// This build has no Vulkan backend.
    NotBuilt,
    /// Registered, with at least one device.
    Ready,
    /// Not usable: no loader on the system (`loader == false`), or a loader
    /// without a driver for any device (`loader == true`).
    Unavailable { loader: bool },
}

/// The state of the Vulkan backend in this process. Valid once the llama.cpp
/// backend is initialised (`init_shared_backend`), which registers it.
pub fn status() -> Vulkan {
    if !cfg!(feature = "vulkan") {
        return Vulkan::NotBuilt;
    }
    // SAFETY: the registry is a process-global the backends fill when they
    // load; the name is a NUL-terminated literal.
    let devices = unsafe {
        let reg = llama_cpp_sys_2::ggml_backend_reg_by_name(c"Vulkan".as_ptr());
        if reg.is_null() {
            0
        } else {
            llama_cpp_sys_2::ggml_backend_reg_dev_count(reg)
        }
    };
    if devices > 0 {
        Vulkan::Ready
    } else {
        Vulkan::Unavailable {
            loader: loader_present(),
        }
    }
}

/// Whether the system has a Vulkan loader the backend could have opened.
fn loader_present() -> bool {
    #[cfg(unix)]
    {
        let names: &[&std::ffi::CStr] = if cfg!(target_os = "macos") {
            &[c"libvulkan.1.dylib", c"libMoltenVK.dylib"]
        } else {
            &[c"libvulkan.so.1", c"libvulkan.so"]
        };
        names.iter().any(|name| {
            // SAFETY: dlopen with a NUL-terminated name; the handle is closed
            // at once, the loader stays loaded if ggml opened it too.
            unsafe {
                let handle = libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
                if handle.is_null() {
                    false
                } else {
                    libc::dlclose(handle);
                    true
                }
            }
        })
    }
    #[cfg(windows)]
    {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        Path::new(&root).join("System32").join("vulkan-1.dll").is_file()
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

/// A card Vulkan would have used and nothing else in this build can: an AMD
/// card with its own memory, or an Intel Arc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdleCard {
    pub name: String,
    pub vram_bytes: Option<u64>,
}

/// Below this an AMD device is the graphics of a Ryzen APU, which llama.cpp
/// leaves alone beside a real card: its memory is a carve-out of the RAM.
const DISCRETE_VRAM_BYTES: u64 = 4 << 30;

/// The cards of the machine only Vulkan can drive, read from the PCI devices
/// in `sysfs` (`/sys/bus/pci/devices` on Linux; nothing elsewhere, where the
/// warning names no card).
pub fn cards_needing_vulkan(sysfs_pci: &Path) -> Vec<IdleCard> {
    let Ok(entries) = std::fs::read_dir(sysfs_pci) else {
        return Vec::new();
    };
    let read = |dir: &Path, file: &str| {
        std::fs::read_to_string(dir.join(file))
            .ok()
            .map(|s| s.trim().to_string())
    };
    let mut cards: Vec<IdleCard> = entries
        .flatten()
        .filter_map(|entry| {
            let dir = entry.path();
            classify(
                &read(&dir, "class")?,
                &read(&dir, "vendor")?,
                &read(&dir, "device")?,
                read(&dir, "mem_info_vram_total").and_then(|v| v.parse().ok()),
            )
        })
        .collect();
    cards.sort_by(|a, b| a.name.cmp(&b.name));
    cards
}

/// Whether a PCI device (its `class`, `vendor` and `device` as `sysfs` writes
/// them, and amdgpu's `mem_info_vram_total`) is a card only Vulkan drives.
fn classify(class: &str, vendor: &str, device: &str, vram: Option<u64>) -> Option<IdleCard> {
    let hex = |s: &str| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok();
    // Display controllers: VGA (0x0300), 3D (0x0302), other (0x0380).
    let class = hex(class)? >> 8;
    if !matches!(class, 0x0300 | 0x0302 | 0x0380) {
        return None;
    }
    let device = hex(device)?;
    match hex(vendor)? {
        0x1002 if vram.is_some_and(|v| v >= DISCRETE_VRAM_BYTES) => Some(IdleCard {
            name: "AMD Radeon".to_string(),
            vram_bytes: vram,
        }),
        // Alchemist (0x56xx) and Battlemage (0xe2xx); every other Intel
        // device is integrated graphics.
        0x8086 if (0x5690..=0x56ff).contains(&device) || (0xe200..=0xe2ff).contains(&device) => {
            Some(IdleCard {
                name: "Intel Arc".to_string(),
                vram_bytes: vram,
            })
        }
        _ => None,
    }
}

/// The command that installs Vulkan here: the loader, or with `loader`
/// already there, the driver. From `/etc/os-release` on Linux.
fn install_hint(loader: bool) -> String {
    if cfg!(windows) {
        return "update the graphics driver of the card (AMD Adrenalin, Intel Arc): it \
                installs Vulkan"
            .to_string();
    }
    let release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
    let family = release
        .lines()
        .filter(|l| l.starts_with("ID=") || l.starts_with("ID_LIKE="))
        .map(|l| l.split('=').nth(1).unwrap_or("").trim_matches('"').to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let has = |id: &str| family.split_whitespace().any(|f| f == id);
    let (apt, dnf, pacman) = if loader {
        (
            "sudo apt install mesa-vulkan-drivers",
            "sudo dnf install mesa-vulkan-drivers",
            "sudo pacman -S vulkan-radeon vulkan-intel",
        )
    } else {
        (
            "sudo apt install libvulkan1 mesa-vulkan-drivers",
            "sudo dnf install vulkan-loader mesa-vulkan-drivers",
            "sudo pacman -S vulkan-icd-loader vulkan-radeon vulkan-intel",
        )
    };
    if has("debian") || has("ubuntu") {
        apt.to_string()
    } else if has("fedora") || has("rhel") || has("centos") {
        dnf.to_string()
    } else if has("arch") {
        pacman.to_string()
    } else {
        format!("{apt}  (Fedora/RHEL: {dnf})")
    }
}

/// What the engine does about Vulkan at start, from the state found, the
/// cards only Vulkan drives and the layers asked of the GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing to say beyond the log.
    Fine,
    /// A box on the terminal and an event in the Monitor: these lines.
    Warn(Vec<String>),
    /// Stop: these lines, then exit.
    Stop(Vec<String>),
}

/// The decision of [`check`], without reading the system.
pub fn verdict(
    vulkan: Vulkan,
    only_gpu_backend: bool,
    gpu_layers: i32,
    idle: &[IdleCard],
    hint: &str,
) -> Verdict {
    let Vulkan::Unavailable { loader } = vulkan else {
        return Verdict::Fine;
    };
    let missing = if loader {
        "Vulkan is installed but finds no graphics card (no driver for it, or no card)"
    } else {
        "Vulkan is not installed (no libvulkan)"
    };
    if only_gpu_backend {
        if gpu_layers == 0 {
            return Verdict::Fine;
        }
        return Verdict::Stop(vec![
            format!("{missing}."),
            "This EuLLM reaches the GPU through Vulkan only: without it every layer".to_string(),
            "would run on the CPU, many times slower.".to_string(),
            String::new(),
            format!("Install it:  {hint}"),
            "Or run on the CPU on purpose:  --gpu-layers 0".to_string(),
        ]);
    }
    if idle.is_empty() {
        return Verdict::Fine;
    }
    let cards = idle
        .iter()
        .map(|c| match c.vram_bytes {
            Some(v) => format!("{} ({:.0} GiB)", c.name, v as f64 / f64::from(1u32 << 30)),
            None => c.name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    Verdict::Warn(vec![
        format!("{missing}."),
        format!("This machine has {cards}, which EuLLM reaches through Vulkan:"),
        "it stays idle, and models larger than the other cards run slower than they".to_string(),
        "could, with part of them on the CPU.".to_string(),
        String::new(),
        format!("Install it:  {hint}"),
    ])
}

/// Check Vulkan once the backend is initialised, and say what the verdict
/// asks: `Err` with the lines to print before exiting when the engine must
/// not go on.
pub fn check(gpu_layers: i32) -> Result<(), Vec<String>> {
    let vulkan = status();
    let only_gpu_backend = cfg!(feature = "vulkan")
        && !cfg!(feature = "cuda")
        && !cfg!(feature = "rocm")
        && !cfg!(feature = "metal");
    let idle = if cfg!(target_os = "linux") {
        cards_needing_vulkan(Path::new("/sys/bus/pci/devices"))
    } else if matches!(vulkan, Vulkan::Unavailable { .. }) && cfg!(windows) {
        // No PCI listing here: name the kind of card, so the box is shown.
        vec![IdleCard {
            name: "an AMD or Intel card, if any".to_string(),
            vram_bytes: None,
        }]
    } else {
        Vec::new()
    };
    let hint = match vulkan {
        Vulkan::Unavailable { loader } => install_hint(loader),
        _ => String::new(),
    };
    match verdict(vulkan, only_gpu_backend, gpu_layers, &idle, &hint) {
        Verdict::Fine => {
            if let Vulkan::Unavailable { loader } = vulkan {
                tracing::info!(
                    "Vulkan unavailable ({}); no card here needs it",
                    if loader { "no driver" } else { "no loader" }
                );
            }
            Ok(())
        }
        Verdict::Warn(lines) => {
            print_box("WARNING: a graphics card is not used", &lines);
            crate::monitor::history::event("warning", lines.join(" "));
            if let Ok(mut notices) = NOTICES.lock() {
                notices.push(Notice {
                    title: "A graphics card is not used: Vulkan is missing".to_string(),
                    text: lines
                        .iter()
                        .filter(|l| !l.is_empty() && !l.starts_with("Install it:"))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" "),
                    command: hint,
                });
            }
            Ok(())
        }
        Verdict::Stop(lines) => Err(lines),
    }
}

/// `lines` in a box on stderr, as the engine's other start-up warnings.
pub fn print_box(title: &str, lines: &[String]) {
    let width = lines
        .iter()
        .map(|l| l.chars().count())
        .chain(std::iter::once(title.chars().count()))
        .max()
        .unwrap_or(0);
    let bar = "═".repeat(width + 4);
    eprintln!();
    eprintln!("╔{bar}╗");
    eprintln!("║  {title:<width$}  ║");
    eprintln!("║  {:<width$}  ║", "");
    for line in lines {
        eprintln!("║  {line:<width$}  ║");
    }
    eprintln!("╚{bar}╝");
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn only_cards_vulkan_alone_drives_are_counted() {
        // RX 6700 XT, 12 GiB.
        let radeon = classify("0x030000", "0x1002", "0x73df", Some(12 * GIB));
        assert_eq!(radeon.unwrap().name, "AMD Radeon");
        // A Ryzen APU's graphics: 512 MiB carved out of the RAM.
        assert_eq!(classify("0x030000", "0x1002", "0x1638", Some(GIB / 2)), None);
        // The NVIDIA card is CUDA's.
        assert_eq!(classify("0x030000", "0x10de", "0x2c05", None), None);
        // Intel: an Arc A770 yes, the UHD of the CPU no.
        assert!(classify("0x030000", "0x8086", "0x56a0", None).is_some());
        assert_eq!(classify("0x030000", "0x8086", "0x4680", None), None);
        // Not a display controller: an AMD SATA controller.
        assert_eq!(classify("0x010601", "0x1002", "0x7901", None), None);
    }

    #[test]
    fn the_pci_devices_are_read_from_sysfs() {
        let root = std::env::temp_dir().join(format!("eullm-sysfs-{}", std::process::id()));
        let dev = |name: &str, files: &[(&str, &str)]| {
            let d = root.join(name);
            std::fs::create_dir_all(&d).unwrap();
            for (f, v) in files {
                std::fs::write(d.join(f), format!("{v}\n")).unwrap();
            }
        };
        dev("0000:01:00.0", &[("class", "0x030000"), ("vendor", "0x10de"), ("device", "0x2c05")]);
        dev(
            "0000:05:00.0",
            &[
                ("class", "0x030000"),
                ("vendor", "0x1002"),
                ("device", "0x73df"),
                ("mem_info_vram_total", "12868124672"),
            ],
        );
        let cards = cards_needing_vulkan(&root);
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].vram_bytes, Some(12868124672));
        assert!(cards_needing_vulkan(Path::new("/nonexistent")).is_empty());
    }

    #[test]
    fn the_engine_stops_only_where_vulkan_was_its_only_gpu() {
        let none = Vulkan::Unavailable { loader: false };
        let radeon = [IdleCard {
            name: "AMD Radeon".into(),
            vram_bytes: Some(12 * GIB),
        }];
        assert!(matches!(verdict(none, true, -1, &[], "x"), Verdict::Stop(_)));
        assert_eq!(verdict(none, true, 0, &[], "x"), Verdict::Fine, "the CPU asked for");
        // Beside CUDA: a box where a card would be idle, nothing elsewhere.
        assert!(matches!(verdict(none, false, -1, &radeon, "x"), Verdict::Warn(_)));
        assert_eq!(verdict(none, false, -1, &[], "x"), Verdict::Fine);
        assert_eq!(verdict(Vulkan::Ready, true, -1, &radeon, "x"), Verdict::Fine);
        assert_eq!(verdict(Vulkan::NotBuilt, false, -1, &radeon, "x"), Verdict::Fine);
        let Verdict::Warn(lines) = verdict(none, false, -1, &radeon, "sudo apt install x") else {
            panic!()
        };
        assert!(lines.iter().any(|l| l.contains("AMD Radeon (12 GiB)")));
        assert!(lines.iter().any(|l| l.contains("sudo apt install x")));
    }
}
