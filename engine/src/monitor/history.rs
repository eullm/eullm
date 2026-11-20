//! The Monitor's history: what the machine and the engine did, minute by
//! minute, kept on disk for as many days as the operator chooses.
//!
//! The sampler ([`super::system`]) runs from the moment the server starts and
//! hands every one-second sample to a [`Recorder`]. At the end of each minute
//! the recorder writes one row with the minimum, the mean and the maximum of
//! every figure (CPU, memory, disk reads, each GPU's load, VRAM, temperature
//! and power, tokens per second, requests running), so a peak at 03:12 is
//! still there a month later even though the seconds around it are not.
//! What happened besides the figures (the server starting, a model loaded or
//! unloaded, a request that failed, a fallback the engine took) is written
//! as an event, the moment it happens.
//!
//! Rows and events go to one JSON Lines file a day, named by its UTC date
//! (`2026-11-20.jsonl`), in `EULLM_METRICS_DIR` (`~/.eullm/metrics` by
//! default; `off` keeps everything in memory only). A day with one GPU is
//! about 1 MB. Files older than the retention are deleted when the server
//! starts and every day after: 30 days unless `EULLM_METRICS_RETENTION_DAYS`
//! says otherwise, up to [`MAX_RETENTION_DAYS`], and the Monitor page can
//! change it (`PUT /api/monitor/storage`), which is kept in the folder's
//! `settings.json` and wins over the environment from then on.
//!
//! The last day of rows and the latest events are kept in memory as well,
//! so the history still answers when the folder cannot be written. Nothing
//! leaves the machine: the files are read by `GET /api/monitor/history`,
//! behind the same perimeter as the rest of `/api`.

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use chrono::{DateTime, Days, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use super::system::Sample;

/// Days kept when nobody chose.
pub const DEFAULT_RETENTION_DAYS: u32 = 30;
/// The longest retention that can be asked for: a year.
pub const MAX_RETENTION_DAYS: u32 = 366;
/// Where a retention chosen from the Monitor page is kept, in the folder.
const SETTINGS_FILE: &str = "settings.json";
/// Rows kept in memory: a day.
const MEMORY_MINUTES: usize = 24 * 60;
/// Events kept in memory.
const MEMORY_EVENTS: usize = 500;
/// The most events one history answer carries: the newest.
const MAX_EVENTS: usize = 1000;
/// The most rows one history answer carries, when no step is asked for.
const TARGET_ROWS: u64 = 720;
const MINUTE_MS: u64 = 60_000;

/// Minimum, mean and maximum of a figure over a minute (or a longer step).
pub type Trio = [f64; 3];

/// One GPU over a minute. What its driver does not say is absent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuRow {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub util: Option<Trio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram: Option<Trio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp: Option<Trio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<Trio>,
}

/// The machine and the engine over one minute, or over a longer step when
/// rows are merged for a long range.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MinuteRow {
    /// The start of the minute (or step), in milliseconds since 1970.
    pub t: u64,
    /// How many one-second samples it was made from.
    pub n: u32,
    /// Every core together, 0-100.
    pub cpu: Trio,
    /// Memory used on the machine, and by this process, in bytes.
    pub ram: Trio,
    pub ram_total: u64,
    pub proc_ram: Trio,
    /// What this process read from disk, in bytes per second.
    pub disk_read: Trio,
    /// Answer tokens per second, every request together.
    pub gen_tps: Trio,
    /// Requests running, and those of them still reading their prompt.
    pub active: Trio,
    pub reading: Trio,
    /// Answer and prompt tokens of the requests, and requests answered and
    /// failed, in the minute.
    pub gen_tokens: u64,
    pub prompt_tokens: u64,
    pub done: u64,
    pub failed: u64,
    #[serde(default)]
    pub gpus: Vec<GpuRow>,
}

/// Something that happened, beside the figures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// When, in milliseconds since 1970.
    pub t: u64,
    /// `start`, `model_loaded`, `model_unloaded`, `request_failed`, `warning`.
    pub kind: String,
    pub message: String,
}

/// A line of a day's file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Line {
    Minute(MinuteRow),
    Event(Event),
}

/// What `GET /api/monitor/history` answers.
#[derive(Debug, Clone, Serialize)]
pub struct History {
    pub from: u64,
    pub to: u64,
    /// The length of a row, in seconds.
    pub step: u64,
    /// Oldest first; a minute with no sample (the server was not running)
    /// has no row.
    pub rows: Vec<MinuteRow>,
    /// Oldest first, the newest [`MAX_EVENTS`] of the range.
    pub events: Vec<Event>,
}

/// What `GET /api/monitor/storage` answers.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// The folder, or none when the history is kept in memory only.
    pub dir: Option<String>,
    pub dir_source: String,
    pub retention_days: u32,
    pub retention_source: String,
    pub max_retention_days: u32,
    /// Every day's file together, in bytes.
    pub total_bytes: u64,
    /// What the retention would take at the size of the days kept so far.
    pub projected_bytes: u64,
    /// Each day's file, oldest first.
    pub days: Vec<DaySize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DaySize {
    pub date: String,
    pub bytes: u64,
}

/// What a deletion freed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Freed {
    pub files: u32,
    pub bytes: u64,
}

/// Where the history goes and for how long, as resolved at startup.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub dir: Option<PathBuf>,
    pub dir_source: String,
    pub days: u32,
    pub days_source: String,
    /// What was set but could not be used, for the startup log.
    pub warnings: Vec<String>,
}

/// The folder: `setting` is `EULLM_METRICS_DIR` and where it was found, the
/// default `~/.eullm/metrics`; `off` (or `none`, `false`, `0`) keeps the
/// history in memory only. Pure, so the precedence is tested without the
/// process environment.
pub fn resolve_dir(setting: Option<(&str, &str)>, home: &Path) -> (Option<PathBuf>, String) {
    match setting
        .map(|(v, s)| (v.trim(), s))
        .filter(|(v, _)| !v.is_empty())
    {
        Some((v, source))
            if matches!(
                v.to_ascii_lowercase().as_str(),
                "off" | "none" | "false" | "0"
            ) =>
        {
            (
                None,
                format!("EULLM_METRICS_DIR={v} ({source}): memory only"),
            )
        }
        Some((v, source)) => (
            Some(PathBuf::from(v)),
            format!("EULLM_METRICS_DIR ({source})"),
        ),
        None => (
            Some(home.join(".eullm").join("metrics")),
            "default".to_string(),
        ),
    }
}

/// The retention: what the Monitor page saved (`saved`, the folder's
/// `settings.json`) wins, then `EULLM_METRICS_RETENTION_DAYS` (`env`, with
/// where it was found), then [`DEFAULT_RETENTION_DAYS`]. A value outside
/// 1..=[`MAX_RETENTION_DAYS`] is not used, and said.
pub fn resolve_days(
    env: Option<(&str, &str)>,
    saved: Option<&str>,
    warnings: &mut Vec<String>,
) -> (u32, String) {
    if let Some(text) = saved {
        match serde_json::from_str::<Saved>(text)
            .ok()
            .and_then(|s| valid_days(s.retention_days))
        {
            Some(days) => return (days, format!("set from the Monitor page ({SETTINGS_FILE})")),
            None => warnings.push(format!(
                "{SETTINGS_FILE} in the metrics folder holds no usable retention_days \
                 (1-{MAX_RETENTION_DAYS}); ignoring it"
            )),
        }
    }
    if let Some((value, source)) = env
        .map(|(v, s)| (v.trim(), s))
        .filter(|(v, _)| !v.is_empty())
    {
        match value.parse::<u64>().ok().and_then(valid_days) {
            Some(days) => return (days, format!("EULLM_METRICS_RETENTION_DAYS ({source})")),
            None => warnings.push(format!(
                "EULLM_METRICS_RETENTION_DAYS={value} is not a number of days between 1 and \
                 {MAX_RETENTION_DAYS}; keeping {DEFAULT_RETENTION_DAYS}"
            )),
        }
    }
    (DEFAULT_RETENTION_DAYS, "default".to_string())
}

fn valid_days(days: u64) -> Option<u32> {
    (1..=u64::from(MAX_RETENTION_DAYS))
        .contains(&days)
        .then_some(days as u32)
}

#[derive(Serialize, Deserialize)]
struct Saved {
    retention_days: u64,
}

/// The history: a folder (or none) and what is kept in memory.
pub struct Store {
    dir: Option<PathBuf>,
    dir_source: String,
    retention: Mutex<(u32, String)>,
    /// Appends, deletions and the settings file, one at a time.
    files: Mutex<()>,
    memory: Mutex<Memory>,
    /// A failed write is said once, not once a minute.
    warned: AtomicBool,
}

#[derive(Default)]
struct Memory {
    rows: VecDeque<MinuteRow>,
    events: VecDeque<Event>,
}

static STORE: OnceLock<Store> = OnceLock::new();

/// The server's history, resolved from the environment the first time.
pub fn store() -> &'static Store {
    STORE.get_or_init(Store::from_env)
}

/// Resolve where the history goes, log it with its source, and delete what
/// is past the retention. Called once by `api::serve`; a folder that cannot
/// be written leaves the history in memory, with a warning: a monitor is no
/// reason to refuse to serve.
pub fn init() {
    let store = store();
    let usage = store.usage();
    match &store.dir {
        Some(dir) => tracing::info!(
            "Monitor history: {} — {} days kept ({}), {} used  [source: {}]",
            dir.display(),
            usage.retention_days,
            usage.retention_source,
            human_bytes(usage.total_bytes),
            store.dir_source
        ),
        None => tracing::info!(
            "Monitor history: in memory only, the last {} hours  [source: {}]",
            MEMORY_MINUTES / 60,
            store.dir_source
        ),
    }
    let freed = store.cleanup(Utc::now().date_naive());
    if freed.files > 0 {
        tracing::info!(
            "Monitor history: deleted {} day(s) past the retention, {}",
            freed.files,
            human_bytes(freed.bytes)
        );
    }
    event(
        "start",
        format!("eullm {} started", env!("CARGO_PKG_VERSION")),
    );
}

/// Record that something happened: written at once, and kept in memory.
pub fn event(kind: &str, message: impl Into<String>) {
    store().record_event(Event {
        t: super::now_ms(),
        kind: kind.to_string(),
        message: message.into(),
    });
}

impl Store {
    #[cfg(test)]
    fn from_env() -> Self {
        // Tests never write to the home folder.
        Self::new(
            None,
            "tests: memory only".into(),
            (DEFAULT_RETENTION_DAYS, "default".into()),
        )
    }

    #[cfg(not(test))]
    fn from_env() -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_or_else(|_| std::env::temp_dir(), PathBuf::from);
        let dir_setting = crate::settings::var_with_source("EULLM_METRICS_DIR");
        let days_setting = crate::settings::var_with_source("EULLM_METRICS_RETENTION_DAYS");
        let resolved = Self::resolve(
            dir_setting.as_ref().map(|(v, s)| (v.as_str(), s.label())),
            days_setting.as_ref().map(|(v, s)| (v.as_str(), s.label())),
            &home,
        );
        for w in &resolved.warnings {
            tracing::warn!("Monitor history: {w}");
        }
        Self::new(
            resolved.dir,
            resolved.dir_source,
            (resolved.days, resolved.days_source),
        )
    }

    /// The folder and the retention, from the two settings; the folder is
    /// created and tried here, and one that cannot be written is dropped
    /// with a warning in [`Resolved::warnings`].
    pub fn resolve(dir: Option<(&str, &str)>, days: Option<(&str, &str)>, home: &Path) -> Resolved {
        let mut warnings = Vec::new();
        let (mut dir, mut dir_source) = resolve_dir(dir, home);
        if let Some(d) = &dir
            && let Err(e) = writable(d)
        {
            warnings.push(format!(
                "{} cannot be written ({e}); keeping the history in memory only, the last \
                 {} hours. Set EULLM_METRICS_DIR to a writable folder to keep it.",
                d.display(),
                MEMORY_MINUTES / 60
            ));
            dir_source = format!("{dir_source}, not writable");
            dir = None;
        }
        let saved = dir
            .as_ref()
            .and_then(|d| fs::read_to_string(d.join(SETTINGS_FILE)).ok());
        let (days, days_source) = resolve_days(days, saved.as_deref(), &mut warnings);
        Resolved {
            dir,
            dir_source,
            days,
            days_source,
            warnings,
        }
    }

    pub fn new(dir: Option<PathBuf>, dir_source: String, retention: (u32, String)) -> Self {
        Self {
            dir,
            dir_source,
            retention: Mutex::new(retention),
            files: Mutex::new(()),
            memory: Mutex::new(Memory::default()),
            warned: AtomicBool::new(false),
        }
    }

    fn memory(&self) -> std::sync::MutexGuard<'_, Memory> {
        self.memory.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn files(&self) -> std::sync::MutexGuard<'_, ()> {
        self.files.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn retention(&self) -> (u32, String) {
        self.retention
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A minute's row: to its day's file and to memory.
    pub fn record_row(&self, row: MinuteRow) {
        self.append(row.t, &Line::Minute(row.clone()));
        let mut memory = self.memory();
        memory.rows.push_back(row);
        while memory.rows.len() > MEMORY_MINUTES {
            memory.rows.pop_front();
        }
    }

    pub fn record_event(&self, event: Event) {
        self.append(event.t, &Line::Event(event.clone()));
        let mut memory = self.memory();
        memory.events.push_back(event);
        while memory.events.len() > MEMORY_EVENTS {
            memory.events.pop_front();
        }
    }

    fn append(&self, t_ms: u64, line: &Line) {
        let Some(dir) = &self.dir else { return };
        let Ok(text) = serde_json::to_string(line) else {
            return;
        };
        let path = dir.join(day_file(date_of(t_ms)));
        let _files = self.files();
        let written = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| writeln!(f, "{text}"));
        match written {
            Ok(()) => self.warned.store(false, Ordering::Relaxed),
            Err(e) => {
                if !self.warned.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "Monitor history: cannot write {} ({e}); keeping it in memory until \
                         it can",
                        path.display()
                    );
                }
            }
        }
    }

    /// Each day's file in the folder, oldest first.
    fn day_files(&self) -> Vec<(NaiveDate, PathBuf, u64)> {
        let Some(dir) = &self.dir else {
            return Vec::new();
        };
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut days: Vec<_> = entries
            .filter_map(Result::ok)
            .filter_map(|e| {
                let path = e.path();
                let date = parse_day_file(path.file_name()?.to_str()?)?;
                let bytes = e.metadata().ok()?.len();
                Some((date, path, bytes))
            })
            .collect();
        days.sort_by_key(|(date, _, _)| *date);
        days
    }

    /// How much the history takes, and how long it is kept.
    pub fn usage(&self) -> Usage {
        let (retention_days, retention_source) = self.retention();
        let days: Vec<DaySize> = self
            .day_files()
            .into_iter()
            .map(|(date, _, bytes)| DaySize {
                date: date.to_string(),
                bytes,
            })
            .collect();
        let total_bytes = days.iter().map(|d| d.bytes).sum();
        // From the full days only: today's file is still growing.
        let today = Utc::now().date_naive().to_string();
        let full: Vec<u64> = days
            .iter()
            .filter(|d| d.date != today)
            .map(|d| d.bytes)
            .collect();
        let per_day = if full.is_empty() {
            days.last().map_or(0, |d| d.bytes)
        } else {
            full.iter().sum::<u64>() / full.len() as u64
        };
        Usage {
            dir: self.dir.as_ref().map(|d| d.display().to_string()),
            dir_source: self.dir_source.clone(),
            retention_days,
            retention_source,
            max_retention_days: MAX_RETENTION_DAYS,
            total_bytes,
            projected_bytes: per_day * u64::from(retention_days),
            days,
        }
    }

    /// Keep `days` days from now on: saved in the folder, so a restart keeps
    /// it, and what is older goes at once.
    pub fn set_retention(&self, days: u64) -> Result<Freed, String> {
        let days = valid_days(days)
            .ok_or_else(|| format!("retention_days must be between 1 and {MAX_RETENTION_DAYS}"))?;
        let source = if let Some(dir) = &self.dir {
            let text = serde_json::to_string_pretty(&Saved {
                retention_days: u64::from(days),
            })
            .map_err(|e| e.to_string())?;
            let _files = self.files();
            fs::write(dir.join(SETTINGS_FILE), text + "\n")
                .map_err(|e| format!("cannot save the retention in {}: {e}", dir.display()))?;
            format!("set from the Monitor page ({SETTINGS_FILE})")
        } else {
            "set from the Monitor page (until the server restarts)".to_string()
        };
        *self.retention.lock().unwrap_or_else(|e| e.into_inner()) = (days, source);
        tracing::info!("Monitor history: {days} days kept, as set from the Monitor page");
        Ok(self.cleanup(Utc::now().date_naive()))
    }

    /// Delete the days past the retention, counted back from `today`: with
    /// 30 days, today and the 29 before it stay.
    pub fn cleanup(&self, today: NaiveDate) -> Freed {
        let (days, _) = self.retention();
        let first_kept = today
            .checked_sub_days(Days::new(u64::from(days.saturating_sub(1))))
            .unwrap_or(NaiveDate::MIN);
        self.delete_before(first_kept)
    }

    /// Delete every day before `before`, from the folder and from memory.
    pub fn delete_before(&self, before: NaiveDate) -> Freed {
        let mut freed = Freed::default();
        {
            let _files = self.files();
            for (date, path, bytes) in self.day_files() {
                if date >= before {
                    break;
                }
                match fs::remove_file(&path) {
                    Ok(()) => {
                        freed.files += 1;
                        freed.bytes += bytes;
                    }
                    Err(e) => {
                        tracing::warn!("Monitor history: cannot delete {} ({e})", path.display())
                    }
                }
            }
        }
        let cut = start_of(before);
        let mut memory = self.memory();
        memory.rows.retain(|r| r.t >= cut);
        memory.events.retain(|e| e.t >= cut);
        freed
    }

    /// The rows and events between `from` and `to` (milliseconds since
    /// 1970), rows merged into steps of `step` seconds (a multiple of a
    /// minute; none chooses one that gives at most [`TARGET_ROWS`] rows).
    pub fn read(&self, from: u64, to: u64, step: Option<u64>) -> History {
        let to = to.max(from);
        let step_ms = choose_step(from, to, step);
        let (rows, events) = if self.dir.is_some() {
            self.read_files(from, to)
        } else {
            let memory = self.memory();
            (
                memory
                    .rows
                    .iter()
                    .filter(|r| (from..to).contains(&r.t))
                    .cloned()
                    .collect(),
                memory
                    .events
                    .iter()
                    .filter(|e| (from..to).contains(&e.t))
                    .cloned()
                    .collect(),
            )
        };
        let mut events: Vec<Event> = events;
        if events.len() > MAX_EVENTS {
            events.drain(..events.len() - MAX_EVENTS);
        }
        History {
            from,
            to,
            step: step_ms / 1000,
            rows: merge_rows(rows, step_ms),
            events,
        }
    }

    fn read_files(&self, from: u64, to: u64) -> (Vec<MinuteRow>, Vec<Event>) {
        let (first, last) = (date_of(from), date_of(to.saturating_sub(1)));
        let mut rows = Vec::new();
        let mut events = Vec::new();
        for (date, path, _) in self.day_files() {
            if date < first || date > last {
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            for line in text.lines() {
                // A line cut short by a crash, or one a later version wrote
                // in a shape this one does not know, is skipped.
                match serde_json::from_str::<Line>(line) {
                    Ok(Line::Minute(r)) if (from..to).contains(&r.t) => rows.push(r),
                    Ok(Line::Event(e)) if (from..to).contains(&e.t) => events.push(e),
                    _ => {}
                }
            }
        }
        rows.sort_by_key(|r| r.t);
        events.sort_by_key(|e| e.t);
        (rows, events)
    }
}

/// Try that `dir` can be written: created if missing, and a file written in
/// it and removed.
fn writable(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let probe = dir.join(".write-test");
    fs::write(&probe, b"")?;
    fs::remove_file(probe)
}

fn choose_step(from: u64, to: u64, step: Option<u64>) -> u64 {
    match step {
        Some(s) => s.div_ceil(60).max(1) * MINUTE_MS,
        None => {
            let minutes = (to - from).div_ceil(MINUTE_MS);
            minutes.div_ceil(TARGET_ROWS).max(1) * MINUTE_MS
        }
    }
}

/// Rows merged into steps of `step_ms`, counted from midnight UTC of
/// 1970-01-01 so that a step is the same rows whenever it is asked for: an
/// hour is 10:00-11:00, not the hour before the click.
fn merge_rows(rows: Vec<MinuteRow>, step_ms: u64) -> Vec<MinuteRow> {
    if step_ms <= MINUTE_MS {
        return rows;
    }
    let mut steps: BTreeMap<u64, MinuteRow> = BTreeMap::new();
    for row in rows {
        let t = row.t / step_ms * step_ms;
        match steps.get_mut(&t) {
            Some(step) => step.merge(&row),
            None => {
                let mut row = row;
                row.t = t;
                steps.insert(t, row);
            }
        }
    }
    steps.into_values().collect()
}

fn merge_trio(a: &mut Trio, an: u32, b: &Trio, bn: u32) {
    let total = f64::from(an + bn).max(1.0);
    a[0] = a[0].min(b[0]);
    a[1] = (a[1] * f64::from(an) + b[1] * f64::from(bn)) / total;
    a[2] = a[2].max(b[2]);
}

fn merge_opt(a: &mut Option<Trio>, an: u32, b: &Option<Trio>, bn: u32) {
    match (a.as_mut(), b) {
        (Some(a), Some(b)) => merge_trio(a, an, b, bn),
        (None, Some(b)) => *a = Some(*b),
        _ => {}
    }
}

impl MinuteRow {
    /// Take `other` into this row: the lowest minimum, the highest maximum,
    /// the mean weighted by samples, the counts added.
    pub fn merge(&mut self, other: &MinuteRow) {
        let (an, bn) = (self.n, other.n);
        merge_trio(&mut self.cpu, an, &other.cpu, bn);
        merge_trio(&mut self.ram, an, &other.ram, bn);
        merge_trio(&mut self.proc_ram, an, &other.proc_ram, bn);
        merge_trio(&mut self.disk_read, an, &other.disk_read, bn);
        merge_trio(&mut self.gen_tps, an, &other.gen_tps, bn);
        merge_trio(&mut self.active, an, &other.active, bn);
        merge_trio(&mut self.reading, an, &other.reading, bn);
        self.ram_total = self.ram_total.max(other.ram_total);
        self.gen_tokens += other.gen_tokens;
        self.prompt_tokens += other.prompt_tokens;
        self.done += other.done;
        self.failed += other.failed;
        for (i, g) in other.gpus.iter().enumerate() {
            match self.gpus.get_mut(i) {
                Some(mine) => {
                    merge_opt(&mut mine.util, an, &g.util, bn);
                    merge_opt(&mut mine.vram, an, &g.vram, bn);
                    merge_opt(&mut mine.temp, an, &g.temp, bn);
                    merge_opt(&mut mine.power, an, &g.power, bn);
                    mine.vram_total = mine.vram_total.max(g.vram_total);
                }
                None => self.gpus.push(g.clone()),
            }
        }
        self.n = an + bn;
    }
}

/// Minimum, sum and maximum of one figure, as samples come.
#[derive(Debug, Clone, Copy, Default)]
struct Acc {
    min: f64,
    max: f64,
    sum: f64,
    n: u32,
}

impl Acc {
    fn add(&mut self, v: f64) {
        if self.n == 0 {
            self.min = v;
            self.max = v;
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.sum += v;
        self.n += 1;
    }

    fn add_opt(&mut self, v: Option<f64>) {
        if let Some(v) = v {
            self.add(v);
        }
    }

    fn trio(&self) -> Option<Trio> {
        (self.n > 0).then(|| [self.min, self.sum / f64::from(self.n), self.max])
    }

    fn trio_or_zero(&self) -> Trio {
        self.trio().unwrap_or_default()
    }
}

/// A trio rounded to `digits` decimals: the files keep what a chart can
/// show, not the float's sixteen digits (a third of a row's size).
fn rounded(t: Trio, digits: i32) -> Trio {
    let scale = 10f64.powi(digits);
    t.map(|v| (v * scale).round() / scale)
}

#[derive(Debug, Clone, Default)]
struct GpuAcc {
    name: String,
    util: Acc,
    vram: Acc,
    vram_total: Option<u64>,
    temp: Acc,
    power: Acc,
}

/// The minute being gathered.
#[derive(Debug, Clone, Default)]
struct Minute {
    t: u64,
    n: u32,
    cpu: Acc,
    ram: Acc,
    ram_total: u64,
    proc_ram: Acc,
    disk_read: Acc,
    gen_tps: Acc,
    active: Acc,
    reading: Acc,
    gen_tokens: u64,
    prompt_tokens: u64,
    done: u64,
    failed: u64,
    gpus: Vec<GpuAcc>,
}

impl Minute {
    fn add(&mut self, s: &Sample) {
        self.n += 1;
        self.cpu.add(f64::from(s.cpu_pct));
        self.ram.add(s.ram_used as f64);
        self.ram_total = self.ram_total.max(s.ram_total);
        self.proc_ram.add(s.process_ram as f64);
        self.disk_read.add(s.disk_read_bps as f64);
        self.gen_tps.add(s.gen_tps);
        self.active.add(f64::from(s.active));
        self.reading.add(f64::from(s.reading));
        self.gen_tokens += s.gen_tokens;
        self.prompt_tokens += s.prompt_tokens;
        self.done += s.done;
        self.failed += s.failed;
        for (i, g) in s.gpus.iter().enumerate() {
            if self.gpus.len() <= i {
                self.gpus.push(GpuAcc {
                    name: g.name.clone(),
                    ..GpuAcc::default()
                });
            }
            let acc = &mut self.gpus[i];
            acc.util.add_opt(g.util_pct.map(f64::from));
            acc.vram.add_opt(g.vram_used.map(|v| v as f64));
            acc.temp.add_opt(g.temp_c.map(f64::from));
            acc.power.add_opt(g.power_w.map(f64::from));
            acc.vram_total = acc.vram_total.max(g.vram_total);
        }
    }

    fn row(&self) -> MinuteRow {
        // Percentages, degrees and watts to a tenth, rates to a hundredth,
        // bytes and counts whole.
        let tenth = |a: &Acc| rounded(a.trio_or_zero(), 1);
        let whole = |a: &Acc| rounded(a.trio_or_zero(), 0);
        let opt = |a: &Acc, d: i32| a.trio().map(|t| rounded(t, d));
        MinuteRow {
            t: self.t,
            n: self.n,
            cpu: tenth(&self.cpu),
            ram: whole(&self.ram),
            ram_total: self.ram_total,
            proc_ram: whole(&self.proc_ram),
            disk_read: whole(&self.disk_read),
            gen_tps: rounded(self.gen_tps.trio_or_zero(), 2),
            active: rounded(self.active.trio_or_zero(), 2),
            reading: rounded(self.reading.trio_or_zero(), 2),
            gen_tokens: self.gen_tokens,
            prompt_tokens: self.prompt_tokens,
            done: self.done,
            failed: self.failed,
            gpus: self
                .gpus
                .iter()
                .map(|g| GpuRow {
                    name: g.name.clone(),
                    util: opt(&g.util, 1),
                    vram: opt(&g.vram, 0),
                    vram_total: g.vram_total,
                    temp: opt(&g.temp, 1),
                    power: opt(&g.power, 1),
                })
                .collect(),
        }
    }
}

/// Turns the sampler's seconds into minute rows, and deletes what is past
/// the retention when the day changes. The sampler thread owns it.
#[derive(Default)]
pub struct Recorder {
    minute: Option<Minute>,
    day: Option<NaiveDate>,
}

impl Recorder {
    /// Take a sample; the minute before it is written when this one starts
    /// a new one.
    pub fn add(&mut self, sample: &Sample) {
        self.add_to(store(), sample);
    }

    fn add_to(&mut self, store: &Store, sample: &Sample) {
        let t = sample.t_ms / MINUTE_MS * MINUTE_MS;
        if self.minute.as_ref().is_some_and(|m| m.t != t)
            && let Some(done) = self.minute.take()
        {
            store.record_row(done.row());
        }
        self.minute
            .get_or_insert_with(|| Minute {
                t,
                ..Minute::default()
            })
            .add(sample);
        let today = date_of(sample.t_ms);
        if self.day.is_some_and(|d| d != today) {
            let freed = store.cleanup(today);
            if freed.files > 0 {
                tracing::info!(
                    "Monitor history: deleted {} day(s) past the retention, {}",
                    freed.files,
                    human_bytes(freed.bytes)
                );
            }
        }
        self.day = Some(today);
    }
}

fn date_of(t_ms: u64) -> NaiveDate {
    DateTime::<Utc>::from_timestamp_millis(t_ms as i64).map_or(NaiveDate::MIN, |t| t.date_naive())
}

fn start_of(date: NaiveDate) -> u64 {
    date.and_hms_opt(0, 0, 0)
        .map_or(0, |t| t.and_utc().timestamp_millis().max(0) as u64)
}

fn day_file(date: NaiveDate) -> String {
    format!("{}.jsonl", date.format("%Y-%m-%d"))
}

fn parse_day_file(name: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(name.strip_suffix(".jsonl")?, "%Y-%m-%d").ok()
}

/// `YYYY-MM-DD`, as `DELETE /api/monitor/history?before=` takes it.
pub fn parse_date(text: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(text.trim(), "%Y-%m-%d").ok()
}

fn human_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= 1024.0 * MIB {
        format!("{:.1} GiB", bytes as f64 / 1024.0 / MIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::system::Gpu;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "eullm-history-{name}-{}-{}",
            std::process::id(),
            super::super::now_ms()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn on_disk(dir: &Path, days: u32) -> Store {
        Store::new(
            Some(dir.to_path_buf()),
            "test".into(),
            (days, "test".into()),
        )
    }

    fn sample(t_ms: u64, cpu: f32, util: Option<f32>) -> Sample {
        Sample {
            t_ms,
            cpu_pct: cpu,
            ram_used: 1000,
            ram_total: 4000,
            gen_tps: f64::from(cpu),
            gen_tokens: 2,
            gpus: vec![Gpu {
                name: "GPU".into(),
                util_pct: util,
                vram_used: Some(10),
                vram_total: Some(100),
                ..Gpu::default()
            }],
            ..Sample::default()
        }
    }

    /// 2026-11-20T10:00:00Z.
    const T0: u64 = 1_795_168_800_000;

    #[test]
    fn a_minute_keeps_its_lowest_mean_and_highest() {
        let store = Store::new(None, "test".into(), (30, "test".into()));
        let mut rec = Recorder::default();
        for (i, cpu) in [10.0, 90.0, 20.0].into_iter().enumerate() {
            rec.add_to(&store, &sample(T0 + i as u64 * 1000, cpu, Some(cpu)));
        }
        // The next minute's first sample writes the one before.
        rec.add_to(&store, &sample(T0 + MINUTE_MS, 50.0, None));
        let h = store.read(T0, T0 + 2 * MINUTE_MS, None);
        assert_eq!(h.rows.len(), 1);
        let row = &h.rows[0];
        assert_eq!((row.t, row.n, row.gen_tokens), (T0, 3, 6));
        assert_eq!(row.cpu, [10.0, 40.0, 90.0]);
        assert_eq!(row.gpus[0].util, Some([10.0, 40.0, 90.0]));
        assert_eq!(row.gpus[0].vram_total, Some(100));
        assert_eq!(row.gpus[0].temp, None);
    }

    #[test]
    fn long_ranges_merge_rows_and_keep_the_peaks() {
        let row = |t: u64, cpu: Trio, n: u32| MinuteRow {
            t,
            n,
            cpu,
            done: 1,
            ..MinuteRow::default()
        };
        let rows = vec![
            row(T0, [5.0, 10.0, 20.0], 60),
            row(T0 + MINUTE_MS, [1.0, 30.0, 95.0], 60),
            row(T0 + 2 * MINUTE_MS, [7.0, 8.0, 9.0], 30),
        ];
        let merged = merge_rows(rows, 2 * MINUTE_MS);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].cpu, [1.0, 20.0, 95.0]);
        assert_eq!((merged[0].n, merged[0].done), (120, 2));
        assert_eq!(merged[1].t, T0 + 2 * MINUTE_MS);
    }

    #[test]
    fn the_step_is_whole_minutes_and_bounded_by_default() {
        assert_eq!(choose_step(0, 3_600_000, None), MINUTE_MS);
        // 30 days: 43,200 minutes in at most 720 rows.
        assert_eq!(choose_step(0, 30 * 86_400_000, None), 60 * MINUTE_MS);
        assert_eq!(choose_step(0, 1, Some(90)), 2 * MINUTE_MS);
        assert_eq!(choose_step(0, 1, Some(0)), MINUTE_MS);
    }

    #[test]
    fn rows_and_events_go_to_their_day_s_file_and_come_back() {
        let dir = temp_dir("files");
        let store = on_disk(&dir, 30);
        let mut rec = Recorder::default();
        rec.add_to(&store, &sample(T0, 33.0, Some(50.0)));
        rec.add_to(&store, &sample(T0 + MINUTE_MS, 1.0, None));
        store.record_event(Event {
            t: T0 + 5,
            kind: "model_loaded".into(),
            message: "gemma".into(),
        });
        let text = fs::read_to_string(dir.join("2026-11-20.jsonl")).unwrap();
        assert!(text.contains(r#""type":"minute""#), "{text}");
        assert!(text.contains(r#""type":"event""#), "{text}");
        // From the file, not from memory: a new store on the same folder.
        let again = on_disk(&dir, 30);
        let h = again.read(T0, T0 + 10 * MINUTE_MS, None);
        assert_eq!(h.rows.len(), 1);
        assert_eq!(h.rows[0].cpu, [33.0, 33.0, 33.0]);
        assert_eq!(h.events.len(), 1);
        assert_eq!(h.events[0].kind, "model_loaded");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_damaged_line_is_skipped() {
        let dir = temp_dir("damaged");
        let row = MinuteRow {
            t: T0,
            n: 1,
            ..MinuteRow::default()
        };
        let good = serde_json::to_string(&Line::Minute(row)).unwrap();
        fs::write(
            dir.join("2026-11-20.jsonl"),
            format!("{good}\n{{\"type\":\"minute\",\"t\":\n{{\"type\":\"later\"}}\n"),
        )
        .unwrap();
        let h = on_disk(&dir, 30).read(T0, T0 + MINUTE_MS, None);
        assert_eq!(h.rows.len(), 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_retention_keeps_today_and_the_days_before_it() {
        let dir = temp_dir("retention");
        for day in ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"] {
            fs::write(dir.join(format!("{day}.jsonl")), "0123456789").unwrap();
        }
        fs::write(dir.join("notes.txt"), "not a day").unwrap();
        let store = on_disk(&dir, 2);
        let freed = store.cleanup(parse_date("2026-11-20").unwrap());
        assert_eq!(
            freed,
            Freed {
                files: 2,
                bytes: 20
            }
        );
        let usage = store.usage();
        let days: Vec<&str> = usage.days.iter().map(|d| d.date.as_str()).collect();
        assert_eq!(days, ["2026-11-19", "2026-11-20"]);
        assert_eq!(usage.total_bytes, 20);
        assert!(dir.join("notes.txt").exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn deleting_before_a_date_frees_its_files() {
        let dir = temp_dir("delete");
        for day in ["2026-11-01", "2026-11-02", "2026-11-03"] {
            fs::write(dir.join(format!("{day}.jsonl")), "abc").unwrap();
        }
        let store = on_disk(&dir, 366);
        let freed = store.delete_before(parse_date("2026-11-03").unwrap());
        assert_eq!(freed, Freed { files: 2, bytes: 6 });
        assert_eq!(store.usage().days.len(), 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_retention_set_from_the_page_is_saved_and_wins_over_the_environment() {
        let dir = temp_dir("settings");
        let store = on_disk(&dir, 30);
        assert!(store.set_retention(0).is_err());
        assert!(store.set_retention(367).is_err());
        store.set_retention(365).unwrap();
        assert_eq!(store.usage().retention_days, 365);
        let r = Store::resolve(
            Some((dir.to_str().unwrap(), "environment")),
            Some(("7", "environment")),
            Path::new("/nowhere"),
        );
        assert_eq!(r.days, 365);
        assert!(r.days_source.contains("Monitor page"), "{}", r.days_source);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_folder_and_the_days_resolve_with_their_source() {
        let home = Path::new("/home/x");
        assert_eq!(
            resolve_dir(None, home),
            (Some(home.join(".eullm/metrics")), "default".into())
        );
        assert_eq!(resolve_dir(Some(("  ", ".env")), home).1, "default");
        let (dir, source) = resolve_dir(Some(("/data/metrics", ".env")), home);
        assert_eq!(dir, Some(PathBuf::from("/data/metrics")));
        assert_eq!(source, "EULLM_METRICS_DIR (.env)");
        assert_eq!(resolve_dir(Some(("OFF", "environment")), home).0, None);

        let mut w = Vec::new();
        assert_eq!(resolve_days(None, None, &mut w).0, DEFAULT_RETENTION_DAYS);
        assert_eq!(
            resolve_days(Some(("90", "environment")), None, &mut w).0,
            90
        );
        assert!(w.is_empty());
        assert_eq!(
            resolve_days(Some(("1000", "environment")), None, &mut w).0,
            30
        );
        assert_eq!(
            resolve_days(Some(("a week", "environment")), None, &mut w).0,
            30
        );
        assert_eq!(w.len(), 2);
        let (days, _) = resolve_days(Some(("90", "environment")), Some("{oops"), &mut w);
        assert_eq!((days, w.len()), (90, 3));
    }

    #[test]
    fn an_unwritable_folder_leaves_the_history_in_memory() {
        let dir = temp_dir("unwritable");
        let file = dir.join("a-file");
        fs::write(&file, "").unwrap();
        // A folder inside a file cannot be made.
        let inside = file.join("metrics");
        let r = Store::resolve(
            Some((inside.to_str().unwrap(), "environment")),
            None,
            Path::new("/x"),
        );
        assert_eq!(r.dir, None);
        assert_eq!(r.warnings.len(), 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_new_day_deletes_what_is_past_the_retention() {
        let dir = temp_dir("newday");
        fs::write(dir.join("2026-11-19.jsonl"), "x").unwrap();
        let store = on_disk(&dir, 1);
        let mut rec = Recorder::default();
        rec.add_to(&store, &sample(T0 - 86_400_000, 1.0, None));
        assert!(dir.join("2026-11-19.jsonl").exists());
        rec.add_to(&store, &sample(T0, 1.0, None));
        assert!(!dir.join("2026-11-19.jsonl").exists());
        let _ = fs::remove_dir_all(dir);
    }
}
