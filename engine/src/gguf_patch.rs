//! Minimal GGUF metadata patcher for Ollama compatibility.
//!
//! Some Ollama GGUF files contain metadata arrays with fewer elements than
//! upstream llama.cpp expects.  For example, `qwen35.rope.dimension_sections`
//! has 3 elements in the Ollama blob but llama.cpp requires 4.
//!
//! This module detects and patches such files during import by inserting
//! zero-filled elements so that the GGUF loads correctly in EULLM's
//! llama.cpp backend.
//!
//! The patcher works at the binary level: it streams through the source
//! file, applies byte-level patches in the metadata section, recalculates
//! alignment padding, and streams the (unchanged) tensor data to the
//! destination.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian

/// Default alignment for tensor data in GGUF v3.
const ALIGNMENT: u64 = 32;

// GGUF value type IDs.
const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

/// Size in bytes of a scalar GGUF value type.
fn scalar_size(t: u32) -> u64 {
    match t {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => 1,
        TYPE_UINT16 | TYPE_INT16 => 2,
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => 4,
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => 8,
        _ => 0,
    }
}

/// A patch to apply: extend an array from `current_count` to `target_count`
/// by appending zero-filled elements.
struct ArrayPatch {
    /// File offset of the 8-byte little-endian count field.
    count_offset: u64,
    current_count: u64,
    target_count: u64,
    /// Size of each element in bytes.
    elem_size: u64,
}

/// Known GGUF metadata keys that Ollama may write with too few array elements.
/// Each entry is (key_name, expected_element_count).
const KNOWN_FIXES: &[(&str, u64)] = &[
    ("qwen35.rope.dimension_sections", 4),
];

/// Check whether a GGUF file needs Ollama compatibility patches.  If so,
/// write a patched copy to `dst` and return `Ok(true)`.  If no patching is
/// needed, return `Ok(false)` without creating `dst` (the caller should do
/// a normal copy).
///
/// The patch inserts zero-filled array elements where the source has fewer
/// elements than llama.cpp expects, then recalculates the alignment padding
/// before the tensor data section.  Tensor data itself is streamed unchanged.
pub fn patch_gguf_if_needed(src: &Path, dst: &Path) -> io::Result<bool> {
    let mut f = std::fs::File::open(src)?;

    // ── Parse GGUF header ────────────────────────────────────────────
    let magic = read_u32(&mut f)?;
    if magic != GGUF_MAGIC {
        return Ok(false);
    }
    let _version = read_u32(&mut f)?;
    let tensor_count = read_u64(&mut f)?;
    let kv_count = read_u64(&mut f)?;

    // ── Scan metadata KV entries ─────────────────────────────────────
    let mut patches: Vec<ArrayPatch> = Vec::new();

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut f)?;
        let vtype = read_u32(&mut f)?;

        if vtype == TYPE_ARRAY {
            let elem_type = read_u32(&mut f)?;
            let count_offset = f.stream_position()?;
            let count = read_u64(&mut f)?;
            let elem_sz = if elem_type == TYPE_STRING {
                // String arrays: skip each string individually
                for _ in 0..count {
                    skip_gguf_value(&mut f, TYPE_STRING)?;
                }
                0 // not a fixed-size element
            } else {
                let sz = scalar_size(elem_type);
                skip_n(&mut f, count * sz)?;
                sz
            };

            // Check if this key needs fixing
            if let Some(&(_, target)) = KNOWN_FIXES.iter().find(|&&(k, _)| k == key.as_str()) {
                if count < target && elem_sz > 0 {
                    patches.push(ArrayPatch {
                        count_offset,
                        current_count: count,
                        target_count: target,
                        elem_size: elem_sz,
                    });
                    tracing::info!(
                        "GGUF patch: {key} has {count} elements, extending to {target}"
                    );
                }
            }
        } else {
            skip_gguf_value(&mut f, vtype)?;
        }
    }

    if patches.is_empty() {
        return Ok(false);
    }

    // ── Parse tensor info to find the header/tensor-data boundary ─────
    for _ in 0..tensor_count {
        let _name = read_gguf_string(&mut f)?;
        let n_dims = read_u32(&mut f)?;
        skip_n(&mut f, n_dims as u64 * 8)?; // dimension sizes
        skip_n(&mut f, 4)?; // tensor type
        skip_n(&mut f, 8)?; // data offset (relative to tensor data start)
    }

    let end_of_header = f.stream_position()?;
    let orig_data_start = align_up(end_of_header, ALIGNMENT);

    // Total extra bytes we are inserting into the metadata section.
    let extra_bytes: u64 = patches
        .iter()
        .map(|p| (p.target_count - p.current_count) * p.elem_size)
        .sum();

    let new_data_start = align_up(end_of_header + extra_bytes, ALIGNMENT);

    // ── Write the patched file ───────────────────────────────────────
    f.seek(SeekFrom::Start(0))?;

    let out = std::fs::File::create(dst)?;
    let mut w = io::BufWriter::with_capacity(8 * 1024 * 1024, out);

    // Sort patches by file offset (they should already be in order, but
    // let's be safe).
    patches.sort_by_key(|p| p.count_offset);

    let mut src_pos: u64 = 0;

    for patch in &patches {
        // Copy everything from current position up to the count field.
        copy_exact(&mut f, &mut w, patch.count_offset - src_pos)?;
        src_pos = patch.count_offset;

        // Read old count (8 bytes), write new count.
        let _old = read_u64(&mut f)?;
        src_pos += 8;
        w.write_all(&patch.target_count.to_le_bytes())?;

        // Copy existing elements.
        let existing = patch.current_count * patch.elem_size;
        copy_exact(&mut f, &mut w, existing)?;
        src_pos += existing;

        // Append zero-filled extra elements.
        let extra = (patch.target_count - patch.current_count) * patch.elem_size;
        let zeros = vec![0u8; extra as usize];
        w.write_all(&zeros)?;
    }

    // Copy remaining header + tensor info up to original padding.
    copy_exact(&mut f, &mut w, end_of_header - src_pos)?;

    // Write new alignment padding.
    let new_padding = new_data_start - (end_of_header + extra_bytes);
    let pad = vec![0u8; new_padding as usize];
    w.write_all(&pad)?;

    // Skip old alignment padding in source.
    let old_padding = orig_data_start - end_of_header;
    skip_n(&mut f, old_padding)?;

    // Stream tensor data (the bulk of the file — may be several GB).
    io::copy(&mut f, &mut w)?;

    w.flush()?;
    Ok(true)
}

// ── I/O helpers ──────────────────────────────────────────────────────────

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Advance the reader by `n` bytes (seek if possible).
fn skip_n(r: &mut (impl Read + Seek), n: u64) -> io::Result<()> {
    r.seek(SeekFrom::Current(n as i64))?;
    Ok(())
}

/// Skip a single GGUF value in the stream (used during header scanning).
fn skip_gguf_value(r: &mut (impl Read + Seek), vtype: u32) -> io::Result<()> {
    match vtype {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => skip_n(r, 1),
        TYPE_UINT16 | TYPE_INT16 => skip_n(r, 2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => skip_n(r, 4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => skip_n(r, 8),
        TYPE_STRING => {
            let len = read_u64(r)?;
            skip_n(r, len)
        }
        TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            if elem_type == TYPE_STRING {
                for _ in 0..count {
                    skip_gguf_value(r, TYPE_STRING)?;
                }
                Ok(())
            } else {
                skip_n(r, count * scalar_size(elem_type))
            }
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF value type {vtype}"),
        )),
    }
}

/// Copy exactly `n` bytes from reader to writer using an 8 KB buffer.
fn copy_exact(r: &mut impl Read, w: &mut impl Write, mut n: u64) -> io::Result<()> {
    let mut buf = [0u8; 8192];
    while n > 0 {
        let chunk = n.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..chunk])?;
        w.write_all(&buf[..chunk])?;
        n -= chunk as u64;
    }
    Ok(())
}

fn align_up(v: u64, alignment: u64) -> u64 {
    v.div_ceil(alignment) * alignment
}
