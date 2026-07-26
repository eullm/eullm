//! The single implementation of "what is safe to hand to the client".
//!
//! Three decode loops produce text: the scheduler's (continuous batching), the
//! sequential engine's (`--batch-size 0`), and the multimodal one. Until
//! v0.6.40 only the scheduler ran the logic in this module; the other two
//! compared `full_output.ends_with(stop)` once per token, which fails in three
//! distinct ways:
//!
//! 1. **A stop sequence that does not land at the end is missed.** A piece of
//!    `"<|im_end|>\n"` leaves the accumulated output ending in `"\n"`, so the
//!    marker is never seen and generation runs on past the turn boundary.
//! 2. **A stop sequence split across two tokens leaks its first half.** The
//!    streaming paths cut the marker out of the *current* piece
//!    (`&piece[..piece.len() - stop.len()]`), but the earlier piece carrying
//!    the first bytes has already been sent and cannot be recalled.
//! 3. **That same slice is a byte index into a UTF-8 string**, so when the cut
//!    lands inside a multi-byte character it panics rather than misbehaves.
//!
//! On top of which `filter_sequences` (the `DEFAULT_HARMONY_FILTERS` that exist
//! to strip `<|channel|>`-style scaffolding) were never applied outside the
//! scheduler at all, so the multimodal path showed them to the user verbatim.
//!
//! The fix is not better checks in three places, it is one place. This module
//! holds the logic the scheduler had already proven in production; the other
//! two loops now call it instead of carrying their own version. Anything added
//! here — a new filter, a new hold-back rule — reaches all three by
//! construction, which is the property the old arrangement lacked.

/// Outcome of feeding one decoded piece through the stop-sequence filter.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PieceOutcome {
    /// Safe-to-stream text (may be empty); generation continues.
    Emit(String),
    /// A full stop sequence was hit. The contained text is whatever preceded
    /// the stop marker and should be streamed before finishing.
    Stop(String),
}

/// Length (in bytes) of the longest suffix of `buf` that is a *proper* prefix
/// of any stop sequence. This is the amount of trailing text that must be held
/// back: it could still grow into a full stop sequence on the next token.
///
/// A full match is handled by the caller (via `find`) before this is consulted,
/// so we never report the entire stop sequence here.
pub(crate) fn stop_prefix_holdback(buf: &str, stops: &[String]) -> usize {
    let mut max = 0;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        // Try the longest possible overlap first; the suffix of `buf` must be a
        // prefix of `s` and strictly shorter than `s` (full matches handled elsewhere).
        let upper = buf.len().min(s.len().saturating_sub(1));
        let mut k = upper;
        while k >= 1 {
            let start = buf.len() - k;
            if buf.is_char_boundary(start) && s.as_bytes().starts_with(&buf.as_bytes()[start..]) {
                if k > max {
                    max = k;
                }
                break;
            }
            k -= 1;
        }
    }
    max
}

/// Feed one decoded `piece` into the per-sequence `pending` hold-back buffer and
/// decide what is safe to stream.
///
/// - If appending the piece completes a stop sequence, everything up to the
///   stop marker is returned as `Stop(..)` and `pending` is cleared.
/// - Otherwise the longest trailing run that could still become a stop sequence
///   is retained in `pending`, and the rest is returned as `Emit(..)`.
///
/// This makes streaming robust against models that spell a turn delimiter out
/// as ordinary text and only then emit an EOG token (e.g. Gemma emitting
/// `<end_of_turn` + EOG): the partial delimiter sits in `pending` and is
/// discarded when the caller observes EOG, instead of leaking to the client.
pub(crate) fn process_piece(
    pending: &mut String,
    stops: &[String],
    filters: &[String],
    piece: &str,
) -> PieceOutcome {
    pending.push_str(piece);

    // 1. Stop sequences win: terminate generation at the earliest hit.
    let mut cut: Option<usize> = None;
    for s in stops {
        if let Some(pos) = pending.find(s.as_str()) {
            cut = Some(cut.map_or(pos, |c| c.min(pos)));
        }
    }
    if let Some(pos) = cut {
        let out = pending[..pos].to_string();
        pending.clear();
        return PieceOutcome::Stop(out);
    }

    // 2. Filter sequences: silently elide every completed occurrence, then
    // continue. Unlike stops, these do not terminate the response.
    for f in filters {
        if f.is_empty() {
            continue;
        }
        while let Some(pos) = pending.find(f.as_str()) {
            pending.replace_range(pos..pos + f.len(), "");
        }
    }

    // 3. Hold back any trailing run that could still complete EITHER a stop
    // or a filter sequence. Reuses `stop_prefix_holdback` for both — it just
    // looks at suffix→prefix overlaps and is agnostic to the list's meaning.
    let holdback = stop_prefix_holdback(pending, stops).max(stop_prefix_holdback(pending, filters));
    let emit_upto = pending.len() - holdback;
    let out = pending[..emit_upto].to_string();
    pending.drain(..emit_upto);
    PieceOutcome::Emit(out)
}

/// Take whatever is still held back, for a generation that ended without a stop
/// sequence (token budget exhausted, or an EOG token).
///
/// The distinction the caller must make: on **EOG** the tail is a partial turn
/// delimiter the model was in the middle of spelling out, and it is dropped.
/// On a **budget** end the tail is ordinary text that simply never grew into a
/// marker, and dropping it would truncate the answer — the scheduler has always
/// emitted it (`std::mem::take(&mut seq.pending)`), and the other loops now do
/// the same instead of silently losing those bytes.
pub(crate) fn flush(pending: &mut String) -> String {
    std::mem::take(pending)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], filters: &[&str], pieces: &[&str]) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let filters: Vec<String> = filters.iter().map(|s| s.to_string()).collect();
        let mut pending = String::new();
        let mut out = String::new();
        for p in pieces {
            match process_piece(&mut pending, &stops, &filters, p) {
                PieceOutcome::Emit(s) => out.push_str(&s),
                PieceOutcome::Stop(s) => {
                    out.push_str(&s);
                    return (out, true);
                }
            }
        }
        (out, false)
    }

    // Defect 1 of the old per-loop check: `output.ends_with(stop)` only sees a
    // marker that finishes the piece. Here the model emits the delimiter and a
    // newline in one token, and the old code ran straight past the turn end.
    #[test]
    fn a_stop_sequence_not_at_the_end_of_a_piece_still_stops() {
        let (out, stopped) = run(&["<|im_end|>"], &[], &["Rome", "<|im_end|>\nUser: "]);
        assert!(stopped, "the marker was inside the piece, not at its end");
        assert_eq!(out, "Rome");
    }

    // Defect 2: split across two tokens. The old streaming paths had already
    // sent the first half before noticing.
    #[test]
    fn a_stop_sequence_split_across_tokens_never_reaches_the_client() {
        let (out, stopped) = run(&["<|im_end|>"], &[], &["Rome", "<|im_", "end|>"]);
        assert!(stopped);
        assert_eq!(out, "Rome", "no fragment of the marker may be emitted");
    }

    // Defect 3: the old cut was a byte slice, so a stop landing mid-character
    // panicked. Multi-byte text either side of the marker must be safe.
    #[test]
    fn multibyte_text_around_a_stop_does_not_panic_and_is_not_cut() {
        let (out, stopped) = run(&["<|im_end|>"], &[], &["Perché", " è così", "<|im_end|>"]);
        assert!(stopped);
        assert_eq!(out, "Perché è così");
    }

    #[test]
    fn a_multibyte_character_split_across_pieces_survives() {
        // "è" is 0xC3 0xA8; a decoder can hand the two halves over separately
        // only as complete pieces, but the hold-back arithmetic must still land
        // on character boundaries when the marker shares a first byte.
        let (out, stopped) = run(&["<|im_end|>"], &[], &["è", "à", "ù"]);
        assert!(!stopped);
        assert_eq!(out, "èàù");
    }

    // Filters were absent outside the scheduler, so this scaffolding reached
    // the user verbatim on the multimodal path.
    #[test]
    fn filter_sequences_are_elided_without_stopping() {
        let (out, stopped) = run(
            &["<|im_end|>"],
            &["<|channel|>analysis"],
            &["Answer: ", "<|channel|>analysis", "42"],
        );
        assert!(!stopped);
        assert_eq!(out, "Answer: 42");
    }

    #[test]
    fn a_filter_split_across_tokens_is_still_elided() {
        let (out, _) = run(
            &["<|im_end|>"],
            &["<|channel|>"],
            &["a", "<|chan", "nel|>", "b"],
        );
        assert_eq!(out, "ab");
    }

    // The hold-back must not swallow text that merely starts like a marker.
    #[test]
    fn text_that_only_resembles_a_marker_is_released() {
        let (out, stopped) = run(&["<|im_end|>"], &[], &["a<|im", "possible"]);
        assert!(!stopped);
        assert_eq!(out, "a<|impossible");
    }

    #[test]
    fn the_tail_is_recoverable_when_generation_ends_without_a_stop() {
        let stops = vec!["<end_of_turn>".to_string()];
        let mut pending = String::new();
        let mut out = String::new();
        for p in ["Rome", "<end_of_turn"] {
            if let PieceOutcome::Emit(s) = process_piece(&mut pending, &stops, &[], p) {
                out.push_str(&s);
            }
        }
        // Held back because it could still become the marker.
        assert_eq!(out, "Rome");
        assert_eq!(pending, "<end_of_turn");
        // A caller ending on the token budget takes it; one seeing EOG drops it.
        assert_eq!(flush(&mut pending), "<end_of_turn");
        assert!(pending.is_empty());
    }

    #[test]
    fn an_empty_stop_list_emits_everything_immediately() {
        let (out, stopped) = run(&[], &[], &["a", "b", "c"]);
        assert!(!stopped);
        assert_eq!(out, "abc");
    }
}
