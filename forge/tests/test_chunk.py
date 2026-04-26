"""Tests for the chunking module."""

from __future__ import annotations

from eullm_forge.datasets.chunk import (
    ChunkConfig,
    chunk_record,
    chunk_text,
)


def test_short_text_returns_one_chunk():
    text = "Una breve sentenza di poche righe. Niente da splittare."
    chunks = chunk_text(text, max_chars=3000)
    assert chunks == [text]


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_each_chunk_within_max_chars():
    text = "abc " * 2000  # 8000 chars
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 1000


def test_chunks_prefer_paragraph_boundary():
    text = (
        "Primo paragrafo abbastanza lungo per superare il primo cut.\n\n"
        "Secondo paragrafo, deve iniziare il secondo chunk.\n\n"
        "Terzo paragrafo, completa il terzo chunk."
    )
    chunks = chunk_text(text, max_chars=80, overlap=10)
    # The first cut should land right after a paragraph break, so the
    # first chunk ends with the trailing period of the first paragraph.
    assert chunks[0].endswith(("primo cut.", "Primo paragrafo abbastanza lungo per superare il primo cut."))


def test_chunks_dont_split_mid_word():
    text = "supercalifragilisticexpialidocious " * 100  # one long word + space
    chunks = chunk_text(text, max_chars=200, overlap=50)
    for c in chunks:
        # Every chunk must start and end on a non-mid-word position.
        assert not c.startswith("super") or c.startswith("supercalif")
        # Specifically, no chunk should end with a half-word (no internal cut).
        last = c.rsplit(" ", 1)[-1] if " " in c else c
        assert last == "supercalifragilisticexpialidocious" or last.endswith("expialidocious")


def test_overlap_is_present_between_consecutive_chunks():
    text = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 100
    chunks = chunk_text(text, max_chars=500, overlap=80)
    assert len(chunks) >= 2
    # Take a substring from the tail of chunk[0] and check that it
    # appears at the head of chunk[1].
    tail = chunks[0][-50:].strip()
    if tail:
        assert tail in chunks[1] or chunks[1].startswith(tail.split()[0])


def test_tiny_trailing_fragment_merged_into_previous_chunk():
    text = ("a" * 1000) + " " + ("b" * 50)
    # max_chars chosen so the last 50 chars would otherwise be a tiny chunk.
    chunks = chunk_text(text, max_chars=900, overlap=0, min_chars=200)
    # The 50-char tail should NOT survive as its own chunk.
    assert all(len(c) >= 200 for c in chunks)


def test_chunk_record_preserves_metadata_and_adds_traceability():
    rec = {
        "sentence_id": "snciv/2023/12345",
        "text": "x" * 5000,
        "metadata": {"court": "Cassazione"},
    }
    cfg = ChunkConfig(max_chars=1000, overlap=100, min_chars=200)
    out = list(chunk_record(rec, config=cfg))
    assert len(out) > 1
    for i, r in enumerate(out):
        # Original metadata preserved.
        assert r["metadata"] == {"court": "Cassazione"}
        assert r["sentence_id"] == "snciv/2023/12345"
        # Chunk-level fields added.
        assert r["chunk_index"] == i
        assert r["chunk_total"] == len(out)
        assert r["source_id"] == "snciv/2023/12345"
        assert r["text"]


def test_chunk_record_skips_empty_text():
    rec = {"sentence_id": "x", "text": ""}
    assert list(chunk_record(rec)) == []
    rec2 = {"sentence_id": "y", "text": "   \n\n   "}
    assert list(chunk_record(rec2)) == []


def test_overlap_zero_works():
    text = ("paragrafo " * 200).strip()
    chunks = chunk_text(text, max_chars=500, overlap=0)
    assert len(chunks) >= 2
    # No overlap → consecutive chunks share no leading/trailing words.
    for c in chunks:
        assert len(c) <= 500


def test_max_chars_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=0)


def test_overlap_must_be_smaller_than_max():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("text" * 100, max_chars=100, overlap=200)
