"""Tests for legislation corpus preparation.

Legislation chunks flow into the same train/val split as court rulings, which
is grouped by document since the chunk-level leakage fix: an article's chunks
must carry the document key they are grouped on, or they scatter across both
sides and the held-out set shares verbatim overlap with train.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from eullm_forge.datasets.chunk import ChunkConfig
from eullm_forge.datasets.training_format import split_indices

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_legislation.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_legislation", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


legislazione = _load()


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _two_chunk_article() -> list[dict]:
    # ~3000 chars at max_chars=500/overlap=0: several chunks, one article.
    return [
        {
            "text": " ".join(f"parola{i}" for i in range(450)),
            "article_num": "2086",
            "article_title": "Direzione e gerarchia",
        }
    ]


def test_legislation_chunks_share_one_sentence_id(tmp_path):
    n = legislazione._write_source(
        "codice_civile",
        _two_chunk_article(),
        tmp_path,
        ChunkConfig(max_chars=500, overlap=0, min_chars=50),
        dry_run=False,
    )
    assert n >= 2
    recs = _read_records(tmp_path / "legislazione_codice_civile.chunks.jsonl")
    assert len(recs) == n
    assert {r["sentence_id"] for r in recs} == {"codice_civile/art_2086"}


def test_legislation_chunks_of_one_article_never_straddle_the_split(tmp_path):
    """The property the per-document split owes the measurement.

    With the sentence_id groups above, an article's chunks always land on
    one side across seeds; with the previous keyless records they straddle.
    """
    legislazione._write_source(
        "codice_civile",
        _two_chunk_article(),
        tmp_path,
        ChunkConfig(max_chars=500, overlap=0, min_chars=50),
        dry_run=False,
    )
    recs = _read_records(tmp_path / "legislazione_codice_civile.chunks.jsonl")
    groups = [r.get("sentence_id") for r in recs]
    assert len(groups) >= 2
    for seed in range(20):
        train_idx, val_idx = split_indices(len(recs), 0.5, seed, groups)
        train, val = set(train_idx), set(val_idx)
        assert (train == set(range(len(recs)))) or (val == set(range(len(recs)))), (
            f"article straddles the split at seed {seed}: train={train_idx} val={val_idx}"
        )


def test_keyless_records_do_straddle_the_split():
    """Control: without the key, the same chunks land on both sides.

    This is what the legislation records were until sentence_id was added —
    each keyless chunk its own group, neighbours split across train/val.
    """
    for seed in range(20):
        train_idx, val_idx = split_indices(2, 0.5, seed, [None, None])
        assert len(train_idx) == 1 and len(val_idx) == 1


# --- the administrative norms: recognised, but never in a default build -----

def _akn(urn):
    return (f'<akomaNtoso><meta><FRBRWork><FRBRthis value="{urn}/!main"/>'
            f'</FRBRWork></meta></akomaNtoso>')


def test_the_administrative_norms_are_recognised_in_an_opendata_zip():
    from eullm_forge.datasets.legal_it import _detect_source_from_akn

    assert _detect_source_from_akn(
        _akn("urn:nir:stato:decreto.legislativo:2010-07-02;104")
    ) == "codice_processo_amministrativo"
    assert _detect_source_from_akn(
        _akn("urn:nir:stato:legge:1990-08-07;241")
    ) == "legge_procedimento_amministrativo"


def test_a_default_build_stays_civil_and_criminal():
    """A vertical is only as focused as its corpus: the administrative norms
    enter a build only when named."""
    from eullm_forge.datasets.legal_it import (
        ALL_NORMATTIVA_LAWS,
        NORMATTIVA_LAWS,
        NORMATTIVA_LAWS_AMMINISTRATIVO,
    )

    default_ids = {law.id for law in NORMATTIVA_LAWS}
    assert not default_ids & {law.id for law in NORMATTIVA_LAWS_AMMINISTRATIVO}
    assert {law.id for law in ALL_NORMATTIVA_LAWS} >= default_ids | {
        "codice_processo_amministrativo", "ricorsi_amministrativi"}


# --- a law is recognised in the document, or the file is skipped ------------

def test_an_akn_without_a_urn_is_not_labelled_as_the_first_law():
    """An empty FRBRthis is a substring of every URN, so the matcher used to
    return the first catalogue entry and write a stranger's articles into
    that law's corpus — the Constitution being first."""
    from eullm_forge.datasets.legal_it import _detect_source_from_akn

    assert _detect_source_from_akn("<akomaNtoso><meta><FRBRWork/></meta></akomaNtoso>") is None
    assert _detect_source_from_akn("<akomaNtoso><FRBRthis value='  '/></akomaNtoso>") is None
    # A URN that is not ours stays unrecognised rather than borrowing a neighbour.
    assert _detect_source_from_akn(
        '<FRBRthis value="urn:nir:stato:decreto.legislativo:1789-07-28;172"/>') is None


def test_the_header_fallback_is_reachable_and_still_identifies_the_law():
    """The fallback exists for documents whose URN is not in a well-formed
    FRBRthis attribute; it was unreachable, because the empty value matched
    before it."""
    from eullm_forge.datasets.legal_it import _detect_source_from_akn

    single_quoted = (
        "<akomaNtoso><meta><FRBRthis "
        "value='urn:nir:stato:legge:1990-08-07;241'/></meta></akomaNtoso>"
    )
    assert _detect_source_from_akn(single_quoted) == "legge_procedimento_amministrativo"
