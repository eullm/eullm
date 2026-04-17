"""Tests for the italgiure.giustizia.it Cassazione fetcher.

The real Solr endpoint is IP-restricted (blocks datacenter egress), so tests
use a mocked requests.Session built from a synthetic Solr JSON response.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eullm_forge.datasets.italgiure import (
    ItalgiureQuery,
    _doc_to_record,
    fetch_italgiure,
    load_italgiure_jsonl,
)


SAMPLE_DOC = {
    "id": "snpen2026513137S",
    "kind": "snpen",
    "numdec": "13137",
    "szdec": "5",
    "ssz": "0",
    "tipoprov": "Sentenza",
    "anno": "2026",
    "datdec": "20251210",
    "datdep": ["20260409"],
    "ecli": "ECLI:IT:CASS:2026:13137PEN",
    "presidente": ["CATENA ROSSELLA"],
    "relatore": ["BELMONTE MARIA TERESA"],
    "filename": ["./20260409/snpen@s50@a2026@n13137@tS.pdf"],
    "ocr": ["E APPELLO di REGGIO CALABRIA visti gli atti, " * 40],
}


def _make_solr_response(docs: list[dict], num_found: int) -> dict:
    return {
        "responseHeader": {"status": 0, "QTime": 15},
        "response": {"numFound": num_found, "start": 0, "docs": docs},
    }


def test_query_lucene_all_sezioni():
    q = ItalgiureQuery(kind="snpen", anno=2026)
    assert q.lucene() == 'kind:"snpen" AND anno:"2026"'
    assert q.slug() == "snpen_2026"


def test_query_lucene_with_sezione():
    q = ItalgiureQuery(kind="snciv", anno=2024, sezione=5)
    assert q.lucene() == 'kind:"snciv" AND anno:"2024" AND szdec:"5"'
    assert q.slug() == "snciv_2024_sez5"


def test_doc_to_record_happy_path():
    rec = _doc_to_record(SAMPLE_DOC)
    assert rec is not None
    assert rec["source"] == "italgiure"
    assert rec["sentence_id"] == "snpen/2026/13137"
    assert rec["article_num"] == "13137"
    assert "E APPELLO di REGGIO CALABRIA" in rec["text"]
    assert rec["metadata"]["ecli"] == "ECLI:IT:CASS:2026:13137PEN"
    assert rec["metadata"]["presidente"] == "CATENA ROSSELLA"
    assert rec["metadata"]["datdep"] == "20260409"
    assert rec["url"].endswith("snpen@s50@a2026@n13137@tS.pdf")


def test_doc_to_record_drops_empty_ocr():
    doc = dict(SAMPLE_DOC, ocr=[])
    assert _doc_to_record(doc) is None


def test_doc_to_record_drops_short_text():
    doc = dict(SAMPLE_DOC, ocr=["abc"])
    assert _doc_to_record(doc) is None


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, pages: list[dict]):
        self._pages = pages
        self._call = 0
        self.headers = {}
        self.get_calls: list[dict] = []

    def get(self, url, params=None, timeout=None):  # noqa: D401
        self.get_calls.append({"url": url, "params": dict(params or {})})
        if not self._pages:
            return _FakeResponse(_make_solr_response([], 0))
        idx = min(self._call, len(self._pages) - 1)
        self._call += 1
        return _FakeResponse(self._pages[idx])


def test_fetch_italgiure_writes_jsonl_and_checkpoint(tmp_path: Path):
    docs_page1 = [dict(SAMPLE_DOC, numdec=str(n)) for n in range(13100, 13103)]
    docs_page2 = [dict(SAMPLE_DOC, numdec=str(n)) for n in range(13200, 13201)]
    pages = [
        _make_solr_response(docs_page1, 4),
        _make_solr_response(docs_page2, 4),
    ]
    session = _FakeSession(pages)

    out = fetch_italgiure(
        tmp_path,
        years=[2026],
        kinds=["snpen"],
        rate_limit_sec=0.0,
        rows=3,
        session=session,
    )

    jsonl = out / "italgiure_snpen_2026.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    first = json.loads(lines[0])
    assert first["source"] == "italgiure"
    assert first["sentence_id"].startswith("snpen/2026/")

    progress = json.loads((out / "_progress.json").read_text(encoding="utf-8"))
    # Slice marked complete (-1 sentinel)
    assert progress["snpen_2026"] == -1

    # Verify Lucene query shape + paging params we actually sent
    first_params = session.get_calls[0]["params"]
    assert first_params["q"] == '(kind:"snpen" AND anno:"2026")'
    assert first_params["wt"] == "json"
    assert first_params["rows"] == "3"
    assert first_params["start"] == "0"
    assert session.get_calls[1]["params"]["start"] == "3"


def test_fetch_italgiure_resumes_from_progress(tmp_path: Path):
    # First run: only get 1 doc (truncated by max_docs_per_query)
    pages = [
        _make_solr_response([dict(SAMPLE_DOC, numdec="1")], 3),
    ]
    session1 = _FakeSession(pages)
    fetch_italgiure(
        tmp_path,
        years=[2026],
        kinds=["snpen"],
        rate_limit_sec=0.0,
        rows=1,
        max_docs_per_query=1,
        session=session1,
    )
    progress = json.loads((tmp_path / "_progress.json").read_text())
    # After fetching 1 row (start=0, rows=1), next offset is 1, not -1
    # because max_docs_per_query truncated the iterator.
    assert progress["snpen_2026"] == 1

    # Second run: resume from offset 1
    pages2 = [
        _make_solr_response(
            [dict(SAMPLE_DOC, numdec="2"), dict(SAMPLE_DOC, numdec="3")],
            3,
        ),
    ]
    session2 = _FakeSession(pages2)
    fetch_italgiure(
        tmp_path,
        years=[2026],
        kinds=["snpen"],
        rate_limit_sec=0.0,
        rows=2,
        session=session2,
    )
    # First call should have used start=1 (resumed), not 0
    assert session2.get_calls[0]["params"]["start"] == "1"

    all_records = load_italgiure_jsonl(tmp_path)
    numdecs = sorted(r["article_num"] for r in all_records)
    assert numdecs == ["1", "2", "3"]


def test_fetch_italgiure_skips_complete_slices(tmp_path: Path):
    (tmp_path / "_progress.json").write_text(
        json.dumps({"snpen_2026": -1}), encoding="utf-8"
    )
    session = _FakeSession([])
    fetch_italgiure(
        tmp_path,
        years=[2026],
        kinds=["snpen"],
        rate_limit_sec=0.0,
        session=session,
    )
    assert session.get_calls == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
