"""Hybrid retrieval with stand-in models: the fusion, the lookup, the reranker.

No model is loaded: document vectors are made by hand, the query function
maps a question to one of them, and the reranker is a lookup table. What is
pinned is the logic around the models, which is what a model swap keeps.
"""

from __future__ import annotations

import pytest

from eullm_forge.eval import NormIndex
from eullm_forge.eval.dataset import EvalItem
from eullm_forge.eval.dense import HybridIndex, describe, rrf
from eullm_forge.eval.norm_exam import retrieval_hits

np = pytest.importorskip("numpy")

FILLER = " Disposizione di dettaglio." * 5
RECORDS = [
    {"code": "codice_civile", "article_num": "", "chunk_index": 0,
     "text": "Art. 1. \n (Recesso) \n Il contraente può sciogliersi dal vincolo." + FILLER},
    {"code": "codice_civile", "article_num": "", "chunk_index": 0,
     "text": "Art. 2. \n (Locazione) \n Il locatore consegna la cosa locata." + FILLER},
    # the second chunk of art. 2: no header of its own
    {"code": "codice_civile", "article_num": "", "chunk_index": 1,
     "text": "Il conduttore deve dare congruo preavviso prima di lasciare l'immobile." + FILLER},
    {"code": "codice_penale", "article_num": "", "chunk_index": 0,
     "text": "Art. 1. \n (Reati) \n Nessuno può essere punito per un fatto non previsto." + FILLER},
]


class FakeReranker:
    model_id = "fake-reranker"

    def __init__(self, prefer: str):
        self.prefer = prefer

    def scores(self, question, docs):
        return [1.0 if self.prefer in d else 0.0 for d in docs]


def _index(query_to_record: dict[str, int], reranker=None, use_bm25=True):
    base = NormIndex(RECORDS)
    vecs = np.eye(len(RECORDS), dtype=np.float32)
    return HybridIndex(base, vecs, lambda q: vecs[query_to_record.get(q, 0)],
                       reranker, use_bm25=use_bm25)


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([[1, 2, 3], [3, 1, 2]])[0] == 1
    assert rrf([[5], [6]]) == [5, 6]          # a tie keeps index order
    assert rrf([]) == []


def test_equal_similarities_come_back_in_index_order():
    """The rule rrf() sorts on, and that its test states, holds at the source.

    rrf() sorts a fused ranking on (-score, i), so a tie keeps index order --
    but it cannot restore an order it is not given. np.argsort's default kind
    is introsort, stable only up to sixteen elements, so with twenty records
    two identical vectors came back the later one first and rrf() had nothing
    left to break. The corpus is 10^5 records.
    """
    twenty = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
               "text": f"Art. {i + 1}. Testo generico numero {i}."} for i in range(20)]
    vectors = np.full((20, 4), 0.2, dtype=np.float32)
    vectors[3] = vectors[17] = np.array([1, 0, 0, 0], dtype=np.float32)
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    index = HybridIndex(NormIndex(twenty), vectors, lambda _: query)

    assert np.array_equal(vectors[3], vectors[17])
    assert index.dense_ranking("Argomento senza nessuna parola in comune", None)[:2] == [3, 17]


def test_the_named_article_still_comes_first():
    idx = _index({"Che cosa prevede l'art. 2 del codice civile?": 0})
    found = idx.search("Che cosa prevede l'art. 2 del codice civile?", k=3)
    assert found[0] is RECORDS[1]


def test_a_topic_question_bm25_misses_is_found_by_meaning():
    q = "Nel codice civile, in materia di disdetta dell'affitto, quale termine?"
    # BM25 shares no word with art. 2's continuation chunk; the embedding does
    idx = _index({q: 2}, use_bm25=False)
    found = idx.search(q, k=1)
    assert found == [RECORDS[2]]
    assert idx.articles_of(found[0]) == ["2"]     # the continuation is art. 2


def test_a_named_code_keeps_the_dense_ranking_inside_it():
    q = "Nel codice penale, quando si è puniti?"
    idx = _index({q: 0}, use_bm25=False)          # nearest vector: a civil-code record
    assert all(r["code"] == "codice_penale" for r in idx.search(q, k=2))


def test_the_reranker_reorders_the_fused_list():
    q = "Nel codice civile, chi consegna la cosa?"
    reranked = _index({q: 0}, FakeReranker("locatore"))
    assert reranked.search(q, k=1) == [RECORDS[1]]
    assert describe(reranked) == "bm25+dense, rerank fake-reranker"
    assert describe(NormIndex(RECORDS)) == "bm25"


def test_hits_count_a_continuation_chunk_as_its_article():
    q = "Nel codice civile, in materia di preavviso del conduttore, quale termine?"
    item = EvalItem(id="norm-termine_argomento-codice_civile-2", domain="legal", lang="it",
                    question=q, metadata={"tipo": "termine_argomento",
                                          "code": "codice_civile", "articolo": "2"})
    idx = _index({q: 2}, use_bm25=False)
    assert retrieval_hits([item], idx, k=1)["termine_argomento"]["top1"] == 1.0
