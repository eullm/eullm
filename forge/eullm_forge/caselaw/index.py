"""Retrieval over rulings (Consiglio di Stato, Corte di cassazione): BM25 and embeddings, by ruling.

The statute index (`eval.retrieval.NormIndex`) keeps one Counter per record,
fine for 10^5 articles and tens of GB for 3.5x10^5 chunks of rulings. Here
BM25 is postings in numpy arrays (for each word, the units holding it and
its weight), and the embeddings of the units are computed in shards cached
on disk, so a two-hour job that runs out of time leaves what it finished.

A **unit** is what is ranked: a chunk of a ruling, the same chunk with a
one-line card prefix in front (`schede.prefix`), or a card itself. Every
unit belongs to a ruling, and what is returned is rulings, best first, each
at the rank of its best unit: a question is answered by a ruling, not by
the third chunk of one.

Fusion is reciprocal rank (k=60, as `eval.dense`), the reranker reorders
the first ``rerank_depth`` units. The statute index is untouched.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.dense import RRF_K
from ..eval.retrieval import tokens


@dataclass
class Unit:
    ruling: str
    text: str


class SparseBM25:
    """BM25 over many documents, with numpy postings instead of a Counter each."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        import numpy as np

        vocab: dict[str, int] = {}
        rows: list[list[int]] = []          # per term: the documents holding it
        tfs: list[list[int]] = []
        lengths = np.zeros(len(texts), dtype=np.float32)
        for d, text in enumerate(texts):
            counts: dict[int, int] = {}
            toks = tokens(text)
            lengths[d] = len(toks)
            for w in toks:
                t = vocab.get(w)
                if t is None:
                    t = vocab[w] = len(vocab)
                    rows.append([])
                    tfs.append([])
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                rows[t].append(d)
                tfs[t].append(c)
        n = len(texts)
        avg = float(lengths.mean()) if n else 1.0
        self.vocab = vocab
        self.n = n
        self.docs = [np.asarray(r, dtype=np.int32) for r in rows]
        self.weights = []
        for r, tf in zip(self.docs, tfs, strict=True):
            tf = np.asarray(tf, dtype=np.float32)
            idf = math.log(1 + (n - len(r) + 0.5) / (len(r) + 0.5))
            norm = tf + k1 * (1 - b + b * lengths[r] / (avg or 1))
            self.weights.append((idf * tf * (k1 + 1) / norm).astype(np.float32))

    def ranking(self, question: str, depth: int) -> list[int]:
        import numpy as np

        scores = np.zeros(self.n, dtype=np.float32)
        hit = False
        for w in tokens(question):
            t = self.vocab.get(w)
            if t is not None:
                np.add.at(scores, self.docs[t], self.weights[t])
                hit = True
        if not hit:
            return []
        top = np.argsort(-scores, kind="stable")[:depth]
        return [int(i) for i in top if scores[i] > 0]


def shard_vectors(texts: Sequence[str], encode: Callable[[list[str]], object],
                  cache_dir: Path | None, key: str, shard: int = 20000):
    """Embeddings of ``texts``, computed shard by shard and cached as .npy files."""
    import numpy as np

    parts = []
    for s in range(0, len(texts), shard):
        path = cache_dir / f"{key}-{s // shard:05d}.npy" if cache_dir else None
        if path and path.is_file():
            parts.append(np.load(path))
            continue
        vec = np.asarray(encode(list(texts[s:s + shard])), dtype=np.float32)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".partial.npy")
            np.save(tmp, vec)
            tmp.replace(path)
        parts.append(vec)
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)


def units_key(units: Sequence[Unit], model_id: str) -> str:
    """Names the cache of one (units, model): changes when either does."""
    h = hashlib.sha256(model_id.encode())
    h.update(str(len(units)).encode())
    for u in units[:: max(1, len(units) // 1000)]:
        h.update(u.ruling.encode())
        h.update(u.text[:200].encode())
    return h.hexdigest()[:16]


def rrf_scores(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))]


@dataclass
class RulingIndex:
    """BM25 (+ embeddings) (+ reranker) over units, answering with rulings."""

    units: list[Unit]
    bm25: SparseBM25 | None = None
    vectors: object = None                       # numpy array, one row per unit
    query_fn: Callable[[str], object] | None = None
    reranker: object = None
    depth: int = 50
    rerank_depth: int = 20
    _by_ruling: dict = field(default_factory=dict, repr=False)

    def ranked_units(self, question: str) -> list[int]:
        import numpy as np

        rankings = []
        if self.bm25 is not None:
            rankings.append(self.bm25.ranking(question, self.depth))
        if self.vectors is not None and self.query_fn is not None:
            q = np.asarray(self.query_fn(question), dtype=np.float32).reshape(-1)
            sims = self.vectors @ q
            rankings.append([int(i) for i in np.argsort(-sims, kind="stable")[:self.depth]])
        fused = rrf_scores(rankings) if len(rankings) > 1 else (rankings[0] if rankings else [])
        if self.reranker is not None and fused:
            head = fused[:self.rerank_depth]
            s = self.reranker.scores(question, [self.units[i].text[:3000] for i in head])
            head = [i for _, i in sorted(zip(s, head, strict=True), key=lambda x: -x[0])]
            fused = head + fused[self.rerank_depth:]
        return fused

    def search(self, question: str, k: int = 10) -> list[str]:
        """Rulings, best first, each at the rank of its best unit."""
        out: list[str] = []
        for i in self.ranked_units(question):
            r = self.units[i].ruling
            if r not in out:
                out.append(r)
                if len(out) >= k:
                    break
        return out


def build_units(rulings: dict, chunks: Sequence[dict], *, cards: dict | None = None,
                prefix_chunks: bool = False, card_units: bool = False,
                court=None) -> list[Unit]:
    """The units of one index setting.

    ``chunks`` are the training chunk records (text, sentence_id); with
    ``prefix_chunks`` each carries its ruling's card prefix, with
    ``card_units`` every card is also a unit of its own (principles and
    norms, never the facts).
    """
    from .courts import CDS
    from .rulings import cited_number
    from .schede import prefix

    units: list[Unit] = []
    for c in chunks:
        rid = str(c.get("sentence_id") or c.get("source_id"))
        text = c.get("text") or ""
        if prefix_chunks and cards and rid in cards:
            r = rulings.get(rid)
            meta = {"sezione": getattr(r, "section", ""), "numero": cited_number(rid),
                    "data": (getattr(r, "meta", {}) or {}).get("DATA_PUBBLICAZIONE"),
                    "esito_openga": cards[rid].get("esito_openga")}
            text = prefix(cards[rid], meta, court or CDS) + "\n" + text
        units.append(Unit(rid, text))
    if card_units and cards:
        for rid, card in cards.items():
            text = (" ".join(card.get("principi", [])) + " Norme: "
                    + "; ".join(card.get("norme", [])))
            units.append(Unit(rid, f"[{card.get('materia', '')}] {text}"))
    return units
