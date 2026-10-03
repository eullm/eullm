"""Retrieval by meaning as well as by words, for the questions BM25 misses.

On the held-out exam BM25 puts the right article in the first three texts
for 97-99% of the questions that name it, and for 66% of those that ask by
topic. A topic question uses the words of the problem ("in materia di
recesso dal contratto"), and the article may use others; that is the miss an
embedding model is for. JuriFindIT (EACL 2026), on Italian statutes, has BM25
at 43.0 Recall@5 and Qwen3-Embedding-8B, untrained, at 74.9.

`HybridIndex` keeps everything `NormIndex` does -- the named article first,
by exact lookup, which no ranking beats -- and replaces the BM25 fill with
reciprocal-rank fusion of BM25 and an embedding model, optionally reordered
by a reranker that reads question and text together. Both models are
Apache-2.0 (Qwen3-Embedding, Qwen3-Reranker) and are used exactly as their
model cards show, with transformers only: no new dependency.

The document embeddings are computed once per (model, legislation files)
and cached as .npy beside the given cache directory, so a second run, or the
exam that uses the index after it was measured, does not pay for them again.

torch is imported only when a model is loaded; the fusion logic is plain
Python and is tested without it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from .retrieval import NormIndex, label, named_code

# One sentence saying what is retrieved, as Qwen3-Embedding and -Reranker
# expect. English, as their cards recommend; the question itself is Italian.
TASK = ("Given a question about Italian law, retrieve the article of the code or "
        "statute that answers it")

# Reciprocal-rank fusion constant: the usual 60, so neither ranking's top
# dominates the other's.
RRF_K = 60


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal-rank fusion of several rankings of record indices."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: (-score[i], i))


def doc_text(record: dict, max_chars: int = 3000) -> str:
    """What a record is embedded as: how it is cited, then its text."""
    return f"{label(record)}\n{record.get('text', '')[:max_chars]}"


class Embedder:
    """Qwen3-Embedding, as on its model card: last-token pooling, left
    padding, normalised vectors, an instruction on the query side only."""

    def __init__(self, model_id: str, *, batch_size: int = 16, max_length: int = 1024):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        cuda = torch.cuda.is_available()
        self.model = AutoModel.from_pretrained(
            model_id, dtype=torch.bfloat16 if cuda else torch.float32)
        self.model.to("cuda" if cuda else "cpu").eval()

    def encode(self, texts: list[str], *, query: bool = False):
        """Unit vectors, one row per text, as a float32 numpy array."""
        import numpy as np

        torch = self.torch
        if query:
            texts = [f"Instruct: {TASK}\nQuery:{t}" for t in texts]
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            # left padding: the last position is every sequence's last token
            vec = torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1)
            out.append(vec.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


class Reranker:
    """Qwen3-Reranker, as on its model card: the probability of "yes" after
    the instruction, the question and the text."""

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based "
              "on the Query and the Instruct provided. Note that the answer can only be "
              "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_id: str, *, batch_size: int = 8, max_length: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16 if cuda else torch.float32)
        self.model.to("cuda" if cuda else "cpu").eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def scores(self, question: str, docs: list[str]) -> list[float]:
        torch = self.torch
        pairs = [f"<Instruct>: {TASK}\n<Query>: {question}\n<Document>: {d}" for d in docs]
        out: list[float] = []
        room = self.max_length - len(self.prefix) - len(self.suffix)
        for i in range(0, len(pairs), self.batch_size):
            enc = self.tok(pairs[i:i + self.batch_size], padding=False,
                           truncation="longest_first", return_attention_mask=False,
                           max_length=room)
            enc["input_ids"] = [self.prefix + ids + self.suffix for ids in enc["input_ids"]]
            batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out.extend(torch.log_softmax(two, dim=1)[:, 1].exp().tolist())
        return out


def files_fingerprint(paths: Sequence[str | Path]) -> str:
    """A short id of the legislation files, so a cache built on other files
    (or on a regenerated corpus) is not reused."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        st = Path(p).stat()
        h.update(f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def cached_doc_vectors(index: NormIndex, embedder: Embedder, cache_dir: Path | None,
                       fingerprint: str):
    """The records' embeddings, from the cache when it holds them."""
    import numpy as np

    path = None
    if cache_dir is not None:
        name = re.sub(r"[^A-Za-z0-9.-]+", "_", embedder.model_id.strip("/"))[-80:]
        path = Path(cache_dir) / f"docvec-{name}-{fingerprint}-{len(index.records)}.npy"
        if path.is_file():
            return np.load(path)
    vecs = embedder.encode([doc_text(r) for r in index.records])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
    return vecs


class HybridIndex:
    """`NormIndex` with BM25 and an embedding model fused, and a reranker on top.

    ``depth`` candidates come from each ranking; the reranker, when given,
    reorders the first ``rerank_depth`` of the fused list. The named article
    still comes first, by exact lookup, exactly as in `NormIndex.search`.
    """

    def __init__(self, base: NormIndex, doc_vectors, query_fn: Callable[[str], object],
                 reranker: Reranker | None = None, *, depth: int = 50,
                 rerank_depth: int = 20, use_bm25: bool = True):
        self.base = base
        self.records = base.records
        self.doc_vectors = doc_vectors
        self.query_fn = query_fn
        self.reranker = reranker
        self.depth = depth
        self.rerank_depth = rerank_depth
        self.use_bm25 = use_bm25
        self._pos = {id(r): i for i, r in enumerate(self.records)}

    # What callers of NormIndex use, passed through.
    def by_article(self, question: str) -> list[dict]:
        return self.base.by_article(question)

    def missing_article_note(self, question: str) -> str:
        return self.base.missing_article_note(question)

    def articles_of(self, record: dict) -> list[str]:
        return self.base.articles_of(record)

    def dense_ranking(self, question: str, code: str | None) -> list[int]:
        import numpy as np

        q = np.asarray(self.query_fn(question), dtype=np.float32).reshape(-1)
        sims = self.doc_vectors @ q
        # Stable, so equal similarities come back in index order: rrf() sorts
        # a fused ranking on (-score, i) and its test says a tie keeps index
        # order, which it cannot restore if the ranking it is given already
        # has them out of it. np.argsort's default kind is introsort, stable
        # only up to 16 elements -- and the corpus is 10^5 records.
        order = np.argsort(-sims, kind="stable")
        out = []
        for i in order:
            if code and self.records[int(i)].get("code") != code:
                continue
            out.append(int(i))
            if len(out) >= self.depth:
                break
        return out

    def ranked(self, question: str) -> list[int]:
        """Record indices, best first, before the named-article lookup."""
        code = named_code(question)
        rankings = [self.dense_ranking(question, code)]
        if self.use_bm25:
            rankings.append([self._pos[id(r)]
                             for r in self.base.bm25(question, self.depth, code=code)])
        fused = rrf(rankings)
        if self.reranker is not None and fused:
            head = fused[:self.rerank_depth]
            s = self.reranker.scores(question, [doc_text(self.records[i]) for i in head])
            head = [i for _, i in sorted(zip(s, head), key=lambda x: -x[0])]
            fused = head + fused[self.rerank_depth:]
        return fused

    def search(self, question: str, k: int = 3) -> list[dict]:
        found = self.by_article(question)[:k]
        seen = {id(r) for r in found}
        for i in self.ranked(question):
            if len(found) >= k:
                break
            r = self.records[i]
            if id(r) not in seen:
                found.append(r)
                seen.add(id(r))
        return found


def build_hybrid(norms: Sequence[str | Path], embedder_id: str, *,
                 reranker_id: str | None = None, cache_dir: Path | None = None,
                 use_bm25: bool = True, heading_boost: int = 0) -> HybridIndex:
    """Load the legislation, the models and the cached embeddings."""
    base = NormIndex.from_files(list(norms))
    if heading_boost:
        base = NormIndex(base.records, heading_boost=heading_boost)
    emb = Embedder(embedder_id)
    vecs = cached_doc_vectors(base, emb, cache_dir, files_fingerprint(norms))
    rr = Reranker(reranker_id) if reranker_id else None
    return HybridIndex(base, vecs, lambda q: emb.encode([q], query=True)[0], rr,
                       use_bm25=use_bm25)


def describe(index) -> str:
    """One line saying which retriever an index is, for logs and answers files."""
    if isinstance(index, HybridIndex):
        parts = ["bm25+dense" if index.use_bm25 else "dense"]
        if index.reranker is not None:
            parts.append(f"rerank {index.reranker.model_id}")
        return ", ".join(parts)
    return "bm25"


__all__ = ["Embedder", "HybridIndex", "Reranker", "build_hybrid", "describe", "rrf",
           "doc_text", "cached_doc_vectors", "files_fingerprint", "TASK"]
