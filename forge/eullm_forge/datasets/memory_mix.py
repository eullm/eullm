"""The training mix of E1-mini: the memory corpus, a replay of the post-training, general Italian.

The legal-it roadmap (``docs/research/legal-it-roadmap.md``, phase 2) trains
the knowledge of the articles into the backbone by continued pretraining
through LoRA, on the corpus of the nine codes (`law_corpus` program rows and
`law_rewrites.py` rewrites), with in every batch a replay of what post-training
taught (the open-book pairs, the false premises: 20-30%) and 20% of general
Italian text, so that the model keeps answering as it was trained to and keeps
its language.

Shares are of TOKENS, not rows: a corpus row is a hundred-odd tokens, an
open-book pair a thousand or more, and a share by rows would make the replay
most of the training. Every epoch takes a fresh draw of replay and general
text, so over the epochs the replay goes through more of its pairs.

Corpus rows of the teacher's elaborating styles (``caso``, ``spiegazione``)
are left out: 3 of 6 cases read were wrong in law (roadmap, phase 2).
Everything here is pure Python and deterministic for a seed; tokenizing,
packing and training are `forge/scripts/memory_cpt.py`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path

# Teacher styles never trained on: they elaborate, and the teacher errs there.
EXCLUDED_STYLES = frozenset({"caso", "spiegazione"})


def read_jsonl(path: Path) -> Iterable[dict]:
    """Rows of a JSONL file; a line cut by a killed writer is skipped."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def corpus_texts(paths: Iterable[Path]) -> list[str]:
    """Texts of the memory corpus: program rows and rewrites, deduplicated.

    A rewrite repeated across sibling files (two chains that crossed) counts
    once, by its id; program rows have no id and are kept by text.
    """
    texts: list[str] = []
    seen: set[str] = set()
    for p in paths:
        for r in read_jsonl(p):
            text = (r.get("text") or "").strip()
            if not text or r.get("forma") in EXCLUDED_STYLES:
                continue
            key = r.get("id") or text
            if key in seen:
                continue
            seen.add(key)
            texts.append(text)
    return texts


def replay_pairs(paths: Iterable[Path]) -> list[dict]:
    """Instruction pairs (``instruction``, ``output``) of the post-training."""
    pairs = []
    for p in paths:
        for r in read_jsonl(p):
            if r.get("instruction") and r.get("output"):
                pairs.append({"instruction": r["instruction"], "output": r["output"]})
    return pairs


def general_texts(paths: Iterable[Path]) -> list[str]:
    """General Italian texts (``text``), e.g. prefetch_general_it.py's Wikipedia sample."""
    return [t for p in paths for r in read_jsonl(p) if (t := (r.get("text") or "").strip())]


def token_targets(corpus_tokens: int, replay_share: float,
                  general_share: float) -> tuple[int, int]:
    """Tokens of replay and of general text an epoch, for those shares of the total."""
    if replay_share < 0 or general_share < 0 or replay_share + general_share >= 1:
        raise ValueError("replay and general shares must be >= 0 and sum to less than 1")
    total = corpus_tokens / (1 - replay_share - general_share)
    return round(total * replay_share), round(total * general_share)


def draw(items: list, sizes: list[int], target: int, rng: random.Random) -> list:
    """Items drawn at random until their sizes reach ``target``.

    Without replacement while the pool lasts, then again from a fresh
    shuffle: a target larger than the pool repeats items, evenly.
    """
    if target <= 0 or not items:
        return []
    out, got = [], 0
    while got < target:
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            out.append(items[i])
            got += max(1, sizes[i])
            if got >= target:
                break
    return out


def pack(token_lists: Iterable[list[int]], eos_id: int, block: int) -> list[list[int]]:
    """Texts joined with an end-of-text token and cut into blocks of ``block`` tokens.

    The continued-pretraining layout: no padding, every token trained on. The
    last, short block is kept if at least a quarter full.
    """
    blocks: list[list[int]] = []
    buf: list[int] = []
    for ids in token_lists:
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= block:
            blocks.append(buf[:block])
            buf = buf[block:]
    if len(buf) >= block // 4:
        blocks.append(buf)
    return blocks


def epoch_examples(corpus: list[list[int]], replay: list[dict], replay_sizes: list[int],
                   general: list[list[int]], *, replay_share: float, general_share: float,
                   eos_id: int, block: int, seed: int,
                   epoch: int) -> tuple[list[dict], dict]:
    """One epoch: packed text blocks and replay examples, shuffled together.

    ``corpus`` and ``general`` are token ids of each text; ``replay`` are
    finished features (``input_ids``, ``labels``: the prompt masked) with
    ``replay_sizes`` their lengths. The corpus is shuffled and packed whole;
    general text and replay are drawn to their token shares. Returns the
    examples (``input_ids``, ``attention_mask``, ``labels``) and the epoch's token counts.
    """
    rng = random.Random(seed * 1000 + epoch)
    corpus_tokens = sum(len(t) + 1 for t in corpus)
    want_replay, want_general = token_targets(corpus_tokens, replay_share, general_share)
    order = list(range(len(corpus)))
    rng.shuffle(order)
    gen = draw(general, [len(t) + 1 for t in general], want_general, rng)
    texts = [corpus[i] for i in order] + gen
    rng.shuffle(texts)
    blocks = pack(texts, eos_id, block)
    rep = draw(replay, replay_sizes, want_replay, rng)
    examples = [{"input_ids": b, "attention_mask": [1] * len(b), "labels": list(b)}
                for b in blocks]
    examples += [{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"]),
                  "labels": r["labels"]} for r in rep]
    rng.shuffle(examples)
    counts = {"corpus": corpus_tokens, "general": sum(len(t) + 1 for t in gen),
              "replay": sum(len(r["input_ids"]) for r in rep),
              "blocks": len(blocks), "replay_examples": len(rep)}
    return examples, counts


def tokenize_texts(texts: list[str], encode: Callable[[list[str]], list[list[int]]],
                   max_tokens: int, batch: int = 1000) -> list[list[int]]:
    """Token ids of each text, in batches, each cut at ``max_tokens``."""
    out: list[list[int]] = []
    for i in range(0, len(texts), batch):
        out.extend(ids[:max_tokens] for ids in encode(texts[i:i + batch]))
    return out
