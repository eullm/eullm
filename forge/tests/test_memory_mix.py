"""E1-mini's mix: shares by tokens, packing, the elaborating styles left out."""

from __future__ import annotations

import json

import pytest

from eullm_forge.datasets.memory_mix import (
    corpus_texts,
    draw,
    epoch_examples,
    pack,
    token_targets,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + "{cut")
    return path


def test_corpus_drops_cases_and_explanations_and_repeats(tmp_path):
    a = _jsonl(tmp_path / "rewrites-a.jsonl", [
        {"id": "c|1|parafrasi|0", "forma": "parafrasi", "text": "Uno."},
        {"id": "c|1|caso|0", "forma": "caso", "text": "Un caso."},
        {"id": "c|1|spiegazione|0", "forma": "spiegazione", "text": "Spiega."},
    ])
    b = _jsonl(tmp_path / "rewrites-b.jsonl", [
        {"id": "c|1|parafrasi|0", "forma": "parafrasi", "text": "Uno, ancora."},
        {"id": "x", "reason": "number", "v": 2},
    ])
    prog = _jsonl(tmp_path / "program.jsonl", [{"forma": "testo", "text": "Art. 1."}] * 2)
    assert corpus_texts([a, b, prog]) == ["Uno.", "Art. 1."]


def test_shares_are_of_tokens():
    assert token_targets(550, 0.25, 0.20) == (250, 200)
    with pytest.raises(ValueError):
        token_targets(100, 0.6, 0.4)


def test_draw_reaches_the_target_and_repeats_evenly():
    import random
    got = draw(["a", "b"], [10, 10], 45, random.Random(0))
    assert len(got) == 5 and abs(got.count("a") - got.count("b")) <= 1
    assert draw(["a"], [1], 0, random.Random(0)) == []


def test_pack_joins_with_eos_and_keeps_a_long_enough_tail():
    blocks = pack([[1, 2, 3], [4, 5]], eos_id=0, block=4)
    assert blocks == [[1, 2, 3, 0], [4, 5, 0]]
    assert pack([[1]], eos_id=0, block=16) == []


def test_an_epoch_mixes_to_its_shares_and_masks_only_the_replay_prompt():
    corpus = [[7] * 99 for _ in range(55)]          # 5,500 tokens with the eos
    general = [[8] * 49 for _ in range(100)]
    replay = [{"input_ids": [9] * 100, "labels": [-100] * 60 + [9] * 40} for _ in range(10)]
    ex, counts = epoch_examples(corpus, replay, [100] * 10, general, replay_share=0.25,
                                general_share=0.20, eos_id=0, block=256, seed=1, epoch=0)
    total = counts["corpus"] + counts["general"] + counts["replay"]
    assert abs(counts["replay"] / total - 0.25) < 0.03
    assert abs(counts["general"] / total - 0.20) < 0.03
    for e in ex:
        assert len(e["input_ids"]) == len(e["labels"]) == len(e["attention_mask"])
    rep = [e for e in ex if -100 in e["labels"]]
    assert len(rep) == counts["replay_examples"] and all(len(e["input_ids"]) == 100 for e in rep)
    assert all(e["labels"] == e["input_ids"] for e in ex if -100 not in e["labels"])
    again, _ = epoch_examples(corpus, replay, [100] * 10, general, replay_share=0.25,
                              general_share=0.20, eos_id=0, block=256, seed=1, epoch=0)
    assert again == ex                                # same seed and epoch, same data
