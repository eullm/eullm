"""tools/answer_check.py, the release gate: the parts that do not need a
binary or a model (those run in release-engine.yml's answer-check job)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("answer_check", HERE / "answer_check.py")
answer_check = importlib.util.module_from_spec(spec)
sys.modules["answer_check"] = answer_check
spec.loader.exec_module(answer_check)


def test_the_kv_cache_line_is_read_from_the_banner():
    log = "eullm ready.\n  Flash attn:    off (--no-flash-attn)\n  KV cache:      K=Q8_0 V=F16\n"
    assert answer_check.kv_line(log) == "K=Q8_0 V=F16"
    assert answer_check.kv_line("no banner") == "?"


def test_the_settings_cover_the_combinations_that_broke():
    labels = [label for label, _ in answer_check.SETTINGS]
    assert len(labels) == len(set(labels))
    flags = dict(answer_check.SETTINGS)
    # 0.9.50-0.10.0 wrote gibberish with exactly this (issue #140).
    assert flags["q4_0-q4_0-nofa"] == ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
                                       "--no-flash-attn"]
    assert flags["f16-f16"] == []


def test_every_question_names_a_lowercase_answer():
    for question, want in answer_check.QUESTIONS:
        assert want == want.lower() and want in {"paris", "144", "mars"}, question
