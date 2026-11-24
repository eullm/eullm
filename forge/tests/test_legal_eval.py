"""The release gate's scoring: v0.1c's answer to art. 2043 must fail it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from eullm_forge.eval import evaluate_qa, load_seed

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "legal_eval.py"
spec = importlib.util.spec_from_file_location("legal_eval", SCRIPT)
legal_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legal_eval)

# Verbatim from the v0.1c smoke test of 25 September.
V01C_2043 = (
    "L'articolo 2043 del codice civile stabilisce che chiunque ha agito in modo "
    "contrario ai doveri di buona fede, come previsto dagli articoli 1176 e 1375, "
    "è tenuto a risarcire il danno cagionato, a meno che non provi di non aver "
    "agito con colpa."
)
RIGHT_2043 = (
    "Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, "
    "obbliga colui che ha commesso il fatto a risarcire il danno."
)


def item(item_id):
    return next(it for it in load_seed() if it.id == item_id)


def test_the_v01c_answer_scores_below_a_correct_one():
    it = item("legal-it-civ-002")
    wrong = evaluate_qa([it], {it.id: V01C_2043})["per_item"][0]["keyword_coverage"]
    right = evaluate_qa([it], {it.id: RIGHT_2043})["per_item"][0]["keyword_coverage"]
    assert right == 1.0
    assert wrong <= 0.25


def test_the_summary_row_counts_full_marks_and_endings():
    items = [item("legal-it-civ-002"), item("legal-it-amm-002")]
    report = evaluate_qa(items, {"legal-it-civ-002": RIGHT_2043,
                                 "legal-it-amm-002": "Non lo so."})
    row = legal_eval.summary_row("x", "m", report, ended=2)
    assert dict(zip(legal_eval.CSV_HEADER, row))["fully_covered"] == 1
    # By name, not by position: row[-1] is not ended_turn any more.
    fields = dict(zip(legal_eval.CSV_HEADER, row))
    assert fields["items"] == 2 and fields["keyword_coverage"] == "0.500"
    assert fields["ended_turn"] == 2 and fields["keyword_items"] == 2


def test_quiet_grading_prints_no_item(tmp_path, capsys):
    """Held-out exam ids name the article asked; --quiet must not print them."""
    import csv
    import json

    spec = importlib.util.spec_from_file_location(
        "judge_answers", SCRIPT.parent / "judge_answers.py")
    ja = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ja)
    from eullm_forge.eval import Grade, ReferenceGrader

    src = tmp_path / "answers-m.jsonl"
    src.write_text("\n".join(json.dumps({"id": f"norm-termine-codice_civile-{n}",
                                         "question": "Q?", "answer": "A.",
                                         "reference": "R."}) for n in (1, 2, 3)) + "\n",
                      encoding="utf-8")

    class Batched:
        calls = 0

        def __call__(self, p):
            return self.batch([p])[0]

        def batch(self, ps):
            Batched.calls += 1
            return ["Grade: correct\nok"] * len(ps)

    grades = ja.grade_file(src, ReferenceGrader(Batched()), batch_size=2, quiet=True)
    assert [g.label for g in grades] == ["correct"] * 3
    assert Batched.calls == 2                       # 3 prompts in batches of 2
    assert "codice_civile" not in capsys.readouterr().out

    # A run whose grader lines we could not read must not produce a score: the
    # mean would be quietly too low, and it would look like a legal verdict.
    assert ja.unreadable_note(grades, src) is None
    one_unreadable = grades + [Grade("unparsed", "Grade: incorrect")]
    note = ja.unreadable_note(one_unreadable, src)
    assert note and "1 of 4" in note and "graded.jsonl" in note
    # The score is over the readable lines; the unreadable one is counted, not zeroed.
    row = ja.scored_row("m", one_unreadable)
    assert row[2] == 4 and row[6] == 1
    assert row[-1] == ja.summary_row("m", grades)[-1]

    # The 0-byte CSV case: the file exists, so a header written on existence
    # alone is skipped and the next row lands in its place, leaving DictReader
    # with nothing.
    out = tmp_path / "graded.csv"
    for module in (ja, legal_eval):
        out.write_text("", encoding="utf-8")
        module.append_csv_row(out, ["2026-09-27T00:00:00", "v0.3", "m", 3, 1, 0.5, 3, 3])
        module.append_csv_row(out, ["2026-09-27T00:01:00", "v0.3", "m2", 3, 1, 0.5, 3, 3])
        with out.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == module.CSV_HEADER, module.CSV_HEADER[:2]
        assert len(rows) == 3, module.CSV_HEADER[:2]
        assert {r[1] for r in rows[1:]} == {"v0.3"}    # no row read as a header
        out.unlink()

    # And on a file that already has a header, nothing is repeated.
    out = tmp_path / "graded2.csv"
    for module in (ja, legal_eval):
        out.unlink(missing_ok=True)
        for label in ("a", "b"):
            module.append_csv_row(out, ["t", label, "m", 1, 1, "1.0", 1, 1])
        with out.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["label"] for r in rows] == ["a", "b"], module.CSV_HEADER[:2]


def test_dangling_retrieval_flags_are_refused_like_absent(monkeypatch, capsys):
    """--embedder/--reranker/--retrieval-cache without their parents were
    silently ignored: the strings never touched, so a run meant to measure
    embedder X came back closed-book/BM25 with exit 0. --absent without
    --norms already refused; these do now. Before any heavy import, so no
    model or dependency is needed to run this."""
    cases = [
        ["model", "--absent"],
        ["model", "--embedder", "NOPE-XYZ"],
        ["model", "--norms", "n.jsonl", "--reranker", "R"],
        ["model", "--retrieval-cache", "c.txt"],
    ]
    for argv in cases:
        monkeypatch.setattr(sys, "argv", ["legal_eval.py", *argv])
        with pytest.raises(SystemExit) as refused:
            legal_eval.main()
        assert refused.value.code == 2
        assert "give --" in capsys.readouterr().err


# --- the chat prompt: thinking off, for every kind of template ---------------

def _tokenizer(template: str):
    """A real tokenizer with a real chat template, and nothing to download."""
    tokenizers = __import__("pytest").importorskip("tokenizers")
    transformers = __import__("pytest").importorskip("transformers")
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=raw)
    tok.chat_template = template
    return tok


# The part of Qwen3's hybrid template that decides whether the model thinks.
HYBRID = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
          "{% if enable_thinking is defined and enable_thinking is false %}"
          "<think>\n\n</think>\n\n{% endif %}{% endif %}")
# A template with no such switch, like Qwen3-4B-Instruct-2507's.
PLAIN = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def test_a_hybrid_model_is_asked_with_thinking_off():
    prompt = legal_eval.chat_prompt(_tokenizer(HYBRID), "Che cosa prevede l'art. 2043 c.c.?")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_a_template_without_the_switch_is_unchanged():
    prompt = legal_eval.chat_prompt(_tokenizer(PLAIN), "Domanda?")
    assert prompt == "<|im_start|>user\nDomanda?<|im_end|>\n<|im_start|>assistant\n"


# --- loading: one GPU, several, or none --------------------------------------

class _Auto:
    calls: list = []

    @classmethod
    def from_pretrained(cls, path, **kw):
        cls.calls.append(kw)
        return cls()

    def to(self, device):
        _Auto.calls.append({"to": device})
        return self


def test_a_model_too_big_for_one_gpu_is_spread_over_several():
    __import__("pytest").importorskip("torch")
    _Auto.calls = []
    legal_eval.load_model(_Auto, "m", 2)
    assert _Auto.calls[0].get("device_map") == "auto"
    assert not any("to" in c for c in _Auto.calls)


def test_one_gpu_or_none_loads_as_before():
    __import__("pytest").importorskip("torch")
    _Auto.calls = []
    legal_eval.load_model(_Auto, "m", 1)
    assert "device_map" not in _Auto.calls[0] and {"to": "cuda"} in _Auto.calls
    _Auto.calls = []
    legal_eval.load_model(_Auto, "m", 0)
    assert _Auto.calls == [_Auto.calls[0]] and "device_map" not in _Auto.calls[0]


class _TextOnly(_Auto):
    @classmethod
    def from_pretrained(cls, path, **kw):
        raise ValueError("Unrecognized configuration class <class 'Mistral3Config'> "
                         "for this kind of AutoModel: AutoModelForCausalLM.")


def test_an_image_and_text_model_is_loaded_by_the_fallback_class():
    __import__("pytest").importorskip("torch")
    _Auto.calls = []
    legal_eval.load_model(_TextOnly, "ministral", 2, fallback_cls=_Auto)
    assert _Auto.calls[0].get("device_map") == "auto"


def test_other_load_errors_are_not_swallowed():
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Unrecognized"):
        legal_eval.load_model(_TextOnly, "ministral", 1)       # no fallback given


def _judge_module():
    spec = importlib.util.spec_from_file_location(
        "judge_answers", SCRIPT.parent / "judge_answers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_judge_link_skips_answers_graded_already(tmp_path):
    import os

    judge = _judge_module()
    answers = tmp_path / "answers-v0.3-open.jsonl"
    answers.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
    graded = tmp_path / "answers-v0.3-open.graded.jsonl"
    assert not judge.already_graded(answers)
    graded.write_text('{"id": "a"}\n', encoding="utf-8")                       # cut short
    assert not judge.already_graded(answers)
    graded.write_text('{"id": "a"}\n{"id": "b"}\n', encoding="utf-8")
    assert judge.already_graded(answers)
    # the model was asked again after grading: the grades are stale
    later = graded.stat().st_mtime + 10
    os.utime(answers, (later, later))
    assert not judge.already_graded(answers)


def test_grades_under_another_rubric_go_to_their_own_directory(tmp_path):
    judge = _judge_module()
    answers = tmp_path / "answers-v0.3-open.jsonl"
    answers.write_text('{"id": "a"}\n', encoding="utf-8")
    (tmp_path / "answers-v0.3-open.graded.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")
    v2 = tmp_path / "devbig-graded-v2"
    assert judge.graded_path(answers, v2) == v2 / "answers-v0.3-open.graded.jsonl"
    assert judge.already_graded(answers)            # under the old rubric, yes
    assert not judge.already_graded(answers, v2)    # under the new one, not yet
