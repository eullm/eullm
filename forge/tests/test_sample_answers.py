"""sample_answers.py: several answers per verifiable question, graded by program."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import eullm_forge.eval.gguf as gguf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_answers.py"


class _Tok:
    eos_token_id = 2

    @classmethod
    def from_pretrained(cls, _path):
        return cls()

    def convert_tokens_to_ids(self, _t):
        return 3

    def apply_chat_template(self, messages, **kw):
        assert kw["enable_thinking"] is False and kw["add_generation_prompt"] is True
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class _Server:
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_every_question_is_asked_n_times_and_graded(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_Tok))
    monkeypatch.setattr(gguf, "LlamaServer", _Server)
    seen = {}

    def fake_generate(server, tok, prompts, *, temperature, seeds, **kw):
        seen.update(prompts=prompts, seeds=seeds, temperature=temperature)
        # each question: two right answers and one wrong
        return [("Il termine è di 60 giorni." if i % 3 else "Sono 30 giorni.", True)
                for i in range(len(prompts))]

    monkeypatch.setattr(gguf, "generate_answers_gguf", fake_generate)
    prompts = tmp_path / "grpo.jsonl"
    rows = [{"id": "q1", "prompt": [{"role": "user", "content": "Domanda 1"}],
             "tipo": "termine", "keywords": ["60 giorni"]},
            {"id": "q2", "prompt": [{"role": "user", "content": "Domanda 2"}],
             "tipo": "contenuto", "keywords": []}]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    system = tmp_path / "sys.txt"
    system.write_text("Sei legal-it.\n", encoding="utf-8")
    out, keep = tmp_path / "samples.jsonl", tmp_path / "informative.jsonl"
    spec = importlib.util.spec_from_file_location("sample_answers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["merged", "--gguf", "m.gguf", "--prompts", str(prompts), "--n", "3",
                     "--system-file", str(system), "--out", str(out),
                     "--informative-out", str(keep)]) == 0
    assert seen["prompts"] == ["system:Sei legal-it.|user:Domanda 1"] * 3   # judged left out
    assert len(set(seen["seeds"])) == 3 and seen["temperature"] == 0.8
    [r] = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert [s["score"] for s in r["samples"]] == [0.0, 1.0, 1.0]
    assert abs(r["pass_rate"] - 2 / 3) < 1e-9
    [k] = [json.loads(ln) for ln in keep.read_text(encoding="utf-8").splitlines()]
    assert "samples" not in k and k["id"] == "q1"
    printed = capsys.readouterr().out
    assert "Domanda" not in printed and "giorni" not in printed
