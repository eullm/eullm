"""drop_prompt_notes.py: the missing-article note out of a share of the prompts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.eval import open_book_prompt

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "drop_prompt_notes.py"


def _load():
    spec = importlib.util.spec_from_file_location("drop_prompt_notes", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REC = {"text": "Art. 2969. Rilievo d'ufficio della decadenza.", "code": "codice_civile",
       "article_num": "2969", "chunk_index": 0}


def test_the_note_the_exam_writes_is_taken_out_and_the_rest_kept():
    mod = _load()
    q = "Che cosa prevede l'art. 3434 del codice civile?"
    noted = open_book_prompt(q, [REC], note="Nota: nella raccolta normativa non è presente "
                                            "art. 3434 del codice civile.")
    bare = mod.without_note(noted)
    assert bare == open_book_prompt(q, [REC])
    assert mod.without_note(open_book_prompt(q, [REC])) is None


def test_the_share_is_taken_out_by_id_and_other_rows_pass(tmp_path, capsys):
    mod = _load()
    q = "Che cosa prevede l'art. 3434 del codice civile?"
    note = "Nota: nella raccolta normativa non è presente art. 3434 del codice civile."
    rows = [{"id": f"r{i}", "tipo": "inesistente",
             "prompt": [{"role": "user", "content": open_book_prompt(q, [REC], note=note)}]}
            for i in range(200)]
    rows.append({"id": "plain", "tipo": "termine",
                 "prompt": [{"role": "user", "content": open_book_prompt(q, [REC])}]})
    src = tmp_path / "p.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    out = tmp_path / "o.jsonl"
    assert mod.main([str(src), "--share", "0.7", "--out", str(out)]) == 0
    got = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(got) == 201 and got[-1] == rows[-1]
    bare = sum(not r["prompt"][0]["content"].startswith("Nota:") for r in got[:200])
    assert 110 <= bare <= 170                     # about 70%, by hash
    # the same rows each time
    out2 = tmp_path / "o2.jsonl"
    mod.main([str(src), "--share", "0.7", "--out", str(out2)])
    assert out2.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
    assert "3434" not in capsys.readouterr().out
