"""pairs_exam_overlap.py traces generated pairs back to their article and
counts the ones that ask about an article of the held-out exam."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eullm_forge.datasets.instruct_gen import GenConfig, make_jobs
from eullm_forge.eval.dataset import EvalItem, save_eval_set

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jupiter" / "pairs_exam_overlap.py"
spec = importlib.util.spec_from_file_location("pairs_exam_overlap", SCRIPT)
overlap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlap)

BODY = ("Il debitore che non esegue esattamente la prestazione dovuta è tenuto al "
        "risarcimento del danno, se non prova che l'inadempimento è stato determinato "
        "da impossibilità della prestazione derivante da causa a lui non imputabile. ") * 3


def corpus(tmp_path: Path) -> Path:
    rows = []
    for n in range(1, 13):
        rows.append({"code": "codice_civile", "article_num": "",
                     "text": f"Art. {n}.\n(Rubrica {n})\n{BODY}"})
    # A chunk without a header continues the article before it.
    rows.append({"code": "codice_civile", "article_num": "", "text": BODY})
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def exam(tmp_path: Path, articles: list[str]) -> Path:
    items = [EvalItem(id=f"norm-testo-codice_civile-{a}", domain="legal", lang="it",
                      question="?", metadata={"tipo": "testo", "code": "codice_civile",
                                              "articolo": a})
             for a in articles]
    items.append(EvalItem(id="norm-inesistente-codice_civile-9999", domain="legal", lang="it",
                          question="?", metadata={"tipo": "inesistente",
                                                  "code": "codice_civile", "articolo": "9999"}))
    path = tmp_path / "norm-exam.jsonl"
    save_eval_set(items, path)
    return path


def test_pairs_on_exam_articles_are_counted_and_left_out(tmp_path, capsys):
    train = corpus(tmp_path)
    records = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    jobs = make_jobs(records, 100, seed=0, cfg=GenConfig())
    assert jobs, "the toy corpus must yield jobs"
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps({"instruction": "q", "output": "a", "task": j.task,
                                         "key": j.key}) + "\n" for j in jobs), encoding="utf-8")
    # Article 12 is also where the header-less last chunk belongs.
    exam_path = exam(tmp_path, ["3", "12"])
    clean = tmp_path / "clean.jsonl"

    assert overlap.main(["--corpus", str(train), "--pairs", str(pairs), "--exam", str(exam_path),
                         "--seeds", "0", "--limit", "100", "--write-clean", str(clean)]) == 0
    out = capsys.readouterr().out

    on_exam = ("Art. 3.", "Art. 12.")
    expected_bad = {j.key for j in jobs if records[j.record]["text"].startswith(on_exam)
                    or j.record == len(records) - 1}
    assert expected_bad and len(expected_bad) < len(jobs)
    assert f"pairs about an exam article: {len(expected_bad)}" in out
    assert "exam articles with pairs about them: 2 of 2" in out
    kept = {json.loads(line)["key"] for line in clean.read_text(encoding="utf-8").splitlines()}
    assert kept == {j.key for j in jobs} - expected_bad


def test_jobs_record_their_source_record():
    records = [{"text": BODY, "code": "c"}, {"text": "corto", "code": "c"},
               {"text": BODY, "code": "c"}]
    jobs = make_jobs(records, 10, seed=1, cfg=GenConfig())
    assert sorted(j.record for j in jobs) == [0, 2]
