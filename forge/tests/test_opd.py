"""Privileged-context distillation: the loss, the prompts, and a run on a tiny model.

The run uses a randomly initialised two-layer Qwen2 with a word-level
tokenizer saved to disk, student and teacher alike: what is pinned is the
loop around the models -- prompts read, answers sampled, the teacher behind
its own prompt, the adapter saved, a stopped run carried on, a teacher with
another vocabulary refused.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reverse_kl_is_zero_for_the_same_distribution_and_masks_positions():
    torch = pytest.importorskip("torch")
    from eullm_forge.opd import reverse_kl

    a = torch.randn(5, 11)
    assert reverse_kl(a, a).item() == pytest.approx(0.0, abs=1e-6)
    b = a.clone()
    b[2] += torch.randn(11) * 3
    assert reverse_kl(a, b).item() > 0
    mask = torch.tensor([1, 1, 0, 1, 1])
    assert reverse_kl(a, b, mask).item() == pytest.approx(0.0, abs=1e-6)
    wide = torch.cat([a, torch.full((5, 4), -1e4)], dim=1)     # padded vocabulary
    assert reverse_kl(a, wide).item() == pytest.approx(0.0, abs=1e-5)


def test_the_teacher_sees_the_source_ruling_and_the_student_does_not():
    from eullm_forge.caselaw.prompts import caselaw_prompt, privileged_prompt, ruling_label

    lab = ruling_label({"sezione": "Sezione Quinta", "numero": "202301234"})
    assert lab == "Cons. Stato, Sezione Quinta, n. 202301234"
    passages = [(lab, "Il termine decorre dalla pubblicazione." + " x" * 2000)]
    s = caselaw_prompt("Da quando decorre il termine?", passages)
    t = privileged_prompt("Da quando decorre il termine?", passages, (lab, "TESTO DELLA SENTENZA"))
    assert s.startswith("Testi di riferimento")
    assert s.endswith("Domanda: Da quando decorre il termine?")
    assert " […]" in s and "TESTO DELLA SENTENZA" not in s
    assert t.startswith("Sentenza da cui proviene la domanda") and t.endswith(s)
    hidden = privileged_prompt("Da quando decorre il termine?", passages,
                               (lab, "TESTO DELLA SENTENZA"), citable=False)
    head = hidden.split("\n", 1)[0]
    assert "202301234" not in head and "non citarla" in head
    assert "TESTO DELLA SENTENZA" in hidden and hidden.endswith(s)


def _corpus(tmp_path: Path):
    rows, cards, index = [], [], []
    for n in range(6):
        num = f"20200{n:04d}"
        text = (f"Sentenza {num}. Il ricorso riguarda materia{n}. " * 30 + "\nDIRITTO\n"
                + f"Il principio{n} si applica. " * 30)
        rows.append({"text": text, "sentence_id": f"cds/{num}", "source_id": f"cds/{num}",
                     "year": 2020, "kind": "cds", "chunk_index": 0})
        cards.append({"id": f"cds/{num}",
                      "principi": [f"Il principio{n} vale sempre in materia{n}."],
                      "norme": ["art. 120 c.p.a."], "esito": "rigetto", "materia": f"materia{n}",
                      "oggetto": "PERMESSO DI SOGGIORNO" if n == 5 else f"APPALTO {n}",
                      "domande_ricerca": [f"Quando si applica il principio{n}?",
                                          f"Cosa prevede la materia{n}?"],
                      "domande_esame": []})
        index.append(f"cds/{num}")
    chunks = tmp_path / "train.jsonl"
    chunks.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    cards_f = tmp_path / "schede.jsonl"
    cards_f.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cards), encoding="utf-8")
    dev = tmp_path / "dev.txt"
    dev.write_text("cds/202000000\n", encoding="utf-8")
    statutes = tmp_path / "prompts.jsonl"
    statutes.write_text("".join(json.dumps({"id": f"s{i}", "prompt": [
        {"role": "user", "content": f"Domanda sulle norme {i}"}]}) + "\n" for i in range(10)), encoding="utf-8")
    return chunks, cards_f, dev, statutes


def test_prompts_leave_out_dev_and_sensitive_rulings_and_mix_statutes(tmp_path, capsys):
    chunks, cards, dev, statutes = _corpus(tmp_path)
    out = tmp_path / "opd" / "prompts.jsonl"
    mod = _load("make_opd_prompts")
    assert mod.main(["--chunks", str(chunks), "--cards", str(cards), "--dev-ids", str(dev),
                     "--statutes", str(statutes), "--mix", "0.25", "--per-ruling", "2",
                     "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    case = [r for r in rows if r["kind"] == "caselaw"]
    assert len(case) == 8                                  # 4 rulings x 2 questions
    assert {r["ruling"] for r in case} == {"cds/202000001", "cds/202000002", "cds/202000003",
                                           "cds/202000004"}
    assert sum(r["kind"] == "statute" for r in rows) == 3   # 25% of the rows
    r = case[0]
    source = r["ruling"].split("/")[1]
    assert f"n. {source}" in r["teacher"][0]["content"].split("\n", 1)[0]
    assert r["teacher"][0]["content"].endswith(r["student"][0]["content"])
    printed = capsys.readouterr().out
    assert "1 development and 1 sensitive" in printed and "principio" not in printed
    assert "source among the passages in 8)" in printed


def test_no_prompt_contains_a_development_ruling_text_anywhere(tmp_path, capsys):
    """Dev rulings are kept out of the sources but used to stay in the index.

    The chosen-source loop skips them, and the assert at the end of main only
    looks at each row's source ruling, so a dev ruling's verbatim text was
    retrieved into a train row's student prompt while the log said "left out
    1 development". The docstring promises no prompt is written from one.
    """
    mark = "DEV-MARKER-ZZZ"
    rows = [{"text": f"Sentenza {num}. Il ricorso riguarda appalto {num} {extra}",
             "sentence_id": f"cds/{num}", "source_id": f"cds/{num}",
             "year": 2020, "kind": "cds", "chunk_index": 0}
            for num, extra in (("202000001", ""), ("202000002", mark))]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "schede.jsonl").write_text("".join(json.dumps(
        {"id": f"cds/{num}", "principi": [f"Il principio vale in appalto {num}."],
         "norme": ["art. 120 c.p.a."], "esito": "rigetto", "materia": "appalto",
         "oggetto": f"APPALTO {num}",
         "domande_ricerca": [f"Quando si applica appalto {num}?"]},
        ensure_ascii=False) + "\n" for num in ("202000001", "202000002")), encoding="utf-8")
    (tmp_path / "dev.txt").write_text("cds/202000002\n", encoding="utf-8")
    out = tmp_path / "prompts.jsonl"
    mod = _load("make_opd_prompts")
    # -k 10 so every retrieved unit lands in the prompt: with the default 3
    # the dev chunk might simply not rank, which would prove nothing.
    assert mod.main(["--chunks", str(tmp_path / "train.jsonl"),
                     "--cards", str(tmp_path / "schede.jsonl"),
                     "--dev-ids", str(tmp_path / "dev.txt"), "-k", "10",
                     "--out", str(out)]) == 0
    prompts = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(prompts) == 1 and prompts[0]["ruling"] == "cds/202000001"
    assert mark not in json.dumps(prompts[0]["student"], ensure_ascii=False)
    assert mark not in json.dumps(prompts[0]["teacher"], ensure_ascii=False)
    assert "left out 1 development" in capsys.readouterr().out


def test_cited_numbers_reads_both_forms():
    mod = _load("cds_answer")
    text = ("Come chiarito da Cons. Stato, Sez. V, n. 202301234 e dalla sentenza n. 45/2021, "
            "nonché dall'art. 120, n. 3 c.p.a.")
    assert mod.cited_numbers(text) == {"202301234", "202100045"}
    assert mod.cited_numbers("") == set()
    # Italian capitalizes "N." at sentence start: uppercase cited nothing,
    # so a hallucinated "N. 999/2019" read cited_ok=True (vacuous empty set)
    # and a correct "N. 45/2021" read source_cited=False.
    assert mod.cited_numbers("Come chiarito da Cons. Stato, N. 999/2019 si applica.") == {
        "201900999"}
    assert mod.cited_numbers("Come chiarito nella sentenza N. 45/2021 del Consiglio.") == {
        "202100045"}


def test_a_law_numbered_like_a_ruling_is_not_a_ruling_cited():
    """"l. n. 241/1990" is a statute: read as a ruling, every answer citing a
    law correctly counted as citing a ruling it was never given."""
    mod = _load("cds_answer")
    for law in ("ai sensi dell'art. 21-octies della l. n. 241/1990",
                "secondo il d.lgs. n. 50/2016", "il D.Lgs. n. 36/2023",
                "il d.P.R. n. 1199/1971", "la legge n. 241/1990",
                "la legge 7 agosto 1990, n. 241/1990", "la l.r. n. 12/2005",
                "il decreto legislativo n. 104/2010", "il regolamento (UE) n. 679/2016",
                "il d.l. n. 76/2020", "il D.P.C.M. n. 5/2020"):
        assert mod.cited_numbers(law) == set(), law
    both = ("La l. n. 241/1990, come letta da Cons. Stato, sez. VI, n. 1234/2023 "
            "e dalla sentenza n. 202301235.")
    assert mod.cited_numbers(both) == {"202301234", "202301235"}


def test_the_exam_asks_only_development_rulings_and_checks_citations(tmp_path, monkeypatch,
                                                                     capsys):
    chunks, cards, dev, _ = _corpus(tmp_path)
    questions = tmp_path / "dev-cards.jsonl"
    questions.write_text("".join(json.dumps({"id": rid, "domande_esame": [
        {"domanda": f"Quando si applica il principio{n}?", "risposta": f"Sempre ({n}).",
         "rubrica": "Dice sempre."}] * 2}) + "\n"
        for n, rid in ((0, "cds/202000000"), (1, "cds/202000001"))), encoding="utf-8")
    mod = _load("cds_answer")
    seen = []

    def fake(args, contents):
        seen.extend(contents)
        return [("Secondo n. 202000000 il principio si applica.", True),
                ("Lo dice la sentenza n. 999/2019.", False)]

    monkeypatch.setattr(mod, "generate", fake)
    out = tmp_path / "answers" / "answers-x.jsonl"
    assert mod.main(["model", "--label", "x", "--questions", str(questions), "--dev-ids",
                     str(dev), "--chunks", str(chunks), "--cards", str(cards),
                     "--answers", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["cds-202000000-0", "cds-202000000-1"]
    assert {"question", "reference", "rubric", "answer"} <= set(rows[0])
    assert all(c.startswith("Testi di riferimento") for c in seen)
    first, second = rows
    assert first["source_retrieved"] and first["context"][0] == "cds/202000000"
    assert first["cited_ok"] and first["source_cited"]
    assert not second["cited_ok"] and not second["source_cited"]
    assert first["ended"] and not second["ended"]
    printed = capsys.readouterr().out
    assert "2 questions" in printed and "principio" not in printed
    assert "cut at --max-new-tokens 0.500" in printed


def test_the_exam_reads_chunks_that_carry_only_source_id(tmp_path, monkeypatch):
    """The corpus key may be sentence_id or source_id (rulings.load_rulings
    and index.build_units both accept either). cds_answer's own filter read
    only sentence_id, so a source_id-only chunk became kind "None" and the
    whole index was empty: the model was asked closed-book and the answer
    file claimed the source ruling was not retrieved."""
    chunks, _cards, dev, _ = _corpus(tmp_path)
    rows = [json.loads(line) for line in chunks.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("sentence_id", None)
        r.pop("kind", None)
        r["source_id"] = f"cds/{r['source_id'].split('/')[1]}"
    chunks.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    questions = tmp_path / "dev-cards.jsonl"
    questions.write_text(json.dumps({"id": "cds/202000000", "domande_esame": [
        {"domanda": "Quando si applica il principio0?", "risposta": "Sempre.",
         "rubrica": "Dice sempre."}]}) + "\n", encoding="utf-8")
    mod = _load("cds_answer")
    seen = []

    def fake(args, contents):
        seen.extend(contents)
        return [("Secondo n. 202000000 il principio si applica.", True)]

    monkeypatch.setattr(mod, "generate", fake)
    out = tmp_path / "answers" / "answers-source-id.jsonl"
    assert mod.main(["model", "--label", "x", "--questions", str(questions),
                     "--dev-ids", str(dev), "--chunks", str(chunks),
                     "--answers", str(out)]) == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "cds/202000000" in row["context"]
    assert row["source_retrieved"]
    assert seen[0].startswith("Testi di riferimento")


def test_the_note_reaches_the_teacher_only_and_leaves_the_rows_alone():
    mod = _load("opd_train")
    teacher = [{"role": "system", "content": "s"}, {"role": "user", "content": "Domanda"},
               {"role": "assistant", "content": "a"}, {"role": "user", "content": "Ancora"}]
    noted = mod.with_note(teacher, "Rispondi in breve.")
    assert noted[3]["content"] == "Ancora\n\nRispondi in breve."
    assert noted[1]["content"] == "Domanda" and teacher[3]["content"] == "Ancora"
    assert mod.with_note(teacher, "") is teacher


@pytest.fixture
def tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")

    words = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "user", "assistant", "Domanda:",
             "Testi", "il", "termine", "ricorso", "sentenza", "principio", "si", "applica"]
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)},
                                                           unk_token="[UNK]"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>",
                                               pad_token="[PAD]")
    tok.chat_template = ("{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} "
                         "<|im_end|> {% endfor %}<|im_start|> assistant")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=256,
                                   eos_token_id=3, pad_token_id=1)
    model = transformers.Qwen2ForCausalLM(cfg)
    path = tmp_path / "tiny"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({
        "student": [{"role": "user", "content": "Domanda: il termine si applica"}],
        "teacher": [{"role": "user", "content": "sentenza principio Domanda: il termine"}]}) + "\n"
        for _ in range(4)), encoding="utf-8")
    return path, prompts


def test_a_tiny_run_saves_the_adapter_and_a_stopped_run_carries_on(tiny_model, tmp_path, capsys):
    path, prompts = tiny_model
    mod = _load("opd_train")
    out = tmp_path / "run"
    base = ["--student", str(path), "--teacher", str(path), "--prompts", str(prompts),
            "--out", str(out), "--student-device", "cpu", "--teacher-devices", "cpu",
            "--batch", "2", "--max-new-tokens", "4", "--rank", "4"]
    assert mod.main(base + ["--steps", "2", "--stop-after", "1e-9"]) == 0
    assert (out / "ckpt" / "state.pt").is_file() and not (out / "adapter").exists()
    assert mod.main(base + ["--steps", "2", "--save-every", "1",
                            "--teacher-note", "Rispondi in breve.", "--anchor-statutes"]) == 0
    assert (out / "adapter" / "adapter_config.json").is_file()
    printed = capsys.readouterr().out
    assert "resuming at step 0" in printed and "step 2/2 kl" in printed
    assert "Domanda" not in printed
    assert mod.main(base + ["--steps", "2"]) == 0          # done: exits at once
    assert "nothing left to do" in capsys.readouterr().out


def test_an_empty_adapter_config_is_not_a_finished_adapter(tmp_path, capsys):
    """A 0-byte adapter_config.json is an interrupted save, not a done run.

    Existence alone used to count as done and exit 0 without even reading the
    prompts, so a killed link looked exactly like a completed one. grpo_train.py
    and stage3_sft.py test size for this reason; opd_train.py was the third
    writer of the same shape and the only one left on existence.
    """
    mod = _load("opd_train")
    out = tmp_path / "run"
    (out / "adapter").mkdir(parents=True)
    (out / "adapter" / "adapter_config.json").touch()  # the killed link's trace
    with pytest.raises(FileNotFoundError):  # it reads the prompts instead
        mod.main(["--student", "s", "--teacher", "t",
                  "--prompts", str(tmp_path / "no-such-prompts.jsonl"),
                  "--out", str(out)])
    assert "nothing left to do" not in capsys.readouterr().out


def test_every_prompts_row_is_validated_before_the_models_load(tmp_path):
    """Only the first row was checked, so a file whose later row lacks
    'teacher' passed the gate and crashed with KeyError at batch time --
    after the student and the 61 GB teacher were loaded."""
    mod = _load("opd_train")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in [
        {"student": [{"role": "user", "content": "q"}],
         "teacher": [{"role": "user", "content": "t"}]},
        {"student": [{"role": "user", "content": "q"}]},
    ]), encoding="utf-8")
    with pytest.raises(SystemExit, match="row 1"):
        mod.load_rows(prompts)


def _wordlevel(vocab: dict[str, int]):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=raw)


def test_a_vocabulary_plus_a_few_tokens_of_its_own_is_the_same_vocabulary():
    """Qwen3.8 is Qwen3.5 plus seven audio tokens after all the others: the
    same ids for every token text can produce. A token inserted among the
    others moves ids and is a different vocabulary."""
    from eullm_forge.opd import MAX_EXTRA_TOKENS, same_vocabulary

    words = ["[UNK]", "il", "ricorso", "è", "respinto"]
    base = {w: i for i, w in enumerate(words)}
    probe = "il ricorso è respinto"
    student = _wordlevel(base)
    audio = _wordlevel({**base, **{f"<audio_{k}>": len(base) + k for k in range(7)}})
    assert same_vocabulary(student, audio, probe) and same_vocabulary(audio, student, probe)
    shifted = _wordlevel({"[UNK]": 0, "<audio>": 1, **{w: i + 1 for w, i in base.items()
                                                         if w != "[UNK]"}})
    assert not same_vocabulary(student, shifted, probe)
    many = _wordlevel({**base, **{f"<x{k}>": len(base) + k
                                  for k in range(MAX_EXTRA_TOKENS + 1)}})
    assert not same_vocabulary(student, many, probe)


def test_a_teacher_with_another_vocabulary_is_refused(tiny_model, tmp_path):
    path, prompts = tiny_model
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    other = tmp_path / "other"
    import shutil
    shutil.copytree(path, other)
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "ciao": 1},
                                                           unk_token="[UNK]"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=raw).save_pretrained(other)
    mod = _load("opd_train")
    with pytest.raises(SystemExit, match="tokenizers differ"):
        mod.main(["--student", str(path), "--teacher", str(other), "--prompts", str(prompts),
                  "--out", str(tmp_path / "r"), "--student-device", "cpu",
                  "--teacher-devices", "cpu"])


def test_statute_rows_are_anchored_to_the_student_before_the_run(tiny_model, tmp_path, capsys):
    """2026-10-07: the case-law teacher on statute rows cost the 4B 15:71 on statutes.

    Anchored, a statute row is taught by the student itself with its adapter
    off: on the first step the adapter is still zero, so the KL is exactly 0,
    whatever the teacher. Without the anchor the teacher -- here one whose
    output layer is scaled, so that it disagrees -- teaches it.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path, prompts = tiny_model
    other = tmp_path / "other"
    model = transformers.AutoModelForCausalLM.from_pretrained(path)
    with torch.no_grad():
        model.lm_head.weight.mul_(50)
    model.save_pretrained(other)
    transformers.AutoTokenizer.from_pretrained(path).save_pretrained(other)
    rows = [dict(json.loads(line), kind="statute") for line in prompts.read_text(encoding="utf-8").splitlines()]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    mod = _load("opd_train")
    base = ["--student", str(path), "--teacher", str(other), "--prompts", str(prompts),
            "--student-device", "cpu", "--teacher-devices", "cpu", "--batch", "2",
            "--max-new-tokens", "4", "--rank", "4", "--steps", "1"]

    def first_kl(out, *extra):
        assert mod.main(base + ["--out", str(tmp_path / out), *extra]) == 0
        line = next(ln for ln in capsys.readouterr().out.splitlines() if "step 1/1 kl" in ln)
        return float(line.split(" kl ")[1].split()[0])

    assert first_kl("anchored", "--anchor-statutes") == 0.0
    assert first_kl("taught") > 1.0


def test_rescoring_reads_the_answers_already_written(tmp_path, capsys):
    """A law cited correctly no longer counts as a ruling invented, without
    asking the questions again."""
    mod = _load("cds_rescore")
    p = tmp_path / "answers-x.jsonl"
    rows = [{"id": "q0", "ruling": "cds/202301234", "context": ["cds/202301234"],
             "answer": "Ai sensi della l. n. 241/1990, come in Cons. Stato n. 1234/2023.",
             "cited_ok": False, "source_cited": True},
            {"id": "q1", "ruling": "cds/202301234", "context": ["cds/202301234"],
             "answer": "Cons. Stato n. 999/2019 lo esclude.", "cited_ok": False,
             "source_cited": False}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert mod.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "citing only rulings given 0.000 -> 0.500" in out
    assert "citing the source 0.500 -> 0.500" in out
    assert "999" not in out and "Cons." not in out


def test_rescoring_a_cassazione_answer_uses_the_same_key_as_cds_answer(tmp_path, capsys):
    """A Cassazione id has three parts (snciv/2023/12345); the old
    split("/", 1)[-1] kept "2023/12345" while cited_numbers yields
    "202312345", so every correct Cassazione citation was rescored as
    invented and every source citation as missing."""
    mod = _load("cds_rescore")
    p = tmp_path / "answers-snciv.jsonl"
    rows = [{"id": "snciv-202312345-0", "ruling": "snciv/2023/12345",
             "context": ["snciv/2023/12345"],
             "answer": "Secondo Cass. civ., Sez. III, n. 12345/2023, il principio vale.",
             "cited_ok": True, "source_cited": True},
            {"id": "snciv-202312345-1", "ruling": "snciv/2023/12345",
             "context": ["snciv/2023/12345"],
             "answer": "Secondo Cass. civ., Sez. III, n. 999/2019, il principio vale.",
             "cited_ok": False, "source_cited": False}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert mod.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "citing only rulings given 0.500 -> 0.500" in out
    assert "citing the source 0.500 -> 0.500" in out


def test_the_cassazione_exam_labels_and_reads_citations_with_the_year(tmp_path, monkeypatch,
                                                                      capsys):
    chunks = tmp_path / "legal_it.jsonl"
    rows = []
    for n in range(4):
        rid = f"snciv/202{n % 2 + 1}/{100 + n}"
        rows.append({"text": f"Corte di cassazione {n}: principio{n} sulla responsabilità.",
                     "sentence_id": rid, "chunk_index": 0, "year": 2021 + n % 2,
                     "kind": "snciv"})
    rows.append({"text": "Consiglio di Stato: altra cosa.", "sentence_id": "cds/202000000",
                 "chunk_index": 0, "year": 2020, "kind": "cds"})
    chunks.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    dev = tmp_path / "snciv-dev-ids.txt"
    dev.write_text("snciv/2021/100\n", encoding="utf-8")
    questions = tmp_path / "snciv-dev-cards.jsonl"
    questions.write_text(json.dumps({"id": "snciv/2021/100", "domande_esame": [
        {"domanda": "Quando vale il principio0 sulla responsabilità?", "risposta": "Sempre.",
         "rubrica": "Dice sempre."}]}) + "\n", encoding="utf-8")
    mod = _load("cds_answer")
    seen = []

    def fake(args, contents):
        seen.extend(contents)
        return [("Secondo Cass. civ., n. 100/2021, il principio vale.", True)]

    monkeypatch.setattr(mod, "generate", fake)
    out = tmp_path / "answers-x.jsonl"
    assert mod.main(["model", "--label", "x", "--kind", "snciv", "--questions", str(questions),
                     "--dev-ids", str(dev), "--chunks", str(chunks), "--answers", str(out)]) == 0
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["id"] == "snciv-202100100-0"
    assert "(sentenze della Corte di cassazione)" in seen[0]
    assert "Cass. civ., n. 100/2021" in seen[0] and "Consiglio di Stato" not in seen[0]
    assert row["source_retrieved"] and row["source_cited"] and row["cited_ok"]
    assert all(r.startswith("snciv/") for r in row["context"])
