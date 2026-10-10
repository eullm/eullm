"""GRPO rewards: right answers score, and the ways to game them do not.

A policy under RL finds whatever the checker pays for. Each case below is a
shortcut it could take, written down so the checker cannot quietly start
paying for it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eullm_forge.rl import abstains, answer_reward, score_answer

SIXTY = ["60 giorni|sessanta giorni"]


@pytest.mark.parametrize("answer", [
    "L'art. 10 prevede che il ricorso sia proposto entro sessanta giorni dalla notifica.",
    "Il termine è di 60 giorni dalla notifica dell'atto.",
    "Non è prevista alcuna proroga: il ricorso va proposto entro 60 giorni.",
])
def test_a_right_deadline_scores(answer):
    assert score_answer(answer, "termine", SIXTY) == 1.0


@pytest.mark.parametrize("answer", [
    "Il termine è di 30 giorni.",                                   # wrong number
    "Il termine può essere di 10 giorni, 30 giorni, 60 giorni o 90 giorni.",  # listing
    "L'art. 10 non esiste. Comunque il termine è di 60 giorni.",    # refuse and answer
    "Tra i testi riportati non trovo la disposizione; forse 60 giorni.",
])
def test_the_shortcuts_to_a_deadline_do_not(answer):
    assert score_answer(answer, "termine_argomento", SIXTY) == 0.0


@pytest.mark.parametrize("answer", [
    "Non è prevista alcuna proroga: il ricorso va proposto entro 60 giorni.",
    "Non esiste alcuna proroga: il ricorso va proposto entro 60 giorni.",
    "Non esiste alcun termine perentorio: il ricorso va proposto entro 60 giorni.",
    "Non esiste alcuna eccezione: il termine è di 60 giorni.",
    "Non esiste alcuna decadenza; il ricorso va proposto entro 60 giorni.",
    "Secondo l'articolo 10 il termine è di 60 giorni e non esiste proroga.",
    "Ai sensi dell'art. 10 il ricorso si propone entro 60 giorni, e non esiste deroga.",
    "Secondo l'articolo 10 il termine è di 60 giorni. Non esiste alcuna proroga.",
])
def test_opening_with_a_negation_about_something_else_still_scores(answer):
    """"Non è prevista" was carved out of the refusal set because it opens right
    deadline answers. "Non esiste" is the more common way to write the same
    sentence, and was in the set, so the same answer was paid for with one
    synonym and not with the other."""
    assert score_answer(answer, "termine", SIXTY) == 1.0


@pytest.mark.parametrize("answer", [
    "L'articolo 10 non esiste, ma il ricorso si propone entro 60 giorni.",
    "Non esiste alcuna norma che disciplini il ricorso; il termine è di 60 giorni.",
    "Non esiste alcun articolo che tratti del ricorso. Il termine è di 60 giorni.",
    "L'articolo 10 è stato abrogato. Non esiste. Il termine è di 60 giorni.",
])
def test_denying_the_article_itself_is_still_a_refusal(answer):
    """What the set is for, in both orders and with the noun spelled out."""
    assert score_answer(answer, "termine", SIXTY) == 0.0


@pytest.mark.parametrize("answer", [
    "Non esiste alcuna proroga. Il termine è di 30 giorni.",
    "Non esiste alcuna proroga; i termini sono 10, 30, 60 o 90 giorni.",
])
def test_a_decorative_negation_does_not_buy_a_wrong_or_a_listed_deadline(answer):
    assert score_answer(answer, "termine", SIXTY) == 0.0


@pytest.mark.parametrize("answer", [
    "entro 120 giorni.",          # ends in the 20 of "20 giorni"
    "entro 1020 giorni.",
    "entro 160 giorni.",          # ends in the 60 of "60 giorni"
    "entro 360 giorni.",
    "entro centoventi giorni.",   # same, in words
    "entro centosessanta giorni.",
    "entro duecentoventi giorni.",
])
def test_a_wrong_deadline_containing_the_right_one_scores_nothing(answer):
    """The keyword is a substring test, so the right digits are enough.

    The value is parsed a few lines up to count the deadlines an answer
    names; it is also what says whether the one it named is the right one.
    """
    keyword = SIXTY if "60" in answer or "sessanta" in answer else ["20 giorni|venti giorni"]
    assert score_answer(answer, "termine", keyword) == 0.0


def test_the_deadline_the_article_also_mentions_is_allowed():
    """An article that sets a deadline and refers to another: naming both is
    a complete answer, not a list."""
    answer = "Il ricorso va proposto entro 60 giorni; l'opposizione entro 30 giorni."
    assert score_answer(answer, "termine", SIXTY) == 1.0


@pytest.mark.parametrize("answer,tipo", [
    ("L'art. 2999 del codice civile non esiste nella raccolta.", "inesistente"),
    ("Tra i testi normativi riportati non trovo la disposizione che risponde.", "assente"),
    ("I testi riportati non contengono la norma richiesta.", "assente"),
])
def test_saying_it_is_not_there_scores_where_it_is_true(answer, tipo):
    assert score_answer(answer, tipo) == 1.0


@pytest.mark.parametrize("answer,tipo", [
    ("Il testo fornito non contiene l'articolo richiesto.", "assente"),
    ("Il brano riportato non contiene la disposizione richiesta.", "assente"),
    ("La raccolta fornita non contiene l'art. 2999 del codice civile.", "inesistente"),
    ("Dagli atti riportati non risulta alcuna disposizione in merito.", "assente"),
    ("Non risulta nessuna norma che disciplini la materia.", "assente"),
    ("Nel brano riportato non compare la disposizione di cui si chiede.", "assente"),
    ("La raccolta non include l'articolo richiesto.", "assente"),
])
def test_the_abstention_is_paid_in_either_number(answer, tipo):
    """"Non contiene" is "non contengono" with one text instead of many.

    Only the plural was in the set, so the singular -- and "non risulta
    alcuna", which is how the answer is usually put -- were correct
    abstentions that scored nothing. Under GRPO an unpaid-but-correct phrasing
    loses, so the policy is trained off it.
    """
    assert score_answer(answer, tipo) == 1.0


@pytest.mark.parametrize("answer", [
    "Il testo contiene l'articolo richiesto.",
    "L'articolo 5 contiene la prescrizione di sessanta giorni.",
    # An inesistente answer that invents the article and says what it leaves
    # out is describing one, not saying it is absent.
    "L'art. 2999 disciplina il trasporto; la norma non include i contratti a termine.",
    "L'art. 3000 prevede che il venditore consegni la cosa; non comprende le spese di trasporto.",
    "L'art. 1500 riguarda la locazione e non risulta alcuna eccezione per gli immobili urbani.",
    "L'articolo stabilisce l'obbligo di custodia, che non riporta limiti di valore.",
])
def test_saying_it_is_there_is_not_an_abstention(answer):
    assert not abstains(answer)


@pytest.mark.parametrize("answer", [
    "L'art. 2999 disciplina la responsabilità del vettore.",           # invents it
    "Non trovo la norma, ma di solito il termine è di 60 giorni.",     # abstains, then answers
])
def test_inventing_or_hedging_on_a_missing_article_does_not(answer):
    assert score_answer(answer, "inesistente") == 0.0


def test_always_abstaining_is_not_a_policy_that_pays():
    """The same refusal on all four kinds: right on two, wrong on the two
    that make up most of the prompts."""
    refusal = "Tra i testi riportati non trovo la disposizione richiesta."
    got = [score_answer(refusal, t, SIXTY) for t in
           ("termine", "termine_argomento", "inesistente", "assente")]
    assert got == [0.0, 0.0, 1.0, 1.0]


def test_an_uncheckable_kind_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        score_answer("qualsiasi", "contenuto")
    with pytest.raises(ValueError):
        score_answer("60 giorni", "termine", [])


def test_the_trl_reward_reads_chat_completions_and_aligned_columns():
    completions = [[{"role": "assistant", "content": "Il termine è di 60 giorni."}],
                   [{"role": "assistant", "content": "Il termine è di 90 giorni."}],
                   "L'articolo non esiste."]
    assert answer_reward(completions, tipo=["termine", "termine", "inesistente"],
                         keywords=[SIXTY, SIXTY, []]) == [1.0, 0.0, 1.0]


def test_the_prompts_file_is_built_from_the_exam_code_and_leaves_exam_articles_out(tmp_path):
    """Prompts come from build_exam and the exam's own retrieval prompt; an
    article of an excluded exam is never drawn."""
    import importlib.util

    from eullm_forge.eval import EvalItem, save_eval_set

    filler = " Il presente articolo contiene disposizioni di dettaglio sufficienti."
    records = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
                "text": f"Art. {n}. \n \n (Materia {n}). \n \n Il ricorso è proposto "
                        f"entro sessanta giorni dalla notifica.{filler * 2}"}
               for n in range(1, 31)]
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                     encoding="utf-8")
    exam = tmp_path / "norm-exam-v9.jsonl"
    save_eval_set([EvalItem(id=f"x{n}", domain="legal", lang="it", question="?",
                            metadata={"code": "codice_civile", "articolo": str(n)})
                   for n in range(1, 11)], exam)

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_grpo_prompts.py"
    spec = importlib.util.spec_from_file_location("make_grpo_prompts", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "prompts.jsonl"
    assert mod.main(["--norms", str(norms), "--exclude-exam", str(exam),
                     "--per-code", "50", "--out", str(out)]) == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    kinds = {r["tipo"] for r in rows}
    assert kinds == {"termine", "termine_argomento", "inesistente", "assente"}
    assert not {r["articolo"] for r in rows if r["tipo"] != "inesistente"} & \
        {str(n) for n in range(1, 11)}
    termine = next(r for r in rows if r["tipo"] == "termine")
    content = termine["prompt"][0]["content"]
    assert content.startswith("Testi normativi di riferimento:") and "Domanda: " in content
    assert termine["keywords"] == SIXTY
    absent = next(r for r in rows if r["tipo"] == "assente")
    assert f"art. {absent['articolo']}\n" not in absent["prompt"][0]["content"]
    for r in rows:   # every row is gradable
        score_answer("x", r["tipo"], r["keywords"])
    assert sum(r["tipo"] == "inesistente" for r in rows) <= 0.2 * len(rows) + 1


def test_nonexistent_articles_are_capped_and_nothing_else_is_dropped():
    import importlib.util
    import random

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_grpo_prompts.py"
    spec = importlib.util.spec_from_file_location("make_grpo_prompts", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [{"tipo": "inesistente"}] * 679 + [{"tipo": "termine"}] * 433
    kept = mod.cap_share(rows, "inesistente", 0.2, random.Random(0))
    assert sum(r["tipo"] == "termine" for r in kept) == 433
    assert sum(r["tipo"] == "inesistente" for r in kept) == 108   # 20% of 541


def test_an_absent_prompt_drops_the_whole_article_not_just_its_first_chunk(tmp_path):
    """An assente row is paid for abstaining, so the article's own text must
    not be in the prompt at all.

    A long article is chunked, and only the first chunk carries a header, so
    the chunks that continue it report no article number -- the answer is
    still in them, and a question about that article ranks them first.
    """
    import importlib.util

    from eullm_forge.eval import EvalItem, NormIndex

    filler = " Il presente articolo contiene disposizioni di dettaglio sufficienti."
    records = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
                "text": f"Art. {n}. \n \n (Materia numero {n}). \n \n "
                        f"Il ricorso è proposto entro sessanta giorni dalla notifica."
                        f"{filler * 2}"} for n in range(1, 21)]
    # art. 15 is long: chunk 0 holds the header, 1 and 2 continue it, and the
    # text the question is about is in the continuation.
    records += [
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": "Art. 15. \n \n (Azione di accertamento). \n \n "
                 f"Disposizione generale in materia di accertamento.{filler * 2}"},
        {"code": "codice_civile", "article_num": "", "chunk_index": 1,
         "text": "Il ricorso è proposto entro venti giorni dalla notifica all'atto "
                 f"emanato, secondo le modalità previste.{filler * 2}"},
        {"code": "codice_civile", "article_num": "", "chunk_index": 2,
         "text": "L'azione di accertamento decade dopo un anno dalla notifica, salvo "
                 f"i casi tassativamente previsti.{filler * 2}"},
    ]
    chunks = tmp_path / "legislazione_y.chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                      encoding="utf-8")
    index = NormIndex.from_files([chunks])

    question = ("Entro quanti giorni dalla notifica si propone il ricorso di cui "
                "all'articolo 15 del codice civile?")
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_grpo_prompts.py"
    spec = importlib.util.spec_from_file_location("make_grpo_prompts2", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    item = EvalItem(id="a15", domain="legal", lang="it", question=question,
                    reference="venti giorni", keywords=["20 giorni|venti giorni"],
                    metadata={"code": "codice_civile", "articolo": "15",
                              "tipo": "termine_argomento"})

    row = mod.prompt_row(item, index, k=3, absent=True)
    assert row is not None and row["tipo"] == "assente"
    content = row["prompt"][0]["content"]
    assert "Art. 15." not in content
    # The continuation chunks carry neither a header nor an article_num, so
    # these are the lines the article's own text would put in the prompt.
    assert "entro venti giorni dalla notifica all'atto" not in content
    assert "L'azione di accertamento decade dopo un anno dalla notifica" not in content
    # ...and other articles are still there, or there is no prompt at all. Not
    # a named one: which of the twenty comes first is the ranking's business,
    # and this test is about article 15 being gone.
    assert len(re.findall(r"^\[\d+\]", content, re.M)) == 3


def test_hybrid_prompts_use_the_hybrid_index_and_still_drop_an_absent_article(
        tmp_path, monkeypatch):
    """--embedder builds the prompts on the index the released models use.

    The embedding model is a stand-in that ranks the continuation chunk of
    art. 15 first, where the answer is: an assente row must still not show it.
    """
    import importlib.util

    np = pytest.importorskip("numpy")
    from eullm_forge.eval import EvalItem, NormIndex, dense

    filler = " Il presente articolo contiene disposizioni di dettaglio sufficienti."
    records = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
                "text": f"Art. {n}. \n \n (Materia numero {n}). \n \n "
                        f"Il ricorso è proposto entro sessanta giorni dalla notifica."
                        f"{filler * 2}"} for n in range(1, 21)]
    records += [
        {"code": "codice_civile", "article_num": "", "chunk_index": 0,
         "text": f"Art. 21. \n \n (Accertamento). \n \n Disposizione generale.{filler * 2}"},
        {"code": "codice_civile", "article_num": "", "chunk_index": 1,
         "text": f"Il ricorso è proposto entro venti giorni dall'atto emanato.{filler * 2}"},
    ]
    norms = tmp_path / "legislazione_z.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                     encoding="utf-8")
    built = {}

    def fake_build_hybrid(paths, embedder_id, *, reranker_id=None, cache_dir=None):
        base = NormIndex.from_files(paths)
        vecs = np.eye(len(base.records), dtype=np.float32)
        built.update(embedder=embedder_id, reranker=reranker_id, cache=cache_dir)
        return dense.HybridIndex(base, vecs, lambda q: vecs[len(base.records) - 1])

    monkeypatch.setattr(dense, "build_hybrid", fake_build_hybrid)
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_grpo_prompts.py"
    spec = importlib.util.spec_from_file_location("make_grpo_prompts3", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    index = dense.build_hybrid([norms], "emb")
    item = EvalItem(id="a21", domain="legal", lang="it",
                    question="Nel codice civile, entro quanti giorni si propone il ricorso?",
                    keywords=["20 giorni|venti giorni"],
                    metadata={"code": "codice_civile", "articolo": "21",
                              "tipo": "termine_argomento"})
    assert "entro venti giorni" in mod.prompt_row(item, index, k=3)["prompt"][0]["content"]
    absent = mod.prompt_row(item, index, k=3, absent=True)["prompt"][0]["content"]
    assert "entro venti giorni" not in absent and "Art. 21." not in absent

    out = tmp_path / "prompts.jsonl"
    assert mod.main(["--norms", str(norms), "--per-code", "20", "--out", str(out),
                     "--embedder", "Qwen/Qwen3-Embedding-0.6B",
                     "--reranker", "Qwen/Qwen3-Reranker-0.6B",
                     "--retrieval-cache", str(tmp_path / "cache")]) == 0
    assert built == {"embedder": "Qwen/Qwen3-Embedding-0.6B",
                     "reranker": "Qwen/Qwen3-Reranker-0.6B", "cache": tmp_path / "cache"}
    assert out.read_text(encoding="utf-8").strip()


def test_judged_content_prompts_carry_what_the_judge_reads(tmp_path):
    """--judged adds "contenuto" rows with the exam's question, article and v2 rubric."""
    import importlib.util

    from eullm_forge.eval.norm_exam import CONTENT_RUBRIC

    filler = " Il presente articolo contiene disposizioni di dettaglio sufficienti."
    records = [{"code": "codice_civile", "article_num": "", "chunk_index": 0,
                "text": f"Art. {n}. \n \n (Materia {n}). \n \n Il ricorso è proposto "
                        f"entro sessanta giorni dalla notifica.{filler * 2}"}
               for n in range(1, 31)]
    norms = tmp_path / "legislazione_x.chunks.jsonl"
    norms.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                     encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_grpo_prompts.py"
    spec = importlib.util.spec_from_file_location("make_grpo_prompts4", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "prompts.jsonl"
    assert mod.main(["--norms", str(norms), "--per-code", "50", "--judged", "4",
                     "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    judged = [r for r in rows if r["tipo"] == "contenuto"]
    assert len(judged) == 4
    for r in judged:
        assert r["question"].startswith("Che cosa prevede l'art. ")
        assert "Il ricorso è proposto" in r["reference"]
        assert r["rubric"] == CONTENT_RUBRIC
        assert r["question"] in r["prompt"][0]["content"]
    assert all("question" not in r for r in rows if r["tipo"] != "contenuto")
    # without --judged the file is what it was
    plain = tmp_path / "plain.jsonl"
    assert mod.main(["--norms", str(norms), "--per-code", "50", "--out", str(plain)]) == 0
    assert "contenuto" not in plain.read_text(encoding="utf-8")
