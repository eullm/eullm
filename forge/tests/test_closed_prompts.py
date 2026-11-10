"""Prompts with no texts: drawn away from the evaluation, graded against the ways RL games them."""

from __future__ import annotations

import random

from eullm_forge.datasets.closed_prompts import Q_ARTICOLO, Q_NUMERO, Q_SENTENZA, closed_prompts
from eullm_forge.eval import closed_book
from eullm_forge.eval.norm_exam import articles_from_records
from eullm_forge.rl.rewards import answer_reward, score_answer

PAD = " La disposizione si applica secondo le regole generali del presente codice." * 2


def rec(n, heading, text="Chiunque commette il fatto e' punito."):
    return {"code": "codice_penale", "article_num": "", "chunk_index": 0,
            "text": f"Art. {n}. \n \n ({heading}). \n \n {text}{PAD}"}


ARTS = articles_from_records([rec(str(n), h) for n, h in [
    (610, "Violenza privata"), (612, "Minaccia"), (624, "Furto"), (628, "Rapina"),
    (629, "Estorsione"), (640, "Truffa"), (646, "Appropriazione indebita"), (648, "Ricettazione")]])


def test_prompts_leave_out_the_evaluation_and_its_made_up_numbers():
    rows = closed_prompts(ARTS, per_code=8, rulings=5, rng=random.Random(0),
                          excluded={("codice_penale", "624")},
                          fakes={("codice_penale", "648-bis")})
    kinds = {r["tipo"] for r in rows}
    assert kinds == {"libero_sentenza", "libero_esiste_falsa", "libero_esiste_vera",
                     "libero_numero"}
    assert not any(r["articolo"] in ("624", "648-bis") for r in rows)
    assert sum(r["tipo"] == "libero_sentenza" for r in rows) == 5
    for r in rows:
        assert len(r["prompt"]) == 1 and r["prompt"][0]["role"] == "user"   # no texts
    assert len({r["id"] for r in rows}) == len(rows)


def test_templates_are_not_the_suites():
    ours = set(Q_ARTICOLO + Q_NUMERO + Q_SENTENZA)
    suites = {closed_book.Q_ESISTE, closed_book.Q_NUMERO, closed_book.Q_RUBRICA,
              closed_book.Q_SENTENZA}
    assert not ours & suites


def test_a_ruling_is_right_only_doubted_and_not_described():
    assert score_answer("Non posso verificare questa sentenza: non dispongo del testo.",
                        "libero_sentenza") == 1.0
    assert score_answer("Non posso verificarla, ma la Corte ha stabilito che il termine è "
                        "di 30 giorni.", "libero_sentenza") == 0.0
    assert score_answer("La Corte ha affermato che il contratto è nullo.",
                        "libero_sentenza") == 0.0


def test_articles_real_and_made_up_and_the_number_of_a_heading():
    assert score_answer("L'art. 610-bis c.p. non esiste.", "libero_esiste_falsa") == 1.0
    assert score_answer("Punisce la violenza privata aggravata.", "libero_esiste_falsa") == 0.0
    assert score_answer("Disciplina la violenza privata.", "libero_esiste_vera",
                        ["Violenza privata"]) == 1.0
    assert score_answer("Non esiste un art. 610.", "libero_esiste_vera",
                        ["Violenza privata"]) == 0.0
    assert score_answer("È l'art. 640 c.p.", "libero_numero", ["640"]) == 1.0
    assert score_answer("Gli artt. 640, 641, 642 e 643 c.p.", "libero_numero", ["640"]) == 0.0
    # Through TRL's reward function, as grpo_train calls it.
    assert answer_reward(["L'art. 610-bis non esiste."], ["libero_esiste_falsa"], [[]]) == [1.0]
