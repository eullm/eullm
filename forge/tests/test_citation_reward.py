"""The citation reward, checked both ways: it pays the right article, not a neighbour."""

from __future__ import annotations

from eullm_forge.rl.citation import citation_reward, cites


def test_the_questions_article_is_paid_and_a_neighbour_is_not():
    assert cites("Secondo l'art. 1120 del codice civile, i condomini possono ...",
                 "codice_civile", "1120")
    assert cites("Ai sensi degli artt. 1176 e 1375 c.c. il debitore ...", "codice_civile", "1375")
    assert cites("L'art. 32 c.p.a. consente la conversione ...",
                 "codice_processo_amministrativo", "32")
    # No code named: the question's own.
    assert cites("Secondo l'art. 2105, il prestatore di lavoro ...", "codice_civile", "2105")
    # The neighbouring number, the same number of another code: not paid.
    assert not cites("Secondo l'art. 197 del codice penale, gli enti ...", "codice_penale", "196")
    assert not cites("Secondo l'art. 107 del codice di procedura civile ...",
                     "codice_processo_amministrativo", "107")


def test_only_judged_rows_with_an_article_are_scored():
    out = citation_reward(
        ["Secondo l'art. 1120 del codice civile ...", "Secondo l'art. 1121 c.c. ...",
         "La domanda va proposta entro 60 giorni.", "Secondo l'art. 5 ..."],
        tipo=["pratica", "contenuto", "termine", "pratica"],
        code=["codice_civile", "codice_civile", "codice_civile", ""],
        articolo=["1120", "1120", "10", ""])
    assert out == [1.0, 0.0, None, None]
    # TRL's conversational completions.
    assert citation_reward([[{"role": "assistant", "content": "Secondo l'art. 1120 c.c."}]],
                           tipo=["pratica"], code=["codice_civile"], articolo=["1120"]) == [1.0]
