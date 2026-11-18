"""Learning from the model's own answers: which are wrong, why, and the right one.

`sample_answers.py` asks a model each verifiable question several times
(the GRPO prompts and the false premises), and `eullm_forge.rl.rewards`
grades every answer by program. From one question's answers come:

* a **preference pair** (``chosen``/``rejected``, for DPO): a right answer and
  a wrong one to the same prompt, both the model's own. Telling them apart
  is the skill of recognising the wrong answer, not only of producing the
  right one;
* a **correction exercise** (SFT): the question, the texts and a proposed
  answer; the target checks it against the texts, says what is wrong and
  why, and writes the right answer. Some proposed answers are right, and
  the target says so: an exercise that is always "it is wrong" teaches to
  find fault, not to check;
* a **pass rate**: the share of right answers. GRPO learns nothing from a
  question whose answers are all right or all wrong (every advantage is 0),
  so `informative` keeps the ones in between for the next round.

What is wrong is said from what the program checked -- the deadline named
against the one the text states, an article described that is not among the
texts -- never from a judge, whose verdicts agree with a person 68-85% of
the time. Only verifiable kinds are used; a right answer is right on the
point checked, which is what the exercise is about.
"""

from __future__ import annotations

import random
import re

from ..eval.metrics import ANY_DEADLINE, DEADLINE_UNITS, number_of
from ..rl.rewards import (
    ABSTAIN_TYPES,
    DEADLINE_TYPES,
    abstains,
    corrects,
    corrects_procedibility,
    mentioned_deadlines,
    refuses,
    score_answer,
    unverified_ruling,
)

VERIFIABLE = DEADLINE_TYPES | ABSTAIN_TYPES | {"premessa_termine", "premessa_abrogato",
                                              "premessa_pena", "premessa_procedibilita",
                                              "premessa_vera", "premessa_sentenza",
                                              "premessa_sentenza_vera"}

ASK = ("\n\nRisposta proposta:\n{answer}\n\nLa risposta proposta è corretta? Verificala sui "
       "testi riportati e, se è sbagliata, indica l'errore e scrivi la risposta corretta.")
_SINGULAR = {"giorni": "giorno", "mesi": "mese", "anni": "anno", "ore": "ora"}


def say(deadline: tuple[int, str]) -> str:
    n, unit = deadline
    return f"{n} {_SINGULAR.get(unit, unit) if n == 1 else unit}"


def right_deadline(keywords: list[str]) -> tuple[int, str] | None:
    """The deadline a deadline question's keyword stands for."""
    for kw in keywords or []:
        for num, unit in ANY_DEADLINE.findall(kw):
            n = number_of(num)
            if n:
                return n, DEADLINE_UNITS[unit.lower()]
    return None


def passage_of(prompt: str, code: str, article: str) -> int | None:
    """The number of the passage that is the article, as the prompt labels it."""
    label = code.replace("_", " ")
    # The suffix is part of the identity: "art. 2-bis" is not "art. 2", and a
    # by-topic question can retrieve both. A plain number also refuses the
    # suffixed labels, so a 2-bis passage is never quoted as 2's.
    base, _, suffix = article.partition("-")
    numbered = re.escape(base)
    if suffix:
        numbered += rf"\s*[-–]?\s*{re.escape(suffix)}\b"
    else:
        numbered += (r"\b(?!\s*[-–]?\s*(?:bis|ter|quater|quinquies|sexies|septies|"
                     r"octies|novies|decies)\b)")
    for m in re.finditer(r"^\[(\d+)\] ([^\n]*)$", prompt, re.MULTILINE):
        if m.group(2).startswith(label) and re.search(
                rf"\bartt?\. {numbered}", m.group(2)):
            return int(m.group(1))
    return None


def diagnosis(answer: str, tipo: str, keywords: list[str], where: str) -> str:
    """Why a wrong answer is wrong, from what the program checked."""
    right = right_deadline(keywords)
    named = mentioned_deadlines(answer)
    if tipo in ("premessa_sentenza", "premessa_sentenza_vera"):
        if not unverified_ruling(answer):
            return ("tratta come verificata una sentenza che non è tra i testi riportati, "
                    "invece di dire che non si può controllare")
        tipo = "premessa_termine" if tipo == "premessa_sentenza" else "premessa_vera"
    if tipo == "premessa_vera" and corrects(answer):
        what = f"il termine di {say(right)}" if right else "quanto"
        return (f"dice che la premessa è errata, ma {where} prevede proprio {what} indicato "
                "nella domanda")
    if tipo == "premessa_vera" and not right:
        return f"non conferma la procedibilità {keywords[0]} che {where} stabilisce"
    if tipo == "premessa_pena":
        if not corrects(answer):
            return "non dice che la premessa sulla pena è errata, e così la lascia in piedi"
        return f"non indica il massimo di pena che {where} prevede, {say(right)}"
    if tipo == "premessa_procedibilita":
        if not corrects_procedibility(answer, keywords[0]):
            return "non dice che la premessa sulla procedibilità è errata"
        return f"non dice che il reato è procedibile {keywords[0]}, come stabilisce {where}"
    if tipo in DEADLINE_TYPES or tipo in ("premessa_termine", "premessa_vera"):
        if tipo == "premessa_termine" and right in named and not corrects(answer):
            return (f"indica il termine giusto, {say(right)}, ma non dice che la premessa "
                    "della domanda è errata, e così la lascia in piedi")
        if refuses(answer) or (abstains(answer) and not named):
            return f"dice che i testi non rispondono, ma {where} prevede un termine di {say(right)}"
        wrong = sorted(d for d in named if d != right)
        if wrong:
            return (f"indica un termine di {say(wrong[0])}, ma {where} prevede un termine di "
                    f"{say(right)}")
        return f"non indica il termine, che {where} fissa in {say(right)}"
    if tipo == "premessa_abrogato":
        return f"tratta l'articolo come vigente, ma {where} riporta che è stato abrogato"
    if tipo == "assente":
        return ("risponde come se i testi contenessero la disposizione, ma nessuno dei testi "
                "riportati risponde alla domanda")
    return ("descrive il contenuto di un articolo che non è tra i testi riportati e non "
            "risulta esistere")


def confirmation(tipo: str, keywords: list[str], where: str) -> str:
    """What a right answer gets right, said the same way."""
    if tipo in DEADLINE_TYPES:
        return f"indica il termine di {say(right_deadline(keywords))} che {where} prevede"
    if tipo in ("premessa_sentenza", "premessa_sentenza_vera"):
        verdict = "corregge" if tipo == "premessa_sentenza" else "conferma"
        return (f"dice che la sentenza citata non è tra i testi e non si può verificare, e "
                f"{verdict} il termine sul testo di legge: {say(right_deadline(keywords))}, "
                f"come prevede {where}")
    if tipo == "premessa_vera":
        right = right_deadline(keywords)
        if not right:
            return f"conferma la procedibilità {keywords[0]} che {where} stabilisce"
        return f"conferma il termine di {say(right)} che {where} prevede"
    if tipo == "premessa_pena":
        return (f"corregge la premessa e indica il massimo di pena di "
                f"{say(right_deadline(keywords))} che {where} prevede")
    if tipo == "premessa_procedibilita":
        return (f"corregge la premessa: il reato è procedibile {keywords[0]}, come stabilisce "
                f"{where}")
    if tipo == "premessa_termine":
        return (f"corregge la premessa e indica il termine di "
                f"{say(right_deadline(keywords))} che {where} prevede")
    if tipo == "premessa_abrogato":
        return f"dice che l'articolo è stato abrogato, come riporta {where}"
    return "dice che i testi riportati non contengono la disposizione, invece di inventarla"


def _where(row: dict) -> str:
    pos = passage_of(row["prompt"][-1]["content"], row.get("code", ""),
                     str(row.get("articolo", "")))
    return f"il testo [{pos}]" if pos else "il testo riportato"


def correction_pair(row: dict, proposed: str, right: str, is_right: bool) -> dict:
    """An SFT exercise: check the proposed answer against the texts, then answer."""
    tipo, kw, where = row["tipo"], row.get("keywords", []), _where(row)
    verdict = (f"è corretta: {confirmation(tipo, kw, where)}" if is_right
               else f"è errata: {diagnosis(proposed, tipo, kw, where)}")
    question = row["prompt"][-1]["content"] + ASK.format(answer=proposed.strip())
    return {"instruction": question,
            "output": f"Verifica della risposta proposta: {verdict}.\nRisposta corretta: "
                      f"{right.strip()}",
            "task": "correzione_" + tipo, "source": row.get("code", ""),
            "key": f"{row['id']}-corr"}


def preference_pair(row: dict, chosen: str, rejected: str) -> dict:
    """A DPO row in TRL's conversational shape."""
    return {"prompt": row["prompt"], "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}], "tipo": row["tipo"],
            "id": row["id"]}


def graded(row: dict) -> tuple[list[str], list[str]]:
    """The row's sampled answers that are right and wrong, by program, each once."""
    right, wrong = [], []
    for a in dict.fromkeys(s["answer"] for s in row.get("samples", []) if s["answer"].strip()):
        (right if score_answer(a, row["tipo"], row.get("keywords")) == 1.0 else wrong).append(a)
    return right, wrong


def informative(row: dict) -> bool:
    """Whether GRPO can learn from the question: some answers right, some wrong.

    Graded again rather than read from the stored scores, so a reward fixed
    after the sampling applies to it without asking the model again.
    """
    scores = [score_answer(s["answer"], row["tipo"], row.get("keywords"))
              for s in row.get("samples", [])]
    return bool(scores) and 0 < sum(scores) < len(scores)


def build(rows: list[dict], rng: random.Random, right_share: float = 0.3
          ) -> tuple[list[dict], list[dict], dict]:
    """Correction exercises and preference pairs from sampled rows, and counts.

    A question gives one of each when the model answered it both right and
    wrong; a question it only got wrong gives nothing (there is no right
    answer of its own to show), one it only got right gives at most a
    "the proposed answer is right" exercise, ``right_share`` of the time.
    """
    corrections, prefs, stats = [], [], {}

    def count(k: str) -> None:
        stats[k] = stats.get(k, 0) + 1

    for row in rows:
        if row.get("tipo") not in VERIFIABLE:
            count("skipped:not_verifiable")
            continue
        right, wrong = graded(row)
        if not right:
            count("skipped:never_right")
            continue
        best = rng.choice(right)
        if wrong:
            prefs.append(preference_pair(row, best, rng.choice(wrong)))
            count("pairs")
        draw = rng.random()
        if wrong and draw >= right_share:
            corrections.append(correction_pair(row, rng.choice(wrong), best, False))
            count("corrections:wrong")
        elif draw < right_share:
            proposed = rng.choice([a for a in right if a != best] or right)
            corrections.append(correction_pair(row, proposed, best, True))
            count("corrections:right")
    return corrections, prefs, stats
