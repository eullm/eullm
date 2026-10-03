"""What a decision model's answers on labelled questions say about it.

The same numbers `bench/reflexbench/qualify.py` gates on, computed here on
the dev split before and after training, from the same readout the engine
makes: each class's full-vocabulary log-probability (the log of the summed
probabilities of its code's spellings), the probabilities renormalized over
the classes, the most probable class as the answer.

* accuracy — the share of answers that are right;
* ECE — expected calibration error of the top answer, 15 equal-width bins,
  as `bench/decision_calibration.py` computes it: how far "90% sure" is
  from being right nine times in ten;
* NLL — mean −log p(right answer), which punishes confident mistakes;
* coverage — the share of the model's probability on a valid code at all.

The temperature fitted on the dev split travels in the exported GGUF, under
`TEMPERATURE_KEY`, for the engine to apply by default; `check_temperature`
holds it to what the engine accepts.
"""

from __future__ import annotations

import math
import struct

ECE_BINS = 15
#: The GGUF key a decision model carries its temperature in, a float32: the
#: one the engine applies to a codes-readout model unless a request gives
#: its own (`eullm-forge decisions export` writes it).
TEMPERATURE_KEY = "eullm.decision.temperature"
#: systemone.rs `MAX_TEMPERATURE`: the highest temperature the engine takes.
MAX_TEMPERATURE = 100.0


def class_result(logprobs: list[float], label: int, kind: str, temperature: float = 1.0) -> dict:
    """One answer from its classes' log-probabilities: decision.rs
    `calibrated_probabilities` with no prior — renormalized over the
    classes after dividing by `temperature`."""
    scaled = [lp / temperature for lp in logprobs]
    top_lp = max(scaled)
    exps = [math.exp(lp - top_lp) for lp in scaled]
    total = sum(exps)
    probabilities = [e / total for e in exps]
    top = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "kind": kind,
        "label": label,
        "logprobs": list(logprobs),
        "answer": top,
        "correct": top == label,
        "confidence": probabilities[top],
        "p_label": probabilities[label],
        "probabilities": probabilities,
        "coverage": min(1.0, sum(math.exp(lp) for lp in logprobs)),
    }


def at_temperature(results: list[dict], temperature: float) -> list[dict]:
    """The same answers read at another temperature."""
    return [class_result(r["logprobs"], r["label"], r["kind"], temperature) for r in results]


MIN_TEMPERATURE = 0.05
MAX_FIT_TEMPERATURE = 20.0


def fit_temperature(results: list[dict]) -> float:
    """The temperature with the lowest NLL on `results`, by golden-section
    search on log T in [0.05, 20] — bench/decision_calibration.py's fit. A
    fine-tuned model is usually too sure of itself (T > 1); the exported
    GGUF carries it (`TEMPERATURE_KEY`), and a request may give another
    (`eullm.temperature`).

    1.0 when the best NLL in that range is at one of its ends, because then
    the search did not land inside the range and its answer is the range, not
    a fit. See the note on the boundary below.
    """

    def nll(log_t: float) -> float:
        t = math.exp(log_t)
        return sum(-math.log(max(r["p_label"], 1e-12)) for r in at_temperature(results, t))

    lo, hi = math.log(MIN_TEMPERATURE), math.log(MAX_FIT_TEMPERATURE)
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(b)
    t = math.exp((lo + hi) / 2)
    # The search shrinks [lo, hi] until it is a rounding error wide, so the
    # midpoint is the edge itself when the minimum was on one: the lowest
    # NLL available was the end of the range, and everything past it is
    # untried. Golden-section search cannot see past hi, so it reports hi the
    # way a convergent search reports its answer.
    #
    # What that answer is not is a fitted temperature, and it used to be
    # written into the GGUF as one. A dev split the model gets wrong with
    # confidence is the case that lands here: the NLL falls the whole way
    # (27.6 at T=0.05 down to 0.83 at T=20 for one sample), so the minimum
    # is the top of the range, and the model shipped carrying T=20 -- as
    # unsure as this search can express -- as the default for every decision
    # it is ever asked. check_temperature() accepts it, because 20 is a legal
    # temperature; nothing downstream could tell it from a fit.
    #
    # 1.0 is what a model with no calibrated temperature gets: the engine's
    # own default, and what export_temperature already returns for a run that
    # fitted none. That is a decision about the search range as much as about
    # this function -- MAX_TEMPERATURE is the engine's limit and the range
    # here is the search's -- so it is written down rather than guessed at.
    edge = 1e-6
    if t <= MIN_TEMPERATURE * (1 + edge) or t >= MAX_FIT_TEMPERATURE / (1 + edge):
        # Said out loud: the report shows 1.0 either way, and only this line
        # tells a fit that landed on 1.0 from one that gave up at the edge.
        import logging

        logging.getLogger(__name__).warning(
            "temperature fit stopped at the edge of [%g, %g] (%.4g): keeping 1.0, "
            "the model is not calibrated", MIN_TEMPERATURE, MAX_FIT_TEMPERATURE, t)
        return 1.0
    return t


def check_temperature(value) -> float:
    """`value` as a GGUF stores it, a float32, when the engine will take it
    as a temperature: finite, above 0 and at most MAX_TEMPERATURE, the test
    systemone.rs puts a request's temperature to — passed before the
    rounding and after it, since 1e-50 is above 0 and its float32 is not.
    Raises ValueError otherwise."""
    try:
        given = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"a temperature is a number, not {value!r}") from None
    try:
        stored = struct.unpack("<f", struct.pack("<f", given))[0]
    except OverflowError:
        stored = math.inf
    for t in (given, stored):
        if not (math.isfinite(t) and 0 < t <= MAX_TEMPERATURE):
            raise ValueError(f"the engine takes a temperature above 0 and at most "
                             f"{MAX_TEMPERATURE:g}, not {value!r}")
    return stored


def ece(results: list[dict], bins: int = ECE_BINS) -> float | None:
    if not results:
        return None
    counts = [[0, 0.0, 0.0] for _ in range(bins)]
    for r in results:
        b = min(int(r["confidence"] * bins), bins - 1)
        counts[b][0] += 1
        counts[b][1] += r["confidence"]
        counts[b][2] += 1.0 if r["correct"] else 0.0
    return sum(abs(hits - conf) for _, conf, hits in counts) / len(results)


def summarize(results: list[dict]) -> dict:
    """Per question type and over all of them."""
    out = {}
    groups = {"all": results}
    for r in results:
        groups.setdefault(r["kind"], []).append(r)
    for name, rows in groups.items():
        if not rows:
            continue
        out[name] = {
            "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "ece": ece(rows),
            "nll": sum(-math.log(max(r["p_label"], 1e-12)) for r in rows) / len(rows),
            "coverage": sum(r["coverage"] for r in rows) / len(rows),
        }
    return out
