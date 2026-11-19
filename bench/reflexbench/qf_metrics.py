"""What the qualification test measures, and what it takes to pass.

Per question type — noul, choice, score — and over all of them:

  * accuracy: the share of right answers, in the mode the model will serve
    in, an answer the server refused to give counted as wrong; with its 95%
    Wilson interval and the accuracy of always giving each question's
    commonest right answer, which a model has to beat;
  * ECE: expected calibration error of the top answer, 15 equal-width bins
    (bench/decision_calibration.py's definition), and NLL;
  * coverage: the share of the model's probability on a valid answer code
    (code readout only);
  * noise between evaluation modes: the same request in `separate`,
    `shared_prefix` and `batched`; the largest difference in any answer's
    probability from `separate`, and the share of answers that change;
  * latency: p50 and p95 of a request as the client sees it, per mode;
  * the temperature: when the candidate's GGUF carries one, whether the
    server applied it to every answer.

Every threshold is configurable; the defaults are below, each next to its
reason, and a check that fails says which reason it broke.
"""

import math
import statistics

ECE_BINS = 15
KINDS = ("noul", "choice", "score")

#: name → (default, reason). None: not checked unless set.
THRESHOLDS = {
    "min_answers": (
        50,
        "below 50 answers of a type, its accuracy is known to no better than ±14 points "
        "(95% Wilson interval at 50%): too little to rest a verdict on",
    ),
    "max_ece": (
        0.10,
        "a decision's probability is what thresholds are set on ('above 0.9, automate'): "
        "on average it may sit at most 10 points from how often the model is right",
    ),
    "max_mode_delta": (
        0.05,
        "a decision must not depend on how it was batched: models trained for decisions "
        "stay under it (Jev-Style 0.8B Q4_K_M: 0.024 on a CPU, 0.039 on a GPU), "
        "Qwen3-0.6B Q4_K_M, which is not, moves by up to 0.53 (docs/engine.md)",
    ),
    "max_mode_flips": (
        0.01,
        "an answer that changes with the evaluation mode cannot be replayed: at most one "
        "answer in a hundred near a tie",
    ),
    "min_coverage": (
        0.90,
        "below it, much of the model's probability goes to something other than an answer "
        "code, and its probabilities describe a minority of what it would say",
    ),
    "max_accuracy_drop": (
        0.02,
        "a replacement may be faster or better calibrated, not less often right: two points "
        "is a small difference, and a measurable one on a few hundred answers",
    ),
    "max_latency_ratio": (
        1.5,
        "callers budget for the decision they have: half again as slow is another budget",
    ),
    "max_p95_ms": (
        None,
        "depends on the hardware and the caller: set it for the machine that will serve",
    ),
}


#: Checks with no threshold to set.
FIXED = {
    "beats_majority": "a model no more often right than one that gives every question its "
    "commonest answer has learnt nothing about the states",
    "gguf_temperature": "the temperature fitted on the model's dev split travels in its GGUF "
    "(eullm.decision.temperature) for the engine to apply by default; a "
    "server that applies another — an engine that does not read the key — "
    "serves, and was measured on, probabilities the fit did not calibrate: "
    "update the engine, or have every request send eullm.temperature",
}
#: How far an applied temperature may be from the GGUF's float32 and still
#: be it: an engine may read the key as text, printed to six decimals.
TEMPERATURE_TOLERANCE = 1e-5


def defaults():
    return {name: value for name, (value, _) in THRESHOLDS.items()}


def wilson(k, n, z=1.96):
    """95% Wilson score interval of k successes in n."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def ece(rows, bins=ECE_BINS):
    """rows: (confidence of the top answer, whether it was right)."""
    if not rows:
        return None
    sums = [[0, 0.0, 0.0] for _ in range(bins)]
    for confidence, right in rows:
        b = min(int(confidence * bins), bins - 1)
        sums[b][0] += 1
        sums[b][1] += confidence
        sums[b][2] += 1.0 if right else 0.0
    return sum(abs(hits - conf) for _, conf, hits in sums) / len(rows)


def percentile(values, q):
    """Nearest-rank percentile, as the other ReflexBench reports have it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def majority_baseline(answers):
    """answers: (question id, right class). The accuracy of always giving
    each question's commonest right answer."""
    by_question = {}
    for qid, right in answers:
        counts = by_question.setdefault(qid, {})
        counts[right] = counts.get(right, 0) + 1
    total = sum(sum(c.values()) for c in by_question.values())
    return sum(max(c.values()) for c in by_question.values()) / total if total else None


def mcnemar(pairs):
    """pairs: (candidate right, current right) per answer. The discordant
    counts and the two-sided exact p-value that the two are equally often
    right."""
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, cur in pairs if cur and not a)
    n = b + c
    if n == 0:
        return {"candidate_only": 0, "current_only": 0, "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n
    return {"candidate_only": b, "current_only": c, "p_value": min(1.0, 2 * tail)}


def summarize(records, serve_mode, reference_mode, modes):
    """records: one per (item, question): `kind`, `qid`, `right`, and per
    mode either the class probabilities or None when the server refused.
    Returns the metrics per type and over all ("all")."""
    groups = {"all": records}
    for r in records:
        groups.setdefault(r["kind"], []).append(r)
    out = {}
    for name, rows in groups.items():
        if not rows:
            continue
        served = [r for r in rows if r["modes"].get(serve_mode) is not None]
        right = [argmax(r["modes"][serve_mode]["p"]) == r["right"] for r in served]
        n = len(rows)
        k = sum(right)
        coverage = [
            r["modes"][serve_mode]["coverage"]
            for r in served
            if r["modes"][serve_mode].get("coverage") is not None
        ]
        entry = {
            "answers": n,
            "refused": n - len(served),
            "accuracy": k / n,
            "accuracy_ci95": wilson(k, n),
            "majority_baseline": majority_baseline([(r["qid"], r["right"]) for r in rows]),
            "ece": ece(
                [
                    (max(r["modes"][serve_mode]["p"]), ok)
                    for r, ok in zip(served, right, strict=True)
                ]
            ),
            "nll": (
                statistics.mean(
                    -math.log(max(r["modes"][serve_mode]["p"][r["right"]], 1e-12)) for r in served
                )
                if served
                else None
            ),
            "coverage": statistics.mean(coverage) if coverage else None,
            "modes": {},
        }
        for mode in modes:
            if mode == reference_mode:
                continue
            compared = [
                r
                for r in rows
                if r["modes"].get(mode) is not None and r["modes"].get(reference_mode) is not None
            ]
            deltas = [
                max(
                    abs(a - b)
                    for a, b in zip(
                        r["modes"][mode]["p"], r["modes"][reference_mode]["p"], strict=True
                    )
                )
                for r in compared
            ]
            flips = sum(
                argmax(r["modes"][mode]["p"]) != argmax(r["modes"][reference_mode]["p"])
                for r in compared
            )
            entry["modes"][mode] = {
                "compared": len(compared),
                "max_delta": max(deltas) if deltas else None,
                "mean_delta": statistics.mean(deltas) if deltas else None,
                "flips": flips,
                "flip_rate": flips / len(compared) if compared else None,
            }
        noise = [m for m in entry["modes"].values() if m["compared"]]
        compared = sum(m["compared"] for m in noise)
        entry["max_mode_delta"] = max((m["max_delta"] for m in noise), default=None)
        entry["mode_flip_rate"] = sum(m["flips"] for m in noise) / compared if noise else None
        out[name] = entry
    return out


def checks(candidate, thresholds, current=None):
    """The candidate's checks, per type: `{passed, check, type, text,
    reason}`. `current`, when given, is the model it would replace,
    measured on the same items."""
    t = thresholds
    out = []

    def check(passed, name, scope, text):
        reason = THRESHOLDS[name][1] if name in THRESHOLDS else FIXED[name]
        out.append(
            {"passed": bool(passed), "check": name, "type": scope, "text": text, "reason": reason}
        )

    for kind in KINDS:
        m = candidate["by_type"].get(kind)
        if not m:
            continue
        check(
            m["answers"] >= t["min_answers"],
            "min_answers",
            kind,
            f"{m['answers']} answers (at least {t['min_answers']})",
        )
        check(
            m["accuracy"] > m["majority_baseline"],
            "beats_majority",
            kind,
            f"accuracy {m['accuracy']:.3f} against {m['majority_baseline']:.3f} for "
            "the commonest answer every time",
        )
        if m["ece"] is not None:
            check(
                m["ece"] <= t["max_ece"],
                "max_ece",
                kind,
                f"ECE {m['ece']:.3f} (at most {t['max_ece']})",
            )
        if m["max_mode_delta"] is not None:
            check(
                m["max_mode_delta"] <= t["max_mode_delta"],
                "max_mode_delta",
                kind,
                f"largest probability change between modes {m['max_mode_delta']:.3f} "
                f"(at most {t['max_mode_delta']})",
            )
            check(
                m["mode_flip_rate"] <= t["max_mode_flips"],
                "max_mode_flips",
                kind,
                f"answers changed by the mode {100 * m['mode_flip_rate']:.1f}% "
                f"(at most {100 * t['max_mode_flips']:.0f}%)",
            )
        if m["coverage"] is not None:
            check(
                m["coverage"] >= t["min_coverage"],
                "min_coverage",
                kind,
                f"coverage {m['coverage']:.3f} (at least {t['min_coverage']})",
            )
        if current and current["by_type"].get(kind):
            c = current["by_type"][kind]
            check(
                m["accuracy"] >= c["accuracy"] - t["max_accuracy_drop"],
                "max_accuracy_drop",
                kind,
                f"accuracy {m['accuracy']:.3f} against {c['accuracy']:.3f} for the "
                f"current model (at most {t['max_accuracy_drop']} lower)",
            )
    carried = candidate.get("temperature_gguf")
    if carried is not None and candidate.get("temperature_asked") is None:
        # Asked for a temperature of its own, the server applies that one:
        # what the GGUF carries is then not what is measured, by choice.
        applied = candidate.get("temperatures_applied") or []
        check(
            applied
            and all(
                math.isclose(
                    t, carried, rel_tol=TEMPERATURE_TOLERANCE, abs_tol=TEMPERATURE_TOLERANCE
                )
                for t in applied
            ),
            "gguf_temperature",
            "all",
            f"temperature applied {', '.join(f'{t:.4g}' for t in applied) or 'not reported'}"
            f", the GGUF carries {carried:.4g}",
        )
    serve = candidate["serve_mode"]
    p95 = candidate["latency_ms"].get(serve, {}).get("p95")
    if t.get("max_p95_ms") is not None and p95 is not None:
        check(
            p95 <= t["max_p95_ms"],
            "max_p95_ms",
            "all",
            f"p95 {p95:.0f} ms in {serve} (at most {t['max_p95_ms']:.0f})",
        )
    if current:
        current_p95 = current["latency_ms"].get(serve, {}).get("p95")
        if p95 is not None and current_p95:
            check(
                p95 <= current_p95 * t["max_latency_ratio"],
                "max_latency_ratio",
                "all",
                f"p95 {p95:.0f} ms against {current_p95:.0f} ms for the current model "
                f"(at most {t['max_latency_ratio']}x)",
            )
    return out
