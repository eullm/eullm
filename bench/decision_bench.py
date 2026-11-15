#!/usr/bin/env python3
"""Shared-prefix benchmark for the decision layer (`POST /v1/systemone`).

Asks Q questions about the same state, for every Q in --questions and every
state size in --states, in each evaluation mode:

  * `shared_prefix` (the server's default): the state is decoded once, then
    every question on its own right after it, so a question's answer depends
    on the state and that question only;
  * `batched`: the state once, then every question together in one batch —
    the fewest decode calls, but an answer moves with the other questions in
    the batch, by the model's own numerical noise;
  * `separate`: each question decoded on its own from scratch, the baseline.

For each case it reports:

  * tokens decoded with the state shared and without, and the ratio — the
    saving the shared prefix is supposed to buy, from the server's own counts;
  * decode time of each mode (prefix + questions, from the server's
    `eullm.timings_ms`, median of --repeat runs), and the speedup of
    `shared_prefix` over `separate`;
  * how far each mode's answers are from the baseline's: the largest
    difference in any raw probability, marked with * when some question's
    top answer differs.

The three modes read the same tokens but hand them to the kernels in
batches of different shapes, and on quantized weights that alone moves an
answer by the model's own numerical noise: the dP columns (see the tests in
engine/src/inference/decision.rs). It is a property of the model, not a bug
— `shared_prefix` guarantees only that the noise does not depend on the
other questions asked — but it is the number to know before comparing
calibrations measured in one mode with answers served in another.

Start the server with a decision model and enough context for the largest
case (a 4k-token state with 64 questions needs about 9k tokens in `batched`
mode):

    eullm serve --decision-model qwen3-0.6b --decision-ctx 16384

then:

    python bench/decision_bench.py --url http://localhost:11434
    python bench/decision_bench.py --states 256,1024 --questions 1,8,64 --repeat 5 \\
        --json results/decision-bench.json

--details prints, for every case, the question whose answer moved most in
each mode, with its coverage; --order-check asks the same questions again in
reverse order, in `shared_prefix` and in `batched` mode, and in
`shared_prefix` mode also the first and last question alone and the whole
request once more with the state the server kept from the request before
(timed: "state kept"). In `shared_prefix` every answer must come back
identical (dP 0.0000); `batched` shows how much an answer moves only because
of where its question sits in the batch.

The timed runs never reuse a kept state: each gets a state of its own, and
a tiny request about another state goes before each mode's first run.

Only the Python standard library is needed. Exit code is nonzero if any
request failed for a reason other than exceeding --decision-ctx, or if a
`shared_prefix` answer changed with the other questions asked.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# Sentences for a synthetic support/legal ticket history. Varied enough that
# the state does not collapse into a few repeated tokens.
SENTENCES = [
    "The customer reports that payouts to the connected bank account have failed since Monday.",
    "A second charge for the September invoice appeared on the card statement on the 14th.",
    "Support replied with the standard refund policy and asked for the transaction reference.",
    "The client's lawyer wrote that the supplier has not delivered the goods for three months.",
    "A hearing before the court of Milan is scheduled for Thursday at ten in the morning.",
    "The account was migrated to the new billing platform during the weekend maintenance window.",
    "Two other merchants in the same region reported delayed settlements on the status page.",
    "The contract includes a penalty clause of two percent of the order value per week of delay.",
    "The customer asked to be called back today and mentioned that payroll is due on Friday.",
    "An internal note says the risk team placed a temporary hold on the account on Tuesday.",
    "The invoice lists VAT at twenty-two percent and a reverse charge note for the EU shipment.",
    "The supplier claims that force majeure applies because of a strike at the port of Genoa.",
]

TOPICS = [
    "a failed payout",
    "a duplicate charge",
    "a court hearing",
    "a contract penalty",
    "a platform migration",
    "an account hold",
    "a VAT question",
    "a delivery delay",
    "a callback request",
    "a refund request",
    "a strike",
    "a legal deadline",
    "a status page incident",
    "payroll",
    "an invoice dispute",
    "a bank transfer",
]


def make_state(target_tokens):
    """About `target_tokens` tokens of ticket history (~4 characters per
    token for English text; the server reports the real count)."""
    target_chars = target_tokens * 4
    lines = []
    length = 0
    while length < target_chars:
        i = len(lines)
        line = f"[day {1 + i // len(SENTENCES)}] {SENTENCES[i % len(SENTENCES)]}"
        lines.append(line)
        length += len(line) + 1
    return "\n".join(lines)


def make_questions(n):
    """`n` distinct questions cycling through the three types, so every
    question's own tokens differ from the others' — as real ones do."""
    questions = {}
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        round_ = i // len(TOPICS)
        kind = ("noul", "choice", "score")[i % 3]
        qid = f"q{i:02d}"
        if kind == "noul":
            questions[qid] = {
                "type": "noul",
                "instructions": f"Does the state mention {topic}? (check {round_ + 1})",
            }
        elif kind == "choice":
            questions[qid] = {
                "type": "choice",
                "instructions": f"Which team should handle {topic}? (check {round_ + 1})",
                "criteria": {
                    "billing": "Charges, invoices, refunds and payouts",
                    "legal": "Contracts, disputes and court deadlines",
                    "tech": "Bugs, outages and migrations",
                    "other": "Anything else",
                },
            }
        else:
            questions[qid] = {
                "type": "score",
                "instructions": f"How urgent is {topic} for the customer? (check {round_ + 1})",
                "criteria": [
                    "Not urgent",
                    "Can wait a week",
                    "Should be handled this week",
                    "Should be handled today",
                    "Needs an immediate response",
                ],
            }
    return questions


class RequestFailed(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RequestFailed(e.code, e.read().decode(errors="replace")) from None


def decode_ms(response):
    t = response["eullm"]["timings_ms"]
    return t["prefix"] + t["questions"]


def raw_values(answer):
    """What an answer was read from: the log-probability of each code (an
    instruction-tuned model), or each option's verdict score (a Jev-Style
    model)."""
    extension = answer["eullm"]
    return extension["logprobs"] if "logprobs" in extension else extension["scores"]


def compare(first, second):
    """Largest raw-probability difference between two responses' answers to
    the questions both asked, the largest difference in what they were read
    from (log-probabilities, or verdict scores), and whether every argmax
    agrees."""
    max_dp = 0.0
    max_dlp = 0.0
    agree = True
    for qid, b in second["answers"].items():
        a = first["answers"][qid]
        pa, pb = a["eullm"]["raw_probabilities"], b["eullm"]["raw_probabilities"]
        la, lb = raw_values(a), raw_values(b)
        logprobs = "logprobs" in a["eullm"]
        for label in pa:
            max_dp = max(max_dp, abs(pa[label] - pb[label]))
            # A code no one would answer carries only noise in its tail.
            if not logprobs or max(la[label], lb[label]) > -15.0:
                max_dlp = max(max_dlp, abs(la[label] - lb[label]))
        if max(pa, key=pa.get) != max(pb, key=pb.get):
            agree = False
    return max_dp, max_dlp, agree


def worst(first, second):
    """The question whose raw probabilities differ most between two runs:
    `(id, largest probability difference, answer in first, answer in second)`."""
    found = None
    for qid, a in first["answers"].items():
        b = second["answers"][qid]
        pa, pb = a["eullm"]["raw_probabilities"], b["eullm"]["raw_probabilities"]
        dp = max(abs(pa[k] - pb[k]) for k in pa)
        if found is None or dp > found[1]:
            found = (qid, dp, a, b)
    return found


def show_worst(mode, response, separate):
    """Print the question that moved most from the baseline, with its
    coverage in both runs: similar coverage and a shifted distribution is
    arithmetic; a coverage that collapses in one run would mean that run
    read the wrong logits."""
    qid, dp, a, b = worst(response, separate)

    def fmt(p):
        return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in p.items()) + "}"

    # A verdict model reads every option at a slot of its own: no coverage.
    coverage = (
        f", coverage {a['eullm']['coverage']:.4f} / {b['eullm']['coverage']:.4f} separate"
        if "coverage" in a["eullm"]
        else ""
    )
    print(f"{'':>11}{mode}: worst {qid} ({a['type']}), dP {dp:.4f}{coverage}")
    print(f"{'':>13}{mode:<14}{fmt(a['eullm']['raw_probabilities'])}")
    print(f"{'':>13}{'separate':<14}{fmt(b['eullm']['raw_probabilities'])}")


def run_case(url, model, state, questions, mode, repeat, timeout):
    """`repeat` requests. The server keeps the state it decoded last and
    skips decoding it again when the next request asks about the same one,
    so every run after the first gets a state of its own — a marker line in
    front — and all of them are timed from a cold state. The first run's
    state is the unmarked one, the same in every mode, so answers compare."""
    runs = []
    for r in range(repeat):
        payload = {
            "state": state if r == 0 else f"(run {r})\n{state}",
            "questions": questions,
            "eullm": {"mode": mode},
        }
        if model:
            payload["model"] = model
        started = time.perf_counter()
        response = post(url, payload, timeout)
        response["_client_ms"] = (time.perf_counter() - started) * 1000.0
        runs.append(response)
    return runs


def evict(url, model, timeout):
    """One tiny request about another state, so the next one cannot start
    from a state the server kept from the request before."""
    payload = {
        "state": "(bench: nothing kept)",
        "questions": {"q": {"type": "noul", "instructions": "Is this empty?"}},
    }
    if model:
        payload["model"] = model
    post(url, payload, timeout)


def median(values):
    return statistics.median(values) if values else float("nan")


def identical(first, second):
    """Whether every question `second` answered got bit-for-bit the same
    log-probabilities (or verdict scores) in `first` (JSON carries every f64
    exactly)."""
    return all(
        raw_values(first["answers"][qid]) == raw_values(b)
        for qid, b in second["answers"].items()
    )


# Evaluation modes, as the server names them, and the key prefix each one's
# numbers get in --json output.
MODES = {"shared_prefix": "shared", "batched": "batched", "separate": "separate"}


def ms_column(case, key, width):
    """A mode's median decode time, or a dash where it did not run."""
    value = case.get(f"{key}_decode_ms")
    return f"{value:>{width}.1f}" if value is not None else f"{'-':>{width}}"


def order_check(ask, runs, questions):
    """The same questions reversed and, in shared_prefix mode, the first and
    last alone: every question reads the same tokens, only what else is in
    the request changes. In shared_prefix mode nothing may move; in batched
    mode what moves, moves because of where its question sits in the batch.

    In shared_prefix mode the same request is also asked once more right
    after the others, about the same state: the server then starts from the
    state it kept (`prefix_reused`), which must not move an answer either.

    Returns the numbers for --json, one line to print per mode, and whether
    every shared_prefix answer came back bit for bit the same. A failed
    request raises RequestFailed."""
    ids = list(questions)
    reversed_questions = dict(reversed(list(questions.items())))
    numbers = {}
    lines = []
    same = True
    for mode in ("shared_prefix", "batched"):
        if mode not in runs:
            continue
        key = MODES[mode]
        base = runs[mode][0]
        others = {"reversed": ask(mode, 1, reversed_questions)[0]}
        if mode == "shared_prefix":
            alone = {"answers": {}}
            for qid in (ids[0], ids[-1]):
                answer = ask(mode, 1, {qid: questions[qid]})[0]
                alone["answers"].update(answer["answers"])
            others["alone"] = alone
            others["kept state"] = ask(mode, 1, questions)[0]
        parts = []
        for name, other in others.items():
            dp, dlp, agree = compare(base, other)
            exact = identical(base, other)
            slug = name.replace(" ", "_")
            numbers.update(
                {
                    f"{key}_{slug}_max_probability_difference": dp,
                    f"{key}_{slug}_max_logprob_difference": dlp,
                    f"{key}_{slug}_argmax_agrees": agree,
                    f"{key}_{slug}_identical": exact,
                }
            )
            parts.append(f"{name} dP {dp:.4f}" + ("" if agree else ", argmax NO"))
            if mode == "shared_prefix":
                same = same and exact
        verdict = ""
        if mode == "shared_prefix":
            kept = others["kept state"]["eullm"]
            numbers["shared_kept_state_reused"] = kept.get("prefix_reused", False)
            numbers["shared_kept_state_decode_ms"] = decode_ms(others["kept state"])
            parts.append(
                f"state kept: {decode_ms(others['kept state']):.1f} ms"
                + ("" if kept.get("prefix_reused") else " (NOT reused)")
            )
            verdict = " — identical" if same else " — NOT IDENTICAL"
        lines.append(f"{'':>11}{mode + ':':<15}{', '.join(parts)}{verdict}")
    return numbers, lines, same


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url", default="http://localhost:11434", help="server base URL"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="decision model to use (default: the one the server has loaded)",
    )
    parser.add_argument(
        "--states",
        default="256,1024,4096",
        help="approximate state sizes in tokens, comma-separated",
    )
    parser.add_argument(
        "--questions",
        default="1,4,8,16,32,64",
        help="questions per request, comma-separated (max 64)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="runs per case; medians are reported"
    )
    parser.add_argument(
        "--max-separate-tokens",
        type=int,
        default=150_000,
        help="skip the separate baseline when it would decode more tokens than this",
    )
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="per-request timeout, seconds"
    )
    parser.add_argument(
        "--json", default=None, help="write every case's numbers to this file"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print the question that differs most from the baseline, per mode and case",
    )
    parser.add_argument(
        "--order-check",
        action="store_true",
        help="also ask the questions in reverse order, and the first and last alone, "
        "and report how far the answers move",
    )
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/v1/systemone"
    states = [int(s) for s in args.states.split(",") if s.strip()]
    counts = [int(q) for q in args.questions.split(",") if q.strip()]
    if any(q < 1 or q > 64 for q in counts):
        parser.error("--questions must be between 1 and 64")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    # Warm-up: loads the model if --model names one, and keeps first-request
    # costs out of the first case.
    try:
        warm = run_case(
            url,
            args.model,
            make_state(64),
            make_questions(2),
            "shared_prefix",
            1,
            args.timeout,
        )[0]
    except (RequestFailed, OSError) as e:
        print(f"Warm-up request failed: {e}", file=sys.stderr)
        return 1
    # Which attention kernels the server runs decisions with: the numbers
    # below depend on it, and a comparison of two runs means nothing if
    # both turn out to have used the same setting.
    flash_attn = warm["eullm"].get("flash_attn", "unknown")
    readout = warm["eullm"].get("readout", "codes")
    print(f"model {warm['model']} ({readout} readout), flash attention {flash_attn}")
    print(
        "tokens: decoded with the state shared / without; speedup: separate over "
        "shared_prefix;\ndP: largest probability difference from separate, "
        "* when a question's top answer differs\n"
    )

    header = (
        f"{'state':>6} {'Q':>3} | {'tokens sh/sep':>15} {'saving':>7} | "
        f"{'ms shared':>10} {'batched':>9} {'separate':>9} {'speedup':>8} | "
        f"{'dP shared':>10} {'batched':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    failed = False
    moved = []
    for state_tokens in states:
        state = make_state(state_tokens)
        for q in counts:
            questions = make_questions(q)
            case = {"state_tokens_target": state_tokens, "questions": q}

            def ask(mode, repeat, asked=questions):
                return run_case(
                    url, args.model, state, asked, mode, repeat, args.timeout
                )

            runs = {}
            for mode, key in MODES.items():
                if (
                    mode == "separate"
                    and "shared_prefix" in runs
                    and runs["shared_prefix"][0]["eullm"]["prompt_tokens"]
                    > args.max_separate_tokens
                ):
                    case["separate_skipped"] = "--max-separate-tokens"
                    continue
                try:
                    evict(url, args.model, args.timeout)
                    runs[mode] = ask(mode, args.repeat)
                except RequestFailed as e:
                    if e.status == 400 and "--decision-ctx" in e.body:
                        case[f"{key}_skipped"] = "exceeds --decision-ctx"
                    else:
                        print(
                            f"{state_tokens:>6} {q:>3} | {mode} failed: {e}",
                            file=sys.stderr,
                        )
                        failed = True
                if mode == "shared_prefix" and mode not in runs:
                    break

            shared = runs.get("shared_prefix")
            if shared is None:
                if "shared_skipped" in case:
                    print(f"{state_tokens:>6} {q:>3} | exceeds --decision-ctx, skipped")
                    results.append(case)
                continue
            s0 = shared[0]["eullm"]
            case.update(
                {
                    "prompt_tokens": s0["prompt_tokens"],
                    "shared_prefix_tokens": s0["shared_prefix_tokens"],
                }
            )
            for mode, responses in runs.items():
                key = MODES[mode]
                first = responses[0]["eullm"]
                timings = [r["eullm"]["timings_ms"] for r in responses]
                case.update(
                    {
                        f"{key}_evaluated_tokens": first["evaluated_tokens"],
                        f"{key}_decode_ms": median([decode_ms(r) for r in responses]),
                        f"{key}_request_ms": median(
                            [r["eullm"]["request_ms"] for r in responses]
                        ),
                        f"{key}_context_ms": median([t["context"] for t in timings]),
                        f"{key}_readout_ms": median([t["readout"] for t in timings]),
                    }
                )

            separate = runs.get("separate")
            dp_columns = {}
            if separate:
                for mode in ("shared_prefix", "batched"):
                    if mode not in runs:
                        continue
                    key = MODES[mode]
                    dp, dlp, agree = compare(runs[mode][0], separate[0])
                    case.update(
                        {
                            f"{key}_max_probability_difference": dp,
                            f"{key}_max_logprob_difference": dlp,
                            f"{key}_argmax_agrees": agree,
                        }
                    )
                    dp_columns[mode] = f"{dp:.4f}{' ' if agree else '*'}"

            saving = s0["prompt_tokens"] / s0["evaluated_tokens"]
            speedup = (
                f"{case['separate_decode_ms'] / case['shared_decode_ms']:>7.1f}x"
                if separate
                else f"{'-':>8}"
            )
            print(
                f"{state_tokens:>6} {q:>3} | "
                f"{s0['evaluated_tokens']:>6}/{s0['prompt_tokens']:<8} {saving:>6.1f}x | "
                f"{ms_column(case, 'shared', 10)} {ms_column(case, 'batched', 9)} "
                f"{ms_column(case, 'separate', 9)} {speedup} | "
                f"{dp_columns.get('shared_prefix', '-'):>10} "
                f"{dp_columns.get('batched', '-'):>9}"
            )
            if args.details and separate:
                for mode in ("shared_prefix", "batched"):
                    if mode in runs:
                        show_worst(mode, runs[mode][0], separate[0])

            if args.order_check and q > 1:
                try:
                    numbers, lines, same = order_check(ask, runs, questions)
                except RequestFailed as e:
                    print(f"{'':>11}order check failed: {e}", file=sys.stderr)
                    failed = True
                else:
                    case.update(numbers)
                    print("\n".join(lines))
                    if not same:
                        moved.append(f"state {state_tokens}, {q} questions")
            results.append(case)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "url": args.url,
                    "model": warm["model"],
                    "flash_attn": flash_attn,
                    "repeat": args.repeat,
                    "cases": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.json}")
    if moved:
        print(
            "\nshared_prefix answers changed with the other questions asked in: "
            + "; ".join(moved)
            + ". They must not: please report it with this output.",
            file=sys.stderr,
        )
    return 1 if failed or moved else 0


if __name__ == "__main__":
    sys.exit(main())
