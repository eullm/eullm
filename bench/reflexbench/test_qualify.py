"""Offline tests for the qualification test: the labelled sets, the traces
reader, the metrics and their thresholds, and the client against a
stand-in server. No network, no model:

    python3 -m unittest discover -s bench/reflexbench
"""

import http.server
import io
import json
import os
import pathlib
import struct
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qf_data  # noqa: E402
import qf_metrics  # noqa: E402
import qualify  # noqa: E402
from rb_methods import ServerError  # noqa: E402

TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Payments", "tech": "Bugs", "other": None}}
URGENT = {"type": "noul", "instructions": "Is it urgent?"}
SEVERITY = {"type": "score", "instructions": "How severe?",
            "criteria": ["Cosmetic", {"label": "Degraded", "description": "slow"}, "Blocking"]}


def labelled_rows(n=60):
    """n requests of three questions, their right answers spread out."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"r{i}",
            "state": f"ticket {i}",
            "questions": {"is_urgent": URGENT, "team": TEAM, "severity": SEVERITY},
            "answers": {"is_urgent": i % 3 == 0, "team": ["billing", "tech", "other"][i % 3],
                        "severity": i % 3},
            "sources": {"is_urgent": "feedback:user", "team": "rules", "severity": "rules"},
        })
    return rows


def write(path, rows):
    pathlib.Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    return path


def right_class(state, qid):
    i = int(state.split()[-1])
    return {"is_urgent": 0 if i % 3 == 0 else 1, "team": i % 3, "severity": i % 3}[qid]


def answer_json(question, p, coverage=0.99):
    labels = qf_data.labels(question)
    if question["type"] == "noul":
        return {"type": "noul", "noul": p[0], "eullm": {"coverage": coverage}}
    best = max(range(len(p)), key=p.__getitem__)
    return {"type": question["type"], "choice": labels[best],
            "probabilities": dict(zip(labels, p)), "eullm": {"coverage": coverage}}


def calibrated(state, qid, question, mode):
    """Right with probability 0.95 on 19 states in 20, wrong as surely on
    the rest: 95% sure and 95% right."""
    n = len(qf_data.labels(question))
    right = right_class(state, qid)
    i = int(state.split()[-1])
    target = right if i % 20 else (right + 1) % n
    return [0.95 if k == target else 0.05 / (n - 1) for k in range(n)]


def stand_in(answer=calibrated, refuse=(), batched_shift=0.0, model="stand-in",
             temperature=1.0):
    """A stand-in `/v1/systemone`: `answer(state, qid, question, mode)`
    gives each question's class probabilities; in `batched` mode they move
    by `batched_shift` towards the next class; a question named in `refuse`
    is refused as EuLLM refuses one, with a 422 naming it. It reports the
    request's temperature as applied, or its own default, `temperature`."""
    sent = []

    def post(url, payload, api_key, timeout):
        sent.append(payload)
        for qid in payload["questions"]:
            if qid in refuse:
                body = {"error": {"code": "invalid_question", "message": "too many options",
                                  "question": qid}}
                raise ServerError(422, json.dumps(body))
        mode = payload["eullm"]["mode"]
        answers = {}
        for qid, question in payload["questions"].items():
            p = answer(payload["state"], qid, question, mode)
            if mode == "batched" and batched_shift:
                k = max(range(len(p)), key=p.__getitem__)
                moved = min(batched_shift, p[k])
                p = list(p)
                p[k] -= moved
                p[(k + 1) % len(p)] += moved
            answers[qid] = answer_json(question, p)
        return {"model": model, "answers": answers,
                "eullm": {"readout": "codes", "request_ms": 2.0, "evaluated_tokens": 30,
                          "temperature": payload["eullm"].get("temperature", temperature)}}

    return post, sent


def gguf_carrying(path, temperature=None):
    """A GGUF of no tensors, carrying `temperature` as `decisions export`
    writes it — a float32 under eullm.decision.temperature — when given."""

    def string(text):
        return struct.pack("<Q", len(text.encode())) + text.encode()

    pairs = [string("general.architecture") + struct.pack("<I", 8) + string("qwen3")]
    if temperature is not None:
        pairs.append(string("eullm.decision.temperature") + struct.pack("<If", 6, temperature))
    pathlib.Path(path).write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs))
                                   + b"".join(pairs))
    return path


class DataTest(unittest.TestCase):
    def test_a_labelled_set_of_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = labelled_rows(2)
            rows[1]["answers"].pop("team")  # a question with no answer is not asked
            path = write(pathlib.Path(tmp) / "set.jsonl", rows)
            labelled = qf_data.from_jsonl(path)
        first, second = labelled.items
        self.assertEqual(list(first.questions), ["is_urgent", "team", "severity"])
        self.assertEqual(first.answers, {"is_urgent": 0, "team": 0, "severity": 0})
        self.assertEqual(list(second.questions), ["is_urgent", "severity"])
        self.assertEqual(second.sources["team"], "rules")

    def test_decision_calibrations_format_is_a_request_of_one_question(self):
        item = qf_data.item_from_row({"state": "s", "question": TEAM, "label": "tech"}, 1)
        self.assertEqual((list(item.questions), item.answers), (["q"], {"q": 1}))

    def test_answers_name_classes(self):
        cases = [(URGENT, True, 0), (URGENT, "no", 1), (URGENT, 1, 0), (TEAM, "other", 2),
                 (SEVERITY, 2, 2), (SEVERITY, "1", 1), (SEVERITY, "Degraded", 1),
                 (SEVERITY, "Blocking", 2)]
        for question, answer, index in cases:
            self.assertEqual(qf_data.answer_index(question, answer), index, answer)
        for question, answer in ((TEAM, "sales"), (SEVERITY, 3), (SEVERITY, True),
                                 (URGENT, "maybe")):
            with self.assertRaises(ValueError):
                qf_data.answer_index(question, answer)

    def test_a_bad_line_names_its_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = labelled_rows(2)
            rows[1]["answers"]["team"] = "sales"
            path = write(pathlib.Path(tmp) / "set.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "line 2"):
                qf_data.from_jsonl(path)

    def test_traces_with_feedback_are_requests(self):
        evaluated = {"type": "choice", "instructions": "Which team?",
                     "options": [{"name": "billing", "description": "Payments"},
                                 {"name": "tech", "description": ""}]}
        decisions = [
            {"schema": 1, "id": "a", "readout": "verdict", "state": '{"text": "refund"}',
             "questions": {"team": evaluated, "is_urgent": URGENT}, "answers": {}},
            {"schema": 1, "id": "b", "state": "no feedback on this one",
             "questions": {"team": TEAM}},
            {"schema": 1, "id": "c", "state": "s", "questions": [dict(TEAM, id="team")]},
        ]
        feedback = [
            {"kind": "feedback", "id": "a", "timestamp": "2026-10-02T10:00:00Z",
             "answers": {"team": "billing"}, "source": "user"},
            {"kind": "feedback", "id": "a", "timestamp": "2026-10-02T09:00:00Z",
             "answers": {"team": "tech", "is_urgent": True}, "source": "rule"},
            {"kind": "feedback", "id": "c", "answers": {"team": "sales"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write(pathlib.Path(tmp) / "decisions.jsonl", decisions)
            path = write(pathlib.Path(tmp) / "feedback.jsonl", feedback)
            with open(path, "a", encoding="utf-8") as f:
                f.write("{cut\n")
            labelled = qf_data.from_traces(tmp)
        [item] = labelled.items
        self.assertEqual(item.id, "a")
        # The structured state goes back in as JSON, for each model to read
        # as it reads JSON.
        self.assertEqual(item.state, {"text": "refund"})
        self.assertEqual(item.questions["team"]["criteria"], {"billing": "Payments", "tech": None})
        # By timestamp, not file order: the 10:00 answer is the later one.
        self.assertEqual(item.answers, {"team": 0, "is_urgent": 0})
        self.assertEqual(item.sources, {"team": "feedback:user", "is_urgent": "feedback:rule"})
        self.assertEqual(labelled.skipped["malformed line"], 1)
        self.assertEqual(sum(n for r, n in labelled.skipped.items() if "sales" in r), 1)

    def test_only_some_sources(self):
        labelled = qf_data.LabelledSet("s", [qf_data.item_from_row(r, 1) for r in labelled_rows(2)])
        kept = qf_data.only_sources(labelled, ["feedback"])
        self.assertEqual([list(i.questions) for i in kept.items], [["is_urgent"]] * 2)
        self.assertIs(qf_data.only_sources(labelled, []), labelled)


class MetricsTest(unittest.TestCase):
    def test_wilson_ece_and_percentiles(self):
        lo, hi = qf_metrics.wilson(25, 50)
        self.assertAlmostEqual(hi - 0.5, 0.5 - lo)
        self.assertTrue(0.13 < 0.5 - lo < 0.14)  # the ±14 points of min_answers
        self.assertEqual(qf_metrics.ece([(1.0, True), (0.5, False), (0.5, True)]), 0.0)
        self.assertAlmostEqual(qf_metrics.ece([(0.9, False)] * 2), 0.9)
        self.assertEqual(qf_metrics.percentile([5, 1, 3, 2, 4], 0.5), 3)
        self.assertEqual(qf_metrics.percentile(list(range(1, 101)), 0.95), 95)
        self.assertIsNone(qf_metrics.percentile([], 0.5))

    def test_the_majority_baseline_and_mcnemar(self):
        self.assertEqual(qf_metrics.majority_baseline([("a", 0), ("a", 0), ("a", 1),
                                                       ("b", 2), ("b", 1)]), 3 / 5)
        m = qf_metrics.mcnemar([(True, False)] * 8 + [(False, True)] * 2 + [(True, True)] * 50)
        self.assertEqual((m["candidate_only"], m["current_only"]), (8, 2))
        self.assertAlmostEqual(m["p_value"], 2 * 56 / 1024)
        self.assertEqual(qf_metrics.mcnemar([(True, True)])["p_value"], 1.0)

    def record(self, kind, right, modes):
        return {"item": "i", "qid": kind, "kind": kind, "right": right, "modes": modes}

    def test_summarize_counts_a_refusal_as_wrong_and_measures_the_modes(self):
        records = [
            self.record("choice", 0, {"separate": {"p": [0.8, 0.2], "coverage": 1.0},
                                      "shared_prefix": {"p": [0.8, 0.2], "coverage": 1.0},
                                      "batched": {"p": [0.45, 0.55], "coverage": 1.0}}),
            self.record("choice", 1, {"separate": None, "shared_prefix": None, "batched": None}),
        ]
        m = qf_metrics.summarize(records, "shared_prefix", "separate",
                                 ["separate", "shared_prefix", "batched"])["choice"]
        self.assertEqual((m["answers"], m["refused"], m["accuracy"]), (2, 1, 0.5))
        self.assertAlmostEqual(m["max_mode_delta"], 0.35)
        self.assertEqual(m["modes"]["batched"]["flips"], 1)
        self.assertEqual(m["modes"]["shared_prefix"]["max_delta"], 0.0)
        self.assertAlmostEqual(m["mode_flip_rate"], 0.5)  # one of two comparisons
        self.assertAlmostEqual(m["ece"], 0.2)

    def candidate(self, **change):
        m = {"answers": 60, "refused": 0, "accuracy": 0.9, "majority_baseline": 0.4,
             "ece": 0.03, "max_mode_delta": 0.01, "mode_flip_rate": 0.0, "coverage": 0.98}
        m.update(change)
        return {"by_type": {"choice": m}, "serve_mode": "shared_prefix",
                "latency_ms": {"shared_prefix": {"p50": 40.0, "p95": 60.0}}}

    def failed(self, candidate, current=None, **thresholds):
        t = dict(qf_metrics.defaults(), **thresholds)
        return {c["check"] for c in qf_metrics.checks(candidate, t, current) if not c["passed"]}

    def test_every_threshold_can_fail(self):
        self.assertEqual(self.failed(self.candidate()), set())
        cases = {"min_answers": {"answers": 49}, "beats_majority": {"accuracy": 0.4},
                 "max_ece": {"ece": 0.11}, "max_mode_delta": {"max_mode_delta": 0.06},
                 "max_mode_flips": {"mode_flip_rate": 0.02}, "min_coverage": {"coverage": 0.8}}
        for check, change in cases.items():
            self.assertEqual(self.failed(self.candidate(**change)), {check}, check)
        self.assertEqual(self.failed(self.candidate(), max_p95_ms=50), {"max_p95_ms"})
        current = self.candidate(accuracy=0.93)
        current["latency_ms"]["shared_prefix"]["p95"] = 30.0
        self.assertEqual(self.failed(self.candidate(), current),
                         {"max_accuracy_drop", "max_latency_ratio"})
        # A verdict model reports no coverage; a single mode, no noise.
        self.assertEqual(self.failed(self.candidate(coverage=None, max_mode_delta=None)), set())
        for check in qf_metrics.checks(self.candidate(ece=0.5), qf_metrics.defaults()):
            self.assertTrue(check["reason"])

    def test_the_server_must_apply_the_temperature_the_gguf_carries(self):
        def served(applied, carried=1.37, asked=None):
            return dict(self.candidate(), temperatures_applied=applied,
                        temperature_gguf=carried, temperature_asked=asked)

        # The GGUF's float32, read back exactly or as text to six decimals.
        self.assertEqual(self.failed(served([1.3700000047683716])), set())
        self.assertEqual(self.failed(served([1.37])), set())
        # An engine that does not read the key applies 1; none reported fails too.
        self.assertEqual(self.failed(served([1.0])), {"gguf_temperature"})
        self.assertEqual(self.failed(served([1.37, 1.0])), {"gguf_temperature"})
        self.assertEqual(self.failed(served([])), {"gguf_temperature"})
        # A temperature asked for is the one measured, by choice; no GGUF, no check.
        self.assertEqual(self.failed(served([2.0], asked=2.0)), set())
        found = qf_metrics.checks(served([1.0], carried=None), qf_metrics.defaults())
        self.assertNotIn("gguf_temperature", {c["check"] for c in found})


class ClientTest(unittest.TestCase):
    item = qf_data.item_from_row(labelled_rows(2)[1], 1)  # ticket 1: no, tech, level 1

    def test_a_request_asks_its_mode_and_model(self):
        post, sent = stand_in()
        with mock.patch.object(qualify, "post", post):
            server = qualify.Server("candidate", "http://x/", "my-model")
            answers, ms, info, refusals = qualify.decide(server, self.item, "batched")
        self.assertEqual(sent[0]["eullm"], {"mode": "batched"})
        self.assertEqual(sent[0]["model"], "my-model")
        self.assertEqual(list(sent[0]["questions"]), ["is_urgent", "team", "severity"])
        # Class order: yes before no, options and levels as listed.
        self.assertEqual([round(p, 3) for p in answers["is_urgent"]["p"]], [0.05, 0.95])
        self.assertEqual([round(p, 3) for p in answers["team"]["p"]], [0.025, 0.95, 0.025])
        self.assertEqual(answers["team"]["coverage"], 0.99)
        self.assertEqual((info["readout"], refusals), ("codes", {}))

    def test_the_temperature_it_will_serve_with_goes_with_every_request(self):
        post, sent = stand_in()
        with mock.patch.object(qualify, "post", post):
            qualify.decide(qualify.Server("c", "http://x", temperature=1.4), self.item, "separate")
            qualify.decide(qualify.Server("c", "http://x"), self.item, "separate")
        self.assertEqual(sent[0]["eullm"], {"mode": "separate", "temperature": 1.4})
        self.assertEqual(sent[1]["eullm"], {"mode": "separate"})

    def test_the_temperature_a_gguf_carries_is_read_with_forges_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            carrying = gguf_carrying(pathlib.Path(tmp) / "decide.gguf", 1.37)
            self.assertAlmostEqual(qualify.gguf_temperature(carrying), 1.37, places=6)
            bare = gguf_carrying(pathlib.Path(tmp) / "bare.gguf")
            self.assertIsNone(qualify.gguf_temperature(bare))

    def test_a_refused_question_is_dropped_and_the_rest_asked_again(self):
        post, sent = stand_in(refuse=("team",))
        with mock.patch.object(qualify, "post", post):
            answers, ms, _, refusals = qualify.decide(qualify.Server("c", "http://x"),
                                                      self.item, "separate")
        self.assertIsNone(answers["team"])
        self.assertIsNotNone(answers["severity"])
        self.assertEqual(list(refusals), ["team"])
        self.assertEqual([list(p["questions"]) for p in sent],
                         [["is_urgent", "team", "severity"], ["is_urgent", "severity"]])

    def test_a_request_refused_whole_refuses_every_question_and_other_errors_stop(self):
        def too_long(url, payload, api_key, timeout):
            raise ServerError(422, json.dumps({"error": {"code": "input_budget_exceeded",
                                                         "message": "state too long"}}))

        with mock.patch.object(qualify, "post", too_long):
            answers, ms, _, refusals = qualify.decide(qualify.Server("c", "http://x"),
                                                      self.item, "separate")
        self.assertEqual(set(answers.values()), {None})
        self.assertIsNone(ms)
        self.assertEqual(len(refusals), 3)

        def not_loaded(url, payload, api_key, timeout):
            raise ServerError(400, '{"error": {"code": "model_not_loaded"}}')

        with mock.patch.object(qualify, "post", not_loaded):
            with self.assertRaises(ServerError):
                qualify.decide(qualify.Server("c", "http://x"), self.item, "separate")


def run_main(args, post):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(qualify, "post", post), redirect_stdout(out), redirect_stderr(err):
        status = qualify.main(args)
    return status, out.getvalue()


class MainTest(unittest.TestCase):
    def test_a_good_candidate_passes_and_a_noisy_one_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = write(pathlib.Path(tmp) / "test.labelled.jsonl", labelled_rows(60))
            report = pathlib.Path(tmp) / "q.json"
            post, sent = stand_in()
            status, out = run_main(["--candidate", "http://c", "--data", str(data),
                                    "--out", str(report)], post)
            self.assertEqual(status, 0, out)
            self.assertIn("PASS: every check passed", out)
            body = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(body["verdict"], "PASS")
            [result] = body["results"]
            self.assertEqual(result["by_type"]["choice"]["answers"], 60)
            self.assertAlmostEqual(result["by_type"]["all"]["accuracy"], 0.95)
            self.assertEqual(result["latency_ms"]["batched"]["requests"], 60)
            # One untimed request, then every request in each of three modes.
            self.assertEqual(len(sent), 1 + 3 * 60)
            self.assertEqual({p["eullm"]["mode"] for p in sent[1:61]}, {"separate"})

            post, _ = stand_in(batched_shift=0.2)
            status, out = run_main(["--candidate", "http://c", "--data", str(data),
                                    "--out", str(report)], post)
            self.assertEqual(status, 1)
            self.assertIn("[FAIL] noul: largest probability change between modes 0.200", out)
            self.assertIn("why it matters", out)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["verdict"], "FAIL")

    def test_against_the_current_model(self):
        def worse(state, qid, question, mode):
            p = calibrated(state, qid, question, mode)
            return p if int(state.split()[-1]) % 4 else p[::-1]

        candidate, _ = stand_in(model="candidate")
        current, _ = stand_in(answer=worse, model="current")

        def post(url, payload, api_key, timeout):
            return (current if url.startswith("http://cur") else candidate)(
                url, payload, api_key, timeout)

        with tempfile.TemporaryDirectory() as tmp:
            data = write(pathlib.Path(tmp) / "set.jsonl", labelled_rows(60))
            report = pathlib.Path(tmp) / "q.json"
            # Stand-ins answer in microseconds, so their latency ratio is noise.
            status, out = run_main(["--candidate", "http://cand", "--current", "http://cur",
                                    "--data", str(data), "--out", str(report),
                                    "--modes", "shared_prefix", "--max-latency-ratio", "1e9"],
                                   post)
            body = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(status, 0, out)
        roles = [(r["role"], r["models"]) for r in body["results"]]
        self.assertEqual(roles, [("candidate", ["candidate"]), ("current", ["current"])])
        self.assertGreater(body["comparison"]["all"]["candidate_only"], 0)
        self.assertIn("McNemar", out)
        checks = {c["check"] for c in body["checks"]}
        self.assertIn("max_accuracy_drop", checks)
        self.assertNotIn("max_mode_delta", checks)  # one mode: nothing to compare

    def test_the_temperature_its_gguf_carries_is_the_one_it_is_qualified_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = write(pathlib.Path(tmp) / "set.jsonl", labelled_rows(60))
            gguf = gguf_carrying(pathlib.Path(tmp) / "decide.gguf", 1.37)
            report = pathlib.Path(tmp) / "q.json"
            args = ["--candidate", "http://c", "--candidate-gguf", str(gguf), "--data",
                    str(data), "--modes", "shared_prefix", "--out", str(report)]
            # An engine that applies the GGUF's temperature by default.
            post, sent = stand_in(temperature=1.3700000047683716)
            status, out = run_main(args, post)
            self.assertEqual(status, 0, out)
            self.assertNotIn("temperature", sent[0]["eullm"])  # the server's own default
            self.assertIn("temperature applied 1.37; its GGUF carries 1.37", out)
            [result] = json.loads(report.read_text(encoding="utf-8"))["results"]
            self.assertAlmostEqual(result["temperature_gguf"], 1.37, places=6)
            self.assertEqual(result["temperatures_applied"], [1.3700000047683716])
            # One that does not read the key applies 1.
            post, _ = stand_in(temperature=1.0)
            status, out = run_main(args, post)
            self.assertEqual(status, 1)
            self.assertIn("[FAIL] all: temperature applied 1, the GGUF carries 1.37", out)
            # A file that is not a GGUF is refused before anything is asked.
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                run_main(args[:3] + [str(data)] + args[4:], post)

    def test_too_few_answers_fail_and_sources_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = write(pathlib.Path(tmp) / "set.jsonl", labelled_rows(60))
            post, sent = stand_in()
            status, out = run_main(["--candidate", "http://c", "--data", str(data),
                                    "--limit", "20", "--sources", "feedback",
                                    "--out", str(pathlib.Path(tmp) / "q.json")], post)
        self.assertEqual(status, 1)
        self.assertIn("[FAIL] noul: 20 answers (at least 50)", out)
        self.assertTrue(all(list(p["questions"]) == ["is_urgent"] for p in sent))


class HttpTest(unittest.TestCase):
    """The same through a real socket: the client's HTTP is the standard
    library's, and an error body must survive the trip."""

    def test_over_http(self):
        post, _ = stand_in(refuse=("severity",))

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    status, body = 200, post(self.path, payload, None, 10)
                except ServerError as e:
                    status, body = e.code, json.loads(e.detail)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            item = qf_data.item_from_row(labelled_rows(2)[1], 1)
            answers, ms, info, refusals = qualify.decide(qualify.Server("c", url), item, "separate")
        finally:
            server.shutdown()
            server.server_close()
        self.assertIsNone(answers["severity"])
        self.assertAlmostEqual(answers["team"]["p"][1], 0.95)
        self.assertIn("too many options", refusals["severity"])
        self.assertGreater(ms, 0)


if __name__ == "__main__":
    unittest.main()
