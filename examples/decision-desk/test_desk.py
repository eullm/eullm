"""The Decision Desk against a fake EuLLM server: the four steps in order, the
rules, the drafting skipped when it should be, the page's two endpoints.

The fake answers `/v1/systemone` with the label each built-in email carries,
so these tests check the desk's plumbing, not a model: `desk.py --check`
against a real server is what measures the model.

Standard library only, run by CI's examples job:

    python -m unittest discover -s examples/decision-desk -v
"""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


desk = load("desk")

BY_SUBJECT = {s["subject"]: s for s in desk.SAMPLES}


def fake_answers(state):
    """What a decision model that is always right would answer."""
    sample = BY_SUBJECT.get(state["subject"]) if isinstance(state, dict) else None
    expect = sample["expect"] if sample else {"team": "billing", "action": "reply"}
    team = expect["team"] or "billing"
    teams = {t: 0.02 for t in desk.TEAMS}
    teams[team] = 1.0 - 0.02 * (len(teams) - 1)
    kind = {"legitimate": 0.9, "phishing": 0.05, "spam": 0.05}
    if expect["action"] in ("quarantine", "spam"):
        name = "phishing" if expect["action"] == "quarantine" else "spam"
        kind = {"legitimate": 0.04, "phishing": 0.03, "spam": 0.03}
        kind[name] = 0.93
    health = 0.9 if sample and sample["id"] == "de-health" else 0.05
    angry = sample is not None and sample["id"] == "fr-returns"
    return {
        "team": {
            "type": "choice",
            "choice": team,
            "probabilities": teams,
            "confidence": 0.9,
        },
        "kind": {
            "type": "choice",
            "choice": max(kind, key=kind.get),
            "probabilities": kind,
            "confidence": 0.8,
        },
        "urgent": {"type": "noul", "noul": 0.8 if angry else 0.2},
        "mood": {
            "type": "score",
            "score": 1.9 if angry else 0.2,
            "legend": {"0": "calm", "1": "dissatisfied", "2": "angry"},
            "probabilities": {"0": 0.05, "1": 0.0, "2": 0.95}
            if angry
            else {"0": 0.8, "1": 0.2, "2": 0.0},
            "confidence": 0.9,
        },
        "health": {"type": "noul", "noul": health},
    }


class FakeEuLLM:
    """Just enough of `eullm serve` for the desk: version, ps, embeddings,
    systemone and a streamed chat, with the requests it got."""

    DRAFT = ("Hallo", " Jonas", ", wir", " prüfen", " das [1].")

    def __init__(self, audit_file=None):
        self.requests = []
        self.audit_file = audit_file
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.requests.append(("GET", self.path, None))
                if self.path == "/api/version":
                    self.reply({"version": "0.7.90"})
                elif self.path == "/api/ps":
                    self.reply(
                        {
                            "models": [
                                {
                                    "name": "gemma-4-e4b",
                                    "eullm": {"slot": "generation"},
                                },
                                {
                                    "name": "qwen3-embedding-0.6b-gguf-q8_0",
                                    "eullm": {"slot": "embedding"},
                                },
                                {"name": "jev-style-2b", "eullm": {"slot": "decision"}},
                            ]
                        }
                    )
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(("POST", self.path, body))
                if self.path == "/v1/embeddings":
                    # A vector per text from its letters: close texts, close vectors.
                    data = []
                    for i, text in enumerate(body["input"]):
                        v = [0.0] * 26
                        for c in text.lower():
                            if "a" <= c <= "z":
                                v[ord(c) - 97] += 1
                        data.append({"index": i, "embedding": v})
                    self.reply({"data": data})
                elif self.path == "/v1/systemone":
                    fake.audit(
                        {
                            "request_type": "systemone",
                            "model": "jev-style-2b",
                            "decision": {
                                "state_sha256": "ab" * 32,
                                "answers": [
                                    {
                                        "id": "team",
                                        "answer": "billing",
                                        "logprobs": [0.1],
                                    }
                                ],
                            },
                        }
                    )
                    self.reply(
                        {
                            "model": "jev-style-2b",
                            "answers": fake_answers(body["state"]),
                            "timing": {"total_ms": 42.0},
                            "eullm": {"audit_id": "b1149e83"},
                        }
                    )
                elif self.path == "/v1/chat/completions":
                    assert body["stream"] is True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for token in fake.DRAFT[: body.get("max_tokens", 99)]:
                        chunk = {"choices": [{"index": 0, "delta": {"content": token}}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    last = {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": len(fake.DRAFT)},
                    }
                    self.wfile.write(
                        f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode()
                    )
                    fake.audit(
                        {"request_type": "chat.completions", "model": "gemma-4-e4b"}
                    )
                else:
                    self.send_error(404)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def audit(self, entry):
        if self.audit_file:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": "x", "timestamp": "t", **entry}) + "\n")

    def posted(self, path):
        return [
            body for method, p, body in self.requests if method == "POST" and p == path
        ]


def options(url, audit_file=None, **over):
    args = SimpleNamespace(
        url=url,
        api_key=None,
        decision_model=None,
        embedding_model="qwen3-embedding-0.6b-gguf-q8_0",
        chat_model="gemma-4-e4b",
        threshold=0.5,
        quarantine=0.7,
        min_team=0.4,
        legal=0.3,
        passages=3,
        max_chars=4000,
        max_tokens=320,
        audit_file=audit_file,
        timeout=10.0,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def sample(sid):
    return next(s for s in desk.SAMPLES if s["id"] == sid)


def stage(events, name):
    return next(e for e in events if e["stage"] == name)


def open_direct(request):
    # The desk is on this machine: never through a proxy in the environment.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=10
    )


def serve_desk(d):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), desk.handler(d))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class WithFake(unittest.TestCase):
    """A fake EuLLM with an empty audit file of its own, for every test."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.audit_file = os.path.join(folder.name, "audit.jsonl")
        Path(self.audit_file).write_text("", encoding="utf-8")
        self.fake = FakeEuLLM(self.audit_file)
        self.addCleanup(self.fake.server.server_close)
        self.addCleanup(self.fake.server.shutdown)

    def run_sample(self, sid, **over):
        d = desk.Desk(options(self.fake.url, **{"audit_file": self.audit_file, **over}))
        events = []
        d.process(sample(sid), events.append)
        return events


class TheFourSteps(WithFake):
    def test_a_reply_goes_through_the_four_steps_in_order(self):
        events = self.run_sample("de-billing")
        stages = [e["stage"] for e in events]
        self.assertEqual(stages[:5], ["start", "find", "decide", "rule", "write"])
        self.assertEqual(stages[-3:], ["written", "audit", "done"])
        tokens = "".join(e["text"] for e in events if e["stage"] == "token")
        self.assertEqual(tokens, "".join(self.fake.DRAFT))

        rule = stage(events, "rule")
        self.assertEqual(
            (rule["action"], rule["team"], rule["priority"]),
            ("reply", "billing", "normal"),
        )
        self.assertEqual(stage(events, "written")["tokens"], len(self.fake.DRAFT))

        decide = stage(events, "decide")
        self.assertEqual((decide["ms"], decide["audit_id"]), (42.0, "b1149e83"))
        self.assertEqual(
            [q["id"] for q in decide["questions"]],
            ["team", "kind", "urgent", "mood", "health"],
        )
        team = decide["questions"][0]["options"]
        self.assertEqual(team[0]["name"], "billing")
        self.assertTrue(team[0]["chosen"])

    def test_the_draft_is_asked_with_the_passages_found_and_in_the_customers_language(
        self,
    ):
        events = self.run_sample("de-billing")
        passages = stage(events, "find")["passages"]
        self.assertEqual(len(passages), 3)
        (chat,) = self.fake.posted("/v1/chat/completions")
        system, user = chat["messages"]
        self.assertIn("same language as the customer's email", system["content"])
        self.assertIn("Billing team", system["content"])
        for n, p in enumerate(passages, 1):
            self.assertIn(f"[{n}] {p['title']}", user["content"])
        self.assertEqual(chat["model"], "gemma-4-e4b")
        self.assertIs(chat["think"], False)

    def test_an_angry_customer_gets_high_priority_and_the_prompt_says_so(self):
        events = self.run_sample("fr-returns")
        self.assertEqual(stage(events, "rule")["priority"], "high")
        (chat,) = self.fake.posted("/v1/chat/completions")
        self.assertIn("high priority", chat["messages"][0]["content"])


class WhatIsNotDrafted(WithFake):
    def test_phishing_is_quarantined_and_the_chat_model_is_never_called(self):
        events = self.run_sample("en-phishing")
        rule = stage(events, "rule")
        self.assertEqual(rule["action"], "quarantine")
        self.assertIsNone(rule["team"])
        self.assertIn("phishing", stage(events, "skip")["why"])
        self.assertFalse(self.fake.posted("/v1/chat/completions"))
        self.assertNotIn("write", [e["stage"] for e in events])

    def test_what_a_person_or_nobody_answers_is_not_drafted(self):
        for sid, action in [
            ("fr-spam", "spam"),
            ("de-health", "person"),
            ("it-legal", "person"),
        ]:
            with self.subTest(sid):
                events = self.run_sample(sid)
                self.assertEqual(stage(events, "rule")["action"], action)
                self.assertFalse(self.fake.posted("/v1/chat/completions"))

    def test_a_letter_that_may_be_legal_goes_to_a_person_even_when_legal_is_second(
        self,
    ):
        # As the Jev-Style 2B answered the lawyer's notice: shipping 0.49, legal 0.45.
        answers = fake_answers({"subject": sample("it-legal")["subject"]})
        teams = {t: 0.0 for t in desk.TEAMS}
        teams.update(shipping=0.49, legal=0.45, returns=0.06)
        answers["team"].update(choice="shipping", probabilities=teams)
        verdict = desk.policy(answers, [], options("http://x"))
        self.assertEqual((verdict["action"], verdict["team"]), ("person", "legal"))
        self.assertIn("legal 0.45", verdict["rule"])
        teams.update(shipping=0.80, legal=0.14)
        self.assertEqual(
            desk.policy(answers, [], options("http://x"))["action"], "reply"
        )

    def test_a_team_the_model_is_unsure_of_goes_to_a_person(self):
        answers = fake_answers({"subject": sample("de-billing")["subject"]})
        answers["team"]["probabilities"]["billing"] = 0.3
        verdict = desk.policy(answers, [], options("http://x"))
        self.assertEqual(verdict["action"], "person")
        self.assertFalse(verdict["draft"])
        self.assertIn("unsure", verdict["rule"])


class PersonalData(WithFake):
    def test_an_iban_is_found_by_code_not_by_the_model(self):
        events = self.run_sample("it-iban")
        self.assertEqual(stage(events, "rule")["flags"], ["IBAN"])
        self.assertEqual(
            desk.personal_data("IBAN IT60X0542811101000000123457"), []
        )  # check digits
        self.assertEqual(
            desk.personal_data("carta 4111 1111 1111 1111"), ["card number"]
        )
        self.assertEqual(desk.personal_data("CF RSSMRA85T10A562S"), ["tax code"])


class Audit(WithFake):
    def test_the_audit_lines_of_one_email_are_read_back_trimmed(self):
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "old", "request_type": "chat"}) + "\n")
        audit = stage(self.run_sample("de-billing"), "audit")
        self.assertEqual(
            [line["request_type"] for line in audit["lines"]],
            ["systemone", "chat.completions"],
        )
        self.assertEqual(
            audit["lines"][0]["decision"],
            {"state_sha256": "ab" * 32, "answers": {"team": "billing"}},
        )

    def test_no_audit_file_means_no_audit_lines(self):
        audit = stage(self.run_sample("de-billing", audit_file=None), "audit")
        self.assertEqual(audit, {"stage": "audit", "file": None, "lines": []})


class WarmUp(WithFake):
    def test_the_named_decision_model_and_the_chat_model_load_before_the_first_email(
        self,
    ):
        d = desk.Desk(
            options(
                self.fake.url, decision_model="jev-style-2b-decision-v3-gguf-q4_k_m"
            )
        )
        d.warm_up()
        self.assertEqual(
            (d.warmup["index"], d.warmup["decision"], d.warmup["chat"]),
            ("ready", "ready", "ready"),
        )
        (decision,) = self.fake.posted("/v1/systemone")
        self.assertEqual(decision["model"], "jev-style-2b-decision-v3-gguf-q4_k_m")
        (chat,) = self.fake.posted("/v1/chat/completions")
        self.assertEqual(chat["max_tokens"], 1)

    def test_without_a_named_decision_model_the_loaded_one_is_left_alone(self):
        d = desk.Desk(options(self.fake.url))
        d.warm_up()
        self.assertEqual(d.warmup["decision"], "waiting")
        self.assertFalse(self.fake.posted("/v1/systemone"))


class Check(WithFake):
    def test_check_measures_every_built_in_email(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(desk.check(desk.Desk(options(self.fake.url))))
        n = len(desk.SAMPLES)
        self.assertIn(f"action right {n}/{n}, team right {n}/{n}", out.getvalue())
        self.assertFalse(self.fake.posted("/v1/chat/completions"))

    def test_every_built_in_email_expects_a_team_the_desk_has(self):
        for s in desk.SAMPLES:
            self.assertIn(s["expect"]["team"], (None, *desk.TEAMS))
            self.assertIn(
                s["expect"]["action"], ("reply", "person", "quarantine", "spam")
            )


class ThePage(WithFake):
    def setUp(self):
        super().setUp()
        server, self.page = serve_desk(
            desk.Desk(options(self.fake.url, self.audit_file))
        )
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

    def test_the_page_and_its_status(self):
        with open_direct(self.page + "/") as r:
            self.assertIn(b"Decision Desk", r.read())
        with open_direct(self.page + "/api/status") as r:
            status = json.load(r)
        self.assertEqual(status["server"], "0.7.90")
        self.assertIsNone(status["error"])
        self.assertEqual(
            {k: m["loaded"] for k, m in status["models"].items()},
            {"embedding": True, "decision": True, "chat": True},
        )
        self.assertEqual(status["models"]["decision"]["name"], "jev-style-2b")
        self.assertEqual(len(status["samples"]), len(desk.SAMPLES))
        self.assertNotIn("expect", status["samples"][0])

    def test_process_streams_one_json_line_per_step(self):
        body = json.dumps(
            {
                "email": {
                    "from": "a@example.eu",
                    "subject": "Hi",
                    "body": "Where is my bike?",
                }
            }
        )
        request = urllib.request.Request(
            self.page + "/api/process",
            data=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        with open_direct(request) as r:
            self.assertEqual(r.headers["Content-Type"], "application/x-ndjson")
            stages = [json.loads(line)["stage"] for line in r.read().splitlines()]
        self.assertEqual((stages[0], stages[-1]), ("start", "done"))

    def test_process_refuses_an_email_without_text(self):
        request = urllib.request.Request(
            self.page + "/api/process",
            data=b'{"email": {"subject": "x", "body": " "}}',
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as e:
            open_direct(request)
        self.assertEqual(e.exception.code, 400)


class NoServer(unittest.TestCase):
    def test_an_unreachable_server_is_an_error_line_not_a_traceback(self):
        server, page = serve_desk(
            desk.Desk(options("http://127.0.0.1:9", None, timeout=2.0))
        )
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        request = urllib.request.Request(
            page + "/api/process",
            data=b'{"sample": "de-billing"}',
            headers={"Content-Type": "application/json"},
        )
        with open_direct(request) as r:
            last = json.loads(r.read().splitlines()[-1])
        self.assertEqual(last["stage"], "error")
        self.assertIn("eullm serve", last["message"])


class VideoTimeline(unittest.TestCase):
    """desk_video.py's clock; the rendering itself needs a browser and
    ffmpeg, which CI does not have."""

    def test_each_email_starts_a_hold_after_the_one_before_ended(self):
        video = load("desk_video")
        emails = [
            {
                "index": 0,
                "events": [
                    {"dt": 0.0, "event": {"stage": "start"}},
                    {"dt": 1.2, "event": {"stage": "done"}},
                ],
            },
            {
                "index": 1,
                "events": [
                    {"dt": 0.0, "event": {"stage": "start"}},
                    {"dt": 0.3, "event": {"stage": "done"}},
                ],
            },
        ]
        steps, seconds = video.timeline(emails, 2.5)
        intro = video.INTRO
        self.assertEqual(
            [(s["index"], s["t"]) for s in steps],
            [(0, intro), (0, intro + 1.2), (1, intro + 3.7), (1, intro + 4.0)],
        )
        self.assertAlmostEqual(seconds, intro + 4.0 + video.OUTRO)


class StartScript(unittest.TestCase):
    def test_start_sh_is_valid_posix_sh(self):
        subprocess.run(["sh", "-n", str(HERE / "start.sh")], check=True)


if __name__ == "__main__":
    unittest.main()
