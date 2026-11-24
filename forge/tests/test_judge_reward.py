"""The judge reward against a stand-in judge speaking the chat API.

The stand-in grades by a word in the answer, so the test knows what each
completion must score. Pinned: judged completions get the judge's grade on
the exam's scale, the others None (the verifiable reward's), the judge reads
the exam's grading prompt with the question, article and rubric, and an
unreadable reply is asked again once and then left unscored, not zeroed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from eullm_forge.rl import answer_reward
from eullm_forge.rl.judge_reward import JudgeReward


@pytest.fixture
def judge():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][0]["content"]
            seen.append(body)
            answer = prompt.split("Answer to grade:", 1)[-1]
            if "GIUSTA" in answer:
                reply = "GRADE: correct\nbecause"
            elif "META" in answer:
                reply = "grade: partial"
            elif "BOH" in answer:
                reply = "non saprei"
            else:
                reply = "Grade: wrong"
            data = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def test_judged_answers_get_the_judges_grade_and_the_others_none(judge):
    url, seen = judge
    reward = JudgeReward(url, parallel=3)
    completions = [[{"role": "assistant", "content": t}]
                   for t in ("risposta GIUSTA", "risposta META", "risposta sbagliata",
                             "Il termine è di 30 giorni")]
    tipo = ["contenuto", "contenuto", "contenuto", "termine"]
    question = ["Che cosa prevede l'art. 1385 del codice civile?"] * 3 + [""]
    reference = ["Testo integrale dell'articolo: caparra confirmatoria"] * 3 + [""]
    rubric = ["Valuta il contenuto essenziale."] * 3 + [""]
    assert reward(completions, tipo, question, reference, rubric) == [1.0, 0.5, 0.0, None]
    assert len(seen) == 3
    prompt = seen[0]["messages"][0]["content"]
    assert "art. 1385" in prompt and "caparra confirmatoria" in prompt
    assert "Valuta il contenuto essenziale." in prompt
    assert all(b["temperature"] == 0 for b in seen)
    # the verifiable reward leaves the judged ones alone
    assert answer_reward(completions, tipo, [[], [], [], ["30 giorni"]]) == [None, None, None, 1.0]


def test_an_unreadable_reply_is_asked_twice_then_left_unscored(judge):
    url, seen = judge
    reward = JudgeReward(url)
    out = reward(["BOH"], ["contenuto"], ["q"], ["ref"], [""])
    assert out == [None] and reward.unscored == 1 and len(seen) == 2


def test_an_unreachable_judge_leaves_the_answer_unscored():
    reward = JudgeReward("http://127.0.0.1:9", timeout=2)
    assert reward(["x"], ["contenuto"], ["q"], ["r"], [""]) == [None]
    assert reward.unscored == 1


@pytest.mark.parametrize("body", [
    {"choices": []},                        # slot exhaustion, restart, proxy strip
    {"choices": None},                      # version mismatch
    {"choices": [{"message": None}]},       # tool-call shape mismatch
])
def test_a_malformed_200_is_unscored_not_a_crash(body):
    """Empty or null choices raised IndexError/TypeError past the except that
    catches the other malformed replies, aborting the whole GRPO step instead
    of counting one unscored completion."""
    seen = []
    data = json.dumps(body).encode()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seen.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        reward = JudgeReward(f"http://127.0.0.1:{srv.server_address[1]}")
        assert reward(["x"], ["contenuto"], ["q"], ["r"], [""]) == [None]
        assert reward.unscored == 1 and len(seen) == 2
    finally:
        srv.shutdown()


def test_grpo_train_refuses_judged_prompts_without_a_judge(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "grpo_train.py"
    spec = importlib.util.spec_from_file_location("grpo_train", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"prompt": [], "tipo": "contenuto", "keywords": []}) + "\n",
                       encoding="utf-8")
    with pytest.raises(SystemExit, match="need a judge"):
        mod.main(["--model", "m", "--prompts", str(prompts), "--out", str(tmp_path / "o")])
