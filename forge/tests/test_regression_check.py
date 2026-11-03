"""regression_check.py: frozen general sets asked of a served model, graded by ReflexBench."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "regression_check.py"


def _mod():
    spec = importlib.util.spec_from_file_location("regression_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Engine(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = body["messages"][-1]["content"]
        reply = "The answer is 42." if "6 x 7" in q else "Answer: B"
        lines = [{"message": {"content": reply}, "done": False}, {"done": True}]
        data = "".join(json.dumps(x) + "\n" for x in lines).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_items_are_graded_by_number_and_letter(tmp_path):
    mod = _mod()
    # As `campaign.py prefetch` freezes a set.
    gsm = tmp_path / "gsm8k.jsonl"
    gsm.write_text("".join(json.dumps(r) + "\n" for r in [
        {"id": "a", "answer": "42", "grader": "number",
         "messages": [{"role": "user", "content": "How much is 6 x 7?"}]},
        {"id": "b", "answer": "40", "grader": "number",
         "messages": [{"role": "user", "content": "How much is 6 x 7?"}]},
    ]))
    arc = tmp_path / "arc-challenge.jsonl"
    arc.write_text(json.dumps({"id": "c", "answer": "B", "grader": "letter",
                               "messages": [{"role": "user", "content": "A or B?"}]}) + "\n")
    srv = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        g = mod.ab_data.from_jsonl(gsm)
        right = mod.ask_all(url, "m.gguf", sorted(g.items, key=lambda i: i.id), 2, 64, 30)
        assert right == [True, False]
        assert mod.ask_all(url, "m.gguf", mod.ab_data.from_jsonl(arc).items, 2, 64, 30) == [True]
        # A server that is not there costs the items, not the run.
        assert mod.ask_all("http://127.0.0.1:9", "m.gguf", g.items, 2, 64, 5) == [False, False]
    finally:
        srv.shutdown()
