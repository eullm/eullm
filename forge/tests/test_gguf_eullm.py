"""The exam through EULLM Engine: the server's command and the answers."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from eullm_forge.eval.gguf import EullmServer, generate_answers_eullm


def test_the_server_serves_the_gguf_with_one_slot_a_question(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    s = EullmServer(gguf, binary="/bin/eullm", parallel=8, ctx_per_slot=8192)
    s.port = 18123
    cmd = s.command()
    assert cmd[:2] == ["/bin/eullm", "serve"]
    assert cmd[cmd.index("--default-model") + 1] == str(gguf.resolve())
    assert cmd[cmd.index("--batch-size") + 1] == "8"
    assert cmd[cmd.index("--ctx-size") + 1] == str(8 * 8192)


def _fake_engine(tmp_path, help_text):
    exe = tmp_path / "eullm"
    exe.write_text(f"#!/bin/sh\necho '{help_text}'\n")
    exe.chmod(0o755)
    return str(exe)


def test_two_servers_on_a_node_open_no_chat_ui(tmp_path):
    # 0.8.5 opens the chat UI on port 11435 for every `serve`: two on a node clash.
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    new = EullmServer(gguf, binary=_fake_engine(tmp_path, "--port --no-ui --ui-port"))
    assert "--no-ui" in new.command()
    (tmp_path / "old").mkdir()
    old = EullmServer(gguf, binary=_fake_engine(tmp_path / "old", "--port --batch-size"))
    assert "--no-ui" not in old.command()


class _Engine(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        stop = "lungo" not in body["prompt"]
        reply = {"response": f" risposta a {body['prompt'][-8:]} ",
                 "done_reason": "stop" if stop else "length"}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_answers_come_back_in_order_greedy_and_raw(tmp_path):
    httpd = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        server = EullmServer(tmp_path / "m.gguf", binary="/bin/eullm")
        server.port = httpd.server_address[1]
        prompts = [f"<s>[INST]domanda {i}[/INST]" for i in range(5)] + ["[INST]lungo[/INST]"]
        out = generate_answers_eullm(server, prompts, max_new_tokens=400, parallel=3)
    finally:
        httpd.shutdown()
    assert [a for a, _ in out[:5]] == [f"risposta a {p[-8:]}" for p in prompts[:5]]
    assert [e for _, e in out] == [True] * 5 + [False]
    body = _Engine.seen[0]
    assert body["raw"] is True and body["stream"] is False
    assert body["options"]["temperature"] == 0.0 and body["options"]["repeat_penalty"] == 1.0
    assert body["options"]["num_predict"] == 400
