"""`rag_enterprise_batch.sh`: one model after the other, unattended.

RAG Enterprise is stood in for by a small HTTP server answering what the
batch and `rag_enterprise_eval.py ask` call (/health, login, query), which
records the model .env named when it started. So the test runs the real
script and the real `ask` against the thing they wait for.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rag_enterprise_batch.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None or shutil.which("curl") is None,
    reason="needs POSIX bash and curl")

FAKE = r'''#!/usr/bin/env python3
import http.server, json, os, re, sys
env = open(".env").read()
model = re.search(r"^EULLM__MODEL_OVERRIDE=(.*)$", env, re.M).group(1)
extra = re.search(r"^EULLM__EXTRA_ARGS=(.*)$", env, re.M)
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"model": model, "extra": extra and extra.group(1)}) + "\n")
if "broken" in model:
    sys.exit(1)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_GET(self): self._send({"status": "ok"})
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/api/auth/login": self._send({"access_token": "t"})
        else: self._send({"answer": "Risposta di " + os.path.basename(model),
                          "sources": [{"filename": "Codice civile, art. 1 - X.txt"}]})
http.server.HTTPServer(("127.0.0.1", int(os.environ["FAKE_PORT"])), H).serve_forever()
'''


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_each_model_is_asked_in_turn_and_the_env_is_put_back(tmp_path):
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "i3k-rag-engine").write_text(FAKE, encoding="utf-8")
    (engine / "i3k-rag-engine").chmod(0o755)
    (engine / ".env").write_text("AUTH__JWT_SECRET=x\nEULLM__MODEL_OVERRIDE=/old.gguf\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    for name in ("a.gguf", "b.gguf", "broken.gguf"):
        (models / name).write_text("gguf", encoding="utf-8")
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({
        "id": f"norm-contenuto-codice_civile-{n}", "question": f"Che cosa prevede l'art. {n} c.c.?",
        "metadata": {"tipo": "contenuto", "code": "codice_civile", "articolo": str(n)}}) + "\n"
        for n in (1, 2)), encoding="utf-8")
    out = tmp_path / "out"
    port = _port()
    env = {**os.environ, "RAG_PASSWORD": "pw", "RAG_URL": f"http://127.0.0.1:{port}",
           "FAKE_PORT": str(port), "FAKE_LOG": str(tmp_path / "started.jsonl"),
           "START_TIMEOUT": "30"}

    proc = subprocess.run(
        ["bash", str(SCRIPT), str(engine), str(items), str(out),
         f"a={models / 'a.gguf'}", f"broken={models / 'broken.gguf'}",
         f"b={models / 'b.gguf'}|--moe-cache auto", f"missing={models / 'none.gguf'}"],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    started = [json.loads(ln) for ln in (tmp_path / "started.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(Path(s["model"]).name, s["extra"]) for s in started] == \
        [("a.gguf", None), ("broken.gguf", None), ("b.gguf", "--moe-cache auto")]
    rows = [json.loads(ln) for ln in (out / "answers-b-v022.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[0]["answer"] == "Risposta di b.gguf"
    assert len((out / "answers-a-v022.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert "broken: RAG Enterprise did not come up" in proc.stdout
    assert "missing: " in proc.stdout and "not found, skipped" in proc.stdout
    # the .env is the one it found, and a second run skips what is done
    assert (engine / ".env").read_text(encoding="utf-8") == \
        "AUTH__JWT_SECRET=x\nEULLM__MODEL_OVERRIDE=/old.gguf\n"
    again = subprocess.run(["bash", str(SCRIPT), str(engine), str(items), str(out),
                            f"a={models / 'a.gguf'}"],
                           env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert "a: already answered, skipped" in again.stdout
