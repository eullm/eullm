#!/usr/bin/env python3
"""A stand-in for the eullm binary, for the tests: `--version`, `list`,
`serve --port N`, whose server answers /api/version, /api/generate and
/api/chat the way the engine does (streamed NDJSON, Ollama's fields), and
`serve --decision-model M`, whose /v1/systemone answers one decision at a
time as the engine's single decision worker does (a model starting with
"missing" stops the server at startup, as a model not in the store does);
called with `-m GGUF` it is llama-server (/health, OpenAI's SSE endpoints),
and as `serve` without `--port` it is Ollama (OLLAMA_HOST). And
`finetune MODEL --output O --report R`, which writes both the way the engine
does: models starting with "huge" are refused as too large for the free
memory, those starting with "notf32" as quantized, and a model file that does
not exist as no model.

Models it knows: every id except those starting with "missing", which get a
404 like an id the store does not have, and those starting with "huge",
refused the way --fit-strict refuses a model too large for the free VRAM.
Every answer ends with "Answer: 4", so a GSM8K-style item whose answer is 4
grades correct. FAKE_EULLM_DELAY_S sets the time per token, and
FAKE_EULLM_DIE_AFTER=N makes the server exit at once (code 134, as an abort)
on the request after the Nth /api/generate or /api/chat, as a server that
crashes in the middle of a soak.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DELAY = float(os.environ.get("FAKE_EULLM_DELAY_S", "0.001"))
DIE_AFTER = int(os.environ.get("FAKE_EULLM_DIE_AFTER", "0"))
SERVED = [0]
SERVED_LOCK = threading.Lock()
WORDS = ["Il", " mare", " era", " calmo", "."]


DECISION_LOCK = threading.Lock()


def decision(body):
    """Answers that depend on the state and the questions only, as a
    deterministic model's do."""
    seed = sum(map(ord, body.get("state", ""))) % 97
    answers = {}
    for k, (qid, q) in enumerate(body.get("questions", {}).items()):
        p = ((seed + k) % 10) / 10
        if q.get("type") == "choice":
            opts = list(q.get("criteria", {}))
            probs = {o: (0.7 if j == (seed + k) % len(opts) else 0.3 / (len(opts) - 1))
                     for j, o in enumerate(opts)}
            answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get),
                            "probabilities": probs}
        elif q.get("type") == "score":
            answers[qid] = {"type": "score", "score": 1 + p * 3}
        else:
            answers[qid] = {"type": "noul", "noul": p}
    return answers


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    flavour = "eullm"
    mtp = 0  # --mtp N: the answers report drafts, as the engine's do

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health" and self.flavour == "llama":
            self._json(200, {"status": "ok"})
        elif self.path == "/api/version" and self.flavour != "llama":
            self._json(200, {"version": "0.0.0-fake"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path == "/v1/systemone":
            started = time.time()
            with DECISION_LOCK:  # one worker: requests wait their turn
                decode_ms = DELAY * 1000 * (2 + len(body.get("questions", {})))
                time.sleep(decode_ms / 1000)
                answers = decision(body)
            self._json(200, {"model": "jev-fake", "answers": answers,
                             "eullm": {"request_ms": (time.time() - started) * 1000,
                                       "timings_ms": {"prefix": decode_ms / 2,
                                                      "questions": decode_ms / 2},
                                       "prompt_tokens": len(body.get("state", "")) // 4,
                                       "evaluated_tokens": len(body.get("state", "")) // 4,
                                       "mode": (body.get("eullm") or {}).get("mode")}})
            return
        if self.path in ("/v1/completions", "/v1/chat/completions"):
            self._openai(body)
            return
        model = body.get("model", "")
        if model.startswith("missing"):
            self._json(404, {"error": f"model '{model}' not found"})
            return
        if model.startswith("huge"):
            self._json(500, {"error": f"Failed to load model '{model}': --fit-strict: model "
                                      f"'{model}' does not fully fit in the currently free VRAM; "
                                      "not loading."})
            return
        if DIE_AFTER:
            with SERVED_LOCK:
                SERVED[0] += 1
                if SERVED[0] > DIE_AFTER:
                    os._exit(134)
        n = int((body.get("options") or {}).get("num_predict") or body.get("num_predict") or 16)
        n = min(n, 32)
        chat = self.path == "/api/chat"
        prompt = body.get("prompt") or json.dumps(body.get("messages"))
        pieces = [WORDS[i % len(WORDS)] for i in range(max(n - 4, 1))] + ["\nAnswer: 4"]
        if not body.get("stream", True):
            time.sleep(DELAY * len(pieces))
            self._json(200, {"model": model, "response": "".join(pieces), "done": True,
                             "eval_count": len(pieces), "prompt_eval_count": len(prompt) // 4,
                             "load_duration": 1000000})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = (json.dumps(obj) + "\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        started = time.time()
        for piece in pieces:
            time.sleep(DELAY)
            send({"model": model, "message": {"content": piece}} if chat
                 else {"model": model, "response": piece, "done": False})
        last = {"model": model, "done": True, "eval_count": len(pieces),
                "prompt_eval_count": len(prompt) // 4,
                "eval_duration": int((time.time() - started) * 1e9),
                "prompt_eval_duration": 0, "load_duration": 2000000}
        if Handler.mtp:
            # As the engine reports them under llama-server's names: here
            # the head drafts `mtp` tokens every other step and half are kept.
            last["draft_n"] = len(pieces) // 2 * Handler.mtp
            last["draft_n_accepted"] = last["draft_n"] // 2
        send(last)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def _openai(self, body):
    """llama-server's OpenAI endpoints, streamed as SSE, with its usage and
    timings in the last chunk."""
    n = min(int(body.get("max_tokens") or 16), 32)
    pieces = [WORDS[i % len(WORDS)] for i in range(max(n - 4, 1))] + ["\nAnswer: 4"]
    chat = self.path.endswith("chat/completions")
    prompt = body.get("prompt") or json.dumps(body.get("messages"))
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Transfer-Encoding", "chunked")
    self.end_headers()

    def send(obj):
        data = f"data: {json.dumps(obj) if isinstance(obj, dict) else obj}\n\n".encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    started = time.time()
    for piece in pieces:
        time.sleep(DELAY)
        send({"choices": [{"delta": {"content": piece}} if chat else {"text": piece}]})
    send({"choices": [], "usage": {"completion_tokens": len(pieces),
                                   "prompt_tokens": len(prompt) // 4},
          "timings": {"predicted_n": len(pieces), "prompt_n": len(prompt) // 4,
                      "predicted_ms": (time.time() - started) * 1000, "prompt_ms": 1.0}})
    send("[DONE]")
    self.wfile.write(b"0\r\n\r\n")
    self.wfile.flush()


Handler._openai = _openai


def flag(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def finetune(argv):
    model = argv[0]
    name = os.path.basename(model)
    if not os.path.exists(model):
        print(f"Error: {model} is neither a .gguf file nor a model in the store", file=sys.stderr)
        return 1
    if name.startswith("huge"):
        print("Error: the run is estimated at 812.0 GiB and 61.2 GiB is free. Shorten --ctx, "
              "train fewer tensors (--train-tensors), use --optimizer sgd, or pass --force to "
              "try anyway.", file=sys.stderr)
        return 1
    if name.startswith("notf32"):
        print(f"Error: {model} is not an F32 model: 197 of 311 tensors are not F32",
              file=sys.stderr)
        return 1
    epochs = int(flag(argv, "--epochs", "2"))
    n_ctx = int(flag(argv, "--ctx", "512"))
    lr = float(flag(argv, "--lr", "1e-6"))

    def pass_(loss, tokens):
        return {"tokens": tokens, "loss": loss, "loss_unc": 0.01, "perplexity": 2.718 ** loss,
                "accuracy": 0.5, "accuracy_unc": 0.01, "seconds": tokens / 4000}

    per_epoch = []
    for e in range(epochs):
        time.sleep(DELAY * 10)
        per_epoch.append({"epoch": e, "lr": lr, "train": pass_(2.0 - 0.3 * e, 8 * n_ctx),
                          "train_tok_s": 4000.0 + e,
                          "validation": pass_(1.9 - 0.3 * (e + 1), n_ctx)})
    report = {
        "schema": "eullm.finetune/1", "engine": "0.0.0-fake", "backend": "cpu",
        "model": model, "data": flag(argv, "--data"), "output": flag(argv, "--output"),
        "arch": "qwen3", "params": 596049920, "trainable_params": 440467456,
        "trainable_tensors": 310,
        "train_tensors": [t for t in (flag(argv, "--train-tensors") or "").split(",") if t],
        "optimizer": flag(argv, "--optimizer", "adamw"), "lr": lr, "epochs": epochs,
        "n_ctx": n_ctx, "baseline": pass_(1.9, n_ctx), "per_epoch": per_epoch,
        "memory_estimate": {"total": 9 * 2**30},
        "dry_run": False,
    }
    with open(flag(argv, "--output"), "wb") as f:
        f.write(b"GGUF")
    with open(flag(argv, "--report"), "w") as f:
        json.dump(report, f)
    print("FINETUNE_RESULT " + json.dumps(report))
    return 0


# What `eullm list` prints with two models installed (engine/src/main.rs,
# cmd_list): the runner reads the table, not the words around it.
LISTING = """Models in /tmp/fake-store [EULLM_MODELS_DIR]

NAME                         SIZE   VRAM STATUS           DESCRIPTION
qwen3-8b                   4.7 GB    6GB  ready            Qwen3 8B
qwen3-4b                   2.3 GB    4GB  ready            Qwen3 4B

Run one with: eullm run <NAME>   (e.g. eullm run qwen3-8b)"""


def main(argv):
    if "--version" in argv:
        print("eullm 0.0.0-fake (test)")
        return 0
    if argv and argv[0] == "list":
        print(LISTING)
        return 0
    if argv and argv[0] == "finetune":
        return finetune(argv[1:])
    if "-m" in argv:  # llama-server
        Handler.flavour = "llama"
        port = int(flag(argv, "--port"))
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    if argv and argv[0] == "serve" and "--help" in argv:
        # As 0.8.5 has it; FAKE_EULLM_NO_UI=0 plays an engine from before it.
        if os.environ.get("FAKE_EULLM_NO_UI", "1") == "1":
            print("      --no-ui  Do not serve the chat UI")
        return 0
    if argv and argv[0] == "serve":
        if "--no-ui" in argv and os.environ.get("FAKE_EULLM_NO_UI", "1") != "1":
            print("error: unexpected argument '--no-ui' found", file=sys.stderr)
            return 2
        Handler.mtp = int(flag(argv, "--mtp", "0"))
        model = flag(argv, "--decision-model")
        if model and model.startswith("missing"):
            print(f"Error: decision model '{model}' not found.", file=sys.stderr)
            return 1
        if "--port" in argv:
            port = int(flag(argv, "--port"))
        else:  # Ollama
            Handler.flavour = "ollama"
            port = int(os.environ["OLLAMA_HOST"].rsplit(":", 1)[1])
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
