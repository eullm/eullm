"""The GGUF exam path against a stand-in llama-server that speaks its protocol.

The stand-in is a real process on a real port, started by `LlamaServer` the
way llama-server is: it answers /health with 503 while "loading" and then
200, and /completion with the reply fields llama-server documents. What is
pinned is everything around the model: the prompt goes as token ids, greedy
and uncached; answers come back in order; "ended its turn" follows the end
token as on the bf16 path; a server that dies or refuses says why.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from eullm_forge.eval.gguf import LlamaServer, generate_answers_gguf

END = 99

FAKE_SERVER = textwrap.dedent(r'''
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    if os.environ.get("FAKE_DIE"):
        print("error: unknown model architecture: 'boom'", flush=True)
        sys.exit(1)
    record = os.environ["FAKE_RECORD"]
    with open(record + ".argv", "w") as f:
        json.dump(args, f)
    ready_at = time.monotonic() + 1.0

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if time.monotonic() < ready_at:
                self._send(503, {"error": {"code": 503, "message": "Loading model"}})
            else:
                self._send(200, {"status": "ok"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(record, "a") as f:
                f.write(json.dumps(body) + "\n")
            ids = body["prompt"]
            if len(ids) > 50:
                self._send(400, {"error": {"message": "exceeds the available context size"}})
                return
            # the answer is the prompt reversed; prompts starting with 7 stop on the
            # end token, the others run out of budget
            out = list(reversed(ids))[: body["n_predict"]]
            stop = "eos" if ids[0] == 7 else "limit"
            tokens = out + ([99] if stop == "eos" else [])
            time.sleep(0.05 * (len(ids) % 3))      # replies come back out of order
            self._send(200, {"content": "ignored", "tokens": tokens, "stop_type": stop,
                             "truncated": False})

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


class FakeTok:
    """Words <-> ints: "7 1 2" encodes to [7, 1, 2]; decode skips the end id."""

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [int(w) for w in text.split()]}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids if not (skip_special_tokens and i == END))


@pytest.fixture
def server_bin(tmp_path, monkeypatch):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    binary = tmp_path / "llama-server"
    binary.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    binary.chmod(0o755)
    gguf = tmp_path / "m-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF")
    record = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    return binary, gguf, record


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shebang")


def test_answers_come_back_in_order_with_their_endings(server_bin, tmp_path):
    binary, gguf, record = server_bin
    prompts = ["7 1 2", "3 4", "7 5", "8 6 6 6"]
    with LlamaServer(gguf, binary=str(binary), parallel=3, ctx_per_slot=1000,
                     log_path=tmp_path / "server.log") as server:
        out = generate_answers_gguf(server, FakeTok(), prompts, max_new_tokens=3,
                                    end_ids=[END], parallel=3)
    assert out == [("2 1 7", True), ("4 3", False), ("5 7", True), ("6 6 6", False)]
    sent = [json.loads(line) for line in record.read_text().splitlines()]
    assert sorted(r["prompt"] for r in sent) == sorted(
        [[7, 1, 2], [3, 4], [7, 5], [8, 6, 6, 6]])                # token ids, as tokenized
    assert all(r["temperature"] == 0 and r["cache_prompt"] is False and r["n_predict"] == 3
               and r["return_tokens"] is True for r in sent)
    argv = json.loads(Path(str(record) + ".argv").read_text(encoding="utf-8"))
    assert argv[argv.index("-np") + 1] == "3" and argv[argv.index("-c") + 1] == "3000"
    assert argv[argv.index("-m") + 1] == str(gguf)


def test_sampling_sends_the_temperature_and_one_seed_per_prompt(server_bin, tmp_path):
    binary, gguf, record = server_bin
    with LlamaServer(gguf, binary=str(binary), parallel=2, ctx_per_slot=1000,
                     log_path=tmp_path / "server.log") as server:
        generate_answers_gguf(server, FakeTok(), ["7 1", "7 2"], max_new_tokens=3,
                              end_ids=[END], parallel=2, temperature=0.8, seeds=[11, 12])
    sent = sorted((json.loads(line) for line in record.read_text().splitlines()),
                  key=lambda r: r["seed"])
    assert [r["seed"] for r in sent] == [11, 12]
    assert all(r["temperature"] == 0.8 and r["top_k"] == 0 and r["top_p"] == 1.0
               for r in sent)


def test_the_server_is_stopped_after_the_run(server_bin, tmp_path):
    binary, gguf, _ = server_bin
    with LlamaServer(gguf, binary=str(binary), log_path=tmp_path / "s.log") as server:
        proc = server.proc
    assert proc.poll() is not None


def test_a_server_that_dies_while_loading_says_why(server_bin, tmp_path, monkeypatch):
    binary, gguf, _ = server_bin
    monkeypatch.setenv("FAKE_DIE", "1")
    with pytest.raises(RuntimeError, match="unknown model architecture"):
        with LlamaServer(gguf, binary=str(binary), log_path=tmp_path / "s.log"):
            pass


def test_a_prompt_too_long_for_the_slot_is_an_error_not_an_empty_answer(server_bin, tmp_path):
    binary, gguf, _ = server_bin
    long_prompt = " ".join(["1"] * 60)
    with LlamaServer(gguf, binary=str(binary), log_path=tmp_path / "s.log") as server:
        with pytest.raises(RuntimeError, match="60 tokens.*400"):
            generate_answers_gguf(server, FakeTok(), [long_prompt], max_new_tokens=3,
                                  end_ids=[END])


def test_missing_binary_or_gguf_fail_before_starting(tmp_path):
    gguf = tmp_path / "m.gguf"
    with pytest.raises(FileNotFoundError, match="llama-server"):
        LlamaServer(gguf, binary=str(tmp_path / "nope")).__enter__()
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    with pytest.raises(FileNotFoundError, match="no GGUF"):
        LlamaServer(gguf, binary=str(binary)).__enter__()


def test_legal_eval_answers_the_exam_with_the_gguf(server_bin, tmp_path):
    """legal_eval.py --gguf end to end: prompts from the HF tokenizer, answers on file."""
    import subprocess

    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"[UNK]": 0, "<|im_end|>": END, **{w: i for i, w in enumerate(
        ["<|im_start|>user", "<|im_start|>assistant", "Che", "cosa", "prevede"], start=7)}}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>")
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m.role }} {{ m.content }} "
                         "<|im_end|> {% endfor %}<|im_start|>assistant")
    model_dir = tmp_path / "merged"
    tok.save_pretrained(model_dir)
    items = tmp_path / "exam-dev.jsonl"
    items.write_text("".join(json.dumps({"id": f"q{i}", "domain": "legal", "lang": "it",
                                         "question": q}) + "\n"
                             for i, q in enumerate(["Che cosa prevede", "Che cosa"])))
    binary, gguf, record = server_bin
    answers = tmp_path / "answers-x-q4.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts" / "legal_eval.py"
    r = subprocess.run([sys.executable, str(script), str(model_dir), "--items", str(items),
                        "--gguf", str(gguf), "--llama-server", str(binary), "--quiet",
                        "--answers", str(answers), "--label", "x-q4"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [json.loads(line) for line in answers.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["q0", "q1"]
    assert all(row["answer"] for row in rows)
    sent = [json.loads(line)["prompt"] for line in record.read_text().splitlines()]
    assert sorted(sent) == sorted([tok(tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True),
        add_special_tokens=False)["input_ids"] for q in ["Che cosa prevede", "Che cosa"]])
    assert "2/2 ended their turn" in r.stdout
    assert (tmp_path / "answers-x-q4.server.log").exists()
