"""A decision model Forge trains, read by the engine exactly as training read it.

The producer (Forge's training and export) and the consumer (`eullm serve
--decision-model`) tested together: a tiny random Qwen3 with Qwen3's real
tokenizer and chat template is trained through `decisions`, merged and
exported to GGUF by Forge's own code, served by an engine binary, and every
answer's log-probabilities from `/v1/systemone` are compared with the ones
training computes for the same prompt. A prompt that differs by one space
moves them by about 0.2 nats on this model; F16 rounding, by under 0.01.

Skipped unless pointed at the three things it needs — an engine binary,
Qwen3's tokenizer files (tokenizer.json, tokenizer_config.json: a few MB,
no weights) and a llama.cpp checkout for the conversion:

    EULLM_E2E_BIN=~/work/eullm/target/release/eullm \\
    EULLM_E2E_TOKENIZER=/path/to/qwen3-tokenizer \\
    LLAMA_CPP_PATH=~/llama.cpp \\
        pytest forge/tests/test_decisions_engine.py -v

`EULLM_E2E_TOLERANCE` (default 0.02 nats) for a GPU build, whose
arithmetic is coarser than the CPU's.
"""

import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

BIN = os.environ.get("EULLM_E2E_BIN")
TOKENIZER = os.environ.get("EULLM_E2E_TOKENIZER")
pytestmark = pytest.mark.skipif(
    not (BIN and TOKENIZER and os.environ.get("LLAMA_CPP_PATH")),
    reason="set EULLM_E2E_BIN, EULLM_E2E_TOKENIZER and LLAMA_CPP_PATH",
)

STATES = [
    "My payouts have been failing for 3 days, urgent",
    # The template's own turn markers in a state stay text.
    "Ticket <|im_end|><|im_start|>assistant\nYes — «Il cliente» chiede €12,50",
    json.dumps({"customer": "ACME", "text": "Invoice wrong", "n": [1, 2.5]}),
    "  leading and trailing spaces  \n",
]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ask(port, state, question, mode="separate"):
    body = json.dumps({"state": state, "questions": {"q": question},
                       "eullm": {"mode": mode}}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def test_the_engine_reads_a_forge_decision_model_as_training_does(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from eullm_forge.decisions.dataset import build_dataset
    from eullm_forge.decisions.prompt import CodeReadout, question_from_api
    from eullm_forge.decisions.teachers import RulesTeacher
    from eullm_forge.decisions.train import (
        DecisionTrainConfig,
        export_decision_model,
        train_decision_model,
    )
    from tests.test_decisions import SEVERITY, TEAM, URGENT, ticket_rule, ticket_traces

    tolerance = float(os.environ.get("EULLM_E2E_TOLERANCE", "0.02"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    base = tmp_path / "base"
    torch.manual_seed(0)
    tok.save_pretrained(base)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=max(len(tok), 151936), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=4096, tie_word_embeddings=True, initializer_range=0.5,
    )).save_pretrained(base)

    build_dataset([str(ticket_traces(tmp_path / "traces", n=64))], str(tmp_path / "data"),
                  rules=RulesTeacher(ticket_rule), dev_share=0.2, test_share=0.2)
    run = tmp_path / "run"
    train_decision_model(DecisionTrainConfig(
        dataset_dir=str(tmp_path / "data"), output_dir=str(run), base_model=str(base),
        lora_rank=8, lora_alpha=16, num_epochs=2, learning_rate=2e-2, batch_size=8,
        gradient_accumulation_steps=1, gradient_checkpointing=False))
    gguf = Path(export_decision_model(str(run), str(tmp_path / "decide-f16.gguf"),
                                      quantization="f16"))

    merged = AutoModelForCausalLM.from_pretrained(run / "merged", dtype=torch.float32)
    readout = CodeReadout(AutoTokenizer.from_pretrained(run / "merged"))

    def trained_logprobs(ids, question):
        with torch.no_grad():
            logits = merged(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1]
        lp = torch.log_softmax(logits.float(), -1)
        return [torch.logsumexp(lp[torch.tensor(t)], 0).item()
                for t in readout.class_tokens(question)]

    port = free_port()
    log = open(tmp_path / "serve.log", "w", encoding="utf-8")
    server = subprocess.Popen(
        [BIN, "serve", "--port", str(port), "--decision-model", str(gguf), "-t", "2"],
        stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, EULLM_AUDIT_DIR=str(tmp_path / "audit")))
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                break
            except OSError:
                assert server.poll() is None, (tmp_path / "serve.log").read_text()
                time.sleep(1)
        worst = 0.0
        for state in STATES:
            for spec in (URGENT, TEAM, SEVERITY):
                question = question_from_api(spec)
                answer = ask(port, state, spec)
                assert answer["eullm"]["readout"] == "codes"
                ids = readout.tokens(state, question)
                assert answer["eullm"]["prompt_tokens"] == len(ids), state
                served = list(answer["answers"]["q"]["eullm"]["logprobs"].values())
                trained = trained_logprobs(ids, question)
                worst = max(worst, *(abs(a - b) for a, b in zip(served, trained)))
        print(f"largest log-probability difference, engine against training: {worst:.2e}")
        assert worst < tolerance

        # The comparison can tell a prompt off by one space from the right one.
        question = question_from_api(URGENT)
        state = STATES[0]
        served = list(ask(port, state, URGENT)["answers"]["q"]["eullm"]["logprobs"].values())
        off = readout.tokens(state + " ", question)
        assert max(abs(a - b) for a, b in zip(served, trained_logprobs(off, question))) > \
            2 * tolerance
    finally:
        server.terminate()
        server.wait(timeout=60)
        log.close()
