"""forge/scripts/ngram_train.py on a tiny saved model, on CPU.

What a chain of 2-hour links relies on:
* the rows and their order are the same in every link;
* two links of two steps end where one run of four does, bit for bit;
* a finished run writes DONE and the next link does nothing;
* the LoRA control is sized to the memory it is compared with.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ngram_train.py"
spec = importlib.util.spec_from_file_location("ngram_train", SCRIPT)
nt = importlib.util.module_from_spec(spec)
sys.modules["ngram_train"] = nt
spec.loader.exec_module(nt)

WORDS = ("articolo comma legge decreto codice civile penale procedura tribunale corte "
         "sentenza ricorso appello contratto obbligazione danno responsabilità termine "
         "prescrizione notifica").split()


def corpus(n=60, seed=0):
    import random

    rnd = random.Random(seed)
    return [" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(5, 30))) for _ in range(n)]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """A tiny Qwen3 and a word-level tokenizer, saved as a model directory."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    d = tmp_path_factory.mktemp("tiny")
    vocab = {"<unk>": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>",
                                                eos_token="<eos>")
    fast.save_pretrained(d)
    cfg = transformers.Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8,
                                   max_position_embeddings=64, eos_token_id=1)
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(d)
    return d


@pytest.fixture()
def data(tmp_path):
    p = tmp_path / "program.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in corpus():
            f.write(json.dumps({"text": t}) + "\n")
        f.write('{"text": "half a li')  # a line cut by a job that died
    return p


def run(tiny_dir, data, out, *extra):
    args = ["--model", str(tiny_dir), "--data", str(data), "--out", str(out),
            "--seq-len", "16", "--batch", "4", "--rows-per-head", "101", "--embed-dim", "32",
            "--heads-per-ngram", "2", "--warmup", "2", "--log-steps", "1", "--save-steps", "100",
            *extra]
    assert nt.main(args) == 0


def test_pack_cuts_rows_of_seq_len():
    rows = nt.pack([[5, 6, 7], [8], [9, 10, 11, 12]], seq_len=3, eos=0)
    assert rows.tolist() == [[5, 6, 7], [0, 8, 0], [9, 10, 11]]


def test_read_texts_skips_broken_and_empty_lines(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"text": "uno"}\n{"text": ""}\n{"x": 1}\n{"text": "du')
    assert nt.read_texts([p]) == ["uno"]


def test_batches_resume_where_they_stopped():
    whole = list(nt.batches(23, 4, seed=3, start_step=0, epochs=2))
    assert len(whole) == 10  # 5 full batches an epoch, the partial one dropped
    assert list(nt.batches(23, 4, seed=3, start_step=7, epochs=2)) == whole[7:]
    first, second = (sorted(i for _, e, b in whole if e == ep for i in b) for ep in (0, 1))
    assert len(set(first)) == 20 and len(set(second)) == 20
    assert [b for _, e, b in whole if e == 0] != [b for _, e, b in whole if e == 1]


def test_two_links_end_where_one_run_does(tiny_dir, data, tmp_path):
    run(tiny_dir, data, tmp_path / "one", "--max-steps", "4")
    run(tiny_dir, data, tmp_path / "two", "--max-steps", "2")
    assert json.loads((tmp_path / "two" / "ckpt" / "state.json").read_text()) == {"step": 2}
    run(tiny_dir, data, tmp_path / "two", "--max-steps", "4")
    from eullm_forge.memory import NGramMemory

    a = NGramMemory.load(tmp_path / "one" / "ckpt" / "memory").state_dict()
    b = NGramMemory.load(tmp_path / "two" / "ckpt" / "memory").state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert a["scale"].abs().item() > 0  # it learned something
    logged = [json.loads(x)["step"] for x in (tmp_path / "two" / "train-log.jsonl").open()]
    assert logged == [1, 2, 3, 4]


def test_a_finished_run_writes_done_and_stops(tiny_dir, data, tmp_path):
    out = tmp_path / "run"
    run(tiny_dir, data, out)
    done = json.loads((out / "DONE").read_text())
    assert done["steps"] > 0 and (out / "memory" / "config.json").exists()
    log_lines = (out / "train-log.jsonl").read_text()
    run(tiny_dir, data, out)
    assert (out / "train-log.jsonl").read_text() == log_lines


def test_rows_are_tokenised_once(tiny_dir, data, tmp_path, monkeypatch):
    out = tmp_path / "run"
    run(tiny_dir, data, out, "--max-steps", "1")
    assert (out / "rows-16.npy").exists()
    monkeypatch.setattr(nt, "read_texts", lambda paths: pytest.fail("tokenised again"))
    run(tiny_dir, data, out, "--max-steps", "2")


def test_stop_after_saves_the_position(tiny_dir, data, tmp_path):
    out = tmp_path / "run"
    run(tiny_dir, data, out, "--stop-after", "1e-9")
    assert json.loads((out / "ckpt" / "state.json").read_text()) == {"step": 0}
    assert not (out / "DONE").exists()


def test_lora_rank_matches_the_parameter_budget(tiny_dir):
    model = transformers.AutoModelForCausalLM.from_pretrained(tiny_dir)
    targets = ["q_proj", "o_proj"]
    per_rank = sum(m.in_features + m.out_features for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in targets)
    assert nt.lora_rank_for(model, 5 * per_rank, targets) == 5
    assert nt.lora_rank_for(model, 1, targets) == 1
    assert nt.lora_rank_for(model, 5 * per_rank, r".*\.(q_proj|o_proj)") == 5


def test_a_run_whose_loss_is_not_finite_stops_and_stays_stopped(tiny_dir, data, tmp_path,
                                                                 monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(torch, "isfinite", lambda x: torch.tensor(False))
    args = ["--model", str(tiny_dir), "--data", str(data), "--out", str(out),
            "--seq-len", "16", "--batch", "4", "--rows-per-head", "101", "--embed-dim", "32",
            "--heads-per-ngram", "2", "--max-bad-steps", "3"]
    assert nt.main(args) == 3
    assert json.loads((out / "DIVERGED").read_text())["step"] == 2
    assert not (out / "DONE").exists()
    monkeypatch.undo()
    assert nt.main(args) == 3  # a later link does not train on from a diverged state


def test_clipping_keeps_the_resume_exact(tiny_dir, data, tmp_path):
    run(tiny_dir, data, tmp_path / "one", "--max-steps", "4", "--clip", "0.01")
    run(tiny_dir, data, tmp_path / "two", "--max-steps", "2", "--clip", "0.01")
    run(tiny_dir, data, tmp_path / "two", "--max-steps", "4", "--clip", "0.01")
    from eullm_forge.memory import NGramMemory

    a = NGramMemory.load(tmp_path / "one" / "ckpt" / "memory").state_dict()
    b = NGramMemory.load(tmp_path / "two" / "ckpt" / "memory").state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k
