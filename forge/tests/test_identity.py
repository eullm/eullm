"""Tests for identity dataset generation."""

import pytest

from eullm_forge.identity import IdentityConfig, generate_identity_dataset


def test_generate_identity_dataset_english():
    config = IdentityConfig(
        identity_name="TestAI",
        languages=["en"],
    )
    examples = generate_identity_dataset(config)
    assert len(examples) >= 6
    assert any("TestAI" in ex["output"] for ex in examples)
    assert any("Who are you" in ex["instruction"] for ex in examples)


def test_generate_identity_dataset_italian():
    config = IdentityConfig(
        identity_name="LegalAI di Studio Rossi",
        languages=["it", "en"],
    )
    examples = generate_identity_dataset(config)
    # Should have both English and Italian examples
    assert any("Chi sei" in ex["instruction"] for ex in examples)
    assert any("Who are you" in ex["instruction"] for ex in examples)
    assert any("LegalAI di Studio Rossi" in ex["output"] for ex in examples)


def test_generate_identity_dataset_german():
    config = IdentityConfig(
        identity_name="MedizinAI",
        languages=["de", "en"],
    )
    examples = generate_identity_dataset(config)
    assert any("Wer bist du" in ex["instruction"] for ex in examples)


def test_generate_identity_dataset_french():
    config = IdentityConfig(
        identity_name="FinanceAI",
        languages=["fr", "en"],
    )
    examples = generate_identity_dataset(config)
    assert any("Qui es-tu" in ex["instruction"] for ex in examples)


def test_generate_identity_dataset_default_name():
    config = IdentityConfig(languages=["en"])
    examples = generate_identity_dataset(config)
    assert any("EULLM Assistant" in ex["output"] for ex in examples)


def test_missing_custom_dataset_path_fails_closed(tmp_path):
    """A non-empty dataset_path that points nowhere must fail, not silently train synthetic.

    Without this, a typo or an unmounted volume falls into the synthetic branch,
    whose "Generated N examples" log is indistinguishable from an intentional run —
    burning 1-2h of GPU and baking generic branding into the merged weights.
    Placed before the heavy imports, so this needs no torch to verify.
    """
    from eullm_forge.identity import fine_tune_identity

    config = IdentityConfig(
        model_path="dummy",
        dataset_path=str(tmp_path / "missing-custom.json"),
    )
    with pytest.raises(FileNotFoundError, match="missing-custom.json"):
        fine_tune_identity(config)


# --- SFT formatting: what the loss is actually computed on --------------------
#
# These use a real HuggingFace tokenizer built offline — a word-level vocabulary
# with the ChatML tokens registered as special tokens, the way Qwen has them —
# so the template rendering, the special-token splitting and the masking are
# exercised as they run in training, with no download.

CHATML_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


def make_tokenizer(texts, chatml=True):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    special = (CHATML_TOKENS if chatml else ["<|endoftext|>"]) + ["[UNK]"]
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    words = texts + ["user assistant system"]
    tok.train_from_iterator(words, trainers.WordLevelTrainer(special_tokens=special))
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]",
        eos_token="<|endoftext|>", pad_token="<|endoftext|>",
        additional_special_tokens=CHATML_TOKENS[:2] if chatml else [],
    )


PAIRS = [
    {"instruction": "Chi sei?", "output": "Sono LegalAI."},
    {"instruction": "Come ti chiami davvero oggi?", "output": "Mi chiamo LegalAI e basta."},
]


def all_text(pairs):
    return [p["instruction"] + " " + p["output"] for p in pairs]


def test_chatml_is_installed_when_the_tokenizer_has_no_template():
    from eullm_forge.identity import CHATML_TEMPLATE, ensure_chat_template

    tok = make_tokenizer(all_text(PAIRS))
    assert ensure_chat_template(tok) is True
    assert tok.chat_template == CHATML_TEMPLATE
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Chi sei?"}], tokenize=False, add_generation_prompt=True,
    )
    assert rendered == "<|im_start|>user\nChi sei?<|im_end|>\n<|im_start|>assistant\n"


def test_an_existing_template_is_left_alone():
    from eullm_forge.identity import ensure_chat_template

    tok = make_tokenizer(all_text(PAIRS))
    tok.chat_template = "{{ messages[0]['content'] }}"
    assert ensure_chat_template(tok) is False
    assert tok.chat_template == "{{ messages[0]['content'] }}"


def test_no_template_and_no_chatml_tokens_is_refused():
    """Inventing the tokens would need embedding rows a LoRA never trains."""
    from eullm_forge.identity import ensure_chat_template

    tok = make_tokenizer(all_text(PAIRS), chatml=False)
    with pytest.raises(ValueError, match="no ChatML tokens"):
        ensure_chat_template(tok)


def test_only_the_answer_and_its_end_of_turn_are_trained_on():
    from eullm_forge.identity import IGNORE_INDEX, build_sft_features, ensure_chat_template

    tok = make_tokenizer(all_text(PAIRS))
    ensure_chat_template(tok)
    features = build_sft_features(PAIRS, tok)

    assert len(features) == len(PAIRS)
    for f, pair in zip(features, PAIRS):
        assert len(f["input_ids"]) == len(f["attention_mask"]) == len(f["labels"])
        # No padding at all: every position is a real token.
        assert all(m == 1 for m in f["attention_mask"])
        assert tok.pad_token_id not in f["input_ids"]

        trained = [t for t in f["labels"] if t != IGNORE_INDEX]
        text = tok.decode(trained)
        # The answer, and the end-of-turn token that teaches the model to stop…
        for word in pair["output"].rstrip(".").split():
            assert word in text
        assert "<|im_end|>" in text
        # …and nothing of the question or the headers.
        assert "Chi" not in text and "chiami" not in text
        assert "user" not in text and "<|im_start|>" not in text

        # The masked part is a prefix: the prompt, exactly as inference sends it.
        n_masked = f["labels"].index(next(t for t in f["labels"] if t != IGNORE_INDEX))
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": pair["instruction"]}],
            tokenize=False, add_generation_prompt=True,
        )
        assert f["input_ids"][:n_masked] == tok(prompt, add_special_tokens=False)["input_ids"]


def test_an_example_that_does_not_fit_is_dropped_not_truncated():
    """A truncated answer loses its end-of-turn token; the model would learn not to stop."""
    from eullm_forge.identity import build_sft_features, ensure_chat_template

    tok = make_tokenizer(all_text(PAIRS))
    ensure_chat_template(tok)
    short, long_ = (len(f["input_ids"]) for f in build_sft_features(PAIRS, tok))
    assert short < long_

    kept = build_sft_features(PAIRS, tok, max_length=short)
    assert len(kept) == 1 and len(kept[0]["input_ids"]) == short

    with pytest.raises(ValueError, match="no identity example fits"):
        build_sft_features(PAIRS, tok, max_length=short - 1)


def test_a_template_that_cannot_separate_prompt_from_answer_is_refused():
    from eullm_forge.identity import build_sft_features

    tok = make_tokenizer(all_text(PAIRS))
    # Renders the assistant header only when there is no answer: the prompt
    # is then not a prefix of the conversation.
    tok.chat_template = (
        "{% for m in messages %}{{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    with pytest.raises(ValueError, match="not render the prompt as a prefix"):
        build_sft_features(PAIRS, tok)


def test_collate_pads_per_batch_and_masks_the_padding():
    pytest.importorskip("torch")
    from eullm_forge.identity import IGNORE_INDEX, collate_sft

    batch = collate_sft(
        [
            {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
            {"input_ids": [8], "attention_mask": [1], "labels": [8]},
        ],
        pad_token_id=0,
    )
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]


def test_fine_tune_identity_runs_end_to_end_on_cpu(tmp_path, monkeypatch):
    """The real training path on a tiny Qwen3, CPU only, nothing patched but the GPU check.

    The adapter's tokenizer must carry the template the weights were trained
    on: the merge copies it, and the GGUF takes its template from there.
    """
    pytest.importorskip("peft")
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from eullm_forge.identity import CHATML_TEMPLATE, fine_tune_identity

    config = IdentityConfig(identity_name="LegalAI", languages=["it", "en"], num_epochs=1)
    tok = make_tokenizer(all_text(generate_identity_dataset(config)))
    model_dir = tmp_path / "student"
    tok.save_pretrained(model_dir)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128,
    )).save_pretrained(model_dir)

    # Force the CPU path so the test means the same on a machine with a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config.model_path = str(model_dir)
    adapter = fine_tune_identity(config)

    from transformers import AutoTokenizer

    saved = AutoTokenizer.from_pretrained(adapter)
    assert saved.chat_template == CHATML_TEMPLATE
    assert (tmp_path / "identity-lora" / "adapter" / "adapter_config.json").exists()


# --- stage 3 on Leonardo: domain pairs + identity, resumable -------------------

DOMAIN_PAIRS = [
    {"instruction": "Chi risponde del danno ingiusto?",
     "output": "Chi lo ha causato con dolo o colpa deve risarcirlo.",
     "task": "qa", "source": "codice_civile", "key": "a1"},
    {"instruction": "Riassumi il seguente testo:\n\nIl ricorso è respinto.",
     "output": "Il giudice ha respinto il ricorso.",
     "task": "riassunto", "source": "cds", "key": "a2"},
]


def test_load_pairs_reads_jsonl_keeps_only_the_pair_and_skips_a_cut_line(tmp_path):
    import json

    from eullm_forge.identity import load_pairs

    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in DOMAIN_PAIRS)
                    + '\n{"instruction": "tagliata a metà', encoding="utf-8")
    pairs = load_pairs(str(path))
    assert pairs == [{"instruction": p["instruction"], "output": p["output"]}
                     for p in DOMAIN_PAIRS]


def test_load_pairs_refuses_a_file_with_no_pairs(tmp_path):
    from eullm_forge.identity import load_pairs

    path = tmp_path / "empty.jsonl"
    path.write_text('{"foo": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no instruction/output pairs"):
        load_pairs(str(path))


def test_a_missing_instruction_file_fails_before_any_heavy_work(tmp_path):
    from eullm_forge.identity import fine_tune_identity

    config = IdentityConfig(model_path="dummy",
                            instruction_path=str(tmp_path / "missing-pairs.jsonl"))
    with pytest.raises(FileNotFoundError, match="missing-pairs.jsonl"):
        fine_tune_identity(config)


def test_stage3_script_trains_resumes_and_merges_on_cpu(tmp_path, monkeypatch):
    """The Leonardo path end to end on a tiny Qwen3: stage3_sft.py → adapter,
    a second submission resumes instead of starting over, and the package
    job's merge produces a model directory with the template in it."""
    pytest.importorskip("peft")
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from eullm_forge.identity import CHATML_TEMPLATE, merge_identity_adapter

    identity = generate_identity_dataset(IdentityConfig(identity_name="EULLM Legal IT",
                                                        languages=["it", "en"]))
    tok = make_tokenizer(all_text(identity) + all_text(DOMAIN_PAIRS))
    base = tmp_path / "merged-step"
    tok.save_pretrained(base)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=256,
    )).save_pretrained(base)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in DOMAIN_PAIRS),
                     encoding="utf-8")
    out = tmp_path / "sft"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "stage3_sft.py"
    spec = importlib.util.spec_from_file_location("stage3_sft", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    argv = ["stage3_sft.py", "--model", str(base), "--pairs", str(pairs), "--out", str(out),
            "--epochs", "1", "--rank", "4", "--max-length", "256", "--batch-size", "2",
            "--grad-accum", "1", "--identity-repeat", "2", "--save-steps", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    assert (out / "adapter" / "adapter_config.json").exists()
    ckpts = sorted(out.glob("checkpoint-*"))
    assert ckpts, "no checkpoint written, so nothing to resume from"

    # A second submission finds the checkpoint and resumes rather than failing.
    assert mod.main() == 0

    merged = merge_identity_adapter(str(base), str(out / "adapter"), str(out / "merged"))
    assert (Path(merged) / "config.json").exists()
    assert AutoTokenizer.from_pretrained(merged).chat_template == CHATML_TEMPLATE

    # The rows of the end-of-turn token were trained and survived the merge,
    # in the embedding AND in the separate output head of this untied model:
    # the head is what decides whether the turn ends.
    before = Qwen3ForCausalLM.from_pretrained(base)
    after = Qwen3ForCausalLM.from_pretrained(merged)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    other = tok.convert_tokens_to_ids("Sono")
    for get in ("get_input_embeddings", "get_output_embeddings"):
        w0, w1 = getattr(before, get)().weight, getattr(after, get)().weight
        assert not torch.equal(w0[end], w1[end]), f"{get}: <|im_end|> row untouched"
        assert torch.equal(w0[other], w1[other]), f"{get}: an ordinary row changed"


# --- the chat-format token rows are trained (v0.1 never ended its turn) --------

def test_format_tokens_are_the_added_tokens_the_template_writes():
    from eullm_forge.identity import ensure_chat_template, format_token_ids

    tok = make_tokenizer(all_text(PAIRS))
    ensure_chat_template(tok)
    ids = format_token_ids(tok)
    assert tok.convert_ids_to_tokens(ids) == ["<|im_start|>", "<|im_end|>"]


def test_the_trainable_rows_reach_the_output_head_tied_or_not():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from eullm_forge.identity import trainable_token_target

    def tiny(tied):
        return Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            tie_word_embeddings=tied,
        ))

    # Shared tensor (Qwen3-4B): PEFT follows the tie from a plain list.
    assert trainable_token_target(tiny(True), [30, 31]) == [30, 31]
    # Separate head (Qwen3-8B): both modules must be named, or the rows of
    # the embedding change how the token is read and not whether it is written.
    assert trainable_token_target(tiny(False), [30, 31]) == {
        "embed_tokens": [30, 31], "lm_head": [30, 31],
    }


def test_the_merge_keeps_the_dtype_the_weights_are_stored_in(tmp_path):
    """Loaded in the default float32, an 8B model is 32 GB: more than a
    serial-partition job may have, for a merge that is only an addition."""
    pytest.importorskip("peft")
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from eullm_forge.identity import merge_identity_adapter

    base = tmp_path / "base"
    tok = make_tokenizer(all_text(DOMAIN_PAIRS))
    tok.save_pretrained(base)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=256,
    )).to(torch.bfloat16)
    model.save_pretrained(base)
    adapter = tmp_path / "adapter"
    get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj"])).save_pretrained(adapter)

    merged = Path(merge_identity_adapter(str(base), str(adapter), str(tmp_path / "merged")))
    with safe_open(str(merged / "model.safetensors"), "pt") as f:
        dtypes = {f.get_tensor(k).dtype for k in f.keys()}
    assert dtypes == {torch.bfloat16}


def test_an_image_text_model_trains_and_merges_its_language_model_only(tmp_path):
    """Ministral-3-8B ships as an image-text Mistral3 model, which
    `AutoModelForCausalLM` refuses. It must load, take LoRA on the language
    model and not on the vision tower (whose layers use the same names), and
    merge back into the same architecture llama.cpp converts."""
    pytest.importorskip("peft")
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, Mistral3Config

    from eullm_forge.identity import (
        load_text_model,
        lora_target_modules,
        merge_identity_adapter,
    )

    base = tmp_path / "base"
    tok = make_tokenizer(all_text(DOMAIN_PAIRS))
    tok.save_pretrained(base)
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(Mistral3Config(
        text_config={"model_type": "mistral", "vocab_size": len(tok), "hidden_size": 32,
                     "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16},
        vision_config={"model_type": "pixtral", "hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 1, "num_attention_heads": 2, "head_dim": 16},
    )).to(torch.bfloat16).save_pretrained(base)

    model = load_text_model(str(base), dtype=torch.bfloat16)
    assert type(model).__name__.startswith("Mistral3For")  # not refused as a causal LM
    peft_model = get_peft_model(model, LoraConfig(r=2, target_modules=lora_target_modules(model)))
    adapted = {n for n, _ in peft_model.named_modules() if n.endswith(".lora_A")}
    assert adapted and all(".language_model." in n for n in adapted), sorted(adapted)
    assert any(".mlp.down_proj." in n for n in adapted)
    for p in peft_model.parameters():          # a trained adapter, not the zero init
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.1)
    peft_model.save_pretrained(tmp_path / "adapter")

    merged = Path(merge_identity_adapter(str(base), str(tmp_path / "adapter"),
                                         str(tmp_path / "merged")))
    before = AutoModelForImageTextToText.from_pretrained(base).state_dict()
    after = AutoModelForImageTextToText.from_pretrained(merged).state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(".language_model." in k for k in changed), sorted(changed)


def test_a_plain_causal_lm_keeps_the_target_list():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from eullm_forge.identity import LORA_TARGETS, lora_target_modules

    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    ))
    assert lora_target_modules(model) == list(LORA_TARGETS)


def test_lora_dropout_is_off_only_for_fused_moe_experts():
    import torch

    from eullm_forge.identity import lora_dropout_for

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Parameter(torch.zeros(4, 8, 16))

    dense = torch.nn.Module()
    dense.down_proj = torch.nn.Linear(16, 8)
    moe = torch.nn.Module()
    moe.attn = torch.nn.Linear(16, 16)
    moe.experts = Experts()
    assert lora_dropout_for(dense) == 0.05
    assert lora_dropout_for(moe) == 0.0
