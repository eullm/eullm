"""legal_eval.py --ple-off: a Flash-Next's n-gram layers switched off, nothing else."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
qwen4 = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "legal_eval.py"
spec = importlib.util.spec_from_file_location("legal_eval_ple", SCRIPT)
le = importlib.util.module_from_spec(spec)
sys.modules["legal_eval_ple"] = le
spec.loader.exec_module(le)


def tiny_flash_next():
    cfg = qwen4.Qwen4ExpTextConfig(
        vocab_size=97, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, ple_layer_ids=[2], ple_embed_dim=32,
        ngram_vocab_size_base=101, heads_per_ngram=2, eos_token_id=1, pad_token_id=0,
        hc_lowrank=8, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=16, indexer_compress_ratio=4, max_position_embeddings=64,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg).eval()


def test_switching_the_ple_off_changes_the_output_and_only_through_it():
    model, ids = tiny_flash_next(), torch.randint(2, 97, (1, 12))
    with torch.no_grad():
        model.model.layers[1].ple.ple_embedding.ngram_embedding.weight.normal_(0, 1.0)
        on = model(input_ids=ids).logits
        assert le.switch_off_ple(model) == 1
        off = model(input_ids=ids).logits
        assert not torch.allclose(on, off)
        # the same as a table of zeros: the layer then adds exactly nothing
        twin = tiny_flash_next()
        twin.load_state_dict(model.state_dict())
        ple = twin.model.layers[1].ple
        for p in ple.parameters():
            p.zero_()
        assert torch.allclose(off, twin(input_ids=ids).logits, atol=1e-5)


def test_a_model_without_ple_layers_is_left_alone():
    cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8)
    assert le.switch_off_ple(transformers.Qwen3ForCausalLM(cfg)) == 0
