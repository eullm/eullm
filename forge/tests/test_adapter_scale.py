"""legal_eval.py --adapter-scale: a LoRA merged at a share of itself."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "legal_eval.py"
spec = importlib.util.spec_from_file_location("legal_eval_scale", SCRIPT)
le = importlib.util.module_from_spec(spec)
sys.modules["legal_eval_scale"] = le
spec.loader.exec_module(le)


def tiny():
    cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8)
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg).eval()


def merged(scale):
    base = tiny()
    torch.manual_seed(1)
    m = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                                  target_modules=["q_proj", "down_proj"]))
    le.scale_lora(m, scale)
    return m.merge_and_unload()


def test_a_half_scaled_adapter_lands_halfway_weight_by_weight():
    base, half, full = tiny().state_dict(), merged(0.5).state_dict(), merged(1.0).state_dict()
    k = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.allclose(base[k], full[k])
    assert torch.allclose(half[k], (base[k] + full[k]) / 2, atol=1e-6)
    zero = merged(0.0).state_dict()
    assert torch.allclose(zero[k], base[k])
