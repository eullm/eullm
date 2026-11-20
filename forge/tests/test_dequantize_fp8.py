"""dequantize_fp8.py: FP8 tensors times their scales in BF16, scales dropped."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dequantize_fp8.py"


def _load():
    spec = importlib.util.spec_from_file_location("dequantize_fp8", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fp8(t):
    return t.to(torch.float8_e4m3fn)


def test_a_mistral_style_checkpoint_comes_out_in_bf16(tmp_path):
    src, dst = tmp_path / "fp8", tmp_path / "bf16"
    src.mkdir()
    lin = torch.tensor([[2.0, -4.0], [8.0, 0.5]])
    experts = torch.stack([torch.full((2, 3), 2.0), torch.full((2, 3), -1.0)])
    shard1 = {"model.layers.0.self_attn.o_proj.weight": _fp8(lin),
              "model.layers.0.self_attn.o_proj.weight_scale_inv": torch.tensor(0.01),
              "model.layers.0.self_attn.o_proj.activation_scale": torch.tensor(0.5),
              "model.norm.weight": torch.ones(2, dtype=torch.bfloat16)}
    shard2 = {"model.layers.0.mlp.experts.gate_up_proj": _fp8(experts),
              "model.layers.0.mlp.experts.gate_up_proj_scale_inv": torch.tensor([0.01, 0.03]),
              "model.layers.0.mlp.experts.gate_up_proj_activation_scale": torch.tensor([1.0, 1.0]),
              "lm_head.weight": torch.full((2, 2), 0.25, dtype=torch.bfloat16)}
    st.save_file(shard1, str(src / "model-00001-of-00002.safetensors"))
    st.save_file(shard2, str(src / "model-00002-of-00002.safetensors"))
    wm = {**{k: "model-00001-of-00002.safetensors" for k in shard1},
          **{k: "model-00002-of-00002.safetensors" for k in shard2}}
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}), encoding="utf-8")
    (src / "config.json").write_text(json.dumps(
        {"model_type": "mistral3", "quantization_config": {"quant_method": "fp8"}}), encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert _load().main([str(src), str(dst)]) == 0

    index = json.loads((dst / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    assert set(index) == {"model.layers.0.self_attn.o_proj.weight", "model.norm.weight",
                          "model.layers.0.mlp.experts.gate_up_proj", "lm_head.weight"}
    out = {}
    for shard in set(index.values()):
        out.update(st.load_file(str(dst / shard)))
    assert all(t.dtype == torch.bfloat16 for t in out.values())
    assert torch.allclose(out["model.layers.0.self_attn.o_proj.weight"].float(), lin * 0.01,
                          rtol=1e-2)
    e = out["model.layers.0.mlp.experts.gate_up_proj"].float()
    assert torch.allclose(e[0], torch.full((2, 3), 0.02), rtol=1e-2)
    assert torch.allclose(e[1], torch.full((2, 3), -0.03), rtol=1e-2)
    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in cfg and cfg["dtype"] == "bfloat16"
    assert (dst / "tokenizer.json").exists()


def test_an_fp8_tensor_without_a_scale_stops_the_conversion(tmp_path):
    src = tmp_path / "fp8"
    src.mkdir()
    st.save_file({"w.weight": _fp8(torch.ones(2, 2))}, str(src / "m.safetensors"))
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w.weight": "m.safetensors"}}), encoding="utf-8")
    (src / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load().main([str(src), str(tmp_path / "out")])


def test_the_output_is_written_in_parts(tmp_path):
    mod = _load()
    src = tmp_path / "fp8"
    src.mkdir()
    tensors = {f"l{i}.weight": _fp8(torch.ones(4, 4)) for i in range(3)}
    tensors.update({f"l{i}.weight_scale_inv": torch.tensor(0.01) for i in range(3)})
    st.save_file(tensors, str(src / "m.safetensors"))
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "m.safetensors" for k in tensors}}), encoding="utf-8")
    (src / "config.json").write_text("{}", encoding="utf-8")
    mod.convert(src, tmp_path / "out", part_bytes=30)
    index = json.loads((tmp_path / "out" / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert len(set(index["weight_map"].values())) == 3
