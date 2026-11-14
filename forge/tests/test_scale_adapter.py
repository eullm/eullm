"""scale_adapter.py: a LoRA adapter scaled and kept in some layers, written to disk."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scale_adapter.py"
spec = importlib.util.spec_from_file_location("scale_adapter", SCRIPT)
sa = importlib.util.module_from_spec(spec)
sys.modules["scale_adapter"] = sa
spec.loader.exec_module(sa)


def test_scaled_and_kept_in_some_layers(tmp_path):
    src = tmp_path / "adapter"
    src.mkdir()
    t = {f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_{ab}.weight": torch.ones(4, 4)
         for i in range(3) for ab in "AB"}
    st.save_file(t, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text(json.dumps({"r": 4}))
    out = tmp_path / "out"
    assert sa.main(["--adapter", str(src), "--out", str(out), "--scale", "0.5",
                    "--layers", "0,2"]) == 0
    got = st.load_file(str(out / "adapter_model.safetensors"))
    B = "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight"
    A = "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight"
    assert torch.all(got[B.format(0)] == 0.5) and torch.all(got[B.format(2)] == 0.5)
    assert torch.all(got[B.format(1)] == 0)
    assert torch.all(got[A.format(1)] == 1)              # A untouched
    assert json.loads((out / "adapter_config.json").read_text()) == {"r": 4}
    with pytest.raises(SystemExit):
        sa.main(["--adapter", str(src), "--out", str(out)])   # refuses an existing output
