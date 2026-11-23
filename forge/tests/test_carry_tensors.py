"""A merged Qwen3.5 checkpoint gets its MTP head back from the base.

The checkpoints are written by hand in the safetensors layout (8-byte header
length, JSON header, raw bytes), so the test needs neither torch nor the
safetensors package -- and checks the copy byte for byte.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

from eullm_forge.identity import _safetensors_header, carry_missing_tensors

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "carry_tensors.py"


def _write(path: Path, tensors: dict[str, bytes]) -> None:
    header, offset = {}, 0
    for name, data in tensors.items():
        header[name] = {"dtype": "BF16", "shape": [len(data) // 2],
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(tensors.values()))


def _read(path: Path, name: str) -> bytes:
    header, base = _safetensors_header(path)
    start, end = header[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + start)
        return f.read(end - start)


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    _write(base / "model-00001-of-00002.safetensors",
           {"model.layers.0.w": b"\x01\x02" * 4, "mtp.fc.weight": b"\x03\x04" * 6})
    _write(base / "model-00002-of-00002.safetensors",
           {"mtp.layers.0.mlp.up_proj.weight": b"\x05\x06" * 3})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.w": "model-00001-of-00002.safetensors",
        "mtp.fc.weight": "model-00001-of-00002.safetensors",
        "mtp.layers.0.mlp.up_proj.weight": "model-00002-of-00002.safetensors"}}), encoding="utf-8")
    return base


def test_the_mtp_head_is_carried_into_a_single_file_merge(tmp_path):
    base = _base(tmp_path)
    merged = tmp_path / "merged"
    merged.mkdir()
    _write(merged / "model.safetensors", {"model.layers.0.w": b"\x09\x09" * 4})

    got = carry_missing_tensors(base, merged)

    assert got == ["mtp.fc.weight", "mtp.layers.0.mlp.up_proj.weight"]
    index = json.loads(
        (merged / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    assert index == {"model.layers.0.w": "model.safetensors",
                     "mtp.fc.weight": "model-carried.safetensors",
                     "mtp.layers.0.mlp.up_proj.weight": "model-carried.safetensors"}
    shard = merged / "model-carried.safetensors"
    assert _read(shard, "mtp.fc.weight") == b"\x03\x04" * 6
    assert _read(shard, "mtp.layers.0.mlp.up_proj.weight") == b"\x05\x06" * 3
    # the merged weights are left alone, and a second run changes nothing
    assert _read(merged / "model.safetensors", "model.layers.0.w") == b"\x09\x09" * 4
    assert carry_missing_tensors(base, merged) == []


def test_a_sharded_merge_keeps_its_index_and_total_size(tmp_path):
    base = _base(tmp_path)
    merged = tmp_path / "merged"
    merged.mkdir()
    _write(merged / "model-00001-of-00001.safetensors", {"model.layers.0.w": b"\x00\x00" * 4})
    (merged / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 8},
        "weight_map": {"model.layers.0.w": "model-00001-of-00001.safetensors"}}), encoding="utf-8")

    carry_missing_tensors(base, merged)

    index = json.loads((merged / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 8 + 12 + 6
    assert index["weight_map"]["model.layers.0.w"] == "model-00001-of-00001.safetensors"
    assert index["weight_map"]["mtp.fc.weight"] == "model-carried.safetensors"


def test_a_base_without_the_head_carries_nothing(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _write(base / "model.safetensors", {"model.layers.0.w": b"\x01\x01"})
    merged = tmp_path / "merged"
    merged.mkdir()
    _write(merged / "model.safetensors", {"model.layers.0.w": b"\x02\x02"})
    assert carry_missing_tensors(base, merged) == []
    assert not (merged / "model.safetensors.index.json").exists()


def test_the_script_runs_on_its_own(tmp_path):
    base = _base(tmp_path)
    merged = tmp_path / "merged"
    merged.mkdir()
    _write(merged / "model.safetensors", {"model.layers.0.w": b"\x09\x09" * 4})
    out = subprocess.run([sys.executable, str(SCRIPT), str(base), str(merged)],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    assert "2 tensors carried" in out
