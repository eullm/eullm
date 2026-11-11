"""gguf_zero_tensors.py: the matching tensors zeroed, everything else as it was."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gguf_zero_tensors.py"
spec = importlib.util.spec_from_file_location("gguf_zero_tensors", SCRIPT)
gz = importlib.util.module_from_spec(spec)
sys.modules["gguf_zero_tensors"] = gz
spec.loader.exec_module(gz)


def write(path, tensors):
    w = gguf.GGUFWriter(str(path), "test")
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read(path):
    return {t.name: np.array(t.data) for t in gguf.GGUFReader(path).tensors}


def test_a_split_gguf_gets_its_matching_tensors_zeroed_and_the_rest_linked(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    a = {"blk.0.attn_q.weight": rng.normal(size=(8, 16)).astype(np.float32),
         "blk.1.ple_value.weight": rng.normal(size=(8, 16)).astype(np.float32)}
    b = {"blk.2.ffn_up.weight": rng.normal(size=(8, 16)).astype(np.float32)}
    write(src / "m-00001-of-00002.gguf", a)
    write(src / "m-00002-of-00002.gguf", b)
    (src / "mmproj-F16.gguf").write_bytes(b"x")
    assert gz.main(["--gguf", str(src), "--match", "ple_value", "--list"]) == 0
    out = tmp_path / "off"
    assert gz.main(["--gguf", str(src), "--match", "ple_value", "--out", str(out)]) == 0
    first = read(out / "m-00001-of-00002.gguf")
    assert not np.any(first["blk.1.ple_value.weight"])
    assert np.array_equal(first["blk.0.attn_q.weight"], a["blk.0.attn_q.weight"])
    assert (out / "m-00002-of-00002.gguf").is_symlink()
    assert (out / "mmproj-F16.gguf").is_symlink()
    assert np.any(read(src / "m-00001-of-00002.gguf")["blk.1.ple_value.weight"])  # source kept
    assert gz.main(["--gguf", str(src), "--match", "nothing", "--out", str(tmp_path / "x")]) == 1
