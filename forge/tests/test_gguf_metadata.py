"""Tests for GGUF metadata written into an exported model (gguf_metadata).

The files here are written by hand from the format, as ggml/src/gguf.cpp
reads it, not by the module under test; where llama.cpp's own Python
package (`gguf`, MIT) is installed, a file written with it is patched and
read back with it too. The export tests run Forge's real export path
against a stand-in llama.cpp, a converter and a quantizer that copy a tiny
GGUF, so they need no torch and no llama.cpp. Measured beside these tests,
not repeated in CI: a real Qwen3-0.6B Q4_K_M patched this way is read by
llama.cpp's C reader with the key as an f32, and served by the engine with
log-probabilities identical to the original's.
"""

import json
import stat
import struct
import sys
from pathlib import Path

import pytest

from eullm_forge import gguf_metadata
from eullm_forge.gguf_metadata import (
    GGUFError,
    read_fields,
    read_metadata,
    set_metadata,
)

KEY = "eullm.decision.temperature"
Q8_0 = 8  # ggml_type: 32 values in 34 bytes


def _string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pad(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def tiny_gguf(path, alignment=None, tensors=True, version=3):
    """A GGUF of four key-value pairs (a string array among them), an
    f32 tensor and a Q8_0 one, padded as llama.cpp pads them."""
    pairs = [
        _string("general.architecture") + struct.pack("<I", 8) + _string("qwen3"),
        _string("general.name") + struct.pack("<I", 8) + _string("tiny «decide»"),
        _string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 3)
        + _string("Yes") + _string("No") + _string("<|im_end|>"),
        _string("qwen3.context_length") + struct.pack("<II", 4, 4096),
    ]
    align = alignment or 32
    if alignment:
        pairs.append(_string("general.alignment") + struct.pack("<II", 4, alignment))
    blobs = [struct.pack("<4f", 0.5, -1.0, 2.0, 3.5), bytes(range(34))] if tensors else []
    infos, offset = b"", 0
    for name, (dims, kind), blob in zip(("a.weight", "b.weight"), (([4], 0), ([32], Q8_0)), blobs):
        infos += _string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", kind, offset)
        offset = _pad(offset + len(blob), align)
    head = b"GGUF" + struct.pack("<IQQ", version, len(blobs), len(pairs)) + b"".join(pairs) + infos
    data = b""
    for blob in blobs:
        data += blob + b"\0" * (_pad(len(blob), align) - len(blob))
    if blobs:
        head += b"\0" * (_pad(len(head), align) - len(head))
    path.write_bytes(head + data)
    return path


def data_section(path):
    layout, _ = gguf_metadata._parse(path, set())
    return layout.data, path.read_bytes()[layout.data:]


@pytest.mark.parametrize("alignment", [None, 64])
def test_a_key_is_added_and_everything_else_kept(tmp_path, alignment):
    path = tiny_gguf(tmp_path / "m.gguf", alignment=alignment)
    before = read_fields(path)
    old_offset, old_data = data_section(path)
    assert set_metadata(path, {KEY: ("float32", 1.37)}) is True
    after = read_fields(path)
    assert after[KEY] == ("float32", pytest.approx(1.37, rel=1e-7))
    assert {k: v for k, v in after.items() if k != KEY} == before
    assert list(after)[-1] == KEY  # a new key goes last
    assert before["tokenizer.ggml.tokens"] == ("array", ["Yes", "No", "<|im_end|>"])
    new_offset, new_data = data_section(path)
    assert new_data == old_data and new_offset > old_offset
    assert new_offset % (alignment or 32) == 0
    assert read_metadata(path, [KEY, "absent"]) == {KEY: pytest.approx(1.37, rel=1e-7)}


def test_a_key_is_replaced_in_place_and_removed_to_the_byte(tmp_path):
    path = tiny_gguf(tmp_path / "m.gguf")
    original = path.read_bytes()
    set_metadata(path, {KEY: ("float32", 1.37), "general.name": ("string", "renamed")})
    order = list(read_fields(path))
    set_metadata(path, {KEY: ("float32", 0.8)})
    assert list(read_fields(path)) == order
    assert read_metadata(path)[KEY] == pytest.approx(0.8, rel=1e-7)
    # Nothing to change: the file is not written.
    stamp = path.stat().st_mtime_ns
    assert set_metadata(path, {KEY: ("float32", 0.8), "absent": None}) is False
    assert path.stat().st_mtime_ns == stamp
    set_metadata(path, {KEY: None, "general.name": ("string", "tiny «decide»")})
    assert path.read_bytes() == original


def test_a_gguf_without_tensors_keeps_no_data(tmp_path):
    path = tiny_gguf(tmp_path / "m.gguf", tensors=False, version=2)
    set_metadata(path, {"general.license": ("string", "Apache-2.0")})
    assert read_metadata(path)["general.license"] == "Apache-2.0"
    assert data_section(path)[1] == b""


def test_values_and_files_it_will_not_write(tmp_path):
    path = tiny_gguf(tmp_path / "m.gguf")
    original = path.read_bytes()
    for update in ({KEY: ("float32", 1e39)}, {KEY: ("uint8", 300)}, {KEY: ("bool", 1)},
                   {KEY: ("float32", "1.5")}, {KEY: ("array", [1])}, {KEY: ("double", 1.0)},
                   {KEY: 1.5}, {"": ("string", "x")},
                   {"general.alignment": ("uint32", 64)}):
        with pytest.raises(GGUFError):
            set_metadata(path, update)
    assert path.read_bytes() == original

    def damaged(name, data):
        (tmp_path / name).write_bytes(data)
        return tmp_path / name

    for bad in (damaged("text.gguf", b"not a gguf at all, nor anything near one"),
                damaged("v1.gguf", original[:4] + struct.pack("<I", 1) + original[8:]),
                damaged("big-endian.gguf", original[:4] + struct.pack(">I", 3) + original[8:]),
                damaged("cut.gguf", original[:60]),
                damaged("short.gguf", b"GGUF")):
        with pytest.raises(GGUFError):
            read_metadata(bad)


def test_a_failure_halfway_leaves_the_gguf_as_it_was(tmp_path, monkeypatch):
    path = tiny_gguf(tmp_path / "m.gguf")
    original = path.read_bytes()

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(gguf_metadata.shutil, "copyfileobj", disk_full)
    with pytest.raises(OSError):
        set_metadata(path, {KEY: ("float32", 1.37)})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_llama_cpps_own_reader_reads_what_is_written(tmp_path):
    """llama.cpp's Python package writes the file, and reads it back."""
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")

    path = tmp_path / "m.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_custom_alignment(64)
    writer.add_name("tiny")
    writer.add_token_list(["Yes", "No", "<|im_end|>"])
    writer.add_tensor("a.weight", np.arange(8, dtype=np.float32))
    writer.add_tensor("h.weight", np.ones((2, 4), dtype=np.float16))
    # One Q8_0 block, as its 34 bytes: 32 values.
    writer.add_tensor("q.weight", np.arange(34, dtype=np.uint8),
                      raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    before = gguf.GGUFReader(str(path))
    tensors = [(t.name, t.tensor_type, t.data.tobytes()) for t in before.tensors]
    fields = {k: (f.types, repr(f.contents())) for k, f in before.fields.items()
              if not k.startswith("GGUF.")}
    del before

    set_metadata(path, {KEY: ("float32", 1.37)})
    after = gguf.GGUFReader(str(path))
    field = after.fields[KEY]
    assert field.types == [gguf.GGUFValueType.FLOAT32]
    assert field.contents() == pytest.approx(1.37, rel=1e-7)
    assert [(t.name, t.tensor_type, t.data.tobytes()) for t in after.tensors] == tensors
    assert {k: (f.types, repr(f.contents())) for k, f in after.fields.items()
            if not k.startswith("GGUF.") and k != KEY} == fields


# --- the export path, against a stand-in llama.cpp -------------------------------------------

@pytest.fixture
def stand_in_llama_cpp(tmp_path, monkeypatch):
    """A llama.cpp whose converter and quantizer copy a tiny GGUF, where
    Forge's export looks for them; returns the log of what was run."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    source = tiny_gguf(tmp_path / "converted.gguf")
    ran = tmp_path / "ran.jsonl"
    (root / "convert_hf_to_gguf.py").write_text(
        "import json, shutil, sys\n"
        f"open({str(ran)!r}, 'a').write(json.dumps(['convert'] + sys.argv[1:]) + '\\n')\n"
        "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
        f"shutil.copyfile({str(source)!r}, out)\n", encoding="utf-8")
    quantize = root / "build" / "bin" / "llama-quantize"
    quantize.write_text(
        f"#!{sys.executable}\nimport json, shutil, sys\n"
        f"open({str(ran)!r}, 'a').write(json.dumps(['quantize'] + sys.argv[1:]) + '\\n')\n"
        "shutil.copyfile(sys.argv[1], sys.argv[2])\n", encoding="utf-8")
    quantize.chmod(quantize.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("LLAMA_CPP_PATH", str(root))
    return ran


def ran_steps(log):
    return [json.loads(line)[0] for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("quant", ["q8_0", "f16"])
def test_export_writes_the_metadata_into_the_gguf(tmp_path, stand_in_llama_cpp, quant):
    from eullm_forge.export import ExportConfig, export_gguf

    model = tmp_path / "merged"
    model.mkdir()
    out = export_gguf(ExportConfig(model_path=str(model), output_path=str(tmp_path / "m.gguf"),
                                   quantization=quant, metadata={KEY: ("float32", 1.25)}))
    assert read_fields(out)[KEY] == ("float32", 1.25)
    assert ran_steps(stand_in_llama_cpp) == (["convert", "quantize"] if quant == "q8_0"
                                             else ["convert"])
    # Neither the F16 intermediate nor the rewrite's partial file is left.
    assert sorted(p.name for p in tmp_path.glob("m*.gguf*")) == ["m.gguf"]


def test_f16_export_replaces_a_target_that_is_already_there(tmp_path, stand_in_llama_cpp):
    """Exporting twice onto the same -o is how an export is re-run.

    The F16 branch moved the converter's output onto the target with
    Path.rename, which on Windows is MoveFileEx without replace semantics and
    raises FileExistsError when the target is there -- and cli.py catches only
    (FileNotFoundError, ValueError, RuntimeError), so the second run ended in a
    traceback with the previous GGUF still in place.
    """
    from eullm_forge.export import ExportConfig, export_gguf

    model = tmp_path / "merged"
    model.mkdir()
    target = tmp_path / "m.gguf"

    def export(temperature):
        return export_gguf(ExportConfig(
            model_path=str(model), output_path=str(target), quantization="f16",
            metadata={KEY: ("float32", temperature)}))

    export(1.25)
    assert read_fields(target)[KEY] == ("float32", 1.25)
    # The second run has to replace it, not refuse it.
    export(2.5)
    assert read_fields(target)[KEY] == ("float32", 2.5)
    # ...and leave neither the F16 intermediate nor a partial behind.
    assert sorted(p.name for p in tmp_path.glob("m*.gguf*")) == ["m.gguf"]


def test_export_refuses_a_value_before_it_converts_anything(tmp_path, stand_in_llama_cpp):
    from eullm_forge.export import ExportConfig, export_gguf

    (tmp_path / "merged").mkdir()
    with pytest.raises(GGUFError):
        export_gguf(ExportConfig(model_path=str(tmp_path / "merged"),
                                 output_path=str(tmp_path / "m.gguf"),
                                 metadata={KEY: ("float32", 1e40)}))
    assert ran_steps(stand_in_llama_cpp) == []


def test_a_decision_models_gguf_carries_its_fitted_temperature(tmp_path, stand_in_llama_cpp,
                                                               monkeypatch):
    """`decisions export`, its merge stood in for: the dev-fitted temperature
    by default, another when given, none when asked, and one the engine
    would refuse refused before the merge."""
    import eullm_forge.identity as identity
    from eullm_forge.decisions.train import FITTED, REPORT, export_decision_model

    merges = []

    def merge(base, adapter, merged):
        merges.append(base)
        Path(merged).mkdir()

    monkeypatch.setattr(identity, "merge_identity_adapter", merge)
    run = tmp_path / "run"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_config.json").write_text('{"base_model_name_or_path": "b"}')
    (run / REPORT).write_text(json.dumps({"base_model": "Qwen/Qwen3-1.7B",
                                          "dev_temperature": 1.3712345678}))
    out = str(tmp_path / "decide.gguf")
    for asked, carried in ((FITTED, 1.3712345678), (2.5, 2.5), (None, None)):
        gguf = export_decision_model(str(run), out, temperature=asked)
        fields = read_fields(gguf)
        if carried is None:
            assert KEY not in fields
        else:
            assert fields[KEY] == ("float32", pytest.approx(carried, rel=1e-7))
    assert merges == ["Qwen/Qwen3-1.7B"] * 3
    with pytest.raises(ValueError, match="at most 100"):
        export_decision_model(str(run), out, temperature=150)
    assert len(merges) == 3

    # A run with no dev split fitted none: the GGUF carries none.
    (run / REPORT).write_text(json.dumps({"base_model": "Qwen/Qwen3-1.7B"}))
    assert KEY not in read_fields(export_decision_model(str(run), out))


def test_the_export_tools_output_is_decoded_as_utf8(tmp_path, stand_in_llama_cpp,
                                                    monkeypatch):
    """The converter's and quantizer's stderr goes into the RuntimeError shown
    to the user: both runs ask for UTF-8, so a file named in non-ASCII text
    arrives as written whatever the host's default encoding is."""
    import eullm_forge.export as export_mod
    from eullm_forge.export import ExportConfig, export_gguf

    class Result:
        stdout = ""
        stderr = "errore: modello non trovato a citt\u00e0\n"
        returncode = 1

    seen = []

    def run(cmd, **kwargs):
        seen.append(kwargs)
        result = Result()
        result.returncode = 0 if len(seen) == 1 else 1  # convert, then quantize fails
        return result

    monkeypatch.setattr(export_mod.subprocess, "run", run)
    model = tmp_path / "merged"
    model.mkdir()
    with pytest.raises(RuntimeError, match="citt"):
        export_gguf(ExportConfig(model_path=str(model),
                                 output_path=str(tmp_path / "m.gguf"),
                                 quantization="q8_0"))
    assert len(seen) == 2 and all(kw.get("encoding") == "utf-8" for kw in seen)
