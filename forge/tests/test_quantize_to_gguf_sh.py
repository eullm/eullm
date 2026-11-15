"""`quantize_to_gguf.sh` never ships a Q4 quantized from an older F16.

The F16 block reconverts when the HF export is newer, but the QUANT skip
only saw "the file exists": with the F16 deleted to save space and HF
re-exported, conversion rebuilt a fresh F16 and the untouched Q4 -- built
from the old F16 -- was reused. The shipped file described a model that no
longer existed. Now a run that (re)builds the F16 re-quantizes beside it.
These tests drive the real script with a stubbed toolchain (the
test_train_sh.py shape, skipped on Windows).
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quantize_to_gguf.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="needs POSIX bash (on Windows, bash is WSL's)")


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    lcpp = tmp_path / "lcpp"
    (lcpp / "build" / "bin").mkdir(parents=True)
    (lcpp / ".git").mkdir()
    monkeypatch.setenv("LCPP_DIR", str(lcpp))
    monkeypatch.setenv("LCPP_SKIP_UPDATE", "1")
    monkeypatch.setenv("GGUF_NAME", "mymodel")
    # the converter stamps which HF export it read; the quantizer copies.
    (lcpp / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
        "ver = open(sys.argv[1] + '/version.txt').read().strip()\n"
        "open(out, 'w').write('F16DATA-built-from-' + ver + chr(10))\n")
    _exe(lcpp / "build" / "bin" / "llama-quantize",
         "#!/usr/bin/env bash\ncp \"$1\" \"$2\"\n")
    _exe(lcpp / "build" / "bin" / "llama-cli",
         "#!/usr/bin/env bash\n"
         'if [ "${1:-}" = "--help" ]; then echo "--single-turn"; exit 0; fi\n'
         "echo 'llama_model_load: ok'\n"
         "echo 'prosa italiana legale di verifica.'\n")
    (lcpp / "build" / "bin" / "llama-perplexity").write_text("")

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text('{"model": "x"}\n')
    out = tmp_path / "out"
    out.mkdir()
    return hf, out


def _run(hf: Path, out: Path) -> str:
    proc = subprocess.run(["bash", str(SCRIPT), str(hf), str(out)],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout + proc.stderr


def _write_export(hf: Path, version: str) -> None:
    (hf / "version.txt").write_text(version + "\n")


def test_a_rebuilt_f16_re_quantizes_the_q4_beside_it(stage):
    hf, out = stage
    _write_export(hf, "v1-old-checkpoint")
    _run(hf, out)
    assert (out / "mymodel-q4_k_m.gguf").read_text().startswith("F16DATA-built-from-v1")
    # F16 deleted to save space, HF re-exported, script re-run.
    (out / "mymodel-f16.gguf").unlink()
    _write_export(hf, "v2-new-checkpoint")

    printed = _run(hf, out)
    assert "skipping quantization" not in printed
    assert (out / "mymodel-f16.gguf").read_text().startswith("F16DATA-built-from-v2")
    assert (out / "mymodel-q4_k_m.gguf").read_text().startswith("F16DATA-built-from-v2")


def test_a_reused_f16_reuses_its_q4(stage):
    hf, out = stage
    _write_export(hf, "v1-old-checkpoint")
    _run(hf, out)
    printed = _run(hf, out)
    assert "skipping conversion" in printed
    assert "skipping quantization" in printed
    assert (out / "mymodel-q4_k_m.gguf").read_text().startswith("F16DATA-built-from-v1")
