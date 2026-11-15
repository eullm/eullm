"""tools/check_utf8.py finds text read or written without its encoding, and
passes what names it, in Python files and in the Python of shell scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_utf8", HERE / "check_utf8.py")
check_utf8 = importlib.util.module_from_spec(spec)
sys.modules["check_utf8"] = check_utf8
spec.loader.exec_module(check_utf8)


def run(tmp_path: Path, name: str, text: str, capsys) -> tuple[int, str]:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    code = check_utf8.main([str(path)])
    return code, capsys.readouterr().out


def test_subprocess_text_without_encoding_is_found_in_python(tmp_path, capsys):
    code, out = run(tmp_path, "a.py", (
        "import subprocess\n"
        "subprocess.run(['ls'], capture_output=True, text=True)\n"
        "subprocess.check_output(['ls'], universal_newlines=True)\n"
    ), capsys)
    assert code == 1
    assert ":2: subprocess.run(text=True)" in out
    assert ":3: subprocess.check_output(text=True)" in out


def test_what_names_its_encoding_passes(tmp_path, capsys):
    code, out = run(tmp_path, "b.py", (
        "import subprocess\n"
        "subprocess.run(['ls'], text=True, encoding='utf-8', errors='replace')\n"
        "subprocess.run(['ls'], capture_output=True)\n"  # bytes: no text to decode
        "open('x')\n"  # in .py files this is ruff's PLW1514, not this check's
    ), capsys)
    assert code == 0, out


def test_the_python_of_a_shell_script_is_checked(tmp_path, capsys):
    code, out = run(tmp_path, "job.slurm", (
        "#!/bin/bash\n"
        "python - \"$CSV\" <<'PY'\n"
        "import sys\n"
        "rows = open(sys.argv[1]).read()\n"
        "open(sys.argv[1], 'rb').read()\n"
        "PY\n"
        "n=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])))' x.json)\n"
        "python3 -c 'import pathlib; pathlib.Path(\"y\").read_text(encoding=\"utf-8\")'\n"
    ), capsys)
    assert code == 1
    lines = out.splitlines()
    assert any(":4: open() in text mode" in line for line in lines), out
    assert any(":7: open() in text mode" in line for line in lines), out
    assert len(lines) == 2, out  # the binary open and the read_text with encoding pass


def test_a_heredoc_with_shell_expansions_is_skipped(tmp_path, capsys):
    code, out = run(tmp_path, "x.sh", (
        "python3 - <<EOF\n"
        "print(${COUNT:-1} +)\n"
        "EOF\n"
    ), capsys)
    assert code == 0, out
