#!/usr/bin/env python3
"""Text read or written by Python in this repository names its encoding.

    python3 tools/check_utf8.py [PATH...]

`open()` and `Path.read_text`/`write_text` without `encoding=` take the
host's locale, and so does a `subprocess` call with `text=True`: what was
written as UTF-8 (a job name, a path, a model's text) is garbled, or fails to
decode, wherever the locale is not UTF-8. Ruff's PLW1514 checks `open()` in
`.py` files and CI runs it; this checks what it does not see:

* `subprocess.run`, `check_output`, `check_call`, `call` and `Popen` with
  `text=True` or `universal_newlines=True` and no `encoding=`, in `.py` files;
* `.read_text()`/`.write_text()` on any object (ruff sees them only where the
  object is visibly a `Path`), the encoding by keyword or by position;
* all of these in the Python that shell scripts and Slurm jobs carry, the
  interpreter named `python3` or by a variable (`"$PY"`): here-documents
  (`$PY - <<'PY' ... PY`) and `-c '...'` or `-c "..."`;
* `Get-Content`, `Set-Content`, `Add-Content`, `Out-File`, `Import-Csv` and
  `Export-Csv` without `-Encoding` in PowerShell scripts, which otherwise use
  the ANSI code page under Windows PowerShell 5.1.

The first version (2026-11-15) knew only `python3` by name, left read_text and
write_text to ruff and did not read PowerShell, and 485 places went unseen.

The bug was found by Gabriele (Gabriele06-local on GitHub), whose fixes of
it, script by script, showed how widespread it was and led to this check.
It prints each place and exits 1 when there is one.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "node_modules", "target", ".venv", "venv", "__pycache__"}
SKIP_UNDER = (ROOT / "engine" / "vendor",)
SUBPROCESS = {"run", "check_output", "check_call", "call", "Popen"}

# The interpreter as scripts name it: python, python3, python3.11, or a
# variable holding it ($PY, "${PYTHON}", "$PY_BIN"). Before 2026-11-18 only
# the first two were recognised, and eight scripts calling "$PY" went unseen.
INTERP = r"(?:\bpython(?:3(?:\.\d+)?)?\b|\"?\$\{?PY(?:THON)?\w*\}?\"?)"
# A here-document fed to it (`$PY - <<'PY'`, `python3 -I - "$a" <<EOF`) and its
# body, up to the delimiter alone on a line; and `-c '...'` or `-c "..."`.
HEREDOC = re.compile(
    INTERP + r"[^\n]*?<<-?\s*(['\"]?)(\w+)\1[^\n]*\n(.*?)\n\s*\2\s*$", re.DOTALL | re.MULTILINE
)
DASH_C_SINGLE = re.compile(INTERP + r"[^\n'\"]*?-c\s+'([^']*)'", re.DOTALL)
DASH_C_DOUBLE = re.compile(INTERP + r"[^\n'\"]*?-c\s+\"((?:[^\"\\]|\\.)*)\"", re.DOTALL)
# PowerShell reads and writes with the system's ANSI code page unless told
# otherwise (Windows PowerShell 5.1, the one install.ps1 meets on most PCs).
PS_TEXT = re.compile(r"\b(Get-Content|Set-Content|Add-Content|Out-File|Import-Csv|Export-Csv)\b",
                     re.IGNORECASE)


def is_true(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def problems(tree: ast.AST, embedded: bool) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "subprocess"
            and name in SUBPROCESS
            and (is_true(kw.get("text", ast.Constant(False)))
                 or is_true(kw.get("universal_newlines", ast.Constant(False))))
            and "encoding" not in kw
        ):
            found.append((node.lineno, f"subprocess.{name}(text=True) without encoding="))
        # read_text/write_text on any object, not only where the type is known to
        # be a Path: ruff's PLW1514 sees `Path(...).read_text()` but not `meta.read_text()`
        # read_text(encoding) and write_text(data, encoding) may also take it by position
        positional = len(node.args) > (0 if name == "read_text" else 1)
        if (isinstance(f, ast.Attribute) and name in ("read_text", "write_text")
                and "encoding" not in kw and not positional):
            found.append((node.lineno, f".{name}() without encoding="))
        if not embedded:
            continue  # ruff's PLW1514 covers the open() builtin in .py files
        # open(file, mode, buffering, encoding): the fourth by position
        if (isinstance(f, ast.Name) and name == "open" and "encoding" not in kw
                and len(node.args) < 4):
            mode = kw.get("mode") or (node.args[1] if len(node.args) > 1 else None)
            binary = (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
                      and "b" in mode.value)
            if not binary:
                found.append((node.lineno, "open() in text mode without encoding="))
    return found


def snippets(text: str):
    """(line, source) of the Python a shell script carries."""
    for m in HEREDOC.finditer(text):
        yield text.count("\n", 0, m.start(3)) + 1, m.group(3)
    for m in DASH_C_SINGLE.finditer(text):
        yield text.count("\n", 0, m.start(1)) + 1, m.group(1)
    for m in DASH_C_DOUBLE.finditer(text):
        # what the shell hands python from inside double quotes
        body = re.sub(r'\\(["\\$`])', r"\1", m.group(1))
        yield text.count("\n", 0, m.start(1)) + 1, body


def powershell(text: str) -> list[tuple[int, str]]:
    """Get-Content and the like without -Encoding, a statement at a time
    (lines ending in a backtick continue on the next)."""
    found = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        first, statement = i + 1, lines[i]
        while statement.rstrip().endswith("`") and i + 1 < len(lines):
            i += 1
            statement = statement.rstrip()[:-1] + " " + lines[i]
        code = statement.split("#", 1)[0]
        for m in PS_TEXT.finditer(code):
            if not re.search(r"-Encoding\b", code, re.IGNORECASE):
                found.append((first, f"{m.group(1)} without -Encoding"))
                break
        i += 1
    return found


def files(paths: list[Path]):
    for base in paths:
        candidates = [base] if base.is_file() else base.rglob("*")
        for p in candidates:
            if not p.is_file() or SKIP.intersection(p.parts):
                continue
            if any(p.resolve().is_relative_to(s) for s in SKIP_UNDER):
                continue
            if p.suffix in (".py", ".sh", ".slurm", ".ps1", ".psm1"):
                yield p


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [ROOT]
    bad = 0
    for path in sorted(files(paths)):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix in (".ps1", ".psm1"):
            places = powershell(text)
        elif path.suffix == ".py":
            try:
                found = problems(ast.parse(text), embedded=False)
            except SyntaxError:
                continue
            places = [(line, why) for line, why in found]
        else:
            places = []
            for first, source in snippets(text):
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue  # shell expansions in an unquoted here-document: not Python as written
                places += [(first + line - 1, why) for line, why in problems(tree, embedded=True)]
        for line, why in places:
            shown = (path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT)
                     else path)
            print(f"{shown}:{line}: {why}")
            bad += 1
    if bad:
        print(f"\n{bad} place(s) read or write text without naming its encoding: "
              "add encoding=\"utf-8\""
              " (and errors=\"replace\" where the bytes may not be UTF-8).", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
