#!/usr/bin/env python3
"""Check that a binary still answers correctly under each KV cache setting.

The release gate of `release-engine.yml`: before anything is published, the
Linux CPU binary of the release answers questions with known answers, at
temperature 0, once for each key/value cache type and flash attention setting
below. An answer passes when it holds the expected word and the model stopped
on its own (`done_reason` "stop"): a model that runs to the token limit is
what gibberish looks like from outside, and timing alone took it for a slow
success.

Why it exists: from 0.9.50 to 0.10.0, `--cache-type-k q4_0 --no-flash-attn`
wrote gibberish on every machine a tester ran it on (issue #140, reported on
github.com, 24-11-2026), and three releases went out with it because the
release workflow built the binaries and published them without running one.
The unit tests could not see it: it needs a model loaded and a cache setting
other than the default.

Standard library only, like smoke_test.py, so it also runs on a Pi:

    python3 tools/answer_check.py --binary ./eullm --pull
    python3 tools/answer_check.py --binary ./eullm --only q4_0-q4_0-nofa

Exit status 0 when every answer passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen3-0.6b"

# (label, extra flags). The default, what people use to save memory, and the
# combinations that have produced nonsense before (four-bit keys, with and
# without flash attention: the engine must raise them to q8_0).
SETTINGS: list[tuple[str, list[str]]] = [
    ("f16-f16", []),
    ("q8_0-f16", ["--cache-type-k", "q8_0"]),
    ("q8_0-q8_0", ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]),
    ("q8_0-q4_0", ["--cache-type-k", "q8_0", "--cache-type-v", "q4_0"]),
    ("q4_0-q4_0", ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]),
    ("q4_1-f16", ["--cache-type-k", "q4_1"]),
    ("f16-f16-nofa", ["--no-flash-attn"]),
    ("q8_0-f16-nofa", ["--cache-type-k", "q8_0", "--no-flash-attn"]),
    ("q4_0-q4_0-nofa", ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--no-flash-attn"]),
    ("q5_0-f16-nofa", ["--cache-type-k", "q5_0", "--no-flash-attn"]),
]

# (question, word the answer must contain, lowercased)
QUESTIONS: list[tuple[str, str]] = [
    ("What is the capital of France? Answer in one sentence.", "paris"),
    ("What is 12 times 12? Answer with the number only.", "144"),
    ("Which planet is known as the Red Planet? Answer in one word.", "mars"),
]

NUM_PREDICT = 160


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_up(api: str, proc: subprocess.Popen, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{api}/api/tags", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def kv_line(log: str) -> str:
    for line in log.splitlines():
        if "KV cache:" in line:
            return line.split("KV cache:", 1)[1].strip()
    return "?"


def run_setting(binary: Path, env: dict, label: str, flags: list[str], logdir: Path,
                timeout: float) -> tuple[bool, list[str], str]:
    port = free_port()
    api = f"http://127.0.0.1:{port}"
    logfile = logdir / f"{label}.log"
    with open(logfile, "w", encoding="utf-8", errors="replace") as out:
        proc = subprocess.Popen([str(binary), "serve", "--port", str(port), *flags],
                                stdout=out, stderr=subprocess.STDOUT, env=env)
    lines: list[str] = []
    ok = True
    try:
        if not wait_up(api, proc, 120):
            return False, ["server did not start"], logfile.read_text(encoding="utf-8",
                                                                      errors="replace")
        for question, want in QUESTIONS:
            try:
                r = post(f"{api}/api/chat", {
                    "model": MODEL, "stream": False, "think": False,
                    "options": {"temperature": 0, "num_predict": NUM_PREDICT},
                    "messages": [{"role": "user", "content": question}],
                }, timeout)
            except (urllib.error.URLError, OSError, ValueError) as e:
                ok = False
                lines.append(f"  FAIL {question!r}: {e}")
                continue
            text = (r.get("message") or {}).get("content", "")
            done = r.get("done_reason")
            passed = want in text.lower() and done == "stop"
            ok &= passed
            shown = " ".join(text.split())[:100]
            lines.append(f"  {'ok  ' if passed else 'FAIL'} {question!r} -> {shown!r} "
                         f"({done}, {r.get('eval_count')} tokens)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return ok, lines, logfile.read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path, help="the eullm binary to check")
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="EULLM_MODELS_DIR for the run (default: the engine's own)")
    ap.add_argument("--pull", action="store_true", help=f"pull {MODEL} first if missing")
    ap.add_argument("--only", action="append", default=[], metavar="LABEL",
                    help="run only these settings (repeatable); "
                         + ", ".join(label for label, _ in SETTINGS))
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds allowed for one answer")
    args = ap.parse_args(argv)

    binary = args.binary.resolve()
    env = dict(os.environ)
    if args.models_dir:
        args.models_dir.mkdir(parents=True, exist_ok=True)
        env["EULLM_MODELS_DIR"] = str(args.models_dir.resolve())
    version = subprocess.run([str(binary), "-V"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=env).stdout.strip()
    print(f"{version}, {MODEL}, temperature 0")
    if args.pull:
        p = subprocess.run([str(binary), "pull", MODEL], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, timeout=1800)
        if p.returncode != 0:
            print(p.stdout + p.stderr)
            print(f"could not pull {MODEL}")
            return 1

    settings = [(label, flags) for label, flags in SETTINGS
                if not args.only or label in args.only]
    unknown = set(args.only) - {label for label, _ in SETTINGS}
    if unknown:
        print(f"unknown setting(s): {', '.join(sorted(unknown))}")
        return 2

    failed = []
    with tempfile.TemporaryDirectory(prefix="answer-check-") as tmp:
        for label, flags in settings:
            t0 = time.monotonic()
            ok, lines, log = run_setting(binary, env, label, flags, Path(tmp), args.timeout)
            print(f"{'PASS' if ok else 'FAIL'} {label:<16} KV cache {kv_line(log)} "
                  f"({time.monotonic() - t0:.0f} s)")
            print("\n".join(lines))
            if not ok:
                failed.append(label)
                print("  --- engine output ---")
                print("\n".join("  " + ln for ln in log.splitlines()[-40:]))
    if failed:
        print(f"\n{len(failed)} of {len(settings)} settings answered wrongly or ran to the "
              f"token limit: {', '.join(failed)}")
        return 1
    print(f"\nall {len(settings)} settings answered correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
