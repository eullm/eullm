#!/usr/bin/env python3
"""Release smoke test for the eullm engine — verifies a *published binary*.

Runs a fixed set of checks against a real engine process and prints a pass/fail
table plus a report file you can attach to an issue. Standard library only: no
pip install, so it works unchanged on a Raspberry Pi or a bare ARM box.

Two things it deliberately does differently from a hand-run test:

* **It never uses ``--daemon``.** In daemon mode the engine redirects its stdout
  and stderr to ``<pidfile>.log`` (``/tmp/eullm.log`` by default), so the startup
  banner — GPU backend, CPU instruction set, context layout, KV cache types —
  and every warning end up in a file that ad-hoc testing usually forgets to
  collect. That banner is the most useful platform-specific evidence there is.
  This script runs the engine in the foreground and captures the whole stream.
* **It separates FAIL from SKIP.** A check that could not run (no GPU, no model,
  UI disabled) is not a failure, and conflating the two is how a green run stops
  meaning anything.

Usage
-----
    # Minimal: pull a small model and run everything
    python3 tools/smoke_test.py --binary ./eullm --pull

    # Against a model already on disk
    python3 tools/smoke_test.py --binary ./eullm --model ./Qwen3-0.6B-Q4_K_M.gguf

    # On a CUDA box, additionally exercise --fit on a model whose GGUF declares
    # an explicit attention.key_length (this is the regression that made --fit
    # under-size the KV cache and die out of VRAM)
    python3 tools/smoke_test.py --binary ./eullm --pull --fit-model qwen3-4b

    # Quick pass, no inference (config/validation checks only, seconds)
    python3 tools/smoke_test.py --binary ./eullm --no-inference

Exit code is 0 only if nothing failed. Skips do not fail the run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "qwen3-0.6b"
STARTUP_TIMEOUT_S = 240  # model load on a slow CPU box can take a while


# ── result plumbing ─────────────────────────────────────────────────────────


@dataclass
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    engine_log: str = ""

    def add(self, name: str, ok: bool | None, detail: str = "") -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.results.append(Result(name, status, detail))
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[status]
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")


# ── HTTP helpers (stdlib only) ──────────────────────────────────────────────


def post(url: str, payload: dict, timeout: float = 300.0):
    """POST JSON. Returns (status_code, parsed_or_text). Never raises on 4xx/5xx."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except Exception as e:  # connection refused, timeout, ...
        return 0, str(e)


def get(url: str, timeout: float = 30.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw), dict(r.headers)
            except json.JSONDecodeError:
                return r.status, raw, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers)
    except Exception as e:
        return 0, str(e), {}


def stream_lines(url: str, payload: dict, timeout: float = 300.0, read_delay: float = 0.0):
    """POST and yield decoded lines as they arrive.

    ``read_delay`` sleeps between reads, which is how the slow-client
    backpressure check makes the engine's per-sequence channel fill up.
    """
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            chunk = r.read(64)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield line.decode("utf-8", "replace")
            if read_delay:
                time.sleep(read_delay)
        if buf.strip():
            yield buf.decode("utf-8", "replace")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── engine process ──────────────────────────────────────────────────────────


class Engine:
    """A foreground engine process with its full output captured."""

    def __init__(self, binary: Path, args: list[str], env: dict, log_path: Path):
        self.binary = binary
        self.args = args
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._lines: list[str] = []

    def __enter__(self) -> "Engine":
        self.log = open(self.log_path, "w", encoding="utf-8")
        env = {**os.environ, **self.env}
        # RUST_LOG on: the tracing lines carry the allowlist source, the audit
        # path, the KV-reuse diagnostics and any scheduler error.
        env.setdefault("RUST_LOG", "eullm=info")
        self.proc = subprocess.Popen(
            [str(self.binary), *self.args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain, daemon=True).start()
        return self

    def _drain(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._lines.append(line)
            self.log.write(line)
            self.log.flush()

    @property
    def output(self) -> str:
        return "".join(self._lines)

    def wait_ready(self, port: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                return False  # died during startup
            code, _, _ = get(f"http://127.0.0.1:{port}/api/version", timeout=2)
            if code == 200:
                return True
            time.sleep(0.3)
        return False

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        time.sleep(0.3)  # let the drain thread flush
        self.log.close()


# ── environment description ─────────────────────────────────────────────────


# ELF e_machine → nome leggibile. Serve per dire "hai scaricato l'asset
# sbagliato" invece di lasciare che execve fallisca con "Exec format error",
# che è vero ma non dice cosa fare.
_ELF_MACHINES = {0x03: "x86", 0x3E: "x86_64", 0x28: "arm", 0xB7: "aarch64",
                 0xF3: "riscv64", 0x15: "ppc64"}
# Come `platform.machine()` nomina le stesse architetture.
_UNAME_ALIASES = {"x86_64": {"x86_64", "amd64"}, "aarch64": {"aarch64", "arm64"},
                  "arm": {"armv7l", "armv6l", "arm"}, "x86": {"i386", "i686"}}


def binary_arch(binary: Path) -> str | None:
    """Architecture an ELF binary targets, or None if not an ELF."""
    try:
        with open(binary, "rb") as f:
            head = f.read(20)
    except OSError:
        return None
    if len(head) < 20 or head[:4] != b"\x7fELF":
        return None
    little = head[5] == 1
    e_machine = int.from_bytes(head[18:20], "little" if little else "big")
    return _ELF_MACHINES.get(e_machine, f"unknown(0x{e_machine:02x})")


def check_arch_match(rep: Report, binary: Path) -> bool:
    """False when the binary cannot possibly run here."""
    arch = binary_arch(binary)
    host = platform.machine()
    if arch is None:
        rep.add("binary is an ELF executable", None,
                "not an ELF file — skipping the architecture check "
                "(fine on macOS, where binaries are Mach-O)")
        return True
    ok = host in _UNAME_ALIASES.get(arch, {arch})
    rep.add(
        "binary architecture matches this host",
        ok,
        f"binary is {arch}, host is {host}" + ("" if ok else
        " — you downloaded the wrong release asset for this machine"),
    )
    return ok


def describe_env(binary: Path) -> dict:
    env = {
        "uname": " ".join(platform.uname()),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "binary": str(binary),
        "binary_sha256": "",
        "cpu_model": "",
        "nvidia_smi": "",
    }
    try:
        import hashlib

        h = hashlib.sha256()
        with open(binary, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        env["binary_sha256"] = h.hexdigest()
    except OSError:
        pass

    # CPU model, best effort per platform.
    try:
        if Path("/proc/cpuinfo").exists():
            txt = Path("/proc/cpuinfo").read_text()
            for key in ("model name", "Model", "Hardware", "cpu model"):
                m = re.search(rf"^{key}\s*:\s*(.+)$", txt, re.M)
                if m:
                    env["cpu_model"] = m.group(1).strip()
                    break
        elif shutil.which("sysctl"):
            env["cpu_model"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True,
            ).stdout.strip()
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            env["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except Exception:
            pass
    return env


# ── checks ──────────────────────────────────────────────────────────────────


def check_binary(rep: Report, binary: Path, expect_version: str | None) -> str:
    """Version string and build variant. Returns the raw `-V` output."""
    try:
        out = subprocess.run(
            [str(binary), "-V"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception as e:
        rep.add("binary runs (`-V`)", False, str(e))
        return ""
    rep.add("binary runs (`-V`)", bool(out), out)
    if expect_version:
        ok = expect_version in out
        rep.add(
            f"version string reports {expect_version}",
            ok,
            out if ok else f"expected {expect_version!r}, got {out!r} — "
            "the bump may not have reached the binary (this happened in 0.6.32)",
        )
    return out


def check_startup_banner_via_run(
    rep: Report, binary: Path, model: str, env: dict, log_dir: Path
) -> None:
    """Capture the platform banner, which only `eullm run` prints.

    `eullm serve` prints six lines and none of the diagnostics: no GPU backend,
    no CPU instruction set, no context layout, no KV cache types. Those live in
    the `run` path only (`main.rs` ~1823-1882, inside `cmd_run`). That is why
    this check spawns a second, short-lived `run` process instead of grepping the
    server's output — and it is also a real gap in the engine, tracked as H3-N in
    docs/backlog-fix-e-hardening.md: the `CPU features` line was added to help
    diagnose issue #140, and anyone testing through `serve` never sees it.
    """
    port = free_port()
    log = log_dir / "engine-run-banner.log"
    args = ["run", model, "--port", str(port), "--no-ui", "--cli", "--batch-size", "1"]
    with Engine(binary, args, env, log) as eng:
        if not eng.wait_ready(port, STARTUP_TIMEOUT_S):
            rep.add("startup banner captured", False,
                    "`eullm run` did not become ready; see " + str(log))
            return
        _check_banner_lines(rep, eng.output)


def _check_banner_lines(rep: Report, log: str) -> None:
    """Record the platform-specific lines. Informational, not pass/fail —
    except that a GPU build reporting a CPU-only backend is worth flagging."""
    wanted = {
        "GPU backend": r"GPU backend:\s*(.+)",
        "CPU features": r"CPU features:\s*(.+)",
        "GPU layers": r"GPU layers:\s*(.+)",
        "Context": r"Context:\s*(.+)",
        "KV cache": r"KV cache:\s*(.+)",
        "Threads": r"Threads:\s*(.+)",
    }
    found = {}
    for label, pat in wanted.items():
        m = re.search(pat, log)
        if m:
            found[label] = m.group(1).strip()
    if not found:
        rep.add(
            "startup banner captured",
            False,
            "no banner lines in `eullm run` output — expected GPU backend / CPU "
            "features / Context / KV cache",
        )
        return
    rep.add("startup banner captured", True, "; ".join(f"{k}={v}" for k, v in found.items()))

    backend = found.get("GPU backend", "")
    if backend:
        rep.add(
            "GPU backend reported",
            None if "none" in backend.lower() else True,
            backend + (" (CPU-only build or no GPU)" if "none" in backend.lower() else ""),
        )


def check_allowlist_source(rep: Report, log: str) -> None:
    m = re.search(r"Allowed source IPs/subnets:\s*(.+?)\s*\[source:\s*(.+?)\]", log)
    if not m:
        rep.add("allowlist source logged", False, "no 'Allowed source IPs' line in the log")
        return
    nets, source = m.group(1), m.group(2)
    rep.add(
        "EULLM_ALLOWED_IPS read from the environment",
        "environment" in source,
        f"{source} → {nets}",
    )
    rep.add(
        "loopback preserved alongside configured entries",
        "127.0.0.1/32" in nets,
        nets,
    )


def check_audit(rep: Report, log: str, audit_dir: Path, expect_records: bool) -> None:
    m = re.search(r"Audit trail:\s*(.+)", log)
    logged = m.group(1).strip() if m else ""
    inside = bool(logged) and str(audit_dir) in logged
    rep.add(
        "audit trail honours EULLM_AUDIT_DIR",
        inside,
        logged or "no 'Audit trail:' line in the log",
    )
    f = audit_dir / "audit.jsonl"
    rep.add("audit file created at startup", f.is_file(), str(f))
    if not expect_records:
        rep.add("audit records written", None, "no inference ran")
        return
    try:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        parsed = [json.loads(l) for l in lines]
    except Exception as e:
        rep.add("audit records written", False, f"unreadable: {e}")
        return
    rep.add("audit records written", len(parsed) > 0, f"{len(parsed)} record(s)")
    if parsed:
        # Known gap, tracked as H1-A: nothing populates user_id, so no record is
        # attributable. Reported as a skip so it stays visible without failing.
        rep.add(
            "audit records carry a user_id",
            None if all(p.get("user_id") is None for p in parsed) else True,
            "always null — no request identity exists yet (backlog H1-A)",
        )


def check_override_validation(rep: Report, api: str) -> None:
    """Out-of-range slot overrides must be refused with 400, in-range accepted."""
    bad = [
        ("batch_size 2**32", {"batch_size": 4294967296}),
        ("batch_size 0", {"batch_size": 0}),
        ("batch_size 65", {"batch_size": 65}),
        ("batch_size -1", {"batch_size": -1}),
        ("batch_size string", {"batch_size": "8"}),
        ("ctx_size 100", {"ctx_size": 100}),
        ("ctx_size 2**32-1", {"ctx_size": 4294967295}),
    ]
    failures = []
    for label, extra in bad:
        code, body = post(
            f"{api}/api/chat",
            {"model": "no-such-model", "messages": [], **extra},
            timeout=60,
        )
        if code != 400:
            failures.append(f"{label}→{code}")
    rep.add(
        "out-of-range slot overrides refused with 400",
        not failures,
        "all 7 cases" if not failures else "unexpected: " + ", ".join(failures),
    )

    # An in-range override must get *past* validation. With a nonexistent model
    # that surfaces as a load error, which is the point: it is no longer a 400.
    code, _ = post(
        f"{api}/api/chat",
        {"model": "no-such-model", "batch_size": 4, "ctx_size": 2048, "messages": []},
        timeout=60,
    )
    rep.add(
        "in-range overrides pass validation",
        code != 400,
        f"http={code} (a non-400 here means validation let it through)",
    )
    # Tracked as H2-I: this ought to be 404, not 500.
    rep.add(
        "unknown model returns 404",
        None if code == 500 else (code == 404),
        f"http={code} — currently 500, should be 404 (backlog H2-I)",
    )


def check_ui_headers(rep: Report, ui: str | None) -> None:
    if not ui:
        rep.add("chat UI security headers", None, "UI not enabled")
        return
    code, _, headers = get(ui + "/", timeout=30)
    if code != 200:
        rep.add("chat UI security headers", False, f"UI returned http={code}")
        return
    lower = {k.lower(): v for k, v in headers.items()}
    missing = [
        h for h in ("content-security-policy", "x-content-type-options", "referrer-policy")
        if h not in lower
    ]
    rep.add(
        "chat UI security headers",
        not missing,
        "CSP + nosniff + referrer-policy" if not missing else f"missing: {missing}",
    )


def check_inference(rep: Report, api: str, model: str) -> dict:
    """Non-streaming, NDJSON and SSE round-trips. Returns timing info."""
    timings: dict = {}

    t0 = time.time()
    code, body = post(
        f"{api}/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: ciao"}],
            "think": False,
            "stream": False,
            "options": {"num_predict": 24, "temperature": 0},
        },
    )
    dt = time.time() - t0
    ok = code == 200 and isinstance(body, dict) and body.get("message", {}).get("content")
    detail = f"http={code} {dt:.1f}s"
    if ok:
        gen = body.get("eval_count", 0)
        pro = body.get("prompt_eval_count", 0)
        detail = f"{gen} tok in {dt:.1f}s, prompt {pro} tok — {body['message']['content'][:40]!r}"
        timings["first_request_s"] = round(dt, 2)
        timings["gen_tokens"] = gen
    rep.add("inference: /api/chat non-streaming", bool(ok), detail)
    if not ok:
        return timings  # nothing else will work either

    # NDJSON: every line must be a standalone JSON object, last one done=true.
    try:
        lines = list(
            stream_lines(
                f"{api}/api/chat",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Count: one two three"}],
                    "think": False,
                    "stream": True,
                    "options": {"num_predict": 16, "temperature": 0},
                },
            )
        )
        objs = [json.loads(l) for l in lines]
        ok = bool(objs) and objs[-1].get("done") is True and all(
            "message" in o for o in objs
        )
        text = "".join(o.get("message", {}).get("content", "") for o in objs)
        rep.add(
            "inference: /api/chat NDJSON streaming",
            ok,
            f"{len(objs)} lines, done={objs[-1].get('done') if objs else None}, {text[:40]!r}",
        )
    except Exception as e:
        rep.add("inference: /api/chat NDJSON streaming", False, str(e))

    # OpenAI SSE: `data: ` prefixed chunks terminated by `data: [DONE]`.
    try:
        lines = list(
            stream_lines(
                f"{api}/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "think": False,
                    "stream": True,
                    "max_tokens": 12,
                    "temperature": 0,
                },
            )
        )
        data = [l[len("data: "):] for l in lines if l.startswith("data: ")]
        ok = bool(data) and data[-1].strip() == "[DONE]"
        terminator = data[-1].strip()[:16] if data else "(none)"
        rep.add(
            "inference: /v1/chat/completions SSE",
            ok,
            f"{len(data)} data lines, terminator={terminator!r}",
        )
    except Exception as e:
        rep.add("inference: /v1/chat/completions SSE", False, str(e))

    return timings


def check_concurrency(rep: Report, api: str, model: str, n: int = 8) -> None:
    """Each concurrent request must answer its OWN question.

    This is the check that matters most for the batching scheduler: if the
    per-sequence logit indices and the decode batch ever disagree, a sequence
    samples from another conversation's distribution and returns fluent text
    that belongs to someone else. Arithmetic makes that visible — a swapped
    answer is unmistakable, where swapped prose would not be.
    """
    answers: dict[int, str] = {}
    errors: list[str] = []

    def ask(k: int) -> None:
        code, body = post(
            f"{api}/api/chat",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"What is {k} multiplied by 10? "
                        "Reply with only the number, nothing else.",
                    }
                ],
                "think": False,
                "options": {"num_predict": 20, "temperature": 0},
            },
        )
        if code != 200 or not isinstance(body, dict):
            errors.append(f"{k}: http={code}")
            return
        answers[k] = body.get("message", {}).get("content", "")

    ks = list(range(2, 2 + n))
    threads = [threading.Thread(target=ask, args=(k,)) for k in ks]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0

    if errors:
        rep.add("concurrency: all requests answered", False, "; ".join(errors[:4]))
        return
    rep.add("concurrency: all requests answered", len(answers) == n, f"{len(answers)}/{n} in {dt:.1f}s")

    # The expected product must appear in that request's own answer.
    wrong = []
    for k in ks:
        expected = str(k * 10)
        if expected not in answers.get(k, ""):
            # Did it instead answer a *different* request's question?
            culprit = next(
                (str(j * 10) for j in ks if j != k and str(j * 10) in answers.get(k, "")),
                None,
            )
            wrong.append(
                f"{k}×10: got {answers.get(k, '')[:24]!r}"
                + (f" (that is {culprit}, another request's answer)" if culprit else "")
            )
    rep.add(
        "concurrency: no answer belongs to another request",
        not wrong,
        f"all {n} correct" if not wrong else "; ".join(wrong[:3]),
    )


def check_backpressure(rep: Report, api: str, model: str) -> None:
    """A slow reader must receive the same bytes as a fast one.

    Streamed pieces used to be dropped when the per-sequence channel filled up,
    so a client that read slowly got a reply with text missing from the middle
    and no error at all. Fixing the seed and temperature makes the two runs
    byte-comparable.

    Honest limitation: the per-sequence channel holds 256 events, so this only
    reaches the full-channel branch when generation outruns the reader for more
    than 256 tokens. On a slow CPU box the reader usually keeps up and the branch
    is never entered — so a PASS here means "slow reading did not corrupt the
    stream", not necessarily "the backpressure path was exercised". The detail
    line reports the event count against that 256 capacity so you can tell which
    of the two you got.
    """
    CHANNEL_CAPACITY = 256
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "List every number from 1 to 200, comma separated, nothing else.",
            }
        ],
        "think": False,
        "stream": True,
        "options": {"num_predict": 500, "temperature": 0, "seed": 42},
    }

    def collect(delay: float) -> tuple[str, int]:
        out, events = [], 0
        for line in stream_lines(f"{api}/api/chat", payload, read_delay=delay):
            events += 1
            try:
                out.append(json.loads(line).get("message", {}).get("content", ""))
            except json.JSONDecodeError:
                pass
        return "".join(out), events

    try:
        fast, n_fast = collect(0.0)
        slow, n_slow = collect(0.02)
    except Exception as e:
        rep.add("slow client receives the full stream", False, str(e))
        return

    if not fast:
        rep.add("slow client receives the full stream", None, "fast run produced no text")
        return

    exercised = "backpressure path plausibly hit" if n_fast > CHANNEL_CAPACITY else (
        f"only {n_fast} events (< {CHANNEL_CAPACITY} channel capacity) — "
        "full-channel branch probably not reached"
    )
    rep.add(
        "slow client receives the full stream",
        fast == slow,
        f"{len(fast)} bytes both ways; {exercised}"
        if fast == slow
        else f"fast={len(fast)}B/{n_fast}ev slow={len(slow)}B/{n_slow}ev — "
        "text diverged, pieces may be dropped",
    )


def check_fit(rep: Report, binary: Path, model: str, env: dict, log_dir: Path) -> None:
    """`--fit` must size the offload without dying out of VRAM.

    CUDA-only by construction: on any other build VRAM cannot be probed, so the
    engine reports "could not size the model" and falls back to --gpu-layers.
    The regression this guards: the KV cache was sized from n_embd/n_head, which
    under-estimates it on architectures that declare attention.key_length
    explicitly (qwen3-4b: 80 assumed vs 128 real), so --fit offloaded more layers
    than fit and the load died — the exact failure --fit exists to prevent.
    """
    port = free_port()
    log = log_dir / "engine-fit.log"
    args = [
        "run", model, "--fit", "--ctx-size", "32768",
        "--port", str(port), "--no-ui", "--cli",
    ]
    with Engine(binary, args, env, log) as eng:
        ready = eng.wait_ready(port, STARTUP_TIMEOUT_S)
        out = eng.output
        if not ready:
            oom = bool(re.search(r"out of memory|OOM|failed to allocate", out, re.I))
            rep.add(
                f"--fit loads {model} without running out of VRAM",
                False,
                "engine did not become ready"
                + (" — OOM in the log, which is the H2-A regression" if oom else ""),
            )
            return
        fit_line = re.search(r"\[EULLM\] --fit:.*", out)
        decision = fit_line.group(0).strip() if fit_line else "(no --fit line)"
        if "could not size the model" in decision:
            rep.add(f"--fit sizes {model}", None, decision + " (expected off CUDA)")
        else:
            rep.add(f"--fit sizes {model}", True, decision)
        code, body = post(
            f"http://127.0.0.1:{port}/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "think": False,
                "options": {"num_predict": 8, "temperature": 0},
            },
        )
        rep.add(
            f"--fit: {model} serves a request after loading",
            code == 200,
            f"http={code}",
        )


# ── main ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Release smoke test for the eullm engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n")[1] if "Usage" in __doc__ else None,
    )
    ap.add_argument("--binary", required=True, type=Path, help="path to the eullm binary")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"catalog id or .gguf path for the inference checks (default: {DEFAULT_MODEL})")
    ap.add_argument("--pull", action="store_true",
                    help="run `eullm pull <model>` first if it is not on disk")
    ap.add_argument("--fit-model", default=None, metavar="ID",
                    help="additionally exercise --fit with this model (CUDA boxes: try qwen3-4b)")
    ap.add_argument("--no-inference", action="store_true",
                    help="config/validation checks only — no model needed")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel requests (default 8)")
    ap.add_argument("--batch-size", type=int, default=4, help="scheduler slots (default 4)")
    ap.add_argument("--expect-version", default=None, metavar="X.Y.Z",
                    help="fail if `-V` does not contain this version")
    ap.add_argument("--models-dir", type=Path, default=None, metavar="DIR",
                    help="model store to use (default: ~/.cache/eullm-smoke/models). "
                         "Persistent on purpose, so a pulled model is reused across runs "
                         "instead of re-downloading it. Point it at your real "
                         "~/.eullm/models to use models you already have.")
    ap.add_argument("--report", type=Path, default=Path("eullm-smoke-report.json"))
    ap.add_argument("--keep-workdir", action="store_true",
                    help="do not delete the temporary models/audit directories")
    args = ap.parse_args(argv)

    binary = args.binary.resolve()
    if not binary.is_file():
        print(f"error: {binary} is not a file", file=sys.stderr)
        return 2
    if not os.access(binary, os.X_OK):
        print(f"error: {binary} is not executable (chmod +x it)", file=sys.stderr)
        return 2

    rep = Report()
    work = Path(tempfile.mkdtemp(prefix="eullm-smoke-"))
    audit_dir = work / "audit"
    audit_dir.mkdir()
    # The audit directory is throwaway (each run must prove it gets created and
    # written), but the model store must NOT be: a fresh store means every run
    # re-downloads hundreds of megabytes, which is unreasonable on a small ARM
    # box and was the first thing that went wrong when this script was tested.
    models_dir = (args.models_dir or Path.home() / ".cache" / "eullm-smoke" / "models").expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    # A non-loopback entry in the environment is what proves the variable is
    # read from there and not only from a .env file; loopback must survive it,
    # or these very checks could not reach the API.
    env = {
        "EULLM_MODELS_DIR": str(models_dir),
        "EULLM_AUDIT_DIR": str(audit_dir),
        "EULLM_ALLOWED_IPS": "203.0.113.5,192.168.7.0/24",
    }

    print(f"\neullm smoke test — workdir {work}")
    rep.env = describe_env(binary)
    for k, v in rep.env.items():
        if v:
            print(f"  {k}: {v}")

    smi = rep.env.get("nvidia_smi", "")
    if smi and ("version mismatch" in smi.lower() or "couldn't communicate" in smi.lower()):
        rep.add(
            "nvidia-smi usable",
            None,
            smi.splitlines()[0][:110] + " — a CUDA build will not find a GPU until "
            "this is resolved (usually a reboot after a driver update)",
        )

    print("\n── binary ──")
    if not check_arch_match(rep, binary):
        print("\nthe binary cannot execute on this host; stopping here.")
        return finish(rep, args, work, keep=args.keep_workdir)
    version_out = check_binary(rep, binary, args.expect_version)
    rep.env["version_output"] = version_out

    model = args.model
    if not args.no_inference:
        is_path = Path(model).exists()
        on_disk = is_path or any((models_dir / model).glob("*.gguf"))
        if not on_disk and args.pull:
            print(f"\n── pulling {model} into {models_dir} ──")
            try:
                r = subprocess.run(
                    [str(binary), "pull", model],
                    env={**os.environ, **env}, capture_output=True, text=True, timeout=7200,
                )
                tail = (r.stdout or r.stderr).strip().splitlines()
                rep.add(f"pull {model}", r.returncode == 0, tail[-1][:120] if tail else "")
                on_disk = r.returncode == 0
            except (OSError, subprocess.SubprocessError) as e:
                rep.add(f"pull {model}", False, f"could not run the binary: {e}")
                on_disk = False
        elif not on_disk:
            rep.add(
                f"model {model} available",
                False,
                f"not a path and no *.gguf under {models_dir / model} — "
                "pass --pull to fetch it, or --models-dir to point at a store that has it",
            )
        if not on_disk:
            args.no_inference = True

    port = free_port()
    ui_port = free_port()
    api = f"http://127.0.0.1:{port}"
    ui = f"http://127.0.0.1:{ui_port}"
    serve_args = [
        "serve", "--port", str(port), "--ui", "--ui-port", str(ui_port),
        "--batch-size", str(args.batch_size),
    ]

    print("\n── server ──")
    log_path = work / "engine-serve.log"
    with Engine(binary, serve_args, env, log_path) as eng:
        if not eng.wait_ready(port, 90):
            rep.add("server becomes ready", False, "no response on /api/version")
            print("\n" + eng.output[-2000:])
            rep.engine_log = eng.output
            return finish(rep, args, work, keep=True)
        rep.add("server becomes ready", True, api)

        log = eng.output
        check_allowlist_source(rep, log)
        check_ui_headers(rep, ui)

        print("\n── request validation ──")
        check_override_validation(rep, api)

        timings: dict = {}
        if args.no_inference:
            rep.add("inference checks", None, "--no-inference")
            check_audit(rep, eng.output, audit_dir, expect_records=False)
        else:
            print("\n── inference ──")
            timings = check_inference(rep, api, model)
            print("\n── concurrency ──")
            check_concurrency(rep, api, model, args.concurrency)
            print("\n── backpressure ──")
            check_backpressure(rep, api, model)
            print("\n── audit ──")
            check_audit(rep, eng.output, audit_dir, expect_records=True)

        rep.env["timings"] = timings
        rep.engine_log = eng.output

    if not args.no_inference:
        # Separate `run` process: the banner is not on the `serve` path at all.
        print("\n── platform banner (via `eullm run`) ──")
        check_startup_banner_via_run(rep, binary, model, env, work)

    if args.fit_model and not args.no_inference:
        print("\n── --fit ──")
        if args.pull:
            subprocess.run([str(binary), "pull", args.fit_model],
                           env={**os.environ, **env}, capture_output=True, timeout=7200)
        check_fit(rep, binary, args.fit_model, env, work)

    return finish(rep, args, work, keep=args.keep_workdir)


def finish(rep: Report, args, work: Path, keep: bool) -> int:
    print("\n" + "─" * 72)
    counts = {s: sum(1 for r in rep.results if r.status == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"PASS {counts['PASS']}   FAIL {counts['FAIL']}   SKIP {counts['SKIP']}")
    if counts["FAIL"]:
        print("\nfailed:")
        for r in rep.results:
            if r.status == "FAIL":
                print(f"  - {r.name}: {r.detail}")

    payload = {
        "env": rep.env,
        "results": [{"name": r.name, "status": r.status, "detail": r.detail} for r in rep.results],
        "counts": counts,
        "engine_log": rep.engine_log,
    }
    args.report.write_text(json.dumps(payload, indent=2))
    print(f"\nreport written to {args.report}  (includes the full engine log — "
          "attach this when reporting a problem)")

    if keep:
        print(f"workdir kept at {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
