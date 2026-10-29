"""Answer exam prompts with a GGUF file, through llama.cpp's llama-server.

What ships is a Q4_K_M GGUF, and every exam so far graded the bf16 weights
it was made from. Quantization costs something, usually little; how much on
these questions is a number, not an assumption, and it is measured by asking
the GGUF the same exam the same way:

* the prompts are built and tokenized by the HF tokenizer of the merged
  model, exactly as for the bf16 run, and sent to the server as token ids,
  so the server adds nothing (no BOS of its own, no template of its own);
* decoding is greedy, the prompt cache off, one request per prompt;
* the generated ids come back (``return_tokens``) and are decoded by the
  same tokenizer, with the same "ended its turn" rule.

The difference between the two answers files is then the quantization and
the runtime, nothing else, and compare_graded.py says whether it matters.

The server is started for one run and stopped after it. Its log goes to a
file and only its last lines are printed when it fails: they name the
problem, never a prompt, so this is safe on the held-out exam.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def default_server_binary() -> str:
    """$LCPP_SERVER, else the CUDA build next to the CPU one in $LCPP_DIR."""
    if os.environ.get("LCPP_SERVER"):
        return os.environ["LCPP_SERVER"]
    lcpp = os.environ.get("LCPP_DIR") or str(Path(os.environ.get("WORK", "~")) / "llama.cpp")
    return str(Path(lcpp).expanduser() / "build-cuda" / "bin" / "llama-server")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class LlamaServer:
    """A llama-server process on a free local port, for the length of a ``with``.

    ``parallel`` requests are decoded together; each has ``ctx_per_slot``
    tokens of context for its prompt and answer.
    """

    def __init__(self, gguf: str | Path, *, binary: str | None = None, parallel: int = 4,
                 ctx_per_slot: int = 16384, log_path: str | Path | None = None,
                 start_timeout: float = 900.0, devices: str | None = None):
        self.gguf = Path(gguf)
        self.binary = binary or default_server_binary()
        self.parallel = parallel
        self.ctx_per_slot = ctx_per_slot
        self.log_path = Path(log_path) if log_path else self.gguf.with_suffix(".server.log")
        self.start_timeout = start_timeout
        self.devices = devices          # CUDA_VISIBLE_DEVICES for this server only
        self.port = 0
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [self.binary, "-m", str(self.gguf), "--host", "127.0.0.1",
                "--port", str(self.port), "-ngl", "999", "-np", str(self.parallel),
                "-c", str(self.parallel * self.ctx_per_slot), "--no-webui"]

    def log_tail(self, n: int = 15) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            return "\n".join(text.splitlines()[-n:])
        except OSError:
            return "(no server log)"

    def __enter__(self) -> LlamaServer:
        if not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"no llama-server at {self.binary} -- build it with "
                                    "sbatch_build_llama_server.slurm, or set LCPP_SERVER")
        if not self.gguf.is_file():
            raise FileNotFoundError(f"no GGUF at {self.gguf}")
        self.port = _free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        env = None
        if self.devices is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.devices}
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        deadline = time.monotonic() + self.start_timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}) while "
                                   f"loading:\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass          # 503 while loading, refused before it listens
            if time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise TimeoutError(f"llama-server not ready after {self.start_timeout:.0f} s:"
                                   f"\n{self.log_tail()}")
            time.sleep(1)

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def complete(self, ids: list[int], max_new_tokens: int, temperature: float = 0.0,
                 seed: int | None = None) -> dict:
        """One completion of a tokenized prompt: the server's JSON reply.

        Greedy by default, as the exams ask. With a ``temperature`` it samples
        from the whole distribution (no top-k, top-p or min-p cut), ``seed``
        making each sample repeatable: several answers to one question, for
        sample_answers.py.
        """
        body = {"prompt": ids, "n_predict": max_new_tokens, "temperature": 0.0,
                "top_k": 1, "cache_prompt": False, "return_tokens": True}
        if temperature > 0:
            body.update(temperature=temperature, top_k=0, top_p=1.0, min_p=0.0)
            if seed is not None:
                body["seed"] = seed
        try:
            return _post(self.url + "/completion", body, timeout=3600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"llama-server refused a prompt of {len(ids)} tokens "
                               f"({e.code}): {detail}") from None


def generate_answers_gguf(server, tok, prompts: list[str], *, max_new_tokens: int,
                          end_ids: list[int], parallel: int = 4, temperature: float = 0.0,
                          seeds: list[int] | None = None,
                          keep_reasoning: bool = False) -> list[tuple[str, bool]]:
    """(answer, ended_its_turn) per templated prompt, in order, like generate_answers.

    A turn has ended when the server stopped on an end-of-generation token
    rather than on the token budget, or when an end token is among the ids it
    returns (cut there, as the bf16 path does). ``temperature`` and one seed
    per prompt sample instead of decoding greedily (`LlamaServer.complete`).
    ``keep_reasoning`` keeps the model's [THINK]/[/THINK] markers, which
    skip_special_tokens would delete before split_reasoning can see them.
    """
    def one(job: tuple[str, int | None]) -> tuple[str, bool]:
        prompt, seed = job
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r = server.complete(ids, max_new_tokens, temperature=temperature, seed=seed)
        if r.get("truncated"):
            raise RuntimeError(f"a prompt of {len(ids)} tokens did not fit the slot's "
                               "context: raise --server-ctx")
        out = list(r.get("tokens") or [])
        cut = next((j for j, t in enumerate(out) if t in end_ids), None)
        ended = cut is not None or r.get("stop_type") == "eos" or bool(r.get("stopped_eos"))
        if not out:           # a server that ignores return_tokens: its own text
            return (r.get("content") or "").strip(), ended
        text = tok.decode(out[:cut] if cut is not None else out,
                          skip_special_tokens=not keep_reasoning)
        return text.strip(), ended

    jobs = list(zip(prompts, seeds or [None] * len(prompts)))
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        return list(pool.map(one, jobs))


class EullmServer:
    """An EULLM Engine server on a free local port with one GGUF, for a ``with``.

    The same exam as `LlamaServer`, through the runtime that ships: the
    prompt templated by the merged model's HF tokenizer and sent as text with
    ``raw`` set, so the engine applies no template and adds no BOS of its own
    (a template's own ``<s>`` is in the text); greedy decoding, no repeat
    penalty. Its ``parallel`` slots of ``ctx_per_slot`` tokens decode the
    questions together, in continuous batching. The answer comes back as
    text, with ``done_reason`` saying whether the turn ended ("stop") or the
    token budget did ("length").
    """

    def __init__(self, gguf: str | Path, *, binary: str | None = None, parallel: int = 4,
                 ctx_per_slot: int = 16384, log_path: str | Path | None = None,
                 start_timeout: float = 900.0, devices: str | None = None):
        self.gguf = Path(gguf).resolve()
        self.binary = binary or os.environ.get("EULLM_BIN", "eullm")
        self.parallel = parallel
        self.ctx_per_slot = ctx_per_slot
        self.log_path = Path(log_path) if log_path else self.gguf.with_suffix(".eullm.log")
        self.start_timeout = start_timeout
        self.devices = devices
        self.port = 0
        self.proc: subprocess.Popen | None = None

    url = LlamaServer.url
    log_tail = LlamaServer.log_tail
    __exit__ = LlamaServer.__exit__

    def command(self) -> list[str]:
        return [self.binary, "serve", "--port", str(self.port),
                "--default-model", str(self.gguf), "--batch-size", str(self.parallel),
                "--ctx-size", str(self.parallel * self.ctx_per_slot)]

    def __enter__(self) -> EullmServer:
        if not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"no eullm binary at {self.binary} -- set EULLM_BIN")
        if not self.gguf.is_file():
            raise FileNotFoundError(f"no GGUF at {self.gguf}")
        self.port = _free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        env = None
        if self.devices is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.devices}
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        deadline = time.monotonic() + self.start_timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"eullm exited ({self.proc.returncode}) while starting:"
                                   f"\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(self.url + "/api/version", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise TimeoutError(f"eullm not answering after {self.start_timeout:.0f} s:"
                                   f"\n{self.log_tail()}")
            time.sleep(1)
        # An empty prompt loads the model and generates nothing, so its load
        # time is not counted in the first question's.
        try:
            _post(self.url + "/api/generate", {"model": str(self.gguf), "prompt": ""},
                  timeout=self.start_timeout)
        except (urllib.error.URLError, OSError) as e:
            self.__exit__(None, None, None)
            raise RuntimeError(f"eullm did not load {self.gguf.name}: {e}\n{self.log_tail()}")
        return self

    def complete_text(self, prompt: str, max_new_tokens: int) -> dict:
        """One greedy completion of a templated prompt: the server's JSON reply."""
        body = {"model": str(self.gguf), "prompt": prompt, "raw": True, "stream": False,
                "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
                            "repeat_penalty": 1.0, "num_predict": max_new_tokens}}
        try:
            return _post(self.url + "/api/generate", body, timeout=3600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"eullm refused a prompt ({e.code}): {detail}") from None


def generate_answers_eullm(server, prompts: list[str], *, max_new_tokens: int,
                           parallel: int = 4) -> list[tuple[str, bool]]:
    """(answer, ended_its_turn) per templated prompt, in order, through EULLM Engine.

    ``parallel`` requests in flight at once, as many as the server has slots,
    so its continuous batching decodes them together.
    """
    def one(prompt: str) -> tuple[str, bool]:
        r = server.complete_text(prompt, max_new_tokens)
        return (r.get("response") or "").strip(), r.get("done_reason") == "stop"

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        return list(pool.map(one, prompts))
