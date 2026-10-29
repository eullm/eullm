"""Measuring one point: start its servers, warm them, measure, stop them.

Two kinds of point.

`throughput` is the method docs/cineca/leonardo.md established and the LUMI
scripts kept: N concurrent /api/generate requests, one prompt, aggregate
output tokens over the wall clock of the batch, a warm-up first and outside
the timed window — repeated, so a number carries its own spread. Streaming
adds what the proposal promised and nobody recorded: time to first token and
the prefill and decode rates per request.

`workload` is sustained load with answers that can be checked: the public
sets ReflexBench pins (GSM8K, ARC, MMLU), sent at a fixed concurrency for a
duration, looped. Every interval is a row of throughput, latency and HBM —
stability under sustained load, the proposal's own words — and the first pass
is graded, so each configuration also has the accuracy its memory and speed
cost. Later passes are compared with the first: decoding is greedy and the
prompt cache is off, so an answer that changes under load is a finding.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from devices import VISIBLE_ENV, cores_for

HERE = os.path.dirname(os.path.abspath(__file__))
REFLEXBENCH = os.path.join(os.path.dirname(HERE), "reflexbench")

# What this runner can run. A point planned by newer code carries `runner`
# above this, or a kind or runtime this file does not know, and is left in
# the queue for a runner that does: jobs keep the code they started with
# for up to 48 hours, while plan adds points at any time.
RUNNER_VERSION = 8
# The engine build this job runs, as `plan --engine-label` names it
# (EULLM_ENGINE_LABEL, exported by submit_campaign.sh). A point planned with a
# label runs only in a job with the same one, and a point planned without one
# only in a job without one: a queue can hold rounds for a new build while
# jobs started with the old binary are still taking points from it, and each
# round is measured on the engine it was planned for.
ENGINE_LABEL = os.environ.get("EULLM_ENGINE_LABEL") or None
KINDS = ("throughput", "workload", "finetune", "decision")
# The servers a point can measure: the engine, and for comparison the stock
# llama.cpp server (the same backend without EuLLM's runtime) and Ollama
# (the API EuLLM is compatible with). Their binaries come from the
# environment: LLAMA_SERVER_BIN, OLLAMA_BIN.
RUNTIMES = ("eullm", "llama-server", "ollama")
RUNTIME_BIN_ENV = {"llama-server": "LLAMA_SERVER_BIN", "ollama": "OLLAMA_BIN"}


def can_run(p: dict, engine_label=None) -> bool:
    label = ENGINE_LABEL if engine_label is None else engine_label or None
    return (p.get("runner", 1) <= RUNNER_VERSION and p.get("kind") in KINDS
            and p.get("runtime", "eullm") in RUNTIMES
            and (p.get("engine_label") or None) == label)


READY_TIMEOUT_S = 900
# How long a killed server may take to go. One unloading hundreds of GB, or
# stuck in a Lustre read, outlives SIGKILL by minutes; until it is gone its
# devices and its port are not free (c02's largest MoE, 06-10-2026).
KILL_WAIT_S = 600
REQUEST_TIMEOUT_S = 1800
WARMUP_TIMEOUT_S = 3600  # a 400 GB model read off Lustre is the slow case
# How long a throughput or decision point may run, servers started to last
# answer, before its servers are killed and it fails: this many times its
# estimate, and never less than the floor. The request timeout bounds one
# read, not a request: a server that streams a token every few minutes never
# reaches it, and DeepSeek-V3.1 with --cpu-moe on SSE2 kernels held a whole
# LUMI-G node for 9.8 hours that way (c10, 15-10-2026). Workload points
# stretch to the time they are given by design and finetune points run no
# server; neither has the limit.
POINT_LIMIT_FACTOR = 3
POINT_LIMIT_MIN_S = 3600

# Filler for synthetic long prompts: plain prose, so it tokenizes like text
# rather than like a repeated token the tokenizer could merge.
FILLER = (
    "Il mare era calmo quella mattina e le barche dei pescatori uscivano dal porto una "
    "dopo l'altra, mentre il sole saliva lentamente sopra le colline. Sulla banchina "
    "qualcuno riparava le reti, qualcun altro contava le casse vuote. "
)
LONG_PROMPT_TAIL = "\n\nRiassumi il testo precedente in una frase."
CHARS_PER_TOKEN = 4


class PointError(Exception):
    pass


class ModelMissing(PointError):
    """The model is not in the store: nothing this point can fix by retrying."""


class Interrupted(PointError):
    pass


class DoesNotFit(PointError):
    """--fit-strict refused the load: this configuration does not fit these
    devices. A measurement — where the memory boundary lies — not a failure."""


FIT_REFUSAL = "does not fully fit"


class RequestError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def is_missing_model(err: RequestError) -> bool:
    text = str(err).lower()
    return err.code == 404 or ("not found" in text and "model" in text)


# ── servers ──────────────────────────────────────────────────────────────


def server_args(p: dict) -> list:
    k, v = p["kv"].split("/")
    return [
        "--batch-size", str(p["batch"]),
        "--ctx-size", str(p["ctx"]),
        "--cache-type-k", k,
        "--cache-type-v", v,
    ] + list(p.get("extra_args", []))


def store_gguf(model_id: str):
    """The GGUF the EuLLM store holds for `model_id` (the first part of a
    split one), or None: llama-server is given the same file."""
    root = os.environ.get("EULLM_MODELS_DIR") or os.path.join(
        os.path.expanduser("~"), ".eullm", "models")
    d = os.path.join(root, model_id)
    try:
        with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
            name = json.load(f).get("gguf_file")
        if name and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    except (OSError, ValueError):
        pass
    try:
        files = sorted(n for n in os.listdir(d) if n.endswith(".gguf") and "mmproj" not in n)
    except OSError:
        return None
    return os.path.join(d, files[0]) if files else None


def mount_root(path: str):
    """The first component of a path once its links are followed: `/scratch`
    or `/flash` on LUMI, without the project and user below it."""
    parts = os.path.realpath(path).split(os.sep)
    return os.sep + parts[1] if len(parts) > 1 and parts[1] else None


def model_storage(p: dict):
    """Where the point's server reads its model from (`mount_root`): a load
    time means nothing without it, and tools/lumi/stage_flash.sh moves a
    model to flash behind links of the same names."""
    if p.get("runtime") == "ollama":
        path = os.environ.get("OLLAMA_MODELS")
    else:
        path = store_gguf(p["model"])
    return mount_root(path) if path else None


def model_files(p: dict) -> list:
    """Every GGUF in the store directory the point's server reads its model
    from (all the parts of a split one); none for Ollama, which reads its
    own copies."""
    path = None if p.get("runtime") == "ollama" else store_gguf(p["model"])
    if not path:
        return []
    d = os.path.dirname(path)
    return sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".gguf"))


def evict(paths) -> int:
    """Ask the kernel to drop these files from the page cache, so the next
    load reads them from the file system whatever ran on the node before (a
    `cold` point). Pages another process has mapped stay. The bytes asked
    for; 0 where posix_fadvise is missing."""
    total = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            total += os.fstat(fd).st_size
        except (OSError, AttributeError):
            pass
        finally:
            os.close(fd)
    return total


def cache_state(p: dict, ctx) -> str:
    """How the point found its model: `evicted` from the page cache on
    purpose (`cold`), else `cold` the first time this job loads it and
    `warm` after."""
    if p.get("cold"):
        return "evicted"
    return "warm" if p["model"] in ctx.model_seen else "cold"


def runtime_command(p: dict, engine: str):
    """(binary, arguments, extra environment) of the server a point measures.
    The same KV pool, slots and cache types for every runtime; EuLLM's own
    flags (`extra_args`) only for EuLLM, `runtime_args` for the others."""
    runtime = p.get("runtime", "eullm")
    if p["kind"] == "decision":
        # The decision model alone, loaded at startup; generation flags do
        # not apply to its slot.
        args = ["--decision-model", p["model"], "--decision-ctx", str(p["decision_ctx"])]
        return engine, args + list(p.get("extra_args", [])), {}
    if runtime == "eullm":
        return engine, server_args(p), {}
    binary = os.environ.get(RUNTIME_BIN_ENV[runtime])
    if not binary or not os.path.exists(binary):
        raise ModelMissing(f"{runtime}: set {RUNTIME_BIN_ENV[runtime]} to its binary")
    k, v = p["kv"].split("/")
    extra = list(p.get("runtime_args", []))
    if runtime == "llama-server":
        gguf = store_gguf(p["model"])
        if gguf is None:
            raise ModelMissing(f"model {p['model']} is not in the EuLLM store")
        return binary, [
            "-m", gguf, "--alias", p["model"], "-c", str(p["ctx"]), "-np", str(p["batch"]),
            "-ngl", "999", "-ctk", k, "-ctv", v,
        ] + extra, {}
    # Ollama: configured through its environment, the model by its name in
    # Ollama's own store (tools/lumi/make_ollama_models.sh).
    env = {
        "OLLAMA_NUM_PARALLEL": str(p["batch"]),
        "OLLAMA_CONTEXT_LENGTH": str(p["slot_ctx"]),
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_KEEP_ALIVE": "-1",
    }
    if (k, v) != ("f16", "f16"):
        env.update(OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE=k)
    return binary, extra, env


class Server:
    def __init__(self, engine, port, args, env, log_path, cores=None, runtime="eullm"):
        self.engine, self.port, self.args = engine, port, args
        self.env, self.log_path, self.cores = env, log_path, cores
        self.runtime = runtime
        self.proc = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ready_path(self) -> str:
        return "/health" if self.runtime == "llama-server" else "/api/version"

    def command(self) -> list:
        prefix = ["taskset", "-c", self.cores] if self.cores else []
        if self.runtime == "llama-server":
            host = ["--host", "127.0.0.1", "--port", str(self.port)]
            return prefix + [self.engine] + host + self.args
        if self.runtime == "ollama":
            return prefix + [self.engine, "serve"] + self.args
        return prefix + [self.engine, "serve", "--port", str(self.port)] + self.args

    def start(self) -> None:
        log = open(self.log_path, "ab")
        # A session of its own, so stopping it stops whatever it started too.
        # Run beside its log: a server that crashes leaves its core dump (and
        # ROCm its gpucore.<pid>) in its working directory, which was the
        # job's, the checkout in /projappl with its 50 GB quota (17-10-2026).
        self.proc = subprocess.Popen(
            self.command(), stdout=log, stderr=subprocess.STDOUT, env=self.env,
            start_new_session=True, cwd=os.path.dirname(os.path.abspath(self.log_path)),
        )
        log.close()

    def wait_ready(self, timeout_s=READY_TIMEOUT_S, stop=None) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if stop is not None and stop.is_set():
                raise Interrupted("stopped while waiting for the server")
            if self.proc.poll() is not None:
                tail = self.log_tail()
                if "not found." in tail and "model '" in tail:
                    raise ModelMissing(tail.strip().splitlines()[-1][:300])
                raise PointError(
                    f"server on port {self.port} exited ({self.proc.returncode}):\n"
                    + self.log_tail()
                )
            try:
                with urllib.request.urlopen(self.url + self.ready_path, timeout=5):
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1)
        raise PointError(f"server on port {self.port} not ready in {timeout_s} s:\n"
                         + self.log_tail())

    def stop(self, kill_wait_s=KILL_WAIT_S) -> bool:
        """Stop the server; whether it is gone. Never raises: it runs in the
        `finally` of a point, where an exception would replace the error
        that ended the point."""
        if self.proc is None or self.proc.poll() is not None:
            return True
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=30)
            return True
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            return True
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=kill_wait_s)
            return True
        except ProcessLookupError:
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill(self) -> None:
        """Immediately, without waiting: for a job whose time is up."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def log_tail(self, lines=40) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except OSError:
            return "(no server log)"


# ── requests ─────────────────────────────────────────────────────────────


def request(url, path, body, timeout=REQUEST_TIMEOUT_S) -> dict:
    """One Ollama-style request, streamed or not, read to the end."""
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    first, pieces, last = None, [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if body.get("stream", True):
                for raw in r:
                    if not raw.strip():
                        continue
                    line = json.loads(raw)
                    if "error" in line:
                        raise RequestError(str(line["error"]))
                    piece = line.get("response")
                    if piece is None:
                        piece = (line.get("message") or {}).get("content", "")
                    if piece and first is None:
                        first = time.perf_counter()
                    pieces.append(piece or "")
                    last = line
            else:
                last = json.loads(r.read())
                if "error" in last:
                    raise RequestError(str(last["error"]))
                message = last.get("message") or {}
                pieces.append(last.get("response") or message.get("content", ""))
    except urllib.error.HTTPError as e:
        raise RequestError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}",
                           code=e.code) from None
    ended = time.perf_counter()
    total_ms = (ended - started) * 1000
    return {
        "text": "".join(pieces),
        "ttft_ms": ((first or ended) - started) * 1000,
        "total_ms": total_ms,
        "eval_count": last.get("eval_count") or 0,
        "prompt_eval_count": last.get("prompt_eval_count") or 0,
        "eval_duration_ns": last.get("eval_duration") or 0,
        "prompt_eval_duration_ns": last.get("prompt_eval_duration") or 0,
        "load_duration_ns": last.get("load_duration") or 0,
        # With --mtp, the drafts the head proposed and the model kept.
        "draft_n": last.get("draft_n") or 0,
        "draft_n_accepted": last.get("draft_n_accepted") or 0,
    }


def accept_pct(drafts: int, kept: int):
    """The share of MTP drafts the model kept, in percent; None without
    drafts (no --mtp, or a runtime that does not report them)."""
    return round(100 * kept / drafts, 1) if drafts else None


def call(server, path, body, timeout=REQUEST_TIMEOUT_S) -> dict:
    """An Ollama-shaped request to whichever runtime `server` is: as it is
    to EuLLM and Ollama, translated to OpenAI's API for llama-server."""
    if getattr(server, "runtime", "eullm") == "llama-server":
        return openai_request(server.url, path, body, timeout)
    return request(server.url, path, body, timeout)


def openai_request(url, path, body, timeout=REQUEST_TIMEOUT_S) -> dict:
    """`body`, an Ollama /api/generate or /api/chat request, sent to
    llama-server's OpenAI endpoints; the answer in `request()`'s shape, its
    counts and timings from llama-server's `usage` and `timings`."""
    options = body.get("options") or {}
    n = options.get("num_predict") or body.get("num_predict") or 128
    stream = body.get("stream", True)
    payload = {"model": body["model"], "max_tokens": n, "stream": stream,
               "cache_prompt": body.get("cache_prompt", True)}
    for key in ("temperature", "top_k", "top_p", "seed"):
        if key in options:
            payload[key] = options[key]
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if path == "/api/chat":
        endpoint = "/v1/chat/completions"
        payload["messages"] = body["messages"]
        payload["chat_template_kwargs"] = {"enable_thinking": bool(body.get("think"))}
    else:
        endpoint = "/v1/completions"
        payload["prompt"] = body["prompt"]
    req = urllib.request.Request(url + endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first, pieces, usage, timings = None, [], {}, {}

    def take(chunk):
        nonlocal first
        if "error" in chunk:
            raise RequestError(str(chunk["error"]))
        for choice in chunk.get("choices") or []:
            piece = choice.get("text")
            if piece is None:
                piece = (choice.get("delta") or choice.get("message") or {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                pieces.append(piece)
        usage.update(chunk.get("usage") or {})
        timings.update(chunk.get("timings") or {})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if stream:
                for raw in r:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    take(json.loads(data))
            else:
                take(json.loads(r.read()))
    except urllib.error.HTTPError as e:
        raise RequestError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}",
                           code=e.code) from None
    ended = time.perf_counter()
    return {
        "text": "".join(pieces),
        "ttft_ms": ((first or ended) - started) * 1000,
        "total_ms": (ended - started) * 1000,
        "eval_count": usage.get("completion_tokens") or timings.get("predicted_n") or 0,
        "prompt_eval_count": usage.get("prompt_tokens") or timings.get("prompt_n") or 0,
        "eval_duration_ns": int((timings.get("predicted_ms") or 0) * 1e6),
        "prompt_eval_duration_ns": int((timings.get("prompt_ms") or 0) * 1e6),
        "load_duration_ns": 0,
        "draft_n": timings.get("draft_n") or 0,
        "draft_n_accepted": timings.get("draft_n_accepted") or 0,
    }


def concurrently(n, fn) -> list:
    """fn(0..n-1) on n threads at once; each slot a result or the exception."""
    out = [None] * n

    def run(i):
        try:
            out[i] = fn(i)
        except Exception as e:  # recorded, counted, and the batch goes on
            out[i] = e

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def point_env(p: dict, workdir: str) -> dict:
    """The point's `env` for its servers, with {workdir} (the job's directory
    of server logs) and {point} (the point's id) filled in: a file a server
    writes, such as a trace, lands beside its log and is the point's own."""
    return {k: v.replace("{workdir}", workdir).replace("{point}", p["id"])
            for k, v in (p.get("env") or {}).items()}


def synthetic_prompt(tokens: int, nonce: str) -> str:
    """About `tokens` tokens of prose, opened by a nonce so no two requests
    share a prefix a cache could reuse."""
    body = (FILLER * (tokens * CHARS_PER_TOKEN // len(FILLER) + 1))[: tokens * CHARS_PER_TOKEN]
    return f"[{nonce}] {body}{LONG_PROMPT_TAIL}"


def generate_body(p: dict, prompt: str) -> dict:
    body = {
        "model": p["model"],
        "prompt": prompt,
        "stream": p["stream"],
        "think": False,
        "num_predict": p["num_predict"],
        "options": {"num_predict": p["num_predict"]},
    }
    if p["prompt_tokens"]:
        body["cache_prompt"] = False
    return body


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def r1(x):
    return None if x is None else round(x, 1)


def rate(count, duration_s):
    return count / duration_s if count and duration_s and duration_s > 0 else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return r1(sum(xs) / len(xs)) if xs else None


# ── one point ────────────────────────────────────────────────────────────


class Context:
    """What the runner gives a point: where it runs and what it may use."""

    def __init__(self, engine, backend, binding, physical, port_base, workdir,
                 sampler=None, stop=None, model_seen=None, sets_dir=None, f32_dir=None,
                 cores=None):
        self.engine, self.backend, self.binding = engine, backend, binding
        # The cores every server of the point runs on, where the point takes
        # those of all the devices it reserved (`"cores": "reserved"`);
        # otherwise each server gets its own devices' (`cores_for`).
        self.cores = cores
        self.physical = list(physical)  # physical ids, in order
        self.port_base, self.workdir = port_base, workdir
        self.sampler = sampler
        self.stop = stop or threading.Event()
        self.model_seen = model_seen if model_seen is not None else set()
        self.sets_dir = sets_dir
        self.f32_dir = f32_dir  # F32 GGUFs for finetune points
        self.servers = []  # set by run(), so the runner can stop them from outside
        self.stragglers = []  # servers still alive after stop(): their devices stay taken


def start_servers(p: dict, ctx: Context) -> list:
    servers = []
    rg = p["replica_gcds"]
    for r in range(p["replicas"]):
        group = ctx.physical[r * rg : (r + 1) * rg]
        env = dict(os.environ)
        env[VISIBLE_ENV[ctx.backend]] = ",".join(group)
        if ctx.backend == "rocm":
            # HIP applies these on top of ROCR_VISIBLE_DEVICES, as indices into
            # what ROCR left visible: a job-wide "0,...,7" from Slurm would
            # point a one-GCD server at devices it no longer has.
            env.pop("HIP_VISIBLE_DEVICES", None)
            env.pop("GPU_DEVICE_ORDINAL", None)
        log = os.path.join(ctx.workdir, f"{p['id']}.server{r}.log")
        # Every request leaves an audit line; on the scratch filesystem, not
        # in a home directory with a quota, unless the job says otherwise.
        if not env.get("EULLM_AUDIT_DIR"):
            env["EULLM_AUDIT_DIR"] = os.path.join(ctx.workdir, "audit")
            os.makedirs(env["EULLM_AUDIT_DIR"], exist_ok=True)
        binary, args, extra_env = runtime_command(p, ctx.engine)
        env.update(extra_env)
        env.update(point_env(p, ctx.workdir))  # the point's own, e.g. LLAMA_MOE_CACHE_STATS
        env["OLLAMA_HOST"] = f"127.0.0.1:{ctx.port_base + r}"
        server = Server(binary, ctx.port_base + r, args, env, log,
                        ctx.cores or cores_for(ctx.binding, group),
                        runtime=p.get("runtime", "eullm"))
        server.start()
        servers.append(server)
    return servers


def warm_up(p: dict, servers: list, ctx: Context, body: dict) -> dict:
    """One request per server, all at once: loads every copy of the model.
    Untimed for throughput; timed here, because loading is a metric."""
    cache = cache_state(p, ctx)
    t0 = time.time()
    got = concurrently(
        len(servers), lambda i: call(servers[i], "/api/generate", body, WARMUP_TIMEOUT_S)
    )
    wall = time.time() - t0
    for i, g in enumerate(got):
        if isinstance(g, RequestError) and FIT_REFUSAL in str(g):
            raise DoesNotFit(str(g)[:300])
        if isinstance(g, RequestError) and is_missing_model(g):
            raise ModelMissing(f"model {p['model']!r} is not in the store: {g}")
        if isinstance(g, Exception):
            raise PointError(f"warm-up failed on server {i}: {g}\n" + servers[i].log_tail())
    ctx.model_seen.add(p["model"])
    return {
        "cache": cache,
        "storage": model_storage(p),
        "wall_s": round(wall, 2),
        "load_ms": [round(g["load_duration_ns"] / 1e6, 1) for g in got],
    }


def run_throughput(p: dict, servers: list, ctx: Context) -> tuple:
    prompt = p["prompt"]
    if p["prompt_tokens"]:
        def body_for(i, rep):
            return generate_body(p, synthetic_prompt(p["prompt_tokens"], uuid.uuid4().hex[:12]))
    else:
        def body_for(i, rep):
            return generate_body(p, prompt)

    reps, first_t, last_t = [], None, None
    for rep in range(p["repeats"]):
        if ctx.stop.is_set():
            raise Interrupted("stopped between repeats")
        t0 = time.time()
        got = concurrently(
            p["concurrency"],
            lambda i: call(servers[i % len(servers)], "/api/generate", body_for(i, rep)),
        )
        t1 = time.time()
        first_t = first_t or t0
        last_t = t1
        ok = [g for g in got if isinstance(g, dict)]
        errors = [str(g)[:200] for g in got if not isinstance(g, dict)]
        wall = t1 - t0
        gen = sum(g["eval_count"] for g in ok)
        ttft = [g["ttft_ms"] for g in ok]
        decode = [rate(g["eval_count"], (g["total_ms"] - g["ttft_ms"]) / 1000) for g in ok]
        prefill = [rate(g["prompt_eval_count"], g["ttft_ms"] / 1000) for g in ok]
        srv_prefill = [rate(g["prompt_eval_count"], g["prompt_eval_duration_ns"] / 1e9) for g in ok]
        srv_decode = [rate(g["eval_count"], g["eval_duration_ns"] / 1e9) for g in ok]
        reps.append({
            "ok": len(ok),
            "failed": len(errors),
            "errors": errors[:3],
            "generated_tokens": gen,
            "prompt_tokens": sum(g["prompt_eval_count"] for g in ok),
            "wall_clock_s": round(wall, 3),
            "aggregate_tok_s": r1(rate(gen, wall)),
            "ttft_ms_p50": r1(percentile(ttft, 0.5)),
            "ttft_ms_p95": r1(percentile(ttft, 0.95)),
            "decode_tok_s_mean": mean(decode),
            "prefill_tok_s_mean": mean(prefill),
            # From the server's own durations; null until the engine reports
            # a measured prompt time (on main it is hard-coded to 0).
            "server_prefill_tok_s_mean": mean(srv_prefill),
            "server_decode_tok_s_mean": mean(srv_decode),
            "draft_n": sum(g.get("draft_n", 0) for g in ok),
            "draft_n_accepted": sum(g.get("draft_n_accepted", 0) for g in ok),
        })
        if not ok:
            raise PointError(f"every request failed in repeat {rep}: {errors[:3]}")

    agg = [r["aggregate_tok_s"] for r in reps if r["aggregate_tok_s"] is not None]
    summary = {
        "aggregate_tok_s_mean": r1(statistics.mean(agg)) if agg else None,
        "aggregate_tok_s_stdev": r1(statistics.stdev(agg)) if len(agg) > 1 else None,
        "aggregate_tok_s_cv_pct": (
            round(100 * statistics.stdev(agg) / statistics.mean(agg), 2)
            if len(agg) > 1 and statistics.mean(agg) else None
        ),
        "repeats": reps,
        "draft_accept_pct": accept_pct(sum(r["draft_n"] for r in reps),
                                       sum(r["draft_n_accepted"] for r in reps)),
    }
    return summary, (first_t, last_t)


def load_items(p: dict, ctx: Context) -> list:
    """The point's sets, from the JSONL files `campaign.py prefetch` wrote on
    a login node: compute nodes have no network, so nothing is downloaded
    here."""
    if REFLEXBENCH not in sys.path:
        sys.path.insert(0, REFLEXBENCH)
    import ab_data

    items = []
    for name in p["sets"]:
        path = os.path.join(ctx.sets_dir or "", f"{name}.jsonl")
        if not os.path.exists(path):
            raise ModelMissing(f"set {name!r} not prefetched ({path}): run campaign.py prefetch")
        items += ab_data.from_jsonl(path, limit=p["limit"]).items
    return items


def run_workload(p: dict, servers: list, ctx: Context, answers_path=None) -> tuple:
    if REFLEXBENCH not in sys.path:
        sys.path.insert(0, REFLEXBENCH)
    import ab_grade
    import ab_methods

    items = load_items(p, ctx)
    n = len(items)
    duration = p.get("duration_s", 0)
    start = time.time()
    deadline = start + duration if duration else None
    lock = threading.Lock()
    counter = [0]
    records = []

    def worker():
        while not ctx.stop.is_set():
            with lock:
                i = counter[0]
                counter[0] += 1
            if deadline is None and i >= n:
                return
            if deadline is not None and time.time() >= deadline:
                return
            item = items[i % n]
            server = servers[i % len(servers)]
            path, body = ab_methods.chat_body(item, p["model"], p["think"], p["max_tokens"])
            rec = {"i": i, "pass": i // n, "item": item.id, "set": item.set}
            try:
                got = call(server, path, body)
                rec.update(ok=True, ttft_ms=got["ttft_ms"], total_ms=got["total_ms"],
                           eval=got["eval_count"], prompt=got["prompt_eval_count"],
                           drafts=got.get("draft_n", 0), kept=got.get("draft_n_accepted", 0),
                           sha=hashlib.sha1(got["text"].encode()).hexdigest()[:16])
                if rec["pass"] == 0:
                    rec["text"] = got["text"]
                    rec["correct"] = bool(ab_grade.correct(item, got["text"]))
            except Exception as e:
                rec.update(ok=False, error=str(e)[:200])
            rec["t"] = time.time()
            with lock:
                records.append(rec)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(p["concurrency"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    if ctx.stop.is_set():
        raise Interrupted("stopped during the workload")

    # A server that died before the runner stopped it is the soak's finding,
    # whatever share of the requests it took with it: the rule below fails a
    # point only when most of them failed, and soaks of Qwen3-32B that
    # crashed in their last hours were counted as passed (LUMI, 16-10-2026).
    # When: the first request on it that failed.
    died = []
    for k, s in enumerate(servers):
        if s.proc is not None and s.proc.poll() is not None:
            failed_at = [r["t"] for r in records if not r["ok"] and r["i"] % len(servers) == k]
            died.append({"server": k, "exit": s.proc.returncode,
                         "after_s": round(min(failed_at) - start, 1) if failed_at else None})

    ok = [r for r in records if r["ok"]]
    if records and len(ok) < len(records) / 2:
        # Carry the death out with the failure: this raise used to discard
        # `died`, and run()'s server_exits are attached only on the success
        # path, so a crash early in a soak reached campaign.py as a bare
        # request-failure count. The retry then succeeded and the report
        # showed a clean measured soak with died=0.
        err = PointError(f"{len(records) - len(ok)} of {len(records)} requests failed: "
                         f"{[r.get('error') for r in records if not r['ok']][:3]}")
        err.servers_died = died
        err.server_exits = [s.proc.returncode if s.proc is not None else None
                            for s in servers]
        raise err

    # Accuracy, from the first pass.
    per_set = {}
    for r in ok:
        if r["pass"] == 0:
            s = per_set.setdefault(r["set"], {"graded": 0, "correct": 0})
            s["graded"] += 1
            s["correct"] += r["correct"]
    for s in per_set.values():
        s["accuracy"] = round(s["correct"] / s["graded"], 4) if s["graded"] else None

    # Consistency: a later pass against the first, item by item.
    first = {r["item"]: r["sha"] for r in ok if r["pass"] == 0}
    later = [r for r in ok if r["pass"] > 0 and r["item"] in first]
    same = sum(r["sha"] == first[r["item"]] for r in later)

    # The time series.
    interval = max(1, int(p["interval_s"]))
    buckets = {}
    for r in records:
        buckets.setdefault(int((r["t"] - start) // interval), []).append(r)
    series = []
    for b in range(int((end - start) // interval) + 1):
        rows = buckets.get(b, [])
        good = [r for r in rows if r["ok"]]
        lat = [r["total_ms"] for r in good]
        b0, b1 = start + b * interval, min(start + (b + 1) * interval, end)
        row = {
            "t_s": b * interval,
            "completed": len(good),
            "errors": len(rows) - len(good),
            "tok_s": r1(rate(sum(r["eval"] for r in good), b1 - b0)),
            "latency_ms_p50": r1(percentile(lat, 0.5)),
            "latency_ms_p95": r1(percentile(lat, 0.95)),
            "ttft_ms_p50": r1(percentile([r["ttft_ms"] for r in good], 0.5)),
        }
        if ctx.sampler is not None:
            dev = ctx.sampler.window(b0, b1, ctx.physical)
            row["vram_peak_mib"] = {d: s["vram_peak_mib"] for d, s in dev.items()}
            row["use_mean"] = {d: s["use_mean"] for d, s in dev.items()}
        series.append(row)

    full = [s for s in series[:-1] if s["tok_s"] is not None] or series
    tenth = max(1, len(full) // 10)
    head = [s["tok_s"] for s in full[:tenth] if s["tok_s"]]
    tail = [s["tok_s"] for s in full[-tenth:] if s["tok_s"]]
    drift = (
        round(100 * (statistics.mean(tail) - statistics.mean(head)) / statistics.mean(head), 2)
        if head and tail else None
    )

    if answers_path:
        with open(answers_path, "w") as f:
            for r in sorted((r for r in ok if r["pass"] == 0), key=lambda r: r["i"]):
                f.write(json.dumps({k: r[k] for k in ("item", "set", "correct", "text",
                                                       "ttft_ms", "total_ms", "eval")}))
                f.write("\n")

    elapsed = end - start
    return {
        "items": n,
        "duration_s": round(elapsed, 1),
        "requests": len(records),
        "failed": len(records) - len(ok),
        "passes": round(len(records) / n, 2) if n else 0,
        "generated_tokens": sum(r["eval"] for r in ok),
        "aggregate_tok_s": r1(rate(sum(r["eval"] for r in ok), elapsed)),
        "latency_ms_p50": r1(percentile([r["total_ms"] for r in ok], 0.5)),
        "latency_ms_p95": r1(percentile([r["total_ms"] for r in ok], 0.95)),
        "latency_ms_p99": r1(percentile([r["total_ms"] for r in ok], 0.99)),
        "accuracy": per_set,
        "consistency": {
            "compared": len(later),
            "identical": same,
            "rate": round(same / len(later), 4) if later else None,
        },
        "throughput_drift_pct": drift,
        "servers_died": died,
        "series": series,
        "draft_accept_pct": accept_pct(sum(r.get("drafts", 0) for r in ok),
                                       sum(r.get("kept", 0) for r in ok)),
    }, (start, end)


class Job:
    """A child process the runner may have to stop: `eullm finetune`."""

    def __init__(self, proc):
        self.proc = proc

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


# How `eullm finetune` says why it would not run, and what each means here.
# Running out of device memory after the estimate let it start is the same
# boundary, found the hard way: recorded with the error, not retried.
FT_DOES_NOT_FIT = ("is estimated at", "out of memory")
# An engine built before `finetune` existed blocks the point too: it waits
# for a binary that has the command, it does not fail.
FT_NOT_RUNNABLE = ("is not an F32 model", "neither a .gguf file",
                   "is not a readable GGUF file", "holds no text", "cannot read",
                   "unrecognized subcommand")


def finetune_command(p: dict, ctx: Context, output: str, report: str) -> list:
    model = p["model"]
    if not os.path.isabs(model):
        model = posixpath.join(ctx.f32_dir or "", model)
    data = p["data"]
    if not os.path.isabs(data):
        data = posixpath.join(ctx.sets_dir or "", data)
    cmd = [
        ctx.engine, "finetune", model,
        "--data", data,
        "--ctx", str(p["ft_ctx"]),
        "--epochs", str(p["epochs"]),
        "--lr", repr(float(p["lr"])),
        "--optimizer", p["optimizer"],
        "--val-split", repr(float(p["val_split"])),
        "--limit-tokens", str(p["limit_tokens"]),
        "--device", "0",
        "--no-progress",
        "--output", output,
        "--report", report,
    ]
    if p["train_tensors"]:
        cmd += ["--train-tensors", ",".join(p["train_tensors"])]
    return cmd + list(p.get("extra_args", []))


def run_finetune(p: dict, ctx: Context, workdir: str) -> tuple:
    """`eullm finetune` on one device, its report as the measurement. The
    trained GGUF is deleted unless the point keeps it: the point measures
    training, it does not produce a model."""
    output = os.path.join(workdir, f"{p['id']}.gguf")
    report = os.path.join(workdir, f"{p['id']}.finetune.json")
    log_path = os.path.join(workdir, f"{p['id']}.finetune.log")
    env = dict(os.environ)
    env[VISIBLE_ENV[ctx.backend]] = ",".join(ctx.physical)
    if ctx.backend == "rocm":
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("GPU_DEVICE_ORDINAL", None)
    cores = cores_for(ctx.binding, ctx.physical)
    cmd = (["taskset", "-c", cores] if cores else []) + finetune_command(p, ctx, output, report)
    start = time.time()
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
    ctx.servers = [Job(proc)]
    while proc.poll() is None:
        if ctx.stop.wait(2):
            Job(proc).kill()
            proc.wait(timeout=30)
            raise Interrupted("stopped during finetune")
    end = time.time()
    try:
        with open(log_path, errors="replace") as f:
            tail = "".join(f.readlines()[-30:])
    except OSError:
        tail = ""
    try:
        if proc.returncode != 0:
            if any(m in tail for m in FT_DOES_NOT_FIT):
                raise DoesNotFit(tail.strip().splitlines()[-1][:300])
            if any(m in tail for m in FT_NOT_RUNNABLE):
                raise ModelMissing(tail.strip().splitlines()[-1][:300])
            raise PointError(f"eullm finetune exited {proc.returncode}:\n{tail}")
        with open(report) as f:
            measured = json.load(f)
    finally:
        if not p.get("keep_output") and os.path.exists(output):
            os.remove(output)
    return measured, (start, end)


# ── decisions ────────────────────────────────────────────────────────────


def decision_request(url, body, timeout=REQUEST_TIMEOUT_S) -> tuple:
    """POST /v1/systemone: (client milliseconds, the response)."""
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RequestError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}",
                           code=e.code) from None
    return (time.perf_counter() - started) * 1000, out


def answers_digest(answers: dict) -> str:
    """The answers, to four decimals: equal digests are the same decision."""
    def values(a):
        for key in ("noul", "choice", "score", "probabilities"):
            v = a.get(key)
            if isinstance(v, (int, float)):
                yield key, round(v, 4)
            elif isinstance(v, dict):
                yield key, {k: round(x, 4) for k, x in sorted(v.items())}
            elif v is not None:
                yield key, v
    canon = {qid: dict(values(a)) for qid, a in sorted((answers or {}).items())}
    return hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]


def answers_decisions(answers: dict) -> dict:
    """What each answer decides, without its probabilities: yes or no for a
    `noul`, the code for a `choice`, the likeliest level for a `score`."""
    def decided(a):
        if a.get("type") == "noul" or ("noul" in a and "probabilities" not in a):
            return "yes" if (a.get("noul") or 0) >= 0.5 else "no"
        if a.get("choice") is not None:
            return a["choice"]
        probs = a.get("probabilities") or {}
        return max(sorted(probs), key=lambda k: probs[k]) if probs else None
    return {qid: decided(a) for qid, a in sorted((answers or {}).items())}


def answers_choices(answers: dict) -> str:
    """`answers_decisions` as a digest: equal digests made the same decisions,
    whatever the digits behind them."""
    canon = answers_decisions(answers)
    return hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]


def decisions_changed(first: dict, other: dict) -> tuple:
    """(questions whose decision differs, questions compared) between two
    answers to one request."""
    shared = [q for q in first if q in other]
    return sum(first[q] != other[q] for q in shared), len(shared)


def answers_probabilities(answers: dict) -> dict:
    """Every probability the answers carry, by question and key: P(yes) of a
    `noul`, each code's or level's of a `choice` or a `score`."""
    out = {}
    for qid, a in (answers or {}).items():
        if isinstance(a.get("noul"), (int, float)):
            out[f"{qid}:yes"] = float(a["noul"])
        for k, v in (a.get("probabilities") or {}).items():
            if isinstance(v, (int, float)):
                out[f"{qid}:{k}"] = float(v)
    return out


def largest_shift(first: dict, other: dict) -> float:
    """The most any probability moved between two answers to one request."""
    return max((abs(other[k] - v) for k, v in first.items() if k in other), default=0.0)


def decision_bodies(p: dict) -> list:
    """`distinct_states` requests: one synthetic ticket history, each opened
    by its own line so no state is the one the server kept from the request
    before (that would measure a cache hit), and the same questions."""
    bench = os.path.dirname(HERE)
    if bench not in sys.path:
        sys.path.insert(0, bench)
    import decision_bench

    state = decision_bench.make_state(p["state_tokens"])
    questions = decision_bench.make_questions(p["questions"])
    return [{"state": f"[ticket {k:04d}]\n{state}", "questions": questions,
             "eullm": {"mode": p["decision_mode"]}} for k in range(p["distinct_states"])]


def run_decision(p: dict, servers: list, ctx: Context) -> tuple:
    """`requests` decisions from `concurrency` clients over the servers.
    Client latency, the server's own time and its decode time apart — the
    difference is the wait behind other requests — and whether a state asked
    again, under other concurrency, got the same answers."""
    bodies = decision_bodies(p)
    n = p["requests"]
    lock = threading.Lock()
    counter = [0]
    records = []

    def worker():
        while not ctx.stop.is_set():
            with lock:
                i = counter[0]
                counter[0] += 1
            if i >= n:
                return
            k = i % len(bodies)
            rec = {"i": i, "state": k}
            try:
                ms, out = decision_request(servers[i % len(servers)].url, bodies[k])
                e = out.get("eullm") or {}
                t = e.get("timings_ms") or {}
                rec.update(ok=True, client_ms=ms, server_ms=e.get("request_ms"),
                           decode_ms=(t.get("prefix") or 0) + (t.get("questions") or 0),
                           prompt_tokens=e.get("prompt_tokens"),
                           evaluated_tokens=e.get("evaluated_tokens"),
                           # Absent when the request was evaluated alone.
                           together=e.get("requests_together") or 1,
                           digest=answers_digest(out.get("answers")),
                           choices=answers_choices(out.get("answers")),
                           decisions=answers_decisions(out.get("answers")),
                           probs=answers_probabilities(out.get("answers")))
            except Exception as ex:
                rec.update(ok=False, error=str(ex)[:200])
            rec["t"] = time.time()
            with lock:
                records.append(rec)

    start = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(p["concurrency"])]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    end = time.time()
    if ctx.stop.is_set():
        raise Interrupted("stopped during the decisions")
    ok = [r for r in records if r["ok"]]
    if records and len(ok) < len(records) / 2:
        raise PointError(f"{len(records) - len(ok)} of {len(records)} decisions failed: "
                         f"{[r.get('error') for r in records if not r['ok']][:3]}")
    by_state = {}
    for r in ok:
        by_state.setdefault(r["state"], []).append(r)
    repeats = [r for rs in by_state.values() for r in rs[1:]]
    same = sum(r["digest"] == rs[0]["digest"] for rs in by_state.values() for r in rs[1:])
    # The digest above counts any change past the fourth decimal; in `batched`
    # mode an answer moves with the requests evaluated beside it, so it also
    # says whether the decisions themselves changed, and by how much at most.
    same_choices = sum(r["choices"] == rs[0]["choices"]
                       for rs in by_state.values() for r in rs[1:])
    shift = max((largest_shift(rs[0]["probs"], r["probs"])
                 for rs in by_state.values() for r in rs[1:]), default=None)
    # One question in 64 decided otherwise makes the whole request differ:
    # the share of questions is what a user of the decisions meets.
    changed = [decisions_changed(rs[0]["decisions"], r["decisions"])
               for rs in by_state.values() for r in rs[1:]]
    flipped, asked = sum(c for c, _ in changed), sum(n for _, n in changed)
    client = [r["client_ms"] for r in ok]
    server = [r["server_ms"] for r in ok if r["server_ms"] is not None]
    decode = [r["decode_ms"] for r in ok]
    wait = [r["server_ms"] - r["decode_ms"] for r in ok if r["server_ms"] is not None]
    elapsed = end - start
    return {
        "requests": len(records),
        "failed": len(records) - len(ok),
        "decisions_per_s": r1(rate(len(ok), elapsed)),
        "questions_per_s": r1(rate(len(ok) * p["questions"], elapsed)),
        "client_ms_p50": r1(percentile(client, 0.5)),
        "client_ms_p95": r1(percentile(client, 0.95)),
        "client_ms_p99": r1(percentile(client, 0.99)),
        "server_ms_p50": r1(percentile(server, 0.5)),
        "decode_ms_p50": r1(percentile(decode, 0.5)),
        "wait_ms_p50": r1(percentile(wait, 0.5)),
        "wait_ms_p95": r1(percentile(wait, 0.95)),
        "prompt_tokens_mean": mean([r["prompt_tokens"] for r in ok]),
        "evaluated_tokens_mean": mean([r["evaluated_tokens"] for r in ok]),
        # Requests the engine evaluated together with each (batched mode):
        # 1 for an engine that takes them one at a time.
        "requests_together_mean": mean([r.get("together", 1) for r in ok]),
        "consistency": {"compared": len(repeats), "identical": same,
                        "rate": round(same / len(repeats), 4) if repeats else None,
                        "same_choices": same_choices,
                        "choices_rate": (round(same_choices / len(repeats), 4)
                                         if repeats else None),
                        "largest_shift": round(shift, 4) if shift is not None else None,
                        "questions_changed": flipped, "questions_compared": asked,
                        "questions_changed_rate": (round(flipped / asked, 5)
                                                   if asked else None)},
        "duration_s": round(elapsed, 1),
    }, (start, end)


WARMUP_BODY = {"prompt": "Ciao.", "stream": True, "think": False, "num_predict": 8,
               "options": {"num_predict": 8}}


def point_limit_s(p: dict):
    """The seconds `p` may run before its servers are killed (above), or
    None for the kinds that have no limit."""
    if p["kind"] not in ("throughput", "decision"):
        return None
    return max(POINT_LIMIT_FACTOR * p.get("est_s", 0), POINT_LIMIT_MIN_S)


def run(p: dict, ctx: Context, answers_path=None) -> dict:
    """Measure `p` on `ctx.physical`; the result, or an exception."""
    if p["kind"] == "finetune":
        measured, window = run_finetune(p, ctx, ctx.workdir)
        stats = ctx.sampler.window(*window, ctx.physical) if ctx.sampler else {}
        return {"finetune": measured, "device_stats": stats}
    evicted = evict(model_files(p)) if p.get("cold") else None
    started = time.time()
    servers = start_servers(p, ctx)
    ctx.servers = servers
    limit = point_limit_s(p)
    expired = threading.Event()

    def expire():
        expired.set()
        for s in servers:
            s.kill()

    timer = threading.Timer(limit, expire) if limit else None
    if timer:
        timer.daemon = True
        timer.start()
    try:
        result = _measure(p, ctx, servers, evicted, started, answers_path)
    except Exception as e:
        if expired.is_set():
            # What the requests saw is the runner's own kill, not a crash: a
            # 235B with --cpu-moe at 32 requests read as a server death in its
            # "Connection refused" (21-10-2026). Say which came first.
            err = PointError(f"{limit_note(p, limit)} (the requests then saw: {e})")
            for key in ("servers_died", "server_exits"):
                if getattr(e, key, None) is not None:
                    setattr(err, key, getattr(e, key))
            raise err from e
        raise
    finally:
        if timer:
            timer.cancel()
        ctx.stragglers = [s for s in servers if not s.stop()]
    if expired.is_set():
        raise PointError(limit_note(p, limit))
    # How each server ended once asked to stop: 0 is a clean exit, -6 an
    # abort, -11 a segfault, -9 our own SIGKILL after it ignored SIGTERM;
    # None if it is still there. On LUMI nearly every engine server aborted
    # while stopping (16-10-2026), which this counts across builds.
    result["server_exits"] = [s.proc.returncode if s.proc is not None else None
                              for s in servers]
    return result


def limit_note(p: dict, limit: float) -> str:
    return (f"stopped after {limit:.0f} s, {POINT_LIMIT_FACTOR} times the estimate of "
            f"{p.get('est_s', 0)} s: the runner stopped its servers, not a crash")


def _measure(p: dict, ctx: Context, servers: list, evicted, started: float, answers_path):
    """`run`'s measurement once the servers are started."""
    for s in servers:
        s.wait_ready(stop=ctx.stop)
    if p["kind"] == "decision":
        # The model loads before the server answers: readiness is the load.
        cache = cache_state(p, ctx)
        ready_s = time.time() - started
        got = concurrently(len(servers), lambda i: decision_request(
            servers[i].url, decision_bodies(dict(p, distinct_states=1))[0], WARMUP_TIMEOUT_S))
        for i, g in enumerate(got):
            if isinstance(g, Exception):
                raise PointError(f"first decision failed on server {i}: {g}\n"
                                 + servers[i].log_tail())
        ctx.model_seen.add(p["model"])
        measured, window = run_decision(p, servers, ctx)
        stats = ctx.sampler.window(*window, ctx.physical) if ctx.sampler else {}
        load = {"cache": cache, "storage": model_storage(p), "wall_s": round(ready_s, 2)}
        if evicted is not None:
            load["evicted_bytes"] = evicted
        return {"load": load, "decision": measured, "device_stats": stats}
    if p["kind"] == "throughput":
        warm_body = generate_body(p, p["prompt"] if not p["prompt_tokens"]
                                  else synthetic_prompt(p["prompt_tokens"], "warmup"))
    else:
        warm_body = dict(WARMUP_BODY, model=p["model"])
    load = warm_up(p, servers, ctx, warm_body)
    if evicted is not None:
        load["evicted_bytes"] = evicted
    if p["kind"] == "throughput":
        measured, window = run_throughput(p, servers, ctx)
    else:
        measured, window = run_workload(p, servers, ctx, answers_path)
    stats = ctx.sampler.window(*window, ctx.physical) if ctx.sampler else {}
    return {"load": load, p["kind"]: measured, "device_stats": stats}
