"""What AutoBench asks the server: the two models' answers, and the routers.

Stage 1 asks both models every item, each by name and deterministically —
temperature 0, top_k 1, seed 1 — so that what a router would have scored is
known for every routing without asking again. Stage 2 asks only for
decisions:

  * `reflex`: `POST /api/route`, the server's own router, with the request
    exactly as a client would send it with `"model": "auto"`; its score is
    the probability it gives the small model;
  * `knn`: the embeddings of the request and of the dev half's, through
    `/v1/embeddings`; its score is the share of the k nearest dev items the
    small model got right (or both got wrong);
  * `length`: the request's length, shorter going small, at a threshold
    fitted on the dev half;
  * `random`, at the share of requests Reflex routes small; `always-small`,
    `always-large`, and the `oracle`, which knows the answers;
  * replays of the question with another wording or option order
    (`--questions`), asked of `/v1/systemone` about the state the server
    built for `/api/route` — the engine stays the only source of the state.
"""

import json
import random
import time
import urllib.error
import urllib.request

import ab_grade
from rb_methods import ServerError, post, unit

# What makes stage 1 deterministic.
GREEDY = {"temperature": 0, "top_k": 1, "seed": 1}

# Greedy decoding is not enough on a GPU: the numbers a prompt produces
# depend on how much of it the server reuses from the request before, so the
# same request could get another answer in stage 3 than in stage 1. Every
# generation decodes its whole prompt instead (llama.cpp's `cache_prompt`).
REPRODUCIBLE = {"cache_prompt": False}


class Answer:
    """One model's answer to one item, and what it cost."""

    def __init__(self, text, model, ttft_ms, total_ms, eval_count=0, prompt_count=0, load_ms=0.0):
        self.text, self.model = text, model
        self.ttft_ms, self.total_ms = ttft_ms, total_ms
        self.eval_count, self.prompt_count, self.load_ms = eval_count, prompt_count, load_ms
        self.correct = None  # set by grading
        self.route = None  # the route the server reported, for "auto"
        self.headers = {}

    def to_json(self):
        return {
            "text": self.text,
            "model": self.model,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "eval_count": self.eval_count,
            "prompt_count": self.prompt_count,
            "load_ms": round(self.load_ms, 2),
            "correct": self.correct,
        }

    @classmethod
    def from_json(cls, row):
        answer = cls(
            row["text"],
            row["model"],
            row["ttft_ms"],
            row["total_ms"],
            row.get("eval_count", 0),
            row.get("prompt_count", 0),
            row.get("load_ms", 0.0),
        )
        answer.correct = row.get("correct")
        return answer


def chat_body(item, model, think, max_tokens, stream=True):
    """The request for `item` as a client sends it: `/api/chat` with its
    messages, or `/api/generate` with its prompt."""
    options = dict(GREEDY, num_predict=max_tokens)
    body = dict(REPRODUCIBLE, model=model, stream=stream, think=think, options=options)
    if item.prompt is not None:
        return "/api/generate", dict(body, prompt=item.prompt)
    return "/api/chat", dict(body, messages=item.messages)


def stream(url, path, body, api_key, timeout):
    """POST a streamed Ollama request and read it to the end: the text, the
    time to the first piece of it, the whole time, the last line, and the
    response headers."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers
    )
    started = time.perf_counter()
    first, pieces, last = None, [], {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_headers = {k.lower(): v for k, v in response.headers.items()}
            for raw in response:
                if not raw.strip():
                    continue
                line = json.loads(raw)
                if "error" in line:
                    raise ServerError(500, line["error"])
                piece = line.get("response")
                if piece is None:
                    piece = (line.get("message") or {}).get("content", "")
                if piece and first is None:
                    first = time.perf_counter()
                pieces.append(piece or "")
                last = line
    except urllib.error.HTTPError as e:
        raise ServerError(e.code, e.read().decode(errors="replace")) from None
    ended = time.perf_counter()
    ttft = ((first or ended) - started) * 1000
    return "".join(pieces), ttft, (ended - started) * 1000, last, response_headers


def generate(url, item, model, api_key, timeout, think=False, max_tokens=768):
    """`model`'s answer to `item`."""
    path, body = chat_body(item, model, think, max_tokens)
    text, ttft, total, last, headers = stream(url, path, body, api_key, timeout)
    answer = Answer(
        text,
        last.get("model", model),
        ttft,
        total,
        last.get("eval_count", 0),
        last.get("prompt_eval_count", 0),
        last.get("load_duration", 0) / 1e6,
    )
    answer.route = (last.get("eullm") or {}).get("route")
    answer.headers = headers
    return answer


JUDGE = (
    "Two assistants answered the same request. Which answer is better: more "
    "correct, more complete, and as helpful? Reply with A, B, or tie, on a line "
    '"Answer: <A, B or tie>".'
)


def judge(url, judge_model, item, small, large, api_key, timeout):
    """The judge's verdict on the two answers to `item`, asked in both
    orders: "small", "large" or "tie" (see `ab_grade.both_orders`)."""
    verdicts = []
    for a, b in ((small, large), (large, small)):
        request = item.text
        content = f"Request:\n{request}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n{JUDGE}"
        body = {
            "model": judge_model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "think": False,
            "options": dict(GREEDY, num_predict=16),
        }
        reply = post(url.rstrip("/") + "/api/chat", body, api_key, timeout)
        verdicts.append(ab_grade.judge_verdict(reply.get("message", {}).get("content", "")))
    return ab_grade.both_orders(*verdicts)


class Decision:
    """A router's decision on one item: its score (P(small)), whether it
    routes the item small on its own, and what deciding took."""

    def __init__(self, score=None, small=None, ms=None, server=None):
        self.score, self.small, self.ms = score, small, ms
        self.server = server or {}

    def to_json(self):
        return {
            "score": self.score,
            "small": self.small,
            "ms": None if self.ms is None else round(self.ms, 2),
            "server": self.server,
        }


class Reflex:
    """`POST /api/route`: the server's own routing, without generating."""

    name = "reflex"

    def __init__(self, url, small, large, api_key, timeout, think=False, max_tokens=768):
        self.url, self.api_key, self.timeout = url, api_key, timeout
        self.small, self.large = small, large
        self.think, self.max_tokens = think, max_tokens

    def decide(self, item):
        _, body = chat_body(item, "auto", self.think, self.max_tokens, stream=False)
        started = time.perf_counter()
        route = post(self.url.rstrip("/") + "/api/route", body, self.api_key, self.timeout)
        ms = (time.perf_counter() - started) * 1000
        names = [c["model"] for c in route.get("candidates", [])]
        for wanted in (self.small, self.large):
            if wanted not in names:
                raise SystemExit(
                    f"the server routes between {names}, not {self.small} and {self.large}: "
                    "start it with --auto-model for both"
                )
        p = {c["model"]: c.get("probability") for c in route["candidates"]}
        server = {
            "reason": route.get("reason"),
            "model": route.get("model"),
            "decision_ms": route.get("decision_ms"),
            "decision_model": route.get("decision_model"),
            "route_id": route.get("route_id"),
            "state": route.get("state"),
            "question": route.get("question"),
        }
        score = p.get(self.small)
        return Decision(score, route.get("model") == self.small, ms, server)


class Replay:
    """The question with another wording or option order, about the state
    the server built for the same request (`decision.server["state"]` from
    `Reflex`), asked of `/v1/systemone`."""

    def __init__(self, spec, url, small, api_key, timeout):
        self.name = f"reflex:{spec['name']}"
        self.instructions = spec.get("instructions")
        self.reverse = bool(spec.get("reverse"))
        self.descriptions = spec.get("descriptions") or {}
        self.url = url.rstrip("/") + "/v1/systemone"
        self.small, self.api_key, self.timeout = small, api_key, timeout

    def decide(self, reflex_decision):
        question = reflex_decision.server.get("question") or {}
        criteria = list((question.get("criteria") or {}).items())
        if len(criteria) < 2:
            return Decision()
        if self.reverse:
            criteria.reverse()
        criteria = {name: self.descriptions.get(name, text) for name, text in criteria}
        payload = {
            "state": reflex_decision.server["state"],
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": self.instructions or question["instructions"],
                    "criteria": criteria,
                }
            },
        }
        started = time.perf_counter()
        body = post(self.url, payload, self.api_key, self.timeout)
        ms = (time.perf_counter() - started) * 1000
        answer = body["answers"]["route"]
        p = answer["probabilities"].get(self.small)
        server = {"request_ms": (body.get("eullm") or {}).get("request_ms")}
        return Decision(p, answer["choice"] == self.small, ms, server)


class Knn:
    """The k nearest dev items by the embeddings of their text: the share
    of them the small model handled is the score."""

    name = "knn"

    def __init__(self, url, model, api_key, timeout, k=10):
        self.url = url.rstrip("/") + "/v1/embeddings"
        self.model, self.api_key, self.timeout, self.k = model, api_key, timeout, k
        self.dev = []  # (unit vector, small_ok)

    def embed(self, texts):
        body = post(self.url, {"model": self.model, "input": texts}, self.api_key, self.timeout)
        return [unit(r["embedding"]) for r in sorted(body["data"], key=lambda r: r["index"])]

    def fit(self, dev_items, small_ok):
        texts = [item.text for item in dev_items]
        vectors = []
        for start in range(0, len(texts), 32):
            vectors += self.embed(texts[start : start + 32])
        self.dev = list(zip(vectors, small_ok, strict=True))

    def decide(self, item):
        started = time.perf_counter()
        query = self.embed([item.text])[0]
        ms = (time.perf_counter() - started) * 1000
        near = sorted(
            self.dev, key=lambda d: -sum(a * b for a, b in zip(query, d[0], strict=True))
        )[: self.k]
        score = sum(ok for _, ok in near) / len(near) if near else 0.0
        return Decision(score, score >= 0.5, ms)


def length_scores(items):
    """Shorter requests score higher: P(small) as the length baseline sees it."""
    return [-len(item.text) for item in items]


def random_routing(n, share, seed):
    """`n` decisions routing small with probability `share`, from `seed`."""
    rng = random.Random(f"random:{seed}")
    return [rng.random() < share for _ in range(n)]


def server_state(url, api_key, timeout):
    """What the report's header records about the server."""

    def get(path):
        request = urllib.request.Request(url.rstrip("/") + path)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, ValueError):
            return None

    return {"version": get("/api/version"), "ps": get("/api/ps")}


def release_embedder(url, model, api_key, timeout):
    """Unload `model` from the server's embedding slot when the kNN baseline
    loaded it there, so the routed candidates have the room it took back
    before stage 3 times them. A reserved companion (`--embedding-model`)
    is left alone: the server keeps room for it whatever happens. Says what
    it did."""
    ps = (server_state(url, api_key, timeout).get("ps") or {}).get("models") or []
    entry = next(
        (
            m
            for m in ps
            if m.get("name") == model and (m.get("eullm") or {}).get("slot") == "embedding"
        ),
        None,
    )
    if entry is None:
        return "not loaded"
    if (entry.get("eullm") or {}).get("reserved_companion"):
        return "reserved, kept"
    unload = {"model": model, "input": "-", "keep_alive": 0}
    post(url.rstrip("/") + "/api/embed", unload, api_key, timeout)
    return "unloaded"


def evictions(state):
    """`generation_evictions` in a `server_state`, or None."""
    version = (state or {}).get("version") or {}
    value = version.get("generation_evictions")
    return value if isinstance(value, int) else None
