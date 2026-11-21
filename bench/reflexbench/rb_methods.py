"""The ways ReflexBench ranks the tools offered with a request.

Every method returns the same thing, a `Ranking`: the candidates from most
to least likely to be needed, what it cost, and, for Reflex, how likely it
found that no tool is needed at all.

  * `bm25`: keyword matching over names and descriptions, no model at all;
  * `embed`: cosine similarity of embeddings from EuLLM's `/v1/embeddings`;
    the tools' vectors are computed once, as a deployment would, so a
    request costs one embedding;
  * `reflex-a`: `/v1/systemone`, the request as the state, the tools as the
    options of a `choice` question, ranked by the model's verdict score;
  * `reflex-b`: the catalog as the state, the request inside the question,
    the options the tool names alone. With the same catalog on every
    request the state is read once and reused, so a request costs its
    question and the names;
  * `two-stage`: the embeddings keep a shortlist, `reflex-a` ranks it.
"""

import json
import math
import re
import time
import urllib.error
import urllib.request

import rb_data

NONE = "none"  # the option that says no tool is needed


class Ranking:
    def __init__(self, order, ms, none_score=None, best_score=None, server=None, refused=False):
        self.order = order  # candidate names, most likely needed first
        self.ms = ms  # wall time of the decision, as the client sees it
        self.none_score = none_score  # Reflex's score for "no tool", if asked
        self.best_score = best_score  # Reflex's score for the first tool
        self.server = server or {}  # tokens and timings reported by EuLLM
        self.refused = refused  # the decision model could not take the request

    def abstains(self):
        """Reflex found "no tool" likelier than any tool."""
        return self.none_score is not None and (
            self.best_score is None or self.none_score > self.best_score
        )


def post(url, payload, api_key, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise ServerError(e.code, e.read().decode(errors="replace")) from None


class ServerError(Exception):
    def __init__(self, code, detail):
        super().__init__(f"HTTP {code}: {detail}")
        self.code, self.detail = code, detail


# --- BM25 -----------------------------------------------------------------


def words(text):
    """Lower-case words, with camelCase and snake_case names split apart."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25, document frequencies from every tool of the set."""

    name = "bm25"

    def __init__(self, dataset, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        docs = {}
        for item in dataset.items:
            for tool in item.candidates:
                docs.setdefault(tool.name + "\0" + tool.description, tool)
        self.n = len(docs)
        self.df = {}
        lengths = []
        for tool in docs.values():
            terms = words(tool.name + " " + tool.description)
            lengths.append(len(terms))
            for term in set(terms):
                self.df[term] = self.df.get(term, 0) + 1
        self.avg = sum(lengths) / max(1, len(lengths))

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def rank(self, item):
        started = time.perf_counter()
        query = words(item.request)
        scored = []
        for tool in item.candidates:
            terms = words(tool.name + " " + tool.description)
            tf = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            norm = self.k1 * (1 - self.b + self.b * len(terms) / self.avg)
            score = sum(
                self.idf(q) * tf[q] * (self.k1 + 1) / (tf[q] + norm) for q in query if q in tf
            )
            scored.append((score, tool.name))
        scored.sort(key=lambda s: -s[0])
        return Ranking([n for _, n in scored], (time.perf_counter() - started) * 1000)


# --- Embeddings ------------------------------------------------------------


class Embeddings:
    """Cosine similarity between the request and each tool, embedded by
    EuLLM with the embedding model named. `query_prefix` goes before each
    request, for the models trained with an instruction on the query side
    only, such as Qwen3-Embedding."""

    name = "embed"
    # Tool vectors by model and text, shared by every set of a run: MetaTool's
    # catalog is embedded once, not once per set.
    seen = {}

    def __init__(self, url, model, api_key, timeout, query_prefix=""):
        self.url = url.rstrip("/") + "/v1/embeddings"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.query_prefix = query_prefix

    def embed(self, texts):
        body = post(
            self.url,
            {"model": self.model, "input": texts},
            self.api_key,
            self.timeout,
        )
        return [row["embedding"] for row in sorted(body["data"], key=lambda r: r["index"])]

    def vectors(self, tools):
        texts = [f"{t.name}: {t.description}" for t in tools]
        missing = [x for x in dict.fromkeys(texts) if (self.model, x) not in self.seen]
        for start in range(0, len(missing), 32):
            batch = missing[start : start + 32]
            for text, vector in zip(batch, self.embed(batch), strict=True):
                self.seen[(self.model, text)] = unit(vector)
        return [self.seen[(self.model, x)] for x in texts]

    def rank(self, item):
        tools = self.vectors(item.candidates)  # once per tool, not timed
        started = time.perf_counter()
        query = unit(self.embed([self.query_prefix + item.request])[0])
        ms = (time.perf_counter() - started) * 1000
        scored = sorted(
            (
                (sum(a * b for a, b in zip(query, v, strict=True)), t.name)
                for t, v in zip(item.candidates, tools, strict=True)
            ),
            key=lambda s: -s[0],
        )
        return Ranking([n for _, n in scored], ms)


def unit(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


# --- Reflex ---------------------------------------------------------------

QUESTION_A = "Which tool is needed to handle this request?"
QUESTION_B = 'Which of the tools listed does this request need? The request: "{request}"'
NONE_TEXT = "none of these tools: the request can be handled without any"


class Reflex:
    """`/v1/systemone`, in layout A or B.

    A question lists 2 to 255 options, and the Jev-Style 0.8B allows 2,048
    tokens for a question with its options, read twice. Each request starts
    with its tools in one question; when the server says a question is too
    long, they are split over twice as many, and so on, up to 64 questions.
    The verdict scores of all of them, one per option, give the ranking —
    compared as they are, although each option was read next to the other
    options of its own question only. "No tool" is offered in every
    question, and weighed against the best tool in that tool's question.

    A request the model cannot read even beside a single tool — a long
    request inside a layout-B question, say — is refused, as the server
    refuses it rather than truncate it, and ranks nothing."""

    def __init__(self, url, layout, model, api_key, timeout, abstain):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.layout, self.model = layout, model
        self.api_key, self.timeout, self.abstain = api_key, timeout, abstain
        self.name = f"reflex-{layout.lower()}"
        # With "none" among the options one tool makes a question; without
        # it, parts of at least 3 tools keep every question at 2 options or
        # more.
        self.largest = 254 if abstain else 255
        self.fewest = 1 if abstain else 3
        self.chunk = self.largest  # tools per question in the last request

    def state_and_questions(self, item, chunks):
        if self.layout == "A":
            state = item.request
            instructions = QUESTION_A
            describe = lambda t: t.description or None  # noqa: E731
        else:
            state = "The tools available:\n" + "\n".join(
                f"- {t.name}: {t.description}" for t in item.candidates
            )
            instructions = QUESTION_B.format(request=item.request)
            describe = lambda t: None  # noqa: E731  (the name alone)
        questions = {}
        for n, chunk in enumerate(chunks):
            criteria = {t.name: describe(t) for t in chunk}
            if self.abstain:
                criteria[NONE] = NONE_TEXT
            questions[f"tools_{n}"] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        return state, questions

    def rank(self, item):
        names = [t.name for t in item.candidates]
        if self.abstain and NONE in names:
            raise ValueError(f"{item.id}: a tool is named {NONE!r}")
        if len(names) == 1 and not self.abstain:
            # One tool and no "none" to weigh it against: nothing to decide.
            return Ranking(names, 0.0, server={"questions": 0})
        # Every request starts from the largest questions: the size a long
        # request needed says nothing about the next one.
        chunk = self.largest
        while True:
            chunks = split(item.candidates, chunk)
            if len(chunks) > 64:
                return Ranking([], 0.0, server={"questions": 0}, refused=True)
            state, questions = self.state_and_questions(item, chunks)
            payload = {"state": state, "questions": questions}
            if self.model:
                payload["model"] = self.model
            started = time.perf_counter()
            try:
                body = post(self.url, payload, self.api_key, self.timeout)
            except ServerError as e:
                # Too long for the model, refused whole: a 422 with the code
                # `input_budget_exceeded` since EuLLM answers /v1/systemone
                # errors in the System One shape, a 400 before; the message
                # says "nothing was truncated" in both.
                if e.code not in (400, 422) or "nothing was truncated" not in e.detail:
                    raise
                if chunk > self.fewest:
                    chunk = max(self.fewest, min(chunk, len(names)) // 2)
                    continue
                return Ranking([], 0.0, server={"questions": 0}, refused=True)
            ms = (time.perf_counter() - started) * 1000
            break
        self.chunk = chunk
        scores, question_of, nones = {}, {}, {}
        for question, answer in body["answers"].items():
            raw = (answer.get("eullm") or {}).get("scores")
            if raw is None:
                raise ValueError(
                    "the decision model does not report verdict scores: "
                    "ReflexBench ranks with a Jev-Style model"
                )
            for name, score in raw.items():
                if name == NONE:
                    nones[question] = score
                else:
                    scores[name], question_of[name] = score, question
        order = sorted(names, key=lambda n: -scores[n])
        info = body.get("eullm", {})
        server = {
            "evaluated_tokens": info.get("evaluated_tokens"),
            "prompt_tokens": info.get("prompt_tokens"),
            "prefix_reused": info.get("prefix_reused"),
            "request_ms": info.get("request_ms"),
            "questions": len(questions),
            "model": body.get("model"),  # the decision model that answered
        }
        best = scores[order[0]]
        return Ranking(order, ms, nones.get(question_of[order[0]]), best, server)


# --- Two stages ------------------------------------------------------------


class TwoStage:
    """The embeddings keep the `shortlist` tools likeliest for the request,
    Reflex ranks those, and the others follow in the embeddings' order: the
    shape a deployment takes when its catalog is too large to read on every
    request. The shortlist goes to Reflex in the catalog's order, not the
    embeddings', so that Reflex judges it on its own. Its cost is both
    stages together."""

    name = "two-stage"

    def __init__(self, embeddings, reflex, shortlist):
        self.embeddings, self.reflex, self.shortlist = embeddings, reflex, shortlist
        self.abstain = reflex.abstain

    def rank(self, item):
        first = self.embeddings.rank(item)
        kept = set(first.order[: self.shortlist])
        short = [t for t in item.candidates if t.name in kept]
        second = self.reflex.rank(rb_data.Item(item.id, item.request, short, item.needed))
        if second.refused:
            # Reflex could not read the shortlist: the embeddings' order stands.
            return Ranking(first.order, first.ms, server=second.server, refused=True)
        order = second.order + first.order[self.shortlist :]
        return Ranking(
            order, first.ms + second.ms, second.none_score, second.best_score, second.server
        )


def split(tools, size):
    """`tools` over as few questions of at most `size` as it takes, as even
    as they can be: 199 in questions of 99 make 67, 66 and 66, not 99, 99
    and 1."""
    n = math.ceil(len(tools) / size)
    each, extra = divmod(len(tools), n)
    parts, start = [], 0
    for i in range(n):
        end = start + each + (i < extra)
        parts.append(tools[start:end])
        start = end
    return parts
