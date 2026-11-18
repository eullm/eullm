"""What the tools decide, and how: the methods ReflexBench measured
(bench/reflexbench, docs/reflex-roadmap.md, MVP 0 and MVP 1).

  * Tool selection, in two stages (`rb_methods.TwoStage`). For one tool out
    of MetaTool's 199, EuLLM's embeddings ranked the right one first 75.4%
    of the time in 51 ms on an RTX 5070 Ti; the Jev-Style 2B reading the
    whole catalog, 73.4% in 439 ms, and 76 s a decision on a 4-core CPU.
    What the model adds is the judgement on few options — which of a
    handful, several at once, or none of them — so the embeddings keep a
    shortlist and the model ranks it next to a "none" option.
  * The RAG gate (`rg_methods.ReflexGate`): one `choice` among answer,
    retrieve_more and abstain, about the question and its passages. With a
    threshold on P(answer) fitted on the domain's own cases it stopped half
    again as many insufficient contexts as the embeddings' best similarity
    at the same cost in good answers; left to its own choice the 2B stopped
    65% of the contexts that did suffice.

The wording of every question, option and state is ReflexBench's, character
for character: the measurements are of these questions, and a threshold
fitted with ragbench.py holds only for the question it was fitted on. The
tests check that the two have not drifted apart.
"""

import collections
import math
import operator
import time
from array import array
from typing import Literal

from pydantic import BaseModel, Field

from eullm_reflex_mcp.eullm import EuLLMError

# --- Tool selection -----------------------------------------------------------

NONE = "none"  # the option that says no tool is needed
QUESTION = "Which tool is needed to handle this request?"
NONE_TEXT = "none of these tools: the request can be handled without any"

# The shortlist ReflexBench's two stages hand the model: 20 tools, about 613
# tokens a decision for the 2B, and the right tool among the first 10 for
# 95.2% of MetaTool's requests (the embeddings alone: 94.6%).
DEFAULT_SHORTLIST = 20

# The most tools the model reads in one call. On BFCL live, catalogs of 2 to
# 37 functions, the 2B ranked the right one first 93–95% of the time against
# the embeddings' 91%; over MetaTool's 199 it ranked no better than the
# embeddings, at nine times their latency on a GPU. Without an embedding
# model a catalog up to this size is read whole; a larger one is refused.
MAX_TOOLS = 37

# Tool vectors kept between calls: an agent sends the same catalog with every
# request, and on a 4-core CPU Qwen3-Embedding-0.6B took 0.14 s for a short
# tool description here and 0.3–0.4 s for a request in ReflexBench, so
# embedding MetaTool's 199 tools again would cost half a minute a call or
# more. 2,048 vectors of its 1,024 dimensions hold 16 MiB.
CACHED_VECTORS = 2048


class Tool(BaseModel):
    name: str = Field(min_length=1, description="The tool's name, unique in the catalog.")
    description: str = Field(
        "", description="What the tool does, as the agent sees it; one or two sentences work best."
    )


class RankedTool(BaseModel):
    name: str
    probability: float | None = Field(
        None,
        description="The decision model's probability that this is the tool the request needs, "
        "against the other tools ranked and 'none'; null when the model did not rank.",
    )


class ToolSelection(BaseModel):
    tools: list[RankedTool] = Field(
        description="The tools the decision model read, most likely needed first. Tools the "
        "embeddings left out of the shortlist are not listed."
    )
    none_probability: float | None = Field(
        None, description="The probability that no tool of the catalog is needed."
    )
    none_wins: bool | None = Field(
        None, description="'none' is likelier than the best tool: the request may need no tool."
    )
    method: Literal["two-stage", "reflex", "embeddings", "no decision"] = Field(
        description="two-stage: the embeddings kept the shortlist, the decision model ranked it. "
        "reflex: the model read the whole catalog. embeddings: the model could not read the "
        "shortlist (see note), the order is the embeddings'. no decision: one tool and no 'none'."
    )
    catalog_size: int
    left_out: int = Field(description="Tools of the catalog not in the shortlist.")
    questions: int = Field(
        description="Questions the tools were asked in: more than one when they did not fit the "
        "model's input budget in one."
    )
    decision_model: str | None = None
    embedding_model: str | None = None
    evaluated_tokens: int | None = Field(None, description="Tokens the decision model read.")
    ms: float = Field(description="Wall time of the call, in milliseconds.")
    note: str | None = None


class SelectionError(ValueError):
    """A catalog or argument the selection cannot work with."""


class Embedder:
    """Cosine similarity between a request and each tool, from EuLLM's
    `/v1/embeddings`, the way ReflexBench's `embed` method computes it: the
    tool as "name: description", the request after the query prefix."""

    def __init__(self, model, query_prefix="", capacity=CACHED_VECTORS):
        self.model, self.query_prefix, self.capacity = model, query_prefix, capacity
        self.cache = collections.OrderedDict()  # tool text -> unit vector, least recent first

    async def similarities(self, eullm, request, tools):
        texts = [f"{t.name}: {t.description}" for t in tools]
        vectors = {}
        for text in dict.fromkeys(texts):
            if text in self.cache:
                self.cache.move_to_end(text)
                vectors[text] = self.cache[text]
        missing = [text for text in dict.fromkeys(texts) if text not in vectors]
        for start in range(0, len(missing), 32):
            batch = missing[start : start + 32]
            for text, vector in zip(batch, await eullm.embeddings(self.model, batch), strict=True):
                vectors[text] = self.keep(text, unit(vector))
        query = unit((await eullm.embeddings(self.model, [self.query_prefix + request]))[0])
        return [dot(query, vectors[text]) for text in texts]

    def keep(self, text, vector):
        self.cache[text] = vector
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return vector


def unit(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return array("d", (x / norm for x in vector))


def dot(a, b):
    return sum(map(operator.mul, a, b))


async def select_tools(eullm, embedder, request, tools, shortlist, allow_none):
    """Rank `tools` for `request`: the embeddings keep `shortlist` when the
    catalog is larger, the decision model ranks what is kept. `embedder` is
    None when no embedding model is configured."""
    started = time.perf_counter()
    names = [t.name for t in tools]
    if not names:
        raise SelectionError("no tools to select from")
    if not 1 <= shortlist <= MAX_TOOLS:
        raise SelectionError(f"shortlist must be between 1 and {MAX_TOOLS}, not {shortlist}")
    if len(set(names)) != len(names):
        twice = sorted({n for n in names if names.count(n) > 1})
        raise SelectionError(f"tool names must be unique; given more than once: {', '.join(twice)}")
    if allow_none and NONE in names:
        raise SelectionError(
            f"a tool is named {NONE!r}, the name of the option that says no tool is needed: "
            "rename it, or pass allow_none=false"
        )

    def result(**fields):
        fields.setdefault("catalog_size", len(tools))
        fields.setdefault("left_out", 0)
        fields.setdefault("questions", 0)
        return ToolSelection(ms=round((time.perf_counter() - started) * 1000, 1), **fields)

    shortlisting = len(tools) > shortlist
    if shortlisting and embedder is None and len(tools) > MAX_TOOLS:
        raise SelectionError(
            f"{len(tools)} tools and no embedding model to shortlist them: set EULLM_EMBED_MODEL "
            f"(e.g. qwen3-embedding-0.6b-gguf-q8_0) in this MCP server's environment. Without "
            f"one, the decision model reads catalogs of up to {MAX_TOOLS} tools whole; over "
            "larger ones it ranks no better than the embeddings, at many times their cost"
        )
    candidates, by_similarity = tools, None
    if shortlisting and embedder is not None:
        similarity = await embedder.similarities(eullm, request, tools)
        by_similarity = sorted(range(len(tools)), key=lambda i: -similarity[i])
        kept = set(by_similarity[:shortlist])
        # In the catalog's order, not the embeddings', so that the model
        # judges the shortlist on its own.
        candidates = [t for i, t in enumerate(tools) if i in kept]
    two_stage = by_similarity is not None
    left_out = len(tools) - len(candidates)
    if len(candidates) == 1 and not allow_none:
        # One tool left and no "none" to weigh it against: the answer is known
        # before the model reads anything, and EuLLM refuses a choice of one.
        # Tested on what the model would actually be given -- after the
        # shortlisting, not before it. Shortlisting to one tool and asking
        # with allow_none=false used to build a choice of a single option,
        # which the engine answers 422 invalid_question to, and the split
        # loop cannot retry: one tool is already the smallest it goes.
        return result(
            tools=[RankedTool(name=candidates[0].name)],
            method="no decision",
            left_out=left_out,
            embedding_model=embedder.model if two_stage else None,
            note=(
                "the shortlist kept one tool and allow_none=false: there is nothing to decide"
                if two_stage
                else "one tool and allow_none=false: there is nothing to decide"
            ),
        )
    try:
        ranking = await rank(eullm, request, candidates, allow_none)
    except EuLLMError as e:
        if not two_stage or e.code != "input_budget_exceeded":
            raise
        # The model could not read the shortlist even one tool a question:
        # the embeddings' order stands, as in ReflexBench's two stages.
        return result(
            tools=[RankedTool(name=tools[i].name) for i in by_similarity[:shortlist]],
            method="embeddings",
            left_out=left_out,
            embedding_model=embedder.model,
            note=f"the decision model could not read the shortlist, so the embeddings rank it: {e}",
        )
    none = ranking.none
    return result(
        tools=[RankedTool(name=n, probability=round(p, 4)) for n, p in ranking.tools],
        none_probability=None if none is None else round(none, 4),
        none_wins=None if none is None else none > ranking.tools[0][1],
        method="two-stage" if two_stage else "reflex",
        left_out=left_out,
        questions=ranking.questions,
        decision_model=ranking.model,
        embedding_model=embedder.model if two_stage else None,
        evaluated_tokens=ranking.evaluated_tokens,
    )


class Ranking:
    def __init__(self, tools, none, questions, model, evaluated_tokens):
        self.tools = tools  # (name, probability), most likely first
        self.none = none  # the probability of "none", if it was offered
        self.questions, self.model, self.evaluated_tokens = questions, model, evaluated_tokens


async def rank(eullm, request, candidates, allow_none):
    """The model ranks `candidates` for `request`: the request is the state,
    the tools with their descriptions the options of a `choice` (layout A).

    All of them start in one question. A question too long for the model's
    input budget — the Jev-Style 0.8B allows 2,048 tokens for a question
    with its options, read twice — is refused whole by EuLLM rather than
    truncated, and the tools are then split over twice as many questions,
    and so on. "None" is offered in every question and weighed against the
    best tool in that tool's question."""
    # With "none" among the options one tool makes a question; without it,
    # parts of at least 3 tools, split evenly, keep every question at 2.
    fewest = 1 if allow_none else 3
    size = len(candidates)
    while True:
        questions = {}
        for n, part in enumerate(split(candidates, size)):
            criteria = {t.name: t.description or None for t in part}
            if allow_none:
                criteria[NONE] = NONE_TEXT
            questions[f"tools_{n}"] = {
                "type": "choice",
                "instructions": QUESTION,
                "criteria": criteria,
            }
        try:
            response = await eullm.systemone({"state": request, "questions": questions})
            break
        except EuLLMError as e:
            if e.code != "input_budget_exceeded" or size <= fewest:
                raise
            size = max(fewest, size // 2)
    try:
        return ranking(response, [t.name for t in candidates], allow_none)
    except (KeyError, TypeError, ValueError) as e:
        raise EuLLMError(
            f"EuLLM's answer to /v1/systemone is not the one expected: {e!r}"
        ) from None


def ranking(response, names, has_none=True):
    """The tools of `names` in order, and the "none" option's probability.

    ``has_none`` says whether the "none" option was offered at all. It is not
    the same question as ``NONE in names``: a catalog may hold a tool called
    "none" -- allow_none=False is the documented way to have one -- and then
    its probability must be the tool's, not an abstention that was never on the
    table.
    """
    answers = response["answers"]
    info = response.get("eullm") or {}
    if len(answers) == 1:
        # One question: EuLLM's own probabilities, calibrated with the
        # model's temperature.
        (answer,) = answers.values()
        probabilities = answer["probabilities"]
        tool = {n: probabilities[n] for n in names}
        none = probabilities.get(NONE) if has_none else None
    else:
        tool, none = merged(answers, names, info.get("temperature") or 1.0, has_none)
    order = sorted(names, key=lambda n: -tool[n])
    return Ranking(
        [(n, tool[n]) for n in order],
        none,
        len(answers),
        response.get("model"),
        info.get("evaluated_tokens"),
    )


def merged(answers, names, temperature, has_none=True):
    """Probabilities over the tools of several questions and one "none":
    the softmax of the verdict scores over the model's temperature — how
    EuLLM computes them within one question — with the scores compared as
    they are, although each tool was read next to its own question's
    options only, as ReflexBench ranks them. "None" is the one from the
    question of the best tool, and only if one was offered."""
    scores, question_of, nones = {}, {}, {}
    for question, answer in answers.items():
        raw = (answer.get("eullm") or {}).get("scores")
        if raw is None:
            raise EuLLMError(
                "the shortlist had to be split over several questions, and the decision model "
                "gives no verdict scores to compare across them (it reads answers by codes): use "
                "a Jev-Style decision model, or shorten the tool descriptions"
            )
        for name, score in raw.items():
            if name == NONE and has_none:
                nones[question] = score
            else:
                scores[name], question_of[name] = score, question
    best = max(names, key=lambda n: scores[n])
    none = nones.get(question_of[best]) if has_none else None
    keys = names + ([NONE] if none is not None else [])
    values = [scores[n] for n in names] + ([none] if none is not None else [])
    top = max(values)
    weights = [math.exp((v - top) / temperature) for v in values]
    total = sum(weights)
    probabilities = {k: w / total for k, w in zip(keys, weights, strict=True)}
    # The probability, not the score it came from, and only when the option was
    # offered at all: with a tool called "none" among the names, that key in
    # `probabilities` is the tool's.
    return {n: probabilities[n] for n in names}, (probabilities.get(NONE) if has_none else None)


def split(tools, size):
    """`tools` over as few questions of at most `size` as it takes, as even
    as they can be: 37 in questions of 18 make 13, 12 and 12, not 18, 18
    and 1."""
    n = math.ceil(len(tools) / size)
    each, extra = divmod(len(tools), n)
    parts, start = [], 0
    for i in range(n):
        end = start + each + (i < extra)
        parts.append(tools[start:end])
        start = end
    return parts


# --- The RAG gate ---------------------------------------------------------------

LABELS = ("answer", "retrieve_more", "abstain")
GATE = "Can the question be answered from these passages alone?"
OPTIONS = {
    "answer": "Yes: together, the passages hold every fact the answer needs.",
    "retrieve_more": (
        "Partly: they hold some of the facts the answer needs, and at least one is missing."
    ),
    "abstain": "No: none of the passages holds anything the answer needs.",
}

UNCALIBRATED = (
    "decided by the model's own choice, with no calibrated threshold: it is far too cautious "
    "(on MuSiQue the Jev-Style 2B stopped 65% of the contexts that did suffice). Calibrate a "
    "threshold on P(answer) on labelled cases of your own and set REFLEX_GATE_THRESHOLD"
)


class GateDecision(BaseModel):
    decision: Literal["answer", "retrieve_more", "abstain"] = Field(
        description="answer: the passages suffice. retrieve_more: some facts are there, at least "
        "one is missing. abstain: nothing there helps."
    )
    decided_by: Literal["threshold", "model_choice"] = Field(
        description="threshold: P(answer) against the calibrated threshold, the likelier of "
        "retrieve_more and abstain below it. model_choice: the model's own, uncalibrated choice."
    )
    threshold: float | None = None
    threshold_source: Literal["argument", "REFLEX_GATE_THRESHOLD"] | None = None
    model_choice: Literal["answer", "retrieve_more", "abstain"]
    probabilities: dict[str, float] = Field(
        description="The decision model's probability of answer, retrieve_more and abstain."
    )
    decision_model: str | None = None
    evaluated_tokens: int | None = None
    ms: float
    note: str | None = None


def gate_state(question, passages):
    """The question and its passages, numbered, as ragbench.py writes them."""
    numbered = "\n\n".join(f"[{n}] {p}" for n, p in enumerate(passages, 1))
    return f"Question: {question}\n\nPassages:\n{numbered}"


async def rag_gate(eullm, question, passages, threshold=None, source=None):
    started = time.perf_counter()
    body = {
        "state": gate_state(question, passages),
        "questions": {"q": {"type": "choice", "instructions": GATE, "criteria": OPTIONS}},
    }
    response = await eullm.systemone(body)
    try:
        answer = response["answers"]["q"]
        probabilities = {label: answer["probabilities"][label] for label in LABELS}
        own = answer["choice"]
    except (KeyError, TypeError) as e:
        raise EuLLMError(
            f"EuLLM's answer to /v1/systemone is not the one expected: {e!r}"
        ) from None
    if threshold is None:
        decision, decided_by, note = own, "model_choice", UNCALIBRATED
    else:
        # ragbench.py fits the threshold as "answer at or above it".
        if probabilities["answer"] >= threshold:
            decision = "answer"
        else:
            decision = max(("retrieve_more", "abstain"), key=probabilities.get)
        decided_by, note = "threshold", None
    return GateDecision(
        decision=decision,
        decided_by=decided_by,
        threshold=threshold,
        threshold_source=source if threshold is not None else None,
        model_choice=own,
        probabilities={k: round(v, 4) for k, v in probabilities.items()},
        decision_model=response.get("model"),
        evaluated_tokens=(response.get("eullm") or {}).get("evaluated_tokens"),
        ms=round((time.perf_counter() - started) * 1000, 1),
        note=note,
    )
