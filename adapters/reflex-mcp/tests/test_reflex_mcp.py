"""Offline tests for the Reflex MCP server: every tool called through the MCP
SDK's in-memory client against a stand-in EuLLM, the settings, the wording
shared with ReflexBench, and the command itself on stdio and on streamable
HTTP. No network beyond 127.0.0.1, no model:

    python3 -m unittest discover -s adapters/reflex-mcp/tests -v
"""

import asyncio
import contextlib
import io
import logging
import math
import os
import pathlib
import socket
import subprocess
import sys
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "src"
BENCH = HERE.parents[2] / "bench" / "reflexbench"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import standin  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402

from eullm_reflex_mcp import reflex  # noqa: E402
from eullm_reflex_mcp.config import Config, ConfigError  # noqa: E402
from eullm_reflex_mcp.server import build_server, main  # noqa: E402

# The SDK logs every tool error at INFO, and many of these tests make one on
# purpose. Configured first, so the SDK's own basicConfig does nothing.
logging.basicConfig(level=logging.WARNING)

EMBED = "qwen3-embedding"
PREFIX = "Instruct: Given a user request, retrieve the tools needed to handle it\nQuery:"
RAIN = "Will it rain in Rome tomorrow?"

CATALOG = [
    {"name": "get_weather", "description": "Current weather and forecast for a city"},
    {"name": "send_email", "description": "Send an email message to a recipient"},
    {"name": "create_invoice", "description": "Create an invoice for a customer"},
    {"name": "search_web", "description": ""},
    {"name": "stock_price", "description": "Latest price of a stock ticker"},
    {"name": "translate", "description": "Translate text between two languages"},
]

# Embeddings: the request is [1, 0, 0]; get_weather is closest, then
# stock_price and translate; the other three are orthogonal to it.
VECTORS = {
    PREFIX + RAIN: [1.0, 0.0, 0.0],
    "get_weather: Current weather and forecast for a city": [0.9, 0.1, 0.0],
    "stock_price: Latest price of a stock ticker": [0.6, 0.8, 0.0],
    "translate: Translate text between two languages": [0.5, 0.0, 0.866],
}
SCORES = {"get_weather": 2.0, "stock_price": -1.0, "translate": 0.5, "none": -2.0}


class ToolFailed(Exception):
    """The tool answered with is_error, as an agent would see it."""


class Agent:
    """An MCP client connected in memory to the server built for `settings`,
    through the JSON-RPC streams and initialize handshake stdio clients use."""

    def __init__(self, url, **settings):
        self.server = build_server(Config(url=url, **settings))

    async def __aenter__(self):
        self.client = Client(self.server, mode="legacy")
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self.client.__aexit__(*exc)

    async def call(self, tool, **arguments):
        return await self.client.call_tool(tool, arguments)


def unwrapped(result):
    """The tool's structured result, or ToolFailed with the text an agent
    reads. Raised outside the client's context: inside it, the SDK's task
    groups would wrap it in an exception group."""
    if result.is_error:
        raise ToolFailed(result.content[0].text)
    return result.structured_content


async def call(url, tool, settings=None, **arguments):
    async with Agent(url, **(settings or {})) as agent:
        result = await agent.call(tool, **arguments)
    return unwrapped(result)


def expected(scores, temperature=0.88):
    return {k: round(v, 4) for k, v in standin.softmax(scores, temperature).items()}


class SelectToolsTest(unittest.IsolatedAsyncioTestCase):
    two_stage = {"embed_model": EMBED, "embed_query_prefix": PREFIX}

    async def test_two_stages_rank_the_embeddings_shortlist(self):
        with standin.StandIn(scores=SCORES, vectors=VECTORS) as eullm:
            result = await call(
                eullm.url, "select_tools", self.two_stage, request=RAIN, tools=CATALOG, shortlist=3
            )
        # The tools as "name: description", the request after the prefix.
        tools_in, query_in = eullm.sent("/v1/embeddings")
        self.assertEqual(tools_in["model"], EMBED)
        self.assertEqual(tools_in["input"], [f"{t['name']}: {t['description']}" for t in CATALOG])
        self.assertEqual(query_in["input"], [PREFIX + RAIN])
        # The three closest, in the catalog's order, and "none".
        (body,) = eullm.sent("/v1/systemone")
        self.assertEqual(body["state"], RAIN)
        self.assertNotIn("model", body)
        self.assertEqual(list(body["questions"]), ["tools_0"])
        question = body["questions"]["tools_0"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["instructions"], reflex.QUESTION)
        self.assertEqual(
            question["criteria"],
            {
                "get_weather": "Current weather and forecast for a city",
                "stock_price": "Latest price of a stock ticker",
                "translate": "Translate text between two languages",
                "none": reflex.NONE_TEXT,
            },
        )
        p = expected(SCORES)
        self.assertEqual(
            result["tools"],
            [
                {"name": "get_weather", "probability": p["get_weather"]},
                {"name": "translate", "probability": p["translate"]},
                {"name": "stock_price", "probability": p["stock_price"]},
            ],
        )
        self.assertEqual(result["none_probability"], p["none"])
        self.assertFalse(result["none_wins"])
        self.assertEqual(result["method"], "two-stage")
        self.assertEqual(
            (result["catalog_size"], result["left_out"], result["questions"]), (6, 3, 1)
        )
        self.assertEqual(result["decision_model"], standin.MODEL)
        self.assertEqual(result["embedding_model"], EMBED)
        self.assertEqual(result["evaluated_tokens"], 42)
        self.assertIsNone(result["note"])

    async def test_the_catalog_is_embedded_once(self):
        with standin.StandIn(scores=SCORES, vectors=VECTORS) as eullm:
            async with Agent(eullm.url, **self.two_stage) as agent:
                results = [
                    await agent.call("select_tools", request=RAIN, tools=CATALOG, shortlist=3)
                    for _ in range(2)
                ]
        for result in results:
            self.assertEqual(unwrapped(result)["tools"][0]["name"], "get_weather")
        inputs = [body["input"] for body in eullm.sent("/v1/embeddings")]
        self.assertEqual(len(inputs[0]), 6)
        self.assertEqual(inputs[1:], [[PREFIX + RAIN], [PREFIX + RAIN]])

    async def test_a_catalog_within_the_shortlist_is_read_whole(self):
        for settings in ({}, self.two_stage):
            with standin.StandIn(scores=SCORES) as eullm:
                result = await call(
                    eullm.url, "select_tools", settings, request=RAIN, tools=CATALOG
                )
            self.assertEqual(eullm.sent("/v1/embeddings"), [], settings)
            self.assertEqual(result["method"], "reflex")
            self.assertIsNone(result["embedding_model"])
            self.assertEqual(result["left_out"], 0)
            (body,) = eullm.sent("/v1/systemone")
            criteria = body["questions"]["tools_0"]["criteria"]
            self.assertEqual(list(criteria), [t["name"] for t in CATALOG] + ["none"])
            # A tool with no description is offered by its name alone.
            self.assertIsNone(criteria["search_web"])

    async def test_without_embeddings_up_to_37_tools_are_read_whole(self):
        catalog = [{"name": f"tool_{n}", "description": f"does {n}"} for n in range(38)]
        with standin.StandIn() as eullm:
            result = await call(eullm.url, "select_tools", request=RAIN, tools=catalog[:37])
            self.assertEqual(result["method"], "reflex")
            self.assertEqual(len(result["tools"]), 37)
            self.assertEqual(
                len(eullm.sent("/v1/systemone")[0]["questions"]["tools_0"]["criteria"]), 38
            )
            eullm.requests.clear()
            with self.assertRaises(ToolFailed) as refused:
                await call(eullm.url, "select_tools", request=RAIN, tools=catalog)
        self.assertIn("38 tools and no embedding model", str(refused.exception))
        self.assertIn("EULLM_EMBED_MODEL", str(refused.exception))
        self.assertEqual(eullm.requests, [])

    async def test_a_question_too_long_is_split(self):
        tools = [{"name": n, "description": f"{n} tool"} for n in "abcd"]
        scores = {
            "a": 0.1,
            "b": -1.0,
            "c": 2.0,
            "d": 0.5,
            ("tools_0", "none"): 5.0,
            ("tools_1", "none"): -3.0,
        }
        with standin.StandIn(scores=scores, max_options=3) as eullm:
            result = await call(eullm.url, "select_tools", request=RAIN, tools=tools)
        # Four tools and "none" refused; two questions of two and "none" fit.
        sent = eullm.sent("/v1/systemone")
        self.assertEqual([len(b["questions"]) for b in sent], [1, 2])
        self.assertEqual(list(sent[1]["questions"]["tools_0"]["criteria"]), ["a", "b", "none"])
        self.assertEqual(list(sent[1]["questions"]["tools_1"]["criteria"]), ["c", "d", "none"])
        self.assertEqual([t["name"] for t in result["tools"]], ["c", "d", "a", "b"])
        # "None" wins among a and b, but is weighed beside c, the best tool:
        # one softmax over the scores, at the model's temperature.
        p = expected({"a": 0.1, "b": -1.0, "c": 2.0, "d": 0.5, "none": -3.0})
        self.assertEqual(result["none_probability"], p["none"])
        self.assertEqual(result["tools"][0]["probability"], p["c"])
        self.assertFalse(result["none_wins"])
        self.assertEqual(result["questions"], 2)

    async def test_a_tool_called_none_survives_the_split_over_questions(self):
        """Merging the questions files a score under "none" as an abstention.

        The tool's own score went into `nones`, so it never reached `scores`
        and the tool dropped out of the ranking altogether -- KeyError on the
        best tool. The same catalog of four answers in one question and split
        over two.
        """
        tools = [
            {"name": "none", "description": "does nothing"},
            {"name": "a", "description": "does a"},
            {"name": "b", "description": "does b"},
            {"name": "c", "description": "does c"},
        ]
        scores = {"none": 2.0, "a": 1.0, "b": 0.0, "c": -1.0}
        with standin.StandIn(scores=scores, max_options=2) as eullm:
            result = await call(
                eullm.url, "select_tools", request=RAIN, tools=tools, allow_none=False
            )
        # The first request is one question of four and is refused as too long;
        # the second is two questions of two, which is the one answered.
        self.assertEqual([len(b["questions"]) for b in eullm.sent("/v1/systemone")], [1, 2])
        self.assertEqual([t["name"] for t in result["tools"]], ["none", "a", "b", "c"])
        self.assertIsNone(result["none_probability"])
        self.assertIsNone(result["none_wins"])

    async def test_a_shortlist_the_model_cannot_read_keeps_the_embeddings_order(self):
        with standin.StandIn(scores=SCORES, vectors=VECTORS, max_options=1) as eullm:
            result = await call(
                eullm.url, "select_tools", self.two_stage, request=RAIN, tools=CATALOG, shortlist=3
            )
        # One question of three tools, then three of one, then no smaller.
        self.assertEqual([len(b["questions"]) for b in eullm.sent("/v1/systemone")], [1, 3])
        self.assertEqual(result["method"], "embeddings")
        self.assertEqual(
            result["tools"],
            [
                {"name": "get_weather", "probability": None},
                {"name": "stock_price", "probability": None},
                {"name": "translate", "probability": None},
            ],
        )
        self.assertIsNone(result["none_probability"])
        self.assertIn("nothing was truncated", result["note"])

    async def test_without_embeddings_a_refusal_is_the_agents_to_read(self):
        with standin.StandIn(scores=SCORES, max_options=1) as eullm:
            with self.assertRaises(ToolFailed) as refused:
                await call(eullm.url, "select_tools", request=RAIN, tools=CATALOG[:2])
        message = str(refused.exception)
        self.assertIn("422 input_budget_exceeded", message)
        self.assertIn("nothing was truncated", message)
        self.assertIn("(question 'tools_0')", message)

    async def test_other_errors_are_not_retried(self):
        with standin.StandIn(decision_model=None) as eullm:
            with self.assertRaises(ToolFailed) as refused:
                await call(eullm.url, "select_tools", request=RAIN, tools=CATALOG)
        self.assertIn("400 model_not_loaded: no decision model is loaded", str(refused.exception))
        self.assertEqual(len(eullm.sent("/v1/systemone")), 1)

    async def test_without_none(self):
        with standin.StandIn(scores=SCORES) as eullm:
            result = await call(
                eullm.url, "select_tools", request=RAIN, tools=CATALOG, allow_none=False
            )
            self.assertNotIn(
                "none", eullm.sent("/v1/systemone")[0]["questions"]["tools_0"]["criteria"]
            )
            self.assertIsNone(result["none_probability"])
            self.assertIsNone(result["none_wins"])
            self.assertAlmostEqual(sum(t["probability"] for t in result["tools"]), 1.0, places=3)
            # One tool and nothing to weigh it against: no question at all.
            eullm.requests.clear()
            alone = await call(
                eullm.url, "select_tools", request=RAIN, tools=CATALOG[:1], allow_none=False
            )
            self.assertEqual(eullm.requests, [])
        self.assertEqual(alone["method"], "no decision")
        self.assertEqual(alone["tools"], [{"name": "get_weather", "probability": None}])

    async def test_a_shortlist_of_one_and_no_none_is_not_a_question_of_one_option(self):
        """Shortlisting can leave one tool, and one option is not a question.

        EuLLM refuses a choice of one, and the split loop cannot retry: one
        tool is already the smallest it goes. The check that skips the
        question was made on the catalog, before the embeddings had picked
        the shortlist.
        """
        with standin.StandIn(scores=SCORES, vectors=VECTORS) as eullm:
            result = await call(
                eullm.url,
                "select_tools",
                self.two_stage,
                request=RAIN,
                tools=CATALOG,
                shortlist=1,
                allow_none=False,
            )
        # The decision model is not asked. The embeddings are -- that is how
        # the shortlist of one was chosen -- and then there is nothing that
        # could go in a question.
        self.assertEqual(eullm.sent("/v1/systemone"), [])
        self.assertEqual(result["method"], "no decision")
        self.assertEqual(result["embedding_model"], EMBED)
        self.assertEqual(result["tools"], [{"name": "get_weather", "probability": None}])
        self.assertEqual((result["catalog_size"], result["left_out"]), (6, 5))
        self.assertIsNone(result["none_probability"])
        self.assertIsNone(result["none_wins"])

    async def test_one_tool_and_none_is_a_decision(self):
        with standin.StandIn(scores={"get_weather": -1.0, "none": 1.0}) as eullm:
            result = await call(eullm.url, "select_tools", request="Hello!", tools=CATALOG[:1])
        self.assertTrue(result["none_wins"])
        self.assertGreater(result["none_probability"], 0.5)

    async def test_names_must_be_unique_and_none_is_reserved(self):
        with standin.StandIn(scores=SCORES) as eullm:
            with self.assertRaises(ToolFailed) as twice:
                await call(eullm.url, "select_tools", request=RAIN, tools=CATALOG + CATALOG[:1])
            self.assertIn("given more than once: get_weather", str(twice.exception))
            named_none = CATALOG[:2] + [{"name": "none", "description": "does nothing"}]
            with self.assertRaises(ToolFailed) as reserved:
                await call(eullm.url, "select_tools", request=RAIN, tools=named_none)
            self.assertIn("allow_none=false", str(reserved.exception))
            self.assertEqual(eullm.requests, [])
            allowed = await call(
                eullm.url, "select_tools", request=RAIN, tools=named_none, allow_none=False
            )
        # allow_none=False is the documented way to have a tool called "none",
        # and it is ranked as the tool it is, at the place its score puts it.
        # The "none" option was not on the table, so there is no probability
        # of abstaining to report -- and it must not be the tool's own, which
        # is what reading the name back out of the answer gave.
        # Resolved the way the stand-in resolves it, default included.
        asked = ("get_weather", "send_email", "none")
        p = expected({n: SCORES.get(n, 0.0) for n in asked})
        self.assertEqual([t["name"] for t in allowed["tools"]], list(asked))
        self.assertAlmostEqual(allowed["tools"][2]["probability"], p["none"], places=3)
        self.assertIsNone(allowed["none_probability"])
        self.assertIsNone(allowed["none_wins"])
        self.assertAlmostEqual(sum(t["probability"] for t in allowed["tools"]), 1.0, places=3)

    async def test_the_arguments_are_checked(self):
        with standin.StandIn() as eullm:
            for arguments in (
                {"tools": CATALOG, "shortlist": 38},
                {"tools": CATALOG, "shortlist": 0},
                {"tools": []},
                {"tools": [{"name": "", "description": "x"}]},
            ):
                with self.assertRaises(ToolFailed, msg=arguments):
                    await call(eullm.url, "select_tools", request=RAIN, **arguments)
        self.assertEqual(eullm.requests, [])

    async def test_a_model_without_verdict_scores_ranks_one_question_only(self):
        tools = [{"name": n, "description": f"{n} tool"} for n in "abcd"]
        with standin.StandIn(scores={"c": 1.0}, readout="codes") as eullm:
            result = await call(eullm.url, "select_tools", request=RAIN, tools=tools)
        self.assertEqual(result["tools"][0]["name"], "c")
        with standin.StandIn(scores={"c": 1.0}, readout="codes", max_options=3) as eullm:
            with self.assertRaises(ToolFailed) as refused:
                await call(eullm.url, "select_tools", request=RAIN, tools=tools)
        self.assertIn("no verdict scores", str(refused.exception))


class EmbedderTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_catalog_larger_than_the_cache(self):
        class EuLLM:
            async def embeddings(self, model, texts):
                return [[1.0, float(len(t))] for t in texts]

        embedder = reflex.Embedder("m", capacity=2)
        tools = [reflex.Tool(name=str(n), description="x" * n) for n in (1, 5, 9)]
        # Each text's vector leans further from the query, [1, 1], the
        # longer the text is.
        first = await embedder.similarities(EuLLM(), "q", tools)
        self.assertEqual(len(embedder.cache), 2)
        self.assertEqual(sorted(first, reverse=True), first)
        self.assertEqual(await embedder.similarities(EuLLM(), "q", tools), first)


class RagGateTest(unittest.IsolatedAsyncioTestCase):
    QUESTION = "Who wrote it?"
    PASSAGES = ["Book: written by Ann", "Ann: a writer"]
    # The model's own choice is abstain; P(answer) is 0.388.
    SCORES = {"answer": 1.0, "retrieve_more": 0.0, "abstain": 1.2}

    async def gate(self, settings=None, **arguments):
        with standin.StandIn(scores=self.SCORES) as eullm:
            result = await call(
                eullm.url,
                "rag_gate",
                settings,
                question=self.QUESTION,
                passages=self.PASSAGES,
                **arguments,
            )
        return result, eullm.sent("/v1/systemone")

    async def test_ragbenchs_question_about_the_passages(self):
        result, (body,) = await self.gate()
        self.assertEqual(
            body["state"],
            "Question: Who wrote it?\n\nPassages:\n[1] Book: written by Ann\n\n[2] Ann: a writer",
        )
        question = body["questions"]["q"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["instructions"], reflex.GATE)
        self.assertEqual(list(question["criteria"].items()), list(reflex.OPTIONS.items()))
        p = expected(self.SCORES)
        self.assertEqual(result["probabilities"], p)
        # No threshold: the model's own choice, and the agent is told so.
        self.assertEqual((result["decision"], result["decided_by"]), ("abstain", "model_choice"))
        self.assertIsNone(result["threshold"])
        self.assertIsNone(result["threshold_source"])
        self.assertIn("REFLEX_GATE_THRESHOLD", result["note"])
        self.assertEqual(result["decision_model"], standin.MODEL)

    async def test_a_calibrated_threshold_decides(self):
        result, _ = await self.gate(threshold=0.3)
        self.assertEqual((result["decision"], result["decided_by"]), ("answer", "threshold"))
        self.assertEqual((result["threshold"], result["threshold_source"]), (0.3, "argument"))
        self.assertEqual(result["model_choice"], "abstain")
        self.assertIsNone(result["note"])
        # Below it, the likelier of the other two.
        result, _ = await self.gate(threshold=0.5)
        self.assertEqual(result["decision"], "abstain")
        # At it, answer: ragbench.py fits "answer at or above it".
        p = standin.softmax(self.SCORES, 0.88)["answer"]
        result, _ = await self.gate(threshold=p)
        self.assertEqual(result["decision"], "answer")
        result, _ = await self.gate(threshold=math.nextafter(p, 1.0))
        self.assertEqual(result["decision"], "abstain")

    async def test_the_threshold_from_the_environment(self):
        result, _ = await self.gate({"gate_threshold": 0.3})
        self.assertEqual(result["decision"], "answer")
        self.assertEqual(result["threshold_source"], "REFLEX_GATE_THRESHOLD")
        # The argument comes first.
        result, _ = await self.gate({"gate_threshold": 0.3}, threshold=0.9)
        self.assertEqual(result["decision"], "abstain")
        self.assertEqual((result["threshold"], result["threshold_source"]), (0.9, "argument"))

    async def test_the_arguments_are_checked(self):
        for arguments in ({"threshold": 1.5}, {"passages": []}, {"question": ""}):
            with standin.StandIn() as eullm:
                with self.assertRaises(ToolFailed, msg=arguments):
                    await call(
                        eullm.url,
                        "rag_gate",
                        **{"question": "q", "passages": ["p"], **arguments},
                    )
            self.assertEqual(eullm.requests, [])


class DecideTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_request_and_the_response_pass_through(self):
        state = {"ticket": "Help! My payouts have been failing for 3 days."}
        questions = {
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle it?",
                "criteria": {"billing": "Payments and payouts", "tech": None},
            },
        }
        with standin.StandIn(scores={"Does this convey urgency?": 0.95, "billing": 3.0}) as eullm:
            result = await call(eullm.url, "decide", state=state, questions=questions)
        self.assertEqual(eullm.sent("/v1/systemone"), [{"state": state, "questions": questions}])
        self.assertEqual(result["answers"]["is_urgent"]["noul"], 0.95)
        self.assertEqual(result["answers"]["team"]["choice"], "billing")
        self.assertEqual(result["eullm"]["mode"], "shared_prefix")
        self.assertEqual(result["model"], standin.MODEL)

    async def test_eullms_error_reaches_the_agent(self):
        one = {"q": {"type": "choice", "instructions": "Which?", "criteria": {"only": None}}}
        with standin.StandIn() as eullm:
            with self.assertRaises(ToolFailed) as refused:
                await call(eullm.url, "decide", state="x", questions=one)
        self.assertIn(
            "EuLLM's /v1/systemone answered 422 invalid_question: a choice question needs 2 to "
            "255 options (question 'q')",
            str(refused.exception),
        )


class ModelInfoTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_decision_model_and_the_settings(self):
        settings = {"embed_model": EMBED, "embed_query_prefix": PREFIX, "gate_threshold": 0.42}
        with standin.StandIn() as eullm:
            result = await call(eullm.url, "model_info", settings)
        self.assertEqual(result["eullm_url"], eullm.url)
        self.assertEqual(result["eullm_version"], "0.7.22")
        self.assertEqual(
            result["decision_model"],
            {
                "name": standin.MODEL,
                "readout": "verdict",
                "context_tokens": 8192,
                "head_max_tokens": 2048,
            },
        )
        self.assertEqual(result["embedding_model"], EMBED)
        self.assertEqual(result["embed_query_prefix"], PREFIX)
        self.assertEqual(result["gate_threshold"], 0.42)
        self.assertEqual(result["max_tools_without_embeddings"], 37)
        self.assertIsNone(result["note"])

    async def test_no_decision_model(self):
        with standin.StandIn(decision_model=None) as eullm:
            result = await call(eullm.url, "model_info")
        self.assertIsNone(result["decision_model"])
        self.assertIn("--decision-model", result["note"])


class Redirecting(standin.StandIn):
    """A 302 that carries a JSON body -- what a gateway in front of EuLLM
    sends when it moves the caller, and the case that used to pass."""

    def route(self, method, path, headers, body):
        if (method, path) == ("GET", "/v1/models"):
            return 302, self.models()
        return super().route(method, path, headers, body)


class ConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_api_key_goes_as_a_bearer_token(self):
        with standin.StandIn(api_key="s3cret") as eullm:
            await call(eullm.url, "model_info", {"api_key": "s3cret"})
            self.assertEqual(eullm.requests[0][2]["Authorization"], "Bearer s3cret")
            with self.assertRaises(ToolFailed) as refused:
                await call(eullm.url, "model_info")
        self.assertIn("401 unauthorized: missing or invalid API key", str(refused.exception))

    async def test_an_eullm_that_is_not_there(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with self.assertRaises(ToolFailed) as refused:
            await call(f"http://127.0.0.1:{port}", "rag_gate", question="q", passages=["p"])
        self.assertIn(f"EuLLM is not reachable at http://127.0.0.1:{port}", str(refused.exception))

    async def test_an_eullm_that_does_not_answer_in_time(self):
        with standin.StandIn(delay=2.0) as eullm:
            with self.assertRaises(ToolFailed) as late:
                await call(eullm.url, "model_info", {"timeout": 0.3})
        self.assertIn("did not answer /v1/models within 0.3 s (EULLM_TIMEOUT)", str(late.exception))

    async def test_a_redirect_is_not_eullm_answering(self):
        """Only a 2xx is an answer. A 3xx used to be read as the model's own
        reply, body and all, so a gateway in front of EuLLM could have the
        tool report a model list EuLLM never sent."""
        with Redirecting() as eullm:
            with self.assertRaises(ToolFailed) as moved:
                await call(eullm.url, "model_info")
        self.assertIn("answered 302 redirect instead of an answer", str(moved.exception))
        self.assertIn("check EULLM_URL", str(moved.exception))


class ToolListTest(unittest.IsolatedAsyncioTestCase):
    async def test_four_read_only_tools_with_their_schemas(self):
        async with Agent("http://localhost:11434") as agent:
            listed = await agent.client.list_tools()
        tools = {t.name: t for t in listed.tools}
        self.assertEqual(set(tools), {"select_tools", "rag_gate", "decide", "model_info"})
        for tool in tools.values():
            self.assertTrue(tool.annotations.read_only_hint, tool.name)
            self.assertFalse(tool.annotations.open_world_hint, tool.name)
            self.assertIsNotNone(tool.output_schema, tool.name)
        shortlist = tools["select_tools"].input_schema["properties"]["shortlist"]
        self.assertEqual((shortlist["default"], shortlist["maximum"]), (20, 37))
        self.assertEqual(tools["rag_gate"].input_schema["required"], ["question", "passages"])
        self.assertEqual(tools["decide"].input_schema["required"], ["state", "questions"])


class ConfigTest(unittest.TestCase):
    def test_only_a_loopback_eullm_unless_allowed(self):
        for url in (
            "http://localhost:11434",
            "http://LOCALHOST:11434/",
            "http://127.0.0.1:11434",
            "http://127.8.9.10:1",
            "http://[::1]:11434",
            "http://[::ffff:127.0.0.1]:11434",
            "https://localhost",
        ):
            self.assertTrue(Config(url=url).loopback, url)
        for url in (
            "http://example.com:11434",
            "http://10.0.0.5:11434",
            "http://localhost.example.com",
            "http://[2001:db8::1]:11434",
        ):
            with self.assertRaises(ConfigError, msg=url):
                Config(url=url)
            self.assertFalse(Config(url=url, allow_remote=True).loopback, url)
        for url in ("ftp://localhost", "localhost:11434", "http://"):
            with self.assertRaises(ConfigError, msg=url):
                Config(url=url, allow_remote=True)

    def test_from_the_environment(self):
        config = Config.from_env({})
        self.assertEqual(config, Config())
        self.assertEqual(config.url, "http://localhost:11434")
        env = {
            "EULLM_URL": "http://127.0.0.1:11500/",
            "EULLM_API_KEY": "k",
            "EULLM_EMBED_MODEL": EMBED,
            "EULLM_EMBED_QUERY_PREFIX": "Instruct: Find tools\\nQuery:",
            "REFLEX_GATE_THRESHOLD": "0.38",
            "EULLM_TIMEOUT": "60",
        }
        config = Config.from_env(env)
        self.assertEqual(config.url, "http://127.0.0.1:11500")
        self.assertEqual((config.api_key, config.embed_model), ("k", EMBED))
        self.assertEqual(config.embed_query_prefix, "Instruct: Find tools\nQuery:")
        self.assertEqual((config.gate_threshold, config.timeout), (0.38, 60.0))
        # An empty variable is an unset one.
        blank = Config.from_env(dict.fromkeys(env, ""))
        self.assertEqual(blank, Config())

    def test_settings_that_cannot_be_used(self):
        for env in (
            {"REFLEX_GATE_THRESHOLD": "1.5"},
            {"REFLEX_GATE_THRESHOLD": "nan"},
            {"REFLEX_GATE_THRESHOLD": "high"},
            {"EULLM_TIMEOUT": "0"},
            {"EULLM_URL": "http://192.168.1.10:11434"},
        ):
            with self.assertRaises(ConfigError, msg=env):
                Config.from_env(env)
        self.assertTrue(Config.from_env({"EULLM_URL": "http://192.168.1.10"}, allow_remote=True))


class WordingTest(unittest.TestCase):
    """The questions are ReflexBench's, character for character: a threshold
    fitted with ragbench.py holds only for the question it was fitted on."""

    @unittest.skipUnless(BENCH.is_dir(), "needs bench/reflexbench from the repository")
    def test_the_questions_are_reflexbenchs(self):
        sys.path.insert(0, str(BENCH))
        try:
            import rb_methods
            import rg_data
            import rg_methods
        finally:
            sys.path.remove(str(BENCH))
        self.assertEqual(reflex.QUESTION, rb_methods.QUESTION_A)
        self.assertEqual((reflex.NONE, reflex.NONE_TEXT), (rb_methods.NONE, rb_methods.NONE_TEXT))
        self.assertEqual(reflex.GATE, rg_methods.GATE)
        self.assertEqual(list(reflex.OPTIONS.items()), list(rg_methods.OPTIONS.items()))
        self.assertEqual(reflex.LABELS, rg_data.LABELS)
        case = rg_data.Case("1", "1", "Who wrote it?", ["Book: by Ann", "Ann: a writer"], "answer")
        self.assertEqual(reflex.gate_state(case.question, case.passages), rg_methods.state(case))

    def test_split_evenly(self):
        def sizes(n, size):
            return [len(part) for part in reflex.split(list(range(n)), size)]

        self.assertEqual(sizes(37, 18), [13, 12, 12])
        self.assertEqual(sizes(7, 3), [3, 2, 2])
        self.assertEqual(sizes(4, 37), [4])
        for n in range(2, 60):
            self.assertGreaterEqual(min(sizes(n, 3)), 2, n)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def environment(**variables):
    """The command's environment: this checkout's package first, and no
    setting of the shell the tests run in."""
    path = os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH")) if p)
    return {"PYTHONPATH": path, **variables}


def inherited():
    return {k: v for k, v in os.environ.items() if not k.startswith(("EULLM_", "REFLEX_"))}


class CommandTest(unittest.IsolatedAsyncioTestCase):
    def test_a_remote_eullm_is_refused_at_start(self):
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"EULLM_URL": "http://example.com:11434"}, clear=True),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(main([]), 2)
        self.assertIn("--allow-remote", stderr.getvalue())

    async def test_stdio(self):
        with standin.StandIn(scores=SCORES) as eullm:
            command = StdioServerParameters(
                command=sys.executable,
                args=["-m", "eullm_reflex_mcp"],
                env=environment(EULLM_URL=eullm.url),
            )
            async with Client(command) as client:
                listed = await client.list_tools()
                result = await client.call_tool("select_tools", {"request": RAIN, "tools": CATALOG})
        self.assertEqual(len(listed.tools), 4)
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(result.structured_content["tools"][0]["name"], "get_weather")

    async def test_streamable_http(self):
        port = free_port()
        with standin.StandIn() as eullm:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "eullm_reflex_mcp",
                    "--transport",
                    "streamable-http",
                    "--port",
                    str(port),
                ],
                env={**inherited(), **environment(EULLM_URL=eullm.url)},
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    with socket.socket() as s:
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            break
                    self.assertIsNone(server.poll(), "the server exited")
                    self.assertLess(time.monotonic(), deadline, "the server did not start")
                    await asyncio.sleep(0.1)
                async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                    result = await client.call_tool("model_info", {})
            finally:
                server.terminate()
                _, stderr = await asyncio.to_thread(server.communicate, timeout=30)
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(result.structured_content["decision_model"]["name"], standin.MODEL)
        self.assertIn(f"EuLLM at {eullm.url}", stderr)


if __name__ == "__main__":
    unittest.main()
