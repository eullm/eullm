#!/usr/bin/env python3
"""
TurboQuant Quality Benchmark — Measure precision impact of KV cache quantization.

Sends identical prompts to the EULLM engine with different KV cache types
and compares output quality. Tests: matrix operations, math, factual Q&A,
logic reasoning, code generation.

Usage:
  # Run against a live engine (start it with the desired --cache-type-k/v)
  python3 bench/turboquant_quality.py --url http://localhost:11434 --label F16

  # Compare multiple result files
  python3 bench/turboquant_quality.py compare results_f16.json results_tq4.json results_tq3.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

# ── Test prompts with verifiable answers ─────────────────────────────────────

TESTS = [
    # ══════════════════════════════════════════════════════════════════════
    # MATRIX (20 tests)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "mat01", "category": "matrix", "check": "exact_normalized",
     "prompt": "Compute [[1,2],[3,4]] × [[5,6],[7,8]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[19,22],[43,50]]"},
    {"id": "mat02", "category": "matrix", "check": "contains_number",
     "prompt": "Determinant of [[3,8],[4,6]]? Return ONLY the number.",
     "expected": "-14"},
    {"id": "mat03", "category": "matrix", "check": "exact_normalized",
     "prompt": "Transpose [[1,2,3],[4,5,6],[7,8,9]]. Return ONLY the matrix.",
     "expected": "[[1,4,7],[2,5,8],[3,6,9]]"},
    {"id": "mat04", "category": "matrix", "check": "contains_number",
     "prompt": "Trace of [[5,1,2],[0,3,1],[2,0,7]]? Return ONLY the number.",
     "expected": "15"},
    {"id": "mat05", "category": "matrix", "check": "exact_normalized",
     "prompt": "Compute [[2,0],[0,3]] × [[1,4],[5,2]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[2,8],[15,6]]"},
    {"id": "mat06", "category": "matrix", "check": "contains_number",
     "prompt": "Determinant of [[1,2,3],[4,5,6],[7,8,9]]? Return ONLY the number.",
     "expected": "0"},
    {"id": "mat07", "category": "matrix", "check": "exact_normalized",
     "prompt": "[[1,0],[0,1]] × [[7,3],[2,9]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[7,3],[2,9]]"},
    {"id": "mat08", "category": "matrix", "check": "contains_number",
     "prompt": "Trace of [[10,0,0],[0,20,0],[0,0,30]]? Return ONLY the number.",
     "expected": "60"},
    {"id": "mat09", "category": "matrix", "check": "exact_normalized",
     "prompt": "Transpose [[1,2],[3,4],[5,6]]. Return ONLY the result.",
     "expected": "[[1,3,5],[2,4,6]]"},
    {"id": "mat10", "category": "matrix", "check": "contains_number",
     "prompt": "Determinant of [[2,0],[0,5]]? Return ONLY the number.",
     "expected": "10"},
    {"id": "mat11", "category": "matrix", "check": "exact_normalized",
     "prompt": "Compute [[3,1],[2,4]] + [[1,5],[3,2]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[4,6],[5,6]]"},
    {"id": "mat12", "category": "matrix", "check": "contains_number",
     "prompt": "What is the rank of [[1,2],[2,4]]? Return ONLY the number.",
     "expected": "1"},
    {"id": "mat13", "category": "matrix", "check": "exact_normalized",
     "prompt": "Multiply scalar 3 by matrix [[1,2],[3,4]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[3,6],[9,12]]"},
    {"id": "mat14", "category": "matrix", "check": "contains_number",
     "prompt": "Determinant of [[1,3],[2,7]]? Return ONLY the number.",
     "expected": "1"},
    {"id": "mat15", "category": "matrix", "check": "exact_normalized",
     "prompt": "Compute [[1,1],[1,1]] × [[1,1],[1,1]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[2,2],[2,2]]"},
    {"id": "mat16", "category": "matrix", "check": "contains_number",
     "prompt": "Trace of the 4x4 identity matrix? Return ONLY the number.",
     "expected": "4"},
    {"id": "mat17", "category": "matrix", "check": "exact_normalized",
     "prompt": "[[2,3],[1,4]] - [[1,1],[1,1]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[1,2],[0,3]]"},
    {"id": "mat18", "category": "matrix", "check": "contains_number",
     "prompt": "Determinant of [[5,3],[2,4]]? Return ONLY the number.",
     "expected": "14"},
    {"id": "mat19", "category": "matrix", "check": "contains_number",
     "prompt": "How many rows in a 3x5 matrix? Return ONLY the number.",
     "expected": "3"},
    {"id": "mat20", "category": "matrix", "check": "exact_normalized",
     "prompt": "Compute [[0,1],[1,0]] × [[0,1],[1,0]]. Return ONLY [[a,b],[c,d]].",
     "expected": "[[1,0],[0,1]]"},
    # ══════════════════════════════════════════════════════════════════════
    # MATH (20 tests)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "math01", "category": "math", "check": "contains_number",
     "prompt": "What is 347 × 283? Return ONLY the number.", "expected": "98201"},
    {"id": "math02", "category": "math", "check": "contains_word",
     "prompt": "Is 997 a prime number? Answer ONLY 'yes' or 'no'.", "expected": "yes"},
    {"id": "math03", "category": "math", "check": "contains_number",
     "prompt": "Square root of 1764? Return ONLY the integer.", "expected": "42"},
    {"id": "math04", "category": "math", "check": "contains_number",
     "prompt": "Next in sequence: 2, 6, 18, 54, ...? Return ONLY the number.", "expected": "162"},
    {"id": "math05", "category": "math", "check": "contains_word",
     "prompt": "Simplify 84/126. Return ONLY the fraction.", "expected": "2/3"},
    {"id": "math06", "category": "math", "check": "contains_number",
     "prompt": "What is 17 × 19? Return ONLY the number.", "expected": "323"},
    {"id": "math07", "category": "math", "check": "contains_number",
     "prompt": "What is 144 ÷ 12? Return ONLY the number.", "expected": "12"},
    {"id": "math08", "category": "math", "check": "contains_word",
     "prompt": "Is 91 a prime number? Answer ONLY 'yes' or 'no'.", "expected": "no"},
    {"id": "math09", "category": "math", "check": "contains_number",
     "prompt": "What is 2^10? Return ONLY the number.", "expected": "1024"},
    {"id": "math10", "category": "math", "check": "contains_number",
     "prompt": "What is 15! / 14!? Return ONLY the number.", "expected": "15"},
    {"id": "math11", "category": "math", "check": "contains_number",
     "prompt": "GCD of 48 and 36? Return ONLY the number.", "expected": "12"},
    {"id": "math12", "category": "math", "check": "contains_number",
     "prompt": "LCM of 4 and 6? Return ONLY the number.", "expected": "12"},
    {"id": "math13", "category": "math", "check": "contains_number",
     "prompt": "What is 25% of 360? Return ONLY the number.", "expected": "90"},
    {"id": "math14", "category": "math", "check": "contains_number",
     "prompt": "Sum of integers from 1 to 10? Return ONLY the number.", "expected": "55"},
    {"id": "math15", "category": "math", "check": "contains_number",
     "prompt": "What is log2(256)? Return ONLY the number.", "expected": "8"},
    {"id": "math16", "category": "math", "check": "contains_number",
     "prompt": "What is 7^3? Return ONLY the number.", "expected": "343"},
    {"id": "math17", "category": "math", "check": "contains_number",
     "prompt": "How many prime numbers between 1 and 20? Return ONLY the count.", "expected": "8"},
    {"id": "math18", "category": "math", "check": "contains_number",
     "prompt": "What is |-7| + |3|? Return ONLY the number.", "expected": "10"},
    {"id": "math19", "category": "math", "check": "contains_number",
     "prompt": "Convert 0.75 to a percentage. Return ONLY the number (no % sign).", "expected": "75"},
    {"id": "math20", "category": "math", "check": "contains_number",
     "prompt": "What is the 10th Fibonacci number (starting F1=1, F2=1)? Return ONLY the number.", "expected": "55"},
    # ══════════════════════════════════════════════════════════════════════
    # FACTUAL (20 tests)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "fact01", "category": "factual", "check": "contains_word",
     "prompt": "Capital of Slovenia? Return ONLY the city name.", "expected": "Ljubljana"},
    {"id": "fact02", "category": "factual", "check": "contains_word",
     "prompt": "Chemical symbol for tungsten? Return ONLY the symbol.", "expected": "W"},
    {"id": "fact03", "category": "factual", "check": "contains_number",
     "prompt": "Year the Euro was introduced for electronic transactions? Return ONLY the year.", "expected": "1999"},
    {"id": "fact04", "category": "factual", "check": "contains_word",
     "prompt": "Planet with most moons in our solar system? Return ONLY the name.", "expected": "Saturn"},
    {"id": "fact05", "category": "factual", "check": "contains_word",
     "prompt": "Max GDPR fine as % of global turnover? Return ONLY the percentage.", "expected": "4%"},
    {"id": "fact06", "category": "factual", "check": "contains_word",
     "prompt": "Capital of Portugal? Return ONLY the city name.", "expected": "Lisbon"},
    {"id": "fact07", "category": "factual", "check": "contains_word",
     "prompt": "Chemical symbol for gold? Return ONLY the symbol.", "expected": "Au"},
    {"id": "fact08", "category": "factual", "check": "contains_number",
     "prompt": "How many countries in the European Union (as of 2024)? Return ONLY the number.", "expected": "27"},
    {"id": "fact09", "category": "factual", "check": "contains_word",
     "prompt": "What is the largest ocean on Earth? Return ONLY the name.", "expected": "Pacific"},
    {"id": "fact10", "category": "factual", "check": "contains_word",
     "prompt": "Who wrote 'The Divine Comedy'? Return ONLY the author's last name.", "expected": "Alighieri"},
    {"id": "fact11", "category": "factual", "check": "contains_word",
     "prompt": "What is the currency of Japan? Return ONLY the name.", "expected": "Yen"},
    {"id": "fact12", "category": "factual", "check": "contains_word",
     "prompt": "What is the chemical formula for water? Return ONLY the formula.", "expected": "H2O"},
    {"id": "fact13", "category": "factual", "check": "contains_number",
     "prompt": "How many bones in an adult human body? Return ONLY the number.", "expected": "206"},
    {"id": "fact14", "category": "factual", "check": "contains_word",
     "prompt": "What is the smallest country in the world by area? Return ONLY the name.", "expected": "Vatican"},
    {"id": "fact15", "category": "factual", "check": "contains_number",
     "prompt": "Speed of light in km/s (rounded to nearest thousand)? Return ONLY the number.", "expected": "300000"},
    {"id": "fact16", "category": "factual", "check": "contains_word",
     "prompt": "Capital of Australia? Return ONLY the city name.", "expected": "Canberra"},
    {"id": "fact17", "category": "factual", "check": "contains_word",
     "prompt": "What element has atomic number 1? Return ONLY the element name.", "expected": "Hydrogen"},
    {"id": "fact18", "category": "factual", "check": "contains_number",
     "prompt": "How many sides does a hexagon have? Return ONLY the number.", "expected": "6"},
    {"id": "fact19", "category": "factual", "check": "contains_word",
     "prompt": "What is the longest river in Europe? Return ONLY the name.", "expected": "Volga"},
    {"id": "fact20", "category": "factual", "check": "contains_word",
     "prompt": "What programming language was created by Guido van Rossum? Return ONLY the name.", "expected": "Python"},
    # ══════════════════════════════════════════════════════════════════════
    # LOGIC & REASONING (20 tests)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "logic01", "category": "logic", "check": "contains_word",
     "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude some roses fade quickly? Answer ONLY 'yes' or 'no'.", "expected": "no"},
    {"id": "logic02", "category": "logic", "check": "contains_number",
     "prompt": "If A=1, B=2, C=3... sum of letters in 'EULLM'? Return ONLY the number.", "expected": "63"},
    {"id": "logic03", "category": "logic", "check": "contains_number",
     "prompt": "Next in: 1, 1, 2, 3, 5, 8, 13, __? Return ONLY the number.", "expected": "21"},
    {"id": "logic04", "category": "logic", "check": "contains_number",
     "prompt": "If a shirt costs $20 after a 20% discount, what was the original price? Return ONLY the number.", "expected": "25"},
    {"id": "logic05", "category": "logic", "check": "contains_word",
     "prompt": "All dogs are animals. All animals are living things. Are all dogs living things? Answer ONLY 'yes' or 'no'.", "expected": "yes"},
    {"id": "logic06", "category": "logic", "check": "contains_number",
     "prompt": "A train travels 60 km/h for 2.5 hours. Distance in km? Return ONLY the number.", "expected": "150"},
    {"id": "logic07", "category": "logic", "check": "contains_number",
     "prompt": "Next in: 3, 6, 12, 24, __? Return ONLY the number.", "expected": "48"},
    {"id": "logic08", "category": "logic", "check": "contains_word",
     "prompt": "Monday is after Sunday. Sunday is after Saturday. Is Monday after Saturday? Answer ONLY 'yes' or 'no'.", "expected": "yes"},
    {"id": "logic09", "category": "logic", "check": "contains_number",
     "prompt": "If 3 workers build a wall in 12 hours, how many hours for 6 workers? Return ONLY the number.", "expected": "6"},
    {"id": "logic10", "category": "logic", "check": "contains_number",
     "prompt": "How many letters in the word 'TURBOQUANT'? Return ONLY the number.", "expected": "10"},
    {"id": "logic11", "category": "logic", "check": "contains_word",
     "prompt": "Some cats are black. Some black things are shoes. Can we conclude some cats are shoes? Answer ONLY 'yes' or 'no'.", "expected": "no"},
    {"id": "logic12", "category": "logic", "check": "contains_number",
     "prompt": "Next in: 1, 4, 9, 16, 25, __? Return ONLY the number.", "expected": "36"},
    {"id": "logic13", "category": "logic", "check": "contains_number",
     "prompt": "You have 3 red balls, 2 blue balls, 4 green balls. Total? Return ONLY the number.", "expected": "9"},
    {"id": "logic14", "category": "logic", "check": "contains_number",
     "prompt": "If today is Wednesday, what day is it in 100 days? Count Wed as day 0. Return ONLY the day name. Actually, return ONLY the number of the weekday (Mon=1..Sun=7).", "expected": "5"},
    {"id": "logic15", "category": "logic", "check": "contains_number",
     "prompt": "A rectangle is 8cm wide and 5cm tall. What is its area in cm²? Return ONLY the number.", "expected": "40"},
    {"id": "logic16", "category": "logic", "check": "contains_number",
     "prompt": "Next in: 2, 3, 5, 7, 11, 13, __? Return ONLY the number.", "expected": "17"},
    {"id": "logic17", "category": "logic", "check": "contains_word",
     "prompt": "If no fish can fly, and a salmon is a fish, can a salmon fly? Answer ONLY 'yes' or 'no'.", "expected": "no"},
    {"id": "logic18", "category": "logic", "check": "contains_number",
     "prompt": "How many vowels in 'ARTIFICIAL INTELLIGENCE'? Return ONLY the number.", "expected": "10"},
    {"id": "logic19", "category": "logic", "check": "contains_number",
     "prompt": "If you fold a paper in half 7 times, how many layers? Return ONLY the number.", "expected": "128"},
    {"id": "logic20", "category": "logic", "check": "contains_number",
     "prompt": "A clock shows 3:15. What is the angle between hour and minute hands in degrees? Return ONLY the number.", "expected": "7.5"},
    # ══════════════════════════════════════════════════════════════════════
    # CODE & TECHNICAL (20 tests)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "code01", "category": "code", "check": "contains_word",
     "prompt": "What does FizzBuzz output for 15? Return ONLY the word.", "expected": "FizzBuzz"},
    {"id": "code02", "category": "code", "check": "contains_word",
     "prompt": "In Python, what built-in function returns the length of a list? Return ONLY the function name.", "expected": "len"},
    {"id": "code03", "category": "code", "check": "contains_word",
     "prompt": "What HTTP status code means 'Not Found'? Return ONLY the number and name.", "expected": "404"},
    {"id": "code04", "category": "code", "check": "contains_word",
     "prompt": "What does SQL stand for? Return ONLY the full name.", "expected": "Structured Query Language"},
    {"id": "code05", "category": "code", "check": "contains_word",
     "prompt": "In git, what command shows the commit history? Return ONLY the command.", "expected": "git log"},
    {"id": "code06", "category": "code", "check": "contains_number",
     "prompt": "What is the default port for HTTPS? Return ONLY the number.", "expected": "443"},
    {"id": "code07", "category": "code", "check": "contains_word",
     "prompt": "What does JSON stand for? Return ONLY the full name.", "expected": "JavaScript Object Notation"},
    {"id": "code08", "category": "code", "check": "contains_word",
     "prompt": "In Python, what keyword defines a function? Return ONLY the keyword.", "expected": "def"},
    {"id": "code09", "category": "code", "check": "contains_number",
     "prompt": "What is the default port for HTTP? Return ONLY the number.", "expected": "80"},
    {"id": "code10", "category": "code", "check": "contains_word",
     "prompt": "What does API stand for? Return ONLY the full name.", "expected": "Application Programming Interface"},
    {"id": "code11", "category": "code", "check": "contains_word",
     "prompt": "What does CSS stand for? Return ONLY the full name.", "expected": "Cascading Style Sheets"},
    {"id": "code12", "category": "code", "check": "contains_word",
     "prompt": "In Rust, what keyword is used to declare a mutable variable? Return ONLY the keyword.", "expected": "mut"},
    {"id": "code13", "category": "code", "check": "contains_number",
     "prompt": "Default SSH port? Return ONLY the number.", "expected": "22"},
    {"id": "code14", "category": "code", "check": "contains_word",
     "prompt": "What does CRUD stand for in databases? Return ONLY the full name.", "expected": "Create Read Update Delete"},
    {"id": "code15", "category": "code", "check": "contains_word",
     "prompt": "In HTML, what tag creates a hyperlink? Return ONLY the tag name (no brackets).", "expected": "a"},
    {"id": "code16", "category": "code", "check": "contains_word",
     "prompt": "What does REST stand for? Return ONLY the full name.", "expected": "Representational State Transfer"},
    {"id": "code17", "category": "code", "check": "contains_word",
     "prompt": "In Python, what exception is raised when dividing by zero? Return ONLY the exception name.", "expected": "ZeroDivisionError"},
    {"id": "code18", "category": "code", "check": "contains_number",
     "prompt": "Default port for PostgreSQL? Return ONLY the number.", "expected": "5432"},
    {"id": "code19", "category": "code", "check": "contains_word",
     "prompt": "What does YAML stand for? Return ONLY the full name.", "expected": "YAML Ain't Markup Language"},
    {"id": "code20", "category": "code", "check": "contains_word",
     "prompt": "In git, what command creates a new branch? Return ONLY the command.", "expected": "git branch"},
]


# ── Answer checking ──────────────────────────────────────────────────────────

def strip_thinking(s: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 thinking mode)."""
    import re
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()


def normalize(s: str) -> str:
    """Remove whitespace, newlines, backticks, markdown, and lowercase."""
    s = strip_thinking(s)
    s = s.replace(" ", "").replace("\n", "").replace("`", "").replace("*", "")
    s = s.replace("\\times", "×").replace("\\cdot", "·")
    return s.strip().lower()


def extract_last_line(s: str) -> str:
    """Get the last non-empty line — often the actual answer after explanation."""
    s = strip_thinking(s)
    lines = [l.strip() for l in s.strip().split('\n') if l.strip()]
    return lines[-1] if lines else s


def check_answer(test: dict, response: str) -> tuple[bool, str]:
    """Check if the response matches the expected answer. Returns (pass, detail)."""
    mode = test["check"]
    resp = strip_thinking(response).strip()
    resp_last = extract_last_line(response)

    if mode == "exact_normalized":
        expected = normalize(test["expected"])
        # Check full response AND last line (model may explain then give answer)
        actual_full = normalize(resp)
        actual_last = normalize(resp_last)
        ok = expected in actual_full or expected in actual_last
        return ok, f"expected={test['expected']} last_line={resp_last[:80]}"

    elif mode == "contains_number":
        expected = test["expected"]
        ok = expected in resp or expected in resp_last
        return ok, f"expected={expected} in response={resp_last[:80]}"

    elif mode == "contains_word":
        expected = test["expected"].lower()
        ok = expected in resp.lower() or expected in resp_last.lower()
        return ok, f"expected='{expected}' in response={resp_last[:80]}"

    elif mode == "contains_any":
        candidates = test["expected_contains"]
        combined = resp.lower() + " " + resp_last.lower()
        ok = any(c.lower() in combined for c in candidates)
        matched = [c for c in candidates if c.lower() in combined]
        return ok, f"matched={matched} in response={resp_last[:80]}"

    return False, "unknown check mode"


# ── API client ───────────────────────────────────────────────────────────────

async def send_prompt(session: aiohttp.ClientSession, url: str, model: str,
                      prompt: str, temperature: float = 0.0) -> str:
    """Send a prompt and return the full response text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 256},
    }
    try:
        async with session.post(f"{url}/api/chat", json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                return f"[ERROR: HTTP {resp.status}]"
            data = await resp.json()
            return data.get("message", {}).get("content", "[no content]")
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Collect mode ─────────────────────────────────────────────────────────────

async def collect(args):
    results = []
    passed = 0
    total = len(TESTS)

    print(f"Running {total} quality tests against {args.url} (label={args.label}, model={args.model})")
    print(f"Temperature: {args.temperature}")
    print()

    async with aiohttp.ClientSession() as session:
        for i, test in enumerate(TESTS):
            response = await send_prompt(session, args.url, args.model, test["prompt"], args.temperature)
            ok, detail = check_answer(test, response)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1

            print(f"  [{status}] {test['id']:<25} {detail}")

            results.append({
                "id": test["id"],
                "category": test["category"],
                "prompt": test["prompt"],
                "response": response,
                "passed": ok,
                "detail": detail,
            })

    score = passed / total * 100
    print(f"\n{'='*60}")
    print(f"  {args.label}: {passed}/{total} passed ({score:.1f}%)")
    print(f"{'='*60}")

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    print(f"\n  {'Category':<12} {'Score':>8}")
    print(f"  {'-'*22}")
    for cat, vals in sorted(categories.items()):
        pct = vals['passed'] / vals['total'] * 100
        print(f"  {cat:<12} {vals['passed']}/{vals['total']} ({pct:.0f}%)")

    # Save JSON
    output = {
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score": {"passed": passed, "total": total, "percent": score},
        "categories": categories,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {args.output}")


# ── Compare mode ─────────────────────────────────────────────────────────────

def compare(args):
    files = args.files
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))

    print(f"\n{'='*70}")
    print(f"  TurboQuant Quality Comparison")
    print(f"{'='*70}\n")

    # Overall scores
    print(f"  {'Cache Type':<20} {'Score':>10} {'Passed':>10} {'Total':>8}")
    print(f"  {'-'*50}")
    baseline_score = None
    for d in data:
        s = d["score"]
        label = d["label"]
        delta = ""
        if baseline_score is None:
            baseline_score = s["percent"]
        else:
            diff = s["percent"] - baseline_score
            delta = f"  ({diff:+.1f}%)"
        print(f"  {label:<20} {s['percent']:>9.1f}% {s['passed']:>8}/{s['total']}{delta}")

    # Category comparison
    all_cats = sorted(set(c for d in data for c in d.get("categories", {})))
    print(f"\n  {'Category':<12}", end="")
    for d in data:
        print(f"  {d['label']:>12}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in data:
        print(f"  {'-'*12}", end="")
    print()

    for cat in all_cats:
        print(f"  {cat:<12}", end="")
        for d in data:
            cats = d.get("categories", {})
            if cat in cats:
                v = cats[cat]
                print(f"  {v['passed']}/{v['total']:>2} ({v['passed']/v['total']*100:3.0f}%)", end="")
            else:
                print(f"  {'n/a':>12}", end="")
        print()

    # Per-test diff (show only failures that differ)
    print(f"\n  Divergent results (different pass/fail across cache types):")
    print(f"  {'-'*60}")
    any_diff = False
    test_ids = [r["id"] for r in data[0]["results"]]
    for tid in test_ids:
        results_for_test = []
        for d in data:
            r = next((x for x in d["results"] if x["id"] == tid), None)
            if r:
                results_for_test.append((d["label"], r["passed"], r["response"][:60]))
        statuses = [r[1] for r in results_for_test]
        if len(set(statuses)) > 1:
            any_diff = True
            print(f"\n  {tid}:")
            for label, passed, resp in results_for_test:
                status = "PASS" if passed else "FAIL"
                print(f"    [{status}] {label:<15} → {resp}")

    if not any_diff:
        print(f"  None — all cache types produced the same pass/fail results!")

    if args.markdown:
        print(f"\n\n### Markdown table\n")
        print(f"| Cache Type | Score | Matrix | Math | Factual | Logic | Code |")
        print(f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for d in data:
            s = d["score"]
            cats = d.get("categories", {})
            row = f"| {d['label']} | {s['percent']:.0f}% |"
            for cat in ["matrix", "math", "factual", "logic", "code"]:
                if cat in cats:
                    v = cats[cat]
                    row += f" {v['passed']}/{v['total']} |"
                else:
                    row += " n/a |"
            print(row)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Quality Benchmark")
    sub = parser.add_subparsers(dest="command")

    # collect
    c = sub.add_parser("collect", help="Run quality tests against a live engine")
    c.add_argument("--url", default="http://localhost:11434")
    c.add_argument("--model", default="qwen3-14b")
    c.add_argument("--label", required=True, help="Cache type label (e.g. F16, TQ4_0, TQ3_0)")
    c.add_argument("--temperature", type=float, default=0.0, help="Use 0.0 for deterministic output")
    c.add_argument("--output", "-o", help="Save results to JSON file")

    # compare
    p = sub.add_parser("compare", help="Compare multiple result files")
    p.add_argument("files", nargs="+", help="JSON result files to compare")
    p.add_argument("--markdown", action="store_true", help="Output markdown table")

    args = parser.parse_args()

    if args.command == "collect":
        asyncio.run(collect(args))
    elif args.command == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
