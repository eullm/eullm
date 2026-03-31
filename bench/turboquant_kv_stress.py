#!/usr/bin/env python3
"""
TurboQuant KV Cache Stress Test — Tests precision of KV cache recall.

Unlike the quality benchmark (which tests model knowledge), this tests
whether the model can accurately recall precise information stored earlier
in the context after filler text pushes the KV cache.

5 test types:
1. Numeric block exact recall (15 lines of digits)
2. Grid cell lookup (20x20 digit grid, specific cell)
3. Delayed matrix multiplication (store matrices, compute after filler)
4. Key-value binding recall (10 name→number pairs)
5. Structured record search (30 records with overlapping attributes)

Each test is run 3 times with different filler lengths to stress the KV cache
at different context distances.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

# ── Filler text generator ────────────────────────────────────────────────────

FILLER_PARAGRAPHS = [
    "The European Union's digital strategy aims to make this transformation work for people and businesses, while helping to achieve its target of a climate-neutral Europe by 2050. The Commission has outlined key policies and frameworks to guide this transition, including the Digital Markets Act and the Digital Services Act, which set new rules for tech platforms operating in Europe.",
    "Recent advances in semiconductor manufacturing have enabled the production of chips at the 3nm node, dramatically increasing transistor density while reducing power consumption. TSMC, Samsung, and Intel are competing to deliver these next-generation processors for both consumer devices and data center applications. The economic implications are significant, with global chip demand expected to double by 2030.",
    "The Mediterranean basin is home to approximately 25,000 plant species, of which about 50% are endemic. This biodiversity hotspot faces threats from urbanization, agricultural intensification, and climate change. Conservation efforts include the establishment of marine protected areas and the restoration of degraded habitats along the coastline from Spain to Turkey.",
    "Cloud computing has fundamentally changed how organizations deploy and manage IT infrastructure. The shift from capital expenditure to operational expenditure models has enabled startups and enterprises alike to scale their operations globally without significant upfront investment. Major providers including AWS, Azure, and Google Cloud now operate data centers on every inhabited continent.",
    "Italian Renaissance art underwent several distinct phases, from the early experiments of Giotto in the 13th century to the High Renaissance works of Leonardo, Michelangelo, and Raphael. The development of linear perspective by Brunelleschi and its systematic application by artists transformed the representation of three-dimensional space on two-dimensional surfaces.",
    "The carbon cycle involves the exchange of carbon between the atmosphere, oceans, terrestrial biosphere, and geological formations. Anthropogenic emissions have disrupted this cycle, leading to an increase in atmospheric CO2 concentrations from approximately 280 ppm in pre-industrial times to over 420 ppm today. Understanding these dynamics is critical for climate modeling.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations that would be impractical for classical computers. While current quantum computers are limited by decoherence and error rates, researchers have demonstrated quantum advantage for specific problems. Applications range from drug discovery to cryptographic analysis.",
    "The logistics industry handles approximately 65 billion parcels annually worldwide, with e-commerce driving a significant portion of this volume. Last-mile delivery remains the most expensive segment, accounting for up to 53% of total shipping costs. Innovations in autonomous delivery vehicles and drone technology promise to reduce these costs in urban areas.",
    "Volcanic activity along tectonic plate boundaries shapes Earth's surface through both constructive and destructive processes. The Ring of Fire, encircling the Pacific Ocean, contains approximately 75% of the world's active volcanoes and accounts for about 90% of the world's earthquakes. Monitoring these geologically active zones is essential for early warning systems.",
    "Machine learning models for natural language processing have grown exponentially in size, from millions of parameters in early models to hundreds of billions in current large language models. This scaling has been accompanied by emergent capabilities not predicted by smaller models, including in-context learning, chain-of-thought reasoning, and cross-lingual transfer.",
    "The agricultural revolution in Europe during the 18th century introduced crop rotation systems, selective breeding of livestock, and mechanical innovations such as the seed drill. These advances dramatically increased food production, supporting population growth and urbanization that would fuel the subsequent Industrial Revolution in Britain and across the continent.",
    "Photovoltaic technology has seen dramatic cost reductions over the past decade, with the levelized cost of solar electricity dropping below that of fossil fuels in many regions. Advances in perovskite cells, bifacial modules, and tracking systems continue to improve efficiency. Global solar capacity exceeded 1 terawatt for the first time in 2022.",
]


def generate_filler(target_tokens: int) -> str:
    """Generate filler text of approximately target_tokens length."""
    # Rough estimate: 1 token ≈ 4 chars
    target_chars = target_tokens * 4
    paragraphs = []
    total = 0
    while total < target_chars:
        p = random.choice(FILLER_PARAGRAPHS)
        paragraphs.append(p)
        total += len(p)
    return "\n\n".join(paragraphs)


# ── Test definitions ─────────────────────────────────────────────────────────

def make_test_numeric_block(filler_tokens: int) -> dict:
    """Test 1: Memorize and recall a numeric block exactly."""
    block = [
        "4839201756", "1029384756", "5647382910", "9182736450", "3748291056",
        "6655443322", "1092837465", "9988776655", "4829103746", "1357924680",
        "2468013579", "9182736455", "5647382901", "1029384755", "4839201755",
    ]
    filler = generate_filler(filler_tokens)
    block_text = "\n".join(block)

    prompt = f"""You must memorize the following numeric block exactly.

BEGIN BLOCK
{block_text}
END BLOCK

Do not reproduce it yet.

{filler}

Now output the original numeric block exactly, line by line, with no explanation. Output ONLY the 15 lines of digits."""

    return {
        "id": f"kv_numeric_block_{filler_tokens}t",
        "category": "kv_recall",
        "prompt": prompt,
        "expected_lines": block,
        "filler_tokens": filler_tokens,
        "check": "line_match",
    }


def make_test_grid_lookup(filler_tokens: int) -> dict:
    """Test 2: Memorize a digit grid, recall a specific cell."""
    random.seed(42)  # deterministic grid
    grid = []
    for r in range(20):
        row = [str(random.randint(0, 9)) for _ in range(20)]
        grid.append(row)

    target_row = 13
    target_col = 17
    expected = grid[target_row - 1][target_col - 1]

    grid_text = ""
    for i, row in enumerate(grid):
        grid_text += f"Row {i+1:02d}: {' '.join(row)}\n"

    filler = generate_filler(filler_tokens)

    prompt = f"""Memorize this 20x20 digit grid.

{grid_text}
Do not answer yet.

{filler}

Answer only with the digit at row {target_row}, column {target_col}. Return ONLY the single digit."""

    return {
        "id": f"kv_grid_lookup_{filler_tokens}t",
        "category": "kv_recall",
        "prompt": prompt,
        "expected": expected,
        "filler_tokens": filler_tokens,
        "check": "contains_number",
    }


def make_test_delayed_matrix(filler_tokens: int) -> dict:
    """Test 3: Store matrices, compute after filler."""
    # A × B = [[3*5+1*6, 3*7+1*8], [4*5+2*6, 4*7+2*8]] = [[21,29],[32,44]]
    filler = generate_filler(filler_tokens)

    prompt = f"""Read these matrices carefully.

A = [[3,1],[4,2]]
B = [[5,7],[6,8]]
C = [[1,0],[0,1]]
D = [[2,3],[1,4]]

Do not compute yet.

{filler}

Now answer: What is A × B?
Output ONLY the resulting 2x2 matrix in format [[a,b],[c,d]]. No explanation."""

    return {
        "id": f"kv_delayed_matrix_{filler_tokens}t",
        "category": "kv_recall",
        "prompt": prompt,
        "expected": "[[21,29],[32,44]]",
        "filler_tokens": filler_tokens,
        "check": "exact_normalized",
    }


def make_test_kv_binding(filler_tokens: int) -> dict:
    """Test 4: Key-value binding recall."""
    bindings = {
        "ALPHA": "48291", "BETA": "10573", "GAMMA": "77420",
        "DELTA": "91826", "EPSILON": "33014", "ZETA": "66589",
        "ETA": "10422", "THETA": "90817", "IOTA": "55120", "KAPPA": "28046",
    }
    target_key = "THETA"
    expected = bindings[target_key]

    binding_text = "\n".join(f"{k} -> {v}" for k, v in bindings.items())
    filler = generate_filler(filler_tokens)

    prompt = f"""Memorize these bindings exactly:

{binding_text}

Do not answer yet.

{filler}

Question: What is the value associated with {target_key}?
Answer with digits only. Return ONLY the 5-digit number."""

    return {
        "id": f"kv_binding_{filler_tokens}t",
        "category": "kv_recall",
        "prompt": prompt,
        "expected": expected,
        "filler_tokens": filler_tokens,
        "check": "contains_number",
    }


def make_test_record_search(filler_tokens: int) -> dict:
    """Test 5: Structured record search with overlapping attributes."""
    records = [
        ("A12", "Turin", 483, "K7"), ("B14", "Milan", 271, "M3"),
        ("C18", "Genoa", 483, "T9"), ("D21", "Turin", 182, "K7"),
        ("E05", "Rome", 395, "K7"), ("F33", "Milan", 182, "M3"),
        ("G09", "Turin", 641, "T9"), ("H17", "Naples", 483, "K7"),
        ("I22", "Genoa", 271, "M3"), ("J44", "Turin", 395, "T9"),
        ("K11", "Milan", 641, "K7"), ("L28", "Rome", 182, "M3"),
        ("M36", "Turin", 271, "T9"), ("N15", "Genoa", 395, "K7"),
        ("O42", "Milan", 483, "M3"), ("P07", "Rome", 641, "T9"),
        ("Q19", "Turin", 483, "K7"), ("R31", "Naples", 271, "M3"),
        ("S25", "Genoa", 182, "T9"), ("T48", "Milan", 395, "K7"),
        ("U13", "Rome", 483, "M3"), ("V37", "Turin", 641, "T9"),
        ("W06", "Naples", 182, "K7"), ("X29", "Genoa", 641, "M3"),
        ("Y41", "Milan", 182, "T9"), ("Z16", "Rome", 271, "K7"),
        ("AA08", "Turin", 395, "M3"), ("BB23", "Naples", 641, "T9"),
        ("CC35", "Genoa", 395, "M3"), ("DD47", "Milan", 641, "K7"),
    ]

    records_text = ""
    for i, (uid, city, num, code) in enumerate(records):
        records_text += f"Record {i+1}: User {uid} lives in {city}, favorite number {num}, project code {code}.\n"

    filler = generate_filler(filler_tokens)

    prompt = f"""You will read a list of records.

{records_text}
Do not answer yet.

{filler}

Question: Which user lives in Turin and has favorite number 182?
Answer with the user ID only. Return ONLY the ID (like D21)."""

    return {
        "id": f"kv_record_search_{filler_tokens}t",
        "category": "kv_recall",
        "prompt": prompt,
        "expected": "D21",
        "filler_tokens": filler_tokens,
        "check": "contains_word",
    }


# ── Build all tests ──────────────────────────────────────────────────────────

def build_tests(filler_levels=None):
    """Build test suite with multiple filler lengths."""
    if filler_levels is None:
        filler_levels = [200, 500, 1000]

    tests = []
    for filler in filler_levels:
        tests.append(make_test_numeric_block(filler))
        tests.append(make_test_grid_lookup(filler))
        tests.append(make_test_delayed_matrix(filler))
        tests.append(make_test_kv_binding(filler))
        tests.append(make_test_record_search(filler))

    return tests


# ── Answer checking ──────────────────────────────────────────────────────────

import re

def strip_thinking(s: str) -> str:
    return re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()

def normalize(s: str) -> str:
    s = strip_thinking(s)
    return s.replace(" ", "").replace("\n", "").replace("`", "").replace("*", "").strip().lower()

def extract_last_line(s: str) -> str:
    s = strip_thinking(s)
    lines = [l.strip() for l in s.strip().split('\n') if l.strip()]
    return lines[-1] if lines else s


def check_answer(test: dict, response: str) -> tuple:
    """Returns (passed, lines_correct, lines_total, detail)."""
    resp = strip_thinking(response).strip()
    resp_last = extract_last_line(response)
    mode = test["check"]

    if mode == "line_match":
        # Compare line by line
        expected_lines = test["expected_lines"]
        resp_lines = [l.strip() for l in resp.split('\n') if l.strip() and l.strip()[0].isdigit()]
        correct = 0
        total = len(expected_lines)
        details = []
        for i, exp in enumerate(expected_lines):
            if i < len(resp_lines) and resp_lines[i].replace(" ", "") == exp:
                correct += 1
                details.append(f"  Line {i+1}: MATCH")
            else:
                got = resp_lines[i].replace(" ", "") if i < len(resp_lines) else "(missing)"
                details.append(f"  Line {i+1}: MISMATCH expected={exp} got={got}")
        passed = correct == total
        return passed, correct, total, f"{correct}/{total} lines correct"

    elif mode == "exact_normalized":
        expected = normalize(test["expected"])
        actual = normalize(resp)
        actual_last = normalize(resp_last)
        ok = expected in actual or expected in actual_last
        return ok, 1 if ok else 0, 1, f"expected={test['expected']} got={resp_last[:60]}"

    elif mode == "contains_number":
        expected = test["expected"]
        ok = expected in resp or expected in resp_last
        return ok, 1 if ok else 0, 1, f"expected={expected} in={resp_last[:60]}"

    elif mode == "contains_word":
        expected = test["expected"].lower()
        ok = expected in resp.lower() or expected in resp_last.lower()
        return ok, 1 if ok else 0, 1, f"expected={expected} in={resp_last[:60]}"

    return False, 0, 1, "unknown check"


# ── API client ───────────────────────────────────────────────────────────────

async def send_prompt(session, url, model, prompt, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }
    try:
        async with session.post(f"{url}/api/chat", json=payload,
                                timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                return f"[ERROR: HTTP {resp.status}]"
            data = await resp.json()
            return data.get("message", {}).get("content", "[no content]")
    except Exception as e:
        return f"[ERROR: {e}]"


# ── Collect ──────────────────────────────────────────────────────────────────

async def collect(args):
    filler_levels = [int(x) for x in args.filler.split(",")]
    tests = build_tests(filler_levels)

    print(f"Running {len(tests)} KV cache stress tests against {args.url}")
    print(f"  Label: {args.label}, Model: {args.model}")
    print(f"  Filler levels: {filler_levels} tokens")
    print(f"  Temperature: {args.temperature}")
    print()

    results = []
    total_pass = 0
    total_lines_correct = 0
    total_lines = 0

    async with aiohttp.ClientSession() as session:
        for test in tests:
            response = await send_prompt(session, args.url, args.model,
                                         test["prompt"], args.temperature)
            passed, lines_ok, lines_total, detail = check_answer(test, response)

            status = "PASS" if passed else "FAIL"
            if passed:
                total_pass += 1
            total_lines_correct += lines_ok
            total_lines += lines_total

            filler = test["filler_tokens"]
            print(f"  [{status}] {test['id']:<35} filler={filler:>5}t  {detail}")

            results.append({
                "id": test["id"],
                "category": test["category"],
                "filler_tokens": filler,
                "passed": passed,
                "lines_correct": lines_ok,
                "lines_total": lines_total,
                "detail": detail,
                "response_preview": strip_thinking(response)[:200],
            })

    total = len(tests)
    pct = total_pass / total * 100 if total > 0 else 0
    line_pct = total_lines_correct / total_lines * 100 if total_lines > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {args.label}: {total_pass}/{total} tests passed ({pct:.1f}%)")
    print(f"  Line-level accuracy: {total_lines_correct}/{total_lines} ({line_pct:.1f}%)")
    print(f"{'='*60}")

    # Breakdown by filler level
    print(f"\n  Filler    Passed   Line accuracy")
    print(f"  {'-'*40}")
    for fl in filler_levels:
        fl_results = [r for r in results if r["filler_tokens"] == fl]
        fl_pass = sum(1 for r in fl_results if r["passed"])
        fl_total = len(fl_results)
        fl_lines_ok = sum(r["lines_correct"] for r in fl_results)
        fl_lines_total = sum(r["lines_total"] for r in fl_results)
        fl_line_pct = fl_lines_ok / fl_lines_total * 100 if fl_lines_total > 0 else 0
        print(f"  {fl:>5}t    {fl_pass}/{fl_total}      {fl_lines_ok}/{fl_lines_total} ({fl_line_pct:.1f}%)")

    # Breakdown by test type
    print(f"\n  Test type              Passed")
    print(f"  {'-'*35}")
    test_types = ["numeric_block", "grid_lookup", "delayed_matrix", "binding", "record_search"]
    type_names = ["Numeric block recall", "Grid cell lookup", "Delayed matrix mult", "Key-value binding", "Record search"]
    for ttype, tname in zip(test_types, type_names):
        t_results = [r for r in results if ttype in r["id"]]
        t_pass = sum(1 for r in t_results if r["passed"])
        t_total = len(t_results)
        print(f"  {tname:<22} {t_pass}/{t_total}")

    output = {
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
        "filler_levels": filler_levels,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score": {"passed": total_pass, "total": total, "percent": pct,
                  "lines_correct": total_lines_correct, "lines_total": total_lines,
                  "lines_percent": line_pct},
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {args.output}")


# ── Compare ──────────────────────────────────────────────────────────────────

def compare(args):
    data = []
    for f in args.files:
        with open(f) as fh:
            data.append(json.load(fh))

    print(f"\n{'='*70}")
    print(f"  TurboQuant KV Cache Stress Test Comparison")
    print(f"{'='*70}\n")

    # Overall
    print(f"  {'Label':<15} {'Tests':>8} {'Lines':>12} {'Test %':>8} {'Line %':>8}")
    print(f"  {'-'*55}")
    for d in data:
        s = d["score"]
        print(f"  {d['label']:<15} {s['passed']}/{s['total']:>3}     "
              f"{s['lines_correct']}/{s['lines_total']:>3}      "
              f"{s['percent']:>5.1f}%   {s['lines_percent']:>5.1f}%")

    # By filler level
    print(f"\n  By filler distance:")
    for d in data:
        print(f"\n  {d['label']}:")
        for fl in d.get("filler_levels", []):
            fl_results = [r for r in d["results"] if r["filler_tokens"] == fl]
            fl_pass = sum(1 for r in fl_results if r["passed"])
            fl_total = len(fl_results)
            fl_lines_ok = sum(r["lines_correct"] for r in fl_results)
            fl_lines_total = sum(r["lines_total"] for r in fl_results)
            fl_line_pct = fl_lines_ok / fl_lines_total * 100 if fl_lines_total > 0 else 0
            print(f"    {fl:>5}t: {fl_pass}/{fl_total} tests, {fl_lines_ok}/{fl_lines_total} lines ({fl_line_pct:.1f}%)")

    # Divergent results
    print(f"\n  Divergent results:")
    print(f"  {'-'*60}")
    any_diff = False
    if len(data) > 1:
        test_ids = [r["id"] for r in data[0]["results"]]
        for tid in test_ids:
            rows = []
            for d in data:
                r = next((x for x in d["results"] if x["id"] == tid), None)
                if r:
                    rows.append((d["label"], r["passed"], r.get("detail", "")))
            statuses = [r[1] for r in rows]
            if len(set(statuses)) > 1:
                any_diff = True
                print(f"\n  {tid}:")
                for label, passed, detail in rows:
                    status = "PASS" if passed else "FAIL"
                    print(f"    [{status}] {label:<15} {detail}")

    if not any_diff:
        print(f"  None — identical pass/fail across all cache types.")

    if args.markdown:
        print(f"\n\n### Markdown\n")
        print(f"| Label | Tests | Lines | Test % | Line % |")
        print(f"|:---:|:---:|:---:|:---:|:---:|")
        for d in data:
            s = d["score"]
            print(f"| {d['label']} | {s['passed']}/{s['total']} | "
                  f"{s['lines_correct']}/{s['lines_total']} | "
                  f"{s['percent']:.1f}% | {s['lines_percent']:.1f}% |")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboQuant KV Cache Stress Test")
    sub = parser.add_subparsers(dest="command")

    c = sub.add_parser("collect")
    c.add_argument("--url", default="http://localhost:11434")
    c.add_argument("--model", default="qwen3-14b")
    c.add_argument("--label", required=True)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--filler", default="200,500,1000",
                   help="Comma-separated filler token counts (default: 200,500,1000)")
    c.add_argument("--output", "-o")

    p = sub.add_parser("compare")
    p.add_argument("files", nargs="+")
    p.add_argument("--markdown", action="store_true")

    args = parser.parse_args()
    if args.command == "collect":
        asyncio.run(collect(args))
    elif args.command == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
