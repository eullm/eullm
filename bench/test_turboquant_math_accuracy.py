"""`contains_number` matches a number, not any digit run.

Expected '160' (S4: 7*8+6*9+5*10) used to pass a model answering '1600' --
plain and inside \\boxed{} -- inflating the pass rate this bench exists to
measure. Same substring class as the contains_word fix in
turboquant_quality.py; this mode is the number one.

Run this with `python -m pytest bench/test_turboquant_math_accuracy.py`;
the Bench job in CI runs it too.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "turboquant_math_accuracy", Path(__file__).resolve().parent / "turboquant_math_accuracy.py")
_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_mod)

S4 = next(t for t in _mod.build_tests([0]) if t.get("expected") == "160")
assert S4["check"] == "contains_number"


def test_a_longer_number_is_not_the_expected_one():
    assert _mod.check_answer(S4, "1600")[0] is False
    assert _mod.check_answer(S4, "The result is \\boxed{1600}")[0] is False
    assert _mod.check_answer(S4, "159")[0] is False


def test_the_right_answer_passes_bare_boxed_and_in_a_sentence():
    assert _mod.check_answer(S4, "160")[0] is True
    assert _mod.check_answer(S4, "\\boxed{160}")[0] is True
    assert _mod.check_answer(S4, "The answer is 160.")[0] is True


def test_every_expected_number_matches_itself():
    for t in _mod.build_tests([0]):
        if t["check"] != "contains_number":
            continue
        assert _mod.check_answer(t, t["expected"])[0] is True, t["id"]
        assert _mod.check_answer(t, "The answer is " + t["expected"] + ".")[0] is True, t["id"]
