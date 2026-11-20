"""Labelled decisions for the qualification test (qualify.py).

An `Item` is one request to `/v1/systemone` — a state and its questions, in
the API's own format — and the right answer to each question. Two sources,
one shape:

  * a labelled set, JSONL, one request per line:

        {"id": "t1", "state": "Payouts failing for 3 days",
         "questions": {"team": {"type": "choice", "instructions": "Which team?",
                                "criteria": {"billing": "Payments", "tech": "Bugs"}},
                       "is_urgent": {"type": "noul", "instructions": "Is it urgent?"}},
         "answers": {"team": "billing", "is_urgent": true},
         "sources": {"team": "feedback:user", "is_urgent": "rules"}}

    `answers`: an option's name for a choice, a level's number (from 0) for
    a score, true/false (or "yes"/"no") for a noul; a question without one
    is not asked. `sources` is optional: where each answer came from, which
    `--sources` filters on. A line in bench/decision_calibration.py's
    format, `{"state", "question", "label"}`, is a request of one question.
    `eullm-forge decisions build` writes its held-out states in this format
    (`dev.labelled.jsonl`, `test.labelled.jsonl`);

  * a traces directory, written by a server started with
    `EULLM_DECISION_TRACES`: every decision with feedback becomes a request
    with the questions it asked and the answers the feedback gave, a later
    feedback line on a question overriding an earlier one. A line that is
    not a JSON object is skipped and counted.
"""

import json
import pathlib

KINDS = ("noul", "choice", "score")


class Item:
    def __init__(self, id, state, questions, answers, sources=None):
        self.id = id
        self.state = state  # a string, or structured JSON as the API takes it
        self.questions = questions  # id → System One question, in asking order
        self.answers = answers  # id → index of the right class
        self.sources = sources or {}  # id → where the right answer came from

    def kind(self, qid):
        return self.questions[qid]["type"]


class LabelledSet:
    def __init__(self, name, items, skipped=None):
        self.name, self.items = name, items
        self.skipped = skipped or {}  # reason → questions left out


def labels(question):
    """The answers' names in class order, as the API keys a response."""
    kind = question["type"]
    if kind == "noul":
        return ["yes", "no"]
    if kind == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def level_names(question):
    """A score's levels as an answer may name them: their text, or a
    `{"label", ...}` level's label."""
    names = []
    for level in question["criteria"]:
        if isinstance(level, dict):
            names.append(level.get("label"))
        else:
            names.append(level)
    return names


def answer_index(question, answer):
    """The class `answer` names; ValueError for one the question cannot have."""
    kind = question["type"]
    if kind == "noul":
        if isinstance(answer, bool):
            return 0 if answer else 1
        if isinstance(answer, (int, float)) and answer in (0, 1):
            return 0 if answer == 1 else 1
        text = str(answer).strip().lower()
        if text in ("yes", "true", "1"):
            return 0
        if text in ("no", "false", "0"):
            return 1
        raise ValueError(f"a noul answer is true or false, not {answer!r}")
    if kind == "choice":
        names = labels(question)
        if isinstance(answer, str) and answer in names:
            return names.index(answer)
        raise ValueError(f"{answer!r} is not one of the options {names}")
    n = len(question["criteria"])
    if isinstance(answer, bool):
        raise ValueError(f"a score answer is a level number, not {answer!r}")
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    if isinstance(answer, str):
        if answer.strip().isdigit():
            answer = int(answer.strip())
        elif answer in level_names(question):
            return level_names(question).index(answer)
    if isinstance(answer, int) and 0 <= answer < n:
        return answer
    raise ValueError(f"{answer!r} is not a level of 0..{n - 1}")


def check_question(question):
    """The question in the API's shape, or ValueError."""
    if not isinstance(question, dict) or question.get("type") not in KINDS:
        raise ValueError("a question needs a type: noul, choice or score")
    if not question.get("instructions"):
        raise ValueError("a question needs instructions")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice" and not (isinstance(criteria, dict) and len(criteria) >= 2):
        raise ValueError("a choice needs criteria: two options or more")
    if kind == "score" and not (isinstance(criteria, list) and len(criteria) >= 2):
        raise ValueError("a score needs criteria: two levels or more")
    return question


def api_question(spec):
    """A traced question in the API's shape: as it is when it has
    `criteria`, rebuilt from the shape the engine evaluated otherwise —
    `options` (`{name, description}` objects, pairs, or an object),
    `levels`, `true_means`/`false_means`."""
    if not isinstance(spec, dict):
        raise ValueError("a question must be an object")
    kind = spec.get("type", spec.get("kind"))
    out = {"type": kind, "instructions": spec.get("instructions")}
    if "criteria" in spec:
        out["criteria"] = spec["criteria"]
    elif kind == "noul":
        means = {k: spec.get(f"{k}_means") for k in ("true", "false")}
        if any(means.values()):
            out["criteria"] = {k: v for k, v in means.items() if v}
    elif kind == "choice":
        raw = spec.get("options")
        if isinstance(raw, dict):
            out["criteria"] = dict(raw)
        elif isinstance(raw, list):
            criteria = {}
            for option in raw:
                if isinstance(option, dict):
                    criteria[str(option.get("name", ""))] = option.get("description") or None
                elif isinstance(option, (list, tuple)) and option:
                    criteria[str(option[0])] = (option[1] if len(option) > 1 else None) or None
                else:
                    criteria[str(option)] = None
            out["criteria"] = criteria
    elif kind == "score":
        out["criteria"] = spec.get("levels")
    return check_question(out)


def item_from_row(row, n):
    """One line of a labelled set; ValueError when it is not one."""
    if "question" in row and "questions" not in row:
        row = {"id": row.get("id"), "state": row.get("state"),
               "questions": {"q": row["question"]}, "answers": {"q": row.get("label")},
               "sources": {"q": row["source"]} if "source" in row else {}}
    if row.get("state") is None or not isinstance(row.get("questions"), dict):
        raise ValueError("a line needs a state and its questions")
    answers = row.get("answers") or {}
    questions, indices = {}, {}
    for qid, question in row["questions"].items():
        if qid not in answers or answers[qid] is None:
            continue
        questions[qid] = check_question(question)
        indices[qid] = answer_index(question, answers[qid])
    if not questions:
        raise ValueError("no question with a right answer")
    return Item(str(row.get("id", n)), row["state"], questions, indices,
                dict(row.get("sources") or {}))


def from_jsonl(path):
    items = []
    for n, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            items.append(item_from_row(json.loads(line), n + 1))
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path}: line {n + 1}: {e}") from None
    return LabelledSet(pathlib.Path(path).stem, items)


def read_jsonl(path):
    """The JSON objects of a file, and how many lines were not one."""
    rows, bad = [], 0
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            bad += 1
    return rows, bad


def request_state(state):
    """The state to send: a traced state that is a JSON object or array went
    in as one, and goes in as one again, so each model reads it as it would
    from the client — indented for the code readout, on one line for a
    verdict model."""
    if isinstance(state, str) and state.strip()[:1] in ("{", "["):
        try:
            value = json.loads(state)
        except json.JSONDecodeError:
            return state
        if isinstance(value, (dict, list)):
            return value
    return state


def from_traces(directory):
    """Every traced decision that has feedback, with the feedback's answers."""
    directory = pathlib.Path(directory)
    files = {stem: sorted(directory.glob(f"{stem}*.jsonl")) for stem in ("decisions", "feedback")}
    if not files["decisions"]:
        raise ValueError(f"no decisions.jsonl in {directory}")
    skipped = {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    decisions, feedback = [], []
    for paths in files.values():
        for path in paths:
            rows, bad = read_jsonl(path)
            for _ in range(bad):
                skip("malformed line")
            for row in rows:
                (feedback if row.get("kind") == "feedback" else decisions).append(row)
    order = sorted(range(len(feedback)), key=lambda i: (str(feedback[i].get("timestamp") or ""), i))
    right = {}
    for i in order:
        row = feedback[i]
        if row.get("id") is not None and isinstance(row.get("answers"), dict):
            entry = right.setdefault(str(row["id"]), ({}, {}))
            for qid, answer in row["answers"].items():
                entry[0][str(qid)] = answer
                entry[1][str(qid)] = f"feedback:{row.get('source') or 'user'}"
    items = []
    for row in decisions:
        if str(row.get("id")) not in right or row.get("state") is None:
            continue
        answers, sources = right[str(row["id"])]
        raw = row.get("questions")
        pairs = list(raw.items()) if isinstance(raw, dict) else [
            (q.get("id", str(n)), q) for n, q in enumerate(raw or []) if isinstance(q, dict)]
        questions, indices = {}, {}
        for qid, spec in pairs:
            if str(qid) not in answers:
                continue
            try:
                question = api_question(spec)
                indices[str(qid)] = answer_index(question, answers[str(qid)])
            except ValueError as e:
                skip(f"feedback not usable: {e}")
                continue
            questions[str(qid)] = question
        if questions:
            items.append(Item(str(row["id"]), request_state(row["state"]), questions, indices,
                              {q: sources[q] for q in questions}))
    return LabelledSet(directory.name, items, skipped)


def only_sources(labelled, prefixes):
    """The answers whose source starts with one of `prefixes`; an answer
    with no recorded source is kept only when no prefix is given."""
    if not prefixes:
        return labelled
    items = []
    for item in labelled.items:
        keep = [q for q in item.questions
                if any(item.sources.get(q, "").startswith(p) for p in prefixes)]
        if keep:
            items.append(Item(item.id, item.state, {q: item.questions[q] for q in keep},
                              {q: item.answers[q] for q in keep},
                              {q: item.sources.get(q) for q in keep}))
    return LabelledSet(labelled.name, items, labelled.skipped)
