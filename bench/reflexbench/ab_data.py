"""Sets for AutoBench, all brought to one shape: a request a chat model can
answer, and what grades the answer.

Each set is downloaded from its publisher on first use, pinned so that a
report can be reproduced — by a commit, an object version, or the SHA-256 of
the file, checked before anything is read from it — and kept in a cache
directory, never in the repository:

  * `gsm8k`: grade-school maths word problems (OpenAI, MIT,
    https://github.com/openai/grade-school-math), graded by the number after
    "Answer:";
  * `arc-easy`, `arc-challenge`: science exam questions with four options
    (AllenAI's ARC, CC BY-SA 4.0, https://allenai.org/data/arc), graded by
    the letter;
  * `mmlu`: four-option questions on 57 subjects (Hendrycks et al., MIT,
    https://github.com/hendrycks/test), graded by the letter.

Together they mix requests a small model answers as well as a large one with
requests it does not, which is what routing has to tell apart. `--data` takes
a set of your own (see `from_jsonl`).
"""

import csv
import hashlib
import io
import json
import pathlib
import random
import tarfile
import urllib.request
import zipfile

from rb_data import cache_dir, fetch

GSM8K = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"

# The ARC release is one 680 MB zip, nearly all of it a corpus under terms of
# its own that this benchmark has no use for. Its test files are read out of
# it with HTTP range requests — a few hundred kilobytes — at a fixed version
# of the object.
ARC = (
    "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip"
    "?versionId=s920702n9UBMzcce3j4P4ZMSNWrUdKSC"
)
ARC_FILES = {
    "arc-easy": (
        "ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl",
        "ce32a0774b4beb1977d2c6b16dda5dd2b24fdc9c7f506882c452678676bd09be",
    ),
    "arc-challenge": (
        "ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl",
        "9fa8ffb3e3a1f88cb302d53dd95aae81e39209ad9bcf974376b319d8fd35717c",
    ),
}

# Not versioned by its server: the SHA-256 is the pin.
MMLU = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLU_SHA256 = "bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b"

SETS = {
    "gsm8k": "GSM8K test: maths word problems, graded by the number",
    "arc-easy": "ARC-Easy test: science questions, four options",
    "arc-challenge": "ARC-Challenge test: harder science questions, four options",
    "mmlu": "MMLU test: 57 subjects, four options",
}

GRADERS = ("number", "letter", "exact", "f1", "judge")

NUMBER_INSTRUCTION = 'Solve the problem. End your answer with a line "Answer: <number>".'
LETTER_INSTRUCTION = 'Answer with the letter of the correct option, on a line "Answer: <letter>".'


class Item:
    """A request and how its answers are graded: `messages` for a chat
    request, or a bare `prompt` for /api/generate; `answer` is the reference
    a grader compares with, none for a `judge` item."""

    def __init__(self, id, set_name, messages, answer, grader, prompt=None):
        self.id, self.set = id, set_name
        self.messages, self.prompt = messages, prompt
        self.answer, self.grader = answer, grader

    @property
    def text(self):
        """The request's own text, for the length baseline and the embeddings."""
        if self.prompt is not None:
            return self.prompt
        return "\n".join(str(m.get("content") or "") for m in self.messages)


class Dataset:
    def __init__(self, name, items):
        self.name, self.items = name, items


def checked(data, sha256, what):
    """`data`, once it is the file the reports were made on."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(f"{what}: SHA-256 {digest}, expected {sha256}: not the pinned file")
    return data


def user(text):
    return [{"role": "user", "content": text}]


def choices_text(question, choices):
    """A multiple-choice question as the model reads it: the question, then
    one line per option."""
    lines = [question.strip(), ""]
    lines += [f"{label}. {text.strip()}" for label, text in choices]
    return "\n".join(lines) + "\n\n" + LETTER_INSTRUCTION


def gsm8k_items(data):
    items = []
    for n, line in enumerate(data.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        text = f"{row['question'].strip()}\n\n{NUMBER_INSTRUCTION}"
        items.append(Item(f"gsm8k-{n}", "gsm8k", user(text), answer, "number"))
    return items


def arc_items(data, set_name):
    items = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = row["question"]
        choices = [(c["label"], c["text"]) for c in question["choices"]]
        # A few questions label their options 1-4: they are asked as A-D,
        # and graded so.
        letters = "ABCDE"
        relabel = {label: letters[i] for i, (label, _) in enumerate(choices)}
        choices = [(relabel[label], text) for label, text in choices]
        answer = relabel.get(row["answerKey"], row["answerKey"])
        text = choices_text(question["stem"], choices)
        items.append(Item(f"{set_name}-{row['id']}", set_name, user(text), answer, "letter"))
    return items


def mmlu_items(rows):
    """`rows`: (subject, [question, a, b, c, d, answer]) in file order."""
    items = []
    for n, (subject, row) in enumerate(rows):
        question, options, answer = row[0], row[1:5], row[5].strip()
        choices = list(zip("ABCD", options, strict=True))
        topic = subject.replace("_", " ")
        text = choices_text(f"({topic}) {question}", choices)
        items.append(Item(f"mmlu-{subject}-{n}", "mmlu", user(text), answer, "letter"))
    return items


class RemoteFile(io.RawIOBase):
    """A file on an HTTP server that answers range requests, read in blocks
    of a megabyte as it is read: enough for `zipfile` to find one member of
    a large archive without downloading the rest."""

    BLOCK = 1 << 20

    def __init__(self, url, timeout=120):
        self.url, self.timeout, self.pos, self.blocks = url, timeout, 0, {}
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def block(self, n):
        if n not in self.blocks:
            start = n * self.BLOCK
            end = min(self.size, start + self.BLOCK) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                if r.status != 206:
                    raise OSError(f"{self.url}: the server does not answer range requests")
                self.blocks[n] = r.read()
        return self.blocks[n]

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.pos
        out = bytearray()
        while size > 0 and self.pos < self.size:
            block = self.block(self.pos // self.BLOCK)
            chunk = block[self.pos % self.BLOCK :][:size]
            out += chunk
            self.pos += len(chunk)
            size -= len(chunk)
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def arc_file(set_name):
    """One ARC test file, out of the release zip, cached once checked."""
    member, sha256 = ARC_FILES[set_name]
    path = cache_dir() / f"arc-{member.rsplit('/', 1)[1]}"
    if path.exists():
        return path.read_bytes()
    with zipfile.ZipFile(RemoteFile(ARC)) as archive:
        data = checked(archive.read(member), sha256, member)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)
    return data


def mmlu_rows():
    """Every MMLU test question as (subject, row), subjects in name order.
    The 166 MB archive is checked, its test questions kept as one JSONL file
    in the cache, and the archive deleted."""
    path = cache_dir() / "mmlu-test.jsonl"
    if not path.exists():
        archive_path = cache_dir() / "mmlu-data.tar"
        if not archive_path.exists():
            part = archive_path.with_name(archive_path.name + ".part")
            with urllib.request.urlopen(MMLU, timeout=600) as r, open(part, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
            part.replace(archive_path)
        digest = hashlib.sha256()
        with open(archive_path, "rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        if digest.hexdigest() != MMLU_SHA256:
            archive_path.unlink()
            raise ValueError(f"{MMLU}: SHA-256 {digest.hexdigest()}: not the pinned file")
        rows = []
        with tarfile.open(archive_path) as archive:
            members = sorted(
                (m for m in archive.getmembers() if m.name.startswith("data/test/")),
                key=lambda m: m.name,
            )
            for m in members:
                if not m.name.endswith("_test.csv"):
                    continue
                subject = m.name[len("data/test/") : -len("_test.csv")]
                text = io.TextIOWrapper(archive.extractfile(m), encoding="utf-8")
                rows += [(subject, row) for row in csv.reader(text) if len(row) == 6]
        part = path.with_name(path.name + ".part")
        part.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        part.replace(path)
        archive_path.unlink()
    lines = path.read_text(encoding="utf-8").splitlines()
    return [tuple(json.loads(line)) for line in lines if line.strip()]


def drawn(items, limit, seed):
    """`limit` items at most (0: all), in a fixed shuffled order."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    return [items[n] for n in (order[:limit] if limit else order)]


def load(name, limit=0, seed=1):
    """One of SETS."""
    if name == "gsm8k":
        items = gsm8k_items(checked(fetch(GSM8K), GSM8K_SHA256, GSM8K))
    elif name in ARC_FILES:
        items = arc_items(arc_file(name), name)
    elif name == "mmlu":
        items = mmlu_items(mmlu_rows())
    else:
        raise ValueError(f"unknown set {name!r}: one of {', '.join(SETS)}")
    return Dataset(name, drawn(items, limit, seed))


def from_jsonl(path, limit=0, seed=1):
    """A set of your own, one JSON object per line:
    {"id": ..., "messages": [...] | "prompt": "...", "answer": optional,
     "grader": "number" | "letter" | "exact" | "f1" | "judge"}.
    `judge` items need no answer: the judge compares the two models' answers
    (`--judge-model`)."""
    items = []
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        grader = row.get("grader", "exact")
        if grader not in GRADERS:
            raise ValueError(f"{path}: line {n + 1}: grader must be one of {', '.join(GRADERS)}")
        if grader != "judge" and row.get("answer") is None:
            raise ValueError(f"{path}: line {n + 1}: a {grader} item needs an answer")
        if ("messages" in row) == ("prompt" in row):
            raise ValueError(f"{path}: line {n + 1}: give either messages or prompt")
        name = pathlib.Path(path).stem
        items.append(
            Item(
                str(row.get("id", n)),
                name,
                row.get("messages"),
                row.get("answer"),
                grader,
                prompt=row.get("prompt"),
            )
        )
    return Dataset(pathlib.Path(path).stem, drawn(items, limit, seed))


def split(dataset, seed=1):
    """Dev and test halves: thresholds and neighbours come from dev, every
    router is scored on test, the same items for all."""
    ids = sorted(item.id for item in dataset.items)
    random.Random(f"split:{seed}").shuffle(ids)
    dev = set(ids[: len(ids) // 2])
    return (
        [i for i in dataset.items if i.id in dev],
        [i for i in dataset.items if i.id not in dev],
    )
