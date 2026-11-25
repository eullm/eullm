#!/usr/bin/env python3
"""Decision Desk: three local models on one EuLLM server, at work on a page.

Customer emails for a fictional e-bike shop arrive in six languages, and
every one goes through four steps that the page draws as they happen:

  1. **Find** — an embedding model (Qwen3-Embedding 0.6B) ranks the shop's
     policy passages by how close they are to the email (`/v1/embeddings`).
  2. **Decide** — a decision model (Jev-Style 2B) answers five typed
     questions about the email at once, with probabilities and nothing
     generated (`/v1/systemone`): which team, legitimate or phishing or
     spam, does it need an answer today, how upset is the sender, does it
     talk about someone's health.
  3. **Rule** — plain code acts on those probabilities, and on what code can
     check exactly (an IBAN, a card number, a tax code, by checksum):
     phishing and spam are set aside, health data and legal matters go to a
     person, a team the model is unsure of goes to a person, the rest gets a
     draft.
  4. **Write** — only then a chat model (Gemma 4 E4B) drafts the reply, in
     the customer's language, from the passages found in step 1
     (`/v1/chat/completions`, streamed). An email set aside costs no
     generation at all, and the page counts what that saved.

Every decision and every draft is a line in EuLLM's audit trail; when the
server runs on this machine the page shows the lines each email added, the
email itself in them only as a SHA-256. Nothing leaves your machine.

    sh examples/decision-desk/start.sh      # pulls the models, starts both

or, with `eullm serve` already running with the three models:

    eullm serve --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \\
                --embedding-model qwen3-embedding-0.6b-gguf-q8_0
    python examples/decision-desk/desk.py   # then http://127.0.0.1:8770

`--check` answers the built-in emails without the page and without drafting,
and says how many went where they should: measure before you show it. Only
the Python standard library is needed.
"""

import argparse
import http.client
import http.server
import json
import math
import os
import pathlib
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

PAGE = pathlib.Path(__file__).with_name("desk.html")

COMPANY = "Ventoverde Bikes"

# The teams the decision model chooses among: name → what the team handles.
TEAMS = {
    "billing": "invoices, payments, double charges, refunds of money",
    "shipping": "deliveries, tracking, late or lost parcels, collecting a parcel",
    "returns": "returns, exchanges, wrong size, defective or broken products, warranty",
    "tech_support": "the website, the app, logging in, the account, error messages",
    "privacy": "requests about personal data: access, correction, deletion (GDPR)",
    "legal": "formal notices, lawyers, courts, legal threats",
    "sales": "dealers, shops, wholesale price lists, partnerships",
    "careers": "job applications and CVs",
}

TEAM_LABELS = {
    "billing": "Billing",
    "shipping": "Shipping",
    "returns": "Returns & warranty",
    "tech_support": "Tech support",
    "privacy": "Privacy",
    "legal": "Legal",
    "sales": "Sales",
    "careers": "Careers",
}

# Asked as a choice, not as "is this phishing?": in decision-triage the
# yes/no form put an angry customer and a lawyer's letter over 0.7, the
# choice kept every legitimate email under 0.5.
KINDS = {
    "legitimate": (
        "a customer, supplier, lawyer, job applicant or partner really writing to the shop"
    ),
    "phishing": (
        "pretends to be a bank, a courier or a service to steal passwords or "
        "money, with a link or a request to pay"
    ),
    "spam": "unsolicited advertising",
}


def questions(teams):
    """The five questions every email is asked, in the order the page shows
    them. The state is read once for all of them."""
    return {
        "team": {
            "type": "choice",
            "instructions": "Which team of the shop should handle this email?",
            "criteria": teams,
        },
        "kind": {
            "type": "choice",
            "instructions": "What kind of email is this?",
            "criteria": KINDS,
        },
        "urgent": {
            "type": "noul",
            "instructions": "The sender needs an answer today.",
        },
        "mood": {
            "type": "score",
            "instructions": "How upset is the sender?",
            "criteria": ["calm", "dissatisfied", "angry"],
        },
        "health": {
            "type": "noul",
            "instructions": (
                "The email talks about a person's health: illness, hospital "
                "stays, operations, therapies."
            ),
        },
    }


QUESTION_LABELS = {
    "team": "Which team?",
    "kind": "Legitimate, phishing or spam?",
    "urgent": "Needs an answer today?",
    "mood": "How upset is the sender?",
    "health": "Talks about someone's health?",
}

# The shop's policies: what the embedding model searches and the chat model
# answers from. Each paragraph is a passage.
KNOWLEDGE = [
    {
        "title": "Returns and exchanges",
        "text": "Any bike or accessory can be returned within 30 days of delivery. "
        "The return label is free everywhere in the EU and is sent by email "
        "within one working day of the request. An exchange for another size "
        "ships as soon as the return is scanned by the courier.",
    },
    {
        "title": "Refunds",
        "text": "Refunds go back to the original payment method within 14 days "
        "of receiving the return. A refund to a bank account (IBAN) is possible "
        "only for cash-on-delivery orders.",
    },
    {
        "title": "Warranty and defects",
        "text": "Every bike has the 2-year EU legal guarantee; batteries are "
        "covered for 2 years or 500 charge cycles. A product that arrives "
        "broken or defective is replaced: send two photos, and the replacement "
        "ships within 5 working days, before the broken one is collected.",
    },
    {
        "title": "Shipping times",
        "text": "Orders ship from our warehouse in Verona and arrive in 3 to 5 "
        "working days in the EU. Every parcel has tracking. When tracking has "
        "not moved for 3 days, we open an investigation with the courier and "
        "answer the customer within 24 hours; a delayed order is upgraded to "
        "express delivery for free.",
    },
    {
        "title": "Holding a parcel",
        "text": "The courier keeps a parcel at its depot for up to 10 working "
        "days on request, or delivers it to another address or a pickup point "
        "at no cost.",
    },
    {
        "title": "Invoices and payments",
        "text": "Invoices are in the customer's account under Orders. A double "
        "charge is verified with the payment provider and the extra amount is "
        "refunded within 5 working days. Business customers get a VAT invoice "
        "on request.",
    },
    {
        "title": "Personal data (GDPR)",
        "text": "Requests to access, correct or delete personal data are "
        "answered within 30 days, as the GDPR requires. Deletion is confirmed "
        "in writing; invoices are kept for 10 years because tax law requires "
        "it.",
    },
    {
        "title": "App and account",
        "text": "Error 500 at login in app version 3.2 on Android 15 is a known "
        "issue fixed in version 3.2.1, available in the store. A forgotten "
        "password is reset from the login page with the link sent by email.",
    },
    {
        "title": "Dealer programme",
        "text": "Shops can join the dealer programme: the price list is sent on "
        "request, with 25% off retail on orders above 5,000 euro and free "
        "shipping on orders above 2,000 euro.",
    },
    {
        "title": "Careers",
        "text": "Every application is read, and candidates hear back within two "
        "weeks. Open positions are listed on the website.",
    },
]

# The built-in inbox: what `--check` measures and the page starts with.
# `expect` is where each email should go, the label the check compares to.
SAMPLES = [
    {
        "id": "de-billing",
        "lang": "DE",
        "from": "Jonas Becker <jonas.becker@example.de>",
        "subject": "Doppelte Abbuchung für Rechnung 2026-0917",
        "body": "Guten Tag, auf meinem Kontoauszug sehe ich zwei Abbuchungen über "
        "je 89,90 Euro für dieselbe Rechnung 2026-0917. Können Sie das prüfen und "
        "mir den doppelten Betrag zurückerstatten? Danke.",
        "expect": {"team": "billing", "action": "reply"},
    },
    {
        "id": "it-shipping",
        "lang": "IT",
        "from": "Giulia Rossi <giulia.rossi@example.it>",
        "subject": "Ordine 48213 non ancora arrivato",
        "body": "Salve, il mio ordine doveva arrivare martedì ma il tracciamento è "
        "fermo da cinque giorni. Mi serve per sabato, è un regalo per mio figlio. "
        "Sapete dirmi dov'è il pacco?",
        "expect": {"team": "shipping", "action": "reply"},
    },
    {
        "id": "fr-returns",
        "lang": "FR",
        "from": "Camille Moreau <camille.moreau@example.fr>",
        "subject": "INADMISSIBLE – batterie défectueuse",
        "body": "C'est la troisième fois que je vous écris ! Le vélo est arrivé "
        "avec une batterie qui ne charge pas, j'ai payé 2 100 euros et je n'ai "
        "aucune réponse. Je veux une solution AUJOURD'HUI.",
        "expect": {"team": "returns", "action": "reply"},
    },
    {
        "id": "en-phishing",
        "lang": "EN",
        "from": "Account Security <security@ventoverde-verify-account.example.net>",
        "subject": "Your shop account will be suspended",
        "body": "Dear customer, we detected unusual activity. To avoid permanent "
        "suspension, confirm your password and card details within 24 hours at "
        "http://ventoverde-login-secure.example.net",
        "expect": {"team": None, "action": "quarantine"},
    },
    {
        "id": "es-privacy",
        "lang": "ES",
        "from": "Lucía Fernández <lucia.fernandez@example.es>",
        "subject": "Supresión de mis datos personales",
        "body": "Buenos días, conforme al artículo 17 del RGPD les solicito que "
        "eliminen todos los datos personales que tienen sobre mí y que me lo "
        "confirmen por escrito.",
        "expect": {"team": "privacy", "action": "reply"},
    },
    {
        "id": "it-legal",
        "lang": "IT",
        "from": "Studio Legale Conti <avv.conti@example.it>",
        "subject": "Diffida ad adempiere – sig. Romano",
        "body": "In nome e per conto del sig. Romano vi diffido a consegnare la "
        "bicicletta pagata il 3 settembre entro 15 giorni dal ricevimento della "
        "presente; in difetto procederemo nelle sedi opportune.",
        "expect": {"team": "legal", "action": "person"},
    },
    {
        "id": "nl-tech",
        "lang": "NL",
        "from": "Daan de Vries <daan.devries@example.nl>",
        "subject": "Kan niet inloggen in de app",
        "body": "Sinds gisteren krijg ik 'error 500' als ik probeer in te loggen "
        "in de app. Ik heb hem al opnieuw geïnstalleerd. Android 15.",
        "expect": {"team": "tech_support", "action": "reply"},
    },
    {
        "id": "en-sales",
        "lang": "EN",
        "from": "Alpine Cycles <purchasing@alpinecycles.example>",
        "subject": "Dealer price list",
        "body": "Hello, we run three bike shops in Tyrol and would like your "
        "dealer price list and your terms for orders above 5,000 euro.",
        "expect": {"team": "sales", "action": "reply"},
    },
    {
        "id": "de-health",
        "lang": "DE",
        "from": "Helga Wagner <helga.wagner@example.de>",
        "subject": "Paket kann nicht angenommen werden",
        "body": "Guten Tag, ich liege nach einer Knieoperation im Krankenhaus und "
        "kann das Paket erst nächste Woche annehmen. Können Sie es so lange "
        "aufbewahren?",
        "expect": {"team": "shipping", "action": "person"},
    },
    {
        "id": "it-iban",
        "lang": "IT",
        "from": "Sara Marino <sara.marino@example.it>",
        "subject": "Reso taglia sbagliata",
        "body": "Ciao, ho ricevuto il casco in taglia M invece di L. Come faccio il "
        "cambio? Per un eventuale rimborso l'IBAN è IT60X0542811101000000123456.",
        "expect": {"team": "returns", "action": "reply"},
    },
    {
        "id": "fr-spam",
        "lang": "FR",
        "from": "Top Référencement <offres@seo-boost.example.com>",
        "subject": "Première page Google en 7 jours !!!",
        "body": "Offre exceptionnelle : votre boutique en première page de Google "
        "en 7 jours, garanti. -70% seulement cette semaine. Cliquez ici pour en "
        "profiter.",
        "expect": {"team": None, "action": "spam"},
    },
    {
        "id": "en-careers",
        "lang": "EN",
        "from": "Elena Costa <elena.costa@example.com>",
        "subject": "Application – bike mechanic",
        "body": "Hello, I am applying for the bike mechanic position on your "
        "website. I have six years of experience with e-bikes and can start "
        "next month. My CV is attached.",
        "expect": {"team": "careers", "action": "reply"},
    },
]

# Qwen3-Embedding is trained with an instruction before the query and none
# before the documents.
QUERY_INSTRUCTION = (
    "Instruct: Given a customer email, retrieve the shop policies that answer it\nQuery:"
)


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class EuLLMError(RuntimeError):
    """EuLLM answered with an error, or could not be reached."""


class EuLLM:
    """One EuLLM server: its URL, its API key when it requires one, and how
    long to wait for each answer."""

    def __init__(self, url, api_key=None, timeout=120.0):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # A server on this machine is asked directly: urllib would send even
        # 127.0.0.1 through $HTTP_PROXY unless $NO_PROXY names it.
        local = urllib.parse.urlparse(self.url).hostname in LOCAL_HOSTS
        handlers = [urllib.request.ProxyHandler({})] if local else []
        self.opener = urllib.request.build_opener(*handlers)

    def request(self, path, payload=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            return self.opener.open(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise EuLLMError(f"{path}: HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise EuLLMError(
                f"{self.url}: {e.reason} — is `eullm serve` running?"
            ) from None
        except TimeoutError:
            raise EuLLMError(f"{path}: no answer within {self.timeout:g}s") from None

    def call(self, path, payload=None, timeout=None):
        """The JSON `path` answers, or `EuLLMError` saying what went wrong."""
        try:
            with self.request(path, payload, timeout) as response:
                return json.load(response)
        except TimeoutError:
            raise EuLLMError(f"{path}: no answer within {self.timeout:g}s") from None
        except json.JSONDecodeError as e:
            raise EuLLMError(
                f"{path}: answered with something that is not JSON ({e})"
            ) from None
        except (http.client.HTTPException, OSError) as e:
            raise EuLLMError(
                f"{path}: the connection dropped mid-answer ({e})"
            ) from None

    def embed(self, texts, model):
        body = self.call("/v1/embeddings", {"model": model, "input": list(texts)})
        rows = sorted(body["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]

    def decide(self, state, asked, model=None):
        payload = {"state": state, "questions": asked}
        if model:
            payload["model"] = model
        return self.call("/v1/systemone", payload)

    def chat_stream(self, payload):
        """`/v1/chat/completions` with `stream: true`: every SSE chunk, parsed."""
        try:
            with self.request(
                "/v1/chat/completions", {**payload, "stream": True}
            ) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    yield json.loads(data)
        except TimeoutError:
            raise EuLLMError(f"chat: no answer within {self.timeout:g}s") from None
        except json.JSONDecodeError as e:
            raise EuLLMError(f"chat: a chunk that is not JSON ({e})") from None
        except (http.client.HTTPException, OSError) as e:
            raise EuLLMError(f"chat: the connection dropped mid-answer ({e})") from None


def unit(vector):
    """The vector scaled to length 1, so that a dot product is a cosine."""
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class Index:
    """The policy passages and their vectors, embedded once."""

    def __init__(self, client, model, passages):
        self.client, self.model, self.passages = client, model, passages
        texts = [f"{p['title']}: {p['text']}" for p in passages]
        self.vectors = [unit(v) for v in client.embed(texts, model)]

    def search(self, text, k):
        """The `k` passages closest to `text`, as (passage, cosine)."""
        query = unit(self.client.embed([QUERY_INSTRUCTION + text], self.model)[0])
        scores = [sum(a * b for a, b in zip(query, v)) for v in self.vectors]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(self.passages[i], scores[i]) for i in ranked[:k]]


def iban_ok(candidate):
    """ISO 13616 check digits: the IBAN, rotated and read as a number, is 1
    modulo 97."""
    rotated = candidate[4:] + candidate[:4]
    digits = "".join(str(int(c, 36)) for c in rotated)
    return int(digits) % 97 == 1


def luhn_ok(number):
    total = 0
    for i, d in enumerate(reversed(number)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def personal_data(text):
    """The personal data code finds exactly, not the model: IBANs and payment
    card numbers whose check digits hold, and Italian tax codes."""
    found = []
    for match in re.finditer(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b", text):
        if iban_ok(match.group().replace(" ", "")):
            found.append("IBAN")
            break
    for match in re.finditer(r"\b\d(?:[ -]?\d){12,18}\b", text):
        if luhn_ok(re.sub(r"[ -]", "", match.group())):
            found.append("card number")
            break
    if re.search(r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b", text.upper()):
        found.append("tax code")
    return found


def options_of(answer):
    """One answer as the page draws it: every option with its probability,
    the chosen one marked."""
    kind = answer.get("type")
    if kind == "noul":
        p = float(answer["noul"])
        return [
            {"name": "yes", "p": p, "chosen": p >= 0.5},
            {"name": "no", "p": 1.0 - p, "chosen": p < 0.5},
        ]
    probabilities = answer.get("probabilities") or {}
    if kind == "score":
        legend = answer.get("legend") or {}
        best = max(probabilities, key=lambda k: probabilities[k], default=None)
        return [
            {"name": str(legend.get(k, k)), "p": float(p), "chosen": k == best}
            for k, p in probabilities.items()
        ]
    chosen = answer.get("choice")
    return [
        {"name": k, "p": float(p), "chosen": k == chosen}
        for k, p in sorted(probabilities.items(), key=lambda kv: -kv[1])
    ]


def policy(answers, found, args):
    """What happens to the email: plain code over the model's probabilities
    and what code found in it. The thresholds are this script's flags.

    Returns the action (`reply`, `quarantine`, `spam`, `person`), the team,
    the priority, the flags, and the rule that decided, in words."""
    team = answers["team"]["choice"]
    p_team = float(answers["team"]["probabilities"][team])
    p_legal = float(answers["team"]["probabilities"].get("legal", 0.0))
    kind = answers["kind"]["probabilities"]
    urgent = float(answers["urgent"]["noul"]) >= args.threshold
    angry = float(answers["mood"]["score"]) >= 1.5
    health = float(answers["health"]["noul"]) >= args.threshold
    priority = "high" if urgent or angry else "normal"
    flags = list(found) + (["health data"] if health else [])

    if kind["phishing"] >= args.quarantine:
        action, rule = (
            "quarantine",
            f"phishing {kind['phishing']:.2f} ≥ {args.quarantine}",
        )
    elif kind["spam"] >= args.quarantine:
        action, rule = "spam", f"spam {kind['spam']:.2f} ≥ {args.quarantine}"
    elif health:
        action, rule = "person", "health data (GDPR art. 9): a person answers"
    elif team == "legal" or p_legal >= args.legal:
        # Legal is a second choice the model can hold just under the first
        # without being wrong: on the Jev-Style 2B the lawyer's notice to
        # deliver a bike came out shipping 0.49, legal 0.45. A letter that
        # may be legal is not one to draft a reply to.
        team = "legal"
        action, rule = "person", f"legal {p_legal:.2f} ≥ {args.legal}: a person answers"
    elif p_team < args.min_team:
        action, rule = "person", f"unsure of the team: {p_team:.2f} < {args.min_team}"
    else:
        action, rule = "reply", f"{TEAM_LABELS.get(team, team)} drafts the reply"
    if action in ("quarantine", "spam"):
        team, priority = None, None
    return {
        "action": action,
        "team": team,
        "team_label": TEAM_LABELS.get(team, team) if team else None,
        "p_team": p_team,
        "priority": priority,
        "flags": flags,
        "rule": rule,
        "draft": action == "reply",
    }


def state_of(mail, max_chars):
    return {
        "from": mail["from"],
        "subject": mail["subject"],
        "text": mail["body"][:max_chars],
    }


def draft_messages(mail, verdict, passages):
    """The chat model's prompt: the team it writes for, the email, and the
    passages found, numbered, to answer from."""
    team = verdict["team_label"]
    system = (
        f"You answer customer emails for the {team} team of {COMPANY}, a "
        "fictional e-bike shop that ships across the EU. Write the reply in the "
        "same language as the customer's email. Use only the facts in the "
        "numbered passages and cite them like [2]; never promise anything they "
        f'do not say. At most 120 words, no subject line. Sign as "{COMPANY} – '
        f'{team}".'
    )
    if verdict["priority"] == "high":
        system += (" The email has high priority: acknowledge it first, and apologise if the "
                   "customer is upset.")
    numbered = "\n".join(
        f"[{n}] {p['title']}: {p['text']}" for n, (p, _) in enumerate(passages, 1)
    )
    user = (
        f"From: {mail['from']}\nSubject: {mail['subject']}\n\n{mail['body']}\n\n"
        f"Passages:\n{numbered}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def default_audit_file():
    folder = os.environ.get("EULLM_AUDIT_DIR") or os.path.join(
        os.path.expanduser("~"), ".eullm", "audit"
    )
    return os.path.join(folder, "audit.jsonl")


class AuditTail:
    """The lines EuLLM appends to its audit trail while one email is
    handled, read from the file when the server runs on this machine."""

    KEEP = (
        "id",
        "timestamp",
        "model",
        "request_type",
        "input_tokens",
        "output_tokens",
        "duration_ms",
    )

    def __init__(self, path):
        self.path = path

    def mark(self):
        try:
            return os.path.getsize(self.path) if self.path else None
        except OSError:
            return None

    def since(self, mark, want=("systemone",), wait=1.0):
        """The lines written after `mark`, trimmed: waits up to `wait`
        seconds for a line of every request type in `want`."""
        if mark is None:
            return []
        deadline = time.monotonic() + wait
        while True:
            lines = self.read(mark)
            seen = {line.get("request_type") for line in lines}
            if all(w in seen for w in want) or time.monotonic() >= deadline:
                return lines
            time.sleep(0.1)

    def read(self, mark):
        try:
            with open(self.path, "rb") as f:
                f.seek(mark)
                raw = f.read(4 << 20)
        except OSError:
            return []
        lines = []
        for chunk in raw.splitlines():
            try:
                entry = json.loads(chunk)
            except ValueError:
                continue  # a line still being written
            if isinstance(entry, dict):
                lines.append(self.trim(entry))
        return lines

    @classmethod
    def trim(cls, entry):
        kept = {k: entry[k] for k in cls.KEEP if k in entry}
        decision = entry.get("decision")
        if isinstance(decision, dict):
            kept["decision"] = {
                "state_sha256": decision.get("state_sha256"),
                "answers": {
                    a.get("id"): a.get("answer")
                    for a in decision.get("answers", [])
                    if isinstance(a, dict)
                },
            }
        return kept


class ClientGone(Exception):
    """The page that asked went away: stop working for it."""


class Desk:
    """The three models and the shop's index, shared by every page."""

    def __init__(self, args):
        self.args = args
        self.client = EuLLM(args.url, args.api_key, args.timeout)
        self.audit = AuditTail(args.audit_file)
        self.lock = threading.Lock()
        self.index = None
        self.warmup = {
            "index": "waiting",
            "decision": "waiting",
            "chat": "waiting",
            "error": None,
        }

    def ensure_index(self):
        with self.lock:
            if self.index is None:
                self.index = Index(self.client, self.args.embedding_model, KNOWLEDGE)
                self.warmup["index"] = "ready"
            return self.index

    def warm_up(self):
        """Embed the passages and load the decision and chat models before the
        first email, so that nobody watches a model load: a plain `eullm
        serve` is enough. Retries until the server is up."""
        while True:
            try:
                self.ensure_index()
                break
            except EuLLMError as e:
                self.warmup["error"] = str(e)
                time.sleep(2.0)
        self.warmup["error"] = None
        if self.args.decision_model:
            # One question about nothing: it loads the model the desk names
            # into the server's decision slot, if it is not there already.
            self.warmup["decision"] = "loading"
            try:
                self.client.decide(
                    "ready",
                    {"ready": {"type": "noul", "instructions": "The desk is ready."}},
                    self.args.decision_model,
                )
                self.warmup["decision"] = "ready"
            except EuLLMError as e:
                self.warmup["decision"] = "failed"
                self.warmup["error"] = str(e)
        self.warmup["chat"] = "loading"
        try:
            for _ in self.client.chat_stream(
                {
                    "model": self.args.chat_model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                }
            ):
                pass
            self.warmup["chat"] = "ready"
        except EuLLMError as e:
            self.warmup["chat"] = "failed"
            self.warmup["error"] = str(e)

    def status(self):
        """What the page shows at the top: the server, its three models, and
        what the desk is set up with."""
        status = {
            "url": self.args.url,
            "company": COMPANY,
            "samples": [
                {k: s[k] for k in ("id", "lang", "from", "subject", "body")}
                for s in SAMPLES
            ],
            "knowledge": [p["title"] for p in KNOWLEDGE],
            "teams": TEAM_LABELS,
            "rules": {
                "quarantine": self.args.quarantine,
                "threshold": self.args.threshold,
                "min_team": self.args.min_team,
                "legal": self.args.legal,
            },
            "warmup": dict(self.warmup),
            "audit_file": self.args.audit_file
            if self.audit.mark() is not None
            else None,
            "models": {
                "embedding": {"name": self.args.embedding_model, "loaded": False},
                "decision": {"name": self.args.decision_model, "loaded": False},
                "chat": {"name": self.args.chat_model, "loaded": False},
            },
            "server": None,
            "error": None,
        }
        try:
            status["server"] = self.client.call("/api/version", timeout=3).get(
                "version"
            )
            running = self.client.call("/api/ps", timeout=3).get("models", [])
        except EuLLMError as e:
            status["error"] = str(e)
            return status
        for m in running:
            slot = (m.get("eullm") or {}).get("slot", "generation")
            name = m.get("name") or m.get("model")
            if slot == "decision":
                status["models"]["decision"] = {"name": name, "loaded": True}
            elif slot == "embedding":
                # Another embedding model in the slot is swapped for ours on
                # the first email: not loaded yet, as far as the page goes.
                status["models"]["embedding"]["loaded"] = (
                    name == self.args.embedding_model
                )
            elif name == self.args.chat_model:
                status["models"]["chat"]["loaded"] = True
        return status

    def process(self, mail, emit, draft=True):
        """Handle one email, telling `emit` every step as it ends. Returns
        the verdict."""
        started = time.perf_counter()
        mark = self.audit.mark()
        mail = {
            "from": str(mail.get("from", ""))[:300],
            "subject": str(mail.get("subject", ""))[:300],
            "body": str(mail.get("body", ""))[: self.args.max_chars],
        }
        emit({"stage": "start", "email": mail})

        t = time.perf_counter()
        hits = self.ensure_index().search(
            f"{mail['subject']}\n{mail['body']}", self.args.passages
        )
        emit(
            {
                "stage": "find",
                "model": self.args.embedding_model,
                "ms": (time.perf_counter() - t) * 1000.0,
                "passages": [
                    {"title": p["title"], "text": p["text"], "score": score}
                    for p, score in hits
                ],
            }
        )

        t = time.perf_counter()
        asked = questions(TEAMS)
        response = self.client.decide(
            state_of(mail, self.args.max_chars), asked, self.args.decision_model
        )
        answers = response["answers"]
        measured = (time.perf_counter() - t) * 1000.0
        emit(
            {
                "stage": "decide",
                "model": response.get("model"),
                "ms": (response.get("timing") or {}).get("total_ms") or measured,
                "audit_id": (response.get("eullm") or {}).get("audit_id"),
                "questions": [
                    {
                        "id": qid,
                        "label": QUESTION_LABELS[qid],
                        "options": options_of(answers[qid]),
                    }
                    for qid in asked
                ],
            }
        )

        found = personal_data(f"{mail['subject']} {mail['body']}")
        verdict = policy(answers, found, self.args)
        emit({"stage": "rule", **verdict})

        if verdict["draft"] and draft:
            self.write(mail, verdict, hits, emit)
        else:
            emit(
                {
                    "stage": "skip",
                    "why": verdict["rule"] if not verdict["draft"] else "drafting off",
                }
            )

        want = (
            ("systemone", "chat.completions")
            if verdict["draft"] and draft
            else ("systemone",)
        )
        lines = self.audit.since(mark, want)
        emit(
            {
                "stage": "audit",
                "file": self.args.audit_file if mark is not None else None,
                "lines": lines,
            }
        )
        emit({"stage": "done", "ms": (time.perf_counter() - started) * 1000.0})
        return verdict, answers

    def write(self, mail, verdict, hits, emit):
        emit({"stage": "write", "model": self.args.chat_model})
        t = time.perf_counter()
        first = None
        tokens = 0
        usage = None
        for chunk in self.client.chat_stream(
            {
                "model": self.args.chat_model,
                "messages": draft_messages(mail, verdict, hits),
                "max_tokens": self.args.max_tokens,
                "temperature": 0.3,
                # EuLLM's switch for a reasoning model's thinking: a short
                # draft from given passages does not need it.
                "think": False,
            }
        ):
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    if first is None:
                        first = time.perf_counter()
                    tokens += 1
                    emit({"stage": "token", "text": text})
        end = time.perf_counter()
        tokens = (usage or {}).get("completion_tokens") or tokens
        writing = end - (first or end)
        emit(
            {
                "stage": "written",
                "model": self.args.chat_model,
                "tokens": tokens,
                "ms": (end - t) * 1000.0,
                "first_token_ms": ((first or end) - t) * 1000.0,
                "tokens_per_s": tokens / writing if writing > 0 else None,
            }
        )


def handler(desk):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/status":
                self.send_json(desk.status())
            else:
                self.send_error(404)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/api/process":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_error(400)
                return
            if not isinstance(request, dict):
                self.send_error(400)
                return
            mail = request.get("email")
            if request.get("sample"):
                mail = next((s for s in SAMPLES if s["id"] == request["sample"]), None)
            if not isinstance(mail, dict) or not str(mail.get("body", "")).strip():
                self.send_json({"error": "an email needs a text"}, 400)
                return

            # NDJSON, one line per step, as each step ends.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(event):
                try:
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
                except OSError:
                    raise ClientGone() from None

            try:
                desk.process(mail, emit, draft=request.get("draft", True) is not False)
            except ClientGone:
                pass
            except (EuLLMError, KeyError, TypeError, ValueError) as e:
                try:
                    emit({"stage": "error", "message": str(e) or type(e).__name__})
                except ClientGone:
                    pass

    return Handler


def check(desk):
    """The built-in emails, decided and ruled on without drafting: where each
    went against where it should, and how long the decisions took."""
    right_team = right_action = 0
    times = []
    for sample in SAMPLES:
        events = []
        verdict, answers = desk.process(sample, events.append, draft=False)
        decide = next(e for e in events if e["stage"] == "decide")
        times.append(decide["ms"])
        expect = sample["expect"]
        team_ok = expect["team"] is None or verdict["team"] == expect["team"]
        action_ok = verdict["action"] == expect["action"]
        right_team += team_ok
        right_action += action_ok
        mark = "✓" if team_ok and action_ok else "✗"
        print(
            f"{mark} {sample['id']:<12} {verdict['action']:<10} "
            f"{(verdict['team'] or '-'):<13} p_team {verdict['p_team']:.2f}  "
            f"phishing {answers['kind']['probabilities']['phishing']:.2f}  "
            f"health {answers['health']['noul']:.2f}  {decide['ms']:.0f} ms"
            + ("" if action_ok else f"   (expected {expect['action']})")
            + ("" if team_ok else f"   (expected team {expect['team']})"),
            flush=True,
        )
    n = len(SAMPLES)
    print(
        f"\naction right {right_action}/{n}, team right {right_team}/{n}, "
        f"median decision {statistics.median(times):.0f} ms"
    )
    return right_action == n and right_team == n


def add_desk_options(parser):
    """The options of the desk itself — server, models, rules — which
    `desk_video.py` takes too."""
    parser.add_argument(
        "--url", default="http://localhost:11434", help="EuLLM server URL"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EULLM_API_KEY"),
        help="API key, when the server requires one (default: $EULLM_API_KEY)",
    )
    parser.add_argument(
        "--decision-model",
        default=None,
        help="decision model (default: the one the server has loaded)",
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding-0.6b-gguf-q8_0",
        help="embedding model (default: qwen3-embedding-0.6b-gguf-q8_0)",
    )
    parser.add_argument(
        "--chat-model", default="gemma-4-e4b", help="chat model (default: gemma-4-e4b)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability from which a yes/no answer counts as yes (default 0.5)",
    )
    parser.add_argument(
        "--quarantine",
        type=float,
        default=0.7,
        help="phishing or spam probability from which an email is set aside (default 0.7)",
    )
    parser.add_argument(
        "--min-team",
        type=float,
        default=0.4,
        help="below this probability for the chosen team, a person answers (default 0.4)",
    )
    parser.add_argument(
        "--legal",
        type=float,
        default=0.3,
        help="from this probability of the legal team, a person answers, "
        "whichever team is first (default 0.3)",
    )
    parser.add_argument(
        "--passages", type=int, default=3, help="passages found per email"
    )
    parser.add_argument(
        "--max-chars", type=int, default=4000, help="email text read at most"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=320, help="longest draft, in tokens"
    )
    parser.add_argument(
        "--audit-file",
        default=None,
        help="EuLLM's audit.jsonl, to show the lines each email adds "
        "(default: $EULLM_AUDIT_DIR/audit.jsonl or ~/.eullm/audit/audit.jsonl, "
        "when the server is on this machine)",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="per-request timeout, seconds"
    )
    return parser


def resolve(args):
    """`args` with the audit file filled in: EuLLM's own, when the server
    runs on this machine."""
    if args.audit_file is None:
        local = urllib.parse.urlparse(args.url).hostname in LOCAL_HOSTS
        args.audit_file = default_audit_file() if local else None
    return args


def main():
    parser = add_desk_options(
        argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="address the page is served on"
    )
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="decide the built-in emails without the page and say how many went right",
    )
    args = resolve(parser.parse_args())

    desk = Desk(args)
    if args.check:
        try:
            sys.exit(0 if check(desk) else 1)
        except EuLLMError as e:
            raise SystemExit(str(e)) from None

    threading.Thread(target=desk.warm_up, daemon=True).start()
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler(desk))
    server.daemon_threads = True
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    address = f"http://{shown}:{args.port}"
    print(
        f"Decision Desk on {address} — models on {args.url}. Ctrl+C to stop.",
        flush=True,
    )
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
