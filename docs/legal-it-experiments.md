# legal-it — what was tried, what it gave, and why the course changed

> Status: living record · Last updated: 2026-10-18

The record of every legal-it training line, from the first distillation on
Cassazione rulings to the GRPO rounds of October. The model cards say how
the *published* models were made; this page says what else was tried and
why it was set aside, so a direction is not dropped and forgotten, or
retried without knowing how it went. Numbers come from the exams on
Leonardo; each section names its source.

## The exams

| Exam | Size | What it measures |
|---|---|---|
| Held-out statute exam | 429 questions | the published models' numbers (model cards); used once per release candidate |
| Development statute exam (`devbig`) | 472 questions | content, deadlines, nonexistent articles, asked open book with hybrid retrieval (BM25 + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B); graded by Qwen3-30B-A3B-Instruct-2507 under rubric v3, and the deadline and absent-article questions (190) also by a program |
| The same, without the note (`-hybnn`) | 472 | the prompt no longer says "nella raccolta normativa non è presente art. N": RAG products built on the models do not write that note, so this is the product condition (`legal_eval.py --no-note`, 2026-10-16) |
| Practical exam | 145 dev / 70 held-out | a practical question needing a group of linked articles, written by Qwen3.6-27B (`eullm_forge.eval.practical`) |
| Case-law exam | 1,200 questions | answering from Consiglio di Stato rulings and citing the right one (`sbatch_cds_exam.slurm`) |

All numbers below are Q4_K_M GGUFs through llama-server unless marked bf16.

## 1. September: distillation on the Cassazione corpus

**What.** `docs/legal-it-4b-strategy.md` and ADR-001. A corpus of 1.13 M
chunks, about 700 M tokens: Cassazione civil and criminal judgments
2021-2026 from ItalGiure (`docs/corpus-acquisition.md`), pseudonymised by
`anonymize.py` (rounds 1-6), deduplicated, plus the codes and the
Constitution. Teacher Qwen3-30B-A3B-Base, student Qwen3-4B-Base, logit
distillation on Leonardo (`legal_it`, `legal_it_v12`; an 8B run in
`legal_it_8b`). Then stage 3 on the distilled student (domain pairs plus
identity): `legal-it-4b` v0.2.

**What it gave.** The student learned the language of the corpus: on 40
held-out chunks of 512 tokens its perplexity at step 12,600 was 3.19
against 6.65 for Qwen3-4B-Base (`legal_it_v12/exports/perplexity.csv`).

**Why the course changed (2026-09-26/27, commits 749a0ee and 689ec50).**
Measured on questions, not on perplexity:

- v0.2 matched Qwen3-4B-Instruct on the keyword score, but read by a person
  four answers in ten were wrong: the TAR deadline given as 30 days (it is
  60), the ricorso straordinario as 30 (it is 120), art. 27 Cost. described
  as art. 25, the AI Act disclosure duty backwards.
- On the held-out exam a 4B model answered about 3% of 233 questions from
  memory, whoever trained it, and 46-60% with the retrieved norms in the
  prompt.
- At exactly that, answering from the text in the prompt, the distilled
  model did worst: restating an article just read, 44 of 86 against 67 for
  Qwen3-4B-Instruct; a question about a nonexistent article got an
  invented answer 18 times out of 18.

So legal-it became a model that answers **from the texts retrieval gives
it**, trained from an instruct model on open-book pairs (section 2). The
distillation did not fail at what it measured; it measured the wrong
thing for the product. Two more reasons weighed on publishing a model
trained on rulings: the anonymiser had once reported a clean run and still
left 59 codici fiscali in the training text (strategy, risks table), and a
published model that memorised a ruling cannot be recalled.

**What is still there.** The checkpoints and exports on `$WORK`
(`legal_it/exports`, `legal_it_v12/exports`, `legal_it_8b`): keep them
until section 6 is decided.

**What was never measured.** Whether the corpus helps on case law or on
practical questions, which did not exist as exams then; and the distilled
student with the open-book recipe on top of it. Every comparison above is
on statute questions, where knowing the rulings cannot help.

## 2. October: the open-book recipe, and the published models

From an instruct model: open-book SFT on 18,972 pairs (85% written by
Qwen3-30B-A3B-Instruct-2507 from one article each, 15% about nonexistent
articles), then GRPO with programmatic rewards (deadlines, abstention).
Statutes only, no rulings. Published 2026-10-04:

| | Base | Held-out (429) | Base model, same retrieval |
|---|---|---|---|
| `legal-it-8b` | Ministral-3-8B-Instruct-2512 (Mistral AI) | 89.0% | 80.9% |
| `legal-it-4b` | Qwen3-4B-Instruct-2507 | 85.1% | 77.6% |

Ministral-3-14B with the same recipe scored 436 on `devbig`, the same as
the 8B (436 bf16, 437 Q4): on this exam a larger model of the same family
added nothing.

**What it cost** (A100 GPU-hours, from `sacct` for 2026-09-25 to 10-05):

| | Job | Wall time | GPU-hours |
|---|---|---|---|
| `legal-it-8b`, SFT | `eullm-s3-ministral-v04` (59092257) | 4 h 09 m on 1 A100 | ~4.2 |
| `legal-it-8b`, GRPO | `eullm-grpo-ministral-v04` (59125822) | 1 h 33 m on 3 A100 | ~4.6 |
| `legal-it-8b`, merge and GGUF | serial partition | 16 m | CPU only |
| Ministral-3-14B, SFT + GRPO | | | 5.8 + 6.4 |
| Shared: open-book pairs written by the teacher (`gen-openbook`) | | | 40.2 |
| Shared: exams (`exam`) and their grading (`judge`) | | | 24.3 + 63.4 |
| `gen-instruct` | | | 22.1 |

The model itself is cheap: **about 9 GPU-hours** of training for the
published 8B. The data and the measuring, shared by every model of the
line, cost far more: the teacher writing the pairs took 40 GPU-hours, the
exams and their grading 88. `gen-instruct`
is the September line's (section 1): it wrote the `instruct-pairs` files
stage 3 used on the distilled student (commit 7040d91). The published 8B's
SFT loaded `openbook-v04.jsonl` alone, 18,972 pairs plus 180 identity
examples (job 59092257's log: `--no-format-tokens --lr 1e-4 --epochs 1
--max-length 4096 --batch-size 1 --grad-accum 16`).

## 3. October: the 27B teacher and the Qwen3.5 students

**What.** Privileged-context distillation for the case-law skill (commit
ba6decf, `opd_train.py`, from 2026-10-06): the student answers with the
passages the case-law index retrieves, the teacher reads the same question
with the source ruling in front, and the student is moved towards the
teacher's distributions at each answer token (reverse KL). For the Qwen3.5
students the teacher was **Qwen3.8-27B** (`opd-q38t-q35-*`). Ministral
was distilled from itself: no available teacher shared its tokenizer.
Then GRPO rounds on top, with the judge as reward (`submit_grpo_round.sh`):
`jp` (practical prompts), `jp2` (a second round on `jp`), `jpa` (with
abstention prompts), `jpn` (the same with the note removed from 70% of the
prompts), `jp3n` and `jpn2` (one more round each).

**Development statute exam (472), correct, with / without the note:**

| Model | With | Without | Program-graded (190), with / without |
|---|---|---|---|
| **legal-it-8b** (published) | 437 | **440** | 183 / 183 |
| **legal-it-4b** (published) | 422 | **393** | 176 / **143** |
| OPD 9B, the start | 403 | 412 | 170 / 173 |
| jp-9b | 437 | 436 | 178 / 181 |
| jp2-9b | **447** | not yet | 177 |
| jp3n-9b | 439 | 437 | 175 / 175 |
| jpn-9b | 434 | 431 | 176 / 176 |
| OPD 4B, the start | 377 | — | 166 |
| jpn-4b | 390 | **427** | 176 / 176 |
| jpn2-4b | 410 | **426** | 177 / 175 |

**Paired (McNemar), without the note:**

- against legal-it-4b: jpn-4b +34 (59:25, p < 0.001), jpn2-4b +33 (56:23,
  p < 0.001), significant strictly and leniently. Almost all of it on
  nonexistent articles (36:2 and 35:0); content even (15:12, 14:11);
  deadlines slightly worse (4:8, 4:9). **The published 4B depends on the
  note** (393 without, 143 of 190 program-graded): without it, it answers
  about articles that are not there. The new 4B does not.
- against legal-it-8b: jp −4 (p 0.65), jp3n −3 (p 0.76), jpn −9 (p 0.24),
  and leniently the 8B is ahead (p 0.002-0.023). **No Qwen3.5-9B beats the
  Mistral 8B.**

**Practical exam (145), correct:** jp2-9b 82, jp3n-9b 81, jpn2-4b 81,
jpa-9b 75, jpn-4b 75, jp-9b 73, jpn-9b 67, OPD 4B 67, OPD 9B 62. The
published models' run is queued.

**Case-law exam (1,200), source ruling cited:** 0.348-0.362 for every
round, with the source ruling retrieved in 0.382 of the questions: the
models cite it about nine times in ten when it is there. **Case law is
held back by retrieval, not by the models.**

**What it says.** The 27B start (403) was far below where the published
models end (437); GRPO did most of the work on the Qwen3.5 line (+30 to
+44). A third round on `jp2` (`jp3n`) did not improve it:
its groups were often all graded the same, nothing left to learn from.

**Why Qwen3.5 may be underused (open).** The strategy noted it in
September (`legal-it-4b-strategy.md`, alternatives table): Qwen3.5's
hybrid architecture makes logit distillation noisier and its tooling
younger. Besides that: it is a reasoning model and every run here keeps
reasoning off; it never had the open-book SFT the published models had;
and `devbig` is near its ceiling (93%). Each is a measurement, not a
belief: reasoning on (needs a `legal_eval.py` option), the same SFT on
Qwen3.5, bf16 against Q4.

## 4. The release candidate

`jpn2-4b` (or `jpn3-4b`, in training) against `legal-it-4b`, without the
note, on the held-out exam, once. If it holds, it replaces `legal-it-4b`:
same content, and it says when an article is not there.

## 5. Teachers, summary

| Teacher | Used for | Published? |
|---|---|---|
| Qwen3-30B-A3B-Base | logit distillation on the Cassazione corpus (September) | no (v0.2) |
| Qwen3-30B-A3B-Instruct-2507 | writes the open-book SFT pairs; the exam judge | yes, as data writer |
| Qwen3.8-27B | privileged-context OPD of Qwen3.5-4B/9B (case-law skill) | no |
| Qwen3.6-27B | writes the practical exam and the practical GRPO prompts | as data writer |

## 6. The Cassazione, again

The corpus is not wasted; what it was used for was the wrong half of the
job. What it can still give, in order of likely return:

1. **Retrieval over rulings.** The case-law exam is capped by retrieval
   (0.382): an index, embedder and reranker that find the right ruling
   move every model at once, and train nothing on personal data.
2. **Open-book case-law training.** As the Consiglio di Stato cards do
   (`export_clean_cards.py`): questions on rulings, the rulings in the
   prompt, answers that cite them, with made-up references and only
   principles of law, then SFT and GRPO on `legal-it-8b`. It teaches the
   skill the product uses, reading rulings, not remembering them.
3. **Knowledge: the distilled corpus model under the open-book recipe.**
   The experiment never run in section 1: continued training on the
   corpus, then the same SFT and GRPO, measured on the practical and
   case-law exams as well as the statutes. Not on top of a published
   model without the SFT and GRPO after it: v0.2 is what continued
   training alone does to the reading skill.

Any model trained on ruling text goes through the memorisation check
(roadmap phase 3) before it is published.

## 7. An external test of legal-it-8b (2026-10-18, from reddit.com)

A legal-tech team tested the published `legal-it-8b` Q4_K_M and sent a
report, after a post about the model on reddit.com. Criminal law and
procedure only; 11 questions in the model card's format (3 retrieved texts,
the missing-article note, temperature 0, 400 tokens), and 5 inside their own
pipeline with 50,000-80,000-token prompts, far from the training format.
Small sample: signals, not rates.

**What held.** It finds the right text among similar ones, also in very
long prompts, and restates its principle faithfully; it corrects a numeric
premise when the right figure is in the text (the penalty for fraud, the
deadline for an appeal); the nonexistent-article answer with the note was
"exemplary"; clean Italian; it fits a 16 GB T4 with 86,000 tokens.

**What failed, and what answers it:**

| Failure | Example | Answer | Where |
|---|---|---|---|
| A false premise accepted with the right text in the prompt | "after the Cartabia reform art. 624-bis c.p. is prosecuted on complaint: by when?" (it is not; the reform concerns art. 624) | false-premise questions written from the statute text, rewarded only when the answer corrects the premise | next GRPO round |
| A wrong ruling number let through | "Sezioni Unite n. 18474/2025" for a principle another ruling in front of it states | `caselaw_wrong_cite` pairs | Cassazione SFT (`make_caselaw_sft.py`) |
| No abstention when the texts do not answer | the expected ruling was not retrieved; the answer was built on another one | absent-context pairs (statutes: `openbook-v05`; case law: `caselaw_absent`); the jpn rounds without the note | Cassazione SFT; 4B candidate |
| Citations that cannot be checked | none of 11 answers used [1]-[3]; the wrong passage, or the operative part instead of the reasoning | every case-law citation ends with its passage number `[n]` | Cassazione SFT |
| Content from memory | art. 360 c.p.c. in a criminal-procedure answer; criteria not in the texts | a reward for citing only articles and rulings present in the prompt (`check_cited_articles.py` as a reward) | next GRPO round |
| Reasoning | art. 444 c.p.p. criterion wrong; art. 4 l. 110/1975 and art. 699 c.p. confused; facts of a case read as legal requirements; contradictions | multi-article practical questions; a stronger teacher (open: reasoning on, Mistral teacher) | open |
| Multi-part questions, the reasoning part dropped | the boundary between two offences, how to frame the appeal | practical questions with a rubric per part | next GRPO round |
| Fines in lire | Normattiva keeps the original lire amounts (art. 640 c.p.) and the model reports them as current | flag lire amounts at indexing time, and train answers to say they are original amounts to be converted | corpus and product |
| A hidden system prompt | with llama.cpp `--jinja` and no system message, Ministral's chat template inserts its default system prompt (~525 tokens, "Le Chat") | check what the training template did, then ship a template without it, or with a short one of our own, trained and served alike | packaging |

Their tests were all criminal; our case-law training is civil only, since
criminal rulings never leave the cluster (section 6). Abstract principles of
criminal law hold no personal data, but whether their cards may be trained
on is a decision still to be taken.

## 8. 2026-10-20: teachers measured, false premises, and what a long SFT undoes

**The candidate teachers on the traps.** The 36 LexIA trap questions
(section 7; the set itself stays private and is never trained on), asked
without the note, as the product asks, graded by the judge:

| Model | Right | Premise corrected (of 18) | Abstained (of 9) | Answered (of 9) |
|---|---|---|---|---|
| Ministral-3-8B, untrained | 6 | 2 | 0 | 4 |
| Ministral-3-8B-Reasoning, reasoning on | 4 | 0 | 1 | 3 |
| Qwen3.6-27B, reasoning on | 10 | 5 | 0 | 5 |
| Qwen3.6-27B | 16 | 6 | 4 | 6 |
| legal-it-8b (published) | 14 | 5 | 5 | 4 |
| legal-it-8b, our system prompt | 16 | 6 | 6 | 4 |

No teacher measured corrects more than 6 false premises of 18, so
distilling one cannot teach it; free reasoning made both reasoning models
worse. On the practical exam the published 8B ties the best students once
"partial" counts as right (jp2-9b 19:18, p=1.0): its answers are half as
long (529 characters against 1,100-1,400), and the judge sides with the
longer answer in two disagreements out of three.

**Our system prompt costs nothing.** The published 8B asked with
`system-prompt.txt` instead of Ministral's "Le Chat" default: devbig 437 and
437 with the note (9:9), 437 and 440 without (0:3, p=0.25), practical 6:5,
LexIA 15 and 16 of 36. The GGUF can be repackaged with it; training uses it
from now on (`stage3_sft.py --default-system-file`).

**False premises, built without a teacher** (`make_premise_pairs.py`): a
wrong deadline, a nonexistent article, a repealed one taken for in force,
from the statute text, with no note, answered in checked steps with the
passage quoted. 1,060 questions; 530 for SFT, 530 for GRPO. The published
8B, 8 samples each (`sample_answers.py`, graded by program):

| Premise | Questions | Pass rate | Always wrong | Mixed | Always right |
|---|---|---|---|---|---|
| nonexistent article | 289 | 0.95 | 8 | 10 | 271 |
| repealed article | 158 | 0.31 | 48 | 101 | 9 |
| wrong deadline | 83 | 0.40 | 27 | 47 | 9 |

Two lessons from reading the answers: the first reward missed right
corrections ("il termine di 20 giorni non è previsto", "60 giorni, non 30")
and scored the deadline kind 0.17 instead of 0.40, so a reward is checked on
real answers before it trains anything; and some "wrong" answers deny an
article that is in the prompt ("non compare l'art. 645 c.p.p."), the
over-abstention the earlier GRPO rounds on nonexistent articles taught.
From the same samples: 158 questions with mixed answers for GRPO (the
rest carry no gradient), 158 DPO pairs and 249 correction exercises, the
model's own right and wrong answers (`make_correction_pairs.py`).

**Citations by program.** `cite_openbook_pairs.py` added the passage number
and a literal quotation to 16,116 of 16,127 grounded statute pairs
(`openbook-v06`), the two things the LexIA report found missing.

**What a long SFT on the published model undid.** The Cassazione SFT
(section 6, item 2: 75,352 case-law pairs and 20,572 statute pairs, 6,007
steps at lr 1e-4) started from the published `legal-it-8b`:

| Exam | Published | After the Cassazione SFT |
|---|---|---|
| devbig with the note | 437 | 415 (10:32, p=0.001) |
| devbig without the note | 440 | 413 (3:30, p<0.001; lenient p=0.002) |
| practical (correct + partial) | 60 + 21 | 57 + 9 (lenient 9:24, p=0.014) |
| LexIA traps without the note | 14 | 4 |

Every LexIA answer was the same sentence, word for word: the fixed answer
of the absent-article pairs ("Tra i testi normativi riportati non trovo la
disposizione…"). Fixed-wording refusals were about 14% of that SFT (1,600
statute pairs and some 11,000 `caselaw_absent`); without the note the model
took the refusal for the safe answer, by-topic deadline questions included
(0:9), since those are the shape of the absent pairs. Two rules follow:

1. a long SFT on a model GRPO has shaped undoes the GRPO; what is added on
   top is short, and GRPO follows it;
2. fixed-wording refusals stay a small share of any SFT mix (the round below
   keeps them near 6%), and a model is examined without the note before it
   is built on.

The round queued that evening, from the published model: a short SFT
(about 7,500 rows: 4,000 cited statute pairs, 2,000 case-law pairs, the
false premises in two formats, the correction exercises) in two arms that
differ only in the premise answers' format, three checked steps or one
paragraph; then GRPO on the mixed-answer questions; every model examined
on devbig with and without the note, the practical exam and the LexIA
traps.

## 9. 2026-10-21/22: what moved the traps, what saturated, and why

**Mistral Small 4 as a teacher.** Measured on JUPITER on the same 36 traps,
answers graded on Leonardo by the same judge (the answer files stay private):
10 of 36 at 400 tokens, 11 at 1,000, 12 with reasoning on, against 16 for
Qwen3.6-27B. It cites articles that are not among the texts, and the judge
marks most of those citations wrong (4 of 6 on the traps, 12 of 17 on the
practical exam with reasoning). Not a better teacher for the traps; its
practical-exam comparison waits for judgements cancelled by mistake on
10-20 and queued again.

**The rounds after r2b, on the traps without the note** (no system prompt,
so the reference is the published model's 14):

| Model | Right | Corrected (of 18) | Abstained (of 9) | Answered (of 9) |
|---|---|---|---|---|
| legal-it-8b (published) | 14 | 5 | 5 | 4 |
| r2b-steps + GRPO | 15 | 7 | 6 | 2 |
| r2d, + true-premise twins | 14 | 6 | 6 | 2 |
| r2e, + practical GRPO | 15 | 7 | 5 | 3 |
| r2f, + penal premises | 15 | 6 | 7 | 2 |
| jpn4 / jpn6 / jpn7 (Qwen3.5-4B) | 6 / 7 / 7 | 1-2 | 4-5 | 0-2 |

No round reaches the promotion rule (20 of 36); the gains are one question,
inside the noise of 36, and every 8B round answers one or two plain
questions fewer than the published model: the over-suspicion is still
there. The 4B line corrects almost no premise.

**Saturation.** The GRPO rounds on generated premises stopped by themselves
(20 steps in a row with every group all-equal: r2d, r2f at step 154, jpn4 at
202). Sampled 6 times, the 8B candidate is already at 0.87-1.00 on every
numbered premise kind; GRPO has nothing to learn from a question it always
gets right or always wrong.

**Two bugs, found by reading answers.** The procedibility reward scored 0
every right 8B answer ("non è procedibile d'ufficio, ma richiede la querela",
"la querela è necessaria"): `corrects()` knew no way of denying a
procedibility. Fixed (`corrects_procedibility`), the same samples rescored
without asking the model again: 0.05 becomes 0.69. And the generator took
art. 624 c.p. ("Si procede, tuttavia, d'ufficio") for complaint-only; articles
naming both procedibilities are now left out. The penalty premises, limited
to articles with a single range, gave 44 questions over the whole penal code;
taking the first-comma range of articles with several (asked "al primo
comma") gives 136.

**Which traps nobody solves.** Per trap, across seven models including the
27B teacher: the nonexistent-article traps and the nonexistent or future
rulings are solved by five to seven models; 11 of 36 by none. They are of
three kinds the numbered premises never ask:

1. a wrong penalty or measure, naming the offence or the institution and no
   article number;
2. a principle attributed to a ruling number that does not say it, which the
   statute texts cannot confirm or deny;
3. a question outside the areas the collection covers (tax, tenancy), where
   the right answer is to abstain.

So the practical ceiling with today's models is about 25 of 36, and reaching
20 means teaching these three. A related data gap: a repealed article keeps
no heading in the corpus ("Art. 323. ((ARTICOLO ABROGATO DALLA L. 9 AGOSTO
2024, N. 114))"), so a question naming "abuso d'ufficio" cannot retrieve it.

**The first of the three, built.** Half of the penalty and procedibility
premises now name the offence by the article's heading ("il reato di truffa
semplice"), when the heading names one offence of the code alone; the answer
still says which article it is (`--named-share`). Since an outside set may
name no article, the articles its questions retrieve are left out of the
generated pairs as well as the ones they name: the traps stay a test. They
are a test now looked at by category, so from here the held-out exam,
asked once, and devbig and the practical exam carry the decision.

**Gemma 4.** Apache 2.0 since 2026-03, Base models included, so admitted
(`forge/CLAUDE.md`). E4B, 12B, 26B-A4B (MoE, 4B active) and 31B, Google's
Q4_0 GGUFs, examined like the teachers on the traps and the practical exam:
E4B against the Qwen3.5-4B line, 26B-A4B as the ready-made sparse model, 31B
as a teacher.

**Queued on 10-22.** Four GRPO rounds on the practical prompts mixed with the
rescored mixed-answer premises (r2g and jpn8 from r2b and jpn4, r2h and jpn9
continuing r2e and jpn6); then r2i and jpn10, which sample r2g and jpn8 on the
by-name premises and train on what they still get wrong. Each round ends in
its own packaging and exams.

**Results of 10-22 (evening).** Against the published 8B (devbig 437 with
the note, 440 without; practical 60 correct + 21 partial) and, for the 4B,
against jpn4; strict counts, McNemar:

| Model | devbig with note | devbig without | Practical (correct + partial) | Traps |
|---|---|---|---|---|
| r2g (r2b + mix) | 443 (+6, p=0.36) | 442 (+2) | 79 + 1 (23:4, p<0.001) | 14 |
| **r2h (r2e + mix)** | 443 (+6, p=0.38) | 435 (-5, p=0.46) | **83 + 2** (29:6, p<0.001) | **16** (7 / 6 / 3) |
| r2i (r2g + by-name premises) | 436 (-1) | 431 (-9; lenient p=0.004) | 79 + 0 (25:6, p=0.001) | 16 (8 / 5 / 3) |
| jpn8 (jpn4 + mix), 4B | 421 (-1) | 440 (0) | 77 + 1 | 7 |
| **jpn9 (jpn6 + mix), 4B** | **441 (+19, p=0.004)** | 447 (+7; lenient p=0.031) | **82 + 1** | 7 |
| Qwen3-30B-A3B-Instruct (teacher) | | | 74 + 0 | |
| Qwen3.6-27B / reasoning on | | | 76 + 0 / 85 + 1 | 16 / 10 |
| Gemma 4 E4B / 12B / 26B-A4B / 31B | | | 71 / 70 / 68 / 74 (+1-2) | 7 / 8 / 11 / 13 |

What it says:

* **The practical GRPO works, for both sizes.** r2h answers 83 practical
  questions right against the published model's 60, more than either
  teacher without reasoning; the 4B jpn9 82. With "partial" counted as
  right the gap closes (the published model's 21 partials), and the judge
  prefers the longer answer, which ours are not (about 740 characters
  against 1,000-1,350), so the strict gain is the conservative reading.
* **No 8B reaches 20 traps.** r2h and r2i are at 16, two over the published
  model; r2i corrects 8 premises of 18, the most so far.
* **A cost to watch: nonexistent articles without the note.** On devbig
  without the note, r2i loses 7 nonexistent-article questions and wins none
  (r2h 0:5, r2g 0:2): the by-name premises and the practical rounds wear
  away the abstention that GRPO only rewards while some answers still get
  it wrong. Every candidate is read on this column before it is promoted.
* **Gemma 4 does not change the plan.** Not a better teacher (31B: 74
  practical, 13 traps, against 76 and 16 for Qwen3.6-27B) and, untrained,
  below our trained models of the same size; 26B-A4B stays the candidate
  for a sparse student, to be tried only with time to train it.
* **By-name premises are not harder.** Sampled on premise4, the 8B passes
  penalty questions 0.93 of the time and procedibility 0.59; 38 questions
  had mixed answers (87 for the 4B), so the premise rounds stop early.

Mistral Small 4 on the practical exam, graded without the "partial" grade
(regrading under rubric v3 is queued): 55-60 correct, against 74-76 for the
two teachers. Below them on both sets.

## 10. 2026-10-23: rulings in the premises, and a gate in front of the model

**Rulings attributed wrongly** (`premessa_sentenza`, `premessa_sentenza_vera`,
premise5): a deadline attributed to a made-up Cassazione ruling, rightly or
wrongly; the answer must say the ruling is not among the texts and check the
deadline on the statute. The first reward missed right answers ("la sentenza
Cass. civ. n. 4767/2025 non è tra i testi", cut at the full stops, and "non
posso confermarne"); fixed and rescored without sampling again, the 8B passes
the false attributions 0.16 of the time and the true ones 0.58, the 4B 0.13
and 0.10. Its usual failures are real: it checks the statute or flags the
ruling, rarely both. r2j trained on the faulty reward, r2k (from r2h too) on
the fixed one; jpn11 and jpn12 compare the 4B with and without the premises.

| Model | devbig with note | devbig without | Practical (correct + partial) | Traps (corr. / abst. / ans.) |
|---|---|---|---|---|
| published 8B | 437 | 440 | 60 + 21 | 14 (5 / 5 / 4) |
| r2h | 443 | 435 | 83 + 2 | 16 (7 / 6 / 3) |
| r2j, faulty ruling reward | **447** (+10, p=0.10) | 441 | 84 + 4 | 13 (7 / 3 / 3) |
| r2k, fixed reward | 445 (+8) | 439 | 84 + 1 (29:5, p<0.001) | 15 (6 / 6 / 3) |
| jpn9 (4B) | 441 | 447 | 82 + 1 | 7 |
| jpn11 (4B, premises + practical) | 445 (+4 on jpn9) | 448 | 83 + 2 | 7 |
| jpn12 (4B, practical only) | 434 (-7) | 446 | 80 + 1 | 7 |

* The fixed reward gives back the abstentions r2j lost (3 to 6) at the
  same practical score; the traps stay at 15-16 for every 8B round.
* For the 4B the premises help (jpn11 against jpn12) on devbig and the
  practical exam, not on the traps.
* Every 8B candidate loses the same 5 nonexistent-article questions of
  devbig without the note (0:5), wins none: the next thing to look at.
* Mistral Small 4 regraded under rubric v3: 55 / 59 / 60 correct, 0-1
  partial, the same as before; it stays below the teachers.

**The context gate** (`eullm_forge.datasets.context_gate`, `eval.gate`,
`submit_gate.sh`). The traps no model solves are mostly questions the texts
in front of the model cannot answer. A small model before the legal one,
Qwen3-Reranker-0.6B fine-tuned with its own instruction, reads the question
and the three retrieved texts and says whether they hold what the answer
needs. Its examples are labelled by construction: an open-book question
with its article among the texts, the same question with it taken out,
questions on nonexistent articles, and questions on matters the collection
does not cover, written by the teacher on 25 matters (tax, traffic,
pensions, immigration, insolvency, building, ...; matters the civil code
covers in part are left out). Development articles and matters are never
trained on. A gated question gets a fixed reply saying what is missing, or a
note to the model (`legal_eval.py --gate-mode`); r2h is examined through it
on the traps (both modes), devbig without the note and the practical exam,
the last two being where a question gated wrongly would lose its answer.
The traps' 20 of 36 stays the target.

**What the answers said, read on 10-23 evening** (`compare_graded.py --show`).

* *The five nonexistent articles every candidate "lost" on devbig without the
  note were not lost.* r2k answered "tra i testi riportati non c'è l'art.
  957 …, e non mi risulta che esista … verifica il numero": right under the
  rubric, which accepts "non compare nei testi" and a suggestion to check the
  reference. The judge failed it as implying the article might exist, while
  passing the published model's vaguer "potrebbe non esistere". The
  judge-free check, printed under every comparison, had them level (1:1).
  For nonexistent articles the judge-free check is the one to read; the
  exam's abstention pattern now knows the three-step phrasings.
* *The judge is close to deterministic.* On identical answers asked twice it
  changed its grade 0.6% of the time on devbig (2 of 341) and 3.3% on the
  practical exam (2 of 60): about 3 questions of devbig, 5 of the practical
  exam and 1 trap move on the judge alone. The +23 on the practical exam
  holds; a one- or two-trap difference does not.
* *Quotations cut short, learnt from our data.* Splitting at every full stop
  cut "a norma dell'art. 2855" after "art."; quote_span then dropped the
  stop. 400 of 16,116 quotations of openbook-v06 and 90 of 667 in the
  premise answers ended that way, and r2h and r2k quote cut short in 19 and
  16 of about 370 answers (the published model hardly quotes). Fixed in the
  splitter, and GRPO now pays nothing for such an answer (r2l, jpn14).
* *The first context gate failed on real questions.* 0.92 on its
  development set, it stopped 31 of 36 traps, 94 of 472 devbig questions and
  77 of 145 practical ones; with its fixed reply the traps fell from 16 to
  2, with the note instead they were 17. Trained on short one-article
  questions, it read a false premise or a case with several facts as
  unanswerable. Version 2 adds those kinds with their article as
  sufficient, caps the threshold at 3% of the sufficient questions gated,
  and only adds the note.
* *4B:* jpn13 (the fixed ruling reward) and jpn11 are level (devbig
  439-445 / 446-448, practical 82-83): the 4B recipe has plateaued.

## 11. 2026-10-24: r2l, the second gate, and the abstention moved into the model

| Model | devbig with note | devbig without | Practical | Quotations cut short | Traps |
|---|---|---|---|---|---|
| published 8B | 437 | 440 | 60 + 21 | 0 (it hardly quotes) | 14 |
| r2k | 445 | 439 | 84 + 1 | 16 / 472 | 15 |
| **r2l** (r2k + the cut-quotation penalty) | **449** (+12, p=0.050; lenient p=0.61) | 442 | **87 + 1** (30:3) | **8 / 472** | 14 |
| jpn13 / jpn14 (4B) | 439 / - | 446 / - | 82 / 82 | 0 / 0 | 9 / 10 |

* r2l is the best 8B on devbig and the practical exam; the penalty halved
  the quotations cut short. The traps stay within the 14-16 band, which
  the judge's own noise covers.
* **The second gate** (false premises and practical questions with their
  article as sufficient, threshold capped at 3% of the sufficient
  development examples): dev balanced accuracy 0.93, threshold 0.014. It
  now stops 27 of 145 practical questions (1:3 against no gate, not
  significant) and 81 of 472 devbig ones, but still 33 of 36 traps, and with
  its note the traps are 16, as without it. The traps are unlike anything it
  was trained on, and it reads them all as unanswerable: it does not tell
  them apart, so it is set aside.
* The judge graded the same answer on a nonexistent article right in one
  run and wrong in the other (arts. 1266, 1432, 1462 c.p.p.): on that kind
  only the judge-free check is read.
* **Next (r2m from r2l, jpn15 from jpn14):** the abstention on matters the
  collection does not cover goes into the model. 400 of the teacher's
  off-topic questions become GRPO prompts rewarded for abstaining
  (`make_offtopic_prompts.py`), in the mix with premise6 (whole quotations,
  offences by name, rulings) and the practical prompts, which keep the model
  from abstaining where the texts answer.

## 12. 2026-10-24/25: the abstention reward taught one sentence, and a lost start model

| Model | devbig with note | devbig without | Practical | Quotations cut short | Traps (corrected of 18) |
|---|---|---|---|---|---|
| **r2l** | **449** | 442 | **87** | 8 / 472 | **14** (6) |
| r2m (r2l + off-topic prompts) | 447 (5:7) | 442 (9:9) | 85 | 5 / 472 | 10 (3) |
| r2n (r2m, again) | 442 | 437 | 87 | - | 7 (4) |
| **jpn14** | - | - | 82 | 0 | **10** (2) |
| jpn15 (jpn14 + off-topic prompts) | 452 (5:5) | 428 (10:13) | 82 | 0 | 8 (2) |
| jpn16 (jpn15, again) | 452 | 412 (+42 partial) | 84 | - | 8 (2) |

**What the answers said** (`compare_graded.py --show`):

* **The three off-topic traps** are still failed by r2m, though it now
  abstains on all three. Their references name the matter and place it
  outside what the service covers; a bare "non trovo la disposizione" is
  graded wrong. Abstaining was never what was asked.
* **r2m lost five traps to r2l and won one.** On the nonexistent articles
  r2l wrote "non c'è l'art. …, e non mi risulta che esista: la domanda parte
  da un riferimento errato"; r2m wrote "non trovo l'art. …: potrebbe non
  esistere, oppure il riferimento potrebbe essere errato", which leaves the
  false premise standing, and the judge is right to fail it. It also
  abstained once where the article was among the texts.
* **The cause is ours.** `premessa_inesistente` was scored as any abstention,
  so nothing told the generic refusal the off-topic prompts rewarded from
  the premise correction, and the refusal spread. A grader checked in one
  direction only, again (see forge/CLAUDE.md). Fixed: a nonexistent premise
  is right only when the answer denies the article exists (`denies_existence`;
  "potrebbe non esistere" is not a denial).
* **jpn16's 16 lost points without the note are mostly the judge.** The
  judge-free line is level (1:1), lenient 2:4 (p=0.69); on art. 1182 c.p.
  jpn16 and jpn15 wrote the same answer word for word and the judge gave one
  "partial" and the other "correct". A small real loss: jpn16 sometimes no
  longer says the article may not exist.
* **premise6 is spent for the 8B:** r2m answers only 21 of its questions
  both right and wrong, so r2n trained almost only on practical and
  off-topic prompts.

**Decision.** The off-topic line is closed: r2m, r2n, jpn15 and jpn16 lose
traps and gain nothing elsewhere. The best models stay r2l and jpn14. Two
pairs of rounds start from them:

* **r2o / jpn17:** premise6 sampled again under the fixed reward (the
  hedged answers now score 0, so the questions are informative again), plus
  the practical prompts, no off-topic ones;
* **r2p / jpn18:** the same plus the off-topic prompts graded by the judge
  (`make_offtopic_prompts.py --kind fuori_ambito`): name the matter and say
  what would be needed, not only that the texts lack it.

**A lost start model.** The scratch cleanup of `submit_grpo_round.sh` kept
only the START of the round being submitted, so the 4B round removed r2l,
the START of the 8B round queued a minute before, and the merged weights of
its ancestors had gone the day before. Nothing was lost but time: each
adapter records the model it was trained on, and `remerge_round.py` walks
back to the first one on disk (the published 8B, through the r2b SFT whose
adapter lives in `premise/sft-r2b-steps`) and merges forward. The cleanup now
keeps the START of every round not yet packaged (`NAME/start`) and leaves
the symlink where the weights were.

**Open question:** the outside test's off-topic traps assume a service
limited to criminal law, while ours covers the civil code too: two of the
three matters are in our collection. Whether the exam should state the
service's scope in a system prompt, as a deployment would, is for the user
to decide.

**10-26, the four rounds of the night** (devbig without the note against the
round each started from):

| Model | devbig with note | without note | Practical | Traps (corrected/abstained/answered) |
|---|---|---|---|---|
| r2l | 449 | 442 | 87 | 14 (6/5/3) |
| r2o (fixed reward) | 448 (2:3) | 439 (4:7) | 83 (3:7, p=0.34) | 13 (6/5/2) |
| r2p (+ off-topic, judged) | 446 (6:9) | 437 (6:11) | 87 | 11 (6/4/1) |
| jpn14 | - | - | 82 | 10 (2/6/2) |
| **jpn17** (fixed reward) | 453 (+1) | **+19** (25:6, p=0.001; judge-free 0:1) | 83 | 9 (2/5/2) |
| jpn18 (+ off-topic, judged) | 453 (+1) | +13 (20:7) | 80 | 8 (1/5/2) |

* The fixed reward did what it was for: the 8B corrects 6 premises again
  (r2m 3), and premise6 is informative again (101 questions answered both
  ways by r2l, against 21 for r2m).
* **jpn17 is the new 4B.** Its gain without the note is on nonexistent
  articles (22:1): jpn14 wrote "non posso rispondere alla domanda basandomi
  sui documenti disponibili" (partial) or invented an article beside the one
  asked (wrong); jpn17 writes that the article was not found and may not
  exist. The judge-free line is level because both abstain; the difference
  is what the abstention says. Its four losses are content omissions and one
  bare refusal, each a single question.
* **r2o is r2l's equal, not better.** On the practical exam each wins
  questions the other loses (mediation, consumer withdrawal and l. 241/1990
  for r2o; probative copies, a minor's representation and concussion for
  r2l), errors of reading with no common shape. r2l stays the 8B.
* **The judged off-topic prompts did not reach the traps.** r2p answers the
  three off-topic traps in the three-step format, says the texts are not
  pertinent, and does not name the matter; on one it states that a civil
  code article in force was repealed. jpn18 gave the same bare refusal to
  all three and was graded right on two and wrong on the third: on these
  three questions the judge is a coin, and they are no signal to decide on.
* **Queued from here:** premise7 (new false premises, seed 7, test articles
  excluded), sampled by r2l and jpn17; r2q from r2l (premise7 x5, premise6
  x2, practical) and jpn19 from jpn17 (premise7 x5, practical), no off-topic
  prompts.

**The traps counted of 33 too** (from 10-26, decided with the user): beside
the official 36, which still decides a promotion, the count without the
three off-topic traps, whose judge is a coin and whose references assume a
service limited to criminal law (`forge/scripts/trap_summary.py`, counts
only). Of 33: r2l 14, r2o 13, r2p 11, r2m 10; jpn14 9, jpn17 8, jpn15 7,
jpn18 6. The 4B's off-topic "wins" were the coin.

**10-26 evening, r2q and jpn19 (premise7):**

| Model | devbig with note | without | Practical | Traps of 36 / of 33 (corrected/abstained/answered) |
|---|---|---|---|---|
| r2l | 449 | 442 | 87 | 14 / 14 (6/5/3) |
| r2q (r2l + premise7) | 445 (3:7, p=0.34) | 438 (4:8, p=0.39) | 86 (2:3) | 16 / 15 (7/5/3) |
| jpn17 | 453 | 450 | 83 | 9 / 8 (2/5/2) |
| jpn19 (jpn17 + premise7) | 452 (5:6) | 453 (7:4) | 79 (5:9, p=0.42) | 10 / 8 (2/7/1) |

* r2q's +2 on the traps is mostly the judge: of its three wins, two are the
  same answer r2l gave, graded the other way (the maintenance question and
  art. 625-bis c.p.p.). One win is real (a ruling not among the texts, which
  r2q says it cannot confirm and r2l discussed anyway), and so is its one
  loss: it declines to define "recidiva reiterata" with art. 99 c.p. among
  the texts. r2q is r2l's equal; jpn19 is jpn17's.
* **Both lines are on a plateau** with generated premises: every difference
  of the last four 8B rounds is inside the noise of every exam. The loss that
  recurs is the one section 9 found: a plain question answered by the texts
  and declined (r2q on "recidiva reiterata", r2m likewise).
* The large trap test built on JUPITER (`make_trap_test.py`) does not
  separate these models: it is drawn by the same generator they trained on,
  and its judge-free check gives 99.5% to models trained on it.

## 13. 2026-10-27: the held-out exam, what the models know with no text, and the outside platform

**The held-out exam** (233 items, asked once; totals only, as its rules
want):

| Model | with note | without | judge-free (deadlines, absent articles) |
|---|---|---|---|
| published 8B | 168 | 169 | 121 / 120 |
| **r2l** | **173** (+5, p=0.44) | **174** (+5, p=0.41) | 114 / 115 |
| r2q | 173 (+5) | 167 (−2) | 113 / 112 |
| jpn14 | 176 | 174 | 120 / 121 |
| **jpn17** | **178** (+2) | **178** (+4) | 121 / 122 |

r2l is the 8B to build on and jpn17 the 4B; the statute gain of the GRPO
rounds over the published 8B is small on the held-out exam (+5, not
significant), their real gain stays the practical exam (60 → 87). The 4B
equals the 8B on statutes. The judge-free column goes the other way from
the judge on deadlines (r2l 4:11 against the published model, the judge
6:3): the longer answers name more deadlines than the check allows, to be
read on devbig, where answers may be read. r2r and jpn20 (the questions by
concept name, only 308 of them) moved nothing, and the GRPO rounds on
Leonardo end here.

**Closed book** (`make_closed_book.py`, 716 items, scored by program):

| Model | right | sure-wrong | real article recognised | made-up denied | number ← heading | heading ← number | terms | penalties |
|---|---|---|---|---|---|---|---|---|
| jpn14 | 4% | 430 | 10% | 0% | 0% | 3% | 0% | 0% |
| jpn17 | 3% | 489 | 8% | 0% | 0% | 2% | 0% | 0% |
| published 8B | 7% | 373 | 4% | 7% | 5% | 8% | 18% | 8% |
| r2l | 30% | 360 | 7% | 59% | 3% | 5% | 13% | 17% |
| r2q | 31% | 401 | 0% | 91% | 2% | 7% | 9% | 17% |

With no text the models know almost nothing of the law, and say otherwise
with confidence. r2l's and r2q's 30% is mostly a reflex the premise rounds
taught: r2q denies 91% of the made-up articles and recognises none of the
real ones. Procedibility (r2l 88%) is inflated because the articles that
state it are mostly "a querela"; the scorer now prints that baseline and
how often a real article is denied. This is the starting point of the
legal-memory plan (`docs/research/legal-memory-plan.md`, E0).

**Five questions put to the outside platform and to our two models**
(same retrieval as the product, no note):

| Question | outside platform | r2l | jpn17 |
|---|---|---|---|
| art. 640-septies c.p. | corrected, answered on 640-ter | corrected, stopped | "may not exist" |
| art. 599-quater c.p.p. | corrected, answered on 599-bis | corrected, stopped | "not found" |
| art. 625-bis for an error of law | (not asked) | right | right |
| abuse of office, fact of 2024 | abolitio criminis, acquittal | "repealed", no consequence drawn | same |
| appeal term across August | 4 September (suspension) | confirms 4 August | "may not exist" |

The platform answered all four, each in four to seven minutes and about
one dollar, through some 30 calls to frontier models over a complete
archive. Two gaps of ours can be taught: after correcting a premise our
models stop ("se mi indichi l'argomento…"), which our own gold answers
taught; and they lack the law outside their texts (art. 2 c.p.'s
consequence, the August suspension of l. 742/1969, not in the index).
The first is now a premise kind of its own (`premessa_numero`: an offence
named with a number that does not exist; the answer corrects it, names the
article that provides for it and answers from it, and the reward pays only
all three); the second is the memory plan and the wider index.

## 14. 2026-11-05: Qwen3.8-Flash-Next on the closed-book suite

Same suite as section 13 (716 questions, no text, `score_closed_book.py`),
Flash-Next as the UD-Q4_K_XL GGUF through llama-server, beside every model
already asked:

| model | right | sure-wrong | denies-real | esist | falso | n<-rub | rub<-n | termi | pena | proc | abrog | sent? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flashnext-q4 | 229 (32%) | 434 | 18% | 24% | 90% | 22% | 18% | 21% | 33% | 4% | 9% | 18% |
| r2q | 220 (31%) | 401 | 45% | 0% | 91% | 2% | 7% | 9% | 17% | 60% | 22% | 100% |
| r2l | 215 (30%) | 360 | 25% | 7% | 59% | 3% | 5% | 13% | 17% | 88% | 45% | 100% |
| moe-i2507 (30B-A3B) | 166 (23%) | 511 | 36% | 9% | 98% | 2% | 6% | 2% | 25% | 16% | 1% | 10% |
| q36-27b | 149 (21%) | 515 | 5% | 14% | 72% | 10% | 8% | 10% | 8% | 0% | 4% | 10% |
| pub8b | 53 (7%) | 373 | 0% | 4% | 7% | 5% | 8% | 18% | 8% | 4% | 1% | 15% |
| jpn14 | 26 (4%) | 430 | 0% | 10% | 0% | 0% | 3% | 0% | 0% | 0% | 1% | 30% |
| jpn17 | 24 (3%) | 489 | 6% | 8% | 0% | 0% | 2% | 0% | 0% | 0% | 1% | 32% |

92% of the procedibility items want "querela", so a model answering it
every time scores 92% there.

**The totals hide what each model knows.** r2l and r2q reach 30-31% mostly
through three columns a reflex can fill: procedibility (88% and 60%, near
the "querela" baseline), made-up rulings (100%) and made-up articles; r2q
also denies 45% of the real articles. Flash-Next's 32% comes from the other
columns: it is first by a wide margin on everything that asks for the
content of an article from memory: the article's number from its heading
(22%, against at most 10% for every other model), the heading from the
number (18%), terms (21%), penalties (33%), and recognising that a real
article exists (24%). It recognises 90% of the made-up articles while
denying only 18% of the real ones, so its "falso" is discrimination, not a
reflex.

**Read with three limits.** It is a 180B model against 8-30B ones, so
size alone explains part of it; the GGUF is 4-bit; and only 355 of its 716
answers ended their turn within the 400 new tokens, so its score is a lower
bound. The run with 1000 tokens (`EXAM_ARGS="--max-new-tokens 1000"`) is the
one to quote: **247/716 (34%)**, sure-wrong 407, denies-real 17%; the
knowledge columns stay where they were (n<-rub 22%, rub<-n 21%, terms 23%,
penalties 29%) and made-up rulings rise to 42%, which a cut answer had not
reached.

**As an open-book teacher it is not better than the one we have.** On
the statute exam of the rounds (`norm-exam-v2`, 231 questions, judge, read
by counts only), 1000 tokens:
Flash-Next 167 right, Qwen3-30B-A3B-Instruct-2507 170; paired, 17 against
14, p = 0.72, no difference; on the part graded without the judge 111/145
against 105/145, p = 0.18. With the texts in hand the 30B reads them as
well as the 180B, at a sixth of the memory. What Flash-Next has more is
what it knows with no text, which is what line A is about.

**Why it matters for line A.** Number and heading of an article are exactly
the lexical, citation-shaped facts a hashed n-gram table can hold, and
Flash-Next is the one model here that has such a table
(`docs/research/flash-next-ngram.md`). That is a hint, not evidence, since
size is confounded with it; the evidence is the same suite with its n-gram
layer switched off (the first measurement listed in that document), and
A2-mini's gate on the 8B.

## 15. 2026-11-08: the gate of A2-mini — the n-gram memory loses to its LoRA control

Line A of the Fast Lane (`docs/cineca/fastlane-plan.md`), experiment E2-mini
of `docs/research/legal-memory-plan.md`. The frozen model plus a hashed
n-gram memory of 0.42 B parameters (`forge/scripts/ngram_train.py`), against
a LoRA of the same size on the same 67 M tokens of the memory corpus
(program rows and teacher rewrites of the articles), same order, 3 epochs.
The first LoRA (lr 1e-3) diverged to NaN and answered nothing; the one
below is lr 1e-4 with gradient clipping.

**Closed book** (716 questions, `score_closed_book.py`):

| model | right | denies-real | esist | falso | n<-rub | rub<-n | termi | pena | proc | sent? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r2l | 215 (30%) | 25% | 7% | 59% | 3% | 5% | 13% | 17% | 88% | 100% |
| r2l + memory, seed 0 | 127 (18%) | 9% | 11% | 25% | 6% | 21% | 19% | 42% | 40% | 38% |
| r2l + memory, seed 1 | 132 (18%) | 10% | 12% | 14% | 6% | 27% | 15% | 25% | 16% | 78% |
| r2l + memory after layer 0 | 143 (20%) | 2% | 15% | 28% | 5% | 17% | 18% | 38% | 12% | 70% |
| r2l + memory ×4 (1.64 B) | 143 (20%) | 11% | 9% | 41% | 4% | 21% | 20% | 50% | 0% | 15% |
| **r2l + LoRA control** | **259 (36%)** | 51% | 0% | 100% | **19%** | **48%** | **34%** | **67%** | 32% | 5% |
| Qwen3-30B-A3B | 166 (23%) | 36% | 9% | 98% | 2% | 6% | 2% | 25% | 16% | 10% |
| Qwen3-30B-A3B + memory | 136 (19%) | 17% | 12% | 71% | 2% | 9% | 11% | 12% | 0% | 8% |

**Open book** (`norm-exam-v2`, 231 questions, judge, counts only): r2l 159;
r2l + memory 149 (10:20, p = 0.10, no difference shown); **r2l + LoRA 102
(9:66, p < 0.001, worse)**; Qwen3-30B-A3B 170, + memory 129 (12:42,
p < 0.001, worse).

**The gate is not passed.** On every column that asks for the content of an
article the LoRA control learnt far more than the memory: heading from
number 48% against 21-27%, penalties 67% against 25-50%, terms 34% against
15-20%, number from heading 19% against 4-6%. A table four times larger or
read at layer 0 does not close the gap. On the MoE the memory adds almost
nothing and costs 41 open-book answers.

**What the memory does do** is keep the open book: r2l loses 10 answers
with it (not significant), 57 with the LoRA. The LoRA bought its knowledge
by forgetting how to answer from texts (and by a new reflex: it denies 51%
of real articles and calls every article asked about nonexistent), which is
what continued pretraining on raw text without replay does.

**What follows.** As the plan states, line A as an n-gram table stops here;
the 7,000 hours of A3 go elsewhere. The finding worth keeping is the
control's: **a LoRA trained on the rewritten articles gives an 8B more
statute knowledge with no text (259, 36%) than any model measured,
Flash-Next's 247 included**. The question it opens is E1 of the memory plan:
the same continued pretraining *with replay* of the open-book data, to keep
both.

## 16. 2026-11-10: continued pretraining with open-book replay keeps the open book

LoRA on the memory corpus (67 M tokens of the nine codes) mixed with
open-book replay (`render_pairs_text.py`: the 20,572 open-book-v06 pairs in
the model's own chat format, 28 M tokens, 29% of the mix), against the same
LoRA without replay. Closed book (716, by program), open book (`norm-exam-v2`,
231, judge, paired against the model the LoRA started from):

| model | closed | denies-real | esist | falso | n<-rub | rub<-n | termi | pena | sent? | open | paired |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| r2l | 215 | 25% | 7% | 59% | 3% | 5% | 13% | 17% | 100% | 159 | — |
| + LoRA, 3 ep, lr 1e-4 | 259 | 51% | 0% | 100% | 19% | 48% | 34% | 67% | 5% | 102 | 9:66, worse |
| + LoRA, 1 ep | 225 | 50% | 1% | 100% | 13% | 33% | 21% | 58% | 10% | 129 | 10:40, worse |
| + LoRA, 3 ep, lr 3e-5 | 167 | 13% | 14% | 27% | 15% | 53% | 26% | 58% | 10% | 150 | 10:19, p = 0.14 |
| **+ LoRA + replay, 3 ep, lr 1e-4** | 167 | 0% | 37% | 6% | **22%** | **62%** | 0% | **67%** | 0% | **156** | **17:20, p = 0.74** |
| + LoRA + replay, 1 ep | 184 | 6% | 44% | 13% | 10% | 48% | 5% | 75% | 75% | 143 | 14:30, worse |
| jpn17 (Qwen3.5-4B) | 24 | 6% | 8% | 0% | 0% | 2% | 0% | 0% | 32% | 154 | — |
| + LoRA, 1 ep | 192 | 15% | 0% | 79% | 7% | 43% | 15% | 50% | 8% | 129 | 8:33, worse |
| + LoRA + replay, 1 ep | 161 | 0% | 26% | 1% | 14% | 50% | 0% | 58% | 98% | 152 | 18:20, p = 0.87 |
| Qwen3.5-9B | 123 | 5% | 18% | 51% | 6% | 7% | 8% | 25% | 18% | 133 | — |
| + LoRA, 1 ep | 243 | 27% | 12% | 98% | 1% | 50% | 19% | 46% | 2% | (pending) | |
| + LoRA + replay, 1 ep | 131 | 0% | 21% | 0% | 13% | 45% | 21% | 71% | 5% | 142 | 35:26, p = 0.31 |
| Qwen3-30B-A3B | 166 | 36% | 9% | 98% | 2% | 6% | 2% | 25% | 10% | 170 | — |
| + LoRA (rank 16, experts too) + replay, 1 ep | 90 | 0% | 10% | 2% | 5% | 36% | 0% | 17% | 62% | 94 | 3:79, worse |


**The replay keeps the open book**: on r2l 156 against 159 (no difference),
on jpn17 152 against 154, on the 9B 142 against 133; without it every LoRA
loses 25-57 answers. **And the knowledge stays in**: on r2l the heading from
the number goes from 5% to 62%, penalties from 17% to 67%, the number from
the heading from 3% to 22%, real articles recognised from 7% to 37%, and
the "non esiste" reflex on real articles is gone (25% to 0%).

**Not yet a verdict, for two columns.** With replay the terms fall to 0%
and the made-up articles and rulings are almost never denied (6%, 0%). It
may be the open-book behaviour the replay teaches (pointing to texts that
were not given, which the closed book scores wrong), or a real loss of the
trap behaviour GRPO had taught, as JUPITER found for its E1-mini
(`docs/jupiter/dossier.md`, 06-11). `score_closed_book.py --texts` counts
the answers that point to texts not given, by kind, without reading any.

**The count answers half of it** (`--texts`, share of answers pointing to
statute texts not given, by kind):

| model | esist | falso | n<-rub | rub<-n | termi | pena | abrog | sent? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r2l | 74% | 68% | 41% | 1% | 10% | 4% | 16% | 0% |
| + LoRA, no replay | 1% | 1% | 0% | 0% | 1% | 0% | 0% | 0% |
| + LoRA + replay | 34% | 50% | 0% | 3% | **100%** | 25% | 85% | 0% |
| jpn17 + LoRA + replay | 50% | 80% | 3% | 13% | 95% | 17% | 100% | 98% |
| 9B + LoRA + replay | 66% | 99% | 36% | 16% | 51% | 4% | 100% | 2% |

* **The terms are not lost, they are not said**: every deadline question
  to r2l + replay is answered by asking for the text (100%), what the
  replay teaches and what the product wants. Whether the model knows the
  deadlines the closed book cannot tell; the open book, where it has the
  text, keeps its score.
* **The made-up articles**: half the answers point to the texts, which the
  closed book rightly does not count as denying the article exists.
* **The made-up rulings are a real loss**: r2l + replay points to texts in
  none of them and denies none, so it confirms them, the trap behaviour
  GRPO taught and continued pretraining wore away, as on JUPITER. The next
  step is the same as there: a GRPO round on top (`r2l-replay-g1`,
  2026-11-10).

**On the MoE the recipe fails** (rank 16 on the fused experts, 1 epoch):
94 open book against 170. Not pursued.

**The GRPO round on top (`r2l-replay-g1`, judge-reward prompts, 750 steps).**
Closed book 188/716: penalties 88% (r2l 17%), heading from number 62%,
deadlines answered again (12%); but made-up rulings denied 2% and made-up
articles 4% (r2l 100% and 59%): a judge-reward round does not bring the
traps back, the premise rounds did. Devbig (472, Q4, hybrid retrieval):
426 against r2l's 449 (6:29, worse), the loss mostly on deadlines (0:9).
Read, the nine deadline items: every number the new model states is in the
article, but it names **one** deadline of an article that has several, often
the first in the text (10 days for the declaration where the item wants the
5 days of the certificate, 24 hours for the notice where it wants the
judge's 20 days, art. 396's term where art. 397 has its own two days); r2l
names more of them. Three of the nine give the right number under the
wrong condition. Not a loss of knowledge, a narrower answer: the next round
pays every deadline the article states.

## 17. 2026-11-12: Flash-Next with its n-gram tables switched off

The question line A could not answer on our own models: in a model trained
with hashed n-gram tables from the start, what do they hold of Italian law?
Qwen3.8-Flash-Next (UD-Q4_K_XL GGUF, 1000 new tokens, the closed book), as
it is and with its one PLE layer switched off (`blk.1.ple_value.weight`
zeroed, `gguf_zero_tensors.py`: the layer then adds exactly nothing):

| | right | sure-wrong | denies-real | esist | falso | n<-rub | rub<-n | termi | pena | proc | abrog | sent? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tables on | 247 (34%) | 407 | 17% | 24% | 91% | 22% | 21% | 23% | 29% | 0% | 12% | 42% |
| tables off | 227 (32%) | 432 | 24% | 16% | 93% | 19% | 18% | 19% | 21% | 8% | 14% | 28% |
| r2l, for scale | 215 (30%) | 360 | 25% | 7% | 59% | 3% | 5% | 13% | 17% | 88% | 45% | 100% |

**The tables hold some of it, not most.** Switched off, every knowledge
column falls (real articles recognised 24% to 16%, penalties 29% to 21%,
number from heading 22% to 19%, heading from number 21% to 18%, terms 23%
to 19%) and real articles are denied more (17% to 24%): 20 answers of 716
in all. Without them Flash-Next still knows several times what any 8-30B
model does (heading from number 18% against at most 10%), so its lead
comes mostly from the rest of a 180B model. It agrees with A2-mini on
r2l: an n-gram table adds a little knowledge; a LoRA of the same size
adds more. (A per-kind difference of three points is two or three items;
the paired count is what says how much of the 20 is noise.)

## 18. 2026-11-12: the premise round does not bring the traps back; double replay loses the open book

* **`r2l-replay-g2`** (GRPO from `r2l-replay-g1` on r2q's prompts:
  premise7 ×5, premise6 ×2, practical): closed book 166/716, made-up
  rulings denied 0% and made-up articles 7% (r2l 100% and 59%), penalties
  92%; devbig 8:28 against r2l (worse, p < 0.001), 8:32 without the note.
  The prompts were filtered for r2l (questions it answered both ways), not
  for this model, and on the closed book the trap is asked with no text,
  which the premise rounds never showed. r2l stays the released 8B.
* **Double replay** (`lora-replay2x`, the open-book pairs twice, 45% of the
  tokens, 3 epochs): open book 7:95 against r2l, worse than no replay at
  all; to be read in its answers before anything is concluded from it.

**Diagnosed (`diagnose_losses.py`, 2026-11-12).**
* Double replay: 141 of 231 answers abstain against r2l's 38, and 79 of
  its 95 losses are abstentions with the text in hand (48 on content, 31 on
  deadlines). The open-book pairs include the abstention examples (the
  article left out of the texts); read twice for three epochs, "non è nei
  testi" became the default answer. Not a loss of reading: an over-learnt
  abstention, which is what the replay share has to be weighed against.
* `g2` on devbig: 28 losses, of which 15 untagged ("altro", 11 on content),
  6 shorter, 5 another deadline of the article, 5 no deadline, 2
  abstentions. The content losses are to be read (`$WORK/eval/g2-losses.txt`).

## 19. 2026-11-12: Flash-Next as the teacher of cases, a pilot

`law_rewrites.py --styles caso,spiegazione --codes codice_civile,codice_penale
--limit 600`, Flash-Next UD-Q4_K_XL served over three GPUs (`--gpus 0+1+2`),
24 minutes. The work list goes style by style, so the 600 were all cases.
Five read against their article:

* art. 219 c.c. (proof of ownership between spouses) and art. 720 c.c.
  (indivisible property in an estate): right;
* art. 415 c.c. (who may be inabilitato): right on the article, silent on
  the amministrazione di sostegno that has replaced it in practice;
* art. 185 c.c.: facts confused (the spouses are Tizio and Caio, the
  property Sempronio's, Tizio wants to sell it), the conclusion right;
* art. 114 c.c. (marriage of the Re Imperatore): applied as living law to a
  member of the "Famiglia Reale", an article without effect since 1946.

Also: Tizio and Caio, the textbook names, are made spouses in three of five,
a marriage Italian law does not have. Two of five with a fault a lawyer
would mark, against three of six for the 30B in October: not the step that
would justify Flash-Next as a teacher with its licence open (forge/CLAUDE.md).
The cases stay out of the default; two corpus fixes follow from the read:
the request should name spouses as Tizio and Mevia, and the articles of the
monarchy (a code's dead letter) should not be rewritten at all.

## 20. 2026-11-13: half a knowledge adapter keeps the open book

The LoRA trained on the memory corpus without replay (`a2-r2l-lora2`:
closed book 259, open book 102) merged into r2l at a share of itself
(`legal_eval.py --adapter-scale`, task-vector scaling):

| share | closed | denies-real | falso | n<-rub | rub<-n | termi | pena | sent? | open book vs r2l |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 (r2l) | 215 | 25% | 59% | 3% | 5% | 13% | 17% | 100% | 159 |
| 0.25 | 186 | 44% | 93% | 5% | 18% | 19% | 25% | 0% | 12:14, p = 0.85 |
| **0.5** | **243** | 31% | 75% | **18%** | **39%** | **25%** | **62%** | 22% | **8:19, p = 0.052** |
| 0.75 | 255 | 39% | 87% | 24% | 50% | 31% | 75% | 2% | 8:39, worse |
| 1 | 259 | 51% | 100% | 19% | 48% | 34% | 67% | 5% | 9:66, worse |

At half the adapter r2l keeps most of what the whole one taught (heading
from number 39% of 48, penalties 62% of 67, terms 25% of 34) and loses 11
open-book answers, short of significance on this exam: no training run of
the month did better on both at once, and it costs a merge. Still to read
before it decides anything: the product's exams (devbig with hybrid
retrieval, practical, traps) at 0.4-0.6, and which layers carry the
knowledge and which the loss (`--adapter-layers`).

Rank 256 with replay (`lora-replay-r256`): closed 92, open 5:109, every
answer abstaining or near it; the over-learnt abstention of the double
replay, larger.

**The share and the layers (2026-11-14).** Same adapter, open book paired
against r2l (159):

| | closed | n<-rub | rub<-n | termi | pena | open book vs r2l |
|---|---|---:|---:|---:|---:|---|
| share 0.4 | 245 | 11% | 29% | 24% | 50% | 7:17, p = 0.064 |
| share 0.5 | 243 | 18% | 39% | 25% | 62% | 8:19, p = 0.052 |
| share 0.6 | 259 | 22% | 45% | 30% | 71% | 7:20, worse (p = 0.019) |
| whole, layers 0-11 only | 196 | 23% | 13% | 13% | 25% | 11:20, p = 0.15 |
| whole, layers 12-23 only | 199 | 2% | 10% | 2% | 12% | 3:39, worse |
| whole, layers 24-35 only | 162 | 9% | 7% | 7% | 12% | 6:19, worse (p = 0.015) |

* **The middle third does the harm and carries almost nothing**: alone it
  costs 36 open-book answers and teaches 2-12%. The first third carries
  the number from the heading (23%, more than the whole adapter's 19%) at
  no significant cost; the heading from the number and the penalties need
  the layers together (48% and 67% whole, 13% and 25% in any one third).
* The 0.5 merge holds on the Qwen3.5 students too: the 9B from 123 to 208
  on the closed book and **better** open book than its base (36:16,
  p = 0.008); jpn17 129 closed, open book level (13:19, p = 0.38).
* Next: the adapter without its middle third (layers 0-11 and 24-35), whole
  and at 0.6; and the product's exams for the two 0.5 merges that hold,
  r2l and the 9B (`scale_adapter.py`, then `submit_grpo_round.sh` LINKS=0).

## 21. 2026-11-14: without the note, half a knowledge adapter reads from memory

The two 0.5 merges of section 20, packaged (`scale_adapter.py`, GGUF
Q4_K_M) and put to the product's statute exam (devbig, hybrid retrieval,
472 questions), with and without the note on a missing article:

| model | with the note | without the note | abstaining (no note) | mean chars (no note) |
|---|---:|---:|---:|---:|
| r2l | 449 | 442 | 83 | 694 |
| r2l + 0.5 knowledge (`r2l-k50`) | 438 (10:21, p = 0.071) | **373 (11:80, worse)** | 28 | 426 |
| best 9B so far (`jp2-q35-9b`) | 447 | 443 | | |
| 9B + 0.5 knowledge (`q35-9b-k50`) | 428 | **376 (16:82 vs r2l, worse)** | 53 | 592 |

With the note r2l-k50 loses nothing proven; without it, 69 answers. The
losses of r2l-k50 without the note (`diagnose_losses.py`, 80): shorter than
r2l's 46, an article neither the question nor the reference names 43, of
which 41 on the questions about articles that do not exist. Read (eight
losses): where r2l quotes the text it was given ("Il testo [1] dice: «…»"),
r2l-k50 recites the article from memory, without the quote and with
mistakes: art. 616 c.p.c. with the paragraph repealed in 2009, art. 446
c.p. said to be repealed, art. 320 c.p. with penalties it does not have.
Asked about an article that does not exist, it describes one instead of
saying the texts do not have it (28 abstentions against 83).

So the knowledge put in the weights does not stay beside the reading: once
nothing in the prompt says the article is missing, it replaces it, and
brings back memory older than the texts. Section 20's open book ran with
the note, which is why it saw no harm; its conclusion ("half an adapter
keeps the open book") holds only with the note, and the product asks
without it. LexIA's traps, also without the note, said the same (9 against
14). The LoRA's knowledge is real (closed book 243 against 215); what it
costs is the habit of reading, and the product needs that habit more.

Next: the share map again on devbig **without** the note, on the bf16
weights (0, 0.25, 0.5), to see whether any share keeps the reading; the
GRPO rounds on false premises over both merges (`r2l-k50-g1`,
`q35-9b-k50-g1`), which reward exactly what was lost (saying an article
does not exist); and no knowledge adapter is merged into a candidate
before devbig without the note has been read.

## 22. 2026-11-15: GRPO gives the reading back; the corpus is checked, and part of it is wrong

**The premise rounds over the 0.5 merges** (`r2l-k50-g1`, `q35-9b-k50-g1`:
750 steps of GRPO on false-premise prompts, judge as reward), against r2l:

| | devbig, note | devbig, no note | LexIA traps (36) | practical (145) |
|---|---|---|---:|---|
| r2l | 449 | 442 | 14 | 87 (+1 partial) |
| r2l-k50 | 10:21 | 11:80, worse | 9 | 68, 8:27, worse |
| **r2l-k50-g1** | 16:12, +4 | **22:13, +9** | 10 | 72 (+12 partial), 8:23 (p = 0.011; lenient 12:16, p = 0.57) |
| q35-9b-k50 | 11:32, worse | 16:82, worse | 11 | 64, worse |
| q35-9b-k50-g1 | 11:23 (p = 0.058) | 19:30 (p = 0.15) | 16 | 75, 11:22 (p = 0.058) |

The habit of reading the texts that the knowledge took away (section 21)
comes back with 750 steps on false premises: without the note r2l-k50-g1
is level with r2l, from 69 answers behind. Not yet a candidate: the
practical exam is still behind on the strict count (its answers are
shorter: 534 characters against 765), the traps did not recover (10 against
14, not significant either way), and whether the knowledge survived the
round is not measured yet (its closed book).

**The share without the note** (bf16, devbig, hybrid retrieval, against
the same weights at share 0): 0.25 is level (440 against 443, 14:17), 0.5
is not (382, 14:75). On the closed book 0.25 gained nothing and 0.4 most
(section 20), so the shares 0.3-0.4 and the first third of the layers
decide whether a merge alone can keep both.

**The corpus, checked** (`verify_rewrites.py`, 60 rows a form, 715 rows,
1,573 copies with a planted error):

| checker | planted errors found | modal | negation | number | party | corpus rows flagged |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next | 96% | 86% | 97% | 98% | 100% | 34% |
| Gemma 4 31B | 93% | 78% | 95% | 98% | 98% | 31% |
| GPT-OSS-120B | 90% | 72% | 92% | 98% | 93% | 32% |
| Qwen3-30B-A3B | 78% | 31% | 84% | 97% | 85% | 12% |

The article's own text is flagged 0-1 times in 60 and the heading
statements 0-4: the checkers do not flag at random. 57 rows of 715 (8%) are
flagged by all four. Sixteen of them read, by where the mistake comes from:

* **our program, 6**: a comma or a numbered item repealed taken for the
  whole article (art. 600 and 617-quater c.p. "non più in vigore"); the
  only penalty range of an article taken for its base penalty when it is the
  aggravated one (art. 507 c.p., "da 2 a 6 anni" for "fino a tre anni") or
  another's (art. 600-septies, confiscation); the procedibility of offences
  named in general taken for the article's (art. 158 c.p., the
  prescription; art. 282-bis c.p.p., a precautionary measure);
* **the teacher, wrong in law, 6**: a case that gives the treasure away
  against the code (art. 988 c.c.), conditions that must all hold listed
  as alternatives (art. 1948 c.c.), the State's law said to govern the
  State where art. 114 Cost. says Rome, a two-thirds majority the
  Constitution does not have (art. 78), a restriction added to a comma
  (art. 78 cod. consumo), three commi where there are two (art. 2481-ter);
* **added, not wrong, 2**: explanations that give purposes and effects the
  text does not state (art. 2199 c.c., 417 c.p.);
* **by design, 2**: questions whose answer is only the article's number
  (`domande`).

The program's mistakes are fixed at their source (`premise_gen`:
`_repealed_sentence` leaves a repealed part alone, `penalty_of` takes only
a range in the first sentence beside the word that punishes,
`procedibility_of` skips offences named in general, `defines_offence` gates
both), which the closed-book suite and the premise prompts share: **the
closed-book suite in use was written with the same mistakes**, so its
penalty, procedibility and repeal columns partly measured agreement with
them, and is to be written again before those columns decide anything. The
teacher's mistakes are where section 13 placed them, in the styles that
elaborate (case, explanation); `recheck_rewrites.py` applies today's
`check_rewrite` to the rewrites already kept. Rebuilt with the fixes (`memory-v2`,
2026-11-15), the program's rows changed: repeals 539 to 375 (164 articles
in force had been called repealed), penalties 252 to 219, procedibility 55
to 46, headings 3,865 to 3,988 and deadlines 523 to 551 (the articles
wrongly repealed now get theirs); `struttura` rewrites 3,167 to 2,728 (436
had the commi counted with the heading); the other 245,000 rewrites pass the
program's check as before, which does not see an error of law. Checker
speed on the sample: Gemma 4 31B 1,900 rows an hour on one GPU, GPT-OSS-120B
1,470 on three, Qwen3-30B-A3B 6,900 on one; the whole corpus (239,836 rows)
is about 130 GPU-hours for Gemma and 490 for GPT-OSS. Next: the corpus rebuilt
(`memory-v2`), the rows every checker flags left out, and the closed-book
of the two g1 rounds.

**The closed book of the g1 rounds** (2026-11-15), without the 42 items the
old functions drew wrongly (`score_closed_book.py --norms`: 32 repeals, 9
procedibility, 1 penalty; 674 left):

| | right | sure-wrong | denies real | made-up articles | n<-rub | rub<-n | terms | made-up rulings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r2l | 196 | 343 | 25% | 59% | 3% | 5% | 13% | 100% |
| r2l, half the knowledge | 234 | 386 | 31% | 75% | 18% | 39% | 25% | 22% |
| **r2l-k50-g1** | **271** | 376 | **43%** | 94% | 16% | 41% | 22% | **8%** |
| q35-9b-k50-g1 | 203 | 439 | 48% | 100% | 0% | 21% | 13% | 5% |

The knowledge survives the round, and the round adds the false premises it
trains. Two things it costs: real articles denied more often (43% against
25%), and made-up rulings: asked about one with no texts, r2l says it cannot
verify it every time, the merge in 22%, g1 in 8%, the others describing a
ruling that does not exist. The knowledge adapter broke it and the premise
round did not mend it; it is the likeliest reason the traps did not come
back. (The procedibility column now holds "querela" items only.)

**Its practical losses** (23, `diagnose_losses.py`): other 15, shorter 5,
an article not given 3. Read (eight): every answer of g1 states the rule
without the text it was given, where r2l cites it ("[1]") and quotes it
("Il testo [1] dice: «…»"); answering from memory it takes one article where
the question needs two (artt. 1743-1744, 2081-2082, 110-111 cod. consumo),
cites a neighbour (art. 134 for 136 c.c.) or mixes two (artt. 379 and 648,
386 and 387 c.p.). The quoting r2l learnt is what the knowledge took and
the premise round, which rewards the verdict and not the quote, did not give
back. Next: distil r2l's way of answering back into g1 on open-book prompts
only (`opd_train.py`, r2l as the teacher of the same vocabulary), which
leaves the closed book, where the knowledge is, out of what is taught.

## 23. 2026-11-16: r2l teaches its way of answering back; the share's edge is 0.35

**On-policy distillation from r2l** (`r2l-k50-g1-opd`: student r2l-k50-g1,
teacher r2l, 300 steps on the 2,948 open-book prompts of the premise round,
nothing closed-book), against r2l:

| | devbig, note | devbig, no note | practical | LexIA traps | closed book (674) |
|---|---|---|---|---:|---:|
| r2l | 449 | 442 | 87 (765 chars) | 14 | 196 |
| r2l-k50-g1 | 16:12 | 22:13 | 72, 8:23 (p = 0.011) | 10 | 271 |
| **r2l-k50-g1-opd** | 10:10 | 12:10 | **81**, 4:10 (p = 0.18; 811 chars) | 11 | 227 |

The practical loss closes (its answers are as long as r2l's again) and
nothing of the product is significantly behind r2l; the closed book keeps a
part of the knowledge (heading from number 32%, terms 20%, penalty 30%,
against r2l's 5%, 13%, 17%) and loses the rest (number from heading 3%,
from 16%). Its real-article existence column is 0% and made-up rulings 0%:
to be read (`--texts`) before it means anything, since a model taught to
answer from the texts given may answer "not among the texts" with none.

**The share without the note** (bf16, devbig, against share 0, 443):
0.25 14:17, 0.3 16:16, 0.35 19:24 (p = 0.54), **0.4 16:47, worse**, 0.5
14:75, the first third of the layers alone 14:55, worse. The reading holds
up to 0.35 and breaks by 0.4; the layer split that looked harmless with the
note is not. Next: the closed book at 0.3 and 0.35, and r2l + 0.35 packaged
for the product's exams: a merge alone, no training after it.

**r2l + 0.35, packaged (Q4_K_M), 2026-11-17**: devbig with the note 11:12,
**without it 15:56, worse** (bf16 at the same share: 19:24); practical
5:14 (p = 0.064), traps 4:7. Closed book: 0.3 226, 0.35 243, 0.4 233 (r2l
196). The share that holds in bf16 does not hold once quantized: 0.35 is
the edge, and the quantization's noise is enough to cross it. A merge alone
has no margin; the candidate stays the distilled one (`r2l-k50-g1-opd`).

**The verified corpus, a preview** (116,456 rewrites checked by Gemma 4
31B and GPT-OSS-120B, 2026-11-17): flagged by both 19%; paraphrase 4%,
elements 11%, questions 14%, case 42%, explanation 77%.

## 24. 2026-11-18: the teacher, not the style — Gemma 4 31B writes cases the checkers pass

The same 200 articles of each style rewritten by bigger teachers
(`teacher_pilot.py`), each judged by the two checkers that are not itself,
beside Qwen3-30B-A3B's rows on the same articles under the same judges:

| teacher | cases flagged | Qwen3-30B, same articles | explanations flagged | Qwen3-30B, same articles |
|---|---:|---:|---:|---:|
| **Gemma 4 31B** (Apache 2.0, 1 GPU) | **14/196, 7%** | 80/169, 47% | **11/190, 6%** | 127/180, 71% |
| Qwen3.8-Flash-Next | 32/161, 20% | 70/169, 41% | 57/191, 30% | 141/180, 78% |
| GPT-OSS-120B | 92/138, 67% | 110/183, 60% | 80/130, 62% | 152/185, 82% |

The cases and explanations were wrong because of who wrote them: a model
with 3B active parameters applying rules to facts. Gemma's are near the
checkers' floor (the article's own text is flagged 0-2%), and three read by
hand were right in law, if close to the text. GPT-OSS lost many answers to
its reasoning (138 and 130 of 200 kept). Flash-Next is not needed, so its
licence question does not arise for the corpus. Next: Gemma rewrites every
case and explanation (`memory-gemma`, eight 1-GPU chains), checked after by
Flash-Next and GPT-OSS; Qwen3.8-27B and Mistral Small 4 join the table when
their rows are checked.

## 25. 2026-11-18: how much distillation, from where

r2l as the teacher (`opd_train.py`, the 2,948 open-book prompts), from
four starts, against r2l (devbig 449 / 442 without the note, practical
87, traps 14, closed book 196 of 674):

| model | devbig, note | devbig, no note | practical | traps | closed book | rub<-n | terms | penalty |
|---|---|---|---|---|---:|---:|---:|---:|
| r2l + whole knowledge (`r2l-k100`) | 5:113, worse | 8:130, worse | 6:59, worse | 4:11 | | | | |
| `r2l-k100-opd` (300 steps) | 9:27, worse | 14:25 | 4:24, worse | 2:7 | **322** | 38% | 31% | 61% |
| `r2l-k50-opd` (300, no GRPO) | 8:9 | 16:9 | 5:12 | 1:5 | 268 | 39% | 20% | 26% |
| `r2l-k50-g1-opd100` | 7:48, worse | 7:42, worse | 5:12 | 5:4 | 286 | 40% | 24% | 30% |
| **`r2l-k50-g1-opd200`** | 4:14 (p = 0.031; lenient 0.077) | 7:9 | 7:11 | 5:5 | **299** | 38% | 27% | 39% |
| `r2l-k50-g1-opd` (300) | 10:10 | 12:10 | 4:10 | 1:4 | 227 | 32% | 20% | 30% |

* The premise round is not needed: from the 0.5 merge directly, the
  distillation reaches r2l's level on every product exam (268 closed book).
* 200 steps from g1 keep more knowledge (299) than 300 (227) at the
  product's level, but 100 steps are worse on devbig than either: the
  steps are not a dial, and the runs vary more than a single run shows. A
  repeat with another seed is the next measure, before a step count is
  chosen.
* From the whole adapter the distillation does not bring the reading back
  in 300 steps (devbig and practical still behind), and keeps the most
  knowledge of all (322).

## 26. 2026-11-19: the second seed, the last teachers, the verified corpus

**Teachers, completed.** Judged by all three checkers (they are not checkers
themselves), against Qwen3-30B on the same articles under the same three:
Qwen3.8-27B cases 21/117 (18%) against 40%, explanations 35/190 (18%)
against 70%; Mistral Small 4 cases 106/176 (60%) against 40%, explanations
129/183 (70%) against 70%. Gemma 4 31B (7% and 6% under two judges, against
47% and 71%) stays the teacher; Mistral Small 4 is no better than the old one.

**Gemma's rewrite of the corpus**, done overnight: 6,357 cases and 6,943
explanations (eight 1-GPU chains), 13,300 checked by Flash-Next and
GPT-OSS-120B.

**The verified corpus** (`memory-v3`, the old corpus without the rows both
Gemma and GPT-OSS flag): 217,902 rows of 253,897; the strict variant without
cases and explanations (`memory-v3s`) 202,570. The knowledge LoRA on it
(`k3s-r2l`, the recipe of `a2-r2l-lora2`) is training.

**The distillation, a second seed** (`r2l-k50-g1-opd200s1`, seed 1): devbig
7:7 and 12:13 without the note, practical 5:13 (p = 0.096), traps 1:3,
closed book 274 (seed 0: 299). 200 steps from g1 hold r2l's level on the
product in both seeds, with 274-299 closed book against r2l's 196: the
candidate. 600 steps from the whole adapter: practical worse (7:26), closed
book 258, below 300 steps (322); more steps take knowledge away without
giving the reading back. The 0.75 merge, alone and distilled, is worse than
both neighbours (distilled: devbig 6:76, closed book 182), an outlier to be
read before it is believed.

## 27. 2026-11-19: the same teacher through EULLM Engine 0.9.0 and llama-server

Gemma 4 31B Q8_0 rewriting the same 200 cases (`law_rewrites.py`, 1 A100, 8
requests in flight, temperature 0), once through llama-server and once
through the engine (`--engine`, the published 0.9.0 data-center build), both
on the same llama.cpp. Four things stood in the way before the two compared,
all of them the engine's, all handed to the engine's own branch:

1. **A projector beside the weights makes the engine sequential.** Gemma's
   folder holds its `mmproj`, the engine loads it on its own
   (`mmproj_beside`) and forces `batch_size = 0` for every request, text
   included: 20 answers in 25 minutes against llama-server's 198 in 6.5. The
   weights hard-linked into a folder without it: the scheduler, 8 slots.
2. **Its KV cache for 8 x 6,144 tokens did not fit** (38,400 MiB on top of
   30.4 GiB of weights) where llama-server, same context, did: the
   sliding-window layers seem to get a full cache. `--fit` had said the
   model fits. 8 x 3,072 tokens did.
3. **Thinking stays on in `/v1/chat/completions`** unless the request says
   Ollama's `think: false`; `chat_template_kwargs.enable_thinking` is not
   read, and Gemma 4's `<|channel>thought` block came back inside `content`
   (13 of 13, 3,712 characters against 650).
4. Its `repeat_penalty` defaults to 1.1 (Ollama's), llama-server's to 1.0:
   now stated in every request.

With the projector away, a context that fits and `think: false`: 198 answers
each, the same mean length (673 and 671 characters), 71 identical word for
word; of the 127 that differ the median first difference is at character
206 (9 before character 20), the arithmetic of a different batch and not a
different prompt. For text work the engine answers as llama-server does once
told the same things.

**Speed**: the engine 198 in 4.2 minutes (47 a minute, 237 prompt and 128
generated tokens a second), llama-server 198 in 6.4 (31 a minute, 156 and
84): half as fast again, the same two refused (a deadline, a length). Not
all of it is the engine's own: its 0.9.0 carries llama.cpp 1.0, whose
few-token kernel speeds up several requests at once on Ampere and later,
while the llama-server of these jobs was built from a checkout of
2026-11-03; and its context per slot was 3,072 tokens against 6,144. To
know what is whose, llama-server again at llama.cpp 1.0 and 3,072 tokens.

**The same pin, 2026-11-20.** The official llama-server built from the
engine's own llama.cpp commit (`b1b770d9`, none of our patches), 3,072
tokens a slot, the same 200 cases: 198 in 5.5 minutes, 36 a minute, 182
prompt and 99 generated tokens a second. The engine 0.9.0 again (a second
run): 198 in 4.1, 48 a minute, 243 and 131. So of the half again, a sixth
was the newer llama.cpp (31 to 36 a minute) and the rest is the engine's:
**a third faster than llama-server on the same llama.cpp**, prompt and
generation alike. The answers: llama-server of 11-03 and of `b1b770d9` agree
word for word on 71 of 198, the engine's two runs on 67 and 57 with the
latter, the same mean length (669-673 characters), no thinking in any. The
engine stands as close to llama-server as two llama-servers to each other.
Next: the engine from main (projector on the scheduler, sliding-window KV
sized to its window), on Gemma's own folder, projector included.

**The engine from main, built beside llama-server, 2026-11-20.** Built on
Leonardo by `build_engine.sh` (10 minutes on one A100) with the same gcc
12.2.0 and CUDA 12.2 as the vanilla llama-server, so nothing but the
program differs. Gemma 4 31B Q8_0, the same 200 cases, 8 in flight:

| run | answers | a minute | prompt tok/s | generated tok/s | identical to llama-server |
|---|---:|---:|---:|---:|---:|
| llama-server `b1b770d9`, 3,072 a slot | 198 | 36 | 182 | 99 | - |
| engine 0.9.0 release (CUDA 12.4), 3,072 | 198 | 48 | 243 | 131 | 57 |
| engine main, `--no-mmproj`, 3,072 | 198 | 45 | 228 | 123 | 57 |
| engine main, projector loaded, 3,072 | 198 | 45 | 225 | 121 | 60 |
| engine main, `--no-mmproj`, 6,144 | 198 | 45 | 227 | 122 | 63 |

- **Same compiler, same CUDA: the engine a quarter faster** (45 against 36
  a minute, prompt and generation alike). The release binary's 48 is three
  more, from its build (CUDA 12.4, gcc-toolset 13) or from noise: the two
  runs of 0.9.0 read 47 and 48.
- **A model with its projector is served in batches** (bb6b8027): Gemma's
  own folder, mmproj on the GPU beside the text model, 198 in about 4.5
  minutes, as fast as text only. Before: 20 answers in 25 minutes.
- **The sliding-window cache sized to its window** (28477e4b, c278e452):
  6,144 tokens a slot now fit, 13,440 MiB of context against 11,520 at
  3,072, where 0.9.0 asked 38,400 MiB and did not load.
- The answers as before: no thinking in any, the same mean length, 57-63
  word for word the same as llama-server's.

**Where the quarter comes from, 2026-11-20: the KV cache's layout.** The
settings compared: `n_batch` 2048, `n_ubatch` 512, flash attention on, F16
cache, 3,072 tokens a slot, for both (llama-server's defaults read in
`common/common.h` at `b1b770d9`, the engine's in its log). One difference
was left: the engine allocates one KV pool for all its slots, llama-server
by default one cache a slot (`kv_unified = 'false'`). llama-server given the
engine's layout (`--kv-unified --kv-unified-per-slot 3072`,
`law_rewrites.py --server-args`), and both at 16 slots too:

| | 8 slots | 16 slots |
|---|---:|---:|
| llama-server, a cache a slot (default) | 36 a minute (182 / 99 tok/s) | 49 (244 / 133) |
| llama-server, one pool, 3,072 a slot | 42 (213 / 116) | 57 (285 / 153) |
| engine main, one pool (its default) | 45 (228 / 123) | 58 (292 / 159) |

- **Most of the quarter is the layout**, not the code around llama.cpp:
  one pool alone takes llama-server from 36 to 42 at 8 slots, 49 to 57 at
  16. On the same layout the engine leads by 7% at 8 slots and 2% at 16,
  where two runs of the same binary differ by 2% (0.9.0: 47 and 48).
- **So what can be said:** with each program's defaults, EuLLM serves this
  model 25% faster than llama-server at 8 requests and 18% at 16, because
  it shares one KV pool among the requests on its own; llama-server matches
  it, within a few percent, when told to. Not "faster than llama.cpp".
- **16 slots instead of 8: a third more** for every program (engine 45 to
  58), and they fit (49,152 tokens of context on the A100 with the 31B).
- The answers: 198 in every run, no thinking, 54-65 word for word the same
  as llama-server's default run.

**The engine with 35540f36 and 687a1275, 2026-11-20.** The micro-batch for
images and the repeat penalty off by default on `/v1`; built on Leonardo
(`b8c189f8`). The penalty changes nothing here (every request states 1.0).

| run | a minute | context MiB | note |
|---|---:|---:|---|
| projector, 8 x 6,144 | 45 | 18,240 (13,440 before) | `Projector: reading prompts 1089 tokens at a time` |
| projector, 16 x 3,072 | 59 | 23,040 | the 1,089 micro-batch did not fit (28,800 MiB asked, `cudaMalloc failed` in the log): fell back to 512, as written |
| text, 16 x 3,072 | 59 | 23,040 | as before (58) |

198 answers in each, no thinking, 55-62 word for word llama-server's. One
thing for the engine: at 16 slots with the projector the fallback works but
llama.cpp writes an ERROR line (`allocating 28800.00 MiB ... out of memory`)
after the engine announced 1,089 tokens at a time; an operator reading the
log would think the load failed. **Decision:** Forge's teacher and checker
jobs on Leonardo use the engine from main (`--engine`, `VR_ENGINE`), 16
requests at 3,072 tokens for a 31B Q8 (`docs/cineca/leonardo.md`, `forge/CLAUDE.md`).

## 28. 2026-11-20: the corpora and the evaluation sets' articles

**What was found.** Asked by the JUPITER work which rows to train on, the
corpora were counted against the evaluation sets: `law_rewrites.py` keeps
out only the closed-book suite's articles, and only of the questions style.
Every other form, and every other set, was written from too: of the Gemma
rows, 141 cases on the articles of practical-dev, 65 on those of
practical-heldout, 323 on devbig's. Exam questions are other cases on the
same articles, so a model trained on them could score higher there than on
articles it never practised on. It should have been seen when the corpus was
built.

**Whether it inflated anything** (`split_by_article.py`, questions split by
whether the corpus the models were trained on, `memory`, has a case or an
explanation on their article; paired against r2l):

| exam | group | questions | r2l | k50 | k50-g1 | k50-g1-opd200 |
|---|---|---:|---:|---:|---:|---:|
| practical-dev | A, article in the corpus | 132 | 83 | 64 (6:25) | 67 (5:21) | 78 (6:11) |
| practical-dev | B, not in it | 13 | 4 | 4 (2:2) | 5 (3:2) | 5 (1:0) |
| devbig | A | 366 | 349 | 334 (6:21) | 350 (12:11) | 347 (3:5) |
| devbig | B | 106 | 100 | 104 (4:0) | 103 (4:1) | 92 (1:9, p=0.02) |

No gain gathers where the corpus saw the article: on practical-dev the
knowledge models lose to r2l there, and on devbig they are level with it.
The results so far are not inflated by the overlap. B is small on
practical-dev (13 questions) and says little; on devbig the candidate's
loss is on the articles without cases (1:9), not a gain on those with.

**From now on: clean files** (`clean_corpus.py`, `$WORK/eullm_runs/clean/`):
no row of any form on an article a held-out set asks about (norm-exam and
its v2-v4, practical-heldout, practical-v1's held-out); no question, case or
explanation on an article of a development set or of the closed-book suite
(their text, paraphrases, elements and structure stay: learning the statute
is the point, and what the closed-book suite measures); nothing on a number
a set makes up. A `manifest.json` of counts and sha256 beside each.

| folder | rows kept | dropped: held-out article | dropped: development, question-like |
|---|---:|---:|---:|
| `clean/v3s` (Qwen3-30B + program) | 184,657 | 14,906 | 3,007 |
| `clean/gemma-ok` (cases, explanations) | 10,194 | 971 | 1,260 |
| `clean/gemma2-ok` (elements, questions) | 11,829 | 1,007 | 334 |
| `clean/gemma3-ok` (all forms) | after the checks, 2026-11-22 | | |

`k7-r2l` and `k7-jpn17` were requeued on the clean files only. The models
trained before (k3s-k6, the k3s50 and k450 chains, the current candidate)
saw the uncleaned corpora: their development results are read with
`split_by_article.py`, and the held-out exams are given only to models
trained on clean files.

## 29. 2026-11-24: the recipe on four corpora, and the clean candidate queued

The candidate's recipe (knowledge adapter at 0.5, one GRPO round of 750
steps, 200 steps of OPD with r2l as teacher) on each knowledge corpus,
against r2l (paired, won:lost; closed book of 674, r2l 196):

| candidate | corpus | devbig | devbig, no note | practical | closed book |
|---|---|---|---|---|---:|
| k50-g1-opd200, seeds 0 and 1 | old memory, unverified | 4:14, 7:7 | 4:13, 7:15 | 7:11, 5:13 | 299, 274 |
| k3s50-g1-opd200 | Qwen, verified | 7:12 | 6:19 | 6:18 | 238 |
| k450-g1-opd200, seeds 0 and 1 | + Gemma cases and explanations | 7:9, 11:11 | 7:20, 10:18 | 8:14, 7:11 | 255, 226 |
| k650-g1-opd200 | + Gemma, all four forms | 7:9 | 7:15 | 8:7 | 240 |
| k750-g1-opd200 | Gemma only, clean, 70M tokens | 4:10 | 5:18 | 7:12 | 231 |

- Every GRPO round reached step 750 (checkpoint-750), so the five compare.
- **k650 is the most even so far**: level with r2l on devbig, the only one
  level on the practical exam (8:7), the least behind without the note,
  and a fifth more on the closed book. One seed: the second is running.
- The old unverified corpus taught more closed-book recall (299) and
  answered devbig worse; the verified corpora trade some recall for
  answers. Gemma alone at 70M tokens (k7) is not enough: Qwen plus Gemma,
  or Gemma at full volume (k8, queued after the checks of samples 5-8).
- The knowledge pass alone, before OPD, is level or ahead of r2l without
  the note (k750-g1 12:12, k650-g1 11:13): OPD buys closed-book recall and
  costs a little there.

**Queued:** k650-g1-opd200 with seed 1; `k9`, the k6 corpus rebuilt from
clean files (Qwen and every Gemma form), on r2l and on the 4B, and behind
k9-r2l the whole recipe (`r2l-k950`, scale, package, g1, OPD, exams), the
first candidate that may sit the held-out exams.

**Also on 2026-11-22/24:** the Booster maintenance of 23-24 November ended
on the 23rd in the evening; the checks of samples 5-8 were cancelled in it
and requeued (they resume from their verdicts). gpt-oss as a checker
through llama-server agreed with itself on 556 of 587 items when run again,
with the engine on 536 and 547: the engine is fit to check, from a new pass
(a pass keeps one server). The Large 4 pilot, with the 64-device build, now
stops at `loading model` and the job's hour of waiting runs out: the load
itself (605 GB, read and sent to 12 remote GPUs) is the next thing to
understand.
