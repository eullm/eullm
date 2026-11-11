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
