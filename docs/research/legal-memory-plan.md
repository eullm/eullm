# A small legal model that knows the law: a persistent legal memory

Research plan, opened 2026-10-26. Sites: JUPITER (training, until the end of
December), LUMI (the engine side only, as declared), MareNostrum 5 if granted.
Results go into this file and, for JUPITER, into `docs/jupiter/dossier.md`.

## Goal

A legal-it model small enough for a laptop that **knows** Italian and EU law,
not one that only reads well what it is handed:

1. it answers and corrects a false premise ("art. 640-septies c.p.", "the
   appeal term is 30 days") **closed book**, without the right article being
   put in front of it;
2. when it is not sure, it **looks the law up itself**, as a lawyer opens the
   code, instead of depending on a pipeline that prepares its context;
3. its knowledge can be **updated** when the law changes, without retraining
   the model.

Six months of Forge produced models that read and cite better than their
bases (practical exam 60 → 87 for the 8B, section 9 onwards of
`legal-it-experiments.md`), but not models that know: every evaluation since
September gives them the texts, and on the outside test a third of the traps
stay unsolved by any of them because the model cannot tell, from what it
holds, that an article does not exist or a term is wrong.

## Why the knowledge did not go in

* **Our training is open book.** SFT, OPD and GRPO all hand the model its
  texts and teach it to use them. Nothing asks it to hold them.
* **The September distillation read each ruling once, as raw text.** Work on
  knowledge capacity (the "Physics of Language Models" series) finds that a
  fact becomes retrievable only after many exposures *in varied forms*
  (paraphrases, questions, the fact cited from different sides), and that a
  model stores a bounded number of facts per parameter. One pass over raw
  rulings stores style, not extractable facts.
* **A 4-8B model has little room.** Italian and EU law plus fifteen years of
  case law is more than its parameters hold as recallable facts.

## Hypotheses, each with a yes/no

| | Hypothesis | Measured by | Yes if |
|---|---|---|---|
| H1 | Knowledge-augmented continued pretraining makes the model recall the law closed book | closed-book suite (below) | ≥ +25 points over r2l before, no loss on the open-book exams |
| H2 | A memory module adds recall at equal active parameters | H1's run with and without the module, same data and tokens | ≥ +10 points closed book |
| H3 | The memory can be updated alone | a simulated reform of 50 articles, only the memory retrained | new text recalled ≥ 80%, old recall within 2 points |
| H4 | The model learns when to look the law up | GRPO with a lookup tool, existing rewards | traps of 33 and practical exam above the best open-book model |

## Architecture

**Backbone: our best model, r2l** (the 8B line, legal-it-8b after the r2
SFT and GRPO rounds: devbig 449, practical 87, traps 14 of 33 on 10-26).
The held-out exam of 10-26 confirms it or puts r2q, its equal on every exam,
in its place. jpn17 (Qwen3.5-4B) is the second line, for a laptop without a
GPU worth the name.

Starting from a post-trained model has one danger we have measured: a long
training on top of a model GRPO shaped undoes the GRPO (the Cassazione SFT
of 10-20, section 8). Two answers, and they are part of what is compared:

* **with the module, the backbone stays frozen**: the knowledge goes into
  the memory and r2l's reading, citing and premise skills cannot be lost;
* **without it (E1), the backbone trains through LoRA with a replay** of the
  post-training data (open book, premises, practical) in every batch, and
  the open-book exams decide whether a short GRPO round has to follow.

**Legal memory module** (H2), in the line of the hashed n-gram embeddings
of Qwen3.8-Flash-Next and DeepSeek Engram and of Gemma 3n's per-layer
embeddings:

* a table of 1-6 B parameters, read at one early layer: for each token the
  hashes of the 2- and 3-token contexts ending on it give a handful of rows,
  concatenated, projected and added to the residual stream through a gate
  initialised at zero, so training starts from the unchanged model;
* the lookup positions are known before the forward pass, so the table lives
  in host RAM and is prefetched: on a laptop the model sits on the GPU (the
  8B in Q4 is about 5 GB) and the memory in RAM;
* trained with a sparse optimiser; the backbone frozen or LoRA;
* **a separate, versioned file** (`lex-memory-YYYY-MM`): an update retrains
  the memory on the changed articles plus a replay sample, not the model (H3).

Variants to compare inside H2: table size (1/3/6 B), layer, n-gram orders,
and a product-key memory (keys learned instead of hashed) as the
alternative family.

**Engine** (LUMI, inference and benchmarks): the lookup as an operator in
our llama.cpp mirror (gather from a host-RAM tensor, project, add), the
table as GGUF tensors, measured on MI250X and on a CPU-only machine.

## Data

The acquisition and processing exist (`docs/corpus-acquisition.md`); what is
new is how each source is written for knowledge to go in.

| Source | Status | Written as |
|---|---|---|
| Normattiva, all acts in force (`prepare_legislation.py --all-acts`) | ready for JUPITER | per article: the text; heading ↔ number both ways; the rule in other words (teacher); deadlines, penalties, procedibility as questions and answers (program, the premise generator's extractors); **non-existence** ("is there an art. 640-septies c.p.? No; the fraud is 640-ter"); repealed articles with the act that repealed them and the date |
| Normattiva, historical versions | to fetch | "until 2022 the article read…, since d.lgs. 150/2022 it reads…": what a reform changed, and the material for H3 |
| Corte di Cassazione 2021-2026, civil and criminal (anonymised) | ready | the principle of law of each ruling, extracted by the teacher, and "Cass. n. X/YYYY holds that…" both ways; the facts and the parties never as targets (privacy, below) |
| Consiglio di Stato 2017-2024 | ready; 2025-26 evaluation only | as the Cassazione |
| EUR-Lex, 17,563 acts in Italian | ready | as Normattiva, per article |
| CJEU judgments in Italian | to fetch | principles, as the Cassazione |
| General Italian text | to choose | 20-30% of every mix, so the model keeps its Italian and its reasoning |

**Held out, enforced as `check_corpus_holdout.py` does:** the Consiglio di
Stato 2025-26, a temporal slice of the Cassazione, and the articles of H3's
reform. For the closed-book suite it is the **questions** that are held out,
not the articles: a model can only know an article it trained on, so the
articles go in as statements, while the made-up numbers the suite asks about
are never said not to exist and no question-and-answer rewrite is written of
an article an evaluation asks about (the knowledge-capacity setting:
biographies trained, questions held out). `eullm_forge.datasets.law_corpus`.

**The legislation part is written** (10-29): `make_law_corpus.py` by program
(the text, a table of contents per code, heading and number, deadlines,
penalties, procedibility, repeals, numbers a code does not have) and
`law_rewrites.py` by the teacher (paraphrase, explanation, case, questions
and answers, elements), each rewrite refused if it adds an article, a
number, a deadline or a penalty its article lacks. First run on Leonardo on
the nine codes there (`sbatch_law_rewrites.slurm`); all acts in force on
JUPITER.

**Size** is measured first (E0): tokens per source, raw and augmented. As a
bound, training costs about 6 × parameters × tokens: for the 8B, 10 B tokens
are ~5e20 FLOPs, of the order of 500-600 GPU-hours on GH200, 50 B tokens
about 2,700. With the backbone frozen (the module's arm) the backward pass
through it is still paid, so the saving is smaller than it looks; it is the
memory, not the compute, that the frozen backbone protects.

## Evaluation, built before any training

A **closed-book suite**, private like the outside test, every item drawn
from held-out material:

* existence: real and made-up article numbers, half and half;
* heading ↔ number; the rule of an article in one sentence;
* deadlines, penalties, procedibility, repealed or in force;
* the principle of a ruling cited by number, and made-up rulings;
* what a reform changed;
* EU articles.

Scored by program where the answer is a fact (existence, numbers), by the
judge otherwise, with **confident wrong answers counted apart**: a model
that knows less but says "I am not sure" beats one that invents.

Every candidate also takes the open-book exams (devbig, practical, the
outside test of 33) so that knowing does not cost reading.

## Experiments

| | What | Where | Cost (GPU-h, order) |
|---|---|---|---|
| E0 | Closed-book suite built; current models measured on it; corpus sizes | Leonardo / JUPITER | < 20 |
| E1 | r2l, continued pretraining on the augmented corpus through LoRA with replay, no module (H1), at 2 and 10 B tokens | JUPITER | 120 + 600 |
| E2 | E1 with the memory module and the backbone frozen, three sizes (H2) | JUPITER | 3 × 600 |
| E3 | Simulated reform: memory retrained alone (H3) | JUPITER | < 50 |
| E4 | The lookup operator in the engine, speed on GPU and CPU | LUMI | bench hours |
| E5 | Lookup tool learnt with GRPO on the winner (H4) | JUPITER | 200-400 |
| E6 | Scale the winner: full corpus, the 8B, and the 4B line if the budget or MareNostrum allows | JUPITER / MN5 | 2,000+ |

E1 at 2 B tokens is the first go/no-go, within two weeks: if knowledge-
augmented pretraining does not move closed-book recall at that scale, the
module is not what is missing and the plan changes before spending more.

## Privacy

The rulings are personal data even anonymised, and a model that memorises
text can give it back. Before any model trained on them is released:

* training targets never include facts or parties, only principles of law;
* an **extraction test**: prompts built from held-out ruling openings, and a
  check that no name, place or amount absent from the prompt comes back;
* canaries (made-up personal details planted in a few training rulings)
  that must not be recalled;
* the EDPB opinion 28/2024 on AI models and personal data read against the
  release.

A failed test blocks the release of that model, not the research.

## Risks

* The memory modules' gains are shown in models trained from scratch;
  grafted onto a trained model they may give less. E2 measures exactly that.
* A small model may still miss numbers; H4 (looking the law up when unsure)
  is part of the goal, not an option.
* Teacher-written paraphrases can be wrong: those of rules are checked by
  program against the article (numbers, terms) before they enter training.

## Results

**E0, 2026-10-27** (details in `legal-it-experiments.md`, section 13). Closed
book, 716 items: published 8B 7% right, r2l 30%, r2q 31%, jpn17 3%; number
from heading 0-5%, heading from number 2-8%, terms 0-18%, penalties 0-17%;
360-489 answers wrong and sure. r2l's and r2q's lead is a reflex ("it does
not exist" said to real and made-up alike: r2q recognises 0% of the real
articles), not knowledge. The models read; they do not know. That is the
gap E1 and E2 measure against.

**Memory corpus, first pass, 2026-10-29.** The nine codes on Leonardo:
14,119 rows by program, 28,071 teacher rewrites kept and 1,264 refused
(4.3%: another article 365, a number 311, a deadline or penalty 305, a
repeal 148, cut short 90), in 92 minutes on two three-GPU jobs. Read
against their articles: the paraphrase (art. 2729 c.c.) and the questions
(art. 2365 c.c.) say what the article says; the case (art. 1429 c.c.)
qualifies a painting's false attribution as an error on the *nature* of the
object, n. 1, where it is the textbook error on a *quality*, n. 2. The
checks cannot see a wrong legal qualification, so the cases stay out of
training until a sample of them has been read; the other styles go in.
