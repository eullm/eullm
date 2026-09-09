# Allocation plan — EHPC-AIF-2026PG01-1147

> 1,250 node-hours on Leonardo Booster, 02/09/2026 → 02/11/2026.
> Operational runbook: [`leonardo-runbook.md`](leonardo-runbook.md).
> Architecture decisions: [`adr-001-offline-distillation.md`](adr-001-offline-distillation.md).

## The number that governs everything

From 09/09 to the 02/11 deadline is 53.5 days, or 1,284 hours. The remaining
allocation is about 1,235 node-hours.

**One node kept busy continuously for the rest of the allocation consumes
essentially all of it.** Not approximately — 1,284 against 1,235 is a 4%
margin over the whole period.

Two consequences, and the second is the one that is easy to get wrong.

### The queue must never be empty

Every idle hour is budget that expires. It cannot be recovered by running
something bigger later, because the ceiling is wall-clock, not throughput.
An imperfect experiment sitting in the queue produces a measurement; an empty
queue produces a line missing from the final report.

This inverts the instinct that node-hours are scarce and should be rationed.
They are not scarce — **calendar** is scarce, and unused allocation is the
actual waste. It also inverts a decision made on 2026-09-09: "do not spend
170 node-hours on the old Phase 2, it is the worse path" was reasoned as if
those hours had an alternative use. They did not. With ~970 node-hours that
would otherwise evaporate, running the old Phase 2 as a genuine end-to-end
A/B baseline is defensible again — see the backlog below.

### Under-use is a reportable outcome

The Final Report to EuroHPC (see below) records what the allocation was used
for. An allocation granted and left 80% idle is visible, and it is the kind
of thing that is remembered at the next call.

## What the current plan actually consumes

| | node-hours |
|---|---:|
| Phase 1 — continued pre-training | ~83 |
| Pilot — designs, K, teacher topology | ~10 |
| Phase 2 — distillation, new pipeline | ~170 |
| Phase 3 — GGUF export (serial partition) | 0 |
| **committed to the critical path** | **~265** |
| **remaining** | **~970** |

Roughly a fifth. The rest needs work that is worth doing, not work invented
to burn hours.

## Backlog — ordered by what it gives the project

Keep this queue fed. Each item is a candidate the moment a node frees up.

1. **Old Phase 2 as an A/B baseline** (~170). Two complete models produced by
   two pipelines, which is the comparison ADR-001 Part 11 replaces with
   paired short runs *on the grounds of cost*. Remove the cost and the
   stronger evidence becomes affordable.
2. **A larger student** (~200-300). Qwen3-8B-Base, or a full fine-tune of the
   4B — the latter needs FSDP or ZeRO on the student, which is where sharding
   is genuinely the right tool. "Ability to train a larger student" is
   already a success criterion in ADR-001.
3. **Several students from one teacher pass** (~100-200). This is what makes
   the offline cache pay: the break-even in ADR-001 Part 2 is a count of
   student runs per teacher version, so a plan with three or four students
   moves the architecture decision as well as filling the queue.
4. **The other two demo models** — `medical-de` and `finance-fr`, already in
   the project's Phase-1 deliverables. Corpus work is the blocker, not compute.
5. **Ablations that improve the report** — K at full scale, LoRA rank, the α
   schedule, teacher precision end to end.

Items 1-3 also happen to be the experiments the report most needs. That is
not a coincidence: the allocation is sized for a research programme, and
using it fully and using it well point the same way.

## Final Report to EuroHPC — obligatory

The Principal Investigator must submit a **Final Report within three months
of the allocation completing**, on the EuroHPC JU template, covering the
results obtained and qualitative feedback on the use of the resources. It
goes to EuroHPC Peer-Review, and **failure to submit can disqualify future
proposals from any member of the research group**.

For this allocation: ends 02/11/2026, so the report is due by **02/02/2027**.

Two things follow.

**It is the same document we are already writing.**
[`paper/legal-it-4b-report-outline.md`](paper/legal-it-4b-report-outline.md)
is the technical report intended for Zenodo; the Final Report is its
obligatory sibling. Write once, submit twice.

**Two details still need confirming with CINECA or the EuroHPC portal**, and
were not extractable from the published call PDFs: the exact Final Report
template, and the required acknowledgement wording for publications. Ask
before the report is drafted, not after.

## Measurements this plan depends on

Every number above rests on one measurement — 4.86 s per optimizer step,
from ten hours of Phase 1. It replaced an earlier estimate of 3.6 s taken
from a fifteen-line sample, which produced a 60-hour epoch estimate against
81 measured. Re-derive the plan when Phase 1 finishes and the real total is
known, rather than carrying today's arithmetic forward as if it were fact.

`forge/scripts/leonardo/budget.sh` reports settled, in-flight and committed
node-hours; `saldo -b` alone lags by everything currently running.
