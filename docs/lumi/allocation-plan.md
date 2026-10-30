# Allocation plan — EHPC-DEV-2026D09-278 (LUMI-G)

> 4,500 node-hours (18,000 GPU-hours) on LUMI-G, `project_465003366`,
> **12-09-2026 → 12-03-2027**. EuroHPC Development Access for the **Engine**,
> not for Forge. What the machine is, how the ROCm build works and what was
> measured on 12-09-2026: [`lumi-g.md`](lumi-g.md). The sibling plan for the
> Leonardo AI-Factory allocation, whose arithmetic this one repeats:
> [`../leonardo-allocation-plan.md`](../leonardo-allocation-plan.md).
>
> Updated 25-10-2026. Billed: **5,704 of 18,000 GPU-hours, 1,426 node-h,
> 31.7%** (`lumi-allocations`, 25-10-2026, the same as `sacct`'s recount) — `tools/lumi/status.sh` prints it
> on every call, beside `campaign.py budget`'s recount from `sacct`. Spent by
> need since 21-10-2026, not by the straight line: [Calendar](#calendar).
> `tools/lumi/nodi.sh` (alias `nodi`) is the short form: node-hours of each
> of the last 7 days, the average a day, what is left and what a day must
> spend to use it by 09-03-2027, the plan's end, three days before the
> allocation's.

## The number that governs everything

Six months against 4,500 node-hours is one whole node busy every day. The
first of those months went by with almost nothing spent, so from 05-10-2026:

| | |
|---|---:|
| calendar elapsed | 12.6% (23 of 181 days) |
| straight-line target to date | ~570 node-h |
| spent | ~0 |
| days left | 158 |
| needed from now | **28.5 node-h/day = 1.2 nodes around the clock** |

**One node is no longer enough.** Queue waits come off whatever is running,
so the campaign runs on **two nodes in parallel** until the deficit is
recovered (`status.sh` shows when), then one.

11-10-2026: 386 node-h spent, 347 behind the straight line (733 to date),
120.7 billed in the last 24 hours on five to six nodes; `budget` now says
what the last day spent and when the gap closes at the current burn (2.9
days at six nodes). The straight line is the stricter measure: the calendar
below asks 750 node-h by the end of October and back-loads the rest
(850-880 a month from November, the long soaks over Christmas), and by it
the six nodes reach October's figure around 13-10. So the burst stops there:
the queued links that would carry superseded engines (the unlabelled chain,
`next3`, `next4`'s soaks) into the week are cancelled, and the queue goes back
to one background chain plus a node for each new engine build.

12-10-2026: 532.6 node-h spent, 225.9 behind the straight line, 145.4 billed
in the last 24 hours. The six nodes' last job was due to end at about 10:00
on 13-10 (about 586 node-h in all by then), a day the machine is left idle
on purpose, and nothing was queued after it: no background chain for the rest
of October, whose 750 leave about 164 node-h for 13-31 October, less than a
node a day. A new engine build gets a node for its before/after round; the
hours the straight line counts as behind stay in the allocation for the peaks
of November to January.

14-10-2026: 557.5 node-h spent (2,230 GPU-hours, `lumi-allocations`), about
240 behind the straight line. The PI's decision: the allocation is spent to
its end on work that leaves measurements, and if it runs out early, more is
asked for. Round `r-next10` (`c01` + `c02` on the engine `next10`, 280
points, about 250 node-h) went on two nodes that night, and `--moe-cache` on
HIP was brought forward from January (`c10-moe-ram`, below).

15-10-2026: round `r-next15` (`c01`, `c02`, `c06`, `c07`, `c08` on the
engine `next15`, 566 points, up to 262 node-h) queued for the night and the
day after, when nobody watches: the two nodes of the `evict` jobs go on to
it, and each gets a second 48-hour link. Its reason is a build flag. `next13`
is the first engine whose CPU code is not SSE2 (x86-64-v3), and every round
before it measured EuLLM's CPU side (sampling, tokenizing, what `--fit`
leaves on the CPU) on SSE2 against a llama-server that ggml builds for the
login node's CPU (`GGML_NATIVE`, on by default: AVX2): `c07`'s comparisons
were not on equal footing until now. `next15` also carries the expert
cache's new eviction (llama.cpp patch 0008).

The same night (645.3 node-h spent, 198.5 behind the straight line, seven
nodes busy) the two `next11` jobs of `c10` were cancelled: `c10-v3` measures
all their points again on `next13`, and one of them had held a whole node
for 9.8 hours on DeepSeek-V3.1 with `--cpu-moe` on SSE2 kernels, a token
every few minutes per request. The request timeout (1,800 s) is per read,
so a server that streams that slowly never reaches it; a limit on a point's
whole duration was the runner's missing piece, added on 17-10-2026: three
times the point's estimate, at least an hour (`point.POINT_LIMIT_FACTOR`). The v3 points of the same
model with `--cpu-moe` at 4 and 16 requests and the `next11` points left in
the queue were moved to `parked/` (one request with `--cpu-moe` and the
480B's points say what the CPU does with the experts).

17-10-2026: LUMI's operations closed access to the machine for security
reasons (a notice to all users, no date to reopen). The build of `next16`
and the `evict2` jobs had been started the night before; what ran, and
whether the queued jobs survive, is to be read once access is back. On
return: follow CSC's instructions first (they may ask for new keys),
check that the checkout matches GitHub and rebuild the binary the next
jobs use rather than trust files that sat on the machine, then
`status.sh`; points claimed by jobs that died go back to the queue by
themselves when the next job starts (`requeue_stale`). Days lost to the
closure count against the calendar, not against the work: if it lasts,
they are a reason to ask EuroHPC for more time.

18-10-2026, the crashes behind the core dumps in the checkout. Reading the
server logs of the last fifteen jobs: nearly every engine server aborted
when the runner stopped it, on every build from `next10` to `next16` and
every model ("double free or corruption", "corrupted size vs. prev_size",
a ggml abort on the way out of `main`). The measurements were already
taken; the cause was `serve` returning with each model still held by its
scheduler thread, so the process's exit handlers took the GPU down under
it. Fixed in the engine (models unloaded and their threads joined before
`serve` returns, then an exit without those handlers), and the runner now
records how every server ended (`server_exits`), which says on the next
build whether it holds. Apart from those, soaks of Qwen3-32B with Q8_0
weights or a q8_0 KV cache at 32 requests aborted while working, and one
Qwen3-30B-A3B Q8_0 server met a GPU memory fault
(`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`); the runner counted those
soaks as passed, since fewer than half their requests failed, and now
records the death (`servers_died`). Their cause is still to be read in
their logs, and was not in them: the engine dropped llama.cpp's own log,
error lines included, so a crash left only `ggml-cuda.cu:109: ROCm error`
and not the line before it that says which call failed and on which
device. `c12-stability` runs them again with llama.cpp's log on; from the
build after `next17` its error lines are in every server's log.

19-10-2026, `c12`'s first pass (`next17`): its five soaks ran their six
hours without a death, and every server stopped with 0, so the clean stop
holds on ROCm. Six hours was too short, since `c01`'s soaks died 8 to 11.5
hours in; its long groups run the same configurations 12 to 16 hours, the
Qwen3-32B ones twice, on one node. In `c01` and `c02` a server whose
logits turned NaN did not recover: every later request failed the same
way and the runner kept asking (`quant-soak`: 30,259,011 of 30,331,204
requests failed; `soak-one-gcd`: 135,126 of 167,211), which points to GPU
memory written over under the model rather than one bad step.

20-10-2026, the cause in the log at last. A `long-32b` server of `c12`'s
long groups (`next17`) died about six hours in: its sequences' logits
turned NaN, then four seconds later `ROCm error: out of memory` in
`ggml_cuda_graph_evaluate_and_capture`, at `hipGraphInstantiate`, and the
abort. llama.cpp builds its ROCm backend with HIP graphs on
(`GGML_HIP_GRAPHS`, on by default), and continuous batching gives the GPU a
graph of another shape nearly every step, so graphs are made and replaced
for hours. Its VRAM had grown too: Qwen3-32B with an f16 KV cache at 32
requests holds 52.3 GB at a throughput point and peaked at 58.5 GB and 61.1
GB at the end of six-hour soaks; Qwen3-32B Q8_0 with a q8_0 cache reached
65.3 GB of the GCD's 65.5. `c12`'s `graphs-*` groups run the same
configurations three hours with the graphs on and off
(`GGML_CUDA_DISABLE_GRAPHS=1`); the node's VRAM samples
(`results/<job>.node.jsonl`) say whether it still grows without them.

The node's samples of that job: every server's VRAM grew in a straight
line, 1.3-2.5 GiB an hour whatever the model (Qwen3-14B 27 → 48 GiB in ten
hours), and the servers died where their GCD was full, at 63.9 GiB: the two
Qwen3-32B Q8_0 at 02:04 UTC, a Qwen3-32B Q4_K_M with an f16 cache at
about 05:45. The Qwen3-30B-A3B fell from 60.4 to 43.2 GiB at 05:29 and went
on answering: llama.cpp's legacy pool, the only one on HIP, frees the
buffers it keeps when an allocation fails and retries, and the GPU
graphs captured before still point into them, so a replay computes NaN
or faults. Patch 0011 drops the graphs after any flush and flushes before
less than 512 MiB would be left; the long groups run again on it
(`next20`, round `r-pool`).

20-10-2026, the first half hour of `graphs-*` (`next19`): with the HIP
graphs on the servers grew 0.6-2.3 GiB an hour (Qwen3-14B 2.11, the
30B-A3B 2.29, Qwen3-32B 0.61-0.65), with them off not at all (0.00 on all
four). The growth comes with the graphs, the MoE included, and may be
the ROCm runtime's rather than the pool's: the pool takes and returns
its buffers the same way with or without a capture. `nographs-*` soaks
the configurations that died 12 to 16 hours with the graphs off, and
`graphs-speed` measures what they are worth (Qwen3-8B and 32B, 1 to 32
requests, on and off), before the engine turns them off on ROCm.

The whole of `graphs-*`, three hours at 32 requests, graphs on → off:
Qwen3-14B 267 → 270 tokens/s, VRAM +2.11 GiB an hour → flat;
Qwen3-30B-A3B 226 → 232, +1.86 → flat; Qwen3-32B 172 → 175 and 172 → 176,
+1.37 and +1.35 → flat. `graphs-speed`, tokens/s at 1, 4, 16 and 32
requests, on → off: Qwen3-32B 37.5 → 37.7, 75.6 → 75.2, 259 → 264, 378 →
384; Qwen3-8B 113 → 112, 228 → 228, 680 → 692, 958 → 953. The graphs buy
nothing on the MI250X, at one request either, and are the growth: EuLLM's
ROCm builds leave them out from `next21` (`GGML_HIP_GRAPHS=OFF` unless the
build sets it). `nographs-*` (`next20` with the graphs off at run time)
checks 12 to 16 hours; `next21` repeats `c12`'s six-hour soaks and `c01`'s
reference and one-GCD groups without the variable.

21-10-2026, the night's soaks. With the graphs off (`nographs-*`,
`next20`), the four configurations that died ran 15.8 hours, none died,
and their VRAM did not move (0.00 GiB an hour): Qwen3-30B-A3B Q8_0 at 42.6
GiB and 227 tokens/s, Qwen3-32B at 51.1 GiB and 174 with an f16 cache,
36.4 and 159 with q8_0, its Q8_0 at 50.1 and 139. `next21`, built without
graphs, repeated `c12`'s six-hour soaks: flat as well, none died, 268,
231, 176, 162 and 137 tokens/s against 267, 227, 169, 158 and 137 with the
graphs (`next17`). Patch 0011 alone, graphs on (`r-pool`): five of eight
died again, 6.3 to 11.9 hours in; its flush carried three to the end (a
Qwen3-32B that reached 64 GiB, the 30B-A3B and the 14B). Leaving the
graphs out is the fix; 0011 stays for builds that keep them.

`c11`'s `load-*` (`next19`, cold, one binary each way), seconds to load
with patch 0010 against without (`GGML_META_GATHER_ROWS=0`): Qwen3-235B
as two servers of four GCDs 352 against 599, Qwen3-Coder-480B over eight
569 against 1,567, gpt-oss-120b as four servers of two 158 against 346.
1.7-2.8 times faster, and still slower than layer split (19-145, 113 and
10-46 s); throughput is the same within one repeat's noise (the 235B 366
and 370 tokens/s, the 480B 238 and 257, gpt-oss-120b 1,298 and 1,518).

The 0.7.90 round (`r-0.7.90`, `next18`, 19-20-10-2026): 142 points, none
failed, and every server stopped with 0. Against the last measurement of
each configuration, the medians of new over old are 0.99-1.01 in every
group; the changes are gains on older builds (dense-one-gcd up to 1.27,
Qwen3-32B over eight GCDs as replicas 1,927 → 2,069 tokens/s). With the
reserve now 1 GiB, Qwen3-32B with an f16 cache and 40,960-token slots at
4 requests loads on one GCD (60.6 GB), where it did not fit before; the
32B Q8_0 with an f16 cache at 32 requests still does not.

21-10-2026, the PI's decision: the allocation is spent on what the work
really needs, without holding to the straight line, and if it runs out
before 12-03-2027 more hours are asked for. `next21` is the first engine
whose soaks hold, and every soak of `c01` and `c02` before it measured
servers dying with the HIP graphs (`quant-soak`'s 30 million failed
requests): the whole round goes on it at once (`r-next21`: `c01`-`c03`,
`c06`-`c11`, about 400 node-h at most), on four nodes
(`TOP_UP=1 submit_campaign.sh 4 2` keeps four chains, counting those
already queued), with `c13` and the 24-hour `soak-node` among its points.
`budget` now says on which day the allocation runs out at the last 24
hours' rate: the date by which more hours have to be asked for.

22-10-2026, why tensor loads are still slow after patch 0010 (3-5 times
layer split): the meta buffer of `--split-mode tensor` has no events or
host buffers, so llama.cpp's loader gives it none of the async uploads from
pinned memory that a GCD of layer split gets. Each weight read into memory
is read whole and then copied, the file idle while the GCDs copy and the
GCDs idle while the file is read, and the meta buffer copies it to its GCDs
one after the other from one thread. Patch 0012 reads the next weight while
the last is copied and copies to every GCD at once, and logs how long each
part took; `c11`'s `load-*` groups measure it three ways on `next22` (both,
threads only, neither), 9 points, up to 6 node-h. DeepSeek-V3.1's tensor
points failed because llama.cpp has no tensor split for its architecture
(`deepseek2`); `c11` keeps it on layer split.

The first measurement (`next22`, job 23062162), seconds to load cold with
0012, with only its threads, and with neither: Qwen3-235B as two servers of
four GCDs 346, 421 and 389; Qwen3-Coder-480B over eight 474, 516 and 577;
gpt-oss-120b as four servers of two 159, 176 and 171. 7-18% off, still about
three times layer split. Its log line said where the rest goes: of the
235B's load, reading the file took 221-272 s and copying to the GCDs 26-53 s
(29 with the threads, 53 without). The read went at 0.6 GB/s where layer
split reads at about 1.2: each weight was read into a buffer allocated for
it, and reading into pages never touched before runs at a fifth of the speed
(0.33-0.49 against 2.6-2.9 GB/s, 512 MiB pieces from memory). 0012 now reads
every weight into the same two buffers and gathers rows into buffers kept
from call to call; `next23` measures it the same way.

`next23` (job 23068061), with layer split in the same job as the reference,
seconds to load cold: Qwen3-235B as two servers of four GCDs 223 with 0012,
278 with its threads only, 260 with neither, 130 by layer; Qwen3-Coder-480B
over eight 285, 292, 327 and 158; gpt-oss-120b as four servers of two 107,
116, 110 and 61. The reused buffers halved the copying (the 235B 13-21 s,
from 26-53) but not the reading (209-253 s): 1.7-1.8 times layer split,
from 3-5 times on `next21`. Layer split reads 1 MiB at a time with Lustre's
read-ahead running ahead of it; a tensor load read each weight with one
read of hundreds of MiB. 0012 now reads it with 8 threads at once in pieces
of 16 MiB; `next24` measures 8 against 1 and 16 (`LLAMA_LOAD_READ_THREADS`).

`next24` (job 23083787), seconds to load cold with 1, 8 and 16 threads
reading each weight, and by layer in the same job: Qwen3-235B 248, 206, 140
and 124; Qwen3-Coder-480B 275, 160, 121 and 162; gpt-oss-120b 120, 83, 59
and 69. Reading the 235B took 234, 192 and 125 s, the 480B 240, 127 and 93,
gpt-oss-120b 112, 76 and 53; copying stayed 8-26 s. With 16 threads a tensor
load is as fast as layer split or faster: the 480B and gpt-oss-120b load
faster split as tensor than by layer, the 235B in 13% more time. From
`next16`'s 943-1,269, 1,679 and 255-626 s, 4 to 14 times faster. 16 is
0012's default from `next25`; tokens/s are the same whichever way a model
loaded (one measurement a point: gpt-oss-120b's four servers wrote 1,226 to
1,722 tokens/s from point to point, whatever the load).

22-10-2026, the night's queue, all of it on `next25` (0012 reading with 16
threads by default), as two jobs of one node each, label `next25`:
`c11`'s `load-*` (16 threads against 32 and 64, layer split beside them;
12 points, up to 7 node-h) and `c12`'s `days-*`: two days at 32 requests on
one GCD each (44-46 hours) of the configurations that died with the HIP
graphs (Qwen3-32B with an f16 and a q8_0 cache, twice each, its Q8_0,
Qwen3-30B-A3B Q8_0) with Qwen3.6-35B-A3B and Qwen3-14B Q8_0 beside them, the
stability objective (6) on the build a release would carry: 8 points, up to
46 node-h. The round `r-next21` goes on on its four chains.

23-10-2026, morning: both `next25` jobs (23088874, 23088875) were still
running twelve hours in and neither had written a summary, so neither had
run `load-*`. Two jobs of one label starting together take the
highest-priority points in turn: `days-*`'s eight went four to each node,
and `load-*`, which wants a whole node, waits for them to end, while four
GCDs of each node have nothing to do. `c12`'s `fill-models` puts eight more
models on them (Qwen3-4B, 8B and its Q8_0, 14B, Qwen3.6-27B, Mistral Small
24B, gpt-oss-20b, Qwen3-30B-A3B Q4_K_M), at 32 requests, one GCD each: the
runner gives a workload point the time until the waiting whole-node point
can start, so they soak until `days-*` ends and `load-*` then has the node.
Points of different widths for one label go to one job, or to jobs started
apart.

24-10-2026, the round `r-next21` read from its jobs' summaries (23029434,
23029435, 23029436, 23029438, 23029440), the first on an engine without HIP
graphs. **No server died in any soak.** `soak-one-gcd`, twelve hours at 32
requests on one GCD, sixteen configurations (Mistral Small 24B, Qwen3-8B,
14B, 32B, Qwen3.6-27B and 35B-A3B, with f16, q8_0 and q4_0 KV caches): every
one to the end, VRAM flat, 41,061 to 329,770 requests each (Qwen3-32B f16
172 tokens/s, q8_0 158, q4_0 144; Qwen3-8B 407-551; Qwen3.6-35B-A3B 306-314).
`quant-soak`, twelve hours, ten configurations of Q8_0 and MoE models
(gpt-oss-20b 403-428 tokens/s, Qwen3-30B-A3B 216-257, Qwen3-32B Q8_0 138):
the soak where a server once failed 30,259,011 of 30,331,204 requests on
NaN logits ran clean. `moe-soak`, 22.5 hours on a whole node: DeepSeek-V3.1
25.1 tokens/s (29,938 requests), Qwen3-235B Q8_0 40.8 (49,925),
Qwen3-Coder-480B 39.7 (28,721), Qwen3-235B Q4_K_M as two servers 97.1
(102,770); VRAM flat on every GCD. Accuracy within the noise of the earlier
rounds (gsm8k 0.91-0.96, arc-easy 0.97-0.99, arc-challenge 0.91-0.97).

`c07` (16 clients, 400 gsm8k questions, tokens/s): EuLLM against
llama-server and Ollama, Qwen3-8B 543, 486 and 170; Qwen3-14B 358, 315 and
141; Qwen3-32B 188, 178 and 104; Qwen3.6-35B-A3B 237, 189 and 29.1. EuLLM is
6-25% ahead of llama-server (the MoE most) and 1.8 to 8 times Ollama, as on
`next6` (533, 349, 191, 221). `--kv-unified` 0-5% either way.

`c06` (see 25-10-2026 below for MTP measured with drafts), one request at a
time over 300 gsm8k questions: Qwen3.5-9B Q4_K_M
98.4 tokens/s without `--mtp`, 97.9-99.4 with `--mtp 1-3`; its Q8_0 80.9 and
79.9-80.5; Qwen3.6-27B Q4_K_M 37.8 and 37.4-38, Q8_0 28.7 and 29; accuracy
within one or two questions of 300. Read at first as MTP buying nothing on a
GCD; corrected 24-10-2026: no draft was made in any `c06` point of any
round. These repositories ship a projector, `eullm pull` fetches it, and a
model with a projector loads on the sequential engine, which does not draft,
while the banner still says MTP is on. MTP has not been measured on the
MI250X yet (below, `mtp-ab`).

`c03`: Qwen3-32B split as tensor over two GCDs, one request, 50.3 tokens/s
(43.3 on `next12`, 45.4 with RCCL on `next14`).

Failed, all known: `place-671b` (no tensor split for DeepSeek's
architecture, gone from `c11`), `ram-480b`, `ram-671b` and two `ram-prompt`
points with every expert on the CPU, stopped at their time limit.

The whole round against the last earlier measurement of each configuration
(`campaign.py compare --round r-next21`, new from 24-10-2026, in every job's
summary since): 744 configurations, tokens/s new over old median 1.01
(c01 1.00-1.02 a group, c02 0.99-1.08, c07 1.00-1.09, c10 1.03-1.23), none of
the 72 groups compared below 0.95 at its median. Of the 27 configurations below 0.95
the largest were Qwen3.6-35B-A3B's long-context points at one and four
requests (0.42 and 0.73), and they were not slower: with an 8,000-token
prompt and one request, decode 83-92 tokens/s, prefill 2,360-2,570 and the
first token at 3.0-3.3 s in every round since 0.7.20, while their tokens/s went from 11.6 to
32.1 from round to round with how many tokens the model wrote before it
stopped. `compare` reads a throughput point by its decode speed first since
then, tokens/s beside it: read so, the round is 1.01 over the rounds before
it at the median of its 744 configurations, 20 of them below 0.95 and no
group.

24-10-2026, 0.7.93 turns the MoE expert cache on by itself: with `--fit`
(on unless `--no-fit`, `--fit-strict` included), no `--cpu-moe`,
`--n-cpu-moe` or `--mtp`, and one GPU of the CUDA backend visible, a MoE
model whose experts do not fit gets `--moe-cache auto`; `--moe-cache off`
turns it off. What it changes on LUMI-G, read from the specs: a point
sees only its own GCDs, so every point on two or more GCDs is untouched
(the cache refuses more than one GPU), and of the one-GCD MoE points every
model but `c10`'s fits a GCD (Qwen3.6-35B-A3B, Qwen3-30B-A3B, gpt-oss-20b),
where the cache plans nothing. Only `c10`'s points without a flag move:
they measured `--fit`'s own split beside the cache and `--cpu-moe`, and on
0.7.93 they would measure the cache twice. They ask for `--moe-cache off`
from now on (new configurations to `compare`, against `[]` until 0.7.92).
The RAM is not a new question there: `c10`'s points are alone on the node,
and DeepSeek-V3.1's 367 GiB of experts were already pinned in a 480 GB job
with the cache. One thing changes for `--fit-strict` everywhere: a MoE
model that does not fit, which it refused, now loads with its experts in
RAM.

What the default is worth on a GCD, from `c10` in `r-next21` (the engine of
21-10-2026, eviction 0009, one measurement a point), tokens/s with
`--fit`'s own split, `--moe-cache auto` and `--cpu-moe`, by requests:

| model (experts in RAM with the split) | 1 | 4-8 | 16-32 |
|---|---|---|---|
| gpt-oss-120b MXFP4 (about 5%) | 107 / 108 / 22 | | 184 / **110** / 52 at 16; 168 / 170 / 41 at 32 |
| Qwen3-235B Q4_K_M (about 60%) | 6.9 / **21.8** / 4.6 | 13.2 / **31.5** / 6.8 at 8 | 15.9 / **25.2** / 9.3 at 32 |
| Qwen3-Coder-480B Q4_K_M (about 80%) | 3.4 / **7.4** / 0.2 | 7.3 / 8.1 / 3.8 at 8 | 8.9 / 9.1 at 32 |
| DeepSeek-V3.1 Q4_K_M (about 90%) | 3.4 / 3.4 | 4.1 / **3.2** at 4 | 5.3 / **3.7** at 16 |

The cache wins where a large share of the experts stays out and it still
holds a good part of them (the 235B 1.6 to 3.2 times, the 480B 2.2 times
at one request); it loses where the model nearly fits, since the split
already keeps nearly every expert on the GCD and the cache adds copies
(gpt-oss-120b 0.60 at 16 requests), and where it holds a tenth of them or
less, since the 64 cores compute the experts faster than the link copies
them (DeepSeek-V3.1 0.70-0.78 at 4 and 16). On the RTX 5070 Ti, where the
default was decided, Qwen3.6-35B-A3B has about half of its experts out.

24-10-2026, what the engine does with it. A first answer (#1090) kept the
default to one CUDA GPU and took it off ROCm. The cause of both losses is
not the backend but where the model sits: the cache gains when a large share
of the experts is out and it still holds a good part of them. So the default
is now `MoeCache::Default`, `auto` only where 25% to 85% of the experts do
not fit (`fit::MOE_CACHE_DEFAULT_OUT`, the share computed by the planner
from the VRAM left and the experts' bytes), on one GPU of the CUDA backend,
CUDA or ROCm; outside it `--fit`'s split, and the log says the share and the
choice. `--moe-cache auto` or a size still asks for the cache whatever the
share. `c10`'s `ram-default` (8 points, `next26`) loads the four models
without a flag at 1 and 16 requests: the 235B and the 480B should take the
cache and gpt-oss-120b and DeepSeek-V3.1 the split, which the summary shows
from the log (`job_summary.sh`), at the tokens/s `ram-*` measured each way.

25-10-2026, `ram-default` read (jobs 23136196 and 23137404, `next26`, no
flag): the band took the faster way at every one of the eight points.

| model | experts out (1 / 16 requests) | way | tokens/s at 1 / 16 | `r-next21`: split, cache |
|---|---|---|---|---|
| gpt-oss-120b MXFP4 | none (fits) / 2% | split | 105 / 184 | 107, 108 at 1; 184, 110 at 16 |
| Qwen3-235B Q4_K_M | 55% / 64% | cache | 20.9 / 25.4 | 6.9, 21.8 at 1; 15.9, 25.2 at 32 |
| Qwen3-Coder-480B Q4_K_M | 79% / 85% | cache | 7.3 / 8.7 | 3.4, 7.4 at 1; 8.9, 9.1 at 32 |
| DeepSeek-V3.1 Q4_K_M | 86% / 88% | split | 3.4 / 5.5 | 3.4, 3.4 at 1; 5.3, 3.7 at 16 |

The upper bound falls between the 480B at 16 requests (85%, cache) and
DeepSeek-V3.1 at one (86%, split), which is where the two ways meet: the
480B wrote 8.9 tokens/s split and 9.1 with the cache at 32 requests,
DeepSeek-V3.1 3.4 each way at one. A share on the wrong side of 85% by a
point or two costs almost nothing; past it the cache loses (DeepSeek-V3.1
3.7 against 5.3 at 16). `MOE_CACHE_DEFAULT_OUT` stays (0.25, 0.85).

25-10-2026, morning: the two 2-hour `mtp-ab` jobs on the text-only twins
(23140145 for `next25`, 23140146 for `next26`) waited the whole night on
priority, with no start time estimated and 362 jobs of higher priority
ahead on `standard-g` (ours 453: 201 of fair share, 52 of age). Points of one
GCD do not need a whole node: the same two jobs went to `small-g` too, four
GCDs each (`submit_campaign.sh 1 1 --partition=small-g --gpus-per-node=4
--cpus-per-task=28 --mem=200G`, which bills 2 GPU-hours an hour against a
node's 4), and whichever starts first measures the points while the other
finds nothing and ends. On a shared node the runner binds no cores where
those of a GCD are not all the job's, and records what shared it; the share
of drafts kept at temperature 0 does not depend on it, and tokens/s are read
against the point without `--mtp` of the same job.

25-10-2026, the engine side of it (`next27`): with `--mtp` on one slot, a
projector found for the model (in the store or beside the weights) gives way,
the model loads text only and drafts, and one log line says so; a projector
passed with `--mmproj` stays and the line says `--mtp` does nothing for it.
The banners show MTP only where the load drafts. `c06` whole on `next27`
(`r-next27`, 56 points: `mtp-graded` and `mtp-throughput` on the models as
pulled, which should now draft, and `mtp-ab` on the twins, which should
match `next26`), on `small-g`: two jobs of four GCDs, the time from
`campaign.py walltime --gcds 4`. It is the first measurement of MTP on the
MI250X, objective (5).

25-10-2026, why nothing started all day, with 47 `small-g` and 58
`standard-g` nodes idle and no maintenance coming (the reservations hold
single broken nodes and two `small-g` nodes for another project): our six
jobs, two hours to nine, `small-g` and `standard-g`, all waited on priority
with no start time, which means the backfill scheduler did not reach them.
Their priority is 453 of a possible ~3,600 (`sprio -w`: age 500, fair share
1,000, job size 1,000, partition 1,000, QOS 100): 52 of age, 201 of fair
share, 200 of partition, 0 of job size (one node of 2,728). The fair share
is the cause: the project's share of the machine is 0.000199
(`sshare -A project_465003366`), its decayed use 0.000434, 2.2 times as
much, from the week of 88 node-hours a day; the factor is 0.20, where a
project that has used nothing gets up to 1. Age adds about 70 a day, the
full 500 after about a week, and the use decays away by itself.

What it means for the pace: spending by need is bounded by the fair share.
A week at four times the allocation's own rate (88 node-hours a day against
22-23) is paid back in days when nothing starts, so the run-out date
`nodi.sh` prints at last week's rate (29-11-2026) is not one the scheduler
would let us reach. What keeps the queue moving is a steady one to two
nodes, the work spread over the weeks rather than in bursts of four nodes;
the requests for more hours still go in when the forecast says so.

26-10-2026, the queue cleared of what no job would run: 59 points set aside
in `$CAMPAIGN_DIR/dropped/2026-10-26/` (moved, not deleted). The 40 `rccl-*`
of the label `rccl` (RCCL measured on `next14` in `r-split-rccl` and
`r-split-rccl2`, linked into every LUMI build since 16-10-2026, so `c03`'s
own groups measure tensor with it), `soak-node` on `next3`, `next15` and
unlabelled and `moe-soak` on `next4` (passed by `soak-node` on `next21` and
the two-day soaks on `next25`), `ram-671b` and `ram-prompt` on `next11` (all
experts on the CPU, stopped at their time limit every time), and `mtp-ab`
on `next27` (the twins measured twice already). Left: `c06`'s
`mtp-graded` and `mtp-throughput` on `next27`, 48 points, in the two
`small-g` jobs waiting.

The pace from here: one chain of one node at a time, with `small-g` for the
points of one GCD, always something of ours waiting, instead of four nodes
for days. A full round on an engine that touches inference is about 400
node-hours; at one or two nodes it runs 10-15 days, which is the 23-30
node-hours a day that ends the allocation on 09-03-2027 with nothing left
over. The next one is on the release that carries the expert-cache band
and the `--mtp` fix, once `c06` on `next27` has run.

26-10-2026, evening: the fair share was not the whole reason. The two
`small-g` jobs of `c06` on `next27`, waiting 29 hours with 9 hours asked,
started within minutes of being lowered to 5 (`scontrol update JobId=J
TimeLimit=5:00:00`), as 23136196 had on 24-10 from 48 hours to 6. With
priority this low the jobs ahead hold the nodes they will start on; what is
left are gaps of some hours before those starts, and a job fits one only if
the time it asks for does. Asking for more time than the work needs keeps a
job waiting at any priority. The 9 hours came from the specs' estimates
(an hour for each of `c06`'s 300-question points, which took 16-60 minutes);
`campaign.py walltime` now times each point by the longest an earlier run of
the same configuration took, the estimate only for one never run.

27-10-2026, `c06` whole on `next27` (jobs 23153303 and 23153304, `small-g`,
four GCDs each, all 48 points overnight): MTP on the MI250X, the six models
as pulled, which now load text only with `--mtp` and draft (their VRAM is
the twins', 6,776 MiB for the 9B at `--mtp 1`). Tokens/s and drafts kept,
`mtp-graded` (300 gsm8k questions, temperature 0) / `mtp-throughput` (the
Leonardo prompt at the default sampling, three repeats):

| model | none | `--mtp 1` | `--mtp 2` | `--mtp 3` |
|---|---|---|---|---|
| Qwen3.5-9B Q4_K_M | 96.4 / 95.7 | 133 (93%) / 124 (80%) | 150 (87%) / 118 (57%) | 124 (81%) / 92.9 (46%) |
| Qwen3.5-9B Q8_0 | 80.1 / 78.8 | 105 (93%) / 103 (78%) | 127 (87%) / 111 (64%) | 134 (81%) / 115 (61%) |
| Qwen3.6-27B Q4_K_M | 37.5 / 36.1 | 55.8 (95%) / 52.4 (80%) | 61.0 (90%) / 53.2 (65%) | 55.3 (85%) / 45.5 (62%) |
| Qwen3.6-27B Q8_0 | 28.7 / 28.8 | 42.8 (95%) / 43.0 (89%) | 53.2 (90%) / 47.4 (74%) | 59.0 (86%) / 42.4 (49%) |
| Qwen3.6-35B-A3B UD-Q4_K_M | 91.3 / 91.0 | 123 (94%) / 118 (75%) | 150 (89%) / 133 (71%) | 121 (83%) / 107 (68%) |
| Qwen3.6-35B-A3B Q8_0 | 96.0 / 96.8 | 131 (94%) / 132 (89%) | 150 (89%) / 131 (67%) | 128 (83%) / 98.0 (51%) |

`--mtp 1` gains everywhere, 29-49% at the default sampling; `--mtp 2` is the
best at temperature 0 (+56-85%), and the Q8_0 models, dearer per token, take
a third draft (27B Q8_0 +106%, 9B Q8_0 +67%). At the default sampling fewer
drafts are kept and `--mtp 2` varies more between repeats (cv 4-18%). The
`r-next21` reading of `c06`, MTP buying nothing on a GCD, measured no draft
at all: it is objective (5)'s first result. Accuracy: within one or two
points of the 300 questions either way, except Qwen3.6-35B-A3B UD-Q4_K_M,
5-7 questions lower with each of the three settings (0.9467 without,
0.923-0.93 with); one run each, watched in the next round. Without `--mtp`
these models still load with their projector, on the sequential engine:
the first token of the 9B and the 35B-A3B came in about 65 ms that way and
23-28 ms on the scheduler with `--mtp`, a difference of the engine and not
of the drafts.

27-10-2026, the round on the release 0.7.94 (`r-v0794`, label `v0794`, the
binary built from the tag): `c01`-`c03`, `c06`-`c11` and `c13`, 835 points,
about 400 node-hours by the estimates and fewer by what the same
configurations took on `next21`. Not on four nodes at once: two chains on
`standard-g` of jobs of at most 8 hours, as many as the work needs
(`submit_campaign.sh 2`, `campaign.py walltime --max-hours 8`), each job
asking for the time its share takes by the earlier measurements. At one to
two nodes it runs about two weeks, the 23-30 node-hours a day the
allocation's pace asks for. `c03`'s `rccl-*` groups go with it: they were
labelled `rccl` in the spec for jobs of an RCCL build, and every LUMI build
has RCCL since 16-10-2026.
Submitted the same day: `eullm 0.7.94 (ROCm) [ac4ec51a6522]` (the tag plus
one commit of tools and docs; the engine is the tag's), 835 points planned,
817 of them timed from an earlier run of the same configuration; two chains
of 14 jobs of 8 hours (heads 23206675 and 23206689), at most about 220
node-hours, most of a chain's later jobs ending at once if the work runs
short of the estimate.

28-10-2026, llama.cpp b11925 before it is merged (branch
`chore/llama-cpp-b11925`, b86d2f0 -> b11925, 443 commits;
`docs/llama-bump-b11925.md` on it). Not merged into this site's branch and
nothing pushed to it: two worktrees on `/scratch`, the branch and its base
be0cf9e (0.7.95), each built into a directory of its own
(`eullm-target/llama-b11925`, `eullm-target/v0795`), with `EULLM_REPO` set
to the worktree, since `campaign_env.sh` exports the main checkout's and
`build_engine.sh` takes it; `r-v0794`'s chains keep `eullm-target/v0794`
and are held (`scontrol hold`) while the test runs, released by a job that
waits on it. The main checks only, old against new, the same models and
flags (`c14-llama-bump`), in one `standard-g` job of 8 hours per binary:

1. `sbatch_smoke.slurm` with the new binary.
2. `bump-moe`, first, on the whole node: the 235B with `--moe-cache auto`
   at 8 and 32 requests, `LLAMA_MOE_CACHE_STATS=64`: the share of experts
   in VRAM, now with 0008 alone in place of llama.cpp's new SLRU.
3. `bump-soak-*`: five hours at 32 requests on one GCD (Qwen3-32B f16 and
   its Q8_0, Qwen3-30B-A3B Q8_0, Qwen3.6-35B-A3B), llama.cpp's log on:
   patch 0011 is halved and calls upstream's graph invalidation.
4. `bump-mtp-graded`, beside the soaks: the six Qwen3.5/3.6 MTP twins
   without `--mtp` and with `--mtp 2`, 100 gsm8k questions at temperature 0.
   On the reference PC b11925 keeps about 5 points fewer drafts at `--mtp 2`
   and writes 2-3.5% slower.

The branch does not build for the MI250X (28-10-2026, 19:30, the first
build, nothing submitted): llama.cpp's `ggml-cuda/topk-moe.cu` instantiates
`topk_moe_cuda<576>` and `group_mask_warp_inplace` finds no match for gfx90a
(`n_experts = 576, experts_per_thread = 18`, "substitution failure"), 13
errors, so `ggml-hip` stops. gfx90a runs 64-wide wavefronts where CUDA's
warp is 32, and a build for a 32-wide AMD card (the release's RDNA) does not
meet it. The release's `eullm-linux-x64-rocm-gfx90a` would fail the same
way. The test waits for the branch to build here; `r-v0794` goes on.

29-10-2026: the fix is on `feat/model-libraries` (4cf28c5), which carries
the same llama.cpp b11925, the model libraries and 0.8.0, and replaces
`chore/llama-cpp-b11925` for the test. The cause was not the 64-wide
wavefront but `__ballot_sync`, which ROCm 6.x declares only with
`HIP_ENABLE_WARP_SYNC_BUILTINS` (ROCm 7 always, so the release's RDNA build
on 7.2.4 never met it); patch 0013 uses a shuffle on HIP for the same mask.
First a build of the branch with this site's ROCm, in a worktree and a
target directory of their own (`eullm-ml`, `eullm-target/model-libraries`):
it has to finish and the binary has to carry gfx90a code
(`amdhsa--gfx90a`, as the release checks it). Then the same `c14` test,
the new binary under the label `v080`.

It did, the same day: `build exit: 0`, `eullm 0.8.0 (ROCm) [4cf28c5d2989]`,
device code `amdhsa--gfx90a`, linked to `libamdhip64.so.6` and
`librccl.so.1`, built with this site's ROCm 6.3 (patch 0013).
The test was submitted next: the old binary `eullm 0.7.95 (ROCm)
[be0cf9ec87be]` built the same way (`eullm-v0795`, `eullm-target/v0795`),
the smoke job 23253715 on the new one, `c14` planned with 18 points per
binary and one job of 8 hours each (23253735 `v0795`, 23253737 `v080`), the
round's waiting jobs held (`JobHeldUser`) and 23253738 on `debug`, waiting on
both, to release them.
The smoke passed on the new binary: one MI250X GCD found (gfx90a, wave
size 64, 65,520 MiB), Qwen3-8B (4.68 GiB) loaded whole on it by `--fit`,
5.9 GB of HBM in use after the request (llama.cpp's breakdown: 4,455 MiB of
model, 576 of context, 108 of compute).

29-30 October 2026, no way in: the EuroHPC SSH certificate service
(`sshca.my-eurohpc.eu`, MyAccessID through eIDAS) stopped issuing
certificates to the PI's Italian identity. Italy was missing from the eIDAS
country list on the evening of the 29th; on the 30th it was back, but SPID
with PosteID answered "Impossibile stabilire l'autenticità della richiesta di
autenticazione", and CIE has always answered that the message does not meet
its security requirements. A SPID identity with another provider (Aruba)
logs in through eIDAS but is a new person to MyAccessID: the Italian eIDAS
identifier differs from one SPID provider to the next. Asked of the EFP
helpdesk: to link it to the existing account rather than open a second
one. The jobs went on without anyone logged in (the `c14` test, the job
releasing `r-v0794`'s chains, then the chains). Once back in, a second
identity is linked to the MyAccessID profile so that one provider failing
does not lock the project out.

`tools/lumi/log_checks.py`, in every job summary from now on, reads the
server logs for the cache's windows and totals, the pool's flushes
("pool[...] freeing"), HSA errors, out of memory and NaN.

25-10-2026, evening, `mtp-ab` on the twins, the first measurement of MTP on
the MI250X with drafts (jobs 23140145 on `next25` and 23140146 on `next26`,
`standard-g`, about ten minutes each; the two `small-g` copies were
cancelled). 100 gsm8k questions one at a time, temperature 0, tokens/s
`next25` / `next26`, drafts kept:

| setting | Qwen3.5-9B Q4_K_M | kept | Qwen3.6-27B Q4_K_M | kept |
|---|---|---|---|---|
| no `--mtp` | 98.9 / 97.7 | | 38.3 / 38.1 | |
| `--mtp 1` | 136 / 133 | 92.7% / 92.7% | 55.8 / 56.2 | 94.3% / 94.3% |
| `--mtp 2` | 150 / 147 | 87.9% / 87.9% | 63.1 / 63.4 | 89.4% / 89.4% |
| `--mtp 3` | 125 / 124 | 81.0% / 80.7% | 55.5 / 55.4 | 84.8% / 84.8% |

3d1c7a0, the head started after the prompt instead of before it, changes
nothing measurable: the share kept is the same to 0.3 points at all six
settings, tokens/s within 2% (the advice from the Leonardo chat answered:
acceptance 81-94% before and after, the same). And MTP pays on the MI250X:
`--mtp 2` is the best on both, +52% on the 9B and +65% on the 27B, as on the
RTX 5070 Ti; `--mtp 3` drafts more than the model keeps. Accuracy moved by
one or two questions of the hundred either way (9B 0.98 without, 0.96-0.99
with; 27B 0.94-0.95), the numeric spread of a token checked among several.
The draft context costs 0.7-0.8 GB on the 9B and 1.0-1.3 GB on the 27B.
`c06` on `next27` measures the six models as pulled, which now draft, and
at the default sampling temperature.

`c11`'s `load-*` on `next25` (job 23137402, one cold load a point, all eight
GCDs, layer split in the same job): seconds to load with 16 threads reading
each weight (the default since `next25`), 32 and 64, and by layer; reading
in brackets.

| model | 16 | 32 | 64 | layer |
|---|---|---|---|---|
| gpt-oss-120b MXFP4 (58 GiB) | 38.6 (32.5) | 43.4 (37.1) | 41.5 (35.0) | 48.6 |
| Qwen3-235B Q4_K_M (132 GiB) | 92.7 (79.0) | 103 (88.7) | 102 (88.2) | 120 |
| Qwen3-Coder-480B Q4_K_M (270 GiB) | 131 (104.5) | 115 (89.6) | 113 (87.5) | 163 |

16 threads stay the default: more are 7-12% slower on the two smaller
models and 12-14% faster only on the 480B, one load each. A tensor load is
now faster than one by layer for all three, by a fifth (20-23% with 16
threads), where `next21` took 3 to 5 times as long; copying stays 8-27 s.
Split as tensor, the same servers also wrote more than by layer:
gpt-oss-120b 1,570 tokens/s against 1,312, the 235B 390 against 222, the
480B 270 against 101 (decode 14.1 against 10.8, 6.5 against 3.6, 4.4
against 1.6), one measurement each.

`c12`'s `days-*` and `fill-models` on `next25` (jobs 23088874 and
23088875), 32 requests on one GCD each: **no server died in any of the
sixteen points, and the VRAM stayed flat in all of them (0.00 GiB an
hour).** Eight soaks of 45.9 hours, the configurations that died within
hours with HIP graphs: Qwen3-32B Q4_K_M four times (158-173 tokens/s,
427,000-468,000 requests each), its Q8_0 (138, 371,138), Qwen3-30B-A3B Q8_0
(229, 697,103), Qwen3.6-35B-A3B (313, 191,679), Qwen3-14B Q8_0 as control
(268, 816,795). Eight more of 35 hours on the GCDs they left free:
Qwen3-4B, 8B and its Q8_0, 14B, Qwen3-30B-A3B Q4_K_M, Qwen3.6-27B, Mistral
Small 24B, gpt-oss-20b (146-711 tokens/s). About nine million requests in
all, accuracy as on short points (gsm8k 0.91-0.96). ROCm without HIP graphs
is settled.

The share of answers written word for word the same when a question comes
back in a later pass is 0.75-0.89 for Qwen3 and Mistral and far lower for
Qwen3.6-35B-A3B (0.23), gpt-oss-20b (0.35) and Qwen3.6-27B (0.49). It
follows the length of the answers (tokens/s times hours over requests):
about 55-70 tokens for the first, 190-270 for the others. With 32 requests
the batch around an answer differs from pass to pass and so do the sums by
the last bits; the longer an answer, the likelier it turns somewhere. The
accuracy does not move with it.

24-10-2026, the MTP half of 0.7.93 (3d1c7a0: the draft head is started after
the prompt is read, not before, which is what the warning `ctx_dft
pos_max=-1` on every request said) measured before and after: `c06`'s
`mtp-ab`, Qwen3.5-9B and Qwen3.6-27B Q4_K_M, without `--mtp` and with 1, 2
and 3, 100 gsm8k questions one at a time at temperature 0, planned once on
`next25` (built 22-10, before it) and once on `next26` (after), one node for
each, about half an hour. The runner records the drafts kept from now on
(`draft_accept_pct`, `kept%` in the report). If the share kept falls after
the fix, the fix is looked at again.

`next25`'s half (job 23137402) came back with no `kept%`: no draft in any of
its eight points, `--mtp 3` included, and the same tokens/s and VRAM with and
without it (9B 97.7 and 96.9-99.5, 27B 37.6 and 37.9-38.2). The cause is in
the store, not in either engine: unsloth's MTP repositories ship
`mmproj-*.gguf`, `eullm pull` fetches the projector with the weights (since
29-09), and a model with a projector is loaded on the sequential engine,
since the scheduler does not route images, and only the scheduler drafts.
`--mtp` is then a flag that does nothing and says nothing. Every `c06` point
of every round was such a point. `tools/lumi/text_only_model.sh` makes
`<id>-text`, the same weights linked under another id with the manifest
without the projector, and `c06`'s `text_only` has `campaign_setup.sh` make
them; `mtp-ab` runs on those, on `next25` and `next26` again, in the jobs
of each label already running. The `mtp-ab` points of `next26` on the
models as pulled, which would have measured nothing, come off the queue.
The engine has to say so too: `--mtp` on a model that will not draft is
logged as on (the banner) and is not.

The three jobs for this (23136196 for `ram-default`, 23137402 and 23137404
for `mtp-ab`) waited on priority with the script's 48 hours for a few hours
of work. Lowered to 6, 6 and 3 hours (`scontrol update JobId=J
TimeLimit=6:00:00`, which a user may do to a waiting job of their own),
23136196 started within the minute: Slurm fits short jobs into the gaps
before the nodes it holds for others, and the runner reads the time it has
from `squeue` and takes only the points that fit. The 48 hours were
`sbatch_campaign.slurm`'s, set for the long chains that drained the first
rounds and kept for every submission after, short rounds too. Since this
day `submit_campaign.sh` asks for the time the label's waiting points need,
spread over the chains, plus the longest point, half as much again and half
an hour, in whole hours up to 48 (`campaign.py walltime`): 2 hours for
`mtp-ab`, 5 for `ram-default`, 48 for a full round or the two-day soaks.

Unused budget is the outcome that has to be explained afterwards; used budget
explains itself if every hour left a measurement behind. That is the design
rule for everything below: the node is never idle, and nothing runs on it
that does not produce a result file with its provenance.

Two facts about LUMI shape how:

- **Single-GCD work cannot spend this allocation.** `small-g` and `dev-g`
  bill 0.125 node-hours per GCD-hour: one GCD busy until March is ~470
  node-hours, a tenth of the budget. The rest has to be whole nodes on
  `standard-g` (48 h walltime).
- **A whole node bills all eight GCDs** whatever runs on it. A job measuring
  one configuration at a time — what every script before the campaign runner
  did — pays for seven idle GCDs.

`small-g` billing: the charge is the largest of GCDs, cores/8 and memory/64
GB. A one-GCD job asking for 16 cores or 128 GB pays for two.

## How the node is kept busy

[`bench/campaign/`](../../bench/campaign/README.md) — tested against a
stand-in engine in CI, run on LUMI by `tools/lumi/sbatch_campaign.slurm`:

- **A queue of points on scratch**, expanded from campaign specs
  (`tools/lumi/campaigns/`). Several jobs drain it at once — two nodes now —
  and a job that hits the walltime puts its unfinished points back.
- **Every GCD always has a point.** Single-GCD points run eight at a time;
  wider points take aligned groups (a pair on one MI250X module, half the
  node, all of it); the next point starts the moment devices free up. A wide
  point that is waiting reserves its devices and narrower ones only take them
  meanwhile if they will be done in time.
- **Workload points stretch to fill.** Sustained-load points run between a
  minimum and a maximum duration and take exactly the time that is free,
  which is what keeps the end of each 48-hour job from being billed idle.
- **Every server is pinned** to the seven cores LUMI documents as closest to
  its GCD, and every result records the cores, the other points on the node
  at the time, and `neighbours-control` measures the same points alone — so
  packing is itself measured rather than assumed harmless.
- **Each job leaves its own usage evidence**: `<job>.summary.json`, the share
  of the job each GCD had work, and `<job>.node.jsonl`, the raw HBM and
  utilisation samples. That is the page of the Final Report that says the
  hours were used.

Workload points are not filler. They send the public, pinned GSM8K, ARC-Easy
and ARC-Challenge sets at a fixed concurrency for hours, grade the first
pass, compare every later pass with it (greedy decoding, prompt cache off: an
answer that changes under load is a bug), and record throughput, latency and
HBM per minute. That is objective (6), stability under sustained load, plus
the accuracy each memory and speed configuration costs — the other half of
objective (1) that a throughput number alone does not give.

## What we committed to

The accepted proposal asks five questions and names the metrics it will be
read against (time-to-first-token, prompt-processing and generation
throughput, HBM use, model-loading time, scaling efficiency, stability under
sustained load; dense and MoE; one device to a full node).

| WP | question | campaign | engine work it waits on |
|---|---|---|---|
| 1 | size × quant × context → memory, throughput | c01 dense/moe/long-context, c02 quant | — |
| 2 | batching as concurrency grows | c01 batch axes; chunked prefill before/after | merge `feat/engine-roadmap` |
| 3 | multi-GPU without disproportionate overhead | c01 ref/scale, c02 235B split vs 2×4 | `--split-mode` incl. tensor |
| 4 | CUDA vs ROCm | the same specs on Leonardo / JUPITER | — (runner is site-neutral) |
| 5 | loading and on-prem transfer | cold/warm load in every result; c02 large MoE | `--moe-cache` on HIP |
| 6 | stability under sustained load | soak groups, 2-24 h | `/metrics` (nice to have) |
| 7 | multi-node ("may be evaluated") | c04, bounded | RPC backend build |

## Campaigns and the budget

`campaign.py plan` prints what each spec costs at most:

| spec | points | node-h | needs |
|---|---:|---:|---|
| `c01-node-baseline` | 205 | ~115 | catalog models only: **runs today** |
| `c02-quant-large-moe` | 75 | ~130 | ~1.2 TB pulled from Hugging Face first |
| ~~`c05-finetune`~~ | — | — | withdrawn from LUMI on 06-10-2026, see below |
| `c06-mtp` | 48 | ~4 | MTP GGUFs (~114 GiB): speculative decoding, objective (5) |
| `c07-runtimes` | 110 | ~6 | llama-server and Ollama (`install_runtimes.sh`): the reference frame for (1) and (2) |
| `c08-decision` | 128 | ~4 | Jev-Style releases: `/v1/systemone` under concurrency, objective (2) for decisions |
| `c09-load` | 45 | ~17 | cold loads from Lustre with and without `--load-threads`, objective (5); a runner of 07-10-2026 or later |
| `c03-split-modes` | 72 | ~38 | `--split-mode` layer and tensor on 2, 4 and 8 GCDs (Qwen3-32B, Qwen3.6-35B-A3B, Qwen3-235B), objective (3); an engine of 15-10-2026 (`next12`); the `rccl-*` groups (40 points) only on an RCCL build, in jobs labelled `rccl` |
| `c10-moe-ram` | 78 | ~57 | large MoE on one GCD with the experts in the node's RAM (`--moe-cache`, `--cpu-moe`, `--fit`'s own split), objective (5); an engine of 14-10-2026 (`next11`) and its runner |
| `c11-placement` | 44 | ~26 | a whole node of 8 GCDs: replicas of a smaller split against one wider one, layer and tensor, at the same total of requests (Qwen3-235B, gpt-oss-120b, Qwen3-Coder-480B, DeepSeek-V3.1 on layer split only, Qwen3-32B as control), objective (3); an RCCL build (`next16`) |
| `c12-stability` | 50 | ~79 | the soaks that crashed while working (Qwen3-32B Q4_K_M and Q8_0 with f16 and q8_0 KV, Qwen3-30B-A3B Q8_0, at 32 requests on one GCD) again, with llama.cpp's own log on and a runner that records a server that dies, Qwen3-14B Q8_0 as control; an engine that stops cleanly (`next17`); six hours was too short, the `long-*` groups run 12-16 h, Qwen3-32B twice, one node; the `days-*` groups two days on `next25` |
| `c13-sites` | 14 | ~1 | the same points on LUMI-G, JUPITER and Leonardo, one device each (`docs/sites-comparison.md`): Qwen3-32B at 32 requests for 90 minutes with CUDA/HIP graphs on and off, whether a CUDA server's VRAM grows as LUMI's did; Qwen3-8B, Qwen3-32B and Qwen3.6-35B-A3B at 1-32 requests; on LUMI in the `next21` job |

Measured honestly, the matrix the proposal describes is cheap: ~250
node-hours a pass, nearly all of it the soaks. What spends 4,500 is doing it
**again for every engine change** — which is the development cycle the
proposal describes ("frequent releases and daily code iterations… experiments
will track exact Git revisions"). A **round** is the same specs planned under
a new label (`ROUND=<engine version> campaign_setup.sh`): every point measured
again, so each engine change has a before and an after on identical
workloads.

| item | node-h |
|---|---:|
| round 1: c01 + c02 on the released 0.7.20 build | ~250 |
| rounds on engine milestones, ~1 every 9 days (~14 × ~250) | ~3,500 |
| c03: row and tensor split, replicas × split, once the engine exposes them | ~300 |
| c04: multi-node exploration, 2-4 nodes, bounded | ~300 |
| iteration on small-g (dev-g for smoke tests of minutes only), reserve | ~150 |
| **total** | **4,500** |

A round needs a reason: a release, a merged branch that touches inference, a
llama.cpp bump, a build flag (RCCL, HIP graphs) — or, once, a deliberate
same-binary repeat to measure day-to-day and node-to-node spread. The soak
`max_duration_s` is the other lever if rounds run short of the pace.

## Engine work, before the hours (no allocation needed)

The four gaps [`lumi-g.md`](lumi-g.md) found by reading the code are still
open on `main` at 0.7.20, and each one ends a round with a before/after.

0. **Merge `feat/engine-roadmap`.** It is 22 commits ahead of `main` and
   carries measured prompt and answer times in every response (`5febfeb` — on
   `main`, `prompt_eval_duration` is hard-coded to 0, so the runner's
   server-side prefill rate is null until this lands) and chunked prefill
   between decode steps (`04d08ca`, roadmap 0.7-D), an objective (2) result.
1. **The banner must report the live backend, not the compiled one.** A
   `rocm` binary that finds no device still prints `GPU backend: ROCm` and
   then bills node-hours at CPU speed. The runner's `--fit-strict` default
   catches the out-of-memory side of this; nothing catches a missing device.
2. **`--split-mode`, `--tensor-split`, `--main-gpu`, device selection in
   `RuntimeOpts`.** The vendored bindings expose all of it, including
   llama.cpp's **experimental `Tensor` split** — real tensor parallelism, the
   one arrangement that could make a *single* request faster across GCDs. The
   12-09 conclusion that splitting adds no throughput was measured on layer
   split only. This is c03. Exposed on 15-10-2026 (`--split-mode`,
   `--tensor-split`, `--main-gpu`, `crate::gpu_split`; no `none`, since one
   GPU is chosen by making only it visible), and `c03-split-modes` measures
   layer and tensor on 2, 4 and 8 GCDs at 1 and 16 requests. Row split is
   not there to measure: every load failed (`r-split`, next12), and so does
   stock llama-bench, "device ROCm0 does not support split buffers"; the
   CUDA backend that ggml builds for ROCm no longer has them (only SYCL and
   Hexagon do), so `row` and `--main-gpu` were dropped from the engine.
3. **The harness (WP0)** — done in `bench/campaign`: one schema
   (`eullm.bench/1`), TTFT, prefill and decode rates, cold/warm load, HBM per
   device, repeats with their spread, and provenance (engine version and
   binary hash, repository revision, ROCm version, cores, neighbours). It
   runs on CUDA unchanged (`--backend cuda --bind none`); what is missing is
   the `sbatch` wrapper for Leonardo and JUPITER, a copy of
   `sbatch_campaign.slurm` with their partitions. A `/metrics` endpoint
   (roadmap 0.7-B) would let a soak see queue depth over time, not just at
   the end.
4. **A placement policy: the smallest split that fits, then replicas.** The
   12-09 numbers give the rule (replicas 4.07× on four GCDs, layer split
   0.98×); c01 and c02 measure it on more models. Turning it into `--fit` on
   a per-device plan, with the engine serving N pinned replicas behind one
   endpoint, is roadmap 1.0-B arriving early, for a measured reason.
5. **NUMA binding** — the runner already pins every server; a round with
   `--bind none` is the before/after that says whether the engine should do
   it itself.
6. **An RCCL build**, `GGML_HIP_RCCL=ON`, and `GGML_HIP_GRAPHS`: experiment
   binaries, each one a round, never the default. RCCL has a measured
   reason since 15-10-2026: tensor parallelism beyond two GCDs reduces
   through it or through ggml's generic path (item 5 of the measured work,
   below). `tools/lumi/build_engine.sh` links it by default since
   16-10-2026, once c03 had measured it (`EULLM_RCCL=0` leaves it out), and
   fails a build that does not link `librccl`.
7. **`--moe-cache` on HIP.** It refuses any non-CUDA device and more than one
   GPU, and its pinned-memory patch is CUDA-only. Objective (5) on MoE.
8. **Loading from Lustre: `--load-threads`** (07-10-2026, `c09`). c02's
   Coder-480B (270 GiB) had not loaded after an hour, and `dd` reading 2 GiB
   pieces of it on a compute node gave 178 MB/s with one stream and 2,283
   MB/s with sixteen (`sbatch_lustre_probe.slurm`). Threads reading the model
   ahead of llama.cpp did not pay: cold, a 132 GiB model loaded in 110 s
   without them and 128 s with 16, since llama.cpp alone reads at 0.7-1.3
   GB/s. The flag is off by default. The hour lost on the 480B was the file
   being mapped: watched on a whole node (`sbatch_load_watch.slurm`, 07-10 and
   08-10-2026), Lustre read it at about 2 GB/s and the VRAM was reserved
   after 3 minutes, then one thread at 95% of a core copied it to the GPUs at
   about 90 MB/s, still unfinished after 70 minutes, the page cache stuck near
   half the RAM. With `--no-mmap` the same model loaded in 151.8 s. c09's
   `load-nommap` groups measured it on every model (cold, seconds):

   | model, GCDs | mapped | read in |
   |---|---:|---:|
   | Qwen3.8-27B Q8, 29 GiB, 1 | 46 | 51 |
   | Qwen3-235B Q4_K_M, 132 GiB, 4 | 110 | 108 |
   | Qwen3-Coder-480B Q4_K_M, 270 GiB, 8 | failed 6 of 6 | 160, 208 |
   | DeepSeek-V3.1 Q4_K_M, 8 | failed 3 of 3 | 226 |

   The tokens per second were the same both ways. The engine now reads a
   model in by itself when it goes to the GPUs whole and its files are more
   than half the memory the process may use (`fit::read_in_whole_on_gpu`);
   `--mmap` keeps it mapped. The `moe-` groups of c02 that failed loading can
   be planned again on that engine: `r-next4` is that round (below). The 480B
   and the 671B loaded read in, in 91 to 284 s; the 235B, still mapped by the
   rule, took 11-32 minutes to load where two servers mapped it at once, and
   failed after the runner's hour three times. Since 13-10-2026 a model that
   goes to the GPUs whole from a network file system is read in whatever its
   size: c09's `load-together-*` groups load several servers of one file in
   14-121 s that way, in 343-2,448 s mapped (below).

Items 0-2 are October. Items 4-6 December. Item 7 January.

## The CUDA half

Leonardo was not awarded for this project. In order:

1. **Leonardo (again) and JUPITER**, requested by the PI in early October,
   decision pending. The campaign runner and the specs run there as they are;
   only the `sbatch` wrapper changes.
2. **Leonardo AI-Factory, before 02-11-2026.** Measuring how the legal-it
   GGUFs that allocation produced serve on an A100 is evaluation of its own
   output, and its queue needs filling too.
3. **The A100 numbers already held** ([`../cineca/leonardo.md`](../cineca/leonardo.md)),
   measured by hand on 04-09-2026 before two llama.cpp bumps.
4. **The RTX 5070 Ti** — the "smaller on-premises system" of objective (5).

## Models

Permissive licences only, as everywhere in the project, checked against the
Hub on 05-10-2026: Qwen and gpt-oss Apache-2.0, DeepSeek-V3.1 MIT. No Llama.

- **c01, catalog** (Q4_K_M): Qwen3 4B/8B/14B/32B, Mistral-Small-24B,
  Qwen3.6-27B, Qwen3.6-35B-A3B; plus the 12-09 reference model,
  `unsloth/Qwen3.8-27B-GGUF` UD-Q8_K_XL (29.3 GiB), so its rows repeat
  exactly. It was gone from LUMI by 05-10 and is pulled again; the engine now
  names it `qwen3.8-27b-gguf-ud-q8_k_xl`, the 12-09 rows
  `qwen3.8-27b-ud-q8_k_xl` — the same file.
- **c02, from Hugging Face**: Qwen3 8B/14B/32B Q8_0 against c01's Q4_K_M;
  Qwen3-30B-A3B Q4_K_M and Q8_0; gpt-oss-20b and -120b (MXFP4, 11 and 59
  GiB); Qwen3-235B-A22B Q4_K_M (132 GiB: 4 GCDs, so one split or two
  replicas of it) and Q8_0 (233 GiB); Qwen3-Coder-480B-A35B Q4_K_M (270 GiB);
  DeepSeek-V3.1 Q4_K_M (378 GiB — only a whole node runs it at all).

About 1.3 TB on `/scratch` in total, inside the 4 TB asked for. Compute nodes
have no network: `campaign_setup.sh` pulls on a login node and leaves the
missing ones blocked, never failed.

The workload sets are GSM8K, ARC-Easy and ARC-Challenge (4,867 graded
questions). MMLU is not: its pinned source,
`people.eecs.berkeley.edu/~hendrycks/data.tar`, answers 404 as of
05-10-2026 — which also breaks `sbatch_autobench.slurm`'s default `SETS`.

## Calendar

Re-paced on 21-10-2026 on what `lumi-allocations` bills: 4,135 of 18,000
GPU-hours (1,034 of 4,500 node-hours, 23.0%), 54 node-hours ahead of the
straight line, 3,466 node-hours left. The straight line (24.5 node-hours a
day to 12-03-2027) is no longer the pace: the PI's decision of 21-10-2026
is to run what the work needs and ask for more hours if they run out
early. The table is what that need looks like now, a forecast and not a
cap: a full round on an engine that touches inference is about 400
node-hours at most on four nodes (`r-next21`), and the engine is expected
to change about every two weeks. `status.sh` prints on every call where
the spend stands and on which day it runs out at the last 24 hours' rate
(`campaign.py budget --allocations`).

| period | days | node-h | GPU-h | engine | on the machine |
|---|---:|---:|---:|---|---|
| 21-31 Oct | 10.6 | ~450 | ~1,800 | `next21`: ROCm without HIP graphs; tensor loads (patch 0010) | the whole round `r-next21` (`c01`-`c03`, `c06`-`c11`) on four nodes; `c13-sites`; `soak-node`, 24 h |
| Nov | 30 | ~800 | ~3,200 | a release with the ROCm fix; placement policy | a round on each release, about two |
| Dec | 31 | ~900 | ~3,600 | served replicas, RCCL | a round; **long soaks queued 20-12 → 06-01**, when nobody is watching and the queue still runs |
| Jan | 31 | ~800 | ~3,200 | load path | a round; `c04` multi-node |
| Feb | ~15 | ~516 | ~2,064 | release candidate | what is left: about mid-February at this pace |
| **total** | | 3,466 | 13,865 | | |

25-10-2026 (`nodi.sh`): 619.5 node-hours in the last 7 days, 88.5 a day,
the week of the whole `r-next21` round on four nodes and the two-day soaks;
392 of them since 21-10, most of the ~450 forecast for 21-31 October. 32.7 a
day since 12-09. Left 3,074 node-hours: 22.7 a day to 09-03-2027 (the plan's
end), about one node around the clock. At last week's rate they run out on
29-11-2026; at the forecast's 27-29 a day for November to January, about
mid-February. Today nothing ran: the `standard-g` queue held 362 jobs of
higher priority than ours, so for now the queue, not the budget, sets the
spend.

At that pace the allocation ends around mid-February, a month early, and
the final round, the data for Zenodo and the Final Report's measurements
need hours after it. The request for more goes in by the January cut-off,
earlier if `status.sh`'s run-out date comes sooner: Development Access
calls have monthly cut-offs and an allocation can be renewed (EuroHPC's
access policy and FAQ say how; to be checked before writing it). The case
is the record this plan keeps: every hour a measurement with its build and
its node, and the crashes found and fixed with them.

## The engine's trainer (`c05-finetune`): withdrawn from LUMI

**Not run on this allocation, as of 06-10-2026.** The JUPITER proposal
(EHPC-AIF-2026PG01-1434, submitted to the same Joint Undertaking at the end
of September) describes this allocation in writing: *"This allocation covers
inference-engine work only; no model training runs on it."* A trainer
benchmark is engine work, but it trains weights, and a statement made to
EuroHPC is not reinterpreted after the fact. The `c05` points were taken off
the queue on 06-10; whatever ran on the night of 05-10, before the conflict
was noticed, is reported as such in the Final Report and not used. The spec
stays in the repository for a machine where training is declared.

What it was, for the record. `eullm finetune` is new engine code: llama.cpp's trainer (ggml-opt), which
the engine did not expose, behind one command that trains an F32 GGUF on a
text file and writes a GGUF the engine serves. `c05` measures it on one GCD
the way the other campaigns measure inference: tokens per second, HBM, and
the does-not-fit boundary by model size (Qwen3 0.6B/1.7B/4B Base), optimizer
(AdamW, SGD), training window (512-2048) and which tensors train (all, or
attention only); then whether the held-out loss falls as it should over
three epochs at three learning rates. The text is GSM8K's training split
(public, MIT); the trained models are deleted when the point ends.

It was planned as software engineering and benchmarking of the runtime, on
public data, with no model as an output; the sentence above settles it.

## Engine work the hours measure (decided 06-10-2026)

With training off this allocation, what fills it is what the proposal asked
for: engine changes, each measured before and after on identical workloads.
In order:

1. **MTP** (`c06`): speculative decoding with the model's own head on
   MI250X, graded at temperature 0 (answers must not change) and at default
   sampling. Objective (5).
2. **The reference frame** (`c07`): the same points on llama-server (same
   llama.cpp commit, without EuLLM's runtime) and on Ollama. What EuLLM's
   scheduler and batching add or cost is the difference. vLLM after a
   like-for-like design (it does not serve GGUF) and a check that its ROCm
   container runs on gfx90a.

   What it found (07-10-2026): level at one request, 8-10% behind at four,
   11% (the MoE) to 36% (8B, 14B) behind at sixteen. The cause, found on
   08-10-2026 with the scheduler's `steps:` lines and
   `tools/lumi/sbatch_concurrency_diag.slurm` (both servers on one GCD, the
   same sampling options): EuLLM's default repeat penalty 1.1 made llama.cpp
   look up all 151,936 tokens of Qwen3's vocabulary for every token of every
   answer, 0.7 ms each on LUMI's CPU, so 11 ms of every step at sixteen;
   llama-server's default has the penalty off. llama.cpp patch 0005 penalizes
   the recent tokens in place (1,196 to 2 µs per token). Qwen3-8B Q4_K_M, one
   GCD, 256 tokens per answer, tokens/s, after the patch (job 22653307):

   | requests | profile | EuLLM | llama-server |
   |---:|---|---:|---:|
   | 1 | each server's defaults | 111 | 112 |
   | 4 | each server's defaults | 221 | 228 |
   | 16 | each server's defaults | 621 | 676 |
   | 16 | penalty 1.1 on both | 630 | 476 |
   | 16 | penalty off on both | 631 | 666 |
   | 16 | greedy | 636 | 701 |

   What is left at sixteen (5-9%): the steps carried 14.9-15.4 sequences on
   average, not 16, because EuLLM reads one waiting prompt per step and a
   round of sixteen starts over sixteen steps, where llama-server reads them
   in one batch; sampling is 12% of a step (0.19 ms per token, on one
   thread). The graded workload's distance (14B 99 against 315 tokens/s) is
   larger than either explains. The likely reason, found on 09-10-2026: with
   a KV cache per slot, llama.cpp runs the model once per run of consecutive
   slot numbers in the order a step lists them (`split_equal`, sequential),
   and EuLLM listed them in the order answers had ended and slots been taken
   again. With requests arriving and ending all the time, as in the graded
   workload, steps took 3.5 to 4 passes on four CPU cores (Qwen3-0.6B, eight
   clients one request after another); in slot order, 1.0 to 1.4, and 60-71
   tokens/s instead of 41-43. The `steps:` lines now count the passes, and the
   diagnosis has closed-loop rounds (`--duration`) and `--kv-unified` to
   measure it on the GCDs.

   On the GCD (job 22663207, 09-10-2026, Qwen3-14B Q4_K_M, the engine with
   both fixes), tokens/s:

   | | EuLLM | EuLLM `--kv-unified` | llama-server |
   |---|---:|---:|---:|
   | 4 clients, one request after another | 145 | 144 | 139 |
   | 16 clients, one request after another | 377 | 391 | 340 |
   | 16 at once, each server's defaults | 449 | | 462 |
   | 16 at once, penalty off on both | 454 | | 456 |
   | 16 at once, penalty 1.1 on both | 454 | | 353 |
   | 16 at once, greedy | 457 | | 474 |

   Closed-loop steps took 1.03-1.10 passes on average (1.00 with
   `--kv-unified`), the prompts read between steps 6-7% of the time.
   The closed loop is a stand-in for the graded workload (short raw prompts,
   answers of 32-384 tokens): a `c07` round on this engine is the check.
   What is left at sixteen at once (3%): the round's prompts start over
   fifteen steps, 15.2 sequences per step on average where llama-server has
   16 from the first. Since 10-10-2026 a step reads the waiting prompts
   itself, beside the answers' tokens, as llama-server does. The diagnosis
   with the engine before it on the same GCD (job 22691743, 10-10-2026,
   Qwen3-14B Q4_K_M, tokens/s, the mean of two rounds at once):

   | | EuLLM before | EuLLM | EuLLM `--kv-unified` | llama-server |
   |---|---:|---:|---:|---:|
   | 16 at once, llama-server's chain on both | 447 | 480 | 447 | 462 |
   | 16 at once, greedy | | 480 | | 472 |
   | 16 at once, each server's defaults | | 469 | | 460 |
   | 16 at once, penalty 1.1 on both | | 478 | | 359 |
   | 16 clients, one request after another | 369 | 366 | 400 | 348 |
   | longest wait for a first token, 16 at once | 0.65-0.84 s | 0.035-0.39 s | 0.034-0.28 s | |
   | longest wait for a first token, 16 clients | 0.94 s | 0.30 s | 0.22 s | |

   A step now carries all sixteen sequences in one pass (16.0 sequences,
   1.00 passes per step) from the round's first. The longer first-token
   waits (0.39 and 0.28 s) are each server's first round, in which the
   prompts are read from scratch; after it the slots hold them. At four,
   all four runs are level (150-153 tokens/s). In the closed loop the
   steps take 1.07-1.17 passes with a KV cache per slot, from the slots
   left idle between a client's requests and the prompts' passes of their
   own; `--kv-unified` has none (400 tokens/s) but reads every slot's cells
   in each answer's attention, which costs it 7% with sixteen at once.

   The `c07` round on that engine (`r-next6`, job 22693068, 10-10-2026, one
   GCD per point, Q4_K_M) against the rounds before it. The graded workload,
   sixteen clients over 400 gsm8k questions, tokens/s (accuracy 0.92-0.96
   everywhere, within the noise of 400 questions):

   | model | EuLLM `r-clean` | EuLLM `r-next3` | EuLLM `r-next6` | llama-server | Ollama |
   |---|---:|---:|---:|---:|---:|
   | Qwen3-8B | 154 | 517 | 533 | 486 | 166 |
   | Qwen3-14B | 102 | 327 | 349 | 317 | 142 |
   | Qwen3-32B | 54 | 186 | 191 | 178 | 105 |
   | Qwen3.6-35B-A3B | 118 | 219 | 221 | 199 | 28 |

   Sixteen requests at once (answers of 150 tokens), tokens/s and the median
   wait for the first token:

   | model | EuLLM `r-next3` | EuLLM `r-next6` | llama-server `r-next6` |
   |---|---:|---:|---:|
   | Qwen3-8B | 613 (205 ms) | 695 (33 ms) | 646 (274 ms) |
   | Qwen3-14B | 423 (306 ms) | 474 (192 ms) | 450 (312 ms) |
   | Qwen3-32B | 234 (600 ms) | 263 (104 ms) | 248 (502 ms) |
   | Qwen3.6-35B-A3B | 244 (445 ms) | 246 (163 ms) | 229 (1963 ms) |

   The distance `c07` found is closed on every model: EuLLM is 7-10% ahead of
   llama-server on the graded workload and 5-8% with sixteen at once, where
   reading the prompts in the step gained 12-13% on the dense models and 1% on
   the MoE. The `r-clean` rows are the engine with the repeat penalty's whole-
   vocabulary scan and the steps out of slot order (0.7.50 fixed both).

   `--kv-unified` against a KV cache per slot (`rt-kvu*`, same job), tokens/s:

   | | Qwen3-8B | Qwen3-14B | Qwen3-32B | Qwen3.6-35B-A3B |
   |---|---:|---:|---:|---:|
   | 4 at once | 232 / 226 | 151 / 149 | 74.5 / 75 | 155 / 154 |
   | 16 at once | 689 / 645 | 472 / 455 | 264 / 246 | 268 / 262 |
   | 16 clients, graded | 531 / 548 | 347 / 362 | 193 / 200 | 218 / 221 |
   | 4 slots of 32k, 16k-token prompts | 13 / 8.1 | 7.5 / 4.1 | | |

   One cache for all gains 1-4% where requests come and go and slots are left
   idle between them, loses 2-7% with sixteen at once, and 38-45% with long
   contexts, where every answer's attention reads all the slots' cells: it
   stays off by default, an option for many short requests.
3. **Decisions** (`c08`): `/v1/systemone` with the Jev-Style releases. The
   engine runs one decision at a time per server; `c08` measures the queueing
   that causes as concurrency grows, and replicas as today's way round it.
   Batching decisions in the engine is the change it is the "before" of:
   since 09-10-2026 `batched` requests that arrive together are evaluated in
   shared decode calls (option 2 of three: only for clients that ask for
   `batched`, so the default stays reproducible bit for bit); a `c08` round on
   that engine is the "after".

   What it found (`r-next5`, 10-10-2026, Jev-Style 0.8B v3, `batched`,
   decisions/s before and after, and how many requests went together):

   | request | 4 at once | 16 at once | 64 at once |
   |---|---:|---:|---:|
   | 256-token state, 8 questions | 8.9 → 11.9 (2) | 9.3 → 11.9 (12.8) | 10.2 → 12.0 (14.8) |
   | 1024-token state, 8 questions | 7.1 → 7.9 (2) | 6.6 → 7.5 (8.1) | 6.5 → 7.5 (8.9) |
   | 64 questions | 1.3-1.4 → 1.5-1.6 (2) | about the same | about the same |
   | 4096-token state | about 3 (1-3) | about 3 | about 3 |

   The median wait at 16 and 64 at once fell by 15-20% where requests went
   together (1555 to 1329 ms, 6278 to 4996 ms). The gain is that small
   because the 0.8B is a hybrid model, whose rounds hold 16 questions at most
   (a recurrent state per sequence): two 8-question requests fill one, and a
   64-question request fills four on its own; the GPU time per question did
   not fall (97 ms for one 8-question request, 1161 ms for 12.8 together).
   `shared_prefix` and the 2B (read in blocks) are never grouped, and did not
   change. Requests evaluated together answered with other digits each time
   (`consist` 0.07-0.56 against 1 alone), as documented; whether the choices
   changed too, or only the digits, is what the decision points measure since
   11-10-2026 (`same choice`, and `max shift`, the most a probability moved).
   What a question costs on the 0.8B, about 10 ms of decode whether alone or
   not, is the next thing to look at.

   The decisions themselves (`r-choices`, next6, 11-10-2026): alone, `batched`
   decided the same every time and no probability moved. Among other requests
   a probability moved by up to 3 points (0.024-0.044) between repeats of one
   request, and the same request decided all its questions the same way in
   50-80% of its repeats (1 for 8 questions on a 1024-token state): the
   borderline questions turn over. The 2B, never grouped, was identical every
   time. Decision points count the share of questions decided otherwise
   (`flipped`) from the next round.

   The rounds of neighbouring lengths (`r-next7`, 11-10-2026, against
   `r-choices` on the same day), Jev-Style 0.8B `batched`, decisions/s:

   | request | 4 at once | 16 at once | 64 at once |
   |---|---:|---:|---:|
   | 256-token state, 8 questions | 11.5 → 12.3 | 11.8 → 13.4 | 12.0 → 13.7 |
   | 256-token state, 64 questions | 1.6 → 2.0 | 1.6 → 2.0 | 1.6 → 2.0 |
   | 1024-token state, 8 questions | 7.9 → 8.2 | 7.5 → 8.0 | 7.4 → 8.0 |
   | 1024-token state, 64 questions | 1.2 → 1.8 | 1.2 → 1.8 | 1.5 → 1.8 |
   | 4096-token state | about 3 and 1.1-1.2, as before | | |

   Alone, 64 questions decode in 546 instead of 623 ms (256-token state), 586
   instead of 670 (1024) and 808 instead of 898 (4096): 10-13%, where the
   passes fell from about 64 to 24. So a pass's own cost is not most of a
   question's; the tokens are: 64 questions are about 6,400 tokens of their
   own (the CPU bench's count), read at about 11,000 a second. Against no
   grouping at all (`r-next5`'s "before"), sixteen
   clients asking 8 questions on a 256-token state now get 13.4 decisions a
   second instead of 9.3, and `shared_prefix`, the default, 5.3.
4. **`--moe-cache` on HIP** (item 7 above): MoE experts in the node's host
   RAM, the direction Strata takes on consumer GPUs (the head-to-head with
   Strata stays on the RTX 5070 Ti, `docs/moe-offload-plan.md`). Objective
   (5). Brought forward on 14-10-2026: the cache and the copies ahead are
   ggml's CUDA backend, which ggml builds for HIP as `ROCm`, so the engine
   (`fit::moe_cache_support_on`) and patch 0003 accept a ROCm GPU now.
   `c10-moe-ram` serves gpt-oss-120b, Qwen3-235B, Qwen3-Coder-480B and
   DeepSeek-V3.1 from one GCD, alone on the node, the server on its 56 cores
   (the runner's `"cores": "reserved"`): with the cache, with every expert
   on the CPU (`--cpu-moe`), and as `--fit` splits them by itself, against
   `c02`'s splits over the GCDs whole. Next, phase 3 of the plan: the CPU
   computing the experts the cache misses while the GPU works, as Strata
   does, which suits a LUMI-G node: 64 cores and 512 GB of RAM beside its
   GPUs.

   First results (`r-moe-ram`, job 22846844, `next11`, 14/15-10-2026; the
   prompt groups and part of DeepSeek still to run). On ROCm the experts
   were pinned in every load, read into memory: 57 GiB of gpt-oss-120b,
   128 of the 235B, 263 of the 480B, 367 of DeepSeek-V3.1. Tokens/s on ONE
   GCD, the experts in RAM, against the same model on the GCDs it needs
   whole (`c02`):

   | model, requests | `--moe-cache auto` | `--fit`'s split | `--cpu-moe` | all on GPU |
   |---|---:|---:|---:|---:|
   | Qwen3-235B Q4_K_M, 1 | 21 | 6.3 | 3.7 | 29 (4 GCDs) |
   | 235B, 8 | 28.3 | 12.2 | 4.2 | 63.7 |
   | 235B, 32 | 12.2 | 15.1 | 9.1 | 98.4 |
   | Qwen3-Coder-480B Q4_K_M, 1 | 6.6 | 3.3 | 0.6 | 27.1 (8 GCDs) |
   | 480B, 8 | 3.7 | 7.3 | 3.0 | 50.2 |
   | 480B, 32 | 8.1 | 8.5 | timed out | 75.9 |
   | DeepSeek-V3.1 Q4_K_M, 16 | 2.7 | 5.7 | | 34 (8 GCDs) |
   | gpt-oss-120b MXFP4, 1 | 42.3 | 87.3 | 20.6 | |
   | gpt-oss-120b, 32 | 156 | 117 | 38.4 | 298 (2 GCDs) |

   One request on the 235B gets 72% of four GCDs' speed from one GCD with
   the cache, 3.3 times `--fit`'s split; with 32 the cache falls below the
   split, as each step routes the tokens to more experts than it holds. A
   model that nearly fits (gpt-oss-120b, 59 GiB) does best with `--fit`'s
   split alone, until 32 requests. Every expert on the CPU (`--cpu-moe`,
   all 56 cores) is the slowest in every row: the node's CPU computes the
   experts far slower than the link copies them, which phase 3 has to
   reckon with. `ram-stats` (runner 7) records the cache's step statistics
   to say where the steps with many requests go.

   What they said (`LLAMA_MOE_CACHE_STATS=64`, 15-10-2026). The link from
   the node's RAM to a GCD carried the cache's copies at 16-24 GB/s. On the
   235B with one request a step took 44 ms: 30 ms of the GPU computing the
   94 layers, 12 ms copying the 2.5% of experts not in VRAM (200 MiB a
   step). With eight, 8 tokens a step, 296 ms: 122 of computing and 170 of
   copying 3.7 GiB, 88% of the experts in VRAM. With 32, the cache found
   0.85% of the experts it was asked for: a step of 32 tokens wants more
   experts than the cache holds, cycling through the layers in the same
   order every step, which is the case where least-recently-used keeps
   evicting what the next step needs, so every expert went over the link
   (7.7 TB in the point). The 480B's working set is larger than its cache
   even for one request (81% in VRAM, 2.5 GiB a step, 100 of 150 ms
   copying). Two changes follow: the cache should stop evicting when a
   step wants more than it holds, keeping what it has and copying the rest,
   and the CPU's share is to be measured again (below) before phase 3.

   The eviction (15-10-2026). `bench/moe_cache_sim` replays the experts the
   routers choose through the cache's policies; on routing calibrated to
   these statistics (it copies 50.7 GiB a step with LRU on the 235B at 32,
   where LUMI copied 50.4) the eviction of llama.cpp patch 0008 keeps 72% of
   the experts in VRAM at 32 instead of 0%, against 81% for Belady's bound,
   and 57% for the 480B at 8 instead of 0%, the same as LRU wherever a step
   fits. On the 235B at 32 that is 14 GiB copied a step instead of 51, about
   0.7 s of copying instead of 2.4 at the measured 22 GB/s, so roughly
   three times the 12.2 tokens/s, above `--fit`'s split (15.1). The `evict`
   group of c10 (12 points, `next15`) measures it against
   `LLAMA_MOE_CACHE_EVICT=lru`, and its points write the routing of the
   235B and 480B at 1, 8 and 32 (`LLAMA_MOE_CACHE_TRACE`, a file per point
   beside its log) to replay the policies on real choices.

   Measured (`r-moe-evict`, `next15`, jobs 22882007 and 22882178,
   15/16-10-2026, one repeat each), tokens/s, the old eviction → the new:

   | requests | Qwen3-235B Q4_K_M | Qwen3-Coder-480B Q4_K_M |
   |---:|---:|---:|
   | 1 | 20.6 → 19.8 | 6.5 → 6.7 |
   | 8 | 28.2 → 26.9 | 3.8 → **7.3** |
   | 32 | 11.9 → **22.3** | 7.1 → 7.5 |

   Where the simulator saw a step overflow the cache (the 235B at 32, the
   480B at 8) the new eviction writes 1.9 times as fast; where a step fits,
   the two are level within one repeat's noise, as they should be, being
   the same eviction there. The 235B at 32 now writes more than `--fit`'s
   split did from one GCD (15.1, on the SSE2 kernels; `c10-v3` measures it
   again), nearly a quarter of what four GCDs whole write (98.4). The 480B
   at 32 hardly moves: a step wants more than twice the experts its cache
   holds, and the simulator gave it 28% of them in VRAM at best.

   The traces (16-10-2026), replayed, agree with the GCD's statistics within
   a few points (the 480B at 8: 57.6% of the experts in VRAM replayed, 56.9%
   measured), so the simulator can choose the next eviction without node
   hours (`bench/moe_cache_sim/README.md`). Counts that decay won where a
   step fit and lost where it overflowed; the two together (`lfu-pass`)
   were the best or level on all six traces: 235B at 8 from 88.9% to 91.0%,
   at 32 from 56.2% to 59.9%, 480B at 8 from 57.6% to 61.4%, against bounds
   of 95.3%, 63.8% and 73.9%. That is llama.cpp patch 0009 (`next16`), and
   c10's `evict2` group measures it against 0008's eviction in the same
   binary, with three repeats each, since one repeat's noise is about the
   size of the difference.

   Measured (`r-moe-evict2`, `next16`, 16/17-10-2026), tokens/s, 0008's
   eviction → 0009's: the 235B 22.0 → 22.1, 30.5 → 31.5, 22.6 → 23.6 at
   1, 8 and 32 requests; the 480B 6.6 → 7.0, 7.9 → 8.1, 8.9 → 9.2. Three
   to six percent where the replay saw fewer copies, never less (the
   repeats' spread 0.8-7%). 0009 stays the default.

   **The CPU kernels of every LUMI build were SSE2 only** until 15-10-2026:
   `tools/lumi/build_engine.sh` set no `target-cpu`, so Rust targeted
   x86-64's baseline and `build.rs` gave ggml no AVX, AVX2, FMA or F16C (the
   published binaries are built for x86-64-v3). Everything the CPU computes
   ran on those kernels: `--cpu-moe` above, and the experts `--fit`'s split
   leaves in RAM, so both columns are low by that much; the cache's copies
   and the GPU's work are not affected. The build now targets x86-64-v3, and
   c10 is measured again on it (`next13`).
5. **Split modes** (`c03`, item 2 above; objective (3)). Layer against
   tensor (`r-split`, `next12`, 15-10-2026, one node a point), tokens/s:

   | model | GCDs | 1, layer | 1, tensor | 16, layer | 16, tensor |
   |---|---:|---:|---:|---:|---:|
   | Qwen3-32B Q4_K_M | 2 | 38 | 43.3 | 265 | 324 |
   | | 4 | 36.1 | 18.2 | 260 | 228 |
   | | 8 | 34.4 | 7.8 | 252 | 91.5 |
   | Qwen3.6-35B-A3B | 2 | 91.9 | 70.8 | 291 | 317 |
   | | 4 | 82.6 | 25.5 | 285 | 283 |
   | | 8 | 73.9 | 11.3 | 273 | 127 |
   | Qwen3-235B Q4_K_M | 4 | 35.6 | 12.3 | 76.8 | 98.9 |
   | | 8 | 33.3 | 4.2 | 75.7 | 55.3 |

   Layer split loses a little with every GCD added, as on 12-09. Tensor is
   the first arrangement that made one request faster across GCDs (the 32B
   on two, +14%; +22% at 16), and the 235B at 16 on four GCDs gains 29%;
   beyond two GCDs one request collapses, to an eighth to a quarter of
   layer's speed on eight. Tensor loads were slow too: 414-1,131 s for the
   235B against about 31 s.

   The cause is in ggml (`ggml-cuda.cu`, `ggml_backend_cuda_comm_init`):
   tensor parallelism sums partial results across the GPUs after every
   cut layer, through NCCL (RCCL on ROCm) when it is compiled in; without
   it, through an internal all-reduce that exists for exactly two devices
   ("internal AllReduce init failed (n_devices != 2?)"); beyond two,
   through a generic exchange of the meta-backend. The engine had no
   RCCL, so 4 and 8 GCDs measured the generic one. An RCCL build
   (`EULLM_RCCL=1 tools/lumi/build_engine.sh`, item 6 above) and c03's
   `rccl-*` groups measure tensor again with it, and on two GCDs the
   internal all-reduce in the same binary (`GGML_CUDA_ALLREDUCE=internal`)
   as control: 20 points, about 9 node-hours.

   With RCCL (`r-split-rccl`, `next14`, job 22880434, 15/16-10-2026),
   tokens/s, tensor without RCCL (`next12`) → with it, layer beside:

   | model | GCDs | 1, tensor | 1, layer | 16, tensor | 16, layer |
   |---|---:|---:|---:|---:|---:|
   | Qwen3-32B | 2 | 43.3 → 45.4 | 38 | 324 → 321 | 265 |
   | | 4 | 18.2 → **52.7** | 36.1 | 228 → **384** | 260 |
   | | 8 | 7.8 → 18.1 | 34.4 | 91.5 → 165 | 252 |
   | Qwen3.6-35B-A3B | 2 | 70.8 → 71.7 | 91.9 | 317 → 315 | 291 |
   | | 4 | 25.5 → 53.9 | 82.6 | 283 → 304 | 285 |
   | | 8 | 11.3 → 23.1 | 73.9 | 127 → 275 | 273 |
   | Qwen3-235B Q4_K_M | 4 | 12.3 → 26.8 | 35.6 | 98.9 → **110** | 76.8 |
   | | 8 | 4.2 → 11.1 | 33.3 | 55.3 → **107** | 75.7 |

   On two GCDs RCCL and ggml's own all-reduce are level (the control, the
   internal one in the same binary: 43.2 and 324 on the 32B, 71.4 and 317
   on the 35B-A3B). Beyond two, RCCL is what tensor parallelism was
   missing: the dense 32B on four GCDs answers one request 46% faster than
   layer split and 39% faster than one GCD alone (about 38), and writes 48%
   more with 16; the 235B with 16 requests writes 41-43% more than layer
   split on four GCDs and on eight. Eight GCDs are still too many for one
   request (the reductions cross the node's slower links), and the small
   MoE gains nothing with one: its experts are small matrices to cut. Loads
   are as slow as before (416-1,128 s for the 235B against about 31). RCCL
   is linked by default in LUMI builds from 16-10-2026 (`EULLM_RCCL=0` to
   leave it out); layer split does not use it. `rccl-batch`, `rccl-480b`
   and `rccl-120b` go on to 32 and 64 requests, the 480B and gpt-oss-120b.

   They (`r-split-rccl2`, `next14`, 16-10-2026), tokens/s, layer → tensor:

   | model | GCDs | 1 | 16 | 32 | 64 |
   |---|---:|---:|---:|---:|---:|
   | Qwen3-32B | 4 | | | 380 → **565** | 521 → **810** |
   | Qwen3-235B Q4_K_M | 4 | | | 116 → **192** | 144 → **264** |
   | Qwen3-Coder-480B Q4_K_M | 8 | 28.1 → 17.6 | 63.9 → **127** | 82 → **192** | |
   | gpt-oss-120b MXFP4, tensor | 2 | 92.2 | 276 | 454 | |
   | | 4 | 75.1 | 309 | 502 | |

   With many requests tensor is the better split wherever a model needs
   more than one GCD: 49-83% more than layer on four GCDs, twice as much on
   the 480B over eight, and gpt-oss-120b on two writes 454 at 32 where c02's
   layer split wrote 298. With one request layer stays ahead on eight GCDs.
   The 480B's tensor load takes 1,506-1,640 s (layer 88-167). Whether
   replicas of a smaller split beat one wider split on a whole node, the
   question the placement policy needs answered, is `c11-placement`.

   It (`next16`, 18-19-10-2026), a whole node of 8 GCDs, tokens/s of the
   node by the requests on it (servers × batch):

   | model | servers × GCDs, split | 16 | 32 | 64 | 128 | 256 |
   |---|---|---:|---:|---:|---:|---:|
   | gpt-oss-120b MXFP4 | 2 × 4, layer | | 445 | 660 | 771 | |
   | | 2 × 4, tensor | | 606 | 934 | 1,271 | |
   | | 4 × 2, layer | | | 905 | 1,323 | 1,508 |
   | | 4 × 2, tensor | | | 1,045 | 1,712 | **2,078** |
   | Qwen3-235B Q4_K_M | 1 × 8, layer | 75.2 | 113 | 142 | | |
   | | 1 × 8, tensor | 96.9 | 184 | 293 | | |
   | | 2 × 4, layer | | 152 | 229 | 286 | |
   | | 2 × 4, tensor | | 220 | 381 | **526** | |
   | Qwen3-Coder-480B Q4_K_M | 1 × 8, layer | | | 102 | | |
   | | 1 × 8, tensor | | | **251** | | |
   | DeepSeek-V3.1 Q4_K_M | 1 × 8, layer | 35.7 | 44.5 | | | |
   | Qwen3-32B | 8 × 1 | | | | **2,046** | |
   | | 4 × 2, tensor | | | | 1,652 | |
   | | 2 × 4, tensor | | | | 1,332 | |

   The placement that follows: a model that fits one GCD runs one server
   per GCD, since tensor's reductions cost more than they give at the same
   requests (Qwen3-32B: eight servers 2,046, four of two GCDs 1,652, with a
   spread of 21-30% between repeats); a larger model runs on the fewest
   GCDs that hold it, split as tensor, as many servers as the node takes
   (gpt-oss-120b four of two GCDs, 2,078 at 256 requests, 38% above layer;
   the 235B two of four, 526 at 128, 1.8 times layer and the one server of
   eight); the 480B needs all eight, where tensor writes 2.5 times layer.
   DeepSeek-V3.1 did not load with tensor (all three points) and stays on
   layer: 21.5 tokens/s at one request, 44.5 at 32. Tensor's loads are
   slow: 943-1,269 s for the 235B against 19-145 with layer, 1,679 s for
   the 480B against 113, 255-626 s for gpt-oss-120b against 10-46. A
   server loaded once a day can afford that, a swap cannot. Why: llama.cpp
   gives a weight split along axis 0 (attention output, feed-forward down,
   experts included) part of every row on each GPU, and copied it as one 2D
   copy from pageable memory; Qwen3-235B's `ffn_down_exps` has 524,288 rows
   a GCD in each of its 94 layers, and 100-230 MB/s, slower the more GCDs, is
   what a copy per row would give. Patch 0010 gathers each GCD's rows in
   memory and copies them whole; c11's `load-*` groups load the 235B, the
   480B and gpt-oss-120b cold with it and without it
   (`GGML_META_GATHER_ROWS=0`), on one binary (`next19`).


Every change that lands in `main` from these is a new round of `c01`+`c02`.

### The first rounds on the October engines (10 to 13-10-2026)

`c01` on `next6` (the engine of 0.7.60, prompts read in the step) and `c02` on
`next4` (the engine that reads large models in), all Q4_K_M unless named.

**One GCD, f16 KV cache** (`c01 dense-one-gcd`), tokens/s at 1, 4, 16 and 32
requests at once:

| model | 1 | 4 | 16 | 32 | 32 over 1 |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 149 | 322 | 921 | 1242 | 8.3× |
| Qwen3-8B | 114 | 229 | 705 | 958 | 8.4× |
| Qwen3-14B | 75.3 | 153 | 470 | 667 | 8.9× |
| Mistral-Small-24B | 53.3 | 109 | 361 | 528 | 9.9× |
| Qwen3-32B | 38 | 76 | 267 | 378 | 9.9× |
| Qwen3.6-27B | 38.3 | 68 | 161 | 201 | 5.2× |

The hybrid 27B (three layers in four recurrent) gains a little over half as
much from batching as the attention-only models, and reads a prompt at half
the 32B's rate (134 against 296 tokens/s alone; the first token after 600 ms
at 32 against 113). On the 35B-A3B, also hybrid, `c07` found llama-server no
faster (229 against EuLLM's 246 with sixteen at once), so the limit is likely
llama.cpp's recurrent layers on gfx90a, not the scheduler; a `c07` point on
the 27B would say.

**A quantized KV cache costs speed on gfx90a and buys only memory**, the
November anomaly, now on every model. At 32 requests q8_0 is 4-11% slower than
f16 and q4_0 12-23% (on Qwen3.6-27B, whose cache is a quarter of its layers,
1-3%); alone, 3-13%. Qwen3-8B at 32 requests holds 23.2 GiB with f16, 15.2
with q8_0, 13.0 with q4_0. With 19,500 tokens of context the cost grows:
decoding with q8_0 is 9-10% slower on Qwen3-14B and the 35B-A3B, 30-33% on
Qwen3-8B and Qwen3-32B.

**Long contexts** (`c01 long-context`, one GCD, one request). The prompts are
the runner's nominal 8,000 and 16,000 tokens, at 4 characters a token, which
Qwen3 reads as about 9,700 and 19,500 (the c01 spec's note). Prompt reading,
tokens/s, and the first token:

| model | 9,700 | 19,500 | first token |
|---|---:|---:|---:|
| Qwen3-8B | 2202 | 1937 | 4.4 / 10.1 s |
| Qwen3-14B | 1400 | 1156 | 7.0 / 16.8 s |
| Qwen3-32B | 688 | 599 | 14.2 / 32.5 s |
| Qwen3.6-35B-A3B | 2565 | 2534 | 3.0 / 6.1 s |

After 19,500 tokens the dense models decode 21-34% slower than with no
context, the 35B-A3B, whose attention is a quarter of its layers, 8%. With
four such prompts at once the median first token comes after 2.6 times one
prompt's (Qwen3-14B, 44.0 s at 19,500): the prompts are read one after
another, oldest first, so the first answer starts after one prompt and not
after four.

**The 235B on a node** (`c02`, Qwen3-235B-A22B), tokens/s at 1, 8 and 32
requests per server:

| | 1 | 8 | 32 |
|---|---:|---:|---:|
| Q4_K_M, one server on 4 GCDs | 29 | 63.7 | 98.4 |
| Q4_K_M, one server on 8 GCDs | 33 | 64.5 | 101 |
| Q4_K_M, two servers of 4 GCDs | 70.1 | 91.1 (cv 79%) | 202 |
| Q8_0, one server on 8 GCDs | 28.2 | 42.7 | 85.4 |

The larger ones, read in on 8 GCDs, tokens/s: Qwen3-Coder-480B 27.1, 50.2
and 75.9 at 1, 8 and 32 requests; DeepSeek-V3.1 21.5, 31.8 and 34 at 1, 4
and 16. A day of the 235B Q8_0 under the graded workload (`moe-soak`): 36.5
tokens/s, 49,013 requests, accuracy 0.96-0.99.

Four more GCDs under one server add nothing (layer split, as on 12-09); the
same four as a second server double the node (2.05× at 32). gpt-oss-120b the
same way: 298 tokens/s on two GCDs at 32, 1181 as four servers of two GCDs on
the node (3.96×). That is item 4's rule, the smallest split that fits and
then replicas, on the models it is for. Eighteen hours of the 235B as two
servers under the graded workload (`moe-235b-soak-x2`): 74.4 tokens/s, 63,431
requests, accuracy 0.95-0.99.

The loads were the open item. 132 and 233 GiB are under half the node's
memory, so `next4` mapped the 235B (item 8), and the points' start times
(job 22687047 on 10-10, 22675825-6 on 09-10) show which loads were slow:

| load, from Lustre | seconds |
|---|---:|
| 235B Q4, one server of 8 GCDs | 59-102 |
| 235B Q4, one server of 4 GCDs alone | 69 |
| 235B Q4, two servers of 4 GCDs at once (side by side, or two replicas) | failed after an hour once; 667-1901, 86 once |
| 235B Q8, one server of 8 GCDs | failed after an hour once; 1161, then 32-89 with the file in the page cache |
| gpt-oss-120b, four servers of 2 GCDs at once | failed after an hour once; 761, then 57-61 with the file in the page cache |
| 480B and 671B, read in | 91-284 |

Mapping a file several servers map at once, or one near half the memory,
is what Lustre does not survive; c09's single-server loads of the 235B
never did it, and read in it costs the same (108 against 110 s). The
three failed hours and the slow loads cost about 5 node-h. Since 13-10-2026
the engine reads in every model that goes to the GPUs whole from a network
file system, whatever its size (`fit::read_in_whole_on_gpu`). c09 measured it
on that engine (`r-next8`, job 22800286, 13-10-2026, `next8`), cold and alone
on a node, one load each, seconds and then tokens/s:

| servers on one file | mapped | read in | the engine's choice |
|---|---:|---:|---:|
| two of the 235B Q4, 4 GCDs each | 2448 (55.1) | 113 (70.6) | 121 (70.2) |
| four of gpt-oss-120b, 2 GCDs each | 1789 (350) | 53 (413) | 63 (414) |
| eight of Qwen3-14B, one GCD each | 343 (589) | 14 (588) | 16 (590) |
| one of the 235B Q8, 8 GCDs | 204 (28.2) | 225 (28.3) | 209 (26.6) |

Several servers mapping one file load 20 to 28 times slower than reading it
in, and the engine now reads it in by itself. One server alone maps even the
233 GiB Q8_0 fine when it starts cold: its hour in c02 came after other
points had filled the page cache. The mapped pairs of the 235B and the four
gpt-oss-120b also answered slower once loaded, the 235B's first token after
245 ms instead of 30; one run each, but it fits the token-embedding table,
which llama.cpp keeps on the CPU and reads from the mapped file, being read
from Lustre again. The round took about two node-hours.

**Qwen3.6-27B** (`c07 rt27-*`, `r-27b`, job 22814646, and
`sbatch_hybrid_profile.slurm`, job 22814618, 14-10-2026, one GCD, `next8`).
The hybrid model against llama-server, tokens/s, with Qwen3-32B beside it:

| requests | 27B EuLLM | 27B llama-server | 32B EuLLM | 32B llama-server |
|---:|---:|---:|---:|---:|
| 1 | 38 | 35.1 | 38 | 37.6 |
| 4 | 66.2 | 61.3 | 75.7 | 75.1 |
| 16 | 162 | 133 | 262 | 252 |
| 32 | 198 | 112 (cv 49%) | 377 | 358 |
| 16 clients, graded gsm8k | 141 | 124 | | |

The 27B's ceiling is llama.cpp's, not the scheduler's: llama.cpp alone
(`llama-batched-bench`, 128 tokens of prompt and 128 of answer a sequence)
writes 33, 69, 172 and 190 tokens/s at 1, 4, 16 and 32 sequences, where
the 32B goes on to 422. With 32 sequences answering the 27B spends twice
the 32B's GPU time (rocprof, 20.3 s against 10.7): 22% in the gated delta
net kernel (0.71 ms a call, 0.02 with one sequence), 22% in a
matrix-vector kernel that the 32B does not run at 32, and 7% gathering the
recurrent states. The matrix-vector one is the linear attention's output
projection (`ssm_out`), whose input llama.cpp shapes one matrix per
sequence, so its weights are read once per sequence (0.71 ms a call,
0.05 with one): llama.cpp patch 0007 (14-10-2026) makes it one product.
Reading a prompt, the two are level (llama-bench, 790 and 684 tokens/s at
512 and 4,096 tokens on the 27B, 689 and 784 on the 32B).

A conversation of five turns over a 6,000-token document (5,400 tokens of
prompt, `multiturn_probe.py`): on the 27B EuLLM read every turn's prompt in
6.5 s, `--ctx-checkpoints 8` no better, llama-server in 0.33-0.67 s from
the second turn; on the 32B EuLLM in 10-70 ms and llama-server in 86-133.
The recurrent state kept before each prompt's end (`next9`, 14-10-2026) is
the fix. The same conversation on it (job 22821238, `eullm-before` the
`next8` engine on the same GCD), prompt reading from the second turn:
Qwen3.6-27B 0.17-0.45 s instead of 6.7 (llama-server 0.31-0.38),
Qwen3.6-35B-A3B 0.10-0.13 s instead of 2.0 (llama-server 0.16-0.43). Each
checkpoint holds 150 MiB on the 27B (156,894,364 bytes) and 63 MiB on the
35B-A3B. The first turn, the server's first request, took 20-29% longer
(10.2 s against 7.9, 2.9 against 2.4); on four CPU cores the cost of the
checkpoints on a first prompt was about 5%, and the probe's second
conversation measures it warm from now on.

Patch 0007 (`next10`, `r-27b-next10`, job 22825342, same day), tokens/s,
against `r-27b`:

| | 27B EuLLM | 27B llama-server | 32B EuLLM (control) |
|---|---:|---:|---:|
| 1 request | 38 → 38.3 | 35.1 → 35.3 | 38 → 37 |
| 4 at once | 66.2 → 67.8 | 61.3 → 61.5 | 75.7 → 75.1 |
| 16 at once | 162 → 186 | 133 → 132 | 262 → 261 |
| 32 at once | 198 → 240 | 112 → 111 | 377 → 378 |
| 16 clients, graded gsm8k | 141 → 165 | 124 → 126 | |

At 32 the 27B now writes 2.2 times llama-server's; it still gains 6.3x
from 1 to 32 against the 32B's 10x. What is left is the gated delta net
kernel itself (22% of the GPU time at 32) and the gathering of the
recurrent states (7%).

**Q8_0 against Q4_K_M** (`c02 quant-one-gcd`): one request is 18% slower on
Qwen3-8B, 24% on 14B, 28% on 32B, and level on Qwen3-30B-A3B (122 against
121; 7% slower at 32). The batched rows are not compared: `next4` reads
prompts between steps, `next6` in them. Under the graded workload
gpt-oss-20b gave the same answer text to a repeated question 23-38% of the
time, Qwen3 80-93%, at the same accuracy (0.93-0.98): its answers vary with
sampling, its grades do not.

## Lines not to cross

- **This is a development allocation for the engine, inference only.** The
  workload is public benchmark sets, graded; nothing it produces feeds Forge,
  and no training of any kind runs here — not Forge, and not the engine's own
  trainer (`c05`, withdrawn). That is what EHPC-AIF-2026PG01-1434 told
  EuroHPC about this allocation.
- **Every hour leaves a result.** A round without an engine change, a point
  that measures nothing new — those are the hours that are hard to explain.
  More rounds tied to more engine changes are not.

## Final Report

The portal already has the slot (page 11 of the consolidated forms: *Final
Report Upload*). To settle while writing, not after:

- **Licence.** The application says Apache-2.0 in three places; the repository
  is AGPL-3.0-or-later since August 2026. Both open source; the report must
  describe the repository as it is.
- **Where the CUDA numbers came from**, per the section above.
- **The `c05` points that ran on 05-10/06-10** before being withdrawn: how
  many, that they trained small public models on public text as an engine
  benchmark, that the models were deleted, and that the track was stopped
  because of the declaration in EHPC-AIF-2026PG01-1434.
- **The deadline and template**: confirm with EuroHPC; the AI-Factory rule is
  three months after the end (12-06-2027 here), and it is reasonable to
  assume the same.

## Decisions needed

1. **Two nodes in parallel** until the deficit is recovered — the default of
   `submit_campaign.sh`.
2. **The CUDA half on Leonardo AI-Factory** before 02-11, or wait for the
   Leonardo/JUPITER decision.
3. **MMLU**: find a pinned mirror for ReflexBench, or leave it out (the
   campaigns already do).
