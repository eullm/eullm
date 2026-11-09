# moe_cache_sim — eviction policies for the MoE expert cache

`--moe-cache` keeps the most used experts of the layers whose experts live
in RAM in a fixed number of VRAM slots, shared by all those layers. At
every layer of every decode step the experts the router chose must all
have a slot; a miss is copied over the CPU-GPU link into a slot taken from
another expert. This program replays the experts a model chose, layer by
layer, through the cache's eviction policies and counts what each one
copies, so that a policy is chosen before it is written into llama.cpp.

Standard library only:

```bash
c++ -O2 -std=c++17 -o moe_cache_sim bench/moe_cache_sim/moe_cache_sim.cpp
```

## Policies

| name | the slot a miss takes |
|---|---|
| `lru` | the least recently used: llama.cpp's `moe_cache_lru::plan` before #30637 |
| `slru` | llama.cpp's `moe_cache_slru::plan` since #30637 (b11777 on): a probation and a protected half, a hit promotes, a miss takes probation's least recently used slot first |
| `stale-mru` | the least recently used while its expert went unused for a whole pass over the layers; when every slot was used within the last pass, the most recently used one outside the current layer. llama.cpp patch `0008` |
| `lfu-decay` | the one whose expert has the lowest count, counts kept for every expert (cached or not) and halved every 4 passes (`--half-life`) |
| `stale-lfu` | `stale-mru` when a pass overflows; among the stale slots, the one whose expert has the lowest decayed count instead of the least recently used |
| `lfu-pass` | `stale-lfu`, and when a pass overflows, the lowest count among the slots used earlier in this pass (the most recently used if there is none); `/32` compares at most 32 slots a miss, as llama.cpp could afford |
| `belady (bound)` | the one whose expert is chosen again furthest in the future: needs the whole trace, so it is the best any policy could do |

The layers come in the same order every step. When a step chooses more
experts than there are slots, the least recently used slot is always an
expert of one of the next layers, the one about to be chosen again, and
`lru` copies every expert. `stale-mru` is `lru` slot for slot while a
step's experts fit, since a stale slot always exists then.

## Input

`run TRACE` reads what llama.cpp writes with patch `0008` and
`LLAMA_MOE_CACHE_TRACE=FILE`: a header (`MOEC`, version 1, the model's
layers, experts and experts per token, the cache's slots and its layers),
then one record per layer planned: the layer, the number of ids, the ids,
all int32. The cache's own slot count is used unless `--slots` says
otherwise. A trace cut short by the end of a job is read up to its last
whole record.

`synth` makes one up, until real ones are there: each layer ranks its
experts for the prompt, an expert of rank r is chosen with weight
(r+1)^-alpha, a sequence keeps each expert of its previous token with
probability `--keep`, and `--drift` pairs of ranks swap every token. With
the same prompt for all sequences (the campaign's throughput points) they
share the ranking; `--own` of their choices come from a ranking of their
own, and `--independent` gives each its own prompt.

## What it said (15-10-2026)

Calibrated against the `c10` statistics from LUMI (one MI250X GCD,
`LLAMA_MOE_CACHE_STATS=64`): Qwen3-235B Q4_K_M with `--alpha 1.8 --keep 0.7
--own 0 --drift 0` and 3,900 slots gives LRU 94.5% at one sequence (LUMI
97.5%), 90.7% at 8 with 2,435 experts a step (LUMI 89%, 2,906), and 0% at
32 with 50.7 GiB copied a step (LUMI 0.85%, 50.4 GiB). Experts in VRAM:

| model, sequences | `lru` | `stale-mru` | `lfu-decay` | bound |
|---|---:|---:|---:|---:|
| 235B, 1 | 94.5% | 94.5% | 94.5% | 95.1% |
| 235B, 8 | 90.7% | 90.7% | 89.5% | 94.7% |
| 235B, 32 | 0% | 71.8% | 65.7% | 81.0% |
| 480B, 1 | 82.0% | 82.0% | 82.4% | 88.8% |
| 480B, 8 | 0% | 57.0% | 52.4% | 74.0% |
| 480B, 32 | 0% | 28.3% | 21.3% | 35.2% |

(480B: 62 layers, 160 experts, 1,350 slots, `--alpha 1.6 --keep 0.5 --drift
1`.) Over 16 other settings of the routing at 8 and 32 sequences, with the
same or different prompts, `stale-mru` was never below `lru`, equal to it
whenever a step fit, and 3-17 points below the bound when it did not;
`lfu-decay` fell to 0% where the steps were largest. The calibration is rough (the 480B's
17% at 8 sequences on LUMI came from steps of every size, the model's from
8 exactly), so the c10 `evict` group measures it on LUMI and records real
traces to replay here.

```bash
./moe_cache_sim synth --layers 94 --experts 128 --alpha 1.8 --keep 0.7 --own 0 --drift 0 \
    --batch 32 --slots 3900 --expert-mib 10.88
./moe_cache_sim run 235b-b32.moe-trace --expert-mib 10.88
```

## Real traces (16-10-2026)

The `c10` `evict` points (`next15`, one MI250X GCD) wrote their routing with
`LLAMA_MOE_CACHE_TRACE`. A quantization mixes tensor types across layers,
and the cache groups layers by layout: the trace holds the first group, 46
of Qwen3-235B's 94 layers and 30 of Qwen3-Coder-480B's 62, with that
group's own slots, which shrink as the batch's KV cache grows. Experts in
VRAM:

| model, requests | slots | `lru` | `stale-mru` | `lfu-decay` | bound | on the GCD, new eviction |
|---|---:|---:|---:|---:|---:|---:|
| 235B, 1 | 2,485 (42%) | 94.3% | 94.3% | 94.3% | 96.1% | 96.3% |
| 235B, 8 | 2,255 (38%) | 88.9% | 88.9% | 90.9% | 95.3% | 88.2% |
| 235B, 32 | 1,438 (24%) | 12.1% | 56.2% | 40.5% | 63.8% | 50.5-56.2% |
| 480B, 1 | 819 (17%) | 77.2% | 77.2% | 78.6% | 88.8% | 82.5% |
| 480B, 8 | 696 (15%) | 16.5% | 57.6% | 60.4% | 73.9% | 56.9% |
| 480B, 32 | 272 (6%) | 5.6% | 16.9% | 6.4% | 22.9% | 14.1% (steps over 8) |

The replay agrees with the GCD's own statistics within a few points, over
all the layers where the GCD's figure covers both groups. The decayed
counts win where the steps fit and lose where they overflow, which is what
`stale-lfu` and `lfu-pass` combine; on the synthetic routing `lfu-pass`
gains 1-4 points over `stale-mru` where a pass overflows and is never
below it.

## llama.cpp's SLRU (27-10-2026)
Upstream replaced the LRU with a 50/50 SLRU (#30637). Replayed on the same
synthetic routing as above (`--no-opt` for the 480B), experts in VRAM:
| model, sequences | `lru` | `slru` | `stale-mru` (`0008`) | `lfu-pass` (`0009`) |
|---|---:|---:|---:|---:|
| 235B, 1 | 94.5% | 94.5% | 94.5% | 94.5% |
| 235B, 8 | 90.7% | 90.6% | 90.7% | 91.1% |
| 235B, 32 | 0% | 0% | 71.8% | 73.1% |
| 480B, 1 | 82.0% | 82.2% | 82.0% | 82.4% |
| 480B, 8 | 0% | 0% | 57.0% | 59.6% |
| 480B, 32 | 0% | 0% | 28.3% | 32.6% |
The SLRU is the LRU slot for slot where a step fits, and copies every expert
where it overflows: its victim is still the least recently used slot, of
probation and then of the protected half, so a pass larger than the cache
evicts the experts of the next layers just as the LRU did. Patches `0008` and
`0009` therefore stay, ported onto the SLRU.
