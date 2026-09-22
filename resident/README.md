# Resident serving — pay the load once

`semif_phase1/cli.py` is one-shot by construction: `load → score → exit`. Every
call therefore re-reads the 20.9 GB checkpoint from disk. On this host that is
**18.9–20.4 s** of a ~59 s call, and because the process dies between calls
nothing carries over.

`resident_scorer.py` keeps the process alive and serves many calls from one load.

## What it does and does not change

**Unchanged, on purpose.** The scoring path is SemIf's own, called as a library:
the reference-tokenizer prompt render, the GGUF/HF vocabulary agreement check
that guards `prompt_sha256`, the shared-state fan-out, and the option-slot
readout. Residency is a *lifecycle* change, so every option score should be
identical to the CLI. Checked, not assumed: across the 1-, 4-, 8- and 16-criterion
runs, and across all three resident calls, every decision matches
`results/decisions.batched.jsonl` on both the argmax option and
`full_vocab_argmax_id`. The 16-criterion single pass is included, so `SEQ_MAX=16`
is verified not to corrupt anything, and the byte-identical repeated prompts inside
the 8- and 16-row batches all decide identically — batching does not bleed state
between sequences.

Probabilities are **bit-identical** to `results/decisions.batched.jsonl` on all four
prompts — all 17 printed digits — even though that baseline ran `threads=6` and this
runs 16. So **thread count does not move these logits at all.** (An earlier draft of
this file, and the handoff, attributed the small delta to thread count. That was
wrong; the measurement below is what corrected it.)

The only numeric spread in the entire set is the *batched* path against the
pre-batching single-sequence one: at most **0.001%** on the winning probability
(c2: `0.99995396519670143` vs `0.99994377885607821`), with no decision changed on any
prompt. That difference predates this work — it is the earlier batching patch — but
this serving path depends on batching, so it is worth stating rather than leaving
implied.

**Honest limit on the lossless claim:** all four fixtures have a dominant option at
p ≈ 0.99998, so they test decision stability *weakly*. A fixture whose top two options
were near-tied would be the real check, and there is none in `examples/`. What is
established is that the arithmetic path is unchanged and the decisions agree on the
fixtures that exist — not that no reachable input could flip.

**Not changed either, and this is the honest limit.** The decode still goes
through llama.cpp's mmap. Read amplification is untouched — see below.

## Measured

8 GiB cgroup cap, page cache evicted first, 4 criteria / 908 tokens, three calls
in one process:

| | load | decode read | wall | s / criterion |
| --- | --- | --- | --- | --- |
| call 1 | 18.92 s | 41.07 GB | 40.27 s | 10.07 |
| call 2 | — | 41.09 GB | 38.52 s | 9.63 |
| call 3 | — | 41.08 GB | 38.90 s | 9.72 |

`process_read_bytes` after call 1 was **61.96 GB** = 20.89 GB load + 41.07 GB
decode, which reproduces the published one-shot figure of 61.6 GB. The instrument
agrees with the baseline it is measuring against.

**So residency buys the load and nothing more: ~59 s → ~38.5 s per call, 1.5×.**

## Why it buys nothing more — measured, not assumed

Read is **flat** across calls: 41.069 / 41.090 / 41.083 GB. There is *zero*
inter-call reuse, and the reason is structural rather than a tuning miss. A pass
sweeps the model's layers in order, so it touches roughly 41 GB against an 8 GiB
cap; by the time it reaches the last layer, the pages it faulted at the first are
already reclaimed. Its own sweep evicts its own early pages. Call 2 then starts at
layer 0, exactly where nothing is warm.

An 8 GiB cap cannot cache a 20.9 GB checkpoint. Residency is worth having — a
coding agent makes many calls, and 20 s of the 59 is pure overhead — but it
cannot rescue the decode, because the decode was never paying for the load.

## Three ways in

```bash
# measurement: one call per fixture, one load
resident/resident_scorer.py --gguf MODEL.gguf --model TOKENIZER_DIR \
    examples/decisions.sample.jsonl

# serving: newline-delimited JSON requests on stdin, one response line each
resident/resident_scorer.py --gguf MODEL.gguf --model TOKENIZER_DIR --serve
{"call_id": "a", "rows": [{"id": ..., "state": ..., "question": ..., "options": [...]}]}

# HTTP, in JEV's own contract: a drop-in local System One provider
resident/systemone_shim.py --gguf MODEL.gguf --model TOKENIZER_DIR --port 8123
# then: export TYPESAFE_BASE_URL=http://127.0.0.1:8123
```

`--repeat N` re-runs the fixture list N times. `--serve-output` also appends
served responses to a file. Every call emits a `resident_call` ledger line on
stderr carrying its own `read_bytes` delta, RSS before and after, and the pass
timing; the process emits `resident_load` at start and `resident_summary` at end.

`read_bytes` from `/proc/self/io` — not wall clock — is the honest measure here.
It includes page-fault-driven reads, so it counts what mmap actually pulled from
the device. A warm run can beat a disk run on the clock while reading nothing.

**The HTTP endpoint generates nothing, by construction.** It maps `state` +
`questions` onto the same `score_shared` rows this scorer already runs, and maps
the option scores back onto the contract's `noul` / `choice` / `score` answers.
`usage.output_tokens` is **0**, and every string it returns is the caller's own
text echoed back. That is the point: a decision is **one prefill pass** however
many options it has, while generating the same answer would pay **212.6 MB per
token** (cold) — see "Generation, measured" in `../README.md`. Its `local`
telemetry carries `cache_state`, which distinguishes a page-cache-warm call from
a real NVMe one, so a `0` in `call_read_bytes` cannot be misread as a dead
counter. Calls are serialised: one context, one call at a time.

Gate it with `test_systemone_shim.py` — 26 contract checks against the real
checkpoint, non-zero exit on any mismatch.

## Running it capped

`run_resident_35b_ssd.sh` mirrors `run_semif_35b_ssd.sh`: it re-execs under
`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`, and evicts the
checkpoint once with `posix_fadvise(DONTNEED)` *before* the resident process
starts. The eviction is what makes the numbers mean anything — a capped scope
does not recharge pages that are already cached, so without it the cap never binds
and every call reads nothing.

```bash
./run_resident_35b_ssd.sh --simulate-8g examples/decisions.sample.jsonl
REPEAT=3 ./run_resident_35b_ssd.sh --simulate-8g
SEQ_MAX=16 ./run_resident_35b_ssd.sh --simulate-8g examples/decisions.n16.jsonl
```

`SEQ_MAX` and `LOAD_MODE` are context/model-creation parameters, so they are fixed
for the life of a process: sweeping them means one process per value, which is
what `scripts/sweep_resident.sh` does — strictly serially, because two capped runs
at once contend for device bandwidth and the measurement *is* bytes over time.

`scripts/analyze_sweep.py results/sweep-*/run.log` fits the read model out of the
ledger lines.

## What one pass costs, and what batching amortizes

Four single-pass runs, 8 GiB cap, `mmap` (`results/sweep-*-token-scale/`,
`results/sweep-*-n16-s16/`). The 16-criterion point needs `SEQ_MAX=16` to stay
single-pass:

| criteria | tokens forwarded | pass read | GB / token | × expert set | s / criterion |
| --- | --- | --- | --- | --- | --- |
| 1 | 253 | 17.55 GB | 69.3 | 0.96 | 14.88 |
| 4 | 908 | 41.05 GB | 45.2 | 2.24 | 9.76 |
| 8 | 1,816 | 64.16 GB | 35.3 | 3.50 | 8.16 |
| 16 | 3,632 | 110.23 GB | 30.3 | 6.01 | 7.50 |

Least squares over those points: `read = 13.8 GB + 26.9 MB × token`.

Two things follow, and they matter more than the residency result.

**A single small call already reads essentially the whole expert set** — 17.55 GB
against 18.33 GB of routed experts, for one criterion of 118 suffix tokens. The
fixed term is not small. It dominates anything a single-criterion caller does, and
it is why one criterion costs 14.9 s while four cost 9.8 s each and sixteen cost
7.5 s each.

**The marginal token gets cheaper as the batch grows** (35.9 → 25.4 MB/token), so
the fixed term is genuinely shared, not merely averaged. Per-criterion read falls
17.5 → 10.3 → 8.0 → 6.9 GB. For a caller that can batch — which is the JEV shape,
one state fanned out over many criteria — that is up to a **2.5× reduction in I/O
per criterion and 2.0× in time per criterion**, available today, with no engine
change and no extra RAM. It needs `SEQ_MAX` to cover the batch: past `n_seq_max`
each extra group is a full extra pass, so a 16-criterion batch at `SEQ_MAX=8` pays
two sweeps for the read volume of one.

Measured, 16 criteria, `mmap`, 8 GiB cap:

| `SEQ_MAX` | passes | read | wall | s / criterion |
| --- | --- | --- | --- | --- |
| 8 (baseline) | 2 | 128.3 GB | 127.1 s | 7.94 |
| 16 | 1 | **110.2 GB** | **119.96 s** | **7.50** |

The saving is exactly one sweep's worth of bytes — 18.1 GB against a fitted fixed
term of 13.8 GB plus the marginal tokens that pass no longer pays. It is a 14%
byte win but only a **1.06× time win**: the single pass runs at 0.92 GB/s against
1.01 GB/s for the two-pass version, because one 8192-wide ubatch carries all 3,632
tokens and so re-reads more within the pass. Sizing `SEQ_MAX` to the batch is
correct and cheap, but it is a small win, not a large one — the large win is
making the pass stop re-reading at all.

## Where the remaining time actually goes

Residency removed the overhead. Batching amortizes the fixed term. What neither
touches is the amplification, and it decomposes cleanly — for the 4-criterion
fixture, whose routing selects about **9.3 GB** of expert bytes (top-8 of 256,
256 experts × 3 projections × 590 KB × 40 layers, from the GGUF table):

| | read | factor |
| --- | --- | --- |
| expert bytes the routing requires | ~9.3 GB | 1× |
| same run at a 16 GiB cap | 19.3 GB | **2.1× gather overfetch** |
| same run at an 8 GiB cap | 41.0 GB | **2.1× cap-driven re-read** |

Two independent 2.1× factors, and both are structural rather than tuning misses.

The cap factor is intra-pass re-read. Within a layer an expert is wanted by
several tokens, but each want is a separate fetch; between them the pass faults
tens of MB of other experts, which under an 8 GiB cap evicts the earlier slice.
Raise the cap and the layer's working set survives, which is exactly what the cap
table shows. An 8 GiB cap cannot cache a 20.9 GB checkpoint, and no amount of
process lifetime changes that — the flat 41.08 GB across three resident calls is
this same fact seen from outside.

The overfetch factor is that even when nothing is evicted, the gather reads ~2×
what the routing asked for.

Together they are the whole remaining prize: **~4.4× less I/O, and therefore
~4.4× less time, at the same 8 GiB cap, with no extra RAM.** Conservatively, it is
at least the measured **2.1×** — an engine that reaches the 16 GiB read volume
while capped at 8 GiB.

It needs the pass to visit experts in expert order: group the batch's tokens by
selected expert, fetch each slice once, use it for every token that wants it, drop
it. Working set becomes a few slices instead of the whole set, which kills the cap
factor; reading each slice exactly once kills the overfetch factor. That is a
decode rewrite, not a flag, and llama-cpp-python ships prebuilt `.so` files, so it
is a rebuild-and-patch project. Every engine in the prior-art survey that works
does exactly this; none of them is a llama.cpp flag.

`direct_io` is *not* that change, and it is worse than the earlier note in
`docs/DEVELOPMENT_LOG.md` suggested. Measured under the 8 GiB cap it does not
merely have a bad footprint — it **cannot load at all**. `O_DIRECT` reads every
tensor into allocated buffers instead of mapping them, so it needs
`CPU model buffer size = 19914.65 MiB` (= 20.88 GB, the whole checkpoint) of
anonymous RAM; the process is SIGKILLed during `load_all_data`, before scoring
anything, and writes no ledger line. It is a load mode for hosts that can hold the
model resident, not for this regime. Recorded in `results/sweep-*-direct-io/`.
