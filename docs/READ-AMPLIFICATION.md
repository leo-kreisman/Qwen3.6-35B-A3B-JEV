# The prefill read amplification: measured, and the two levers that remove it

Two things are true about our prefill, and they explain why the JEV scorer is slow:

1. **llama.cpp re-reads the same expert once per ubatch.** A 2048-token prompt split
   into 4 ubatches of 512 pulls every routed expert off the disk up to 4 times. The
   weights are identical each time.
2. **The GGUF stores one expert in three separate file regions.** Fetching a single
   expert takes three scattered reads instead of one contiguous one.

Both are measured below. The first is a scheduling fix, the second a layout fix, and
they are independent — they multiply.

---

## Lever 1: row-grouping — MEASURED, 3.49×

The mechanism and the name come from **SSD-LLaMA** (arXiv:2609.18110, "SSD-Native
Inference for Trillion-Parameter MoE at 1+ Token/s on a Consumer PC"), §4.3:

> "During prefill, many prompt tokens are processed together… These prompt tokens are
> distributed unevenly among the selected experts… SSD-LLaMA groups all prompt rows
> assigned to the same expert and forms one task for each active expert. … It allows
> each expert's weights to be **loaded once and reused** for all rows assigned to that
> expert, thereby **avoiding repeated weight reads during prefill**."

It reports **prefill 1.90×–3.61× over llama.cpp** and does it *"without speculative
reads or modifications to the router-selected expert set"* — no training, no accuracy
change.

llama.cpp has no cross-ubatch expert scheduling, so the lever available today is the
ubatch size itself: **one ubatch spanning the whole prompt is the degenerate case where
every expert is loaded exactly once.** So we measure the disease from the other end —
watching storage traffic fall as the pass is split into fewer ubatches.

`scripts/probes/probe_readamp_capped.sh`, cold, page cache evicted, **8 GiB cap
verified binding** (memory.peak 8.0 GiB and `memory.events max` 123,656–236,558 in every
run — all three genuinely streamed), prompt 2048, `-ngl 0`, `-t 6`, model
`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`:

| ubatch | ub/pass | prefill t/s | infer read | MB/prefill tok | effective GB/s | seconds |
|---|---|---|---|---|---|---|
| 512 | 4 | **29.09** | **135.08 GB** | 66.0 | 1.92 | 70.41 |
| 1024 | 2 | **55.30** | **72.06 GB** | 35.2 | 1.95 | 37.03 |
| 2048 | 1 | **101.59** | **37.81 GB** | 18.5 | 1.88 | 20.16 |

- **3.49× prefill, 3.57× fewer bytes** (135.08 → 37.81 GB) from ub 512 → 2048.
- Reads scale **linearly with ubatch count**: 4 ubatches = 135.08 GB, 2 = 72.06, 1 = 37.81.
  That is the signature of per-ubatch redundancy, not of a different access pattern.
- **Effective GB/s is flat at 1.88–1.95.** The gain is *entirely* fewer bytes read, not
  better scheduling. Nothing about the disk got faster.

This confirms, on our hardware and under a verified cap, exactly what the paper claims —
and it is the same effect seen earlier at 2.79× when ubatch was raised 512 → 2048 on the
warm path. The cold capped number is **3.49×**.

### Residual amplification, still unexplained

At ubatch 2048 a single pass still reads **2.09× the full expert pool per layer**:

    per layer at ub 2048 :  37.81 GB / 40 layers      = 0.945 GB
    full expert pool/layer: 256 experts × 1.6875 MiB  = 0.453 GB
    ratio                                             = 2.09×   (ub 512: 7.46×)

The part not accounted for by per-ubatch redundancy is re-reads of the always-active
non-expert set under the RAM cap, plus routing imbalance. Not decomposed — flagged, not
claimed.

---

## Lever 2: expert-pack layout — MODELLED from our own I/O measurements

**The layout is broken, verified by parsing our GGUF directly.** blk.0's three expert
tensors:

    974,839,808    blk.0.ffn_down_exps   [512, 2048, 256] Q4_K
    1,126,555,648  blk.0.ffn_gate_exps   [2048, 512, 256] Q4_K
    1,280,376,832  blk.0.ffn_up_exps     [2048, 512, 256] Q4_K

**305 MB apart, in three separate regions** — exactly the pathology SSD-LLaMA describes:

> "In the original GGUF layout, the three tensors belonging to one expert may reside in
> different file regions or shards. Consequently, retrieving one selected expert requires
> **several separate file accesses rather than one contiguous read**."

Its fix is an offline **lossless** "expert pack": gate+up+down back-to-back in one aligned
block, one aligned `O_DIRECT` request per expert, plus a manifest mapping each expert to
its offset and tensor boundaries. *"This reorganization changes only the physical layout
on SSD; the tensor contents, quantization formats, routing decisions, and model
computation remain unchanged."*

**The arithmetic, from our own tensor table:**

    one slab (gate, or up, or down) = 2048 × 512 weights at Q4_K (4.5 bpw)
                                    = 589,824 B = 576 KiB      <- exact
    one complete expert             = 3 × 576 KiB = 1,687.5 KiB = 1.688 MiB

**576 KiB is not a coincidental block size — it is this model's expert slice.** And we
already measured this exact granularity on this exact NVMe
(`results/nvme-io-20260922/io_results.csv`, raw io_uring vs `pread`, `O_DIRECT` random):

| block | `pread` qd1 | io_uring qd8 | ratio |
|---|---|---|---|
| **576 KiB** | 1364.3 MiB/s | **3383.8 MiB/s** | **2.48×** |
| 2 MiB | 2722.7 MiB/s | 3378.2 MiB/s | 1.24× |
| 4 MiB | 3054.8 MiB/s | 3382.9 MiB/s | 1.11× |

So per expert fetch:

- **today** — 3 scattered 576 KiB reads, `pread` qd1 @ 1364 MiB/s → **1.27 ms**
- **packed** — 1 contiguous 1.688 MiB read, ~2722 MiB/s → **0.62 ms** (**2.0×**)
- **packed + io_uring batched** — 3380 MiB/s → **0.50 ms** (**2.5×**)
- requests per expert: **3 → 1**

And the ceiling this puts on the whole path:

    effective GB/s measured in our prefill today : 1.88–1.95 GB/s
    io_uring batched O_DIRECT ceiling on 576 KiB : ~3380 MiB/s
    headroom                                     : up to ~1.75×

**Modelled, not measured** — the 2.0–2.5× is arithmetic on top of measured I/O rates,
not an end-to-end run of a repacked file. Repacking has not been built.

---

## What this does for the JEV scorer

This is the answer to "why is the scorer slow", and it is the same disease.

The scorer's own committed table (`README.md:119`) reads **35.3 MB per token-forward**
at 16 criteria (128.3 GB / 3,632). Our measurements bracket that:

    my ub 512 : 66.0 MB/token-forward
    their scorer : 35.3 MB/token-forward   <- equivalent to about ub 1024
    my ub 2048: 18.5 MB/token-forward

So the scorer is running at roughly ub-1024-equivalent economics. At ub 2048 the same
prefill work would cost:

    3,632 token-forwards × 18.5 MB = 67 GB   (vs 128.3 GB measured)
    3,632 / 101.59 t/s             = 35.8 s  (vs 127.1 s measured at 16 criteria)

**≈3.5× less wall time and ≈1.9× less read on the scorer's own committed numbers.** This
is an estimate projected from measured per-token costs, not a scorer run — the scorer's
own ubatch is set by `_ubatch()` in `semif/src/semif_phase1/llamacpp_backend.py`, which
returns 512 as committed.

---

## Where this comes from outside AI

Lever 2 is not a new idea. It is **ROMIO's data sieving and two-phase collective I/O**
(Thakur et al., Argonne) — the 1990s MPI-IO answer to exactly this shape of problem,
"accesses to a large number of small, noncontiguous pieces of data": instead of issuing N
little requests, read the large contiguous region and discard what you did not need.
Seekable zstd (independently compressed frames + a seek table) is the non-AI answer to the
compression step. The AI-specific contribution of SSD-LLaMA is applying both to experts
that are individually addressable and immutable — which is what makes packing and
per-expert compression safe here.

## Also found, ranked

| # | lever | status | effect |
|---|---|---|---|
| 1 | row-grouped prefill (expert read once per pass) | **measured** | **3.49× prefill, 3.57× fewer bytes** |
| 2 | expert-pack layout (3 scattered reads → 1) | modelled | ~2.0× per fetch, 3 → 1 requests |
| 3 | io_uring batched O_DIRECT + pinned pool | measured in isolation | up to 2.48× at 576 KiB |
| 4 | lossless expert-pack compression | paper | −33.2% bytes (8.89 → 5.93 GB/token) |
| 5 | prerouter / expert prediction | paper | needs 33 distilled heads + ~2M rows |

Lever 5 is the only one requiring training, and SSD-LLaMA reaches up to 3.61× on prefill
without it. Edge0's prerouter needs a **full-token lead**, not Pre-gated MoE's one-layer
lead — the paper says the per-layer schedule *"does not survive contact with a streaming
engine"* (30–100 ms of pipeline drain per step).

**Not applicable to us:** MoE-Prefill (arXiv:2605.02960) is a *distributed* system
(AsyncEP, expert-weight AllGather); its three redundancies are about expert parallelism
across devices, not one box. SSD-LLaMA's own numbers — RTX 5090, 16 GB host DRAM, PCIe 5.0
NVMe at 9 GiB/s — do not transfer; ours is ~2.4 GB/s. Its mechanisms do.

## Reproduce

```bash
cd ~/Projects/JEV_experiment
UB_LIST="512 1024 2048" PROMPTS=2048 CAP=8G THREADS=6 ./scripts/probes/probe_readamp_capped.sh
```

Raw: `results/probe-readamp-20260922-174920/configs.jsonl` (and `-174540`, which contains
the first ub=512 run whose summary line was lost to an arg-index bug now fixed).
I/O table: `results/nvme-io-20260922/io_results.csv`. Papers:
`~/.hermes/cache/scratch/2609.18110.txt`, `2605.02960.txt`, `2609.18063.txt`, `romio.txt`.
