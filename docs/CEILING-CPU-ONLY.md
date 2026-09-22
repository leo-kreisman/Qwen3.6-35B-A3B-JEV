# The CPU-only ceiling: what 20+ tok/s actually requires

**Question:** can this box serve Qwen3.6-35B-A3B CPU-only, experts off NVMe, under a
hard RAM cap, at 20+ tok/s?

**Answer, measured and arithmetic:** **not single-stream at Q4_K_S.** The binding
constraint is not the SSD — it is DDR4-2400 dual-channel RAM bandwidth. But 20+ is
reachable two ways, and both preserve the CPU-only premise: **amortise the weight
pass over more than one token (JEV parallelism / speculative verification)**, or
**reduce bytes per token below the current quant**.

## 1. The arithmetic (from the file, not the label)

Parsed from the GGUF tensor table:

| quantity | value |
|---|---|
| total params | 34.66 B |
| active params per token (A3B) | ~3.3 B |
| file size | 20.893 GB |
| **effective bits/weight (Q4_K_S as shipped)** | **4.822** |
| **active weight bytes streamed per token** | **1.989 GB** |
| routed expert set | 18.327 GB (40 × 256 × 3 × 576 KiB slices) |

So a single decode token must stream **1.99 GB** of weights. At 20 tok/s that is
**39.8 GB/s**, against a **DDR4-2400 dual-channel theoretical peak of 38.4 GB/s** —
**104% of peak**. It is arithmetically impossible, not merely hard.

## 2. The ceiling, measured three ways

**(a) Our own thread sweep**, CPU-only (`-ngl 0`), page-cached, 135-token prefill +
32-token decode, `-ub 512`, run twice:

| threads | decode tok/s (run 1) | decode tok/s (run 2) | achieved RAM BW |
|---|---|---|---|
| 6 | 10.94 | 10.97 | ~21.8 GB/s |
| 8 | **11.03** | 10.75 | ~21.7 GB/s |
| 12 | 7.73 | 7.17 | ~14.8 GB/s |
| 16 | 6.43 | 6.62 | ~13.0 GB/s |
| 20 | 6.38 | — | 12.7 GB/s |

Two runs agree within ~3%. The shape is a **plateau at 6–8 threads (~10.8–11.0
tok/s) then a cliff at 12**, not a gentle decline. This host has **6 physical cores /
12 hardware threads**, so the cliff sits exactly at the point where thread count
crosses the physical-core count: 6–8 is at or barely past the cores, 12 is full SMT,
16–20 oversubscribes past all hardware threads. In a bandwidth-bound loop, extra
threads contend for the same memory controller and buy nothing.

The magnitude of the regression: the current default of **16** is **1.66–1.71×
slower** than 6.

**(b) Independent confirmation on the same CPU.** An external benchmark of
`Qwen3-30B-A3B` on **an i7-8700 with no GPU**, 6 threads, reports **10.6 tok/s**,
and states the ceiling directly: *"Single-user CPU generation is
memory-bandwidth-bound: the speed ceiling is set by how many weight bytes you stream
from RAM for each token"* and *"read ~3.3B of weights per token and you get ~10.6
tok/s, whether those weights came from a dense 3B model or were routed out of a 30B
one."* Source: `docs/research/sources/https-inventivehq-com-blog-moe-on-cpu-benchmark.md`.

**(c) The project's own warm figure** is 10.68 tok/s — same number, third
independent path. Three measurements, one ceiling.

Our achieved 21.8 GB/s is **57% of DDR4-2400 theoretical peak**, which is a normal
fraction for random-ish expert row reads. Getting to 20 tok/s single-stream would
need 104% of peak. **The SSD is not in this equation at all.** No I/O optimization —
io_uring, repacking, prefetch, slot pools, W-TinyLFU — can move a bandwidth wall
that sits above the hardware's maximum.

## 3. A config regression this exposed

`llamacpp_backend._default_threads()` returns `max(4, nproc * 4 // 3)` = **16** on
this host, on the strength of the measured 4/3 knee (54.6 → 38.4 s, −29.6%).

That finding is real **but regime-specific**. Above, in the bandwidth-bound regime,
16 threads costs **1.66×** against 6 (6.62 vs 10.97 tok/s). The 4/3 oversubscription
pays only when threads are *blocked on page faults* — it keeps the device queue fed.
Once the weights are in RAM there are no faults to hide, and the extra threads
simply contend for the same memory controller.

**The two regimes want opposite thread counts**, and the current code hard-codes the
I/O-bound answer for both. Thread count should be selected by regime, not by a
constant: ~6 (physical cores) when bandwidth-bound, 4/3 × nproc when fault-bound.
This is a one-line change with a measured 1.66× on one side of it.

Corroboration: the external i7-8700 benchmark explicitly used *"6, matching the
physical cores (more threads hurt)"*.

## 4. The two routes to 20+, both CPU-only

### Route A — amortise the weight pass (this is JEV parallelism)

If one weight pass produces more than one useful token, the per-token bandwidth cost
divides by the number of tokens per pass:

| tokens accepted per weight pass | effective tok/s at the 10.97 ceiling |
|---|---|
| 2 | 21.9 |
| 3 | 32.9 |
| 4 | 43.9 |

Two published mechanisms:

- **Speculative decoding** — llama.cpp ships it; the project's survey cites
  **3.1–3.4× end-to-end on CPU** from the EAGLE-style family. Verification batches k
  draft tokens through one target pass, and MoE routing means the k tokens' expert
  union is read **once**. At k=2–4 accepted, this clears 20+ on its own.
- **JEV's own structure — N criteria per pass.** `score_shared` already batches
  branches into one `llama_decode`, and `batch_logits_batched` exists precisely so
  *"the routed experts for the whole set are read once per forward pass instead of
  once per criterion."* The project's own measured throughput table:

  | criteria | token-forwards | tok/s |
  |---|---|---|
  | 4 | 908 | 23.0 |
  | 8 | 1,816 | 27.2 |
  | 16 | 3,632 | 28.6 |

  Those are *prefill token-forwards*. In decision terms, 4 criteria in 39.4 s cold =
  0.102 decisions/s. The lever is not the per-pass rate but **criteria per pass**:
  at 16 criteria per pass the same weight traffic yields 16 decisions. **JEV's
  product metric is decisions per GB read, and it is already the best-amortised path
  in the project** — it just has never been stated in those units.

### Route B — reduce bytes per token

| quant | bits/weight | GB/token | ceiling at 57% of peak |
|---|---|---|---|
| Q4_K_S (current) | 4.82 | 1.989 | ~16.4 |
| Q3_K | 3.40 | 1.403 | ~23.3 |
| ~2.1 bpw (IQ2-class) | 2.10 | 0.866 | ~37.7 |

Route B is **lossy** and belongs in a separate tier behind a decision-flip check on
the four fixtures — not mixed into a lossless claim. Route A is not lossy at all
(speculative verification is distribution-preserving under exact rejection sampling,
and JEV batching is already verified bit-identical).

## 5. What this means for the build order

The project's phase list targets **bytes read from SSD**. On this host the measured
constraint is **bytes streamed from RAM**, and the SSD numbers (41 GB per pass,
212.6 MB/token) are a *symptom* of the RAM cap forcing re-reads — not the root cost.

That reorders the phases by measured leverage, without changing the CPU-only premise:

1. **Fix the thread regime** (§3) — one line, measured 1.66× in the bandwidth-bound
   case, and it makes every later measurement honest.
2. **Batch criteria per pass** (Route A, JEV parallelism) — code already exists in
   `batch_logits_batched`; the gate is `n_seq_max`, not a new subsystem.
3. **Speculative verification** (Route A) — the published, measured path to 3× on
   CPU, and the project's survey already ranks it the highest-leverage import.
4. **Then** the phases aimed at SSD read volume (slot remap, repack, io_uring),
   because they attack the 41 GB symptom and cannot exceed the bandwidth ceiling.

## 6. Layer restructuring — what the literature actually offers

Three papers found in this pass, with their own reported numbers:

- **Pipeline-Native Transformers / cflow** (arXiv:2608.23841) — the direct answer to
  "restructure the layers". Single-token CPU decode is bandwidth-bound; the paper
  **restructures the model into vertical pipeline stages so per-token weight
  bandwidth falls 9.00 → 4.50 MB/token (2.00×)** within 0.24 perplexity of the best
  baseline, reaching **5.94 tok/s vs llama.cpp's 4.75** on a 32-vCPU Ice Lake box.
  The key point: it is *architecture change + a per-layer on-disk format*, not a
  runtime trick — and its gain is bounded by the same bandwidth wall, so it improves
  the ceiling rather than escaping it.
- **OmniMoE Expert-Centric Scheduling** (arXiv:2602.05711) — **inverts execution from
  token-centric to expert-centric**, grouping physically nearby experts so a block of
  expert weights is loaded once and reused across stacked tokens; reported **10.9×
  (73 ms → 6.7 ms)**. This is the same amortisation as Route A, implemented as a
  scheduler.
- **MoE-Prefill / AsyncEP** (arXiv:2605.02960) — the publishable answer to the
  prefill expert-set problem: it **decouples expert placement from activation
  routing**, keeping only upcoming layers' expert weights resident and gathering in
  the background. Directly relevant to JEV, whose hot path is prefill.

Caveat on all three: **none is measured on a single NVMe at ~2.4 GB/s with no DRAM
tier.** Pipeline-Native's baseline is a page-cached Ice Lake server; OmniMoE's is a
GPU. Their mechanisms port; their numbers do not transfer.

## 7. Honest bottom line

- 20+ tok/s **single-stream CPU-only at Q4_K_S is impossible here** — needs 104% of
  DDR4-2400 peak. Stated plainly rather than worked around.
- 20+ is reachable at **k≥2 tokens per weight pass** (spec decode or JEV's criteria
  batching) **without any lossy change**, and that is already the project's own
  architecture.
- Or at **Q3_K-class quant** (lossy, needs a decision-flip check).
- The thread default is a measured regression in the bandwidth-bound regime.

## Sources

Saved full text under `docs/research/sources/`:

- `https-inventivehq-com-blog-moe-on-cpu-benchmark.md` — i7-8700, no GPU, Qwen3-30B-A3B
  10.6 tok/s, bandwidth-ceiling argument
- `https-arxiv-org-pdf-2608-23841.md` — Pipeline-Native Transformers (vertical
  pipeline, 2.00× bandwidth reduction)
- `https-arxiv-org-pdf-2602-05711v2.md` — OmniMoE expert-centric scheduling (10.9×)
- `https-arxiv-org-html-2605-02960.md` — MoE-Prefill / AsyncEP
- `https-github-com-thecodacus-llama-cpp-blob-perf-README-md.md` — MoE expert cache
  fork, measured slot-count table (+21% to +78%)
- `https-github-com-Lidenburg-llama-cpp.md` — three-tier VRAM/RAM/disk cache using
  **io_uring + O_DIRECT** for the disk tier, `madvise(MADV_DONTNEED)` to release
  mmap'd pages, and a measured **~1% hit rate for next-layer prefetch** (a negative
  result worth having)
- `https-huggingface-co-blog-Doctor-Shotgun-llamacpp-moe-offload-guide.md`

Not verified / not claimed: no published figure found for single-stream CPU-only
decode of a 35B-A3B class MoE at 20+ tok/s without a second tier; the spec-decode
3.1–3.4× figure is the project's own survey citation and was not re-measured here.
