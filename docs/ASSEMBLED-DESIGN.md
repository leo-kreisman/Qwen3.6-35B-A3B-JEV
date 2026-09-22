# The assembled design — best-of the surveyed engines, on our substrate

Assembled from source-level teardowns of every repo in `qwen3.6-35b-a3b-ssd-offload.md`
(clones in `offload_projects/`, comparison in `OFFLOAD_PROJECTS_ANALYSIS.md`). Each element
names its source and the measurement behind it. Nothing here is README-level branding —
`slipstream-dwijenpatel` is a fork of TurboFieldfare, `Mference` carries the same files, and
Edge0's prerouter accuracy is never measured in-repo. Where a number is a vendor claim it is
marked as such.

**Target substrate: llama.cpp/GGUF, x86 Linux, no Apple Silicon.** Edge0's only backend is
MLX, which is the lock-in we are avoiding — so we take its *ideas*, not its code.

---

## 1. Container and I/O — the layer everything else depends on

**Repack the experts into one container with per-expert contiguous slabs, 16 KiB aligned,
plus a sidecar offset table.** This is the single most repeated structural idea in the set:
`apus` packs gate_up+down into a 6,291,456 B slab and verifies single-shard placement;
`flash-moe` uses a fixed 7,077,888 B stride with 9 components at fixed in-expert offsets;
`qwen-fieldfare`'s `Q4EXP02` has a 4 KiB header and a uniform 1,769,472 B stride at 16 KiB
alignment; `slipstream-schero94`'s PGRN v1 sets `PGRN_ALIGN=16384`; `siphon.cpp` uses a
16 KiB-aligned sidecar manifest. Alignment is what makes a routed expert **one extent**.

**Read with `pread`/`preadv` against that table — never mmap.** Every working engine does
this. Evidence: `flash-moe` measured mmap of expert files as **5× worse** than pread; every
working engine in the table uses pread; the only mmap entries are Edge0 (page-cache
dependent) and `Qwen-MoE-Router-exp` (a stub).

**Do not "fix" amplification by suppressing read-ahead.** Our own measurement: `MADV_RANDOM`
cut reads 42.0 → 26.0 GB but cost 56.0 → **368.0 s**. The ~2.15× amplification on a >RAM MoE
is **load-bearing prefetch**, not waste.

**Fire the three projections concurrently** (`qwen-fieldfare`) to collapse 9 reads toward 3.
**Optional: lazy `O_DIRECT` twin fds** (`samosa-chat` measured 0.8 GB/s buffered vs 2.3+ GB/s
O_DIRECT). Note this is *per-expert* O_DIRECT reads — categorically different from our
whole-file `O_DIRECT` load mode, which is SIGKILLed during load and closed.

## 2. Slot pool — the engine owns the buffers, not the page cache

**Bounded slot pool with a frequency-aware policy.** Plain LRU is **wrong** for
layer-monotonic access — `q36` explicitly skips past layers during prefill for that reason.
Take one of: LFU-aging (`Mference`, `slipstream-dwijenpatel` — halve counts every 32 plans);
decayed routing frequency with LRU tiebreak (`q36`, halved every 16 tokens); SLRU
(`qwen-fieldfare`); CLOCK-LRU-K with HOT/WARM tiers (`slipstream-schero94`); ARC + execution
pins (`siphon.cpp`); per-layer LRU + LFRU pins + slab freelist (`apus`).

**Floor the pool at 8 slots/layer — this is load-bearing, not a tuning knob.** With top-8
routing, a layer below 8 slots misses *every* token (`qwen-fieldfare`).

**A hard RAM admission gate.** `slipstream-schero94` refuses a cache that would induce
swapping (`peregrine_admission.c:15-71`). `qwisp`'s field data is the argument: a 128-slot
tier (~11.4 GB wired) **collapsed ~10× on a 16 GB Mac**; 64 slots recovered 6×.

**This is the rule our own cache-warming tests hit from the other side.** Under a fixed cap,
stage into the engine's own buffers — never the page cache. Layer-ordered
`POSIX_FADV_WILLNEED` moved `read_bytes` by 0.14 GB across a whole run (the kernel caps an
advisory window to a small multiple of `read_ahead_kb`), and a real `preadv` prewarmer was
*worse*: 42.0 → 56.1 GB, because clean unreferenced pages are the first reclaim victims.

## 3. Prefetch — and the one place the prior art does not answer us

**Decode:** predict layer L+1 from L's router. `apus` reports **79.6% recall** — a real
in-repo measurement, unlike Edge0's prerouter whose accuracy is never measured anywhere.
`siphon.cpp` is better than fixed lookahead: it sets prefetch **distance** from an EWMA of
transfer_us vs layer_compute_us. `q36` runs a dedicated prewarm pthread with a 512 MiB window
and seeds a hotlist at startup. `qwisp` uses a bounded 2-arena background `pread` pipeline.

**Prefill is the gap.** All of this is decode-time, and a one-token lookahead does nothing
for a one-shot prefill — the whole layer's expert set is touched. `q36` is the only engine
that even names the problem (it skips past layers during prefill). **Our expensive case is
prefill**, and the prior art is silent on it. Any honest plan has to say what we do here
rather than importing a decode optimisation and calling it solved.

## 4. Scheduling — ours, and in none of the surveyed engines

- **Thread oversubscription at 4/3 of hardware threads: 54.6 → 38.4 s (−29.6%), lossless.**
  A thread waiting on a page fault is not runnable, so an expert-streaming decode needs *more*
  runnable threads than cores. The prior art is silent because those engines drive I/O from
  their own worker pools instead of demand faults. Reads unchanged (41.2–42.1 GB) — pure
  scheduling, and proof the decode is not purely I/O-bound.
- **`n_batch = n_ubatch = sequences × 512`: 67.0 → 42.0 GB.** `n_ubatch`, not `n_batch`, is
  the parameter that decides I/O; the default 512 silently split a 908-token group into two
  ubatches that each re-walked all 40 layers.

## 5. The llama.cpp seam — built twice, by other people

`siphon.cpp` intercepts the MoE top-k tensor in an eval callback, repoints `w->data` at a
slot pool, shrinks `w->ne[2]`, and rewrites the id tensor to slot ids so `MUL_MAT_ID` sees a
small pool. `slipstream-schero94` ships **the same trick as a 4,488-line patch against pinned
llama.cpp `79bba02a6741`** — the closest thing to a drop-in, and the cheapest place to start.

## 6. Lossy levers — held back until the lossless path is measured

- **REAP expert pruning** (arXiv:2510.13999): one-shot, no fine-tuning, 50% compression at
  ~96–97% retention.
- **Batch-aware routing** (OEA arXiv:2511.02237, Lynx arXiv:2411.08982): decode latency is
  linear in the count of *distinct* experts touched, not batch size; 39% decode-latency cut
  on Qwen3-30B at batch 16 for <1 pp loss.

## 7. Measured dead ends — do not re-litigate

`MADV_RANDOM`; lossless compression of 4-bit streams (incompressible, and the best rANS
decoder runs 114–120 MB/s against our 750 MB/s SSD); elaborate eviction
(arXiv:2608.07911 — causal rules pick the optimal victim 3.4% of the time vs LRU's 20.6–22.1%);
co-activation expert reordering (`flash-moe` measured 0%); page-cache warming from outside
the engine (tested twice, worse both times); whole-file `O_DIRECT` load mode (SIGKILLed).

---

## Build order

| phase | work | gate |
|---|---|---|
| 0 | **Run our decode path.** ✅ **Done 2026-09-22** — `scripts/probes/probe_decode.py`: cold under a verified-binding 8 GiB cap **3.67 tok/s at 212.6 MB/token**, warm **10.68**. | a real tok/s number on the goal axis |
| 1 | Port `slipstream-schero94`'s 4,488-line slot-remap patch onto our llama.cpp | decode bytes fall at the same cap |
| 2 | Repack experts into an aligned slab container + sidecar table (§1) | amplification < 1.2× |
| 3 | Replace LRU with LFU-aging or decayed-frequency; floor 8 slots/layer; admission gate (§2) | hit rate + cap stability |
| 4 | Decode prefetch: router L+1, adaptive distance (§3) | measured, against phase 0 |
| 5 | Solve prefill — the gap no surveyed engine answers (§3) | — |
| 6 | Only then: REAP / batch-aware routing (§6) | quality delta measured first |

Phase 0 is complete (3.67 cold / 10.68 warm), so every later phase now has a baseline to be
judged against. The measured gap to close: 212.6 MB read per generated token.

A note on the survey size, since the count drifted in earlier revisions: **12 repositories
were cloned, 11 were marked CERTIFIED, and there are ~9 distinct engines** —
`slipstream-dwijenpatel` is a TurboFieldfare fork, `Mference` carries the same files
verbatim, and `Qwen-MoE-Router-exp` is a stub.
