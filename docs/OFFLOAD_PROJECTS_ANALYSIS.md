# SSD expert-streaming: source-level teardown of the 12 candidate projects

Scope: every repo in `qwen3.6-35b-a3b-ssd-offload.md` (the 11 "certified" entries plus
`flash-moe`), cloned into `offload_projects/` and read at **source level**, not README level.
Plus a web pass over the published algorithms. Written against one concrete problem:

> Qwen3.6-35B-A3B, 40 layers, 256 experts, top-8, 19.5 GiB Q4_K_S, x86 Linux, **8 GiB RAM cap**,
> via llama.cpp + mmap. Measured: 60.2 GB read and 56.6 s for 4 decisions.

The list's own warning was that "certified" only ever meant "the documentation asserts it."
That gap was real, and section 1 is what fell out of closing it.

---

## 1. First finding: the list overcounts independent implementations

Three of the four Apple repos are **one codebase**. `slipstream-dwijenpatel` is a fork of
`TurboFieldfare` (`AGENTS.md:5`), and `Mference` carries the same streaming files almost
verbatim (`Infrastructure/Streaming/PreadExpertStreamer.swift`, `ExpertStreamer.swift`,
`ResidentExpertStreamer.swift`). So "Apple Silicon 7" is closer to 4 independent designs:
TurboFieldfare-lineage (×2), `qwisp`, `qwen-fieldfare`, `q36`, `samosa-chat`, and Edge0.

Second, the certification standard did not survive contact with the source:

| Claim | Source-level reality |
|---|---|
| Edge0 "prerouter routing prediction" | Implemented (`prerouter/heads.py`, 33 MLP heads, start_layer 7) — but **prediction accuracy is never measured anywhere in the repo**. No recall counter in `layer.stats()` (`layer.py:923`). The "+59%" is a docstring number with no artifact. |
| Edge0 "SSD expert offload" | Real, but implemented as **whole-file mmap + `madvise(MADV_WILLNEED)`** (`streaming/mmap.py:27,43-50`). It has a real offset table yet never uses it for a positioned read: per-expert access is a numpy view (`mmap.py:89-93`). I/O parallelism is delegated to the page cache. MLX/Metal-bound. |
| Edge0 "Recover-LoRA" | Plain unmerged LoRA (`adapters/lora.py:1-10`), base int4 stays byte-identical. **Reduces quality loss, not bytes** — it reads *more*. README branding. |
| Edge0 measured numbers | `examples/bench.py` reports tok/s and peak GiB only — **no GB-read, no disk throughput, no hit rate**. 14.9–17.7 tok/s is real; the streaming-efficiency claims are not measured in-repo. |
| `samosa-chat` "2-bit MoE" | Two 4-bit planes joined — `refine_join_projection` rebuilds q4 nibbles from base+residual (`qwen36b.c:1698-1710`); base-only forces low 2 bits to 1, giving codes {−7,−3,1,5} (`:142-147`). Bytes on disk drop; the kernel still eats q4. |
| `Qwen-MoE-Router-exp` | **Stub.** Zero disk code; a launcher plus cgroup/SIGSTOP governor. Its README self-admits no per-expert eviction. Useful only as the honest negative baseline. |
| `flash-moe` ordering experiments | Expert *ordering* by co-activation measured **0%** gain — "NVMe ignores scatter at 7 MB granularity". The win is coalescing, not grouping. |
| `siphon.cpp` 6 GB VRAM | Real mechanism, but the 50+/20 tok/s figures have **no committed logs**, only harnesses. |

Two more negative results worth not repeating: `dwijenpatel`'s routing-prediction prefetch
cut disk wait 15.95 → 7.01 ms/token yet was **net neutral-to-negative** (6% slower elsewhere,
`README.md:338-345`); and Mference's own docs report its `mmap` + GPU-fault residency strategy
paging at **~0.7 GB/s** (`ResidentExpertStreamer.swift:19-21`).

---

## 2. What every working engine actually does

Without exception, the engines that work read experts with **`pread` against a per-expert
offset table**, and the ones that use whole-file mmap are the slow or broken ones.

| Project | Disk path | Offset table | Cache policy | Prefetch |
|---|---|---|---|---|
| **Edge0** | mmap + `MADV_WILLNEED` | safetensors header, per-tensor; expert offset by **integer division** | global cross-layer **LRU**, 64 slots, count-bounded | trained prerouter (33 heads) + previous-token set; lookahead **1 token**; `staged_replace=False` in all presets |
| **Mference** | `pread` + coalesced **`preadv`** | `layout.json`, page-aligned stride validated at load | **LFU-aging** (halve counts every 32 plans) | speculative reservations on a global queue; 16.4% gain claimed |
| **slipstream-dwijenpatel** | `pread`, `F_NOCACHE`; no mmap | `layout.json`, `expertStride % pageSize == 0` enforced | **LFU-aging**, 64 slots/layer | RoutePredictionLedger with real `recall(distance:)`; **off by default**, honestly reported neutral |
| **qwisp** | `pread` **directly on safetensors** — no repack | safetensors header at runtime; no alignment needed | LRU arena, **RAM-tiered** slot count | bounded 2-arena background `pread` pipeline |
| **qwen-fieldfare** | `pread`, `Q4EXP02` container, 4 KiB header | **implicit** — uniform 1,769,472 B stride, 16 KiB alignment | **SLRU** (probationary+protected), per-layer capacity distributed by demand, **floor of 8 slots/layer** | speculative L+1 into spare slots; 3 projections fired concurrently |
| **q36** | per-expert `pread` on the GGUF fd; `posix_fadvise(RANDOM)`, `DONTNEED` | GGUF `abs_offset` + per-slot gate/up/down offsets | decayed routing-frequency victim + LRU tiebreak, halved every 16 tokens | dedicated prewarm pthread, 512 MiB window; hotlist seeded at startup |
| **apus** | `pread`, mmap explicitly rejected | `apus.index.json` `expert_slabs{layer,expert,shard,offset,nbytes}`, verified single-shard | per-layer LRU + LFRU pins + **RSS guard** + slab freelist | pthread pool; pilot predicts L+1 from L's router, **79.6% recall** |
| **flash-moe** | per-expert `pread` into Metal buffers | none at runtime; 60 flat per-layer files, fixed stride | **none — "trust the OS"** (~71% natural page-cache hit) | persistent I/O pool, A/B double-buffered expert sets |
| **samosa-chat** | `pread` + `posix_fadvise`, **lazy `O_DIRECT` twin fds** (buffered 0.8 vs O_DIRECT 2.3+ GB/s measured) | safetensors JSON + streaming `expert_offsets[]/expert_sizes[]` | byte-budget LRU/2Q with per-layer floors | `WILLNEED` per expert only — no threads, no prediction |
| **siphon.cpp** | **GDS/cuFile** `cuFileBatchIOSubmit` + `O_DIRECT`, 20 worker threads | GGUF tensor offsets + `expert_stride = total/n_experts`, 16 KiB pool alignment, sidecar manifest | **hybrid ARC** (recent/frequent) + hit-count retention, execution pins | EWMA of transfer_us vs layer_compute_us sets prefetch distance |
| **slipstream-schero94** | `pread` only — "mmap is never used"; `POSIX_FADV_RANDOM` + `DONTNEED` | PGRN v1 container, `PGRN_ALIGN=16384`, 26-byte dir records | fixed-slot **CLOCK-LRU-K**, HOT/WARM tiers, hard resident cap, RAM admission gate | parallel cold reads, `io_width` threads |
| **Qwen-MoE-Router-exp** | mmap (inherited, deliberately) | none | none | none |

### The convergent design

1. **`pread` with a per-expert offset table**, not mmap. `flash-moe` measured mmap of expert
   files as **5× worse** than pread (per-page faults).
2. **Coalesce each expert into one contiguous slab** so a routed expert is *one* read:
   `apus` packs gate_up+down into a 6,291,456 B slab; `flash-moe` uses a fixed
   7,077,888 B expert stride with 9 components at fixed in-expert offsets; `qwen-fieldfare`
   fires the 3 projections **concurrently** to collapse 9 reads. This is the single most
   repeated structural idea.
3. **16 KiB alignment**, everywhere (`PGRN_ALIGN`, `flash-moe` stride, `qwen-fieldfare`
   header, siphon pool alignment). Page/block alignment is what makes a read one extent.
4. **A bounded slot pool with a frequency-aware eviction policy** — LFU-aging, decayed
   routing frequency, SLRU, CLOCK-LRU-K, ARC. Notably **plain LRU is wrong** for
   layer-monotonic access (`q36` skips past layers during prefill for exactly this reason),
   and `qwen-fieldfare`'s **floor of 8 slots/layer** is load-bearing: with top-8 routing a
   layer below 8 slots misses *every* token.
5. **A hard RAM admission gate** — `slipstream-schero94` refuses a cache that would induce
   swapping (`peregrine_admission.c:15-71`). `qwisp`'s own field data: a 128-slot tier
   (~11.4 GB wired) **collapsed ~10× on a 16 GB Mac**; 64 slots recovered 6×.

### The proven way to stream experts *inside llama.cpp*

Two repos independently implement the same seam, which is the most directly applicable
finding here for us:

- **`siphon.cpp`**: an eval callback intercepts a synthetic graph tensor
  `moe_selected_experts_ds-<layer>` (`llama-context.cpp:3528`), reads the ids back to host
  (`:3596-3606`), streams the experts, **repoints `w->data` at the slot pool and shrinks
  `w->ne[2]` to the slot count** (`:3679-3727`), then rewrites the id tensor in place with
  slot ids (`:3730-3737`) so `MUL_MAT_ID` runs over a tiny pool. Restored after `ffn_moe_out`.
- **`slipstream-schero94`**: a 4,488-line patch pinned to llama.cpp `79bba02a6741`
  (`patches/slipstream-seams.patch`). `build_moe_ffn` gains `compact_slots`; duplicates the
  top-k tensor on callback `"pgrn_moe_topk_raw"` (`:2406-2423`) and swaps
  `selected_experts` → `expert_slots` in every `mul_mat_id`/`add_id` (`:2432-2506`).

`slipstream-schero94` is the closest thing to a drop-in: it is a llama.cpp fork and it is the
only repo here whose modifications are shipped as a reviewable patch against a pinned commit.
Its own benchmarks: 35B Q4 on internal NVMe goes **5.5 → 19 tok/s as the resident cache grows
2 → 14 GiB** (21% → 86% hit rate), decode is ~90% SSD-fetch-bound, and `io=8-16` with
`ubatch 2048` gave **2.7–3.2× prefill**. Caveat it states itself: Linux/CUDA is untested.
The largest documented lever in that file is not code at all — **2.7× just from moving the
streamed file to the internal NVMe.**

---

## 3. Our own measurement contradicts the top recommendation

The web pass ranked "kill mmap fault-around read-ahead" first, citing llama.cpp #23324
(+53% from an I/O-path swap). We tested the cheap version of it, and it is a trap.

`madvise(MADV_RANDOM)` on llama.cpp's GGUF mapping, applied via `/proc/self/maps` (the
descriptor belongs to llama.cpp, but the VMA is ours to advise), same evicted cache and 8 GiB
cap:

| | default | `MADV_RANDOM` |
|---|---|---|
| decode bytes read | 42.0 GB | **26.0 GB** |
| decode time | 56.0 s | **368.0 s** |
| effective throughput | 750 MB/s | **71 MB/s** |

Reads fell 38% and the run got **6.6× slower (368 s)**. Disabling read-ahead turns scattered
expert faults into synchronous random reads, and NAND random-read *latency* — not bandwidth —
takes over. **The 2.15× "amplification" is prefetch that pays for itself.**

The distinction the web pass blurred: #23324 replaced mmap with an **async `pread` pool over
exact expert extents with pipelining**, which is not the same as suppressing read-ahead on
demand faults. That is a rewrite of the weight-access path, and it is what the two llama.cpp
forks above actually do.

---

## 4. What the literature says is worth doing — and what is dead

**Dead ends, with reasons:**

- **Lossless compression of quantized weights.** Tested and rejected in print
  (arXiv:2508.19263): 4-bit value streams are essentially incompressible — only block scale
  factors compress, and those are ~11% of a Q4_K block. The best rANS decoder runs
  **114–120 MB/s**, far below our 750 MB/s. Decompression would cost more than the I/O saved.
- **Sophisticated cache-eviction policies.** arXiv:2608.07911 finds a 44–46% gap to the
  offline-optimal eviction exists, but a causal next-use rule recovers **none** of it (−11.4%)
  and picks the optimal victim 3.4% of the time versus LRU's 20.6–22.1%. Separately, against
  LRU an explicit co-activation prefetcher *wastes* bandwidth (LRU already retains co-firing
  experts). The cache-policy literature substantially overstates what simple policies recover.
- **Expert reordering by co-activation.** `flash-moe` measured 0%: at 3.9–7 MB per expert,
  NVMe ignores scatter at that granularity. Coalescing helps; grouping does not.
- **Learned next-expert predictors.** Every prediction result found is evaluated
  GPU-resident (PCIe/HBM-shaped), and Pre-gated MoE requires retraining. The one macOS
  implementation that shipped it (`dwijenpatel`) measured it neutral-to-negative.

- **Warming the page cache from outside the engine.** Tested twice, both worse. Layer-ordered
  `POSIX_FADV_WILLNEED` over the 40 layers' expert extents changed `read_bytes` by **0.14 GB
  across a whole run** — the kernel caps an advised window to a small multiple of
  `read_ahead_kb` (128 KB here), so 16 MB chunks became ~128 KB requests and the advice was
  never acted on. Replacing it with a real `preadv` that *did* read all 18.3 GB made the run
  worse: read 42.0 → **56.1 GB**, time 56.0 → **57.56 s**, because clean unreferenced pages are
  the first reclaim victims under an 8 GiB cap — evicted before the decoder arrives, then
  faulted in a second time. The prior art's "hard RAM admission gate" is the same lesson from
  the other side: under a fixed cap, stage in the engine's own buffers, never the page cache.

**Worth doing:**

- **Minimize the number of *distinct* experts touched per pass (T), not the batch size.**
  OEA's latency model is `b·T + a·Bk` with R²>0.99 — latency is linear in T. Batch-aware
  routing (Lynx, arXiv:2411.08982 — AffinityBinning; OEA, arXiv:2511.02237) remaps
  token→expert assignments within a batch so tokens piggyback experts already loaded:
  **39% MoE decode-latency cut on Qwen3-30B at batch 16**, <1 pp accuracy loss. Our ubatch is
  already ≥16, so the reuse pool is larger than their evaluation. **Lossy.**
- **REAP router-weighted expert pruning** (arXiv:2510.13999, ICLR 2026): one-shot, **no
  fine-tuning**, 50% compression at ~96–97% retention on Qwen3-Coder-480B. Pruning provably
  beats merging — merging destroys the router's input-dependent modulation. Halves bytes read.
  **Lossy.**
- **Expert-contiguous repacking**: measured **36× fewer page faults** (192 vs 6,912 pages per
  expert) and, with Q4 experts (~2.2 MB), pipelining can hide I/O entirely — compute-bound
  even at 0% cache hit. **Lossless.**

---

## 5. Ranked recommendations for our path

Given our measured state (60.2 GB / 56.6 s, ~100% I/O-bound, 8 GiB cap):

1. **Do not kill read-ahead.** Measured 6.6× regression. Leave mmap alone until there is a
   real pread pool to replace it with.
2. **Adopt the slot-remap seam inside llama.cpp** — port `slipstream-schero94`'s patch or
   replicate siphon's `w->data` repoint. This is the only route to *controlling* which
   experts are read. Two independent implementations exist; one ships as a patch against a
   pinned commit. **Lossless, highest ceiling.**
3. **Repack experts to a contiguous, 16 KiB-aligned slab, one read per expert.** Lossless,
   and `apus` shows the exact layout (`expert_slabs`: layer, expert, shard, offset, nbytes).
4. **Bound the cache with a RAM admission gate, and floor it at 8 slots/layer.** `qwisp`'s
   10× collapse at 11.4 GB wired and `qwen-fieldfare`'s 8-slot floor both apply directly to
   an 8 GiB budget.
5. **If quality budget allows, REAP 50% pruning** — the only lossy lever with a one-shot,
   no-fine-tuning 50% byte reduction.
6. **Skip:** lossless weight compression, elaborate eviction policies, co-activation
   reordering, learned predictors. Each has a measured negative or an inapplicable regime.
7. ~~**Cheap and independent of all the above:** move the GGUF to the internal NVMe — the
   largest single documented lever (2.7×) in the closest repo to ours.~~ **Already taken, and
   retracted.** Measured: the GGUF is *already* on `/dev/nvme0n1p5` (Kingston SFYRD4000G,
   `rotational=0`), and `dd iflag=direct bs=1M` reads it at **2.4 GB/s**. Scoring achieves
   ~750 MB/s — ~31% of the device — so the gap is access pattern, not placement. This was
   the one recommendation in this list that cost nothing and it was void before it was read.

8. **Mildly oversubscribe the CPU threads.** Not in any of the 12 repos, and the largest
   measured win found after the survey: **54.6 s → 38.4 s (−29.6%)** at 4/3 of hardware
   threads, lossless, read volume unchanged. A thread waiting on an expert page fault is not
   runnable, so an expert-streaming decode wants more runnable threads than the CPU has.
   The project's own runner was pinning half the host for no recorded reason.

**The honest summary:** every working engine here uses pread over an offset table with a
bounded slot cache, and the two that stream *inside llama.cpp* both do it by rewriting the
expert-id tensor so `MUL_MMAT_ID` sees a small pool. Our mmap-based path is the outlier, and
its 2.15× overhead is prefetch rather than waste — which is why the cheap fix backfired and
the real fix is a weight-access rewrite.

**Follow-up, measured after the survey** (`SEMIF_LLAMACPP_SSD.md` §6c): of the seven
recommendations above, the only one that paid was the one *not* in the prior art —
**CPU thread oversubscription, −29.6%**. The two prefetching routes (#1 and #2's cheap
cousin) both measured worse, the storage lever (#7) was already taken, and the ~2× read
amplification turned out not to be a second ubatch sweep. The remaining real lever is still
#2/#3, and it is a rewrite, not a knob.

---

## Appendix — artifacts and method

- Clones: `offload_projects/` (12 repos, shallow). The two `slipstream` repos collided on
  clone; the second is `slipstream-schero94`.
- Our experiment: `/tmp/probe_madvise.py`, `/tmp/run_madvise.sh` (evicted cache, 8 GiB cap,
  `MADV_RANDOM` applied to the GGUF VMA found in `/proc/self/maps`).
- Phase attribution of reads: `/tmp/probe_phases.py` (load vs decode; `/proc/self/io`).
- Five parallel source/web passes over: Edge0; `q36`+`apus`+`flash-moe`;
  `Mference`+`qwisp`+both `slipstream`s+`qwen-fieldfare`; `samosa-chat`+`slipstream-schero94`+
  `siphon.cpp`+`Qwen-MoE-Router-exp`; and the published-algorithms web pass.
- file:line citations throughout are from the clones at the revisions fetched 2026-09-21.
