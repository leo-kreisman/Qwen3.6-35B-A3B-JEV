# 03 — Fragmentation and coupling

The layer above the engine surveys. `docs/ARCHITECTURE-DECOMPOSITION.md` answers *who solved each
part first, in another field*. This dossier answers the two harder questions:

1. **Where does the pipeline actually fragment?** A streamed MoE engine is eleven separable
   parts, and the losses are not evenly spread. They are concentrated at the seams.
2. **What does each part cost, in the currency we measure?** Every import below is priced in
   *bytes read per decision* or *decisions per second*, against our own measured baseline, so
   "this is worth doing" is an arithmetic claim and not a taste claim.

**Provenance tags used throughout.** `[verified-here]` = measured in this repo
(`docs/STATUS-2026-09-22.md`, `docs/SEMIF_LLAMACPP_SSD.md`, `docs/ASSEMBLED-DESIGN.md`).
`[verified-source]` = read from the primary source during this review (title + abstract or the
quoted passage), URL given. `[unverified]` = secondary claim or a number I could not confirm in
the primary text. **Nothing here is estimated silently.**

---

## 0. The baseline everything is measured against

| figure | value | source |
|---|---|---|
| cold decode, 8 GiB cap verified binding | **3.67 tok/s at 212.6 MB read/token** | `[verified-here]` |
| warm decode, checkpoint page-cached | 10.68 tok/s, 0 bytes read | `[verified-here]` |
| prefill (the product's hot path), cold | 23.0 / 27.2 / 28.6 token-forwards/s; 41.0 / 64.1 / 128.3 GB for 4 / 8 / 16 decisions | `[verified-here]` |
| routed expert set, one prefill over the fixed prompt | 9.3 GB | `[verified-here]` |
| raw NVMe | ~2.4 GB/s | `[verified-here]` |
| predicted decode ceiling | 2.4 GB/s ÷ 566 MB/token ≈ 4.2 tok/s | `[verified-here]` (cold lands under it) |

Two numbers govern every ranking that follows, derived from the table above with arithmetic only:

- **Prefill amplification**: 41.0 GB served for 9.3 GB of routing-selected bytes = **4.41×**.
  Per decision: 10.25 GB read against a per-decision routed set of 9.3 GB = **1.10×**.
  So per *decision* the read path is already near-optimal; the 4.41× is entirely the
  8 GiB **cgroup-driven intra-pass re-read** of the same experts across 4 sequential decisions.
- **Decode amplification**: 212.6 MB/token against a 566 MB routed-set figure = **2.66×**
  intra-run reuse already banked. The remaining lever is not hitting more, it is **reading less
  per token**, or **paying one read for several tokens**.

---

## 1. Decomposition — eleven separable parts, and the two coupling seams

### 1.1 The parts

| # | part | interface it owns | measured cost driver |
|---|---|---|---|
| (1) | **On-disk layout / blocking** of variable-size expert data + alignment | "expert → extent(s)" | 4096 B logical blocks ⇒ a 590 KB–7 MB expert is 144–1700 extents, not one |
| (2) | **Offset / index table and its residency** | "expert id → offset" | one extra random read per miss unless co-located or resident |
| (3) | **Read path** | "extent → pinned user buffer" | queue depth, not bandwidth, when extents are small |
| (4) | **Slot pool / admission + eviction** | "expert → resident slot" | hit rate; admission dominates eviction under Zipf routing |
| (5) | **Working-set protection** | "this pass may not evict itself" | cap-driven 2.12× re-read `[verified-here]`: 19.3 → 41.0 GB |
| (6) | **Prefetch / prediction** | "issue the read before demand" | decode-solvable, prefill-unsolved |
| (7) | **I/O scheduling / queue depth** | "keep the device busy" | our 4/3×nproc result is a *substitute* for missing qd |
| (8) | **Work grouping / batching** | "one pass, many tokens" | `n_batch = n_ubatch = seq×512`: 67.0 → 42.0 GB `[verified-here]` |
| (9) | **Weight coding** | "bytes per expert" | Q4_K = 4.5 bits/weight, 11.1% of which is fp16 metadata |
| (10) | **Compute ↔ layout coupling** | "the layout the dequant kernel wants" | `q4_K_8x8` wants a repack that costs residency |
| **(11)** | **Residency control as an explicit gate** | "what may be resident at all" | `qwisp` collapsed ~10× without a gate; cap stability |
| **(12)** | **Execution-graph / prefix-sharing seam** | "can branches share a prefilled prefix?" | `llama_memory_seq_cp` SIGABRT on `qwen35moe` ⇒ 908 token-forwards where 100 would do `[verified-here]` |

**Parts (11) and (12) are additions to the ten in `ARCHITECTURE-DECOMPOSITION.md`, and the
requested eleventh/twelfth parts — yes, there are two, not one.** They are the two that
compound rather than optimize:

- **(11) Residency admission as a first-class gate.** Not "which victim", but "may this be
  resident *at all*". Best-in-class in another field: the **kernel's swap-vs-cache decision**
  (`vm.swappiness`, `memory.max`, and cgroup file-page charging) — i.e., admit into a *bounded
  engine-owned pool* or refuse, never "let the page cache have it and find out". The measured
  argument is our own and it is the strongest one in the repo `[verified-here]`: a real `preadv`
  prewarmer made things **worse** (42.0 → 56.1 GB) because clean unreferenced page-cache pages
  are the first reclaim victims under a cgroup cap, while an engine-owned bounded pool is
  reclaimable exactly once, at the slot level, where the policy can see the routing. Best
  LLM-specific solution: **`slipstream-schero94`'s admission gate** (`peregrine_admission.c:15-71`,
  refuses a cache that would induce swapping) with `qwisp`'s field data as the argument
  (128-slot ≈ 11.4 GB wired **collapsed ~10×** on a 16 GB Mac; 64 slots recovered 6×).
- **(12) The execution-graph / prefix-sharing seam.** This is a *coupling*, not a component: the
  scorer's own batching is 2× **slower** warm because hybrid recurrent/attention state cannot be
  copied per sequence, so each of 4–16 typed decisions re-walks all 40 layers and re-reads every
  expert slab. Best-in-class in another field: **distributed-systems request coalescing /
  single-flight** (Nginx `proxy_cache_lock`, memcached's lease/single-flight) — one execution for
  N identical prefixes, with latecomers waiting on the first. Best LLM-specific solution:
  **vLLM's prefix caching + RadixAttention and PrefillOnly's KV lifetime management**
  (`https://arxiv.org/abs/2605.02960` cites PrefillOnly, SIGOPS 2025). Our blocker is specific
  and verified: `llama_memory_seq_cp` SIGABRTs on this checkpoint `[verified-here]`. **Named
  owner of the part: vLLM prefix caching / PrefillOnly.**

### 1.2 The two coupling seams (what "fragmentation and coupling" actually means here)

| seam | the tension | who resolves it best | resolution |
|---|---|---|---|
| **A. Kernel ↔ disk layout** | the byte layout `q4_K_8x8` dequantizes fastest is not the layout that makes an expert one extent | **Apache Arrow / Parquet dictionary order + Blosc2 NDim two-level partitioning** (`https://blosc.org/docs/Exploring-MilkyWay-SciPy2023-paper.pdf`) | make the *storage unit* the dequant block-group, not the tensor; a repack is a *derived artifact*, and the derived artifact is what goes on disk |
| **B. Placement ↔ routing** | where an expert sits decides whether routing can be executed at all, and routing decides what must be moved | **MoE-Prefill / AsyncEP** (`https://arxiv.org/abs/2605.02960`) | decouple them: gather experts **by weight** on a static layer schedule instead of routing them **by activation** |

Seam B is the single most important sentence in this dossier, and it is not in
`ARCHITECTURE-DECOMPOSITION.md`. Quoting the primary source `[verified-source]`:

> "We observe that these overheads stem from coupling expert placement with synchronous
> activation routing — a design inherited from the decoding era. The long, compute-bound forward
> passes of large-batch prefill open a per-layer window wide enough to stream expert weights in
> the background, replacing per-layer activation AllToAll with asynchronous weight AllGather fully
> overlapped with computation."

That is a statement about *our exact problem* — prefill, where the whole layer's expert set is
touched and prediction is useless — written by the Snowflake AI Research group, with numbers:
**1.35–1.37× throughput over the strongest distributed baseline, up to 1.59× on long-context
synthetic workloads, 29.8–36.2% per-GPU MFU on Qwen3-235B-A22B across four hardware/precision
configurations, deployable envelope widened from "≥4 GPUs" to "1–8 GPUs"** `[verified-source]`.
It also confirms the product framing is real and production-sized: **prefill-only workloads are
"65.3% of all input tokens served"** in their measured production cluster `[verified-source]`.

**Its central mechanism is the resolution of the prefill expert-set problem.**
`T_EP` (the time to move one layer's expert set) is compared against the layer's compute window
`T_layer`; while `T_layer ≥ T_EP`, offloaded experts "appear always-resident by construction".
The "saturation threshold" `T` is enforced by the *frontend* as a batch-size rule. **That is a
prediction-free prefetch: it needs no oracle, only a big enough batch.** The stated limitation is
the mirror image of ours: "on low-bandwidth interconnects `t_EP` and thus `T` grow, potentially
re-exposing some transfers on the critical path" `[verified-source]`. Our interconnect is the
slowest one there is (a single NVMe at 2.4 GB/s), so we cannot import it whole — but we can
import its *invariant* and ask what batch size makes it hold, and when it cannot hold, we are in
the regime where speculation (below) is the only lever left.

---

## 2. Part-by-part: best-in-class, elsewhere and in LLM land

Format per part: **primitive → best from another field → best LLM-specific → import → the
measurement it moves.**

### (1) On-disk layout, blocking of variable-size experts, alignment

- **Primitive.** How a variable-size expert becomes one addressable extent on a device with
  4096 B logical blocks.
- **Elsewhere — EROFS fixed-output compression.** SquashFS cuts fixed *input* (128 KiB chunks →
  variable output); EROFS cuts fixed *output* (pclusters). `[verified-source]` for the design
  distinction (USENIX ATC'19 EROFS slides, `https://www.usenix.org/sites/default/files/conference/protected-files/atc19_slides_gao.pdf`,
  which show fixed-sized-input compression → read amplification, and fixed-sized-output → reduced
  read amplification + in-place decompression). `[unverified]` for the **165 MB of I/O to serve
  16 MB of random reads** figure, the "at most two clusters per request" claim and the LZ4
  30.9 vs 26.3 MiB/s benchmark — those are carried from `ARCHITECTURE-DECOMPOSITION.md` and from
  `https://sigma-star.at/blog/2022/07/squashfs-erofs/`, and I did not re-derive them.
- **Elsewhere, second opinion — Zarr v3 sharding codec.** Packs many inner chunks into one
  storage object ("shard") with a **16-byte-per-chunk index**, and the index may be placed at
  `start` or `end` (`https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/`)
  `[verified-source]`. This is the *variable-size-member, one-object* pattern we need.
- **LLM-specific.** Every working engine in our survey converges on the same shape: one
  contiguous slab per expert at 16 KiB alignment (`apus` 6,291,456 B; `flash-moe` 7,077,888 B
  stride; `qwen-fieldfare` `Q4EXP02` uniform 1,769,472 B stride; `slipstream-schero94`
  `PGRN_ALIGN=16384`) `[verified-here]`.
- **Import.** Cut on **output** boundaries (16 KiB pclusters), never on 4096 B input blocks;
  one expert = one extent; the extents that a single layer touches are co-located so a layer's
  set is a *range*, not a scatter.
- **Moves:** prefill amplification **4.41× → ≤1.2×** (design gate already in
  `ASSEMBLED-DESIGN.md`, phase 2). At 9.3 GB routed set and 2.4 GB/s, the prefill floor is
  **≈3.9 s/decision** versus the measured **9.85 s/decision** — a **2.5×** headroom that is
  physical, not algorithmic `[arithmetic on verified-here numbers]`.

### (2) The offset/index table and where it lives

- **Primitive.** Resolve an expert id to bytes without paying a second random read.
- **Elsewhere — LSM/SSTable trailing index block** (index adjacent to the data it indexes);
  **Zarr shard index** (16 B/chunk, at start or end of the shard) `[verified-source]`;
  **git multi-pack-index / reverse index**, which exists precisely to make "which packfile, what
  offset" a single lookup instead of a scan across many packs
  (`https://git-scm.com/docs/multi-pack-index`) `[verified-source, doc-level]`;
  **TileDB fragment metadata**, which consolidates per-fragment metadata to stop a growing count
  of fragments from degrading reads (`https://github.com/TileDB-Inc/TileDB/wiki/Architecture`;
  the design paper is `https://people.csail.mit.edu/stavrosp/papers/vldb2017/VLDB17_TileDB.pdf`,
  `[unverified]` for exact figures).
- **LLM-specific.** `qwen-fieldfare`'s 4 KiB header inside the container; `siphon.cpp`'s 16 KiB
  aligned sidecar manifest; `flash-moe`'s fixed in-expert component offsets (the table is
  arithmetic, not stored).
- **Import, two options, ranked.** (a) **Fixed-stride arithmetic** when every expert of a layer
  has equal size — zero table bytes, zero lookups (`flash-moe`'s pattern); (b) otherwise a
  **header per shard** plus a **resident** global table. Do not leave it as a separate file whose
  every access is its own random read.
- **Moves:** removes a whole class of second-order latency. At 40 layers × 8 experts × 1 read/token
  a separate index file is up to 320 extra 4 KiB reads/token; against 212.6 MB/token at 4096 B
  granularity that is **≈1.3 MB/token, ≈0.6%** — small in bytes, but it is 320 *more extents in
  the queue*, and extents, not bytes, are what part (3) says is the real constraint `[arithmetic]`.

### (3) The read path — the clearest measured gap

- **Primitive.** Get 590 KB–7 MB extents off NVMe into pinned buffers at depth.
- **Elsewhere — the DBMS consensus, measured.** `https://github.com/nazq/io-uring-bench`
  `[verified-source, README table]`:

  | strategy | IOPS | throughput |
  |---|---|---|
  | sync `pread`, 1 thread | 28,418 | 111 MiB/s |
  | threaded `pread`, 24 cores | 564,188 | 2,204 MiB/s |
  | **io_uring qd=32** | 752,937 | 2,941 MiB/s |
  | **io_uring qd=128** | **974,495** | **3,807 MiB/s** |

  Corroborating study: "High-Performance DBMSs with io_uring: When and How to use it"
  (`https://arxiv.org/pdf/2512.04859v1`) `[verified-source, abstract]`. Registered buffers /
  fixed files remove per-op fd lookup and page pinning
  (`https://kernel-internals.org/io-uring/fixed-buffers/`,
  `https://man7.org/linux/man-pages/man7/io_uring_registered_buffers.7.html`) `[verified-source,
  doc-level]`.
- **LLM-specific.** Every working engine uses `pread`/`preadv` against a slab table; `flash-moe`
  measured **mmap as 5× worse than pread** for expert files `[verified-here, in-repo survey]`.
  `samosa-chat` measured **0.8 GB/s buffered vs 2.3+ GB/s O_DIRECT** `[verified-here]`.
- **The reframe our own result already implies.** Our **4/3×nproc** thread-oversubscription win
  (54.6 → 38.4 s, **−29.6%**, reads unchanged at 41.2–42.1 GB `[verified-here]`) is not a law of
  expert streaming: it is the *substitute for queue depth*. A thread blocked on a page fault is
  not runnable, so we added runnable threads. io_uring supplies depth directly, and at qd=128 it
  is **1.73× beyond the best 24-core threaded pread** in the benchmark above `[arithmetic on
  verified-source]`.
- **Caveat that must stay attached.** On a warm page cache io_uring is *slower* (≈7 vs ≈8 GB/s)
  `[unverified — carried from ARCHITECTURE-DECOMPOSITION.md §3, not re-verified here]`; it only
  wins cold with `O_DIRECT`. Our whole measurement regime *is* cold with `O_DIRECT` available.
- **Import.** io_uring + `O_DIRECT` at qd 32–128, registered buffers, one ring per I/O worker,
  `scheduler=none`, `nomerges=2`, `rq_affinity=2`.
- **Moves:** decode 3.67 → up to the 4.2 tok/s ceiling only if reads fall too (depth alone buys
  latency, not bytes); **prefill** is where it pays because prefill is extent-limited, not
  byte-limited: 10.25 GB/decision served at 4.4× amplification.

### (4) Slot pool / cache admission and eviction

- **Primitive.** Which experts stay resident.
- **Elsewhere — W-TinyLFU (Caffeine).** `[verified-source]` from three places:
  the Caffeine wiki states ARC "is also patented and cannot be used without a license agreement
  with IBM" (`https://github.com/ben-manes/caffeine/wiki/Efficiency`); the TinyLFU paper states
  "for most traces, **even an LRU eviction policy enhanced with a TinyLFU admission policy
  obtained the same (best) results**", that W-TinyLFU "tops or equals all other cache management
  policies we have experimented with, and it is the only policy that performed so well
  consistently", and that a window of **1%** of total cache outperformed or tied all schemes on
  the majority of traces (`https://arxiv.org/pdf/1512.00727`, `[verified-source]`).
- **The reframe:** **admission dominates eviction under skewed access.** Our routing is Zipf-ish;
  the expensive half of cache design is the half we do not need.
- **LLM-specific, and it is a real zoo.** `LFU-aging` (`Mference`, `slipstream-dwijenpatel`:
  halve counts every 32 plans); `decayed routing frequency + LRU tiebreak` (`q36`: halve every
  16 tokens); `SLRU` (`qwen-fieldfare`); `CLOCK-LRU-K with HOT/WARM tiers`
  (`slipstream-schero94`); `ARC + execution pins` (`siphon.cpp`); `per-layer LRU + LFRU pins +
  free-list` (`apus`) — all `[verified-here, in-repo survey]`.
- **Newer published LLM-specific results (with URLs).**
  - **FlashMoE** (arXiv:2601.17063): a lightweight ML policy predicting **next-use distance**
    (an explicit Belady approximation), SSD-backed expert offload, "**up to 51% higher cache hit
    rate** over LRU/LFU and **up to 2.6× speedup**" on a user-grade desktop; also: SSD loading is
    **>70% of decoding latency**, and the load stage loads each unique expert **exactly once per
    layer** for the whole input batch `[verified-source, abstract + alphaXiv system-architecture
    summary]`. This is the only published *SSD-tier* expert-cache result I found, and its
    prefill behaviour is the same idea as (8) below.
  - **MoE-Infinity** (arXiv:2401.14361): request-level activation tracing drives replacement and
    prefetch; **3.1–16.7× per-token latency improvement** over vLLM/Ollama/DeepSpeed/BrainStorm
    on personal machines `[verified-source, abstract]`. Its traced pathologies — selective
    activation, **group activation**, **skewed reuse** — are the three things a policy must
    respect. It also reports the anti-result that matters to us: prediction without correct
    semantics gave **934 ms TPOT (BrainStorm) vs 485 ms (on-demand vLLM)** `[verified-source]`.
  - **Reproducible Evaluation of MoE Expert Caching** (arXiv:2608.07911): after correcting three
    evaluation artifacts, a large offline-optimal gap remains (**44.2–45.9%** across 13 frozen
    workloads), but a *causal* next-use predictor recovers **0%** of it and picks the optimal
    victim **3.4%** of the time versus **20.6–22.1% for LRU and LFRU** `[verified-source,
    abstract]`. Read correctly: **the gap does not mean a better policy exists**; it means
    recency/frequency heuristics are already near what causality buys. Do not spend the roadmap
    here.
- **Import.** W-TinyLFU **admission** (doorkeeper Bloom + CountMinSketch, ~8 B/entry, no ghost
  entries) over a cheap LRU window; keep the load-balancing *floor*. Which floor depends on the
  regime (below).
- **Moves:** hit rate at fixed resident bytes. On our traces the honest ceiling is not 51% — it
  is whatever remains between LRU/LFRU's 20.6–22.1% optimal-victim rate and the 44–46% offline
  gap, and the arXiv:2608.07911 result says the causal part is ~3.4%. **Rank it low.** The gate
  for adopting it: it must beat LRU on *our* trace, or it is not adopted.

### (5) Working-set protection against self-eviction

- **Primitive.** A pass must not evict pages it will re-touch before it finishes.
- **Elsewhere — the Linux kernel's MGLRU.** Four generations instead of active/inactive, plus
  **`lru_gen_min_ttl`**, a minimum time-to-live that protects the working set from reclaim
  (docs: `https://docs.kernel.org/7.1/mm/multigen_lru.html`, LWN
  `https://lwn.net/Articles/894859/`); and **PostgreSQL `pg_prewarm`** with
  `autoprewarm.interval` / dump-and-restore of the buffer contents across restarts
  (`https://postgresql.org/docs/current/pgprewarm.html`) `[verified-source, doc-level]`.
- **Also elsewhere — Denning's working-set model (1968)**, cited directly in the MoE caching
  literature (`https://arxiv.org/html/2608.07911v3` bibliography) `[verified-source]`: the
  working set is *defined* per pass, not per item.
- **LLM-specific.** `slipstream-schero94`'s execution pins (ARC + pins); `apus`'s LFRU pins over
  slab free-lists `[verified-here]`.
- **Our measurement this answers, exactly.** At a 16 GiB cap the same pass reads 19.3 GB; at
  8 GiB, **41.0 GB** — a **2.12×** amplification factor measured with no access-pattern theory
  involved `[verified-here]`. That is self-eviction: layer 0's experts are gone by the time
  layer 39 finishes.
- **Import, cheapest possible version first.** Because our access is **layer-monotonic within a
  pass**, the kernel's generation number is almost free for us: stamp each slot with
  `pass_id` (already available), and **refuse to evict slots whose `pass_id` equals the current
  pass** until the pool is exhausted; only then fall back to LRU/admission order. No ghosts, no
  sketches, no predictor.
- **Moves:** the 2.12× cap factor. Ceiling: 41.0 → 19.3 GB per pass at an 8 GiB cap is
  **1.1 s → 0.5×... ** concretely, prefill per-decision read 10.25 GB → **4.83 GB**, and the
  prefill floor drops from 9.85 s to **≈4.6 s/decision** `[arithmetic]`. **This is the highest
  bytes-per-effort import in the entire dossier, and it is a one-afternoon change.**

### (6) Prefetch / prediction, including the unsolved prefill case

- **Primitive.** Issue the read before demand.
- **Elsewhere — DBMS prefetching, mature and measured:** **GrASP** (LSTM as multi-label
  classification over LBA deltas; generalizes to datasets **250× larger than training**; **+45%
  hit ratio, −60% I/O time**, `https://arxiv.org/html/2510.11011`); **Pythia** (transformer
  encoder inside the Postgres buffer manager, up to **6×** on DSB OLAP,
  `https://openproceedings.org/2025/conf/edbt/paper-102.pdf`); **SeLeP** (encoder-decoder LSTM as
  time-series forecasting; **+40% hit ratio, −45% I/O**,
  `https://dl.acm.org/doi/10.14778/3659437.3659458`) — all `[unverified: carried from
  ARCHITECTURE-DECOMPOSITION.md, not re-read here]`. **Also elsewhere:** `pg_prewarm`'s
  autoprewarm dump/restore, and OS read-ahead — with our own measurement showing the kernel caps
  an advisory window to a small multiple of `read_ahead_kb`
  (`POSIX_FADV_WILLNEED` moved `read_bytes` by **0.14 GB** across a whole run) `[verified-here]`.
- **LLM-specific (decode):**
  - **`apus`: 79.6% recall** predicting layer L+1 from L's router — a real in-repo measurement
    `[verified-here]`. Contrast Edge0's 33-head prerouter, whose accuracy is **never measured in
    its repo** `[verified-here]`.
  - **`siphon.cpp`:** prefetch **distance** from an EWMA of `transfer_us` vs `layer_compute_us` —
    better than fixed lookahead, because it self-tunes to the device `[verified-here]`.
  - **`q36`:** dedicated prewarm pthread, 512 MiB window, startup hotlist `[verified-here]`.
  - ****Fate** (arXiv:2502.12224): gate inputs from *adjacent layers* predict routing; reported
    "up to **4.5×** over Load-on-Demand and **1.9×** over Expert-Activation-Path" (v1), and the
    ACM version states "**1.34×–5.07× prefill speedup and 1.26×–4.41× decoding speedup**"
    `[verified-source for both quotes; the two differ by version — cite the version you mean]`.
  - ****ST-MoE** (arXiv:2606.15453): spatio-temporal expert prefetching, "**85% expert prediction
    accuracy**", average speedups **2.5× / 2.2× / 1.5×** over GPU / Adap-Gating / Pre-gated MoE
    `[verified-source, abstract]`. Note it needs **reconfigurable hardware** — the predictor alone
    is not the artifact.
  - ****MoE-SpeQ** (arXiv:2511.14102): the draft model *is* the quantized model, exploiting
    "remarkable fidelity in its expert activation patterns relative to its full-precision
    parent" — a zero-training-cost oracle `[verified-source]`. **This is directly relevant to us:
    our Q4_K_S model can be its own predictor.**
  - ****ExpertFlow** (arXiv:2410.17954, DAC'26): a transformer routing-path predictor over *all*
    MoE layers in one forward pass, plus token scheduling and a predictive cache; "reducing GPU
    memory usage by up to **93.72%** and improving throughput by up to **10×** over strong
    offloading baselines on a single GPU" `[verified-source, abstract]`.
- **The unsolved case — and the three published things that are *not* the same thing.** Everyone
  is right that a one-token lookahead does nothing for a one-shot prefill. The published answers
  that exist are of a *different kind*:
  1. **MoE-Prefill/AsyncEP** — no prediction at all: make the layer's compute window cover the
     transfer time (part (1) and (8) above) `[verified-source]`.
  2. **Speculation** — the verification pass reads the *union*, one read for k tokens (part (7)
     below) `[verified-source]`.
  3. **FlashMoE's prefill stage** — for the input batch, load each unique expert **exactly once
     per layer**, i.e. **batch-level dedup of the routed set** `[verified-source, secondary
     summary of the paper's §Pre-filling]`.
  **None of them predicts a prefill expert set, and I found no paper that does.** What they do is
  remove the *need* to predict it. That distinction — the answer is structural, not predictive —
  is this dossier's main correction to the project's framing of the prefill gap.
- **Import.** Decode: router-L+1 with EWMA distance (`apus` recall + `siphon.cpp` distance).
  Prefill: `FlashMoE`'s once-per-layer unique-expert load **plus** AsyncEP's `T_layer ≥ T_EP`
  batch sizing **plus** the part-(5) pass-pin. Nothing else.
- **Moves:** decode miss bytes; prefill, indirectly, by making one pass read the union exactly
  once (9.3 GB, not 10.25–41.0 GB).

### (7) I/O scheduling and queue depth

- **Primitive.** Keep the device queue fed without burning a core per request.
- **Elsewhere.** io_uring SQPOLL/IOPOLL, registered buffers/files; thread-per-core runtimes
  (Seastar) as the *CPU-side* analogue; DBMS group commit as the *coalescing* analogue.
- **LLM-specific.** Our own 4/3×nproc result is the only LLM-specific scheduling measurement in
  the survey `[verified-here]`; the prior art drives I/O from its own worker pools and never needs
  oversubscription `[verified-here]`.
- **Import.** io_uring at depth with a **single submission thread per NUMA node** and completion
  polling; retire the 16-thread oversubscription to a fallback.
- **Moves:** the 54.6 → 38.4 s window is already banked. What remains to be won is *latency per
  extent*, which is exactly what the DBMS table measures, and it is **not quantifiable from our
  numbers today** — the honest form of the claim is: our 4/3×nproc result proves the decode is not
  purely I/O-bound, so depth will not move bytes, and its payoff must be measured against the
  post-ranks-1–2 baseline. **No projected seconds are offered here; that would be invention.**

### (8) Work grouping / batching

- **Primitive.** One pass over many tokens, one read per expert.
- **Elsewhere.** Database group commit / WAL batching (amortize fsync); **MapReduce
  combiner/local aggregation** (aggregate before the shuffle); GPU kernel fusion.
- **LLM-specific, and it is the best-understood knob we have.**
  - `n_batch = n_ubatch = sequences × 512`: **67.0 → 42.0 GB** `[verified-here]` — the default
    silently split a 908-token group into two ubatches, each re-walking all 40 layers.
  - **Chunked prefill / Sarathi-Serve** (arXiv:2403.02310): splits a prefill into near-equal
    chunks and creates **stall-free schedules** — but note MoE-Prefill's counter-observation that
    *small chunks hurt MoE* (they shrink GEMM dimensions and MFU) `[verified-source]`. For an
    SSD-streamed engine the trade is different from a datacenter one and must be measured, not
    assumed.
  - **MoE-Lightning/CGOPipe** (arXiv:2411.11217), best-in-class among *offloading* systems:
    a CPU–GPU–I/O pipelining schedule with paged weights, "up to **10.3×** higher throughput than
    state-of-the-art offloading-enabled LLM inference systems for Mixtral 8x7B on a single T4
    (16 GB)", reaching the throughput upper bound with **2–3× less CPU memory**
    `[verified-source, abstract]`; and it is the baseline SpecMoEOff beats by 2.1× average
    `[verified-source]`.
  - **MoE-Prefill's prefill-only atomic operation** formalizes exactly our product:
    `t* = argmax_{t∈C} logit(t|c)`, one pass, no decoding, with **abundant prefix sharing**
    `[verified-source]`. Also: "**KV-cache-free execution for prefill-only workloads**" — for a
    single prefill there is no KV to cache, which removes a whole allocation class.
- **Import.** (i) Explicit **batch-size-as-overlap-window** batching (AsyncEP's saturation
  threshold); (ii) once-per-layer unique-expert gathering; (iii) revisit `n_ubatch` upward as a
  first-class I/O parameter, not a memory parameter.
- **Moves:** prefill bytes/decision: 10.25 GB at amplification 1.10 with the 8 GiB cap's 2.12×
  removed by part (5) → the true routed floor **9.3 GB**.

### (9) Weight coding — the lossless question, corrected

- **Primitive.** Fewer bytes per expert, **no quality loss**.
- **Elsewhere — scientific/AI model storage:** **QStore** (VLDB 2025; paper
  `https://dl.acm.org/doi/10.14778/3778092.3778100` — **returned HTTP 403 to me**, so the paper
  itself is `[unverified]` here; the repository `https://github.com/illinoisdata/qstore`
  `[verified-source]` states the artifact "saving storage footprint by 2.2x and reducing load time
  by 1.8x", and the arXiv version is 2505.04081). It stores the model at two
  precisions plus the residual needed for lossless reconstruction; the "10.7–11.5 bits/weight vs
  24" figure is `[unverified]` — carried from ARCHITECTURE-DECOMPOSITION.md, not re-read.
  **ZipNN / NeuZip / Huff-LLM** (exponent streams
  are skewed; Huffman only, no LZ) and **Blosc shuffle/bitshuffle** (byte-plane transposition so
  a general codec can work) `[unverified — carried forward; not re-read here]`.
- **Our own dead-end note was testing the wrong stream, and now it is priced.**
  `block_q4_K` is 144 bytes for 256 weights = **4.5 bits/weight**, of which
  `d` + `dmin` (2 × fp16) + `scales` (12 B) = **16 of 144 bytes = 11.1%** is metadata — exactly
  the skewed-exponent data ZipNN/QStore target. Huffman to 0.7× on the metadata alone yields a
  **3.33% total byte reduction** (212.6 → 205.5 MB/token, 3.67 → 3.80 tok/s); to 0.5×, **5.56%**
  (200.8 MB/token, 3.89 tok/s) `[arithmetic on the GGUF block layout, which is a documented
  format]`.
- **Import.** Separate the **metadata stream from the payload stream** (byte-plane transposition),
  Huffman the metadata only, leave the 4-bit payload raw. Do it *inside* the 16 KiB pcluster of
  part (1) so decompression stays in-place.
- **Moves:** **3.3–5.6%** of all bytes read. Small — and therefore **rank it below every
  structural fix in this dossier**, which is the opposite of the ranking in
  `ARCHITECTURE-DECOMPOSITION.md` §The compounded proposal. That correction is deliberate: it
  buys single-digit percent and costs a compressor in the hot path.

### (10) Compute–layout coupling

- **Primitive.** The byte layout the dequant kernel wants versus the layout the disk wants.
- **Elsewhere — cache-hierarchy-aware array formats:** **Blosc2 NDim** two-level partitioning
  (chunks → blocks) mapped to L1/L2/L3, and **Btune** auto-selecting codec/filter per array
  (`https://blosc.org/docs/Exploring-MilkyWay-SciPy2023-paper.pdf`) `[unverified — carried
  forward]`. **Parquet dictionary/ordering** and **Arrow's columnar layout** are the same idea in
  the analytics world: lay out for the reader, store the reader's unit.
- **LLM-specific.** `use_extra_bufts=false` (repack off) is required to avoid OOM, but repack
  exists because **`q4_K_8x8` is faster to compute** `[verified-here]`. We traded compute for
  residency.
- **Import — the seam resolution.** Do not choose. Make the **repacked layout the on-disk
  layout**: a build-time pass emits the `q4_K_8x8`-friendly artifact, 16 KiB pcluster-aligned,
  one expert per slab. Then the kernel and the disk want the same bytes and the OOM pressure
  disappears because nothing needs a second copy. Size the slot pool against the **cache
  hierarchy** (L2/L3-resident hot experts, DRAM-resident warm, NVMe cold) rather than DRAM alone.
- **Moves:** removes the repack trade entirely; enables part (4)'s pool to be honest about what
  "resident" costs.

### (11) Residency admission gate, and (12) prefix/execution sharing

Covered in §1.1. Their imports:
- **(11)** a **hard admission gate** (`slipstream-schero94`, `peregrine_admission.c:15-71`) plus
  the rule that staging happens **into engine-owned buffers, never the page cache**; the measured
  argument is ours: a `preadv` prewarmer made reads **worse** (42.0 → 56.1 GB)
  `[verified-here]`. **Moves: cap stability at 8 GiB — the precondition for every other number in
  this dossier being reproducible.**
- **(12)** one execution for N identical prefixes (single-flight / request coalescing). **Moves:
  the scorer's whole per-decision cost**, because 4–16 typed decisions over the same prompt stop
  being 4–16 full 40-layer walks. Blocked today by `llama_memory_seq_cp` SIGABRT on
  `qwen35moe` `[verified-here]`; the workarounds are (a) fix/copy the recurrent state, or
  (b) batch at the *sequence* level with a shared prefill that the graph accepts, or
  (c) accept N passes and exploit (8) instead.

---

## 3. The MoE-specific literature, with URLs and numbers

### 3.1 Expert-level pipelining and double buffering

- **Fiddler** (ICLR'25, `https://arxiv.org/abs/2402.07033`, code
  `https://github.com/efeslab/fiddler`): each layer is placed on CPU or GPU; the **optimal
  execution strategy is chosen per layer from the number of input tokens per expert**;
  frequently-used experts are placed on GPU by **offline popularity profiling**.
  Numbers: **1.26× single-batch, 1.30× long prefill, 11.57× beam search**; **6.5× over
  DeepSpeed-MII on Phi-3.5-MoE on average** `[verified-source, abstract + §7]`.
  **Relevance: it is the only system that explicitly treats *long prefill* as a first-class case,
  and it wins there by 1.30×.**
- **Pre-gated MoE** (ISCA'24, `https://arxiv.org/abs/2308.12066`, camera-ready
  `https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf`):
  a **pre-gate function** in block N−1 selects block N's experts, so the migration overlaps
  execution; single-GPU deployment of large MoEs. `[verified-source: abstract and mechanism. The
  commonly quoted "1.5×" is `[unverified]` — the abstract I read states improvements without the
  number.]`
- **ST-MoE** and **Fate** as in (6) above. **SP-MoE** (arXiv:2510.10302): speculative expert
  prefetching that exploits structural correspondence between draft and target models + a
  **cutoff-layer policy** bounding per-layer prefetch depth + async prefetch threads with batched
  I/O: **1.07–3.5× TPOT speedup** `[verified-source, abstract]`.

### 3.2 Activation locality

- Mixtral's own finding: adjacent tokens select the same expert with probability above the
  12.5% random baseline for 8 experts, "**sometimes near 30%**"
  (`https://arxiv.org/pdf/2511.05814v1`, quoting Jiang et al.) `[verified-source]`.
- **MoE-Infinity**'s request-level traces name the three regimes that matter — **selective
  activation, group activation, skewed reuse** — and report that in some sequences
  "experts 48, 44, 23, and 3 exhibit higher reuse than others" `[verified-source, extracted
  text]`. Its warning is the important half: with a **uniform** distribution "the expert cache
  will fail to find which experts are more likely to be reused", and naive prediction performed
  *worse than on-demand* (934 ms vs 485 ms TPOT) `[verified-source]`.
- **Our own locality is the mirror image**: layer-monotonic within a pass (which is why plain LRU
  is actively wrong, and why the part-(5) pass-pin is nearly free).

### 3.3 Expert caching by popularity vs recency — and what is unusable

| policy | status | evidence |
|---|---|---|
| **LRU** | free; baseline | optimal-victim rate **20.6–22.1%** `[verified-source, arXiv:2608.07911]` |
| **LFU** | free | wins on MoE expert traces in `https://arxiv.org/pdf/2511.05814v1` (frequency beats recency); FlashMoE measures against LRU *and* LFU `[verified-source]` |
| **LFU-aging / decayed frequency** | free; **the LLM-shipped default** | `Mference`/`slipstream-dwijenpatel` halve counts every 32 plans; `q36` every 16 tokens `[verified-here]` |
| **W-TinyLFU (window + TinyLFU admission)** | **free, unpatented**, ~8 B/entry, no ghosts | TinyLFU paper: LRU *with* TinyLFU admission tied or beat everything on most traces; window 1%–40% `[verified-source]` |
| **SLRU / segmented LRU** | free (classical). Individual patent hits exist (`US6904501` mentions segmented-LRU prefetching) `[verified-source, patent text located, claim scope not analysed]` — **treat as free but check jurisdiction if shipping commercially** |
| **CLOCK** (and LRU-K/CLOCK-LRU-K) | free; the low-overhead LRU approximation | `slipstream-schero94`'s HOT/WARM CLOCK-LRU-K `[verified-here]` |
| **ARC / CAR** | **PATENTED — unusable without an IBM licence.** Caffeine: "It is also patented and cannot be used without a license agreement with IBM" `[verified-source]`; patent **US6996676B2** "System and method for implementing an adaptive replacement cache policy", IBM `[verified-source, patent located]`; CAR (Bansal & Modha) is the same IBM line `[verified-source, paper]` |
| **LIRS** | patent status **undetermined here** `[unverified]`. Free to implement from the paper; the ARC patent's own definitions section *discusses* LIRS in detail, which is the reason to have counsel look before shipping. Cost: needs ghost/large stack (~3× cache) `[unverified, carried forward]` |
| **LFRU** | free-ish; used in `apus`, and is the *second* best victim-picker in the arXiv:2608.07911 evaluation (20.6–22.1% range with LRU) `[verified-source]` |
| **Belady/offline-optimal as a *learned* target** | free to attempt | FlashMoE's ML next-use-distance predictor: +51% hit rate, 2.6× `[verified-source]`; but the causal predictor in arXiv:2608.07911 recovered 0% `[verified-source]`. **The two results disagree in direction; the difference is the workload and the SSD tier. Measure, do not assume.** |
| **MoE-SpeQ's "entropy-aware hierarchical caching" + lookahead-aware eviction** | unpublished-license | arXiv:2511.14102 `[verified-source, abstract]` |
| **FlashMoE's ML policy** | research artifact | arXiv:2601.17063 `[verified-source]` |

**Legal summary, stated plainly:** of the schemes named in the task, **ARC (and CAR) are patented
and cannot be used**; **W-TinyLFU, LFU/aging, CLOCK, SLRU and (probably) LIRS are free**;
**LIRS's status I did not determine** `[unverified]`. Practically this removes nothing we wanted:
TinyLFU is the better fit for Zipf anyway.

### 3.4 Prefill expert-set prediction

**No paper I found predicts a prefill expert set.** The honest statement, with what does exist:

- **Prefill touches most of the layer's expert set** — this repo's own framing, and consistent
  with MoE-Prefill's "the percentage of activated experts grows with the number of tokens"
  (`https://arxiv.org/pdf/2510.10302v1` discussing multi-token verification, `[verified-source]`).
- The three *structural* escapes (AsyncEP window; union-by-verification; batch dedup of the
  routed set) are in §1.2 and §2(6).
- Prefill-specific prediction attempts exist only as *cross-layer gate transfer* for serving
  (`Fate`, `ST-MoE`) and none claims a one-shot, full-length prefill number.
- **Consequence for the roadmap: the prefill expert-set problem is not solved by prediction.
  It is dissolved by (a) exact dedup within the batch pass, (b) pass-pinning so a pass reads its
  own set once, and (c) — if a single pass still cannot fit — one verification read shared by
  many tokens. Our product is already a single prefill pass, so (a)+(b) is where the win is, and
  the correct framing is "make the pass read 9.3 GB exactly once", not "predict the set".**

### 3.5 Amortizing a read across many tokens by speculative verification

This is the strongest imported idea in the survey, and the papers are consistent about *why*:
verification fetches the **union** of the draft tokens' experts — one read for k accepted tokens.

| system | mechanism | number | URL |
|---|---|---|---|
| **SpecMoE** (DAC'26) | self-assisted SD: the *target MoE itself* is the drafter; per-expert temporal activation; affinity-based expert selection; draft experts pinned resident | throughput up to **4.30×**; **CPU→GPU transfer reduced by up to 76.73%**; **2.25× retained when experts are on SSD instead of CPU DRAM**, where latency-hiding alone fails | `https://arxiv.org/abs/2604.10152`, `https://arxiv.org/pdf/2604.10152v1.pdf` `[verified-source: abstract + body + alphaXiv summary; the 2.25× is from the alphaXiv body summary, `[unverified]` at the level of the exact table cell]` |
| **MoE-SpeQ** | draft model predicts the expert *sequence*; Expert Lookahead Buffer; **Amortization Roofline Model**: maximize `E[accepted tokens]·S_token ÷ E[synchronous I/O bytes]`, with an I/O roof and per-k compute roofs; governor solves for k* online | **2.34× max speedup** over SOTA offloading on Phi-MoE | `https://arxiv.org/abs/2511.14102`, `https://arxiv.org/html/2511.14102v1` `[verified-source: model equations `I_amort(k)`, `Θ(k)`, `T_cycle(k)` read directly]` |
| **SpecMoEOff** (SC'25) | SD to *enlarge* each expert's workload; theoretical + empirical roofline; CPU chunked-attention verification kernel; hyperparameter optimizer | **2.5× average decode over SOTA**, **2.1× average over MoE-Lightning**, **11.3–27.5× (avg 17.7×) over DeepSpeed-ZeRO-Inference**; note it says SD-OnDemand exists and MoE-SpAc is a baseline | `https://arxiv.org/abs/2508.21706`, `https://arxiv.org/html/2508.21706v1` `[verified-source: abstract + §6.2]` |
| **DraftExpert** | reframes the unit: "**accepted tokens per expert-set expansion**"; fixed-footprint shared+top-1+draft-expert drafter; confidence–expansion truncation; target-expert prefetching | its own measurement that a top-1 drafter is 4.7× over target, top-2 2.7×, top-3 2.0× — i.e. **cheap drafting buys acceptance with expert loads** | `https://arxiv.org/html/2607.24434` `[verified-source: abstract + motivation]` |
| **BigMoMo** | mobile MoE runtime using the multi-token verification window to decouple expert movement from single-token execution: "weight reuse, contiguous flash reads, load-compute overlap"; on-flash expert reorganization by **runtime co-loading patterns** | qualitative; TPOT comparison vs MoE-SpAc / SpecMoEOff / SP-MoE in Fig. 1 | `https://arxiv.org/pdf/2609.14643v1.pdf` `[verified-source: abstract; the figure's values are `[unverified]` — not read numerically]` |
| **SP-MoE** | SD-aware offloading: draft-target prefetch correspondence, cutoff-layer policy, pipelined runtime | **1.07–3.5× TPOT** | `https://arxiv.org/abs/2510.10302` `[verified-source]` |
| **MoE-Spec** | verification-time **expert budgeting**: fixed per-layer expert capacity, drop the long tail by aggregate routing probability | **10–30% higher throughput than EAGLE-3** at comparable quality | `https://arxiv.org/pdf/2602.16052v1` `[verified-source: abstract + Fig. 1]` |
| **AcceptMoE** | verifier-side expert selector: target-router scores + offline commitment probabilities, self-sizing expert sets; **conditions eligibility on cache residency** | **1.290×** throughput with all experts in GPU memory, **2.06× under physical expert offloading**, **73.6–77.1% less host→device traffic**, **0.27 pp** mean accuracy loss across 12 model–task pairs | `https://arxiv.org/abs/2608.02989` `[verified-source: abstract]` |
| **Speculative weight streaming (SWS)** | lifts speculation into the *weight* domain: a micro-draft proposes an **assembly blueprint**, a verifier checks it against the real router path, misses fall back to exact reassembly | a governing inequality `(selection_accuracy × bandwidth_saved) > (miss_rate × (recompute_cost + fetch_penalty))` | `https://awesome.ecosyste.ms/projects/github.com%2Fiblameandrew%2Fspeculative-weight-streaming` (project description) `[unverified — a github-adjacent description, not a paper, and it self-describes as work-in-progress]` |

**Two things to take from this table.** First, the amortization roofline is the right *objective
function* for this project: with `E[accepted tokens]` in the numerator and **synchronous I/O
bytes** in the denominator, our measured 212.6 MB/token is the denominator. Second, **every one
of these systems assumes a draft that is cheap relative to the target.** For us the drafter must
not read from NVMe at all — which is exactly why SpecMoE's *self*-assisted draft (the quantized
model predicting its own routing, `MoE-SpeQ`'s stated fidelity result) is the only version that
can work on an SSD-only substrate.

### 3.6 Expert pruning / merging (REAP)

- **REAP** — "Router-weighted Expert Activation Pruning", Cerebras, `https://arxiv.org/abs/2510.13999`
  (v2), code `https://github.com/CerebrasResearch/reap`, blog `https://www.cerebras.ai/blog/reap`
  `[verified-source: abstract]`. Core claims read directly: expert **pruning beats merging** for
  generative tasks because merging "introduce[s] an irreducible error due to the loss of
  fine-grained routing control"; the criterion combines **router gate-values and expert activation
  norms** to minimize a reconstruction-error bound; evaluated across 20B–1T models;
  "**near-lossless compression on code generation tasks with Qwen3-Coder-480B and Kimi-K2, even
  after pruning 50% of experts**".
- For an SSD-streamed engine this is a **bytes lever, not a latency lever**: 50% fewer experts is
  ~50% fewer bytes per read. But it is lossy and one-shot at model level, and our product is a
  decision scorer where a small probability shift can flip an answer. **Gate: measure the
  decision-flip rate on our four fixtures *before* any pruning.** Note the repo already found
  batching moves the winning probability by ≤0.001% and threads are bit-identical
  `[verified-here]` — so we have the harness to measure a flip rate honestly.

### 3.7 Batch-aware routing

- **OEA** — "Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without
  Retraining", `https://arxiv.org/pdf/2511.02237v1` `[verified-source: abstract]`. Mechanism:
  tokens "piggyback experts that have already been loaded into memory due to being crucial to
  other tokens within the same batch". Numbers: **39% (Qwen3-30B) and 15% (Qwen3-235B) MoE-layer
  decode-latency reduction at batch 16 with no statistically significant accuracy loss**. Its
  framing is the one to steal: "MoE latency is governed by the number of **activated experts**",
  and load grows more slowly than batch size, so a memory-bound regime persists.
- Related: **SERE** (arXiv:2602.07616) similarity-based expert re-routing for batch decoding;
  **Minimum Expert Token ROuting** (`[unverified]`, surfaced only via a
  ResearchGate listing).
- **Note the collision with our product**: at batch 1 (one decision, one prompt) there is no
  cross-request piggyback to exploit. OEA pays off only if we batch *decisions*, which loops back
  to part (12) — the prefix-sharing seam.

### 3.8 Multi-GPU / multi-tier placement across VRAM–DRAM–NVMe

Two **idle RTX 5060 Ti 16 GiB** cards exist in this host and are unused by the current design.
That is the largest unexploited physical resource in the picture, so this section is deliberately
blunt.

- **DeepSpeed ZeRO-Inference** (`https://www.deepspeed.ai/2022/09/09/zero-inference.html`,
  paper `https://arxiv.org/pdf/2207.00032`) `[verified-source: blog]`: pins **all** weights in CPU
  or **NVMe** and streams layer-by-layer to the GPU; reduces GPU memory by up to two orders of
  magnitude (Megatron-Turing-530B FP16: 1 TB → 10 GB); **explicitly optimized for throughput with
  large batches**; and its central design conclusion is quotable and counter-intuitive —
  "**when a model does not fit in GPU, using GPU memory to increase batch size rather than to
  partially fit the model leads to faster token generation**". Its limitation is also stated:
  "DeepSpeed-ZeRO-Inference suffers from small batch size due to its design of storing all
  requests' KV cache on GPU, which leads to low throughput" (SpecMoEOff §6.2) `[verified-source]`.
- **MoE-Lightning** (ASPLOS'25; `https://arxiv.org/abs/2411.11217`, `.../html/2411.11217v1`):
  **CGOPipe** CPU-GPU-I/O pipelining with paged weights + **HRM** hierarchical-roofline policy
  search; **up to 10.3×** over offloading baselines on one T4 16 GB; throughput upper bound at
  **2–3× less CPU memory**; scales to Mixtral-8x22B/DBRX on 2–4 T4s `[verified-source: abstract]`.
- **MoE-Prefill/AsyncEP** (see §1.2) — the multi-GPU *and* single-GPU case, with the
  offload-to-DRAM extension `[verified-source]`.
- **KTransformers** (`https://github.com/kvcache-ai/ktransformers`, tutorial
  `https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md`)
  `[verified-source: README + tutorial, fetched raw]` — **this is the published system with the
  hard CPU/GPU-offload MoE numbers the task asks for, and the numbers are on a 24 GB GPU + 382 GB
  DRAM, not NVMe**:
  - 671B DeepSeek-V3/R1, Q4_K_M, **14 GB VRAM + 382 GB DRAM**;
  - decode: **8.73 tok/s (32 cores) → 11.26 (2×32 cores) → 13.69 tok/s (6 experts)** vs
    **4.51 tok/s for llama.cpp** on 2×32 cores = up to **3.03×**;
  - prefill: 54.21 → 74.36 → 255.26 → **286.55 tok/s**, up to **27.79×** over llama.cpp
    (prefill figures are **up to 28× faster** than llama.cpp);
  - the stated cause of the win: **AMX (matrix extension) + a "specially designed cache friendly
    memory layout"** — i.e. part (10) of this decomposition, solved by the vendor of the CPU;
  - and the honest limit they state themselves: "the inference process is still constrained by the
    CPU's computational speed and memory bandwidth".
  - Later releases add **3-layer (GPU–CPU–disk) prefix cache reuse** (June 30 2025) and
    **CPU-GPU expert scheduling** (Jan 22 2026) `[verified-source, README changelog]`.
- **Fiddler** (CPU+GPU per-layer placement, 1.26×/1.30×/11.57×) and **FlashMoE** (the SSD-tier
  system, 2.6×, +51% hit rate) as above.

**Does any published system do GPU/VRAM offload of MoE experts with hard numbers? Yes — several,
and the numbers are comparable across them only if you check the tier.** Sorted by the memory
tier each one actually uses:

| system | tier it streams from | hard number |
|---|---|---|
| KTransformers | CPU DRAM (+ GPU for attn) | 13.69 tok/s decode, 286 tok/s prefill, 671B model, 14 GB VRAM `[verified-source]` |
| MoE-Lightning | CPU DRAM / GPU | up to 10.3× on one T4 16 GB `[verified-source]` |
| Fiddler | CPU DRAM | 1.26× / 1.30× / 11.57× vs baselines `[verified-source]` |
| DeepSpeed ZeRO-Inference | CPU DRAM **or NVMe** | 10× scale reduction; throughput-oriented; beaten 11.3–27.5× by SpecMoEOff `[verified-source]` |
| SpecMoE | CPU DRAM **and SSD** | 2.25× retained on SSD where latency-hiding fails `[unverified at cell level]` |
| SpecMoEOff | CPU DRAM | 2.5× avg decode; 2.1× avg over MoE-Lightning `[verified-source]` |
| MoE-Prefill | GPU HBM + CPU DRAM | 1.35–1.37×, 29.8–36.2% MFU, 1–8 GPUs `[verified-source]` |
| FlashMoE | **SSD** | +51% hit rate, 2.6× over LRU/LFU systems `[verified-source]` |
| **ours** | **NVMe, no GPU** | **3.67 tok/s cold decode; 23–28.6 prefill tok-fwd/s** `[verified-here]` |

**The honest reading of that table:** no published system streams MoE experts from NVMe *without*
a GPU or a large DRAM tier, and the two that come closest — FlashMoE (SSD, on-device) and
SpecMoE's SSD variant — report *relative* improvements over their own baselines rather than
absolute tok/s on a fixed host. **Our 3.67 tok/s under a verified-binding 8 GiB cap is still a
number nobody else has published.** That is the project's one defensible unique claim, and this
dossier's imports are the plan to improve it without needing anyone else's hardware.

---

## 4. The ranked "best compound"

Ranked by **bytes or decisions moved per unit of engineering**, not by novelty. Each entry names
the measurement it must move, and the gate that kills it if it does not.

| rank | import | from | measurement it must move | kill gate |
|---|---|---|---|---|
| **1** | **Pass-pinned slots** — a slot whose `pass_id` equals the current pass is not evictable (part 5). Kernel MGLRU's `lru_gen_min_ttl` is the same idea. | Linux MM | prefill: 10.25 GB → **4.83 GB** per decision; 9.85 s → **≈4.6 s**/decision `[arithmetic on verified-here]` | if per-pass read does not fall below the 16 GiB-cap figure (19.3 GB/pass), the access pattern is not what we think it is |
| **2** | **On-disk repack = 16 KiB output-blocked pclusters, one expert = one extent, layer's set contiguous, plus the `q4_K_8x8` layout itself (seams A and part 10)** | EROFS + Zarr shard + Blosc2 NDim | amplification **4.41× → ≤1.2×**; prefill floor **9.85 s → 3.9 s**/decision at 2.4 GB/s | bytes read per decision does not drop ⇒ the amplification is not layout but cap-driven (then rank 1 carries everything) |
| **3** | **io_uring + `O_DIRECT` at qd 32–128, registered buffers, SQPOLL/IOPOLL** (part 3) | DBMS | retire the 4/3×nproc crutch; target the residual I/O wait implied by `T_layer − T_mem`; benchmark parity at **1.73×** the best threaded pread | warm-cache regression (io_uring slower on warm cache) ⇒ keep a buffered fast path |
| **4** | **Batch-size-as-overlap-window + once-per-layer unique-expert gather (AsyncEP's invariant, FlashMoE's prefill stage)** (parts 8, 6, 11) | MoE-Prefill + FlashMoE | prefill reads **9.3 GB exactly once**; MFU/window condition checked per layer | `T_layer < T_EP` on our NVMe ⇒ the window cannot open; then rank 5 is the only lever |
| **5** | **Speculative verification with a *resident* drafter (self-assisted, no NVMe for the draft)** (part 7) | SpecMoE / MoE-SpeQ / AcceptMoE / MoE-Spec | denominator of the amortization roofline: **212.6 MB per accepted token → 212.6/k MB**; target 1.5–2× on decode | drafter reads from NVMe, or acceptance `< ~1.4` at k=4 ⇒ the union of experts costs more than it saves (DraftExpert's own 4.7→2.0× warning) |
| **6** | **W-TinyLFU admission over a cheap LRU window, replacing pure LRU** (part 4) | Caffeine / TinyLFU | hit rate at fixed resident bytes, measured on *our* trace | if LRU+pass-pin already matches it, drop it — admission is a means, not a goal |
| **7** | **Metadata/payload stream separation with Huffman on the fp16 metadata only** (part 9) | QStore / ZipNN / Blosc bitshuffle | **3.33–5.56%** of all read bytes (212.6 → 205.5/200.8 MB/token) | decode CPU budget: if the Huffman cost exceeds the 3–6% byte saving, drop it |
| **8** | **Decode-only router-L+1 prefetch with EWMA distance** (part 6) | `apus` (79.6% recall) + `siphon.cpp` | decode miss bytes, **after** ranks 1–6 have moved the denominator | it must be measured against the *new* baseline, not the old one |
| **9** | **REAP pruning / batch-aware routing** (lossy) | REAP / OEA / SERE | bytes per read (pruning) or experts per batch (routing) | **decision-flip rate on the four fixtures** — if any fixture flips, stop |
| **10** | **Two idle RTX 5060 Ti 16 GiB** | MoE-Lightning / ZeRO-Inference / Fiddler | move the *hot* subset of experts and the dequant/reduce kernels off the streaming path; ZeRO-Inference's rule applies — spend VRAM on **batch size**, not on partially fitting the model | if the PCIe path becomes the bottleneck at our batch sizes, spend the VRAM on the scorer's attention instead |

**Why this ordering differs from `ARCHITECTURE-DECOMPOSITION.md`'s compound.** That document ranks
io_uring (3), weight coding (7), speculation (5) and TinyLFU (6) as the compound. Three
corrections, each from a measurement or a primary source:

1. **Weight coding drops from #2 to #7**: the arithmetic is 3.3–5.6%, not the "2.2× storage" of
   QStore (which is 2× *storage*, on a different bit-width, not 2× *read path*).
2. **Pass-pinning enters at #1 and is absent from that document**: it is the direct answer to the
   repo's own measured 2.12× cap amplification, it is a handful of lines, and it needs no model.
3. **"Solve prefill" changes character**: it is not a prediction problem. It is
   read-the-routed-set-once (rank 4) plus do-not-evict-yourself (rank 1), with AsyncEP's window as
   the published existence proof that the coupling — not the access pattern — was the obstacle.

### The smallest compounding set

If only three changes are ever made, they are **1, 2, 3** and they compound because each makes
the next one's ceiling higher: pinning makes the pool's residency meaningful (1), the repack makes
an expert one extent instead of 144–1700 (2), and io_uring makes many small extents cheap to
request (3). Rank 4 is the one that changes the *regime*, and rank 5 is the one that changes the
*currency* — but neither is worth attempting before 1–3, because both are judged against a
baseline that ranks 1–3 will have moved by a factor of ~2.

---

## 5. What is still unsolved

Stated as problems, not as plans.

1. **The prefill expert-set problem.** No published system predicts a one-shot prefill's expert
   set, and on this model 40 layers must be walked once and most of each layer's 256 experts are
   touched. The published escapes are structural (window, dedup, union-by-verification) and
   **every one of them assumes a faster tier than ours exists** — CPU DRAM in MoE-Prefill,
   a batch large enough to saturate, or an SSD tier at 2.25× *relative* gain. **Whether the
   AsyncEP window condition `T_layer ≥ T_EP` can be satisfied at all on a single NVMe at
   2.4 GB/s, for a 9.3 GB per-pass routed set, is not answered by any paper I found.** It is our
   problem, and it is measurable in one afternoon with our existing harness.
2. **Prefix sharing on hybrid recurrent/attention checkpoints.** `llama_memory_seq_cp` aborts on
   this architecture `[verified-here]`, which blocks the single highest-value fix in the product
   path (one execution for N typed decisions). No upstream fix, no published workaround, and no
   paper in the MoE-offload literature addresses hybrid-state copying. **This is unsolved
   infrastructure, not unsolved research.**
3. **Absolute NVMe-only numbers.** Every MoE-offload system with hard numbers streams from CPU
   DRAM or has a GPU; the SSD-tier systems report relative gains. Nobody has published absolute
   tok/s for MoE experts streamed from NVMe with no DRAM or VRAM tier. Our 3.67 tok/s cold under a
   verified-binding cap is the only such figure.
4. **Energy.** SSD offload measures up to **~12× the per-token energy** of HBM with the SSD at
   **73–80% of total energy**, and "prefetching hides *latency*, not energy"
   (`https://arxiv.org/html/2508.06978v1`, `[verified-source: abstract]`). Every improvement in
   this dossier is a speed claim; **none of them is an efficiency claim**, and the paper's own
   forward-looking condition (Flash read energy improving ~10×) is out of our control.
5. **Expert-cache policy has no reliable published winner.** arXiv:2608.07911 shows a 44–46%
   offline gap with only 3.4% recovered causally; FlashMoE shows +51% hit rate. They cannot both
   be the last word, and the difference is the tier. **For SSD-tier expert caches the evidence
   base is one paper.**
6. **Acceptance-rate physics for a *self*-assisted, NVMe-only drafter.** SpecMoE keeps the draft
   in GPU memory and reports 2.25× on SSD; MoE-SpeQ's draft lives on-device. Our drafter would run
   on CPUs sharing the same bus as the expert stream. **The amortization roofline's numerator
   (accepted tokens per cycle) and denominator (synchronous I/O bytes) both change when the
   drafter competes for the device.** No measurement exists for that configuration.
7. **The 8 GiB cap is the real product constraint, not the SSD.** Every number in this dossier
   improves faster if the cap rises than if any algorithm changes: 3.67 tok/s at 8 GiB, 10.68 warm
   with no cap. **The cheapest "speedup" available to this project is more RAM**, and honesty about
   that belongs in the same document as the imports.

---

## Appendix — provenance ledger

- **Measured in this repo** (`docs/STATUS-2026-09-22.md`, `docs/SEMIF_LLAMACPP_SSD.md`,
  `docs/ASSEMBLED-DESIGN.md`, `docs/ARCHITECTURE-DECOMPOSITION.md`, `README.md`):
  3.67 / 10.68 tok/s; 212.6 MB/token; 41.0 / 64.1 / 128.3 GB prefill reads;
  19.3 GB vs 41.0 GB at 16 vs 8 GiB cap; 9.3 GB routed set; 4/3×nproc = −29.6%;
  `n_batch=n_ubatch` 67.0 → 42.0 GB; `MADV_RANDOM` 42.0 → 26.0 GB but 56.0 → 368.0 s;
  page-cache prewarmer 42.0 → 56.1 GB; `POSIX_FADV_WILLNEED` moved 0.14 GB;
  `llama_memory_seq_cp` SIGABRT; `use_extra_bufts=false`; the 12-clone / 12-repo survey.
- **Read from primary sources during this review** (title + abstract or quoted passage):
  arXiv 2605.02960, 2604.10152, 2511.14102, 2508.21706, 2510.10302, 2607.24434, 2608.02989,
  2602.16052, 2510.13999, 2511.02237, 2401.14361, 2402.07033, 2411.11217, 2601.17063, 2606.15453,
  2608.07911, 2511.05814, 2308.12066, 2312.17238, 2508.06978, 1512.00727; the Caffeine wiki;
  patent US6996676B2; USENIX ATC'19 EROFS slides; Zarr shard-index spec; git multi-pack-index
  docs; DeepSpeed ZeRO-Inference blog; KTransformers README + DeepSeek tutorial (raw).
- **Carried forward unverified from `docs/ARCHITECTURE-DECOMPOSITION.md`** and marked as such
  inline: GrASP / Pythia / SeLeP figures; ZipNN / NeuZip / Huff-LLM / Blosc ratios; QStore's
  bits-per-weight breakdown; the io_uring warm-cache regression; EROFS LZ4 throughput figures;
  LIRS ghost-entry overhead; the SigmaStar benchmark numbers.
- **Not determined:** LIRS / SLRU patent status in the jurisdictions that matter for shipping
  (ARC/CAR are clearly IBM-patented and excluded). Consult counsel before shipping any adaptive
  replacement policy other than TinyLFU-family, CLOCK-family or plain LFU/LRU.
