# Architecture decomposition — each part, and who solved it first

The project's architecture is not novel in any single part. It is ten ordinary primitives
assembled into one pipeline. Each part below is answered *better* somewhere else, usually for
an unrelated reason — databases, filesystems, CPU caches, scientific data formats, version
control. This is that map: what the part is, who solves it best, and what is importable.

**Read the "Import" lines as the work list. The "Evidence" lines are why it is worth doing.**

---

## 1. On-disk layout and blocking

**Primitive:** how model bytes are cut into addressable units on the device.

**Who else solves it:** read-only compressed filesystems. **EROFS** uses *fixed-output*
compression (4 KiB pclusters → variable input) while **SquashFS** uses fixed *input*
(128 KiB chunks → variable output). That single difference is why SquashFS suffers brutal
read amplification on random access — ATC'19 measured **~165 MB of I/O to serve 16 MB of
random reads** — while EROFS reads at most two clusters per request and every byte read is
usable. EROFS also **interleaves data with metadata** for locality; SquashFS stores them apart.

**Best-in-class:** EROFS block-aligned fixed-output clusters, zstd, with in-place
decompression (99.6% of blocks qualify) and `cache_strategy=` control.

**Import:** cut expert slabs on **output** boundaries, not input. 16 KiB-aligned (already in
the convergent design), and co-locate the offset/index with the data it indexes (see §2).

**Evidence:** EROFS beats SquashFS on every compressor in the sigma-star benchmark (LZ4:
30.9 vs 26.3 MiB/s), at a modest density cost.

## 2. Addressing / offset table

**Primitive:** finding a named expert's bytes without an extra random read.

**Who else solves it:** LSM-tree storage engines. The **SSTable trailing index block** pattern
keeps the index adjacent to the data so a lookup is one seek, not two. Zarr/HDF5 chunk grids,
CDN piece maps and torrent piece hashes are the same idea in other clothes.

**Import:** the sidecar offset table is a *separate file* today — every lookup risks its own
random read. Fold the index into the slab. This is nearly free and removes a whole class of
second-order latency.

## 3. Read path — the biggest measured gap

**Primitive:** getting 590 KB–7 MB extents off NVMe into CPU buffers.

**Who else solves it:** databases. The consensus is unambiguous:

| strategy | IOPS | throughput |
|---|---|---|
| sync `pread`, 1 thread | 28,418 | 111 MiB/s |
| threaded `pread`, 24 cores | 564,188 | 2,204 MiB/s |
| **io_uring, qd=32** | 752,937 | **2,941 MiB/s** |
| **io_uring, qd=128** | **974,495** | **3,807 MiB/s** |

Sources: [nazq/io-uring-bench](https://github.com/nazq/io-uring-bench),
[io_uring for High-Performance DBMSs](https://arxiv.org/html/2512.04859),
[Fixed Buffers and Files](https://kernel-internals.org/io-uring/fixed-buffers/).

**Registered buffers** cut 30–40 µs CPU/op on 4 KiB random reads at depth 32; IOPOLL +21%,
SQPOLL +32%. **Caveat:** on warm page cache io_uring is *slower* (~7 vs ~8 GB/s) — it only
wins cold with `O_DIRECT`.

**Import — and this reframes our own best finding.** We use synchronous `pread` through mmap
demand faults. Our **thread-oversubscription result (4/3 nproc, 54.6 → 38.4 s, −29.6%)** is
not a fundamental law — it is the *substitute* for missing queue depth. We over-subscribe
threads because a faulting thread is not runnable and we have no async queue. io_uring gives
the same queue depth directly, and at qd=128 goes **1.7× beyond the best threaded pread**.
Also: `read_ahead_kb` is irrelevant under `O_DIRECT`; `scheduler=none`, `nomerges=2`,
`rq_affinity=2` are the NVMe knobs ([kernel queue-sysfs](https://kernel-internals.org/io/tuning-storage/)).

## 4. Cache / slot pool and eviction

**Primitive:** which experts stay resident.

**Who else solves it:** CDN and application caches. **W-TinyLFU** (Caffeine) is the only scheme
reported to do well on *all* traces: near-optimal hit rate, ~8 bytes/entry overhead, **no ghost
entries**, and **not patented**. ARC is stronger on some traces but is **patented by IBM** —
excluded on licensing grounds alone. LIRS needs ~3× cache for ghost entries.

**The reframe that matters:** TinyLFU's result is that **admission policy dominates eviction
policy**. Under skewed (Zipf) access — which MoE routing is — *LRU plus good admission* nearly
matches true LFU. So the expensive part (exotic eviction) is largely unnecessary; the cheap
part (a doorkeeper filter + CountMinSketch admission) is where the win is.

**Import:** replace LRU with **W-TinyLFU admission**. Keep the 8-slots/layer floor.
Sources: [Caffeine efficiency](https://github.com/ben-manes/caffeine/wiki/Efficiency),
[TinyLFU](https://arxiv.org/pdf/1512.00727).

## 5. Admission control / working-set protection

**Primitive:** refusing to admit what will cause reclaim — and not evicting your own working set.

**Who else solves it:** the Linux kernel. **MGLRU** (6.1+) replaces the active/inactive pair
with four generations and adds `lru_gen_min_ttl` — a minimum time-to-live that protects the
working set from being reclaimed. It also exploits that **unmapped pages need no TLB flush**.

**Import:** our measurement showed a pass evicting its own early pages before finishing (call 2
starts at layer 0 with nothing warm — "do not look for a residency scheme that compounds").
`lru_gen_min_ttl` is the kernel's name for exactly that failure mode, and the fix is the same:
protect the working set by age, not by position in an LRU. Sources:
[MGLRU docs](https://docs.kernel.org/7.1/mm/multigen_lru.html),
[LWN](https://lwn.net/Articles/894859/).

## 6. Prefetch / prediction

**Primitive:** issuing the read before the demand arrives.

**Who else solves it:** DBMS research, where this is mature — and it is the adult version of
Edge0's 33-head prerouter whose accuracy is never measured in-repo.

- **GrASP** — LSTM framed as multi-label classification over LBA deltas; generalizes to
  datasets **250× larger than training**; +45% hit ratio, −60% I/O time.
- **Pythia** — transformer encoder into the Postgres buffer manager; up to **6× on DSB OLAP**.
- **SeLeP** — encoder-decoder LSTM as time-series forecasting; +40% hit ratio, −45% I/O.

**The honest limit:** all of it is **decode-time**. Our expensive case is **prefill**, where a
one-token lookahead is useless — the whole layer's expert set is touched. None of the surveyed
engines solves prefill prefetch either. This remains the open problem.
Sources: [GrASP](https://arxiv.org/html/2510.11011), [Pythia](https://openproceedings.org/2025/conf/edbt/paper-102.pdf),
[SeLeP](https://dl.acm.org/doi/10.14778/3659437.3659458).

## 7. Scheduling / I/O dispatch

**Primitive:** keeping the device queue fed.

**Who else solves it:** thread-per-core servers (Seastar), work-stealing runtimes, and
io_uring's SQPOLL/IOPOLL modes. Our 4/3-nproc result is real and measured, but §3 shows it is a
workaround for synchronous I/O rather than a property of the problem.

## 8. Work grouping / batching

**Primitive:** amortizing one pass over many items.

**Who else solves it:** database group commit and WAL batching; GPU kernel fusion. Our version
is `n_batch = n_ubatch = sequences × 512` (67.0 → 42.0 GB). The deeper version is §11.

## 9. Weight coding — the lossless question

**Primitive:** fewer bytes per expert, **no quality loss**.

**Who else solves it:** scientific data compression and AI-model storage. Four directly usable
results:

- **QStore** (VLDB 2025) — stores the model at two precisions: low-precision **plus the residual
  needed for lossless reconstruction**. Groups floats **by quantization function (block scale),
  then by post-quantization value**, and Huffman-codes per subgroup. **2.2× storage reduction,
  10.7–11.5 bits/weight vs 24.** This is precisely our structure: Q4_K_S = 4-bit quants + fp16
  block scales/mins.
- **ZipNN** (IBM) — the compressibility is in the **exponent**, which is highly skewed (top 12
  values ≈ 99.9% of parameters). Extract exponents into their own stream, group by byte,
  **Huffman only** (skip LZ — there is no multi-byte repetition). BF16 ~33% savings; "clean"
  FP16 up to >50%; **80 GB/s decompress** multi-threaded, 1.65 GB/s single-thread.
- **Huff-LLM** — Huffman over FP16 sign/exponent/mantissa-MSB/LSB as separate streams;
  24–32%, single-cycle hardware decoder.
- **Blosc shuffle/bitshuffle** — byte/bit-plane transposition so a general codec can work;
  lz4-bitshuffle runs ~21 GB/s compression; ratios to 83× on structured scientific data.

**Import — and a correction to our own dead-end note.** We recorded "lossless compression of
quantized weights" as dead because 4-bit streams are incompressible and rANS decodes at
114–120 MB/s. **That tested the wrong stream.** The 4-bit quants *are* incompressible; the
**fp16 scales and mins are not** — their exponents are exactly the skewed data ZipNN targets.
The move is to **separate the metadata stream from the payload stream** (byte-plane
transposition), compress only the metadata with Huffman, and keep the payload raw. QStore is
the published version of this. Sources: [QStore](https://www.vldb.org/pvldb/vol19/p388-li.pdf),
[ZipNN](https://arxiv.org/html/2411.05239), [NeuZip](https://arxiv.org/html/2410.20650),
[Blosc bitshuffle](https://blosc.org/posts/new-bitshuffle-filter/).

## 10. Compute–layout coupling

**Primitive:** the byte layout the dequant kernel wants vs the layout the disk wants.

**Who else solves it:** cache-hierarchy-aware array formats. **Blosc2 NDim** uses two-level
partitioning (chunks → blocks) deliberately **mapped to L1/L2/L3**, and **Btune** auto-selects
codec/filter per array. Our own tension is the mirror image: `use_extra_bufts=false` (repack
off) is required to avoid OOM, but repack exists because `q4_K_8x8` is faster to compute. We
traded compute for residency. Sizing the slot pool to the **cache hierarchy** rather than to
RAM alone is the unexplored middle. Source:
[Blosc2 NDim](https://blosc.org/docs/Exploring-MilkyWay-SciPy2023-paper.pdf).

## 11. The one that compounds everything — amortization by verification

**Primitive:** paying the expert read once for *many* tokens.

**Who else solves it:** speculative decoding for MoE offload, which is a direct, published
answer to our measured 212.6 MB/token:

- **SpecMoE** — self-assisted speculative decoding; multiple tokens generated **per expert
  parameter retrieval**; on **SSD offloading ~2.25× throughput**; up to 4.30× overall;
  **76.73% reduction in CPU→GPU transfer**. Works on less-skewed models too.
- **MoE-SpeQ** — states the goal formally as an **"Amortization Roofline Model"**: maximise
  *accepted tokens per byte of synchronous I/O*.
- **SpecMoEOff** — 2.5× decode over MoE-Lightning. **DraftExpert** — reframes around
  **"accepted tokens per expert-set expansion"**.

Sources: [SpecMoE](https://www.arxiv.org/pdf/2604.10152), [MoE-SpeQ](https://arxiv.org/html/2511.14102v1),
[SpecMoEOff](https://arxiv.org/pdf/2508.21706), [DraftExpert](https://arxiv.org/html/2607.24434v1).

**Why this is the highest-leverage import in the survey:** our cost is *per token* expert bytes,
and our measured cold decode is **3.67 tok/s at 212.6 MB/token**. A verification pass fetches
the **union** of experts for k draft tokens — one read serving k accepted tokens. It attacks the
exact quantity we measured, it is published with SSD numbers, and **none of the surveyed
engines does it.**

## Honest counterweight — do not present without it

**[SSD Offloading for LLM MoE Weights Considered Harmful in Energy Efficiency](https://arxiv.org/html/2508.06978v1)**
measures SSD offload at up to **12× the per-token energy of HBM** and ~3× CPU DRAM, with SSD at
**73–80% of total energy**. Prefetching hides *latency*, not energy. Any claim about this
architecture should say "faster", never "more efficient".

---

## The compounded proposal

Three imports compound, and one is ours:

1. **io_uring + O_DIRECT at qd 32–128** (§3) — replaces thread oversubscription and goes past it.
2. **Metadata/payload stream separation with Huffman on the metadata only** (§9, QStore/ZipNN) —
   the one lossless lever our own dead-end note wrongly closed.
3. **Speculative verification** (§11) — amortizes the measured 212.6 MB/token across accepted tokens.
4. **W-TinyLFU admission** (§4) — cheap, unpatented, and the evidence says admission beats eviction.

Our own contribution stands: the thread-oversubscription measurement, and a **cold 8 GiB-capped
decode number (3.67 tok/s)** that no one else in this space has published — Edge0's figures are
on a 24 GB machine with the checkpoint effectively page-cached.
