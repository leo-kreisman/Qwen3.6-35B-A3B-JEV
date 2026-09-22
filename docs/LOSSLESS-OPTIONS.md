# Lossless levers — nothing here changes a decision

Tier definition used throughout: **lossless = the same bytes reach the dequant kernel, so
logits and decisions are unchanged by construction** (bit-identical, or trivially
decision-stable because only the *layout* or the *schedule* moves). Anything that changes the
numerics is marked, and the check it needs is named.

Baseline being moved (all `[verified-here]`, `docs/STATUS-2026-09-22.md`): cold 8 GiB cap
verified binding, page cache evicted, **3.67 tok/s at 212.6 MB read per generated token**;
warm/no cap 10.68 tok/s; routed expert set 18.33 GB; raw device ~2.4 GB/s; predicted
decode ceiling 2.4 GB/s ÷ 566 MB/token ≈ **4.2 tok/s**.

New evidence collected for this tier (this session, `[verified-here]`):

| probe | result | file |
|---|---|---|
| per-stream compressibility, byte-weighted over **all 120** expert tensors | payload 96.3% of raw (1.04×), packed6 scales/mins 89.1% (1.12×), **fp16 d/dmin 78.6% (1.27×)**; all streams together **5.11%** of the expert set | `results/q4ks-weighted.json` |
| metadata-only split | metadata is **11.1%** of the file; compressed to 73.5% ⇒ **2.95%** file reduction if the payload stays raw | `results/q4ks-decomp-speed.json` |
| decompression speed | zstd-3 whole 846 MB/s 1-thread, **5.28 GB/s at 8-way over 4 MiB frames**; LZ4 4 MiB frames **7.0 GB/s** at 8-way and 92.6% of raw | `results/q4ks-decomp-speed.json` |
| O_DIRECT random reads on **this GGUF, this NVMe** (unprivileged C harness, 512 MiB budget/run) | 4 KiB qd=128 registered: 1,271 MiB/s vs sync pread 53 MiB/s (**24×**); **589,824 B (=one expert) qd=8 uring 3,384 vs pread 1,658 MiB/s (2.0×)**; 4/7 MiB ~3.3–3.4 GB/s both ways | `results/nvme-io-20260922/io_results.csv` |
| per-layer spread | payload zstd ratio is **layer-dependent** (1.113 at blk.0 vs 1.030 at blk.18/27/36) — a per-tensor figure, not a file constant | `results/q4ks-layer-spread.log` |
| `-ncmoe` sweep under the 8 GiB cap | decode **ncmoe=99 → 8.2 tok/s vs ncmoe=32 → 20.4 tok/s (2.5×)**; no cap, ncmoe=99 37.4 / 40 36.3 / 32 40.3 / 24 50.7 tok/s | `results/bench-8g-20260922-035837/` |

---

## A. Bounded tier — same bytes, only the layout or the schedule changes

1. **Read the expert gather with io_uring + `O_DIRECT`, registered buffers, qd 32–128.**
   Same file, same offsets, same bytes. Measured on this device at the exact extent size a
   routed expert has (589,824 B): **3,384 vs 1,658 MiB/s, 2.0×**; at 4 KiB it is 1,271 vs 53
   MiB/s. Kills the need for the 4/3×nproc crutch (that result is the *substitute* for missing
   queue depth, not a law). **Ceiling: 4.2 tok/s — depth buys latency, not bytes**, so this is
   worth up to ~1.15× on decode but much more on prefill, which is extent-limited.
2. **Repack: 16 KiB output-blocked slabs, one expert = one extent, offset table folded into
   the slab.** Pure byte shuffling; the quant bytes are unchanged. Today a 590 KB–7 MB expert
   is 144–1700 separate 4 KiB extents. Directly attacks the ~2.2× *gather overfetch*, the
   factor that is currently inferred from the cap table and needs per-layer read tracing to
   confirm. Projection: amplification 4.41× → ≤1.2×.
3. **Pass-pinned working set (`lru_gen_min_ttl` semantics) — do not evict a page the current
   pass will re-touch.** This is the measured 2.12× *cap-driven* re-read (41.0 GB read for a
   9.3 GB routed set). No numerics involved. Projection: 10.25 GB → 4.83 GB per decision.
   Must stay paired with (7), because the re-read is also what makes the prefetch work.
4. **Stage into engine-owned buffers, never the page cache.** Measured: a `preadv` prewarmer
   made reads *worse* (42.0 → 56.1 GB) because clean unreferenced pages are reclaim victim #1.
   Lossless by construction — it changes where bytes land, not which bytes.
5. **`n_batch = n_ubatch = sequences × 512`.** Measured 67.0 → 42.0 GB, logits bit-identical.
6. **Thread oversubscription at 4/3 × hardware threads.** Measured 54.6 → 38.4 s, −29.6%,
   reads unchanged, **identical probabilities**.
7. **Do not "fix" amplification by suppressing read-ahead.** `MADV_RANDOM` cut reads
   42.0 → 26.0 GB but cost **56.0 → 368.0 s**. The amplification on a >RAM MoE is load-bearing
   prefetch. Listed as a lever *against* the tempting lossless-looking one.
8. **More RAM — the cheapest lossless speedup in the project.** 3.67 tok/s at a verified-binding
   8 GiB vs **10.68 tok/s** warm with no cap. Raising the cap is a **2.9×** lever and changes
   nothing numerically; it belongs in the same table as the imports.

## B. Lossless-by-data — fewer bytes on the wire, decoder must stay out of the hot path

9. **Metadata/payload stream separation: byte-plane transpose, compress the fp16 metadata
   only, keep the 4-bit payload raw.** The self-correction stands: the payload is incompressible
   (1.04× over all 120 tensors) but **fp16 `d`/`dmin` compresses to 78.6% (1.27×)** and the
   packed-6 scales/mins to 89.1%. File-level saving is **2.95%** (212.6 → 206.3 MB/token).
   Cheap to decode: only 11.1% of the bytes are touched. Huffman rather than zstd is the
   published form (ZipNN/QStore/Huff-LLM target exactly this skewed-exponent stream).
10. **Whole-file lossless recompression is a smaller lever than it looks, and it is now
    priced.** Compressing *everything* (payload included) saves **5.11%** of the expert set
    ⇒ 212.6 → 201.7 MB/token, 3.67 → ~3.87 tok/s (**+5.4%**), at a cost of **0.251 s of
    single-threaded zstd per token against a 0.272 s/token budget** — i.e. it only pays if the
    decode is multi-threaded (5.28 GB/s at 8-way over 4 MiB frames → 0.040 s) or LZ4 is used
    (7.0 GB/s at 8-way, but only 7.4% of raw saved). **This is single-digit percent for a
    compressor in the hot path; rank it below every structural lever in section A.**
11. **`-ncmoe N` — put part of the expert set in the two idle RTX 5060 Ti and stop reading it
    from NVMe.** Measured under the same binding 8 GiB cap: decode **8.2 → 20.4 tok/s at
    ncmoe=32** (2.5×), and the disk-resident working set shrinks by (40 − N)/40 (N=32 removes
    ~3.7 GB of expert bytes from the stream). **Flagged, not filed as bit-identical:** the
    weights are the same but the compute path becomes CUDA kernels, so this needs the same
    decision-equivalence check the batching patch got (`results/decisions.batched.jsonl`).
    If a near-tie fixture ever exists, re-verify against it — all four current fixtures sit at
    p ≈ 0.99998 for the winner, which is a weak test.
12. **Speculative verification (SpecMoE / MoE-SpeQ) with a resident drafter.** The published
    mechanism is *distribution-preserving under exact rejection sampling*, so it can belong in
    this tier — but only in that form and only with a drafter that does **not** read NVMe, since
    the union of experts for k drafts must cost less than k sequential reads. Measured gate:
    acceptance ≥ ~1.4 at k=4, else the union costs more than it saves. **Unverified on this
    checkpoint; listed as a conditional member of the lossless tier, not an accepted one.**

---

## What is *not* in this tier (kept separate on purpose)

Lossy or numerics-changing: smaller quant / bpw reduction, REAP pruning, expert merging,
batch-aware routing overrides, any router approximation or lookahead that can change a
top-8 selection. Each must be presented with its own decision-flip rate on the four fixtures
(`examples/`), measured the way the batching patch was — a lossy lever may never be reported
with a lossless one.

## Kill gates

- (1) warm-cache regression: io_uring is slower on a warm cache ⇒ keep a buffered fast path.
- (2)/(3) if bytes read per decision does not fall, the amplification is not layout — then (3)
  carries everything and (2) is only a latency win.
- (9)/(10) if decompression CPU cost exceeds the 3–5% byte saving on the same cores that do the
  MoE math, drop it. The 0.251 s vs 0.272 s numbers above are the reason this gate is explicit.
- (11) decision-equivalence on all four fixtures, and on a near-tie fixture if one is ever added.
- (12) acceptance < ~1.4 at k=4, or a drafter that touches NVMe.
