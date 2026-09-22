# Piece-wise architecture optimization — where the bytes actually are

**Why this file exists.** The project optimized one piece: the routed experts
(18.33 GB of file, 88% of storage). That is where the *disk* bytes are. It is **not**
where the *per-token* bytes are. This is the decomposition, priced in the unit the
physics charges — **bytes streamed per token** — with each piece's best-in-class
solution from the literature.

## 1. The correction

Measured from the GGUF tensor table (`results/gguf-byte-breakdown.json`):

| class | stored | **read per token** | share of per-token traffic | stored at |
|---|---|---|---|---|
| attention / ssm | 1.379 GB | **1.379 GB** | **53.3%** | **Q8_0** |
| expert gate/up/down | 18.327 GB | 0.573 GB (8/256 active) | 22.1% | Q4_K |
| output proj | 0.417 GB | **0.417 GB** | **16.1%** | **Q6_K** |
| dense ffn | 0.134 GB | 0.134 GB | 5.2% | Q8_0 |
| router | 0.084 GB | 0.084 GB | 3.3% | **F32** |
| token_embd | 0.540 GB | ~0 (one row lookup) | 0% | Q8_0 |
| **total** | **20.88 GB** | **2.587 GB / token** | | |

**The non-expert weights are 77.9% of per-token traffic, and they are stored at 8,
6 and 32 bits.** The single largest per-token term is not an expert at all — it is
the attention/SSM block at Q8_0, worth more traffic than all 256 experts of a layer
combined.

**Validation of the model.** 2.587 GB/token × the measured 10.97 tok/s = **28.4 GB/s
= 74% of DDR4-2400 dual-channel peak (38.4 GB/s)**. That is a realistic achieved
fraction for dense matvec over partially-random expert rows, and it lands
independently of the breakdown — the two measurements corroborate each other. The
earlier 1.989 GB/token figure was *active-params only* and understated the real cost
by 30%.

## 2. Every piece, and what optimizing it buys

| piece | change | GB/token saved | verdict |
|---|---|---|---|
| attention/ssm Q8_0 → Q4_K | 1.379 → 0.831 | **−0.548** | biggest single win in the whole model |
| output proj Q6_K → Q4_K | 0.417 → 0.335 | −0.082 | cheap, always active |
| output proj vocab-prune to 40% | 0.335 → 0.134 | −0.201 | needs a quality check on the decision task |
| cold experts 2-bit (tiered) | 0.573 → 0.267 | −0.306 | lossy-but-diluted; flash-moe precedent |
| dense ffn Q8_0 → Q4_K | 0.134 → 0.080 | −0.054 | small, always active |
| router F32 | **do not touch** | 0 | it *chooses* the experts — any change flips decisions |
| token_embd Q8_0 | **trap** | **0** | one row is read per token; it is a lookup, not a stream. Optimizing it helps RAM footprint, not tok/s |

### The ceiling, if you stacked them

| configuration | GB/token | ceiling @peak | ceiling @74% |
|---|---|---|---|
| baseline Q4_K_S as shipped | 2.587 | 14.8 | 10.9 ✓ *matches measured 10.97* |
| **P1**: attention + output + dense ffn → Q4_K (experts untouched, lossless-ish) | **1.957** | **19.6** | **14.5** |
| P2: P1 + tiered experts (hot 25% 4-bit / cold 2-bit) | 1.705 | 22.5 | 16.6 |
| P3: P2 + output-proj vocab prune to 40% | 1.504 | 25.5 | 18.8 |
| P5: P1 + prune 4 of 40 layers | 1.761 | 21.8 | 16.1 |

**P1 alone is a 1.32× bandwidth reduction → ~14.5 tok/s, from requantizing three
tensor classes the project has never touched, with the expert path untouched.**

## 3. Amortization — the only route past the single-stream wall

P1 changes the *constant*; it cannot break the wall. For that, more than one token
per weight pass:

| tokens accepted per weight pass | effective tok/s |
|---|---|
| 2 | 29.0 |
| 3 | 43.6 |
| 4 | 58.1 |

(both spec-decode and JEV's own multi-criteria batching)

That is lossless by construction: speculative verification is distribution-preserving
under exact rejection sampling, and `batch_logits_batched` is already verified
bit-identical.

## 4. Per-piece literature, with each source's own numbers

### On-disk format / expert packaging
- **flash-moe, tiered expert quantization** (`Alexintosh/flash-moe`, branch
  `feature/tiered-expert-quantization`, "Implemented and verified") — hot ~25% of
  experts by frequency (covering ~80% of activations) stay 4-bit, **cold experts
  requantized to 2-bit: 1,769,472 → 983,040 bytes per expert, −44.4%**. Per-expert
  kernel dispatch selects 4-bit or 2-bit dequant. Explicitly notes *"uniform 2-bit
  breaks JSON/tool calling"* — tiering is what preserves quality. Also reports the two
  negatives worth having: **LZ4 compression −13%** and **speculative prefetch −38%
  (cache pollution, bandwidth waste)**.
- **RotaryQuant** (arXiv:2608.08081) — a three-axis compression system whose
  *weight* axis is exactly the mixed-precision assignment we lack: **4-bit dense
  layers, 2-bit routed experts, 8-bit for the shared expert whose high activation
  kurtosis resists aggressive compression**. Reported **9–19 tok/s interactively with
  ΔPPL < 0.002** across three architecturally distinct models. This is the closest
  published analogue to the per-piece policy argued here, and it independently
  confirms the *asymmetry*: bit-width should follow architectural role, not be uniform.
- **OmniMoE** (arXiv:2602.05711) — **expert-centric scheduling** inverts
  token-centric execution to group physically nearby experts so expert blocks load
  once and are reused across stacked tokens; reported **10.9× (73 ms → 6.7 ms)**.

### Layer restructuring
- **Pipeline-Native Transformers / cflow** (arXiv:2608.23841) — the direct treatment
  of "restructure the layers". It notes CPU decode is bandwidth-bound (~1 TFLOP/s
  compute vs ~50 GB/s memory) and **restructures the model into vertical pipeline
  stages, cutting per-token weight bandwidth 9.00 → 4.50 MB/token (2.00×)** within
  0.24 perplexity of the best baseline; **5.94 tok/s vs llama.cpp's 4.75** on a
  32-vCPU Ice Lake box. It is architecture change plus a per-layer on-disk format
  (`.cflow`), not a runtime flag.
- **LayerSkip** (ACL 2024) — training recipe (layer dropout + early exit loss) making
  early exit viable: **1.34–2.16× speedups on summarization and 1.82–2.16× on
  coding/translation**. Early exit is the natural fit for a *classifier* like JEV,
  which may not need all 40 layers to choose between two options.
- **2D early-exit optimisation** (arXiv:2604.18592) — coordinates layer-wise and
  sentence-wise exiting **specifically for classification tasks**. Directly on-point
  for JEV's decision use case.
- **llama.cpp `--prune-layers`** — the tool already exists upstream for exactly this.

### Vocabulary / output
- **VocabTrim** (arXiv:2506.22694) — reconstruct the LM head over a limited token
  subset, **training-free**, for drafter speed.
- **Vocabulary trimming by language heuristics** (arXiv:2311.09709) — VT reduces
  small-model memory **by nearly 50%** with an upper bound of **25% generation-speed
  improvement**.
- llama.cpp already exposes `--token-embedding-type` and `--output-tensor-type`.

### CPU compute path
- **KTransformers** (SOSP'25, Tsinghua MADSys) — treats the CPU as first-class for
  MoE, with **AMX-optimized kernels**; runs 671B DeepSeek-V3 on one 24 GB GPU plus
  RAM. Its AMX backend offers **BF16 / AMXINT8** weight formats. Note: our i7-8700
  (Coffee Lake) has **no AMX** (Sapphire Rapids+), so this specific lever does not
  apply to this host — recorded so nobody re-tries it.
- **llama.cpp `--no-extra-bufts`** — the weight-repack switch, already a measured
  trap in this project (repack materialises experts into anonymous RAM and OOM-kills
  a >RAM checkpoint during load).
- **llama.cpp quantize `--tensor-type NAME=TYPE`, `--output-tensor-type`,
  `--token-embedding-type`, `--prune-layers`, `--exclude-weights`, `--imatrix`** —
  per-tensor control already upstream, which is what makes P1/P2 executable without
  writing a quantizer.

### Memory-bandwidth-bound CPU work generally
- **Asynchronous KV cache prefetching** (arXiv:2504.06319, AAAI) — L2-cache-oriented
  async prefetch for the memory-bound regime. Relevant because the KV stream, not
  the weights, is the other always-active reader at long context.
- The i7-8700-independent benchmark (saved in `sources/`) states the regime law:
  *"Single-user CPU generation is memory-bandwidth-bound: the speed ceiling is set by
  how many weight bytes you stream from RAM for each token"* and reports
  **10.6 tok/s on a 30B-A3B in an i7-8700 with no GPU** — matching this project's
  10.68 warm and this file's 10.97 measurement to within 4%.

## 5. Ranked by bytes-saved × feasibility, all in one CPU-only premise

| # | change | GB/tok | lossless? | executable today with |
|---|---|---|---|---|
| 1 | **attention+output+dense ffn → Q4_K** | −0.63 | ~yes (standard recipe) | `llama-quantize --tensor-type` |
| 2 | **k≥2 tokens per weight pass** | ÷k | yes | spec decode, or `batch_logits_batched` |
| 3 | cold experts → 2-bit, hot stay 4-bit | −0.31 | lossy, diluted | flash-moe's repack script (ported) |
| 4 | output-proj / vocab pruning | −0.20 | needs decision check | `--output-tensor-type` + rebuild |
| 5 | early exit for the classifier | −(layers skipped)/40 | needs decision check | `--prune-layers`; LayerSkip recipe |
| 6 | 2-bit expert plane format | −0.28 | lossy | port flash-moe / `.moet2pf` |
| 7 | io_uring + O_DIRECT for the disk tier | 0 (latency only) | yes | ctypes io_uring; Lidenburg precedent |
| 8 | pass-pinned slot pool, W-TinyLFU admission | 0 (cap relief) | yes | engine patch |

## 6. What this reorders

The project's phases 1–3 target SSD read volume. On this host the measured binding
constraint is **per-token RAM bandwidth**, so the highest-leverage changes are the
ones that reduce *bytes per token* — and the top one is a quantization recipe for
tensor classes nobody has looked at, executable with the upstream tool already on
disk.

**Two beliefs this pass overturned, both of them mine:**
1. "Non-expert weights are ~10% of traffic" → they are **77.9%**.
2. "The experts are the problem" → per token they are the **smallest term** outside
   the router, because only 8 of 256 fire.

## Sources

`docs/research/sources/` — flash-moe tiered quantization, RotaryQuant (2608.08081),
Pipeline-Native (2608.23841), LayerSkip (ACL 2024), 2D early exit (2604.18592),
VocabTrim (2506.22694), vocab trimming (2311.09709), OmniMoE (2602.05711),
KV async prefetch (2504.06319), llama.cpp quantize README, i7-8700 MoE-on-CPU
benchmark, thecodacus MoE expert cache fork, Lidenburg three-tier cache.

**Not claimed:** no published end-to-end number exists for a 35B-A3B-class MoE at
20+ tok/s CPU-only with no second tier. The per-piece savings above are arithmetic on
measured file bytes and measured bandwidth; the *quality* deltas of the lossy rows
are the papers' own numbers, not ours — each needs its own decision-flip measurement
on the four fixtures before any speed claim is made.
