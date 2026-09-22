# The best approach — 20+ tok/s equivalent, CPU-only, no GPU, no training

**Status: this is the answer.** Two independent sources confirm the target is real on
this class of hardware, and the mechanism we were missing is now identified. Three of
the four levers are training-free; the fourth is optional.

---

## 0. The target is real — two independent confirmations

| source | hardware | result | memory |
|---|---|---|---|
| **Edge0 paper** (arXiv:2609.18063, Sept 2026, AutoArk) | Mac mini M4 Pro, 24 GB | **20.4 tok/s** at K=4; **14.9–17.7 tok/s** at K=8 | **2.9 GiB** peak active |
| **i7-8700 CPU-only benchmark** (saved in `sources/`) | i7-8700, **no GPU** | **10.6 tok/s** on a 30B-A3B | — |

The Edge0 abstract states it plainly: *"On a single 24 GB machine, Edge0 serves a 35B
MoE at ≈20 tok/s inside 3 GiB of peak active memory."* That is our model class, our
target, and 3 GiB is *less than our 8 GB cap*.

**The most important comparison in the paper:** on the same machine, a vanilla
mlx-lm server with all 19.5 GB of 4-bit weights **resident** decodes at **3.9 tok/s**
occupying 18.2 GiB, while Edge0's streaming K=4 profile decodes at **20.4 tok/s**
occupying **2.9 GiB**. *5.2× faster than the fully-resident baseline, using 6× less
memory.* Streaming from SSD is not a tax to be minimised — **it is the winning
configuration outright**, because the resident path's memory is unreclaimable and
crowds out KV and the OS.

This is why our 3.67 tok/s cold number is not a ceiling: it is the resident-ish regime
losing. The offload regime, done properly, is 5× faster.

---

## 1. What we were missing: prediction **as** routing

Edge0's core insight is not prefetching. It is this: under SSD streaming, layer N+1's
experts must be chosen before layer N's output exists, so naive streaming **serialises
disk latency into every layer of every step**. The fix is a **prerouter**: a small
per-layer head (fc1 → erf-gelu → fc2, fp16, hidden 512) that predicts layer N+1's
routing from layer N's hidden state **one token ahead**, and — the twist — the
prediction is *consumed as the routing itself*:

> "The prediction is consumed as the routing itself: the staged expert set and the
> routed set are identical by construction, and the approximation that other
> pre-gating schemes absorb at inference time through fallback loads and dropped
> tokens is instead paid once, in training, and recovered there."

Cost of the approximation is paid once, in training — so **zero drops at inference**.
Measured advantage on a machine where the weights do not fit (16 GB MacBook M2,
18.4 GiB checkpoint): **+80% / +82% / +84%** decode at K=2 / K=4 / K=8.

**This is our regime exactly.** JEV runs cold under an 8 GB cap with a 20 GB
checkpoint — "a machine on which the weights do not fit". Edge0's +80–84% applies to
our 3.67 tok/s cold path, not to the 10.68 warm path.

**A zero-training version exists.** The paper's head description: *"fc1 → erf-gelu →
fc2, plus a linear residual path ℓ that the training script **warm-starts from the next
layer's router weight** (its default is zero), so training begins from 'apply the next
router directly to this hidden state'."* That warm-start point is itself a usable
predictor: **feed layer N's hidden state through layer N+1's own router weights** to
predict N+1's routing, one token ahead. No training, no parameters, no new checkpoint.
Worth measuring first — it is the default initialisation the trained MLP only corrects.

---

## 2. The single biggest lever: training-free K-halving on *our exact model*

**arXiv:2609.04575, "Training-Free Halving of Activated Experts in Fine-Grained
Mixture-of-Experts Models"** (Xing Chen, Hengshuai Yao). Its primary object of study is
**Qwen3.6-35B-A3B — our model**, quoted numbers and all:

> "In Qwen3.6-35B-A3B, our primary object of study, all 40 layers are MoE layers with
> 256 experts each and k=8 selected per token; the routed experts hold 32.2B of the
> model's 35.9B parameters (89.6%), while each token touches only 3.1% of them."

**The finding:** reducing k changes *two* things — which experts fire, **and the
strength of the expert branch**, because renormalising over the surviving k implicitly
recalibrates the output gain. Separate them by **activating the top k₁ experts while
normalising by the probability mass of the top k₂ experts** (k₂ > k₁):

> "On Qwen3.6-35B-A3B, reducing from 8 to 4 experts causes a **4.65-point MMLU drop
> under standard renormalization but only 0.35 points with k₂=16**, while **halving
> routed-expert compute**." (one integer; no parameters, no training, no measurable
> compute overhead)

Replicated on the 11× larger Qwen3.5-397B-A17B (10→5 experts: 0.55 points). And a
warning worth keeping: *"perplexity and downstream accuracy favor different k₂,
cautioning against selecting MoE compression settings using unlabeled text alone"* —
pick k₂ on the labelled decision task, not on ppl.

**Why this is decisive for us.** Edge0's *first* speed lever was narrowing K=8→K=4,
and their own table prices it: **3.3 → 6.4 tok/s, "nearly doubles decode"**, with lower
peak memory. But Edge0 could only do it by **training** — a three-phase distillation
campaign (~2M SFT rows) producing a recovery LoRA, and they state the reason:
*"below 4 bits, error grows until the model is unusable, and no post-hoc technique
recovers it."* They paid for K=4 with a training run.

2609.04575 shows that for **routing width specifically**, the training-free
renormalisation fix recovers almost all of it: **−0.35 MMLU instead of −4.65**, on our
exact checkpoint. So the one lever Edge0 needed a training campaign for, we get for
**one integer**.

Effect on per-token bytes: the expert term **0.573 → 0.287 GB/token** (8/256 →
4/256 of 18.327 GB), measured from our own tensor table. That is a **pure
bandwidth reduction on the binding constraint**, and it also halves expert *compute*.

---

## 3. Our own finding, still valid: the non-expert weights are 77.9% of traffic

From the GGUF tensor table (`results/gguf-byte-breakdown.json`), per decode token:

| class | read per token | share | stored at |
|---|---|---|---|
| attention / ssm | **1.379 GB** | **53.3%** | **Q8_0** |
| experts (8 of 256 fire) | 0.573 GB | 22.1% | Q4_K |
| output proj | 0.417 GB | 16.1% | **Q6_K** |
| dense ffn | 0.134 GB | 5.2% | Q8_0 |
| router | 0.084 GB | 3.3% | F32 |
| token_embd | ~0 | 0% | Q8_0 (lookup, not a stream) |
| **total** | **2.587 GB** | | |

Cross-check: 2.587 GB/token × the measured 10.97 tok/s = **28.4 GB/s = 74% of
DDR4-2400 dual-channel peak**. The two measurements corroborate each other
independently, so the decomposition is sound.

Requantising attention + output + dense ffn to Q4_K (experts untouched) removes
**0.63 GB/token** — attention alone saves 0.548, the largest single non-expert win. The
router must stay exact: it *chooses* the experts, so a precision change there flips
decisions. token_embd is a trap — one row is read per token, so optimising it helps RAM
footprint, not bandwidth.

---

## 4. The projection on our measured baseline

Baseline: **10.97 tok/s**, CPU-only, `-ngl 0`, 6 threads, page-cached, measured twice
(10.94 / 11.03 at 6/8 threads; 1.66–1.71× faster than the current default of 16).

| step | change | GB/token | lossless? | projected tok/s |
|---|---|---|---|---|
| — | baseline Q4_K_S | 2.587 | — | **10.97 (measured)** |
| 1 | **K=8→4 with k₁/k₂ renormalisation** (2609.04575) | 2.301 | −0.35 MMLU | 12.3 |
| 2 | **+ attention/output/ffn → Q4_K** | **1.617** | Q4_K standard | **17.6** |
| 3 | + tiered cold experts ⇒ 2-bit (flash-moe: −44.4%/expert) | 1.47 | lossy, diluted | 19.3 |
| 4 | + expert-reuse-aware spec decode (EcoSpec, ≤1.62×) | — | lossless | **≥20 ✓** |

Steps 1–2 are **1.60× on bytes alone, training-free, with the expert path and the
router untouched**. Steps 3–4 clear 20 with margin. Step 4 is EcoSpec
(arXiv:2607.12696), whose finding is exactly the objection we raised earlier against
speculative decoding in MoE: confidence-driven draft selection causes **expert
scattering** — draft tokens route to disjoint experts, so verification activates the
*union* and *increases* weight traffic. EcoSpec folds predicted **marginal expert
activation cost** into draft selection, favouring draft paths that reuse experts already
covered by the verification set — *without changing the target verification rule* —
for up to **1.62×** on DeepSeek-V3.1/Qwen3-235B/GPT-OSS-120B. That is what makes spec
decode a *bytes* win in MoE instead of a bytes loss.

**Honest note on the prerouter's value here:** its +80–84% removes *stall* time, not
bytes. On the warm path we are bandwidth-bound at 74% of peak, so there is less stall
to hide and the gain will be smaller than Edge0's. Its real target is the **cold,
8 GB-capped** regime, where every token faults experts in from SSDs — that is exactly
the "weights do not fit" case Edge0 measured, and it is where our 3.67 tok/s lives.

---

## 5. The remaining levers, priced

| # | lever | source | gain | cost |
|---|---|---|---|---|
| 1 | **Persistent sticky-slot stack** (`incr_stack`) instead of per-step stack rebuild | Edge0 §3.1, App. B | **+34% decode** | engineering only |
| 2 | **Cross-token prerouter, prediction-as-routing** | Edge0 §3.2 | **+80–84%** (weights-don't-fit regime) | needs a small head; zero-training warm-start variant available |
| 3 | **K=8→4 with k₁/k₂ renorm** | 2609.04575 | **halves expert bytes + compute** | −0.35 MMLU, one integer, no training |
| 4 | **EcoSpec cost-aware verification** | 2607.12696 | **≤1.62×** | draft model |
| 5 | **Non-expert → Q4_K** | our measurement | **−0.63 GB/token** | standard quantize |
| 6 | **Bigger contiguous reads** (O_DIRECT + io_uring, 576 KiB extents) | our `io_results.csv` | 1364–1696 → **3211–3383 MiB/s** (2.0–2.4×) | ctypes io_uring |
| 7 | **Page-cache economics** — read in fewer, larger loads | Edge0 §5.3 | per-load cost = **1.17 ms + 1.33 ms × cold**; 0.75–4.4 ms | follows from 6 |
| 8 | **Tiered experts** (hot 25% 4-bit / cold 2-bit) | flash-moe | **−44.4% per cold expert** | lossy; uniform 2-bit breaks tool calling |
| 9 | **Expert-centric scheduling** | OmniMoE (2602.05711) | **10.9×** | batching regime |
| 10 | **SpecPrefetch** router-preserving adapters | 2607.24787 | up to **+20%** | adapter training |

### Things that looked good and are not
- **LZ4 compression of experts**: −13% (flash-moe, measured).
- **Speculative prefetch of experts**: **−38%** — cache pollution plus wasted
  bandwidth (flash-moe, measured). Our own earlier negative result agrees.
- **Uniform 2-bit experts**: **breaks JSON/tool calling** (`\name\` instead of `"name"`).
- **Same-token pre-gating**: "every same-token variant we measured fell below a plain
  LRU baseline" (Edge0) — head evaluation drains the pipeline and the next layer's
  attention is not computed yet. One token of lead is the minimum that works.
- **AMX kernels (KTransformers)**: our i7-8700 Coffee Lake has no AMX — Sapphire
  Rapids and later only. Do not re-try.
- **Whole-layer prefill**: fastest only when the checkpoint is page-cache-resident
  (0.24 s vs 0.37 s warm) and **slowest when it is not** (5.3 s / 4.06 GiB cold vs
  0.6–1.1 s / 0.4–0.8 GiB on-demand). Cold is our regime ⇒ on-demand.
- **Thread count**: 16 is 1.66–1.71× *slower* than the 6–8 plateau in the
  bandwidth-bound regime. The 4/3-of-cores knee only pays when threads block on page
  faults.

---

## 6. Where the bytes go, and the order to attack them

Bytes per token are the binding constraint. Ranked by bytes saved per unit of work:

1. **attention/ssm Q8_0 → Q4_K** — 0.548 GB/token, no training, upstream quantize.
2. **K=8→4 + k₁/k₂ renormalisation** — 0.286 GB/token, one integer, −0.35 MMLU.
3. **output proj Q6_K → Q4_K** — 0.082 GB/token.
4. **dense ffn Q8_0 → Q4_K** — 0.054 GB/token.
5. **tiered 2-bit cold experts** — 0.145 GB/token, lossy, diluted.
6. **output-proj / vocab pruning** — 0.201 GB/token, needs a decision-flip check.
7. **early exit for the classifier** — 2604.18592 / LayerSkip; JEV's scorer may not need
   all 40 layers to choose between options.

Then, and only then, the stall levers (prerouter, sticky slots, io_uring) — which do
not reduce bytes but stop waiting on them, and matter most in the cold 8 GB regime.

---

## 7. What is missing and what it would take

- **Edge0's training code is not released.** The repo ships `prerouter/heads.py`,
  `stager.py`, `install.py`, `state.py` (inference + install only) and the trained
  adapters as `.safetensors`. `scripts/` holds only `convert_adapters_legacy.py`,
  `upload_hf.py`, `strip_vision_weights.py`. So the training recipe is described in
  §3.2/§4 of the paper but **not in the repo** — the zero-training warm-start variant
  (§1) is therefore not a shortcut but the only route that avoids reproducing their
  training campaign.
- **The published checkpoints are MLX int4 with pre-trained adapters** — not loadable
  in llama.cpp on CPU. The *mechanisms* port; the weights do not.
- **K=4 needs the k₁/k₂ trick**, not plain truncation: plain 8→4 costs 4.65 MMLU
  points on our model, which would destroy the decision accuracy JEV exists to serve.
- **Spec decode needs EcoSpec-style costing**, or it loses: in MoE the union of
  experts across a draft tree raises bytes/token.
- **Every lossy item above needs its own decision-flip measurement** on the JEV
  fixtures before any of it is claimed as a speed win. The papers' quality deltas are
  their numbers on their tasks; ours must be measured on ours.

**Not claimed:** no published end-to-end number exists for *this* exact configuration
(CPU-only x86, no GPU, 8 GB cap, Q4_K_S llama.cpp, 20+ tok/s, single stream). The
projection in §4 is arithmetic on measured bytes and measured bandwidth, with the
sources' own deltas applied one at a time; the compound effect is not measured.

---

## 8. Does Memory Caching (arXiv:2602.24281) help? — verdict: not the tok/s, but yes the idea

Fetched everything: the arXiv full text (via `pdftotext`, the HTML was truncated), the
HuggingFace paper page, the OpenReview record (**ICLR 2026 submission #24014**, Oct
2025), the community breakdown, and the one implementation that exists.

**What MC is.** Segment the sequence, cache the **end-of-segment hidden-state
checkpoints** of a recurrent model, and at each token retrieve from all cached states
plus the current online state: `y_t = Agg({M_L^(1)…M_L^(s-1)}; M_t^(s); q_t)`.
Four variants: Residual Memory, Gated Residual Memory (context-dependent gates
`γ_t^(i) = ⟨u_t, MeanPooling(S^(i))⟩`), Memory Soup, and **Sparse Selective Caching
(SSC)** — "a Mixture-of-Experts style router… measures the contextual similarity of
each token to its past segments and chooses a subset". Complexity O(N·L), between the
O(L) of RNNs and the O(L²) of Transformers.

**Why it cannot lift our decode tok/s — four hard reasons:**

1. **It is a training-time architecture change.** Every result is on 760M / 1.3B models
   trained from scratch on 30B / 100B FineWeb tokens (SWLA, DLA, Titans, all retrained).
   It is not a checkpoint transformation and cannot be applied to our Q4_K_S file.
2. **It *adds* cost at inference.** "the retrieval process requires forward pass over
   all cached memory and so needs O(N) operations per token." The authors' own framing:
   MC variants "provide a middle ground between Transformers and RNNs" — i.e. they give
   up part of the RNN's efficiency to buy recall. Its efficiency figure is *training
   throughput vs Transformers* at long context, not decoding speed.
3. **It does not reduce the binding constraint.** Our decode is weight-streaming-bound
   at 2.587 GB/token. MC touches KV/state traffic, which at agent-scale context is the
   smaller term. It cannot move 10.97 toward 20.
4. **It doesn't port onto Qwen3.6 any more than onto any other trained model.** The
   paper's own §4.1 shows the hybrid equivalence holds only "for an oversimplified
   version"; the proof of concept is retrained models. Our 30 GDN layers were trained
   with a fixed-size recurrence and cannot be re-cast without training.

Confirmed on our own file: **no MTP tensors** — 733 tensors, `blk.0`–`blk.39`, none
matching `mtp`/`nextn`/`eh_proj`. So the cheap-draft route needs the MTP variant of the
checkpoint, not this one. (Also confirmed: **30 GDN layers + 10 full-attention at
interval 4** — `attn_v` present only at blk 3,7,11,…,39.)

**But three things from this branch do help, and two are measured:**

**(a) Application-layer MC — the biggest measured lever for the OpenCode use case.**
`growing-memory` (sypherin) implements MC's SSC variant *over the prompt*: chunk
history, compress each chunk to a checkpoint, retrieve the relevant few. Published
numbers: **95% accuracy at 1,027 tokens vs 90.3% at ~15,000 for full context = 14.6×
fewer tokens and higher accuracy than stuffing**; a second benchmark **100% at 384
tokens (7× fewer)**. The repo's own note: *"The paper proves the saving in theory but —
as the ICLR reviewer flagged — never measured real tokens/latency. This repo does."*

Why this is the one that matters: a coding agent **resends a large prompt every turn**,
and under SSD offload **prefill is the expensive phase**. Cutting request tokens 14.6×
is a direct cut in per-turn prefill traffic — and it *raises* the decision quality the
JEV scorer is graded on. This is the agent-layer half of the problem that no amount of
expert-prefetch engineering addresses.

**(b) The sibling family is built for our exact architecture** (the real find):

| paper | what | number |
|---|---|---|
| **DASC** (2608.30386) | Decay-aware state compression for **Gated DeltaNet** + Kimi Delta Attention — derives retention horizons from the weights, packs long-horizon state units into a ragged checkpoint layout | **KDA state compressed 2.63×**; at fixed state-checkpoint budget **TTFT −42.6%, input throughput +68.4%** |
| **MARCH** (2608.12435) | Content-routed **state anchors** over Gated DeltaNet — attends over cached state anchors with content-conditioned keys; "consistently outperforms multiple linear attention variants" on LongBench / in-context retrieval | architecture-level (retrained) |
| **DART** (2608.02032) | Decodes token-conditioned keys *and* values from Mamba-2 chunk states and does attention over them | **75% savings at chunk 256 / state 128** — our `state_size` is exactly **128** |

DASC is the directly relevant one: the *problem* it solves is that "a recurrent state is
overwritten as tokens arrive and does not preserve earlier prefix boundaries. Prefix
caching therefore materializes full state checkpoints at regular token intervals" — and
**prefix reuse across turns is precisely what a coding agent needs.** Under our 8 GB
cap, state checkpoints compete with the expert cache; DASC's decay-aware packing is how
you get both.

**(c) MC's *selector* is the same shape as our cache-admission question.** SSC =
"router measures contextual similarity… chooses a subset" — the retrieval policy over a
set of cached states. Our slot pool needs exactly a relevance-gated admission policy
(pass-pinned + similarity), and DASC/MARCH supply the GDN-specific scoring.

**Net:** MC does not make our decode faster and cannot be bolted on. Its *concept* pays
in three places that do matter: the agent-side 14.6× prompt-token cut (measured), the
GDN-specific state/prefix-reuse path (DASC: −42.6% TTFT), and the admission-policy
shape.

---

## 9. The metric was wrong — for a coding agent it is not single-stream decode tok/s

Reframing for the OpenCode use case changes which levers matter. From
`slipstream-schero94`'s measured coding-agent run:

| measured, coding-agent workload | number |
|---|---|
| **~30k-token agent prompt prefill** | **75 → 208 tok/s (2.7×)** with `ubatch 2048` + parallel I/O threads |
| decode vs cache size | 2 GiB → ~5.5 tok/s @ **21% hit**; 10 GiB → ~13 @ **78%**; 14 GiB → ~19 @ **86%** |
| storage tier (PGRN) | USB SSD 0.72 → internal NVMe 1.95 (**2.7×**) → +DFlash spec draft 2.36 (**3.3×**) → +larger cache 2.83 (**3.9×**) |

**Cache hit rate, not raw decode, is the control variable** — and it is the one we are
starving under an 8 GB cap. Note also that `ubatch 2048` + parallel I/O threads moved
prefill 2.7× — a config change, not a new algorithm.

**The on-disk layout law (mbolt), which is quantified and self-limiting:**

> `speedup = 1 / ((1 − s) + s/g)` — `g` = per-read coalescing gain, `s` = cold-read share.
> "Coalescing pays **iff** misses are latency-bound: fine-grained MoEs and low-bit quants
> (~≤ 1 MB slices), not big-expert models."

Measured points: one expert = **9 sub-slices** (gate/up/down × weight/scales/biases),
3.19 MiB, **scattered across a 2.5 GB in-file span**; coalescing to one contiguous block
cuts per-cold-expert I/O **3.0×** (1396 → 463 µs) → **~20% decode, ~10% prefill**, 9×
fewer decode read-ops (486 → 54/token). On Qwen3-235B at 0.2 resident: **+32.3% decode,
−26.3% TTFT**. It weakens with slice size (gpt-oss-120b, 4 MB slices: only +9.7%).

**Our expert slice is 576 KiB (589,824 B, 16 KiB-aligned) — squarely in the regime where
this pays.** And the 6 tiny 32 KiB scales/biases reads per expert at 96 µs each are pure
IOPS waste, which our `pread`/`io_uring` measurements already showed is where the
latency lives (4 KiB reads: 53–55 MiB/s; 576 KiB: 1364–1696 MiB/s pread, 3211–3383
io_uring).

**The predictor, zero-training (PILOT family).** Router-lookahead: run layer L+1's
router on layer L's pre-MoE state, normalised with L+1's layernorm. Recall measured at
**73.6%**; a two-step refinement (compute the **shared expert** first — already
resident, zero disk I/O — add it to the residual, then route) gives **76.7%, +3.1
points**. Independent confirmation: "Speculating Experts" (arXiv:2603.19289, UMD/LLNL)
calls it the "quasi-hidden state". Reference points: predicting from the previous token
= **36.3%**; from the layer input skipping attention = **81.7%**.

Critical honest caveat from the implementer, which matches our situation: *"No tok/s
improvement on our hardware… `cap lowered 64->2` — with 24 GB RAM, the expert cache
holds only 2 experts per layer. There's almost nothing to prefetch into; disk bandwidth
is already saturated — WILLNEED hints don't create more bandwidth."* **This is a
hardware ceiling, not an algorithm failure** — and it is exactly why cache *size* and
coalescing come before prediction in our order. Prediction only converts to tok/s once
there is a cache to prefetch into and the reads are latency-bound.

## Sources

`docs/research/sources/` — Edge0 paper (arXiv:2609.18063) + `docs/prerouter.md` +
`docs/streaming.md`; K-halving (arXiv:2609.04575, fulltext); EcoSpec (arXiv:2607.12696);
SpecPrefetch (arXiv:2607.24787); S2-MoE (arXiv:2608.15018); Memory Caching
(arXiv:2602.24281); flash-moe tiered quantization; Pipeline-Native (2608.23841);
OmniMoE (2602.05711); RotaryQuant (2608.08081); LayerSkip (ACL 2024); 2D early exit
(2604.18592); VocabTrim (2506.22694); i7-8700 CPU-only MoE benchmark; Kimi K3 8 GB C99
engine; Colibri.
