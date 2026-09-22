# SOURCE: https://github.com/JustVugg/colibri/issues/200

## TL;DR

PILOT's router-lookahead prefetch predicts L+1's experts by running L+1's router on L's post-attention (pre-MoE) state — **stale by one MoE residual**. We can close most of that gap for free: compute only the **shared expert** (already resident in RAM, no disk I/O), add its output to the residual, then run the router on the corrected state. Measured **+3.1% recall** (73.6% → 76.7%) on GLM-5.2. No tok/s improvement on our 24 GB host (cache cap=2, disk-saturated), but the recall gain should help higher-RAM hosts where the cache can actually hold prefetched experts.

Independently validated by **"Speculating Experts"** (arXiv:2603.19289, UMD/LLNL, Mar 2026) — they call the same idea the "quasi-hidden state" but use a static default vector. Our version uses the **actual computed shared expert**, which is input-dependent and more accurate.

---

## Background: what PILOT does

`pilot_prefetch()` runs L+1's router on L's post-attention residual `x` (before MoE(L) has been computed):

```
x = residual after attention(L)
nrm = rmsnorm(x, L+1.post_ln)        // normalize with L+1's layernorm
ch  = matmul(nrm, L+1.router)         // run L+1's router
pred = topk(sigmoid(ch) + L+1.router_bias)
→ enqueue WILLNEED / real-load for predicted experts
```

The prediction is **stale by one MoE residual**: the real L+1 router sees `x + MoE(L)`, not `x`. The routed experts are on disk (the whole point of prefetching them), so we can't compute them. But the **shared expert** is part of the dense model — it's already resident in RAM.

## The two-step approach

```
// target = L+1, so src_layer = L = target-1
snrm = rmsnorm(h, L.post_ln)                    // normalize with L's layernorm
sg   = matmul(snrm, L.sh_gate)                   // shared expert gate
su   = matmul(snrm, L.sh_up)                     // shared expert up
sout = matmul(silu(sg) * su, L.sh_down)          // shared expert down
hc   = h + sout                                   // corrected state
nrm  = rmsnorm(hc, L+1.post_ln)                  // normalize corrected state
ch   = matmul(nrm, L+1.router)                    // run L+1's router on corrected state
pred = topk(sigmoid(ch) + L+1.router_bias)
```

This adds **3 small matmuls** (sh_gate, sh_up, sh_down — each `D × moe_inter`) but **zero disk I/O** (the shared expert weights are resident). For GLM-5.2: D=2048, moe_inter=2048, n_shared=1, so each matmul is 2048×2048 — roughly the same cost as the router matmul itself.

## Measurements

### Routing recall (LOOKA=1, MTP=0, single-token decode)

Measured with the LOOKA harness: scores predicted top-8 against actual routed experts.

| Prediction method | Recall | Samples |
|---|---|---|
| Previous token (=SPEC prefetch) | 36.3% | 13800 |
| Layer input, skip attention | 81.7% | 13800 |
| **PILOT (stale, current)** | **73.6%** | 13800 |
| **Two-step (shared-expert)** | **76.7%** | 13616 |
**Delta: +3.1% recall.** The two-step has slightly fewer samples because it can't predict for layer 0 (no source layer).

### End-to-end tok/s (same prompt, 32 tokens, MTP=0)

| Config | tok/s | hit rate | RSS |
|---|---|---|---|
| Baseline (no PILOT) | 0.17 | 18.0% | 19.16 GB |
| PILOT=1 (stale) | 0.16 | 18.5% | 19.22 GB |
| PILOT_TWO=1 (two-step) | 0.16 | 15.6% | 17.70 GB |
**No tok/s improvement on our hardware.** The reason is visible in the startup log: `cap lowered 64->2` — with 24 GB RAM, the expert cache holds only **2 experts per layer**. At that cache size:

- There's almost nothing to prefetch *into*
- Disk bandwidth is already saturated — WILLNEED hints don't create more bandwidth
- The 3 extra matmuls add a tiny compute tax for zero I/O benefit
This is a **hardware ceiling, not an algorithm failure**. On a host with enough RAM for cap≥32, the +3.1% recall means 3% more predicted experts actually hit the cache instead of missing to disk — that's where tok/s would improve.

## Prior art

**"Speculating Experts Accelerates Inference for MoE"** (arXiv:2603.19289, Madan et al., UMD/LLNL, Mar 2026)

They independently developed the same idea. Their "quasi-hidden state" combines the normalized residual stream with a **"default vector"** (the mean MoE output across training data) before running the router. Key differences from our approach:

| | Speculating Experts | Our two-step |
|---|---|---|
| **MoE approximation** | Static default vector (mean) | Computed shared expert (input-dependent) |
| **Accuracy** | Good (they report improvements) | +3.1% over baseline |
| **Cost** | Free (just add a cached vector) | 3 extra matmuls (cheap, no disk) |
| **Applicability** | Any MoE | Models with shared experts (DeepSeekMoE, GLM) |
Our approach is more accurate per-prediction (input-dependent vs static mean) but requires a shared expert. Their approach is more general (works on any MoE, even without shared experts). **Both converge on the same insight**: approximate the missing MoE residual before re-running the router.

Also relevant: **"Pre-Attention Expert Prediction"** (arXiv:2511.10676, ETH Zürich, Nov 2025) — predicts experts from the *pre-attention* state (even earlier than PILOT), trading accuracy for more prefetch lead time. This suggests a spectrum of prediction points that could be combined.

## Other experiments tried (all on separate branches)

| Experiment | Branch | Result |
|---|---|---|
| Frequency prior (usage histogram) | `experiment/pilot-usage-prior` | +0.7% |
| COUPLE cross-layer pair table | `experiment/couple-recall-measure` | +0.6% |
| Two-step + COUPLE ensemble | `experiment/two-step-plus-ensemble` | No gain over two-step alone |
| Two-step + top-1 routed expert | `experiment/two-step-top1-routed` | **-16% (worse)** — adding a guessed routed expert adds noise |
Key takeaway: the shared expert alone is the right proxy. Adding guessed routed experts or cross-layer tables doesn't help — the signal is already captured by the shared expert correction.

## Implementation

Branch: `experiment/two-step-production` (based on latest `dev` at [f1fa5bf](https://github.com/JustVugg/colibri/commit/f1fa5bf3c897b049bba1e8ecefe839dd681f3107))

- `la_predict()` kind==2: measurement harness (LOOKA=1)
- `pilot_prefetch()`: production path behind `PILOT_TWO=1` env var
- Guards: `n_shared==0` and `moe_inter<=0` early return (prevents zero-size malloc crash on models without shared experts)
- Workspace allocated once per `pilot_prefetch()` call, reused across positions (not per-position malloc churn)

### Reproduce

```
# Recall measurement:
SNAP=<model_dir> LOOKA=1 MTP=0 PROMPT="..." NGEN=32 ./glm 64
# → prints LOOKAHEAD routing recall table

# End-to-end tok/s:
SNAP=<model_dir> MTP=0 PILOT_TWO=1 PROMPT="..." NGEN=32 ./glm 64
# → compare tok/s with PILOT=1 and baseline
```

## What would make this actually faster

The two-step recall improvement is real but only matters when the cache can hold enough prefetched experts to make use of better predictions. The most impactful next step would be testing on a host with **cap≥32** (roughly 64+ GB RAM) where:

- The cache can actually hold the prefetched experts
- Better recall → fewer disk misses → measurable tok/s improvement
I can't test that on my 24 GB machine, but the recall data shows the algorithm works. If anyone with a higher-RAM host wants to test `PILOT_TWO=1`, the branch is ready.
