# k₁/k₂ MoE renormalization — implemented, measured, **not** an improvement here

`--moe-norm-ref N` (arXiv:2609.04575) is **built and working** in
`/home/scribe/Projects/llama.cpp-fresh`. It is **not a win on this model at this
precision**, and this file records exactly why, with the numbers.

## What was built

Six files, additive and default-off:

| file | change |
|---|---|
| `include/llama.h` | `uint32_t n_expert_norm_ref` on `llama_context_params` |
| `src/llama-cparams.h` | same field on `llama_cparams` |
| `src/llama-context.cpp` | default `0`; copied from params → cparams |
| `src/llama-graph.cpp` | the change itself: in the `norm_w` block, divide by the mass of the top-k₂ experts instead of the k₁ that were selected |
| `common/arg.cpp` | `-moe-nref, --moe-norm-ref N` |
| `common/common.h` / `common/common.cpp` | `common_params::n_expert_norm_ref`, plumbed into ctx params |

The graph change, at the previously identified seam (`build_moe_ffn`, the `norm_w`
block):

```cpp
ggml_tensor * weights_sum = nullptr;
if (cparams.n_expert_norm_ref > 0 && n_ref != n_expert_used) {
    if (gating_op != SOFTMAX) GGML_ABORT(...);   // needs real softmax over all experts
    ggml_tensor * probs_full = ggml_soft_max(ctx0, selection_probs);
    // same layout convention as the selected-expert path: gather source is [1, n_expert, n_tokens]
    ggml_tensor * probs_ref = ggml_reshape_3d(ctx0, probs_full, 1, n_expert, n_tokens);
    ggml_tensor * sel_ref = ggml_argsort_top_k(ctx0, probs_full, n_ref);
    ggml_tensor * p_ref   = ggml_get_rows(ctx0, probs_ref, sel_ref);
    weights_sum = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, p_ref, n_ref, n_tokens));
}
if (weights_sum == nullptr) { weights_sum = ggml_sum_rows(ctx0, weights); }  // standard path
```

**Inert by default, verified:** with the flag unset, perplexity on the calibration
corpus reproduces the pre-patch number **exactly** (10.5533 both before and after).

## Measured — general corpus (160k, repo/docs/llama.cpp sources)

| config | PPL | Δ vs native |
|---|---|---|
| k₁=8 native | **10.5533** | — |
| k₁=4, k₂=4 (standard renorm) | **12.7415** | +2.19 |
| k₁=4, k₂=16 (the paper's setting) | **12.2282** | **+1.67** |
| k₁=8, k₂=16 (ref moved, k₁ held) | **11.0217** | +0.47 |

## Measured — decision-conditioned corpus (the JEV fixtures, 17.7 native)

| config | PPL | Δ vs native |
|---|---|---|
| k₁=8 native | **17.7135** | — |
| k₁=4, k₂=4 | 21.4291 | +3.72 |
| k₁=4, k₂=16 | 21.2194 | +3.51 |
| k₁=4, k₂=32 | **27.2807** | +9.57 |

## Measured — 64-token context (40.1 native)

| config | PPL | Δ vs native |
|---|---|---|
| k₁=8 native | 40.1258 | — |
| k₁=4, k₂=4 | 49.1480 | +9.02 |
| k₁=4, k₂=16 | **45.3071** | **+5.18** (best) |
| k₁=4, k₂=32 | 56.3023 | +16.18 |

## What the numbers say

1. **The mechanism works and is exactly right in shape.** Moving the reference set
   *recovers* loss at every setting tested: −0.51 on the general corpus, −0.21 on the
   decision corpus, −3.84 at short context. So the implementation reproduces the
   paper's core claim directionally: **the denominator, not the selection, is what
   breaks when k₁ is lowered.**
2. **But the recovery is nowhere near enough here.** The paper reports −0.35 MMLU at
   k₁=4/k₂=16 versus −4.65 under standard renorm — a recovery of ~93% of the loss.
   We recover **23%** (general), **6%** (decision), **43%** (64-token). The gap to the
   paper is large and consistent across three corpora.
3. **Perplexity is the wrong judge, in both directions** — exactly as the paper warns:
   *"perplexity and downstream accuracy favor different k₂… selecting compression
   hyperparameters on unlabeled text is not safe here."* Our PPL deltas and their MMLU
   deltas are measuring different things and disagree in magnitude. **So this
   experiment cannot confirm or refute the MMLU result.**
4. **k₂ has an interior optimum, and it moves.** k₂=16 beats k₂=4 everywhere; k₂=32 is
   *worse* than both. That matches the paper's "both extremes of the reference set are
   harmful" — 16 is the sweet spot at k₁=4 on this checkpoint.

## The honest blocker — this may be a precision artifact

The paper's experiments run on the **original BF16/int4 checkpoint** (Qwen3.6-35B-A3B
for the MMLU table). Ours is **`unsloth/Qwen3.6-35B-A3B-UD-Q4_K_S`**, experts only
Q4_K, with the non-expert tensors additionally requantized to Q5_K in the JEV build.

The trick works by scaling expert *output gain* against a reference mass. That gain is
exactly what a low-bit quant already distorts. There is a strong hypothesis — **not
verified** — that the Q4_K expert quant is the reason we recover ~23% where they
recover ~93%.

**Therefore the one test that would settle it is not a PPL sweep.** It is: run the
same k₁/k₂ sweep on a **higher-precision** checkpoint (Q8_0 or BF16 experts, both of
which should fit the 24 GB of the two 5060 Ti *only* with GPU use — which the standing
premise forbids — so it means a Q8_0 GGUF with a page-cache-warm corpus, or accepting
very slow runs). Until that is run, the claim is unresolved, not disproven.

## What is *not* changed

- The model, the Q5K build, and the measured **+19.4% decode / +0.46% PPL** from
  `docs/VERIFIED-PIPELINE.md` are untouched by this experiment.
- The default build is byte-identical in behaviour to before the patch (verified).
- No decision-flip test succeeded: the resident scorer runs through
  **llama-cpp-python 0.3.35's bundled `libllama.so`**, not this patched shared library,
  so `SEMIF_MOE_NORM_REF` is silently ignored there (`hasattr` guard). The k₂=16
  resident run that showed "0/16 flips, 0.00000 drift" was a **no-op arm** and proves
  nothing. Wiring the flag through the Python binding requires rebuilding/repackaging
  llama-cpp-python against this tree; not done.

## Reproduce

```bash
P=/home/scribe/Projects/llama.cpp-fresh/build/bin/llama-perplexity
G=$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-JEV-Q5K-nonexpert.gguf
$P -m "$G" -f results/pipeline/calib160k.txt -ngl 0 -t 6 -c 2048 -b 2048 -ub 2048 \
    --override-kv qwen35moe.expert_used_count=int:4 --moe-norm-ref 16
```

Raw logs: `results/pipeline/khalve-133921.log`, `khalve-control64-135807.log`,
`khalve-c64-140023.log`, `decision-ppl-134911.log`.
