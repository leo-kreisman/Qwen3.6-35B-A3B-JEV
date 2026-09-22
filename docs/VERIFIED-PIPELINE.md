# Verified pipeline results — 2026-09-22

Everything below is measured on this host (i7-8700K, 6c/12t, DDR4-2400, NVMe
KINGSTON SFYRD4000G), CPU-only (`-ngl 0`), no GPU. Nothing committed.

## Lever 1 — non-expert requantization: **+19.4% decode for +0.46% PPL**

The finding (see `docs/ARCHITECTURE-PIECEWISE.md`): the non-expert weights are
**77.9% of per-token traffic** and are stored at Q8_0/Q6_K, while the project has only
ever optimized the expert path (22.1%). Attention/SSM alone is 53.3% of traffic.

**What was built.** A GGUF with only the *always-active* tensors downshifted
Q8_0→Q5_K, everything else pinned to its existing type:

- 250 tensors requantized: `attn_qkv`, `attn_q`, `attn_k`, `attn_v`, `attn_output`,
  `attn_gate`, `ssm_out`, `ffn_{up,gate,down}_shexp`
- **experts untouched** (all 18.33 GB stay Q4_K/Q6_K, byte-identical)
- **router untouched** (`ffn_gate_inp` stays F32 — it *chooses* experts; a precision
  change there flips decisions)
- **`token_embd` untouched** (Q8_0 — one row is read per token, it is a lookup, not a
  stream; optimizing it helps RAM, not bandwidth)

**Method — the safety detail that matters.** `llama-quantize --tensor-type` only
overrides the tensors you *list*; everything else would be re-quantized to the base
type, which would have lossily re-quantized all 18.33 GB of experts. So the type file
**pins all 733 tensors** to their current type and changes only the 250 intended.

```bash
# 1. imatrix over a domain-relevant corpus (repo docs+python + llama.cpp sources)
llama-imatrix -m Qwen3.6-35B-A3B-UD-Q4_K_S.gguf -f calib160k.txt \
              -o imatrix.gguf -ngl 0 -t 6 -c 2048 -b 2048 -ub 2048 --chunks 60

# 2. full tensor-type pin file generated from the source tensor table (733 lines)
python3 gen_ttypes.py     # -> ttypes_full.txt, 250 changes Q8_0->Q5_K

# 3. requantize
llama-quantize --allow-requantize --imatrix imatrix.gguf \
               --tensor-type-file ttypes_full.txt \
               Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
               Qwen3.6-35B-A3B-JEV-Q5K-nonexpert.gguf Q4_K_S 6
```

**Size:** 19,914.65 MiB (4.82 BPW) → 19,422.78 MiB (4.70 BPW).

**Per-token traffic, verified from the new file's tensor table:**

| class | before GB/token | after | delta |
|---|---|---|---|
| attn_qkv | 0.5348 | 0.3460 | −0.1887 |
| ssm_out | 0.2674 | 0.1730 | −0.0944 |
| attn_gate | 0.2674 | 0.1730 | −0.0944 |
| attn_q | 0.1783 | 0.1153 | −0.0629 |
| attn_output | 0.0891 | 0.0688 | −0.0203 |
| ffn_*_shexp | 0.1338 | 0.0864 | −0.0471 |
| attn_k / attn_v | 0.0222 | 0.0144 | −0.0078 |
| **experts (3 classes)** | 0.5726 | **0.5726** | **0.0000** ✓ |
| router (F32) | 0.0839 | 0.0839 | 0.0000 ✓ |
| **TOTAL** | **2.5874** | **2.0716** | **−0.5158 (−19.9%)** |

**Speed — same-session alternating A/B, 2 rounds, `-p 2048 -n 64 -ub 2048 -t 6 -r 2`:**

| model | decode r1 | decode r2 | prefill r1 | prefill r2 |
|---|---|---|---|---|
| baseline Q4_K_S | 10.73 | 10.71 | 567.62 | 566.17 |
| **JEV Q5K non-expert** | **12.84** | **12.73** | 577.29 | 578.89 |

- **decode 10.72 → 12.79 tok/s = +19.4%**
- prefill 566.9 → 578.1 tok/s = +2.0%
- prediction was 1.249×, measured 1.193× — same direction, ~5% optimistic

**Quality gate (`llama-perplexity`, 23 chunks, n_ctx 2048):**

| model | PPL | ± |
|---|---|---|
| baseline Q4_K_S | **10.5051** | 0.19326 |
| JEV Q5K non-expert | **10.5533** | 0.19462 |

**Δ = +0.0482 PPL = +0.46%, inside the ±0.19 error bars.** Per-chunk the variant is
worse at all 23 chunks by ~0.05, so it is a real but very small degradation — far
smaller than a typical quant step.

## Lever 2 — prefill config: **2.79× at zero quality cost**

`ubatch` was the lever (slipstream measured 2.7× on agent-scale prompts with
`ubatch 2048` + parallel I/O threads). Measured here, 4096-token prompt, `-b 8192`:

| threads | ubatch | prefill tok/s | decode tok/s |
|---|---|---|---|
| 6 | 512 | 203.24 | 11.05 |
| 8 | 512 | 203.25 | 10.90 |
| **6** | **2048** | **567.27** | **11.08** |
| 8 | 2048 | 566.84 | 10.92 |

**Prefill 203.2 → 567.3 tok/s = 2.79×. Decode is unchanged (11.05 → 11.08).** Pure
config, lossless, and it matches slipstream's independently measured 2.7×.

This matters more than any decode number for the coding-agent use case: an agent
resends a large prompt every turn and **prefill is the expensive phase under SSD
offload**.

## Corrections to the project's own config, now measured

| setting | project value | measured best | effect |
|---|---|---|---|
| `_ubatch()` | 512 | **2048** | prefill 2.79× (unchanged decode) |
| `_default_threads()` | 16 | **6–8** | 16 is 1.66–1.71× *slower* in the bandwidth-bound regime |

## Files

- Model: `/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-JEV-Q5K-nonexpert.gguf`
- Imatrix: `results/pipeline/imatrix.gguf`
- Pin file: `results/pipeline/ttypes_full.txt` (733 lines, 250 changes)
- Calibration corpus: `results/pipeline/calib160k.txt`
- Raw A/B: `results/pipeline/ab-124355.jsonl`; PPL: `results/pipeline/ppl-124559.log`
- Prefill sweep: `results/prefill-tuning/sweep-123623.jsonl`

## Where this leaves the 20+ tok/s target

Measured now: **12.79 decode / 578 prefill**, from a 10.72 / 567 baseline. Still short
of 20 decode.

Next lever, seam identified but **not built**: the **k₁/k₂ renormalization** from
arXiv:2609.04575 — activate the top k₁ experts but normalize by the top k₂≥16 mass
(`−0.35 MMLU instead of −4.65` on this exact model). It halves expert bytes/token
(0.573 → 0.287, −13.8% of current total ⇒ ~14.8 tok/s) **and** halves expert compute.
The code site is `src/llama-graph.cpp:2134` — the `norm_w` block sums `weights` over the
selected k₁; the fix is to sum over a top-k₂ set instead:

```cpp
// at ~line 2120, alongside the k1 selection
ggml_tensor * sel2  = ggml_argsort_top_k(ctx0, selection_probs, n_norm_experts);
ggml_tensor * p2    = ggml_get_rows(ctx0, probs, sel2);        // [1, k2, n_tokens]
ggml_tensor * sum2  = ggml_sum_rows(ggml_reshape_2d(ctx0, p2, n_norm_experts, n_tokens));
// then in the norm_w block, divide by sum2 instead of sum over k1
```

It also needs one integer plumbed from the CLI (`common/arg.cpp`) through
`llama_context_params` to `build_moe_ffn`. **Not started, not tested** — it changes
routing, so it needs its own decision-flip measurement before any claim.
