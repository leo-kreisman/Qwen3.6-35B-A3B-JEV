# SOURCE: https://theneuralfeed.com/article/i-pushed-kimi-k3-onto-one-cpu-with-8-gb-of-ram/7y794Sd3

# Developer squeezes 1.56TB Kimi K3 onto 8GB RAM with C99 tricks

r/LocalLLaMA August 02, 2026

16 of 896 experts fire per token, streamed from NVMe on demand—no GPU needed.

Deep Dive

Fareed Khan, an engineer who previously deployed Kimi K3 on 32 H100s, wanted to poke at the model on his own machine. So he wrote a minimal inference engine in C99—six files, libm, OpenMP, and a 176KB binary with no BLAS, no framework, and no GPU path. The key insight is that Kimi K3 is a mixture-of-experts model: 93% of its 1.56TB checkpoint is routed experts, but only 16 of 896 fire per token. Khan never loads experts into RAM resident; instead, they're read off NVMe on demand and multiplied directly from their packed 4-bit form, skipping dequantization entirely.

His benchmark machine—2x EPYC 7763 with NVMe and four idle GPUs—hit 8.24GB peak RSS at the smallest preset (~33s/token). Bumping to 128GB of RAM yields ~20s/token, the fastest he measured, with byte-identical outputs at every memory budget. The dense trunk is repacked into a single file with known layer offsets, streamed one layer at a time. Khan is clear this isn't practical for serving: it takes half a minute per token and needs 1.7TB of free disk. It's a learning exercise to understand the architecture by implementing it. The repo includes a test mode that builds a 13-layer version and validates greedy decode plus incremental KV-cache paths against PyTorch fixtures, all in about a minute without downloading weights.

Key Points

- C99 engine runs Kimi K3's 1.56TB MoE checkpoint on CPU with 8.24GB peak RSS and no GPU
- Only 16 of 896 experts activate per token, streamed from NVMe in packed 4-bit without dequantization
- Byte-identical outputs across memory budgets; 33s/token at 8GB, 20s/token at 128GB

### Why It Matters

Proves massive MoE models can run on commodity hardware via smart streaming, opening local inference to low-RAM devices.

📬 Get the top 10 AI stories daily
