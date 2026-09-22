# SOURCE: https://inventivehq.com/blog/moe-on-cpu-benchmark

Mixture-of-Experts (MoE) models have a confusing spec sheet. A model labeled **30B-A3B** stores 30 billion parameters but only *activates* about 3 billion of them per token. So when you go to run one locally, which number actually governs how fast it goes — the big total, or the small active slice?

On a GPU the answer barely matters, because GPUs have compute and bandwidth to spare. On a **CPU**, where single-user decoding is bottlenecked on memory bandwidth, it's the whole ballgame. So we measured it. We ran a **Qwen3-30B-A3B** MoE against a ladder of dense models — 1.5B, 3B, 7B, and 8B — on a plain **Intel i7-8700**, an 8-year-old desktop CPU with no GPU in the loop. Everything below is reproducible on a recent [llama.cpp](https://github.com/ggml-org/llama.cpp) build, and all of our raw data and the harness are [open source](https://github.com/InventiveHQ/local-llm-benchmarks/tree/main/experiments/moe-on-cpu).

### The result, up front

The 30B-A3B MoE decoded at **10.6 tok/s** on the i7-8700 — essentially the speed of a dense **3B** (11.8 tok/s), a model with roughly 10× fewer total parameters. A genuinely *dense* 30B would crawl near **~1.16 tok/s** on the same CPU, so the MoE is about **9.2× faster** than its total size suggests, for free, by construction. And it isn't trading quality for that speed: it scored **8/10** on our graded math/reasoning set versus the dense 3B's **1/10**.

| Model | Kind | Active B | Total B | Size (MB) | tok/s | Quality |
|---|---|---|---|---|---|---|
| 1.5B dense | dense | 1.5 | 1.5 | 986 | 22.4 | 1/10 |
| 3B dense | dense | 3.1 | 3.1 | 1930 | 11.8 | 1/10 |
| **30B-A3B MoE** | **MoE** | **3.3** | **30.5** | **18557** | **10.6** | **8/10** |
| 7B dense | dense | 7.6 | 7.6 | 4683 | 4.9 | 5/10 |
| 8B dense | dense | 8.0 | 8.0 | 4921 | 4.2 | 1/10 |
The short version: on a bandwidth-bound CPU, an MoE runs at the speed of its **active** size but answers at the quality of its **total** size. That makes it the best model you can run on a RAM-rich box with no usable GPU. Below is exactly why, and how to size your own machine for it.

## The question: do you pay for the parameters you store, or the ones you use?

Dense models read *every* weight to generate *every* token. An MoE is built differently: its layers hold many parallel expert sub-networks, and a small router picks just a few to run for each token. A 30B-A3B model stores 30 billion parameters but activates only about 3.3 billion per token.

That distinction barely matters on a fast GPU, but it's decisive on a CPU. Single-user CPU generation is **memory-bandwidth-bound**: the speed ceiling is set by how many weight bytes you stream from RAM for each token. If only the active experts are read, an MoE should decode at roughly the speed of a dense model its *active* size — while answering like something far bigger. This experiment tests exactly that, on hardware most people already own.

## How MoE routing saves bandwidth

Both models must be fully **resident in RAM** — every expert might be needed on the next token. But each individual token only *reads* the active experts, and on a bandwidth-bound CPU, bytes-read-per-token is what sets your tok/s.

> The catch is memory *capacity*, not speed: all 30B of experts have to fit in RAM, because which ones fire changes token to token. At Q4 that's ~18 GB — fine on a 32 GB+ machine, impossible to keep resident on most 16 GB GPUs. CPU plus lots of cheap system RAM is the MoE's natural home. (For the GPU side of this trade-off, see our guide on [how much VRAM you need to run an LLM](/blog/how-much-vram-to-run-an-llm).)

## Why not just use speculative decoding?

A fair question, since "speculative decoding" is the other headline trick for going faster. NVIDIA's dFlash and the EAGLE family hit their big multiples with a *trained, target-conditioned draft head* running in TensorRT-LLM on Blackwell GPUs at high concurrency — not something that exists in llama.cpp or on a CPU, and there's no published draft head for this model anyway. And as our [first benchmark in this series](/blog/local-llm-performance-what-to-expect) discussed, at batch size 1 on a single device a generic draft model is overhead-governed and often *loses*. So the interesting lever here isn't a bolt-on decoder trick — it's the MoE **architecture itself**. That's what we measure.

## The setup

We kept the methodology deliberately boring so the numbers are comparable.

| Component | What we used |
|---|---|
| **Device** | CPU only (`-ngl 0`), 6 threads = physical cores |
| **CPU** | Intel Core i7-8700 (6 cores / 12 threads, Coffee Lake) |
| **RAM** | 64 GB DDR4-2400 (4 × 16 GB, dual-channel) |
| **Runtime** | llama.cpp, a build recent enough for the `qwen3moe` architecture |
| **MoE** | Qwen3-VL-30B-A3B-Instruct, Q4_K_M (30.5B total / ~3.3B active) |
| **Dense refs** | Qwen2.5-Coder 1.5B / 3B / 7B and Llama-3.1-8B, all Q4_K_M |
| **Speed test** | Shared 12-prompt suite, greedy (temp 0), 256 tokens — server `predicted_per_second` |
| **Quality** | 10 multi-step math/reasoning problems, exact integer match (no code executed) |
Because this run is entirely on the CPU, the lever is **DDR4-2400 dual-channel** memory bandwidth — DDR5 or more channels should lift the whole curve. The number of threads matters too; we used 6, matching the physical cores (more on that in our look at [CPU thread scaling for LLM inference](/blog/cpu-thread-scaling-llm-inference)).

## Result 1: the MoE runs at active-parameter speed

Plot tok/s against each model and the pattern is unmistakable. The 30B-A3B MoE (green) lands right next to the dense 3B, exactly where its **active** size predicts — nowhere near the dashed red line where a genuinely dense 30B would sit.

Every model — dense and MoE alike — falls on one bandwidth curve. The MoE isn't special-cased: read ~3.3B of weights per token and you get ~10.6 tok/s, whether those weights came from a dense 3B model or were routed out of a 30B pool. The total size only governs how much RAM you need to hold the model, not how fast it runs. This is the single most useful fact about running MoE locally: **budget your speed by active size, your RAM by total size.**

## Result 2: same speed as a 3B, the smarts of something much bigger

Speed is only half the story. If the MoE merely matched the 3B's *quality* too, none of this would be interesting. It doesn't. On the graded set the dense models trace the usual tradeoff — faster means smaller means lower scores — and the MoE breaks straight through it.

Up-and-to-the-right is better. The dense models sit along the dashed frontier; the MoE clears it — the small-model speed, a bigger-model score. Read the quality numbers within reason: the dense references span two families (Qwen2.5-*Coder* and a general Llama-3.1-8B), and the graded set is math and word problems, so the Llama-8B's low score reflects task fit, not raw capability. The cleanest read is *within* the Qwen family: 3B (1/10) → 7B (5/10) → the 30B-A3B MoE (8/10), with the MoE both the most accurate and the second-fastest of the entire set. (For more on how small a model can get before quality falls off, see [the smallest LLM that can code](/blog/smallest-llm-that-can-code).)

## What this means in practice

**The bottleneck is RAM, and RAM is cheap.** The same MoE will decode at very different speeds on different machines, depending entirely on memory bandwidth — dual-channel DDR4-2400 (this box) sits at the low end; DDR5-6000 or a quad/eight-channel workstation moves the whole curve up. But unlike VRAM, system RAM is cheap and plentiful. That puts a 30B-class model within reach of a machine with no usable GPU at all. You're no longer gated by a 16 GB VRAM ceiling — only by how many DIMMs you can afford.

**This is a quality *proxy*, on a small set.** The 10-question graded set measures multi-step reasoning by exact-answer match — a signal, not a full benchmark, and no model code is executed. Treat the quality numbers as "which tier is this model in," not a leaderboard. The speed result, by contrast, is robust: it's a direct consequence of how many bytes the runtime streams per token. (If you're calibrating what to expect from local hardware generally, our [local LLM performance guide](/blog/local-llm-performance-what-to-expect) sets the baselines.)

**The practical takeaway:** if your machine has lots of RAM and no big GPU, an MoE is the highest-quality model you can run at a usable speed. A 30.5B-A3.3B model needs ~18 GB of RAM but decodes like a 3B (10.6 tok/s here) while answering like something far larger. Pick the MoE whose *active* size your CPU can decode fast enough, then give it as much RAM as its *total* size needs. Faster memory raises the speed; it doesn't change the rule. On a fast GPU the dense 3B and the MoE feel similarly quick, so the MoE's edge is invisible. On a CPU, the dense model that could match the MoE on quality — a real 30B — would be ~9× slower, unusable for interactive work. The MoE gives you that quality tier at small-model speed. For CPU inference, that's the difference between "too slow to use" and "fine."

## Reproduce it — and send us your numbers

Get a recent llama.cpp build (the MoE architecture is new), download the GGUFs into your models directory, then run the CPU-only sweep:

```bash
 # CPU-only sweep — missing models are skipped automatically python scripts/bench.py --threads 6 python scripts/aggregate.py # writes results/data.json python scripts/inject.py # bakes data into the report
```

Memory bandwidth is the whole story here, so DDR5 and more memory channels should shift your curve up — submissions across memory configurations are especially valuable. Contributing is one command and a pull request:

```bash
 python scripts/make_submission.py --name "ryzen7950x-ddr5" --notes "DDR5-6000, llama.cpp bXXXX"
```

That opens a PR-ready `results/community/<name>.json` with your hardware auto-captured. This benchmark is experiment #12 in our [open-source local-LLM benchmark series](https://github.com/InventiveHQ/local-llm-benchmarks/tree/main/experiments/moe-on-cpu), and the point is to build a community map of how MoE decode speed scales across real memory configurations. An AMD box on DDR5? An eight-channel Threadripper? A laptop? Every data point sharpens the picture.

---

*New to running models on your own hardware? Start with our [complete guide to running local AI](/blog/running-local-ai-the-complete-guide), then size your machine with [how much VRAM you need to run an LLM](/blog/how-much-vram-to-run-an-llm) and [what tokens-per-second to expect](/blog/local-llm-performance-what-to-expect).*
