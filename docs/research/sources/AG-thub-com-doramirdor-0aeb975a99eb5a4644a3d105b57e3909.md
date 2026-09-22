# SOURCE: https://gist.github.com/doramirdor/0aeb975a99eb5a4644a3d105b57e3909

Show Gist options

- You must be signed in to star a gist
- You must be signed in to fork a gist

- Save doramirdor/0aeb975a99eb5a4644a3d105b57e3909 to your computer and use it in GitHub Desktop.
Save doramirdor/0aeb975a99eb5a4644a3d105b57e3909 to your computer and use it in GitHub Desktop.

 mbolt × mlx-lm#1438: profile-guided expert layout for MLX SSD-offload (Qwen3.6-35B-A3B-8bit) — coalescing = ~20% decode / ~10% prefill, profile-free; reorder marginal

# mbolt × mlx-lm #1438 — profile-guided expert layout for MLX SSD-offload

> **Update 2026-07-20 — cross-model null, and the law that explains it.** The lever does NOT strengthen with model scale. Measured on gpt-oss-120b-MXFP4 (4 MB weight slices): A2 drops to 1.29× I/O ≈ +9.7% decode — the misses are already bandwidth-bound, so the per-op latency the coalesce reclaims is gone. Rule of thumb from the four measured points: `gain ≈ 1 + saved_ops · t_op / (slice_bytes / BW)` (t_op ≈ 100 µs, BW ≈ 6 GB/s). **Coalescing pays iff misses are latency-bound: fine-grained MoEs and low-bit quants (~≤1 MB slices), not big-expert models.** Full story in the issue thread.
>
> **Update 2026-07-22 — GLM-5.2 point lands; dense-prefix models now handled.** A downstream team (pierre427, mlx-lm#1588) ran the pipeline on GLM-5.2-affine-4bit (~20.25 MiB experts, 407.69 GB side-file, bit-exact 128/128): **measured A2-vs-9-scattered g = 1.195 against a pre-registered 1.2 (band 1.15–1.3)** — the law's first registered-then-measured point. Their decomposition refines the t_op term: it's a *locality* cost, not per-op overhead (adjacent 9-sub reads beat one block pread by ~7%; true scatter costs ~64 µs at 8.2 GB/s), so A1 is not an intermediate rung once subs are adjacent. Tooling fix from their run: `st_map` completeness validation now iterates the discovered MoE layer set (`STMap.layers`) instead of `range(n_layers)` — GLM's first 3 layers are dense and tripped the assert. `coalesce_rewrite.py` writes sparse stacks correctly (manifest schema unchanged; entries always carried absolute `layer` ids, plus new top-level `moe_layers`). Trace-coupled analysis scripts (readmodel/cluster_local) still assume trace layer i = i-th MoE layer; map through `STMap.layers` for dense-prefix models when a GLM trace exists.
>
> **Update 2, same day — first in-engine run, and the completing term.** An independent implementer ran `coalesce_rewrite.py` unmodified on Qwen3-235B-4bit (119 GB side-file, bit-exact 64/64) against his own byte-range reader at 0.2 resident (table = 2.6× RAM): **A1 = +32.3% decode, −26.3% TTFT**, preads/token exactly 3× down, text identical. Not the per-read lever strengthening (g ≈ 1.34 there, weaker than the 8-bit case) — the cold-read share s ≈ 0.96 at deep offload. Full decomposition: `speedup = 1 / ((1−s) + s/g)`; g falls with slice size, s → 1 as the fraction falls. Deep offload pays double digits even with a modest per-read gain.

Companion code + results for [ml-explore/mlx-lm#1438](https://github.com/ml-explore/mlx-lm/issues/1438) (MoE expert streaming / SSD offload on Apple Silicon).

Measures the on-disk **expert-layout read cost** for `mlx-community/Qwen3.6-35B-A3B-8bit` under an explicit-read offload streamer, using @mabaeyens' Mira routing trace. Method mirrors [mbolt](https://github.com/doramirdor/mbolt): read-op counts under a per-layer LRU + a measured cold-SSD per-op wall on the real shards (bytes-are-bytes, so a contiguous read from a cold region is a faithful proxy — no 33 GB rewrite).

## Result (honest, end-to-end)

Routed experts are 90.7% of the shards; one expert = **9 sub-slices** (gate/up/down × weight/scales/biases) = 3.19 MiB **scattered across a 2.5 GB in-file span**. Profile-free **coalescing** of those 9 into one contiguous block:

- **~20% faster decode, ~10% faster prefill** in the over-DRAM regime. (Cold expert reads are ~30% of the decode token / ~14% of prefill; coalescing cuts the per-cold-expert I/O **3.0×**, measured 1396 → 463 µs, so `0.70 + 0.30/3.0 = 0.80`.)
- **9× fewer decode read-ops** (486 → 54 / token @ 0.3 resident). It's an IOPS/latency win, not bandwidth (bytes/token 190 → 182): the 6 tiny 32 KiB scales/biases reads per expert (96 µs each) get folded into the contiguous block.
- Co-activation **reorder** adds only ~12% on decode (generalizes cross-topic; bigger on prefill). Optional polish — ship the deterministic coalesce.
Two layouts matter: **A1** (coalesce {weight,scales,biases} per projection — no trace, fits a per-module lazy fetch) and **A2** (coalesce the whole expert — needs fetch-at-dispatch). See `REPORT.md` for full tables and the three validations against @mabaeyens' machine.

## Files

| file | what |
|---|---|
| `st_map.py` | safetensors expert-slice map (the 9 stacked `switch_mlp` tensors/layer) |
| `cluster_local.py` | co-activation clustering → clique/chain perms (port of `mbolt.cluster`) |
| `trace_load.py` | Mira routing-trace loader |
| `readmodel.py` | read-op counts under a per-layer LRU; layouts stock / A1 / A2 / A2B |
| `wall.py` | physical cold-SSD per-op microbench (F_NOCACHE) |
| `coalesce_rewrite.py` | emit the coalesced expert side-file + manifest (bit-exact, verified) for the offload `fetch_fn` — one file serves A1 & A2, `--reorder` gives A2+B |
| `analyze.py` | driver: signal + read model + cross-topic transfer |
| `report.py`, `chart.py` | `REPORT.md` + `summary.png` |
| `REPORT.md`, `*.json`, `summary.png` | generated results + per-layer perms (`perms_full.json`) |

## Reproduce

```
 pip install numpy networkx matplotlib export MBOLT_MODEL=/path/to/Qwen3.6-35B-A3B-8bit export MBOLT_TRACE=/path/to/doramirdor_trace.jsonl export MBOLT_OUT=./out python analyze.py && python wall.py && python report.py && python chart.py
```

- Trace: [https://gist.github.com/mabaeyens/de8314ba6d90e50ce46d5dd328682e4a](https://gist.github.com/mabaeyens/de8314ba6d90e50ce46d5dd328682e4a)
- Model: [https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit)
Layouts are bit-exact relayouts of the same expert bytes. QD1-serial wall (matches the "decode faults serialize" regime); the read-model above uses a bytes-are-bytes proxy.

To measure the **true end-to-end wall**, `coalesce_rewrite.py` emits the physical coalesced expert file + manifest so an offload `fetch_fn` can read it directly:

```
 # A1/A2 (natural order, profile-free) — full model ~34 GB, ~40 s python coalesce_rewrite.py --out ./out/experts_coalesced.bin # A2+B (co-activation reorder) — needs perms_full.json from analyze.py python coalesce_rewrite.py --reorder --perms ./out/perms_full.json --out ./out/experts_A2B.bin
```

Each writes `<file>.manifest.json`: per layer, `expert_offset[e]` (byte offset of original expert id e's block), `block_bytes`, the 9 `subs` (rel-offset/size/dtype/per-expert shape), and `proj_ranges` (one contiguous range per projection, for A1). The rewrite is offline and verified bit-exact against the source (`--verify-samples`). Non-expert tensors are untouched — load them from the original model; only the routed-expert fetch points at this file.

# mbolt × mlx-lm #1438 — safetensors expert-layout results

**Model:** `mlx-community/Qwen3.6-35B-A3B-8bit` (40 MoE layers, 256 experts, top-8). **Trace:** mabaeyens' Mira routing trace, 13,203 tokens (code/prose/agentic). **Method:** hardware-independent read-op counts under a per-layer LRU (his own methodology) + measured cold-SSD per-op wall on the real shards (bytes-are-bytes proxy, no 33 GB rewrite). Routed experts = 90.7% of shards; one expert = 9 sub-slices (gate/up/down × weight/scales/biases) = 3.19 MiB, scattered across a 2.5 GB in-file span today.

## Independent validations against his measurements

- decode hit rate @ 0.3 resident: **0.829** (his ~0.83)
- stock opens/token @ 0.3: **486** (his ~471)
- cold 1.05 MiB read: **286 µs** (his 254 µs / 1.1 MB)

## Signal on this model

- heat entropy 7.37/8 bits (92% of max)
- **32% of expert pairs co-activate >2× expected** (range 24%–42% across layers); median 5 clusters/layer → real clique structure.

## Measured per-cold-expert read wall (QD1, this SSD)

| fetch | median µs | vs stock |
|---|---|---|
| stock 9 scattered (1396 µs) | 1396 | 1.0× |
| A1 3 contiguous proj blocks | 619 | 2.25× |
| A2 1 contiguous expert block | 463 | 3.02× |
The win is IOPS/latency, not bandwidth: bytes/token barely move (190→182). The 6 tiny 32 KiB scales/biases reads/expert are latency-bound (96 µs each → 325 MB/s); coalescing folds them into the contiguous block.

## Decode

### Decode, 0.3 resident (his regime) (cap=77 = 30% resident, hit=0.829)

| layout | reads(=opens)/tok | vs stock | MB/tok | I/O ms/tok (QD1) | speedup |
|---|---|---|---|---|---|
| stock (9 scattered/expert) | 485.8 | 1.0× | 190.1 | 75.3 | 1.00× |
| A1 per-proj coalesce (3/expert) | 161.9 | 3.0× | 182.1 | 33.4 | 2.25× |
| A2 per-expert coalesce (1/expert) | 54.0 | 9.0× | 182.1 | 25.0 | 3.02× |
| A2+B coalesce + clique reorder | 47.7 | 10.2× | 182.1 | 22.1 | 3.41× |

## Prefill / cold (no effective retention — his 'warm prefill = cold')

### Cold, no retention (cap=0 = 0% resident, hit=0.000)

| layout | reads(=opens)/tok | vs stock | MB/tok | I/O ms/tok (QD1) | speedup |
|---|---|---|---|---|---|
| stock (9 scattered/expert) | 2800.2 | 1.0× | 1115.3 | 434.4 | 1.00× |
| A1 per-proj coalesce (3/expert) | 933.8 | 3.0× | 1069.5 | 192.7 | 2.25× |
| A2 per-expert coalesce (1/expert) | 311.3 | 9.0× | 1069.5 | 144.0 | 3.02× |
| A2+B coalesce + clique reorder | 227.8 | 12.3× | 1069.5 | 105.4 | 4.12× |

## Lever B (reorder) generalization — cross-topic

Perm trained on code+prose, evaluated on held-out **agentic** @ 0.3 resident:

- A2 (no perm): 66.4 reads/tok
- A2+B held-out perm: 59.9 (in-sample perm: 58.3)
B **generalizes** across topics (held-out ≈ in-sample), unlike heat-pinning — but its marginal value over profile-free A2 is small on decode (~13%) and larger on cold/prefill (~37%).

## Bottom line

- **Coalescing (A2, profile-free) is the win: 9× fewer decode read-ops (486→54/tok), 3.0× faster per cold expert.** Deterministic, bit-exact, no trace needed.
- **A1 (per-projection, deploys under the current per-module lazy fetch, no fetch-at-dispatch) already gets 2.25×** — the minimal-repack option.
- **Reorder (B) is marginal on decode (+13%), more useful on prefill (+37%), and generalizes** — optional polish that only earns the profiling step for prefill-heavy use.
- Confirms the prior: A carries it, B is conditional.
