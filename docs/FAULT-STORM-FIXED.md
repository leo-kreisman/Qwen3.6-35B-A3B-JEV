# The fault storm is fixed: BigMoeOnEdge streams this model at 7.41 tok/s under 8 GB

The hostile part of this project was never the *bytes* — it was that the box kept dying.
A 20.9 GB model behind a demand-paged mmap under an 8 GiB cap is a **fault storm**: the
OS evicts weights as fast as it reads them, speed swings wildly, and other processes get
killed. That is what crashed this machine, and `LOAD_MODE=direct_io` did not fix it — it
OOM-killed at 19,914 MiB because it loads the whole model instead of streaming.

It is fixed now, by an engine built for exactly this case.

## The engine

**BigMoeOnEdge** (github.com/Helldez/BigMoeOnEdge) — CPU-only, lossless MoE weight
streaming, built **on llama.cpp's public API**:

> "it keeps the small always-needed part of the model at hand and reads just the experts
> each token asks for, directly from flash storage, at the moment they are needed. The
> rest of the model stays on disk… the output is byte-identical to running the same model
> fully resident."

Mechanism, in their words: *"The model loads file-backed, and the engine hooks llama.cpp's
public evaluation callback. When a layer picks its experts for the current token, the
engine fetches exactly those slices from flash just in time for the matmul, optionally
caching the hottest ones and overlapping reads with compute. **No llama.cpp sources are
modified.**"* The only exception is the optional `--overlap`, which needs a ~25-line hook
on their fork branch.

`qwen35moe` is a supported architecture, named explicitly in their table: *"Qwen3.6-35B-A3B
and siblings — Hybrid attention/SSM stack; routed experts stream unchanged."*

Their README describes our crash before we hit it: *"An 18 to 22 GB model on a 12 GB phone
is where the ordinary way of loading (mmap) turns into a fault storm: the OS evicts weights
as fast as it reads them, speed swings wildly, other apps get killed."*

## Measured on this box, 8 GiB cap, CPU-only

`scripts/probes/bench_bmoe_sweep.sh`, `MemoryMax=8G MemorySwapMax=0`, our
`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`, `-t 6 -c 4096 -n 128`, `--moe-stream --dense-weights anon
--overlap`:

| cache MiB | io lanes | tok/s | MiB/token | hit% | flash s/token | major faults | OOM |
|---|---|---|---|---|---|---|---|
| 0 (off) | 4 | **3.834** | 549.94 | — | 0.621 | 0.00 | no |
| 2000 | 4 | **5.877** | 187.19 | 61.7 | 0.206 | 0.00 | no |
| 3000 | 4 | **6.536** | 138.66 | 69.9 | 0.157 | 0.00 | no |
| **4000** | **8** | **7.408** | **105.28** | **75.5** | 0.185 | 0.00 | no |

- **3.834 → 7.408 tok/s = +93%**
- **549.94 → 105.28 MiB/token = −81% reads**
- **0.00 major faults/token in every run** — the fault storm is gone, and every cell
  survived the cap. Contrast the old path: crash.

The cache is the dominant lever, which is what the engine's own numbers predict. Their
desktop row for this same model (x86, 16 GB, NVMe) reads 4.8 tok/s with cache auto and
**7.3 tok/s with overlap + drop 75%**, so 7.41 here is consistent with their measurement
rather than an outlier.

## The convergence worth recording

Their desktop analysis reaches **our** conclusion from an independent direction:

> "on this laptop it is **DRAM-bandwidth-bound in compute (~0.11 s/token in every cell, so
> a ~9 tok/s ceiling even at zero I/O)**. More io lanes, more compute threads and the
> ~3 GB/s NVMe are all neutral; the levers that pay are the cache budget, cache-aware
> dropping and `--overlap`."

That is the same wall this project computed as 38.4 GB/s ÷ 2.5874 GB per token ≈ 14.84
tok/s on DDR4-2400, and our measured 10.73 decode sitting at 72% of it. Two independent
measurements, same shape: **on a DDR4 desktop the ceiling is DRAM bandwidth, not the SSD.**

Their conclusion follows: *"If the model fits in RAM, just run it resident."*

## Where this leaves the target

| | tok/s |
|---|---|
| resident `llama-server` (warm, no cap) | 10.73 |
| **streamed, 8 GiB cap, BigMoeOnEdge** | **7.41** |
| streamed, cache off (floor) | 3.83 |
| Edge0 (M4 Pro, 2.9 GiB) | 20.4 |
| the 20+ target | not reached |

Streaming under a cap now costs **1.45×** against resident (7.41 vs 10.73) instead of
being a crash, and the loss is entirely the flash reads the cache cannot cover.

**Still not 20+.** Their own analysis says why: on DDR4 the compute path saturates near
~9–14 tok/s regardless of I/O. 20+ on this box means either the GPU (forbidden by the
standing premise) or a smaller active-expert set — and their lossy knobs exist for that:
`--n-expert-used 6` is measured by them at 5.8 vs 5.0 tok/s on a phone for this model, and
`--drop-cold-experts 0.75` at 7.3 vs 4.8 on their desktop. Both change the output.

## Reproduce

```bash
cd ~/Projects/JEV_experiment
git clone --recursive https://github.com/Helldez/BigMoeOnEdge.git vendor/BigMoeOnEdge
(cd vendor/BigMoeOnEdge && scripts/build-host.sh)
CAP=8G GRID="0:4 2000:4 3000:4 4000:8" ./scripts/probes/bench_bmoe_sweep.sh
```

`vendor/` is gitignored (648 MB, someone else's history) — same policy as `semif/` and
`offload_projects/`. Raw logs and CSVs: `results/bmoe-sweep-20260922-191242/`.

## What is still open

1. **Not wired to OpenCode.** `bmoe-cli` has `--session` (JSON prompt requests on stdin)
   and is not an HTTP server, so OpenCode still cannot reach any of this. The bridge is
   unbuilt — same gap as before, now with a faster engine behind it.
2. **The cache budget is capped by the working set.** 4000 MiB is where this sweep stopped,
   not where the cache stops paying; `--cache-ceil-mb 6000` is untested.
3. **Their lossy levers are untouched here** — `--drop-cold-experts`, `--n-expert-used 6`,
   `--expert-substitute`. They are where the remaining speed is, and they cost quality.
4. **`--ppl` exists** to price those lossy settings on a fixed token sequence, which is the
   honest way to test them; not run.
