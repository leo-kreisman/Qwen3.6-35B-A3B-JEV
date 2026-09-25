# Tiered expert storage: implemented and tested, not enabled by default

2026-09-25. The best supported deployment choice from this experiment is to keep
the existing weights. The conservative tiered arm saved 4% of expert I/O, did not
improve decision latency, and flipped a correct answer. The implementation and
measurement tools are retained as an opt-in research path.

## What was built

Inspired by [Alexintosh's Flash-MoE model](https://huggingface.co/alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE)
and its [tiered quantization design](https://github.com/Alexintosh/flash-moe/blob/main/docs/tiered-expert-quantization.md).
No Apple checkpoint was downloaded. The local source is the already-existing
`Qwen3.6-35B-A3B-JEV-Q5K-nonexpert.gguf`, SHA256
`5cacb2a89facde696385d12a2eacb7a031feff0456d3c7d05ce73a2ae68a6f24`.
Its routed experts remain the original Q4_K/Q6_K set before applying a sidecar.

The current CPU graph assigns one quantization type to an entire expert tensor.
To avoid rewriting that graph or llama.cpp, this implementation uses a **storage
codec**. Hot experts keep their original bytes; cold Q4_K blocks use two-bit
indices into a small codebook, with the original scales/minima preserved. An
81-byte stored block expands to a 144-byte Q4_K block on read. All Q6_K expert
slices are preserved. Scalar and SSSE3 decoders are provided.

This is not Flash-MoE's quantizer. In particular, its constrained nibble codebooks
do not optimize new per-group scales the way that implementation does. The
negative quality result here does not establish the quality of the linked model.

Files and capabilities:

- `scripts/tiered/build.py`: routing profile -> hot set -> bounded-memory sidecar
  builder, with source SHA256 and calibration trace hashes.
- `patch/tiered-bmoe/changes.patch`: reproducible patch against BigMoeOnEdge
  `74ba18f53d308e3271a6b7dbbdaf8e59af64cdde`; no llama.cpp edits.
- `--tiered-pack`: direct-I/O reader, per-lane staging, source/geometry validation,
  native-format expansion and read/expansion telemetry. Default off.
- `--choices-only`: final-label scoring of rendered chat prompts, without
  generating text or computing prompt perplexity. Rejects multi-token labels.
- `scripts/tiered/score.py`: JEV JSONL input, SemIf prompt rendering, conditional
  option probabilities and an enforced 8 GiB/no-swap scope. Questions are scored
  separately inside one loaded process; this does not replace the existing batched
  resident HTTP scorer.
- `run_guarded.py`, `analyze.py`, `materialize.py`, fixtures and tests: cold-cache
  measurement, row-level comparison and an expanded same-weights reference.

The original GGUF is still required: **sidecars add installed disk usage**.
The compact byte counts below describe the representation of the expert payload,
not total installed artifacts. Expanded expert-cache size and CPU weight traffic
do not shrink. This limitation prevents importing Flash-MoE's cache-capacity gain.

## Experiments

Profiling used baseline routing, without dropping or substitutions. The first
profile used 12 synthetic support-ticket prompts. The conservative profile added
the four existing JEV page-state examples. Coverage is the share of observed
routing events assigned to the preserved hot set, not guaranteed coverage of
unseen workloads or a quality target.

| Profile | Mean hot experts/layer | Logical expert payload | Change |
|---|---:|---:|---:|
| Original | 256 | 18.327 GB | - |
| 90% observed coverage | 69.25 | 12.682 GB | -30.80% |
| 99% observed coverage | 164.48 | 15.564 GB | -15.08% |

Sidecar files themselves are 7.258 GB and 3.553 GB; original hot data remains in
the source file. Their manifests are copied into the results directory.

### First arm: failed numerical quality check

The 90% arm increased token-weighted perplexity from **6.9189 to 11.2488** on 12
separate support-ticket prompt texts, approximately **+62.6%**. This is a fixed-text
stress check, not a production quality benchmark: the ordinary perplexity path
tokenizes chat delimiters as literal text. Those scores must not be presented as
correctly rendered JEV decision scores. Build work overlapped this diagnostic, so
its timing is not used for a speed claim.

### Conservative arm: real final-label scoring, binding cap, cold cache

Twelve new synthetic binary UI decisions, separate from the four profile examples.
Prompts were rendered with SemIf's `direct_messages()` and the local tokenizer,
with thinking disabled. Correct labels were specified before the candidate run.
This small set can expose regressions but cannot estimate production accuracy or
calibration. Reusing prompt *structure* is intentional; states and questions differ.

Both arms: 6 CPU compute threads, 4 I/O lanes, 3000 MiB expert cache,
context/ubatch 256, 8 GiB total cgroup cap, swap disabled, no GPU and no tracing.
Model/sidecar page residency was verified as zero before each arm. Both reached
exactly 8 GiB peak and recorded cap pressure, with zero OOM kills.

| Metric | Original weights | 99% tiered storage |
|---|---:|---:|
| Correct decisions | 11 / 12 | 10 / 12 |
| Total scoring time, excluding initial model load | 85.1977 s | 85.5333 s |
| End-to-end process time | 101.1343 s | 101.5424 s |
| Expert physical read bytes during scoring | 125.598 GB | 120.583 GB |
| Whole process-tree device reads, including load | 149.806 GB | 144.787 GB |
| Decision flips relative to baseline | - | 1 |
| Maximum conditional-probability change | - | 0.17545 |

**Reads fell 3.99%; speedup was 0.996x.** This single paired run establishes no
speed advantage; the sub-percent timing difference is not a meaningful slowdown
claim either. Since the quality gate failed, a broad speed sweep was not warranted.

The flipped item is `ui-0`: origin Lisbon, destination Madrid, goal Lisbon to
Madrid; does origin match? Baseline probability of Yes was **0.52272**; tiered was
**0.34727**, changing a correct Yes into No. Preserving experts responsible for
99% of calibration activations did not protect this near-tie decision.

The SIMD arm spent **5.113 summed lane-seconds** expanding cold slices over all
12 decisions. This is parallel worker time, not 5.113 seconds to subtract from
wall time. Expanded cache buffers and the main CPU math were unchanged.

## Correctness and integration checks

- Four test cases passed: independent codec decode/metadata checks, malformed
  lengths/codebooks, routing profile validation, and a native direct-I/O reader
  gate covering two lanes, reopening, invalid extents and wrong-source rejection.
- All **13 existing BigMoeOnEdge CTest gates passed**, including streamed-versus-
  resident and split-shard tests. The first attempt hit an installed Python `gguf`
  version mismatch; using the bundled `gguf-py` resolved that environment issue.
- For two complete prompts, the tiered reader and a separately materialized GGUF
  containing the **same lossy weights** returned exactly equal answer log-
  probabilities at 17-digit output precision. Maximum recorded difference: **0**.
  This controls the storage implementation, not quantization quality.
- The new JEV adapter scored `examples/decisions.one.jsonl`, selected the correct
  destination input, and returned its conditional probability as **0.99999056**.
  The response labels probabilities uncalibrated. The default adapter invocation
  uses no tiered pack.

## Reproduce and inspect

Instructions: [scripts/tiered/README.md](../scripts/tiered/README.md).
Evidence root: `results/tiered-20260925/`.

Key files:

- `jev/comparison-r1.json`: all 12 paired scores, predictions, times and read counts.
- `jev/base-r1.json`, `jev/tiered99-r1.json`: commands, scope peaks/events, verified
  zero residency, process-tree reads and exit status.
- `jev/reader-reference-proof.json`: same-weights reference equality.
- `pack90.manifest.json`, `pack99.manifest.json`: profile counts and full source hash.
- `adapter-smoke.jsonl` and `.run.json`: actual JEV input adapter smoke test.
- `tests.txt`, `ctest.txt`: successful verification output.

Local weight artifacts, deliberately outside git:

```
/home/scribe/models/jev-pack/jev-cold90.tier
/home/scribe/models/jev-pack/jev-cold99.tier
/home/scribe/models/jev-pack/jev-cold90-expanded.gguf
```

## Decision

Keep both lossy packs opt-in and preserve the existing default scorer and model.
The useful deliverable is a working CPU experiment with quality/read/latency
gates, including a near-tie regression that easy fixtures would have missed.

A further attempt should change the mechanism: e.g. a quantizer selected by
measured sensitivity and a CPU path that keeps mixed-precision experts compressed
inside the cache. Repeating frequency-only coverage sweeps on this storage codec
is not justified by the measured gain. Nothing here disproves native mixed-
precision execution, demonstrates a hardware ceiling, or establishes 20 tok/s.
