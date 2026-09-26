# CPU setup reuse and overlapping SSD transport

2026-09-26. Implements the first bounded steps of the
[cross-industry audit](RESUMABLE-CROSS-INDUSTRY-PERFORMANCE-AUDIT.md): clean timing,
reusable executor setup, and known-demand read/compute overlap. CPU-only,
unchanged model weights, 8 GiB/no swap. The resident service and upstream sources
are unchanged.

## Result and attribution

**Overlapping reads with computation reduces native single-layer replay time by
about 31% on both tested work shapes, with byte-identical outputs and identical
SSD read bytes.** This is a measured transport/execution improvement, not a claim
of a 31% faster complete model. The larger engine still has only one packed layer
integrated through the experimental bypass.

The setup-only changes reduce repeated operations but did **not** establish a
meaningful whole-request speedup. Keeping these findings separate preserves the
earlier scheduling result: 40.28 → 18.11 GB was a reduction in cumulative scoring
reads, not a smaller checkpoint file and not the result measured here.

## Implemented pieces

| Code | Change | Contract retained |
|---|---|---|
| [probe.cpp](../scripts/resumable/probe.cpp) | `diagnostics: false` disables route/tensor logging while retaining bypass control callbacks; stock baseline needs no callback | Validation-only check/replace requires diagnostics; unsupported combinations fail explicitly |
| [native_tiles.h](../scripts/resumable/native_tiles.h) | Share gate/up prepared inputs when native activation formats match | Same Q8_K conversion and native quantized dot kernels; no requantized weights |
| Same executor | Allocate temporary arrays once per admitted group, reuse across experts; omit unused staged-path qhidden allocation | Bounded lifetime; large scratch does not persist across calls |
| Same executor | Reuse native SwiGLU graph; rebind all shapes/strides/pointers and re-plan each invocation | Native activation, fixed tile width, complete down-projection reduction |
| [native_tile_io.h](../scripts/resumable/native_tile_io.h) | One reader worker and two aligned buffers, with producer/consumer ownership | Only known routed bundles; no predicted expert selection or skipped model work |
| Same transport | Handle EINTR; reject short reads; propagate worker errors; join on completion or failure | No use of incomplete data and no reuse of a buffer before the CPU releases it |
| Same executor | Include second buffer, descriptor capacity allowance and conservative worker allowance in local admission budget | Whole-process 8 GiB cap remains separately enforced |
| [analyze.py](../scripts/resumable/analyze.py) | Reject duplicate IDs, incomplete logit arrays and nonfinite results | Exact comparisons cannot silently accept truncated `zip` input or NaNs |
| [run_performance.py](../scripts/resumable/run_performance.py) | Alternate full-model baseline/control/candidate order; verify no diagnostic files, exact outputs and skipped-op counts | Separate clean control binary from candidate implementation |
| [native_benchmark.cpp](../scripts/resumable/native_benchmark.cpp), [run_layer_matrix.py](../scripts/resumable/run_layer_matrix.py) | Replay captured real inputs/routes under the cap, with balanced variants and output SHA-256 / equal-I/O gates | Explicit single-layer scope; captured readiness is not an online scheduler |

The transport starts loading a known upcoming bundle into the other buffer while
the CPU consumes the current one. At most two bundles are buffered. Gate/up weights
are consumed directly; staged quantized down columns retain the validated native
reduction. The CPU publishes release only after the current bundle's last use.

## Setup-only full-model comparison

Four rounds, alternating baseline/control/candidate and reverse order. Each run
starts in a fresh guarded process with pre-load cold model/pack checks. Loading
is reported separately; loading can itself warm pages before scoring.

| Variant | Median scoring seconds | Median scoring physical reads |
|---|---:|---:|
| Stock graph, diagnostics off | 35.16535 | 18.09581 GB |
| Original tile executor bypass, diagnostics off | 35.16082 | 18.08428 GB |
| Setup-reuse tile executor bypass, synchronous I/O | 35.03314 | 18.08604 GB |

All candidate/control selected logits and probabilities matched the stock graph
exactly. Every tile variant fetched exactly 445,906,944 bytes for the packed layer.
Process-wide bytes include other model activity and kernel readahead.

Candidate-minus-control paired time differences were **+0.1895, +0.0217, -0.3060,
+0.1422 seconds**. Three of four pairs were slower despite the slightly smaller
candidate median. This is **not reliable evidence of a full-model speedup**.

Within the candidate, activation-graph constructions fell from the original
code's 504 per group to one per executor lifetime; input-row quantizations fell
from 2,024 to 1,012. Native layer time medians in these full-model runs were
0.52692 seconds (control) and 0.52350 seconds (candidate). These setup costs were
not the dominant bottleneck on this workload.

[Full-model evidence](../results/resumable-performance-20260926/timing/comparison.json).
The `control-source` and `p1-source` directories in the evidence directory retain
the source variants; the control tile header's hash was checked against the
pre-edit static audit. Binary hashes are included in the comparison.

## Isolated native transport comparison

For each work shape: four alternating outer trials per variant, three repeated
executions per process. Values below are the median of the four per-trial medians.
The control uses original executor code; synchronous and overlap variants use
the same final optimized binary. Direct I/O bypasses the pack's page cache.
CPU caches, allocator state, and repeated executor setup can warm within a trial.

| Captured work shape | Original executor | Setup reuse + sync | Setup reuse + overlap | Overlap reduction vs sync |
|---|---:|---:|---:|---:|
| One equalized group, 1,012 rows | 0.52908 s | 0.51958 s | **0.35926 s** | **30.85%** |
| Three fragmented groups, 848 + 38 + 22 rows | 0.77779 s | 0.76776 s | **0.52868 s** | **31.14%** |

Overlap won every paired outer trial: approximately 26–34% on equalized work and
27–33% on fragmented work. This is a more consistent result than setup-only timing.
These are twelve internal repetitions per variant, clustered into four processes;
they are not twelve independent end-to-end model trials.

| Measurement per replay | Equalized sync | Equalized overlap | Fragmented sync | Fragmented overlap |
|---|---:|---:|---:|---:|
| Direct read bytes | 445,906,944 | 445,906,944 | 962,592,768 | 962,592,768 |
| Direct read calls | 504 | 504 | 1,088 | 1,088 |
| Median consumer I/O wait | 175.66 ms | **13.89 ms** | 378.38 ms | **141.16 ms** |
| Maximum measured cgroup memory across these trials | 156.29 MB | 156.62 MB | 141.24 MB | 144.40 MB |

Wait values are medians of per-trial cumulative wait divided by three repetitions.
Synchronous wait is measured time inside `pread`; overlap wait includes the wait
and locking needed to acquire a completed buffer. They measure exposed consumer
delay, not pure SSD latency. For equalized work, worker read-service time remained
about 181 ms versus about 176 ms synchronously: **the reads were hidden, not made
faster or eliminated**. Worker durations overlap computation and must not be
added to whole-request wall time.

Output-file SHA-256, direct bytes and read counts matched across every variant
and outer trial. All guarded replays completed without OOM or swap.

[Equalized evidence](../results/resumable-performance-20260926/layer-padded/comparison.json),
[fragmented evidence](../results/resumable-performance-20260926/layer-fragmented/comparison.json).

## Validation

The final native suite passes **27 checks** across Q4_K, Q5_K and Q6_K, including
grouped/serial execution, repeated shrinking/growing activation shapes, asynchronous
output equality, repeated pipeline construction, short-read error propagation,
graph-operation restoration and unsupported-input rejection. Five Python tests
pass, including stronger comparison validation.

The full integration test combined non-flash attention, an eight-question
shared-prefix checkpoint, two successive requests in one context, and overlap
bypass in both prefix/suffix phases. All selected logits and probabilities matched
the stock reference exactly. Four expert groups ran through the replacement;
each of the original gate/up/activation/down operations was skipped four times,
and all graph metadata was restored. Interior original-weight pages totaling
452,972,544 bytes were protected against access (boundary pages remain excluded).

Across those four groups the executor used one activation-graph construction,
1,540 bundle reads, 1,362,493,440 direct bytes and 71,793,920 bytes of peak accounted
local work, including the pipeline allowance. The checkpoint size was 67,053,944
bytes. Final selected-output agreement is a numerical fidelity result; it does
not improve the baseline's task accuracy.

This one integration comparison took 56.95 seconds for stock, 57.91 seconds for
the original synchronous tile control, and 55.43 seconds for the overlap candidate,
across two successive requests per process. It validates combined operation; its
single ordered trial does not establish a full-model speedup.

All **39 guarded runs** passed with an 8 GiB/no-swap configuration and no OOM.
Python syntax, report links and whitespace checks passed.
[Combined evidence](../results/resumable-performance-20260926/checkpoint-overlap/comparison.json),
[validation and final source/binary hashes](../results/resumable-performance-20260926/validation.json).

## How to select the experiment

In a frozen probe input:

```json
{
  "diagnostics": false,
  "tile_io": "overlap"
}
```

Use this with the existing `PACK bypass staged` arguments. Synchronous I/O remains
the default; no resident-service configuration was changed. `flash_attention`
is an independent diagnostic control and must not be switched silently while
comparing execution policies.

This completes a bounded implementation of the audit's timing/setup/transport
steps. Cross-call cooperative scheduling, a persistent expert cache, all-layer
replacement and selective vocabulary projection remain separate integration work.
