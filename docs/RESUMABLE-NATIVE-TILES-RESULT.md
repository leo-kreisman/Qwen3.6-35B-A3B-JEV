# Native executable tiles and segmented-evaluation diagnosis

2026-09-25. Continues the [first measurements](RESUMABLE-DECISION-RESULT.md)
without changing the CPU/SSD/8 GiB objective, model weights, or service defaults.

## What was added

`NativeTiles` in [native_tiles.h](../scripts/resumable/native_tiles.h) consumes the
existing lossless gate/up/down bundles directly with GGML CPU quantized dot
kernels. It accepts currently ready input rows and actual expert IDs, groups the
rows by expert, reads each required expert's bundles once, and scatters outputs
back to their original row/route positions. It does not assume future routes or
expand weights into float32 matrices.

The executor has an explicit 256 MiB work-admission budget. Its accounting covers
input/route arrays, output, quantized activations, ready-row indices, activation
scratch and direct-I/O buffers. This local budget is separate from the complete
model's 8 GiB cgroup cap. Shapes above the bounded prototype's admission limits
are rejected rather than silently increasing the working set.

The public evaluation callback now supports checking the executor against an
actual layer output and substituting its result before downstream model work.
**This is a validation hook:** the original layer still executes first. It is not
a speed-optimized replacement backend, and total hook-run timing cannot establish
a service speedup. Only layer zero is packed and integrated in these runs.

No llama.cpp source or BigMoeOnEdge streamer code was changed.

## A numerical failure was refined into an exact result

The first native executor used scalar SiLU and summed independent down-projection
partial products. Its layer output differed by only 5.22e-8 at maximum, but feeding
that output downstream changed one of eight challenge decisions. Staging the full
quantized down projection while retaining scalar SiLU still changed the decision.
These failures concern numerical execution, not lossless storage: the original
pack's byte-identity verification remains valid.

The successful executor makes two precise adjustments:

1. It calls the native GGML SIMD SwiGLU activation through a small public-API graph.
2. It stages the quantized down columns and complete intermediate activation for
   the active expert, then uses the original down-projection reduction length.

Gate/up computation still consumes tiles as they arrive. Down weights remain
quantized: staging costs 589,824 bytes for this layer's active expert, rather than
expanding the expert into 12 MiB of float32 weights. This is a deliberate tradeoff
between immediate partial accumulation and preserving numerical behavior.

On the eight-question challenge batch:

| Measurement | Native activation + staged down |
| --- | ---: |
| Ready token rows | 872 |
| Routed row/expert pairs | 6,976 |
| Required experts | 225 |
| Direct reads | 450 |
| Direct bytes | 398,131,200 |
| Float32-expanded weight bytes | 0 |
| Peak accounted executor work | 112,771,840 bytes |
| Executor time, one observation | 0.508 seconds |
| Maximum layer error against native output | **0** |
| Final selected-logit difference | **0** |
| Final decision changes | **0 / 8** |

The final validation hook refuses substitution unless every output value matches
numerically exactly. It also rejects nonfinite candidate outputs. Exact equality
on these fixtures is a local acceptance test, not a universal model guarantee.
The original baseline must still pass its own decision-quality checks.

Evidence is under `results/resumable-native-20260925/`: `live-check`,
`live-replace` (initial failed partial reduction), `live-staged` (failed scalar
activation), and `live-native-activation` (successful native activation). All
runs use unchanged model weights. Intermediate experimental binaries are local
rebuild/debug artifacts; the maintained implementation is the final strict hook.

The final strict hook was additionally tested on the original four-question
fixture in both execution shapes. Both completed with zero layer-output and
final selected-logit differences:

| Ready-work shape | Executor direct bytes | Expert visits | Bundle reads | Executor seconds |
| --- | ---: | ---: | ---: | ---: |
| Three fragmented groups | 962,592,768 | 544 | 1,088 | 0.764 |
| One equalized group | 445,906,944 | 252 | 504 | 0.554 |

These executor times are single observations within validation runs, not repeated
end-to-end speed results. The direct-read reduction is 53.7%, despite the equalized
batch containing more token positions. The current executor reuses across rows
within each ready group; it cannot retain unfinished work across the three
separate graph traversals. [Exact output comparisons](../results/resumable-native-20260925/strict-comparison.json).

## Save/restore is not the cause of the reproduced split discrepancy

The earlier problematic question was isolated as one sequence, with the same
58-token split boundary:

| Execution | Yes probability |
| --- | ---: |
| One uninterrupted prefill | 0.5227237458770636 |
| Prefix then suffix, without serialization | 0.1358466528413265 |
| Prefix, serialize/clear/restore, then suffix | 0.1358466528413265 |

The latter two have identical selected logits. Thus, for this reproduction,
serialization preserves the state used by segmented evaluation; the discrepancy
already exists without serialization. This preserves the usefulness of the
checkpoint mechanism while identifying a separate segmented-evaluation problem.

Capturing normalized expert inputs across the whole model found exact agreement
at layers 0, 1 and 2. The first differences appear at layer 3, the first full
attention layer. Even prefix token 1 differs, before the split boundary: maximum
prefix difference there is 0.0090056, and suffix difference is 0.0126784. This rules
out interpreting the symptom solely as bad restored suffix state. Detailed attention captures show matching query/key/value projections and
normalized query/key prefixes, followed by a 0.0123689 maximum difference in
`attn_pregate-3`. Automatic CPU Flash Attention was enabled. Thus attention
computation is the first observed divergence; the underlying fused-kernel cause
is not yet established. Use `attention-comparison-indexed.json`, which matches
repeated tensor names by occurrence, rather than the preliminary name-only file.

## Controlled attention workaround

An explicit test input with `flash_attention: disabled` retained the model,
weights, CPU threads and memory limit while selecting the non-flash attention
path. On the isolated question, uninterrupted, split and split-plus-restore
execution then had **identical selected logits**, with yes probability
0.27762711057082945. [Isolation comparison](../results/resumable-native-20260925/checkpoint-isolation.json).

This supports CPU Flash Attention as the source of the reproduced segmentation
sensitivity. It does not identify the exact instruction or kernel defect. No
upstream kernel was patched and no live service setting was changed. The expected
answer on this fixture is yes, so the consistent non-flash answer is still wrong:
execution consistency and task accuracy must remain separate gates.

The eight-question non-flash test also produced **identical selected logits and
probabilities** for full equalized prompts and shared-prefix equalized suffixes.
The former took 43.20 seconds and the latter 104.32 seconds in single untraced
observations, with 16.74 GB and 28.12 GB scoring reads respectively. These are not
repeated timing estimates and do not establish the isolated cost of disabling
Flash Attention. They do establish that consistency alone is insufficient to
select a faster execution policy. All eight answers agree; seven match the
fixture labels. [Branch comparison](../results/resumable-native-20260925/branch-comparison.json).

## Combined check passed

The final run combined non-flash attention, prefix serialization/restoration into
eight branches, and strict native tile substitution at layer zero in both prefix
and suffix passes. Both substituted layer outputs matched exactly; all eight
final selected-logit vectors matched the non-flash full-prompt reference exactly.
It completed within 8 GiB with no swap or OOM.

The executor consumed 681,246,720 direct bytes in 770 bundle reads, with 54,107,392
bytes of peak accounted work and no expanded float32 weights. The serialized
prefix was 67,053,944 bytes; snapshot/branch work took 0.118 seconds. This proves
the components work together on this fixture. It does not prove a service speedup:
original expert computation still runs in the validation hook.

This traced combined run took 35.93 seconds for scoring, versus 104.32 in the
untraced shared diagnostic. That large difference reinforces why these diagnostic
runs must not be presented as a controlled latency estimate or attributed solely
to one implementation change. [Combined evidence](../results/resumable-native-20260925/combined-comparison.json).

## Validation and limits

The native tests exercise Q4_K, Q5_K and Q6_K synthetic packs, ready-row grouping
versus separate requests, comparison to an independent native whole-expert graph,
invalid expert rejection and missing-tile rejection. Twelve checks pass. Existing
Python tile/I/O/cache tests remain relevant and are rerun separately.

The next execution integration must avoid calculating the original expert output
before the tile consumer. The current hook intentionally cannot claim that saving.
Likewise, it does not yet coordinate ready work across independent graph calls:
it consumes all rows ready at its current graph frontier. Those are explicit
remaining integration tasks, not evidence against the working transport format.

All **19 guarded runs** completed successfully with an 8 GiB cap and no OOM.
The 12 native checks and four Python tests pass; syntax and whitespace checks pass.
See [validation](../results/resumable-native-20260925/validation.json),
[guard audit](../results/resumable-native-20260925/guard-validation.json),
[code/binary provenance](../results/resumable-native-20260925/provenance.json),
and the [execution contract](../results/resumable-native-20260925/execution-contract.json).
