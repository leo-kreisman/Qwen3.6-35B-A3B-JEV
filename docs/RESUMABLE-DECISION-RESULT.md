# Resumable CPU decisions: first implementation and measurements

2026-09-25. Implements the first experiments from
[the fixed architecture research](RESUMABLE-DECISION-ENGINE-RESEARCH.md).
Scope: CPU execution, an 8 GiB cgroup limit with swap disabled, SSD-resident
weights, unchanged model precision, and useful decision work per weight transfer.

## What now exists

[Reproducible experiment tools](../scripts/resumable/README.md) implement native
prefix save/restore, actual expert-route tracing, schedule comparisons, a
memory-budgeted trace simulator, and lossless executable expert tiles. These are
standalone experiments; the production streamer and service defaults are unchanged.

The native probe uses the existing CPU llama/ggml libraries through public APIs
and a memory-mapped GGUF. This measures complete decision forward passes, not the
resident BigMoeOnEdge HTTP path. The model is the existing
`Qwen3.6-35B-A3B-JEV-Q5K-nonexpert.gguf`; its digest, library revisions and input
hashes are recorded in the evidence directory. The filename is not a claim that
this experiment used a newly downloaded model.

## A concrete source of repeated work

Four existing flight-form questions have 908 prompt-token positions, including a
common 141-token prefix. Giving all four prompts to one large native batch still
produced **three internal model traversals**: layer-zero groups of 848, 38 and
22 rows, repeated across 40 expert layers. Increasing the microbatch capacity
alone did not remove this fragmentation: it arose from unequal recurrent sequence
lengths.

An experimental schedule appends EOS tokens **after each question's requested
answer readout**, making lengths equal while preserving original answer indices.
It processes 1,012 positions, but the trace shows one traversal rather than three.
This exchanges some extra arithmetic for substantially fewer repeated weight reads.
It is an execution-shape experiment, not prompt truncation or weight compression.

Padding is not a general resumable-state solution: the final recurrent state has
advanced past the answer position. Continued generation would need a checkpoint
at the real boundary. Changed kernel shapes also change numerical results.

## Three cold runs per schedule

Six CPU threads; no GPU offload; MemoryMax=8G; MemorySwapMax=0. Each run used a new
process and verified that no model pages were cached initially. Modes were run
sequentially with alternating order. Times cover scoring, including shared-state
work where applicable, and exclude model loading. Reads are process physical-read
counters, including page readahead; GB below is decimal.

| Schedule | Median scoring seconds | Median read GB | Scoring seconds in three runs |
| --- | ---: | ---: | --- |
| Full prompts, unequal lengths | 58.75 | 40.28 | 48.03, 58.75, 59.98 |
| Full prompts, equalized lengths | 41.68 | 18.11 | 39.01, 41.68, 46.58 |
| Shared prefix, equalized suffixes | 43.52 | 32.51 | 35.07, 43.52, 43.65 |

Full-prompt equalization reduced median scoring latency **29.1%** (1.41× throughput
for this fixed workload) and reads **55.0%**. All runs completed without OOM under
the cap. Timing variance is substantial; three runs are not a statistical claim
about arbitrary workloads. Tracing was disabled during these timing runs.

The four selected answers agreed in every run. Equalization changed conditional
option probabilities by at most 0.00001235, but selected raw logits changed by up
to 0.875. Saturated probabilities hide differences; this is not bit-exact execution
or a demonstrated general accuracy guarantee. A separate decision check is
reported below.

Evidence: [timing comparison](../results/resumable-20260925/comparison.json),
individual results and guards under `results/resumable-20260925/timing/`.

## Decision checks exposed execution sensitivity

Eight constructed flight-form challenge questions include a previously observed
near-tie. These are diagnostic fixtures, not an independent benchmark or
calibration dataset, despite the `holdout` artifact names. Each mode was tested
once; these runs were intended for correctness comparison, not another robust
performance estimate.

For “Does the origin match the goal?” with origin Lisbon and goal Lisbon to
Madrid, the expected answer is yes:

| Execution | Conditional probability of yes | Correct questions out of eight |
| --- | ---: | ---: |
| Full unequal-length batch | 0.433988 | 7 |
| Full equalized batch | 0.522724 | 8 |
| Independent full prompts, serial | 0.522724 | 8 |
| Shared prefix, equalized suffixes | 0.135847 | 7 |
| Shared prefix, unequal-length suffixes | 0.135847 | 7 |

**The equalized batch and serial reference have exactly the same printed selected
logits and probabilities on all eight questions.** The unequal-length batch
changes one decision relative to serial, and the shared-prefix schedule also
changes that decision. Thus a difference from the old batch does not by itself
mean a regression: the serial reference matters. Conversely, eight fixtures do
not establish that the equalized schedule is universally correct.

The unpadded and padded shared-prefix runs also have identical selected logits
on all eight questions. Removing suffix padding therefore does not resolve this
shared-prefix discrepancy. It is material. The precise cause—kernel shape,
prefill segmentation, or state handling—has not been isolated, and it should not
be dismissed as harmless floating-point rounding. Successful serialization and
restoration does not establish semantic equivalence. No shared execution mode
has been promoted to a service default.

Evidence: [challenge comparisons](../results/resumable-20260925/holdout-comparison.json)
and [quality versus serial](../results/resumable-20260925/decision-quality.json),
with frozen prompts/expected answers in `holdout.jsonl`.

## Prefix state can actually be resumed

The public sequence serialization API saved a 68,755,760-byte prefix snapshot and
restored independent branches. It avoided the previously unsuccessful sequence
copy path. Snapshot/restore for four branches took 0.080 seconds in the traced run.
The unpadded shared run matched the full traced run's printed selected logits
on the first four fixtures. That agreement is limited to those fixtures; it does
not establish numerical equivalence for other prefix/suffix shapes.

Prefix sharing reduces evaluated positions from 908 to 485 (589 with equalized
suffixes). Nevertheless it requires a prefix traversal followed by a suffix
traversal. Its median reads exceeded the one-traversal full-padded schedule, and
it was slightly slower. **Eliminating repeated arithmetic and eliminating weight
transfers are different objectives.** Shared evidence is useful state machinery,
but these measurements do not justify enabling it unconditionally.

## Executable transport tiles are implemented

For all 256 experts in layer zero, each SSD tile contains matching gate rows, up
rows and down columns for 256 intermediate coordinates. The CPU retains input
vectors and an output accumulator while consuming the two tiles per expert.
Original quantization blocks are rearranged without requantization. Bundles are
4 KiB aligned and directly readable with O_DIRECT.

The pack is 452,984,832 bytes, exactly the original layer payload size. Every
original projection was reconstructed **byte-for-byte** (256 experts × three
projections). This establishes a lossless transport representation, not a smaller
model. [Verification](../results/resumable-20260925/pack-verified.json).

A separate single-layer replay used captured real inputs/routes from the three
internal groups. It compared consuming each group separately with grouping ready
rows by expert. All four arms used direct reads, six BLAS threads and the same
8 GiB/no-swap guard.

| Ready-work schedule / representation | Seconds | Direct read MB | Read calls | Peak decoded weights |
| --- | ---: | ---: | ---: | ---: |
| Separate / original projections | 2.689 | 969.28 | 1,632 | 12 MiB |
| Grouped / original projections | 1.441 | 449.00 | 756 | 12 MiB |
| Separate / executable tiles | 2.949 | 962.59 | 1,088 | 6 MiB |
| Grouped / executable tiles | 0.985 | 445.91 | 504 | 6 MiB |

These are **single observations of an isolated layer**, not repeated full-model
benchmarks. Tiling alone was slower in this sample. Grouping removed repeated
expert reads; paired tiles halved the decoded-weight working set. The grouped
tile arm's cgroup peak was 226,566,144 bytes; the 6 MiB figure counts only decoded
weights, not inputs, accumulators, Python or I/O buffers.

Maximum absolute output difference from the separate original-weight float32
reference was 8.95e-8; relative L2 error was at most 1.43e-7. All outputs were finite.
The original and tiled paths use GGML weight dequantization plus float32 BLAS;
they do not reproduce GGML's quantized-activation kernels. Outputs are expert
contributions before router-weight merging. This is not full-model quality
validation. [Layer comparisons](../results/resumable-20260925/layer-comparison.json).

## Scheduling opportunity and the RAM ledger

The captured route-order replay demands 26.44 GB of expert payload. Deduplicating
ready experts at each layer frontier reduces it to 13.59 GB, a 48.6% opportunity.
In this trace, LRU caches through 4 GiB cannot retain useful experts between the
three complete traversals. This illustrates why coordinating unfinished work can
matter more than replacing the eviction policy.

The corrected ledger includes actual allocated attention/recurrent context
(347,340,800 bytes), a temporary serialized prefix (68,755,760 bytes), dense
weights, activations, graph scratch, I/O buffers and explicit runtime/margin
reserves. With 4 GiB for experts, total planned use is 8,146,774,832 bytes.

This is an offline scheduling estimate with observed future routes, not a working
online scheduler or proof of actual allocator usage. Prefix/suffix phase fences
are retained. Payload bytes exclude dense rereads and OS read amplification.
Use [simulation-budgeted.json](../results/resumable-20260925/simulation-budgeted.json)
as the current result; earlier simulation files preserve preliminary accounting.

## What remains before live integration

The bounded prototype establishes actual resumption, lossless executable blocks,
and a measurable repeated-traversal problem within the RAM constraint. It does
not implement all 48 research components or an online layer-frontier scheduler.
The next integration work is to expose ready layer work explicitly, budget its
live state, and consume original quantized tiles with native CPU kernels. This
would avoid using padding as a permanent scheduling mechanism and remove the
float32 replay's temporary expansion. It needs decision-quality checks as well
as repeated cold and warm service-path benchmarks.

## Validation and provenance

Four focused unit tests pass: quantization-block round trips, rejecting unaligned
tile cuts, direct-I/O byte fidelity/buffering, and byte-weighted LRU behavior.
Python syntax and tracked whitespace checks pass. Native execution, complete
layer reconstruction and numerical replays provide integration evidence beyond
the unit tests.

See [validation](../results/resumable-20260925/validation.json),
[final code/library hashes](../results/resumable-20260925/provenance-final.json),
and [hardware/cap settings](../results/resumable-20260925/hardware.json).
Large binaries, activation captures and the generated weight pack are ignored
by Git; their manifests, hashes, measurements and regeneration tools are retained.
