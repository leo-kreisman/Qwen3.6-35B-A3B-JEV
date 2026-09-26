# Removing duplicate expert computation: bounded CPU bypass

2026-09-25. Follow-up to the [native tile validation](RESUMABLE-NATIVE-TILES-RESULT.md).
The previous stage was committed and pushed as `f120adc` before this work began.

## What changed

The standalone probe now has an explicit `bypass` action. At the packed expert
layer it suppresses the original gate, up, SwiGLU and down operations, groups the
currently ready rows by expert, consumes the lossless quantized SSD tiles, and
writes the resulting expert outputs before router weighting and subsequent model
layers run. Thus this path no longer computes both the original expert and its
tile replacement.

The existing `check` and strict `replace` actions retain their original validation
roles. Bypass cannot compare its output with an original calculation it did not
execute; it is evaluated against separately captured baseline outputs.

## The execution boundary is experimental

The existing eval callback officially observes operations; it is not a supported
skip-operation API. [native_bypass.h](../scripts/resumable/native_bypass.h) therefore
implements **version-sensitive graph interception**: temporarily set the target
node to `GGML_OP_NONE` before its CPU graph segment executes, synchronize at the
callback boundary, then restore the original operation. The CPU dispatcher skips
`NONE` nodes. Downstream nodes receive the native tile result.

This changes graph metadata in the standalone process. It does not patch llama.cpp
source, update the vendor submodule, change the streamer, or enable a service
default. It is restricted to the tested CPU graph with plain split SwiGLU experts;
expert scales/biases, differing model paths, mismatched weight types/shapes and
unexpected dependencies are rejected. It requires staged down reduction and the
native activation implementation established by the prior exact-output tests.

This is evidence for an executable replacement boundary, not a supported general
engine extension. A durable backend integration still needs an explicit supported
operation-replacement interface. Revalidate against the recorded library hashes
before considering a different engine build or concurrent serving.

## Correctness and proof that weights are bypassed

The first attempted run correctly failed when router IDs appeared as a strided
view rather than a contiguous array. The corrected reader respects tensor byte
strides. The failed run is retained as evidence and did not produce accepted
answers.

The next eight-question run skipped one gate/up/activation/down group at layer
zero and matched every reference selected logit exactly.

A stronger Linux-only diagnostic makes the **whole interior pages of the original
expert-weight tensors unreadable** with `mprotect(PROT_NONE)`. The original mapped
weights remain inaccessible across repeated graph calls. The separate tile file
is read with direct I/O. The protected original region totals **452,972,544 bytes**;
boundary pages are excluded to avoid affecting adjacent tensors. A CPU operation
reading those protected pages would fault. Protection is removed before model
teardown.

With this diagnostic enabled:

* Three executions in the same context completed, each matching the independent
  reference's selected logits exactly. Three of each original operation were
  suppressed, and graph operations were restored.
* The non-flash shared-prefix/eight-branch execution completed with two suppressed
  expert groups (prefix and suffix). All eight selected-logit vectors matched the
  non-flash full-prompt reference exactly.

Both tests ran under MemoryMax=8G and MemorySwapMax=0 without OOM. Only layer zero
is replaced. [Correctness evidence](../results/resumable-bypass-20260925/correctness.json).

The memory-access diagnostic is opt-in through `protect_bypassed_weights` in a
frozen input. `repetitions` requests up to eight evaluations in one context. These
are probe controls, not model or service defaults.

## Reproduce and assess performance

[run_bypass.py](../scripts/resumable/run_bypass.py) runs a cold, capped baseline
followed by bypass on the same frozen input. It requires exact selected logits,
matching operation counts for every captured expert group, restored operators,
and protected weight pages when requested. Its report retains timing/read data
but explicitly labels a single ordered pair as insufficient for a speed claim.

```bash
mkdir -p results/resumable-bypass-local
bash scripts/resumable/build.sh results/resumable-bypass-local/probe
python scripts/resumable/run_bypass.py \
  --model /path/to/model.gguf --input /path/to/frozen-input.json \
  --pack /path/to/verified-pack --probe results/resumable-bypass-local/probe \
  --output results/resumable-bypass-local/pair --mode full --protect-weights
```

The metadata model path must refer to the pack's original source model. The pack
was byte-verified in the preceding stage; path/type/shape checks are not a new
cryptographic verification of the entire checkpoint on every request.

## Fragmented-batch comparison

The automated pair on the original four-question fixture passed. Baseline tracing
identified three layer-zero expert groups; bypass suppressed exactly three of
each gate/up/activation/down operation, restored their metadata, and retained the
452,972,544-byte protection guard. All four selected-logit vectors matched exactly.

| Measurement | Baseline | Bypass |
| --- | ---: | ---: |
| Scoring seconds, one ordered pair | 47.254 | 47.083 |
| Process scoring reads, decimal GB | 40.121 | 39.929 |

This is **not a demonstrated speedup**. The change replaces only one of 40 layers,
and a single ordered pair cannot resolve such a small timing difference. The
result establishes replacement correctness and skipped computation, while leaving
the earlier batching/reuse improvements intact. [Pair evidence](../results/resumable-bypass-20260925/paired/comparison.json).

The final native suite passes 18 checks, including operator restoration and
rejection of unsupported graph operations, plus the four Python tests. Five
successful capped model runs and the initial explicit stride rejection are
retained. No run recorded OOM. [Validation](../results/resumable-bypass-20260925/validation.json)
and [provenance](../results/resumable-bypass-20260925/provenance.json).

## Remaining scope

This removes duplicate computation for one packed layer in the experimental
execution path. It does not yet replace all model layers, provide an upstream
supported backend, or coordinate ready rows across independent graph calls.
Checkpoint consistency still uses the separately tested non-flash path, and
matching a baseline does not fix that baseline's incorrect decisions.
