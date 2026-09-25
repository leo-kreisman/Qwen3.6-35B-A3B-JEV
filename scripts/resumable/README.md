# Resumable CPU decision experiments

This is a bounded implementation of the first experiments in
[the architecture research](../../docs/RESUMABLE-DECISION-ENGINE-RESEARCH.md).
It uses CPU-only execution and unchanged model weights. It does not change the
resident service, SemIf defaults, BigMoeOnEdge, or llama.cpp.

## What is implemented

* `prepare.py`: exact SemIf prompts, verified one-token answer slots, reference
  tokenizer IDs, and a token-level common prefix. No prompt truncation.
* `probe.cpp`: standalone llama.cpp public-API scoring with five schedules:
  `full`, `serial`, `shared`, `full-padded`, and `shared-padded`.
  Shared modes serialize one sequence's entire prefix state and restore it into
  independent sequence IDs. They do not call the previously failing `seq_cp`.
* Padded modes append EOS tokens **after each requested readout position** to
  equalize branch lengths, keeping the original answer indices. This can avoid
  recurrent microbatch fragmentation. Causality preserves the intended question,
  but changed kernel shapes can alter floating-point outputs; test near ties.
* Optional graph callbacks record real routed expert IDs and normalized inputs
  for one selected layer. Trace timings are diagnostic, not benchmark evidence.
* `simulate.py`: byte-weighted LRU replay and an offline layer-frontier scheduling
  opportunity estimate, with explicit state/activation/I/O/reserve accounting.
  This is not a latency prediction or an implemented arbitrary graph scheduler.
* `tiles.py`: lossless quantization-block-aligned expert bundles. Each contains
  gate/up rows and corresponding down columns. Replay keeps input rows and output
  accumulators alive while consuming tiles, with separate or grouped ready rows.
  `verify` reconstructs and compares every packed expert projection byte-for-byte.
* `dequant.cpp`: calls GGML's native weight decoding for the layer replay.
  Replay math uses float32 NumPy/BLAS, not GGML's quantized activation kernels.
  Numerical equivalence here is to a dequantized original-weight reference;
  it is not a full-model or native-kernel equivalence claim.
* `run_matrix.py`: sequential alternating-order cold A/B comparisons under
  systemd MemoryMax=8G and MemorySwapMax=0, with the existing guarded runner.
* `analyze.py`: medians, measured reads, cap checks, decision flips, probability
  and logit differences. Model quality is separate from agreement with a baseline.

## Reproduce

Run from the repository root. Use a fresh output directory; measurements are
create-only. The native build uses the already-built CPU libraries in
`vendor/BigMoeOnEdge/build/bin` and their matching headers. No upstream patch is
needed. The SemIf environment supplies NumPy and the reference tokenizer.

```bash
mkdir -p results/resumable-local
bash scripts/resumable/build.sh results/resumable-local/probe
semif/.venv/bin/python scripts/resumable/prepare.py \
  --input examples/decisions.sample.jsonl \
  --tokenizer /path/to/reference-tokenizer \
  --output results/resumable-local/input.json
python scripts/resumable/run_matrix.py \
  --model /path/to/model.gguf \
  --input results/resumable-local/input.json \
  --probe results/resumable-local/probe \
  --output results/resumable-local/timing --rounds 3
python scripts/resumable/analyze.py results/resumable-local/timing \
  --output results/resumable-local/comparison.json
```

Capture actual layer-zero routes/inputs by adding `0 6` after the schedule when
invoking the probe through `scripts/tiered/run_guarded.py`. The last arguments
mean capture layer zero and use six CPU threads. Omitting both disables tracing
and still uses six threads. A capture layer of -1 records routes without inputs.

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- \
  python scripts/tiered/run_guarded.py \
  --output results/resumable-local/full-trace.guard.json \
  --evict /path/to/model.gguf -- \
  results/resumable-local/probe /path/to/model.gguf \
  results/resumable-local/input.json results/resumable-local/full-trace full 0 6
```

The tile intermediate width must divide the model's expert width and align to
the original quantization block (256 values for supported K quants). This first
prototype supports split gate/up/down Q4_K/Q5_K/Q6_K tensors. It fails unsupported
geometry rather than silently requantizing. A packed layer is an extra research
artifact, not a replacement for the full model.

```bash
src=vendor/BigMoeOnEdge/third_party/llama.cpp
libs=$(realpath vendor/BigMoeOnEdge/build/bin)
g++ -shared -fPIC -O2 scripts/resumable/dequant.cpp \
  -I"$src/ggml/include" -L"$libs" -Wl,-rpath,"$libs" -lggml-base \
  -o results/resumable-local/dequant.so
semif/.venv/bin/python scripts/resumable/tiles.py build \
  --model /path/to/model.gguf --layer 0 --tile 256 \
  --output results/resumable-local/pack
semif/.venv/bin/python scripts/resumable/tiles.py verify \
  --model /path/to/model.gguf --pack results/resumable-local/pack \
  --output results/resumable-local/pack-verified.json
```

Replay through the guarded runner, cold-evicting both the original model and the
pack, with `OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6`. Compare all four combinations
of `--schedule separate|grouped` and `--representation original|tiles`:

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 \
  semif/.venv/bin/python scripts/resumable/tiles.py replay \
  --model /path/to/model.gguf --pack results/resumable-local/pack \
  --trace results/resumable-local/full-trace/routes.jsonl --phase full \
  --decoder results/resumable-local/dequant.so \
  --schedule grouped --representation tiles \
  --output results/resumable-local/grouped-tiles.json
```

Do not run competing benchmarks or pack builds while timing inference. Full
model cold eviction can fail when another process retains pages; the guard
refuses to label that run cold. A cgroup cap does not count the entire host OS.

## Interpretation limits

One route record represents a real expert-up graph node; repeated records at the
same layer expose internal model sweeps. The callback does not label every
physical SSD read with its expert. Process read counters include kernel read
amplification. The simulator counts demanded expert payload and cannot be equated
to those counters. It retains phase fences and reports its reserve assumptions.

The layer replay uses actual model inputs and routing, but all rows have already
been captured: readiness is provided by the offline experiment. Making them ready
online under a memory cap is a separate graph-scheduling task. Its speed must not
be multiplied into a full-model prediction. The native padded schedules are real
end-to-end forward passes and have their own independent measurements.

```bash
semif/.venv/bin/python -m unittest discover -s scripts/resumable -p 'test_*.py'
```

Measured outcomes and limitations are in
[the result report](../../docs/RESUMABLE-DECISION-RESULT.md).

For the scheduling simulation, supply allocated context memory from the native
engine log, separately from the serialized snapshot size:

```bash
semif/.venv/bin/python scripts/resumable/simulate.py \
  --model /path/to/model.gguf \
  --trace results/resumable-local/full-trace/routes.jsonl \
  --context-bytes 347340800 --snapshot-bytes 68755760 --branches 4 \
  --output results/resumable-local/simulation.json
```

Those byte counts belong to the recorded four-question fixture; measure them
again for different context/branch geometry. Padded execution leaves state past
the requested readout and is not directly reusable for answer continuation.

## Native ready-row executor and live validation

[Native integration results](../../docs/RESUMABLE-NATIVE-TILES-RESULT.md) describe
`native_tiles.h`. The maintained path keeps weights quantized, uses native SIMD
SwiGLU, and stages the active expert's quantized down columns to preserve the
native reduction. `evaluate(inputs, routes, rows, top_k)` consumes only currently
ready work; its local admission budget is 256 MiB. It groups ready rows by expert
and returns expert outputs in original row/route order.

The extended probe accepts:

```text
probe MODEL INPUT NEW_OUTPUT MODE [capture_layer] [threads] [PACK check|replace staged|tiled]
```

Use `staged` (the default). `tiled` retains the partial-reduction experiment for
shadow comparison; its differing output cannot pass the strict replacement gate.
`check` compares without substitution. `replace` substitutes only exactly equal
finite output. This public callback runs **after** the ordinary expert computation:
it validates downstream integration and does not bypass that original work.

Example, using a fresh directory and the previously generated layer-zero pack:

```bash
bash scripts/resumable/build.sh results/resumable-local/native-probe
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- \
  python scripts/tiered/run_guarded.py \
  --output results/resumable-local/native.guard.json \
  --evict /path/to/model.gguf \
  --evict results/resumable-local/pack/weights.bin -- \
  results/resumable-local/native-probe /path/to/model.gguf \
  results/resumable-local/input.json results/resumable-local/native \
  full-padded -1 6 results/resumable-local/pack replace staged
```

The live hook is currently restricted to the tested `qwen35moe` architecture.
The pack must correspond to that model and layer; an output mismatch fails the
strict hook. This is not a service configuration change.

`split-serial` evaluates each sequence in two calls without state serialization.
`restore-serial` inserts serialize/clear/restore at the same boundary. Comparing
these modes isolates checkpoint behavior from prefill segmentation. Capture layer
`-2` records normalized expert inputs at every layer; `-3` records named float32
intermediates at layer 3 for the diagnosed attention divergence. Repeated tensor
names require occurrence matching; `compare_tensors.py` handles that.

A frozen experiment input may explicitly set `"flash_attention": "disabled"`.
Otherwise the probe retains `auto`. This controls the diagnostic process only.

Build and run the native checks in addition to the Python suite:

```bash
src=vendor/BigMoeOnEdge/third_party/llama.cpp
libs=$(realpath vendor/BigMoeOnEdge/build/bin)
g++ -std=c++17 -O2 -fopenmp -Wall -Wextra \
  scripts/resumable/native_tests.cpp \
  -I"$src/ggml/include" -I"$src/vendor" \
  -L"$libs" -Wl,-rpath,"$libs" -lggml-cpu -lggml -lggml-base \
  -o results/resumable-local/native-tests
results/resumable-local/native-tests results/resumable-local/new-test-fixtures
```
