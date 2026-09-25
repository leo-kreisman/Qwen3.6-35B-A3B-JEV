# Profile-guided tiered expert storage (experimental)

The measured configurations did **not** earn deployment: the conservative arm
saved 4% of expert reads, took the same time, and flipped a correct UI decision.
See [the result](../../docs/TIERED-STORAGE-RESULT.md). Original weights and the
existing resident HTTP scorer remain the default.

This adapts the *idea* of Flash-MoE's hot/cold precision tiers to the existing CPU
engine. It does not load the linked Apple/Metal checkpoint. Cold Q4_K payloads
are reduced to two-bit codebook indices, then expanded into native Q4_K buffers
on read. Metadata and hot experts stay unchanged. This saves transfer bytes but
does not increase expert-cache capacity or reduce CPU weight traffic. The sidecar
supplements the original GGUF, so it is not a disk-space-saving distribution.

## Build

The checked-out `vendor/BigMoeOnEdge` already has the experimental patch. For a
fresh checkout, apply `patch/tiered-bmoe/changes.patch` at the base commit recorded
in its `BASE_COMMIT`, then build:

```bash
git -C vendor/BigMoeOnEdge apply --check "$PWD/patch/tiered-bmoe/changes.patch"
git -C vendor/BigMoeOnEdge apply "$PWD/patch/tiered-bmoe/changes.patch"
cmake -S vendor/BigMoeOnEdge -B vendor/BigMoeOnEdge/build -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/BigMoeOnEdge/build -j 4
mkdir -p results/my-tiered-experiment
g++ -O3 -std=c++17 -shared -fPIC scripts/tiered/codec.cpp \
  -o results/my-tiered-experiment/codec.so
python3 -m unittest discover -s scripts/tiered -v
PYTHONPATH="$PWD/vendor/BigMoeOnEdge/third_party/llama.cpp/gguf-py" \
  ctest --test-dir vendor/BigMoeOnEdge/build --output-on-failure
```

Do not reapply the patch to the already-modified checkout. No llama.cpp files are
changed. The reader gates require a built `libbmoe_core.a` and Linux direct I/O.

## Profile and build

Collect `--route-trace routes.csv` from **baseline generation/session requests**
representative of your workload. Traced inference is not a speed benchmark.
The existing `--ppl` path emits a trace header but no routing rows; the builder
rejects such empty traces. Session JSON must contain `"cmd":"generate"`.

```bash
python3 scripts/tiered/build.py \
  --model /path/to/source.gguf --trace /path/to/routes.csv \
  --coverage .99 --codec results/my-tiered-experiment/codec.so \
  --output /path/to/cold99.tier
```

Multiple `--trace` arguments combine workloads. Coverage is the fraction of
observed routing *events*, not a quality guarantee. Unobserved experts are cold.
Q6_K and other quantization types are preserved. Use a new path for every build;
the builder refuses overwrites. Keep its JSON manifest with the sidecar.

## Score JEV rows

The adapter reads the existing `{id,state,question,options}` JSONL schema and
uses SemIf's prompt renderer. It loads the model once per invocation and scores
questions sequentially. It does not replace the resident API or implement
shared-prefix batching. Run with the existing SemIf environment:

```bash
semif/.venv/bin/python scripts/tiered/score.py \
  --input examples/decisions.sample.jsonl --output results/my-decisions.jsonl \
  --model /path/to/source.gguf --tokenizer /path/to/tokenizer
```

Add `--pack /path/to/cold99.tier` only for an explicit lossy experiment. The adapter
enforces an 8 GiB scope with no swap, records prompt hashes, and reports conditional
option probabilities as **uncalibrated**. Both llama.cpp and the tokenizer must
match the intended checkpoint. Original full-vocabulary probabilities are used
internally; only the requested labels are normalized in the result.

## Cold-cache measurement

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- \
  python3 scripts/tiered/run_guarded.py --output results/arm.json \
  --evict /path/to/source.gguf --evict /path/to/cold99.tier -- \
  vendor/BigMoeOnEdge/build/cli/bmoe-cli \
  -m /path/to/source.gguf --tiered-pack /path/to/cold99.tier \
  --moe-stream --dense-weights anon --cache-mb 3000 --io-threads 4 \
  -t 6 -c 256 --ubatch 256 --choices-only \
  --ppl-list /path/to/rendered-prompts.list --ppl-choices A,B
```

Use the same flags without `--tiered-pack` for baseline. The guard verifies cold
cache with `fincore`, fails if eviction did not work, and records scope peak/events,
wall time, exit status and process-tree device reads. Only successful runs count.
Use `analyze.py` on paired final-label logs and a fixture JSON containing expected
labels. Keep accuracy, probability drift, storage traffic and time separate.

`materialize.py` creates a separate expanded GGUF from a sidecar, with a full
source SHA256 check. It is for **same-weights correctness/transport controls** and
costs another checkpoint's disk space; it is not a compact model. It never changes
the source. `make_jev_fixtures.py` renders the original examples for profiling and
creates separate synthetic UI checks; those checks are not a production benchmark.
