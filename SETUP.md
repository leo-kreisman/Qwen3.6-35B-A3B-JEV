# Setup

How to install, run, and invoke the scorer. Written for the case this was built
for: **a coding agent calling it as a tool on a shared workstation.**

Read `README.md` first for what the numbers mean — in particular that nothing is
generated and the `tok/s` figures are prefill throughput. This file is the
operational path only.

---

## What you get, and what you do not

**You get** a scorer: given one page state, one question, and a list of candidate
options, a probability per option. Deterministic, one pass, no sampling.

**You do not get** a chat model, a resident server, or calibrated confidence.
`cli.py` is one-shot — `load → score → exit`. Every call pays a ~19.8 s model
load, and nothing carries over between calls. See [Latency](#latency-what-to-budget).

## Prerequisites

| Need | Why |
| --- | --- |
| Linux x86-64 | what this was measured on; llama.cpp goes wider, the runner does not |
| ~25 GB free disk | 20.9 GB GGUF + venv |
| Python 3.11 (3.10+) | SemIf requires ≥3.10; 3.11 is what was tested |
| `uv` | for the venv and for `uvx` to fetch the tokenizer |
| `git` | to fetch SemIf |
| cgroup v2 + `systemd --user` | **only** for `--simulate-8g` (the cap path) |
| C++ toolchain | only if `llama-cpp-python` has no wheel for your platform |

You do **not** need a GPU. The whole point is CPU-only with experts on disk.

---

## Step 1 — the code

```sh
git clone https://github.com/leo-kreisman/Qwen3.6-35B-A3B-JEV
cd Qwen3.6-35B-A3B-JEV
```

## Step 2 — the model and the tokenizer

The exact artifact measured (20,893,015,008 bytes):

```sh
uvx --from huggingface_hub hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --local-dir ~/models/unsloth/Qwen3.6-35B-A3B-GGUF
```

Worth verifying, since every number in this repo is against *this* file:

```sh
sha256sum ~/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
# expect a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c
```

The runner fetches the tokenizer on first use into
`~/models/Qwen3.6-35B-A3B-tokenizer`. SemIf renders the prompt with the reference
HF tokenizer and then re-tokenizes through the GGUF vocabulary, **refusing to
score if the two disagree** — so the tokenizer must come from the same family as
the GGUF. A mismatched pair is a hard error, not a silent quality loss.

## Step 3 — SemIf, the patch, and the venv

```sh
./patch/apply.sh                 # clones SemIf at the pinned commit, applies the patch
cd semif
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[test,llamacpp]'
```

`patch/apply.sh` refuses to touch an existing `semif/`, so it is safe to re-run.
To do it by hand instead, see `patch/README.md`.

Two notes on the install:

- The `llamacpp` extra pins `llama-cpp-python==0.3.35`. If no wheel exists for
  your platform it builds from source, which needs a C++ toolchain and a while.
- SemIf's base dependencies include `torch`, because its default backends are
  PyTorch and MLX. The llama.cpp path does not use torch, but the dependency set
  asks for it anyway (~2.5 GB). There is no supported way to skip it.

Confirm the patch landed:

```sh
.venv/bin/python -m pytest -q          # expect: 90 passed, 3 skipped
```

The 3 skips are the MLX backend's, on a host without MLX.

## Step 4 — sanity, warm

```sh
cd ..                                  # back to the repo root
./run_semif_35b_ssd.sh
```

This leaves the page cache **warm on purpose**. On a host with enough RAM the
checkpoint is served from cache and the run is fast — that is expected, and it is
*not* the disk path. Use it to confirm the model loads and the answers look sane.

```sh
cat results/decisions.out.jsonl | head -1
```

## Step 5 — the real thing

```sh
./run_semif_35b_ssd.sh --simulate-8g
```

This re-execs under a `MemoryMax=8G` scope with swap off, evicts the GGUF from
page cache with `posix_fadvise(DONTNEED)`, and then scores. Budget **~40 s** on an
idle box, **~74 s** on a busy one.

> A memory cap alone does **not** exercise the disk path. A cgroup does not
> recharge a page that is already cached, so a fresh capped scope reuses a warm
> file for free — measured: `memory.peak` 632 MB, `read_bytes` 0. The eviction
> step is what makes the cap bind. That is why both are in the runner together.

### Confirming you actually hit the disk

Never trust wall-clock here: a warm run can be *faster* than a disk run, and that
inversion is the tell. Check the counters instead.

```sh
fincore ~/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
# want: PAGES 0 after the eviction step
```

For `read_bytes` you need the process, and the CLI exits before you can read
`/proc/<pid>/io`. `scripts/probes/oneshot_wrapper.py` runs the CLI in-process via
`runpy` and prints `read_bytes` afterwards — that is exactly what it is for.

---

## Invoking it from an agent

### Input — one JSON object per line

```json
{"id": "c1-destination",
 "state": "Page: Google Flights (one-way search form). Observed elements: [0] input 'Where from?' value 'Zürich'; [1] input 'Where to?' value ''; ...",
 "question": "Which observed element should receive the destination value 'London'?",
 "options": [{"id": "e0", "description": "Element 0, the origin input, which already holds Zürich."},
             {"id": "e1", "description": "Element 1, the destination input, which is currently empty."}]}
```

| Field | Notes |
| --- | --- |
| `id` | unique per row; echoed back |
| `state` | a **string** — the rendered page state. Not a dict. |
| `question` | the criterion |
| `options` | 2+; `id` and `description` both required |

**In `MODE=shared` (the default, and the JEV pattern) every row must carry one
identical `state`.** That is the whole trick: one page state prefilled once, many
criteria scored against it in parallel. For rows with *differing* states use
`MODE=direct` — that scores each row independently and is much slower per
criterion, because each row re-prefills.

Fixtures: `examples/decisions.sample.jsonl` (4 criteria, one shared state),
`examples/decisions.one.jsonl` (1), `examples/decisions.n8.jsonl` and `.n16.jsonl`
(8 and 16, for scaling — n16 exercises the multi-group path).

### Output — one JSON object per line

```json
{"id": "c1-destination",
 "option_ids": ["e0", "e1", "e2", "e3"],
 "probabilities": [1.27e-05, 0.9999854, 1.36e-06, 4.81e-07],
 "option_logits": [18.547, 29.819, 16.314, 15.272],
 "answer_token_ids": [32, 33, 34, 35],
 "input_tokens": 253,
 "allowed_token_mass": 0.9999351,
 "full_vocab_argmax_id": 33,
 "prompt_sha256": "e4c77f47...",
 "prompt_version": "direct-options-v1",
 "readout": "quantized branch last-position logits over a restored prefix state",
 "probability_status": "conditional option score over quantized weights; uncalibrated as decision confidence",
 "shared_timing": {"total_seconds": 35.68, "prefix_tokens": 135, "prefill_seconds": 19.24,
                   "suffix_forward_seconds": 16.44, "batch_size": 1, ...},
 "model": {"gguf": {"file": "...", "bytes": 20893015008, "sha256": "a8138f18..."},
           "threads": 6, "load_mode_name": "mmap", "use_extra_bufts": false, ...}}
```

Three fields matter more than the rest:

- **`probability_status` reads "uncalibrated as decision confidence".** Take it
  literally. These are conditional option scores over quantized weights: good for
  *ranking* options, and **not** a calibrated probability. Do not threshold them
  as though 0.9 meant 90% of the time. Use the argmax, or rank.
- **`full_vocab_argmax_id`** is the argmax over the whole 248,320-token vocabulary.
  When it equals `answer_token_ids[argmax]`, the model's preferred continuation
  really was one of your option labels. When it does not, the model wanted to say
  something you did not offer, and the option ranking is the least of your
  problems. This is the cheapest sanity check available — use it.
- **`allowed_token_mass`** is how much probability stayed on the option labels. Low
  values mean the question is being answered outside the option set.

Every row also records the model, thread count, and mode under `model`, so a
result file is self-describing.

### Pick the argmax

```sh
./run_semif_35b_ssd.sh --simulate-8g
jq -r '. as $r
  | ($r.probabilities | to_entries | max_by(.value) | .key) as $i
  | "\($r.id)\t\($r.option_ids[$i])\t\($r.probabilities[$i])"' results/decisions.out.jsonl
```

### Latency: what to budget

| Scenario | Per call |
| --- | --- |
| one-shot CLI, idle box, 4 criteria | ~38–40 s |
| one-shot CLI, ~12 competing threads | **~74 s** |
| the load alone, every call | **~19.8 s (33%)** |

**Plan for ~74 s per call, not 38 s**, if this runs on a workstation that is doing
other work — and it is meant to. The thread default is worth *more* under
contention (2.1×) than idle (1.4×), so do not pin it down to be polite to the
other workloads; that measurably backfires.

The CLI is create-only and refuses to overwrite its output file. The runner
`rm -f`s it first. If you call the CLI directly, do the same.

---

## Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `SEMIF_DIR` | `<repo>/semif` | where SemIf lives |
| `PY` | `$SEMIF_DIR/.venv/bin/python` | interpreter |
| `GGUF` | `~/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` | the checkpoint |
| `TOKENIZER_DIR` | `~/models/Qwen3.6-35B-A3B-tokenizer` | fetched if absent |
| `INPUT` / `OUTPUT` | `examples/decisions.sample.jsonl` / `results/decisions.out.jsonl` | |
| `MODE` | `shared` | `shared` or `direct` |
| `THREADS` | `nproc * 4 / 3` | the measured knee; override to pin |
| `MAX_TOKENS` | `8192` | prompt budget |
| `LOAD_MODE` | `mmap` | keep it; file-backed and demand-paged |
| `EXTRA_BUFT` | `0` | **keep it 0**; see below |

Backend-level overrides, read by the patched `llamacpp_backend.py`:

| Variable | Default | Notes |
| --- | --- | --- |
| `SEMIF_LLAMA_SEQ_MAX` | `8` | branches per group; past this each group is a full extra pass |
| `SEMIF_LLAMA_UBATCH` | `seq_max * 512` | physical batch — this, not `n_batch`, decides I/O |
| `SEMIF_GGUF_DIGEST` | `cache` | `compute` or `skip` |

---

## Troubleshooting

**`EXTRA_BUFT=1` OOM-kills during load.** Not a bug in the patch. The repack
switch defaults to `true` upstream, which rewrites `q4_K` expert tensors into
`q4_K_8x8` into anonymous RAM at load time — a >RAM checkpoint dies during load
rather than merely slowing down. This is the single most important setting here.

**`llama_decode failed` past 8 criteria.** Fixed in this patch; if you see it you
are running unpatched code. `score_shared` used to clear the context once before
the group loop, so groups after the first reused sequence ids whose KV still held
a longer prompt. Verify with `examples/decisions.n16.jsonl`.

**Two settings that look right and do nothing.** `params.use_mmap` /
`params.use_mlock` were *removed* from `llama_model_params` in this llama.cpp — a
ctypes `Structure` accepts arbitrary attributes, so setting them succeeds, reads
back plausibly, and has no effect. The real controls are `load_mode` and
`use_extra_bufts`. Likewise `GGML_CPU_REPACK=0` is bogus: `CPU_REPACK` is a
buffer-type *name*, not an env key.

**`--simulate-8g` finishes suspiciously fast.** The cap did not bind — the pages
were already cached. `systemd-run --user` must be available, and check
`fincore` reports `PAGES 0`.

**No `systemd --user` (container, CI).** Skip `--simulate-8g`, cap the container
yourself, and evict the file before each run — `scripts/probes/measure_evicted.sh`
and `run_lifecycle.sh` show the eviction step in isolation. Without *both*, you
are measuring page cache.

**Threads look absurd (`nproc * 4/3`).** That is deliberate and it is the single
biggest win in the repo: a thread blocked on an expert page fault is not runnable,
so a disk-served decode wants more runnable threads than cores. Measured knee at
4/3; flat from there to 20. Full sweep in `docs/SEMIF_LLAMACPP_SSD.md` §6c.

---

## Things that look like they will help and do not

All measured, all written up in `docs/SEMIF_LLAMACPP_SSD.md` §6c. Listed so you
do not spend a day re-deriving them:

- **Adding `-ot` / `--n-cpu-moe` / `--cpu-moe`.** These are *partitioning*
  controls — they keep experts on CPU while other layers go to the GPU. SemIf sets
  `n_gpu_layers = 0`, so every layer is already on CPU and there is nothing to
  partition. No passthrough is needed.
- **Prefetching with `POSIX_FADV_WILLNEED`.** The kernel caps an advised readahead
  window to a small multiple of `read_ahead_kb` (128 KB here), so large-chunk
  advice is silently truncated to ~128 KB requests. Moved `read_bytes` by 0.14 GB.
- **A real `preadv` prewarmer.** Made it *worse*: 42.0 → 56.1 GB, 56.0 → 57.56 s.
  Under a cap, clean unreferenced prefetched pages are the first reclaim victims —
  evicted before use, then faulted in twice.
- **`MADV_RANDOM`.** Cuts reads 41 → 26.0 GB but costs 6.6× in time.
- **Sizing the cap near the fit boundary.** 12 GiB measured *bimodal*: 49.98 s then
  34.91 s on identical reads (±18%), against ±2.5% at 8 GiB. Leave a clear margin.
