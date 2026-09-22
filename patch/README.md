# The SemIf patch

This directory is the change to SemIf that makes it score typed decisions with
Qwen3.6-35B-A3B streamed from SSD. SemIf itself is **not** in this repository.

    upstream:  https://github.com/TheoLeeCJ/SemIf
    license:   MIT, Copyright (c) 2026 TheoLeeCJ
    pinned at: 1f2dea3e25379f9dfc98cb83c324f00ab5deda37
               "Merge pull request #18 from chenqy2018/llamacpp-cpu-backend"

## Why a patch and not a fork

`semif/` on the machine this was measured on is a clone of the upstream repo with
three files modified. Publishing it here would mean either carrying a foreign
history into this repository or silently forking someone else's project. A diff
against a pinned commit does neither: it says exactly what changed, against
exactly what, and leaves upstream as the source of truth.

Both forms are provided, because they serve different readers:

| Path | Use |
| --- | --- |
| `semif-ssd-scorer.patch` | `git apply` onto a checkout at the pinned commit |
| `files/` | the three complete post-patch files, to drop in directly |

`files/` is for reading and for the case where you want the file, not the delta.
The patch is authoritative: it is what was tested.

## Apply

```sh
./apply.sh /path/to/where/semif/should/live
```

The script clones upstream at the pinned commit, applies the patch, and then runs
the new test file to confirm the result. To do it by hand:

```sh
git clone https://github.com/TheoLeeCJ/SemIf.git
cd SemIf
git checkout 1f2dea3e25379f9dfc98cb83c324f00ab5deda37
git apply /path/to/semif-ssd-scorer.patch
```

## What the patch changes

Three files, **+343 / -32** on the tracked files plus one new test file.

### `src/semif_phase1/llamacpp_backend.py` — the substantive work

- **Batched branch scoring** (`branch_logits_batched`, `score_shared`). Each
  branch carries the whole prompt as its own sequence with positions restarting
  at 0, because `qwen35moe` is a hybrid model whose `llama_memory_hybrid::seq_cp`
  aborts — so branches *cannot* share one prefilled prefix KV. This is the whole
  reason the batched path costs 908 token-forwards instead of 503.
- **`_ubatch()`** — sets `n_batch = n_ubatch = sequences * 512`. `n_ubatch`, not
  `n_batch`, is the parameter that decides I/O: a decode larger than `n_ubatch`
  splits into physical ubatches, and each one re-walks all 40 layers and re-reads
  their experts. This was the single largest read-volume win (67.0 → 42.0 GB).
- **`_default_threads()`** — `max(4, nproc * 4 // 3)`. Mild oversubscription,
  because a thread waiting on an expert page fault is not runnable, so a
  disk-served decode wants more runnable threads than the CPU has. Measured knee
  at 4/3 of hardware threads: 54.6 s → 38.4 s, lossless.
- **`_gguf_digest()`** — caches the SHA-256 in a `<name>.sha256.json` sidecar
  keyed on size + `mtime_ns`, so a cold run stops reading the checkpoint twice
  before scoring a token. `SEMIF_GGUF_DIGEST=compute|skip` overrides.
- **`load_mode=mmap` + `use_extra_bufts=false`** as the defaults. The repack
  switch is the one that matters: its default (`true`) rewrites `q4_K` experts
  into `q4_K_8x8` at load time into anonymous RAM, which OOM-kills a >RAM
  checkpoint during load rather than merely slowing it.
- **The bug fix** — `score_shared` cleared the context once *before* the group
  loop rather than per group. Any caller sending more than `n_seq_max=8`
  criteria reused sequence ids whose KV still held a longer prompt and got a hard
  `llama_decode failed` from group 2 onward. Regression test in
  `files/tests/test_llamacpp_batched.py`.

### `tests/test_llamacpp.py`, `tests/test_llamacpp_batched.py`

Coverage for the above. The new file is the batched-path suite: per-sequence
positions, logit flags across ubatch splits, ubatch sizing, the thread policy
table, the group-clearing regression, and the digest cache.

## Traps this patch encodes

Two settings that look right and do nothing, both found the hard way:

- **`params.use_mmap` / `params.use_mlock` were removed from
  `llama_model_params`** in the llama.cpp this was built against. A ctypes
  `Structure` accepts arbitrary Python attributes, so setting them succeeds,
  reads back plausibly, and has no effect. The controls are `load_mode` and
  `use_extra_bufts`.
- **`GGML_CPU_REPACK=0` is bogus.** `CPU_REPACK` is a buffer-type *name*, not an
  env key.

Both are commented in the source so the next reader does not re-derive them.

## Verify

```sh
cd /path/to/semif && .venv/bin/python -m pytest -q
```

Expected: `90 passed, 3 skipped` (3 skipped are the MLX backend's, on a host
without MLX).
