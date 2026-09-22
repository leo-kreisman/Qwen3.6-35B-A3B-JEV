# Qwen3.6-35B-A3B decisions from SSD — the path we took

**Verdict: neither of the two options on the table.** Not Path A (point the existing
backend at a GGUF and hope), not Path B (write an Edge0 backend). The optimal path is
a **~20-line patch to SemIf's existing llama.cpp backend**, because that backend already
satisfies the precondition for SSD expert streaming — it just never says so out loud.

---

## 1. Why no expert-offload flags are needed

The whole `-ot ".ffn_.*_exps.=CPU"` / `--cpu-moe` / `--n-cpu-moe N` family exists to
keep **experts on the CPU while the other layers go to the GPU**. They are a
*partitioning* control.

SemIf's backend already sets:

```python
params.n_gpu_layers = 0     # src/semif_phase1/llamacpp_backend.py
```

Zero GPU layers means every layer — attention and experts alike — is already on the
CPU. The partitioning those flags perform has nothing left to partition. So there is
no flag to pass and no passthrough to build.

Streaming turns out to be the **default**, expressed through one field — and blocked by
a second one:

| field | default | what it does here |
|---|---|---|
| `load_mode` | `LLAMA_LOAD_MODE_AUTO` | Resolves to `mmap` (`llama-model-loader.cpp:554`). The GGUF stays a mapped file: a routed expert is read from disk on first touch and its page stays in the page cache. |
| `use_extra_bufts` | `true` | **The one that has to change.** It registers the CPU backend's extra buffer types, which include *repack* — the step that materialises expert tensors into anonymous RAM at load time. See §2b. |

`mmap` without `mlock` is exactly the "whole-file `mmap`, page-cache thrash" mechanism
the earlier research flagged — but here it is the *intended* path, not a fallback.
`LLAMA_LOAD_MODE_MLOCK` is what would kill it: it pins every expert in RAM, and a
checkpoint larger than memory then fails to load rather than merely running slow.

**A trap worth naming.** The obvious spelling of this — `params.use_mmap = True` /
`params.use_mlock = False` — is a **silent no-op** on this llama.cpp. Those fields were
removed from `llama_model_params`; the struct carries `load_mode` instead. Because a
ctypes `Structure` accepts arbitrary Python attributes, the assignment *succeeds*, and
`bool(params.use_mmap)` reads back as `True` — so metadata built from it is
self-confirming and wrong. The first version of this patch did exactly that and
therefore changed nothing; the flags and the metadata both had to move onto
`load_mode` / `use_extra_bufts` before either meant anything.

That is also why "it loaded fine on the big host" proved nothing: streaming was already
on by default, so a no-op patch and a working patch look identical there.

## 2. What the patch changes

`src/semif_phase1/llamacpp_backend.py`, inside `_cpu_model_params`:

- Sets `load_mode` **explicitly** (default `mmap`) instead of inheriting `AUTO`, with a
  `SEMIF_LLAMA_LOAD_MODE` override accepting `auto|none|mmap|mlock|mmap_mlock|direct_io`.
  An unknown value raises rather than silently falling back.
- Sets `use_extra_bufts = False` explicitly, with a `SEMIF_LLAMA_EXTRA_BUFT` override.
  This is the load-time repack switch (§2b), and it is the difference between a
  >RAM checkpoint loading and being OOM-killed.
- Records `load_mode`, `load_mode_name` and `use_extra_bufts` in the result metadata, so
  every output row is self-describing about how its weights were served.
- Docstring states the streaming contract, including why `use_mmap` / `use_mlock` are
  *not* the controls here.

`pytest tests/test_llamacpp.py` passes — the tests assert `load_mode` and
`use_extra_bufts`, and additionally assert the returned params object has **no**
`use_mmap` / `use_mlock` attribute, so a regression back to the inert spelling fails
loudly instead of silently doing nothing.

`direct_io` is exposed but not the path: it opens the GGUF with `O_DIRECT`, which reads
tensors into allocated buffers rather than mapping them (`llama-mmap.cpp:199`). That is
a *faster load*, not a smaller footprint — it makes residency worse, not better.

## 2b. The catch: llama.cpp repacks expert tensors into RAM

`load_mode=mmap` is **necessary but not sufficient**. The 8 GiB capped
run was **OOM-killed**, and it did not die during inference — it died during load:

```
repack: repack tensor blk.18.ffn_up_exps.weight with q4_K_8x8
run_semif_35b_ssd.sh: line 81: 1123053 Killed
```

Layer 18 of 40. The culprit is the CPU backend's **repack** step: for certain quantized
types (here `q4_K` experts) llama.cpp converts each tensor into a SIMD-friendly
interleaved layout (`q4_K_8x8`) at load time. That conversion **materialises the weights
into anonymous memory**, so those tensors stop being file-backed and demand-paged — the
exact property the streaming path depends on.

So on a small-RAM machine the failure mode is not "slow inference", it is **"cannot load
at all"**, and it presents as an out-of-memory kill rather than a configuration problem.
This is precisely the thing that would have blown up on the M3, and it is invisible if
you only ever test on a big host.

### The control is `use_extra_bufts` — and it is not an environment variable

Repack is reached through the CPU backend's *extra buffer types*.
`make_cpu_buft_list` adds them only when `use_extra_bufts` is set
(`llama-model.cpp:942`), and the repack buffer type is one of them — it comes from
`ggml_backend_cpu_repack_buffer_type()` (`ggml-cpu.cpp:65`). Because
`llama_model_default_params()` sets `use_extra_bufts = true`, **repack is on by default**,
and no command-line flag governs it: it has to be turned off in the params struct.

Measured on the real checkpoint — two loads, counting `repack` lines:

| `use_extra_bufts` | repack lines | load |
|---|---|---|
| `false` | **0** | loaded |
| `true` (default) | **119** | loaded |

The patch therefore sets it `false`. The cost is prompt-processing throughput, since the
interleaved kernels are the faster ones; `SEMIF_LLAMA_EXTRA_BUFT=1` restores llama.cpp's
default whenever the checkpoint fits in RAM and speed matters more than residency.

**Two wrong answers worth recording, because both looked right.** The first was
`GGML_CPU_REPACK=0`, inferred from the `CPU_REPACK` string in `libggml-cpu.so`. It is
**disproven** — the run still emitted 117 repack lines. That string is the buffer type's
*name*, returned by `ggml_backend_buft_name`, not an environment key. The second was
reading `use_extra_bufts` off a Python-constructed `llama_model_params()` and seeing
`False`. That struct is zero-initialised by the binding and is not the one llama.cpp
uses, so it reported the exact opposite of the real default (`true`,
`llama-model.cpp:2489`). Both errors had the same shape: a plausible-looking value read
from the wrong place.


## 3. Prerequisites found on disk

| Thing | Status |
|---|---|
| Target GGUF | **already local** — `/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`, 19,926 MB |
| GGUF identity | `general.architecture=qwen35moe`, `general.name=Qwen3.6-35B-A3B`, metadata `model_type=qwen3_5_moe` |
| SemIf install | **absent** at first — fresh clone; needs `uv venv --python 3.11` + `llama-cpp-python==0.3.35` (now done, `.venv/`) |
| Tokenizer | **was not cached** — fetched `Qwen/Qwen3.6-35B-A3B` (`vocab.json` + `merges.txt`; this repo ships **no** `tokenizer.json`) |
| Pinned `llama-cpp-python` | `0.3.35` ships **sdist only, zero wheels on PyPI** — a source build is unavoidable. The 0.2.39 in `~/.local` uses the **old C API** (`llama_load_model_from_file`, no `llama_get_memory`) and cannot run this backend |

The tokenizer requirement is worth stating plainly: SemIf's llama.cpp path does not
need the model's `model_type` to be supported — it only needs the **tokenizer** from HF,
because it re-tokenizes through the GGUF vocabulary and refuses to score if the two
disagree (`_verify_vocabulary`). That is why a `qwen3_5_moe` GGUF works here even though
`core.py` only recognises `qwen3_5` / `qwen3_5_text` for the torch backend.

## 4. The measurement problem — read this before trusting any number

**This host has 62 GiB RAM.** The checkpoint is 19.5 GiB. It fits in page cache, so
`mmap` will serve nearly every expert from RAM and the SSD path will never be exercised.
Any latency measured here is a **page-cache latency, not an SSD latency.**

### A cgroup cap alone does NOT fix this — measured, not assumed

`--simulate-8g` runs the scorer under `MemoryMax=8G` with swap disabled. It **does not
force a single disk read**, because a cgroup does not re-charge a page that is already
cached:

| evidence | value |
|---|---|
| `fincore` on the GGUF | **19.5 G resident** — the whole file is in the host page cache |
| `read_bytes` for the scored process, sampled every 3 s across the whole run | **0** |
| capped scope `memory.peak` | **632 MB** |
| capped scope `memory.events` `max` / `oom` | **0 / 0** — the 8 G limit never bound |

Only the anonymous working set (KV cache, branch states, compute buffers) is charged to
the new scope. The file pages were faulted first by the long-lived terminal scope, and
that is the cgroup they belong to — a fresh `systemd-run` scope reuses them for free.
The host's session scope shows a **40 GB** peak, which is where that charge actually sits.

So the cap is inert while the page cache is warm, and any "8 GiB capped" number taken
without evicting first is a page-cache number wearing a cap. The failure is silent:
nothing errors, the run simply finishes fast — and it will look *faster* under the cap
than without it, which is the tell.

### What actually forces the SSD path

Evict the file from page cache immediately before the run. Unprivileged
`posix_fadvise(POSIX_FADV_DONTNEED)` is sufficient — no root required:

```
before: RES 19.5G  PAGES 5100834
after:  RES    0B  PAGES       0
```

With the cache dropped, the run's own reads are charged to its scope, the cap binds, and
`read_bytes` becomes direct evidence that storage was touched. Use that counter as the
check rather than wall-clock time: a warm run and a disk run can look similar if the SSD
is fast, but `read_bytes` cannot be faked. `sudo` is not needed and `drop_caches` is not
needed — which matters, since this box has no passwordless sudo.

**And the harness has to be checked, not trusted.** The first version of the eviction step
was gated on `$SIMULATE`, which the re-exec'd child never saw: it passed `"$0"` with no
arguments, so `--simulate-8g` was dropped and the run silently took the warm path —
`fincore` still reporting 19.5 G, `total_seconds` still ~9 s. The flag was applying the
cgroup cap (that came from `systemd-run` itself) while quietly skipping everything else
gated on it. Eviction is now gated on an exported `SEMIF_CAPPED`, and the script prints
the `fincore` line before every run, so cache state is observed rather than assumed.
Verify a measurement harness by checking that it *fails* when it should, not only that it
runs.

The second thing to expect: **prefill and decode are not symmetric.**

- **Decode** touches ~8 experts per layer per token — small, sequential, SSD-friendly.
- **Prefill** over N tokens touches most of the 256 experts per layer — large, scattered,
  SSD-hostile.

SemIf reads decision logits from a **prefill**. A JEV decision is therefore the
*expensive* case, not the cheap one. This is the reason `--max-tokens` matters: the
default is **4096**, and a DOM snapshot will exceed it. The script defaults to 8192 and
will need tuning upward.

This is also the honest re-read of Edge0's 113–140 tok/s cold prefill claim: at ~3.3k
tokens that figure is arithmetically impossible if the whole ~20 GB model is read per
pass. Its `prerouter` predicts the *next token's* routing one token ahead so expert
loading overlaps generation (`docs/prerouter.md`) — a decode-time optimisation. It does
not make a one-shot prefill cheap.

## 5. Where this leaves Edge0

Edge0 is now a **reference implementation, not a dependency**:

- Its MLX-only backend (`src/edge0/backends/mlx`) is the Apple lock-in you said you
  wanted to avoid, and llama.cpp is your preferred substrate anyway.
- Its real assets are `docs/prerouter.md`, `moe/routing.py`, and the streaming contract
  in `docs/streaming.md` — worth reading for the expert-slot cache and the
  `staged_replace` / `pin_bonus` ideas **if** the cgroup-capped run shows the mmap path
  thrashing too hard.
- The fix for thrashing is deliberate I/O (`O_DIRECT` / `pread` with an expert offset
  table) instead of fault-driven paging. That is what Edge0 does and what unmerged
  llama.cpp PR #25294 does. **The gate on this is now passed**: the capped run thrashed
  measurably (112.8 GB read for 4 decisions, 70,584 limit hits), so this is the next
  engineering step rather than a speculative one.

## 6. Run it

```bash
./run_semif_35b_ssd.sh --simulate-8g     # measures the real streaming path
./run_semif_35b_ssd.sh                   # full host RAM; page cache hides the disk
```

Input rows are `state` + `question` + typed `options` → native option logits →
probabilities, via `--mode shared`, which prefills shared state once and evaluates the
criteria in parallel. That is the SemIf scoring interface a JEV-style decider wants.

**`--mode shared` has a hard precondition, and it is the JEV contract made explicit:**
every row must carry **one identical, nonempty `state`**. It refuses otherwise with
`ValueError: Shared scoring requires one nonempty exact state`. That is correct
behaviour, not a bug — "prefill the shared state once" is meaningless if the rows do not
share a state. So the input shape is *one page snapshot → many criteria*, which is
exactly the browser-decider pattern. Use `MODE=direct` for rows whose states differ.
`decisions.sample.jsonl` models the shared-state shape.

### Confirmed against the real checkpoint

The 19.5 GiB GGUF loads and reports:

- `load_mode = mmap` — the streaming path is active, not a fallback
- `general.architecture = qwen35moe`, `n_layer = 40`, `n_expert = 256`, `n_expert_used = 8`
- all 40 layers `assigned to device CPU`
- `file type = Q4_K - Small`, 19.45 GiB, 4.82 BPW

Each result row records its own serving evidence under the `model` key —
`load_mode: 1`, `load_mode_name: "mmap"`, `use_extra_bufts: false`, `n_gpu_layers: 0`,
`llama_cpp_python_version: 0.3.35`, `serving_config:
"llamacpp-state-restore-shared-v1"` — so a result file states how its weights were
served rather than requiring you to remember. Note these are read back off the params
struct llama.cpp actually consumed, which is the part the first version of this patch
got wrong.

### Measured baseline — **warm page cache, so this is not the SSD number**

Host: 62 GiB RAM, `threads=6`, `max-tokens=8192`, `MODE=shared`, `batch_size=4`.

**Read the caveat first:** this run predates the repack fix, so it executed the
**repacked** `q4_K_8x8` kernels — the faster ones. The `--simulate-8g` run cannot use
them (`use_extra_bufts=false`). So these two runs differ in *two* variables at once,
kernel choice and residency, and must not be compared as if only residency changed.

| | |
|---|---|
| Decisions correct | **4 / 4** |
| `total_seconds` (all 4 criteria) | 11.88 s |
| `encode_seconds` | 0.011 s |
| `prefix_tokens` (shared state) | 135 |
| `prefill_seconds` | 3.61 s |
| `replicate_seconds` | 0.020 s |
| `suffix_forward_seconds` | 8.24 s |
| `branch_state_bytes` | 68.6 MB |
| `true_suffix_tokens` / `padded` | 368 / 368 |

Read this carefully:

- The answers were **right**: the destination question picked the empty destination
  input at p≈0.99999, the date question rejected `Sat, Oct 18` for `2026-10-20` at
  p≈0.99994, and both well-formedness criteria returned `yes` at p≈0.99999. A 35B MoE
  does produce usable typed decisions through this path.
- `total_seconds` is charged to **all four** criteria, because `shared` mode prefills
  the state once and then replicates. So the marginal cost of one more criterion is
  roughly `suffix_forward / batch_size` ≈ **2.1 s**, not 11.9 s.
- **`suffix_forward` (8.24 s) exceeds `prefill` (3.61 s)**, and both are ~37–45 tok/s
  on CPU. That is the expert-weight read cost, and it is exactly the term that
  changes when the reads stop being page-cache hits.
- **This run does not measure the SSD path.** With 62 GiB of RAM the 19.5 GiB
  checkpoint is resident in page cache. The number that matters is the evicted run below.

### The SSD measurement — cache evicted, 8 GiB cap, and the cap actually binds

Same host, same input, `threads=6`, `max-tokens=8192`, `batch_size=4`, repack off. The
GGUF is evicted with `posix_fadvise(DONTNEED)` first, so the run's own reads are charged
to its scope.

| | warm (no cap) | **capped at 8 GiB** | ratio |
|---|---|---|---|
| `total_seconds` (4 criteria) | 9.29 s | **72.79 s** | **7.8×** |
| `prefill_seconds` (135-token state) | 2.45 s | **16.03 s** | 6.5× |
| `suffix_forward_seconds` (368 tok) | 6.82 s | **56.74 s** | 8.3× |
| marginal cost per extra criterion | ≈1.7 s | **≈14.2 s** | 8.3× |
| decisions correct | 4 / 4 | **4 / 4** | — |
| **bytes read from storage** | 20.9 GB | **112.8 GB** | **5.4×** |
| cgroup `memory.peak` | 40.3 GB (uncapped) | **exactly 8 GiB** | — |
| cgroup `memory.events` `max` | 0 | **70,584** | — |

Read this carefully:

- **The cap genuinely binds this time.** `memory.peak` is exactly 8 GiB and the limit was
  hit **70,584 times**, each hit a reclaim that evicts expert pages. Contrast the warm
  capped run: 632 MB peak, 0 hits, 0 bytes read. That difference is the whole validity of
  the measurement, and it is visible only because `read_bytes` was watched.
- **One shared-state run of 4 decisions read 112.8 GB from SSD** — 5.8× the size of the
  checkpoint. Subtracting the ~20.9 GB that any cold run pays once (SHA-256 pass plus first
  touch), roughly **92 GB is re-reading evicted expert pages**. That is the thrash, and it
  is now a number rather than a worry.
- **Answers did not degrade.** The same 4/4 typed decisions, at essentially identical
  probabilities (0.99995–1.0). Streaming costs time, not correctness.
- **The measured storage throughput during the read phase was ~800 MB/s.**
- **Reproduced independently.** The same measurement via `./run_semif_35b_ssd.sh
  --simulate-8g` (rather than the ad-hoc harness) gave **79.28 s** total, prefill 17.77 s,
  suffix 61.48 s, with the identical 4/4 answers and probabilities. So quote the budget as
  **≈73–79 s for 4 decisions on a 135-token state, ≈14–15 s marginal per criterion** — the
  spread is run-to-run variance, not a different regime.

**Two things this implies that are worth stating plainly:**

1. **Repack-off is not a speed cost on this machine.** Warm and repack-off is **9.29 s**
   against **11.88 s** warm and repack-on — it is ~22% *faster* here, not slower. So the
   setting that makes a >RAM checkpoint loadable also happens to be free on this CPU. The
   earlier assumption that disabling repack trades speed for residency did not hold.
   (Caveat: the 11.88 s figure was measured in an earlier session; the two warm runs
   differ by that one setting as far as the configuration recorded, but not under
   identical process conditions.)
2. **The JEV case is worse than this table shows.** `prefix_tokens` is only **135** —
   a Google Flights form snapshot. Prefill cost scales with tokens × experts touched, and
   a real DOM snapshot is thousands of tokens. The 16 s prefill is the *cheap* end of the
   JEV case. Budget the prefill term against a state two orders of magnitude larger before
   extrapolating to production.

## 6b. Optimising the scoring path — 112.8 GB → 60.2 GB, 72.8 s → 56.6 s

The first SSD measurement cost **112.8 GB and 72.8 s** for four decisions, and the cost grew
with the criteria count (≈13.2 GB marginal per criterion on a ≈60 GB fixed overhead). Two
mechanisms caused that, and neither was visible from the run output — they had to be
attributed by phase.

**Mechanism 1 — a single-sequence context.** `_Engine` created the context with
`n_seq_max = 1`. A context that can hold only one sequence has no way to carry four
criteria at once, so `score_shared` serialised them: restore a saved prefix state, decode
one branch, restore, decode the next. Five full model passes for four criteria.

**Mechanism 2 — the physical batch, which is the thing that actually decides I/O.** Simply
raising `n_seq_max` and putting all four branches in one `llama_decode` call did *not* fix
it: reads went 112.8 → 108.1 GB and the run got slightly **slower**. The log said why:

```
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs = 8, n_outputs = 8
```

A decode of 908 tokens against llama.cpp's default `n_ubatch` of 512 is split into **two
ubatches**, and every ubatch walks all 40 layers and re-reads their experts. So one
"single pass" in Python was two full model sweeps underneath. Setting `n_batch = n_ubatch`
to the whole group size (4096 for 8 sequences) is the fix that worked.

### What was measured, by phase

Adding `n_ubatch` alone, with reads attributed:

| phase | time | bytes read | note |
|---|---|---|---|
| load | 68.7 s | **41.8 GB** | 20.9 GB my SHA-256 + 20.9 GB llama.cpp's own load |
| decode | 78.0 → **56.0 s** | 67.0 → **42.0 GB** | one ubatch instead of two, per the fix above |

`n_ubatch` is a real lever with a real cost: the CPU compute buffer went from 90 MiB to
586 MiB (it is sized by `n_ubatch` and allocated whether or not a group fills it).

**The load was the other half — and it reads the checkpoint twice.** Isolated with no
SHA-256 pre-pass and no tokenizer, `llama_model_load_from_file` alone reads **20.9 GB**
(18.3 s) — the entire file — even with `load_mode = mmap` and `use_extra_bufts = false`.
`check_tensors` is `False` in the params this builds, so it is not a validation read; it is
this llama.cpp's mmap path touching the whole mapping. That is not controllable through
`llama_model_params`, but the *other* full read is: the SHA-256 integrity pass. It now
caches its digest in a `<name>.sha256.json` sidecar keyed on size + `mtime_ns`, so a
repeated run skips 20.9 GB. `SEMIF_GGUF_DIGEST=compute|skip` forces or drops it, and every
result row names which path was taken in `model.gguf.digest_source`.

### Result — same cache eviction, same 8 GiB cap, same input

| | before | after | |
|---|---|---|---|
| `total_seconds` (scoring) | 72.79 s | **56.57 s** | 1.29× |
| bytes read, whole run | 112.8 GB | **60.2 GB** | 1.87× |
| bytes read per criterion | 28.2 GB | **15.1 GB** | 1.87× |
| prompt throughput | 6.9 tok/s | **16.1 tok/s** | |
| decisions correct | 4 / 4 | **4 / 4** | identical probabilities |
| cgroup `memory.peak` | 8 GiB | 8 GiB | cap still binds |
| cgroup `memory.events max` | 70,584 | **39,005** | fewer refill stalls |

The 4/4 answers and their probabilities are **bit-identical** to the pre-optimisation run
(0.999982 / 0.999955 / 0.999992 / 0.999989), which is the check that matters — the batched
pass changes how the branches are scheduled, not what they compute.

### The trade-off, stated honestly

**Warm, the batched path is slower: 18.53 s against 9.29 s.** Each branch now carries the
whole prompt as its own sequence, so the 135-token shared state is recomputed per criterion:
908 token-forwards instead of 503. Warm, that cost is real and unmasked. The obvious fix —
prefill the prefix once and copy its KV into each branch with `llama_memory_seq_cp` —
**does not work on this checkpoint**: `qwen35moe` is a *hybrid* model, interleaving
recurrent (linear-attention) layers with attention layers, and
`llama_memory_hybrid::seq_cp` aborts on the recurrent half (measured: SIGABRT inside
`llama_kv_cache::seq_cp`). So the choice is redundant compute or redundant I/O, and on a
checkpoint that cannot be resident, I/O is the term worth paying compute to avoid.

Batching is therefore the right default **only for the regime this work targets** — a
checkpoint that does not fit in RAM. On a host with room for the full 19.5 GiB, the
single-sequence path is faster.

**Tested and refuted — the read amplification is load-bearing prefetch, not waste.** The
decode reads 42.0 GB for one ubatch whose logical expert footprint is ~19.5 GB (≈2.15×), which
looks like mmap fault-around read-ahead pulling pages that are never used.
`madvise(MADV_RANDOM)` on the GGUF mapping (`/proc/self/maps` gives the VMA even though
llama.cpp owns the descriptor) suppresses exactly that read-ahead:

| | default | `MADV_RANDOM` |
|---|---|---|
| decode bytes read | 42.0 GB | **26.0 GB** |
| decode time | 56.0 s | **368.0 s** |
| effective throughput | 750 MB/s | **71 MB/s** |
| decisions | 4/4 | 4/4, identical |

Reads fell 38% and the run got **6.6× slower**. Killing read-ahead converts scattered expert
accesses into synchronous random reads, and NAND random-read latency — not bandwidth —
takes over: effective throughput drops ~10×. So the "amplification" is prefetch that pays
for itself, and the naive fix is catastrophic.

The real target is a different design, not a smaller number: an explicit async `pread` pool
that reads **exact expert extents** in large aligned chunks with a worker pool and overlaps
layer N+1's reads with layer N's compute. That is what every pread-based engine here does,
and what llama.cpp issue #23324 measured at +53% (0.43 → 0.66 t/s) over mmap. It is a
rewrite of the weight-access path inside llama.cpp, not a tuning knob. See
`OFFLOAD_PROJECTS_ANALYSIS.md`.

## 6c. Thread oversubscription — 56.6 s → 38.4 s, lossless, no code path change

The biggest remaining lever turned out to be one nobody had swept: **the thread count**.
The runner pinned `THREADS=6` on a 6-core/12-thread host — half the machine — and
`load_model`'s own default was `os.cpu_count()`, which is exactly the hardware thread
count. Both are wrong for a decode served from disk.

The reason is that a thread waiting on an expert page fault is not runnable. With a
checkpoint that does not fit in RAM the decode is continuously faulting, so it wants
**more runnable threads than the CPU has**, to keep the device queue fed. Swept over one
fixed pass (4 branches, 908 token-forwards, ~41 GB read, cache evicted, 8 GiB cap):

| threads | decode | effective throughput |
|---|---|---|
| 6 (the pinned default) | 54.62 s | 768 MB/s |
| 8 | 50.35 s | 836 MB/s |
| 12 (hardware threads) | 47.87 s | 860 MB/s |
| **16 (= 4/3 × 12)** | **38.44 s** | **1069 MB/s** |
| 20 | 38.42 s | 1054 MB/s |

Monotonic to a knee at 16, then flat. Read volume is unchanged across the sweep
(41.2–42.1 GB), so this buys wall clock without touching I/O — it is *not* an I/O
improvement, and it is the evidence that the decode is not purely I/O-bound after all.

**−29.6% (54.62 → 38.44 s), lossless, identical probabilities, one parameter.** The policy
now lives in `_default_threads()` (4/3 of `os.cpu_count()`, floor 4) and the runner
computes the same value instead of pinning 6.

### Two prefetching variants, both measured worse — do not retry

The decode reads ~41 GB for a checkpoint whose routed experts are 18.33 GB of the 20.89 GB
file, i.e. ~2× the file. That looks like read-ahead waste, and it is — but it cannot be
fixed by warming the cache, because the 8 GiB cap makes speculative warming
self-defeating. A layer-ordered prewarmer (`/tmp/probe_prefetch.py`) walks the 40 layers'
expert extents in the same order the decoder does, so the page-cache victim is always an
already-consumed layer:

| decode | read | time |
|---|---|---|
| no prewarmer | 42.0 GB | 56.0 s |
| `POSIX_FADV_WILLNEED`, 1.3 GB/s | 42.1 GB | 55.95 s |
| real `preadv`, 1.3 GB/s | **56.1 GB** | **57.56 s** |

Two distinct negatives. `WILLNEED` moved read_bytes by **0.14 GB across a whole run**: the
kernel caps an advised readahead window to a small multiple of `read_ahead_kb` (128 KB
here), so the 16 MB chunks became ~128 KB requests and the advice was never acted on. The
real-`preadv` version did read all 18.3 GB — and made the run *worse*: the prewarmed pages
are clean and unreferenced, so they are the **first** reclaim victims under the cap, evicted
before the decoder reaches them and faulted in a second time. We paid for those bytes twice.

This is the same lesson the prior art states as a rule — a **hard RAM admission gate**, and
`pread` into the engine's *own* bounded buffers rather than the kernel page cache
(`OFFLOAD_PROJECTS_ANALYSIS.md` §2). Under a fixed cap, the page cache is the wrong place
to stage.

Also refuted here: the theory that the ~2× is a **second expert sweep**. With 8 branches ×
908 tokens = 7,264 tokens it would have been — but the sample's group is 4 branches / 908
tokens, which already fits one 4,096 ubatch (sweeps = 1). Raising `SEMIF_LLAMA_UBATCH` to
8192 changed nothing (41.93 GB / 54.62 s). The two-sweep mechanism from §6b is real but is
not what is inflating this measurement; `n_ubatch` already covers the group.

## 6d. Deployment-shaped tests — the invocation lifecycle, contention, budget and scaling

Every measurement up to here was a single run on an idle box with a synthetic cap. The scorer
is meant to be **invoked by a code agent on a workstation that is doing other things**, so §6c's
numbers are best-case. Four tests in that shape, plus one real bug.

### The bug: more criteria than `n_seq_max` failed outright

`score_shared` cleared the context **once, before** the group loop. Every branch carries the
whole prompt as its own sequence, so positions restart at 0 within each group — and a group that
does not release the previous group's cells reuses sequence ids whose KV still holds a longer
prompt. Measured: **16 criteria against `n_seq_max=8` raised `llama_decode failed` from group 2
onward.** A hard failure, not a slow answer, and exactly what an agent sending a batch will hit.
Fixed by clearing per group; 16 criteria now return 16 correct answers with identical
probabilities. Regression test: `test_score_shared_clears_between_groups_so_sequence_ids_can_be_reused`.

### 1. Lifecycle — `cli.py` is one-shot and nothing carries over

3 invocations of `load → score → exit`, against one resident process:

| | 3 one-shot calls | resident (1 load + 3 calls) |
|---|---|---|
| wall | 58.2 / 61.2 / 60.6 = **180.0 s** | 19.8 + 40.3 / 39.5 / 39.4 = **139.0 s** |
| bytes read | 61.9 / 61.6 / 61.7 = **185.2 GB** | **144.1 GB** |

Calls 2 and 3 read the *same* 61.6 GB as call 1: with ~41 GB per call against an 8 GiB cap,
nothing survives between invocations. So the load (20.9 GB, ~19.8 s) is paid in full every call —
**33% of per-call wall clock**. Residency saves **22.8% over 3 calls**, **33.8% per call** once
loaded, and more as calls accumulate (10 calls: 600 vs 417 s). This is an architecture decision,
not a knob.

### 2. Contention — the thread default holds, and matters more under load

Cache evicted, 8 GiB cap, background busy loops as the workstation's other work:

| scorer threads | idle | +6 busy | +12 busy |
|---|---|---|---|
| 16 (new default) | 38.4 s | 55.75 s | **74.17 s** |
| 6 (old default) | 54.62 s | 83.54 s | **155.34 s** |

Under 12 competing threads 16 beats 6 by **2.1×**, wider than idle's 1.4×, and the old default
degrades worse (2.8× vs 1.9×). Reads stay ~41 GB throughout, so contention costs time, not bytes.
§6c's default is therefore not an idle-box artifact. Realistic agent-facing latency on a busy
box is **~74 s per call**, not 38 s.

### 3. Memory budget — more cache is not monotonically better

| cap | decode reads | decode | |
|---|---|---|---|
| 8 GiB | 41.0 GB | 38.4–40.3 s | stable (±2.5%, 4 samples) |
| 12 GiB | 30.2 / 30.5 GB | 49.98 / **34.91** s | **unstable (±18%)** |
| 16 GiB | 19.28 GB | 29.31 s | |
| 24 GiB | 0.00 GB | 16.02 s | fully cached |

Reads fall monotonically with the cap; time does not. The 24 GiB point cross-checks the warm
floor (16.02 s against 16–18.5 s warm), which is the sanity check that these are real. 12 GiB
straddles the fit boundary and goes **bimodal** — identical read volume, 15 s of spread. Size the
budget with a clear margin on one side, never near the fit boundary. Note 8 GiB is not a handicap
despite reading the most: it also achieves the highest per-byte rate (1.07 GB/s against 0.61 at
12 GiB), which is why the 8 GiB operating point was never the problem.

### 4. Criteria scaling — cost grows faster than the criterion count

| N | passes | tokens | reads | time | s/criterion | tok/s |
|---|---|---|---|---|---|---|
| 4 | 1 | 908 | 41.0 GB | 39.4 s | 9.85 | 23.0 |
| 8 | 1 | 1816 | 64.1 GB | 66.7 s | 8.34 | 27.2 |
| 16 | 2 | 3632 | 128.3 GB | 127.1 s | 7.94 | 28.6 |

Batching pays a little (9.85 → 7.94 s/criterion) but is nowhere near free: one pass over 8
branches reads 64 GB against 41 GB for 4, because the amplification grows with tokens-per-pass.
Past `n_seq_max=8`, each additional group is a full extra pass. **The amplification is now the
thing to explain, and it scales with tokens — still pointing at the pread-over-exact-extents
rewrite, not at any page-cache knob.**

## 7. Open risks

1. **Tokenizer/GGUF vocabulary agreement** is enforced at load; if the fetched tokenizer
   disagrees the run fails loudly rather than scoring wrong. Expect this to be the first
   thing that breaks.
2. **`--mode shared` vs `--mode direct`** differ in whether state is prefilled once;
   `shared` is the right default for a decision server, but its timing block is recorded
   separately and should be read before quoting a per-decision latency.
3. **Prompt length — now the dominant risk, and measured.** 4096 is a hard cap; the
   measured `prefix_tokens` was only **135**, and prefill already cost **16 s** under an
   8 GiB budget. Prefill scales with tokens × experts touched, so a real DOM snapshot is
   the case to be afraid of. Raising `--max-tokens` also grows KV memory, competing for
   exactly the RAM streaming needs.
4. **`--llama-threads` defaults to every core.** Saturating cores while doing fault-driven
   disk reads usually loses; the script defaults to 6 and this is still unmeasured under
   the capped configuration.
5. **The next step is justified, not speculative.** The thrash is now quantified
   (§6), so deliberate I/O — `pread` against an expert offset table, or `O_DIRECT` —
   is the measured response. Note `SEMIF_LLAMA_LOAD_MODE=direct_io` is **not** that fix:
   it applies `O_DIRECT` to *loading*, reading tensors into allocated buffers
   (`llama-mmap.cpp:199`), which is a faster load and a worse footprint.
6. **One measurement caveat to keep.** The warm/repack-on baseline (11.88 s) was taken in
   an earlier session. The warm/repack-off figure (9.29 s) was reproduced twice under
   current conditions, so the "repack-off is free" conclusion rests on the latter pair;
   re-measure repack-on under current conditions before leaning on the 22% figure.
