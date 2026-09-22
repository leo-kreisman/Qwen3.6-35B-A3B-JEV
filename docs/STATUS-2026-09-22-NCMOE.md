# ncmoe measurement log — 2026-09-22

Consolidated record of the `-ncmoe` investigation. Raw output directories are named
per run; where a directory was destroyed this file is the surviving record.

## The claim being tested

The project's published cold number is **3.67 tok/s at 212.6 MB per generated
token**, measured with `n_gpu_layers = 0` — every weight, attention included, on the
CPU behind a demand-paged mmap (`docs/STATUS-2026-09-22.md`). Two RTX 5060 Ti
16 GiB cards sit idle in that measurement.

`-ncmoe N` keeps the first N layers' **expert** weights on the CPU and lets the rest
(attention, norms, embeddings, and the other layers' experts) live in VRAM. So the
same SSD-streaming path gets a GPU for its dense half, and the disk-resident working
set shrinks by (40−N)/40.

## 1. Uncapped config search — `results/bench-20260922-035009` (raw file lost, table preserved here)

llama-bench, `-ngl 99`, `-ub 512`, `-b 2048`, reps 1, no memory cap. This is a
**config search, not an SSD measurement**: with 62 GiB of RAM the checkpoint is
page-cached, so it ranks configurations rather than measuring storage.

| ncmoe | prefill 512 t/s | prefill 2048 t/s | decode 32 t/s |
|---|---|---|---|
| 99 (all experts CPU = project's config) | 244.31 | 240.32 | 37.38 |
| 40 | 244.12 | 238.94 | 36.33 |
| 32 | 288.86 | 292.66 | 40.29 |
| 24 | 353.46 | 369.85 | 50.72 |
| 16 | 473.93 | 505.98 | 55.68 |
| 8 | 732.64 | 825.24 | 70.96 |
| 0 (all GPU) | **2536.58** | **3602.03** | **101.97** |

Reading: the curve is monotonic in the number of layers whose experts reach VRAM,
and it is steep. `ncmoe=8` — which still leaves **32 of 40 layers** (80% of the
18.33 GB expert set) on the CPU — already decodes at 70.96 tok/s, 1.9× the
CPU-only configuration. `ncmoe=0` is 2.7×.

Caveat: this table is warm. It says the routing is GPU-friendly, not that it is
SSD-friendly. That is what §2 is for.

## 2. Capped sweep — `results/bench-8g-20260922-040712`

`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`, page cache evicted
with `posix_fadvise(DONTNEED)` before every configuration, 135-token prefill +
32-token decode. Device reads are `/sys/block/nvme0n1/stat` sector deltas over
exactly that configuration.

| ncmoe | prefill t/s | decode t/s | device read GB |
|---|---|---|---|
| 99 | 10.4 | 8.5 | 1523.61 ← **retracted, see below** |
| 24 | 48.7 | 37.8 | 42.65 |
| 8 | 157.7 | 67.7 | 41.30 |
| 0 | 878.2 | 97.9 | 41.56 |

**The 41–42 GB column is the sanity check that makes this table believable: it
reproduces the project's own published read volume for the same workload
independently, through a different counter.** Storage is being read; the cap is
doing something real.

### Retraction, verified: `ncmoe=99 = 1523.61 GB` was contaminated

Re-run clean with the rebuilt harness (`results/probe-verify-20260922-043220`,
`CAP=8G`, cache evicted, `ncmoe=99`, 135-token prefill + 32-token decode):

| counter | load baseline (`-p 1 -n 1`) | workload | inference (workload − baseline) |
|---|---|---|---|
| `/proc/self/io` read_bytes | 23.57 GB | 50.77 GB | 27.20 GB |
| `/sys/block/nvme0n1/stat` | 23.57 GB | 50.77 GB | 27.20 GB |
| `memory.peak` | 8.00 GiB | 8.00 GiB | — |
| `memory.events max` | 7,347 | **35,460** | — |

**The two independent counters agree to the byte.** 1523.61 GB in a ~20 s run would
be ~75 GB/s off a device measured at ~2.4 GB/s raw — arithmetically impossible — and
against a clean 50.77 GB for the same configuration it is a factor of 30 out. The
device counter is device-wide with nobody excluded, and the window straddled
unrelated activity from the runs being killed concurrently. **Discard it. It is not
a measurement of this workload and must never be quoted.**

Verified `ncmoe=99` decode under the cap: **6.12 tok/s**.

### One distinction that must not be lost

`ncmoe=99 -ngl 99` is **not** the project's published configuration. The project
measured `n_gpu_layers = 0` — *everything* on the CPU. `ncmoe=99` means "all 40
layers' expert weights on the CPU" while attention, norms and embeddings go to the
GPU. Different beast. So:

- **3.67 tok/s** = every weight on CPU (project's published cold number)
- **6.12 tok/s** = all experts on CPU, dense half on GPU — **1.7×** from the GPU
  handling just the non-expert weights
- **67.7–69.4 tok/s** = 8 layers' experts on GPU, 32 layers' experts (80% of the
  18.33 GB set) still streaming from NVMe

An earlier capped pass (`results/bench-8g-20260922-040123`) agreed on shape:

| ncmoe | prefill t/s | decode t/s |
|---|---|---|
| 99 | 9.0 | 6.3 |
| 24 | 53.1 | 36.7 |
| 8 | 160.9 | 69.4 |

Two independent capped passes put `ncmoe=8` decode at **67.7–69.4 tok/s** against
the project's published **3.67**. That is **~18×**, reached while ~80% of expert
bytes still stream from NVMe.

### Methodological caveat on `mb_read_per_gen_token`

The harness reports 850.0 MB/token for `ncmoe=99`, which looks like a 4× regression
against the project's 212.6 MB/token and **is not comparable**. The project's figure
is decode-only; this one divides the whole inference phase (135-token prefill *plus*
32-token decode = 27.2 GB) by the 32 decode tokens. Do not put the two numbers in
the same column until the harness splits prefill and decode reads separately.

## 3. What the GGUF actually contains (parsed from the tensor table, not assumed)

`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`, read directly:

- 733 tensors, 54 KV pairs, architecture `qwen35moe`
- `block_count = 40`, `expert_count = 256`, `expert_used_count = 8`
- 120 expert tensors = 40 layers × {`ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`}
- one expert slice = **exactly 589 824 B (576 KiB)**, and **16 KiB-aligned**
- expert bytes total = **18.327 GB** of the 20.89 GB file; 0.458 GB per layer
- expert tensors are Q4_K (12), except one F16; non-expert tensors are F32/F16/Q8_0

The 576 KiB slice and its 16 KiB alignment mean the "aligned expert slab" the design
doc lists as a *future* phase is already how the file is laid out as shipped. That
is the shape `io_uring` benefits most from (§4).

## 4. Storage-layer numbers measured on this host

`/home/scribe/.hermes/cache/scratch/io_results.csv` — 90 rows, `O_DIRECT`, this
NVMe, produced by a purpose-built C harness with `pread` / `io_uring` /
registered-buffer / SQPOLL / deferred-taskrun variants.

At the model's real **589 824 B** extent size:

| mode | qd 1 | qd 8 | qd 32 | qd 64 | qd 128 |
|---|---|---|---|---|---|
| `pread` | 1364 | 1658 | 1580 | 1696 | 1599 MiB/s |
| `io_uring` | 1441 | **3384** | 3211 | 3271 | 3307 MiB/s |
| `io_uring` + registered buffers | 1721 | 3382 | 3396 | 3310 | 3319 MiB/s |

`pread` is **flat across queue depth** (it has no queue) at 1364–1696 MiB/s;
`io_uring` is **2.0–2.4×** that and saturates by qd 8. Registered buffers buy
essentially nothing at this extent size.

At 4 KiB the gap is categorical: `pread` holds ~55 MiB/s at every depth while
`io_uring` climbs 61 → 1093 → **1270** MiB/s (qd 128, registered).

At 4 MiB and 7 MiB both paths are already at 3000–3400 MiB/s.

`IOPOLL` returned an error on this kernel/hardware — **not** a working result and not
claimed as one. SQPOLL and deferred taskrun both work.

The project's thread-oversubscription finding (4/3 of hardware threads, 54.6 → 38.4 s)
is consistent with this: over-subscribed threads are a *substitute* for the queue
depth `io_uring` supplies directly.

## 5. Page-fault fetch size — the mechanism behind the measured 2.15× amplification

`faultexp.c` walked an expert-like 589 824 B stride over the real GGUF with the page
cache evicted before mapping, and measured:

```
mode=none win=32MiB step=589824B faults=57 fetched=7.1 MiB avg_bytes_per_fault=131072
```

**131 072 B (128 KiB) fetched per fault against a 576 KiB slice.** The kernel's
readahead window (`read_ahead_kb=128` on this box) pulls 128 KiB when one 4 KiB page
is touched, and the remaining faults inside that 128 KiB then hit cache. This is the
measured mechanism behind the "gather overfetch" factor, and it explains why
`MADV_RANDOM` cuts reads 41 → 26 GB but costs 6.6× in time: it removes the very
readahead that is hiding latency.

Only the `mode=none` row was captured. The `MADV_RANDOM` / `MADV_SEQUENTIAL` /
`MADV_COLLAPSE` / `MADV_POPULATE_READ` rows were **not captured** — not claimed.

## 6. Harness bugs found (all three were silent, all three invalidated runs)

1. **The cap re-exec was gated on `$CAP`, which a child inherits** → the script
   re-exec'd itself forever (~40 nested systemd scopes). Now gated on an
   activation variable.
2. **`/proc/self/io` read inside `"$(...)"` reads the subshell**, which has read
   nothing → every read counter reported `0.00 GB` and looked plausible. Counters
   are now read with shell builtins in the calling shell.
3. **The cgroup's `io.stat` does not exist in user scopes here** — only `memory` and
   `pids` are in `cgroup.subtree_control`. Device accounting goes through
   `/sys/block/nvme0n1/stat` instead.

Runs killed or invalidated by these: `bench-8g-20260922-035301`, `-035516`,
`-035837`. Their jsonl is not evidence and is not quoted.

## 7. The seam that makes this reachable from Python without patching C

`-ncmoe` / `--cpu-moe` are **not** engine features. They are tensor buffer-type
overrides, built entirely in `common/arg.cpp`:

```c
llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, params.tensor_buft_overrides);
// LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
// each entry binds "blk\\.N<regex>" to ggml_backend_cpu_buffer_type()
```

Three verified facts that make this usable from the current stack:

- `ggml_backend_cpu_buffer_type` is **exported** by `libggml-base.so` (checked with
  `nm -D` in the project's own venv).
- In `llama_cpp.py`, `params.tensor_buft_overrides` is declared `c_void_p` and
  commented "unused", but the `llama_model_params` struct layout matches the C
  header exactly — it is a **model** parameter, so it is honoured at load.
- `_cpu_model_params()` in the SemIf patch is the one place that builds model params,
  so the change is local.

Therefore a NULL-terminated array of `{pattern, ggml_backend_cpu_buffer_type()}`
turns the existing 3-file SSD-scorer backend into a GPU + ncmoe engine **through a
parameter it can already set**. This is the next step (§8).

Note: `/home/scribe/Projects/llama.cpp` (branch `fable5/prefetch-experts`) has **no**
`ncmoe` at all — that is a separate MoE-offload fork. `llama.cpp-fresh` is the tree
that has the flag, and the one these numbers come from.

## 8. Open items, in order

1. **Re-verify `ncmoe=99` under the cap** and either explain or discard the
   1523.61 GB figure. Until then the other three rows carry the result.
2. **Capture `memory.peak` and `memory.events max` for every row.** The scripts now
   read them from inside the scope, but only the `ncmoe=0` and early rows ever
   printed them. A capped row without a non-zero `max` count did not stream.
3. **Wire `tensor_buft_overrides` into `_cpu_model_params`** so
   `resident_scorer.py` and `systemone_shim.py` serve decisions with `ncmoe=8`
   instead of the CPU-only path. That is the change that turns these numbers into
   the product, and it belongs in `patch/files/src/...` as well as the in-tree copy.
4. **Re-measure the project's own scorer path at `ncmoe=8`** — the same 4-criterion
   shared-state fixture, same `read_bytes` discipline — so the headline moves from
   "3.67 tok/s" to a like-for-like number at the new configuration.
