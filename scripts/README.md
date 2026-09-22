# Measurement scripts

Every number in `docs/` came from one of these. They are the evidence, not
tooling — kept because a claim you cannot re-run is a claim you have to take on
faith, and several of these were written to *disprove* a hypothesis that had
already been written down.

Nothing here is needed to use the scorer. `../SETUP.md` and
`../run_semif_35b_ssd.sh` are the operational path.

## Two rules these all obey

**Measure `read_bytes`, never wall-clock.** A warm run can be *faster* than a
disk run. Wall-clock cannot tell you which one you got; `/proc/<pid>/io
read_bytes` can. Every script here samples it.

**Evict the page cache first, or you measure nothing.** A cgroup cap does not
re-charge a page that is already cached, so a fresh capped scope reuses a warm
file for free and the cap never binds — measured: `memory.peak` 632 MB,
`read_bytes` 0. `posix_fadvise(POSIX_FADV_DONTNEED)` is unprivileged and drops
the file's clean pages. Confirm with `fincore "$GGUF"` (you want `PAGES 0`).

## Probes

| File | Answers |
| --- | --- |
| `probe_phases.py` | Attributes reads **by phase** (prefill / suffix_forward). This is how the two real wins were found — neither was visible in the run output. |
| `probe_ubatch.py` | Whether the ~41 GB decode is a second ubatch sweep. It is not: 908 tokens fit one 4096 ubatch, and `SEMIF_LLAMA_UBATCH=8192` changed nothing (41.93 GB / 54.62 s). Reads `SEMIF_INPUT`, `SEMIF_THREADS`. |
| `probe_prefetch.py` | Layer-ordered prewarmer with a GGUF tensor-table parser (`gguf_layout`, `expert_extents`). Modes: `advise` (POSIX_FADV_WILLNEED) and `read` (real `preadv`). Both are negative results — see below. |
| `probe_madvise.py` | `MADV_RANDOM`: cuts reads 41 → 26.0 GB but costs 6.6× in time. |
| `probe_load.py` | The load phase alone. Found that `llama_model_load_from_file` reads the entire 20.9 GB file even with mmap and repack off. |
| `probe_lifecycle.py` | One-shot CLI vs resident scorer: the same 61.6 GB is re-read every call. |
| `oneshot_wrapper.py` | Runs the shipped CLI in-process via `runpy` so the process's block-device reads are still readable after exit. The one-shot path is `load → score → exit`, so `/proc/<pid>/io` is gone by the time it returns otherwise. |

## Runners

Each wraps a probe in the evict-then-cap dance and a read-bytes sampler.

| File | Knobs |
| --- | --- |
| `run_probe.sh` | generic wrapper |
| `run_n.sh` | criteria-count scaling. `IN_FILE`, `THREADS`. This is what produced 4/8/16-criteria and reproduced the `llama_decode failed` bug at N=16. |
| `run_contention.sh` | `CAP`, `THREADS`, `LOAD`. Starts `LOAD` busy loops as competing workstation work, runs the probe, kills them. The thread default is worth *more* under load (2.1×) than idle (1.4×). |
| `run_lifecycle.sh` | `VARIANT=oneshot` (3 sequential CLI calls) or `resident`. Evicts **once**, deliberately not between calls — the question is whether anything carries over, and evicting between calls would assume the answer. |
| `run_ubatch.sh`, `run_madvise.sh`, `run_prefetch.sh`, `run_load.sh` | one per probe above |

## Older shell measurements

`measure_io.sh` (capped vs warm, sampling `read_bytes`), `measure_evicted.sh`
(evict immediately before each run), `measure_one.sh` (one criterion under the
cap), `measure_batched.sh` (the batched-branch path, evicted + capped).
Superseded by the `run_*.sh` wrappers but kept — these are what established the
evict-then-cap method in the first place.

## The parser

`probe_prefetch.py` contains a small GGUF tensor-table reader. It is how the
expert layout was established: **40 layers × 3 extents ≈ 151 MB each = 18.33 GB
of routed experts in a 20.89 GB file.** That number is the denominator for every
read-amplification claim in `docs/`, so it is worth being able to re-derive.

## Two negative results worth keeping

Both are recorded in full in `docs/SEMIF_LLAMACPP_SSD.md` §6c, and both look
like they should work:

- **`POSIX_FADV_WILLNEED` is silently capped.** The kernel limits an advised
  readahead window to a small multiple of `read_ahead_kb` (128 KB here), so
  16 MB chunks became ~128 KB requests and moved `read_bytes` by 0.14 GB. The
  advice was never acted on — this is not "prefetch is futile", it is "the
  syscall did nothing".
- **A real `preadv` prewarmer made it worse** — 42.0 → 56.1 GB, 56.0 → 57.56 s.
  Prefetched clean unreferenced pages are the first reclaim victims under a
  memory cap, so they were evicted before use and faulted in twice.
