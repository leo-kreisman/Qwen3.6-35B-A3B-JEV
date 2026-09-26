# Qwen3.6-35B-A3B-JEV

Latest experiment: [overlapping SSD reads and CPU computation](docs/RESUMABLE-PERFORMANCE-RESULT.md) reduces native tile replay time by about 31% with identical outputs and read bytes. Full-model speedup remains unproven; this is an opt-in, single-layer experiment.

[CPU tile bypass](docs/RESUMABLE-BYPASS-RESULT.md) skips the original expert calculation at one layer and tests exact decisions with original weight pages protected. This is an opt-in, version-sensitive graph experiment.

Follow-up: [native quantized executable tiles](docs/RESUMABLE-NATIVE-TILES-RESULT.md) now feed live model evaluation under a strict output-equality gate; checkpoint and segmented-attention behavior are tested separately.

2026-09-25 experiment: [resumable CPU decisions and executable SSD tiles](docs/RESUMABLE-DECISION-RESULT.md). Native schedule tests reduce repeated reads; lossless tiles are implemented. Numerical decision changes keep scheduling experimental.

2026-09-25 experiment: [profile-guided tiered expert storage](docs/TIERED-STORAGE-RESULT.md)
is implemented and tested. It saved reads but failed the quality gate without a
measurable latency gain, so it remains opt-in.

Running **Qwen3.6-35B-A3B** as a JEV-style **typed-decision scorer** with its
expert weights served from **SSD** instead of resident RAM — on a machine that
cannot hold the checkpoint, via llama.cpp (no Apple Silicon lock-in).

Given a page state, a question, and a list of candidate options, it returns a
probability per option. One pass, one logit vector. It is a **classifier**.

    upstream engine:  SemIf (MIT) + a 3-file patch, in patch/
    measured on:      6-core / 12-thread host, 62 GiB RAM, NVMe, 8 GiB cgroup cap

---

## Read this first: what the numbers mean

**The scorer generates no tokens. Nothing is decoded autoregressively.** It runs
a *prefill* over a fixed prompt and reads the logits of the option-label tokens.
So every `tok/s` figure below is **prompt-prefill throughput** — token-forwards
per second — not generation speed.

The distinction is not pedantic, because prefill and generation are opposite
cases for SSD streaming:

| | experts touched per layer | SSD behaviour |
| --- | --- | --- |
| **prefill** (what the scorer does) | most of 256 | SSD-hostile |
| **generation** | ~8 | easy |

The scorer sits in the *hostile* case and still lands at 23–29 token-forwards/s.
Presenting that as "27 tokens per second of output" would be a large
overstatement, so: **23/27/28 tok/s is per-pass prefill throughput.** End-to-end
per call, including the model load, it is **15.1 tok/s one-shot** and **23.0
resident** — and on a busy workstation, **~74 s per call**.

### Generation, measured (2026-09-22)

Generation was previously unmeasurable here because **no generation code
existed** — `--max-tokens` is an *input* prompt cap, not a generation length.
`scripts/probes/probe_decode.py` closes that, and the first real decode numbers
are:

| run | tok/s | MB read per token |
| --- | --- | --- |
| cold, cache evicted, **8 GiB cap verified binding** | **3.67** | **212.6** |
| warm, checkpoint page-cached | **10.68** | 0 |

So the honest reading: **the scorer path is the fast path.** A typed decision
costs one prefill whether it has 2 options or 255; generating the same answer as
text would pay 212.6 MB *per token*. That is why `resident/systemone_shim.py`
implements the TypeSafe System One contract directly instead of prompting a
generation — point `TYPESAFE_BASE_URL` at it and the local SSD-served model
becomes a drop-in provider.

The predicted ceiling was 2.4 GB/s ÷ 566 MB/token ≈ 4.2 tok/s; cold lands at 3.67.

---

## Results

Four criteria against one shared 135-token page state, 8 GiB cap, page cache
evicted first so the reads are real. Verified with `/proc/<pid>/io read_bytes`,
never wall-clock — a warm run can be *faster* than a disk run, and that inversion
is the tell.

| Configuration | Wall clock | Process reads |
| --- | --- | --- |
| as shipped upstream (branches serialised, `n_ubatch` 512) | 72.8 s | 112.8 GB |
| + batched branches, `n_batch = n_ubatch = seq × 512` | 56.6 s | 60.2 GB |
| + thread oversubscription **(this repo)** | **38.4 s** | **61.6 GB** |

**1.9× faster, identical probabilities.** The answers never changed — streaming
costs time, not correctness. Reads after the fix are ~20.9 GB load + ~41 GB
decode; the checkpoint itself is 20.89 GB, of which **18.33 GB is routed experts**
(40 layers × 3 extents × ~151 MB).

> The read column is not monotonic across the last two rows, and that is not a
> regression: **the thread change does not move read volume at all** (flat
> 41.2–42.1 GB across the whole sweep, which is the evidence the decode is not
> purely I/O-bound). 60.2 and 61.6 GB are the same quantity measured on different
> runs, within run-to-run variance. Only the wall clock improves.

The three changes, in order of size:

1. **Thread oversubscription** (−29.6%, 54.6 → 38.4 s, lossless, one parameter).
   A thread waiting on an expert page fault is **not runnable**, so a disk-served
   decode wants *more* runnable threads than the CPU has, to keep the device queue
   fed. Swept cold over one fixed 908-token pass: 6 threads 54.62 s, 8 → 50.35,
   12 → 47.87, **16 → 38.44**, 20 → 38.42. Knee at **4/3 of hardware threads**.
   Read volume is flat across the sweep (41.2–42.1 GB) — which is itself the
   evidence that the decode is **not** purely I/O-bound, contrary to what the
   earlier work concluded. Not in any of the surveyed engines.
2. **`n_ubatch`, not `n_batch`, decides I/O.** A decode carrying more tokens than
   `n_ubatch` splits into physical ubatches, and *each* re-walks all 40 layers and
   re-reads their experts. llama.cpp's default 512 silently split the 908-token
   group in two. Setting `n_batch = n_ubatch = sequences × 512` took the decode
   from 67.0 → 42.0 GB.
3. **Batching the branches at all.** `n_seq_max = 1` made the context
   single-sequence, so `score_shared` serialised branches via save/restore — five
   full model passes for four criteria.
4. **Residency — paying the load once** (`resident/`). `cli.py` is one-shot, so the
   ~19.8 s load is paid on every call. Keeping the process alive is worth **~59 →
   ~38.5 s per call (1.5×)**, and worth *only* the load: the decode read is flat at
   41.07 / 41.09 / 41.08 GB across three calls, so there is no inter-call reuse to
   collect. Item 1 and the read model are two faces of one device: at fixed tokens
   more threads move time but not bytes (latency-bound), while at fixed threads
   time tracks bytes at ~1.03 GB/s (bandwidth-bound).

Caveat worth stating plainly: batching is right **only** in the I/O-bound regime.
`qwen35moe` is a hybrid model whose `llama_memory_hybrid::seq_cp` **aborts**, so
branches cannot share one prefilled prefix KV; each carries the whole prompt, 908
token-forwards instead of 503. On a host where the checkpoint fits in RAM the
batched path is **2× slower** (18.53 s vs 9.29 s warm).

---

## Throughput

| Criteria | Passes | Token-forwards | Time | Read | tok/s | s / criterion |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 908 | 39.4 s | 41.0 GB | 23.0 | 9.85 |
| 8 | 1 | 1,816 | 66.7 s | 64.1 GB | 27.2 | 8.34 |
| 16 | 2 | 3,632 | 127.1 s | 128.3 GB | 28.6 | 7.94 |

Batching pays a little per criterion, but past `n_seq_max = 8` each group is a
full extra pass — cost grows *faster* than the criterion count. Read
amplification also grows with tokens per pass: 2.3× at 908 tokens, 3.5× at 1,816,
against the 18.33 GB expert set.

## Under load — the deployment case

This is meant to be invoked by a coding agent on a **shared** workstation, and
the thread default is worth *more* under contention, not less:

| Competing busy threads | 16 threads | 6 threads |
| --- | --- | --- |
| 0 (idle) | 38.4 s | 54.6 s |
| 6 | 55.8 s | 83.5 s |
| 12 | **74.2 s** | **155.3 s** |

2.1× better under load, against 1.4× idle. Reads stay ~41 GB either way —
contention costs time, not bytes. **Budget ~74 s per call on a busy box.**

## Memory cap

Not monotonically better, and **12 GiB is a trap**:

| Cap | Read | Time |
| --- | --- | --- |
| 8 GiB | 41.0 GB | 38.4–40.3 s (stable ±2.5%) |
| 12 GiB | 30.2 GB | 49.98 s then 34.91 s (**bimodal ±18%**) |
| 16 GiB | 19.3 GB | 29.3 s |
| 24 GiB | 0.0 GB | 16.0 s (fully cached) |

Reads fall with the cap; time does not. Size with a clear margin, **never at the
fit boundary**.

---

## What was measured and rejected

Twelve existing MoE-offload / SSD-streaming repositories were cloned and surveyed
(`docs/OFFLOAD_PROJECTS_ANALYSIS.md`); deduplicating two forks and one stub that is **~9
distinct engines**. Every one that works uses `pread` against a
per-expert offset table with a bounded slot pool; the mmap-based ones are the slow
ones. Four plausible optimisations were tested here and are **dead ends** — they
are written down so nobody re-tries them:

- **`POSIX_FADV_WILLNEED` does nothing.** The kernel caps an advised readahead
  window to a small multiple of `read_ahead_kb` (128 KB here), so 16 MB chunks
  became ~128 KB requests: read_bytes moved by 0.14 GB. Not "prefetch is futile" —
  the syscall was never acted on.
- **A real `preadv` prewarmer made it worse**: 42.0 → 56.1 GB and 56.0 → 57.56 s.
  Under a memory cap, clean unreferenced prefetched pages are the *first* reclaim
  victims, so they get evicted before use and faulted in twice.
- **`MADV_RANDOM`** cuts reads 41 → 26.0 GB but costs 6.6× in time.
- **The 41 GB decode is not a second ubatch sweep.** 908 tokens fit one 4096
  ubatch; `SEMIF_LLAMA_UBATCH=8192` changed nothing (41.93 GB / 54.62 s).

Also retracted: moving the GGUF to NVMe was ranked as the cheapest win and was
**already taken** — the file was already on the NVMe doing 2.4 GB/s raw
(`dd iflag=direct`) while scoring gets ~1,069 MB/s. Storage is not the bottleneck.

## What is not solved

- **Read amplification** is the remaining cost, and it is two ~2.1× factors, not
  one. For the 4-criterion fixture the routing selects about **9.3 GB** of expert
  bytes (top-8 of 256, from the GGUF tensor table); the same run reads **19.3 GB**
  at a 16 GiB cap (gather overfetch) and **41.0 GB** at 8 GiB (cap-driven
  intra-pass re-read). Removing both needs the pass to visit experts in expert
  order — fetch each selected slice once and use it for every token that wants it,
  working set a few slices rather than the whole set. That is the proven seam from
  the prior art: `pread` over exact expert extents with the engine owning a bounded
  slot pool. It is a decode rewrite, not a flag; llama-cpp-python ships prebuilt
  `.so` files, so it is a rebuild-and-patch project. See `resident/README.md`.
- **`direct_io` is closed, not open.** It was listed as an untested load mode. Under
  an 8 GiB cap it cannot load at all: `O_DIRECT` reads every tensor into allocated
  buffers rather than mapping them, needs 20.88 GB of anonymous RAM, and is
  SIGKILLed during `load_all_data` before scoring anything.
- `--max-tokens` (the *input* prompt cap) has not been tuned. It is not a
  generation length — see "Generation, measured" above.

## Residency — solved

`cli.py` is one-shot (`load → score → exit`), so the 20.9 GB / ~19.8 s load —
**33% of per-call wall clock** — was paid on every call and nothing carried over:
calls 2 and 3 read the *same* 61.6 GB as call 1. `resident/resident_scorer.py`
keeps the process alive, and it is worth exactly the load and no more:

| | load | decode read | wall |
| --- | --- | --- | --- |
| one-shot, per call | 18.9–20.4 s | 41.07 GB | ~59 s |
| resident call 1 | 18.92 s | 41.07 GB | 40.27 s |
| resident calls 2–3 | — | 41.09 / 41.08 GB | 38.52 / 38.90 s |

**~59 s → ~38.5 s per call, 1.5×.** The decode read is *flat* across calls —
41.069 / 41.090 / 41.083 GB — so there is zero inter-call reuse, and that is
structural: a pass touches ~41 GB against an 8 GiB cap, so it evicts its own early
pages before it finishes, and call 2 starts at layer 0 where nothing is warm.
Scoring is unchanged (SemIf's own path, called as a library) and decision-identical.
Batching against the shared state is the other available win: per-criterion read
falls 17.5 GB at 1 criterion to 6.9 GB at 16, for up to **2.5× less I/O and 2.0×
less time per criterion**, with no engine change.

## Serving typed decisions — the System One endpoint

`resident/systemone_shim.py` serves **JEV's own contract** locally:

    POST /v1/systemone   {"model", "state", "questions"} -> {"model", "answers", "usage"}
    GET  /v1/models

All three question types are implemented, each returning the shape the contract
specifies: `noul` (a float, no confidence, no probabilities), `choice`
(winner + confidence + probabilities over the caller's own keys, up to 255), and
`score` (a probability-weighted mean over an ordered level array, plus a `legend`
keyed by level index). Invalid requests return **422 `invalid_request`**;
`usage.output_tokens` is **0 by construction**, because nothing is generated and
every string returned is the caller's own text echoed back. Extras live under a
non-conflicting top-level `local` key, including `cache_state`, which says
whether the call was served from page cache or from NVMe.

Point the documented extension variable at it and nothing downstream needs a fork:

    python resident/systemone_shim.py --gguf <gguf> --model <tokenizer-dir> --port 8123 &
    export TYPESAFE_BASE_URL=http://127.0.0.1:8123

Calls are **serialised** — one llama.cpp context, one call at a time — which is
the honest shape of a single-box SSD-streamed model.

## Repository layout

    SETUP.md                        install, run, and invoke it — start here
    run_semif_35b_ssd.sh            the operational runner (the tested path)
    run_resident_35b_ssd.sh         resident runner: load once, serve many calls
    resident/
      resident_scorer.py            the resident scorer (library use or --serve)
      systemone_shim.py             the TypeSafe System One endpoint (typed answers)
      test_systemone_shim.py        end-to-end test for it: 26 contract checks
      README.md                     what residency buys, and what it does not
    patch/                          the SemIf change: diff + complete files + apply.sh
    docs/
      SEMIF_LLAMACPP_SSD.md         the main lab notebook, every measurement
      OFFLOAD_PROJECTS_ANALYSIS.md  the survey of the cloned engines (~9 distinct), rankings
      qwen3.6-35b-a3b-ssd-offload.md  the original design note
      DEVELOPMENT_LOG.md            session handoffs, including failed paths
    scripts/
      README.md                     which probe answers which question
      fetch_prior_art.sh            re-clone the 12 surveyed repositories
      sweep_resident.sh             the serial residency sweep (one process per knob)
      analyze_sweep.py              fit the read model out of the ledger lines
      probes/                       the measurement scripts (19 files)
    examples/                       input JSONL fixtures
    results/                        output JSONL, the evidence for the tables above

## Quick start

```sh
git clone https://github.com/leo-kreisman/Qwen3.6-35B-A3B-JEV
cd Qwen3.6-35B-A3B-JEV
$EDITOR SETUP.md      # 5 steps; needs the 20.9 GB GGUF and ~45 GB free disk
```

Sanity check that does not need the disk path at all:

```sh
cd semif && .venv/bin/python -m pytest -q     # expect: 90 passed, 3 skipped
```

## License

MIT (`LICENSE`). The scorer is a patch to **SemIf** (MIT, Copyright (c) 2026
TheoLeeCJ), which is not redistributed here — see `THIRD_PARTY_NOTICES.md` for
what is derived, what is depended on, and what is merely surveyed. Model weights
and the tokenizer are fetched separately and carry their own licenses.
