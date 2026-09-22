# Qwen3.6-35B-A3B-JEV

Running **Qwen3.6-35B-A3B** as a JEV-style **typed-decision scorer** with its
expert weights served from **SSD** instead of resident RAM — on a machine that
cannot hold the checkpoint, via llama.cpp (no Apple Silicon lock-in).

Given a page state, a question, and a list of candidate options, it returns a
probability per option. One pass, one logit vector. It is a **classifier**.

    upstream engine:  SemIf (MIT) + a 3-file patch, in patch/
    measured on:      6-core / 12-thread host, 62 GiB RAM, NVMe, 8 GiB cgroup cap

---

## Read this first: what the numbers mean

**No tokens are generated. Nothing is decoded autoregressively.** The scorer runs
a *prefill* over a fixed prompt and reads the logits of the option-label tokens.
So every `tok/s` figure below is **prompt-prefill throughput** — token-forwards
per second — not generation speed. A 35B MoE *generating* text from SSD would be
a different and much worse story; that is not what this does.

The distinction is not pedantic, because prefill and generation are opposite
cases for SSD streaming:

| | experts touched per layer | SSD behaviour |
| --- | --- | --- |
| **prefill** (what this does) | most of 256 | SSD-hostile |
| **generation** | ~8 | easy |

The scorer sits in the *hostile* case and still lands at 23–29 token-forwards/s.
Presenting that as "27 tokens per second of output" would be a large
overstatement, so: **23/27/28 tok/s is per-pass prefill throughput.** End-to-end
per call, including the model load, it is **15.1 tok/s one-shot** and **23.0
resident** — and on a busy workstation, **~74 s per call**.

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
   earlier work concluded. Not in any of the twelve surveyed engines.
2. **`n_ubatch`, not `n_batch`, decides I/O.** A decode carrying more tokens than
   `n_ubatch` splits into physical ubatches, and *each* re-walks all 40 layers and
   re-reads their experts. llama.cpp's default 512 silently split the 908-token
   group in two. Setting `n_batch = n_ubatch = sequences × 512` took the decode
   from 67.0 → 42.0 GB.
3. **Batching the branches at all.** `n_seq_max = 1` made the context
   single-sequence, so `score_shared` serialised branches via save/restore — five
   full model passes for four criteria.

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

Twelve existing MoE-offload / SSD-streaming engines were surveyed
(`docs/OFFLOAD_PROJECTS_ANALYSIS.md`). Every one that works uses `pread` against a
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

- **The largest unattacked win is residency, and it does not exist.** `cli.py` is
  one-shot (`load → score → exit`), so the 20.9 GB / ~19.8 s load — **33% of
  per-call wall clock** — is paid on every call, and nothing carries over: calls 2
  and 3 read the *same* 61.6 GB as call 1. A resident process gives 3 calls in
  139.0 s against 180.0 s one-shot, **33.8% saved per call**, widening with call
  count. A resident scorer is the next real move.
- **Read amplification** (41 GB against an 18.33 GB expert set) needs a
  weight-access rewrite: `pread` over exact expert extents with the engine owning
  a bounded slot pool. That is the proven seam from the prior art.
- `--max-tokens` has not been measured.

## Repository layout

    SETUP.md                        install, run, and invoke it — start here
    run_semif_35b_ssd.sh            the operational runner (the tested path)
    patch/                          the SemIf change: diff + complete files + apply.sh
    docs/
      SEMIF_LLAMACPP_SSD.md         the main lab notebook, every measurement
      OFFLOAD_PROJECTS_ANALYSIS.md  the survey of twelve engines, and the rankings
      qwen3.6-35b-a3b-ssd-offload.md  the original design note
      DEVELOPMENT_LOG.md            session handoffs, including failed paths
    scripts/
      README.md                     which probe answers which question
      fetch_prior_art.sh            re-clone the twelve surveyed engines
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
