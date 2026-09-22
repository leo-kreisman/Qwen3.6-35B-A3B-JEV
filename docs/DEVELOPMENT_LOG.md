# DEVELOPMENT_LOG

## Session Handoff — 2026-09-22 (d) — the first decode number, and a working System One endpoint

### Goal

Four things, in the order they were asked for: (1) **measure generation**, which had never
been done; (2) the **research mandate** — decompose the architecture into ~10 parts and find,
for each part, where the same primitive is solved better elsewhere and what is importable;
(3) **use JEV**, i.e. wire the typed-decision path up so the cheap path is actually reachable;
(4) do (3) **properly, not crude** — and do not stop until it works end to end.

### Failed Paths

1. **The `subprocess.PIPE` deadlock — the whole "shim never became healthy" failure.**
   `test_systemone_shim.py` started the shim with `stdout=subprocess.PIPE` and drained that
   pipe only *after* `server.poll()` returned non-None. The checkpoint load prints ~800 lines
   of llama.cpp loader output, which filled the 64 KiB pipe buffer, so the shim **blocked on
   write before it ever bound the socket** — and the process therefore never exited, so the
   drain never happened either. It presented as "shim never became healthy" plus a **masked
   exit 0** (the run was piped through `| tail -70`). Diagnosis took a foreground run with
   the shim's own output going to a file, where it loaded normally in ~3 s. Fix: log to a
   file, `tail` it only on failure, and close it in `finally`.
2. **`ThreadingHTTPServer` over a non-thread-safe llama.cpp context** — the first draft of
   the shim. One context, many request threads. Fixed with a module-level `call_lock` around
   `scorer.call(rows)`, and the endpoint now advertises that calls are **serialised**.
3. **Two contract shapes I guessed instead of reading.** `score.criteria` is an **ordered
   array** of level descriptions, not a numeric-keyed map; and `state` is **not necessarily a
   string** — it may be a string, an object, or an array of strings. Both were wrong in the
   first draft and both are now handled, with the array form treated as canonical.
4. **`--max-tokens` was called "the last unmeasured lever" for decode. False.** Grep showed
   **no generation code existed in the project at all**; `--max-tokens` is an *input* prompt
   cap. Generation was not under-measured, it was *unreachable* — there was no decode loop.
5. **My "arithmetically impossible" claims about Edge0's prefill figure — retracted.**
   3,300 tok at 113 tok/s is 29.2 s, so even a full 20.88 GB read is 0.72 GB/s, ~7× under an
   M4 Pro SSD. Corrected in `DEVELOPMENT_LOG.md` and `SEMIF_LLAMACPP_SSD.md` §4. Both numbers
   are also on a 24 GB page-cached box, so the honest like-for-like is **our warm 10.68 vs
   Edge0's 14.9 — 1.4×, not 3.5×**.

### Final Solution

**1. Generation, measured — `scripts/probes/probe_decode.py` + `run_decode.sh`.** Cold, page
cache evicted, under an **8 GiB cap verified binding** (`memory.peak` exactly 8 GiB,
`memory.events max` 25,499): **3.67 tok/s at 212.6 MB per generated token**; prefill 25 tok in
7.52 s reading 9.82 GB; load 19.87 s; 44.32 GB total process read. Warm, no cap: **10.68 tok/s,
0 bytes read**. The 4.2 tok/s ceiling I had computed from 2.4 GB/s ÷ 566 MB/token was sound and
the measurement lands under it.

**2. The decomposition — `docs/ARCHITECTURE-DECOMPOSITION.md`.** Eleven parts, each with
Primitive / Who else solves it / Best-in-class / Import / Evidence. The load-bearing findings:
io_uring qd=128 at **3,807 MiB/s** vs sync `pread` 111 MiB/s — and our 4/3-nproc result is the
*substitute* for missing queue depth, not a law; **W-TinyLFU** (ARC is IBM-patented, excluded);
the **QStore/ZipNN** correction that our rANS dead-end tested the wrong stream (the 4-bit
quants are incompressible, the **fp16 scales/mins are not**); and **speculative verification**
(SpecMoE ~2.25× on SSD, 76.73% transfer reduction) as the highest-leverage import — it attacks
the measured 212.6 MB/token directly and **none of the surveyed engines does it**. Plus the
necessary counterweight: SSD offload costs up to **12× the per-token energy** of HBM.

**3. A working System One endpoint — `resident/systemone_shim.py`.** Serves `POST /v1/systemone`
and `GET /v1/models` with all three question types in their exact contract shapes (`noul`
returns a bare float with no confidence; `choice` returns a caller key + probabilities; `score`
returns a probability-weighted mean over ordered levels + `legend` keyed by index-as-string).
422 `invalid_request` on invalid input, `usage.output_tokens` **0 by construction**, extras
under a non-conflicting `local` key. `resident/test_systemone_shim.py` passes **26/26** checks
against the real checkpoint, including a semantic winner check (`e1`, the empty destination
input), all four validation cases, and `state` as string, object and array.

**Why this is the answer to "JEV is to make it faster":** a decision is **one prefill pass**
whose cost does not grow with the option count, while generating the same answer would pay
**212.6 MB and ~0.27 s per token**. The scorer path is not a compromise against generating —
it is strictly the cheaper route to the same typed answer.

### Unresolved

- **We still do not beat Edge0.** 10.68 warm vs 14.9 — a **1.4×** gap, and it is an
  **algorithm** (prerouter + fixed staged slots), not a platform, so it ports to llama.cpp.
  Not started.
- **Prefill prefetch is unsolved everywhere.** Every published prefetcher (GrASP, Pythia,
  SeLeP) is decode-time; our expensive case is prefill, where a one-token lookahead is
  useless. The prior art is silent on it.
- **Nothing is pushed.** The working tree has the new probe, the shim, the test and the doc
  corrections; no commit or push was made this session.
- **The shim has no auth.** Errors are modelled (401/429/529 documented) but unimplemented;
  it binds `127.0.0.1` only, which is the reason it is acceptable for now.

## Session Handoff — 2026-09-22 (c) — residency, and pricing the read amplification

### Goal

Branch `resident-expert-serving`: implement the serving path that "leaves SemIf
behind", and improve the measured results under the standing premise — **8 GiB RAM
plus SSD**, no Apple Silicon lock-in.

### Failed Paths

1. **The duplicate-expert-storage hypothesis is dead.** `scripts/probes/gguf_tensor_layout.py`
   reads only the header and tensor table: 733 tensors, extents sum to 20.88 GB against a
   20.89 GB file, 0.0 MB unaccounted, **0 identical ranges and 0 partial overlaps**, and
   only three routed suffixes per layer (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`).
   So the amplification is not double-stored weights, and `ffn_gate_up_exps` is an internal
   fused *view*, not a second copy. Cost of learning this: one probe, plus a count printed
   as `-40` because `"exps"` and `"shexp"` are distinct substrings, so the two filters do
   not nest — read `len(experts) - len(shared)` and it looks negative.
2. **Inter-call page reuse does not exist and cannot be engineered into existence.**
   Three resident calls read **41.069 / 41.090 / 41.083 GB** — flat to 0.05%. The reason is
   structural, not a tuning miss: a pass touches ~41 GB against an 8 GiB cap, so it evicts
   its own early pages before it finishes, and call 2 then starts at layer 0 where nothing is
   warm. Any plan that assumes residency compounds must die here.
3. **`direct_io` is closed, not merely untested.** Listed as an open lever in two earlier
   sessions. Under the 8 GiB cap it is **SIGKILLed during load** (exit 137): `O_DIRECT` reads
   every tensor into allocated buffers instead of mapping them, needing
   `CPU model buffer size = 19914.65 MiB` (20.88 GB) of anonymous RAM, and it dies in
   `load_all_data` before writing a single ledger line. It is a load mode for hosts that can
   hold the model, not for this regime.
4. **`use_extra_bufts` (repack) stays dead** at this size — it needs the model resident.
5. **`read_ahead_kb` stays dead** — WILLNEED is capped at ~128 KB; moved `read_bytes` by
   0.14 GB. Re-confirmed as not worth retrying.
6. **My own first read model was wrong.** "18.33 GB fixed sweep + 25.4 MB/token" rested on
   two points and on the wrong token denominator. `branch_logits_batched` gives every branch
   the *whole* prompt as its own sequence, so the honest counter is `batched_tokens`
   (= suffix + prefix × rows), not `true_suffix_tokens`. Refit over four single-pass runs:
   **read = 13.8 GB + 26.9 MB × token.**

### Final Solution

A resident scorer plus a strictly serial sweep, both measured cold under
`systemd-run --scope -p MemoryMax=8G -p MemorySwapMax=0` with the checkpoint evicted
once beforehand via `posix_fadvise(DONTNEED)`.

**Residency buys the load, and exactly the load.** ~59 s → ~38.5 s per call (**1.5×**);
`process_read_bytes` after call 1 was 61.96 GB, reproducing the published 61.6 GB, so the
instrument agrees with the baseline it measures. Scoring is SemIf's own path called as a
library — argmax and `full_vocab_argmax_id` identical on all four fixtures.

**One correction I made to my own account, after measuring instead of asserting.**
I first wrote here that probabilities differed from the committed baselines because those
ran `threads=6` and this runs 16. **That is wrong.** The `threads=16` output is
*bit-identical* to `results/decisions.batched.jsonl` on all four prompts, all 17 printed
digits — thread count does not move these logits at all. The only spread in the whole set
is *batched* vs the pre-batching single-sequence path: at most **0.001%** on the winning
probability, no decision changed. That difference is the earlier batching patch, not this
work, but this path depends on batching so it is disclosed rather than implied. And the
lossless claim is weaker than it sounds: all four fixtures sit at p ≈ 0.99998 for the
winner, so a near-tie fixture would be the real test and there is none in `examples/`.

**Batching is the other available win, and it is free.** Per-criterion read falls
17.5 → 10.3 → 8.0 → 6.9 GB for 1 / 4 / 8 / 16 criteria (up to **2.5× less I/O, 2.0× less
time per criterion**), because the marginal token gets *cheaper* as the batch grows
(35.9 → 25.4 MB/token) — the fixed term is genuinely shared. It needs `SEQ_MAX` to cover the
batch: a 16-criterion batch at `SEQ_MAX=8` pays two full sweeps.

**`SEQ_MAX ≥ batch width` is correct but small:** 128.3 → 110.2 GB and 127.1 → 119.96 s
(**1.06×**) — the saving is exactly one sweep's worth of bytes, but the single 8192-wide
ubatch runs at 0.92 GB/s against 1.01 GB/s, so 14% of the bytes buys 6% of the time.

**The amplification is two ~2.1× factors, and the cap table separates them** (top-8 of 256,
confirmed from the GGUF): the routing selects ~**9.3 GB** of expert bytes for the 4-criterion
fixture; the same run reads **19.3 GB** at a 16 GiB cap (gather overfetch) and **41.0 GB** at
8 GiB (cap-driven intra-pass re-read). Removing both is worth **2.1× conservatively, ~4.4× if
overfetch also falls** — at the same 8 GiB, with no extra RAM. It needs an expert-ordered
decode: group the batch's tokens by selected expert, fetch each slice once, use it for every
token that wants it, drop it.

### Unresolved

- **The expert-ordered decode is not started.** It is the only remaining lever on the read
  axis, and it is a decode rewrite, not a flag: llama-cpp-python 0.3.35 ships prebuilt `.so`
  files, so it is a rebuild-and-patch project. Every working engine in the survey does
  exactly this; none of them is a llama.cpp flag.
- **The ~2.1× overfetch is inferred from the cap table, not directly observed.** Confirming
  it needs per-layer read tracing.
- **A smaller quant is an untested lever with a real number behind it.** The expert set
  scales with bpw and the amplification is cap-driven, so a ~25% smaller expert set should
  cut reads ~25–33%. Not tested: it trades answer quality, and that is the user's call.
- `--max-tokens` has never been measured.
  **Correction (2026-09-22, session d):** this was written as if `--max-tokens` were the
  remaining *generation* lever. It is not — it is an **input prompt cap**. Generation was
  unmeasured because **no generation code existed** in the project, not because a knob was
  unturned. It is now measured: **3.67 tok/s cold at 212.6 MB/token under a binding 8 GiB
  cap, 10.68 warm** (`scripts/probes/probe_decode.py`).
- **Committed on the branch only — not pushed.**

## Session Handoff — 2026-09-22 (b) — publishing the repo: setup.md, patch, prior art

### Goal

Publish all of it — a `setup.md` an agent can follow, the scripts, and every research file —
to `https://github.com/leo-kreisman/Qwen3.6-35B-A3B-JEV`.

### Failed Paths

1. **`gh` is not installed**, so there was no CLI to create or inspect the repo.
   Not a blocker in the end: the repo already existed and was **empty**, and
   `git-credential-manager` is configured globally (ten repos already push to
   `leo-kreisman`), so plain `git push` was the whole answer.
2. **My first reachability check lied to me.** `git ls-remote … | head -5; echo $?` reports
   *head's* exit status, which is always 0, so the empty output was ambiguous between "empty
   repo" and "error swallowed". Re-run **without the pipe**: exit 0, no stdout, no stderr —
   that is the signature of an existing empty repo. Check the thing, not the pipe.
3. **`results/decisions.one.jsonl` is not a result.** Its keys are
   `id/options/question/state` — it is the one-criterion *input* fixture, and it had been filed
   with the outputs. Moved to `examples/` after checking every file's keys rather than its name.
4. **Vendoring `semif/` was the obvious move and the wrong one** — see below.
5. **Git identity is configured nowhere** (`git config --list --show-origin` → no
   `user.name`/`user.email`), so the first commit would have failed outright. Took the identity
   from `MiniCPM5-2B`'s *local* config rather than inventing one.

### Final Solution

**SemIf is published as a patch, not a fork.** `semif/` is a clone of `TheoLeeCJ/SemIf`
(MIT, Copyright (c) 2026 TheoLeeCJ) with three files modified. Committing it would have meant
either carrying a foreign history into this repo or silently forking someone else's project.
`patch/` instead holds a **754-line diff against pinned commit `1f2dea3`**, the three complete
post-patch files for drop-in use, and `apply.sh`. The diff's exactness was verified by
`git apply --check -R` against the working tree — it reverse-applies, so it describes the tree
precisely, and the working tree was left untouched (`git reset` after `git add -N`).

- **The 12 surveyed engines and `edge0/` are not vendored** (474 MB of other people's code).
  `scripts/fetch_prior_art.sh` re-clones each at the pinned commit that was actually read, so
  every claim in the survey stays checkable without redistributing anything.
- **19 probes rescued from `/tmp`** into `scripts/probes/`, with a README mapping each probe to
  the question it answers. They were a `/tmp` cleanup away from being lost, and they are the
  evidence — four of the findings in §6c exist *because* a probe was written to disprove a
  hypothesis already written down.
- **The runner was generalized.** It had `/home/scribe/...` hardcoded in six places, so it would
  not have run for anyone else. Now env-overridable with repo-relative defaults. Every hard-won
  comment was kept; they carry more than the code does.
- **The framing was corrected in the deliverable, not just in conversation.** README.md and
  SETUP.md both state up front that **nothing is generated** — 23/27/28 tok/s is *prompt-prefill*
  throughput for a classifier, against 15.1 one-shot / 23.0 resident end-to-end including the
  load. A repo whose headline could be read as "27 tokens/s of output from SSD" would mislead
  every reader who acted on it.
- **`probability_status` is surfaced to the agent author.** The field literally reads
  "uncalibrated as decision confidence"; SETUP.md quotes it and says to rank on it, not threshold
  it. `full_vocab_argmax_id` is documented as the cheap sanity check that the model's preferred
  token was one of the offered options.
- **Published:** 48 files, 310 KB, commit `b25fd22` — no `semif/`, `edge0/`, `offload_projects/`,
  `.venv`, weights, or `INVALID-*.jsonl`. Secret-scanned first (clean).

### Unresolved

- **Read amplification** — 41 GB against an 18.33 GB expert set. Needs the weight-access rewrite
  (`pread` over exact expert extents, engine-owned bounded slot pool), which is the seam every
  working prior-art engine uses.
- **No resident scorer.** The one-shot CLI re-pays a 19.8 s load every call — 33% of wall clock —
  and nothing carries over between calls (calls 2 and 3 read the same 61.6 GB). A resident
  process saved 33.8% per call over three calls. This is the largest unattacked win, and it is
  still only sketched by `scripts/probes/probe_lifecycle.py`.
- **`--max-tokens` is still unmeasured.** It is now the last unmeasured lever.
- **The repo is published but the push has not been independently re-verified** — the check was
  blocked by a transient classifier denial, so the evidence is the push's own output
  (`* [new branch] main -> main`, exit 0). Re-confirm with `git ls-remote origin` next session.

## Session Handoff — 2026-09-22 — deployment-shaped tests: lifecycle, contention, budget, scaling

### Goal

The scorer is meant to be **invoked by a code agent** (OhMyPi, OpenCode, …) on a
**workstation with other work on it**. Every prior measurement was a single run on an idle
box with a synthetic cap. Re-test in the deployment's shape, in the order: lifecycle →
contention → memory budget → criteria scaling → variance.

### Failed Paths

1. **Nothing failed here, but the shape of the tests mattered.** Evicting the page cache
   between agent calls would have overstated the one-shot cost: a real second call inherits
   whatever the first process left cached. Evicted once at the start of each variant only.
2. **First N=16 run errored and my grep hid it** (`2>/dev/null` on the probe). The failure was
   real and became the most important finding of the session — see below.
3. **My prediction that 8 criteria would be free was wrong.** I expected one 8-branch group to
   cost the same as one 4-branch group. It cost 64 GB against 41 GB.
4. **The 12 GiB cap result was not reproducible** (49.98 s, then 34.91 s on identical reads), so
   the "more RAM is worse" inversion I first wrote up is not a stable property — 12 GiB is
   simply erratic. Corrected rather than published.

### Final Solution — one real bug fixed, four measured answers

**Bug: `score_shared` cleared the context once, before the group loop.** Every branch carries
the whole prompt as its own sequence, so positions restart at 0 for each group; a group that
does not release the previous group's cells reuses sequence ids whose KV still holds a longer
prompt, and `llama_decode` fails outright. **Measured: 16 criteria against `n_seq_max=8` raised
`llama_decode failed` from group 2 onward** — a hard failure, not a slow answer. Fixed by
clearing per group; 16 criteria now return 16 correct answers with identical probabilities.
Regression test added. Only reachable beyond `n_seq_max` criteria, which is exactly what an
agent sending a batch will do.

**1. Lifecycle — `cli.py` is one-shot, and nothing carries over.** 3 calls each of
`load → score → exit`, vs one resident process (`/tmp/probe_lifecycle.py`):

| | 3 one-shot calls | resident (1 load + 3 calls) |
|---|---|---|
| wall | 58.2 / 61.2 / 60.6 = **180.0 s** | 19.8 + 40.3 / 39.5 / 39.4 = **139.0 s** |
| bytes | 61.9 / 61.6 / 61.7 = **185.2 GB** | **144.1 GB** |

Calls 2 and 3 read the *same* 61.6 GB as call 1 — with 41 GB per call against an 8 GiB cap,
the cache retains nothing between invocations. Residency saves **22.8% over 3 calls** and
**33.8% per call** once loaded (60.0 → 39.7 s); the saving grows with call count (at 10 calls,
600 vs 417 s). End-to-end per call the load is 33% of wall clock. **This is an architecture
decision — a resident scorer — not a tuning knob, and it is the largest unattacked cost.**

**2. Contention — the thread default holds, and is worth *more* under load.** Decode seconds,
cache evicted, 8 GiB cap, background busy loops representing other workstation work:

| scorer threads | idle | +6 busy | +12 busy |
|---|---|---|---|
| 16 (new default) | 38.4 | 55.75 | **74.17** |
| 6 (old default) | 54.62 | 83.54 | **155.34** |

Under 12 competing threads 16 beats 6 by **2.1×** (idle: 1.4×) — the gap *widens*, and the old
default degrades worse (2.8× vs 1.9×). Read volume stays ~41 GB throughout, so contention costs
time, not bytes. Realistic agent-facing figure on a busy box: **~74 s per call, not 38 s.**

**3. Memory budget — the cap is not a handicap, and near the fit boundary it is erratic.**

| cap | decode reads | decode | note |
|---|---|---|---|
| 8 GiB | 41.0 GB | 38.4–40.3 s | stable (±2.5%, 4 samples) |
| 12 GiB | 30.2 / 30.5 GB | 49.98 / **34.91** s | **unstable (±18%)** |
| 16 GiB | 19.28 GB | 29.31 s | |
| 24 GiB | 0.00 GB | 16.02 s | fully cached |

Reads fall monotonically with the cap, but time does not, and the 24 GiB point cross-checks the
warm floor exactly (16.02 s against the 16–18.5 s warm measurements). 12 GiB straddles the
fit boundary and goes **bimodal** — do not size the budget there; take a clear margin on one side.

**4. Criteria scaling — cost grows faster than the criterion count.**

| N | passes | tokens | reads | time | s/criterion | tok/s |
|---|---|---|---|---|---|---|
| 4 | 1 | 908 | 41.0 GB | 39.4 s | 9.85 | 23.0 |
| 8 | 1 | 1816 | 64.1 GB | 66.7 s | 8.34 | 27.2 |
| 16 | 2 | 3632 | 128.3 GB | 127.1 s | 7.94 | 28.6 |

Batching pays a little (9.85 → 7.94 s/criterion) but nowhere near free: one pass over 8 branches
reads 64 GB against 41 GB for 4, because reads grow with tokens-per-pass. Beyond `n_seq_max=8`
each extra group is a **full additional pass**.

### Unresolved

- **The read amplification now grows with tokens-per-pass** (2.3× the 18.33 GB expert set at
  1,816 tokens/908 tokens, 3.5× at 1,816). Not a second ubatch sweep, not fixable by prewarming.
  Still points at the pread-over-exact-extents rewrite.
- **No resident scorer exists.** Largest known win, needs building.
- Variance at the operating point (8 GiB) is ±2.5% over 4 samples — enough for a timeout, but
  p95 across many invocations is not yet characterised.
- `--max-tokens` remains unmeasured.

### Artifacts

`/tmp/probe_lifecycle.py` (resident: load once, N calls), `/tmp/oneshot_wrapper.py` (true
one-shot path incl. `/proc/self/io`, via runpy), `/tmp/run_lifecycle.sh`,
`/tmp/run_contention.sh` (CAP/THREADS/LOAD matrix), `/tmp/run_n.sh`, `/tmp/decisions.n8.jsonl`,
`/tmp/decisions.n16.jsonl`. Docs: `SEMIF_LLAMACPP_SSD.md` §6d.


## Session Handoff — 2026-09-21 (c) — "do the fastest one": thread oversubscription

### Goal

"Do the fastest one" — implement the biggest-win optimisation from
`OFFLOAD_PROJECTS_ANALYSIS.md`'s ranked list. The list's #1 cheap item was storage placement
("move the GGUF to the internal NVMe, documented 2.7×").

### Failed Paths

1. **Storage placement — void before it was read.** The device inventory refuted the premise
   the recommendation rested on: the GGUF is *already* on `/dev/nvme0n1p5` (Kingston
   SFYRD4000G, `rotational=0`, ext4 `noatime`), and `dd iflag=direct bs=1M` reads it at
   **2.4 GB/s**. Scoring achieves ~750 MB/s ≈ **31% of the device**, so the gap is access
   pattern, not placement. Lever already taken.
2. **`POSIX_FADV_WILLNEED` prewarmer — no effect at all.** A layer-ordered thread issuing
   WILLNEED over the 40 layers' expert extents (parsed from the GGUF tensor table, 18.33 GB
   of routed experts) moved read_bytes by **0.14 GB across a whole run** (42.07 vs 41.93 GB)
   and time by 0.05 s. Not "prefetch is futile" — the *advice was never acted on*: the kernel
   caps an advised readahead window to a small multiple of `read_ahead_kb` (128 KB here), so
   16 MB chunks became ~128 KB requests.
3. **Real `preadv` prewarmer — actively worse.** Replacing the advice with a genuine
   `preadv` into a reused 16 MB buffer did read all 18.3 GB, and the run got worse:
   read 42.0 → **56.1 GB**, time 56.0 → **57.56 s**. The prewarmed pages are clean and
   unreferenced, so under the 8 GiB cap they are the *first* reclaim victims — evicted before
   the decoder arrives, then faulted in a second time. We paid for those bytes twice.
4. **"Two expert sweeps" theory — refuted.** 8 branches × 908 tokens = 7,264 would exceed
   `n_ubatch` 4096 and cause two full layer sweeps. But this sample's group is 4 branches /
   **908 tokens**, already inside one 4096 ubatch (sweeps = 1). `SEMIF_LLAMA_UBATCH=8192`
   changed nothing (41.93 GB / 54.62 s).
5. **My own test expectation was wrong again**, not the code: `_default_threads()` with
   `cpu_count() is None` returns 5 (fallback 4 cores, then the 4/3 rule), not 4.

### Final Solution

**CPU thread oversubscription.** The runner pinned `THREADS=6` on a 6-core/12-thread host —
half the machine — and `load_model` defaulted to `os.cpu_count()`, exactly the hardware
thread count. Both wrong for a decode served from disk: **a thread waiting on an expert page
fault is not runnable**, so an expert-streaming decode wants more runnable threads than the
CPU has, to keep the device queue fed. Swept over one fixed pass (cache evicted, 8 GiB cap):

| threads | decode | throughput |
|---|---|---|
| 6 (pinned default) | 54.62 s | 768 MB/s |
| 8 | 50.35 s | 836 MB/s |
| 12 (hardware threads) | 47.87 s | 860 MB/s |
| **16 (4/3 × 12)** | **38.44 s** | **1069 MB/s** |
| 20 | 38.42 s | 1054 MB/s |

Knee at 16, then flat; read volume unchanged (41.2–42.1 GB), so this buys wall clock without
touching I/O — and it is the evidence that the decode is **not** purely I/O-bound, contrary
to the earlier conclusion. Result: **−29.6%, lossless, identical probabilities**.

Changes: `_default_threads()` added to `llamacpp_backend.py` (4/3 of `os.cpu_count()`, floor
4) and used in `load_model`; `run_semif_35b_ssd.sh` computes `$(nproc)*4/3` instead of
pinning 6; 6 parametrised cases added to `tests/test_llamacpp_batched.py`.

`pytest -q` → **89 passed, 3 skipped**. End-to-end through the shipped runner: metadata
records `threads 16`, probabilities 0.999986 / 0.999954 / 0.999996 / 0.999988 — identical to
the controlled probe runs.

### Unresolved

- **The ~2× read amplification stands unexplained.** ~41 GB decode for 18.33 GB of routed
  experts in a 20.89 GB file, ~1,069 MB/s against a device that does 2.4 GB/s. Not a second
  ubatch sweep, not fixable by prewarming, and `MADV_RANDOM` (which cuts reads to 26.0 GB)
  costs 6.6× in time. The remaining explanation is read-ahead re-reads under cap pressure.
- **The real lever is still the weight-access rewrite** — `pread` over exact expert extents
  with the engine owning a bounded slot pool (the slot-remap seam, `OFFLOAD_PROJECTS_ANALYSIS.md`
  §5 #2/#3). Two independent implementations exist; `slipstream-schero94` ships it as a
  4,488-line patch against pinned llama.cpp `79bba02a6741`. Not a knob.
- **Warm is still 2× slower than the single-sequence path** (§6b) — batching pays only in the
  I/O-bound regime.
- `--max-tokens` remains unmeasured; threads are now settled.

### New — probes and docs

`/tmp/probe_prefetch.py` (layer-ordered prewarmer, `advise` and `read` modes; GGUF tensor-table
parser for expert extents), `/tmp/probe_ubatch.py` (sweep-count attribution, `SEMIF_THREADS`),
`/tmp/run_prefetch.sh`, `/tmp/run_ubatch.sh`. Docs: `SEMIF_LLAMACPP_SSD.md` §6c (new),
`OFFLOAD_PROJECTS_ANALYSIS.md` (§4 dead end added, #7 retracted, #8 added).


## Session Handoff — 2026-09-21 (b) — optimising the scoring path

### Goal

"revisit and optmise, it should be doing at least the same" — the first SSD measurement
(72.79 s, 112.8 GB for 4 decisions) looked implausibly slow next to Edge0's streamed
13 tok/s, so the scoring path had to be made at least as fast.

### Failed Paths

1. **"Raise `n_seq_max` and batch the branches" — necessary but not sufficient.** It was
   the right diagnosis (a single-sequence context serialised the branches) and the wrong
   fix on its own: reads went 112.8 → **108.1 GB** and the run got *slower* (77.2 s). The
   log line `graph_reserve: ... ubatch with n_tokens = 512` was the clue — llama.cpp's
   default `n_ubatch` split the 908-token group into two physical ubatches, and each ubatch
   walks all layers and re-reads their experts, so "one pass" was two sweeps underneath.
   Fixed by sizing `n_batch = n_ubatch` to the whole group.
2. **`llama_memory_seq_cp` to share one prefilled prefix — SIGABRT.** The cheaper design was
   to prefill the 135-token state once and copy its KV into each branch. This checkpoint is
   a *hybrid*: `qwen35moe` interleaves recurrent (linear-attention) layers with attention
   layers, and `llama_memory_hybrid::seq_cp` aborts inside `llama_kv_cache::seq_cp`. So each
   branch must carry the whole prompt itself, which is exactly why the batched path is
   *slower* warm (18.53 s vs 9.29 s) than the sequential one.
3. **Assuming the load was cheap because `load_mode = mmap`.** It is not: isolated with no
   SHA-256 pre-pass and no tokenizer, `llama_model_load_from_file` reads **20.9 GB (18.3 s)**
   — the entire file. `check_tensors` is `false`, so this is not a validation read; it is
   this llama.cpp's mmap path touching the whole mapping, and no `llama_model_params` field
   turns it off.

### Final Solution

Three changes in `semif/src/semif_phase1/llamacpp_backend.py`, plus a new test file:

1. **Multi-sequence batched scoring.** `_Engine` takes `sequences` (`SEMIF_LLAMA_SEQ_MAX`,
   default 8) and sets `n_seq_max` / `n_outputs_max`. New `branch_logits_batched(prefix,
   suffixes)` puts every branch — each carrying the *whole* prompt as its own sequence —
   into a single `llama_decode`, flagging only each branch's last token and reading logits
   back per batch slot. `score_shared` uses it and loses the per-branch save/restore loop
   entirely.
2. **`n_batch = n_ubatch` sized to the group** (`_ubatch()`, default `sequences × 512`).
   This is the change that actually cut reads; `SEMIF_LLAMA_UBATCH` / `..._PER_SEQ` pin it.
3. **Cached GGUF digest** (`_gguf_digest()`): the SHA-256 pass reads the whole 20.9 GB file,
   and llama.cpp's loader then reads it again, so a cold run paid two full sweeps before
   scoring a token. Now cached in a `<name>.sha256.json` sidecar keyed on size + `mtime_ns`;
   `SEMIF_GGUF_DIGEST=compute|skip` overrides, and every row records which path was taken in
   `model.gguf.digest_source`.

### Result — same eviction, same 8 GiB cap, same input

| | before | after |
|---|---|---|
| `total_seconds` (scoring) | 72.79 s | **56.57 s** |
| bytes read, whole run | 112.8 GB | **60.2 GB** |
| prompt throughput | 6.9 tok/s | **16.1 tok/s** |
| decisions | 4/4 | **4/4, identical probabilities** |

**Warm is the trade-off and it went the other way: 9.29 s → 18.53 s**, because each branch
recomputes the shared prefix (908 token-forwards vs 503). Batching is correct for the
regime this work targets and wrong for a host where the checkpoint fits in RAM.

Verified: `pytest tests/` → **79 passed, 3 skipped**, including a new
`tests/test_llamacpp_batched.py` covering the batched slot mapping, multi-ubatch flagging,
the batch-sizing rules, and digest-cache reuse/invalidation.

### Unresolved

- **The ≈2× read amplification on the decode pass.** One 908-token ubatch reads 42.0 GB
  against a ~19.5 GB logical expert footprint — consistent with mmap fault-around
  read-ahead pulling unused pages. `posix_fadvise(POSIX_FADV_RANDOM)` / `madvise(MADV_RANDOM)`
  would suppress it, but both must be applied to llama.cpp's own descriptor or mapping,
  which this backend does not own. Worth ≈25 s if it holds, and it is the largest single
  remaining term.
- **The load's 20.9 GB is unattacked** and is now roughly a third of the run.
- **The warm regression is unmitigated** — no fallback path is wired, deliberately, to keep
  one hot path rather than two. Revisit if a RAM-resident deployment matters.
- **`--max-tokens` and `--llama-threads` remain unmeasured**, and prefill still scales with
  tokens × experts touched, so a real DOM snapshot is far worse than the 135-token case.

### New — parallel project analysis (complete)

Cloned all 12 repos from `qwen3.6-35b-a3b-ssd-offload.md` into `offload_projects/`
(11 certified + `flash-moe` for its `paper/` and `repack_experts.py`), ran five agents
(Edge0; the C/portable cluster; the Apple cluster; the remainder; a web-research pass on
published algorithms), and wrote the synthesis to **`OFFLOAD_PROJECTS_ANALYSIS.md`**.
The two `slipstream` repos collided on clone; the second is `slipstream-schero94`.

Headline findings:

- **The list overcounts implementations.** `slipstream-dwijenpatel` is a fork of
  TurboFieldfare and `Mference` carries the same files almost verbatim — three of the four
  Apple repos are one codebase. Certification at README level also did not survive: Edge0's
  prerouter accuracy is **never measured in-repo**, its "Recover-LoRA" reduces quality loss
  and **zero I/O**, `samosa-chat`'s "2-bit" is two 4-bit planes, and
  `Qwen-MoE-Router-exp` is a stub with no disk code.
- **Every working engine uses `pread` over a per-expert offset table.** The mmap-based ones
  are the slow or broken ones. Convergent extras: coalesce each expert into one contiguous
  16 KiB-aligned slab, bound the cache with a hard RAM admission gate, and floor it at
  8 slots/layer (top-8 routing means a layer below 8 slots misses *every* token).
- **The proven way to stream experts inside llama.cpp** is a slot-remap seam implemented
  independently by `siphon.cpp` (repoint `w->data`, shrink `w->ne[2]`, rewrite the id tensor)
  and `slipstream-schero94` (a 4,488-line patch against pinned commit `79bba02a6741`).
  The latter is the closest thing to a drop-in and the most directly applicable artifact found.

**A measured correction to our own doc.** The web pass ranked "kill mmap fault-around
read-ahead" first (+53% claimed, llama.cpp #23324). We tested the cheap version —
`madvise(MADV_RANDOM)` on the GGUF VMA via `/proc/self/maps` — and it is a trap: reads fell
42.0 → **26.0 GB** while the run went 56.0 → **368.0 s (6.6× slower)**, effective throughput
750 → 71 MB/s. Disabling read-ahead turns scattered expert faults into synchronous random
reads and NAND *latency* takes over. **The 2.15× overhead is load-bearing prefetch, not
waste**; `SEMIF_LLAMACPP_SSD.md` §6b has been corrected to say so. #23324 measured an async
`pread` pool over exact expert extents with pipelining, which is a different change.

Also recorded as dead ends with reasons: lossless compression of quantized weights
(incompressible, decoder 114–120 MB/s vs our 750 MB/s SSD), elaborate eviction policies
(arXiv:2608.07911: causal rules recover none of the offline-optimal gap), and co-activation
expert reordering (`flash-moe` measured 0% — NVMe ignores scatter at 7 MB granularity).


## Session Handoff — 2026-09-21

### Goal

Run **Qwen3.6-35B-A3B** as the JEV-style typed-decision scorer, with expert weights
served from **SSD** rather than resident RAM, on an **M3** but without depending on
Apple Silicon — the user prefers the llama.cpp substrate over MLX.

### Failed Paths

1. **Path A — "point SemIf's llama.cpp backend at a 35B GGUF with MoE-offload flags."**
   Half right, but premised on a wrong assumption. The `-ot ".ffn_.*_exps.=CPU"` /
   `--cpu-moe` / `--n-cpu-moe N` family are *partitioning* flags: they keep experts on
   CPU while other layers go to the GPU. SemIf already sets `n_gpu_layers = 0`, so every
   layer is on CPU and there is nothing left to partition. No flag to pass, and no
   flag-passthrough patch needed. **The entire "Path A needs a passthrough patch"
   premise was wrong.**

2. **Path B — "add Edge0 as a SemIf backend / peer module."** Dropped. Edge0's only
   backend is `src/edge0/backends/mlx`, which is exactly the Apple lock-in the user
   asked to avoid. Its value is as a reference implementation, not a dependency.

3. **Laya weight-conversion route.** Earlier established as impossible: Laya is
   encoder + decision Transformer + scoring head + action head, non-autoregressive,
   421M, 512–1024 ctx. A 35B causal MoE cannot be converted into a Laya checkpoint.
   Only its *I/O contract* is convertible. Moot now — the llama.cpp path supersedes it.

4. **`--n-cpu-moe 0`** in a pasted transcript was backwards (zero expert layers on CPU =
   all experts to GPU = OOM). Correct is `--n-cpu-moe 99` / `--cpu-moe`. Now moot.

5. **README-level "certification"** of 11 SSD-offload repos was weaker evidence than the
   conclusions drawn from it. Disclosed in `qwen3.6-35b-a3b-ssd-offload.md`. Superseded
   by reading actual source here.

### Final Solution

**A small patch to SemIf's existing llama.cpp backend** — `src/semif_phase1/llamacpp_backend.py`:

- `load_mode` is set **explicitly** (default `mmap`) with a `SEMIF_LLAMA_LOAD_MODE`
  override accepting `auto|none|mmap|mlock|mmap_mlock|direct_io`; an unknown value
  raises rather than silently falling back.
- `use_extra_bufts` is set to **`false`**, with a `SEMIF_LLAMA_EXTRA_BUFT` override. This
  is the load-time repack switch, and the difference between a >RAM checkpoint loading
  and being OOM-killed during load.
- Both recorded in the result metadata (`load_mode`, `load_mode_name`,
  `use_extra_bufts`) so every output row is self-describing about how its weights were
  served.
- Docstring states the streaming contract, including why `use_mmap` / `use_mlock` are
  **not** the controls here.

`mmap` keeps the GGUF file-backed and demand-paged, so a routed expert is read from disk
on first touch and its page stays evictable — which is what lets a 19.5 GiB checkpoint
score on a small machine. `LLAMA_LOAD_MODE_MLOCK` would pin every expert in RAM and the
load would simply fail.

**Correction — the first version of this patch was a silent no-op.** It set
`params.use_mmap` / `params.use_mlock`. Those fields were **removed** from
`llama_model_params` in this llama.cpp; the struct carries `load_mode` instead. A ctypes
`Structure` accepts arbitrary Python attributes, so the assignment *succeeded* and
`bool(params.use_mmap)` read back as `True` — the patch changed nothing, and the
metadata it wrote was self-confirming. Streaming was already on by default anyway
(`load_mode = AUTO` → `mmap`, `llama-model-loader.cpp:554`), which is exactly why a
big-RAM host could not tell a working patch from a no-op one.

**Correction — the repack diagnosis.** `GGML_CPU_REPACK=0` was tested and **disproven**
(117 repack lines persisted); `CPU_REPACK` is the buffer type's *name*, not an env key.
Reading `use_extra_bufts` off a Python-built `llama_model_params()` also misled: that
struct is zero-initialised by the binding and reported `false`, when the real default is
`true` (`llama-model.cpp:2489`). The actual gate is `use_extra_bufts` in
`make_cpu_buft_list` (`llama-model.cpp:942`), reached by
`ggml_backend_cpu_repack_buffer_type()` (`ggml-cpu.cpp:65`) — and it is confirmed by
measurement, below.

Verified against disk:

| Prerequisite | Finding |
|---|---|
| Target GGUF | already local — `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`, 19,926 MB |
| GGUF identity | `general.architecture=qwen35moe`, `general.name=Qwen3.6-35B-A3B`, `model_type=qwen3_5_moe` — confirmed correct model |
| Tokenizer | not cached; fetched `Qwen/Qwen3.6-35B-A3B` (`vocab.json` + `merges.txt`; this repo has **no** `tokenizer.json`) |
| `llama-cpp-python==0.3.35` | **sdist only, zero wheels on PyPI** — source build (cmake + compiler) is unavoidable |
| 0.2.39 in `~/.local` | uses the **old C API** (`llama_load_model_from_file`, no `llama_get_memory`) — cannot run this backend |
| SemIf install | absent; fresh clone, venv created at `semif/.venv` |

**Verified by execution, not by reading:**

- `pytest tests/test_llamacpp.py` → **19 passed, 1 skipped**. The streaming test asserts
  `load_mode` and `use_extra_bufts`, and additionally asserts the returned params have
  **no** `use_mmap` / `use_mlock` attribute — so a regression back to the inert spelling
  fails loudly instead of silently doing nothing.
- **Repack is gated by `use_extra_bufts`, measured on the real checkpoint** — two loads,
  counting `repack` lines: `false` → **0 lines**, `true` (the default) → **119 lines**.
  Both loaded. This is the fix for the OOM kill, and the run script now pins it to `0`.
- The 19.5 GiB checkpoint **loads and reports `load_mode = mmap`** with all 40 layers
  `assigned to device CPU`, `n_expert = 256`, `n_expert_used = 8`, `file type = Q4_K - Small`.
- **Warm-cache baseline (NOT an SSD number)**: 4/4 decisions correct, 11.88 s for all four
  criteria at `threads=6`, `max-tokens=8192`, `batch_size=4` — prefill 3.61 s,
  suffix_forward 8.24 s (≈2.1 s marginal per extra criterion). `suffix_forward` exceeding
  `prefill` is the expert-read term, i.e. the one that changes on the SSD path.
- **`--mode shared` requires all rows to carry one identical nonempty `state`** and refuses
  otherwise (`ValueError: Shared scoring requires one nonempty exact state`). This is the
  JEV contract made explicit — one page snapshot, many criteria — and it is why the first
  sample input (heterogeneous states) failed. `decisions.sample.jsonl` now models the
  shared-state shape; `MODE=direct` is the escape hatch for differing states.

Note the llama.cpp path never inspects `model_type` — it needs only the HF **tokenizer**,
re-tokenizes through the GGUF vocabulary and refuses to score if the two disagree
(`_verify_vocabulary`). That is why a `qwen3_5_moe` GGUF works even though `core.py`
recognises only `qwen3_5` / `qwen3_5_text` for the torch backend.

Artifacts: `SEMIF_LLAMACPP_SSD.md` (full rationale), `run_semif_35b_ssd.sh`,
`decisions.sample.jsonl`.

Result files, named so the valid ones cannot be confused with the invalid ones:

| file | what it is |
|---|---|
| `decisions.ssd8g_evicted.jsonl` | **the real SSD measurement** — evicted cache, cap bound |
| `decisions.scriptcheck.jsonl` | independent reproduction via the run script |
| `decisions.warm_evicted.jsonl` | warm control, cache evicted, no cap |
| `decisions.warm_noRepack.jsonl` | warm, repack off (the 9.29 s figure) |
| `INVALID-warmcache.ssd8g*.jsonl` | capped runs taken **before** eviction was added — cap never bound; kept only as evidence of the failed method, **not** valid SSD numbers |

### Unresolved

- **The SSD number exists — and the first attempt at it was invalid.** The `--simulate-8g`
  run (cgroup `MemoryMax=8G`, `MemorySwapMax=0`) completed after the repack fix and gave
  4/4 decisions in 9.35 s — but it measured **nothing**, because a cgroup does not
  re-charge an already-cached page. Measured: `read_bytes = 0` for the whole run, scope
  `memory.peak = 632 MB`, `memory.events max = 0` — the 8 G limit never bound. The
  19.5 GiB GGUF was fully resident (`fincore`: 19.5 G) and charged to the long-lived
  terminal scope (40 GB peak), so the fresh scope reused it free. A capped run can come
  out *faster* than an uncapped one, which is the tell.
  **Fix:** evict first with unprivileged `posix_fadvise(POSIX_FADV_DONTNEED)` (`fincore`
  19.5 G → 0 pages, no root needed; this box has no passwordless sudo).
  `run_semif_35b_ssd.sh` does this under `--simulate-8g` and prints the `fincore` line so
  cache state is visible rather than assumed. **The check to trust is
  `/proc/<pid>/io read_bytes`, not wall-clock.**
  **Second bug, caught by the end-to-end re-check:** the eviction step was gated on
  `$SIMULATE`, but the re-exec passed `"$0"` with **no arguments**, so the child never
  saw `--simulate-8g`, `SIMULATE` was empty, and the run silently took the warm path
  (`fincore` 19.5 G, `total_seconds` 9.17 s). The flag applied the cgroup cap — that came
  from `systemd-run` directly — while skipping everything else gated on it. Now gated on
  an exported `SEMIF_CAPPED` and `"$@"` is forwarded. Lesson recorded: verify a
  measurement harness by checking it *fails* when it should, not only that it runs.

**The SSD measurement (evicted + capped at 8 GiB, repack off, `threads=6`, 4 criteria):**

| | warm (no cap) | capped 8 GiB | ratio |
|---|---|---|---|
| `total_seconds` | 9.29 s | **72.79 s** | 7.8× |
| `prefill_seconds` (135-token state) | 2.45 s | **16.03 s** | 6.5× |
| `suffix_forward_seconds` (368 tok) | 6.82 s | **56.74 s** | 8.3× |
| marginal per extra criterion | ≈1.7 s | **≈14.2 s** | 8.3× |
| decisions correct | 4 / 4 | **4 / 4** | — |
| **bytes read from storage** | 20.9 GB | **112.8 GB** | 5.4× |
| cgroup `memory.peak` | 40.3 GB (uncapped) | **exactly 8 GiB** | — |
| cgroup `memory.events max` | 0 | **70,584** | — |

The cap genuinely bound (8 GiB peak, 70,584 limit hits), so this is a real streaming
measurement: **one shared-state run of 4 decisions read 112.8 GB from SSD — 5.8× the
checkpoint size.** Subtracting the ~20.9 GB any cold run pays once (SHA-256 + first
touch), ~92 GB is re-reading evicted expert pages. Answers were unchanged (4/4, same
probabilities), so streaming costs time, not correctness. Storage throughput during the
read phase was ~800 MB/s.

**Reproduced independently** through `./run_semif_35b_ssd.sh --simulate-8g` rather than
the ad-hoc harness: **79.28 s** total, prefill 17.77 s, suffix 61.48 s, identical 4/4
answers. Quote the budget as **≈73–79 s for 4 decisions on a 135-token state, ≈14–15 s
marginal per criterion**; the spread is run-to-run variance, not a different regime.

**Also measured: repack-off is not a speed cost here.** Warm + repack-off is **9.29 s**
vs **11.88 s** warm + repack-on — ~22% *faster*, and reproduced twice (9.35 s, 9.29 s).
So the setting that makes a >RAM checkpoint loadable is also free on this CPU, contrary
to the earlier assumption that it trades speed for residency. Caveat: the 11.88 s figure
came from an earlier session, so re-measure repack-on under current conditions before
leaning on the 22%.

**Caveat that limits extrapolation:** `prefix_tokens` was only **135** — a small form
snapshot. Prefill scales with tokens × experts touched, so a real DOM snapshot is the
expensive case and 16 s is the *cheap* end.

2. **Prefill is the expensive case, not decode.** Decode touches ~8 experts/layer/token
   (SSD-friendly); prefill over N tokens touches most of 256 experts/layer (SSD-hostile).
   SemIf reads decision logits from a **prefill**, so a JEV decision sits in the worst
   case.
   **Retraction (2026-09-22):** this entry previously added "This also re-reads Edge0's
   113–140 tok/s cold-prefill claim as arithmetically implausible for a full-model read
   per pass." **That is wrong and is withdrawn.** 3,300 tokens at 113 tok/s is 29.2 s; a
   full 20.88 GB read in that window is 0.72 GB/s, roughly **7× under** an M4 Pro's SSD.
   The figure was never impossible. Edge0's actual mechanism is `staged_k4()`'s four fixed
   expert slots plus a 33-head prerouter, which engineer away the per-token full routed-set
   read — the same conclusion as `SEMIF_LLAMACPP_SSD.md` §5. Both numbers are also on a
   24 GB machine with the checkpoint effectively page-cached, so the like-for-like row is
   our **warm 10.68 tok/s vs Edge0's 14.9**, not a 3.5× gap.

3. **`--max-tokens` default is 4096** — a hard prompt cap that a DOM snapshot will
   exceed. Script defaults to 8192; needs real tuning, since raising it grows KV memory
   and competes with the page cache for exactly the RAM streaming depends on.

4. **`--llama-threads` defaults to every core**; script defaults to 6. Unmeasured.

5. **If the capped run shows mmap thrashing**, the fix is deliberate I/O (`O_DIRECT` /
   `pread` against an expert offset table) rather than fault-driven paging. Edge0 does
   this; unmerged llama.cpp PR #25294 does too. That is a *later* step, gated on
   measurement — do not start it before the `--simulate-8g` number exists.
   Note `SEMIF_LLAMA_LOAD_MODE=direct_io` is **not** that fix: it opens the GGUF with
   `O_DIRECT` and reads tensors into allocated buffers (`llama-mmap.cpp:199`), which is a
   faster *load* and a worse *footprint*. It is exposed for completeness, not recommended.

6. **Source build of `llama-cpp-python` in flight.** venv created with
   `uv venv --python 3.11`, which resolved to **3.11.0rc1** (the only other local
   interpreter is stable 3.10.12). If the extension misbehaves, rebuild on 3.10.12.
