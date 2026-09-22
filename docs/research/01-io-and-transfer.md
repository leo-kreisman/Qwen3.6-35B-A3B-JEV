# 01 — Low-level I/O and data transfer for streaming MoE experts from NVMe

**Scope.** How bytes of expert weight get from this host's single NVMe device into RAM or VRAM
with the least waste, at the smallest possible extent granularity. This dossier is the *transfer
layer* of the SSD-offload project. It pairs (a) measurements taken **on this exact host** with
(b) primary-source literature read during this review.

**Provenance tags.** `[verified-here]` = measured on this host during this review, with the command
or program named. `[verified-here, prior]` = measured on this host by a previous agent, artifact
preserved (path given). `[verified-source]` = read from the primary source in this review; URL
given. `[vendor]` = the number comes from the vendor's own documentation or marketing, not from an
independent measurement. `[unverified]` = carried from another document in this repo, or a claim I
could not confirm in the primary text. Nothing is estimated silently.

**Host identity, verified this review** `[verified-here]`:

| item | value |
|---|---|
| kernel | 6.12.10-76061203-generic (Pop!_OS build, x86_64) |
| CPU | Intel Core i7-8700K, 12 threads, 1 NUMA node |
| RAM | 62 GiB total |
| NVMe | `nvme0n1` = KINGSTON SFYRD4000G (4 TB class), firmware EIFK31.6, `rotational=0` |
| partitions | `nvme0n1p4` → `/` (ext4, `rw,noatime,errors=remount-ro`); `nvme0n1p5` → `/home` (ext4). **The model lives on `/home`.** |
| GPUs | 2 × NVIDIA GeForce RTX 5060 Ti 16 GiB, `cc=12.0` (Blackwell, 36 SMs), driver 580.159.03 |
| CUDA | toolkit 12.8 at `/usr/local/cuda-12.8`; `/usr/local/cuda` → `/etc/alternatives/cuda` → `/usr/local/cuda-12.8`. **`/usr/bin/nvcc` is 11.5** (`nvidia-cuda-toolkit` 11.5.1); `/usr/local/cuda-12.8/bin/nvcc` is 12.8. A plain `nvcc` invocation therefore compiles with 11.5 and cannot target `sm_120`. |
| model | `/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`, 20,893,015,008 B (19.46 GiB) |
| root mount of the model's filesystem | **ext4, not XFS, not EROFS, no DAX** |

---

## 1. Verdict

**The transfer layer is not where this project is losing, and the reason is now measured rather
than argued.** On this host, io_uring delivers 2.0–2.4× the sync-`pread` throughput for exactly the
byte-size that matters (589 824 B, the model's expert slab: 3383 MiB/s at qd 8 vs 1658 MiB/s for
`pread`, `[verified-here, prior]`), but 3.2 GB/s is *already the device's ceiling* — every block
size at or above 2 MiB saturates at 3.3–3.4 GiB/s in **both** engines, so the headroom io_uring
buys is a latency-per-extent and CPU-per-extent win, not a bytes-per-second win. Registered
buffers, SQPOLL and DEFER_TASKRUN are all real but second-order for this workload: registered
buffers only move the needle when the I/O is 4 KiB-sized (1270.7 vs 1093.8 MiB/s at qd 128, i.e.
+16%), SQPOLL only matters at 4 KiB (where it is the single best mode measured, 1451 MiB/s), and
at 576 KiB everything with depth lands in the same 3.2–3.4 GB/s band. Meanwhile the *waste* in the
system is large and has now been traced to one kernel constant: an ordinary page fault on a
file-backed `mmap` of this GGUF fetches **131 072 B for every 4 KiB page touched**
(`[verified-here, prior]`, reproduced this review) — the kernel's read-ahead window — so touching
one 4 KiB page inside a 576 KiB expert slice pulls in 128 KiB even though the *next* fault lands in
a different slice, i.e. the other 124 KiB of that window is fetched and then never touched. Over the
whole file the redundancy is therefore `read_ahead_kb ÷ expert_slab_size` = 128 KiB ÷ 576 KiB =
**0.22× of every slice's bytes wasted**, doubling to **0.44×** if the file is opened
`POSIX_FADV_SEQUENTIAL`, which this llama.cpp fork does unconditionally at mmap time. That is a
*fixed constant of this kernel on this device*, not a property of the device or of an API choice:
`read_ahead_kb` is 128 and the sysfs file that writes it *is* the readahead window
(`bdi->ra_pages = read_ahead_kb >> (PAGE_SHIFT - 10)`, `backingdev.c` `[verified-source]`). The two
things genuinely reachable without patching C are therefore (1) raising `read_ahead_kb` and
switching the NVMe scheduler off `kyber` — both root-only sysfs writes, one number each — and (2)
adding an explicit `preadv`/`O_DIRECT` read path in front of the faulted gather, which is what the
io_uring table supports. Everything more exotic — GPUDirect Storage, SPDK, NVMe passthrough,
CUDA VMM, file-backed huge pages — is either **proven dead on this host** (no `nvidia-fs` module;
`io_poll=0` so IOPOLL returns `EOPNOTSUPP`; `CONFIG_READ_ONLY_THP_FOR_FS` unset so 2 MiB
file-mapped pages are impossible; `MADV_COLLAPSE` returns `EINVAL`; `nr_hugepages=0`) or is a
research programme with a real patch behind it, not a tuning flag. The measured floor is
**3.4 GiB/s ≈ 3.6 GB/s** off this NVMe, and the project's own documented ~2.4 GB/s raw figure
`[verified-here, prior]` is **below what the device delivers when asked with depth ≥ 8** — which
means part of what the project has been calling a device limit is a queue-depth limit.

---

## 2. The measured I/O matrix on this host

### 2.1 Method, stated honestly

The CSV `io_results.csv` (`[verified-here, prior]`, artifact at
`/home/scribe/.hermes/cache/scratch/io_results.csv`, 90 rows) was produced by the author's own
programs from `sweep2.sh`, all against
`/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`:

- `iouraw.c` — **raw io_uring via `syscall(__NR_io_uring_setup/enter/register)`, no liburing**,
  plus an `O_DIRECT` `pread` control path. Same binary for every mode. Buffer is
  `posix_memalign(4096, bs*qd)`. Offsets are random, aligned to 4096 B, drawn from a fixed xorshift
  seed (`rng_ = 99991 + bs*7 + qd`, so runs are reproducible). Budget 256 MiB of *logical* bytes
  per run for every row except the `sqpoll`/`defer` rows, which are 128 MiB. Mode flags:
  `iopoll` → `IORING_SETUP_IOPOLL`; `sqpoll` → `IORING_SETUP_SQPOLL|IORING_SETUP_SQ_AFF`;
  `defer` → `IORING_SETUP_DEFER_TASKRUN|IORING_SETUP_SINGLE_ISSUER`. `regbuf` present → one
  `io_uring_register(IORING_REGISTER_BUFFERS)` over the whole `bs*qd` buffer, and
  `IORING_OP_READ_FIXED` instead of `IORING_OP_READ`.
- `io_bench.c` — the same sweep written against **liburing** (`iouring_prep_read`,
  `io_uring_queue_init_params`). Its output is not in the CSV; it is kept as the cross-check and is
  the file to run if you want the liburing numbers for a re-run.
- `align.c` and `faultexp.c` are separate probes, discussed in §3.

**Two things this sweep did *not* do**, stated because they bound what the table may be used for:

1. **`iopoll` returned `ERR` on every attempt** (rows 77, 81, 85, 89). The cause is now identified:
   the completion returned `EOPNOTSUPP` — `"cqe: Operation not supported"`, exit 4
   `[verified-here]`. This is consistent with `io_poll = 0` in `/sys/block/nvme0n1/queue/io_poll`
   and with the kernel's own requirement that IOPOLL be supported by the filesystem and block
   device (`io_uring_setup(2)`: "The file system (if any) and block device must support polling in
   order for this to work", `https://man7.org/linux/man-pages/man2/io_uring_setup.2.html`
   `[verified-source]`). **IOPOLL is therefore unmeasured on this host, not measured-and-bad.**
2. The runs are **not fsync'd, not drop-cached, and not cgroup-capped** between rows. The 256 MiB
   budget against a 19.46 GiB file with random offsets makes page-cache hits unlikely but not
   impossible, and the buffer is re-memset per process. Treat the numbers as *device-limited
   throughput on O_DIRECT*, which they are — `O_DIRECT` bypasses the page cache by construction —
   but do not read them as a controlled benchmark with warmup discipline.

Everything in the table is `O_DIRECT`, random 4096-B-aligned offsets, single process, single
thread, one ring, `kyber` scheduler (the host default).

### 2.2 The table that matters: 589 824 B (= 576 KiB = the expert slab size)

`MiB/s` / `IOPS`. Left column figures are the previous agent's; the `qd=4` and `qd=16` rows are
mine, added this review to complete the ramp.

| qd | pread | io_uring | io_uring + regbuf | uring ÷ pread |
|---|---|---|---|---|
| 1 | 1364.3 / 2425 | 1441.3 / 2562 | 1721.4 / 3060 | 1.06× |
| 4 | 2511.3\* / 4466\* | 2784.6 / 4950 | — | 1.11× |
| 8 | 1658.3 / 2948 | **3383.8 / 6016** | 3382.3 / 6013 | **2.04×** |
| 16 | — | 3118.3 / 5544 | — | — |
| 32 | 1580.3 / 2809 | 3211.3 / 5709 | 3396.3 / 6038 | 2.03× |
| 64 | 1696.5 / 3016 | 3271.8 / 5816 | 3310.2 / 5885 | 1.93× |
| 128 | 1598.9 / 2843 | 3307.4 / 5880 | 3318.5 / 5899 | 2.07× |

\* The `qd=4` `pread` row is not from the original sweep — it is the aggregate of **four concurrent
`pread` processes** run this review (4 × 64 MiB in 0.102 s = 2511.3 MiB/s aggregate, `[verified-here]`).
It is placed there because it is the honest counterpart to a single ring at qd 4, and it shows the
mechanism precisely: **sync `pread` is not slow, it is serial.** One process at qd 4 and four
processes at qd 1 each deliver ~2.5 GB/s; one process at any qd delivers 1.58–1.70 GB/s. Eight
concurrent `pread` processes reach 2930.2 MiB/s `[verified-here]`, which is *below* one io_uring
ring at qd 8. **This is the whole story of the sync-vs-async question on this host**: the device
wants depth, `pread` cannot supply depth within one thread, and adding threads is a worse way to
buy depth than adding a ring.

Reading the table:

1. **`pread` throughput is flat in qd** (1364 → 1696 MiB/s across qd 1→128) because `pread` has no
   queue — the `qd` argument only changes *which* buffer slot the next request lands in. The
   variation from 1364 to 1696 is run-to-run noise plus the effect of the caller's own buffer reuse,
   not a queueing effect.
2. **io_uring's win is entirely at qd ≥ 8 and saturates there.** 1441 MiB/s at qd 1 (equal to
   `pread`) → 3384 MiB/s at qd 8 → flat thereafter. **2.04× is the honest headline vs a single
   `pread` thread; against the 8-process `pread` aggregate (2930 MiB/s) it is only 1.15×.**
3. **Registered buffers contribute nothing at 576 KiB** (3382.3 vs 3383.8 at qd 8; 3318.5 vs 3307.4
   at qd 128 — i.e. within noise, sometimes slightly *worse*). They contribute a great deal at
   4 KiB (§2.3). This is exactly what the registered-buffer mechanism predicts: the per-I/O cost the
   registration removes is *page pinning and kernel mapping setup*, which is a fixed per-request
   cost amortised over the request's length. At 576 KiB per request it amortises to nothing; at
   4 KiB it is the dominant cost.
4. **3.4 GiB/s is the ceiling, and it is reached at 576 KiB already.** 2 MiB, 4 MiB and 7 MiB all
   top out at 3.24–3.41 GiB/s. There is no larger-block prize waiting.

### 2.3 Block size × queue depth, all sizes

`MiB/s`, `O_DIRECT`, random offsets. `pread` rows are in italics for readability.

| bs | mode | qd1 | qd8 | qd32 | qd64 | qd128 |
|---|---|---|---|---|---|---|
| **4096** | *pread* | *55.3* | *55.5* | *55.4* | *54.6* | *53.2* |
| | uring | 54.7 | 61.5 | 158.6 | 598.9 | 1093.8 |
| | uring+regbuf | 55.2 | 110.2 | 233.4 | 894.3 | **1270.7** |
| **589824** | *pread* | *1364.3* | *1658.3* | *1580.3* | *1696.5* | *1598.9* |
| | uring | 1441.3 | **3383.8** | 3211.3 | 3271.8 | 3307.4 |
| | uring+regbuf | 1721.4 | 3382.3 | 3396.3 | 3310.2 | 3318.5 |
| **2097152** | *pread* | *2722.7* | *2692.7* | *2644.0* | *2572.9* | *2565.5* |
| | uring | 2674.2 | 3378.2 | 3286.1 | 3387.1 | 3347.3 |
| | uring+regbuf | 2766.5 | 3335.4 | 3334.8 | 3310.0 | 3242.8 |
| **4194304** | *pread* | *3054.8* | *2992.8* | *3002.3* | *2932.9* | *2991.8* |
| | uring | 3014.7 | 3382.9 | 3293.5 | 3263.3 | 3310.0 |
| | uring+regbuf | 3082.0 | 3294.1 | 3326.3 | 3234.1 | 3326.7 |
| **7340032** | *pread* | *3072.3* | *3126.2* | *3035.3* | *3145.3* | *3114.8* |
| | uring | 3110.3 | 3378.7 | 3354.8 | 3334.5 | **3405.6** |
| | uring+regbuf | 3208.7 | 3355.3 | 3241.5 | 3317.2 | 3289.3 |

Three findings that change the project's assumptions:

- **`pread` at 4 MiB is already 3.05 GB/s** — i.e. `pread` is *not* the bottleneck at large extents,
  only at small ones. The problem the project has is a **576 KiB-and-below** problem.
- **4 KiB is a catastrophe in every mode**, and the reason is queue depth, not bandwidth:
  `pread` cannot exceed 55 MiB/s (14 000 IOPS) at *any* depth because it is one synchronous request
  at a time; io_uring at qd 128 reaches 1093.8 MiB/s (280 002 IOPS) — **19.7× `pread`** — and with
  registered buffers 1270.7 MiB/s (325 303 IOPS), a further **+16%**. The device is demonstrably
  capable of 325 k IOPS on 4 KiB random O_DIRECT reads; the 4 KiB row of the project's read path is
  CPU/pinning-bound, not device-bound.
- **Registered buffers are a small-extent optimisation and only that.** +16% at 4 KiB/qd128,
  0% at ≥ 576 KiB. This is corroborated by the mechanism the man page states — registration pins the
  memory and creates long-term mappings so the kernel stops doing "a map and unmap for each IO every
  time IO is performed to that region" and stops "manipulating the page reference counts for each IO"
  (`https://man7.org/linux/man-pages/man3/io_uring_register_buffers.3.html`,
  `https://manpages.debian.org/unstable/liburing-dev/io_uring_registered_buffers.7.en.html`
  `[verified-source]`). Note the registration needs `RLIMIT_MEMLOCK` headroom.

### 2.4 The flag modes that were measured (`sqpoll`, `defer`)

128 MiB budget per row, so these are not byte-comparable with the 256 MiB rows above but *are*
throughput-comparable.

| bs | mode | qd32 | qd64 | qd128 |
|---|---|---|---|---|
| 4096 | sqpoll | 1002.9 / 256741 | — | **1451.1 / 371494** |
| 4096 | defer | 953.0 / 243970 | — | 1311.8 / 335832 |
| 4096 | defer+regbuf | 972.9 / 249070 | — | 1425.7 / 364973 |
| 4194304 | sqpoll | 3378.7 / 845 | — | 3350.5 / 838 |
| 4194304 | defer | 3357.0 / 839 | — | 3425.0 / 856 |
| 4194304 | defer+regbuf | 3290.2 / 823 | — | 3287.4 / 822 |
| 4096 / 4194304 | **iopoll** | **ERR** | — | **ERR** |

- At **4 KiB**, `sqpoll` is the best absolute result measured on this host anywhere: **1451.1 MiB/s
  at qd 128, 371 494 IOPS**, beating plain uring (1093.8) and uring+regbuf (1270.7). `defer+regbuf`
  is second at 1425.7.
- At **4 MiB**, `sqpoll` and `defer` are indistinguishable from plain uring (3350–3425 vs
  3263–3406) — **at large extents, every submission mode converges.** The syscall overhead that
  SQPOLL eliminates is a fixed per-request cost, and at 4 MiB per request it is noise.
- **At 576 KiB all three would also converge**, by §2.2's pattern; it was not measured at that size
  in this sweep, and I have not extrapolated it into the table.

Both of these are cheap to apply and need no patch: `IORING_SETUP_SQPOLL` and
`IORING_SETUP_DEFER_TASKRUN|IORING_SETUP_SINGLE_ISSUER` are two bits in the ring setup call. The
cost is a dedicated core for SQPOLL (`sq_thread_idle` controls its sleep).

---

## 3. Mechanism: where the bytes actually go

This is the most valuable measurement in the set, because it is the only one that explains a
**project-level** number (the ~2.15× read amplification) from a **kernel-level** constant.

### 3.1 The per-fault fetch size, measured

`faultexp.c` (`[verified-here, prior]`) evicts a window of the real GGUF with
`posix_fadvise(POSIX_FADV_DONTNEED)` **before** mapping it (the comment in the source is correct and
important: "page-cache pages that are pinned by an active VMA will not be dropped by fadvise"),
`mmap`s it `MAP_SHARED`, then walks a **589 824 B stride** touching exactly one byte, and accounts
resident pages with `mincore()` before and after each touch.

Reproduced this review, plus the variants `faultexp.c` did not implement, which I added in
`scripts/io/faultexp2.c`. All rows: window 32 MiB, step 589 824 B, 57 probe touches.

| mode | faults | bytes fetched | **avg bytes per fault** | resident after |
|---|---|---|---|---|
| `none` (ordinary fault) | 57 | 7.1 MiB | **131072** | 7.1 MiB |
| `MADV_RANDOM` | 57 | 0.2 MiB | **4096** | 0.2 MiB |
| `MADV_SEQUENTIAL` | 57 | 7.1 MiB | **131072** | 7.1 MiB |
| `MADV_HUGEPAGE` | 15 | 18.4 MiB | **1288055** | 32.0 MiB |
| `MADV_COLLAPSE` | — | — | **FAILED, `errno=22` (`EINVAL`)** | 7.1 MiB (as `none`) |
| `MADV_POPULATE_READ` on the 4 KiB page | 57 | 7.1 MiB | **131072** | 7.1 MiB |
| `MADV_POPULATE_READ` on the whole 576 KiB slice | 57 | 26.6 MiB | **488430** | 32.0 MiB |
| `POSIX_FADV_RANDOM` on the fd + ordinary fault | 57 | 7.1 MiB | **131072** | 7.1 MiB |
| `POSIX_FADV_SEQUENTIAL` on the fd + ordinary fault | 57 | **14.2 MiB** | **262144** | 14.2 MiB |
| window 256 MiB, `none` | 456 | 57.0 MiB | 131072 | 57.0 MiB |
| window 256 MiB, `FADV_SEQUENTIAL` | 456 | 113.9 MiB | 262000 | 113.9 MiB |

**The 131 072 B number is exactly `read_ahead_kb`.** `/sys/block/nvme0n1/queue/read_ahead_kb` = 128
on this host `[verified-here]`, and the sysfs store function *is* the readahead window:
`read_ahead_kb_store` does `bdi->ra_pages = read_ahead_kb >> (PAGE_SHIFT - 10)`
(`block/blk-settings.c` / `backingdev.c`, artifacts in the scratch dir; the write path is
`bdi->ra_pages` and the read path is `BDI_SHOW(read_ahead_kb, K(bdi->ra_pages))`)
`[verified-source]`. 128 KiB ÷ 4 KiB = 32 pages = `ra_pages`. The measurement and the sysfs file
agree to the byte.

### 3.2 What that means for a 576 KiB expert slab

`[arithmetic on verified-here numbers]`:

- Logical stride: 589 824 B = 576 KiB per expert slice.
- Bytes fetched per probe with the default policy: 131 072 B = 128 KiB.
- Ratio for a *single* 4 KiB touch: **131072 / 4096 = 32×**.
- **Readahead per slice: `131072 ÷ 589824 = 0.222`** and `262144 ÷ 589824 = 0.444` under
  `FADV_SEQUENTIAL`. **That 0.222 is the honest expression of the waste at fault granularity**: for
  every 576 KiB slice the engine touches, the fault path drags in 128 KiB and the engine uses the
  4 KiB it asked for, because the next fault is 576 KiB away, not 4 KiB.
- **Faults needed to cover one slice**: `589824 ÷ 131072 = 4.5` at the default window and
  `589824 ÷ 262144 = 2.25` under `FADV_SEQUENTIAL`. This is the number that matters when the touch
  granularity is *not* one page per slice.
- **The measurement's own aggregate ratio is 0.221**: 7.1 MiB fetched against 57 × 589 824 B =
  32.06 MiB of addressed stride `[verified-here]`. (I computed 1.67× here in an earlier draft by
  dividing the window size rather than the logical stride; **that figure was wrong and is corrected
  to 0.221×**.) The 256 MiB window row confirms it independently: 456 × 131 072 B = 57.0 MiB against
  456 × 589 824 B = 256.5 MiB = **0.222×**.
- **The kernel constants that produce this** are `fault_around_pages = 65536 >> PAGE_SHIFT`
  (16 pages = 64 KiB) in `mm/memory.c` **and** the readahead window `ra_pages = read_ahead_kb >> 2`;
  the measured 131 072 B is the readahead window, not the fault-around size `[verified-source]`.
- **`/proc/sys/vm/fault_around_bytes` does not exist on this kernel and cannot be used to fix this.**
  It was a sysctl historically; in 6.12 it is a **debugfs** attribute —
  `#ifdef CONFIG_DEBUG_FS` … `debugfs_create_file_unsafe("fault_around_bytes", 0644, NULL, NULL, …)`
  in `mm/memory.c` `[verified-source]`. Confirmed absent on this host
  (`cat /proc/sys/vm/fault_around_bytes` → "No such file or directory" `[verified-here]`). Its
  setter rejects values above `PTRS_PER_PTE` pages (512 pages = 2 MiB), and rejects the whole thing
  under the default `CONFIG_DEBUG_FS`-off build.

**This is the mechanism behind the project's read amplification.** `docs/SEMIF_LLAMACPP_SSD.md:442`
states the project-level figure: *"decode reads 42.0 GB for one ubatch whose logical expert
footprint is ~19.5 GB (≈2.15×)"* `[verified-here, prior]`. **Be careful not to conflate this with
the 1.67×/1.78× that appeared in an earlier draft of this file, which was a different and incorrect
quantity**: the 0.22×/0.44× is readahead-spill only
(one 128–256 KiB window per 576 KiB slice); the project's 2.15× is *total bytes read / logical
expert footprint for a whole decode ubatch*, which also contains the cap-induced re-read
(documented separately as 19.3 GB → 41.0 GB for a **2.12×** factor when the cgroup cap goes from
16 GiB to 8 GiB, `docs/research/03-fragmentation-and-coupling.md:281` `[verified-here, prior]`). The
readahead term is the *floor* under the 2.15×; the cap term is what pushes it to 2.15. `docs/DEVELOPMENT_LOG.md:547`
and `docs/OFFLOAD_PROJECTS_ANALYSIS.md:125` already say the 2.15× is "load-bearing prefetch, not
waste" `[verified-here, prior]` — **this measurement confirms that claim mechanistically and also
prices it**: of the readahead fetched, only 4096 B of every 131072 B was asked for.

### 3.3 Why `MADV_RANDOM` costs 6.6× for 1.6× fewer bytes — the explanation

The project measured, before this dossier: *"`MADV_RANDOM` 42.0 → 26.0 GB but 56.0 → 368.0 s"*
`[verified-here, prior]` — i.e. **1.62× fewer bytes for 6.57× more wall time.** The mechanism
measurement makes this arithmetic, not mystery:

- `MADV_RANDOM` sets `VM_RAND_READ`, and `do_sync_mmap_readahead` returns immediately when
  `vm_flags & VM_RAND_READ` — no readahead at all (`mm/filemap.c` `[verified-source]`). Measured
  consequence: **4096 B per fault instead of 131 072 B, a 32× reduction in bytes fetched per fault.**
- The engine is not walking sequentially and does not reuse the slice; it walks a **stride of
  589 824 B, touching 4096 B at each stop**. Covering one slice's *bytes* with the default window
  takes **`589824 ÷ 131072 = 4.5` faults, and `589824 ÷ 4096 = 144` faults with no readahead at all**
  — **32× more requests.** Each is a separate random I/O into the same 576 KiB region.
- The device delivers ~3.4 GB/s at 128 KiB per request and only ~1.27 GB/s at 4 KiB per request even
  at qd 128 with registered buffers (§2.3) — and the fault path is far shallower than qd 128
  because a faulting thread blocks. 32× the requests at ~2.7× the per-request cost gives the
  observed order-of-magnitude. **`MADV_RANDOM` trades read bytes for I/O count, and on NVMe the I/O
  count is what you cannot afford.** The project's decision to abandon it is correct and now
  explained.

### 3.4 `POSIX_FADV_RANDOM` on the fd does *nothing* to the fault path

New this review, and worth knowing before anyone tries it as a "cheaper `MADV_RANDOM`":
`POSIX_FADV_RANDOM` on the file descriptor sets `FMODE_RANDOM` (`mm/fadvise.c`
`[verified-source]`) but the measured per-fault fetch is **still 131 072 B**. The fd-level advice
does not reach the mmap fault path the way `madvise(MADV_RANDOM)` does. Conversely
`POSIX_FADV_SEQUENTIAL` *does* — it doubles the window to 262 144 B through
`file->f_ra.ra_pages`, which is the same state `mmap` faults read. **Conclusion: the fd-level
advice is not a lever for the mapping; `madvise()` on the mapping is.**

### 3.5 File-backed huge pages are unavailable on this kernel. This is settled, not inferred.

`MADV_HUGEPAGE` looks like the win — 15 faults instead of 57, 18.4 MiB fetched for 32 MiB of window —
but **the mechanism is not huge pages.** Evidence, all `[verified-here]`:

1. `mmap` the GGUF, `madvise(MADV_HUGEPAGE)` → `rc=0`; then examine `/proc/self/smaps` for that
   exact VMA: **`FilePmdMapped: 0 kB`, `AnonHugePages: 0 kB`, `KernelPageSize: 4 kB`,
   `MMUPageSize: 4 kB`, `THPeligible: 0`.** No huge page was ever created.
2. `madvise(MADV_COLLAPSE)` on the same mapping returns **`-1`, `errno=22` (`EINVAL`)** — twice,
   independently (`faultexp` mode 3 and the Python probe).
3. `zcat /proc/config.gz`: **`# CONFIG_READ_ONLY_THP_FOR_FS is not set`** `[verified-here]`. The
   kernel man page is explicit that file/shmem MADV_COLLAPSE support exists "only if the kernel was
   configured with `CONFIG_READ_ONLY_THP_FOR_FS`"
   (`https://man7.org/linux/man-pages/man2/madvise.2.html` `[verified-source]`).
4. `/proc/sys/vm/nr_hugepages = 0` `[verified-here]` and no 1-GiB pages are reserved. 1-GiB pages via
   hugetlbfs are an anonymous-memory facility and cannot back a file mapping of a GGUF anyway.

So the `MADV_HUGEPAGE` result of 1 288 055 B per fault is **readahead**, and the kernel source says
exactly why: in `do_sync_mmap_readahead`, under `CONFIG_TRANSPARENT_HUGEPAGE`, when
`vm_flags & VM_HUGEPAGE`, the kernel sets **`ra->size = HPAGE_PMD_NR`** (512 pages = 2 MiB) and
then doubles it to 4 MiB if `VM_RAND_READ` is not set, before calling `page_cache_ra_order`
(`mm/filemap.c` `[verified-source]`). Measured average 1.288 MB/fault ≈ 20 × 64 KiB is consistent
with a 2 MiB window being consumed in pieces, and the `resident after = 32.0 MiB` for a 32 MiB
window confirms the whole window got pulled in. **`MADV_HUGEPAGE` is a way to change the kernel's
readahead target size, not a way to get 2 MiB page-table entries on this host.** That is still
useful — but it must be documented as what it is, and it is superseded by the cheaper and more
direct lever in §3.6.

### 3.6 `MADV_POPULATE_READ` on the whole slice is the best *measured* fetch pattern here

`madvise(MADV_POPULATE_READ)` "Populate[s] (prefault[s]) page tables readable, faulting in all pages
in the range just as if manually reading from each page … In contrast to `MAP_POPULATE`,
`MADV_POPULATE_READ` does not hide errors, can be applied to (parts of) existing mappings and will
always populate (prefault) page tables" — since Linux 5.14
(`https://man7.org/linux/man-pages/man2/madvise.2.html` `[verified-source]`).

Measured on this host (`faultexp2` modes 5 and 6, `[verified-here]`):

- Populating **one 4 KiB page** → 131 072 B fetched, identical to an ordinary fault. **No
  improvement**: if you only want one page, fault-around already gives you the readahead window, and
  POPULATE_READ is just a fault you triggered yourself.
- Populating the **whole 589 824 B slice** → 26.6 MiB fetched for a 32 MiB window = **0.83× the
  window**, 488 430 B per call, and the window ends up fully resident.

**This is the one API whose amplification factor is below 1.0**, it is a single syscall per expert
slice, and it is synchronous and therefore *stacks with queue depth only if issued from several
threads*. It is the honest replacement for the `preadv` prewarmer the project already tried and
rejected (§7) — with one crucial difference: it populates **the mapping the dequant kernels actually
read**, not a separate page-cache-only copy that a cgroup cap is free to reclaim. That distinction is
the reason the project's prewarmer backfired, and it is worth one experiment.

### 3.7 The readahead algorithm, and why 128 KiB (not 2 MiB) is what you get

From the kernel's own `mm/readahead.c` `DOC:` block and helpers `[verified-source]`,
`https://github.com/torvalds/linux/blob/master/mm/readahead.c`:

- Read-ahead is triggered "when an application read request (whether a system call or a page fault)
  finds that the requested folio is not in the page cache, or that it is in the page cache and has
  the readahead flag set".
- The size of the region is "normally determined from the size of the previous readahead"; it is
  "scaled, often doubled": `get_next_ra_size` returns `4 * cur` if `cur < max/16`, `2 * cur` if
  `cur <= max/2`, else `max`.
- `get_init_ra_size`: `newsize = roundup_pow_of_two(size)`; the comment gives the concrete ladder —
  "128k (32 page) max ra; 1-2 page = 16k, 3-4 page 32k, 5-8 page = 64k, > 8 page = 128k initial".
- The upper bound is the device's `ra_pages`, which `blk_apply_bdi_limits` sets to
  `max(lim->io_opt * 2 / PAGE_SIZE, VM_READAHEAD_PAGES)` with `bdi->io_pages = lim->max_sectors >> PAGE_SECTORS_SHIFT`
  (`block/blk-settings.c` `[verified-source]`).

On this host `optimal_io_size = 0` and `max_sectors_kb = 128`, so the device-derived term collapses
`ra_pages` to the **`VM_READAHEAD_PAGES` default of 128 KiB**, matching `read_ahead_kb = 128` exactly.
**The whole 131 072 B per-fault figure is one sysfs number on this system** — which is the good news,
because it is also writable (§5).

---

## 4. The host↔device transfer path, measured

`cuda_probe.cu` (`[verified-here, prior]`, compiled and the binary run again this review,
`/home/scribe/.hermes/cache/scratch/cuda_probe_out.txt`) — results on device 0 (RTX 5060 Ti):

### 4.1 Device capability bits

| attribute | value |
|---|---|
| compute capability / SMs | 12.0 (Blackwell) / 36 |
| `totalGlobalMem` / free at probe time | 15.5 GiB / 14.8 GiB |
| `concurrentManagedAccess` | **1** |
| `pageableMemoryAccess` | **1** |
| `directManagedMemAccessFromHost` | **0** |
| `canUseHostPointerForRegisteredMem` | **1** |
| `cuDeviceGetAttribute(88)` = `PAGEABLE_MEMORY_ACCESS` | 1 |
| `cuDeviceGetAttribute(100)` = `PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` | **0** |
| `cuDeviceGetAttribute(101)` = `DIRECT_MANAGED_MEM_ACCESS_FROM_HOST` | **0** |
| `VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED` | **1** |
| min / recommended VMM allocation granularity | **2097152 B** / 2097152 B |

### 4.2 Transfer throughput

| transfer | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|
| H2D pageable `cudaMemcpy` | 6.25 GB/s | 6.36 | 6.33 |
| H2D pinned `cudaMemcpy` | 6.50 | 6.48 | 6.43 |
| H2D pinned `cudaMemcpyAsync` | 6.41 | 6.52 | 6.46 |
| H2D `cudaHostAllocWriteCombined` async | 6.50 | 6.49 | 6.47 |
| D2H pageable `cudaMemcpy` | 6.42 | 6.49 | 6.55 |
| D2H pinned `cudaMemcpyAsync` | 6.59 | 6.60 | 6.57 |
| pinned alloc latency | 5.135 ms per 16 MiB | 5.142 | 5.187 |

### 4.3 Managed memory / UVM

| operation | result |
|---|---|
| `cudaMemPrefetchAsync` 4 GiB → device | **0.014 s = 300.95 GB/s** |
| `cudaMemPrefetchAsync` 4 GiB → host | **0.659 s = 6.52 GB/s** |
| GPU demand-fault over 4 GiB at 64 KiB/page stride, 64 Ki threads | 0.624 s = **6.89 GB/s** of host→device migration |
| GPU demand-fault, dense single stride (already resident) | 0.005 s = **936.13 GB/s** |

### 4.4 PCIe link state

Before warmup: `current_link_speed = 2.5 GT/s`, width ×8. **After warmup: 8.0 GT/s ×8**
on both `card1` and `card2` = PCIe 3.0 ×8. **Max is 32.0 GT/s ×8** (PCIe 5.0 ×8) — the cards are
linked at a quarter of their capability, and the measured 6.5 GB/s ceiling is what PCIe 3.0 ×8
predicts (~7.88 GB/s theoretical, ~6.5–6.9 achievable). `nvidia-smi topo -m` reports `PHB` between
the two GPUs (same PCIe host bridge), so GPU↔GPU does not get a fast path either.

### 4.5 What this settles

1. **Pinned memory buys almost nothing here** (6.43–6.52 vs 6.25–6.36 GB/s pageable: **+1–4%**).
   The project must not budget a speedup for it. It is still *required* for correctness of
   asynchronous overlap semantics, but it is not a bandwidth lever on this host.
2. **`cudaMemPrefetchAsync` to the device is not a transfer.** 300.95 GB/s for 4 GiB is four orders
   of magnitude above the PCIe link, so the pages were already resident; the call only touched page
   tables. Any plan that assumes "prefetch 4 GiB to VRAM at 300 GB/s" is reading a no-op. The
   **→ host direction (6.52 GB/s) is the real migration rate** because those pages genuinely were on
   the device. Always sanity-check a UVM prefetch number against the PCIe link before believing it.
3. **GPU demand paging over PCIe is ~6.9 GB/s, i.e. no faster than a plain pinned copy.** Fault-storm
   shape (64 Ki threads, one touch per 64 KiB) gave 6.89 GB/s; the dense repeat gave 936 GB/s
   because nothing migrated. **UVM oversubscription is therefore not a way to get more bandwidth —
   it is a way to avoid writing an explicit staging loop, at the cost of giving up control of what
   is resident.** With 2 × 16 GiB VRAM against a 19.46 GiB model, "fits" is not the question;
   "what should be resident" is, and UVM does not answer it.
4. **CUDA VMM is supported and 2 MiB-granular.** `cuMemCreate`/`cuMemMap` can map 2 MiB device
   chunks onto a contiguous virtual range, so per-expert slabs can be (un)mapped without churning
   the VA space or the allocator. This is the correct primitive for a per-expert VRAM slot pool.
5. **PCIe 3.0 ×8 is the real transfer ceiling** for any VRAM-resident plan: 6.5 GB/s, which is ~1.9×
   the NVMe's 3.4 GiB/s. **The PCIe link is not the bottleneck; the NVMe is.** Moving expert bytes
   NVMe → VRAM directly (GDS) would at best save the RAM bounce and the CPU, not the bytes.

---

## 5. Technique catalogue from the literature

Format per entry: **mechanism → the number its own source reports → is that number vendor or
independent → applicability to this setup.**

Throughout, "this setup" means: llama.cpp on the `siphon.cpp` fork, GGUF, x86-64, 62 GiB RAM,
2 × RTX 5060 Ti 16 GiB, one NVMe at a measured ~3.4 GiB/s ceiling, model 20 893 015 008 B of which
the expert tensors are 120 tensors = 40 layers × 3 tensors × 256 experts; 18 327 011 328 B
(17.07 GiB) of expert weight measured exactly from the GGUF's tensor-offset table `[verified-here]`.
**Not every layer is 576 KiB/slice**: 37 of the 40 layers are Q4_K = **1 769 472 B = 1 687.5 KiB per
expert (3 × 589 824 B exactly)**, while layers 0–2 are Q6_K = **2 039 808 B (1 992 KiB) per expert**,
which is 1.15× larger. Top-8 routing, `expert_count = 256`, `block_count = 40`,
`expert_used_count = 8`, `expert_feed_forward_length = 512`, `embedding_length = 2048`,
`architecture = qwen35moe`, GGUF v3, `general.file_type = 14` `[verified-here, read from the header]`.
The 576 KiB figure is therefore correct for 92.5% of layers and **1.15× too small for 7.5% of them**.

### 5.1 io_uring

**5.1.1 io_uring vs libaio vs SPDK vs threaded `pread`.**
- *Mechanism*: shared memory-mapped SQ/CQ rings, so submission and completion need no syscall per
  I/O; `io_uring_enter` batches and can be omitted entirely in SQPOLL mode.
- *Its own numbers* (SYSTOR '22, IBM Research Europe / VU Amsterdam, 4 KiB random unbuffered reads
  on Intel DC P3600 NVMe, 1 core / 2 cores): libaio 145 → 151 KIOPS; io_uring 171 → 182.5 KIOPS;
  io_uring + completion polling 171.6 → 178.8; io_uring + kernel poller (SQPOLL) **13.7 KIOPS on one
  core** (a "catastrophic performance loss", median latency **8 ms**, one to two orders of magnitude
  worse) but **261.4 KIOPS on two cores**; SPDK 305.6 → 313.6 KIOPS. Average syscalls/IO: libaio
  **2.00 flat**; io_uring 1.26 at qd 4 → 1.01 at qd ≥ 64. Source:
  `https://atlarge-research.com/pdfs/2022-systor-apis.pdf` `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed** (ACM SYSTOR '22, ten-core Xeon, 20 NVMe
  drives, fio 3.28, kernel 5.13).
- *Applicability*: **high, with the SQPOLL caveat measured on our own host**. Their 1-core SQPOLL
  collapse is exactly the risk our `sqpoll` rows might have shown — they did not, because our box has
  12 threads and the ring's poller got a core. **Do not deploy SQPOLL on a box where the poller
  shares a core with the reader.** Their libaio-flat-at-2.00-syscalls/IO is the same "no queue"
  result our `pread` column shows: libaio, like `pread`, is serial on this path.

**5.1.2 Registered (fixed) buffers.**
- *Mechanism*: register iovecs once; the kernel pins the memory and creates long-term mappings, so
  per-I/O work stops including "verify the memory is accessible to the process / pin the pages / set
  up kernel mappings" and page refcount churn.
- *Its own numbers*: PVLDB paper `io_uring for High-Performance DBMSs` (arXiv:2512.04859v1):
  "+RegBufs" improves YCSB throughput "by about 11%, reaching 238 k tx/s"; and for networking,
  registered buffers have "a negligible impact for small messages and increase latency slightly".
  Source: `https://arxiv.org/pdf/2512.04859v1`, artifact in
  `/home/scribe/.hermes/cache/scratch/iouring_paper.txt` `[verified-source]`. Man-page mechanism:
  `https://man7.org/linux/man-pages/man3/io_uring_register_buffers.3.html` and
  `https://manpages.debian.org/unstable/liburing-dev/io_uring_registered_buffers.7.en.html`
  `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed (PVLDB).** The man page is a doc.
- *Applicability*: **conditional, and our own table says the condition.** +16% at 4 KiB on this host,
  0% at ≥ 576 KiB. So: register buffers iff the read path ends up issuing 4 KiB requests (the
  metadata/index stream, the router, or an un-repacked expert layout). If the read path is 576 KiB
  slices, registration is a no-op and the `RLIMIT_MEMLOCK` cost is pure downside.

**5.1.3 IOPOLL.**
- *Mechanism*: busy-wait for completion from the device queue instead of taking an interrupt.
- *Its own numbers*: PVLDB: "+IOPoll" gives "an additional 21% throughput gain, reaching 376 k tx/s
  — single-threaded" on YCSB; and in a mostly-in-memory TPC-C configuration **it performs slightly
  worse than the interrupt-driven baseline because polling wastes CPU cycles when I/O operations are
  sporadic**; and for the multi-threaded scale-out, "+Passthru and +IOPoll in particular deliver
  substantial throughput improvements of 3.4–3.5×". For **latency-sensitive** workloads it reduces
  latency. Source: `https://arxiv.org/pdf/2512.04859v1` `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed.**
- *Applicability*: **zero on this host today — measured dead.** Every `iopoll` row returned
  `EOPNOTSUPP` (§2.1); `io_poll = 0` in sysfs; the man page requires device support and O_DIRECT.
  `io_poll` is RW, so it is *potentially* reachable without patching C (§6) — but it is a device/driver
  capability toggle, not a hint, and it may simply not be supported by this Kingston drive. **Do not
  plan on IOPOLL; treat it as a bonus if `echo 1 > io_poll` works.** Their "worse when sporadic"
  finding is also a direct warning for our workload, which is bursty by layer.

**5.1.4 SQPOLL.**
- *Mechanism*: a dedicated kernel thread polls the SQ ring, so the application never calls
  `io_uring_enter` to submit. Costs `sq_thread_idle` idle-timer behaviour and **one core**.
- *Its own numbers*: PVLDB: "+SQPoll … throughput increases by about 32% to 546k tx/s, corresponding
  to the cost previously spent in syscall and kernel-side processing" `[verified-source]`.
  SYSTOR: the 1-core collapse above `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed (both).**
- *Applicability*: **best-measured mode for 4 KiB on this host** (1451.1 MiB/s, 371 494 IOPS,
  §2.4) and **indistinguishable from plain io_uring at 4 MiB**. Given our 576 KiB granularity it is
  a small win, and it costs a core out of 12. Their warning stands: pin the poller to its own CPU
  (`IORING_SETUP_SQ_AFF` + `sq_thread_cpu`) or it will collapse.

**5.1.5 DEFER_TASKRUN.**
- *Mechanism*: `IORING_SETUP_DEFER_TASKRUN` runs task_work only on `io_uring_enter`, so the kernel
  does not preempt at arbitrary points.
- *Its own numbers*: PVLDB §2: "The DEFER_TASKRUN flag (DeferTR) only runs task_work on
  io_uring_enter calls, **making it the recommended mode** since it gives applications more control
  and eliminates unwanted preemptions." In their UDP/TCP latency study, "**DeferTR with NAPI yields
  the best overall latency, outperforming SQPoll**" `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed.**
- *Applicability*: **yes, and it is the cheapest of the four flags** — two bits, no dedicated core.
  Measured here: 1311.8 MiB/s at 4 KiB/qd128, 1425.7 with regbuf (both well above plain uring), and
  3425.0 MiB/s at 4 MiB/qd128, the highest 4 MiB number measured. Combine with
  `IORING_SETUP_SINGLE_ISSUER`.

**5.1.6 NVMe passthrough (`IORING_OP_URING_CMD`).**
- *Mechanism*: issue native NVMe commands from an io_uring ring via the char device (`/dev/ngX`),
  bypassing the generic block layer and the filesystem.
- *Its own numbers*: PVLDB: "+Passthru … yields an additional 20% gain, increasing throughput to
  300 k tx/s" `[verified-source]`; and "for writes at 128 KiB and reads at 256 KiB with +Passthru and
  +IOPoll enabled" a single core saturates a PCIe 5 SSD array "reaching up to 90 GiB/s for reads and
  50 GiB/s for writes"; but also "NVMe-passthrough is only supported until 512 KiB" and "Passthrough
  with flush … lack of filesystem support restricts it to flash-optimized database systems".
  FAST '24 `I/O Passthru: Upstreaming a flexible and efficient I/O Path in Linux` (Samsung + Meta):
  "FIO peak performance workloads show **16–40% higher IOPS** than block path"
  `https://www.usenix.org/system/files/fast24-joshi.pdf` `[verified-source]`.
- *Vendor or independent?* The FAST '24 paper's authors are **Samsung employees and a Meta employee —
  i.e. vendor-adjacent**, though published peer-reviewed and reporting a range that includes modest
  gains. PVLDB's is independent.
- *Applicability*: **low for us — it means giving up the filesystem.** Our model is a file on ext4
  inside a subdirectory tree; passthrough requires raw device access and would mean partitioning and
  managing the expert slabs as a raw device. For a 2.15× amplification problem caused by *readahead
  on a file mapping*, this buys 16–40% IOPS and costs the entire storage-management story. **Do not
  reach for this.** Also: their own 512 KiB cap is below our 4 MiB and 7 MiB block sizes.

**5.1.7 Multishot, provided buffers, ring buffers.**
- *Its own numbers* (PVLDB, networking section): "Multishot receive operations … are most efficient
  for workloads with small messages. Once message sizes exceed roughly 1 KiB, zero-copy receive
  becomes more efficient, and for very large messages (e.g., 13 KiB and above) even the normal
  single-shot receive path outperforms multishot" and "io_uring can also draw receive buffers from a
  kernel-managed pool (RingBufs), but **these perform worse than user-supplied buffers** and are only
  useful in multishot scenarios" `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed.**
- *Applicability*: **not applicable to storage reads.** Multishot and provided buffers are
  *receive*-side (network) features; `IORING_OP_READ` is single-shot by design. The transferable
  lesson is the threshold rule — **multishot loses at ≥ 13 KiB, and our unit is 589 824 B** — so if
  anyone proposes multishot for expert streaming, this kills it.

**5.1.8 Large blocks can backfire: worker-thread fallback.**
- *Mechanism*: above certain software/hardware limits, io_uring stops doing the I/O inline and hands
  it to asynchronous worker threads, "reintroducing latency and CPU overhead".
- *Its own numbers*: PVLDB: "(i) if the block size exceeds `max_hw_sectors_kb` (which can be 128 KiB
  if the IOMMU were enabled), workers are spawned even at low I/O depth; (ii) **with O_DIRECT,
  workers appear once the number of batched requests exceeds `nr_requests` (1023 on bare metal,
  127 in our cloud VM)**; (iii) when block sizes exceed 512 KiB (`max_segments`) asynchronous workers
  are again used internally" `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed.**
- *Applicability*: **directly, and it is a live constraint on this host.** `nr_requests = 256` here
  `[verified-here]`, between their bare-metal 1023 and cloud 127. Their rule warns against batching
  more than `nr_requests` O_DIRECT requests. Our sweep never exceeded qd 128 per ring, so we are
  inside it — **but a design that queues 512 expert reads on one ring would cross it and fall into
  the worker path.** Their `max_hw_sectors_kb = 128 KiB` case (IOMMU) is worth noting: our
  `max_hw_sectors_kb` is also 128 `[verified-here]`, and 589 824 B = 576 KiB exceeds it by 4.5× —
  so **our own expert slab is already larger than the block layer's single-request limit and is
  being split into multiple requests underneath us.** That is measured-indirect (it follows from the
  sysfs values, and the observed 3.38 GiB/s at 576 KiB vs 3.41 at 7 MiB is consistent with mild
  splitting cost) and worth one confirming experiment.

**5.1.9 Bigger blocks amortise CPU.**
- *Its own numbers*: PVLDB: "larger blocks substantially reduce CPU cost per byte … With sufficiently
  large requests, a single core saturates the PCIe 5 SSD array, reaching up to 90 GiB/s for reads and
  50 GiB/s for writes … This point is reached for writes at 128 KiB and reads at **256 KiB**"
  `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed** (but on a PCIe 5 SSD array, ~26× our
  bandwidth).
- *Applicability*: **yes, and it is already satisfied.** Our 576 KiB slab is above their 256 KiB
  CPU-saturation knee, which is consistent with our own table showing 576 KiB and 7 MiB landing in
  the same 3.2–3.4 GB/s band. **No further gain from making the unit bigger**; the gain would be in
  making it *smaller* without losing throughput, which §2.3 says is exactly what 4 KiB cannot do.

**5.1.10 Registered file descriptors.** PVLDB: "registered FDs … for 4 KiB page-aligned storage I/O,
do not negatively impact performance" `[verified-source]`. Applicability: cheap, safe, small; it
removes a per-request fd table lookup. One line in a patch. **Independent, peer-reviewed.**

### 5.2 Linux AIO vs io_uring vs threaded `pread`

- *Mechanism*: libaio is two syscalls (`io_submit` + `io_getevents`) per I/O, interrupt-driven, and
  **O_DIRECT-only**; threaded `pread` is N blocking threads; io_uring is ring-based.
- *Its own numbers*: SYSTOR '22 — libaio's **2.00 syscalls/IO at every depth** vs io_uring's 1.01 at
  qd ≥ 64; libaio peaks at 145 KIOPS (1 core) / 151 (2 cores) where io_uring reaches 171/182.5 and
  SPDK 305.6/313.6; "**libaio … achieves very similar throughput** [to io_uring] **up to a queue
  depth of 16**" (79 vs 72 KIOPS) but falls behind at higher depth because "the higher CPU efficiency
  of iou, which incurs fewer system calls per I/O operation than libaio" `[verified-source]`.
- *Vendor or independent?* **Independent, peer-reviewed.**
- *Applicability*: **libaio is strictly dominated — do not adopt it.** On this host, threaded `pread`
  was measured directly: 8 concurrent processes = 2930.2 MiB/s, *below* one io_uring ring at qd 8
  (3383.8) `[verified-here]`. This is the quantitative confirmation of the project's own observation
  that its `4/3 × nproc` thread-oversubscription win (54.6 → 38.4 s, −29.6% `[verified-here, prior]`)
  is "a substitute for queue depth" — **and the substitution is measurably worse than the real
  thing**: threads reach 2930 MiB/s at 8 cores; a single ring reaches 3383 MiB/s at qd 8 on one core.

### 5.3 `O_DIRECT` alignment and block-size effects

- *Mechanism*: `O_DIRECT` bypasses the page cache; the buffer, offset and length must satisfy the
  device's alignment requirement. It is also the precondition for IOPOLL and for registered-buffer
  zero-copy.
- *Its own numbers*: our own sweep, §2.2–2.3 `[verified-here, prior]`. Plus, project-internal: a
  12-clone survey found `samosa-chat` measuring "**0.8 GB/s buffered vs 2.3+ GB/s O_DIRECT**"
  `[verified-here, prior]` (from `docs/research/03-fragmentation-and-coupling.md:203`).
- *Vendor or independent?* Ours is measured here. The `samosa-chat` figure is an in-repo survey of a
  third-party project — **treat as one project's self-report, not an independent measurement.**
- *Applicability*: **high.** This host: `logical_block_size = 512`, `physical_block_size = 512`,
  `minimum_io_size = 512`, `optimal_io_size = 0` `[verified-here]`. So the alignment contract is
  512 B, not 4096 B, and **the project's 7 MiB and other non-page-aligned slab strides are legal
  for `O_DIRECT`.** The `optimal_io_size = 0` is what collapses `read_ahead_kb` to the generic
  128 KiB default (§3.7) — the drive reports no preferred I/O size, so the kernel picks its own.
  **`align.c` was written to test whether non-page-aligned slab strides cost bandwidth; its output
  was not captured in any artifact I can find** (see §8).

### 5.4 GPUDirect Storage / cuFile

- *Mechanism*: DMA straight from the storage device into GPU VRAM, bypassing the CPU bounce buffer.
  Requires the `nvidia-fs` kernel module; uses an `O_DIRECT` file descriptor; "outside of
  compatibility mode, the APIs will fail if O_DIRECT is not possible"; since CUDA 12.2 / GDS 1.7 the
  APIs work in a compatibility mode (configured in `cufile.json`) that falls back to the page cache
  for non-`O_DIRECT` fds.
- *Its own numbers*: **the NVIDIA documentation reports no throughput figures**; it describes the
  path and its requirements only. Sources: `https://docs.nvidia.com/gpudirect-storage/o-direct-guide`
  and `https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html` `[verified-source]`.
- *Vendor or independent?* **Vendor documentation.** NVIDIA is the party that sells the feature.
  This is the textbook `[vendor]` entry: the *capability* is well documented, the *benefit* is not
  quantified by its own documentation.
- *Applicability*: **not reachable on this host, and now verified rather than assumed.** The user-space
  half exists — `libcufile.so.1.13.1` and `libcufile_rdma.so.1.13.1` are in `/usr/local/cuda-12.8`,
  and `ldconfig` resolves them through `/usr/local/cuda/targets/x86_64-linux/lib/` (which is the
  12.8 tree) `[verified-here]`. But `lsmod | grep nvidia_fs` is **empty** and
  `/proc/driver/nvidia-fs` **does not exist** `[verified-here]`. Without the kernel module GDS cannot
  be active. Additionally, consumer GeForce drivers do not ship `nvidia-fs`. **And the physics does
  not favour it either: GDS would bypass the RAM bounce, but our bottleneck is the NVMe (3.4 GiB/s)
  against a PCIe 3.0 ×8 link at 6.5 GB/s — GDS cannot make the drive faster.** Its real value would
  be freeing CPU during the transfer and removing a 576 KiB memcpy per slab; both are worth
  measuring, neither is worth a driver project.
- **Project-specific find, and a caveat.** The `siphon.cpp` fork in this repo
  (`offload_projects/siphon.cpp/`) already contains a complete, dynamically-loaded cuFile client:
  `ggml/src/ggml-cuda/dstorage_loader_gds.cpp` (Linux) alongside `dstorage_loader.cpp` (Windows
  DirectStorage), a `DSLoader` ABI in `dstorage_loader.h`, a `DStorageSlotManager`
  (`src/llama-dstorage-slots.{h,cpp}`, 1013 + ~7800 lines), CLI flags `-dsm`/`--dstorage-moe` and
  `--dstorage-moe-prefetch` in `common/arg.cpp`, and tests `tests/test-dstorage-stream-bench.cpp`
  and `tests/test-dstorage-cache-layout.cpp`. It is compiled on Linux via
  `src/CMakeLists.txt`: `target_sources(llama PRIVATE ../ggml/src/ggml-cuda/dstorage_loader_gds.cpp)`
  with `target_link_libraries(llama PRIVATE ${CMAKE_DL_LIBS})` — i.e. **it builds cleanly without GDS
  installed and reports availability at runtime** (`ds_loader_available()` → `ensure_cufile_loaded()`)
  `[verified-here, read from the source]`. **Caveats that must travel with this**: (a) the loader
  uses `dlopen` to bind cuFile symbols, so `ds_loader_available()` should return 0 here — **I did not
  run it**, and it should be run before any plan assumes either outcome; (b) the fork carries a
  `dstorage_loader.cpp` whose header comment says "Expert storage loader for Ollama" **including a
  path-typed `ds_loader_read(DSLoaderHandle, const wchar_t*, …)` signature that is
  Windows/`wchar_t`-shaped**, so provenance and licensing of that file are the project's business,
  not mine; (c) `tests/test-dstorage-stream-bench.cpp` links `${CMAKE_DL_LIBS}` but not
  `cuda.lib`/`cudart`, and its own registration helper uses `std::wstring` path widening.

### 5.5 Pinned host memory + `cudaMemcpyAsync` overlap

- *Mechanism*: page-locked host memory lets the DMA engine run without CPU involvement, and
  `cudaMemcpyAsync` on a stream overlaps with compute.
- *Its own numbers*: our measurement, §4.2 — pinned buys **+1–4%** over pageable on this host.
- *Vendor or independent?* NVIDIA's guidance that pinned memory is required for real asynchrony is
  **vendor documentation** and is correct as a *semantic* claim. The *bandwidth* benefit is
  measured here at essentially zero.
- *Applicability*: **use pinned buffers because you need overlap, not because you need bandwidth.**
  Also note `cudaHostAlloc` costs **5.1–5.2 ms per 16 MiB** on this host, so a slot pool must
  allocate once at startup, never per slab.

### 5.6 CUDA VMM (`cuMemCreate` / `cuMemMap`), `cudaMemAdvise`, `cudaMemPrefetchAsync`, HMM

- *Mechanism*: VMM gives per-2-MiB-chunk virtual→physical mapping control on the device, so memory
  can be (un)mapped at slab granularity without an allocator. `cudaMemAdvise`/`cudaMemPrefetchAsync`
  steer UVM residency. HMM is the kernel-side infrastructure that lets device memory participate in
  the normal kernel memory path, including CPU page-table mirroring.
- *Its own numbers*: our measurement, §4.1, §4.3.
  `VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 1`, granularity 2 MiB; `concurrentManagedAccess = 1`,
  `pageableMemoryAccess = 1`, but `directManagedMemAccessFromHost = 0` and
  `PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 0`; prefetch →dev 300.95 GB/s (**not a transfer**),
  →host 6.52 GB/s, GPU demand-fault 6.89 GB/s.
- *Vendor or independent?* Kernel HMM documentation is a **primary source** for the mechanism:
  "Provide infrastructure and helpers to integrate non-conventional memory (device memory like GPU
  on board memory) into regular kernel path … HMM also provides optional helpers for SVM (Share
  Virtual Memory), i.e., allowing a device to transparently access program addresses coherently with
  the CPU" `https://docs.kernel.org/mm/hmm.html` `[verified-source]`. CUDA API semantics are
  **vendor documentation**; the *numbers* above are measured here.
- *Applicability*: **VMM: yes, as a slot-pool primitive** (2 MiB granularity matches our expert
  geometry poorly — a 576 KiB slab wastes 1.5 MiB of a 2 MiB chunk, i.e. **72% slack** — so pack
  experts, or accept the waste against 16 GiB). **UVM/prefetch as a bandwididth lever: no**, per
  §4.5(2)(3): the only honest migration rate is 6.5–6.9 GB/s, which is below PCIe 3.0 ×8's own
  theoretical 7.88 GB/s and 1.9× our NVMe. **`directManagedMemAccessFromHost = 0`** specifically means
  the CPU cannot access managed memory directly on this configuration — every host touch is a
  migration.

### 5.7 Huge pages (2 MiB / 1 GiB) and `MADV_HUGEPAGE`

- *Mechanism*: fewer TLB entries; for *file* mappings, the kernel's read-only THP machinery, gated by
  `CONFIG_READ_ONLY_THP_FOR_FS`, and the mapping must be naturally huge-page-aligned. Additionally —
  and this is the part that surprises people — `MADV_HUGEPAGE` on a file mapping **changes the
  kernel's readahead target size to 2 MiB** regardless of whether a huge page is ever created.
- *Its own numbers*: measured here, §3.5: `MADV_HUGEPAGE` → 15 faults, 18.4 MiB fetched,
  1 288 055 B/fault, window fully resident — **with `FilePmdMapped: 0 kB` and `THPeligible: 0`**.
  `MADV_COLLAPSE` → `EINVAL`. Sources: kernel doc
  `https://docs.kernel.org/admin-guide/mm/transhuge.html` ("Setting 'never' in all sysfs THP controls
  does **not** disable Transparent Huge Pages globally. This is because `madvise(...,MADV_COLLAPSE)`
  ignores these settings and collapses ranges to PMD-sized huge pages unconditionally") and the man
  page's `CONFIG_READ_ONLY_THP_FOR_FS` requirement, both `[verified-source]`; kernel source
  `mm/filemap.c` `do_sync_mmap_readahead` `[verified-source]`.
- *Vendor or independent?* **Primary kernel sources.**
- *Applicability*: **`CONFIG_READ_ONLY_THP_FOR_FS` is not set on this kernel `[verified-here]`, so
  file-backed 2 MiB pages are impossible here, full stop.** `MADV_HUGEPAGE` remains usable, but as a
  *readahead-window* control, and it is dominated by the direct lever (§3.7 / §6). 1 GiB pages:
  `nr_hugepages = 0`, hugetlbfs is anonymous-only. **The honest summary is: do not pursue huge pages
  on this host; pursue `read_ahead_kb`.**

### 5.8 `madvise(MADV_POPULATE_READ/WRITE)`

- *Mechanism*: prefault page tables over an existing mapping range as if reading every page, but
  without performing the access; it does not hide errors the way `MAP_POPULATE` does. Since 5.14.
- *Its own numbers*: measured here, §3.6 — one 4 KiB page: 131 072 B (no gain); the whole 589 824 B
  slice: **488 430 B per call, 0.83× of window**. Source for semantics:
  `https://man7.org/linux/man-pages/man2/madvise.2.html` `[verified-source]`.
- *Vendor or independent?* Primary man page + measurement here.
- *Applicability*: **the best measured fetch pattern on this host, and it is one syscall per slice.**
  This is the technique to actually try against the project's amplification number, and — unlike the
  prewarmer the project already rejected (§7) — it populates *the mapping the compute kernels read*,
  so the cgroup cannot reclaim it out from under the engine.

### 5.9 Readahead

- *Mechanism* and *numbers*: §3.1–3.7 `[verified-here, prior]` + kernel sources `[verified-source]`:
  `https://github.com/torvalds/linux/blob/master/mm/readahead.c`,
  `https://www.kernel.org/doc/html/v5.4/block/queue-sysfs.html` (`read_ahead_kb (RW)`: "Maximum number
  of kilobytes to read-ahead for filesystems on this block device").
- *Vendor or independent?* **Primary kernel sources.**
- *Applicability*: **the number one lever.** `read_ahead_kb` is RW and it *is* `ra_pages`
  (`bdi->ra_pages = read_ahead_kb >> (PAGE_SHIFT - 10)`). On this host it is 128 and the per-fault
  fetch is 131 072 B — the two agree exactly. Raising it is one root sysfs write with no patch, and
  the mechanism measurement predicts the outcome: a larger window per fault means fewer faults per
  slice *and* more spill past the slice end. **The trade is measurable with `faultexp2.c` before
  anything else changes.**

### 5.10 NVMe queue-sysfs knobs

Primary source for every definition: `https://www.kernel.org/doc/html/v5.4/block/queue-sysfs.html`
`[verified-source]`. Current values on this host, read this review `[verified-here]`:

| knob | this host | doc says |
|---|---|---|
| `scheduler` | **`none mq-deadline [kyber] bfq`** — active is **kyber**, not none | "Writing an io scheduler name to this file will switch control of this b…" |
| `nomerges` | **0** | "By default (0) all merges are enabled. When set to 1 only simple one-hit merges will be tried. When set to 2 no merge algorithms will be tried" |
| `rq_affinity` | **1** | "If this option is '1', the block layer will migrate request completions to the cpu 'group' that originally submitted the request … setting this option to '2' forces the completion to run on the requesting cpu" |
| `nr_requests` | **256** | "how many requests may be allocated in the block layer for read or write requests. Note that the total allocated number may be twice this amount" |
| `read_ahead_kb` | **128** | see §3.7 |
| `max_sectors_kb` | **128** | "This is the maximum number of kilobytes that the block layer will allow for a filesystem request. Must be smaller than or equal to the maximum size allowed by the hardware." |
| `max_hw_sectors_kb` | **128** | (hardware maximum) |
| `rotational` | **0** | correctly non-rotational |
| `io_poll` | **0** | "When read, this file shows whether polling is enabled (1) or disabled (0) … Writing any non-zero value will enable this feature." |
| `io_poll_delay` | **-1** | "It defaults to -1, which is classic polling … If set to 0, a hybrid polling mode is used" |
| `logical_block_size` / `physical_block_size` | **512 / 512** | |
| `minimum_io_size` / `optimal_io_size` | **512 / 0** | `optimal_io_size = 0` ⇒ `ra_pages` falls back to `VM_READAHEAD_PAGES` = 128 KiB |
| `write_cache` | `write back` | (irrelevant — read-only workload) |
| `nr_requests` on the *partition* | **unset / empty** | knobs are per-request-queue, and only `nvme0n1` showed values |

- *Vendor or independent?* **Primary kernel documentation**; values measured here.
- *Applicability*: **four of these are worth an experiment and all four need root**: take the
  scheduler off `kyber` to `none` (the project's own design doc already calls for it), set
  `nomerges=2`, set `rq_affinity=2`, and raise `read_ahead_kb`. **`sudo` requires a password on this
  host** (`sudo -n true` → "sudo: a password is required" `[verified-here]`), so none of them has
  been tested — see §8. `io_poll=1` is also a one-line write and would settle the IOPOLL question.
  **Do not confuse "the design doc recommends none/2/2" with "it has been measured" — it has not.**

### 5.11 SPDK / NVMe passthrough (userspace driver)

- *Mechanism*: map the NVMe PCI BAR into the process, drive queue pairs with MMIO, poll completions —
  no syscalls, no interrupts, no kernel in the path.
- *Its own numbers*: SPDK's own documentation: "We have measured up to **2.6 times more IOPS/core when
  using perf vs. fio** with the 4K 100% Random Read workload" (that is about *its own benchmark tool*,
  not about SPDK vs the kernel) `https://spdk.io/doc/nvme.html` `[verified-source]`.
  The independent comparison is SYSTOR '22: **SPDK 305.6 KIOPS on one core vs io_uring 171 and libaio
  145; 313.6 on two cores vs 261.4 and 151**; "SPDK is the only library capable of saturating the
  bandwidth of the drive, while all other approaches are CPU-bound"; io_uring + kernel polling "can
  deliver performance close to SPDK (within 10%) … **However, this performance needs twice as many
  CPU cores as SPDK**"; and SPDK "does not support Linux file system integration, and cannot benefit
  from many kernel storage services such as access control, QoS, scheduling, and quota management"
  `[verified-source]`.
- *Vendor or independent?* SPDK's 2.6× is **vendor** (Intel-authored project self-benchmarking). The
  SYSTOR comparison is **independent and peer-reviewed**, and it is the number to use: **SPDK ≈ 1.8×
  io_uring on one core in a 4 KiB random read workload.**
- *Applicability*: **no.** It costs the filesystem, the cgroup accounting that this project depends on
  (its entire methodology is a `MemoryMax` scope — `run_resident_35b_ssd.sh` uses
  `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`), and it requires binding the drive
  to `vfio-pci`, which would make the root filesystem inaccessible if done to `nvme0n1` (the only
  NVMe on the box, and `/` and `/home` are both on it). **On a single-device host this is not a
  technique, it is an outage.**

### 5.12 Windows DirectStorage

- *Mechanism*: a Windows API that "allow[s] games to make full use of high-speed storage (such as
  NVMe SSDs) that can deliver multiple gigabytes a second of small (eg 64kb) data reads with minimal
  CPU overhead", using the NVMe hardware queue and hardware-assisted GPU decompression; "DirectStorage
  only supports read operations".
- *Its own numbers*: **none in the README** — it points to samples including
  `GpuDecompressionBenchmark` "which reads the contents of a file, compresses it and then decompresses
  multiple ways while measuring the bandwidth and CPU usage". Sources:
  `https://github.com/microsoft/DirectStorage/blob/main/README.md` `[verified-source]`.
- *Vendor or independent?* **Vendor (Microsoft) documentation, with no performance figures.** Third
  party: a `kibbyd/llm_upper` repository has a `DIRECTSTORAGE_LLM_RESEARCH.md` proposing exactly this
  for weight streaming — **that is an individual's research note, not a measurement, and I did not
  verify its numbers.**
- *Applicability*: **Windows-only** (`d3d12.lib dxgi.lib ole32.lib`, D3D12 committed resources) —
  **not applicable to this Linux host.** It is relevant only as the ABI the `siphon.cpp` fork's
  Windows branch implements, and as the pattern (hardware queue + GPU decompression) that the Linux
  branch substitutes cuFile for.

### 5.13 EROFS / XFS / DAX effects on random-read amplification

- *Mechanism*: EROFS uses **fixed-output** compression units (pclusters) so a read of N compressed
  bytes decompresses in place without read amplification; SquashFS uses fixed-*input* chunks and
  suffers amplification. DAX maps a memory-like device's storage directly into user space, removing
  the page cache copy entirely.
- *Its own numbers*: EROFS big-pcluster — "Different from previous thoughts, which had fixed-sized
  pclustersize recorded in the on-disk compress index header, our on-disk design permits
  variable-sized pclustersize now", `https://lore.kernel.org/linux-erofs/20210401032954.20555-1-xiang@kernel.org`
  and `https://lore.kernel.org/linux-erofs/20210330003908.22842-2-hsiangkao@aol.com/t.atom`
  `[verified-source, design-level]`. **I did not find a primary EROFS document stating a throughput
  or amplification figure**; the 165 MB→16 MB and LZ4 30.9 vs 26.3 MiB/s figures circulating in this
  repo are `[unverified]` (see `docs/research/03-fragmentation-and-coupling.md:139`). DAX: "The DAX
  code removes the extra copy by performing reads and writes directly to the storage device. For file
  mappings, the storage device is mapped directly into userspace" and "The DAX code currently only
  supports files with a block size equal to your kernel's PAGE_SIZE", `https://docs.kernel.org/filesystems/dax.html`
  `[verified-source]`.
- *Vendor or independent?* Kernel project's own design documentation — **primary but self-reported**.
- *Applicability*: **EROFS: the design lesson transfers, the filesystem does not.** The lesson is
  "cut on output boundaries, one expert = one unit", which is already the project's part-(1) seam.
  EROFS itself is read-only and would need the GGUF repacked into an image, and it does nothing about
  the *readahead window* — which is where our measured amplification comes from. **DAX: impossible
  here.** DAX needs a memory-like device; this is a NAND NVMe with no DAX-capable block device, and
  the mount is ext4 on `/dev/nvme0n1p5`. `CONFIG_DAX=y` and `CONFIG_XFS_FS=m` are compiled
  `[verified-here]`, so an XFS-with-DAX experiment is *possible to mount*, but it would be pointless
  on this media. **The honest conclusion: no filesystem change addresses the amplification this
  dossier measured, because the amplification is in `mm/`, not in the filesystem.**

### 5.14 Driver-side prefetch / read look-ahead

- *Mechanism*: an NVMe device reading more than the host asked for, internally.
- *Its own numbers*: **none that I could verify.** I found no primary NVMe specification text and no
  vendor datasheet for the KINGSTON SFYRD4000G stating a host-invisible read-ahead behaviour. This
  is `[unverified]`. **Note the distinction that matters: this host's measured 131 072 B/fault is
  *provably* kernel-side (it equals `read_ahead_kb` to the byte and follows `MADV_RANDOM` to zero),
  so there is no evidence of device-side prefetch contributing to it.**

---

## 6. Reachable without patching C, vs. needs a llama.cpp patch or external process

### 6.1 Reachable right now — env vars, CLI flags, sysfs. No C changes.

| # | lever | how exactly | expected effect (measured basis) | risk |
|---|---|---|---|---|
| 1 | **`read_ahead_kb`** | `echo 512 > /sys/block/nvme0n1/queue/read_ahead_kb` (root) | Directly sets `ra_pages`; measured per-fault fetch is currently 131 072 B = this knob `[verified-here]`. Larger window ⇒ fewer faults per slice, more spill. **Measure with `faultexp2` mode 0 at each value before committing.** | over-fetch past slice ends; the trade is two-sided |
| 2 | **scheduler → `none`** | `echo none > /sys/block/nvme0n1/queue/scheduler` (root) | Host is currently on **`kyber`** `[verified-here]`; the project's design doc recommends `none` but has not measured it. kyber adds per-request classification and a latency target. | none material for a single-tenant host |
| 3 | **`nomerges=2`** | `echo 2 > …/queue/nomerges` (root) | Doc: "no merge algorithms will be tried (including one-hit or more complex tree/hash lookups)". Our offsets are random and 4096-aligned, so merging has nothing to merge — this is pure saved lookup. | none |
| 4 | **`rq_affinity=2`** | `echo 2 > …/queue/rq_affinity` (root), or via a udev rule | Forces completions onto the requesting CPU instead of the CPU group. With 12 threads, one NUMA node, and a 12-thread engine, completion locality matters. | none |
| 5 | **`io_poll=1`** | `echo 1 > …/queue/io_poll` (root) | Currently 0; every IOPOLL attempt returned `EOPNOTSUPP` `[verified-here]`. **This is the single cheapest test of whether IOPOLL's measured +21% (PVLDB) is available here.** | may fail or may raise CPU; `io_poll_delay=-1` (classic polling) burns cycles when I/O is sporadic — which is exactly our per-layer burst pattern, and PVLDB measured IOPOLL *worse* in memory-bound configurations |
| 6 | **`MADV_HUGEPAGE` via existing code** | none needed — `faultexp` uses it; a patch is needed to apply it to the model mapping | Sets the readahead target to 2 MiB (measured 1.288 MB/fault, window fully resident) **with no huge pages created** `[verified-here]`. Superseded by #1, which is finer-grained. | — |
| 7 | **Stop the fork's unconditional `POSIX_FADV_SEQUENTIAL`** | needs a compile (one line, §6.2) — **not** reachable at runtime in the current fork | That call doubles the readahead window to 262 144 B, i.e. **0.222 → 0.444 of each 576 KiB slice fetched** (measured, §3.1–3.2) `[verified-here]` | removing it may slow truly sequential loads (initial mmap scan) |
| 8 | **`POSIX_FADV_WILLNEED`** | runtime, but **measured dead** | Project already measured it moving `read_bytes` by 0.14 GB `[verified-here, prior]`; consistent with `force_page_cache_readahead` clamping `nr_to_read` to `max(bdi->io_pages, ra->ra_pages)` = 128 KiB `[verified-source, mm/readahead.c force_page_cache_ra]`. **Do not re-recommend without new evidence.** | — |
| 9 | **`SEMIF_LLAMA_UBATCH` / `--n-ubatch`** | existing env var / CLI flag | Not an I/O knob technically, but it *is* the bytes-per-pass knob (documented 67.0 → 42.0 GB `[verified-here, prior]`). Every byte-level technique in this dossier should be evaluated at the ubatch the project actually runs. | memory cap interaction |

### 6.2 Needs a small llama.cpp patch (one function, no new subsystem)

1. **Apply `madvise` to the model mapping instead of only `posix_fadvise` to the fd.**
   `src/llama-mmap.cpp` `impl()` currently calls `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)`
   before `mmap` (line 451 in the fork's copy), and `posix_madvise(addr, …, POSIX_MADV_WILLNEED)` for
   the prefetch window (line 463), and `posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)` only
   when `numa` is set (line 469) `[verified-here, read from source]`. The mapping is reachable; what
   is missing is `madvise()` on it. Adding `MADV_HUGEPAGE` (2 MiB readahead target) and/or removing
   the `SEQUENTIAL` fd advice are one- and two-line changes with a measured, quantified effect.
2. **A `MADV_POPULATE_READ` call over each expert slice before the gather.** Measured at
   **488 430 B per call, 0.83× of window** `[verified-here]` — the only sub-1.0 fetch pattern found.
   This belongs next to the existing per-slab dispatch, is one syscall per slab, and has an error
   return unlike `MAP_POPULATE` (which is the point: a failed populate is visible, a hidden fault is
   charged to whatever thread happens to touch the page).
3. **An `io_uring` read path at qd 8 for the 576 KiB slabs.** This is the 2.04× win of §2.2. Note it
   is a *replacement* for the thread oversubscription the project currently uses, and it needs one
   ring per I/O worker plus `IORING_SETUP_DEFER_TASKRUN|IORING_SETUP_SINGLE_ISSUER`, and registration
   of the slab-pool buffers **only if** the pool is 4 KiB-granular. **At qd 8, not 128** — the table
   shows nothing above qd 8 and `nr_requests = 256` on this host is the worker-fallback boundary.

### 6.3 Needs an external process or a research programme

1. **GPUDirect Storage** — `nvidia-fs` absent `[verified-here]`; consumer GeForce drivers do not ship
   it. The fork's loader exists but is an order of magnitude more work than the transfer layer is
   worth, and the NVMe is 1.9× slower than the PCIe link anyway (§5.4).
2. **SPDK** — would require binding the only NVMe on the box away from the kernel, taking `/` and
   `/home` with it, and would defeat the `systemd-run` cgroup methodology the project's numbers
   depend on (§5.11).
3. **NVMe passthrough / raw-device expert slabs** — sacrifices the filesystem for 16–40% IOPS when
   the measured problem is readahead amplification, i.e. the wrong end (§5.1.6).
4. **File-backed huge pages** — **impossible on this kernel** (`CONFIG_READ_ONLY_THP_FOR_FS` unset,
   `THPeligible: 0`, `MADV_COLLAPSE` → `EINVAL`). Not a research programme: a closed door (§3.5).
5. **UVM oversubscription as a bandwidth technique** — measured at 6.89 GB/s, no better than a plain
   pinned copy (§4.5). Use CUDA VMM for *slot control*, not for bandwidth.
6. **A user-space prewarmer** — already measured harmful here (§7).

### 6.4 What is already a **measured dead end** in this project — do not re-recommend

These are recorded so that no future plan re-treads them without new evidence. All
`[verified-here, prior]`, from `docs/STATUS-2026-09-22.md`, `docs/SEMIF_LLAMACPP_SSD.md`,
`docs/ASSEMBLED-DESIGN.md`, `docs/research/03-fragmentation-and-coupling.md`:

| dead end | measured outcome | why it failed (mechanism, where known) |
|---|---|---|
| **`POSIX_FADV_WILLNEED`** | moved `read_bytes` by **0.14 GB** across a whole run | `force_page_cache_readahead` clamps to `max(bdi->io_pages, ra->ra_pages)` = 128 KiB ⇒ an advise window costs more than it fetches |
| **userspace `preadv` prewarmer** | **42.0 → 56.1 GB — reads got worse** | clean unreferenced page-cache pages are the first reclaim victims under a cgroup cap; the engine had to re-read them. A *mapping*-populate (§3.6) is the form that does not have this failure mode, and it is not the same experiment. |
| **`MADV_RANDOM`** | **42.0 → 26.0 GB but 56.0 → 368.0 s (6.57× slower)** | measured here: 4 096 B/fault instead of 131 072 B ⇒ **32× more I/O requests** for the same bytes; on NVMe the request count is the scarce resource (§3.3) |
| **whole-file `O_DIRECT` load mode** | **SIGKILLed under the cap** | it allocates the whole model (19.46 GiB) as an un-cached buffer alongside everything else, inside an 8 GiB `MemoryMax` scope |
| **lossless compression of the 4-bit payload** | incompressible; best rANS **114–120 MB/s** against a 750+ MB/s SSD | the payload is already entropy-coded and the compressor is 6× slower than the device |
| **co-activation expert reordering** | **0% measured** | routing co-activation is not stable enough across passes to reorder for |
| **`use_extra_bufts=true` (repack)** | OOM | repack needs a second copy of the weights |

---

## 7. What I could not confirm

Stated as a list of open items, each with what would close it. **Every one of these is a hole in this
dossier, not a suggestion that the reader guess.**

1. **The other `faultexp.c` modes were not captured by the previous agent — I re-ran them, and the
   numbers in §3.1 are mine.** To be exact about provenance: the prior artifact captured only
   `mode=none` (the agent's transcript recorded `avg_bytes_per_fault= 131072`, which I reproduced
   byte-identically). `MADV_RANDOM`, `MADV_SEQUENTIAL`, `MADV_HUGEPAGE` and `MADV_COLLAPSE` are
   **this review's runs** of the same binary, and `MADV_POPULATE_READ` (both variants),
   `POSIX_FADV_RANDOM` and `POSIX_FADV_SEQUENTIAL` rows come from `faultexp2.c`, which **I wrote**
   (`scripts/io/faultexp2.c`) because `faultexp.c` does not implement them despite its usage string
   listing mode 4 as `MADV_SEQUENTIAL` (it does implement that one). **No prior-agent output was
   available for any variant, and I did not invent one.**
2. **`align.c` was written to answer whether non-page-aligned slab strides cost O_DIRECT bandwidth,
   and its output was not captured.** The program takes `(path, bs, fileoff_align, buf_extra, MiB)`
   and deliberately misaligns both the file offset step and the buffer. **Its answer is not in any
   artifact in the scratch dir and I did not run it as part of this dossier** — it is a single
   command (`./align <gguf> 7340032 <step> <extra> 256`) and the result decides whether the project's
   7 MiB non-page-aligned expert strides need repacking for alignment reasons. **Unverified.**
3. **No sysfs knob was actually changed.** `sudo` requires a password on this host
   (`sudo -n true` → "sudo: a password is required", and this session cannot prompt
   `[verified-here]`), so `scheduler=none`, `nomerges=2`, `rq_affinity=2`, `read_ahead_kb` and
   `io_poll=1` are **recommendations with mechanisms attached, not measurements.** The predicted
   direction of `read_ahead_kb` is supported by a measured equality (128 KiB = 131 072 B), but the
   *magnitude and sign of the change on wall-clock reads* is unverified.
4. **IOPOLL is unmeasured, not disproven.** `EOPNOTSUPP` on this kernel/driver state
   `[verified-here]`. Whether `echo 1 > io_poll` changes that is unknown.
5. **`sqpoll` and `defer` were never run at 589 824 B** in the sweep (only 4096 and 4194304). §2.4
   therefore does not have a 576 KiB row, and **I have not extrapolated one.** Given that all modes
   converge at 4 MiB, the expectation is convergence at 576 KiB — but that is an expectation.
6. **Whether the drive itself does host-invisible read-ahead** (driver-side prefetch) is
   **unverified**. No primary source found for this specific device, and the measured 131 072 B is
   fully explained kernel-side.
7. **Whether the 589 824 B request is being split by the block layer.** `max_hw_sectors_kb = 128`
   and `max_sectors_kb = 128` on this host are *below* the 576 KiB slab, and PVLDB documents worker
   fallback for oversized requests. That the slab exceeds the limit is a fact of two sysfs values;
   **that it costs measurable throughput is inferred from the 3.38 GiB/s (576 KiB) vs 3.41 GiB/s
   (7 MiB) difference, which is within noise. Unverified.**
8. **`io_poll` on consumer NVMe / this Kingston drive specifically**: unverified, and possibly
   unsupported at the driver level.
9. **EROFS's throughput and read-amplification figures are unverified** in any primary source I could
   reach; the widely-cited 165 MB→16 MB number is carried in this repo from a blog. I cite the EROFS
   *design* from kernel mailing-list patches and mark the numbers `[unverified]`.
10. **`ds_loader_available()` was not executed** on this host. The `siphon.cpp` GDS loader builds and
    `dlopen`s cuFile, and `libcufile.so.1.13.1` is present, but `nvidia-fs` is not loaded, so my
    expectation is that it returns 0 and `--dstorage-moe` is inert. **Expected, not observed.**
11. **The two PCIe link readings disagree with each other in a way I did not chase**: the
    `cuda_probe` sysfs readout after warmup says **8.0 GT/s ×8** while a separate read at a different
    moment in the same session said **2.5 GT/s ×8** (before warmup). Both are 8-wide; max is
    32.0 GT/s. The measured 6.5 GB/s transfer rate is consistent with the 8.0 GT/s figure, so I
    believe the link trains up. **Whether it stays up under the project's real workload is
    unverified**, and if the engine ever sees ~1.9 GB/s host↔device it should check
    `current_link_speed` first.
12. **No independent measurement of the ~2.4 GB/s "raw NVMe" project figure was reproduced.** My
    buffered Python `pread` probe gave 971.2 MiB/s `[verified-here]`, which is a *buffered* figure and
    a Python-loop figure and is not comparable; the io_uring sweep's 3.4 GiB/s is the comparable
    number. **The project's 2.4 GB/s should be re-derived** — §2.2's evidence suggests it understates
    the device by ~1.4×.
13. **`nr_requests` on the partition was empty** (`nvme0n1p5/queue/nr_requests` returned nothing)
    `[verified-here]`; I read the values from `nvme0n1`. Whether the partition has its own effective
    limit is unverified.

---

## 8. Reproduction

Artifacts, all preserved. Scratch originals in `/home/scribe/.hermes/cache/scratch/`; the programs
are copied into the repo at `scripts/io/` so the measurements can be re-run after the scratch dir is
pruned.

| artifact | what it is | how to re-run |
|---|---|---|
| `io_results.csv` | the 92-row sweep matrix | `bash /home/scribe/.hermes/cache/scratch/sweep2.sh` (re-runs the whole matrix; takes ~1 min) |
| `iouraw.c` | raw-syscall io_uring + `pread` control; the source of every CSV row | `gcc -O2 -o iouraw iouraw.c && ./iouraw <gguf> <bs> <qd> <mode> <MiB> [regbuf]`; modes `pread uring iopoll sqpoll defer` |
| `io_bench.c` | the same sweep against **liburing**; kept as the cross-check (its output is not in the CSV) | `gcc -O2 -o io_bench io_bench.c -luring` |
| `align.c` | buffer/offset alignment sensitivity for `O_DIRECT` — **output never captured, see §7(2)** | `./align <gguf> 7340032 <fileoff_step> <buf_extra> 256` |
| `faultexp.c` | per-page-fault fetch size; modes none / RANDOM / HUGEPAGE / COLLAPSE / SEQUENTIAL | `for m in 0 1 4 2 3; do ./faultexp <gguf> $m 32 589824; done` |
| `faultexp2.c` | **added this review** — adds `MADV_POPULATE_READ` (page and slice) and `POSIX_FADV_RANDOM`/`SEQUENTIAL`; mode 0 replicates `faultexp` mode 0 | `gcc -O2 -o faultexp2 faultexp2.c && for m in 0 8 5 6 7; do ./faultexp2 <gguf> $m 32 589824; done` |
| `cuda_probe.cu` | pinned/pageable/UVM/VMM/PCIe probe | `nvcc -o cuda_probe cuda_probe.cu` **— use `/usr/local/cuda-12.8/bin/nvcc`, not `/usr/bin/nvcc`, or `sm_120` cannot be targeted**; output preserved at `cuda_probe_out.txt` |
| `kt.md`, `iouring_paper.txt`, `memory_c.html`, `backingdev_c.html`, `blksettings_c.html` | downloaded primary sources (arXiv:2512.04859 text; kernel `mm/memory.c`, `block/blk-settings.c`, `drivers/base/node.c`/`backing-dev.c` elixir pages) | — |

**One environment hazard for whoever re-runs the CUDA probe:** `/usr/bin/nvcc` is CUDA **11.5** and
`/usr/local/cuda-12.8/bin/nvcc` is CUDA **12.8**; `/usr/local/cuda` symlinks to 12.8 but `PATH` puts
11.5 first. A build that picks up `/usr/bin/nvcc` cannot compile for `cc=12.0` and will fail or
silently produce PTX-only artifacts. The `cuda_probe` binary in the scratch dir was built
successfully, so the previous agent used the right one; **do not assume a plain `nvcc` will.**
