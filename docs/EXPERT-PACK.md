# The expert pack — repacking the routed experts into one aligned slab each

Status: **file rewrite built and verified lossless; measurement complete. The
llama.cpp side that reads the pack is NOT built** (see "What is not done").

## The defect this removes

In this checkpoint the three tensors that make up one routed expert live in
three separate file regions. blk.0:

```
974,839,808   ffn_down_exps   [512, 2048, 256]  Q4_K
1,126,555,648 ffn_gate_exps   [2048, 512, 256]  Q4_K
1,280,376,832 ffn_up_exps     [2048, 512, 256]  Q4_K
```

~305 MB apart, three regions. Reading one routed expert is three scattered
`pread`s. Every engine in `docs/ASSEMBLED-DESIGN.md` §1 fixes this the same way:
`apus` packs gate_up+down into one 6,291,456 B slab, `flash-moe` uses a fixed
7,077,888 B stride, `qwen-fieldfare`'s Q4EXP02 has a uniform 1,769,472 B stride
at 16 KiB alignment, `slipstream-schero94`'s PGRN v1 sets `PGRN_ALIGN=16384`,
`siphon.cpp` uses a 16 KiB-aligned sidecar manifest.

## A second defect, found while checking the first

**Every one of the 120 routed-expert tensors sits at an offset that is 480 mod
4096.** The GGUF tensor data starts at 10,990,048 (480 mod 4096) and the header
alignment is only 32 bytes, so no expert tensor is block-aligned:

```
data_start = 10,990,048   (mod 4096 = 480)
expert tensors whose absolute start is NOT 4096-aligned: 120/120
  blk.0.*  abs mod 4096 = 480
  blk.1.*  abs mod 4096 = 1248
```

Consequence: **the experts of this GGUF cannot be read with `O_DIRECT` at all.**
Not tuned badly — cannot be issued. Measured directly
(`results/expertpack-20260922-213111/odirect-check.txt`):

```
source   O_DIRECT: 0/3 ok, 3 failed  (errno 22 Invalid argument;
                                     first at offset 1147376096 = 480 mod 4096)
pack     O_DIRECT: 3/3 ok, 0 failed
```

and at full scale, every request in a routed set:

```
source GGUF, io_uring + O_DIRECT, qd=8
  3840/3840 requests have a non-4096-aligned offset or length
  requests 3840   ok 0   errors 3840   wall 0.002 s
  first error: errno 22 (Invalid argument)
```

This matters because `O_DIRECT` is the mechanism every surveyed engine relies on
to keep the page cache out of the way, and `docs/LOSSLESS-OPTIONS.md` lever 1
(the 3,384 vs 1,658 MiB/s result) assumed it was available. On the unmodified
file it is not.

## The container

`scripts/io/repack_experts.py`. Bytes copied verbatim: no dequantisation, no
requantisation, no reordering within a row. The tool never parses a weight, so
losslessness is structural — and it is also *checked*, see below.

```
[ 16 KiB header ] [ blk.0 slabs ] [ blk.1 slabs ] ... [ blk.39 slabs ]
slab = gate_row(e) || up_row(e) || down_row(e) [ || zero pad to 16 KiB ]

slab_offset(layer, expert) = header + layer_base[layer] + expert * stride[layer]
```

**This checkpoint is not stride-uniform, which the prior art did not anticipate.**
117 of 120 routed-expert tensors are Q4_K with a 589,824 B per-expert row, but
`blk.34/38/39.ffn_down_exps` are **Q6_K with an 860,160 B row**:

```
stride 1,769,472 B (108 x 16384)  gate/up/down = 589,824 Q4_K   x 37 layers
stride 2,048,000 B (125 x 16384)  down = 860,160 Q6_K           x 3 layers (34, 38, 39)
```

So the header carries an explicit per-layer table (stride + three component
offsets) plus each layer's base offset, rather than one global stride. A first
version of the tool asserted a uniform stride and **failed loudly on blk.34** —
that assertion is what surfaced the Q6_K outlier; it is kept as a per-layer check
rather than relaxed.

The 1,769,472 B stride for 37 of 40 layers is exactly qwen-fieldfare's Q4EXP02
number, independently derived from this file's dimensions rather than copied.

```
container   18,333,319,168 B (18.333 GB) vs source 20.893 GB
padding     6.3 MB (0.034%) — declared in the manifest, carries no content
build       27.4 s (0.67 GB/s)
```

## Losslessness

```
$ python3 scripts/io/repack_experts.py --verify --verify-samples 300
  VERIFY OK: 300 random slabs, 900 components bit-identical to source;
             header geometry round-trips
```

300 random (layer, expert) pairs; all three components of each compared byte for
byte against the source slices by `mmap`-free `pread`. Any difference is reported
with its byte offset. The header is parsed back and its strides and base offsets
are cross-checked against the geometry the build used, so the format has one
reader and it is exercised on every verify.

## The measurement

`scripts/io/bench_expert_pack.{c,sh}`. A routed set = 8 experts × 40 layers,
4 passes = 1,280 experts × 1,769,472 B ≈ 2,185 MiB of payload. Page cache
evicted before **every** arm (verified `cached before: 0 bytes`) and each arm runs
in its own process; device reads are `/proc/self/io read_bytes`, not wall clock —
the discipline in `scripts/README.md`.

| arm | requests | device read | wall | MiB/s |
|---|---|---|---|---|
| source GGUF, buffered pread | 3,840 | 2,655.2 MiB (1.22×) | 3.533 s | **618.5** |
| pack, buffered pread | **1,280** | 2,240.8 MiB (1.03×) | 1.579 s | **1,384.2** |
| pack, io_uring + O_DIRECT qd8 | **1,280** | 2,185.5 MiB (**1.00×**) | **0.648 s** | **3,373.3** |
| source GGUF, io_uring + O_DIRECT | 3,840 | — | — | **3840/3840 EINVAL** |

```
requests      3,840 -> 1,280    3.0x fewer
device read   2,655.2 -> 2,185.5 MiB   -17.7%
amplification 1.22x -> 1.00x
throughput    618.5 -> 3,373.3 MiB/s   5.45x
wall          3.533 -> 0.648 s         5.45x
```

The whole table reproduces across two independent runs to within 1%
(`expertpack-20260922-213007`, rebuilt as `-213111`: 3,404.4 vs 3,373.3 MiB/s
for the same arm, identical request counts and byte totals).

The 5.45× is **two effects stacked**: the pack removes the scatter (3 requests →
1, +2.2× on its own, buffered row), and alignment makes O_DIRECT possible at all
(+2.4× on top). Neither alone gets there.

### Two things this number is not

**The 1.22× device-read amplification on the source is not waste, and the pack's
−17.7% is not a pure gain.** `docs/ASSEMBLED-DESIGN.md` and
`docs/LOSSLESS-OPTIONS.md` lever 7 record that this amplification is
**load-bearing read-ahead**: `MADV_RANDOM` cut reads 42.0 → 26.0 GB but cost
56.0 → 368.0 s. Removing it without replacing the prefetch makes things worse.
The defensible wins here are the **request count** (3× fewer is unambiguously
less work) and the **5.4× achievable read-rate**, not the byte reduction.

**This is the I/O path only.** The compute column from the BigMoeOnEdge sweep is
0.088–0.094 s/token and is unchanged by any file layout, so the decode ceiling
stays **1 ÷ 0.088 ≈ 11.4 tok/s**. And routed experts are only 22.1% of per-token
traffic — always-active tensors are 77.9%. Even a perfect expert path is bounded
by the part it does not touch. This lever cannot reach 20 tok/s, and nothing in
this document claims it does.

## What is not done

**llama.cpp cannot read the pack.** This is a file rewrite plus a measurement;
the loader and graph side is the remaining work, and it is real work:

- GGUF's addressable unit is the *tensor*, not a slice inside one, so a packed
  layout is not expressible as a standard GGUF tensor reorder.
- Route A: split each expert tensor into one tensor per expert (3 × 256 × 40 =
  **30,720 tensor entries**) — legal GGUF, but the graph must select among 30k
  tensors: a model-architecture change, not a file rewrite.
- Route B: packed layout + this header, read by a custom loader. **BigMoeOnEdge
  already does its own `O_DIRECT` reads**, so this is the natural host and needs
  no graph change.
- A third option, simpler and untested: use the pack for *staging* — prefetch
  routed sets from it into a slot pool while the graph keeps reading ordinary
  GGUF tensors. The pack then fixes the read side without touching the graph.

Until one of those is built, the honest statement is: **the file is fixed, the
reader is not written, and no end-to-end tok/s number has moved.**

## Reproduce

```bash
python3 scripts/io/repack_experts.py --dry-run          # geometry + free space
python3 scripts/io/repack_experts.py --verify           # build, then verify
bash scripts/io/bench_expert_pack.sh                    # 4 arms, evicted, own process
```

Artifacts: `~/models/jev-pack/qwen3.6-35b-a3b-expertpack.bin` (+`.manifest.json`,
+`.source_regions.tsv`). The 18.3 GB pack is not in the repo.
