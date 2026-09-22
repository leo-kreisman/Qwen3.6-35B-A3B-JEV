# Qwen3.6-35B-A3B → **SSD** offload: certified list

Scope: Qwen3.6-35B-A3B. **No architecture filter.** 76+ repos gathered across ~26
architecture-agnostic queries, then each README read to certify two things:

1. Does it actually run **Qwen3.6-35B-A3B**?
2. Does it actually **stream expert weights from SSD/NVMe** — not RAM/VRAM offload?

Last verified 2026-09-21.

> **Correction (2nd revision).** An earlier version of this file claimed Edge0 had no
> GitHub repo. It does — [Edge0-AI/Edge0](https://github.com/Edge0-AI/Edge0), 2,036★ —
> and it is now the largest certified entry here.
>
> **Methodology gap that caused it:** my query set was model-first ("35B-A3B" + SSD terms).
> That structurally misses general-purpose engines that *support* Qwen3.6-35B-A3B but
> don't lead with the model name in their description. `FreeToken` (13,460★) was also
> missed the same way.

---

## CERTIFIED — 11 repos

| Repo | ★ | Arch / Platform | Certifying evidence (verbatim) |
|---|---|---|---|
| [Edge0-AI/Edge0](https://github.com/Edge0-AI/Edge0) | **2036** | Apple Silicon (MLX); CUDA pluggable | "An open-source streaming MoE inference framework — **SSD expert offload** + Recover-LoRA + prerouter routing prediction." `edge0-35b` = `Edge0/Edge0-35B-A3B-preview`, "built on open sparse-MoE base models (**Qwen3.6-35B-A3B**)". Apache-2.0. |
| [NeelM0906/Mference](https://github.com/NeelM0906/Mference) | 118 | Apple Silicon | "…the current token from SSD. The complete model does not have to fit in RAM." |
| [deepanwadhwa/samosa-chat](https://github.com/deepanwadhwa/samosa-chat) | 55 | Apple Silicon (Metal) | "2-bit MoE with routed experts streamed from SSD"; "Qwen performance is dependent on SSD throughput." |
| [Ninnix/q36](https://github.com/Ninnix/q36) | 53 | **Vulkan + Metal; AMD BC-250** | "8 GB Macs are intended to use SSD streaming." Primary target is an AMD BC-250. |
| [penta2himajin/qwisp](https://github.com/penta2himajin/qwisp) | 12 | Apple Silicon | "RAM sets the tier automatically: `<32 GB` streams experts from flash." |
| [Manohar20006/siphon.cpp](https://github.com/Manohar20006/siphon.cpp) | 9 | **CUDA (Windows/Linux)** | "Run Qwen3.6 35B-A3B MoE locally on a 6 GB VRAM laptop… with **SSD-based MoE expert streaming**." |
| [dwijenpatel/slipstream](https://github.com/dwijenpatel/slipstream) | 5 | Apple Silicon (Metal) | "reads, for each token, only the eight experts that token routes to" from SSD; 31.1 tok/s in 5.6 GB. |
| [Schero94/slipstream](https://github.com/Schero94/slipstream) | 3 | Apple Silicon (llama.cpp fork) | "Qwen3.6-35B-A3B (Q4), streamed from the internal NVMe." |
| [tanishqpatil/qwen-fieldfare](https://github.com/tanishqpatil/qwen-fieldfare) | 2 | Apple Silicon (Metal) | "Streams active MoE experts from SSD." |
| [bricesommers/apus-qwen3.6-35B-A3B](https://github.com/bricesommers/apus-qwen3.6-35B-A3B) | 0 | **C engine (portable)** | "routed experts from NVMe through a bounded RAM cache"; "~1.1 GiB/token of reads". |
| [anwarth/Qwen-MoE-Router-exp](https://github.com/anwarth/Qwen-MoE-Router-exp) | 0 | llama.cpp (GPU-less PC) | Disk-backed via llama.cpp mmap. **Caveat:** README warns the streaming "is effectively inactive" if the OS doesn't page. |

### Architecture
Apple Silicon 7 · CUDA 1 · portable C 1 · Vulkan/AMD 1 · llama.cpp generic 1.

---

## REJECTED — looked like hits, are not

| Repo | ★ | Why rejected |
|---|---|---|
| [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) | **13460** | Supports Qwen3.6-35B-A3B, but **not SSD**. README + all four docs (`models/quickstart/cli/install.md`) contain **zero** disk/SSD/NVMe terms. Mechanism is VRAM expert caches ("dynamic runtime VRAM re-allocation between expert caches and KV memory") + CPU–GPU co-execution. Also `freetoken-kernel-cache/`. |
| [AEON-7/…heretic-NVFP4-DFlash](https://github.com/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash) | 144 | Regex matched "**Re-validated**". vLLM/DGX Spark — weights in VRAM. |
| [Sandermage/sndr_core_engine](https://github.com/Sandermage/sndr_core_engine) | 132 | Matched "**spreading activation**". Knowledge-graph app, not an inference engine. |
| [lastloop-ai/vllm-blackwell-guide](https://github.com/lastloop-ai/vllm-blackwell-guide) | 23 | Matched "**batch processing**". vLLM, VRAM-resident. |
| [hanxiao/Qwen3.6-35B-A3B-MTP-L4](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) | 18 | "SSD" was a **GCS bucket** staging step. |
| [hanxiao/Qwen3.6-35B-A3B-L4-deploy](https://github.com/hanxiao/Qwen3.6-35B-A3B-L4-deploy) | 17 | `--boot-disk-type=pd-ssd` is a **GCE boot-disk flag**. |

## Satellite repos — Edge0 ecosystem, not engines
- [dogebi/edge0-35b-atlas](https://github.com/dogebi/edge0-35b-atlas) — WebGL tensor atlas for Edge0-35B-A3B. Visualization.
- [hi-sch/basechat](https://github.com/hi-sch/basechat) — SwiftUI frontend that drives the `edge0` CLI.
- [Mesutcydev/ondevice-core](https://github.com/Mesutcydev/ondevice-core) — iPhone app bundling the Edge0 runtime.
- [bayerhazard/aimighty-freetoken](https://github.com/bayerhazard/aimighty-freetoken) — Olares packaging of FreeToken (⇒ not SSD, inherits the rejection above).

## Also excluded
- **`danveloper/flash-moe` (4147★)** — most-cited repo here, but runs **Qwen3.5-397B-A17B**.
  Its `paper/` and `repack_experts.py` remain the best technical writeup on SSD expert
  streaming, but it is not a 35B runtime.

---

## ⚠️ What "certified" means here

**README-level certification only.** I read READMEs and API metadata (`raw.githubusercontent`
+ GitHub API). I did **not** clone, read source, build, or run any of these. So
"certified" = "the documentation asserts it," the same standard I warned you not to
trust for tok/s numbers. An SSD claim can be a design goal, a stub, or stale text.

To actually confirm, each repo needs: a real disk read path (`pread`/`O_DIRECT`/`io_uring`
on a file descriptor for expert tensors, not whole-file `mmap`), a populated expert offset
table, and evidence it builds. **`siphon.cpp` (9★) and `apus` (0★) are the two non-Mac
picks where this matters most** — and Edge0, despite 2k stars, is unverified at source level.
