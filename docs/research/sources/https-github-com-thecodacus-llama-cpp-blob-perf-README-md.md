# SOURCE: https://github.com/thecodacus/llama.cpp/blob/perf/README.md

## File metadata and controls

278 lines (212 loc) · 16 KB

# llama.cpp

[![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

## ⚡ This fork — Fable's MoE-offload prefill optimizations

Two **opt-in** optimizations for large MoE models whose experts are offloaded to system RAM (`--n-cpu-moe`), found and implemented by Fable. Both are **off by default**, toggled via environment variables, and produce **token-identical** output to mainline.

| Env var | What it does |
|---|---|
| `GGML_CUDA_REGISTER_HOST=1` | Page-locks (pins) the mmap'd CPU expert weights so host→device copies go straight over DMA instead of through the driver's hidden bounce buffer (~6–7 → ~20 GB/s). |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | Prefetches each layer's experts on a second CUDA stream, so the weight uploads overlap compute instead of stalling the GPU. |

### Benchmark

Measured on an **RTX 3060 12GB** with **Qwen3.6-35B-A3B** (`--n-cpu-moe 26`), prompt-processing at 2048 (`MODEL` = path to your `.gguf`):

```
# baseline (patches off):
./build/bin/llama-bench -m MODEL -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048

# patched (both optimizations on):
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
./build/bin/llama-bench -m MODEL -ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048
```

Result: **~1143 → ~1880 t/s** prefill (**+64%**) — same GPU, same settings, token-identical.

Branches: [`fable5/host-register`](https://github.com/thecodacus/llama.cpp/tree/fable5/host-register) (pinning only) · [`fable5/prefetch-experts`](https://github.com/thecodacus/llama.cpp/tree/fable5/prefetch-experts) (both — this branch).

## ⚡ This fork — MoE expert cache (VRAM-resident hot experts)

For MoE models whose routed experts live in system RAM (`--n-cpu-moe`), this fork can keep the most-frequently-routed experts of each layer **resident in VRAM**. Decode runs the hot experts on GPU and only the cold remainder on CPU; the two halves are merged exactly, so output is **bit-identical** to baseline. Opt-in, off by default.

Measured on an RTX 3060 12GB (`-ngl 99 -ncmoe 99 -fa 1`):

| Model | Baseline tg | Cached tg | Prefill |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M (256 experts/layer) | 42.3 | **51.3 (+21%)** @ 124 slots | +14% |
| — same, stacked with `--spec-type draft-mtp` | 41.7 | **69.3 (+66%)** @ 112 slots | — |
| — same, plus async CPU splits (default on) | 41.7 | **74.2 (+78%)** @ 88 slots | — |
| GLM-4.7-Flash Q4_K_M (64 experts/layer) | 32.1 | **46.3 (+44%)** @ 40 slots | +64% |
| Laguna-S-2.1-118B-A8B IQ4_XS (256 experts/layer) | 11.5 | **12.1 (+5%)** @ 36 slots | +12% |
| Qwen3.8-Flash-Next 177B UD-IQ3_XXS (512 experts/layer, 10 active) | 16.6 | **24.4 (+47%)** @ 56 slots, with `--spec-type draft-mtp` | — |
Supported architectures: `qwen35moe`, `qwen4exp` (Qwen3.8-Flash-Next), `deepseek2`, `laguna` (plain fused-SILU gated expert FFN, separate gate/up/down tensors). Other architectures run unchanged.

### Quick start

**1. Capture a routing profile** (one time per model — records which experts the router picks):

```
MOE_TRACE_OUT=mymodel-code.csv ./build/bin/llama-moe-trace -m model.gguf \
  -ngl 99 -ncmoe 99 -fa 1 -c 4096 -n 512 -p "<a code-flavored prompt>"

MOE_TRACE_OUT=mymodel-chat.csv ./build/bin/llama-moe-trace -m model.gguf \
  -ngl 99 -ncmoe 99 -fa 1 -c 4096 -n 512 -p "<a chat-flavored prompt>"

cat mymodel-code.csv mymodel-chat.csv > mymodel-merged.csv
```

512 generated tokens per prompt is enough. Merge traces from contrasting workloads — a merged profile measures within 1% of per-workload specialist profiles, so one merged CSV per model is all you need.

**2. Serve with the cache:**

```
./build/bin/llama-server -m model.gguf -ngl 99 -ncmoe 99 -fa 1 \
  --moe-cache-profile mymodel-merged.csv --moe-cache-slots 112
```

Also works per model in a `--models-preset` INI section (`moe-cache-profile = ...`, `moe-cache-slots = ...`), and as env vars `LLAMA_ARG_MOE_CACHE_PROFILE` / `LLAMA_ARG_MOE_CACHE_SLOTS` (or legacy `GGML_MOE_CACHE_PROFILE` / `GGML_MOE_CACHE_SLOTS`, which `llama-bench` also accepts).

**3. Confirm it engaged** — look for this line at load:

```
init_moe_expert_cache: expert cache: 40 layers x 112 slots, 8164.00 MiB uploaded to CUDA0
```

A warning instead of this line means the cache fell back to baseline (see Tuning).

### Tuning

- **`--moe-cache-slots` is the main knob** — experts cached per layer. Throughput rises with slot count until the pack no longer fits in VRAM. The pack is all-or-nothing: an oversized request logs `pack allocation failed - expert cache disabled` and runs at baseline speed (it does not partially fill). The warning reports the per-slot cost and the maximum count that could fit — set slots to that, minus headroom for KV/compute buffers which allocate afterwards.
- **The cold and hot chains overlap by default.** CPU graph splits run on a worker thread so the GPU hot chain executes concurrently with the CPU cold chain (`--no-sched-async-cpu` to disable; `llama-bench --sched-async-cpu 0,1` benches both). Worth +4-5% with speculative decoding, ~±2% without it; outputs stay bit-identical either way.
- **Leave ~900 MB of VRAM free beyond the pack.** A slot count that loads can still crash on the first large prompt: runtime CUDA pool growth allocates beyond what the load-time check sees. Size slots against the biggest prompt you will serve, not against "it loaded".
- **Fill VRAM to just under the ceiling, don't sweat the split.** Near the maximum, a marginal MB is worth about the same as cache slots or as fully-resident layers (lower `--n-cpu-moe`). Pure `-ncmoe 99` + max slots is the simple default; a hybrid (e.g. `-ncmoe 30` + fewer slots) buys ~1% decode and ~3% prefill at best.
- **Context size competes with the pack.** KV grows with `-c` and shrinks the viable slot count. Compressing the KV cache (`-ctk`/`-ctv`, e.g. TurboQuant types) frees VRAM that converts directly into slots — often worth more than the KV precision costs.
- **Speculative decoding stacks multiplicatively.** `--spec-type draft-mtp` composes with the cache (+48% cache × +12% MTP ≈ +66% on Qwen); reserve ~1 GB for the draft context by dropping a few slots.
- **Expert size decides the payoff.** Big experts (GLM: ~5 MiB each) gain the most per slot; small experts need high slot counts before the win beats the dual-path overhead (~25% traffic coverage is roughly break-even). If VRAM only fits <15% of the expert count, expect single-digit gains (Laguna above).
- **Profiles are model-specific, workload-tolerant.** A wrong-workload profile still helps (+28% measured on Qwen worst-case) but loses about half the win; the merged profile recovers nearly all of it. Regenerate only if your usage changes character entirely.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `cannot open profile '...'` | Path not visible to the process (e.g. not mounted into the container). |
| `pack allocation failed` | Slot count too big — read the fit math in the warning and reduce. |
| `no CPU-resident MoE layers` | Experts are already on GPU (no `--n-cpu-moe`) — nothing to cache. |
| No init line, no warning | Architecture not wired for the cache — model runs unchanged. |
| Model loads, then context creation OOMs | Pack fits but KV/compute don't — drop a few slots or shrink/compress KV. |

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](/thecodacus/llama.cpp/blob/perf/docs/multimodal.md)
- VS Code extension for FIM completions: [https://github.com/ggml-org/llama.vscode](https://github.com/ggml-org/llama.vscode)
- Vim/Neovim plugin for FIM completions: [https://github.com/ggml-org/llama.vim](https://github.com/ggml-org/llama.vim)
- Hugging Face Inference Endpoints now support GGUF out of the box! [ggml-org#9669](https://github.com/ggml-org/llama.cpp/discussions/9669)
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- WebGPU support is now available in the browser, see a blog/demo introducing it [here](https://reeselevine.github.io/llamas-on-the-web/).

---

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit [https://llama.app](https://llama.app) and follow the instructions
- Run with Docker - see our [Docker documentation](/thecodacus/llama.cpp/blob/perf/docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](/thecodacus/llama.cpp/blob/perf/docs/build.md)
Once installed:

```
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

| [![VLM session with `llama cli`](https://private-user-images.githubusercontent.com/1991296/629069102-88726b48-1713-48aa-a525-95a02e78afc4.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODkwOTY2MTksIm5iZiI6MTc4OTA5NjMxOSwicGF0aCI6Ii8xOTkxMjk2LzYyOTA2OTEwMi04ODcyNmI0OC0xNzEzLTQ4YWEtYTUyNS05NWEwMmU3OGFmYzQucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDkxMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA5MTFUMDMxMTU5WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YWE2MTlmOTEyM2FjYjEyNTlmZmJmOGE1MDM4MDJlYjg4NjUzY2ViOGZmNjJlNTc3OGE2ODNmN2QwYjk5MDE3MCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.i3v1gL60L_W985IbtBgwYMhm6tHpZoC_bp80nqMYeb8)](https://private-user-images.githubusercontent.com/1991296/629069102-88726b48-1713-48aa-a525-95a02e78afc4.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODkwOTY2MTksIm5iZiI6MTc4OTA5NjMxOSwicGF0aCI6Ii8xOTkxMjk2LzYyOTA2OTEwMi04ODcyNmI0OC0xNzEzLTQ4YWEtYTUyNS05NWEwMmU3OGFmYzQucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDkxMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA5MTFUMDMxMTU5WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YWE2MTlmOTEyM2FjYjEyNTlmZmJmOGE1MDM4MDJlYjg4NjUzY2ViOGZmNjJlNTc3OGE2ODNmN2QwYjk5MDE3MCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.i3v1gL60L_W985IbtBgwYMhm6tHpZoC_bp80nqMYeb8) *VLM session with **llama cli*** | [![Built-in web UI against `llama serve` running Qwen 3.6](https://private-user-images.githubusercontent.com/1991296/628539505-b402f972-2e32-4def-8771-8d849f08cf2e.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODkwOTY2MTksIm5iZiI6MTc4OTA5NjMxOSwicGF0aCI6Ii8xOTkxMjk2LzYyODUzOTUwNS1iNDAyZjk3Mi0yZTMyLTRkZWYtODc3MS04ZDg0OWYwOGNmMmUucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDkxMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA5MTFUMDMxMTU5WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YTI1MjUyMjhlOWY1ZjgwNGQxZmExMDk3Njc2NzViMjIzOWYxZWEwYzVmNDFkZTY2MDhhZmFmODUwZTY4NmNkNiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.5jq9mKe6O1gZNi4pxRi3n2_PTANK8Jp97JCr4xptzdY)](https://private-user-images.githubusercontent.com/1991296/628539505-b402f972-2e32-4def-8771-8d849f08cf2e.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODkwOTY2MTksIm5iZiI6MTc4OTA5NjMxOSwicGF0aCI6Ii8xOTkxMjk2LzYyODUzOTUwNS1iNDAyZjk3Mi0yZTMyLTRkZWYtODc3MS04ZDg0OWYwOGNmMmUucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDkxMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA5MTFUMDMxMTU5WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YTI1MjUyMjhlOWY1ZjgwNGQxZmExMDk3Njc2NzViMjIzOWYxZWEwYzVmNDFkZTY2MDhhZmFmODUwZTY4NmNkNiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.5jq9mKe6O1gZNi4pxRi3n2_PTANK8Jp97JCr4xptzdY) *Built-in web UI against **llama serve*** |
|---|---|

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity
The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
|---|---|
| [BLAS](/thecodacus/llama.cpp/blob/perf/docs/build.md#blas-build) | All |
| [BLIS](/thecodacus/llama.cpp/blob/perf/docs/backend/BLIS.md) | All |
| [CANN](/thecodacus/llama.cpp/blob/perf/docs/build.md#cann) | Ascend NPU |
| [CUDA](/thecodacus/llama.cpp/blob/perf/docs/build.md#cuda) | Nvidia GPU |
| [HIP](/thecodacus/llama.cpp/blob/perf/docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](/thecodacus/llama.cpp/blob/perf/docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](/thecodacus/llama.cpp/blob/perf/docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](/thecodacus/llama.cpp/blob/perf/docs/build.md#musa) | Moore Threads GPU |
| [Metal](/thecodacus/llama.cpp/blob/perf/docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](/thecodacus/llama.cpp/blob/perf/docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](/thecodacus/llama.cpp/blob/perf/docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](/thecodacus/llama.cpp/blob/perf/docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](/thecodacus/llama.cpp/blob/perf/docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](/thecodacus/llama.cpp/blob/perf/docs/build.md#vulkan) | GPU |
| [WebGPU](/thecodacus/llama.cpp/blob/perf/docs/build.md#webgpu) | All |
| [ZenDNN](/thecodacus/llama.cpp/blob/perf/docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](/thecodacus/llama.cpp/blob/perf/tools/cli/README.md)
- [completion](/thecodacus/llama.cpp/blob/perf/tools/completion/README.md)
- [server](/thecodacus/llama.cpp/blob/perf/tools/server/README.md)
- [GBNF grammars](/thecodacus/llama.cpp/blob/perf/grammars/README.md)

#### Development

- [How to build](/thecodacus/llama.cpp/blob/perf/docs/build.md)
- [Running on Docker](/thecodacus/llama.cpp/blob/perf/docs/docker.md)
- [Build on Android](/thecodacus/llama.cpp/blob/perf/docs/android.md)
- [Multi-GPU usage](/thecodacus/llama.cpp/blob/perf/docs/multi-gpu.md)
- [Performance troubleshooting](/thecodacus/llama.cpp/blob/perf/docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](/thecodacus/llama.cpp/blob/perf/docs/xcframework.md)
- [Completions](/thecodacus/llama.cpp/blob/perf/docs/completions.md)
- [Models](/thecodacus/llama.cpp/blob/perf/docs/models.md)
- [Release process](/thecodacus/llama.cpp/blob/perf/docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](/thecodacus/llama.cpp/blob/perf/CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
