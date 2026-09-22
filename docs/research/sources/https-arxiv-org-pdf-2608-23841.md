# SOURCE: https://arxiv.org/pdf/2608.23841

Pipeline-Native Transformers:

Co-Designing Model Architecture and CPU
Inference
for Bandwidth-Efficient Autoregressive Decode

An Independent Research Report

Tom Poperszky

2026

arXiv:2608.23841v1  [cs.AR]  24 Aug 2026


Abstract

Single-token autoregressive decode on CPUs is bound by memory bandwidth, not
arithmetic: a modern CPU sustains roughly 1 TFLOP/s of compute but only about
50 GB/s from main memory, and each generated token must stream every active
weight once. This report argues that the most effective response is to co-design the
model architecture and the inference runtime together. It presents cflow, a CPU-
first streaming engine, alongside a family of pipeline-native transformer architectures
whose inter-layer dependency graphs are constructed to permit a vertical, stage-major
execution schedule.
cflow stores weights as L2-sized tiles in compute-consumption order, reads only
the top-k experts of each mixture-of-experts layer, fuses projections, and executes a
delay-aware schedule from per-model dependency parameters. Across five architectures
trained on TinyStories, one (arch2_4_combined) achieves a 2.00× reduction in critical-
path weight bandwidth (9.00 →4.50 MB/token) within 0.24 perplexity of the best
candidate, and the tile layout incurs 7.29× fewer L1-data read misses than a row-major
baseline. On a 30.9-billion-parameter pipeline-native MoE, cflow decodes at 5.94
tokens/s (tok/s) on a 32-vCPU Ice Lake server, ahead of llama.cpp (4.75) and the
vLLM CPU backend (1.65) on comparably sized dense models. Realizing the expert-
delay window as asynchronous I/O overlap on a disk-resident expert tier yields a
further net win of up to 1.68×, matching the overlap model within 1%. Measurement
refutes one of the eight design claims and leaves a second inconclusive; both are
reported in full, with the conditions under which they would hold.

i


Note on the Use of AI Tools

The implementation and writing of this report were carried out with the assistance of
an AI coding assistant (Anthropic’s Claude), used in two specific capacities: helping
to implement the cflow runtime and its supporting code, and helping to draft and
edit this report. All research direction, architectural and experimental design, analysis
of results, and conclusions are my own, and I take full responsibility for the contents
of this report, including any errors.

ii


Contents

Abstract
i

Note on the Use of AI Tools
ii

1
Introduction
1
1.1
The Bandwidth Bottleneck in LLM Inference . . . . . . . . . . . . . .
1
1.2
The GPU-First Retrofit Problem
. . . . . . . . . . . . . . . . . . . .
2
1.3
The Co-Design Opportunity . . . . . . . . . . . . . . . . . . . . . . .
2
1.4
Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3
1.5
Summary of Key Results . . . . . . . . . . . . . . . . . . . . . . . . .
4
1.6
Scope and Limitations
. . . . . . . . . . . . . . . . . . . . . . . . . .
5
1.7
Report Organization . . . . . . . . . . . . . . . . . . . . . . . . . . .
6

2
Background
7
2.1
CPU Memory Hierarchy and the Roofline Model . . . . . . . . . . . .
7
2.1.1
The Memory Hierarchy . . . . . . . . . . . . . . . . . . . . . .
7
2.1.2
The Roofline Model . . . . . . . . . . . . . . . . . . . . . . . .
7
2.1.3
Arithmetic Intensity of Single-Token Transformer Decode . . .
8
2.2
The Standard Pre-Norm Transformer . . . . . . . . . . . . . . . . . .
9
2.2.1
Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . .
9
2.2.2
Multi-Head Attention . . . . . . . . . . . . . . . . . . . . . . .
9
2.2.3
Feed-Forward Network . . . . . . . . . . . . . . . . . . . . . .
10
2.2.4
Mixture-of-Experts Feed-Forward Layers . . . . . . . . . . . .
10
2.2.5
The Layer Dependency DAG . . . . . . . . . . . . . . . . . . .
10
2.3
Quantization
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11
2.3.1
Q4 Quantization
. . . . . . . . . . . . . . . . . . . . . . . . .
11
2.3.2
GGUF and llama.cpp . . . . . . . . . . . . . . . . . . . . . . .
11
2.4
Existing CPU Inference Runtimes . . . . . . . . . . . . . . . . . . . .
12
2.5
Mixture-of-Experts at Scale
. . . . . . . . . . . . . . . . . . . . . . .
12
2.6
Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
13
2.6.1
Weight Streaming and Offloading . . . . . . . . . . . . . . . .
13
2.6.2
Speculative Decoding . . . . . . . . . . . . . . . . . . . . . . .
13
2.6.3
Structured Sparsity and Expert Routing . . . . . . . . . . . .
13

iii


CONTENTS
iv

2.6.4
Cache-Aware Deep Learning . . . . . . . . . . . . . . . . . . .
14
2.6.5
Model-Runtime Co-Design . . . . . . . . . . . . . . . . . . . .
14

3
The cflow Runtime
15
3.1
Design Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
3.2
Tile-Native Weight Format . . . . . . . . . . . . . . . . . . . . . . . .
16
3.2.1
Tile Geometry . . . . . . . . . . . . . . . . . . . . . . . . . . .
16
3.2.2
TileHeader Format . . . . . . . . . . . . . . . . . . . . . . . .
16
3.2.3
MappedTile: Zero-Copy Slice into mmap . . . . . . . . . . . .
17
3.3
The .cflow File Format . . . . . . . . . . . . . . . . . . . . . . . . .
17
3.3.1
GlobalHeader . . . . . . . . . . . . . . . . . . . . . . . . . . .
17
3.3.2
Per-Layer Tile Ordering
. . . . . . . . . . . . . . . . . . . . .
18
3.3.3
Expert Offset Table and Conditional Loading
. . . . . . . . .
18
3.4
The .vflow File Format . . . . . . . . . . . . . . . . . . . . . . . . .
18
3.4.1
Vertical Groups . . . . . . . . . . . . . . . . . . . . . . . . . .
19
3.4.2
File Layout
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
19
3.5
Fused Projection Kernels . . . . . . . . . . . . . . . . . . . . . . . . .
19
3.5.1
Tiled Matrix-Vector Product . . . . . . . . . . . . . . . . . . .
19
3.5.2
Fused QKV and Gate+Up . . . . . . . . . . . . . . . . . . . .
20
3.6
Q4 Inner-Product Pipeline . . . . . . . . . . . . . . . . . . . . . . . .
20
3.6.1
Q4_0 Quantization Format
. . . . . . . . . . . . . . . . . . .
20
3.6.2
Scalar Reference Implementation
. . . . . . . . . . . . . . . .
21
3.6.3
AVX2 + FMA Path
. . . . . . . . . . . . . . . . . . . . . . .
21
3.6.4
AVX-512 Path . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
3.7
Attention Pipeline
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
3.7.1
Grouped-Query Attention with KV Cache . . . . . . . . . . .
22
3.7.2
Rotary Position Embedding . . . . . . . . . . . . . . . . . . .
22
3.7.3
V-Norm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
3.8
Normalization Layers . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
3.9
Conditional Expert Prefetch . . . . . . . . . . . . . . . . . . . . . . .
23
3.9.1
Two-Thread Pipeline . . . . . . . . . . . . . . . . . . . . . . .
23
3.9.2
Linear and Conditional Commands . . . . . . . . . . . . . . .
23
3.9.3
Negative Result: PREFETCHT0 at RAM-Bottleneck Scale . .
24
3.9.4
Staged Direct-I/O Expert Fetch . . . . . . . . . . . . . . . . .
24
3.10 Safetensors Converter . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
3.10.1 Architecture Dispatch
. . . . . . . . . . . . . . . . . . . . . .
24
3.10.2 DelayedMoESource . . . . . . . . . . . . . . . . . . . . . . . .
25
3.10.3 Tile Quantization . . . . . . . . . . . . . . . . . . . . . . . . .
25
3.11 Implementation Correctness . . . . . . . . . . . . . . . . . . . . . . .
25


CONTENTS
v

4
Pipeline-Native Transformer Architectures
27
4.1
The Layer Dependency Problem . . . . . . . . . . . . . . . . . . . . .
27
4.1.1
Formal Statement . . . . . . . . . . . . . . . . . . . . . . . . .
27
4.1.2
The Vertical Pipeline Idea . . . . . . . . . . . . . . . . . . . .
28
4.1.3
Bandwidth Implications
. . . . . . . . . . . . . . . . . . . . .
28
4.2
DAG Rewriting: Dense Delay and Expert Delay . . . . . . . . . . . .
29
4.2.1
The Dense Delay Transform . . . . . . . . . . . . . . . . . . .
29
4.2.2
The Expert Delay Transform . . . . . . . . . . . . . . . . . . .
29
4.2.3
The CombineStyle Variants
. . . . . . . . . . . . . . . . . . .
30
4.3
The Five Candidate Architectures . . . . . . . . . . . . . . . . . . . .
30
4.3.1
Arch1: Decoupled Residual Streams . . . . . . . . . . . . . . .
31
4.3.2
Arch2_4_combined: Dense-and-Expert Delay . . . . . . . . .
31
4.3.3
Arch2_4_sync: Synchronous Variant . . . . . . . . . . . . . .
31
4.3.4
Arch3: Pipeline Registers
. . . . . . . . . . . . . . . . . . . .
32
4.3.5
Arch4: Asynchronous Experts . . . . . . . . . . . . . . . . . .
32
4.3.6
Arch5: Fixed-Point Iteration with Weight Sharing . . . . . . .
32
4.4
The Delay-Aware Scheduler
. . . . . . . . . . . . . . . . . . . . . . .
33
4.4.1
Ring-Buffered Residual History
. . . . . . . . . . . . . . . . .
33
4.4.2
Multi-Layer Driver . . . . . . . . . . . . . . . . . . . . . . . .
33
4.5
Bandwidth Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33
4.5.1
Critical-Path Analysis
. . . . . . . . . . . . . . . . . . . . . .
33
4.5.2
Dense Delay vs. Expert Delay as Complementary Knobs
. . .
35
4.6
The Co-Design Trade-off . . . . . . . . . . . . . . . . . . . . . . . . .
36

5
Evaluation
38
5.1
Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . .
38
5.1.1
Hardware
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
38
5.1.2
Software and Build . . . . . . . . . . . . . . . . . . . . . . . .
39
5.2
Training Quality: Five Architectures
. . . . . . . . . . . . . . . . . .
39
5.2.1
Training Protocol . . . . . . . . . . . . . . . . . . . . . . . . .
39
5.2.2
Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
39
5.2.3
Large-Scale Validation: arch2_4_8k_4l
. . . . . . . . . . . .
40
5.3
Claim 1: Conditional Expert Loading . . . . . . . . . . . . . . . . . .
40
5.4
Claim 2: Tile-Streaming Cache Locality
. . . . . . . . . . . . . . . .
41
5.4.1
Wall-Clock Proxy (Windows, May 2026)
. . . . . . . . . . . .
41
5.4.2
PMU Hardware Counter Measurement (Linux KVM, May 2026) 41
5.5
Claim 3: AVX2 Q4 Kernels
. . . . . . . . . . . . . . . . . . . . . . .
42
5.6
Claim 4: Fused Projections . . . . . . . . . . . . . . . . . . . . . . . .
42
5.7
Claim 5: Compute-Order File Layout . . . . . . . . . . . . . . . . . .
43
5.8
Claim 6: Explicit Prefetch (PREFETCHT0) . . . . . . . . . . . . . .
43
5.8.1
64 MB Scale (arch2_4_combined, Ryzen 5 2600)
. . . . . . .
43
5.8.2
4.7 GB Scale (arch2_4_8k_4l, Windows SATA SSD) . . . . .
43


CONTENTS
vi

5.9
Claim 7: Vertical Pipeline Bandwidth Reduction . . . . . . . . . . . .
44
5.9.1
Method
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
44
5.9.2
Result . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
45
5.10 Claim 8: Stage-Major Disk Layout Readahead Benefit . . . . . . . . .
45
5.10.1 64 MB Scale . . . . . . . . . . . . . . . . . . . . . . . . . . . .
45
5.10.2 4.7 GB Scale . . . . . . . . . . . . . . . . . . . . . . . . . . . .
45
5.11 PyTorch Parity Validation . . . . . . . . . . . . . . . . . . . . . . . .
46
5.11.1 arch2_4_combined . . . . . . . . . . . . . . . . . . . . . . . .
46
5.11.2 arch4_async_experts . . . . . . . . . . . . . . . . . . . . . . .
46
5.11.3 Test Suite Coverage . . . . . . . . . . . . . . . . . . . . . . . .
46
5.12 Summary of Results
. . . . . . . . . . . . . . . . . . . . . . . . . . .
47
5.13 Head-to-Head Comparison with llama.cpp and vLLM . . . . . . . . .
48
5.13.1 Benchmark Model and Hardware
. . . . . . . . . . . . . . . .
48
5.13.2 cflow Decode Optimization
. . . . . . . . . . . . . . . . . .
48
5.13.3 Comparison with llama.cpp and vLLM . . . . . . . . . . . . .
49
5.13.4 Interpretation . . . . . . . . . . . . . . . . . . . . . . . . . . .
50
5.14 Wall-Clock Realization of the Expert-Delay Window . . . . . . . . . .
50
5.14.1 Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
50
5.14.2 Part 1 — Desktop (SATA tier): mechanism proven, net a wash
51
5.14.3 Part 2 — EC2 r6id.8xlarge (NVMe tier): the net win . . . . .
51
5.14.4 Part 3 — Parallel fetch: a null that locates the ceiling . . . . .
52
5.14.5 What this adds to Claims 7 and 8 . . . . . . . . . . . . . . . .
53

6
Discussion
54
6.1
The Co-Design Philosophy . . . . . . . . . . . . . . . . . . . . . . . .
54
6.1.1
What Co-Design Means Here
. . . . . . . . . . . . . . . . . .
54
6.1.2
The Scope of the Result
. . . . . . . . . . . . . . . . . . . . .
54
6.1.3
Relationship to Existing Work . . . . . . . . . . . . . . . . . .
55
6.2
Scaling Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
56
6.2.1
How the Bandwidth Reduction Scales . . . . . . . . . . . . . .
56
6.2.2
Attention-Path Delays and Future Work . . . . . . . . . . . .
57
6.2.3
Expert Scaling
. . . . . . . . . . . . . . . . . . . . . . . . . .
57
6.3
Speculative Pipeline Recovery . . . . . . . . . . . . . . . . . . . . . .
58
6.3.1
The Approach A Hypothesis . . . . . . . . . . . . . . . . . . .
58
6.3.2
The Approach B Hypothesis . . . . . . . . . . . . . . . . . . .
58
6.4
Limitations
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
59

7
Conclusion
60
7.1
Summary of Contributions . . . . . . . . . . . . . . . . . . . . . . . .
60
7.2
The Co-Design Thesis, Restated . . . . . . . . . . . . . . . . . . . . .
61
7.3
Implications for Production CPU Inference . . . . . . . . . . . . . . .
62
7.3.1
When the Result Matters
. . . . . . . . . . . . . . . . . . . .
62


CONTENTS
vii

7.3.2
What Practitioners Should Take Away
. . . . . . . . . . . . .
62
7.4
Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
62
7.5
Closing Remarks
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
63

Bibliography
65


List of Figures

4.1
The bandwidth–quality trade-off across the trained candidates (data of
Table 4.1). The dense delay moves a model left (arch2_4_combined and
its sync ablation are the only points at 4.50 MB/token); the expert delay
moves it down (arch4’s pre-dense routing reaches the best perplexity
but stays at the undelayed bandwidth). arch5’s weight sharing trades
bandwidth for parameter efficiency. . . . . . . . . . . . . . . . . . . .
35

5.1
The expert-overlap A/B on EC2 r6id.8xlarge (3-run medians of Table 5.6;
NVMe expert tier). The sync arm pays the serial sum C +IO and grows
as threads shrink; the staged arm pins at the disk floor max(C, IOeff)
at every thread count. Annotations give the measured net speedup; the
(C+IO)/ max(C, IOeff) model reproduces all four within 1%.
. . . . .
52

viii


List of Tables

1.1
Thesis scorecard: eight claims, their status, and the headline measured
result.
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5

4.1
Pipeline-native architecture summary: training results and bandwidth
analysis. All architectures trained 10K steps on TinyStories, d=512,
L=6. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36

5.1
Wall-clock speedup of tiled over naive matrix-vector products at arch2_4_8k_4l
geometry (d = 8192). Ratio > 1.0 means tiled is faster. Working-set
size relative to 32 KB L1-d. . . . . . . . . . . . . . . . . . . . . . . . .
41
5.2
Hardware PMU: L1-d read miss counts, tiled vs. naive, Xeon E5-2650
KVM. Lower miss count is better. Ratio = naive / tiled. Raw counts
were recorded only for the dense-down workload; for the remaining
rows the benchmark harness logged the ratio alone. . . . . . . . . . .
42
5.3
Thesis scorecard: evaluation status for all eight claims. . . . . . . . .
47
5.4
cflow single-token decode throughput on arch2_4_8k_16l (30.9B,
Q4), AWS r6i.8xlarge, 32 threads. Changes are cumulative. . . . . . .
48
5.5
End-to-end single-token decode throughput, same AWS r6i.8xlarge
CPU. Decode rate is reported as steady-state tokens per second; it is
insensitive to sampling temperature and prompt for a fixed model. . .
49
5.6
Expert-overlap A/B on EC2 r6id.8xlarge (NVMe expert tier), 3-run
medians. “Hidden” is the reduction in per-token expert-read stall.
Identical greedy output in all 24 cells. . . . . . . . . . . . . . . . . . .
52

ix


Chapter 1

Introduction

1.1
The Bandwidth Bottleneck in LLM Inference

Almost all of the engineering effort in language-model serving has gone into GPUs:
pipeline parallelism across devices, batching, FlashAttention, speculative decoding.
CPU inference, by comparison, is usually treated as a fallback — what you run when
there is no accelerator to be had. The assumption underneath that treatment is that
CPU inference is GPU inference, only slower.
It is not. The two are limited by different things, and the CPU limit is the simpler
one to state: arithmetic intensity. A CPU has far more arithmetic throughput than it
can keep fed from memory, and single-token decoding never supplies enough work per
byte loaded to close that gap. The rest of this section makes the claim quantitative.
A modern desktop or server CPU executes approximately one teraflop per second
of floating-point arithmetic while moving approximately 50 gigabytes per second
from main memory to the processor. The ratio—the machine balance—is roughly
20 FLOP/byte. For inference to be compute-bound, each byte loaded from memory
must participate in at least 20 arithmetic operations before being evicted. Single-token
autoregressive decoding does not come close to this threshold.
Consider a transformer weight matrix stored in Q4 quantization: 0.5 bytes per
parameter. Computing the matrix-vector product with a single token’s activation
vector requires two floating-point operations per weight element (one multiply, one
add), yielding an arithmetic intensity of 2 FLOP / 0.5 bytes = 4 FLOP/byte. This
is five times below the machine balance. Every cycle the CPU spends waiting for
weights to arrive from RAM is a cycle wasted.
So on a CPU, single-token decode latency is set almost entirely by how many bytes
move, not by how many operations run. Anything that cuts the byte traffic cuts the
latency with it, close to one-for-one. That is the principle the rest of this work is built
on.

1


CHAPTER 1. INTRODUCTION
2

1.2
The GPU-First Retrofit Problem

The dominant CPU inference ecosystem—llama.cpp [10], ExLlama2 [24], and their
derivatives—emerged not from first-principles CPU design but from GPU inference
code adapted to run on CPU. The adaptations are real and substantial: hand-written
SIMD kernels replace GPU shader code, memory-mapped files replace VRAM, and
scalar attention replaces batched CUDA kernels. But several fundamental design
decisions inherited from GPU execution remain, and they are systematically wrong
for CPU-optimal single-token decoding.
Weight layout. GPU-optimized matrix multiplication arranges weights in row-
major order, designed for warp-level coalesced access across many threads operating
on a wide activation batch. On a CPU executing a single-token forward pass, the
activation vector is narrow (one row of the batch), and each weight row is accessed once
and discarded. There is no coalescing opportunity, and the access pattern generates
L1-cache misses in proportion to the weight matrix size, not in proportion to the
number of tokens being decoded.
Execution order. GPU execution pipelines multiple tokens through a layer using
spatial parallelism across tensor cores. CPU execution of a single token processes one
layer completely before advancing to the next. The file layout of weights in GPU-
derived runtimes reflects GPU execution order, which may not match the order in which
a CPU reads them during single-token decode. Mismatches introduce non-sequential
memory access patterns that defeat hardware prefetchers.
Expert loading in mixture-of-experts models. Modern MoE models such
as Mixtral-8×7B [12] and Gemma 4 26B-A4B [11] select a small number of active
experts per token from a large expert pool. A naive runtime loads all expert weight
matrices during the forward pass and discards the unselected ones. At the geometry
of Gemma 4—128 experts, top-8 selection—this loads 128/8 = 16× the expert weight
bytes actually consumed. On a CPU where bandwidth is the bottleneck, this is a
15-fold avoidable overhead.
None of these is a bug to be patched. They are assumptions baked into the data
layout and the execution order, and unpicking them means starting over with the
CPU memory hierarchy as the first constraint rather than an afterthought.

1.3
The Co-Design Opportunity

The two halves of a deployed inference system—the model architecture and the
inference runtime—are conventionally treated as independent. A model is trained
to minimize perplexity under a standard transformer recipe; a runtime is written to
execute any model that conforms to the standard transformer interface. The runtime
does not know what the model is doing; the model does not know how the runtime
will execute it.
This independence is a valuable abstraction in the GPU regime, where per-token


CHAPTER 1. INTRODUCTION
3

latency is dominated by compute and the memory access pattern of the runtime is
largely irrelevant at the hardware level. It becomes a liability in the CPU regime,
where every byte read from memory has cost, and the order in which bytes are read
determines whether the hardware prefetcher can hide latency.
The central thesis of this report is as follows:

By co-designing the model architecture and the inference runtime together,
with the CPU memory hierarchy as the shared optimization target, it
is possible to achieve per-token memory bandwidth reductions that are
unavailable to either the runtime or the architecture acting independently.

The runtime contribution is a tile-streaming weight format and execution engine,
called cflow, that lays weights out in L2-cache-sized tiles and reads them in precisely
the order required for single-token decode. The architecture contribution is a family
of pipeline-native transformers: models whose inter-layer dependency graphs are
rewritten so that the runtime’s vertical pipeline schedule—reading weights for multiple
layers in stage-major order—is mathematically valid. Neither contribution is useful
without the other: the pipeline schedule saves bandwidth only when the model
architecture permits it, and the model architecture is only beneficial when the runtime
understands and exploits its relaxed dependency constraints.

1.4
Contributions

This report makes the following contributions:

1. The cflow inference runtime. A CPU-first inference engine for transformer
models, built around a tile-native weight format (128×256 Q4 tiles, approx-
imately 18 KB each, sized to fit in L2 cache) and two on-disk formats: a
per-layer format (.cflow) for general single-token decode and a stage-major
format (.vflow) for vertical pipeline execution. The runtime includes fused
QKV and gate-up projections, AVX2-accelerated Q4 inner-product kernels, and
a staged direct-I/O expert-fetch mechanism that reads only the top-k selected
experts’ tiles — driven by the MoE router output and asynchronously overlapped
with compute under the expert-delay schedule (Section 5.14).

2. A taxonomy of pipeline-native transformer architectures. A formal
analysis of the layer dependency DAG in standard pre-norm transformers
demonstrates why stage-major execution is mathematically invalid for single-
token autoregressive decode: layer ℓ+1 requires the complete residual output
of layer ℓ, not its input. I introduce two dependency-relaxation operations—
dense_delay and expert_delay—and three corresponding CombineStyle vari-
ants (ParallelSqrt2, DelayedSum, AsyncExperts) that rewrite the dependency
graph so that the stage-major schedule becomes valid while preserving the model’s
capacity to learn.


CHAPTER 1. INTRODUCTION
4

3. Five trained pipeline-native architectures. I define and train five candi-
date architectures spanning the bandwidth–quality trade-off space: a baseline
with no dependency relaxation (arch1), the bandwidth-optimizing architecture
(arch2_4_combined, dense_delay = 1, expert_delay = 2), a pipeline-register
variant (arch3), the quality-optimizing architecture (arch4_async_experts,
expert_delay = 2, routing from pre-dense activations), and a weight-sharing
exploration (arch5). All five train stably to convergence on TinyStories at
10,000 steps with no gradient explosions or divergence. The reference architec-
ture, arch2_4_combined, achieves a test perplexity of 6.50 and a critical-path
bandwidth reduction of 2.00× relative to the undelayed baseline.

4. A delay-aware multi-layer execution scheduler. A scheduler that reads
each architecture’s dense_delay and expert_delay parameters at runtime,
constructs a ring-buffered residual history, and injects delayed expert outputs
at the correct layer offsets. The scheduler’s output is validated against pinned
PyTorch traces for both arch2_4_combined and arch4_async_experts: the
Rust runtime matches the Python reference to within 0.006% relative norm
error, with exact agreement on the argmax token at position 9760.

5. Empirical evaluation across the thesis scorecard. I measure the following
results against a defined set of eight claims:

• Tile-streaming achieves 7.29× fewer L1-d cache read misses on the
dense-down projection at the trained 8.34B-parameter geometry (Xeon E5-
2650, hardware PMU via perf_event_open);

• The delay-aware scheduler achieves a 2.00× critical-path bandwidth re-
duction (naive 9.00 MB/token →delayed 4.50 MB/token) on arch2_4_combined;

• Claims 6 (PREFETCHT0 explicit prefetch) and 8 (stage-major disk layout)
fail direct measurement: both are tested precisely at two scales (64 MB
and 4.7 GB, direct I/O) and found to provide no measurable benefit when
storage bandwidth is the bottleneck, with mechanistic explanations for why
the benefit cannot manifest until compute dominates I/O.

1.5
Summary of Key Results

Table 1.1 summarizes the eight thesis claims and their measured status: the structural
and bandwidth claims are proven with direct experimental evidence; Claim 6 is refuted
and Claim 8 is inconclusive, each with a results with precise mechanistic explanations.
A separate end-to-end tokens-per-second comparison against llama.cpp and vLLM
— not one of the eight structural claims — is now reported in Section 5.13: cflow
sustains 5.94 tok/s on a 30.9B-parameter pipeline-native MoE, ahead of llama.cpp’s
4.75 tok/s on a parameter-comparable dense model on the same CPU.


CHAPTER 1. INTRODUCTION
5

Table 1.1: Thesis scorecard: eight claims, their status, and the headline measured
result.

#
Claim
Status / Headline Result

1
Conditional expert loading
Proven (structural): only top-k expert
tiles are read
2
Tile-streaming cache locality
Proven: 7.29× fewer L1-d misses
(PMU, Xeon E5-2650, dense-down)
3
AVX2 Q4 kernels
Proven:
implemented and validated
against reference
4
Fused projections
Proven: QKV and gate-up from one
activation cache load
5
Compute-order file layout
Proven by format construction
6
PREFETCHT0 prefetch
Refuted: PF=1 is ≈4% worse at 4.7 GB
direct-I/O
7
Delay-aware pipeline schedule
Proven: 2.00× bandwidth reduction
on arch2_4_combined; realized in wall-
clock at up to 1.68× net (§5.14)
8
Stage-major disk layout
Inconclusive: peak SSD bandwidth iden-
tical; median gap confounded by SSD
cache state

The five-architecture comparison, presented in full in Chapter 4 and evaluated in
Chapter 5, reveals a clean qualitative structure: dense_delay is the bandwidth
knob and expert_delay is the quality knob. The architecture that delays both
the dense FFN read and the expert read (arch2_4_combined) achieves the largest
bandwidth reduction (2.00×). The architecture that delays only the expert read but
routes from pre-dense activations (arch4_async_experts) achieves the best perplexity
(6.26 vs. 6.50), because the router sees a cleaner activation signal before the dense
transformation. These two architectures sit at opposite corners of the achievable
trade-off space; the remaining three probe the boundary between them.

1.6
Scope and Limitations

The experiments in this report target single-token autoregressive decode: the memory-
bandwidth-intensive regime in which each weight byte is loaded once per generated
token. Batched inference, where multiple sequences are decoded simultaneously, shifts
the arithmetic intensity toward compute-bound operation and reduces the relative
benefit of bandwidth optimization. I do not claim that the techniques presented here
generalize to batched decode, though I discuss the conditions under which they might
in Chapter 6.


CHAPTER 1. INTRODUCTION
6

The trained architectures are proof-of-concept models trained on TinyStories [7],
a synthetic children’s story dataset with a vocabulary of 50,257 tokens. The goal is
to validate that pipeline-native architectures can be trained stably with competitive
perplexity relative to each other, not to produce state-of-the-art language models.
The cache-locality and bandwidth claims are validated at the 8.34-billion-parameter
geometry of arch2_4_8k_4l, trained on Lambda cloud infrastructure with eight A100
80 GB GPUs under FSDP, providing a hardware-realistic experimental basis for the
bandwidth analysis.
Claims 6 and 8 are explicitly not proven. I include them in the report because the
failure modes are instructive: PREFETCHT0 prefetches data from RAM to L1 cache, but
when the bottleneck is storage to RAM (12–17 seconds of I/O versus 96 milliseconds of
compute at the 4.7 GB scale), there is nothing for it to overlap. Stage-major disk layout
provides a theoretical readahead advantage only when the runtime can asynchronously
stream stage ℓwhile computing stage ℓ−1; the current single-token scheduler reads
the entire file sequentially and the I/O and compute phases do not overlap. These are
not experimental failures; they are experimentally confirmed structural limitations.

1.7
Report Organization

Chapter 2: Background. I introduce the CPU memory hierarchy and the roofline
model for single-token decode, establish the arithmetic intensity argument formally,
survey existing CPU inference runtimes and their inherited GPU assumptions, and
review the transformer architecture and MoE routing mechanisms that the rest of the
report builds on.
Chapter 3: The cflow Runtime.
I describe the design of the tile-native
weight format, the .cflow and .vflow file formats, the fused projection kernels, the
conditional expert prefetch mechanism, and the AVX2 inner-product pipeline.
Chapter 4: Pipeline-Native Transformer Architectures. I formalize the
dependency problem in standard pre-norm transformers, introduce the dense_delay
and expert_delay rewriting operations, describe all five pipeline-native architectures,
present the delay-aware scheduler, and derive the bandwidth model.
Chapter 5: Evaluation. I report training quality across the five architectures,
the PMU-measured cache locality result, the bandwidth reduction measurement, the
storage I/O experiments (including the negative results for claims 6 and 8), and the
Rust–Python parity validation.
Chapter 6: Discussion. I examine the co-design philosophy, characterize the
bandwidth–quality trade-off space, project the delay-aware pipeline to production-scale
MoE architectures, and discuss the speculative pipeline recovery directions that define
the next research phase.
Chapter 7: Conclusion. I restate the co-design thesis with supporting evidence,
enumerate the validated contributions, and situate the work within the broader
trajectory of CPU-first inference research.


Chapter 2

Background

2.1
CPU Memory Hierarchy and the Roofline Model

2.1.1
The Memory Hierarchy

Modern CPUs present a layered memory hierarchy in which storage capacity increases
and access latency increases as one moves away from the processor die. A representative
configuration from the experimental hardware used in this report (Intel Xeon E5-2650,
Sandy Bridge microarchitecture) illustrates the key parameters:

• L1-d cache: 32 KB per core, 4-cycle latency, bandwidth ≈400 GB/s (peak,
cache-resident data).

• L2 cache: 256 KB per core, 12-cycle latency, bandwidth ≈200 GB/s.

• L3 cache (shared): 20 MB across 8 cores, ≈30–40 cycles, bandwidth ≈100 GB/s.

• Main memory (DDR3): Unbounded capacity, 40–100 cycles, bandwidth
≈40–50 GB/s per socket.

What matters here is the size of the gap: roughly 8–10× between L1 and main
memory. Two computations with identical arithmetic can differ in throughput by that
factor alone, depending only on whether their working set stays in L1 or has to be
streamed from RAM. That gap is what the tile-streaming design in Chapter 3 is built
to exploit.
For the purposes of latency analysis, what matters is the sustained bandwidth to
main memory: roughly 40–50 GB/s for a modern desktop or server CPU. This is the
bandwidth that determines single-token inference latency.

2.1.2
The Roofline Model

The roofline model [29] characterizes whether a given computation is compute-bound
or memory-bandwidth-bound by comparing its arithmetic intensity (FLOPs per byte

7


CHAPTER 2. BACKGROUND
8

loaded from memory) against the machine’s compute-to-bandwidth ratio (peak FLOPs
per second divided by peak memory bandwidth in bytes per second).

Definition 2.1 (Arithmetic Intensity). For a computation requiring F floating-point
operations and loading B bytes from memory, the arithmetic intensity is I = F/B
(FLOP/byte).

Definition 2.2 (Machine Balance). For a processor with peak compute throughput
Pmax (FLOP/s) and peak memory bandwidth Bmax (byte/s), the machine balance is
R = Pmax/Bmax (FLOP/byte).

When I < R the computation is memory-bandwidth-bound: memory cannot feed
the arithmetic units fast enough to keep them busy. When I > R it is compute-bound,
and the memory subsystem supplies data faster than the units can consume it.
For a CPU with Pmax = 1 TFLOP/s and Bmax = 50 GB/s:

R =
1012 FLOP/s
50 × 109 byte/s = 20 FLOP/byte.

Any computation with arithmetic intensity below 20 FLOP/byte is memory-bandwidth-
bound.

2.1.3
Arithmetic Intensity of Single-Token Transformer De-
code

Consider a single matrix-vector product, the dominant operation in transformer
inference: a weight matrix W ∈Rm×n applied to an activation vector x ∈Rn. The
computation requires 2mn floating-point operations (one multiply and one add per
weight element). In Q4 quantization, each weight parameter occupies 0.5 bytes. The
arithmetic intensity is:

Idecode =
2mn FLOP
0.5 · mn bytes = 4 FLOP/byte.

This is five times below the machine balance of 20 FLOP/byte. Single-token trans-
former inference is firmly in the memory-bandwidth-bound regime: the CPU spends
most of its time waiting for weight bytes to arrive from RAM, not performing arith-
metic.
The intensity of 4 FLOP/byte is an upper bound; in practice it is often lower. The
weight matrix must be loaded from memory exactly once per token, and the activation
vector (one row of the batch) fits entirely in L1 cache. The FLOPs performed on cached
activations do not change the bandwidth requirement. The latency of a complete
single-token forward pass through all layers is therefore:

τtoken ≈bytes(all weights)

Bmax
,


CHAPTER 2. BACKGROUND
9

where bytes(all weights) is the total weight data transferred from RAM during one
forward pass. This is the quantity that the cflow runtime and the pipeline-native
architectures are designed to minimize.

2.2
The Standard Pre-Norm Transformer

2.2.1
Architecture

The standard pre-norm transformer decoder [26, 30] processes a sequence of tokens
t = (t1, t2, . . . , tT) through an embedding layer, a stack of L identical transformer
layers, and an output projection. Each layer applies two sub-operations to a residual
stream x ∈Rd: a multi-head self-attention block and a feed-forward network, both
preceded by layer normalization (the pre-norm convention). The residual stream is
updated by adding each sub-operation’s output:

x ←x + Attn(Norm(x)) ,
(2.1)
x ←x + FFN(Norm(x)) .
(2.2)

After all L layers, a final normalization and linear projection produce logits over
the vocabulary. For autoregressive generation, inference proceeds token by token:
the model is evaluated on the current context to produce a distribution over the
next token, one token is sampled, and the process repeats. Only the final position’s
activation vector needs to be processed; the key-value projections for all prior positions
are cached (the KV cache) and reused.

2.2.2
Multi-Head Attention

The attention sub-operation computes queries, keys, and values by applying learned
projection matrices to the normalized residual stream, then computes scaled dot-
product attention:

Attn(x) = ConcatH
h=1

�
softmax

�QhK⊤
h
√dk

�
Vh

�
WO,
(2.3)

where Qh = xW h
Q, Kh = xW h
K, Vh = xW h
V are the query, key, and value projections
for head h, dk is the head dimension, and WO is the output projection. For single-token
decode, the query vector for the new token attends to all cached key-value pairs; only
the projections WQ, WK, WV , and WO must be loaded from memory (the KV cache
itself is already in RAM or L3 cache).
Grouped-query attention (GQA). In grouped-query attention [1], the number
of key-value heads HKV is smaller than the number of query heads HQ, with HQ/HKV
query heads sharing each key-value head. GQA reduces the size of the KV cache and


CHAPTER 2. BACKGROUND
10

the bandwidth cost of loading KV projections, while keeping the query projection at
full width. The trained architectures in this report use GQA with varying HQ and
HKV ratios depending on scale.

2.2.3
Feed-Forward Network

The feed-forward sub-operation applies a two-layer fully-connected network with a
gated activation function. The GeGLU variant [19] used throughout this report
computes:

FFN(x) = (xWgate ⊙GELU(xWup)) Wdown,
(2.4)

where Wgate, Wup ∈Rd×dff and Wdown ∈Rdff×d are learned weight matrices, ⊙
denotes elementwise multiplication, and dff is the feed-forward hidden dimension. The
bandwidth cost per token is dominated by loading these three matrices: 3×d×dff×0.5
bytes in Q4.

2.2.4
Mixture-of-Experts Feed-Forward Layers

Mixture-of-experts (MoE) layers [20, 8] replace the single FFN with a collection of
E independent expert FFNs and a router that selects a small number k of them to
activate for each token. The router applies a linear projection to the normalized
residual stream, selects the top-k experts by score, and computes a weighted sum of
their outputs:

MoE(x) =
�

i∈top-k
si · FFNi(x),
(2.5)

where si are softmax-normalized routing weights for the selected experts.
The bandwidth implication is significant. A naive runtime loads all E expert
weight matrices during the forward pass, uses k of them, and discards the rest. The
wasted bandwidth ratio is (E −k)/k. At the geometry of Gemma 4 26B-A4B [11]
(E = 128, k = 8), the naive runtime moves 128/8 = 16× more expert weight bytes
than it consumes. A runtime that reads only the selected experts’ tiles eliminates
this waste entirely, requiring prior knowledge of which experts are selected—which is
produced by the router in the forward pass.

2.2.5
The Layer Dependency DAG

The computational dependencies of the standard pre-norm transformer form a strict
chain. Let xin
ℓdenote the residual stream entering layer ℓand xout
ℓ
denote the residual
stream leaving it. From Equations (2.1) and (2.2):

xout
ℓ
= xin
ℓ+ Attnℓ
�
Norm(xin
ℓ)
�
+ FFNℓ
�
Norm
�
xin
ℓ+ Attnℓ(Norm(xin
ℓ))
��
,


CHAPTER 2. BACKGROUND
11

where the FFN input is itself dependent on the attention output within the same
layer. The key constraint is:
xin
ℓ+1 = xout
ℓ.
(2.6)

Layer ℓ+1 cannot begin until layer ℓhas produced its complete output, including both
the attention residual and the FFN residual. This is the dependency structure that
prevents stage-major execution for single-token decode, as formalized in Chapter 4.

2.3
Quantization

2.3.1
Q4 Quantization

Quantization reduces the per-parameter storage cost by representing weights at lower
precision than the training format. The Q4_0 format used throughout this report
stores each weight value as a 4-bit integer, with a per-block scale factor shared across
a block of 32 consecutive values. Each weight therefore costs 4/8 = 0.5 bytes, plus a
negligible overhead for the scale factor.
The dequantization operation, applied just before each matrix-vector product,
recovers an approximate floating-point value:

ˆwi = scale × (qi −offset),

where qi is the 4-bit stored value and offset is a fixed midpoint (8 for unsigned Q4). In
AVX2 SIMD implementations, this dequantization can be fused with the dot product,
operating on 32 values per iteration with a single scale load. The dequantized values
never need to be materialized in memory; they flow directly through the arithmetic
pipeline, keeping bandwidth demand at the Q4 rate of 0.5 bytes per parameter.

2.3.2
GGUF and llama.cpp

The GGUF file format [9] is the de facto standard for distributing quantized transformer
models for CPU inference. It stores weights in row-major order under the Q4_K_M
or similar quantization scheme, with a global header and per-tensor metadata. The
llama.cpp runtime [10] reads GGUF files and executes transformer inference using
hand-written SIMD kernels for x86 and ARM processors.
The cflow format is designed as a direct alternative to GGUF. The key differences
are: (1) weights are stored in tile-major order (128×256 tiles in compute sequence,
not row-major), (2) expert tiles are stored in a separate random-access bank to
enable conditional loading, and (3) the header carries architecture-specific fields
(dense_delay, expert_delay, CombineStyle) that the runtime uses to construct the
execution schedule. Chapter 3 describes the format in detail.


CHAPTER 2. BACKGROUND
12

2.4
Existing CPU Inference Runtimes

I survey the major CPU inference runtimes, focusing on the design decisions that
distinguish them from cflow.
llama.cpp [10] is the most widely deployed CPU inference runtime for large
language models. It supports a broad range of quantization formats (Q4_0, Q4_K_M,
Q8_0, and others) and model families (LLaMA, Mistral, Gemma, Mixtral). Its kernel
design targets high arithmetic throughput using architecture-specific SIMD paths
(AVX2, AVX512, ARM NEON, Apple Metal).
The weight layout is row-major,
optimized for batched execution across multiple tokens. For single-token decode,
llama.cpp loads each weight row sequentially, with no tile-level L2 reuse. Expert
loading in MoE models follows the naive all-experts path: all expert weight matrices
for the current layer are loaded, the router selects k, and the unselected experts’ weight
data is discarded.
ExLlama2 [24] focuses on GPU inference with extreme quantization (EXL2
format, sub-4-bit per weight), with a CPU execution path added for compatibility.
The CPU path inherits the GPU data layout and is not designed for bandwidth-optimal
single-token decode.
MLC-LLM [15] uses a compilation approach: model execution plans are compiled
ahead of time using TVM [5], targeting CPUs, GPUs, and mobile platforms. The
compilation pipeline enables architecture-specific optimizations but operates on the
standard per-layer execution order; vertical pipelining is not a target.
ctransformers and CTranslate2 [18] are similar in spirit to llama.cpp: quan-
tized weights loaded from disk (or memory-mapped), per-layer execution, no tile-
streaming.
For all their differences, these runtimes share an origin: each is designed for, or
carried over from, GPU execution. None co-designs the model with the runtime to
make cross-layer reordering possible in the first place.

2.5
Mixture-of-Experts at Scale

The motivation for the conditional expert prefetch design in cflow is most clearly
illustrated by Gemma 4 26B-A4B [11], a publicly released MoE model that exemplifies
the bandwidth gap between naive and selective expert loading. Gemma 4 26B-A4B has
128 total experts per MoE layer and selects the top-8 for each token. Its architecture
includes 30 layers (25 sliding-window attention layers and 5 full-attention layers), with
a hidden dimension of 2,816. The expert hidden dimension is 704.
Under naive loading, a single forward pass through one layer loads all 128 expert
weight matrices (gate, up, and down projections), discarding 120 of them after the


CHAPTER 2. BACKGROUND
13

router selects the top 8. The bandwidth waste is:

waste = 128 −8

8
= 120

8
= 15 × .

Conditional loading—reading only the 8 selected experts’ tiles—eliminates this waste.
At Q4, the expert parameters for a single top-8 selection amount to approximately
8 × 3 × 704 × 2816 × 0.5 ≈23.7 MB per layer, versus 16 × 23.7 ≈379 MB for all
experts. This is the motivating bandwidth gap that the cflow expert bank design
addresses.
This report uses Gemma 4 as a sizing reference for the bandwidth analysis in
Chapters 3 and 5. I do not target Gemma 4 for end-to-end inference or seek parity
with the Gemma 4 PyTorch implementation; the architecture dimensions appear in
the bandwidth analysis to provide a realistic context for the theoretical savings.

2.6
Related Work

2.6.1
Weight Streaming and Offloading

FlexGen [21] addresses the problem of running large models on limited GPU memory
by offloading weights to CPU memory or disk and streaming them back as needed.
The primary target is high-throughput batched inference rather than latency-optimal
single-token decode, and the weight layout is standard (row-major, layer-sequential).
DejaVu [14] identifies contextual sparsity in attention and FFN weights and skips the
computation and loading of near-zero-contribution parameters, achieving throughput
improvements on GPU hardware.
Both works address bandwidth indirectly (by
reducing data volume) rather than by restructuring weight layout for cache-optimal
access.

2.6.2
Speculative Decoding

Speculative decoding [4, 13] accelerates autoregressive generation by running a small
draft model to propose candidate tokens and a large target model to verify them in
parallel. The technique is orthogonal to the work presented here: it reduces the number
of forward passes required to generate a token but does not reduce the bandwidth
cost of each forward pass.

2.6.3
Structured Sparsity and Expert Routing

Switch Transformer [8] and Mixtral [12] demonstrate that MoE architectures can
maintain model quality with sparse expert activation. Recent work on expert routing
efficiency [16] explores whether the top-1 or top-2 selection can be tuned without
quality degradation. This report is complementary: I accept the routing mechanism as


CHAPTER 2. BACKGROUND
14

given and optimize the bandwidth cost of executing it on CPU, rather than modifying
the routing policy.

2.6.4
Cache-Aware Deep Learning

Tiling for GEMM operations is a classical topic in high-performance computing.
Modern BLAS implementations such as OpenBLAS [17] and BLIS [25] use multi-level
tiling to maximize reuse in L1, L2, and L3 caches. These optimizations target the
batched GEMM case (large matrices, many tokens) where tiling along both the output
and input dimensions is beneficial. For the matrix-vector case (single token, tall-and-
thin weight matrix), classical BLAS tiling does not improve L1-d reuse because the
activation vector is already small; the bottleneck is loading the weight matrix rows.
The tile-streaming approach in cflow is specifically designed for this regime, where
the activation fits in L1 and the challenge is loading weight tiles at L2 granularity.

2.6.5
Model-Runtime Co-Design

The idea of co-designing a model architecture with its execution environment has
precedents in hardware-aware neural architecture search (HW-NAS) [3, 27], which
optimizes model structure for latency or energy on a target device. These approaches
modify the macro-structure of the model (number of layers, filter widths, skip connec-
tions) to match the device’s compute profile, but they do not modify the inter-layer
dependency graph. The pipeline-native architectures introduced in this report make a
more specific change: they rewrite the dependency structure of the transformer itself so
that a particular execution schedule—vertical stage-major pipelining—becomes mathe-
matically valid. The closest architectural precedent is the parallel attention–FFN block
of GPT-J [28] and PaLM [6], which computes the FFN from the pre-attention residual
and thereby removes the intra-layer attention→FFN dependency. The dense-delay
transform generalizes this across layers (the FFN reads a residual from δd layers
earlier), adds an expert-delay counterpart, and — the part with no precedent I am
aware of — co-designs the rewritten dependency graph with the runtime execution
schedule that exploits it.


Chapter 3

The cflow Runtime

This chapter describes the design and implementation of the cflow runtime: a CPU-
first streaming inference engine for transformer models built from scratch around CPU
memory hierarchies. The runtime has four primary components: (1) a tile-native weight
format that pre-slices matrices into L2-sized chunks stored in compute-consumption
order; (2) two on-disk formats, .cflow for per-layer streaming and .vflow for vertical
pipeline execution; (3) a fused compute pipeline spanning AVX2 inner products,
attention, and normalization; and (4) a conditional expert prefetch mechanism that
eliminates unused MoE weight traffic.

3.1
Design Principles

Three ideas run through the whole runtime and account for most of its design.
Zero-copy from storage. Weights are memory-mapped from disk using mmap
on POSIX systems and CreateFileMapping/MapViewOfFile on Windows.
Tiles
are parsed directly from the mapped region via unsafe pointer casts into repr(C,
packed) structs; no deserialization step materializes copies. The MappedTile type
holds references into the mapped region and is consumed directly by the compute
kernels.
Compute-order layout. Bytes on disk appear in the same sequence in which they
are consumed during a forward pass. Sequential mmap reads feed the OS readahead
mechanism, achieving sustained near-peak bandwidth for the non-expert weight stream.
The expert tiles, whose access pattern is data-dependent, are stored in a separate
random-access bank and loaded

..._This content has been truncated to stay below 50000 characters_...
