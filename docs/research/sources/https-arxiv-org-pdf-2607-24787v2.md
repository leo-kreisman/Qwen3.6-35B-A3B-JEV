# SOURCE: https://arxiv.org/pdf/2607.24787v2

SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation
Models

Jinwei Kong1,2,*, Runqi Meng2,*, Xiaolong Zhong1, Fanyi Wang1, Wentao Qiu1,3,
Haotian Hu1, Yongjian Zhou4, Zhenhua Ge1,†

1StepX, 2ShanghaiTech University, 3Xiamen University, 4Chongqing University
*Equal contribution; †Corresponding author.
kongjw2023@shanghaitech.edu.cn,
gezhenhua@ustc.edu

Abstract

Sparse Mixture of Experts (MoE) models enable scalable
foundation models through conditional computation, but their
large expert pools pose significant memory challenges for
deployment. Expert offloading reduces accelerator memory
usage by storing inactive experts in host memory or storage,
yet introduces additional transfer latency because required
experts are only identified after native routing. This reactive
loading places expert transfer on the inference critical path.
We propose SpecPrefetch, a parameter-efficient and router-
preserving framework for expert prefetching in offloaded MoE
inference. SpecPrefetch employs lightweight layer-specific
adapters to predict next-layer expert priorities solely for asyn-
chronous transfer, while preserving the frozen native router
for final expert selection. To avoid excessive transfers beyond
the achievable overlap capacity, SpecPrefetch introduces a
window-aware prefetch budgeting policy based on the pro-
filed computation and transfer overlap window. This design
improves expert availability without modifying pretrained
routing behavior. Experiments on Qwen3-VL-30B-A3B and
DeepSeek-VL2-Tiny demonstrate that SpecPrefetch achieves
superior next-layer expert coverage across diverse workloads
with substantially fewer trainable parameters than learned pre-
dictor baselines. Real-device evaluation on Snapdragon 8 Elite
further shows up to 20% decoding throughput improvement
under storage-constrained settings over a compute-optimized
offloading runtime, validating the effectiveness of prediction-
guided and budget-aware expert prefetching for resource-
constrained MoE deployment. Code and weights will be re-
leased upon acceptance.

1
Introduction
Recent progress in large language models (LLMs) and vi-
sion language models (VLMs) has been driven by contin-
ued growth in model capacity, training data, and compu-
tation. However, dense scaling incurs substantial inference
costs in memory, computation, and latency. Mixture of Ex-
perts (MoE) architectures provide a more efficient scaling
paradigm by retaining a large parameter capacity while acti-
vating only a small subset of experts for each token. Existing
MoE models adopt either routed expert pools or hybrid strat-
egy with shared and routed experts (Qwen Team 2025a; Jiang
et al. 2024; Dai et al. 2024). Although sparse activation re-
duces per token computation, inference still requires access
to the complete expert pool. Consequently, the large expert
footprint becomes a major obstacle to deploying MoE models

Figure 1: Motivation for window-aware expert prefetching.
(a) Per layer decoding latency under resident and on-demand
expert loading. Exposed expert loading accounts for 67% of
per layer latency on the GPU and 52% under mobile cold
cache, while MoE computation provides an opportunity for
transfer overlap. (b) Measured prefetch window and candi-
date transfer demand on Qwen3-VL-30B-A3B. Curves report
per layer averages, while percentages denote the fraction of
event level demands exceeding their corresponding prefetch
windows. The window supports about two transfers, while
top-4 and top-8 exceed it in 37% and 89% of events.

on commodity GPUs and resource constrained edge devices.
Expert offloading reduces memory pressure by storing inac-
tive experts in CPU memory or host storage and transferring
selected experts during inference.
Nevertheless, expert offloading introduces substantial data
movement because the required experts remain unknown un-

arXiv:2607.24787v2  [cs.AI]  30 Jul 2026


Figure 2: Comparison of expert-loading timelines in of-
floaded MoE inference. On-demand execution exposes
expert-transfer latency after native routing. SpecPrefetch pre-
dicts next-layer expert candidates in advance and overlaps
their transfer with ongoing computation, while the native
router remains authoritative for final expert execution.

til the native router evaluates the corresponding hidden rep-
resentations. Consequently, conventional on-demand loading
places expert transfer on the inference critical path and stalls
the accelerator before MoE execution. Figure 1(a) quantifies
the resulting stall and further shows that MoE FFN compu-
tation accounts for a substantial fraction of per layer latency.
Meanwhile, the relatively long FFN stage provides an op-
portunity to overlap next-layer expert transfer with ongoing
computation. Exploiting the overlap window requires expert
demand to be predicted before the target router is executed.
However, the available window may be insufficient to transfer
all predicted candidates. As shown in Figure 1(b), candidate
transfer demand exceeds the available prefetch capacity in
multiple layers. Although enlarging the candidate set can im-
prove expert coverage, redundant transfers consume limited
bandwidth and memory. Therefore, efficient offloaded MoE
inference requires both accurate next-layer expert prediction
and a bounded prefetch budget selected from the profiled
computation and transfer overlap window.
Existing approaches address only individual aspects of
the expert transfer bottleneck, as summarized in Figure 2.
Quantization and compression reduce expert size and trans-
fer latency, but expert demand remains unavailable before
native routing (Yan et al. 2025). Training free methods, such
as FATE, exploit cross layer routing correlations with negli-
gible prediction overhead (Fang et al. 2025b). However, lim-
ited prediction accuracy often requires a larger candidate set,
thereby increasing redundant transfers and cache pressure.
Training based methods improve predictive capacity through
auxiliary predictors or pre gating mechanisms (Hwang et al.
2024; Chen et al. 2025), but may introduce additional latency
and memory overhead or interfere with native routing deci-
sions. More importantly, prediction accuracy alone does not
determine prefetching effectiveness. A predicted expert con-
tributes to latency reduction only when its transfer overlaps
useful computation, whereas incorrect predictions consume
limited bandwidth and memory resources. Effective expert

prefetching must balance expert coverage, prediction over-
head, and a window-derived prefetch budget, while preserv-
ing native routing.
To address these challenges, we propose SpecPrefetch, a
parameter-efficient and router preserving framework for ex-
pert prefetching in offloaded MoE inference. SpecPrefetch
employs lightweight layer-specific adapters to predict next-
layer expert priorities and initiate asynchronous transfers,
while retaining the native router for final expert selec-
tion. Moreover, a window-aware budgeting policy limits
priority-based asynchronous prefetching according to a max-
imum budget selected from the profiled overlap window.
By separating transfer prediction from execution routing,
SpecPrefetch improves expert readiness without altering the
routing behavior of the pretrained MoE model.
We evaluate SpecPrefetch at both the model and system
levels. Model level experiments on Qwen3-VL-30B-A3B and
DeepSeek-VL2-Tiny demonstrate accurate next-layer expert
prediction and high native expert coverage with limited train-
able overhead across language and multimodal workloads.
Furthermore, real device deployment on a Snapdragon 8
Elite platform confirms that the improved expert readiness
translates into practical acceleration, yielding up to a 20% im-
provement in decoding throughput under storage constrained
offloading conditions.
Our main contributions are as follows.

• We formulate expert prefetching as a router preserving
transfer problem that jointly considers expert coverage,
prediction overhead, and the prefetch capacity provided
by the computation and transfer overlap window.
• We propose SpecPrefetch, which combines parameter-
efficient next-layer expert prediction with window-aware
prefetch budgeting while preserving native routing for
execution.
• We evaluate representative MoE models and deploy on a
real mobile platform, demonstrating accurate expert pre-
diction with low overhead and up to a 20% improvement
in decoding throughput.

2
Related Work
2.1
Sparse MoE Foundation Models
Mixture of Experts (MoE) scales foundation models through
conditional computation. In sparse MoE layers, a router acti-
vates only a small subset of experts for each token, increasing
total parameter capacity while limiting activated computa-
tion (Shazeer et al. 2017). Early systems such as GShard,
Switch Transformer, and GLaM show that sparse routing im-
proves model capacity and language modeling performance
with controlled computation cost, while introducing chal-
lenges in routing stability, load balancing, communication,
and expert specialization (Lepikhin et al. 2021; Fedus, Zoph,
and Shazeer 2022; Du et al. 2022).
Recent MoE models further diversify expert architecture.
Mixtral uses a pure routed expert pool (Jiang et al. 2024),
DeepSeekMoE separates shared and routed experts (Dai et al.
2024), and sparse upcycling initializes MoE models from
dense checkpoints to reduce training cost (Komatsuzaki et al.


Figure 3: Overview of SpecPrefetch. (a) A lightweight adapter is attached alongside each MoE block. (b) During execution at
layer l, the adapter predicts next-layer expert priorities and asynchronously prefetches candidate experts, while the native router
Rl+1 remains unchanged for final expert selection.

2023). These designs differ in how expert capacity is con-
structed, shared, and routed, which directly affects offloading
and expert prefetching.

2.2
Offloaded MoE Inference and Expert
Prefetching

Expert offloading enables large sparse MoE models to op-
erate under limited accelerator memory by storing inactive
experts in CPU memory or host storage. Existing systems
primarily optimize expert placement and data movement
through caching, heterogeneous memory management, quan-
tization, compression, expert splitting, and asynchronous
transfer (Shen et al. 2022; Xue et al. 2024; He et al. 2024; Yu
et al. 2025a; Fang et al. 2025a; Yan et al. 2025; Tang et al.
2024; Yu et al. 2025b). In parallel, scheduling and prefetching
methods overlap expert transfer with computation through
predictive scheduling, token rebatching, pipeline execution,
and expert aware memory organization (He et al. 2024; Fang
et al. 2025a; Yu et al. 2025a; Abhimanyu Bambhaniya 2024;
Sun and Li 2025). FATE further exploits adjacent layer rout-
ing correlations for training free expert prefetching on edge
devices (Fang et al. 2025b). However, many existing methods
either optimize data movement after expert demand becomes
available or rely on stable routing regularity and throughput
oriented execution mechanisms.
Learned prefetching methods predict future expert activa-
tions through auxiliary predictors, pre gating mechanisms, or
parameter-efficient adaptation (Hwang et al. 2024; Chen et al.
2025; Gavhane et al. 2025; Zhu et al. 2025), while speculative
generation creates additional transfer windows through draft
and verification stages (Leviathan, Kalman, and Matias 2023;
Li et al. 2024; Wang et al. 2025). In contrast, SpecPrefetch
targets standard offloaded MoE inference and predicts next-
layer candidates solely for asynchronous transfer. The native
router remains responsible for final top-K expert selection,
allowing SpecPrefetch to advance expert availability without
altering the routing behavior of the pretrained model.

3
Method
As illustrated in Figure 3, SpecPrefetch separates expert pre-
diction for transfer from native routing for execution. Sec-
tion 3.1 formulates router preserving prefetching under a
finite overlap window. Section 3.2 introduces a lightweight
adapter that estimates next-layer expert priorities. Section 3.3
uses the predicted priorities and a profiled maximum prefetch
budget to initiate asynchronous transfers. Finally, Section 3.4
presents the training objective for the prediction adapters.

3.1
Problem Formulation
We formulate SpecPrefetch as an early expert transfer prob-
lem under a finite overlap window while preserving the native
routing policy. Consider an MoE model with L MoE layers
and n routed experts per layer. Let xl
b,s ∈RD denote the
router input of token (b, s) at layer l, and let Ωdenote the
set of valid tokens in the current batch. The native router Rl
selects K experts for each token:

T l
b,s = TopK
�
Rl �
xl
b,s
��
.

The corresponding batch-level native expert demand is

T l
batch =
�

(b,s)∈Ω
T l
b,s.

The set T l
batch contains all experts required by the cur-
rent batch at layer l and is determined exclusively by the
native router. Before routing is performed at layer l+1, how-
ever, T l+1
batch remains unavailable. For each l = 1, . . . , L −1,
SpecPrefetch therefore estimates an expert priority vector
sl+1 ∈Rn from the router inputs available at layer l. The
prediction guides only early transfer, while the native router
remains responsible for final expert selection.
Let tpred
l
denote the time at which the prediction result at
layer l becomes available, and let texec
l+1 denote the time at
which routed expert computation begins at layer l + 1. The
available prefetch overlap window is


Wl = texec
l+1 −tpred
l
.
Within Wl, SpecPrefetch issues priority-based asyn-
chronous transfers under a profiled maximum prefetch bud-
get. Redundant transfers consume limited transfer capacity,
while missed experts require on-demand loading. Since the
native router remains unchanged, prediction errors affect in-
ference efficiency rather than model outputs.

3.2
Expert Prediction Adapter
SpecPrefetch attaches an external low rank adapter to each
adjacent pair of MoE layers. Given the router input xl
b,s at
layer l, the adapter predicts the expert logits of layer l + 1 as

ˆzl+1
b,s = BlAlxl
b,s.

Here,

Al ∈Rr×D,
Bl ∈Rn×r,
and r ≪D denotes the adapter rank. The corresponding
token level expert distribution is

ˆπl+1
b,s = softmax
�
ˆzl+1
b,s
�
.

Because expert transfers are scheduled at the batch level,
token level predictions are aggregated into expert priorities,

sl+1
e
=
�

(b,s)∈Ω
ˆπl+1
b,s,e,
e = 1, . . . , n.

The resulting priority vector is

sl+1 =
�
sl+1
1
, . . . , sl+1
n
�
.

The score sl+1
e
represents the total predicted routing mass
assigned to expert e and therefore prioritizes experts expected
to serve more tokens in the current batch. For model-level
evaluation with M predicted candidates, the candidate set is

Cl+1
M
= TopM
�
sl+1�
.
Assuming uniform hidden dimensions and expert counts
across layers, adapters for all adjacent MoE layer pairs intro-
duce

(L −1)r(D + n)
trainable parameters.

3.3
Window-Aware Prefetch Budgeting
The predicted priority vector sl+1 ranks the experts at layer
l + 1 according to their expected routing demand. Since
expert transfer can reduce latency only when it overlaps with
the computation preceding the target MoE layer, the number
of prefetched experts must be constrained by the available
overlap window.
Let tpred
l
denote the time at which the prediction becomes
available and texec
l+1 denote the start time of routed expert
computation at layer l + 1. The available prefetch window is

Wl = texec
l+1 −tpred
l
.

We profile the computation window and expert transfer la-
tency on the target platform to determine a maximum prefetch
budget Mmax. This budget limits the number of transfer re-
quests issued for each target MoE layer and prevents exces-
sive prefetching from introducing redundant I/O and cache
pressure.
At runtime, let Al+1(t) and Ul+1(t) denote the experts al-
ready available and currently being transferred, respectively.
These experts are excluded from additional transfer requests.
The remaining experts are ranked according to sl+1
e
, and the
highest-ranked candidates are selected as

Pl+1(t) = TopMmax
��
sl+1
e
| e /∈Al+1(t) ∪Ul+1(t)
��
.

The selected experts are asynchronously transferred in de-
scending order of predicted priority. The actual number of
issued transfers ml satisfies

ml =
��Pl+1(t)
��≤Mmax.

At layer l + 1, the native router independently determines
the experts used for execution. Any required expert that is
not available is loaded on demand. Therefore, the prefetch
budget affects only expert availability and transfer overhead,
while the native routing policy and model outputs remain
unchanged.

3.4
Training Objective

The frozen native router at layer l + 1 provides the target
routing distribution

πl+1
b,s = stopgrad
�
softmax
�
Rl+1 �
xl+1
b,s
���
.

The prediction adapters are optimized by minimizing

Lpred =
1
L −1

L−1
�

l=1

1
|Ω|

�

(b,s)∈Ω
KL
�
πl+1
b,s ∥ˆπl+1
b,s
�
.

Distilling the complete routing distribution preserves the
relative expert ordering required for priority based prefetch-
ing. Only the adapter parameters {Al, Bl}L−1
l=1 are optimized,
while the pretrained MoE model remains frozen.

4
Experiment

4.1
Experimental Setup

Models, Training Data, and Benchmarks.
We evaluate
SpecPrefetch on Qwen3-VL-30B-A3B and DeepSeek-VL2-
Tiny, covering diverse sparse MoE architectures and routing
behaviors (Qwen Team 2025b; Wu et al. 2024). Predictors are
trained on a heterogeneous multimodal instruction corpus,
while evaluation is performed on disjoint VLM and LLM
benchmarks. Detailed training data and benchmark settings
are provided in the supplementary material.


Table 1: Next-layer expert coverage of SpecPrefetch and baseline predictors across LLM and VLM workloads. The evaluated
candidate counts represent limited, native, and expanded prediction settings for each model.

Category
Benchmark
Method
Qwen3-VL-30B-A3B
DeepSeek-VL2-Tiny

R@4†
R@8
R@10
R@3†
R@6
R@8

LLM
Workloads

GSM8K
FATE
47.51
81.40
88.38
48.62
84.66
93.07
Draft Model
–
–
–
46.52
78.57
87.56
ProMoE
48.45
82.97
89.94
46.50
77.80
86.40
SpecPrefetch
48.83
85.60
92.31
48.36
84.35
92.47

HumanEval
FATE
47.22
81.16
88.11
47.43
81.51
90.14
Draft Model
–
–
–
42.68
70.14
80.13
ProMoE
47.30
79.62
86.69
41.50
66.70
75.50
SpecPrefetch
48.68
85.48
92.16
47.61
81.51
90.65

VLM
Workloads

OCRBench
FATE
48.72
84.21
92.05
48.59
84.84
93.07
Draft Model
48.51
84.54
93.01
47.83
83.39
91.36
ProMoE
49.06
87.93
94.01
48.80
86.90
94.20
SpecPrefetch
49.43
89.29
95.29
49.36
89.29
96.45

ChartQA-Test
FATE
48.47
83.88
90.85
48.43
83.70
91.99
Draft Model
49.04
85.88
92.96
48.53
83.76
92.27
ProMoE
49.48
89.01
95.21
49.50
87.60
95.30
SpecPrefetch
49.51
89.40
95.48
49.63
89.61
96.83

HallusionBench
FATE
48.95
85.01
92.80
49.13
86.21
94.98
Draft Model
48.98
85.61
93.36
49.18
87.33
95.19
ProMoE
49.37
88.14
94.38
48.80
86.30
93.80
SpecPrefetch
49.49
89.33
95.34
49.25
88.72
95.96

R@M denotes ExpertRecall@M with M predicted candidates. The evaluated budgets are {K/2, K, K + 2}, where K = 8 for Qwen3-VL-
30B-A3B and K = 6 for DeepSeek-VL2-Tiny. The maximum R@4 and R@3 are 50%.

Baselines.
We compare SpecPrefetch with FATE, Draft
Model, and ProMoE. FATE exploits adjacent layer routing
regularity without training, while Draft Model and ProMoE
use learned predictors for next layer expert demand. All meth-
ods use matched candidate budgets and the same transfer
only protocol. Predicted experts are used only for prefetch-
ing, while the frozen native router independently determines
the executed experts.

Evaluation Metrics.
We report ExpertRecall@M for
model-level prediction quality and ReadyRecall for system-
level expert availability. For each valid token (b, s) ∈Ω,
T l+1
b,s
is the native Top-K expert set, Cl+1
M
is the Top-M
predicted candidate set, and Al+1(t) is the set of experts
available on the accelerator at time t. The metrics are

ExpertRecall @M = 1

|Ω|

�

(b,s)∈Ω

���T l+1
b,s
∩Cl+1
M
���

K
.

ReadyRecall = 1

|Ω|

�

(b,s)∈Ω

���T l+1
b,s
∩Al+1 �
texec
l+1
����

K
.

We additionally report trainable predictor parameters and
decoding throughput in tokens per second.

On Device Evaluation.
We deploy the 4 bit DeepSeek-
VL2-Tiny model on a Qualcomm Snapdragon 8 Elite plat-
form with an Adreno 825 GPU. The model contains 11 MoE
layers, each with 64 routed experts and native Top-K rout-
ing with K = 6, and each quantized expert occupies 1.94
MB. We compare load on-demand execution, a compute op-
timized runtime, and SpecPrefetch built on the same runtime.
Experiments use physical NVMe storage and emulate slower
storage through controlled loading delays.

Implementation Details.
The predictors are trained for
one epoch using AdamW with β = (0.9, 0.999), ϵ = 10−8,
and learning rate 10−4. We use cosine scheduling, a warmup
ratio of 0.03, gradient clipping at 0.5, and bf16 precision.
The maximum input length is 4096 for DeepSeek-VL2-Tiny
and 2048 for Qwen3-VL-30B-A3B. Predictor training uses
PyTorch, HuggingFace Trainer, DeepSpeed, and FlashAtten-
tion 2 on 2 × 8 NVIDIA A800-SXM4-80GB GPUs.

4.2
Comparison Results
Table 1 compares SpecPrefetch with FATE, Draft Model,
and ProMoE under a unified next-layer prediction protocol.


Table 2: Predictor design ablation across all workloads and candidate budgets, together with trainable parameter comparison.

(a) Qwen3-VL-30B-A3B

Single MLP
SpecPrefetch

Benchmark
R@4†
R@8
R@10 R@4†
R@8
R@10

GSM8K
47.56
79.36
86.29
48.83
85.60
92.31

HumanEval
44.77
71.98
78.63
48.68
85.48
92.16

OCRBench
48.23
85.35
91.52
49.43
89.29
95.29

ChartQA-Test
49.31
87.76
94.20
49.51
89.40
95.48

HallusionBench
48.84
85.66
92.03
49.49
89.33
95.34

(b) DeepSeek-VL2-Tiny

Single MLP
SpecPrefetch

Benchmark
R@3†
R@6
R@8
R@3†
R@6
R@8

GSM8K
39.90
64.30
73.30
48.36
84.35
92.47

HumanEval
33.70
53.40
61.60
47.61
81.51
90.65

OCRBench
46.30
78.70
87.40
49.36
89.29
96.45

ChartQA-Test
47.20
79.10
88.20
49.63
89.61
96.83

HallusionBench
46.40
78.70
87.40
49.25
88.72
95.96

R@M denotes ExpertRecall@M with M predicted candidates. R@4 and R@3 have a maximum possible value of 50%.

Table 3: Trainable parameters of different expert predictors.

Model
Single
Draft
ProMoE SpecPrefetch
MLP
Model

DeepSeek-VL2-Tiny
0.78M
26.40M
16.35M
0.82M
Qwen3-VL-30B-A3B 12.19M 19.19M 207.23M
6.48M

Table 4: Runtime overhead of prediction and prefetch con-
trol under cold-cache NVMe. PRED0 executes the complete
prediction and control path but issues no expert transfers.

Configuration
Throughput
Gain vs. OFF
(tok/s)
(%)

OFF
3.753 ± 0.164
–
PRED0
3.730 ± 0.113
−0.53 ± 3.05
SpecPrefetch (Mmax = 8)
4.055 ± 0.129
+8.12 ± 2.79

R@M denotes ExpertRecall@M, where M is the number
of predicted expert candidates.
SpecPrefetch achieves the best or tied-best recall in 27 of
the 30 settings. It ranks first across all settings on Qwen3-
VL-30B-A3B and across HumanEval and all VLM bench-
marks on DeepSeek-VL2-Tiny. Its largest gains over FATE
are 5.52 percentage points at R@8 on ChartQA-Test with
Qwen3-VL-30B-A3B and 5.91 percentage points at R@6
on ChartQA-Test with DeepSeek-VL2-Tiny. The only excep-
tions are the three DeepSeek-VL2-Tiny GSM8K settings,
where FATE leads by 0.26, 0.31, and 0.60 percentage points.
SpecPrefetch also consistently outperforms the learned
predictor baselines, with particularly clear advantages on
VLM workloads. These results demonstrate reliable next-
layer expert coverage across different model architectures,
workloads, and candidate budgets.

4.3
Predictor Design and Parameter Efficiency
Table 2 compares SpecPrefetch with a Single MLP under
identical supervision and candidate budgets. SpecPrefetch
outperforms the Single MLP in all 30 settings, with particu-
larly large gains on DeepSeek-VL2-Tiny. At the native can-
didate budget R@6, it improves recall by 20.05 and 28.11

Table 5: Matched-budget comparison of expert prediction
strategies on Snapdragon 8 Elite with Mmax = 8.

Policy
Cold NVMe
Slow Storage
ReadyRecall
Gain (%)
Gain (%)
(%)

Naive Prefetch
−1.08 ± 2.83
−2.90 ± 2.66
23.4 ± 7.3
SpecPrefetch
6.97 ± 2.56
15.43 ± 2.96
91.7 ± 3.1

percentage points on GSM8K and HumanEval, respectively,
and by more than 10 percentage points on all three VLM
benchmarks. The consistent gains across both models indi-
cate that layer-specific predictors capture cross-layer routing
transitions more effectively than a shared predictor.
As shown in Table 3, these improvements are achieved
with limited trainable overhead. On Qwen3-VL-30B-A3B,
SpecPrefetch uses 6.48M parameters, approximately half
the size of the Single MLP and only 3.1% of ProMoE.
On DeepSeek-VL2-Tiny, its 0.82M parameters are compa-
rable to the 0.78M Single MLP and correspond to only 3.1%
and 5.0% of Draft Model and ProMoE, respectively. These
results demonstrate that the layer-specific low-rank design
improves expert coverage without relying on a substantially
larger predictor.

4.4
On-Device System Evaluation
We evaluate whether improved expert prediction translates
into practical decoding acceleration on a Snapdragon 8 Elite
platform. The experiments use physical NVMe storage, to-
gether with controlled loading delays to emulate slower stor-
age conditions. Unless otherwise specified, the main through-
put comparisons use controlled paired measurements against
the compute-optimized offloading runtime.

Prediction Overhead and End-to-End Benefit.
Table 4
decomposes the runtime effect of prediction-guided prefetch-
ing under cold-cache NVMe. OFF disables both predic-
tion and prefetching, while PRED0 executes the complete
prediction and control path without issuing expert trans-
fers. PRED0 achieves 3.730 ± 0.113 tokens/s, compared
with 3.753 ± 0.164 tokens/s for OFF, corresponding to a
measured difference of only −0.53%. This indicates that


Figure 4: Impact of the maximum prefetch budget under cold-
cache NVMe. Throughput peaks at Mmax = 8, supporting
bounded prefetching under the overlap window.

Figure 5: Storage sensitivity of on device decoding through-
put under physical NVMe and emulated loading delays. The
benefit of SpecPrefetch increases as expert loading becomes
more exposed on the inference critical path.

the prediction and control path introduces limited runtime
overhead. When asynchronous expert transfers are enabled,
SpecPrefetch achieves 4.055 ± 0.129 tokens/s, yielding an
8.12% improvement over OFF. These results show that the la-
tency hidden by expert prefetching outweighs the additional
prediction and control overhead.

Prediction Quality under a Matched Prefetch Budget.
Table 5 compares SpecPrefetch with Naive Prefetch under
the same maximum prefetch budget of Mmax = 8. Naive
Prefetch uses the experts activated by the preceding token,
whereas SpecPrefetch selects candidates according to the
learned expert priorities. SpecPrefetch increases ReadyRe-
call from 23.4% to 91.7%, resulting in throughput gains of
6.97% under cold NVMe and 15.43% under slow storage. In
contrast, Naive Prefetch provides no acceleration and intro-
duces additional overhead. These results show that accurate
prediction translates a limited prefetch budget into higher
expert readiness and throughput.

Figure 6: Cold cache component analysis on Snapdragon 8
Elite after clearing the operating system page cache. The
comparison illustrates the complementary roles of compute
optimization and prediction guided expert prefetching.

Prefetch Budget Sensitivity.
Figure 4 examines the effect
of Mmax under physical cold-cache NVMe. Increasing the
budget improves ReadyRecall from 82.1% to 97.5%, but
throughput peaks at Mmax = 8 and decreases with larger
budgets. This shows that higher expert coverage does not nec-
essarily yield greater acceleration, since excessive prefetch-
ing introduces redundant transfers and additional I/O pres-
sure. It therefore supports using a bounded prefetch budget
determined by the computation and transfer overlap window.

Storage Sensitivity.
Figure 5 evaluates decoding through-
put under physical NVMe and emulated storage delays.
SpecPrefetch provides limited additional benefit when expert
loading is largely hidden under fast NVMe, but its advantage
increases as storage latency becomes more exposed on the in-
ference critical path. Compared with the compute-optimized
runtime, the throughput improvement reaches approximately
20% under the Slow UFS setting and 17.02% under the slow-
est SD-card setting. The experiment evaluates sensitivity to
storage latency rather than shared bandwidth contention.

Complementarity with Compute Optimization.
Fig-
ure 6 compares prediction-guided prefetching with compute
optimization after clearing the operating system page cache.
Their combination achieves the highest throughput, indicat-
ing that the two techniques address complementary bottle-
necks: compute optimization reduces execution and dispatch
overhead, while SpecPrefetch hides exposed expert-loading
latency. Because mobile throughput is sensitive to thermal
state, this experiment is used as a component analysis rather
than as the primary quantitative comparison.

5
Conclusion

We
present
SpecPrefetch,
a
router-preserving
expert
prefetching framework for efficient sparse MoE inference.
By predicting next-layer expert priorities with lightweight
adapters and performing window-aware asynchronous trans-
fers, SpecPrefetch improves expert availability without modi-
fying native routing decisions. Experiments on representative
MoE models and real-device deployment demonstrate con-
sistent gains in inference efficiency, validating the effective-
ness of prediction-guided and budget-aware expert prefetch-
ing for practical MoE deployment.


References

Abhimanyu Bambhaniya, T. K., Sashankh Chengavalli Ku-
mar. 2024. MoE-ERAS: Efficient Runtime and Scheduling
for Mixture-of-Experts Serving. Preprint.
Chen, Y.; Li, C.; Zhang, M.; Wang, W.; Liu, J.; and Guo,
M. 2025. SPMoE: Accelerating Sparse Mixture-of-Experts
Inference with Learned Expert Prediction. Preprint.
Dai, D.; Deng, C.; Zhao, C.; Xu, R. X.; Gao, H.; Chen,
D.; Li, J.; Zeng, W.; Yu, X.; Wu, Y.; Xie, Z.; Li, Y. K.;
Huang, P.; Luo, F.; Ruan, C.; Sui, Z.; and Liang, W. 2024.
DeepSeekMoE: Towards Ultimate Expert Specialization in
Mixture-of-Experts Language Models. arXiv:2401.06066.
Du, N.; Huang, Y.; Dai, A. M.; Tong, S.; Lepikhin, D.; Xu,
Y.; Krikun, M.; Zhou, Y.; Yu, A. W.; Firat, O.; Zoph, B.;
Fedus, L.; Bosma, M.; Zhou, Z.; Wang, T.; Wang, Y. E.;
Webster, K.; Pellat, M.; Robinson, K.; Meier-Hellstern, K.;
Duke, T.; Dixon, L.; Zhang, K.; Le, Q. V.; Wu, Y.; Chen,
Z.; and Cui, C. 2022. GLaM: Efficient Scaling of Language
Models with Mixture-of-Experts. In Proceedings of the 39th
International Conference on Machine Learning.
Fang, Z.; Huang, Y.; Hong, Z.; Lyu, Y.; Chen, W.; Yu, Y.;
Yu, F.; and Zheng, Z. 2025a. Klotski: Efficient Mixture-
of-Expert Inference via Expert-Aware Multi-Batch Pipeline.
arXiv:2502.06888.
Fang, Z.; Yang, T.; Wang, Y.; Xu, J.; Liu, Y.; Chen, B.; Xing,
E.; Zhang, Z.; and Chen, Y. 2025b. FATE: Fast Edge In-
ference of Mixture-of-Experts Models via Cross-Layer Gate
Reuse. arXiv:2502.12224.
Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch Trans-
formers: Scaling to Trillion Parameter Models with Simple
and Efficient Sparsity. Journal of Machine Learning Re-
search, 23(120): 1–39.
Gavhane, N.; Mehrotra, A.; Chawla, R.; and Proenca, P. 2025.
MoE-Beyond: Learning-Based Expert Activation Prediction
on Edge Devices. arXiv:2508.17137.
He, X.; Zhang, S.; Tang, K.; Shi, S.; Wang, Y.; Zeng,
Z.; Tang, Z.; Chu, X.; Yin, H.; Tsang, I. W.; and Ong,
Y. S. 2024. ExpertFlow: Efficient Mixture-of-Experts Infer-
ence via Predictive Expert Caching and Token Scheduling.
arXiv:2410.17954.
Hwang, R.; Wei, J.; Cao, S.; Hwang, C.; Tang, X.; Cao,
T.; and Yang, M. 2024.
Pre-gated MoE: An Algorithm-
System Co-Design for Fast and Scalable Mixture-of-Expert
Inference. In Proceedings of the 51st Annual International
Symposium on Computer Architecture.
Jiang, A. Q.; Sablayrolles, A.; Roux, A.; Mensch, A.; Savary,
B.; Bamford, C.; Chaplot, D. S.; de las Casas, D.; Hanna,
E. B.; Bressand, F.; Lengyel, G.; Bour, G.; Lample, G.;
Lavaud, L. R.; Saulnier, L.; Lachaux, M.-A.; Stock, P.;
Le Scao, T.; Lavril, T.; Wang, T.; Lacroix, T.; and El Sayed,
W. 2024. Mixtral of Experts. arXiv:2401.04088.
Komatsuzaki, A.; Puigcerver, J.; Lee-Thorp, J.; Ruiz, C. R.;
Mustafa, B.; Ainslie, J.; Tay, Y.; Dehghani, M.; and Houlsby,
N. 2023.
Sparse Upcycling: Training Mixture-of-Experts
from Dense Checkpoints. arXiv:2212.05055.

Lepikhin, D.; Lee, H.; Xu, Y.; Chen, D.; Firat, O.; Huang, Y.;
Krikun, M.; Shazeer, N.; and Chen, Z. 2021. GShard: Scaling
Giant Models with Conditional Computation and Automatic
Sharding. In International Conference on Learning Repre-
sentations.
Leviathan, Y.; Kalman, M.; and Matias, Y. 2023. Fast Infer-
ence from Transformers via Speculative Decoding. In Pro-
ceedings of the 40th International Conference on Machine
Learning.
Li, Y.; Wei, F.; Zhang, C.; and Zhang, H. 2024. EAGLE:
Speculative Sampling Requires Rethinking Feature Uncer-
tainty. arXiv:2401.15077.
Qwen
Team.
2025a.
Qwen3
Technical
Report.
arXiv:2505.09388.
Qwen
Team.
2025b.
Qwen3-VL
Technical
Report.
arXiv:2511.21631.
Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.;
Hinton, G.; and Dean, J. 2017. Outrageously Large Neural
Networks: The Sparsely-Gated Mixture-of-Experts Layer. In
International Conference on Learning Representations.
Shen, L.; Wu, Z.; Gong, W.; Hao, H.; Bai, Y.; Wu, H.; Wu,
X.; Bian, J.; Xiong, H.; Yu, D.; and Ma, Y. 2022. SE-MoE: A
Scalable and Efficient Mixture-of-Experts Distributed Train-
ing and Inference System. arXiv:2205.10034.
Sun, Q.; andLi, Y. 2025. DS-MoE: Dynamic ExpertSchedul-
ing for Efficient MoE-based LLM Inference. In Proceedings
of the 5th International Conference on Intelligent Technology
and Embedded Systems, 90–95.
Tang, P.; Liu, J.; Hou, X.; Pu, Y.; Wang, J.; Heng, P.-A.; Li, C.;
and Guo, M. 2024. HOBBIT: A Mixed Precision Expert Of-
floading System for Fast MoE Inference. arXiv:2411.01433.
Wang, W.; Liu, J.; Hou, X.; Xia, X.; Tang, P.; Zhang, M.; Li,
C.; and Guo, M. 2025. MoE-SpeQ: Speculative Quantized
Decoding with Proactive Expert Prefetching and Offloading
for Mixture-of-Experts. arXiv:2511.14102.
Wu, Z.; Chen, X.; Pan, Z.; Liu, X.; Liu, W.; Dai, D.; Gao,
H.; Ma, Y.; Wu, C.; Wang, B.; et al. 2024. DeepSeek-VL2:
Mixture-of-Experts Vision-Language Models for Advanced
Multimodal Understanding. arXiv:2412.10302.
Xue, L.; Fu, Y.; Lu, Z.; Sun, C.; Mai, L.; and Marina, M. 2024.
MoE-Infinity: Efficient MoE Inference on Personal Machines
with Sparsity-Aware Expert Cache. arXiv:2401.14361.
Yan, J.; Liu, J.; Xu, H.; and Huang, L. 2025. Accelerat-
ing Mixture-of-Expert Inference with Adaptive Expert Split
Mechanism. arXiv:2509.08342.
Yu, E.; Dong, D.; Zhang, Z.; Bai, Z.; Yang, W.; Wang, H.;
Li, D.; Wu, Y.; and Liao, X. 2025a. LayerScope: Predic-
tive Cross-Layer Scheduling for Efficient Multi-Batch MoE
Inference on Legacy Servers. arXiv:2509.23638.
Yu, H.; Cui, X.; Zhang, H.; Wang, H.; and Wang, H.
2025b.
Taming Latency-Memory Trade-Off in MoE-
Based LLM Serving via Fine-Grained Expert Offloading.
arXiv:2502.05370.
Zhu, S.; Bohl, S.; Oester, R.; and Alonso, G. 2025. Pre-
Attention Expert Prediction and Prefetching for Mixture-of-
Experts Large Language Models. arXiv:2511.10676.


