# SOURCE: https://arxiv.org/pdf/2602.05711v2

OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale

Jingze Shi 1 Zhangyang Peng 1 Yizhang Zhu 1 Yifan Wu 1 Guang Liu 2 Yuyu Luo 1

Abstract
Mixture-of-Experts (MoE) architectures are
evolving towards finer granularity to improve pa-
rameter efficiency. However, existing MoE de-
signs face an inherent trade-off between the gran-
ularity of expert specialization and hardware ex-
ecution efficiency. We propose OmniMoE, a
system–algorithm co-designed framework that
pushes expert granularity to its logical extreme.
OmniMoE introduces vector-level Atomic Ex-
perts, enabling scalable routing and execution
within a single MoE layer, while retaining a
shared dense MLP branch for general-purpose
processing. While this atomic design maximizes
capacity, it poses severe challenges for routing
complexity and memory access. To address these,
OmniMoE adopts a system-algorithm co-design:
(i) a Cartesian Product Router that decomposes
the massive index space to reduce routing com-
plexity from O(N) to O(
√

N); and (ii) Expert-
Centric Scheduling that inverts the execution
order to turn scattered, memory-bound lookups
into efficient dense matrix operations. Validated
on seven benchmarks, OmniMoE (with 1.7B ac-
tive parameters) achieves 50.9% zero-shot ac-
curacy across seven benchmarks, outperforming
coarse-grained and fine-grained baselines. Cru-
cially, OmniMoE reduces inference latency from
73ms to 6.7ms (a 10.9× speedup) compared
to PEER, demonstrating that massive-scale fine-
grained MoE can be fast and accurate. Our code
is open-sourced at https://github.com/
HKUSTDial/omni-moe.

1. Introduction

Mixture-of-Experts (MoEs) has emerged as a key approach
to mitigating scaling bottlenecks by partially decoupling

1The Hong Kong University of Science and Technology
(Guangzhou) 2Beijing Academy of Artificial Intelligence. Corre-
spondence to: Yuyu Luo <yuyuluo@hkust-gz.edu.cn>.

Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).

FNN

A$n

Transformer 

Backbone

(a) Coarse-grained MoE
Performance:
Latency:

(b) Fine-grained MoE
Performance:
Latency:

Skip

(c) OmniMoE (Ours)
Performance:
Latency:

Expert
Ac;vated Expert
Non-ac;vated Expert
Block on Chip

Valid Ac;va;on
Redundant Ac;va;on
Inac;ve
Data Flow

Figure 1. Activation Patterns and System Optimization. (a)
Coarse-grained MoE activates large experts, inevitably involving
redundant parameters and wasting computation. (b) Fine-grained
MoE improves parameter efficiency, but suffers from bandwidth
bottlenecks due to scattered, fragmented memory accesses. (c) Our
OmniMoE employs a universally activated shared dense MLP, and
uses expert-centric scheduling to reorganize fine-grained expert
fetches into contiguous, coalesced memory accesses, achieving
both high parameter efficiency and hardware-efficient execution.

model capacity from per-token computation (Fedus et al.,
2022). By activating only a subset of experts for each token,
MoEs allow for massive parameter scaling while maintain-
ing manageable inference budgets. A central design choice
in MoE is the granularity of experts, which largely deter-
mines both routing precision and system efficiency. Broadly,
existing designs fall into two categories: coarse-grained
MoEs and fine-grained MoEs.

Coarse-Grained MoEs. Coarse-grained architectures rep-
resent the dominant paradigm in contemporary large-scale
language models. Representative systems (Du et al., 2022;
Jiang et al., 2024; Zoph et al., 2022) such as DeepSeek-
V3 (DeepSeek-AI et al., 2025) (256 experts) and KIMI-
K2 (Team et al., 2025) (384 experts) instantiate each expert
as a complete dense FFN, benefiting from hardware-efficient
dense matmuls (via Tensor Cores), contiguous VRAM ac-
cess, and shared-expert general knowledge and training
stability (Dai et al., 2024; Nguyen et al., 2025; DeepSeek-
AI et al., 2025; Team et al., 2025). Despite the success,

1

arXiv:2602.05711v2  [cs.CL]  1 Jul 2026


OmniMoE

coarse-grained MoEs inherently suffer from imprecise acti-
vation (Szatkowski et al., 2024) and low flexibility. Specif-
ically, activating large expert blocks incurs computation
on parameters irrelevant to specific tokens (orange nodes
in Figure 1 (a)), leading to computational waste (Cheng
et al., 2025; Li et al., 2023; Szatkowski et al., 2024). More-
over, their rigid size hinders adaptation to limited hardware:
coarse granularity restricts scaling flexibility, forcing steep,
discrete memory increments when adjusting expert counts.

Fine-grained MoEs. Fine-grained architectures seek to
maximize expressivity by utilizing millions of lightweight
experts (e.g., embeddings).
MoE scaling-law analy-
ses (Ludziejewski et al., 2024; Clark et al., 2022) suggest
that, under a fixed training-token budget, performance im-
proves with the total number of activated experts. This mo-
tivates fine-grained MoEs (He, 2024; Nogueira dos Santos
et al., 2024) that use extra lightweight experts. For example,
PEER (He, 2024) scales to millions of experts by adopting
a Product Key Memory (PKM (Lample et al., 2019)) style
design, enabling precise routing and fine-grained control
over both model capacity and activated parameters through
smooth scaling.

However, scaling fine-grained experts to massive magni-
tudes introduces three system challenges. (i) Limited ex-
pressivity: existing designs (e.g., PEER (He, 2024)) reduce
experts to static parameter vectors. This restricts the expert
computation to linear vector aggregation, stripping away
the token-dependent nonlinear transformations (e.g., MLP
projections) essential for modeling complex linguistic de-
pendencies. (ii) Routing overhead: scaling to a large expert
pool increases routing cost and load imbalance, resulting
in skewed expert utilization at scale. (iii) Hardware in-
efficiency: scattered activations trigger random memory
I/O, shifting execution from compute-bound to memory-
bound and degrading GPU utilization. As illustrated in
Figure 1(b), while fine-grained experts ensure precise acti-
vation, the active parameters are inherently scattered across
memory, which triggers frequent, non-contiguous memory
accesses, inevitably shifting the execution bottleneck from
computation to memory bandwidth.

While coarse-grained MoEs benefit from hardware-friendly
architecture, the fine-grained ones leverage high activation
efficiency and flexibility. This raises a key question: Is
it possible to reconcile the parameter efficiency of fine-
grained models with the hardware efficiency of coarse-
grained architectures? Realizing this synergy is non-trivial.
It requires a holistic orchestration that simultaneously en-
hances the expressivity of fine-grained experts, minimizing
routing overhead in large expert spaces, and reshaping irreg-
ular sparse accesses into hardware-efficient execution.

Our Methodology and Contributions. To address the
aforementioned challenges, we propose OmniMoE, a

system-algorithm co-designed MoE framework that syn-
ergizes the precise parameter activation of fine-grained ex-
perts with the hardware efficiency of coarse-grained designs.
The core architectural innovation lies in a hybrid parallel
design that combines a shared dense MLP for capturing
general semantic knowledge, with a massive pool of routed
fine-grained experts that specialize in long-tail knowledge
retrieval. To orchestrate the activation of these fine-grained
experts at scale, we introduce three tightly integrated contri-
butions that jointly resolve the key bottlenecks.

First, to maximize model capacity and routing precision, we
push expert granularity to its logical extreme by introducing
the Atomic Expert, a minimal routable unit parameterized
by a pair of vectors, and propose a corresponding Dynamic
Expert Assembly (DEA) mechanism to organize and com-
pose massive experts. This formulation enables scaling
to a massive expert pool and supports highly specialized,
token-specific, high-expressivity parameter compositions.

However, orchestrating large-scale atomic experts poses
an unprecedented routing challenge: standard approaches
would incur prohibitive routing overhead. To address this,
we introduce the Cartesian Product Router. It decomposes
the massive, 1D expert index space into a two-dimensional
grid. By factorizing the routing computation into two inde-
pendent, low-dimensional projections, it reduces the routing
cost from linear in N to proportional to
√

N, making large-
scale expert routing practical and efficient.

With routing no longer the bottleneck, the key challenge
in our orchestration shifts to hardware inefficiency: fine-
grained routing induces highly scattered atomic expert ac-
cesses, leading to poor locality and low GPU efficiency. To
overcome this final obstacle, we developed Expert-Centric
Scheduling. This systemic contribution inverts the exe-
cution paradigm from token-centric to expert-centric. By
reordering computations, it groups requests targeting the
same experts, thereby converting scattered memory lookups
into contiguous, reusable reads and enabling the use of high-
throughput Grouped GEMM operations.

In summary, OmniMoE is a system-algorithm co-designed
framework that orchestrates fine-grained expert activation,
from atomic expert formulation to efficient routing and
hardware-aware scheduling, achieving both high model
expressivity and hardware efficiency at scale. As illus-
trated in Figure 1, our heterogeneous architecture elim-
inates redundant computation inherent to coarse-grained
MoEs, while our scheduling strategy resolves bandwidth
bottlenecks caused by scattered, non-contiguous memory
accesses under fine-grained routing. Extensive experiments
demonstrate that OmniMoE achieves superior performance
with 10.9× speedup compared to state-of-the-art baselines.
Our code is open-sourced at https://github.com/
HKUSTDial/omni-moe.

2


OmniMoE

Score
Top-K

C
Router

R
Router

σ

MLP

Longtail Knowledge Retrieval

General Seman4cs Handling

Inputs

Outputs

(a) Dynamic Expert Assembly

(b) Shared Expert

: Rou7ng indices 
  for top-K experts;

: Global parameter 
  matrix;

,

: Compact assemble 
  parameter matrix.

,

: Rou7ng scores;

Figure 2. Overview of the OmniMoE Architecture. The framework operates via two parallel pathways to balance efficiency and
expressivity. (a) Dynamic Expert Assembly (Top): For Longtail Knowledge Retrieval objective, we employ a Cartesian Product
Router (decomposed into Row/Column routers) to efficiently compute routing scores gx and identify the top-K expert indices Ix. Then
the system dynamically retrieves specific parameter slices from the global matrices W, V to assemble compact, token-dependent parameter
blocks wx, vx for the final gated projection. (b) Shared Expert (Bottom): A dense MLP which is always active to handling General
Semantics. The final output is obtained by aggregating the outputs from the sparse, routed branch and the shared dense branch.

2. Methodology

We first formalize the general MoE architecture. A stan-
dard MoE layer comprises a pool of N experts E =
{E1, . . . , EN} and a routing function G(·) : Rd →RN.
Several MoE variants (Dai et al., 2024; DeepSeek-AI et al.,
2025; Team, 2025) incorporate a shared dense MLP that
remains universally activated for all inputs. For each input
token with representation x ∈Rd, where d is the dimension
of hidden states, the router computes routing scores and
selects a subset of K experts identified by indices Ix.

Ix = (I0, . . . , IK−1) = TopK(G(x), K)
(1)

where Ii is the index of the i-th activated expert. And the
routing weight gi for each expert EIi could be calculated by

gi = Softmax(G(x)[Ix])i, i ∈[0, K)
(2)

Finally, the layer output is the weighted sum of these acti-
vated experts:

y =
�

i∈[0,K)
gi · EIi(x) + MLP(x)
(3)

OmniMoE Overview. Figure 2 illustrates the overall archi-
tecture of OmniMoE. Our design follows the standard MoE
formulation in Eq. 3. Furthermore, OmniMoE instantiates
the activated experts as Atomic Experts (Section 2.1) and
uses our Dynamic Expert Assembly (DEA) mechanism to
retrieve and assemble token-conditioned parameters on the
fly, enabling the routed branch to operate at a much finer
granularity than conventional FFN experts. In parallel, we
retain a dense MLP as a shared expert to provide general
semantic reasoning and stable capacity that is independent
of routing. The final representation is obtained by summing
the shared dense branch and the routed fine-grained branch.
The remainder of this section introduces (i) how we param-
eterize and store atomic experts efficiently (Section 2.1), (ii)

how we route over massive expert spaces (Section 2.2), and
(iii) how we schedule the resulting sparse computations to
maximize hardware efficiency (Section 2.3).

2.1. Atomic Experts and Dynamic Expert Assembly

In this section, we introduce the core components of our fine-
grained expert design. We first define the atomic expert as
the minimal routable computational unit. We then present
Dynamic Expert Assembly (DEA), our proposed mech-
anism for logically organizing these atomic units. Specif-
ically, DEA enables the model to dynamically retrieve a
sparse set of atomic experts from a global pool and compose
them into a token-conditioned assembled expert.

An atomic expert Ei is defined as a minimal, lightweight
computational unit parameterized by an input vector win
i
∈
Rd and an output vector wout
i
∈Rd. Given a token repre-
sentation x ∈Rd, its computation is:

Ei(x) = σ(xwin
i
⊤)wout
i
(4)

where σ(·) is a non-linear activation. Throughout Omni-
MoE, we instantiate σ(·) with SWIGLU (Shazeer, 2020).
While a single atomic expert exhibits limited expressivity,
the strength of our approach arises from the dynamic com-
position of these experts.

The Dynamic Expert Assembly (DEA) mechanism gov-
erns this composition process. For each input token, DEA
consists of two steps: (i) Retrieval, where it selects a sparse
subset of the most relevant atomic experts from massive
global experts, and (ii) Assembly, which composes the re-
trieved parameters into a computational block.

To make this DEA computationally feasible at scale, the pa-
rameters of all N atomic experts are not stored individually.
Instead, they are consolidated into two global parameter
matrices, W, V ∈RN×d, which serve as a centralized pa-

3


OmniMoE

rameter repository for efficient retrieval and composition.

W = [win
0 , . . . , win
N−1]⊤,
V = [wout
0
, . . . , wout
N−1]⊤

(5)
For a given input token x, the routing mechanism identifies
Ix ⊂{0, ..., N −1}, the indices of the top-K most relevant
atomic experts, similar to Eq. 1. The Retrieval step of DEA
is implemented by gathering the rows indexed by Ix from
the global parameter matrices, yielding compact, token-local
parameter blocks:

wx = W[Ix] ∈RK×d,
vx = V [Ix] ∈RK×d
(6)

Simultaneously, the associated routing scores are collected
into a vector gx = [gI0, . . . , gIK−1] ∈RK. The Assembly
step is then performed by composing these retrieved param-
eters and their corresponding weights into a single, fused
computation:

y = (gx ⊙σ(xw⊤
x ))vx + MLP(x).
(7)

This formulation demonstrates how DEA effectively con-
structs a unique, powerful assembled expert for each to-
ken by composing simple, reusable atomic experts. This
approach ensures that every retrieved parameter is com-
putationally active for the target token, achieving extreme
parameter efficiency while maintaining high expressivity.

2.2. Cartesian Product Router

The primary role of the router is to efficiently select a sparse
set of atomic expert indices I for the DEA mechanism.
However, scaling to massive expert pools presents a se-
vere indexing challenge. A standard top-K router com-
putes routing scores for all experts via a projection matrix
Wg ∈Rd×N and set G(x) = xWg in Eq. 2. When N
reaches millions, the computational cost of G(x) (O(Nd))
and the memory required to store Wg becomes prohibitively
expensive and often dominates the total inference latency.

Intuition. The key insight is that the one-dimensional expert
index space of size N can be decomposed into the Cartesian
product of two lower-dimensional subspaces. Therefore,
to overcome this bottleneck, we introduce the Cartesian
Product Router. Rather than scoring all N experts with
a single N-way classifier, we view an expert id n as a 2D
coordinate (i, j) on a Nr ×Nc grid (with N = NrNc). The
router predicts two low-dimensional distributions over rows
and columns, and composes them to score any expert on
the grid. This is analogous in spirit to product-structured
indexing (e.g., PKM (Lample et al., 2019)): We replace one
prohibitively large projection with two small projections,
while still addressing the full N-sized expert space.

Implicit Scoring via Factorized Projections. Our model-
ing assumption is that the joint probability distribution over

the expert grid can be approximated by the product of two
independent marginal distributions:

p(i, j|x) ≈pr(i|x) · pc(j|x).
(8)

We replace the single routing matrix Wg with two smaller
matrices, Wr ∈Rd×Nr and Wc ∈Rd×Nc. For an input
token x, the row and column logits are computed as

sr = xWr , sc = xWc.
(9)

Then the log-probabilities for each subspace are obtained
via the LogSoftmax function for numerical stability:

pr = LogSoftmax(sr) , pc = LogSoftmax(sc).
(10)

Since pr and pc are log-probabilities (via LogSoftmax),
the factorized product becomes additive in log-space:
(log p(i, j | x) ≈pr[i] + pc[j]). The score for an ex-
pert at coordinate (i, j) is the sum of the corresponding
log-probabilities, which implicitly defines a score matrix
S ∈RNr×Nc without its materialization:

Sij = pr[i] + pc[j]
(11)

Parallel Top-K Selection. Although S ∈RNr×Nc is never
materialized, its entries can be computed on-the-fly as de-
scribed in Eq. 11 from the global vectors pr and pc. We
therefore partition the implicit grid into tiles and assign each
tile to parallel GPU thread blocks. Each block computes
scores for its tile and extracts local top-K candidates. These
candidates are then merged via a lightweight reduction to
obtain the global top-K expert indices Ix. The routing
weights gx are computed by normalizing the corresponding
top-K scores according to Eq. 2.

Complexity Analysis. The factorized router reduces the
projection cost (and router parameter size) from O(Nd) to
O(
√

Nd). Top-K selection is performed on the implicit
grid via a tiled GPU search and reduction; although the total
score-evaluation work still scales with N, it is highly par-
allel and incurs negligible wall-clock overhead in practice.
See Appendix B for the full derivation and details.

2.3. Expert-Centric Scheduling

The Cartesian Product Router (Section 2.2) operates on a
per-token basis: for each input token x, it efficiently selects
the top-K expert indices Ix and computes gating weights
gx. In practice, however, execution processes a batch of to-
kens X = {xl}L−1
l=0 . While routing is efficient, token-centric
execution becomes a bottleneck at batch scale: each token
independently fetches its selected expert parameters from
HBM via scattered accesses, fragmenting memory traffic
and preventing vectorization, thus limiting throughput.

To address this, we propose Expert-Centric Scheduling.
Unlike standard static approaches that iterate over all ex-
perts, our method dynamically organizes computation based

4


OmniMoE

More Load

7
6
5
4
3
2
1
0

0 1 2 3 4 5 6 7 Expert ID SM on Chip

Token ID

7
6
3
2
5
4
1
0

0 1 2 3 4 5 6 7

7
6
5
4
3
2
1
0

0 1 2 3 4 5 6 7

Token ID

Expert ID SM on Chip

Re-ordering

Token ID

More OccupaDon
Less Load
Less OccupaDon
(a) Conventional
(b) Our Approach

Group 1
Group 2
Group 1
Group 2
Group 1
Group 2

Figure 3. Comparison of Execution Paradigms: Token-Centric vs. Expert-Centric Scheduling. (a) Conventional: Tokens
independently fetch parameters from scattered experts, leading to random memory accesses (high load overhead) and fragmented
vector-vector computations that underutilize on-chip SMs. (b) Our Approach: We invert the execution order using expert-centric
scheduling. Left-to-Right: First, tasks are reordered: we compress active experts into dense groups (e.g., experts 0–3 are grouped into
Group 1) and sort tasks by Token ID within each group. Matrix Fusion: This reorganization allows us to merge individual token-expert
pairs into dense tensors. Instead of scattered ops, the GPU executes efficient Grouped GEMM kernels (rightmost block), where a block
of expert weights is loaded once and reused across stacked tokens, maximizing Tensor Core utilization and memory bandwidth.

on active experts. The pipeline proceeds as follows: we first
collect routed computation tasks, group physically nearby
experts, then reorder tasks according to these groups to im-
prove locality, and finally execute the resulting workloads
using high-throughput GEMM kernels.

Task Collection and Active Expert Compression. For a
batch of L tokens X = {xl}L−1
l=0 with top-K routing, the
router returns, for each token l, an ordered expert index
list Il = (Il,0, . . . , Il,K−1) and the corresponding gating
weights gl = (gl,0, . . . , gl,K−1), where gl,k denotes the
routing weight assigned to expert Il,k. We first flatten these
decisions into a list of M = L × K tasks:

˜T =
�
(xl, Il,k, gl,k)
��l ∈[0, L), k ∈[0, K)
�
(12)

and collect the set of unique active experts Eactive =
�

x∈X Ix, sorted by global expert ID. We then partition
this ordered list into contiguous groups of size B, so
that experts with nearby IDs are placed in the same group.
Specifically, the τ-th expert in Eactive is assigned to group
qτ = ⌊τ/B⌋.
Consequently, the number of execution
groups is determined solely by the active sparsity, i.e.,
Ngroups = ⌈|Eactive|/B⌉, ensuring that every compute
group (except the last) is fully populated.

Hierarchical Sorting. We reorganize the tasks ˜T by per-
forming a hierarchical sort. The primary key is the Group
ID q, and the secondary key is the Token ID l.

T = Sort( ˜T , keys = (q, l))
(13)

This sorting strategy improves hardware efficiency at two
levels: (i) Inter-Group Locality: tasks targeting the same
set of B active experts are clustered together; (ii) Intra-
Group Coalescing: within each group, processing tasks in
increasing order of token ID l improves coalescing for input
reads and output scatters.

Grouped GEMM Execution. After hierarchically sorting
the tasks, we process each of the Ngroups active groups. For
a given group q, we first gather the corresponding expert
parameters into dense blocks Wq, Vq ∈RB×d. Concur-
rently, we stack the input tokens and gating weights for all
tasks assigned to this group, forming a dense input tensor
Xq ∈RTq×d and a gating vector Gq ∈RTq×B, where Tq
denotes the number of tasks in group q. The entire compu-
tation is then performed by a single fused operation:

Oq = (Gq ⊙σ(XqW ⊤
q ))Vq
(14)

where Oq ∈RTq×d is the output block. The per-task out-
puts are subsequently written back via scatter-add, preserv-
ing the semantics of Eq. 7.

Why Expert-Centric Scheduling is Efficient. As illus-
trated in Figure 3, unlike the token-centric paradigm, our
expert-centric approach clusters spatially proximate experts
into contiguous groups, reorders tasks by token ID within
each group, and fuses the resulting workloads into a small
number of high-throughput Grouped GEMM kernels. By
executing tasks in expert-centric order and sorting by token
ID within each group, we increase parameter reuse and im-
prove memory locality, which raises effective bandwidth
and enables high-throughput Grouped GEMM execution. A
detailed complexity analysis is deferred to Appendix A.

3. Experiments

3.1. Experimental Setup

Model Architectures. We compare six FFN variants: (i)
Dense (standard MLP), (ii) Gshard (Lepikhin et al., 2021),
(iii) DeepSeekMoE (Dai et al., 2024), (iv) PKM (Lam-
ple et al., 2019), (v) PEER (He, 2024), and (vi) Omni-
MoE (ours). All models adopt Grouped Query Attention
(GQA) (Ainslie et al., 2023). For fair comparison, we keep

5


OmniMoE

the Transformer backbone identical across methods (depth,
width, and attention configuration) and vary only the FFN
module. It is important to emphasize that we prioritize ar-
chitectural comparison via controlled pre-training from
scratch rather than comparing against off-the-shelf check-
points. We define the activated-parameter budget as the
number of unique parameters utilized in the forward pass of
a single token, including embeddings, attention weights, the
shared dense FFN, router projections, and the top-K active
MoE experts. For efficiency baselines, we ensure state-of-
the-art implementations (see Appendix B for details).

We evaluate OmniMoE under three complementary settings.
For Speed and Memory Benchmarking, we fix a 200M
backbone and sweep the activated-parameter budget (Act
Params) and the number of activated tokens (Act Tokens)
as summarized in Table A (see Appendix B). For scaling-
law experiments, we train MoE families with 280M-A80M,
800M-A200M, 2.7B-A680M, and 6.4B-A1.7B configu-
rations (where A denotes the activated parameter budget)
alongside their Dense counterparts with matched activated
parameters, using matched backbones and training recipes
across baselines (Table B in Appendix B). For downstream
evaluation, we report zero-shot results using the 6.4B-
A1.7B models.

Training Data and Tokenization. Models are pre-trained
on the SmolLMCorpus (Allal et al., 2025), a high-quality
corpus of 40 billion tokens spanning Web, Textbook, Code,
and Math domains. This diverse mixture establishes funda-
mental linguistic proficiency and broad general knowledge.
We employ the NeoX tokenizer (Black et al., 2022) with a
vocabulary size of 128,256 tokens.

Training Strategy and Hyper-Parameters. We use the
AdamW optimizer (Loshchilov & Hutter, 2017) with the
WSD learning rate scheduler (H¨agele et al., 2024). Hy-
perparameters follow optimal scaling laws (Li et al., 2025)
and Chinchilla compute-optimality protocols (Hoffmann
et al., 2022). We run experiments in the NVIDIA PyTorch
container (NVIDIA, 2022) with Hugging Face Transform-
ers (Wolf et al., 2020). All inference/evaluation runs use a
single node with 8× NVIDIA A100 GPUs.

Evaluation Benchmarks. We evaluate downstream and
reasoning performance with Hugging Face LightEval (Four-
rier et al., 2023) on seven commonsense benchmarks:
MMLU (Hendrycks et al., 2021a) (multitask knowledge),
TriviaQA (Joshi et al., 2017) (factual recall), ARC (Clark
et al., 2018) (science reasoning), PIQA (Bisk et al.,
2020) (physical commonsense), HellaSwag (Zellers et al.,
2019) (commonsense inference), OBQA (Mihaylov et al.,
2018) (open-book QA), and Winogrande (Sakaguchi et al.,
2019) (coreference resolution); and five extended bench-
marks: GSM8K (Cobbe et al., 2021) (math reasoning),
MATH (Hendrycks et al., 2021b) (competition math),

BBH (Suzgun et al., 2022) (hard reasoning), MBPP (Austin
et al., 2021) (code generation), and HumanEval (Chen et al.,
2021) (code synthesis).

3.2. Main Results

Main Results. We summarize the main empirical findings
of OmniMoE from two complementary perspectives: model
quality under a fixed active-parameter budget, and system
efficiency (latency/memory) when executing the correspond-
ing routed feed-forward computation.

Downstream Performance. As shown in Table 1, our 6.4B-
A1.7B model achieves the best average zero-shot accuracy
(50.9), outperforming both coarse-grained (e.g., +0.7 vs.
DeepSeekMoE) and fine-grained (+2.0 vs. PEER) baselines.
The results highlight the benefits of our heterogeneous de-
sign: compared to coarse-grained models, OmniMoE’s pre-
cision on knowledge-intensive tasks like TriviaQA (+1.1)
and OBQA (+1.4) is superior. Conversely, compared to
fine-grained models with limited expressivity, OmniMoE’s
shared dense expert boosts performance on reasoning-heavy
benchmarks such as ARC (+3.6) and HellaSwag (+4.6).

Reasoning Performance and Throughput. To further
validate generalization beyond commonsense tasks, we re-
port results on reasoning benchmarks alongside end-to-end
throughput. As shown in Table 2, OmniMoE achieves the
highest throughput among all MoE variants (∼14.2k tok/s),
comparable to the Dense baseline (∼14.8k tok/s) and signifi-
cantly outperforming fine-grained baselines (PEER: ∼11.6k
tok/s, PKM: ∼7.9k tok/s). This confirms that Expert-Centric
Scheduling effectively eliminates the memory bandwidth
bottleneck at the full-model level. On the extended bench-
marks, OmniMoE achieves an average score of 42.5, outper-
forming DeepSeekMoE (40.5) by +2.0 and PEER (27.0) by
+15.5. The large gap over PEER indicates that purely fine-
grained designs without a shared dense backbone struggle
with sustained multi-step logic, while OmniMoE’s hybrid ar-
chitecture effectively combines precise knowledge retrieval
with stable reasoning capacity.

End-to-End Efficiency and Scalability. Figure 4 demon-
strates that OmniMoE is substantially more efficient. De-
spite activating a comparable or even larger number of pa-
rameters (OmniMoE: 28M, PEER: 26M, DeepSeekMoE:
28M), OmniMoE achieves substantially lower latency, re-
ducing inference time from 73 ms (PEER) and 102 ms
(DeepSeekMoE) to 6.7 ms at 4,096 tokens, 10.9× and
15.2× speedup respectively, while maintaining a memory
footprint comparable to coarse-grained MoEs. This gain
stems directly from Expert-Centric Scheduling, which trans-
forms scattered memory accesses into coalesced, reusable
reads, thus shifting the execution from memory-bound to
compute-bound.

6


OmniMoE

Table 1. Performance on Downstream Benchmarks for 6.4B-A1.7B MoE Models and the 1.7B Dense Baseline.. The best results for
each size are in bold, and the second-best results are underlined. For the pre-trained base model, OmniMoE performs well on most tasks,
demonstrating its effectiveness.

MODEL
MMLU
TRIVIAQA
ARC
PIQA
HELLASWAG
OBQA
WINOGRANDE
AVG.
ACC ↑
ACC ↑
ACC ↑
ACC ↑
ACC ↑
ACC ↑
ACC ↑
AVG ↑

Dense
35.4
9.4
53.4
72.9
56.1
37.0
57.3
45.9
Gshard
36.7
16.7
58.3
75.3
59.3
38.7
59.5
49.2
DeepSeekMoE
37.1
17.4
60.7
77.2
61.2
38.9
59.1
50.2
PKM
36.3
12.2
53.6
73.8
52.7
38.2
56.7
46.2
PEER
37.4
16.9
57.4
75.9
56.3
39.1
59.4
48.9
OmniMoE (ours)
37.5
18.5
61.0
78.7
60.9
40.3
59.7
50.9

Table 2. Performance and Throughput on Reasoning Benchmarks for 6.4B-A1.7B MoE Models and the 1.7B Dense Baseline.
Throughput is measured with 40 batched parallel requests (∼512 input tokens, ∼8192 output tokens) on 8× A100 GPUs. The best results
are in bold, and the second-best are underlined. For the fine-tuned models, OmniMoE performs well on most tasks and achieves the
highest throughput.

MODEL
THROUGHPUT
MMLU
BBH
GSM8K
MATH
ARC-C
MBPP
HUMANEVAL
AVG.
(tok/s)
ACC ↑
ACC ↑
ACC ↑
ACC ↑
ACC ↑
ACC ↑
ACC ↑
↑

Dense
∼14.8k
46.4
37.7
46.3
11.7
37.4
40.0
6.7
32.3
Gshard
∼14.0k
48.2
41.5
61.9
18.9
44.1
47.4
8.5
38.6
DeepSeekMoE
∼13.4k
49.1
43.0
65.6
21.3
45.6
49.9
9.1
40.5
PKM
∼7.9k
36.3
24.8
18.9
4.1
30.2
17.6
1.8
19.1
PEER
∼11.6k
43.8
35.6
31.4
7.9
35.9
29.7
4.9
27.0
OmniMoE (ours)
∼14.2k
50.2
44.6
69.8
24.1
46.8
52.3
9.8
42.5

0

25

50

75

100

Latency (ms)

(a)
(b)

4
8
16
32
Activated Param (M)

2000

2500

3000

Memory (MiB)

(c)

2
4
8
16
Number of Tokens (K)

(d)

GShared
DeepSeekMoE

PKM
PEER

OmniMoE (w/o ECS)
OmniMoE (Ours)

Figure 4. End-to-End Efficiency Comparison. (a, b) Infer-
ence latency and (c, d) peak memory versus activated parameters
(left column) and input token count (right column). Baselines
include Dense, Gshard, DeepSeekMoE, PKM, and PEER. Omni-
MoE achieves consistently lower latency than DeepSeekMoE and
fine-grained baselines (PKM/PEER), while maintaining a peak
memory footprint comparable to coarse-grained MoEs.

Interestingly, although DeepSeekMoE uses coarse-grained
FFN experts, it can be slower than fine-grained PEER
at large token counts or activated budgets.
This is
largely due to packing/alignment overhead in tiled coarse-
grained kernels, where routed tokens must be reordered and

1017
1019
FLOPs

5 × 100

101

2 × 101

3 × 101

Perplexity

(a)

80M
200M 500M 1B1.6B
Activated Param

(b)

Dense
GShared

DeepSeekMoE
PKM

PEER
OmniMoE (Ours)

Figure 5. Scaling Laws. Validation perplexity (lower is better)
versus (a) training FLOPs and (b) activated parameters. OmniMoE
consistently outperforms all baselines, achieving the best trade-off
between model quality and computational cost.

padded to fixed block sizes, causing redundant computa-
tion and extra memory traffic. In contrast, OmniMoE re-
shapes fine-grained activations into compact, expert-centric
batched matrix operations. We report strict end-to-end la-
tency for all methods, including all scheduling/reordering
overheads for OmniMoE and the corresponding layout-
transformation/alignment costs for baselines, confirming
net gains from improved hardware utilization. Furthermore,
we verify the scalability of OmniMoE in distributed training

7


OmniMoE

settings (Appendix C). We observe that the communication
overhead saturates once the expert pool size exceeds the
active token count, demonstrating that OmniMoE can scale
to millions of experts with constant communication cost.

Scaling laws (perplexity vs. compute/activated parame-
ters). Figure 5 compares the scaling behavior of different
FFN variants. Under matched training FLOPs, OmniMoE
consistently achieves the lowest perplexity among all base-
lines, indicating superior compute efficiency. Moreover,
when controlling for the activated parameter budget, Omni-
MoE also attains the lowest perplexity, demonstrating higher
parameter efficiency. As the activated capacity increases,
OmniMoE benefits more steadily from additional experts,
reflecting the complementary roles of fine-grained activation
for long-tail knowledge and the shared dense MLP branch
for stable general reasoning.

3.3. Ablation Studies

We ablate the three core components of OmniMoE to iso-
late their individual contributions.
Table 3 isolates the
impact of three core components in OmniMoE: (i) the
Shared Dense MLP for general stability, (ii) the Carte-
sian Product Router for routing quality, and (iii) Expert-
Centric Scheduling for system efficiency. All metrics
are normalized to the full model (lower is better for La-
tency/Memory/PPL/Unevenness). To characterize expert
utilization, we follow PKM (Lample et al., 2019) and
PEER (He, 2024) and report two distribution metrics based
on the normalized expert retrieval frequency z ∈RN:

• Expert Usage: The fraction of experts activated at
least once, defined as 1

N |{i | zi > 0}|.

• Unevenness: The KL divergence from a uniform dis-
tribution, computed as DKL(z∥U) = �
i zi log(Nzi),
where lower values indicate more balanced load.

Effect of the shared dense MLP. Removing the shared
dense MLP slightly improves efficiency (0.86× latency,
0.98× memory) but hurts both perplexity (1.2×) and down-
stream performance (0.91× knowledge and 0.79× reason-
ing).This suggests that the shared dense branch serves as
a critical foundational backbone complementary to fine-
grained retrieval. It handles common linguistic patterns
and reasoning steps, allowing the routed branch to focus
exclusively on fetching token-specific long-tail knowledge.

Effect of the Cartesian Product Router. Replacing the
Cartesian Product Router with a standard dense routing
projection leads to a dramatic efficiency regression (30.6×
latency and 337.5× memory), driven by the cost of com-
puting and storing full-dimension logits. Crucially, it also
degrades model quality (1.4× PPL). Notably, expert usage
collapses to only 4% and unevenness increases from 0.24 to

The negative coefficient on education should not be

interpreted causally because it likely reflects omitted

variable bias or multicollinearity rather than a true negative

effect. The high variance inflation factors indicate severe

multicollinearity, which distorts coefficient estimates and

makes causal claims unreliable.

Omitted variable bias arises when unobserved factors

correlated with both education and wages are excluded,

potentially reversing the sign of the coefficient.

Multicollinearity, evident here, inflates standard errors and

can cause coefficient sign reversals, even when the true

effect is positive. Suppression effects occur when adding

variables reveals a negative relationship masked by

confounding, but multicollinearity complicates this

interpretation.

A high \( R^2 \) indicates the model fits the data well for

prediction but does not imply causality, as it captures

correlations rather than causal mechanisms. The model's

improved fit simply means the added variables explain

additional variance in wages, not that the negative education

coefficient is valid.

The most defensible interpretation is that the negative

coefficient is a statistical artifact due to multicollinearity

and omitted variables, not evidence of a true negative causal

effect. Researchers should address multicollinearity and

consider alternative models or data to isolate education's

effect more reliably.

Shared MLP
Mixed
Routed MoE

Branch dominance

Figure 6. Token-Level Branch Dominance Analysis. Each token
is colored by which branch contributes more to the final prediction:
the Shared MLP (red) or the Routed MoE (green). Factual entities
and rare tokens are predominantly served by the routed atomic ex-
perts, while common linguistic patterns and reasoning connectives
are handled by the shared dense MLP, empirically validating the
functional specialization of the two branches.

0.77. Despite rigorous tuning of the standard auxiliary load-
balancing loss for this baseline during training, the naive
gate fails to learn distinct specializations over the massive
expert space, collapsing into a few dominant experts.

Effect of Expert-Centric Scheduling. Reverting Expert-
Centric Scheduling to the standard token-centric execution
mechanism preserves quality (metrics remain at 1.0×) but
incurs a massive system cost (24.8× latency and 417.7×
memory). The peak memory increase corresponds to the
materialization of full routing tensors required by the stan-
dard baseline, which our scheduling avoids. This confirms
that our scheduling strategy is the primary source of ac-
celeration: by inverting the loop order to process experts
sequentially, we transform strictly random HBM accesses
into streaming reads and maximize on-chip tensor reuse,
eliminating the memory bandwidth bottleneck inherent to
fine-grained MoEs.

Branch Specialization Analysis. To directly validate the

8


OmniMoE

Table 3. Ablation Study. All metrics are reported relative to the full model. Lower is better for Latency, Memory, PPL, and Unevenness;
higher is better for Knowledge Performance, Reasoning Performance, and Expert Usage.

METHODS
LATENCY
MEMORY
PPL
KNOWLEDGE PERF.
REASONING PERF.
EXPERT USAGE
UNEVENNESS
↓
↓
↓
↑
↑
↑
↓

Full
1.0×
1.0×
1.0×
1.0×
1.0×
100%
0.24
w/o Shared Dense MLP
0.86×
0.98×
1.2×
0.91×
0.79×
100%
0.27
w/o Cartesian Product Router
30.6×
337.5×
1.4×
0.66×
0.79×
4%
0.77
w/o Expert-Centric Scheduling
24.8×
417.7×
1.0×
1.0×
1.0×
100%
0.24

division of labor between the two branches, we visualize
per-token branch dominance in Figure 6. For each token, we
measure the relative contribution of the shared dense MLP
versus the routed atomic experts to the final hidden state.
The results reveal a clear functional specialization: factual
entities and domain-specific terms are predominantly served
by the routed branch, while common function words and
reasoning connectives rely on the shared MLP. This pattern
is consistent with the ablation findings in Table 3, where re-
moving the shared MLP disproportionately hurts reasoning
performance while preserving expert usage, confirming that
the two branches fulfill complementary roles.

4. Related Work

MoE Architectures. Conditional computation via Mixture-
of-Experts (MoE) enables scaling model capacity with
bounded per-token cost (Shazeer et al., 2017; Sun et al.,
2025; Mu & Lin, 2025; Liu et al., 2025). Early MoE Trans-
formers predominantly adopt coarse-grained FFN experts
with lightweight routing, exemplified by Switch Transform-
ers’ Top-1 gating (Fedus et al., 2022) and GShard (Lep-
ikhin et al., 2021). Motivated by scaling-law evidence for
improved specialization, recent work trends toward finer-
grained expert designs (Ludziejewski et al., 2024; Tian et al.,
2025); modern LLMs (e.g., DeepSeek-V3 (DeepSeek-AI
et al., 2025), KIMI-K2 (Team et al., 2025)) scale to hun-
dreds of experts and often include shared experts to stabilize
general knowledge (Dai et al., 2024). Pushing granularity
to the extreme, PKM (Lample et al., 2019) and PEER (He,
2024) replace FFNs with million-scale tiny experts (e.g.,
embeddings), improving routing precision but reducing
per-expert expressivity. Additionally, prior work reports
redundant activation in FFNs/MoEs (Li et al., 2023; Sza-
tkowski et al., 2024; Zhou et al., 2025; Yang et al., 2024);
MoNE (Cheng et al., 2025) prunes computation within acti-
vated coarse-grained experts but retains expert-level top-K
routing, whereas OmniMoE routes over atomic experts to
enable finer-grained control of activated parameters.

Efficient MoE Systems. System optimizations for MoE
generally focus on kernel fusion and communication
scheduling for coarse-grained experts. Frameworks such as
DeepSpeed-MoE (Rajbhandari et al., 2022), Fast-MoE (He
et al., 2021), and MegaBlocks (Gale et al., 2023) opti-

mize GEMM kernels and handle variable-length sequences
to mitigate padding overheads in coarse-grained MoEs,
whereas our Expert-Centric Scheduling targets fine-grained
atomic experts, converting scattered memory accesses into
contiguous batched operations. Recent work like Sonic-
MoE (Guo et al., 2025) improves efficiency with memory-
efficient algorithms, minimal activation caching, and tile-
aware token rounding to reduce padding waste in Grouped
GEMM kernels. Other works, including PIT (Zheng et al.,
2023) and ScatterMoE (Tan et al., 2024), further exploit
dynamic sparsity to prune invalid computations within acti-
vated coarse-grained experts but retain expert-level top-K
routing. In contrast, OmniMoE introduces Expert-Centric
Scheduling to transform scattered memory accesses into
hardware-efficient batched operations. Concurrent works
Klotski (Fang et al., 2025) and ExpertFlow (He et al., 2026)
also batch tokens per expert but target inter-request schedul-
ing and cross-device load balancing for deployed coarse-
grained MoEs; our scheduling instead operates intra-batch
on a single device, addressing the scattered HBM reads
unique to million-scale atomic experts.

5. Conclusion

In this paper, we presented OmniMoE, a system-algorithm
co-designed MoE framework that integrates a shared dense
MLP for general-purpose reasoning with massive atomic
experts for long-tail knowledge retrieval, thereby enabling
more precise parameter activation. To make large-scale
expert activation practical, OmniMoE orchestrates the ac-
tivation of atomic experts via two key innovations: the
Cartesian Product Router and Expert-Centric Schedul-
ing. Together, these components yield a dramatic 10.9×
inference speedup over the state-of-the-art fine-grained
baseline, PEER. Moreover, under comparable activated-
parameter budgets, OmniMoE consistently improves av-
erage accuracy and outperforms strong baselines on most
benchmarks. These results demonstrate that, with holis-
tic co-design, massive-scale fine-grained MoEs can be both
accurate and highly efficient. Our current implementation re-
lies on custom Triton kernels optimized for NVIDIA GPUs;
portability to other hardware backends remains unexplored.
Additionally, scaling behavior beyond 6.4B total parameters
has not yet been validated empirically and is left for future
work.

9


OmniMoE

Acknowledgements

This paper was supported by the NSF of China (62402409);
Youth S&T Talent Support Programme of Guangdong
Provincial Association for Science and Technology
(SKXRC2025461); the Young Talent Support Project
of Guangzhou Association for Science and Technology
(QT-2025-001); Guangzhou Basic and Applied Basic Re-
search Foundation (2026A1515010269, 2025A04J3935,
2023A1515110545); and Guangzhou-HKUST(GZ) Joint
Funding Program (2025A03J3714). We gratefully acknowl-
edge the FlagOS open-source community and the OpenSeek
project for providing the computational resources essential
to this work.

Impact Statement

This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.

References

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y.,
Lebr´on, F., and Sanghai, S. GQA: training generalized
multi-query transformer models from multi-head check-
points. In Bouamor, H., Pino, J., and Bali, K. (eds.), Pro-
ceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing, EMNLP 2023, Singa-
pore, December 6-10, 2023, pp. 4895–4901. Association
for Computational Linguistics, 2023. doi: 10.18653/V1/
2023.EMNLP-MAIN.298. URL https://doi.org/
10.18653/v1/2023.emnlp-main.298.

Allal, L. B., Lozhkov, A., and Bakouch, E. et al. Smollm2:
When smol goes big–data-centric training of a small lan-
guage model. arXiv preprint arXiv:2502.02737, 2025.

Austin, J., Odena, A., and Nye, M. et al. Program synthesis
with large language models, 2021. URL https://
arxiv.org/abs/2108.07732.

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. PIQA: Reason-
ing about physical commonsense in natural language. In
Proceedings of the AAAI conference on Artificial Intelli-
gence, volume 34, 2020.

Black, S., Biderman, S., and Hallahan, E. et al. Gpt-neox-
20b: An open-source autoregressive language model.
arXiv preprint arXiv:2204.06745, 2022.

Chen, M., Tworek, J., and Jun, H. et al. Evaluating large
language models trained on code, 2021.

Cheng, R., Guan, Y., Ding, Y., Hu, Q., Wei, Y., Yuan,
C., Shen, Y., Chen, W., and Gong, Y. Mixture of neu-

ron experts, 2025. URL https://arxiv.org/abs/
2510.05781.

Clark, A., de Las Casas, D., and Guy, A. et al. Unified
scaling laws for routed language models. In International
Conference on Machine Learning, ICML 2022, 17-23
July 2022, Baltimore, Maryland, USA, volume 162 of
Proceedings of Machine Learning Research, pp. 4057–
4086. PMLR, 2022. URL https://proceedings.
mlr.press/v162/clark22a.html.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
question answering? try ARC, the AI2 reasoning chal-
lenge. arXiv preprint arXiv:1803.05457, 2018.

Cobbe, K., Kosaraju, V., and Bavarian, M. et al. Train-
ing verifiers to solve math word problems, 2021. URL
https://arxiv.org/abs/2110.14168.

Dai, D., Deng, C., and Zhao, C. et al. Deepseekmoe: To-
wards ultimate expert specialization in mixture-of-experts
language models.
arXiv preprint arXiv:2401.06066,
2024.

DeepSeek-AI, Liu, A., and Feng, B. et al. DeepSeek-V3
Technical Report, February 2025.

Du, N., Huang, Y., and Dai, A. M. et al. Glam: Efficient scal-
ing of language models with mixture-of-experts. In Inter-
national Conference on Machine Learning, ICML 2022,
17-23 July 2022, Baltimore, Maryland, USA, volume 162
of Proceedings of Machine Learning Research, pp. 5547–
5569. PMLR, 2022. URL https://proceedings.
mlr.press/v162/du22c.html.

Fang, Z., Huang, Y., Hong, Z., Lyu, Y., Chen, W., Yu, Y.,
Yu, F., and Zheng, Z. Klotski: Efficient mixture-of-expert
inference via expert-aware multi-batch pipeline, 2025.
URL https://arxiv.org/abs/2502.06888.

Fedus, W., Zoph, B., and Shazeer, N. Switch transform-
ers: Scaling to trillion parameter models with simple
and efficient sparsity. J. Mach. Learn. Res., 23:120:1–
120:39, 2022. URL https://jmlr.org/papers/
v23/21-0998.html.

Fourrier, C., Habib, N., Kydl´ıˇcek, H., Wolf, T., and
Tunstall, L. Lighteval: A lightweight framework for
llm evaluation, 2023. URL https://github.com/
huggingface/lighteval.

Gale, T., Narayanan, D., Young, C., and Zaharia, M.
Megablocks: Efficient sparse training with mixture-of-
experts. In Song, D., Carbin, M., and Chen, T. (eds.),
Proceedings of the Sixth Conference on Machine Learn-
ing and Systems, MLSys 2023, Miami, FL, USA, June 4-8,
2023. mlsys.org, 2023.

10


OmniMoE

Guo, W., Mishra, M., Cheng, X., Stoica, I., and Dao, T.
Sonicmoe: Accelerating moe with io and tile-aware opti-
mizations, 2025. URL https://arxiv.org/abs/
2512.14080.

H¨agele, A., Bakouch, E., Kosson, A., Von Werra, L., Jaggi,
M., et al. Scaling laws and compute-optimal training
beyond fixed training durations. Advances in Neural
Information Processing Systems, 37:76232–76264, 2024.

He, J., Qiu, J., Zeng, A., Yang, Z., Zhai, J., and Tang, J.
FastMoE: A Fast Mixture-of-Expert Training System,
March 2021.

He, X., Zhang, S., and Tang, K. et al. Expertflow: Ef-
ficient mixture-of-experts inference via predictive ex-
pert caching and token scheduling, 2026. URL https:
//arxiv.org/abs/2410.17954.

He, X. O.
Mixture of A million experts.
CoRR,
abs/2407.04153, 2024.
doi: 10.48550/ARXIV.2407.
04153.
URL https://doi.org/10.48550/
arXiv.2407.04153.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,
Song, D., and Steinhardt, J. Measuring massive multitask
language understanding. In International Conference on
Learning Representations, 2021a.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring math-
ematical problem solving with the math dataset, 2021b.
URL https://arxiv.org/abs/2103.03874.

Hoffmann, J., Borgeaud, S., and Mensch, A. et al. An em-
pirical analysis of compute-optimal large language model
training. Advances in Neural Information Processing
Systems (NeurIPS), 35:30016–30030, 2022.

Jiang, A. Q., Sablayrolles, A., and Roux, A. et al. Mixtral
of experts

..._This content has been truncated to stay below 50000 characters_...
