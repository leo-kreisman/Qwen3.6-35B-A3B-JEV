# SOURCE: https://arxiv.org/pdf/2608.12435

2026-08-14

MARCH: Scaling Recurrent Memory with
Content-Routed State Anchors

Ming Zhang1,2∗, Kaisen Yang2∗, Shu Yu1,3, Ermo Hua1,2, Ning Ding1,2, Xia Hu1, Bowen Zhou1,2,
Chaochao Lu1†♯, Youbang Sun1,2‡♯

1 Shanghai AI Laboratory
2 Tsinghua University
3 Fudan University
∗Equal Contribution.
† Project Lead.
‡ Technical Lead.
♯Corresponding Authors.

Abstract | Transformers owe much of their strong long-context retrieval capability to a token-level memory
that grows with context length. This flexibility, however, incurs a quadratic computation complexity during
training and a key–value cache that grows linearly during autoregressive inference. Recurrent alternatives
offer efficient decoding by compressing the entire history into a fixed-size state, but often underperform
on recall-intensive tasks since earlier associations usually get overwritten by subsequent updates, and
only the most recent contextual information is retained. In this paper, we introduce Memory-Anchor
Routing across Context History (MARCH), a network architecture that effectively scales state-space models
beyond a fixed-size dimension, while maintaining computational efficiency over long-sequences. MARCH
periodically caches cumulative recurrent-state checkpoints as state anchors and associates each anchor
with a compact, content-conditioned anchor key. This lets MARCH maintain a memory bank, which can
grow as context length increases, providing a controllable trade-off between historical resolution and
memory cost. At each token, MARCH produces an anchor query to attend all causally available state
anchors, and the output is calculated as an attention-style aggregation over all historical anchors along the
current state. We show that after standard pretraining, MARCH consistently outperforms multiple linear
attention variants across commonsense reasoning, LongBench, and in-context retrieval. These results
demonstrate that content-routed state caching substantially strengthens recurrent long-range memory
while preserving its native computation path.

0

20

40

60

80

100

Score (%)

55.3

81.7

Single NIAH

+47.7%

16.9

32.9

Complex NIAH

+94.8%

20.6

30.8

RULER

+49.6%

Gated-DeltaNet
MARCH

State Space Update 

.

Vanilla SSM

State Anchor Bank 

Selector

Readout

MARCH

�1
�2
�1
�2

Figure 1 | Overview of MARCH. Left: long-context retrieval performance at a context length of 8K
for MARCH and Gated DeltaNet. Right: State anchors enable content-based retrieval of historical
recurrent states.

arXiv:2608.12435v1  [cs.LG]  12 Aug 2026


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of
language understanding and generation tasks. However, many real-world applications—including
long-document understanding, multi-turn interaction, and in-context learning—require models to
integrate information distributed across extended sequences [Bai et al., 2025; Kwan et al., 2024; Zou
et al., 2025]. Supporting such contexts involves more than simply increasing the number of input
tokens: models must retain relevant information across long intervening spans and reliably retrieve it
when needed [Liu et al., 2024a; Hsieh et al., 2024]. Effective long-context modeling therefore hinges
on a model’s ability to manage memory—determining what information to preserve, how to represent
it, and when to retrieve it [Wang et al., 2023; Behrouz et al., 2024]. A useful perspective is to view a
sequence model as a memory system with two basic operations: writing, which incorporates each new
input into memory, and reading, which retrieves information relevant to the current input [Behrouz
et al., 2024]. Under this view, standard self-attention maintains a growing token-level memory in
its key–value cache [Vaswani et al., 2017]: it writes by appending each new key–value pair without
compressing the existing cache, and reads by matching the current query against all stored keys and
combining their associated values. This uncompressed, token-level memory provides a direct path to
every preceding token, enabling accurate recall of fine-grained details and distant dependencies. Its
flexibility, however, entails quadratic computation during training and a key–value cache that grows
linearly with sequence length during autoregressive inference, making self-attention increasingly
costly as context windows expand.

Linear attention and modern recurrent sequence models make the opposite trade-off [Katharopoulos
et al., 2020; Dao and Gu, 2024]. They compress the causal prefix into a fixed-size, matrix-valued
recurrent state, update this state with each new input, and read from it by applying the current
query. This design enables constant-memory recurrent decoding, but the same compressed write
that makes it efficient also limits its long-range memory. At each step, information from the new
token is written into a state already shared by the entire history. Much recent work has therefore
focused on improving the write operation. Selective state-space models such as Mamba introduce
input-dependent state transitions and forgetting [Gu and Dao, 2023], whereas DeltaNet and Gated
DeltaNet use data-dependent delta-rule updates to revise existing associations before incorporating
new information [Schlag et al., 2021; Yang et al., 2025]. These mechanisms improve state tracking
and mitigate indiscriminate accumulation, but they do not eliminate the underlying fixed-state
bottleneck: the entire history must still share one evolving state, and only its latest version remains
available for reading. Indeed, although recent linear recurrent models can match or surpass softmax
attention in short-context settings, their performance often degrades as the evaluation context grows
[Arora et al., 2024a; Wang et al., 2026]. Once an earlier association has been weakened by forgetting
or modified by subsequent writes, the model has no direct path to its earlier representation and
cannot recover it from the latest state alone.

Recent works have relaxed the fixed-state bottleneck along two broad directions. One approach
increases the memory capacity available at each step through partitioned sparse states, large memories
with sparse reads and writes, or routed mixtures of independent states [Pan et al., 2025; Cabannes
et al., 2026; Du et al., 2025]. A second line preserves a temporally structured collection of compressed
states through logarithmic hierarchies, adaptive state construction and merging, or recurrent-state
caching [Wang et al., 2026; Guo et al., 2026; Behrouz et al., 2026]. Collectively, these approaches
demonstrate that expanding memory capacity or temporal coverage can improve long-range recall.
However, with the exception of certain instances in [Behrouz et al., 2026], all of the existing works still
maintain a finite or upper-constrained state space dimension, though the dimensionality of which is
increased. Moreover, as multiple states become available, the primary bottleneck shifts from memory

2


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

construction to memory retrieval: the model must determine which state retains the information
most relevant to the current token. In particular, how to construct context-dependent representations
for historical checkpoints that facilitate effective and efficient query-dependent retrieval remains
underexplored.

In this work, we introduce Memory-Anchor Routing across Context History (MARCH), a memory-
augmented recurrent architecture that enables selective retrieval from earlier versions of recurrent
memory. Without modifying the underlying recurrence, MARCH periodically preserves cumulative
states as state anchors, giving later tokens access to earlier versions of the evolving memory. Each
anchor is associated with a compact learned descriptor, allowing the model to route each token to
relevant historical states when additional context is needed and combine their contents with the
current-state readout. MARCH thereby complements efficient recurrent processing with selective
access to preserved historical memory. The resulting mechanism remains causal and is trained end to
end with the standard language-modeling objective.

Our main contributions are summarized as follows:

• Routable memory-state bank. We introduce MARCH, which expands fixed-state recurrence
into a growing bank of historical memory states which increases its capacity as context grows,
alleviating the single-state memory bottleneck without modifying the underlying recurrence.
• Content-conditioned historical retrieval. MARCH brings attention-style content routing to
recurrent memory by applying a standard softmax over compact keys for a temporally sparse
set of state anchors rather than token-level key–value pairs. A learned null route allows the
model to suppress the historical branch when the current recurrent state is sufficient, while
residual fusion preserves the original recurrent path and supports end-to-end training.
• Extensive empirical validation. We demonstrate consistent improvements over strong recurrent
baselines across commonsense reasoning, LongBench, in-context retrieval, and NIAH evaluations,
including robust extrapolation beyond the training context length.

2. Preliminaries

Full and Linear Attention as Memory.
Let x𝑡∈ℝ𝑑denote the hidden representation at position 𝑡.
The corresponding query, key, and value vectors are obtained through learned linear projections:

q𝑡= W𝑞x𝑡,
k𝑡= W𝑘x𝑡,
v𝑡= W𝑣x𝑡,
(1)

where W𝑞, W𝑘∈ℝ𝑑𝑘×𝑑and W𝑣∈ℝ𝑑𝑣×𝑑, such that q𝑡, k𝑡∈ℝ𝑑𝑘and v𝑡∈ℝ𝑑𝑣. Following the memory-
system perspective adopted in prior work [Behrouz et al., 2024], we view a causal sequence mixer as
an online memory system. Let M𝑡denote the memory state after processing the first 𝑡tokens, with
M0 denoting its initial state. At each position, the current key–value pair is first written into memory,
after which the updated memory is queried using the current query:

M𝑡= Write�M𝑡−1; k𝑡, v𝑡
�
,
o𝑡= Read�M𝑡; q𝑡
�
,
(2)

where o𝑡∈ℝ𝑑𝑣denotes the memory readout at position 𝑡. For causal softmax attention, the memory
explicitly retains all projected key–value pairs observed up to position 𝑡:

M𝑡= �K≤𝑡, V≤𝑡
�
,
(3)

where K≤𝑡∈ℝ𝑡×𝑑𝑘and V≤𝑡∈ℝ𝑡×𝑑𝑣stack the keys and values row-wise, respectively. Writing appends
(k𝑡, v𝑡) to these matrices, whereas reading performs content-based retrieval:

o𝑡= V⊤
≤𝑡softmax
�K≤𝑡q𝑡
√

𝑑𝑘

�

.
(4)

3


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

This explicit storage keeps individual tokens directly retrievable and enables fine-grained retrieval
from the entire causal prefix. However, processing a sequence of length 𝑇requires O(𝑇2) query–key
interactions, while autoregressive decoding maintains a key–value cache of size O(𝑇(𝑑𝑘+ 𝑑𝑣)) per
attention head [Vaswani et al., 2017].

Linear attention instantiates the memory state M𝑡as a fixed-size matrix S𝑡∈ℝ𝑑𝑣×𝑑𝑘. Its write and
read operations are given by
S𝑡= S𝑡−1 + v𝑡k⊤
𝑡,
o𝑡= S𝑡q𝑡.
(5)

Each write therefore adds a rank-one key–value association to the shared matrix, while each read
retrieves a query-dependent superposition of the stored values. This enables constant-memory
recurrent decoding, but introduces interference as the compressed history grows.

Gated DeltaNet (GDN).
To mitigate the interference caused by the additive write rule of linear
attention, GDN retains the same state-based read operation but introduces input-dependent retention
and a targeted delta-rule write [Yang et al., 2025]:

S𝑡= 𝛼𝑡S𝑡−1 + 𝛽𝑡(v𝑡−𝛼𝑡S𝑡−1k𝑡) k⊤
𝑡,
o𝑡= S𝑡q𝑡,
(6)

where 𝛼𝑡∈(0, 1) is an input-dependent retention gate, and 𝛽𝑡∈[0, 1] modulates the strength of the
targeted delta update. Despite its more adaptive state dynamics, GDN still compresses the entire
causal history into a single fixed-size recurrent state. Because all associations share this evolving state,
information weakened or overwritten by subsequent updates has no direct retrieval path, limiting
reliable long-context recall.

Scaling recurrent memory.
Recent work has sought to relax the fixed-state bottleneck along
two broad directions. Capacity-expansion methods enlarge the current recurrent state, whereas
temporal-expansion methods retain multiple versions of the state along its trajectory:

Mcap
𝑡
:= �S𝑡=
�
S(1)
𝑡
| · · · | S(𝑃)
𝑡
�
∈ℝ𝑑𝑣×𝐷mem,
𝐷mem =

𝑃
∑︁

𝑝=1
𝑑𝑝,

Mtemp
𝑡
:= �S𝑡,1, . . . , S𝑡,𝑀𝑡
�
,
S𝑡,𝑚∈ℝ𝑑𝑣×𝑑𝑘,
1 ≤𝜏𝑡,1 < · · · < 𝜏𝑡,𝑀𝑡≤𝑡.

(7)

In the capacity formulation, 𝑃is the fixed number of state partitions and 𝐷mem is their total memory
dimension. Such methods increase 𝐷mem while using sparse access to keep computation tractable [Pan
et al., 2025; Cabannes et al., 2026]. In the temporal formulation, 𝑀𝑡is the number of retained state
representations, each associated with a temporal boundary 𝜏𝑡,𝑚. These methods preserve states from
distinct temporal regions or earlier stages of the recurrent trajectory [Wang et al., 2026; Guo et al.,
2026; Behrouz et al., 2026]. MARCH follows the latter direction by retaining cumulative snapshots of
a continuously evolving recurrent state.

3. Method

Figure 2 illustrates MARCH, a content-routed recurrent memory framework that enables selective
retrieval from earlier versions of an evolving recurrent state. Rather than routing over predefined
state indices or temporal scales, MARCH matches each query against individual historical states
based on their contents. As tokens are processed, MARCH periodically checkpoints the cumulative
recurrent state, producing a bank of state anchors. Each checkpoint is paired with an occurrence of a
shared learned anchor token, whose hidden representation yields a compact routing key. For each

4


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

(a) MARCH matches anchor keys, then reads their states

…
…
ξ₁
ξ₂
now
…

κ₁
κ₂
route ρ
read q

ρ · κ₁
ρ · κ₂
null

softmax
weights π

Checkpoint
current state

S

π₁
π₂
×
×

Σ
+
output

(b) MARCH augments every

recurrent layer

tokens + anchor positions

Layer 1
MARCH
FFN

⋮

Layer ℓ

MARCH
FFN

Layer N
MARCH
FFN

×N

model output

current state

checkpoint bank

κ₁
A₁
κₘ
Aₘ

Token positions directly project keys and queries

Figure 2 | Architecture of MARCH. Left: Text tokens update one continuous Gated DeltaNet state,
while periodic checkpoints form state anchors. Token-dependent routing combines causally visible
anchors, and the resulting historical readout is added to the current-state readout. Right: MARCH
augments every recurrent layer and rebuilds state-aware routing keys across layers.

text token, a routing query scores all causally visible anchors alongside a learned null option, allowing
the model to use historical memory only when useful. The resulting routing probabilities define a
weighted combination of the visible anchor states, which is read using the token’s standard recurrent
query. This historical readout is added to the current-state readout, preserving the native recurrent
path while introducing a content-dependent route to earlier memory. Together, state anchoring
and content-routed retrieval turn the otherwise transient state trajectory into a persistent source of
long-range memory.

3.1. Continuous Recurrent-State Anchoring

Anchor placement.
Let T = [𝑡1, . . . , 𝑡𝐿] be a sequence of 𝐿text tokens. An anchoring policy specifies
an ordered set of text boundaries B = {𝑏𝑚}𝑀
𝑚=1, where 0 = 𝑏0 < 𝑏1 < · · · < 𝑏𝑀≤𝐿. We insert an anchor
position after each boundary:

�T =
𝑀
∥
𝑚=1

�
[𝑡𝑏𝑚−1+1, . . . , 𝑡𝑏𝑚] ∥[𝜉𝑚]
�
∥[𝑡𝑏𝑀+1, . . . , 𝑡𝐿].
(8)

where ∥denotes sequence concatenation, and 𝜉𝑚is the 𝑚-th occurrence of a shared learned anchor
embedding 𝜉. Text and anchor positions serve different computational roles. Text positions apply
the base recurrent update, allowing the matrix-valued state to evolve continuously across anchor
boundaries. Immediately after processing 𝑡𝑏𝑚, MARCH checkpoints the resulting cumulative state to
form the 𝑚-th state anchor. The following anchor position 𝜉𝑚does not modify the recurrent state;
instead, its hidden representation provides the routing metadata associated with that checkpoint.
Thus, each anchor boundary produces two coupled objects: a snapshot of the recurrent memory and
a compact representation through which that snapshot can later be retrieved. We formalize these
two operations next.

5


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

Cumulative recurrent-state checkpointing.
At layer ℓ, MARCH leaves the underlying recurrent
update unchanged and carries the state S(ℓ)
𝑡
∈ℝ𝑑𝑣×𝑑𝑘continuously across anchor boundaries. At each
boundary 𝑏𝑚, it snapshots the current state:

A(𝑚,ℓ) = S(ℓ)
𝑏𝑚∈ℝ𝑑𝑣×𝑑𝑘,
𝑚= 1, . . . , 𝑀.
(9)

Because the recurrence is not reset between anchor boundaries, A(𝑚,ℓ) encodes the cumulative prefix
up to position 𝑏𝑚, rather than only the segment since the preceding boundary. We therefore refer to
it as a state anchor. The ordered bank {A(1,ℓ), . . . , A(𝑀,ℓ)} traces the temporal evolution of a single
recurrent memory, preserving earlier versions before subsequent decay and delta updates attenuate
or modify their contents.

Content-conditioned anchor metadata.
Let u(ℓ)
𝑚denote the normalized input representation of
anchor position 𝜉𝑚at layer ℓ. The anchor position reads only its aligned state checkpoint:

q(ℓ)
𝑚= W(ℓ)
𝑞u(ℓ)
𝑚,
o(ℓ)
𝑚= A(𝑚,ℓ)q(ℓ)
𝑚.
(10)

The same input representation is projected into a compact routing key:

𝜿(ℓ)
𝑚= W(ℓ)
𝑘u(ℓ)
𝑚
∈ℝ𝑑𝑟.
(11)

The aligned readout is incorporated into the anchor position through the standard output projection
and residual pathway. Consequently, u(ℓ+1)
𝑚
depends on A(𝑚,ℓ), and the routing key produced at layer
ℓ+ 1 becomes conditioned on the content retained by the aligned state anchor. Thus, although all
anchor positions share the same learned input embedding, they acquire distinct, state-dependent
representations after the first layer. This cross-layer construction makes routing explicitly dependent
on what each state anchor contains, rather than only on its temporal index.

3.2. Content-Routed Historical Reading

Content-based routing.
For a text token at position 𝑡, the causally available state anchors are
indexed by V𝑡= {𝑚∈{1, . . . , 𝑀} | 𝑏𝑚< 𝑡}. MARCH projects the normalized hidden state x𝑡into a
routing query and scores it against the key of each visible anchor:

𝝆𝑡= W𝑅x𝑡,
𝑎𝑡,𝑚= 𝝆⊤
𝑡𝜿𝑚,
𝑚∈V𝑡.
(12)

To allow the model to bypass historical memory, we augment the visible anchor set with a null
option ∅, whose payload is fixed to zero, A(∅) = 0. Its query-dependent logit is 𝑛𝑡= w⊤
∅x𝑡+ 𝑏∅. Let
�
V𝑡= V𝑡∪{∅} denote the augmented candidate set. We define the logit of each candidate 𝑗∈�
V𝑡as

𝑠𝑡,𝑗=

�
𝑎𝑡,𝑗,
𝑗∈V𝑡,

𝑛𝑡,
𝑗= ∅,
𝜋𝑡,𝑗=
exp(𝑠𝑡,𝑗)
∑︁

𝑟∈�
V𝑡
exp(𝑠𝑡,𝑟)
.
(13)

Since the selected routing probabilities directly weight the historical state readouts, their scores
remain jointly optimized by the language-modeling objective. The routing query 𝝆𝑡determines which
anchors to retrieve, whereas the state-read query q𝑡reads their matrix-valued contents. If no anchor
is visible, the null option receives all probability mass.

We note that the aggregation formulation in Equation (13) readily admits a sparse variant by restricting
aggregation to the 𝐾highest-scoring visible anchors (Top-𝐾). This sparse approach exhibits natural
connections with the hierarchical sparse attention approaches [Lu et al., 2025; Hu et al., 2026], while
preserving dense token-level processing rather than relying on hard token-level pruning. Our ablation
studies in Section 5 show that sparse routing substantially reduces aggregation cost with minimal
performance degradation.

6


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

Historical retrieval and residual fusion.
Given the routing probabilities, the causally visible state
anchors are aggregated into a query-dependent historical state, which is read using the same state-
read query as the current state. The resulting historical readout is then added to the current-state
readout:
o𝑡= S𝑡q𝑡+
∑︁

𝑗∈�
V𝑡

𝜋𝑡,𝑗A( 𝑗)q𝑡.
(14)

This additive formulation preserves the original recurrent path and introduces historical retrieval
as an auxiliary residual branch, without modifying the underlying recurrent update. Since the
routing probabilities directly affect the layer output, the routing queries and anchor-derived keys are
optimized end-to-end with the language-modeling objective.

3.3. Implementation

We implement MARCH as a two-stage producer–reader computation. Following the hardware-efficient
chunkwise formulation of Gated DeltaNet [Yang et al., 2025], the producer processes recurrent updates
in blocks amenable to tensor-core acceleration, computes each token’s current-state output, and
checkpoints the recurrent state at each anchor boundary. The resulting state anchors are consumed
by the historical reader. Inspired by the I/O-aware principles of FlashAttention [Dao et al., 2022],
the reader jointly tiles query tokens and state anchors, reuses each anchor tile across a block of
queries, and fuses routing-score computation, online softmax updates, and the accumulation of
weighted state readouts into a streaming reduction. This fused schedule avoids materializing either
the dense token-to-anchor routing matrix or the substantially larger tensor of per-anchor candidate
readouts, thereby reducing intermediate storage and the associated HBM traffic. As shown in Figure 4,
despite the cost of historical retrieval, our fused dense implementation exceeds FlashAttention-2 in
throughput at 64K and above and incurs lower core runtime from 32K onward.

4. Experiments

MARCH is designed to extend the long-range memory of recurrent models while preserving their
general language capabilities. In this paper, we verify the effectiveness of MARCH by training from
scratch, and evaluate across a diverse suite of benchmarks spanning zero-shot commonsense reason-
ing, long-context understanding, and in-context retrieval. Across these tasks, MARCH consistently
outperforms existing recurrent baselines, with particularly strong gains on retrieval-intensive and
long-context benchmarks.

4.1. Experimental Setup

Training configuration.
Following the academic-scale protocol used by Log-Linear Attention [Guo
et al., 2026], we pretrain the models from scratch on 50B tokens from the Long-Data-Collections
dataset, using a sequence length of 16K. The main configurations use 21 layers and a hidden size
of 1536. The Transformer (693M) uses 16 attention heads and a RoPE base of 500K, while Gated
DeltaNet (793M) and its variants use six value heads. To control for parameter count in addition
to model depth, we also include a 24-layer Transformer with (778M) parameters, closely matching
the size of the Gated DeltaNet. For MARCH, we set the routing dimension to 𝑑𝑟= 64 and use a
periodic anchoring interval of 𝐶= 512 text tokens. We train all models with a global batch size of
approximately 4.2M tokens using the fused AdamW optimizer, with 𝛽1 = 0.9, 𝛽2 = 0.95, 𝜖= 10−8,
and a weight decay of 0.1. The peak learning rate is set to 4 × 10−4 with a warmup-stable-decay
schedule. All models use the same training data, token budget, context length, and optimization
configuration.

7


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

Table 1 | Zero-shot performance of MARCH and baseline models on eight commonsense reasoning
benchmarks. Results are reported using accuracy (acc) or normalized accuracy (acc_n), as indicated
in the column headers; higher is better (↑). The best result among Gated DeltaNet variants in each
column is highlighted in bold.

Model
LMB.
PIQA
Hella.
Wino.
ARC-e
ARC-c
OBQA
CSQA
Avg.
acc ↑
acc ↑
acc ↑
acc ↑
acc_n ↑
acc_n ↑
acc_n ↑
acc ↑

Transformer
49.4
66.5
33.9
52.1
47.8
26.4
32.0
22.1
41.3
w/ 24 Layers
50.3
67.6
34.4
50.6
46.3
25.8
31.2
24.7
41.4

Gated DeltaNet
48.5
66.1
33.1
50.8
45.9
25.3
30.0
21.1
40.1
w/ Log-Linear
47.7
65.7
33.2
51.9
44.3
24.9
30.4
21.7
40.0
w/ MARCH
49.5
66.9
34.8
52.6
47.1
25.6
32.8
22.5
41.5

Baselines.
Our primary comparisons are against standard GDN [Yang et al., 2025] and GDN
augmented with Log-Linear Attention [Guo et al., 2026]. MARCH and these two baselines use matched
architectural configurations and the same pretraining setup, enabling a controlled comparison of
their memory mechanisms. To contextualize their performance against full attention, we additionally
include two Transformer baselines: a 21-layer model matched in depth to the recurrent models and a
24-layer model approximately matched to them in parameter count.

Evaluation tasks.
For short-context generalization, we use eight zero-shot commonsense bench-
marks: LAMBADA [Paperno et al., 2016], PIQA [Bisk et al., 2020], HellaSwag [Zellers et al., 2019],
WinoGrande [Sakaguchi et al., 2020], ARC-Easy and ARC-Challenge [Clark et al., 2018], Open-
BookQA [Mihaylov et al., 2018], and CommonsenseQA [Talmor et al., 2019]. We additionally
evaluate long-context understanding on LongBench [Bai et al., 2024], covering single-document
QA, multi-document QA, summarization, and few-shot learning. The long-context retrieval evalu-
ation covers six single-neddle and multi-needle tasks from RULER [Hsieh et al., 2024] at 4K, 8K,
and 16K context lengths. Finally, the in-context retrieval suite contains SQuAD [Rajpurkar et al.,
2018], TriviaQA [Joshi et al., 2017], SWDE [Lockard et al., 2019], FDA [Arora et al., 2023], Natural
Questions [Kwiatkowski et al., 2019], and DROP [Dua et al., 2019]. We follow the evaluation protocol
of prior work [Wang et al., 2026] and use the LM-Evaluation-Harness [Gao et al., 2021].

4.2. Main Results

Commonsense reasoning.
As shown in Table 1, MARCH consistently outperforms both the vanilla
and Log-Linear variants of Gated DeltaNet across all eight zero-shot commonsense reasoning bench-
marks. It improves the average accuracy from 40.1 and 40.0 to 41.5, respectively, with the largest
gain over the vanilla baseline observed on OpenBookQA (+2.8 points), aligning with our findings
in retrieval tasks presented below. Moreover, MARCH achieves a higher average score than both
Transformer baselines, surpassing the standard Transformer on six of eight tasks and the 24-layer
Transformer on four. These results indicate that MARCH consistently strengthens the Gated DeltaNet
backbone while remaining competitive with comparable full-attention models on short-context lan-
guage understanding tasks.

Needle-in-a-haystack retrieval.
We evaluate long-context associative retrieval using the needle-in-a-
haystack (NIAH) suite from RULER [Hsieh et al., 2024], where a model must recover values associated
with keys embedded among irrelevant context. All models are trained with a maximum context length

8


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

Table 2 | Results on twelve LongBench tasks. The best result among Gated DeltaNet and its variants
is shown in bold for each task. The relative average gain over the Gated DeltaNet is shown in green
parentheses.

Single-Doc QA
Multi-Doc QA
Summarization
Few-shot Learning

Model
NQA QQA MFQ HQA 2WM Mus GvR QMS MNs TRC TQA
SSM
Avg.↑

Transformer
4.4
4.1
15.9
7.5
9.9
4.1
10.7 11.6 14.5 21.0 33.9
28.3
13.8
w/ 24 Layers
3.5
11.1 18.3
7.8
9.8
4.1
11.3 12.9 12.8 22.5 47.5
23.2
15.4

Gated DeltaNet
3.0
4.8
13.3
5.6
8.7
2.1
2.6
11.3 13.0 18.0 38.6
21.9
11.9
w/ Log-Linear
3.6
6.2
13.6
7.1
8.2
3.3
6.3
13.2 13.5 17.0 32.8
25.1
12.5
w/ MARCH
4.2
7.8
14.6
7.4
11.5
4.8
8.2
17.4 14.1 19.0 43.1
26.3
14.9 (↑25%)

of 16K; Figure 3 reports results from 4K to 32K, making 32K a zero-shot length-extrapolation setting.
Across the 24 task–length combinations, MARCH outperforms the stronger recurrent baseline in 19
settings and matches it in the remaining five. On the multi-needle tasks, it wins in 11 of 12 settings.
At 32K, MARCH achieves the best result on all six tasks, retaining perfect accuracy on S-NIAH-1
and nonzero accuracy on the remaining tasks, whereas both Transformer variants and Log-Linear
Gated-DeltaNet score zero throughout. This contrast is consistent with RoPE extrapolation in the
Transformers and the state-index-dependent coefficients of Log-Linear Gated-DeltaNet. MARCH
instead shares the same content-based router across all anchors, allowing longer contexts to introduce
additional anchors without requiring new anchor-specific routing parameters.

Long-context understanding.
Table 2 reports results across four LongBench task categories.
MARCH consistently outperforms both vanilla Gated DeltaNet and its log-linear variant on all twelve
tasks. The improvements are particularly pronounced on multi-document QA: relative to the stronger
of the vanilla and log-linear Gated DeltaNet baselines, MARCH raises the 2WikiMultihopQA score

0.0

0.2

0.4

0.6

0.8

1.0

S-NIAH-1

0.0

0.2

0.4

0.6

0.8

1.0

S-NIAH-2

0.0

0.2

0.4

0.6

0.8

1.0

S-NIAH-3

4K
8K
16K
32K

0.0

0.2

0.4

0.6

0.8
MK-NIAH-1

4K
8K
16K
32K

0.0

0.2

0.4

0.6

MQ-NIAH

4K
8K
16K
32K

0.0

0.1

0.2

0.3

0.4

0.5
MV-NIAH

Transformer
Transformer (24 Layers)

Gated-DeltaNet
Log-Linear Gated-DeltaNet

MARCH

Figure 3 | NIAH performance on three single-needle and three multi-needle tasks. The Transformer
achieves perfect accuracy on both S-NIAH-1 and S-NIAH-2 at context lengths of 4K, 8K, and 16K.

9


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

Table 3 | In-context retrieval accuracy (↑). The best result among Gated DeltaNet and its variants
is marked in bold for each benchmark, with the relative gain over the stronger of the two Gated
DeltaNet baselines shown in green parentheses.

Model
SQuAD↑
SWDE↑
FDA↑
TriviaQA↑
DROP↑
NQ↑
Avg.↑

Transformer
41.3
59.3
80.4
2.2
2.9
1.7
31.3
w/ 24 Layers
40.4
64.9
83.7
4.0
3.4
2.5
33.2

Gated DeltaNet
34.8
45.0
31.4
1.1
2.2
0.8
19.2
w/ Log-Linear
33.7
46.1
38.2
1.3
2.6
1.0
20.5
w/ MARCH
37.7 (↑8%)
51.9 (↑13%)
44.6 (↑17%)
1.6 (↑23%)
2.9 (↑12%)
1.2 (↑20%)
23.3 (↑14%)

from 8.7 to 11.5 and the MuSiQue score from 3.3 to 4.8, corresponding to relative gains of 32% and
45%, respectively. The benefits also extend to summarization, where the QMSum score increases from
13.2 to 17.4 (32%), and to all three few-shot learning tasks. These results show that content-routed
state anchors improve long-context understanding across diverse task formats, rather than benefiting
only retrieval-oriented question answering.

In-Context Retrieval.
Following [Arora et al., 2024b], we evaluate in-context retrieval on six real-
world, recall-intensive benchmarks. As shown in Table 3, MARCH consistently outperforms both
vanilla Gated DeltaNet and its Log-Linear variant across all tasks. Relative to the stronger of the vanilla
and log-linear Gated DeltaNet baselines on each benchmark, MARCH yields relative improvements
ranging from 8% on SQuAD to 23% on TriviaQA and raises the average accuracy from 20.5 to 23.3,
corresponding to a 14% relative improvement. These consistent gains across heterogeneous retrieval
tasks demonstrate that MARCH improves the retrieval capability of the Gated DeltaNet backbone
beyond a particular dataset or input format. Together, these results establish content-routed state
anchors as an effective mechanism for strengthening fine-grained retrieval in recurrent models.

8192
16384
32768
65536
131072
Sequence Length

1 × 105

6 × 104

2 × 105

3 × 105

4 × 105

Throughput (tokens/s)

Transformer
Gated-DeltaNet

MARCH (Dense)
MARCH (TopK4)

8192
16384
32768
65536
131072
Sequence Length

100

101

102

103

Runtime (ms)

Transformer
Gated-DeltaNet

MARCH (Dense)
MARCH (TopK4)

Figure 4 | Training efficiency across sequence lengths. Left: end-to-end training throughput in
tokens per second (higher is better). Right: forward–backward runtime of the core sequence-mixing
operation in milliseconds (lower is better). MARCH (Top-4) retains only the four highest-scoring
state anchors for each token and head during historical retrieval.

Training efficiency.
Figure 4 compares the end-to-end throughput and core forward–backward
runtime of FlashAttention-2, Gated DeltaNet, dense MARCH, and its Top-4 implementation. Sparse
routing becomes increasingly beneficial as the context grows. At 128K tokens, Top-4 MARCH more

10


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

than doubles the training throughput of dense MARCH and reduces its core runtime by roughly an
order of magnitude. It also achieves higher throughput than FlashAttention-2 at this length, although
vanilla Gated DeltaNet remains faster because it incurs no historical-retrieval overhead.

5. Ablation Studies

Effect of chunk size.
The training chunk size 𝐶determines how frequently MARCH checkpoints the
recurrent state. Smaller chunks create denser candidate anchors and offer finer temporal resolution,
at the cost of a larger anchor cache and higher historical-routing overhead. We vary 𝐶from 256 to
2048 while keeping all other model and training settings fixed. Table 4 reports performance on three
in-context retrieval benchmarks and the average over the six NIAH tasks at context lengths of 4K,
8K, and 16K. As an additional inference test, we organize the MARCH state bank according to the
Fenwick tree scheme used by Log-Linear Attention [Guo et al., 2026; Fenwick, 1994]. This scheme
arranges state anchors hierarchically and retains O(log𝑇) anchors as the context grows. We report
performance close to [Guo et al., 2026]. This demonstrates that the learned router in MARCH shows
great generalizability and flexibility across various state bank organization schemes.

Table 4 | Effect of chunk size on in-context retrieval and NIAH. Panel (a) evaluates checkpoints using
the same chunk size during training and inference. Panel (b) varies the inference-time chunk size for
the checkpoint trained with chunk size 512. NIAH scores are averaged over six tasks at each context
length. Bold indicates the selected setting in each panel and the best result in each column.

In-context Retrieval
NIAH
Chunk Size
# Anchors
SQuAD
SWDE
FDA
4K
8K
16K

(a) Matched training and inference chunk sizes
256
64
36.76
49.23
44.83
58.17
49.25
44.83
512
32
37.70
51.85
44.56
59.58
54.96
39.46
1024
16
36.49
48.51
43.28
53.21
43.21
33.54
2048
8
34.15
44.19
30.76
59.54
39.83
27.96

(b) Inference-time chunk size
64
256
39.41
53.11
46.01
63.33
55.00
39.67
128
128
39.08
52.84
46.91
63.33
57.33
39.17
256
64
40.35
53.38
47.46
62.83
56.17
40.83
512
32
37.70
51.85
44.56
59.58
54.96
39.46
1024
16
37.23
43.74
34.57
55.33
48.33
32.67
2048
8
37.23
33.93
23.23
51.00
40.33
34.00
Fenwick Tree
6
36.16
36.79
22.35
52.17
39.56
33.14

In Panel (a), 𝐶= 512 provides the best overall balance between retrieval quality and anchor count.
Smaller chunks improve some long-context results but incur higher memory and routing costs,
whereas larger chunks generally degrade retrieval because the resulting checkpoints are too sparse.
We therefore adopt 𝐶= 512 as the default. Panel (b) shows that changing the chunk size at inference
provides a flexible accuracy–memory trade-off: denser anchors generally improve retrieval at higher
cost, while overly sparse anchors lead to substantial degradation. The Fenwick tree row additionally
evaluates hierarchical organization of the state bank at inference.

Routing design.
We ablate the router’s query–key dimension 𝑑𝑟, routing sparsity, and learned
null option. Table 5 reports aggregate results across general language understanding, long-context
benchmarks, and NIAH. The default uses dense routing with 𝑑𝑟= 64 and includes the null option.

11


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

Table 5 | Routing-design ablations. The first row is the default; each subsequent row changes one
component. Commonsense (CS), LongBench, and Retrieval are macro-averages over 8, 12, and 6
benchmarks, respectively. NIAH scores are averaged over six tasks, and Avg. over the three context
lengths. Column-wise best results are bold.

Configuration
Benchmark Averages
NIAH
Routing
𝑑𝑟
Null
CS
LongBench Retrieval
4K
8K
16K
Avg.

Dense (default)
64
Yes
41.48
14.87
23.31
59.58 54.96 39.46 51.33

Dense
192
Yes
40.94
13.88
24.52
56.71
51.33
37.38
48.47
Top-4
64
Yes
41.38
13.79
23.17
57.29
46.04
31.21
44.85
Dense
64
No
41.04
14.11
22.86
54.88
47.92
35.13
45.98

Increasing 𝑑𝑟to 192 yields higher retrieval capability but reduces general performance across other
tasks, making 𝑑𝑟= 64 a more balanced choice overall. Top-4 nearly matches dense routing on
commonsense and retrieval but trails on NIAH, making it an efficiency-oriented operating point when
considered alongside Figure 4. Removing the null option degrades every aggregate, confirming the
benefit of bypassing irrelevant historical states.

6. Related Work

Efficient Attention Mechanisms.
Efficient attention reduces the quadratic cost of full self-attention
through local windows, kernelization, or systems optimization. Local sliding-window attention limits
each query to a bounded neighborhood [Wang et al., 2025; Cabannes et al., 2025]. Performer,
Nystromformer, and Linear Attention replace the softmax kernel with feature maps and exploit
associativity for linear-time computation [Katharopoulos et al., 2020; Choromanski et al., 2021; Xiong
et al., 2021]. FlashAttention-2, sequence parallelism, and chunkwise algorithms instead improve
hardware efficiency without changing the dense attention pattern [Dao, 2024; Sun et al., 2024].

Sparse Attention.
Sparse attention retains content-based softmax retrieval but restricts each query
to a small subset of token-level key–value pairs. Early methods rely on predefined connectivity:
Sparse Transformer factorizes the attention pattern, while Longformer and BigBird combine local
windows with global or random links [Child et al., 2019; Beltagy et al., 2020; Zaheer et al., 2020].
Later methods make the sparse pattern input dependent: Routing Transformer clusters tokens by
content, H2O evicts low-utility cache entries, and Quest selects KV-cache pages conditioned on the
current query [Roy et al., 2021; Zhang et al., 2023; Tang et al., 2024]. More recent trainable designs
route queries to relevant blocks, as in MoBA, or combine compressed, selectively retrieved, and local
branches with hardware-aligned kernels, as in Native Sparse Attention [Lu et al., 2025; Yuan et al.,
2025]. These approaches reduce attention-score computation or memory traffic, but their accuracy
hinges on token or block selection and they still store or manipulate token-level KV memories. In
contrast, MARCH routes over compact keys associated with historical recurrent-state snapshots,
retrieving compressed prefix states rather than sparsifying token-to-token attention.

State Space Models and Gated Linear Recurrences.
State space models (SSMs) and linear
recurrent networks compress the prefix into a recurrent state. Linear Attention and its kernelized
variants share this view through decayed outer-product updates and query-based reads [Katharopoulos
et al., 2020; Dao and Gu, 2024; Chou et al., 2024]. S4 uses structured linear dynamics, while Mamba
and Mamba-2 use selective transitions; RetNet, RWKV, HGRN, and LRU combine associative memories

12


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

with structured recurrences [Dao and Gu, 2024; Gu and Dao, 2023; Gu et al., 2022; Sun et al.,
2023; Peng et al., 2023; Qin et al., 2024; Orvieto et al., 2023; Liu et al., 2024b]. GLA introduces
input-dependent decay, DeltaNet and GDN use delta-rule corrections, and GDN-2 decouples erase
and write through channel-wise gates [Yang et al., 2025, 2024; Hatamizadeh et al., 2026; Siems
et al., 2025; Grazzi et al., 2025]. Despite these advances, most models retain one fixed-capacity state,
whose dimension directly controls update and read cost; this bottleneck contributes to the retrieval
gap with Transformers [Arora et al., 2024b; Wen et al., 2024]. We preserve efficient recurrence while
expanding memory into selectively accessed states.

State Expansion and Associative Memory.
Long-context studies identify recurrent state capacity
as a central limitation of Linear Attention models [Arora et al., 2024a,b]. Multi-State RNNs, HGRN2,
and Log-Linear Attention expand or hierarchically organize recurrent states [Guo et al., 2026; Qin
et al., 2024; Oren et al., 2024]. Mixture-of-Memories, Sparse State Expansion, Product Key Memory,
and Fast-weight Product Key Memory use memory experts or sparse banks, while Sparse Delta Memory
sparsifies GDN reads and writes [Pan et al., 2025; Cabannes et al., 2026; Du et al., 2025; Lample
et al., 2019; Berges et al., 2024; Zhao and Jones, 2026; Afzal et al., 2026]. Context-compression
methods instead retrieve at the token or chunk level, using learned summary tokens or selective
chunk reopening [Chevalier et al., 2023; Mu et al., 2023; Zhang et al., 2024; Deng et al., 2025; Petrov
et al., 2025; Mao et al., 2026]. Our method treats recurrent states as retrieval units, expands total
capacity, and reads only selected states, decoupling capacity from dense per-token updates.

7. Limitations and Future Work

MARCH adopts periodic checkpointing and organizes all historical states in a single homogeneous
anchor bank. Although this design is simple and efficient, fixed-interval anchoring does not account
for the non-uniform evolution of recurrent memory: it may create redundant anchors in stable regions
while providing insufficient resolution when the state changes rapidly. A natural extension is to develop
adaptive anchoring mechanisms according to state novelty or update magnitude and to consolidate
or evict redundant anchors given a memory budget. More broadly, MARCH improves access to earlier
states but does not explicitly increase or specialize the capacity of the underlying memory. Future
work could combine state anchoring with larger-capacity memory and multiple memory partitions
specialized for different temporal scales or information types. For example, short-term context, salient
episodic events, and slowly consolidated knowledge could be maintained through distinct write,
retention, and forgetting mechanisms, while a hierarchical router determines both which memory
partition and which stored state should serve each query. In addition, MARCH has the potential
to support external memory modules for optimized performance over specific downstream tasks,
knowledge consolidation from experience to parametric information, and other memory manipulation
mechanisms, opening up new scaling directions for test-time training and continual learning.

8. Conclusion

We introduce MARCH, a novel attention architecture which augments recurrent models with content-
routed state anchors. By preserving cumulative state checkpoints, MARCH enables selective access to
earlier recurrent states without modifying the underlying recurrence. It consistently outperforms
strong recurrent baselines across commonsense reasoning, LongBench, in-context retrieval, and NIAH.
Ablations further show a controllable retrieval–efficiency trade-off through checkpoint density and
sparse routing. These results establish historical-state retrieval as a practical approach to scaling
recurrent memory beyond a single evolving state.

13


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

References

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu,
Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning on
realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 3639–3664, 2025. doi: 10.18653/v1/
2025.acl-long.183. URL https://aclanthology.org/2025.acl-long.183/.

Wai-Chung Kwan, Xingshan Zeng, Yuxin Jiang, Yufei Wang, Liangyou Li, Lifeng Shang, Xin Jiang,
Qun Liu, and Kam-Fai Wong. Mt-eval: A multi-turn capabilities evaluation benchmark for large
language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing, pages 20153–20177, 2024. doi: 10.18653/v1/2024.emnlp-main.1124. URL https:
//aclanthology.org/2024.emnlp-main.1124/.

Kaijian Zou, Muhammad Khalifa, and Lu Wang. On many-shot in-context learning for long-context
evaluation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 25605–25639, 2025. doi: 10.18653/v1/2025.acl-long.1245. URL

https://aclanthology.org/2025.acl-long.1245/.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. Lost in the middle: How language models use long contexts. Transactions of the Association
for Computational Linguistics, 12:157–173, 2024a. URL https://aclanthology.org/2024.
tacl-1.9/.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang
Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language
models?
In First Conference on Language Modeling, 2024. URL https://openreview.net/
forum?id=kIoBbc76Sy.

Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting
language models with long-term memory. Advances in Neural Information Processing Systems,
36:74530–74543, 2023. URL https://proceedings.neurips.cc/paper_files/paper/
2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv
preprint arXiv:2501.00663, 2024. URL https://arxiv.org/abs/2501.00663.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Lukasz Kaiser, and Illia Polosukhin.
Attention is all you need.
In Isabelle Guyon, Ulrike
von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Ro-
man Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Confer-
ence on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA,
USA, pages 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/
3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are
rnns: Fast autoregressive transformers with linear attention.
In Proceedings of the 37th In-
ternational Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, vol-
ume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR, 2020. URL
https://proceedings.mlr.press/v119/katharopoulos20a.html.

Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through
structured state space duality. In Proceedings of the 41st International Conference on Machine

14


MARCH: Scaling Recurrent Memory with Content-Routed State Anchors

Learning, volume 235 of Proceedings of Machine Learning Research, pages 10041–10071. PMLR,
2024. URL https://proceedings.mlr.press/v235/dao24a.html.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv
preprint arXiv:2312.00752, 2023. URL https://arxiv.org/abs/2312.00752.

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight
programmers, 2021. URL https://arxiv.org/abs/2102.11174.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with delta
rule. In International Conference on Learning Representations, 2025. URL https://openreview.
net/forum?id=r8H7xhYPwz.

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra,
and Christopher Ré. Zoology: Measuring and improving recall in efficient language models. In
Proceedings of 12th International Conference on Learning Representations (ICLR). ICLR, 2024a. URL
https://openreview.net/forum?id=LY3ukUANko.

Xin Wang, Hui Shen, Boyuan Zheng, Xueshen Liu, Minkyoung Cho, Zhongwei Wan, Zesen Zhao,
Zhuoqing Mao, Shen Yan, and Mi Zhang. Dynamic linear attention. arXiv preprint arXiv:2606.10650,
2026. URL https://arxiv.org/abs/2606.10650.

Yuqi Pan, Yongqi An, Zheng Li, Yuhong Chou, Ruijie Zhu, Xiaohui Wang, Mingxuan Wang, Jin-
qiao Wang, and Guoqi Li. Scaling linear attention with sparse state expansion. arXiv preprint
arXiv:2507.16577, 2025. URL https://arxiv.org/abs/2507.16577.

Loïc Cabannes, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Matthijs Douze, Maria Lomeli,
Ilze Amanda Auzina, Justin Carpentier, Gabriel Synnaeve, and Hervé Jégou. Sparse delta memory:
Scaling the state of linear rnns through sparsity. arXiv preprint arXiv:2607.07386, 2026. URL
https://arxiv.org/abs/2607.07386.

Jusen Du, Weigao Sun, Disen Lan, Jiaxi Hu, and Yu Cheng. MoM: Linear sequence modeling with
mixture-of-memories. arXiv preprint arXiv:2502.13685, 2025. URL https://arxiv.org/abs/
2502.13685.

Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, and Yoon Kim. Log-linear attention. In
International Conference on Learning Representations, 2026. URL https://openreview.net/
forum?id=mOJgZWkXKW.

Ali Behrouz, Zeman Li, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, and Vahab Mirrokni. Memory
caching: Rnns with growing memory. In Forty-third International Conference on Machine Learning,
2026. URL https://arxiv.org/abs/2602.24281.

Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He,
Enming Yuan, Yuzhi Wang, et al. MoBA: Mixture of block attention for long-context LLMs. arXiv
preprint arXiv:2502.13189, 2025. URL https://arxiv.org/abs/2502.13189.

Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang

..._This content has been truncated to stay below 50000 characters_...
