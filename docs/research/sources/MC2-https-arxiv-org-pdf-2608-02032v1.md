# SOURCE: https://arxiv.org/pdf/2608.02032v1

Preprint

DART: DECODED ATTENTION
OVER RECURRENT
STATES FOR EFFICIENT LONG-CONTEXT SEQUENCE
MODELING

Yixiao Qian1
Song Chen2,∗
Pengkai Wang3
Jiaxu Liu4
Shengze Cai1,∗
Chao Xu1,∗

1College of Control Science and Engineering, Zhejiang University
2Department of Mathematics, National University of Singapore
3Hong Kong Polytechnic University
4School of Science, Huzhou Normal University
yixiaoqian@zju.edu.cn, song.chen@nus.edu.sg, pengkai.wang@connect.polyu.hk
jiaxuliu@zju.edu.cn, shengze cai@zju.edu.cn, cxu@zju.edu.cn
∗Corresponding authors.

ABSTRACT

Modern language models are built primarily from Transformers, recurrent models,
and their hybrid architectures. Transformers rely on token-level attention memo-
ries, while recurrent models such as state space models (SSMs) and linear atten-
tion maintain compact recurrent states. These architectures are typically instan-
tiated separately or interleaved at the layer level, leaving open whether a shared
memory representation can support both recurrent compression and attention-style
retrieval. We study this question through the state space duality (SSD) view of
Mamba-2, where the SSM state can be interpreted as a compressed associative
key–value (KV) cache. We observe that Mamba-2 decodes token-conditioned
values from this state but does not decode token-conditioned keys. Based on
this observation, we propose DART (Decoded Attention over Recurrent sTates),
which retains the chunk state contributions produced by the Mamba-2 chunked
scan as chunk state memories, decodes token-conditioned keys and values from
these memories, and performs state-memory attention (SMA) over the resulting
KV pairs. The retrieved output is then combined with the native Mamba-2 out-
put through a gated residual connection. DART supports practical training by
reusing the Mamba-2 chunked scan and implementing SMA as a FlashAttention-
style computation. Our analysis and experiments show that DART substantially
reduces the length-dependent inference cache compared with a matched attention
baseline (e.g., 75% savings when the chunk size is S = 256 and the state size is
N = 128). Compared with Mamba-2, DART substantially improves associative
recall and retrieval while preserving general language-modeling quality.

1
INTRODUCTION

Modern large language models (LLMs) are built mainly from Transformers, recurrent models, and
their hybrid architectures. Transformers are the dominant backbone of LLMs and can directly attend
to all previous tokens (Vaswani et al., 2017; Brown et al., 2020; Touvron et al., 2023). However, full
attention has quadratic training compute in sequence length, and autoregressive inference requires a
token-level KV cache that grows linearly with context length. Recurrent models provide a represen-
tative alternative for addressing this efficiency bottleneck, ranging from classical RNNs and LSTMs
to modern linear-time language models such as RWKV, RetNet, Mamba, and DeltaNet (Elman,
1990; Hochreiter & Schmidhuber, 1997; Peng et al., 2023; Sun et al., 2023; Gu & Dao, 2023; Dao
& Gu, 2024; Yang et al., 2024). These models process sequences through recurrent state updates
and maintain a compact recurrent state during inference. While these advantages make recurrent
models attractive for long sequences, compressing long contexts into compact states creates a mem-
ory bottleneck. Prior studies on associative recall, copying, and position-structured retrieval show

1

arXiv:2608.02032v1  [cs.LG]  3 Aug 2026


Preprint

that recurrent models can lose long-range recall capacity relative to attention (Arora et al., 2024a;
Jelassi et al., 2024; Waleffe et al., 2024).

These complementary tradeoffs have motivated hybrid architectures that combine attention and re-
current models. By mixing efficient recurrent layers with attention layers, hybrid models can im-
prove the trade-off between compute efficiency and recall. Recent systems have achieved strong
language-modeling and retrieval performance (Lieber et al., 2024; Glorioso et al., 2024; Ren et al.,
2025; Lahoti et al., 2026). Most hybrid designs, however, keep token-level attention memories and
compact recurrent states as separate memory forms and combine them by interleaving or stacking
layers. This leaves open a more basic question: can the same memory representation support both
recurrent compression and attention-style retrieval?

Mamba-2 provides a natural starting point through state space duality (SSD) (Dao & Gu, 2024). SSD
shows that a selective state space model (SSM) layer can be written as structured causal attention,
connecting the SSM coefficients (A, B, C, X) with the attention roles of decay, keys, queries, and
values. In this view, the Mamba-2 state is a compressed KV cache. The recurrent update writes
token-dependent information into this cache, while the readout CtHt extracts the value needed by
the current token. SSD also leads to the chunked scan algorithm of Mamba-2, which makes the
recurrent computation efficient on modern GPUs.

The key observation of this paper is that Mamba-2 reads values from the compressed KV cache but
does not extract keys from it. If keys can also be decoded from this compressed state, then attention
need not operate over all token-level keys and values. It can instead retrieve from a much smaller
set of compressed memories produced by the recurrent scan. This suggests a state-memory attention
(SMA) mechanism between full token-level attention and purely recurrent models: recurrent models
learn to compress input tokens into compact states, and attention retrieves from the compressed
memories according to the current query.

Based on this observation, we propose DART (Decoded Attention over Recurrent sTates), an archi-
tecture that augments Mamba-2 with an SMA branch for attention-style retrieval over compressed
recurrent states. DART keeps the Mamba-2 chunked scan, treats each chunk state contribution as a
chunk state memory, extracts token-conditioned keys and values from these memories, and adds the
retrieved state readout as a gated residual correction to the Mamba-2 output. Our contributions can
be summarized as follows:

• We introduce DART, an architecture that connects recurrent compression with attention-
style retrieval through SMA. The Mamba-2 scan produces chunk state contributions that
serve as chunk state memories, and SMA retrieves from them with token-conditioned keys
and values.

• We show that DART is practical to train by reusing the Mamba-2 chunked scan and imple-
menting SMA with a FlashAttention-style computation (Dao et al., 2022; Dao, 2024).

• We evaluate DART on MQAR, NIAH, real-world retrieval tasks, Pile pretraining, and zero-
shot downstream evaluations. Compared with Mamba-2, DART substantially improves
associative recall and long-context retrieval while preserving language-modeling quality.
Compared with matched attention, DART substantially reduces the length-dependent in-
ference cache without relying on sparse or sliding-window mechanism.

2
BACKGROUND

State Space Models.
Inspired by continuous dynamical systems, state space models (SSMs) map
a sequence x ∈RL to y ∈RL through an implicit latent state h:

˙h(t) = Ah(t) + Bx(t),
y(t) = Ch(t).
(1)

Modern SSMs can be broadly grouped into non-selective and selective variants. Non-selective SSMs
such as S4, S4D, and S5 use input-independent dynamics and exploit their structure for efficient
convolution or scan algorithms (Gu et al., 2021; 2022; Smith et al., 2022). Selective SSMs such
as Mamba, Mamba-2, and Mamba-3 make the dynamics input-dependent, allowing the model to
choose what to write, retain, and read from the state at each token (Gu & Dao, 2023; Dao & Gu,
2024; Lahoti et al., 2026).

2


Preprint

State Space Duality.
Mamba-2 introduces state space duality (SSD), which connects selective
SSMs and attention through an equivalent structured causal attention form. For X, Y ∈RL×P and
scalar transition At = atIN, Mamba-2 can be written as

Ht = AtHt−1 + B⊤
t Xt,
Yt = CtHt,
(2)

where Ht ∈RN×P , Bt ∈R1×N, and Ct ∈R1×N. The update B⊤
t Xt compresses the current input
Xt into the state Ht, while the readout CtHt decodes the value needed by the current token from
this state. We write

Au:v = AvAv−1 · · · Au =

v
�

k=u
Ak,
u ≤v,
(3)

and set Au:v = IN when u > v. Let B, C ∈RL×N stack the token-wise vectors Bt and Ct row-
wise. Without the SSD decay, the product CB⊤X has the same algebraic form as linear attention,
namely QK⊤V with Q = C, K = B, and V = X. SSD adds a structured causal decay mask to
this kernel and writes the Mamba-2 layer as structured causal attention:

Y =
�
LSSD ◦CB⊤�
X,
Q = C,
K = B,
V = X.
(4)

Here (LSSD)ij = �i
k=j+1 ak for i ≥j, and (LSSD)ij = 0 for i < j. SSD shows that Mamba-2 can
be viewed as an iterative form of structured attention. In this view, Ht is a compressed KV cache,
and CtHt extracts the value needed by the current token from this cache.

Memory Issues of Recurrent Models.
Recurrent models reduce memory cost by compressing
history into compact states, but this compression creates a bottleneck for precise long-range recall.
In selective SSMs, for example, selectivity can be viewed as a learned compression policy: dur-
ing training, the model learns which tokens should be preserved in the state and which tokens can
be forgotten. This compression is powerful but learned from training data. When a past token is
processed, the model cannot know which future query will need it. A straightforward way to store
more history is to use a larger state, which increases memory capacity but does not fundamentally
address the lack of attention-like search over past tokens. Several studies have identified this limi-
tation on associative recall, copying, and position-structured retrieval, where recurrent models can
have weaker long-range recall than attention (Arora et al., 2024a; Jelassi et al., 2024; Waleffe et al.,
2024). Recent work has also developed more explicit memory-editing mechanisms: delta-rule recur-
rent models improve compact states by subtracting stale reads, applying targeted updates, combining
such updates with adaptive gating, or decoupling erase and write controls (Yang et al., 2024; 2025;
Hatamizadeh et al., 2026).

Bottlenecks of Long-Context Attention.
Modern LLMs are built primarily on the Transformer
architecture, but extending Transformers to long contexts remains challenging. Many efforts have
therefore focused on reducing the training and inference costs of long-context attention. FlashAt-
tention exploits the GPU memory hierarchy to remove the quadratic activation-memory cost of
traditional attention (Dao et al., 2022; Dao, 2024), but very long-context inference is still limited
by token-level KV-cache memory and bandwidth. Existing systems reduce KV-cache cost through
sliding-window or block-sparse patterns (Beltagy et al., 2020; Zaheer et al., 2020; Xiao et al., 2024),
dynamic cache selection during decoding (Zhang et al., 2023; Li et al., 2024), latent KV represen-
tations such as multi-head latent attention (MLA) (Liu et al., 2024a), or KV-cache quantization (Liu
et al., 2024b). While effective, these approaches mainly optimize, approximate, or redesign the
token-level KV cache itself, rather than exploiting the compact recurrent state already maintained
by the model.

Hybrid Architectures.
The complementary strengths of attention and recurrent models have mo-
tivated hybrid sequence architectures. Existing interleaved systems usually place the two compo-
nents in separate layers, and their attention–recurrent ratio is typically selected empirically, as in
representative systems such as Jamba, Zamba, and Samba (Lieber et al., 2024; Glorioso et al., 2024;
Ren et al., 2025). Recent chunk-level attention methods have also moved beyond token-level KV
memory. Attamba and RAT both perform attention over chunk-level representations: Attamba uses
SSMs to compress token chunks into key-value states, while RAT applies recurrence within chunks
and softmax attention across chunks (Akhauri et al., 2024; Wei et al., 2025). These methods mainly
focus on using recurrent or SSM modules to produce compressed K/V for an attention layer. Figure 1

3


Preprint

illustrates the distinction between DART and interleaved hybrid architectures: interleaved hybrids
insert attention and recurrent modules as separate layers, whereas DART performs attention-style
retrieval inside the recurrent block by reading its chunk state memories.

Interleaved hybrid

SSM block

SWA block

DART memory-level hybrid

SSM
SMA

Attention and recur-
rent modules are in-
serted as separate layers.

Attention-style re-
trieval is performed over
chunk state memories.

Chunk state memory ∆H[c]

Vt,c ∈RP

value features P

key
features N

Ct

Kt,c ∈RN
Et

The same chunk state memory
decodes values through Ct and
token-conditioned keys through Et.

Figure 1: Comparison between interleaved hybrid architectures and DART. Standard interleaved
hybrids place attention and recurrent modules in separate layers, whereas DART integrates attention-
style retrieval within each recurrent block through SMA over chunk state memories. The matrix
view shows that a chunk state contribution ∆H[c] ∈RN×P supports both value readout Ct∆H[c]
and token-conditioned key decoding ∆H[c]Et.

3
METHODOLOGY

DART augments the Mamba-2 chunked scan with state-memory attention (SMA) over its chunk
state contributions, which DART retains as chunk state memories. In the SSD view, the recurrent
state Ht is a compressed KV cache: Mamba-2 reads values through CtHt, while DART also decodes
token-conditioned keys from chunk state memories. We refer to the original Mamba-2 scan and
readout as the SSM branch, and to the retrieval module over chunk state memories as the SMA
branch. More details about Mamba-2 chunked scan are provided in Appendix A.

Chunk State Memories.
For a sequence of length L and chunk size S, DART divides the sequence
into chunks. We write [c] for the set of token indices in the c-th chunk, let c(t) = ⌈t/S⌉denote the
chunk containing token t, and let ∆H[c] denote its chunk state contribution:

∆H[c] =
�

s∈[c]
As+1:e(c)B⊤
s Xs ∈RN×P .
(5)

Here e(c) is the last index in [c], and As+1:e(c) =
��e(c)
k=s+1 ak
�
IN, with the empty product equal
to the identity. In the recurrent scan, we refer to ∆H[c] as a chunk state contribution because it is
the contribution of chunk c to the chunk-boundary state. When retained for SMA, the same tensor
serves as a chunk state memory. The matrix ∆H[c] has a key axis N and a value axis P. Reading it
from the left with Ct ∈R1×N produces a value, while contracting its value axis with Et ∈RP ×1
produces a key:

Ct∆H[c] ∈R1×P
(value-side readout),
∆H[c]Et ∈RN×1
(key-side readout).
(6)

State-Memory Attention.
Let Ut ∈R1×dmodel denote the input representation to the DART block.
For a single SSM head and token t, DART forms three token-conditioned vectors from the block
input Ut:

Qt = UtWQ ∈R1×N,
Ct = UtWC ∈R1×N,
Et = (UtWE)⊤∈RP ×1,
(7)

where WQ, WC ∈Rdmodel×N and WE ∈Rdmodel×P . Here Qt is a query vector in the SSD key-side
space and is used for chunk routing. The vector Ct is the native Mamba-2 read vector shared with

4


Preprint

the SMA branch, and Ct∆H[c] extracts the value needed by token t from the chunk state memory.
Empirically, this sharing is important for optimization, as an independent SMA read vector did not
learn useful recall behavior in our experiments. The vector Et is a value-side verifier, and ∆H[c]Et
turns each chunk state memory into a token-conditioned key.

For each historical chunk c, DART computes:

Vt,c = Ct∆H[c] ∈R1×P ,
Kt,c = ∆H[c]Et ∈RN×1.
(8)

And the chunk logit is:

ℓt,c = QtKt,c
√

N
∈R.
(9)

Gated Residual Integration.
The SMA logits are normalized over historical chunks:

gt,c = softmaxc<c(t)(ℓt,c),
Rt =
�

c<c(t)
gt,cVt,c ∈R1×P .
(10)

For tokens in the first chunk, the historical memory set is empty; in this case we define the SMA
readout as Rt = 0. In practice, the score branch uses RMS-normalized Qt, Et, and chunk states
for stable routing, with details provided in Appendix B. DART then adds the SMA readout as a
scalar-gated residual correction to the SSM branch readout:

Gt = SiLU(UtWG) ∈R,
Yt = CtHt + GtRt,
(11)

where WG ∈Rdmodel×1. We initialize WG to zero, so the SMA residual starts from GtRt = 0 and
is learned as a residual correction during training.

The architecture of DART is shown in Figure 2. The SSM branch follows the Mamba-2 block and
passes chunk state memories to the SMA branch. The SMA branch uses token-conditioned Qt and
Et for retrieval, and its readout is added to the SSM output through a gated residual connection.

N

+
G

SSM
SMA

Conv

A X
B
C
C
Q
E

U

DART

Linear projection

Sequence transformation

Nonlinearity (activation,
normalization, multiplication)

+
Gated residual connection

Figure 2: Architecture of DART. The SSM branch provides chunk state memories to the SMA
branch, whose retrieved readout is combined with the SSM output through a gated residual connec-
tion. In practice, the B, C, Q, and E projections use a single projection group (G = 1) shared
across all heads, analogous to multi-value attention (MVA) in Mamba-2.

4
EFFICIENT TRAINING AND COMPLEXITY ANALYSIS

DART can be trained efficiently because it reuses the Mamba-2 chunked scan to produce chunk
state memories, and implements SMA with FlashAttention-style streaming and recomputation. The
complexity analysis shows that DART can substantially reduce inference cache size. With a common
setting of chunk size S = 256 and state size N = 128, DART saves 75% cache memory compared
with token-level attention.

5


Preprint

Efficient Training.
Figure 3 illustrates the Mamba-2 chunked scan and SMA readout used by
DART. DART reuses the efficient Mamba-2 chunked SSD scan (Dao & Gu, 2024), which produces
both the SSM branch output CtHt and the chunk state contributions ∆H[c]. The SMA branch is
implemented with a custom FlashAttention-style kernel (Dao et al., 2022; Dao, 2024). Following
FlashAttention, the kernel fixes a query tile and streams over historical chunks. For each historical
chunk c, the kernel constructs Kt,c and Vt,c from ∆H[c] on chip, updates the online softmax over
chunks, and accumulates the SMA readout Rt. Appendix C provides the background on online
softmax and recomputation.

(a) Mamba-2 chunked scan

Outputs
Y

States
H[c]

Inputs
X

chunk 1
chunk 2
chunk 3
∆H[1]
∆H[2]
∆H[3]

(b) State-memory attention

∆H[1]
∆H[2]
∆H[3]

Kt,1
Vt,1

Et
Ct

Kt,2
Vt,2

Et
Ct

Kt,3
Vt,3

Et
Ct

State-Memory Attention

Figure 3: Construction and retrieval of chunk state memories in DART. Left: the Mamba-2 chunked
scan computes local outputs and chunk state contributions while propagating compact boundary
states. Right: for each token t, Et and Ct decode a key Kt,c and value Vt,c from each historical
chunk state memory. SMA attends over the decoded key–value pairs and produces the readout Rt.

Algorithm 1 FlashAttention-Style State-Memory Attention

Input: Matrices Q, C ∈RL×N, E ∈RP ×L, chunk state memories ∆H[1], . . . , ∆H[M] ∈RN×P
in HBM, chunk size S, token tile size T.
Output: SMA readout R ∈RL×P and logsumexp Llse ∈RL.

1: Divide the sequence into M = ⌈L/S⌉chunks; divide each chunk into token tiles of size T.
2: for target chunk i = 1, . . . , M do
3:
for token tile τ inside chunk i do
4:
Load Qτ, Cτ ∈RT ×N and Eτ ∈RP ×T from HBM to on-chip SRAM.

5:
On chip, initialize O(0)
τ
= 0 ∈RT ×P , d(0)
τ
= 0 ∈RT , and m(0)
τ
= −∞∈RT .
6:
for historical chunk j < i do
7:
Load ∆H[j] from HBM to on-chip SRAM.

8:
On chip, compute K(j)
τ
= (∆H[j]Eτ)⊤∈RT ×N.

9:
On chip, compute ℓ(j)
τ
= rowsum(Qτ ⊙K(j)
τ )/
√

N ∈RT .

10:
On chip, compute V (j)
τ
= Cτ∆H[j] ∈RT ×P .

11:
On chip, compute m(j)
τ
= max(m(j−1)
τ
, ℓ(j)
τ ) and �g(j)
τ
= exp(ℓ(j)
τ
−m(j)
τ ).

12:
On chip, update d(j)
τ
= em(j−1)
τ
−m(j)
τ d(j−1)
τ
+ �g(j)
τ .

13:
On chip, update O(j)
τ
= diag(em(j−1)
τ
−m(j)
τ )O(j−1)
τ
+ diag(�g(j)
τ )V (j)
τ
.
14:
end for
15:
On chip, compute Rτ = diag(d(i−1)
τ
)−1O(i−1)
τ
, keeping Rτ = 0 when no historical chunk
exists.
16:
On chip, compute (Llse)τ = m(i−1)
τ
+ log d(i−1)
τ
, keeping (Llse)τ = −∞when no histor-
ical chunk exists.
17:
Write Rτ to HBM as the corresponding tile of R.
18:
Write (Llse)τ to HBM as the corresponding tile of Llse.
19:
end for
20: end for
21: Return the SMA readout R and logsumexp Llse.

Algorithm 1 summarizes this forward kernel. For clarity, the algorithm omits the RMS normaliza-
tion applied to the SMA score branch; the normalized formulation is provided in Appendix B. The
forward pass writes only the SMA readout and logsumexp statistics to HBM. The chunk keys, chunk
values, logits, and chunk weights remain on-chip temporaries and are recomputed in the backward
pass.

6


Preprint

Training Complexity.
Let the sequence length be L, chunk size be S, number of chunks be M =
⌈L/S⌉, state size be N, and head dimension be P. In training, the SMA branch lets each token
read from at most M −1 historical chunk state memories. For each token–chunk pair, the kernel
forms a token-conditioned key Kt,c ∈RN and value Vt,c ∈RP through bilinear contractions with
∆H[c] ∈RN×P . The resulting compute cost is

O(LMNP) = O(L2NP/S).
(12)

Compared with Mamba-2, DART adds this SMA branch on top of the linear SSD scan, trading ad-
ditional compute for query-dependent access to historical compressed states. Compared with token-
level attention implemented by FlashAttention, the SMA branch reads M chunk state memories
rather than L token-level KV entries. Its retrieval compute has a leading-order ratio

O(L2NP/S)

O(L2P)
= N

S
(13)

relative to token-level attention per head, and is smaller when N < S.

Inference Cache.
DART is compared against a matched token-level attention baseline during
autoregressive inference. This baseline stores keys and values for each token, giving a length-
dependent KV cache of size O(2LP) per head. DART instead stores one chunk state memory
per chunk, with cache size O(MNP) = O(LNP/S). The cache ratio to this matched attention
baseline is therefore
LNP/S

2LP
= N

2S .
(14)

For example, with chunk size S = 256 and state size N = 128, the chunk state-memory cache is
25% of the token-level KV cache, saving 75% of the length-dependent cache memory.

5
EXPERIMENTS

We evaluate DART on synthetic MQAR to test associative recall, pretrain it on the Pile for 100B
tokens, and evaluate on general language-modeling and retrieval tasks. We further analyze runtime
and memory behavior of the SMA, with more details provided in Appendix D.

5.1
SYNTHETIC TASKS: ASSOCIATIVE RECALL

Following Mamba-2 (Dao & Gu, 2024), we use synthetic associative recall to test whether DART
improves the ability to look up information from context. We adopt multi-query associative recall
(MQAR), which requires an autoregressive model to memorize multiple in-context key–value as-
sociations and predict the associated value when queried by a previously seen key (Arora et al.,
2024a). We compare DART against Mamba-2 under the same sequence length, model width, and
SSD state size, and include the Transformer++ attention baseline (Dao & Gu, 2024) under the same
sequence length and model width. The chunk size is set to S = 16. For DART, this relatively small
chunk size tests whether SMA can retrieve the stored associations from chunk state memories.

32
64
128
Model dimension

0.0

0.2

0.4

0.6

0.8

1.0

Accuracy

Sequence Length: 256

32
64
128
Model dimension

Sequence Length: 512

32
64
128
Model dimension

Sequence Length: 1024

Transformer++
DART (N=16)
DART (N=64)
Mamba-2 (N=16)
Mamba-2 (N=64)

Figure 4: MQAR recall accuracy. Each panel fixes the sequence length L and sweeps the model
width dmodel. DART and Mamba-2 are evaluated with SSD state sizes N ∈{16, 64}. All markers
report test accuracy.

7


Preprint

Results are shown in Figure 4. We sweep sequence length L ∈{256, 512, 1024} and model width
dmodel ∈{32, 64, 128}. For Mamba-2 and DART, we evaluate SSD state sizes N ∈{16, 64}. More
details are provided in Appendix D.1. DART consistently improves MQAR recall over Mamba-2
under matched model width and state size. With N = 64, DART reaches near-perfect accuracy
across all tested sequence lengths and model widths. With the smaller state size N = 16, DART
also remains highly accurate, and even exceeds Mamba-2 with N = 64 in the matched-width con-
figurations. In contrast, Mamba-2 degrades substantially when either the state size or model width is
small, especially at longer sequence lengths. The Transformer++ baseline is strong when the width
is large enough, but is unreliable in low-width and long-sequence settings. These results indicate that
SMA makes compressed SSD memories much more accessible than the native Mamba-2 readout,
especially when recurrent state capacity is limited.

5.2
LANGUAGE MODELING

Following standard LLM protocols, we train and evaluate DART on autoregressive language mod-
eling tasks. We compare pretraining perplexity on the Pile, zero-shot evaluations on standard down-
stream tasks, and retrieval performance on real-world extraction tasks and synthetic NIAH bench-
marks.

General Sequence Modeling.
We start from the baseline Mamba-2 model sizes of 130M, 370M,
and 780M parameters, and construct DART by adding the SMA branch to the same backbone shapes.
All models are pretrained on the Pile for 100B tokens, and the resulting models are evaluated with
Pile validation metrics and zero-shot downstream evaluations. Details of the Pile pretraining setup
are provided in Appendix D.2.

Table 1 reports Pile validation perplexity and accuracy for the trained models, while Table 2 reports
zero-shot downstream results. For a public full-attention reference trained under a comparable token
budget, we also include Pythia checkpoints (Biderman et al., 2023) released at approximately 100B
pretraining tokens in the downstream evaluation. DART adds fewer than 3% parameters through the
SMA branch at these model sizes. This comparison tests whether the added state-memory retrieval
improves recall and retrieval while preserving general sequence-modeling quality. Across the eval-
uated model sizes, DART preserves Pile perplexity, improves Pile validation accuracy, and obtains
higher average zero-shot downstream accuracy than Mamba-2. The gains are especially clear at the
smaller 130M and 370M scales, indicating that DART preserves general language-modeling capac-
ity. Most standard downstream examples in Table 2 are short-context inputs, with prompt-answer
lengths below 256 tokens. Thus, the S = 256 setting mainly evaluates whether training with SMA
preserves the general capability of the SSM branch. The smaller evaluation chunk sizes S = 64
and S = 32 invoke state-memory retrieval on more examples and show that DART maintains down-
stream performance when the SMA branch is active at evaluation time.

Table 1: Model parameters and Pile validation metrics after 100B-token pretraining. DART adds
the SMA branch on top of the Mamba-2 backbone, resulting in a small parameter increase and
competitive Pile validation metrics across the completed model scales. Lower perplexity and higher
accuracy are better.

Mamba-2
DART
Mamba-2
DART
Mamba-2
DART
Model scale
130M
133M
370M
380M
780M
795M
Parameters
128.99M
132.70M
368.35M
378.22M
780.16M
794.95M
Pile PPL ↓
12.36
12.25
10.18
10.10
9.19
9.20
Pile Acc. ↑
50.58
50.75
53.27
53.45
54.86
54.88

Retrieval Capacity.
We evaluate retrieval on synthetic needle-in-a-haystack (NIAH) tasks and
real-world tasks covering semi-structured extraction, passage completion, and question answering;
details are provided in Appendix D.3. As shown in Table 3, DART gives the clearest gains on
context-conditioned extraction tasks. Across all three model scales, DART substantially improves
SWDE and FDA over Mamba-2. The gains on QA-style retrieval tasks are smaller but generally
positive. DART also improves many NIAH settings that require retrieving information from the
provided context. The largest gains appear on NIAH-Single-2 and NIAH-Single-3 at 1024 and 2048
tokens, where Mamba-2 often has very low recall but DART recovers a large fraction of the targets.

8


Preprint

Table 2: Zero-shot downstream evaluation of models pretrained on the Pile. DART uses S = 256 as
the pretraining and main inference setting. We also include S = 64 and S = 32 as evaluation-time
chunk-size ablations of the same trained checkpoint to show how chunk size affects performance.
LAMBADA PPL is reported as perplexity; all other entries are accuracies in percent, and Avg. is the
mean over the seven accuracy metrics. Pythia models are public full-attention references released at
approximately 100B pretraining tokens. Best and second-best entries within each model-size group
are shown in bold and underline, respectively.

Model
Chunk Size
LAMBADA PPL ↓
LAMBADA
HellaSwag
PIQA
ARC-E
ARC-C
WinoGrande
OpenBookQA
Avg.
Mamba-2 130M
–
22.44
37.94
32.34
62.40
39.73
23.21
49.49
28.40
39.07
DART 133M
256
22.77
38.35
32.69
62.95
41.29
24.23
51.14
29.80
40.06
DART 133M
64
25.28
36.66
32.66
62.79
41.25
24.23
51.14
29.80
39.79
DART 133M
32
24.13
37.07
32.68
62.79
41.16
23.81
50.59
30.00
39.73
Pythia 160M
–
35.67
33.30
31.10
62.40
40.70
23.38
50.20
26.40
38.21
Mamba-2 370M
–
13.53
46.85
38.93
66.49
45.41
24.15
51.07
29.40
43.18
DART 380M
256
12.68
47.45
39.17
68.72
43.43
24.66
50.99
30.80
43.60
DART 380M
64
13.18
46.40
39.07
68.72
43.43
24.57
52.17
30.80
43.59
DART 380M
32
12.92
47.22
39.08
68.88
43.52
24.40
52.09
30.80
43.71
Pythia 410M
–
16.20
43.76
37.09
65.56
44.07
24.15
51.07
30.40
42.30
Mamba-2 780M
–
9.50
52.32
44.42
68.34
48.11
25.94
52.88
31.40
46.20
DART 795M
256
9.44
52.22
44.03
69.10
50.21
27.13
54.46
30.80
46.85
DART 795M
64
9.87
51.27
44.16
69.26
50.17
27.05
54.22
31.20
46.76
DART 795M
32
9.76
51.50
44.18
69.15
50.17
27.05
54.46
31.00
46.79
Pythia 1B
–
10.77
50.81
42.47
67.46
47.52
24.91
51.78
30.80
45.11

The hardest 4096-token NIAH-Single-2/3 settings remain challenging for both models. Table 3 re-
ports DART at its pretraining chunk size S = 256 as the main configuration, and includes evaluation-
time chunk-size ablations at S = 64 and S = 32 using the same trained checkpoints. At S = 256,
DART already substantially improves over Mamba-2 across extraction tasks and many NIAH set-
tings. The chunk-size ablations further measure sensitivity to inference granularity: smaller chunks
improve several extraction-heavy metrics such as SWDE and FDA, whereas S = 64 is strongest on
some NIAH-Single-2/3 settings. Pythia remains a strong full-attention reference on several QA and
NIAH columns, indicating that DART narrows the retrieval gap to attention while operating under
the compact-memory constraints of recurrent states.

Table 3: Retrieval evaluation of models pretrained on the Pile.
The real-world suite includes
extraction-heavy tasks (SWDE, SQuAD Completion, and FDA) and QA-style retrieval tasks (Triv-
iaQA, NQ, and DROP). DART uses S = 256 as the pretraining and main inference setting. We
also include S = 64 and S = 32 as evaluation-time chunk-size ablations of the same trained check-
point to show how chunk size affects performance. NIAH reports synthetic retrieval accuracy with
500 examples per context length. All real-world tasks are evaluated with contains accuracy. Pythia
models are public full-attention references released at approximately 100B pretraining tokens and
are evaluated up to their native 2048-token context length. All entries are percentages, and higher is
better. Best and second-best entries within each model-size group are shown in bold and underline,
respectively.

Real-world retrieval
NIAH-Single-1
NIAH-Single-2
NIAH-Single-3
Model
Chunk Size
SWDE
SQD
FDA
TQA
NQ
DROP
1024
2048
4096
1024
2048
4096
1024
2048
4096
Mamba-2 130M
–
17.1
25.6
15.4
38.4
5.7
16.6
98.6
97.0
74.2
2.4
1.0
1.6
1.0
1.0
1.2
DART 133M
256
34.9
23.3
19.7
37.8
9.1
17.3
100.0
99.4
57.2
59.0
47.2
1.6
36.0
46.6
5.0
DART 133M
64
44.8
26.6
26.2
38.8
7.1
18.1
100.0
99.4
65.2
65.6
42.2
1.4
71.4
47.2
4.4
DART 133M
32
46.1
26.7
29.0
38.4
6.4
18.2
100.0
98.4
66.8
50.6
27.6
1.4
37.0
24.8
3.0
Pythia 160M
–
50.9
26.5
37.0
35.4
9.0
18.2
98.2
99.2
–
98.0
83.2
–
52.8
25.4
–
Mamba-2 370M
–
33.2
29.9
16.6
45.9
9.5
20.9
100.0
100.0
95.4
1.4
1.4
1.8
2.6
3.4
2.8
DART 380M
256
46.9
29.8
31.3
46.3
12.6
18.9
100.0
100.0
95.8
13.0
14.0
3.2
2.8
4.4
5.8
DART 380M
64
56.7
31.2
45.8
45.3
11.6
18.9
99.8
99.4
86.6
24.0
32.6
1.8
22.8
18.4
4.6
DART 380M
32
56.2
31.5
50.4
44.4
10.4
19.0
99.6
97.0
65.8
36.0
35.4
1.4
35.4
34.8
3.4
Pythia 410M
–
68.8
32.9
62.7
49.1
14.3
20.8
99.2
99.2
–
99.6
85.4
–
98.2
91.8
–
Mamba-2 780M
–
39.9
32.6
17.2
50.5
11.8
21.3
85.8
91.2
99.8
40.6
24.6
1.8
70.6
19.4
3.6
DART 795M
256
56.5
32.4
28.9
52.0
16.1
23.5
100.0
100.0
99.8
60.6
70.0
1.4
67.6
43.4
2.6
DART 795M
64
63.7
35.5
41.4
51.1
15.7
23.6
100.0
100.0
100.0
83.0
90.6
1.4
99.6
80.6
2.4
DART 795M
32
65.9
34.8
47.9
50.4
14.5
23.4
100.0
100.0
100.0
80.6
89.4
1.4
99.2
88.8
2.4
Pythia 1B
–
68.7
34.8
66.6
52.5
17.6
23.2
99.0
97.4
–
99.2
99.8
–
91.0
97.0
–

5.3
ABLATIONS AND PERFORMANCE ANALYSIS

SMA Branch.
We assess the contribution of the SMA branch through an evaluation-time ablation.
For each trained DART checkpoint, we remove the SMA residual readout during evaluation and keep
only the native SSM branch. The resulting performance change measures how much associative-
recall and retrieval capacity is provided by the SMA branch beyond the recurrent state readout.

9


Preprint

Table 4 reports the retrieval-task ablation, and Table 5 reports the MQAR ablation. Removing SMA
substantially weakens MQAR and real-world extraction performance, indicating that the retrieval
gains mainly come from making chunk state memories explicitly addressable. The NIAH results are
more task-dependent, showing that SMA improves access to chunk state memories but does not by
itself solve all synthetic long-context retrieval variants.

Table 4: Evaluation-time ablation of the SMA branch on NIAH and real-world extraction tasks for
Pile-pretrained checkpoints. The ablated rows use the same trained checkpoints as the full model,
but remove the SMA residual readout during evaluation. All evaluations use S = 256, and all entries
are accuracies in percent.

NIAH-Single-1
NIAH-Single-2
NIAH-Single-3
Real
Model
1024
2048
4096
1024
2048
4096
1024
2048
4096
SWDE
FDA
DART 133M w/o SMA
12.8
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
4.2
0.7
DART 133M
100.0
99.4
57.2
59.0
47.2
1.6
36.0
46.6
5.0
34.9
19.7
DART 380M w/o SMA
100.0
95.8
66.2
4.6
4.2
3.0
2.4
4.6
3.2
13.8
7.1
DART 380M
100.0
100.0
95.8
13.0
14.0
3.2
2.8
4.4
5.8
46.9
31.3
DART 795M w/o SMA
99.8
98.0
90.8
11.6
13.2
3.0
29.6
18.6
3.2
17.3
3.0
DART 795M
100.0
100.0
99.8
60.6
70.0
1.4
67.6
43.4
2.6
56.5
28.9

Sharing the SSM Read Vector.
DART uses the native Mamba-2 read vector Ct as the value-side
read vector in the SMA branch. This design aligns the SMA readout with the SSM branch when
decoding historical chunk state memories. We train otherwise matched variants that replace Ct with
an independently parameterized SMA read vector CSMA
t
= UtW SMA
C
. Table 6 reports the MQAR
comparison in the memory-constrained setting N = 16. The shared design consistently succeeds
across widths and sequence lengths, whereas the independently parameterized read vector fails to
learn useful recall behavior in this setting and stays at zero accuracy. This shows that aligning the
SMA value readout with the native SSM readout is important for decoding historical chunk state
memories.

Table 5:
Evaluation-time ablation of the
SMA branch on MQAR. Each row uses a
DART model with SSD state size N = 16
and chunk size S = 16. All entries are test
accuracies in percent.

Model
dmodel
L = 256
L = 512
L = 1024
DART w/o SMA
32
12.40
0.30
0.11
DART
32
96.70
96.67
95.94
DART w/o SMA
64
0.55
1.07
0.06
DART
64
99.70
99.83
99.88
DART w/o SMA
128
0.23
0.08
0.08
DART
128
99.94
99.99
99.98

Table 6: Ablation of Ct sharing on MQAR with
SSD state size N = 16 and chunk size S = 16.
All entries are test accuracies in percent.

Model
dmodel
L = 256
L = 512
L = 1024
DART, shared Ct
32
96.70
96.67
95.94
DART, independent CSMA
t
32
0.00
0.00
0.00
DART, shared Ct
64
99.70
99.83
99.88
DART, independent CSMA
t
64
0.00
0.00
0.00
DART, shared Ct
128
99.94
99.99
99.98
DART, independent CSMA
t
128
0.00
0.00
0.00

Kernel Forward Time and Inference Cache.
We evaluate the system efficiency of the SMA
branch in terms of kernel forward time and inference cache size. For runtime, we report forward
time for the kernels under matched batch size, head dimension, state size, chunk size, and bf16
inputs on a single A800 80GB GPU. The comparison measures FlashAttention-2 with precomputed
Q, K, V , SSD with precomputed X, ∆, A, B, C, and SMA with precomputed Q, E, C, chunk state
memories {∆H[c]}. For inference, we compare the persistent decode cache associated with these
sequence modules: attention KV cache for FlashAttention-2, conv/recurrent state cache for SSD,
and chunk state-memory cache for SMA. As discussed in Section 4, the ratio of the chunk state-
memory cache to the matched attention KV cache is N/(2S) per head, giving a 25% ratio when
N = 128 and S = 256.

6
CONCLUSION

We presented DART, an architecture that augments Mamba-2 with state-memory attention over com-
pact recurrent states. The Mamba-2 chunked scan produces chunk state contributions that DART
retains as chunk state memories. SMA decodes token-conditioned keys and values from these mem-
ories and adds its readout as a gated correction to the SSM branch. Experiments on MQAR, Pile
pretraining, downstream evaluation, retrieval benchmarks, and efficiency analysis show that DART

10


Preprint

1k
2k
4k
8k
16k
32k
64k
Sequence length

10−1

100

101

102

Forward time (ms)

-37.5%

Kernel Forward

1k
2k
4k
8k
16k
32k
64k
Sequence length

10−2

10−1

100

101

Cache allocation (GB)

-74.0%

Inference Cache

FlashAttention-2
SSD
SMA

Figure 5: Kernel forward time and inference cache comparison. Left: forward time of the kernels
on an A800 80GB GPU with batch size 1, 24 heads, head dimension 64, SSD state size 128, chunk
size 256, and bf16 inputs. Right: measured CUDA allocation for persistent inference cache tensors
in a 24-layer, 24-head model. The results show that SMA consistently has lower kernel forward time
than FlashAttention-2 in this setting, while also requiring substantially smaller persistent inference
cache than token-level attention.

improves associative recall and retrieval while preserving general language-modeling quality. These
results demonstrate that recurrent compression and attention-style access can be supported by the
same recurrent-state representation.

REFERENCES

Yash Akhauri, Safeen Huda, and Mohamed S Abdelfattah.
Attamba: Attending to multi-token
states. arXiv preprint arXiv:2411.17685, 2024.

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Y Zou, Atri
Rudra, and Christopher R´e. Zoology: Measuring and improving recall in efficient language mod-
els. In International conference on learning representations, volume 2024, pp. 15664–15730,
2024a.

Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley,
James Zou, Atri Rudra, and Christopher R´e. Simple linear attention language models balance
the recall-throughput tradeoff. arXiv preprint arXiv:2402.18668, 2024b.

Simran Arora, Aman Timalsina, Aaryan Singhal, Sabri Eyuboglu, Xinyi Zhao, Ashish Rao, Atri
Rudra, and Christopher R´e. Just read twice: closing the recall gap for recurrent language models.
In Workshop on Efficient Systems for Foundation Models II@ ICML2024, 2024c.

Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer.
arXiv preprint arXiv:2004.05150, 2020.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric
Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al.
Pythia: A suite for analyzing large language models across training and scaling. In International
conference on machine learning, pp. 2397–2430. PMLR, 2023.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. PIQA: Reasoning about physical
commonsense in natural language. In Proceedings of the AAAI conference on artificial intelli-
gence, volume 34, pp. 7432–7439, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning chal-
lenge. arXiv preprint arXiv:1803.05457, 2018.

11


Preprint

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In Inter-
national Conference on Learning Representations, volume 2024, pp. 35549–35562, 2024.

Tri Dao and Albert Gu.
Transformers are SSMs: Generalized models and efficient algorithms
through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-
efficient exact attention with IO-awareness. Advances in neural information processing systems,
35:16344–16359, 2022.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner.
DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In
Proceedings of the 2019 Conference of the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp.
2368–2378, 2019.

Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179–211, 1990.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason
Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The Pile: An 800GB dataset of diverse
text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam
Ibrahim, and Beren Millidge.
Zamba: A compact 7B SSM hybrid model.
arXiv preprint
arXiv:2405.16712, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv
preprint arXiv:2312.00752, 2023.

Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured
state spaces. arXiv preprint arXiv:2111.00396, 2021.

Albert Gu, Karan Goel, Ankit Gupta, and Christopher R´e. On the parameterization and initialization
of diagonal state space models. Advances in neural information processing systems, 35:35971–
35983, 2022.

Qiang Hao, Rui Cai, Yanwei Pang, and Lei Zhang. From one tree to a forest: a unified solution for
structured web data extraction. In Proceedings of the 34th international ACM SIGIR conference
on Research and development in Information Retrieval, pp. 775–784, 2011.

Ali Hatamizadeh, Yejin Choi, and Jan Kautz. Gated DeltaNet-2: Decoupling erase and write in
linear attention. arXiv preprint arXiv:2605.22791, 2026.

Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural computation, 9(8):
1735–1780, 1997.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Jia Fei, Zhang
Yang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language
models? arXiv preprint arXiv:2404.06654, 2024.

Samy Jelassi, David Brandfonbrener, Sham M Kakade, and Eran Malach. Repeat after me: Trans-
formers are better than state space models at copying. arXiv preprint arXiv:2402.01032, 2024.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meet-
ing of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1601–1611,
2017.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a
benchmark for question answering research. Transactions of the Association for Computational
Linguistics, 7:453–466, 2019.

12


Preprint

Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and
Albert Gu. Mamba-3: Improved sequence modeling using state space principles. arXiv preprint
arXiv:2603.15569, 2026.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle
Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before
generation. Advances in Neural Information Processing Systems, 37:22947–22970, 2024.

Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi,
Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. Jamba: A hybrid transformer-
Mamba language model. arXiv preprint arXiv:2403.19887, 2024.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong
Ruan, Damai Dai, Daya Guo, et al. DeepSeek-V2: A strong, economical, and efficient mixture-
of-experts language model. arXiv preprint arXiv:2405.04434, 2024a.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi
Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2-bit quantization for KV cache. arXiv
preprint arXiv:2402.02750, 2024b.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct elec-
tricity? a new dataset for open book question answering. In Proceedings of the 2018 conference
on empirical methods in natural language processing, pp. 2381–2391, 2018.

Denis Paperno, Germ´an Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi,
Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fern´andez. The LAMBADA dataset:
Word prediction requiring a broad discourse context. In Proceedings of the 54th annual meeting
of the association for computational linguistics (volume 1: Long papers), pp. 1525–1534, 2016.

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman,
Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, et al. RWKV: Reinventing RNNs for
the transformer era. In Findings of the association for computational linguistics: EMNLP 2023,
pp. 14048–14077, 2023.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions
for squad. In Proceedings of the 56th Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers), pp. 784–789, 2018.

Liliang Ren, Yang Liu, Yadong Lu, Chen Liang, Weizhu Chen, et al. Samba: Simple hybrid state
space models for efficient unlimited context language modeling. In International Conference on
Learning Representations, volume 2025, pp. 53551–53575, 2025.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An ad-
versarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for
sequence modeling. arXiv preprint arXiv:2208.04933, 2022.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and
Furu Wei. Retentive network: A successor to transformer for large language models. arXiv
preprint arXiv:2307.08621, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee
Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and
efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural informa-
tion processing systems, 30, 2017.

Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert
Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, et al. An empirical study of Mamba-
based language models. arXiv preprint arXiv:2406.07887, 2024.

13


Preprint

Xiuying Wei, Anunay Yadav, Razvan Pascanu, and Caglar Gulcehre. RAT: Bridging RNN efficiency
and attention accuracy via chunk-based sequence modeling. Advances in Neural Information
Processing Systems, 38:34735–34762, 2025.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks. In International Conference on Learning Representations,
volume 2024, pp. 21875–21895, 2024.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transform-
ers with the delta rule over sequence length. Advances in neural information processing systems,
37:115491–115522, 2024.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with
delta rule. In International Conference on Learning Representations, volume 2025, pp. 29687–
29707, 2025.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago
Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big Bird: Transformers for
longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi.
HellaSwag: Can a
machine really finish your sentence? In Proceedings of the 57th annual meeting of the association
for computational linguistics, pp. 4791–4800, 2019.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,
Yuandong Tian, Christopher R´e, Clark Barrett, et al. H2O: Heavy-hitter oracle for efficient gen-
er

..._This content has been truncated to stay below 50000 characters_...
