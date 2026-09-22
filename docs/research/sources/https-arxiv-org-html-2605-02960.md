# SOURCE: https://arxiv.org/html/2605.02960

^1^^1^footnotetext: Part of work was done during Zhaoyuan’s internship at Snowflake AI Research.

## 1 Introduction

Large language models (LLMs) are increasingly used beyond text generation for a class of discriminative tasks—classification [61, 52], moderation [29], recommendation [69], factual verification [44]—where the answer is one choice from a small, predefined candidate set and is fully determined by the logits of a single forward pass, with no autoregressive decoding required. Measurements from an anonymized production cluster (Fig. 1) show that such __prefill-only workloads__ already account for 65.3% of all input tokens served. Their system profile differs from generative serving along three axes: they are throughput-oriented rather than latency-sensitive, run at extremely large batch sizes that yield long compute-bound forward passes, and exhibit abundant prefix sharing (shared system prompts, user profiles, document headers) both within and across batches.

Meanwhile, state-of-the-art open-weight LLMs have shifted to the Mixture-of-Experts (MoE) architecture [4, 57], whose total parameter counts grow far faster than per-GPU HBM (§4.1) and force operators into distributed execution purely to hold the weights. Unfortunately, distributed parallelism relieves memory pressure only by introducing __three redundancies__ on the MoE prefill critical path: redundant computation from sub-saturated per-device GEMMs and routing-imbalance stragglers, redundant memory from full-resident expert weights, $k$-way activation expansion, and duplicated prefix KV caches, and redundant communication from two synchronous AllToAll on every MoE layer under expert parallelism. As we quantify in §4, these overheads collectively push MoE prefill model FLOPs utilization (MFU) below $16\%$ across all mainstream parallel strategies on 8$\times$A100, with several baselines plateauing or regressing beyond four GPUs.

These redundancies are not inherent to MoE serving—they are inherited from a decoding-era design choice that __couples expert placement with synchronous activation routing__. The long, compute-bound forward passes of large-batch prefill open a wide per-layer compute window; if expert weights are streamed into the window in the background, per-layer AllToAll can be removed from the critical path entirely. Concretely, experts are gathered __by weight__ rather than routed __by activation__, so every GPU holds the complete expert set for the current layer, and dispatch becomes a local operation.

We propose MoE-Prefill, a prefill-only serving system that eliminates all three redundancies on the MoE prefill critical path. Its core execution paradigm AsyncEP (Asynchronous Expert Parallelism) replaces per-layer activation AllToAll with background expert-weight AllGather, overlapped with computation and optionally extended to CPU-DRAM offloading. The overlap is governed by a __saturation threshold__ $T$ (§6.2) derived analytically from hardware, workload, and model configuration. The MoE-Prefill frontend enables $T$ as a per-GPU batching rule and performs prefix-aware routing, and true-FLOPs compute tracking on top of it, closing the frontend–backend co-design loop.

We implement MoE-Prefill on top of vLLM and evaluate it across four hardware/precision combinations (8$\times$A100 BF16, 8$\times$H100 BF16/FP8, 8$\times$H200 FP8) against the five distributed MoE strategies supported by vLLM. On an aggregated real-world prefill-only workload, MoE-Prefill achieves 1.35–1.37$\times$ end-to-end throughput over the strongest baseline in every hardware/precision/parallel-degree cell, and up to 1.59$\times$ on long-context synthetic workloads. Its feasible deployment envelope widens from “$\geq$4 GPUs required to hold Qwen3-235B-A22B” to “1–8 GPUs” (a $4\times$ broader hardware range), while sustaining 29.8–36.2% per-GPU MFU.

Contributions. This paper makes four contributions:

*    •

  A workload characterization formalizing the atomic __prefill-as-a-service__ operation and identifying three structural opportunities of prefill-only serving (§3).
*    •

  A quantitative analysis of MoE prefill bottlenecks unified in a per-device per-layer communication table over seven parallel-strategy combinations (§4).
*    •

  AsyncEP, an MoE execution paradigm that gathers experts __by weight__ rather than __by activation__, removing per-layer AllToAll and routing-imbalance stragglers from the critical path (§6).
*    •

  MoE-Prefill, a co-designed serving system bound by a physically-derived saturation threshold, delivering $1.35$–$1.37\times$ throughput and a $4\times$ broader deployable envelope on large MoE models (§7–§8).

## 2 Background

### 2.1 LLM Inference: Prefill and Decoding

LLM serving executes in two phases: prefill processes the entire input context of length $S$ in a single forward pass, while decoding autoregressively emits output tokens one at a time. To amortize attention across decoding steps, engines such as vLLM [36] and SGLang [71] cache per-layer key/value projections, whose footprint scales linearly with batch size and sequence length (full memory accounting in §4.1).

### 2.2 Prefill-Only LLM Workloads

A growing class of production tasks—classification, moderation, recommendation, verification—can be answered from the logits of a single prefill pass, with no autoregressive decoding. We characterize these __prefill-only workloads__ in §3.

### 2.3 Mixture-of-Experts (MoE) Models

Most recent open-weight LLMs—Qwen3 [65], DeepSeek-V3 [10], Mixtral [30], gpt-oss [1]—adopt the Mixture-of-Experts (MoE) architecture to scale capacity without proportional growth in per-token FLOPs. In a MoE Transformer, the dense FFN of (some or all) decoder blocks is replaced by $E$ parallel experts, each a two-layer MLP; a lightweight __router__ dispatches each token to its top-$k$ experts (typically $k{=}2$–$8$) and sums their outputs. Sparse activation packs hundreds of billions of total parameters while activating only a small subset per token, but the cost is paid on the __system__ side: total parameters far exceed per-GPU HBM, and data-dependent top-$k$ routing is typically imbalanced across experts and GPUs.

A structural feature of MoE—central to the design space of this paper—is that attention weights and expert weights are disjoint parameter groups. The two stacks can therefore adopt __independent__ parallelization strategies, a freedom unavailable in dense models; this decoupling makes combinations such as DP $\times$ EP or TP $\times$ EP meaningful and is exploited by our design in §6.

### 2.4 Distributed Parallelism for MoE Serving

Because large MoE models rarely fit on a single GPU, serving systems resort to distributed execution. Five strategies are in common use: DP [40] (replicate weights, shard requests), TP [59] (shard weight matrices), PP [26, 45] (pipeline layers), SP [41, 35] (shard along sequence), and the MoE-specific EP [37] (shard experts), each with distinct on-path communication semantics. Because attention and expert stacks are disjoint (§2.3), MoE systems can compose different parallel strategies for the two stacks; §4 (Table 1) gives a per-device per-layer comparison of the relevant combinations.

## 3 Workload Analysis

### 3.1 Prefill-Only Workloads in Practice

Definition and atomic operation. __Prefill-only workloads__ are inference tasks whose output is selected from a predefined candidate token set via the logits of a single prefill forward pass, without autoregressive decoding. We formalize the atomic __prefill-as-a-service__ operation: given an input context $c$ and a candidate token set $\mathcal{C}=\{t_{1},\dots,t_{k}\}$, the system performs one prefill pass and returns $t^{*}=\arg\max_{t\in\mathcal{C}}\text{logit}(t\mid c)$. Unlike autoregressive serving, this operation has no iterative decoding and no per-token KV updates, making it inherently parallelizable and throughput-friendly.

Representative categories. A broad range of industrial tasks reduces to this atomic operation, directly or via simple reformulation:

*    •

  Single-Token Classification (binary sentiment, toxicity, factual verification): the candidate set is a small set of single-token labels, served by one atomic call.
*    •

  Single-Choice Selection (multiple-choice QA, intent classification): assigning each candidate a symbolic identifier (A/B/C/D) in the prompt reduces the task to selecting one identifier token—again one atomic call.
*    •

  Multi-Selection (multi-label classification, recommendation, ranking): we __decompose__ one $N$-candidate request into $N$ independent binary sibling requests, served by $N$ parallel atomic operations. The $N$ siblings share the same prefix; prefix sharing (§3.3) absorbs the duplication, keeping total prefill compute comparable to a single undecomposed pass while eliminating decoding entirely.

Insight #1. A wide class of production LLM workloads can be served through prefill-as-a-service, motivating serving systems specialized for prefill-only execution.

### 3.2 Workload Characteristics

Unlike latency-sensitive interactive serving, prefill-only workloads are __throughput-oriented__: the primary objective is to minimize total time over a large batch, not per-request latency. These workloads run in offline pipelines [72, 63] that process millions to billions of requests in bulk, with input lengths spanning short reviews to long documents of tens of thousands of tokens. Each prefill forward therefore processes an enormous token count and runs in a heavily compute-bound interval, fundamentally different from the memory-bandwidth-bound per-step of autoregressive decoding.

Insight #2. The long compute-bound forward passes of large-batch prefill open an opportunity to overlap weight streaming with computation, motivating weight offloading as a first-class design choice that unlocks MoE models larger than GPU memory.

### 3.3 Prefix Sharing Opportunities

Prefix reuse appears at two scales. In-batch, multi-selection decomposition (§3.1) and similar reformulations produce batches in which many requests share a common prefix (e.g., $N$ candidate-scoring requests all prefixed by the same user profile); since transformer attention is causal, the prefix attention output is identical across siblings, so computing it once eliminates nearly all prefix-stage compute and collapses $N$ redundant copies of the prefix KV into a single HBM copy. Cross-batch, production pipelines repeatedly process requests built from shared templates, stable user profiles, or document headers (e.g., the same user profile scored against successive candidate sets); caching the prefix KV across batches skips both recomputation and memory-bandwidth cost for subsequent same-prefix requests, and the cached prefix occupies HBM only once.

Insight #3. Prefill-only workloads exhibit abundant prefix sharing, both within and across batches, motivating prefix-aware scheduling and caching as a design principle.

## 4 System Analysis

Having identified the opportunities of prefill-only workloads (§3), we analyze whether existing MoE serving systems can capture them. We answer negatively, identifying a causal chain of three inefficiencies: __memory pressure forces distributed parallel execution, which in turn degrades compute efficiency and introduces heavy on-path communication__. Throughout, we use the following notation: $B$ batch size, $S$ sequence length, $L$ layers, $H$ hidden size, $N_{kv}$ KV heads, $d_{h}$ head dim, $h$ MoE intermediate size, $k$ top-$k$ routing fan-out, $b$ bytes/element (BF16: $b{=}2$, FP8: $b{=}1$).

### 4.1 Memory Pressure

Three components contend for GPU HBM during prefill-only MoE serving:

Model weights. MoE scales by adding experts and widening expert MLPs, so total parameters grow far faster than per-GPU HBM capacity across hardware generations (Figure 2). The gap persists even under FP8, placing moderately sized MoE deployments beyond the reach of a single GPU.

KV caches. $V_{\text{KV}}=2LBS\,N_{kv}d_{h}\,b$, linear in both $B$ and $S$. Under the long-context, large-batch regime of §3, KV grows without bound with batch and sequence length, rapidly overtaking weights to become the dominant HBM consumer.

Activations. Approximating peak activation by the two largest MLP intermediates, $V_{\text{act}}\approx 2\,BS\,k\,h\,b$—linear in $B{\cdot}S$ and amplified by a factor of $k$ from top-$k$ routing, unlike dense models where activations traverse a single MLP path. Under aggressive batching and wide MoE, activations themselves consume a non-trivial slice of HBM.

Insight #4. Large-MoE prefill serving faces severe HBM pressure from weights, KV caches, and activations simultaneously, forcing existing systems into distributed parallel execution.

### 4.2 Computation Inefficiency

Once distributed execution is chosen, compute efficiency degrades at two levels: insufficient per-device token batches (model level) and MoE routing imbalance (layer level).

Model level: MFU requires large token batches. Model FLOPs Utilization [7] ($\text{Achieved FLOPs}/\text{Peak FLOPs}$) is primarily determined by the effective token batch $B{\times}S$ and the resulting GEMM geometry—larger batches improve arithmetic intensity and Tensor Core occupancy.

Figure 3 shows that small chunks (2K, by default on A100) constrain the per-kernel token batch and reduce GEMM dimensions; larger or disabled chunking recovers MFU, and longer contexts further improve it by enlarging per-request token volume. In short, __MFU degrades whenever distributed execution or aggressive chunking drives per-device token batches below the GEMM-saturation threshold.__

Layer level: expert-load imbalance fragments GEMMs. Even with a large global batch, MoE top-$k$ routing distributes tokens unevenly across experts [74, 75].

Figure 4 shows that in Qwen3-30B-A3B the max/min token count across experts reaches $16.15\times$ aggregated over all 48 MoE layers (Fig. 4(a)) and is even more skewed within individual layers (Fig. 4(b)). This imbalance (i) makes per-expert GEMMs small and irregular, reducing Tensor Core utilization; and (ii) turns the heaviest-loaded GPUs into stragglers that dictate layer completion under EP.

Insight #5. Distributed parallel execution degrades compute efficiency through small per-device batches and MoE routing-induced expert-load imbalance.

Table 1: Per-device, per-layer communication cost of distributed parallel strategies for MoE prefill-only inference.
Notation follows §4 with $P$ denoting the parallel degree. The communication sizes assume ring-style implementations of collective primitives and ignore routing imbalance across experts. The last row, AsyncEP, is our proposed design, detailed in §5.

|Attention Parallelism | Expert       Parallelism | Collective Operations                 (ring-style) | Per-Device Forward Communication         (bytes) | Communication Volume |
| --- | --- | --- | --- | --- |
|DP | DP | None | $\approx 0$ | Low |
|DP | TP | 1 $\times$ All-Gather + 1 $\times$ Reduce-Scatter | $\approx 4\cdot\frac{P-1}{P}\cdot B\cdot S\cdot H\cdot b$ | Medium |
|DP | EP | 2 $\times$ All-to-All | $\approx 4\cdot k\cdot\frac{P-1}{P}\cdot B\cdot S\cdot H\cdot b$ | High |
|TP | TP | 2 $\times$ All-Reduce | $\approx 4\cdot\frac{P-1}{P}\cdot B\cdot S\cdot H\cdot b$ | Medium |
|TP | EP | 1 $\times$ All-Reduce + 2 $\times$ All-to-All | $\approx(2+4k)\cdot\frac{P-1}{P}\cdot B\cdot S\cdot H\cdot b$ | High |
|PP | PP | 2 $\times$ Send/Recv | $\approx 2\cdot B\cdot S\cdot H\cdot b$ | Medium |
|SP | TP | 2 $\times$ All-Gather + 1 $\times$ Reduce-Scatter | $\approx(2\cdot N_{kv}\cdot d_{h}+4\cdot H)\cdot\frac{P-1}{P}\cdot B\cdot S\cdot b$ | High |
|SP | EP | 1 $\times$ All-Gather + 2 $\times$ All-to-All | $\approx(2\cdot N_{kv}\cdot d_{h}+4\cdot k\cdot H)\cdot\frac{P-1}{P}\cdot B\cdot S\cdot b$ | High |
|DP | AsyncEP | 1 $\times$ All-Gather (Async) | $\bm{\approx 0}$ | Low |

### 4.3 Communication Overhead

Distributed execution introduces two on-path overheads per layer: __collective primitives__ (All-Reduce, All-to-All) that exchange activations, and __auxiliary kernels__ (reshaping, token permutation/alignment) that prepare tensors for them. Both sit on the critical path of every layer.

A distinctive property of MoE serving—absent in dense models—is that attention weights and expert weights are disjoint parameter groups, so the attention stack and the expert stack can adopt __different__ parallel strategies. The common choices are DP (replicate weights, shard requests), TP (shard weight matrices, All-Reduce per layer), PP (partition layers into stages with Send/Recv at boundaries), SP (shard along sequence, All-Gather/Reduce-Scatter to reconstruct full-sequence ops), and the MoE-specific EP (shard experts with two All-to-All per MoE layer).

Table 1 summarizes per-device, per-layer communication volume for each combination. DP$\times$DP is communication-free but memory-expensive; the strategies feasible for large MoE models (DP$\times$EP, TP$\times$EP) incur traffic proportional to $B\cdot S\cdot H$, amplified under EP by the top-$k$ fan-out. The last row, MoE-Prefill’s DP$\times$AsyncEP, achieves near-zero on-path traffic and is a design preview.

Insight #6. Distributed parallel execution introduces frequent on-path collective communication and auxiliary kernels, particularly under expert parallelism where per-layer All-to-All traffic scales with the routing fan-out $k$.

## 5 MoE-Prefill: Design Overview

§3–§4 surface six insights: three __opportunities__ from prefill-only workloads (Insights #1–#3) and three __constraints__ in existing MoE serving (Insights #4–#6). MoE-Prefill distills these into three design principles (§5.1), realized by a two-tier architecture (§5.2) whose frontend and backend are co-designed (§5.3).

### 5.1 Design Principles

MoE-Prefill targets three classes of redundancy on the MoE prefill-only critical path, mirroring its title slogan __Zero Redundancy Overheads__: P1 (computation): strip decoding-stage machinery and co-locate same-prefix requests so each prefix is executed exactly once (Insights #1, #3); P2 (memory): exploit the compute window to offload expert weights to CPU DRAM and stream them back asynchronously, share prefix KV via affinity routing, and optionally disable KV storage (Insights #2, #4); P3 (communication): gather experts __by weight__ rather than routing __by activation__, replacing on-path collectives with background weight transfers fully overlapped with compute (Insights #5–#6).

### 5.2 System Architecture and Workflow

MoE-Prefill is a two-tier serving system (Figure 5). The frontend (§7) first normalizes each request—single-token classification, single-choice selection, or multi-selection decomposed into binary siblings (§3)—into prefill-only form, then assembles per-GPU batches under three simultaneous rules: co-locate same-prefix requests to maximize KV reuse, measure load in true FLOPs after prefix-sharing credit, and stop admitting new requests to a GPU once its accumulated FLOPs reach the backend-derived __saturation threshold__ $T$ (Eq. (1)). The backend (§6) runs each batch as pure DP attention with __asynchronous expert parallelism__ (AsyncEP): expert weights for upcoming MoE layers are gathered in the background via D2D AllGather over NVLink (and optionally prefetched from CPU DRAM via PCIe), while the current layer computes. Because $T$ is enforced by the frontend, the per-layer compute window always dominates the slowest ongoing transfer, so no collective appears on the critical path. A KV-cache-free mode further disables KV storage for workloads without prefix reuse.

### 5.3 Frontend–Backend Co-Design

Most LLM serving systems treat the scheduler and the execution engine as independently tunable components connected only by a request queue. MoE-Prefill departs from this pattern: frontend and backend share three explicit interfaces. (1) The __saturation threshold $T$__ is defined by the backend from measurable hardware quantities (Eq. (1)) and enforced by the frontend as the per-GPU admission condition (§7.4). (2) __Prefix-KV affinity__: the frontend’s prefix-aware routing and the backend’s KV block table operate on the same per-GPU cache, so routing decisions and KV layout share a single source of truth (§7.2). (3) A __true-FLOPs cost model__ measures load after prefix-sharing credit, exactly matching the work the backend performs—one prefix pass plus per-sibling suffix passes (§7.3). All three interfaces originate in the backend, so we present it (§6) before the frontend (§7).

## 6 Backend: Asynchronous Expert-Parallel Execution

The backend of MoE-Prefill executes the computation scheduled by the frontend, with the goal of keeping every GPU on useful compute rather than waiting on data transfers. We present it as a progression of execution modes, each relaxing one more constraint from §4.

### 6.1 Design Constraints

Building on Insights #4–#6 (§4), the backend must jointly satisfy, in descending order of priority: C1 (feasibility) the model’s weights, KV cache, and activations fit per-GPU HBM; C2 (performance) no synchronous collective sits on the per-layer critical path; and C3 (balance) no GPU becomes a per-layer straggler under expert-load skew. C1 determines __whether__ the model runs, C2 __how fast__ it runs, and C3 __how evenly__ GPUs utilize compute. Conventional synchronous DP+EP (Figure 6) violates all three: every GPU permanently stores $\frac{1}{N}$ of every layer’s experts (C1), two All-to-Alls per layer block computation (C2), and the barrier amplifies routing skew into stragglers (C3). MoE-Prefill’s backend addresses them through a progressive design: AsyncEP via D2D (§6.2) eliminates C2 and C3; AsyncEP with offloading (§6.3) additionally relaxes C1 on __weights__; and a KV-cache-free mode (§6.3) further relaxes C1 on __KV__ for prefix-sparse workloads.

### 6.2 Asynchronous Expert Parallelism

AsyncEP eliminates activation routing from the critical path by restructuring how expert weights are managed. The key insight is that in prefill-only serving, a GPU does not need to hold experts for all layers simultaneously—it only needs the __complete__ set of experts for the layer currently being computed.

Weight layout and execution. As shown in Figure 7(a), each GPU holds a full replica of attention weights (identical to DP+EP) and $\frac{1}{N}$ of the expert weights partitioned by expert index, with all GPUs additionally replicating the __complete__ expert set for the first MoE layer. After computing attention locally, each GPU evaluates the current MoE layer entirely on-device and simultaneously initiates a background D2D AllGather over NVLink to collect the next layer’s complete expert weights from the $\frac{1}{N}$ shards distributed across GPUs. By the time the current layer finishes, the gather has completed and the GPU is ready to execute the next layer locally; the process repeats layer by layer.

Why the overlap works: the saturation threshold $T$. For a layer’s D2D AllGather to be fully hidden, the layer’s compute must last at least as long as the gather. Let $t_{\text{EP}}$ denote the time for the slowest EP data transfer per layer (D2D AllGather, or H2D PCIe transfer when offloading is enabled in §6.3) and $F_{\text{GPU}}$ the GPU’s peak FLOP rate. A GPU’s per-batch compute fully overlaps the transfer whenever its FLOPs exceed the __saturation threshold__

$$T=t_{\text{EP}}\times F_{\text{GPU}}\times\gamma,$$

where $\gamma\geq 1$ (e.g., $1.2$) absorbs transient jitter. $T$ is a physical quantity computed once at startup from hardware and model configuration. Prefill-only workloads naturally exceed $T$ because aggressive batching produces long, compute-bound per-layer forward passes, and the frontend (§7) enforces $T$ as the per-GPU admission condition, closing the co-design loop. Two All-to-Alls per MoE layer thus disappear from the critical path (C2), and routing imbalance no longer creates per-layer stragglers because all top-$k$ dispatch is local (C3).

### 6.3 AsyncEP with Offloading and KV-Cache-Free Execution

The D2D-only AsyncEP resolves C2 and C3 but does not fully address C1: each GPU must still permanently store $\frac{1}{N}$ of expert weights for __every__ layer, a footprint that for very large MoE models crowds out the HBM available for KV caches and activations. MoE-Prefill therefore extends AsyncEP with two independent __memory knobs__—hybrid weight offloading on the __weights__ dimension and KV-cache-free execution on the __KV__ dimension—both shrink the per-GPU HBM footprint and extend deployment to cheaper hardware.

Weights dimension: hybrid offloading. As shown in Figure 7(b), each GPU now retains $\frac{1}{N}$ expert weights for only a small number of upcoming layers; the complete expert weights for all layers live in CPU pinned memory as an offloaded backing store. Execution becomes a __two-channel pipeline__: while the current layer computes, a D2D AllGather over NVLink assembles the immediately-next layer’s complete expert set from shards already resident on each GPU, __and__ an H2D PCIe prefetch concurrently transfers $\frac{1}{N}$ shards for further-ahead layers from CPU memory, staging them for future AllGathers. The NVLink and PCIe channels run in parallel, so by the time each AllGather starts, the shards it needs are already on-GPU. Because the slowest per-layer transfer now shifts from D2D AllGather to H2D PCIe, the saturation threshold $T$ in Eq. (1) automatically absorbs this by setting $t_{\text{EP}}$ to the H2D latency—the frontend still schedules enough compute to cover the slower channel.

KV dimension: KV-cache-free execution. For workloads without significant prefix sharing (e.g., classification over diverse documents), prefix KV caches offer no reuse benefit but still consume substantial HBM. MoE-Prefill supports a __KV-cache-free__ mode that disables KV storage entirely and computes attention on the fly [9]. The trade-off is explicit—high-prefix-reuse workloads lose the reuse dividend—but for prefix-sparse workloads the KV footprint is pure overhead.

The two knobs are orthogonal and compose: with both enabled, each GPU’s HBM holds only attention weights, a few layers of expert shards, and activation workspace—a fraction of static EP’s footprint (C1). Freed memory is reinvested in larger KV caches and more aggressive batching, or traded for lower-HBM GPUs that broaden the deployable envelope. Together with AsyncEP (C2+C3), the three components form a progressive, composable design that executes with zero on-path communication, provided the frontend meets the saturation threshold $T$, which is the co-design interface.

## 7 Frontend: Prefix-, Compute-, and Overlap-Aware Scheduling

The backend (§6) hides communication behind compute via AsyncEP transfers. The frontend’s role is to make this overlap effective in practice: supply each GPU with __enough__ compute to cover the transfer latency, while simultaneously maximizing prefix reuse and avoiding redundant work.

### 7.1 Design Constraints for Prefill-Only Scheduling

Combining workload opportunities (§3) with backend properties (§6), the frontend must jointly satisfy: F1 (prefix reuse) co-locate same-prefix requests so the prefix is computed and cached exactly once (Insight #3); F2 (compute awareness) account for prefix-sharing credit, since the true cost of $N$ same-prefix requests is one prefix pass plus $N$ suffix passes, not $N$ full ones; and F3 (overlap) keep each GPU’s per-batch FLOPs above the saturation threshold $T$ (Eq. (1)) so AsyncEP’s transfers stay hidden on the critical path. F1 decides __where__ a request goes, F2 __how__ to measure its cost, and F3 __when__ a GPU has enough work.

Why existing schedulers fail. Existing LLM serving systems optimize mixed prefill–decode workloads with simple load proxies (e.g., vLLM’s $\textit{waiting}{\times}4+\textit{running}$) that work only when per-request cost is dominated by decoding. They violate all three constraints: they ignore prefix sharing and scatter same-prefix requests (F1); even a token-count metric would estimate $10\cdot(4096{+}S_{\text{sfx}})$ tokens for 10 same-prefix requests whose true cost is one prefix pass plus 10 short suffix passes, perversely driving the scheduler to disperse them (F2); and they lack a backend-derived overlap threshold, so decoding-calibrated heuristics are misaligned with AsyncEP (F3).

### 7.2 Prefix-Aware Routing

To satisfy F1, MoE-Prefill routes each request to the GPU whose resident prefix KV cache shares the __longest common prefix__ with the request’s input. Matching operates at __block granularity__ (e.g., 16 tokens per block), aligned with the KV management unit used by PagedAttention [36] and RadixAttention [71]: each block is hashed and the scheduler queries each GPU’s block table for the longest contiguous match. This reduces matching cost from $O(P)$ token-level lookups to $O(P/B)$ block-level lookups and reuses the block table already maintained by the serving engine.

In-batch vs. cross-batch reuse. The same longest-match rule covers two time scales without any additional machinery, though the savings differ. __Within a round__ (in-batch), sibling requests share a single prefix forward pass: the prefix self-attention is identical across siblings and is computed once, and a single prefix KV copy is stored in HBM. __Across rounds__ (cross-batch), each GPU’s block table retains prefix KV blocks under LRU eviction; a later request hitting an already-resident prefix and reuses the cached prefix KV cache. Both behaviors emerge from a single routing policy, with no explicit batch-assembly or replication mechanism; the precise FLOPs are captured by $C_{\text{pfx}}(P_{r}{-}M_{r})+C_{\text{sfx}}(S_{r},P_{r})$ in §7.3.

### 7.3 Compute-Aware Tracking

To satisfy F2, the scheduler maintains a running cost estimate $L_{i}$ per GPU that reflects actual FLOPs after prefix reuse. When request $r$ with prefix length $P_{r}$ and suffix length $S_{r}$ is assigned to GPU $i$ with $M_{r}$ prefix tokens already cached, the cost increment is

$$\Delta_{r}=C_{\text{pfx}}(P_{r}-M_{r})+C_{\text{sfx}}(S_{r},\,P_{r}),$$

where $C_{\text{sfx}}(S,P)=C_{\text{FFN}}(S)+C_{\text{self}}(S)+C_{\text{cross}}(S,P)$ combines linear feed-forward, $O(S^{2})$ suffix self-attention, and $O(S{\cdot}P)$ cross-attention of suffix tokens to prefix KV. When the full prefix is cached ($M_{r}=P_{r}$), the prefix term vanishes and only the suffix cost remains. After each assignment, $L_{i}\!\leftarrow\!L_{i}+\Delta_{r}$.

This cost model correctly captures prefix-sharing savings: assigning 10 same-prefix requests to a GPU adds one prefix cost plus 10 suffix costs, not 10 full-request costs. The scheduler therefore treats such a GPU as __lightly__ loaded—the opposite of a token-based metric—and keeps routing same-prefix requests there until saturation.

### 7.4 Overlap-Aware Balancing

To satisfy F3, the scheduler enforces the saturation threshold $T$ (Eq. (1)) derived from the backend’s EP communication latency.

At each scheduling round, the scheduler drains a global FIFO queue. For each request, it evaluates all __unsaturated__ GPUs ($L_{i}<T$), selects the one with the longest block-level prefix match (F1), updates its load via the true-cost model (F2), and marks it __saturated__ once $L_{i}\geq T$. The round ends when all GPUs are saturated or the queue is exhausted. A popular prefix that saturates the primary GPU early naturally spills to the next-best-matching GPU through the same logic—no explicit replication policy is needed. Because GPUs are marked saturated one at a time, each GPU’s final load lies in $[T,\;T+\Delta_{\text{last}}]$, so load balance emerges __for free__ as a structural byproduct of saturation, not an explicit objective. Pseudocode of one scheduling round is given in App. A.

## 8 Evaluation

|Dataset | Workload | Req Context (tokens) | # Request | Total (tokens) | Prefix Share |
| --- | --- | --- | --- | --- | --- |
|MoralStories | Moral reasoning | $\sim$100–200 | 24K | $\sim$3.3M | High |
|MMLU | Multiple-choice QA | $\sim$50–500 | 24K | $\sim$5.8M | Low |
|BoolQ | Binary QA | $\sim$200–600 | 12K | $\sim$2.7M | Low |
|IMDB | Sentiment classification | $\sim$300–2K | 12K | $\sim$5.5M | Low |
|QuALITY | Long-document QA | $\sim$4K–12K | 1.2K | $\sim$10.5M | High |
|ArXiv Class. | Document classification | $\sim$6K–128K | 600 | $\sim$10.1M | Medium |
|Aggregated | Prefill-only | $\sim$50–128K | 73.8K | $\sim$37.9M | Mixed |

Table 2: Composition of the aggregated prefill-only workload. Per-source rows describe the contribution of each benchmark; the aggregated workload (last row) is the sole workload used in §8.2–§8.3. All requests are reformulated as prefill-only; output length is fixed to 1.

We evaluate MoE-Prefill along five axes: end-to-end throughput on real-world prefill-only workloads (§8.2), the contribution of each co-design tier (§8.3), generalization to synthetic workloads without prefix reuse (§8.4), memory scalability and per-GPU compute efficiency (§8.5), and correctness of the prefill-only reformulation (§8.6).

### 8.1 Evaluation Setup

Datasets. To reflect the heterogeneity of production prefill-only traffic, we construct a single __aggregated workload__ by mixing requests sampled from six public benchmarks (Table 2): short-context classification and QA (MoralStories [14], MMLU [25], BoolQ [8]), medium-length document analysis (IMDB [43], QuALITY [49]), and long-context document classification (ArXiv Classification [24], up to 128K tokens). Together they span the full prefix-share spectrum—from high (shared system prompts) through medium to low (independent documents)—and yield a workload of 73.8K requests and $\sim$37.9M tokens, mirroring a production endpoint that interleaves classification, QA, and long-document requests.

Hardware, models, and baselines. We evaluate MoE-Prefill on three datacenter GPU platforms—$8{\times}$A100 (80GB, BF16), $8{\times}$H100 (80GB, BF16 and FP8), and $8{\times}$H200 (141GB, FP8)—using __Qwen3-235B-A22B__ [65] (128 experts, top-8 routing, $\sim$22B activated parameters). We compare against two groups of baselines, swept at parallel degrees 1, 2, 4, and 8. Group (i): the five distributed strategies natively supported by vLLM for large-MoE inference—DP+EP, DP+TP, TP+EP, TP+TP, and PP+PP (Table 1). Group (ii): four PrefillOnly-augmented configurations. PrefillOnly [12] targets dense LLMs on a single GPU; its hybrid-prefilling and suffix-KV-cache-discarding techniques are engineered to enable such deployment by reducing per-GPU memory. Since Qwen3-235B-A22B exceeds the HBM of every GPU tested, these techniques have a negligible impact. We therefore port its scheduler, shortest-prefill-first, with continuous prefill-time estimation into vLLM v0.11.0 and pair it with two distributed backends, DP+EP and PP+PP (the strongest Group (i) baseline), each at two batch sizes: max-num-seqs=1 (PrefillOnly’s paper default setting) and a value matched to MoE-Prefill. For every hardware–precision cell we report MoE-Prefill’s improvement over the best baseline per parallel degree.

Implementation and metrics. MoE-Prefill is implemented on top of vLLM (details in App. B) and enabled via a single flag. We report (i) throughput in tokens/s (total prefilled tokens divided by end-to-end batch completion time); (ii) MFU as achieved FLOPs over the GPU’s peak FLOPs at the deployed precision; (iii) peak HBM usage and maximum feasible batch/context; and (iv) task accuracy. All runs fix the output length to 1, isolating prefill execution.

### 8.2 End-to-End Throughput on Real-World Workloads

Figure 9 reports throughput on the aggregated real-world workload across all four hardware–precision combinations, sweeping the parallel degree from 1 to 8 GPUs.

Uniform dominance across precisions. MoE-Prefill achieves the highest throughput in every hardware–precision–parallel-degree cell, delivering 1.37$\times$ on 8$\times$A100 (BF16), 1.36$\times$ on 8$\times$H100 (BF16), 1.35$\times$ on 8$\times$H100 (FP8), and 1.37$\times$ on 8$\times$H200 (FP8) over the strongest baseline in each cell. MoE-Prefill’s $\sim$1.35$\times$ relative gain is stable even as FP8 roughly doubles absolute throughput, consistent with the analytical prediction in §4 that on-path All-to-All traffic scales with $B{\cdot}S{\cdot}H$ independent of element size.

Near-linear scaling. Baselines exhibit the sub-linear scaling predicted by Table 1: adding GPUs amortizes the weight footprint but injects more per-layer collective traffic, and several baselines plateau or regress past 4 GPUs. MoE-Prefill scales near-linearly because its only cross-GPU traffic, AsyncEP’s D2D AllGather, is background-overlapped with compute, decoupling throughput from collective-scaling limits.

### 8.3 Necessity of Frontend–Backend Co-Design

Because DP+AsyncEP (backend only) already resolves the on-path-collective bottleneck (§6), a natural question is whether the frontend of §7 contributes enough to justify its complexity. Fig. 10 decomposes the gain into two tiers: DP+EP (synchronous MoE), DP+AsyncEP under vLLM’s default scheduler, and MoE-Prefill (backend + frontend co-design).

Fig. 10 annotates each bar with the throughput gain of adding the frontend (i.e., MoE-Prefill vs. DP+AsyncEP under vLLM’s default scheduler). At 8 GPUs, the frontend improves throughput by +18% (A100, BF16), +17% (H100, BF16), +18% (H100, FP8), and +16% (H200, FP8). The contribution __grows with parallel degree__: at low GPU counts, random dispatch still achieves reasonable prefix hits, but adding GPUs scatters same-prefix requests and dilutes reuse, whereas MoE-Prefill’s longest-block-match routing sustains high reuse at any scale. The gain is further amplified under FP8, where higher compute density shrinks the per-layer window and gives saturation-driven admission more room to contribute.

### 8.4 Generalization Across Context Regimes

To isolate the effect of context length __independent__ of prefix reuse, we sweep synthetic workloads with uniformly random prompts (no prefix sharing) across four context regimes: short ($256{\times}40960$), medium ($4\text{K}{\times}2560$), long ($32\text{K}{\times}320$), and ultra-long ($128\text{K}{\times}80$), where $S{\times}N$ denotes sequence length $\times$ number of requests (10M tokens per regime).

MoE-Prefill’s backend alone (DP+AsyncEP) improves over the best baseline by 1.30$\times$ (short), 1.30$\times$ (medium), 1.52$\times$ (long), and 1.49$\times$ (ultra-long) at 8 GPUs. Even without prefix reuse, AsyncEP delivers $\geq$1.30$\times$, confirming that P3 (zero redundant communication) produces gains independent of P1. The improvement __grows__ with context length because the baselines degrade on two fronts: on-path All-to-All volume scales linearly with $B{\cdot}S$, making EP-based strategies proportionally more costly; and causal attention creates an inherent load imbalance across pipeline stages, early stages process shorter attention contexts, while later stages handle the full context and bear disproportionately heavier work, exposing pipeline bubbles that further penalize PP+PP. MoE-Prefill is immune to both: AsyncEP’s weight transfers are fully hidden behind compute regardless of sequence length, and pure DP attention eliminates pipeline partitioning.

### 8.5 Memory Scalability and Compute Efficiency

Distributed MoE serving usually sacrifices efficiency on two axes at once: HBM capacity (235B-class models require multi-GPU deployment just to hold the weights) and per-GPU MFU (smaller per-device batches fall below the GEMM-saturation threshold, and on-path collectives steal cycles from useful compute; Insight #5). MoE-Prefill’s P2 (hybrid offloading + KV-cache-free execution) and P3 (AsyncEP) attack the two bottlenecks jointly. We reuse the synthetic sweep of §8.4 (8$\times$H100 FP8, Qwen3-235B-A22B, four context regimes, parallel degrees 1/2/4/8), measuring MFU as achieved FLOPs normalized by the 1979 TFLOPS FP8 peak reported in the H100 datasheet [47].

Memory scalability at no MFU cost. Qwen3-235B-A22B occupies 235 GB in FP8 (470 GB in BF16), which exceeds the aggregate 160 GB HBM of two H100s even before activations. Every baseline in Table 1 therefore requires $\geq$4 GPUs simply to hold the weights—the N/A columns at 1–2 GPUs in Fig. 12 make this gap concrete. Under P2, MoE-Prefill retains only attention weights plus a sliding window of upcoming expert shards on-GPU while streaming the rest from CPU pinned memory over PCIe, combined with KV-cache-free execution for prefill-only workloads; the per-GPU HBM footprint then fits within 80 GB, pushing the deployable envelope from “$\geq$4 GPUs” down to a single commodity 80 GB GPU. Crucially, this does not come at a compute-efficiency cost: on 1–2 GPUs MoE-Prefill sustains 32.0–36.2% MFU, statistically indistinguishable from its 29.8–34.7% MFU at 8 GPUs, because the large per-device token batch in the memory-constrained regime fully overlaps H2D expert transfers behind compute.

Dominance in the multi-GPU regime. From 4 to 8 GPUs, every baseline degrades monotonically in every context regime (e.g., DP+EP drops 1.90$\times$ at short context), confirming the two predicted mechanisms: sub-saturation GEMMs and on-path collective overhead. DP+AsyncEP is the top bar in __every__ (context, parallel degree) cell at 29.8–36.2% MFU, beating the strongest baseline by 1.29$\times$ (short), 1.30$\times$ (medium), 1.59$\times$ (long), 1.49$\times$ (ultra-long). Strikingly, even the baselines’ best 8-GPU cell (TP+TP at 128K, 20.09%) falls below MoE-Prefill’s __worst__ cell across the sweep (29.84%). The feasible hardware envelope therefore widens from “$\geq$4 GPUs” to “1–8 GPUs” (a 4$\times$ broader range), and within that entire envelope per-GPU MFU varies by only 1.03–1.09$\times$—validating Insight #5 and the zero-redundancy design of P2+P3.

### 8.6 Accuracy of Prefill-Only Reformulation

Correctness decomposes into two orthogonal claims mirroring the two-tier design of §5.3. __Tier 1 (execution fidelity):__ AsyncEP only changes __when__ expert weights arrive, hybrid offloading only relocates them between HBM and CPU memory, and KV-cache-free execution recomputes attention under the same formulation; none of these modifies GEMM math, attention kernels, or quantization, so per-layer logits are bit-identical to vLLM up to FP non-associativity by construction, empirically confirmed on a 25K-sample same-model run (App. C). __Tier 2 (reformulation fidelity):__ on a nine-task production classification harness with two open-source models, prefill-only logit scoring lies within $\pm$3.6 pp of autoregressive decoding on 7 of 9 tasks (Table 3); a same-model three-way ablation (Table 4) isolates the decoding-mode contribution to $\sim$1 pp. Together these validate the end-to-end accuracy of prefill-as-a-service under MoE-Prefill.

## 9 Related Work

MoE-Prefill intersects four lines of prior work; the structural difference throughout is that MoE-Prefill gathers experts __by weight__ rather than routing activations, co-designed with a physically-derived saturation invariant.

LLM inference engines and prefill optimizations. Production engines [36, 71, 48, 3, 68] and techniques such as continuous batching [68], chunked prefill [2], speculative decoding [38], and prefill/decode disaggregation [50, 73, 53] all presume prefill is amortized over many decoding steps. PrefillOnly [12] first targets prefill-only workloads via KV-cache lifetime and job-level scheduling but leaves the MoE execution stack untouched; our analysis (§4) shows execution-level overheads dominate regardless of KV policy, making MoE-Prefill complementary.

Mixture-of-experts serving. MoE scales capacity via sparse activation [15, 37, 13, 75] and is adopted by most recent open-weight models [30, 10, 65, 1]. System-side MoE work [22, 27, 54, 17, 23, 39, 5, 31, 66, 28, 51, 20, 60] optimizes traffic, placement, or kernels __within__ the synchronous activation-routed EP model, keeping two per-layer AllToAlls on the critical path. MoE-Prefill inverts this model: experts are gathered by weight, removing AllToAll and dissolving routing-imbalance stragglers (§6).

Prefix caching and attention reuse. PagedAttention [36], RadixAttention [71], and related caching systems [42, 53, 18, 67, 16, 32] treat prefix reuse as a __passive cache effect__. MoE-Prefill’s frontend turns it into an __active scheduling decision__ via longest-block-match routing, enabled by AsyncEP freeing the HBM needed for large batches that make prefix co-location meaningful. BlendServe [70] overlaps __requests__ with complementary resource profiles atop a synchronous engine; MoE-Prefill overlaps __layers__ via async weight gathering, and the two compose.

Expert and weight offloading. Dense offloading systems [58, 56, 55, 21] and MoE extensions [28, 34, 33, 64] either transfer on demand (on the critical path) or rely on popularity heuristics that degrade under large-batch routing skew. MoE-Prefill exploits the __deterministic layer order of prefill__ and sizes the compute window via $T$ (Eq. (1)), making offloaded experts appear always-resident by construction.

## 10 Discussion

Applicability and limitations. MoE-Prefill targets throughput-oriented, batch-driven prefill-only serving on MoE models that exceed single-GPU HBM. It is __not__ designed for latency-critical interactive serving, arrivals too bursty to sustain the saturation threshold, or dense models with no expert stack. Three limitations merit note: (i) on low-bandwidth interconnects $t_{\text{EP}}$ and thus $T$ grow, potentially re-exposing some transfers on the critical path; (ii) $T$ is calibrated once at startup—severe workload drift can briefly revert layers to synchronous behavior, though prefill-only batches refill quickly; and (iii) KV-cache-free mode and prefix-aware routing are workload-dependent knobs that contribute nothing on truly random traffic but do not regress (§8.4).

Broader implications. The core insight—decoupling expert placement from activation routing—applies whenever the per-layer compute window covers transfer latency, including long-context reasoning and speculative-decode verification. More broadly, __expert weights should be treated as schedulable resources rather than static parameters__, dissolving routing-imbalance stragglers and turning HBM capacity into a deployment knob.

## 11 Conclusion

Prefill-only workloads expose three structural redundancies in distributed MoE serving, all rooted in coupling expert placement with synchronous activation routing. MoE-Prefill inverts this coupling: AsyncEP replaces per-layer AllToAll with background weight AllGather fully overlapped with compute, while a co-designed frontend enforces a saturation threshold to guarantee the overlap. On Qwen3-235B-A22B, MoE-Prefill delivers $1.35$–$1.37\times$ throughput over the strongest baseline and sustains $29.8$–$36.2\%$ MFU across 1–8 GPUs. The broader lesson: expert weights are best treated not as static parameters but as schedulable resources driven by the execution schedule.

## Appendix

## Appendix A Scheduling Algorithm Pseudocode

Algorithm 1 summarizes one scheduling round of MoE-Prefill’s frontend (§7), integrating prefix-aware routing (F1), compute-aware tracking (F2), and overlap-aware saturation (F3) in a single per-request assignment pass.

## Appendix B System Implementation

We implement MoE-Prefill on top of vLLM [36] (version v0.11.0), reusing its request ingestion, tokenizer, and model execution paths. The additions span three layers—a backend weight streaming engine (§6), a frontend DP router (§7), and a piggyback event channel that keeps the router’s shadow state in sync with the backend—and require no changes to model definitions, kernels, or the tokenizer pipeline.

### B.1 Backend: Weight Streaming Engine

The backend presents three execution modes—one-GPU H2D streaming, multi-GPU D2D AllGather over NVLink (§6.2), and concurrent H2D prefetch + D2D AllGather (§6.3)—carried by two helpers on dedicated CUDA streams. The __MoE gatherer__ issues layer $i{+}1$’s AllGather as soon as layer $i$ starts computing and synchronises it with a single event wait at layer $i{+}1$’s entry. The __H2D offloader__ pins the CPU backing store and prefetches a sliding window of $k$ layers ahead of the gatherer. In hybrid mode the two pipelines saturate PCIe and NVLink concurrently, and the model runner sees only the slower channel (captured by $t_{\text{EP}}$ in Eq. (1)). KV-cache-free execution is an orthogonal launch flag that skips the KV allocator entirely.

### B.2 Frontend: Prefix-Aware DP Router

The frontend replaces vLLM’s default weighted-count DP load balancer with an event-driven state machine that tracks, per engine $i$, (a) a __committed__ table $\mathcal{C}_{i}$ of block hashes already materialised on the GPU, updated from the engine’s block-cache lifecycle events (captures cross-batch reuse); (b) a __pending__ table $\mathcal{P}_{i}$ of hashes promised by requests routed but not yet executed, updated speculatively at routing time (captures in-batch reuse); and (c) a FLOPs-denominated work counter $L_{i}$ computed from a single per-token constant $f_{\text{tok}}$ derived analytically from the model config. Because both enqueue and decay use the same $f_{\text{tok}}$, $L_{i}$ is drift-free even though the engine reports progress in tokens.

For each request the router runs the F3$\to$F1$\to$F2 cascade in a single pass: filter by saturation ($L_{i}<T$), pick the engine with the longest $\mathcal{C}_{i}\cup\mathcal{P}_{i}$ prefix match (ties broken by $L_{i}$), fall back to $\arg\min_{i}L_{i}$ if no engine has a positive match, then speculatively add the uncached hashes to $\mathcal{P}_{i^{*}}$ and increment $L_{i^{*}}$ by $\D

..._This content has been truncated to stay below 50000 characters_...
