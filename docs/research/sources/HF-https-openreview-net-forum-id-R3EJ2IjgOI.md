# SOURCE: https://openreview.net/forum?id=R3EJ2IjgOI

# Memory Caching: RNNs with Growing Memory | OpenReview
URL: https://openreview.net/forum?id=R3EJ2IjgOI
Published: 2025-10-08

Memory Caching: RNNs with Growing Memory | OpenReview

## Memory Caching: RNNs with Growing Memory

### Ali Behrouz, Zeman Li, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni

Submitted to ICLR 2026Everyone Revisions BibTeX CC BY 4.0

Keywords: Recurrent Neural Network, Attention, Transformers

Abstract: Transformers have been established as the de-facto backbones for most recent advances in sequence modeling, mainly due to their growing memory capacity that scales with the context length. While plausible for retrieval tasks, it causes quadratic complexity and so has motivated recent studies to explore viable subquadratic recurrent alternatives. Despite showing promising preliminary results in diverse tasks, such recurrent architectures underperform Transformers in recall-intensive tasks, {often attributed to their fixed-size memory. In this paper, we introduce Memory Caching (\mc), a simple yet effective technique that enhances recurrent models by caching checkpoints of their memory states (a.k.a. hidden states). \mc{} allows the effective memory capacity of RNNs to grow with sequence length, offering a flexible trade-off that interpolates between the fixed memory ($\mathcal{O}(L)$ complexity) of RNNs and the growing memory ($\mathcal{O}(L^2)$ complexity) of Transformers. We propose four variants of MC, including gated aggregation and sparse selective mechanisms, and discuss their implications on both linear and deep memory modules.} Our experimental results on language modeling, and long-context understanding tasks show that \mc{} enhances the performance of recurrent models, supporting its effectiveness. In in-context recall tasks, our results indicate that while Transformers still achieve the best performance, our MC variants show competitive performance, close the gap with Transformers, and performs better than state-of-the-art recurrent models.

Primary Area: foundation or frontier models, including LLMs

Submission Number: 24014

Loading
