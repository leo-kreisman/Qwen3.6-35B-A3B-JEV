# SOURCE: https://arxiv.org/abs/2602.24281

# Memory Caching: RNNs with Growing Memory
URL: https://arxiv.org/abs/2602.24281

Memory Caching: RNNs with Growing Memory

arXiv is now an independent nonprofit! Learn more×

# Memory Caching: RNNs with Growing Memory

Ali Behrouz Zeman Li Yuan Deng Peilin Zhong [7pt] Meisam Razaviyayn Vahab Mirrokni Affiliation:† Correspondence: alibehrouz@google.com

###### Abstract

Transformers have been established as the de-facto backbones for most recent advances in sequence modeling, mainly due to their growing memory capacity that scales with the context length. While plausible for retrieval tasks, it causes quadratic complexity and so has motivated recent studies to explore viable subquadratic recurrent alternatives. Despite showing promising preliminary results in diverse domains, such recurrent architectures underperform Transformers in recall-intensive tasks, often attributed to their fixed-size memory. In this paper, we introduce Memory Caching (MC), a simple yet effective technique that enhances recurrent models by caching checkpoints of their memory states (a.k.a. hidden states). MC allows the effective memory capacity of RNNs to grow with sequence length, offering a flexible trade-off that interpolates between the fixed memory (i.e., $\mathcal{O}(L)$ complexity) of RNNs and the growing memory (i.e., $\mathcal{O}(L^{2})$ complexity) of Transformers. We propose four variants of MC, including gated aggregation and sparse selective mechanisms, and discuss their implications on both linear and deep memory modules. Our experimental results on language modeling, and long-context understanding tasks show that MC enhances the performance of recurrent models, supporting its effectiveness. The results of in-context recall tasks indicate that while Transformers achieve the best accuracy, our MC variants show competitive performance, close the gap with Transformers, and performs better than state-of-the-art recurrent models.

## 1 Introduction

Transformers (Vaswani et al., 2017) are the foundation of recent advances in machine learning across diverse domains (Jumper et al., 2021; Dosovitskiy et al., 2021; Comanici et al., 2025). This success often is attributed to their ability to learn at scale (Kaplan et al., 2020) and in-context (Brown et al., 2020), both of which are the byproduct of their primary building block–attention module–that acts as an associative memory with growing capacity (Ramsauer et al., 2021; Bietti et al., 2024; Behrouz et al., 2026). While effective for many retrieval tasks (Arora et al., 2024b), this growing memory incurs quadratic complexity and high inference-time memory usage (KV-caching). This has motivated the development of sub-quadratic architectures that aim to improve efficiency while maintaining performance (Dai et al., 2019; Child et al., 2019; Poli et al., 2023).

In particular, recurrent neural networks that aim to compress the past data into their memory state, maintaining a fixed size over the entire input sequence, have regained attention in recent years (Katharopoulos et al., 2020; Irie et al., 202
