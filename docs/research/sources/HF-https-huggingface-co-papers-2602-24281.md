# SOURCE: https://huggingface.co/papers/2602.24281

# Paper page - Memory Caching: RNNs with Growing Memory
URL: https://huggingface.co/papers/2602.24281
Author: Ali Behrouz    ,

# Memory Caching: RNNs with Growing Memory

Published on Feb 27

 

 taesiri

- +6

Authors:

Ali Behrouz ,

Zeman Li ,

Yuan Deng ,

Peilin Zhong ,

Meisam Razaviyayn ,

Vahab Mirrokni 

## Abstract

Memory Caching enhances recurrent models by allowing their memory capacity to scale with sequence length, bridging the gap between traditional RNNs and Transformers in long-context tasks.

Transformers have been established as the de-facto backbones for most recent advances in sequence modeling, mainly due to their growing memory capacity that scales with the context length. While plausible for retrieval tasks, it causes quadratic complexity and so has motivated recent studies to explore viable subquadratic recurrent alternatives. Despite showing promising preliminary results in diverse domains, such recurrent architectures underperform Transformers in recall-intensive tasks, often attributed to their fixed-size memory. In this paper, we introduce Memory Caching (MC), a simple yet effective technique that enhances recurrent models by caching checkpoints of their memory states (a.k.a. hidden states). Memory Caching allows the effective memory capacity of RNNs to grow with sequence length, offering a flexible trade-off that interpolates between the fixed memory (i.e., O(L) complexity) of RNNs and the growing memory (i.e., O(L^2) complexity) of Transformers. We propose four variants of MC, including gated aggregation and sparse selective mechanisms, and discuss their implications on both linear and deep memory modules. Our experimental results on language modeling, and long-context understanding tasks show that MC enhances the performance of recurrent models, supporting its effectiveness. The results of in-context recall tasks indicate that while Transformers achieve the best accuracy, our MC variants show competitive performance, close the gap with Transformers, and performs better than state-of-the-art recurrent models.

### Community

 

Mar 2

This is an automated message from the Librarian Bot. I found the following papers similar to this paper. 

The following papers were recommended by the Semantic Scholar API 

- RAM-Net: Expressive Linear Attention with Selectively Addressable Memory (2026)
- Towards Compressive and Scalable Recurrent Memory (2026)
- HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing (2026)
- AllMem: A Memory-centric Recipe for Efficient Long-context Modeling (2026)
- CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling (2026)
- Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models (2026)
- Out of the Memory Barrier: A Highly Memory Efficient Training System for LLMs with Million-Token Contexts (2026)

 Please give a thumbs up to this comment if you found it helpful!

 If you want recommendations for any Paper on Hugging Face checkout this Space

 You can directly ask Librarian Bot for paper recommendations by tagging it in a c
