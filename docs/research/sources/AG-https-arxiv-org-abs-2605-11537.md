# SOURCE: https://arxiv.org/abs/2605.11537

[Submitted on 12 May 2026]

# Title:Fast MoE Inference via Predictive Prefetching and Expert Replication

[View PDF](/pdf/2605.11537)[HTML (experimental)](https://arxiv.org/html/2605.11537v1)

> Abstract:The Mixture of Experts (MoE) architecture has become a fundamental building block in state-of-the-art large language models (LLMs), improving domain-specific expertise in LLMs and scaling model capacity without proportionally increasing their computational overhead. However, MoE inference often suffers from suboptimal GPU utilization, load imbalance, and elevated latency arising from multiple tokens waiting on the same experts for their computation which arises from sparsity of expert activation. To address these challenges, we propose a dynamic expert replication strategy that predicts which experts are likely to be overloaded and replicates them for upcoming batches of tokens. The replicated experts process batch tokens concurrently across layers, which leads to improved parallelism, shorter GPU idle time, and significantly faster inference. Experimental evaluations conducted on large-scale MoE models, including Switch-base-128 and Switch-base-256, demonstrate that our method achieves near-complete GPU utilization (approx 100%), leading to upto 3x improvement in inference speed while preserving approximately 90-95% of the performance of baseline architectures
