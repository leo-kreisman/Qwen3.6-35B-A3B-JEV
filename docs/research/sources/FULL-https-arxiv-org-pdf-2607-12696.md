# SOURCE: https://arxiv.org/pdf/2607.12696

# Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts
URL: https://arxiv.org/pdf/2607.12696

Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts

arXiv is now an independent nonprofit! Learn more×

# Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts

Jincheng Xie Affiliation: Department of Mathematical Sciences, Tsinghua University, Beijing, China Runheng Liu Affiliation: School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China Heyan Huang Affiliation: School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China Yawen Ling Affiliation: JDT AI Infra Hanbin Dai Affiliation: JDT AI Infra Yu Zheng Affiliation: School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu, Sichuan, China Affiliation: School of Cyber Engineering, Xidian University, Xi’an, Shaanxi, China Wen Hu Affiliation: JDT AI Infra Correspondence to: huwen.31@jd.com

###### Abstract

Sparse Mixture-of-Experts (MoE) models have become an important approach for scaling Large Language Models (LLMs), but their inference efficiency depends strongly on expert activation patterns. Speculative decoding (SD) accelerates autoregressive generation by verifying multiple draft tokens in parallel, yet existing draft selection strategies primarily optimize acceptance likelihood. In large-scale MoE models, however, selecting draft tokens also determines the union of experts activated during verification. We observe that confidence-driven SD can introduce expert scattering: high-probability draft tokens may route to disjoint experts, increasing expert-weight memory traffic and reducing the speedup from speculation. Motivated by this observation, we revisit draft-tree selection under the non-uniform memory-cost structure of MoE inference. We propose EcoSpec, a cost-aware speculative decoding framework that incorporates predicted marginal expert activation cost into draft selection. With a lightweight expert predictor and a dynamic expert buffer, EcoSpec favors draft paths that preserve high acceptance likelihood while reusing experts already covered by the current verification set, without modifying the target-model verification rule. We evaluate EcoSpec on three large-scale MoE models, including DeepSeek-V3.1 (671B), Qwen3-235B-A22B, and GPT-OSS-120B, across reasoning, coding, question-answering, and dialogue benchmarks. EcoSpec consistently reduces active expert footprints and improves end-to-end decoding speed, achieving up to $1.62\times$ speedup. These results show that accounting for expert activation cost is important for efficient speculative decoding in large-scale MoE models.

###### Keywords:

Machine Learning, ICML

††affiliationnotice: Equal contribution $\ddagger$ Work done while interning at JDT AI Infra.

## 1 Introduction

Figure 1: The Bandwidth Bottleneck in MoE Speculative Decoding. (a) Verification latency scales linearly with the number of active experts ( $\mathcal{E}$ ), creating a strict latency penalty for r
