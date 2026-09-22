# SOURCE: https://arxiv.org/pdf/2609.04575

# Training-Free Halving of Activated Expertsin Fine-Grained Mixture-of-Experts Models
URL: https://arxiv.org/pdf/2609.04575

Training-Free Halving of Activated Expertsin Fine-Grained Mixture-of-Experts Models

arXiv is now an independent nonprofit! Learn more×

# Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models

Xing Chen Note: Correspondence: raincchio@gmail.com Hengshuai Yao

###### Abstract

Modern fine-grained Mixture-of-Experts (MoE) models route each token to a small number of experts and renormalize their router probabilities. We show that this renormalization implicitly calibrates expert output gain to the training top- $k$ : reducing $k$ at inference changes not only which experts are used but also the strength of the expert branch. We separate these effects by activating the top $k_{1}$ experts while normalizing by the probability mass of the top $k_{2}$ experts, introducing one integer with no parameters, training, or measurable compute overhead. On Qwen3.6-35B-A3B, reducing from 8 to 4 experts causes a 4.65-point MMLU drop under standard renormalization but only 0.35 points with $k_{2}=16$ , while halving routed-expert compute. The result replicates on the $11\times$ larger Qwen3.5-397B-A17B, where reducing from 10 to 5 experts loses only 0.55 points with an appropriate reference set. Removing renormalization entirely is catastrophic, showing that preserving a suitable reference mass is crucial. We further find that perplexity and downstream accuracy favor different $k_{2}$ , cautioning against selecting MoE compression settings using unlabeled text alone. Analyses also show that expert identity matters substantially more than expert weighting, while balanced and domain-specialized routing leaves limited room for expert pruning.

## Introduction

Sparse Mixture-of-Experts (MoE) architectures decouple parameter count from per-token compute by activating only a few of many expert sub-networks (Shazeer et al. 2017; Lepikhin et al. 2021; Fedus et al. 2022). Recent open-weight models have pushed this design toward fine-grained sparsity: rather than 8 or 16 large experts, they use hundreds of small ones and route each token to a handful of them (Dai et al. 2024; DeepSeek-AI 2024; Qwen Team 2025). In Qwen3.6-35B-A3B, our primary object of study, all 40 layers are MoE layers with 256 experts each and $k{=}8$ selected per token; the routed experts hold $32.2$ B of the model’s $35.9$ B parameters (89.6%), while each token touches only $3.1\%$ of them.

This memory–compute asymmetry—all experts must be resident, few are used—is what makes MoE compression attractive and awkward at the same time. Reducing the number of experts (pruning or merging) saves memory but not compute, since $k$ is unchanged; reducing $k$ saves compute but not memory. A large body of training-free work attacks the memory side by identifying and removing redundant experts (Lu et al. 2024; Li et al. 2024). The compute side, by contrast, is usually treated as trivially adjustable: $k$ is an inference-time argument, so one simply lowers it, renormalizes over the s
