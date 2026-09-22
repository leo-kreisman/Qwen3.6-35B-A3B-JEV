# SOURCE: https://arxiv.org/abs/2609.04575

# Computer Science > Machine Learning

 [Submitted on 4 Sep 2026]

# Title:Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models

[View PDF](/pdf/2609.04575) [HTML (experimental)](https://arxiv.org/html/2609.04575v1)

> Abstract:Modern fine-grained Mixture-of-Experts (MoE) models route each token to a small number of experts and renormalize their router probabilities. We show that this renormalization implicitly calibrates expert output gain to the training top-$k$: reducing $k$ at inference changes not only which experts are used but also the strength of the expert branch. We separate these effects by activating the top $k_1$ experts while normalizing by the probability mass of the top $k_2$ experts, introducing one integer with no parameters, training, or measurable compute overhead. On Qwen3.6-35B-A3B, reducing from 8 to 4 experts causes a 4.65-point MMLU drop under standard renormalization but only 0.35 points with $k_2=16$, while halving routed-expert compute. The result replicates on the $11\times$ larger Qwen3.5-397B-A17B, where reducing from 10 to 5 experts loses only 0.55 points with an appropriate reference set. Removing renormalization entirely is catastrophic, showing that preserving a suitable reference mass is crucial. We further find that perplexity and downstream accuracy favor different $k_2$, cautioning against selecting MoE compression settings using unlabeled text alone. Analyses also show that expert identity matters substantially more than expert weighting, while balanced and domain-specialized routing leaves limited room for expert pruning.

| Subjects: | Machine Learning (cs.LG); Artificial Intelligence (cs.AI) |
|---|---|
| Cite as: | [arXiv:2609.04575](https://arxiv.org/abs/2609.04575) [cs.LG] |
| | (or [arXiv:2609.04575v1](https://arxiv.org/abs/2609.04575v1) [cs.LG] for this version) |
| | [https://doi.org/10.48550/arXiv.2609.04575](https://doi.org/10.48550/arXiv.2609.04575) |

## Submission history

 From: Xing Chen [[view email](/show-email/3ecd94cc/2609.04575)]
**[v1]** Fri, 4 Sep 2026 00:11:28 UTC (59 KB)

# Bibliographic and Citation Tools

# Code, Data and Media Associated with this Article

# Recommenders and Search Tools

# arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community? [**Learn more about arXivLabs**](https://info.arxiv.org/labs/index.html).
