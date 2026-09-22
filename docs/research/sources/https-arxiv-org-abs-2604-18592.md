# SOURCE: https://arxiv.org/abs/2604.18592

[Submitted on 27 Mar 2026]

# Title:Two-dimensional early exit optimisation of LLM inference

[View PDF](/pdf/2604.18592)[HTML (experimental)](https://arxiv.org/html/2604.18592v1)

> Abstract:We introduce a two-dimensional (2D) early exit strategy that coordinates layer-wise and sentence-wise exiting for classification tasks in large language models. By processing input incrementally sentence-by-sentence while progressively activating deeper layers, our method achieves multiplicative computational savings that exceed those from optimizing either dimension independently. Experimental evaluation across four state-of-the-art LLMs (Llama 3.1, Llama 3.2, Gemma, Qwen; 3B-8B parameters) on three sentiment classification datasets demonstrates additional speed-ups of 1.4--2.3$\times$ over optimal layer-wise early exit for simpler tasks with vanilla models, with graceful degradation on complex multi-class problems. Fine-tuning reduces but does not eliminate this advantage. The approach is model-agnostic, requires only lightweight classification adapters, and is orthogonal to complementary efficiency methods such as quantization and pruning. Our findings indicate that 2D early exit strategies excel when semantic information accumulates predictably across input structure, suggesting possible applicability to sequence-processing tasks beyond sentiment classification.
