# SOURCE: https://arxiv.org/html/2609.04575v1

# Training-Free Halving of Activated Experts
in Fine-Grained Mixture-of-Experts Models

###### Abstract

Modern fine-grained Mixture-of-Experts (MoE) models route each token to a small number of experts and renormalize their router probabilities. We show that this renormalization implicitly calibrates expert output gain to the training top-: reducing at inference changes not only which experts are used but also the strength of the expert branch. We separate these effects by activating the top experts while normalizing by the probability mass of the top experts, introducing one integer with no parameters, training, or measurable compute overhead. On Qwen3.6-35B-A3B, reducing from 8 to 4 experts causes a 4.65-point MMLU drop under standard renormalization but only 0.35 points with , while halving routed-expert compute. The result replicates on the larger Qwen3.5-397B-A17B, where reducing from 10 to 5 experts loses only 0.55 points with an appropriate reference set. Removing renormalization entirely is catastrophic, showing that preserving a suitable reference mass is crucial. We further find that perplexity and downstream accuracy favor different , cautioning against selecting MoE compression settings using unlabeled text alone. Analyses also show that expert identity matters substantially more than expert weighting, while balanced and domain-specialized routing leaves limited room for expert pruning.

## Introduction

Sparse Mixture-of-Experts (MoE) architectures decouple parameter count from per-token compute by activating only a few of many expert sub-networks (Shazeer et al. 2017; Lepikhin et al. 2021; Fedus et al. 2022). Recent open-weight models have pushed this design toward *fine-grained* sparsity: rather than 8 or 16 large experts, they use hundreds of small ones and route each token to a handful of them (Dai et al. 2024; DeepSeek-AI 2024; Qwen Team 2025). In Qwen3.6-35B-A3B, our primary object of study, all 40 layers are MoE layers with 256 experts each and selected per token; the routed experts hold B of the model’s B parameters (89.6%), while each token touches only of them.

This memory–compute asymmetry—all experts must be resident, few are used—is what makes MoE compression attractive and awkward at the same time. Reducing the number of experts (pruning or merging) saves memory but not compute, since is unchanged; reducing saves compute but not memory. A large body of training-free work attacks the memory side by identifying and removing redundant experts (Lu et al. 2024; Li et al. 2024). The compute side, by contrast, is usually treated as trivially adjustable: is an inference-time argument, so one simply lowers it, renormalizes over the survivors, and measures the loss (Chitty-Venkata et al. 2025). The loss is large, and the reported remedies are training-based—retraining with sampled expert counts, or distilling a half-expert student (Wang et al. 2025; Gu et al. 2025; Lv et al. 2026).

We argue this treatment overlooks a detail that turns out to dominate the outcome. Nearly all modern MoE implementations renormalize the selected router probabilities, , so that the mixture weights sum to one. In a coarse-grained router this is nearly a no-op, because the top few probabilities already carry most of the mass. In a fine-grained router it is not: in the model we study, the top-8 of 256 probabilities sum to only , so renormalization multiplies them by on average. That amplification is not a free-standing design choice—it is a quantity the model was trained under, and it is a function of . Lowering from 8 to 4 while keeping the renormalization discards four experts *and* redistributes their weight onto the survivors, inflating the expert branch’s contribution to the residual stream beyond anything the model saw in training. The measured degradation of “top-4” therefore conflates two effects: the loss of four experts, and a gain miscalibration that has nothing to do with expert capacity.

Our contribution is to separate them, using the observation that the numerator and the denominator of the renormalization need not use the same expert set. We activate the top experts but normalize by the mass of the top :

| | | |
|---|---|---|
where controls compute and controls gain. Setting recovers standard renormalization; removes it; anchors the gain to the trained value; and pushes the gain below its trained value, adaptively per token. Only expert FFNs are ever evaluated, and the router already computes all probabilities, so varying costs nothing measurable.

The empirical payoff is large, and it is largest exactly where prior practice reports the reduction of as unaffordable. On 2000-question, 5-shot MMLU with paired McNemar testing, activating 4 of 256 experts per layer instead of 8—halving routed-expert compute—costs accuracy points under standard renormalization (), but only points at (), which the paired test cannot distinguish from the top-8 baseline. The same experiment on Qwen3.5-397B-A17B, an larger model with 512 experts and , gives points at versus points at ().

We summarize our findings as follows.

- •
The renormalization reference set is a free variable. Lowering without decoupling the denominator conflates capacity loss with gain miscalibration. Decoupling it is a one-line change that makes it possible to halve the number of activated experts with no statistically detectable MMLU loss on two models spanning in scale.
- •
The gain, not the weights, is what breaks. Removing renormalization entirely () costs MMLU points and perplexity, so the model is not indifferent to weight magnitudes; and with held fixed, both shrinking and enlarging the reference set away from its optimum degrade the model. The damage is specific to the reference set being wrong, and it is two-sided.
- •
Which experts, not how much. With the number of activated experts held fixed at 8, replacing the router’s weights with uniform ones costs perplexity, while sampling experts instead of taking the argmax costs and random selection . Expert *identity* is worth several times more than expert *weighting*, which rules out approximate or stochastic routing on this model but is precisely what makes a cheap global gain correction viable.
- •
Perplexity picks the wrong operating point. On the 35B model perplexity is minimized at and MMLU at ; the perplexity-optimal setting is significantly worse on MMLU ( points, ). Selecting compression hyperparameters on unlabeled text is not safe here, and we recommend paired significance testing on a downstream task.
- •
Expert pruning has less headroom than this. The router is well balanced with essentially no dead experts, and its specialization is strongly domain-dependent: WikiText and code top-64 expert sets overlap at , below the random baseline, so domain-pruned models do not transfer and the union of two domains’ top-128 sets already covers 198 of 256 experts.

All results are obtained without any training, distillation, or fine-tuning.

### Sparse MoE language models.

Conditional computation via learned routing dates to Shazeer et al. (2017), and was scaled to modern transformers by GShard (Lepikhin et al. 2021) and Switch Transformer (Fedus et al. 2022), which also documented the training instabilities that motivate auxiliary load-balancing losses (Zoph et al. 2022). Alternatives to token-choice top- routing include expert-choice routing (Zhou et al. 2022). Open-weight decoder MoEs such as Mixtral (Jiang et al. 2024) use coarse granularity (8 experts, top-2), whereas DeepSeekMoE (Dai et al. 2024) argued for many small experts plus always-on shared experts, a design adopted by DeepSeek-V3 (DeepSeek-AI 2024) and the Qwen3 series (Qwen Team 2025), and pushed further by He (2024). Our objects of study belong to this fine-grained family, and the effect we describe is a direct consequence of fine granularity: it is the flatness of a 256-way router’s top- mass that makes the renormalization gain large and thus makes miscalibrating it expensive.

### Analyses of learned routing.

Several studies have asked whether experts specialize semantically. OpenMoE (Xue et al. 2024) reported context-independent, largely token-identity-driven routing, and Lo et al. (2024) analyzed routing behavior across layers in open MoE models. Our routing measurements are consistent with the general picture that routers are well balanced, but we reach a sharper conclusion on specialization for this model: the domain effect exceeds a within-domain resampling noise floor by nearly four orders of magnitude, and cross-domain expert overlap falls *below* chance. We also separate layers by attention type—both models interleave linear-attention and full-attention layers in a 3:1 pattern—and find the two classes behave differently, a grouping that prior analyses of homogeneous architectures had no reason to consider.

### Training-free MoE compression.

Expert pruning removes experts judged redundant (Lu et al. 2024) and merging consolidates them using routing statistics (Li et al. 2024); Lasby et al. (2025) report near-lossless one-shot pruning at 50% expert reduction with a router-weighted criterion. These target memory, as does quantization (Frantar et al. 2023; Lin et al. 2024), which suits MoE well since the routed experts are a large homogeneous block; Huang et al. (2025) combine both in a training-free pipeline. On the compute side the closest method to ours is LExI (Chitty-Venkata et al. 2025), which picks a per-layer number of active experts data-free—and renormalizes the surviving router weights, which in our parameterization is , precisely the setting we find worst on both models. The two compose: LExI decides *how many* experts a layer keeps, the reference set decides *how strongly* the survivors drive the residual stream. More generally, we are not aware of prior work that scans the renormalization denominator separately from , and prior top- reduction numbers on fine-grained models are, for this reason, likely to be pessimistic. Some architectures do expose a related knob—GLM-style configurations ship a routed_scaling_factor applied after renormalization—but as a fixed scalar chosen at training time.

### Changing after training.

That MoE models degrade sharply when the number of activated experts is altered at inference is well documented—Wang et al. (2025) call the degradation “precipitous”—and the reported remedies are *training-based*: retraining with randomly sampled expert counts (Wang et al. 2025), training expert combinations so that can be scaled *up* (Gu et al. 2025), null experts that let the effective vary per token (Zeng et al. 2024), and, closest to our headline result, self-distilling a post-trained MoE that skips roughly half its experts (Lv et al. 2026). That last work also argues, as we do, that renormalizing the surviving weights inflates the effective scale of the expert residual branch relative to what pre-training calibrated. Our contribution relative to it is twofold: our correction needs no training, and we find the choice is not the binary “renormalize or do not”—those are the endpoints and of a family whose useful settings are interior. Which endpoint is less wrong depends on the model’s native forward pass: for a model that does not renormalize, not renormalizing preserves the trained gain, whereas both models here renormalize by construction and removing it costs MMLU points. The invariant is not a rule about renormalization but the reference set.

## Background and Setup

### Models.

We study two fine-grained MoE models from the same series but very different scales (Table 1). Qwen3.6-35B-A3B has 40 layers, hidden size 2048, and 256 routed experts of intermediate size 512 per layer with , plus one always-on shared expert per layer modulated by a learned scalar gate; every layer is an MoE layer. Qwen3.5-397B-A17B has 60 layers, hidden size 4096, and 512 routed experts of intermediate size 1024 with . Both interleave linear-attention (gated delta-net style) and full-attention blocks in a repeating 3:1 pattern. For the 35B model we enumerated tensors in the released checkpoint: routed experts hold 89.6% of all parameters (B of B; the remainder is B of attention, embeddings, shared experts and norms, a B multi-token-prediction head, and a B vision tower). Each token activates of 256 experts per layer, i.e. of expert parameters—a memory-to-compute asymmetry that organizes the rest of this paper.

| | Qwen3.6 | Qwen3.5 |
|---|---|---|
| | 35B-A3B | 397B-A17B |
| Layers (all MoE) | 40 | 60 |
| Hidden size | 2048 | 4096 |
| Experts per layer | 256 | 512 |
| Trained | 8 | 10 |
| Expert interm. size | 512 | 1024 |
| Shared experts | 1 | 1 |
| Top- router mass | 0.182 | 0.192 |
| Renorm. gain | | |
| Act. expert params, native | 1.132 B | 8.305 B |
| at / | 0.881 B | 6.795 B |
| at | 0.629 B | 4.530 B |

### Router and the implicit gain.

For hidden state at a given layer, the router computes over all experts, selects the index set of the largest entries, and forms the layer output

| | | | (1) |
|---|---|---|---|
The denominator is what concerns us. Write for the retained probability mass. Renormalization multiplies the raw probabilities by , and since increases with , this gain is implicitly tied to the used in training. We measure on WikiText and on code for the 35B model, i.e. average amplifications of and ; for the 397B model , i.e. . Dropping to reduces and hence *raises* the gain applied to the surviving experts, pushing the expert branch out of the regime the model was trained in. This is the miscalibration we correct.

### Method: decouple the reference set.

We keep the selection rule and the numerator, and replace only the denominator’s index set:

| | | | (2) |
|---|---|---|---|
with when . The parameterization is a strict generalization of common practice:

- •
: standard renormalization (Eq. 1);
- •
 (trained value): the gain is anchored to what training calibrated;
- •
: no renormalization at all, since ;
- •
: gain below the trained value, applied per token.
The weights no longer sum to one when ; they sum to , which is a *per-token, input-adaptive* down-scaling rather than a constant. Averaged over tokens this ratio is for and for on the 35B model, so the discrete grid we scan spans the range a tuned global scalar would explore, without introducing a continuous hyperparameter.

Only expert FFNs are evaluated, so the compute saving is exactly that of reducing to . The extra cost of a larger is a wider top- over router logits that are computed for all experts anyway—negligible against expert FFNs. Both are implemented by replacing the router’s forward at inference time; no weights are modified.

### Evaluation protocol.

Perplexity is measured on 100K-token samples of WikiText-103 (Merity et al. 2017) and a Python subset of CodeParrot (Tunstall et al. 2022), at a 2048-token context, with a held-out WikiText split (drawn beyond row 400,000, disjoint from the tuning split) used to check that the choice of does not overfit. All configurations are scored on byte-identical token chunks. Downstream accuracy is measured on MMLU (Hendrycks et al. 2021): 2000 questions sampled with a fixed seed from all 14,042 questions across all 57 subjects, 5-shot, scored by comparing the logits of the single tokens A–D; on C-Eval (Huang et al. 2023) (1300 questions, 5-shot, Chinese); and, as a generation task, on GSM8K (Cobbe et al. 2021) (500 questions, 0-shot, greedy to 512 tokens, reasoning disabled via the chat template, exact-match on the final number). All configurations run on *identical* question sets, compared to the baseline with a paired McNemar test (McNemar 1947). Pairing is not optional at these effect sizes: a power analysis at gives a minimum detectable difference of paired versus unpaired, and our effects are –.

## How the Router Actually Behaves

Before modifying the router we characterize it, using forward hooks on all gates to record the selected expert indices and softmax weights over 500K tokens per domain (WikiText and code), giving an expected 15,680 activations per expert per layer on the 35B model.

### Load is well balanced; there is no obvious fat to cut.

Normalized activation entropy is (WikiText) and (code), with Gini coefficients of and . The number of never-activated experts per layer has median and maximum across all layers and both domains. The 397B model behaves similarly (entropy , Gini , dead experts median of 512). Figure 1(b) shows the resulting retention curves: keeping the top 50% of experts per layer retains only (WikiText) and (code) of routing mass, and the worst layer retains just . A router trained with load-balancing pressure leaves little slack for pruning, which is the first reason we look elsewhere for compute savings.

### Experts are strongly domain-specialized.

Comparing per-layer expert distributions across domains yields a mean Jensen–Shannon divergence of nats. To calibrate this we estimate a noise floor by multinomial resampling within a domain at matched counts, obtaining nats; the observed divergence is roughly the floor, so it is not a sampling artifact. More strikingly, the top-64 expert sets of the two domains overlap at , *below* the random baseline of : the domains do not merely prefer different experts, they actively avoid each other’s. Specialization increases with depth (Figure 1(a)), from nats at layer 0 to at layer 39.

### But specialization does not make pruning transferable.

Pruning to the top-128 experts using code-derived rankings retains of code routing mass; using WikiText-derived rankings retains only on the same domain—a factor of two. Meanwhile the union of the two domains’ top-128 sets covers 198 of 256 experts. Strong specialization is thus a double-edged result: it licenses *domain-specific* pruned models, but it forecloses a single general-purpose pruned model, because any expert set broad enough to serve both domains is barely smaller than the original.

### Attention type structures routing.

Grouping layers by attention type reveals structure that a layer-agnostic average hides. Full-attention layers share expert preferences with each other (cross-layer top-64 overlap versus the random baseline), whereas linear-attention layers are mutually near-independent (), and cross-type pairs are indistinguishable from chance (). The shared-expert gate follows the same split: mean on full-attention layers versus on linear-attention layers, rising with depth to at layer 39. We report this as a caution for routing analyses of the increasingly common hybrid-attention architectures: layers are not interchangeable samples.

## Expert Selection Matters More Than Expert Weighting

If a global correction to the router weights is to work, the model must be more sensitive to expert selection than to the precise weight values. We test this directly with three ablations that all keep the number of activated experts at exactly 8, so compute is held constant (Table 2).

| | WikiText | | Code | |
| Routing rule | PPL | | PPL | |
|---|---|---|---|---|
| top- (native) | 6.64 | — | 2.59 | — |
| top-, uniform | 9.95 | +49.7% | 3.11 | +20.3% |
| sampled without repl. | 27.22 | +310% | 6.09 | +135% |
| uniformly random | 117.39 | +1667% | 18.69 | +623% |
Discarding the router’s weights entirely but keeping its choices costs perplexity on WikiText. Keeping the router’s distribution but sampling from it instead of taking the argmax costs , and choosing experts uniformly at random costs . Selection is worth roughly six times more than weighting on both domains.

This has two consequences. Negatively, it rules out a family of efficiency methods on this model: approximate top-, locality-sensitive-hashing routers, and any scheme that injects stochasticity into selection will be catastrophic, because exact argmax is effectively a hard requirement. Positively, it is what makes our method plausible. Since the model tolerates substantial distortion of the weight *magnitudes* as long as the selected set is right, correcting the systematic component of that distortion is enough—and reducing produces exactly a systematic, not idiosyncratic, magnitude error.

## The Reference Set Controls the Damage

### The router’s top- mass is small.

Table 3 reports the mean softmax mass by rank on the 35B model. The single highest-scoring expert receives only of the probability mass, the top-8 together only , and even the top-32 only . The distribution is remarkably flat, as one would expect from a 256-way router trained with load balancing. Renormalization therefore performs a amplification, and that factor is a property of .

| Domain | rank 1 | top-4 | top-6 | top-8 | top-16 |
|---|---|---|---|---|---|
| WikiText | 0.046 | 0.122 | 0.155 | 0.182 | 0.262 |
| Code | 0.046 | 0.116 | 0.146 | 0.170 | 0.245 |

### Both extremes of the reference set are harmful.

Two observations bracket the effect. Removing renormalization altogether () at raises WikiText perplexity from to () and drops MMLU by points; the model depends on the amplification rather than merely tolerating it. At the other extreme, shrinking the reference set along with () costs perplexity and MMLU points at , and MMLU points at . Between them, in the range recovers nearly all of it. Perplexity as a function of is thus U-shaped with an interior minimum, and MMLU is inverted-U with an interior maximum (Figure 2); neither optimum sits at an endpoint of the scanned grid.

### Perplexity.

Table 4 gives the perplexity scan. On the 35B model at , standard renormalization costs , while yields —slightly *better* than the native top-8 baseline while activating 22% fewer expert parameters. The held-out WikiText split, disjoint from the split used to choose , reproduces the ranking exactly ( at ), as does code (), so the choice is not an artifact of the tuning split. The 397B model shows the same ordering at , with the best of the grid on all three corpora (WikiText, held-out, and code). Across both models and all six corpus/model combinations, perplexity is minimized at equal to the model’s trained and degrades monotonically as moves away in either direction.

| Configuration | WikiText | Held-out | Code |
|---|---|---|---|
| *Qwen3.6-35B-A3B* (trained , ) | | | |
| native | 6.643 | 6.986 | 2.585 |
| , | +4.08% | +4.69% | +1.72% |
| , | –1.61% | –0.57% | +0.68% |
| , | +1.81% | +1.42% | +3.29% |
| , | +23.25% | +19.04% | +13.00% |
| , | +356% | +313% | +158% |
| *Qwen3.5-397B-A17B* (trained , ) | | | |
| native | 4.159 | 3.198 | 2.249 |
| , | +1.06% | +1.54% | +0.42% |
| , | +0.46% | +0.56% | +0.34% |
| , | +4.98% | +4.95% | +2.18% |
| , | +17.18% | +18.11% | +6.87% |
| , | +128% | +146% | +56.04% |

## Downstream Validation

Perplexity on 100K tokens is weak evidence for a claim about model capability. We therefore validate on MMLU under the paired protocol described earlier (Table 5), and confirm the conclusion on a generation task (GSM8K) and a Chinese benchmark (C-Eval) below.

| Configuration | Acc. | (pp) | flips | McNemar |
|---|---|---|---|---|
| *Qwen3.6-35B-A3B*, | | | | |
| native | 81.65% | — | — | — |
| , | 79.05% | –2.60 | 82 / 30 | |
| , | 80.55% | –1.10 | 53 / 31 | 0.021 |
| , | 82.40% | +0.75 | 74 / 89 | 0.27 |
| , | 54.20% | –27.45 | 637 / 88 | |
| , | 77.00% | –4.65 | 166 / 73 | |
| , | 78.55% | –3.10 | 97 / 35 | |
| , | 81.30% | –0.35 | 99 / 92 | 0.66 |
| *Qwen3.5-397B-A17B*, | | | | |
| native | 89.45% | — | — | — |
| , | 89.60% | +0.15 | 23 / 26 | 0.78 |
| , | 89.00% | –0.45 | 28 / 19 | 0.24 |
| , | 88.45% | –1.00 | 46 / 26 | 0.024 |
| , | 87.35% | –2.10 | 78 / 36 | |
| , | 88.90% | –0.55 | 42 / 31 | 0.24 |
| , | 87.75% | –1.70 | 64 / 30 | |
| , | 70.90% | –18.55 | 410 / 39 | |

### Halving activated experts is free, if the denominator is not halved too.

Activating 4 of 8 experts on the 35B model costs points under standard renormalization but at , with a near-symmetric flip distribution (); activating 5 of 10 on the 397B model costs points at but at (). The reference set recovers and points at identical compute. A non-significant result is not proof of equivalence—the paired design only bounds any residual loss below the -point resolution at .

### The optimal is model-dependent, and lies in .

The 35B model prefers at both we tried; the 397B model prefers ; neither prefers once the cut is deep. We therefore do not propose a universal value but that be scanned over whenever is reduced. The consistent finding is the negative one: , every implementation’s default, is the worst choice in that grid.

### The ablation structure matters.

Had we evaluated only the baseline and our final candidate—common practice—the 35B result would read as “top-4 is simply lossless” and attribute nothing to the reference set; only the row shows the reduction is damaging by default. Conversely, the 397B block, where the default is already fine (), shows the opposite: reporting only that row would conclude renormalization never matters.

### Perplexity picks a significantly worse operating point.

On the 35B model perplexity is minimized at but MMLU at (Figure 2(a)): the perplexity-optimal setting is significantly worse on MMLU ( points, ), and the MMLU-optimal one worse on perplexity. Since selecting on unlabeled text is exactly the cheap protocol one would reach for, we regard this as the main methodological finding—compression settings chosen on perplexity alone should be treated as unvalidated. This sharpens the known pattern that pruned LLMs retain perplexity while degrading on knowledge tasks (Jaiswal et al. 2024): here the two metrics are not merely of different sensitivity but optimized at different points, so perplexity mis-ranks configurations.

### Generation: the effect survives autoregressive decoding.

A single-token multiple-choice score cannot reveal damage that accumulates over a generated sequence, so we repeat the experiment on GSM8K (Table 6). The pattern matches MMLU, and is if anything sharper. Reducing to under standard renormalization costs points (35B) and (397B)—larger than the corresponding MMLU losses, as expected if a miscalibrated gain compounds token by token—both highly significant. Anchoring the reference set to the native erases the loss: , costs points (35B, ) and , costs (397B, ), neither detectable. The truncation rate tracks the accuracy loss closely—it reaches at the worst 35B setting and falls to once the reference set is corrected—so the default reduction does not merely change answers but degrades the model into unterminated outputs.

| Configuration | Acc. | (pp) | McNemar | trunc. |
|---|---|---|---|---|
| *Qwen3.6-35B-A3B*, | | | | |
| native | 95.00% | — | — | 2.8% |
| , | 92.80% | –2.20 | 0.027 | 6.0% |
| , | 95.00% | +0.00 | 1.00 | 2.2% |
| , | 88.40% | –6.60 | | 10.4% |
| , | 94.40% | –0.60 | 0.68 | 1.4% |
| *Qwen3.5-397B-A17B*, | | | | |
| native | 96.60% | — | — | 2.6% |
| , | 94.60% | –2.00 | 0.006 | 3.8% |
| , | 96.20% | –0.40 | 0.63 | 2.2% |
| , | 92.40% | –4.20 | | 7.0% |
| , | 95.80% | –0.80 | 0.34 | 1.6% |

### C-Eval: insensitivity is model-specific, not benchmark-specific.

On C-Eval (1300 Chinese questions, 5-shot) the 35B model shows no significant difference for *any* configuration, including the default reduction (, : , ; , : , ; baseline ). Read alone, this benchmark would license the default reduction that MMLU and GSM8K reject. Yet the same benchmark on the 397B model rejects it: , loses points (; baseline ), and anchoring to the native recovers most of it (, : ; , : , ). The recovery direction is consistent with every other result—anchoring helps, is worst—but a benchmark’s *sensitivity* to the intervention is not a property of the benchmark alone: the very C-Eval that certifies the default reduction on one model refutes it on another. Validating on a single (benchmark, model) pair risks certifying an artifact of that pair—our perplexity caution, one level up.

### Cost accounting.

On the 35B model, halving from 8 to 4 cuts routed-expert compute in half (B to B activated routed parameters per token) and the total expert term by (B to B), a end-to-end reduction against the B parameters per token—though the expert term dominates weight traffic during memory-bound decoding. Memory is unchanged (all experts stay resident), so this is a pure compute-side lever, orthogonal to quantization.

## Discussion

### Practical recommendation.

When compute or decode bandwidth binds, decouple the reference set from the activation count whenever is changed and scan with paired testing: one integer, three evaluations, and the difference between a significant – point MMLU regression and none. It is also a control for any published top- reduction result, which absent a decoupled overstates the cost of reducing .

### Why we do not recommend expert pruning here.

Structural compression looks like the harder path on this architecture: the router is well balanced with essentially no dead experts, retention falls off quickly (Figure 1(b)), and specialization is domain-bound, so a general-purpose pruned model has almost no room ( experts in the two-domain union). Since the routed experts are homogeneous and of parameters, quantization looks better matched to the memory problem and composes with our compute-only method. This claim rests on activation-mass retention and cross-domain transfer; it does not contradict the near-lossless single-domain pruning of Lasby et al. (2025).

### Limitations.

Both models are from the same series; though they differ by in parameters and in depth, expert count, and trained , cross-architecture generalization remains a hypothesis. The mechanism predicts the effect scales with router flatness, so coarse-grained MoEs (e.g. 8 experts, top-2) should benefit little; a preliminary check on a third architecture (Appendix A) is consistent but ran under an earlier scalar parameterization and is not directly comparable. Our corpora are English and Python only, long-context is untested, and the perplexity-optimal is already domain-dependent. Downstream we cover multiple-choice (MMLU, C-Eval) and short-form generation (GSM8K) but not long-form generation. Finally, we scan over a geometric grid; a finer grid, or a per-layer or per-domain , may do better.

## Conclusion

Training-free reduction of activated experts in a fine-grained MoE is usually presented as a compute/accuracy trade-off. We showed it is confounded by an implicit variable: renormalization applies a gain calibrated to the trained , and lowering shrinks the denominator with the activation count, silently miscalibrating the expert branch. Decoupling the two—activate experts, normalize by the top- mass—turns a point MMLU regression at half the activated experts into points (35B) and a point regression into points (397B), for one integer and no measurable compute. Because perplexity and downstream accuracy select different reference sets, this knob should be validated on a downstream task with paired testing, not on perplexity.

## References

- LExI: layer-adaptive active experts for efficient MoE model inference. arXiv preprint arXiv:2509.02753. Cited by: Introduction, Training-free MoE compression..
- Training verifiers to solve math word problems. Cited by: Evaluation protocol..
- DeepSeekMoE: towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), Cited by: Introduction, Sparse MoE language models..
- DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437. Cited by: Introduction, Sparse MoE language models..
- Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research 23 (120), pp. 1–39. Cited by: Introduction, Sparse MoE language models..
- GPTQ: accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations (ICLR), Cited by: Training-free MoE compression..
- Elastic MoE: unlocking the inference-time scalability of mixture-of-experts. arXiv preprint arXiv:2509.21892. Cited by: Introduction, Changing after training..
- Mixture of a million experts. arXiv preprint arXiv:2407.04153. Cited by: Sparse MoE language models..
- Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), Cited by: Evaluation protocol..
- Mixture compressor for mixture-of-experts LLMs gains more. In International Conference on Learning Representations (ICLR), Cited by: Training-free MoE compression..
- C-Eval: a multi-level multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: Evaluation protocol..
- Compressing LLMs: the truth is rarely pure and never simple. In International Conference on Learning Representations (ICLR), Cited by: Perplexity picks a significantly worse operating point..
- Mixtral of experts. arXiv preprint arXiv:2401.04088. Cited by: Sparse MoE language models..
- REAP the experts: why pruning prevails for one-shot MoE compression. arXiv preprint arXiv:2510.13999. Cited by: Training-free MoE compression., Why we do not recommend expert pruning here..
- GShard: scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations (ICLR), Cited by: Introduction, Sparse MoE language models..
- Merge, then compress: demystify efficient SMoE with hints from its routing policy. In International Conference on Learning Representations (ICLR), Cited by: Introduction, Training-free MoE compression..
- AWQ: activation-aware weight quantization for on-device LLM compression and acceleration. In Proceedings of Machine Learning and Systems (MLSys), Cited by: Training-free MoE compression..
- A closer look into mixture-of-experts in large language models. arXiv preprint arXiv:2406.18219. Cited by: Analyses of learned routing..
- Not all experts are equal: efficient expert pruning and skipping for mixture-of-experts large language models. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), Cited by: Introduction, Training-free MoE compression..
- Post-trained MoE can skip half experts via self-distillation. arXiv preprint arXiv:2605.18643. Cited by: Introduction, Changing after training..
- Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12 (2), pp. 153–157. Cited by: Evaluation protocol..
- Pointer sentinel mixture models. In International Conference on Learning Representations (ICLR), Cited by: Evaluation protocol..
- Qwen3 technical report. arXiv preprint arXiv:2505.09388. Cited by: Introduction, Sparse MoE language models..
- Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations (ICLR), Cited by: Introduction, Sparse MoE language models..
- CodeParrot: a GPT-2 model trained to generate Python code. Note: https://huggingface.co/datasets/codeparrot/codeparrot-clean Cited by: Evaluation protocol..
- Training matryoshka mixture-of-experts for elastic inference-time expert utilization. arXiv preprint arXiv:2509.26520. Cited by: Introduction, Changing after training..
- OpenMoE: an early effort on open mixture-of-experts language models. arXiv preprint arXiv:2402.01739. Cited by: Analyses of learned routing..
- AdaMoE: token-adaptive routing with null experts for mixture-of-experts language models. In Findings of the Association for Computational Linguistics: EMNLP, Cited by: Changing after training..
- Mixture-of-experts with expert choice routing. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: Sparse MoE language models..
- ST-MoE: designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906. Cited by: Sparse MoE language models..

## Appendix A Appendix A: A Third Architecture

Before adopting the parameterization we ran a preliminary study on Gemma-4-26B-A4B (30 layers, 128 experts, top-8). It used a global scalar gain —a fixed, rather than per-token, version of the correction studied above—and a different evaluation protocol, so its numbers are not directly comparable to the main results and we report them as directional evidence only. They are nonetheless informative on two points: whether a scalar gain helps on a third architecture, and whether the flatness mechanism predicts the size of the effect.

### The scalar gain helps but does not fully recover.

On MMLU (, 5-shot, chat template, instruction-tuned variant, baseline ), reducing to at the native costs points (Table 7). Scanning recovers up to point: the best value reaches (), the only non-significant setting, but unlike the two Qwen models it does not return all the way to the baseline. As on Qwen, the “natural” guess is not the empirical optimum.

| Configuration | Acc. | (pp) | McNemar |
|---|---|---|---|
| native | 80.40% | — | — |
| , | 78.10% | –2.30 | 0.0018 |
| , | 78.60% | –1.80 | 0.015 |
| , | 78.60% | –1.80 | 0.020 |
| , | 79.10% | –1.30 | 0.079 |
| , | 78.80% | –1.60 | 0.033 |

### The amplitude degree of freedom is untrained on all three models.

Gemma exposes two learnable scale parameters in its router. The pre-softmax temperature router.scale was trained hard: it moved from its initialization of to a near-constant across all 30 layers (an effective gain of on the RMSNorm-scaled logits). The post-selection amplitude per_expert_scale, by contrast, stayed at its initialization: all values are , essentially untouched, because it is mathematically redundant with each expert’s down_proj and the gradient flows to the larger-magnitude weights instead. The amplitude—the exact quantity our controls—is therefore a degree of freedom that training never optimized on any of the three models we examined: Qwen exposes no such parameter, Gemma has one but left it at , and GLM-style configurations set it by hand. This strengthens rather than weakens the main claim: we are not correcting an already-optimized quantity but filling a gap the training procedure leaves open.

### The flatness mechanism predicts the effect size.

The mechanism in the main text predicts the effect should scale with router flatness. Gemma’s top-8 router mass is versus for Qwen3.6-35B-A3B, i.e. a renormalization gain of rather than . Correspondingly the gain is far less load-bearing: on the base variant, removing renormalization entirely at () raises WikiText perplexity only from to (), and is in fact *better* than standard renormalization at the same ()— whereas the identical intervention costs on Qwen. The selection-over-weighting result also replicates: with the eight activated experts fixed, uniform weights cost WikiText perplexity, sampling , and random selection . A matched replication on this architecture remains future work.

### A protocol caution.

Gemma is a reasoning model whose base and instruction-tuned variants must be evaluated differently: the instruction-tuned variant scores WikiText perplexity of under plain completion but for the base variant, because bare completion triggers degenerate repetition. All Gemma perplexity numbers above use the base variant; the MMLU scan uses the instruction-tuned variant under its chat template. This protocol dependence is the same one noted in Appendix B.

## Appendix B Appendix B: Reproducibility Notes

### Infrastructure.

All experiments use PyTorch with the HuggingFace Transformers library; the method is a drop-in replacement of the router’s forward, and no weights are modified. We verified that the router computation is numerically identical across Transformers versions 5.3–5.14 (only variable names changed), reproducing the WikiText baseline perplexity exactly. Qwen3.6-35B-A3B runs in bf16 on a single 98 GB accelerator (72 GB resident); Qwen3.5-397B-A17B is sharded across the machine (752 GB in bf16). All routing and downstream configurations are inference-only.

### Protocol pitfalls.

Several protocol details changed our results by more than the effects we set out to measure, and we record them for others. An initial 0-shot, per-option-loglikelihood MMLU protocol scored the 35B baseline at —implausible for a model of this size—and standard 5-shot single-letter scoring restored it to ; conclusions drawn under the first protocol would have been meaningless. The model emits reasoning traces spontaneously even under plain completion formatting, which exhausts the token budget on generation tasks unless thinking is explicitly disabled via the chat template. The released generation config defaults to sampling, which must be disabled for paired comparison. Evaluation protocol must be matched to model form, a failure that produced the Gemma perplexity anomaly discussed in Appendix A and that we initially misattributed to a framework bug. And, as noted, unpaired testing at cannot resolve differences below , larger than most effects reported here.
