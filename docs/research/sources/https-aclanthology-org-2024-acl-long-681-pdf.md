# SOURCE: https://aclanthology.org/2024.acl-long.681.pdf

Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12622–12642
August 11-16, 2024 ©2024 Association for Computational Linguistics

LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding

Mostafa Elhoushi1∗†, Akshat Shrivastava1∗†, Diana Liskovich2†, Basil Hosmer1, Bram
Wasti2, Liangzhen Lai3, Anas Mahmoud4, Bilge Acun1, Saurabh Agrawal6, Ahmed
Roman7, Ahmed A Aly3, Beidi Chen1,5, Carole Jean-Wu1,
1FAIR at Meta
2GenAI at Meta
3Reality Labs at Meta
4University of Toronto
5Carnegie Mellon University
6University of Wisconsin-Madison
7Dana-Farber Cancer
Institute
m.elhoushi@ieee.org
akshats@meta.com

Abstract

We present LayerSkip, an end-to-end solution
to speed-up inference of large language mod-
els (LLMs). First, during training we apply
layer dropout, with low dropout rates for ear-
lier layers and higher dropout rates for later
layers, and an early exit loss where all trans-
former layers share the same exit. Second, dur-
ing inference, we show that this training recipe
increases the accuracy of early exit at earlier
layers, without adding any auxiliary layers or
modules to the model. Third, we present a
novel self-speculative decoding solution where
we exit at early layers and verify and correct
with remaining layers of the model. Our pro-
posed self-speculative decoding approach has
less memory footprint than other speculative
decoding approaches and benefits from shared
compute and activations of the draft and ver-
ification stages. We run experiments on dif-
ferent Llama model sizes on different types
of training: pretraining from scratch, contin-
ual pretraining, finetuning on specific data do-
main, and finetuning on specific task.
We
implement our inference solution and show
speedups of up to 2.16× on summarization for
CNN/DM documents, 1.82× on coding, and
2.0× on TOPv2 semantic parsing task. We
open source code at https://github.com/
facebookresearch/LayerSkip.

1
Introduction

Large Language Models (LLMs) have been de-
ployed to many applications, yet their high com-
pute and memory requirements lead to high fi-
nancial and energy costs when deployed to GPU
servers Samsi et al. (2023). Acceleration solutions
do exist to deploy to commodity GPUs on lap-
tops but they suffer from significant drop in accu-
racy Zhu et al. (2023). Accelerating LLMs further
to mobile or edge devices is still an active research

*Equal Contribution.
†Core Contributors.

T1
T2
T3
T4

LM Head

T5

Self Speculation Decoding

T1
T2
T3
T4

LM Head

Early Exit Inference

Train using Layer Dropout + Early Exit Loss....
... enables inference with subset of layers
 with higher accuracy...
... and we can improve accuracy by verifying 
and correcting with remaining layers

Skipped

Computed

Skipped w. Probability

Legend

Cached

T1
T2
T3
T4

LM Head

Training 
with Layer Dropout 
& Early Exit Loss

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Figure 1: Overview of our end-to-end solution, Layer-
Skip, showing its 3 components.

area Çöplü et al. (2023); Liu et al. (2024). While a
large portion of LLM acceleration approaches re-
duce number of non-zero weights Xia et al. (2023)
(a.k.a. sparsity), number of bits per weight Xiao
et al. (2023) (a.k.a. quantization), number of heads
per layer Shim et al. (2021) (a.k.a. head pruning),
a smaller portion of approaches focus on reduc-
ing number of layers Fan et al. (2020); Elbayad
et al. (2020). In this paper, we explore reducing the
number of layers required for each token by exit-
ing early during inference. Unlike quantization or
sparsity, acceleration by reducing number of layers
does not require specialized hardware or software
kernels.
Moreover, a popular research trend in LLM ac-
celeration is speculative decoding Leviathan et al.
(2023); Chen et al. (2023) that has no drop in ac-
curacy, where a large model, referred to as the
main model, is accompanied with a faster model,
referred to as the draft model. The advantage of
speculative decoding is that it leads to faster infer-
ence compared to the main model, but requires a
larger memory footprint and complexity in imple-
mentation to maintain key-value (KV) cache in two
different models. In addition to exiting early, this
paper also proposes combining exiting early with

12622


speculative decoding to propose a self-speculative
decoding approach that does not require an addi-
tional model or auxiliary layers.
The contribution of this paper is an end-to-end
solution:

• a training recipe that combines layer dropout
and early exit loss, that leads to,
• inference that is more robust to exiting at ear-
lier layers of the model, essentially creating
different sized sub-models within the same
model, and
• a self-speculative decoding solution that de-
codes with earlier layers and verifies and cor-
rects with later layers.
The solution achieves speedups between 1.34×
and 2.16× depending on the task. We provide an
overview of the solution in Figure 1.

2
Motivation

2.1
Exiting Earlier in LLMs

To motivate our approach, we investigate, with an
example prompt, what happens in each layer in a
LLM. In Figure 2a, we provide the first prompt
from the HumanEval coding dataset Chen et al.
(2021) to a pretrained Llama1 7B model Touvron
et al. (2023a). The prompt consists of a Python
function header and a docstring, and the model au-
tocompletes it by defining the function body. When
generating each token, we probe each transformer
layer in the LLM by projecting its output embed-
dings on the language model (LM) head (that con-
sists of the model’s final layer normalization and
linear layer), applying softmax, and then obtain-
ing the index of the output element with highest
value. The resulting index corresponds to the pre-
dicted token at this layer. This operation is referred
to in some literature as the unembedding opera-
tion Phuong and Hutter (2022); Cancedda (2024),
as it converts an embedding to an index. Unembed-
ding at each layer is equivalent to early-exit at that
layer, i.e., it is equivalent to skipping the remaining
transformer layers to the model’s LM head.
The token predictions across layers in Figure 2b
illustrate the evolution of embeddings from an in-
put token fed to the model to the predicted next
token by the model. When analyzing the token
prediction in each layer in Figure 2b, we make a
few observations. First, token predictions in earlier
layers appear to be irrelevant as they correspond to
the previous token projected on the model’s embed-
ding layer’s weights, which are different from the

weights of the LM head. In later layers, token pre-
dictions converge to the final prediction. Second,
we do not always need all the layers to predict the
correct token. In fact, most of the time, the final
token prediction is predicted fewer layers before
the end. We also notice that intermediate layers are
sometimes hesitant and “change their minds”, e.g.,
for Token 05, the model was predicting “range” as
early as Layer 07, but changed its mind between
Layer 22 and Layer 26, before settling again on
“range”.

Similar analysis was done in Geva et al. (2022)
on a GPT2 model Radford et al. (2019) as it devel-
oped predictors to estimate when prediction satu-
rates to exit early. For the particular example we
present in Figure 2, we find, on average, a token
requires 23.45 layers out of the model’s 32 lay-
ers. Hence, even if we have a perfect predictor
that has zero compute overhead, we can only save
up to 26% of computation. Therefore, there is a
need to make LLM models require fewer layers to
predict each token, and spend less compute being
hesitant or “changing its mind”. By default, deep
learning models are not motivated to predict their
final output early and instead spread their compute
across all layers Voita et al. (2019, 2023). We see
in Figure 2b, that tokens we would consider easy or
straightforward to predict, e.g., Token 02 that starts
a for-loop, required all 32 layers to predict “for”.
We would like our model to be less reliant on later
layers and only use later layers for harder tokens.
We would like our models to be more reliant on ear-
lier layers than later layers. To do that, we propose
skipping layers during training, which we refer to
as layer dropout. However, we use higher dropout
rates for later layers and lower dropout rates for
earlier layers, to make the model less reliant on
later layers.

Moreover, LM heads in LLMs are trained to
unembed embeddings from the last transformer
layer. They were not trained to unembed from
earlier layers. Therefore, our solution also adds a
loss function during training to make LM heads
better “understand” embeddings of earlier layers.
While most papers that explored early exit Schus-
ter et al. (2022); Elbayad et al. (2020) trained a
dedicated LM head for each transformer layer, and
some have introduced additional modules for each
early exit Zhang et al. (2019), we chose to have
a shared LM head for all transformer layers in
the model.
This makes training faster, require

12623


Prompt:
from typing import List
def has_close_elements(numbers: List[float], threshold:
float) -> bool:

"""
Check if in given list of numbers, are any two
numbers closer to each other than given threshold.

>>> has_close_elements([1.0, 2.0, 3.0], 0.5)
False
>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0,
2.0], 0.3)

True
"""

Generation:
for i in range(len(numbers)):\n

for j in range(i+1, len(numbers)):\n

if abs(numbers[i] - numbers[j]) < threshold:\n

return True\n
return False\n

(a)

Token 01
Token 02
Token 03
Token 04
Token 05
Token 06
Token 07
Token 08
Token 09
Token 10
Token 11
Token 12
Token 13
Token 14
Token 15
Token 16
Token 17

Layer 00
...
//
eground
externs
EV
Anleitung
this
зня
**
hips
academ
...
//
eground
unction
EV
Anleitung

Layer 01
and
//
instance
Pod
response
atel
self
View
self
AS
timing
everybody //
sure
osh
hib
ous

Layer 02
Dark
#
ego
Pod
pse
ula
**
tern
self
AT
__
SO
//
instance
osh
isti
ula

Layer 03
<<
//
ego
tak
ula
s
łu
self
Helper
Hem
Rein
//
_,
tak
ula

Layer 04
dorf
//
ego
range
town
pract
Nar
zero
paces
Hem
`.`
//
isu
input
iter
ula

Layer 05
Pay
if
instance
Sold
range
widet
etra
Hem
paces
fib
ioned
if
isu
fish
range
ula

Layer 06
(...)
if
isu
iter
iska
̂
тика
Trace
SR
fib
ioned
if
za
Gree
iter
iri

Layer 07
hoff
if
_,
range
("@
̂
тика
,
piel
Kn
ioned
if
pat
Gree
range
("@

Layer 08
\n
if
instance
range
("@
len
тика
ulp
paces
AT
ioned
if
pat
range
("@

Layer 09
Cow
return
loop
range
stag
self
of
this
cope
ori
ioned
if
wards
forg
range
(

Layer 10
return
return
i
range
stag
len
ect
Maz
smallest
Cop
ioned
if
i
Sn
range
till

Layer 11
return
return
i
range
val
self
<=
this
ut
ida
ioned
if
i
enda
range
iras

Layer 12
return
return
i
話
range
stag
len
kir
din
plit
cep
ioned
if
i
pent
range
iras

Layer 13
return
return
i
range
stag
len
<=
self
plit
oss
ioned
if
i
oth
range
iras

Layer 14
return
return
i
range
len
len
<=
self
plit
lag
ioned
if
i
next
range
(

Layer 15
return
return
i
range
len
len
<=
len
list
{
ioned
if
i
unction
range
(

Layer 16
  
#
i
range
len
len
ovo
self
paces
 
ioned
if
i
ura
range
(

Layer 17
return
#
i
range
len
len
<=
self
paces
\n
ioned
if
ii
ura
range
(

Layer 18
  
#
pair
ota
range
len
len
len
numbers
plit
\n
irm
if
i
i
range
(

Layer 19
  
#
i
ota
range
len
len
len
numbers
plit
for
  
if
i
i
range
(

Layer 20
  
#
i
ota
range
len
len
len
numbers
plit
for
ioned
if
i
i
range
(

Layer 21
  
distance
i
ota
range
len
len
<=
numbers
plit
for
ioned
for
i
i
range
(

Layer 22
  
return
i
_
numbers
():
len
(
numbers
plit
for
irm
if
i
in
range
(

Layer 23
  
return
i
_
range
(
len
(
numbers
):
for
irm
if
i
in
range
(

Layer 24
  
return
number
in
numbers
(
len
(
numbers
[:
for
irm
for
j
in
range
(

Layer 25
  
return
i
in
range
(
len
(
numbers
):
for
      
for
j
in
range
(

Layer 26
  
return
i
in
numbers
(
len
):
numbers
)):
\n
      
for
j
in
range
(

Layer 27
  
return
i
in
range
(
len
):
numbers
)):
\n
      
for
j
in
range
(

Layer 28
  
return
i
in
range
(
len
(
numbers
)):
\n
      
for
j
in
range
(

Layer 29
  
return
i
in
range
(
len
(
numbers
)):
\n
      
for
j
in
range
(

Layer 30
  
return
i
in
range
(
len
(
numbers
)):
\n
for
j
in
range
(

Layer 31
  
for
i
in
range
(
len
(
numbers
)):
\n
      
for
j
in
range
(

(b)

Figure 2: (a) A prompt from the HumanEval dataset Chen et al. (2021) and corresponding text generated by Llama1
7B. The color of each generated token corresponds to the earliest layer in the model that predicted it. (b) Token
prediction at each layer in Llama1 7B.

T1
T2
T3
T4

LM Head

T1
T2
T3
T4

LM Head

Train once using Layer Dropout + Early Exit....

T1
T2
T3
T4

LM Head

T1
T2
T3
T4

LM Head

T1
T2
T3
T4

LM Head

... to create different sized models with shared weights 

Skipped

Computed

Skipped w. Probability

Legend

Cached

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Figure 3: We propose using layer dropout and early exit
loss during training to create a model that is equivalent
to an ensemble of models of various depths.

less memory consumption for both training and
inference, and eases deployment and maintenance.
Hence, as shown in Figure 3, we train a deep learn-
ing model that is equivalent to an ensemble of mod-
els of various depths, capable of skipping from
different transfomer layers to the LM head.

2.2
Correcting if we Exit Too Early

Regardless if we use heuristics or predictors (as
Schuster et al. (2022); Geva et al. (2022)) to exit
early, or if we modify the training procedure to
make models predict early (as Elbayad et al. (2020);
Zhang et al. (2019) and this paper as well), it is
likely that exiting early during inference will lead
to a reduction in accuracy. It will be ideal if there
is a way to verify if an early prediction is accurate,
and correct it by executing remaining layers. Some
approaches like Zhang et al. (2019) proposed a con-
fidence heuristic to decide after executing an early

exit if the remaining layers are needed. Here, we
leverage speculative decoding techniques to verify
the early exit prediction and correct it. Specula-
tive decoding benefits from the fact that verifying
the prediction of a group of tokens is faster than
generating each token auto-regressively. Hence,
we present a self-speculative decoding approach
where we use early exit to generate each token
auto-regressively, and use the remaining layers to
verify a group of tokens in parallel, and correct
them.

3
Related Work

Dropout
Dropout was first introduced by Sri-
vastava et al. (2014) and involved stochastically
replacing a portion of output elements of fully-
connected layers with zeros during training. We
refer to this variant of dropout as unstructured
dropout. It presented a regularization effect for
training, with the purpose of reducing over-fitting.
Unstructured dropout was commonly used in con-
volutional neural networks (CNNs) before batch
normalization Ioffe and Szegedy (2015) replaced
it as a means to improve generalization. However,
the introduction of transformers brought it back to
light as Vaswani et al. (2017) used a dropout rate of
0.1. However, dropout faded again when pretrain-
ing dataset sizes increased, e.g., large scale models
like Llama Touvron et al. (2023a) and GPT3 Brown
et al. (2020) do not mention dropout in their papers.

Layer Dropout
Skipping layers stochastically
during training is referred to in literature with differ-
ent terms such as stochastic depth or layer dropout.
It was first explored in ResNets by Huang et al.
(2016) and is used to train ConvNext Liu et al.

12624


(2022).In language models, LayerDrop Fan et al.
(2020) applied dropout to every other transformer
layer, which increased its robustness to pruning
layers at inference time. Zhang and He (2020) in-
creased the pretraining speed of BERT by applying
a dropout rate that progressively increased every
iteration as well as every layer. To the best of our
knowledge, layer dropout for training decoder-only
models, or scaling language models to large model
sizes or large datasets has not been explored. More-
over, our paper is the first to propose using layer
dropout to improve early exit inference.

Early Exit
Exiting early in deep learning has
first been explored in CNNs Panda et al. (2016);
Teerapittayanon et al. (2017). They added branch
modules at different exit points in a deep learning
network and introduced additional loss functions
during training to improve the accuracies of those
early exits.
In language models Elbayad et al. (2020) added
a dedicated LM head for each decoder layer in an
encoder-decoder translation model.CALM Schus-
ter et al. (2022) built upon that and started with
a model pretrained with early exit losses, and fo-
cused on finding optimal criteria to decide which
layer to exit at during inference. Din et al. (2023)
started with pretrained models and finetuned auxil-
iary fully-connected layers to map the embeddings
outputted by earlier layers to later layers. In our
proposed solution, we do not introduce any addi-
tional modules or linear layers for early exit, and
instead used a shared exit for all layers.

Speculative
Decoding
Speculative
decod-
ing Leviathan et al. (2023); Chen et al. (2023)
is a popular acceleration technique for language
models. It is based on the fact that auto-regressive
decoding of decoder models are slow as they
generate one token a time, while measuring the
likelihood of a group of generated tokens in
parallel is faster. It uses a fast, less accurate model,
referred to as the draft model, to generate multiple
tokens auto-regressively, and a large, slower,
more accurate main model, to verify the tokens in
parallel, and correct them when needed. The draft
model could have the same or different architecture
as the main model, or could be a compressed
version of the model. Zhang et al. (2023) recently
proposed a self-speculative decoding approach
where the draft model is the same as the main
model, but with a group of intermediate attention

and feed forward network (FFN) layers skipped.
The advantage of our proposed solution compared
to Zhang et al. (2023) is that verification and
correction stages can reuse the activation and KV
cache from the draft stage as both stages execute
the same early layers in the same order, while
Zhang et al. (2023) can not reuse them as it skips
intermediate layers.
Hooper et al. (2024) used
shared transformer layer groups and a shared LM
head to exit each token at a different layer and
execute different layer groups in a pipeline fashion.

4
Proposed Solution

Our approach has three different stages:

1. Training using Layer Dropout & Early Exit
Loss
2. Inference using Early Exit
3. Verification and Correction using Speculative
Decoding
We explain each stage in the following sub-
sections.

4.1
Training using Layer Dropout & Early
Exit Loss

We denote the input tokens to a transformer model
as X and its output as Y , with an embedding layer
that maps the token indices to token embeddings,
x0, and a transformer model with L transformer
layers, where transformer layer l evolves embed-
dings outputted from its previous layer, xl+1 =
xl + fl(xl), and a final LM head that maps the
embedding outputs of the last layer, xL to logits,
eL = g(xL). We denote the cross entropy loss
function that is usually used to train language mod-
els as JCE(eL, Y ).

4.1.1
Layer Dropout
The first modification we apply to common train-
ing recipes, is to apply layer dropout. Hence the
transformer layer operation at layer l and training
iteration t changes to:

xl+1,t = xl,t + M(pl,t)fl(xl,t)
(1)

where pl,t is the dropout rate of layer l at iteration
t, M(p) is a Bernoulli function that returns 0 with
probability p and returns 1 with probability 1 −p.
We apply the dropout operation on each sample
separately within a batch. We remove the dropped
samples from a batch, apply the transformer oper-
ation fl on the remaining samples, and then con-
catenate the output with the dropped samples. To

12625


ensure higher speedup during training, we seed the
random number generator for each GPU with the
same seed, so that each transformer layer at each
iteration will drop the same number of samples.
The dropout rate can be different at each layer l
and training iteration t, pl,t:

pl,t = S(t)D(l)pmax
(2)

where pmax is a hyperparameter that sets the max-
imum dropout rate in the model during training,
D(l) is a per-layer scaling function, and S(t) is a
per-time step scaling function. We found that the
best per-layer scaling is to increase dropout rate
exponentially across layers from 0.0 in layer 0, to
1.0 in last layer, L −1:

D(l) = e
lln2
L−1 −1
(3)

For scaling across time, S(t), we found that if we
start with a pre-trained model and perform contin-
ual pre-training or finetuning, it is best to not scale
across time and hence set S(t) = 1. However, for
pretraining from scratch, we found that an expo-
nential curriculum, Sexp(t), lead to best accuracies
for T training steps:

Sexp(t) = e
tln2
T −1 −1
(4)

4.1.2
Early Exit Loss
To boost prediction accuracy of lower layers, we
need to ensure that the model’s LM head, g, is ca-
pable of unembedding outputs of different layers.
Hence, during training, we augment layer dropout
with early exit loss at each layer. During train-
ing we supervise the model directly to connect the
early exit layers to the LM head, this enables us
to directly supervise the lower layers for the lan-
guage modeling task. The total loss of the model
at iteration t is:

J(X, Y, t) =

l=L−1
�

l=0
˜e(t, l)JCE(g(xl+1), Y )
(5)

Where ˜e(t, l) is a normalized per-layer loss scale,
whose sum across all layers is equal to 1:

˜e(t, l) =
C(t, l)e(l)
�i=L−1
i=0
C(t, i)e(i)
(6)

C(t, l) is a binary curriculum function that deter-
mines if we enable early exit of layer l at iteration
t. We build upon Elbayad et al. (2020) and set a

scale that increases across layers, such as the scale
at one layer is proportional to the sum of the scales
of all previous layers:

e(l) =

�
escale
�i=l
i=0 i,
if 0 ≤l < L −1
L −1 + escale
�i=L−2
i=0
i,
if l = L −1

This way, we penalize later layers with quadrat-
ically higher weight, as predicting in later layers
is easier. 0 ≤escale ≤1 is a hyperparameter that
controls the scale of early exit loss.
Note that we do not add additional LM heads as
proposed in other early exit papers Elbayad et al.
(2020); Schuster et al. (2022), as we essentially use
the same LM head for all layers.

Early Exit Loss Curriculum
We find that
adding early exit loss of all layers at all iterations
during training slows down training and reduces
accuracy of the last layer. To overcome this, we in-
troduce a curriculum, C(t, l). We have explored 2
different curricula. First, we explored a rotational
early exit curriculum, Crot,R, where we enable early
exit at every R layers, and perform circular rotation
at each iteration. This way, early exit at each layer
is enabled once every R iterations. Hence, at each
training iteration, only ⌈L/R⌉unembedding oper-
ations are applied. Second, we explored a gradual
early exit curriculum, Cgrad, where we gradually
enable early exit loss from layers L −1 to 0, one
layer at a time every T/2L iterations.

4.2
Inference using Early Exit

When generating each token during autoregressive
decoding, we run the first E transformer layers in
a model, and skip to the model’s LM head, i.e.,
the model’s final output becomes g(xE). We ex-
plore with different values of E and provide the
accuracies in the Results section.

4.3
Inference using Self-Speculative Decoding

With layer dropout and early exit loss in training,
we show it is possible to speedup autoregressive
generation by exiting early, but this comes at an
accuracy cost compared to using the full model.
Speculative decoding Leviathan et al. (2023); Chen
et al. (2023) is able to leverage a faster yet less
accurate model to speedup generation without ac-
curacy cost. However, this requires storing and
training 2 models.
We introduce a novel self-speculative decoding
algorithm built on top of early exit, enabling us to

12626


T1
T2
T3
T4

LM Head

Verify: Reuse the draft states,
only recompute the remaining model

T1
T2
T3
T4

LM Head

Draft: Use first N layers of model to compute
 next token

T1
T2
T3
T4

LM Head

Autoregressive Generation: 
Compute each token 

T5

T5

Cost: L layers per token

Cost: E layers per token 
E < L
Cost: Prefill Latency for L-E layers
L for new token generation

Self Speculation Decoding

Autoregressive Decoding

T1
T2
T3
T4

LM Head

Verify: Use main model to verify draft
tokens through prefill

T1
T2
T3
T4

LM Head

Draft: Use draft model to generate candidate
tokens T1....T4

T5

Cost: L' layers per token

Cost: Prefill Latency for L layers
L for new token generation

Speculative Decoding

Skipped
Computed

Cached

Legend

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Layer 4

Layer 3

Layer 2

Layer 1

Layer 2

Layer 1

Figure 4: Comparison between autoregressive decoding,
speculative decoding, and our proposed self-speculative
decoding.

reduce memory through the use of a single model
and latency of traditional speculative decoding
through re-using hidden states in draft and verify
steps. As shown in Figure 4, our self-speculation
algorithm consists of 2 key steps (1) Self-Drafting,
using the early exit to draft tokens from the same
model (2) Self-Verification, using the remaining
layers to validate the prediction. To enable re-use
in (1) and (2), we develop a novel Cache Reuse
technique that unifies the KV cache and storing
the exit query. We provide a high level description
of the algorithm in sections §4.3.1 and 4.3.2 and
provide pseudo code in A.6.

4.3.1
Self-Drafting

The first step in speculative decoding is to define a
set of draft tokens D0...d−1. In our algorithm, we
compute the first d draft tokens through early exit.
We refer to d as the number of speculations. We
leverage a subset of the LLM and conduct auto-
regressive inference exiting at layer E.
Our training recipe enabled us to train the model
once to get an ensemble of different candidate draft
models at each layer depth. We can evaluate exiting
at different layers and observe a trade off between
latency and accuracy.

4.3.2
Self-Verification

The next step in speculative decoding is verification.
Verification leverages the full LLM to predict the
next token for each draft token in a single forward

pass. We then assess to see where the draft tokens
and verified tokens agree. All the draft tokens up
till the disagreement point are added to the output
along with the next verified token and the process
continues from the draft stage.
In our self-speculative decoding algorithm, the
self-verification stage critically only requires com-
puting the remaining layers of the model that were
not used in the draft stage. For a model with L
layers, the number of verification layers is L −E.
In order to re-use the first E layers from the draft
stage we employ some modifications to the KV
cache as we show in the subsequent subsection.

4.3.3
Reusing the Cache
In autoregressive transformers, KV cache is a criti-
cal component of efficient generation, allowing us
to avoid recomputing prior KV pairs in each layer.
As our draft stage uses the first E layers of the
model and the verification stage uses the remaining
L −E layers, we are able to re-use a significant
amount of compute between the 2 stages:

• Single KV Cache As draft and verification
stages operate on the same model using the
same order of layers, the first E layers are
shared in both steps. Hence, in the draft stage,
the KV cache in the first E layers are already
computed, so we are able to effectively main-
tain a single KV cache for the draft and verify
steps, reducing memory and latency.
• Exit Query Cache: To further reduce com-
putation of the first E layers, we introduce an
exit query cache that saves the query vector
of exit layer E −1 for verification to directly
continue from layer E to last layer L. Criti-
cally note that we need to save only the query
for the exit layer. We term the union of the
KV cache and the exit query as KVQ cache.

5
Experiments

We would like to evaluate our training recipe on dif-
ferent types of training, whether pretraining from
scratch or finetuning. To verify our approach, we
run different types of training experiments:

• Continual Pretraining:
start with a pre-
trained model and continue pretraining on
52B tokens from a corpus of diverse data con-
taining natural language text and code. We ex-
periment using pretrained Llama2 7B (32 lay-
ers), with pmax = 0.1, escale = 0.2, Crot,R=8,
and Llama2 13B (40 layers), with pmax = 0.1,
escale = 0.1, Crot,R=39.

12627


• Pretraining from Scratch: start with ran-
domly initialized model and pretrain on 26B
tokens from a corpus of diverse data contain-
ing natural language text and code. We ex-
periment with Llama2 1.5B (a custom small
Llama-like model with 24 layers) (see A.3.1
for architecture details) with pmax = 0.1,
escale = 0.2, Crot,R=23 and Llama2 7B (32
layers) with pmax
=
0.2, escale
=
0.2,
Crot,R=31. Following Srivastava et al. (2014)
we use higher learning rates when layer
dropout is greater than 0.0.
• Finetuning on Code Data: see §A.2 for de-
tails and §A.4 for results.
• Finetuning on Task-Specific Dataset: see
§A.2 for details and §A.4 for results.
We try different variants of LayerSkip:
layer
dropout only (LD), early exit loss only (EE), and
both layer dropout and early exit loss (LD+EE).
We provide more details about training hyperpa-
rameters in Appendix A.3.

6
Results

6.1
Early Exit Inference Results

After training each model configuration, we evalu-
ate accuracy of exiting early at different layers.

Continual Pretraining
In Figure 5, we present
our results for Llama2 7B and 13B on a diverse set
of evaluation tasks (see § A.3.2 for task details) and
compare with the baseline model from Touvron
et al. (2023b). In Table A4 we zoom in and show
the specific values of accuracies for the last layer
and middle layer of each model. In Figure A1 we
show sample text generations for exiting at earlier
layers for both models with and without continual
pretraining with LayerSkip. Overall, for earlier
layers, LayerSkip is clearly better than the baseline.
For last layer accuracy, LayerSkip has minimal
drop in accuracy compared to baseline.

Pretraining from Scratch
In Figure A2, we
present our results for Llama2 1.5B and 7B pre-
trained from scratch on 26B tokens using Lay-
erSkip on a diverse set of evaluation tasks (see
§ A.3.2 for task details) and compare with the same
models pretrained on the same number of tokens
from scratch without LayerSkip. In Figure A3 we
show sample text generations for exiting at earlier
layers. Results show that introducing our proposed
training recipe leads to higher accuracy than the
baseline on earlier layers. On the last layer, we

do see a slight drop in accuracy in some down-
stream tasks, while in other tasks we see LayerSkip
leading to higher accuracy.

6.2
Self-Speculative Decoding Results

We evaluate the self-speculative decoding algo-
rithm introduced in §4.3 on different trained mod-
els. We report quality metrics, EM (exact match)
and ROUGE-2 Ganesan (2018), token acceptance
rate for the self speculation algorithm (how of-
ten verification accepts each of the draft tokens),
throughput measured as tokens per second aver-
aged over the sampled dataset, and speed up com-
pared to autoregressive decoding. For our early
exit and our self-speculative decoding experiments,
we denote layer we exit at as E. We compare
with Draft & Verify Zhang et al. (2023) on com-
mon models and tasks evaluated in both papers.
All experiments were performed with greedy de-
coding and generated a maximum of 512 tokens
for each sample. Following Zhang et al. (2023),
speedup is calculated as acceleration of average
inference time per token compared to “Autoregres-
sive” baseline. “Autoregressive” experiments use
baseline models that were pretrained or finetuned
without LayerSkip, while “Early Exit” and “Self
Speculative” experiments use our models trained
or finetuned with LayerSkip. Our implementation
leverages HuggingFace Wolf et al. (2020).

Continual Pretraining
In Table 1, we evaluate
the continual pre-training of Llama2 7B and 13B
with and without LayerSkip on various tasks: CN-
N/DM Nallapati et al. (2016), XSUM Narayan et al.
(2018) abstractive summarization tasks, and Hu-
manEval Chen et al. (2021) coding task. The exper-
iments were performed on NVIDIA H100 GPUs.
The number of speculations, i.e., the number of to-
kens generated in the draft stage, is denoted d. We
obtain speedups between 1.34× and 2.16× depend-
ing on model or task. In general, we observe higher
speedups for the smaller 7B compared to the larger
13B model. Comparing with Draft & Verify, we are
significantly faster on CNN/DM (1.81× vs. 1.5×)
and slightly slower on XSUM (1.34× vs. 1.48×).

Pretraining from Scratch
Experiments were
performed on H100 GPUs and results presented
in Table 2. We found an opposite trend to continual
pretraining: bigger model has a bigger speedup,
reaching 2.16× speedup.

12628


102
104

PPL

Wikipedia Test

102

PPL

Books Test

102
104

PPL

The Stack Test

40

60

80

Acc

BoolQ

60
70

Acc

PIQA

35

40

45

Acc

SIQA

30
40
50

Acc

HellaSwag

50
60
70

Acc

Winogrande 1.1

40
60

Acc

ARC Easy

20
30
40

Acc

ARC Challenge

20

30

Acc

OBQA

60

80

Acc

COPA

20

40

60

Acc

RACE Middle

30

40

Acc

RACE High

30
40

Acc

MMLU

0
10
20

EM

NQ

0
25
50

EM

TQA

0

10

EM

GSM8K

0

2

EM

MATH

32
24
16
8
4

Layer

0

10

Pass@1

HumanEval

32
24
16
8
4

Layer

0
10
20

Pass@1

MBPP

Baseline
LayerSkip-LD+EE

(a) Llama2 7B

102
104

PPL

Wikipedia Test

102

PPL

Books Test

102
104

PPL

The Stack Test

70
80

Acc

BoolQ

60
70
80

Acc

PIQA

35
40
45

Acc

SIQA

40

60

Acc

HellaSwag

50
60
70

Acc

Winogrande 1.1

40
60
80

Acc

ARC Easy

20

40

Acc

ARC Challenge

20
30

Acc

OBQA

70
80
90

Acc

COPA

40

60

Acc

RACE Middle

30
40

Acc

RACE High

30
40
50

Acc

MMLU

0

20

EM

NQ

0
25
50

EM

TQA

0

20

EM

GSM8K

0
2
4

EM

MATH

40
30
20
10
5

Layer

0

10

20

Pass@1

HumanEval

40
30
20
10
5

Layer

0

20

Pass@1

MBPP

Baseline
LayerSkip-LD+EE

(b) Llama2 13B

Figure 5: Early exit evaluation of continual pretraining

7
Ablation Studies

Many ablation studies are in the Appendix, but we
summarize some here.

Scaling with Pretraining Tokens
Figure A5,
shows that without LayerSkip pretraining increases
perplexity of earlier layers by orders of magnitude.

KV Cache in Self-Speculation
Table A7 shows
that our proposed re-use of KV cache consistently
saves us 9-20 ms per token depending on the task.

Selecting Parameters for Self Speculation
Self
speculation relies on 2 core parameters (1) early
exit layer and (2) number of speculations. There
exists a tradeoff where selecting too low of an exit
point and too many tokens are rejected, too high

and the latency cost of the exit layer reduces the
benefits of speculation. We find that these parame-
ters are task dependent. Figure 6 shows how range
of decoding parameters varies for different tasks.

8
Conclusion

We show that combining layer dropout & early exit
loss with curriculum, improves accuracy of early
exit during inference, and developed a novel self-
speculative decoding solution that led upto 1.86×
speedup. We hope this encourages researchers to
adopt the proposed recipe in pretraining and fine-
tuning. In the future, we can increase accuracy
of earlier layers to obtain better speedups for self-
speculative decoding, e.g., by combining with dy-
namic conditions (like Schuster et al. (2022)).

12629


Llama2 7B
Llama2 13B

Generation
E
d
ROUGE-2

Token
Acc.

Tokens
per Sec.
Speedup
E
d
ROUGE-2

Token
Acc.

Tokens
per Sec.
Speedup

CNN-DM
One-Shot Abstractive Summarization

Autoregressive
-
-
0.079
-
62.7
1.00×
-
-
0.098
-
37.2
1.00×
Early Exit
8
-
0.012
-
232.4
-
15
-
0.016
-
105.5
-
Self Speculative
8
12
0.078
68.9%
127.9
1.86×
15
12
0.098
74.5%
70.2
1.81×

Draft and Verify
n/a
n/a
n/a
n/a
n/a
n/a
-
-
0.107
n/a
n/a
1.56×

XSUM
Abstractive Summarization

Autoregressive
-
-
0.073
-
63.4
1.00×
-
-
0.124
-
43.8
1.00×
Early Exit
8
-
0.002
-
228.0
-
15
-
0.009
-
110.6
-
Self Speculative
8
12
0.073
54.6%
104.7
1.54×
15
4
0.124
67.7%
60.5
1.34×

Draft and Verify
n/a
n/a
n/a
n/a
n/a
n/a
-
-
0.126
n/a
n/a
1.48×

HumanEval
Coding

Autoregressive
-
-
0.041
-
62.9
1.00×
-
-
0.055
-
48.9
1.00×
Early Exit
8
-
0.003
-
225.4
-
15
-
0.0005
-
244.3
-
Self Speculative
8
6
0.042
67.1%
122.8
1.83×
7
4
0.055
57.0%
84.2
1.66×

Table 1: Generation results for Llama2 continually pretrained with and without LayerSkip.

Llama2 1.5B - 26B Tokens
Llama2 7B - 26B Tokens

Generation
E
ROUGE-2

Token
Acc.

Tokens
per Sec.
Speedup
E
ROUGE-2

Token
Acc.

Tokens
per Sec.
Speedup

CNN-DM
One-Shot Abstractive Summarization

Autoregressive
-
0.063
-
91.6
1.00×
-
0.060
-
64.5
1.00×
Self Speculative
8
0.063
77.4%
167.4
1.76×
8
0.067
77.8%
145.6
2.16×

Table 2: Generation results for Llama2 pretrained from scratch on 26B tokens with and without LayerSkip.

4
6
8
10
12
14
16
18
20
Exit Layer

2

4

6

8

10

12

14

16

18

20

Number of Speculations

Tokens Per Second

48

60

72

84

96

108

120

132

(a) Llama2 7B CNN-DM Self-Speculation

2
4
6
8
10
12
14
16
Exit Layer

2

4

6

8

10

12

14

16

18

20

Number of Speculations

Tokens Per Second

40

52

64

76

88

100

112

124

(b) Llama2 7B HumanEval Self-Speculation

Figure 6: Self Speculation Decoding Parameters Sweep.

12630


9
Limitations

• Our self-speculative decoding solution re-
quires finetuning a model or pretraining it with
our recipe, while the self-speculative decoding
approach propoposed in Zhang et al. (2023)
does not require changing a model’s weights.
• The introduced hyperparameters, pmax for
layer dropout, escale and R for early exit, re-
quires tuning in order to avoid a drop in last
layer accuracy.
• When pretraining with layer dropout from
scratch, increasing the learning rate is required
to maintain accuracy, and tuning learning rate
to get optimal accuracy could be tricky and
time consuming.

Acknowledgements

We would like to thank:

• Volker Seeker, Artem Korenev, and Ilia Ku-
likov for logistic support,
• Fabian Gloeckle, Andrey Gromov, Francisco
Massa, Daniel Haziza, Aaditya Singh, Karen
Hambardzumyan, Nicola Cancedda, for dis-
cussions,
• FAIR’s clusters’ support team members, es-
pecially, Henry Estela, Hongsheng Song,
Shubho Sengupta, and Nabib Ahmed, for their
help in maintaing our clusters,
• Kamila Benzina, Carolyn Krol, Helen Klein,
and Philippe Brunet for legal support.

12631


References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten
Bosma, Henryk Michalewski, David Dohan, Ellen
Jiang, Carrie Cai, Michael Terry, Quoc Le, and
Charles Sutton. 2021. Program synthesis with large
language models.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng
Gao, and Yejin Choi. 2020. Piqa: Reasoning about
physical commonsense in natural language. In Thirty-
Fourth AAAI Conference on Artificial Intelligence.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens
Winter, Chris Hesse, Mark Chen, Eric Sigler, Ma-
teusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. 2020.
Language models are few-shot learners.
In Ad-
vances in Neural Information Processing Systems,
volume 33, pages 1877–1901. Curran Associates,
Inc.

Nicola Cancedda. 2024. Spectral filters, dark signals,
and attention sinks.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving,
Jean-Baptiste Lespiau, L. Sifre, and John M. Jumper.
2023. Accelerating large language model decoding
with speculative sampling. ArXiv, abs/2302.01318.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming
Yuan, Henrique Ponde de Oliveira Pinto, Jared Ka-
plan, Harri Edwards, Yuri Burda, Nicholas Joseph,
Greg Brockman, Alex Ray, Raul Puri, Gretchen
Krueger, Michael Petrov, Heidy Khlaaf, Girish Sas-
try, Pamela Mishkin, Brooke Chan, Scott Gray,
Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz
Kaiser, Mohammad Bavarian, Clemens Winter,
Philippe Tillet, Felipe Petroski Such, Dave Cum-
mings, Matthias Plappert, Fotios Chantzis, Eliza-
beth Barnes, Ariel Herbert-Voss, William Hebgen
Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie
Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain,
William Saunders, Christopher Hesse, Andrew N.
Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan
Morikawa, Alec Radford, Matthew Knight, Miles
Brundage, Mira Murati, Katie Mayer, Peter Welinder,
Bob McGrew, Dario Amodei, Sam McCandlish, Ilya
Sutskever, and Wojciech Zaremba. 2021. Evaluating
large language models trained on code.

Xilun Chen, Asish Ghoshal, Yashar Mehdad, Luke
Zettlemoyer, and Sonal Gupta. 2020. Low-resource
domain adaptation for compositional task-oriented
semantic parsing. In Proceedings of the 2020 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 5090–5100, Online. As-
sociation for Computational Linguistics.

Christopher Clark, Kenton Lee, Ming-Wei Chang,
Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. 2019. Boolq: Exploring the surprising
difficulty of natural yes/no questions. In NAACL.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,
Ashish Sabharwal, Carissa Schoenick, and Oyvind
Tafjord. 2018. Think you have solved question an-
swering? try arc, the ai2 reasoning challenge. ArXiv,
abs/1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems.

Alexander Yom Din, Taelin Karidi, Leshem Choshen,
and Mor Geva. 2023. Jump to conclusions: Short-
cutting transformers with linear transformations.

Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael
Auli. 2020. Depth-adaptive transformer. In In Proc.
of ICLR.

Angela Fan, Edouard Grave, and Armand Joulin. 2020.

Reducing transformer depth on demand with struc-
tured dropout. In International Conference on Learn-
ing Representations.

Kavita Ganesan. 2018. Rouge 2.0: Updated and im-
proved measures for evaluation of summarization
tasks.

Leo Gao, Stella Biderman, Sid Black, Laurence Gold-
ing, Travis Hoppe, Charles Foster, Jason Phang,
Horace He, Anish Thite, Noa Nabeshima, Shawn
Presser, and Connor Leahy. 2020.
The Pile: An
800gb dataset of diverse text for language modeling.
arXiv preprint arXiv:2101.00027.

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Gold-
berg. 2022. Transformer feed-forward layers build
predictions by promoting concepts in the vocabulary
space. In Proceedings of the 2022 Conference on
Empirical Methods in Natural Language Process-
ing, pages 30–45, Abu Dhabi, United Arab Emirates.
Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2021a. Measuring massive multitask language under-
standing. In International Conference on Learning
Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul
Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. 2021b. Measuring mathematical
problem solving with the math dataset. NeurIPS.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh,
Hasan Genc, Kurt Keutzer, Amir Gholami, and
Sophia Shao. 2024. Speed: Speculative pipelined
execution for efficient decoding.

12632


Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kil-
ian Weinberger. 2016. Deep networks with stochastic
depth.

Sergey Ioffe and Christian Szegedy. 2015. Batch nor-
malization: Accelerating deep network training by
reducing internal covariate shift.

Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk,
Jonghyun Choi, Ali Farhadi, and Hannaneh Ha-
jishirzi. 2017. Are you smarter than a sixth grader?
textbook question answering for multimodal machine
comprehension. 2017 IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages
5376–5384.

Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia
Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine
Jernite, Margaret Mitchell, Sean Hughes, Thomas
Wolf, Dzmitry Bahdanau, Leandro von Werra, and
Harm de Vries. 2022. The stack: 3 tb of permissively
licensed source code. Preprint.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Matthew Kelcey,
Jacob Devlin, Kenton Lee, Kristina N. Toutanova,
Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: a benchmark for question answering
research. Transactions of the Association of Compu-
tational Linguistics.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang,
and Eduard Hovy. 2017. RACE: Large-scale ReAd-
ing comprehension dataset from examinations. In
Proceedings of the 2017 Conference on Empirical
Methods in Natural Language Processing, pages 785–
794, Copenhagen, Denmark. Association for Compu-
tational Linguistics.

Yaniv Leviathan, Matan Kalman, and Yossi Matias.
2023. Fast inference from transformers via spec-
ulative decoding. In Proceedings of the 40th Interna-
tional Conference on Machine Learning, ICML’23.
JMLR.org.

Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen
Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong,
Ernie Chang, Yangyang Shi, Raghuraman Krish-
namoorthi, Liangzhen Lai, and Vikas Chandra. 2024.
Mobilellm: Optimizing sub-billion parameter lan-
guage models for on-device use cases.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Fe-
ichtenhofer, Trevor Darrell, and Saining Xie. 2022. A
convnet for the 2020s. Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recog-
nition (CVPR).

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish
Sabharwal. 2018. Can a suit of armor conduct elec-
tricity? a new dataset for open book question answer-
ing. In Conference on Empirical Methods in Natural
Language Processing.

Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos
santos, Caglar Gulcehre, and Bing Xiang. 2016.
Abstractive text summarization using sequence-to-
sequence rnns and beyond.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata.
2018. Don’t give me the details, just the summary!
Topic-aware convolutional neural networks for ex-
treme summarization. In Proceedings of the 2018
Conference on Empirical Methods in Natural Lan-
guage Processing, Brussels, Belgium.

Priyadarshini Panda, Abhronil Sengupta, and Kaushik
Roy. 2016. Conditional deep learning for energy-
efficient and enhanced pattern recognition.

Mary Phuong and Marcus Hutter. 2022. Formal algo-
rithms for transformers. ArXiv, abs/2207.09238.

Alec Radford, Jeff Wu, Rewon Child, David Luan,
Dario Amodei, and Ilya Sutskever. 2019. Language
models are unsupervised multitask learners.

Melissa Roemmele, Cosmin Adrian Bejan, and An-
drew S. Gordon. 2011. Choice of plausible alter-
natives: An evaluation of commonsense causal rea-
soning. In Logical Formalizations of Commonsense
Reasoning, Papers from the 2011 AAAI Spring Sym-
posium, Technical Report SS-11-06, Stanford, Cali-
fornia, USA, March 21-23, 2011. AAAI.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle,
Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi
Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom
Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish
Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wen-
han Xiong, Alexandre Défossez, Jade Copet, Faisal
Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier,
Thomas Scialom, and Gabriel Synnaeve. 2023. Code
llama: Open foundation models for code.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat-
ula, and Yejin Choi. 2019. Winogrande: An adver-
sarial winograd schema challenge at scale.

Siddharth Samsi, Dan Zhao, Joseph McDonald, Baolin
Li, Adam Michaleas, Michael Jones, William Berg-
eron, Jeremy Kepner, Devesh Tiwari, and Vijay Gade-
pally. 2023. From Words to Watts: Benchmarking
the Energy Costs of Large Language Model Infer-
ence. arXiv e-prints, page arXiv:2310.03003.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan
Le Bras, and Yejin Choi. 2019. Social IQa: Com-
monsense reasoning about social interactions. In
Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP), pages 4463–
4473, Hong Kong, China. Association for Computa-
tional Linguistics.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani,
Dara Bahri, Vinh Q. Tran, Yi Tay, and Donald Met-
zler. 2022. Confident adaptive language modeling.
In Advances in Neural Information Processing Sys-
tems.

12633


Kyuhong Shim, Iksoo Choi, Wonyong Sung, and Jung-
wook Choi. 2021. Layer-wise pruning of transformer
attention heads for efficient language modeling. In
2021 18th International SoC Design Conference
(ISOCC), pages 357–358.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
Ilya Sutskever, and Ruslan Salakhutdinov. 2014.
Dropout: A simple way to prevent neural networks
from overfitting. Journal of Machine Learning Re-
search, 15(56):1929–1958.

PyTorch Team. 2024. gpt-fast. https://github.com/
mostafaelhoushi/gpt-fast.

Surat Teerapittayanon, Bradley McDanel, and H. T.
Kung. 2017. Branchynet: Fast inference via early
exiting from deep neural networks.

Hugo Touvron, Thibaut Lavril, Gautier Iz

..._This content has been truncated to stay below 50000 characters_...
