# SOURCE: https://github.com/sypherin/growing-memory

# growing-memory

**Long-term memory for LLM apps — an application-layer implementation of Memory Caching ([Behrouz et al., 2026](https://arxiv.org/abs/2602.24281)).**

> Give any chatbot or agent a memory that grows with the conversation — without the quadratic cost of a giant context window, and without training a model.

`growing-memory` brings the core idea of Google's **Memory Caching** paper ("RNNs with Growing Memory", arXiv:2602.24281) to the application layer: chunk your history into segments, cache a **compressed checkpoint** of each, and at query time retrieve **only the few relevant checkpoints**. Your model effectively remembers everything; your prompt stays tiny.

```
from growing_memory import MemoryCache, sentence_transformer_embed, openai_chat

mc = MemoryCache(
    llm=openai_chat("http://localhost:8001/v1", "your-model"),  # compresses segments
    embed=sentence_transformer_embed("all-mpnet-base-v2"),       # for retrieval
)

mc.add(months_of_conversation_or_a_long_document)   # chunks → compresses → stores
context = mc.recall("what did we decide about pricing?", k=5)
# → drop `context` into your next prompt. That's it.
```

Three lines: `add()`, `recall()`, done. Pure-stdlib core (sqlite + cosine), pluggable LLM + embedder (OpenAI / local llama.cpp / vLLM / sentence-transformers).

## Why

| | context window | what it costs |
|---|---|---|
| **Stuff everything** (full context) | grows with history | O(L²) compute, huge KV-cache / token bill |
| **Sliding window** (last K) | fixed | cheap, but **forgets** anything older |
| **growing-memory** | grows, but you only *pay* for the relevant few | cheap **and** remembers |
This is the trade-off the paper formalizes for the model architecture (interpolating between O(L) RNNs and O(L²) Transformers). `growing-memory` applies the same idea where you can use it today — over the prompt, with the models you already have.

## Does it actually save anything? (measured)

The paper proves the saving in *theory* but — as the ICLR reviewer flagged — never measured real tokens/latency. This repo does.

**SQuAD v2** — 80 real Wikipedia paragraphs across 24 topics, 40 *human-written* ground-truth questions, answered by a real local LLM (gemma-4-26B):

| strategy | accuracy | avg tokens |
|---|---|---|
| full-context (stuff all 80 paragraphs) | 90% | 14,975 |
| fixed-window (last 5) | 5% | 875 *(forgets)* |
| **growing-memory** (retrieve relevant 5) | **95%** | **1,027** |
**14.6× fewer tokens — and *higher* accuracy than full-context** (retrieving the relevant few avoids the "lost in the middle" distraction of a stuffed prompt).

Synthetic needle-in-haystack recall, for the isolated case:

| strategy | recall | avg tokens |
|---|---|---|
| full-context | 100% | 2,704 |
| fixed-window (K=4) | 8% *(forgets)* | 416 |
| **growing-memory** | **100%** | **384** *(7× fewer)* |
Reproduce both yourself:

```
pip install -e ".[demo]"
python benchmark/needle_recall.py     # synthetic needle recall
python benchmark/squad_recall.py      # SQuAD v2, real public data
```

## How it maps to the paper

Memory Caching (MC) caches the compressed memory **checkpoints** of an RNN per segment and aggregates them. `growing-memory` mirrors the paper's **Sparse Selective Caching** variant at the application layer: a retriever (the paper's MoE-style router) selects only the relevant cached checkpoints. See [`TECHNICAL_NOTES.md`](/sypherin/growing-memory/blob/master/TECHNICAL_NOTES.md) for the full breakdown of the mechanism, the four variants, the complexity analysis, and what's genuinely new vs prior art (Compressive Transformer, Transformer-XL, Titans).

## Install

```
pip install growing-memory                 # core (stdlib only)
pip install "growing-memory[demo]"         # + sentence-transformers for local embeddings
```

## Use it as a chat memory

```
mc = MemoryCache(llm=my_llm, embed=my_embed, namespace=f"user:{user_id}")
mc.add(new_messages)                       # after each exchange / session
prior = mc.context("the user's question")  # retrieve, prepend to the prompt
```

## License

MIT — fork it, ship it.

---

## Built by Altronis

We're **[Altronis](https://altronis.sg)** — we build **private, on-premise AI systems** for businesses that can't send their data to the cloud: local LLMs, document automation, and memory/retrieval like this, running on hardware you own.

`growing-memory` is the open-source version of a problem we solve in production: making AI remember long histories and large document sets *cheaply and privately*. If your business needs:

- an assistant that remembers customers/cases across months — without a cloud vendor holding the data,
- document processing (invoices, delivery orders, contracts) with on-prem AI,
- or a local AI stack where your data never leaves the building,
…that's literally what we do. **→ [Talk to us at altronis.sg](https://altronis.sg)** or open an issue.

⭐ If this saved you tokens, a star helps others find it.

---

Keywords: Memory Caching, growing memory, RNNs with growing memory, Behrouz, arXiv:2602.24281, long-term memory for LLMs, LLM long context, subquadratic, Titans, retrieval-augmented memory, checkpoint memory, KV-cache reduction, context compression, chatbot memory, on-prem AI, private LLM, local AI, Singapore AI consulting.
