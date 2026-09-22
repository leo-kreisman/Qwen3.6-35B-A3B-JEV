"""Attribute decode bytes-read to the physical ubatch sweep count.

The 42 GB decode on an 18.33 GB expert set is not fault-around amplification: with
8 branches x 908 tokens = 7264 tokens in one pass and n_ubatch = 8 x 512 = 4096,
llama.cpp runs TWO physical ubatches per pass, and every ubatch walks all 40 layers
and re-reads their experts. 2 x 18.33 GB ~= 36.7 GB, plus ~2.5 GB of non-expert
tensors, is the measured 42 GB.

So the lever is the ubatch size, and it is one environment variable.

Run as:  SEMIF_LLAMA_UBATCH=8192 python probe_ubatch.py
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/scribe/Projects/JEV_experiment/semif/src")
from semif_phase1 import llamacpp_backend as L

GGUF = "/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
TOK = "/home/scribe/models/Qwen3.6-35B-A3B-tokenizer"
IN = os.environ.get("SEMIF_INPUT", "/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl")


def rd():
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return -1


rows = [json.loads(line) for line in open(IN)]
THREADS = int(os.environ.get("SEMIF_THREADS", "6"))
base = rd()
model, tokenizer, metadata = L.load_model(TOK, "local", GGUF, threads=THREADS, context_tokens=8192)
after_load = rd()

# Token count of one branch group, which is what the ubatch has to cover.
encoded = [model.encode_verified(row, 8192) for row in rows]
width = max(1, model.engine.sequences)
group = encoded[:width]
branch_tokens = sum(len(ids) for ids, _, _ in group)

print(f"LOAD read_bytes = {(after_load-base)/1e9:.2f} GB")
print(f"threads = {THREADS}  n_ubatch = {model.engine.ubatch}  "
      f"branches/group = {len(group)}  tokens/group = {branch_tokens}")
print(f"expected sweeps per pass = {-(-branch_tokens // model.engine.ubatch)}")

results, timing = L.score_shared(model, tokenizer, rows, metadata, max_tokens=8192)
print(f"DECODE read_bytes = {(rd()-after_load)/1e9:.2f} GB")
print(f"scoring total_seconds = {timing['total_seconds']:.2f}")
print(f"batched_forward_seconds = {timing['batched_forward_seconds']:.2f}")
print(f"batched_passes = {timing['batched_passes']}  batched_tokens = {timing['batched_tokens']}")
print(f"effective throughput = {(rd()-after_load)/timing['total_seconds']/1e6:.0f} MB/s")
print("answers:", [(r["id"], round(max(r["probabilities"]), 6)) for r in results])
