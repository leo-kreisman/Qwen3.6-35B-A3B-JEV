"""Resident scorer: load once, then serve repeated scoring calls.

The comparison this exists for: the shipped CLI is one-shot (load -> score -> exit), so
an agent pays the whole 20.9 GB model load on every call. Here the load is paid once and
each subsequent call pays only its decode. Both are cold-cache per call -- 41 GB per call
against an 8 GiB cap means nothing survives between calls -- so the only thing residency
removes is the load.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/scribe/Projects/JEV_experiment/semif/src")
from semif_phase1 import llamacpp_backend as L

GGUF = "/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
TOK = "/home/scribe/models/Qwen3.6-35B-A3B-tokenizer"
IN = "/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl"
CALLS = int(os.environ.get("SEMIF_CALLS", "3"))


def rd():
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return -1


rows = [json.loads(line) for line in open(IN)]
process_start = time.perf_counter()
base = rd()
model, tokenizer, metadata = L.load_model(TOK, "local", GGUF, context_tokens=8192)
after_load = rd()
load_seconds = time.perf_counter() - process_start
print(f"LOAD {load_seconds:.2f}s {(after_load-base)/1e9:.2f} GB "
      f"threads={metadata['threads']}")

for call in range(1, CALLS + 1):
    mark = time.perf_counter()
    before = rd()
    results, timing = L.score_shared(model, tokenizer, rows, metadata, max_tokens=8192)
    after = rd()
    print(f"CALL {call} {time.perf_counter()-mark:.2f}s {(after-before)/1e9:.2f} GB "
          f"passes={timing['batched_passes']} "
          f"probs={[round(max(r['probabilities']), 6) for r in results]}")

print(f"TOTAL {time.perf_counter()-process_start:.2f}s {(rd()-base)/1e9:.2f} GB")
