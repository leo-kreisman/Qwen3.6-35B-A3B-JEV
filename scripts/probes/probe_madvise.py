"""Test whether MADV_RANDOM on llama.cpp's GGUF mapping kills read amplification.

Baseline (measured): one 908-token ubatch reads 42.0 GB against a ~19.5 GB logical
expert footprint. Hypothesis: mmap fault-around read-ahead is pulling in pages the
MoE never uses, and MADV_RANDOM disables it for that VMA.

We do not own llama.cpp's descriptor, but the mapping is visible in
/proc/self/maps, so MADV_RANDOM can still be applied to the exact VMA.
"""
import ctypes
import json
import os
import sys
import time

sys.path.insert(0, "/home/scribe/Projects/JEV_experiment/semif/src")
from semif_phase1 import llamacpp_backend as L

MADV_RANDOM = 1
MADV_WILLNEED = 3
GGUF = "/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
TOK = "/home/scribe/models/Qwen3.6-35B-A3B-tokenizer"
IN = "/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl"


def rd():
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return -1


def gguf_mappings(path):
    """Return (start, end) for every file-backed VMA of `path`, with its advice."""
    target = os.path.realpath(path)
    found = []
    with open("/proc/self/maps") as fh:
        for line in fh:
            if target not in line:
                continue
            span, perms, _offset, _dev, _inode, name = line.rstrip("\n").split(None, 5)
            if "r" not in perms:  # only readable mappings can be advised
                continue
            start, end = (int(part, 16) for part in span.split("-"))
            found.append((start, end, line.strip()))
    return found


libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int

rows = [json.loads(line) for line in open(IN)]
base = rd()
model, tokenizer, metadata = L.load_model(TOK, "local", GGUF, threads=6, context_tokens=8192)
print(f"load read_bytes = {rd()-base:,} ({(rd()-base)/1e9:.1f} GB)")

spans = gguf_mappings(GGUF)
print(f"GGUF mappings: {len(spans)} VMA(s)")
total = 0
for start, end, line in spans:
    total += end - start
    rc = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), MADV_RANDOM)
    err = ctypes.get_errno()
    print(f"  {line[:100]}")
    print(f"    madvise(MADV_RANDOM) on {end-start:,} B -> rc={rc} errno={err}")
print(f"total mapped = {total/1e9:.1f} GB")

after_advice = rd()
t0 = time.perf_counter()
results, timing = L.score_shared(model, tokenizer, rows, metadata, max_tokens=8192)
elapsed = time.perf_counter() - t0
print(f"DECODE: {elapsed:.2f}s  read_bytes = {rd()-after_advice:,} ({(rd()-after_advice)/1e9:.1f} GB)")
print(f"TOTAL read_bytes = {rd()-base:,} ({(rd()-base)/1e9:.1f} GB)")
print("timing:", json.dumps({k: timing[k] for k in ("total_seconds", "suffix_forward_seconds", "batched_tokens")}))
print("answers:", [(r["id"], round(max(r["probabilities"]), 6)) for r in results])
