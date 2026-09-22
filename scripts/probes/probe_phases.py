"""Attribute SSD reads to load vs decode, and sample the decode's I/O over time."""
import json, os, sys, time, threading

sys.path.insert(0, "/home/scribe/Projects/JEV_experiment/semif/src")
from semif_phase1 import llamacpp_backend as L


def rd():
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return -1


GGUF = "/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
TOK = "/home/scribe/models/Qwen3.6-35B-A3B-tokenizer"
IN = "/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl"

rows = [json.loads(line) for line in open(IN)]
base = rd()
print(f"read_bytes at start = {base}")

mark = time.perf_counter()
model, tokenizer, metadata = L.load_model(TOK, "local", GGUF, threads=6, context_tokens=8192)
print(f"LOAD: {time.perf_counter()-mark:.2f}s  read_bytes delta = {rd()-base:,} ({(rd()-base)/1e9:.1f} GB)")
after_load = rd()

samples = []
stop = threading.Event()


def watch():
    while not stop.is_set():
        samples.append((round(time.perf_counter() - t0, 1), (rd() - after_load) / 1e9))
        time.sleep(2)


t0 = time.perf_counter()
watcher = threading.Thread(target=watch, daemon=True)
watcher.start()
results, timing = L.score_shared(model, tokenizer, rows, metadata, max_tokens=8192)
stop.set()
watcher.join(timeout=3)
print(f"DECODE: {time.perf_counter()-t0:.2f}s  read_bytes delta = {rd()-after_load:,} ({(rd()-after_load)/1e9:.1f} GB)")
print(f"TOTAL read_bytes = {rd()-base:,} ({(rd()-base)/1e9:.1f} GB)")
print("profile (s, GB):", samples[::2])
print("timing:", json.dumps(timing))
print("answers:", [(r["id"], round(max(r["probabilities"]), 6)) for r in results])
