"""Isolate llama.cpp's own load cost, with no SHA-256 pre-pass and no tokenizer."""
import sys, time

sys.path.insert(0, "/home/scribe/Projects/JEV_experiment/semif/src")
import llama_cpp
from semif_phase1 import llamacpp_backend as L


def rd():
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return -1


GGUF = "/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
mode = sys.argv[1] if len(sys.argv) > 1 else "mmap"

base = rd()
params = llama_cpp.llama_model_default_params()
params.n_gpu_layers = 0
params.load_mode = L._LOAD_MODES[mode]
params.use_extra_bufts = False
mark = time.perf_counter()
model = llama_cpp.llama_model_load_from_file(GGUF.encode(), params)
elapsed = time.perf_counter() - mark
print(f"LOAD_ONLY[{mode}]: {elapsed:.2f}s  read_bytes = {rd()-base:,} ({(rd()-base)/1e9:.1f} GB)")
llama_cpp.llama_model_free(model)
