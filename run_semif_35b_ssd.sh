#!/usr/bin/env bash
# Score typed decisions with Qwen3.6-35B-A3B streamed from SSD through SemIf's
# llama.cpp backend.
#
# Why this works with no expert-offload flags: SemIf loads the GGUF with
# n_gpu_layers=0, so every layer -- experts included -- is already on the CPU.
# The -ot / --n-cpu-moe / --cpu-moe flags exist to keep experts on CPU *while
# other layers go to the GPU*; with zero GPU layers that precondition is already
# met. Streaming then comes from load_mode=mmap (experts are file-backed and
# demand-paged) plus use_extra_bufts=0 (no repack into anonymous RAM).
#
# Usage:
#   ./run_semif_35b_ssd.sh                 # host RAM; page cache hides the disk
#   ./run_semif_35b_ssd.sh --simulate-8g   # cap at 8 GiB to force real reads
#   MODE=direct ./run_semif_35b_ssd.sh     # rows with differing states
#
# Paths are env-overridable so a fresh clone runs. See SETUP.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMIF_DIR="${SEMIF_DIR:-$REPO_ROOT/semif}"
PY="${PY:-$SEMIF_DIR/.venv/bin/python}"
GGUF="${GGUF:-$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$HOME/models/Qwen3.6-35B-A3B-tokenizer}"
INPUT="${INPUT:-$REPO_ROOT/examples/decisions.sample.jsonl}"
OUTPUT="${OUTPUT:-$REPO_ROOT/results/decisions.out.jsonl}"

# Mildly oversubscribe the hardware threads: an expert-streaming decode stalls on page
# faults, so extra runnable threads keep the device queue fed. Measured knee at 4/3 of
# nproc (6 threads 54.6 s -> 16 threads 38.4 s on a 12-thread box). Override THREADS to pin.
# Note this is worth MORE under contention, not less: 12 competing busy threads make
# 16 threads beat 6 by 2.1x, against 1.4x idle.
THREADS="${THREADS:-$(( $(nproc) * 4 / 3 ))}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
# shared = one page state prefilled once, every criterion scored against it in
# parallel (the JEV pattern; requires all rows to carry one identical state).
# direct = each row scored independently; use it for rows with differing states.
MODE="${MODE:-shared}"

SIMULATE=""
if [[ "${1:-}" == "--simulate-8g" ]]; then
  SIMULATE="8G"
  shift
fi

# --- 1. tokenizer -----------------------------------------------------------
# SemIf renders the prompt with the reference HF tokenizer and then re-tokenizes
# through the GGUF vocabulary, refusing to score if the two disagree. So the
# tokenizer must come from the same family as the GGUF.
if [[ ! -f "$TOKENIZER_DIR/tokenizer_config.json" ]]; then
  echo "==> fetching tokenizer into $TOKENIZER_DIR"
  mkdir -p "$TOKENIZER_DIR"
  command -v uvx >/dev/null || {
    echo "need uvx on PATH, or pre-place $TOKENIZER_DIR (see SETUP.md step 2)" >&2; exit 1; }
  uvx --from huggingface_hub hf download \
    Qwen/Qwen3.6-35B-A3B \
    "tokenizer_config.json" "vocab.json" "merges.txt" \
    --local-dir "$TOKENIZER_DIR"
fi

# --- 2. memory cap ----------------------------------------------------------
# The host that measured this has 62 GiB RAM against a 20.9 GB checkpoint, so mmap
# serves nearly every expert from page cache and the SSD path is never exercised.
# A cgroup memory cap is only half the fix, though: a cgroup does NOT recharge a
# page that is already cached, so a fresh scope reuses the warm file for free and
# the cap never binds (measured: memory.peak 632 MB, read_bytes 0). The GGUF has
# to be evicted from page cache too, or --simulate-8g silently measures nothing.
# Re-exec under the cap so every variable above is inherited.
if [[ -n "$SIMULATE" && -z "${SIMULATE_ACTIVE:-}" ]]; then
  echo "==> capping memory at $SIMULATE with swap disabled to force eviction"
  export SIMULATE_ACTIVE=1
  # SEMIF_CAPPED is exported rather than re-derived from "$@" in the child: the
  # re-exec used to pass "$0" with no arguments, so the child never saw
  # --simulate-8g and every SIMULATE-gated step below silently took the warm
  # path. Passing "$@" fixes the re-parse, but the env var makes it independent
  # of argument handling entirely.
  export SEMIF_CAPPED=1
  exec systemd-run --user --scope -p MemoryMax="$SIMULATE" -p MemorySwapMax=0 -- "$0" "$@"
fi

# Evict the checkpoint so the run's own reads come from disk. Unprivileged:
# posix_fadvise(DONTNEED) drops the file's clean cached pages, no root needed.
# A capped scope must re-fault them itself, which is what charges its cgroup and
# makes the cap bite. Verify with `fincore "$GGUF"` (want PAGES 0) and watch
# /proc/<pid>/io read_bytes -- that counter, not wall-clock, is the real check.
# Only under --simulate-8g: without it the run is deliberately left warm, so the
# two modes differ in cache state as well as the cap, and both are labelled.
if [[ -n "${SEMIF_CAPPED:-}" ]]; then
  "$PY" - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF
  echo "==> page cache for GGUF: $(fincore "$GGUF" 2>/dev/null | tail -1)"
else
  echo "==> page cache for GGUF: $(fincore "$GGUF" 2>/dev/null | tail -1) (warm run)"
fi

# --- 3. sanity --------------------------------------------------------------
[[ -f "$PY" ]]   || { echo "missing venv: uv venv --python 3.11 $SEMIF_DIR/.venv" >&2; exit 1; }
[[ -f "$GGUF" ]] || { echo "missing GGUF: $GGUF (see SETUP.md step 2)" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"  # the CLI is create-only and refuses to overwrite

echo "==> model   $GGUF ($(du -h "$GGUF" | cut -f1))"
echo "==> mode    $MODE   threads $THREADS   max-tokens $MAX_TOKENS"

# --- 4. score ---------------------------------------------------------------
# load_mode=mmap keeps the GGUF file-backed and demand-paged; use_extra_bufts=0
# disables the load-time repack that would otherwise materialise the expert
# tensors into anonymous RAM and OOM-kill a >RAM checkpoint during load.
# Both are recorded per row under the "model" key so every result file is
# self-describing. Override LOAD_MODE / EXTRA_BUFT to experiment.
SEMIF_LLAMA_LOAD_MODE="${LOAD_MODE:-mmap}" \
SEMIF_LLAMA_EXTRA_BUFT="${EXTRA_BUFT:-0}" \
"$PY" -m semif_phase1.cli \
  --backend llamacpp \
  --mode "$MODE" \
  --model "$TOKENIZER_DIR" --revision local \
  --gguf "$GGUF" \
  --llama-threads "$THREADS" \
  --max-tokens "$MAX_TOKENS" \
  --input "$INPUT" --output "$OUTPUT"

echo "==> wrote $OUTPUT"
