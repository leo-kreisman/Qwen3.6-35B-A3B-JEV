#!/usr/bin/env bash
# Serve typed decisions from a RESIDENT scorer under a memory cap, so the
# 20.9 GB checkpoint is read once instead of once per call.
#
# run_semif_35b_ssd.sh answers "how fast is one call from SSD". This answers
# "how fast is the Nth call", which is the question a coding agent actually
# asks, because it makes many calls. The difference is the load: ~19.8 s of the
# ~60 s per-call wall clock, paid every time by the one-shot CLI and once here.
#
# Usage:
#   ./run_resident_35b_ssd.sh                              # host RAM, warm
#   ./run_resident_35b_ssd.sh --simulate-8g                # real disk reads
#   ./run_resident_35b_ssd.sh --simulate-8g examples/decisions.n16.jsonl
#   REPEAT=3 ./run_resident_35b_ssd.sh --simulate-8g       # 3 calls per fixture
#   SEQ_MAX=16 ./run_resident_35b_ssd.sh --simulate-8g examples/decisions.n16.jsonl
#   LOAD_MODE=direct_io ./run_resident_35b_ssd.sh --simulate-8g
#
# Output goes to a fresh, timestamped directory under results/ -- never an
# existing one, because the scorer is create-only by design.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$REPO_ROOT/semif/.venv/bin/python}"
GGUF="${GGUF:-$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$HOME/models/Qwen3.6-35B-A3B-tokenizer}"
# Same 4/3 knee as the one-shot runner; worth more with a resident process
# because the load no longer hides the difference between thread settings.
THREADS="${THREADS:-$(( $(nproc) * 4 / 3 ))}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
REPEAT="${REPEAT:-1}"
# Branches scored per forward pass. Past n_seq_max each extra group is a full
# model sweep, so this must cover the largest batch you intend to score.
SEQ_MAX="${SEQ_MAX:-8}"
LOAD_MODE="${LOAD_MODE:-mmap}"

SIMULATE=""
if [[ "${1:-}" == "--simulate-8g" ]]; then
  SIMULATE="8G"
  shift
fi
FIXTURES=("$@")
if [[ ${#FIXTURES[@]} -eq 0 ]]; then
  FIXTURES=("$REPO_ROOT/examples/decisions.sample.jsonl")
fi

# --- tokenizer --------------------------------------------------------------
if [[ ! -f "$TOKENIZER_DIR/tokenizer_config.json" ]]; then
  echo "==> fetching tokenizer into $TOKENIZER_DIR"
  mkdir -p "$TOKENIZER_DIR"
  command -v uvx >/dev/null || {
    echo "need uvx on PATH, or pre-place $TOKENIZER_DIR (see SETUP.md step 2)" >&2; exit 1; }
  uvx --from huggingface_hub hf download Qwen/Qwen3.6-35B-A3B \
    "tokenizer_config.json" "vocab.json" "merges.txt" --local-dir "$TOKENIZER_DIR"
fi

# --- memory cap -------------------------------------------------------------
# Re-exec under the cap so every variable above is inherited. SEMIF_CAPPED is
# exported rather than re-derived from "$@" so the child cannot silently take
# the warm path if its argument parsing changes -- the same trap the one-shot
# runner documents.
if [[ -n "$SIMULATE" && -z "${SIMULATE_ACTIVE:-}" ]]; then
  echo "==> capping memory at $SIMULATE with swap disabled to force eviction"
  export SIMULATE_ACTIVE=1
  export SEMIF_CAPPED=1
  exec systemd-run --user --scope -p MemoryMax="$SIMULATE" -p MemorySwapMax=0 -- "$0" "$@"
fi

# Evict the checkpoint ONCE, before the resident process starts. A capped scope
# does not recharge pages that are already cached, so without this the cap never
# binds (measured: memory.peak 632 MB, read_bytes 0). Evicting once is also what
# makes the measurement meaningful: call 1 pays the cold cost, and every call
# after it is the residency result rather than a page-cache artefact.
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

# --- sanity -----------------------------------------------------------------
[[ -f "$PY" ]]   || { echo "missing venv: uv venv --python 3.11 $REPO_ROOT/semif/.venv" >&2; exit 1; }
[[ -f "$GGUF" ]] || { echo "missing GGUF: $GGUF (see SETUP.md step 2)" >&2; exit 1; }
for fixture in "${FIXTURES[@]}"; do
  [[ -f "$fixture" ]] || { echo "missing fixture: $fixture" >&2; exit 1; }
done
# Timestamped and create-only: an existing results directory is never touched.
OUTDIR="${OUTDIR:-$REPO_ROOT/results/resident-$(date +%Y%m%d-%H%M%S)}"
LOG="$OUTDIR/run.log"
mkdir -p "$OUTDIR"

echo "==> model    $GGUF ($(du -h "$GGUF" | cut -f1))"
echo "==> threads  $THREADS   seq-max $SEQ_MAX   repeat $REPEAT   load-mode $LOAD_MODE"
echo "==> fixtures ${FIXTURES[*]}"
echo "==> output   $OUTDIR"

# --- serve ------------------------------------------------------------------
# SEQ_MAX and LOAD_MODE are context/model-creation parameters, so they are fixed
# for the life of the process: sweeping them means one process per value.
SEMIF_LLAMA_SEQ_MAX="$SEQ_MAX" \
SEMIF_LLAMA_LOAD_MODE="$LOAD_MODE" \
SEMIF_LLAMA_EXTRA_BUFT="${EXTRA_BUFT:-0}" \
"$PY" "$REPO_ROOT/resident/resident_scorer.py" \
  --gguf "$GGUF" --model "$TOKENIZER_DIR" --revision local \
  --llama-threads "$THREADS" --max-tokens "$MAX_TOKENS" \
  --output-dir "$OUTDIR" --repeat "$REPEAT" \
  "${FIXTURES[@]}" 2>&1 | tee "$LOG"

echo "==> wrote $OUTDIR"
