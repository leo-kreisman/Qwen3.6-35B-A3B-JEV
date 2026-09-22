#!/bin/bash
# The generation measurement, on the same footing as every other SSD number in
# this repo: page cache evicted immediately before the run, hard 8 GiB cgroup
# cap, cap confirmed to have bound, reads taken from /proc/<pid>/io.
set -u
ROOT=/home/scribe/Projects/JEV_experiment
PY=$ROOT/semif/.venv/bin/python
TOK=/home/scribe/models/Qwen3.6-35B-A3B-tokenizer
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
NTOK="${NTOK:-64}"
OUT="$ROOT/results/decode-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

evict() {
  "$PY" - "$GGUF" <<'EOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
EOF
  fincore "$GGUF" | tail -1
}

run_one() {
  local tag="$1" samples="$2"
  : > "$samples"
  SEMIF_LLAMA_LOAD_MODE=mmap SEMIF_LLAMA_EXTRA_BUFT=0 \
  "$PY" "$ROOT/scripts/probes/probe_decode.py" \
    --model "$TOK" --revision local --gguf "$GGUF" \
    --ntok "$NTOK" --threads "${THREADS:-8}" --ctx 8192 \
    --report "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1 &
  local pid=$! cg peak=0 rb=0
  cg=$(awk -F: '{print $3}' /proc/$pid/cgroup 2>/dev/null)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    rb=$(awk '/^read_bytes/{print $2}' /proc/$pid/io 2>/dev/null || echo "$rb")
    [ "${rb:-0}" -gt "$peak" ] 2>/dev/null && peak=$rb
    echo "t=${SECONDS}s read_bytes=${rb:-?}" >> "$samples"
  done
  wait "$pid"; local rc=$?
  echo "TAG=$tag exit=$rc peak_read_bytes=$peak" >> "$samples"
  [ -n "${cg:-}" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ] && \
    echo "memory.peak=$(cat /sys/fs/cgroup$cg/memory.peak)" >> "$samples"
  [ -n "${cg:-}" ] && [ -r "/sys/fs/cgroup$cg/memory.events" ] && \
    { echo "--- memory.events ---"; cat "/sys/fs/cgroup$cg/memory.events"; } >> "$samples"
}

echo "=== CAPPED 8G + evicted cache ==="
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- \
  bash -c "$(declare -f run_one evict); PY=$PY ROOT=$ROOT TOK=$TOK GGUF=$GGUF NTOK=$NTOK OUT=$OUT THREADS=${THREADS:-8}; \
           evict; run_one capped_ev $OUT/samples_capped.txt" 2>&1 | tail -3

echo "=== WARM (no cap) + evicted cache ==="
bash -c "$(declare -f run_one evict); PY=$PY ROOT=$ROOT TOK=$TOK GGUF=$GGUF NTOK=$NTOK OUT=$OUT THREADS=${THREADS:-8}; \
         evict; run_one warm_ev $OUT/samples_warm.txt"

echo "OUTDIR=$OUT"
echo "ALLDONE"
