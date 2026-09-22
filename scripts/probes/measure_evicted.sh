#!/bin/bash
# Measure the SSD path for real: evict the GGUF from page cache immediately
# before each run, then sample /proc/<pid>/io read_bytes (actual storage reads)
# and the cgroup's memory.events (whether the cap ever bound).
set -u
PY=/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python
TOK=/home/scribe/models/Qwen3.6-35B-A3B-tokenizer
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
IN=/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl

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
  local tag="$1" out="$2" log="$3" samples="$4"
  : > "$samples"
  SEMIF_LLAMA_LOAD_MODE=mmap SEMIF_LLAMA_EXTRA_BUFT=0 \
  "$PY" -m semif_phase1.cli --backend llamacpp --mode shared \
    --model "$TOK" --revision local --gguf "$GGUF" \
    --llama-threads 6 --max-tokens 8192 \
    --input "$IN" --output "$out" > "$log" 2>&1 &
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

rm -f /tmp/samples_capped_ev.txt /tmp/samples_warm_ev.txt
echo "=== CAPPED 8G + evicted ==="
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- \
  bash -c "$(declare -f run_one evict); PY=$PY TOK=$TOK GGUF=$GGUF IN=$IN; \
           evict; run_one capped_ev /home/scribe/Projects/JEV_experiment/decisions.ssd8g_evicted.jsonl \
                   /tmp/capped_ev.log /tmp/samples_capped_ev.txt" 2>&1 | tail -3
echo "=== WARM (no cap) + evicted ==="
bash -c "$(declare -f run_one evict); PY=$PY TOK=$TOK GGUF=$GGUF IN=$IN; \
         evict; run_one warm_ev /home/scribe/Projects/JEV_experiment/decisions.warm_evicted.jsonl \
                 /tmp/warm_ev.log /tmp/samples_warm_ev.txt"
echo "ALLDONE"
