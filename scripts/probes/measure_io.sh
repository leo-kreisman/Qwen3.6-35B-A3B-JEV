#!/bin/bash
# Run the scorer twice -- capped and warm -- sampling /proc/<pid>/io read_bytes.
# read_bytes counts bytes actually fetched from the storage layer, so it is the
# direct evidence of whether the SSD path is being exercised, rather than an
# inference from wall-clock time.
set -u
PY=/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python
TOK=/home/scribe/models/Qwen3.6-35B-A3B-tokenizer
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
IN=/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl

run_one() {
  local tag="$1" out="$2" log="$3" samples="$4"
  : > "$samples"
  SEMIF_LLAMA_LOAD_MODE=mmap SEMIF_LLAMA_EXTRA_BUFT=0 \
  "$PY" -m semif_phase1.cli --backend llamacpp --mode shared \
    --model "$TOK" --revision local --gguf "$GGUF" \
    --llama-threads 6 --max-tokens 8192 \
    --input "$IN" --output "$out" > "$log" 2>&1 &
  local pid=$!
  local cg peak=0 rb=0
  cg=$(awk -F: '{print $3}' /proc/$pid/cgroup 2>/dev/null)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 3
    rb=$(awk '/^read_bytes/{print $2}' /proc/$pid/io 2>/dev/null || echo "$rb")
    [ "${rb:-0}" -gt "$peak" ] 2>/dev/null && peak=$rb
    echo "t=${SECONDS}s read_bytes=${rb:-?}" >> "$samples"
  done
  wait "$pid"; local rc=$?
  echo "TAG=$tag exit=$rc peak_read_bytes=$peak cgroup=${cg:-none}" >> "$samples"
  if [ -n "${cg:-}" ] && [ -r "/sys/fs/cgroup$cg/memory.events" ]; then
    echo "--- memory.events ---" >> "$samples"
    cat "/sys/fs/cgroup$cg/memory.events" >> "$samples"
  fi
  if [ -n "${cg:-}" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ]; then
    echo "memory.peak=$(cat /sys/fs/cgroup$cg/memory.peak)" >> "$samples"
  fi
}

rm -f /tmp/samples_warm.txt /tmp/samples_capped.txt
echo "=== WARM (no cap, repack off) ==="
run_one warm /home/scribe/Projects/JEV_experiment/decisions.warm_noRepack.jsonl \
        /tmp/warm_norepack.log /tmp/samples_warm.txt
echo "=== CAPPED (8G, swap off, repack off) ==="
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- \
  bash -c "$(declare -f run_one); PY=$PY TOK=$TOK GGUF=$GGUF IN=$IN; \
           run_one capped /home/scribe/Projects/JEV_experiment/decisions.ssd8g2.jsonl \
                   /tmp/capped2.log /tmp/samples_capped.txt" 2>&1 | tail -3
echo "ALLDONE"
