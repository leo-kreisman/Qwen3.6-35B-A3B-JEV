#!/bin/bash
# One-criterion run under the cap with the cache evicted, sampling read_bytes.
# If scoring cost is linear in criteria count, this should read ~3 model passes
# (SHA-256 + prefill + 1 branch ~= 58 GB) rather than the ~117 GB a 4-criteria
# run implies (6 passes).
set -u
PY=/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python
TOK=/home/scribe/models/Qwen3.6-35B-A3B-tokenizer
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
IN=/home/scribe/Projects/JEV_experiment/decisions.one.jsonl
OUT=/home/scribe/Projects/JEV_experiment/decisions.one_capped.jsonl
LOG=/tmp/one_capped.log
SAMPLES=/tmp/samples_one.txt
: > "$SAMPLES"

SEMIF_LLAMA_LOAD_MODE=mmap SEMIF_LLAMA_EXTRA_BUFT=0 \
  "$PY" -m semif_phase1.cli --backend llamacpp --mode shared \
  --model "$TOK" --revision local --gguf "$GGUF" \
  --llama-threads 6 --max-tokens 8192 \
  --input "$IN" --output "$OUT" > "$LOG" 2>&1 &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 5
  rb=$(awk '/^read_bytes/{print $2}' /proc/$pid/io 2>/dev/null || echo "$rb")
  [ "${rb:-0}" -gt "$peak" ] 2>/dev/null && peak=$rb
  echo "t=${SECONDS}s read_bytes=${rb:-?}" >> "$SAMPLES"
done
wait "$pid"; echo "exit=$? peak_read_bytes=$peak" >> "$SAMPLES"
cg=$(awk -F: '{print $3}' /proc/$$/cgroup)
[ -r "/sys/fs/cgroup$cg/memory.peak" ] && echo "memory.peak=$(cat /sys/fs/cgroup$cg/memory.peak)" >> "$SAMPLES"
echo "ALLDONE" >> "$SAMPLES"
