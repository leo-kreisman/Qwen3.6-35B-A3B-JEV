#!/bin/bash
# Batched-branch measurement: evicted page cache + cgroup cap, sampling read_bytes.
# Re-execs itself under systemd-run so the cap applies to the whole subtree.
set -u
PY=/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python
TOK=/home/scribe/models/Qwen3.6-35B-A3B-tokenizer
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
IN=/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl
OUT=${OUT:-/home/scribe/Projects/JEV_experiment/decisions.batched.jsonl}
LOG=${LOG:-/tmp/batched.log}
SAMPLES=${SAMPLES:-/tmp/samples_batched.txt}

if [[ -z "${BATCHED_ACTIVE:-}" ]]; then
  export BATCHED_ACTIVE=1
  exec systemd-run --user --scope -p MemoryMax="${CAP:-8G}" -p MemorySwapMax=0 -- "$0" "$@"
fi

echo "==> cgroup: $(awk -F: '{print $3}' /proc/$$/cgroup)"
"$PY" - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF
echo "==> page cache for GGUF: $(fincore "$GGUF" 2>/dev/null | tail -1)"

: > "$SAMPLES"
SEMIF_LLAMA_LOAD_MODE="${LOAD_MODE:-mmap}" SEMIF_LLAMA_EXTRA_BUFT="${EXTRA_BUFT:-0}" \
SEMIF_LLAMA_SEQ_MAX="${SEQ_MAX:-8}" \
  "$PY" -m semif_phase1.cli --backend llamacpp --mode shared \
  --model "$TOK" --revision local --gguf "$GGUF" \
  --llama-threads "${THREADS:-6}" --max-tokens 8192 \
  --input "$IN" --output "$OUT" > "$LOG" 2>&1 &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  sleep 5
  rb=$(awk '/^read_bytes/{print $2}' /proc/$pid/io 2>/dev/null || echo "$rb")
  [ "${rb:-0}" -gt "$peak" ] 2>/dev/null && peak=$rb
  echo "t=${SECONDS}s read_bytes=${rb:-?}" >> "$SAMPLES"
done
wait "$pid"; rc=$?
echo "exit=$rc peak_read_bytes=$peak" >> "$SAMPLES"
cg=$(awk -F: '{print $3}' /proc/$$/cgroup)
[ -r "/sys/fs/cgroup$cg/memory.peak" ] && echo "memory.peak=$(cat /sys/fs/cgroup$cg/memory.peak)" >> "$SAMPLES"
[ -r "/sys/fs/cgroup$cg/memory.events" ] && grep -E '^(max|oom)' "/sys/fs/cgroup$cg/memory.events" | sed 's/^/memory.events /' >> "$SAMPLES"
echo "ALLDONE" >> "$SAMPLES"
