#!/bin/bash
set -u
if [[ -z "${PROBE_ACTIVE:-}" ]]; then
  export PROBE_ACTIVE=***
  exec systemd-run --user --scope -p MemoryMax="${CAP:-8G}" -p MemorySwapMax=0 -- "$0" "$@"
fi
PY=/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
TOK=/home/scribe/models/Qwen3.6-35B-A3B-tokenizer
IN=/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl
# Evict ONCE. A real agent's second call inherits whatever the first process left
# cached, so evicting between calls would overstate the one-shot cost.
"$PY" - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF

export SEMIF_LLAMA_LOAD_MODE=mmap SEMIF_LLAMA_EXTRA_BUFT=0

if [[ "${VARIANT:-oneshot}" == "oneshot" ]]; then
  for i in 1 2 3; do
    t0=$(date +%s.%N)
    "$PY" /tmp/oneshot_wrapper.py --backend llamacpp --mode shared \
      --model "$TOK" --revision local --gguf "$GGUF" \
      --max-tokens 8192 --input "$IN" --output "/tmp/life_$i.jsonl" 2>/dev/null \
      | grep READ_BYTES | sed "s/^/CALL $i /"
    t1=$(date +%s.%N)
    echo "CALL $i WALL $(echo "$t1 - $t0" | bc)"
  done
else
  "$PY" /tmp/probe_lifecycle.py
fi
