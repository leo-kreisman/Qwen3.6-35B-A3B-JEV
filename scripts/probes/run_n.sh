#!/bin/bash
set -u
if [[ -z "${PROBE_ACTIVE:-}" ]]; then
  export PROBE_ACTIVE=***
  exec systemd-run --user --scope -p MemoryMax="${CAP:-8G}" -p MemorySwapMax=0 -- "$0" "$@"
fi
PY=/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
"$PY" - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF
export SEMIF_INPUT="${IN_FILE}"
SEMIF_THREADS="${THREADS:-16}" "$PY" /tmp/probe_ubatch.py 2>&1 | grep -vE "^(graph_reserve|resolve_fused|sched_reserve|create_tensor|load_tensors|print_info|ggml_|llama_|~)"
