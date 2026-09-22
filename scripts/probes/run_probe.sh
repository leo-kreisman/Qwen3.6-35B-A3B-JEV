#!/bin/bash
set -u
if [[ -z "${PROBE_ACTIVE:-}" ]]; then
  export PROBE_ACTIVE=1
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
echo "==> cache: $(fincore "$GGUF" 2>/dev/null | tail -1)"
SEMIF_LLAMA_LOAD_MODE=mmap SEMIF_LLAMA_EXTRA_BUFT=0 SEMIF_LLAMA_SEQ_MAX="${SEQ_MAX:-8}" \
  "$PY" /tmp/probe_phases.py
