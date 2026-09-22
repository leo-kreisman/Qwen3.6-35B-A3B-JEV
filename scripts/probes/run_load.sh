#!/bin/bash
set -u
if [[ -z "${PROBE_ACTIVE:-}" ]]; then
  export PROBE_ACTIVE=1
  exec systemd-run --user --scope -p MemoryMax="${CAP:-8G}" -p MemorySwapMax=0 -- "$0" "$@"
fi
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF
/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python /tmp/probe_load.py mmap 2>&1 | grep -E "LOAD_ONLY|error|abort"
