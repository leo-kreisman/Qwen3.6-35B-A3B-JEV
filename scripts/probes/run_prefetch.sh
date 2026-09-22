#!/bin/bash
set -u
if [[ -z "${PROBE_ACTIVE:-}" ]]; then
  export PROBE_ACTIVE=***
  exec systemd-run --user --scope -p MemoryMax="${CAP:-8G}" -p MemorySwapMax=0 -- "$0" "$@"
fi
PY=/home/scribe/Projects/JEV_experiment/semif/.venv/bin/python
GGUF=/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
# Evict the checkpoint from page cache so every run starts cold, exactly as the
# 56.0 s / 42.0 GB baseline was measured.
"$PY" - "$GGUF" <<'PYEOF'
import os, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
print("cache evicted", file=sys.stderr)
PYEOF
"$PY" /tmp/probe_prefetch.py
