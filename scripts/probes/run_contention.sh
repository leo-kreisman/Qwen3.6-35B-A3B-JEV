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
# Background CPU load representing "other things processing on the workstation".
PIDS=()
for _ in $(seq "${LOAD:-0}"); do
  bash -c 'while :; do :; done' &
  PIDS+=($!)
done
sleep 1
SEMIF_THREADS="${THREADS:-16}" "$PY" /tmp/probe_ubatch.py 2>/dev/null \
  | grep -E "^(DECODE|scoring|effective)"
for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null; done
wait 2>/dev/null
