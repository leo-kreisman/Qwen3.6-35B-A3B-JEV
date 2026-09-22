#!/usr/bin/env bash
# Find the fastest way to serve Qwen3.6-35B-A3B on THIS box.
#
# The project's headline problem: the 20.9 GB checkpoint does not fit a small RAM
# budget, so every expert byte comes off NVMe (measured: 3.67 tok/s decode at
# 212.6 MB/token under an 8 GiB cap). Two RTX 5060 Ti (16 GiB each) sit idle in
# that measurement. `-ncmoe N` keeps the first N layers' experts on the CPU while
# attention + the other layers' experts live in VRAM -- so the disk-resident
# working set shrinks by (40 - N)/40 for free.
#
# Round 1 (this script, no cap): config search. Which ncmoe fits, what does it buy.
# Round 2 does the same configs under a verified-binding 8 GiB cap with the page
# cache evicted first -- that is the number the project can quote.
#
# Output is create-only, one timestamped directory per invocation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-/home/scribe/Projects/llama.cpp-fresh/build/bin/llama-bench}"
GGUF="${GGUF:-$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
NCMOE_LIST="${NCMOE_LIST:-99 36 32 28 24}"
PROMPTS="${PROMPTS:-512}"
NGEN="${NGEN:-32}"
UB="${UB:-512}"
REPS="${REPS:-2}"
CAP="${CAP:-}"

OUTDIR="${OUTDIR:-$REPO_ROOT/results/bench-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
# Only the outermost invocation tees into run.log; the re-exec'd (capped) child
# appends to the same file instead, so a capped run does not fork a logging
# process *inside* the cgroup whose memory budget it is trying to bind.
if [[ -n "${BENCH_ACTIVE:-}" ]]; then
  exec >> "$OUTDIR/run.log" 2>&1
else
  exec > >(tee -a "$OUTDIR/run.log") 2>&1
fi

echo "==> bin      $BIN"
echo "==> gguf     $GGUF ($(du -h "$GGUF" | cut -f1))"
echo "==> ncmoe    $NCMOE_LIST   prompts $PROMPTS   ngen $NGEN   ub $UB   reps $REPS"
echo "==> cap      ${CAP:-none}"
echo "==> output   $OUTDIR"

[[ -x "$BIN" ]] || { echo "missing $BIN" >&2; exit 1; }
[[ -f "$GGUF" ]] || { echo "missing $GGUF" >&2; exit 1; }

evict() {
  python3 - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF
  echo "    page cache: $(fincore "$GGUF" 2>/dev/null | tail -1)"
}

# Per-run storage accounting.
#
# /proc/self/io read_bytes of this shell DOES include its children -- verified on
# this host (a `dd iflag=direct` child reading 512 MB moved the parent's
# read_bytes by 536.9 MB). read_bytes counts only reads that reached the storage
# layer, so a page-cache hit contributes nothing: it is the honest per-run
# measure of what came off the device, and the counter this project already
# trusts. (The cgroup's own io.stat would be a second option, but the io
# controller is not delegated into user scopes here -- only memory and pids are
# in cgroup.subtree_control -- so io.stat does not exist.)
cgdir() {
  local rel
  rel="$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)"
  echo "/sys/fs/cgroup${rel}"
}

# NOTE: do NOT read the counters through a command substitution around awk --
# inside "$(...)" /proc/self/io resolves to *awk*, and awk has read nothing, so
# the whole sweep silently reports 0.00 GB. Bash builtins keep it in this shell.
read_gb() {
  local bytes=""
  while IFS= read -r line; do
    case "$line" in
      read_bytes:*) bytes="${line#read_bytes: }" ;;
    esac
  done < /proc/self/io
  awk -v b="${bytes:-0}" 'BEGIN{printf "%.2f", b/1e9}'
}

device_gb() {
  awk '{printf "%.2f", $3*512/1e9}' /sys/block/nvme0n1/stat
}

cg_report() {
  local dir; dir="$(cgdir)"
  echo "    cgroup          $dir"
  echo "    memory.peak     $(cat "$dir/memory.peak" 2>/dev/null) (the scope is gone by now, so read this from the log above)"
  echo "    memory.events   $(tr '\n' ' ' < "$dir/memory.events" 2>/dev/null)"
}

run_one() {
  local ncmoe="$1"
  echo "--- ncmoe=$ncmoe ---"
  # Evict before EVERY configuration. Evicting once at script start makes config 1
  # cold and every later config a page-cache measurement of the previous config's
  # reads -- exactly the trap this project documented for the scorer harness, and
  # it would rank ncmoe values by run order rather than by merit.
  [[ -n "${BENCH_ACTIVE:-}" ]] && evict
  local before after dbefore dafter
  before="$(read_gb)"; dbefore="$(device_gb)"
  "$BIN" -m "$GGUF" -ngl 99 -ncmoe "$ncmoe" \
    -p "${PROMPTS// /,}" -n "$NGEN" -ub "$UB" -b 2048 \
    -r "$REPS" -o jsonl
  after="$(read_gb)"; dafter="$(device_gb)"
  # Two independent counters, both deltas over exactly this configuration:
  #   process read_bytes -- this shell plus its children, reads that reached storage
  #   /sys/block/nvme0n1/stat -- sectors read by the whole device, nobody excluded
  # They should agree; if they do not, something else on the box read during the run.
  echo "# ncmoe=$ncmoe process_read_gb=$(awk -v a="$after" -v b="$before" 'BEGIN{printf "%.2f", a-b}') device_read_gb=$(awk -v a="$dafter" -v b="$dbefore" 'BEGIN{printf "%.2f", a-b}')"
}

if [[ -n "$CAP" && -z "${BENCH_ACTIVE:-}" ]]; then
  evict
  echo "==> capping memory at $CAP, swap disabled"
  export BENCH_ACTIVE=1
  exec systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 \
    -- "$0" "$@"
fi
# Reached only inside the cap scope (or when CAP is empty). The guard above is on
# BENCH_ACTIVE, not on CAP: gating the re-exec on CAP alone makes the child inherit
# CAP and re-exec itself forever.
if [[ -z "${BENCH_ACTIVE:-}" ]]; then
  echo "==> NOT capped (BENCH_ACTIVE unset and CAP empty)"
else
  echo "==> inside cap scope $CAP: $(cat "$(cgdir)/memory.max" 2>/dev/null)"
fi

: > "$OUTDIR/results.jsonl"
for ncmoe in $NCMOE_LIST; do
  run_one "$ncmoe" >> "$OUTDIR/results.jsonl" 2>&1 || echo "    FAILED ncmoe=$ncmoe (see log)" >&2
done

echo "==> cgroup evidence (the cap was in force for these runs)"
cg_report
echo "==> wrote $OUTDIR/results.jsonl"
python3 - "$OUTDIR/results.jsonl" <<'PYEOF'
import json, sys
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
if not rows:
    print("no rows")
    raise SystemExit(0)
print(f"{'ncmoe':>6} {'test':>10} {'t/s':>10} {'±':>8}")
for r in rows:
    print(f"{r.get('n_cpu_moe', -1):>6} {r.get('n_prompt', 0)}p/{r.get('n_gen', 0)}g {r.get('avg_ts', 0):>10.2f} {r.get('stddev_ts', 0):>8.2f}")
PYEOF
