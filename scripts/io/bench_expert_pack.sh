#!/usr/bin/env bash
# Measure whether the expert pack reduces device reads for a routed expert set.
#
# Discipline (scripts/README.md): evict the page cache before every arm, and
# measure /proc/self/io read_bytes rather than wall clock. A warm file is faster
# than a disk file and wall clock cannot tell you which one you got.
#
# Each arm runs in its own process, because a cold arm followed by a second arm
# in the same process would measure the first arm's page cache.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$HERE/bench_expert_pack"

SRC="${SRC:-$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
PACK="${PACK:-$HOME/models/jev-pack/qwen3.6-35b-a3b-expertpack.bin}"
REGIONS="$PACK.source_regions.tsv"

EXPERTS="${EXPERTS:-8}"     # top-8 routing
SETS="${SETS:-4}"           # 4 passes over all 40 layers
QD="${QD:-8}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="$ROOT/results/expertpack-$STAMP"
mkdir -p "$OUTDIR"

if [[ ! -x "$BIN" ]]; then
  echo "building $BIN"
  ( cd "$HERE" && cc -O2 -o bench_expert_pack bench_expert_pack.c ) || exit 1
fi
for f in "$SRC" "$PACK" "$REGIONS"; do
  [[ -r "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

# Drop the file's clean pages from the page cache. posix_fadvise(DONTNEED) is
# unprivileged; confirm with fincore.
evict() {
  python3 - "$@" <<'PY' >/dev/null 2>&1
import ctypes, os, sys
lib = ctypes.CDLL("libc.so.6", use_errno=True)
POSIX_FADV_DONTNEED = 4
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        lib.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
PY
  sync
  sleep 0.5
}

cached_pages() {
  fincore --bytes "$1" 2>/dev/null | awk 'NR==2 {print $2}'
}

run_arm() {
  local name="$1"; shift
  # .txt, not .log: the repo gitignores *.log, and this output is the evidence
  # the doc cites, so it has to be tracked.
  local log="$OUTDIR/$name.txt"
  echo "=== arm: $name" | tee -a "$OUTDIR/run.txt"
  evict "$SRC" "$PACK"
  local before_cached
  before_cached="$(cached_pages "$SRC")"
  echo "  cached before: ${before_cached:-?} bytes" | tee -a "$OUTDIR/run.txt"
  "$BIN" --pack "$PACK" --source "$SRC" --regions "$REGIONS" \
         --experts "$EXPERTS" --sets "$SETS" --qd "$QD" --arm "$name" 2>&1 | tee "$log"
  grep -hE "requests|device read" "$log" | sed "s/^/  [$name] /" >> "$OUTDIR/summary.txt"
  echo "" >> "$OUTDIR/summary.txt"
  evict "$SRC" "$PACK"
}

echo "src   $SRC" | tee "$OUTDIR/run.txt"
echo "pack  $PACK" | tee -a "$OUTDIR/run.txt"
echo "set   $EXPERTS experts x 40 layers x $SETS passes" | tee -a "$OUTDIR/run.txt"
echo "" | tee -a "$OUTDIR/run.txt"

: > "$OUTDIR/summary.txt"
"$BIN" --pack "$PACK" --source "$SRC" --regions "$REGIONS" --check 2>&1 | tee "$OUTDIR/odirect-check.txt"

run_arm source
run_arm pack
run_arm pack-uring
run_arm source-uring

echo ""
echo "=== summary ==="
cat "$OUTDIR/summary.txt"
echo ""
echo "results in $OUTDIR"
