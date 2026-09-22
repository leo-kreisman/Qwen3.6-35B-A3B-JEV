#!/usr/bin/env bash
# Sweep BigMoeOnEdge's two measured levers on THIS box, for Qwen3.6-35B-A3B.
#
# Why this exists: the box crashed under mmap + an 8 GiB cap because a 20.9 GB
# model behind a demand-paged mapping turns into a fault storm -- the OS evicts
# weights as fast as it reads them. BigMoeOnEdge avoids that by streaming only
# the routed experts with O_DIRECT, and its own README says the fault storm is
# exactly what it fixes. So memory is capped hard here, deliberately: if a
# configuration only survives by eating RAM, this is the sweep that says so.
#
# The two levers that matter, from the engine's own telemetry:
#   --cache-ceil-mb  the LRU expert cache. It is I/O-bound (flash ~0.25 s/token
#                    vs compute ~0.09), so hit rate is the whole game.
#   --io-threads     parallel read lanes. Their device measurement saturates at
#                    two lanes; this box is PCIe 3.0 x4 on the NVMe, so more.
#
# Every run goes through systemd-run with MemoryMax, never a bare invocation:
# the crash this replaces came from trusting the process to stay in budget.
#
# Usage:
#   ./bench_bmoe_sweep.sh                  # 8 GiB cap, default sweep
#   ./bench_bmoe_sweep.sh --warm           # no cap (only if you mean it)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLI="${CLI:-$ROOT/vendor/BigMoeOnEdge/build/cli/bmoe-cli}"
GGUF="${GGUF:-$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
CAP="${CAP-8G}"
THREADS="${THREADS:-6}"
NGEN="${NGEN:-128}"
CTX="${CTX:-4096}"
OUTDIR="${OUTDIR:-$ROOT/results/bmoe-sweep-$(date +%Y%m%d-%H%M%S)}"
PROMPT="${PROMPT:-Explain briefly what a mixture of experts model is.}"

mkdir -p "$OUTDIR"
[[ -x "$CLI"  ]] || { echo "missing $CLI (run vendor/BigMoeOnEdge/scripts/build-host.sh)" >&2; exit 1; }
[[ -f "$GGUF" ]] || { echo "missing $GGUF" >&2; exit 1; }

# cache-ceil : io-threads   -- the grid. 0 = cache off (pure streaming floor).
GRID="${GRID:-0:4 2000:4 3000:4 4000:8}"

echo "==> cli      $CLI"
echo "==> model    $GGUF ($(du -h "$GGUF" | cut -f1))"
echo "==> cap      ${CAP:-none}   threads $THREADS   n $NGEN   ctx $CTX"
echo "==> grid     $GRID"
echo "==> output   $OUTDIR"

run_one() {
  local cache="$1" lanes="$2"
  local tag="cache${cache}_io${lanes}"
  local log="$OUTDIR/$tag.log"
  local csv="$OUTDIR/$tag.csv"
  local cflag
  if [[ "$cache" == "0" ]]; then cflag="--cache-mb 0"; else cflag="--cache-mb auto --cache-ceil-mb $cache"; fi

  echo "--- $tag ---"
  set +e
  systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 -- \
    "$CLI" -m "$GGUF" \
      --moe-stream --dense-weights anon --overlap \
      $cflag --io-threads "$lanes" \
      -t "$THREADS" -c "$CTX" -n "$NGEN" --chatml --no-think \
      --progress --csv "$csv" \
      -p "$PROMPT" > "$log" 2>&1
  local rc=$?
  set -e

  python3 - "$tag" "$log" "$cache" "$lanes" "$rc" <<'PYEOF'
import json, re, sys
tag, log, cache, lanes, rc = sys.argv[1:6]
txt = open(log, errors="replace").read()

perf = {}
m = re.search(r"=== perf ===(.*?)(?:\n===|\Z)", txt, re.S)
body = m.group(1) if m else txt

def grab(pat, cast=float, default=None, src=None):
    mm = re.search(pat, src if src is not None else txt)
    if not mm: return default
    try: return cast(mm.group(1))
    except Exception: return default

res = {
    "tag": tag,
    "cache_ceil_mb": int(cache),
    "io_threads": int(lanes),
    "exit_code": int(rc),
    "tok_s": grab(r"generation:\s*\d+ tokens,\s*[\d.]+ s/token\s*\(([\d.]+) tok/s\)", src=body),
    "s_per_token": grab(r"generation:\s*\d+ tokens,\s*([\d.]+) s/token", src=body),
    "prefill_tok_s": grab(r"prefill:.*?\(([\d.]+) tok/s\)", src=body),
    "load_s": grab(r"model load ([\d.]+) s", src=body),
    "ttft_s": grab(r"TTFT ([\d.]+) s", src=body),
    "cpu_occupancy_pct": grab(r"compute:\s*([\d.]+)% CPU occupancy", src=body),
    "majflt_per_token": grab(r"([\d.]+) major faults/token", src=body),
    "read_mib_total": grab(r"moe-stream: read ([\d.]+) MiB", src=body),
    "mib_per_token": grab(r"\(([\d.]+) MiB/token\)", src=body),
    "flash_io_s_per_token": grab(r"flash I/O ([\d.]+) s/token", src=body),
    "flash_mib_s": grab(r", ([\d.]+) MiB/s\)", src=body),
    "compute_s_per_token": grab(r"compute ([\d.]+) \+", src=body),
    "cache_hit_pct": grab(r"moe-cache: ([\d.]+)% hit", src=body),
    "cache_budget_mb": grab(r"budget (\d+) MiB", cast=int, src=body),
    "evictions": grab(r"([\d.]+) evictions", src=body),
    "overlap_stall_s": grab(r"stall ([\d.]+) s/token", src=body),
    "oom_killed": "oom-kill" in txt or int(rc) == -9,
}
print(json.dumps(res, allow_nan=False))
PYEOF
}

: > "$OUTDIR/sweep.jsonl"
for pair in $GRID; do
  cache="${pair%%:*}"; lanes="${pair##*:}"
  run_one "$cache" "$lanes" >> "$OUTDIR/sweep.jsonl" || true
done

echo "==> wrote $OUTDIR"
python3 - "$OUTDIR/sweep.jsonl" <<'PYEOF'
import json, sys
rows=[]
for line in open(sys.argv[1]):
    line=line.strip()
    if line.startswith("{"): rows.append(json.loads(line))
if not rows: print("no rows"); raise SystemExit(0)
h = f"{'cache MiB':>10} {'io':>3} {'tok/s':>7} {'MiB/tok':>8} {'hit%':>6} {'flash s/t':>10} {'majflt':>7} {'oom':>4}"
print(h); print("-"*len(h))
for r in rows:
    print(f"{r['cache_ceil_mb']:>10} {r['io_threads']:>3} {str(r['tok_s']):>7} "
          f"{str(r['mib_per_token']):>8} {str(r['cache_hit_pct']):>6} "
          f"{str(r['flash_io_s_per_token']):>10} {str(r['majflt_per_token']):>7} "
          f"{'YES' if r['oom_killed'] else 'no':>4}")
PYEOF
