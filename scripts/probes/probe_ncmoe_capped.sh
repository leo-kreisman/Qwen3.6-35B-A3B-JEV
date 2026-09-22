#!/usr/bin/env bash
# How fast can this box serve Qwen3.6-35B-A3B from SSD under a hard host-RAM cap,
# using the two idle RTX 5060 Ti cards?
#
# The project's published cold number is 3.67 tok/s at 212.6 MB per generated
# token, measured with n_gpu_layers=0 -- every weight, attention included, on the
# CPU behind a demand-paged mmap. `-ncmoe N` keeps the first N layers' *expert*
# weights on the CPU and lets everything else (attention, norms, embeddings, the
# remaining layers' experts) live in VRAM. So the same SSD-streaming path gets a
# GPU for its dense half, and the disk-resident working set shrinks by (40-N)/40.
#
# METHOD, and why each part is here:
#   * systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0  -- a real cap.
#   * posix_fadvise(DONTNEED) on the GGUF BEFORE EVERY CONFIGURATION. A cgroup does
#     not re-charge an already-cached page, so without eviction the cap never binds
#     and the run measures the page cache. Evicting once at the top would rank the
#     configs by run order instead of by merit.
#   * TWO device counters per run, both deltas over exactly that run:
#       /proc/self/io read_bytes   -- this shell + its reaped children
#       /sys/block/nvme0n1/stat    -- the whole device, nobody excluded
#     They must agree; if they diverge, something else on the box read during the
#     run. read_bytes counts only reads that reached storage, so a page-cache hit
#     contributes nothing -- this is the counter the project already trusts.
#   * A LOAD BASELINE per configuration (`-p 1 -n 1`). llama-bench reports tok/s
#     and nothing about bytes, and the model load alone reads the checkpoint; without
#     the baseline the workload's own storage traffic is unattributable.
#   * memory.peak and memory.events read from INSIDE the scope after every run, so
#     the cap's effect is observed rather than assumed. A run whose `memory.events
#     max` is 0 did not stream -- it was served from cache.
#
# TRAP THIS SCRIPT EXISTS TO AVOID: /proc/self/io read inside "$(...)" reads the
# *subshell*, which has read nothing, so the whole sweep reports 0.00 GB and looks
# plausible. Counters are therefore read with shell builtins in this shell.
#
# Usage:
#   CAP=8G NCMOE_LIST="99 32 24 16 8 0" ./probe_ncmoe_capped.sh
#   NCMOE_LIST="99 0" PROMPTS=135 NGEN=32 ./probe_ncmoe_capped.sh   # no cap (warm)
#
# Output: create-only timestamped directory under results/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-/home/scribe/Projects/llama.cpp-fresh/build/bin/llama-bench}"
GGUF="${GGUF:-$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
DEV="${DEV:-/sys/block/nvme0n1/stat}"
NCMOE_LIST="${NCMOE_LIST:-99 32 24 16 8 0}"
PROMPTS="${PROMPTS:-135}"
NGEN="${NGEN:-32}"
UB="${UB:-512}"
REPS="${REPS:-1}"
CAP="${CAP:-}"
THREADS="${THREADS:-}"

OUTDIR="${OUTDIR:-$REPO_ROOT/results/probe-ncmoe-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
if [[ -n "${PROBE_ACTIVE:-}" ]]; then
  exec >> "$OUTDIR/run.log" 2>&1
else
  exec > >(tee -a "$OUTDIR/run.log") 2>&1
fi

# --- counters (shell builtins; see the trap note above) ----------------------
PROC_BYTES=0
DEV_BYTES=0

snap() {                     # snap <proc_var> <dev_var>
  local line
  while IFS= read -r line; do
    case "$line" in
      read_bytes:*) PROC_BYTES="${line#read_bytes: }" ;;
    esac
  done < /proc/self/io
  DEV_BYTES="$(awk '{print $3*512}' "$DEV")"
  eval "$1=$PROC_BYTES"; eval "$2=$DEV_BYTES"
}

gb() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1e9}'; }

cgdir() {
  local rel
  rel="$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)"
  echo "/sys/fs/cgroup${rel}"
}

cg_peak() {
  local p; p="$(cat "$(cgdir)/memory.peak" 2>/dev/null || echo 0)"
  awk -v b="$p" 'BEGIN{printf "%.2f", b/1073741824}'
}

cg_max_events() {
  awk '/^max /{print $2}' "$(cgdir)/memory.events" 2>/dev/null || echo "?"
}

evict() {
  python3 - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF
}

# --- one configuration -------------------------------------------------------
# Emits, per configuration:
#   baseline   load-only reads and wall time
#   workload   prefill + decode reads and wall time
#   cap        memory.peak and memory.events max, read from inside the scope
row_header_printed=0
run_config() {
  local ncmoe="$1"
  echo "--- ncmoe=$ncmoe ---"

  evict
  local b_proc b_dev a_proc a_dev
  snap b_proc b_dev
  "$BIN" -m "$GGUF" -ngl 99 -ncmoe "$ncmoe" -p 1 -n 1 -ub "$UB" -b 2048 \
      ${THREADS:+-t "$THREADS"} -r 1 -o jsonl > "$OUTDIR/.baseline.jsonl" 2>/dev/null || true
  snap a_proc a_dev
  local load_proc=$((a_proc-b_proc)) load_dev=$((a_dev-b_dev))
  local peak_load; peak_load="$(cg_peak)"
  local ev_load; ev_load="$(cg_max_events)"

  evict
  snap b_proc b_dev
  "$BIN" -m "$GGUF" -ngl 99 -ncmoe "$ncmoe" -p "$PROMPTS" -n "$NGEN" -ub "$UB" -b 2048 \
      ${THREADS:+-t "$THREADS"} -r "$REPS" -o jsonl > "$OUTDIR/.workload.jsonl" 2>/dev/null || true
  snap a_proc a_dev
  local work_proc=$((a_proc-b_proc)) work_dev=$((a_dev-b_dev))
  local peak_work; peak_work="$(cg_peak)"
  local ev_work; ev_work="$(cg_max_events)"

  python3 - "$ncmoe" "$OUTDIR/.workload.jsonl" "$load_proc" "$load_dev" \
             "$work_proc" "$work_dev" "$peak_load" "$peak_work" "$ev_load" "$ev_work" \
    <<'PYEOF'
import json, sys
(ncmoe, path, load_proc, load_dev, work_proc, work_dev,
 peak_load, peak_work, ev_load, ev_work) = sys.argv[1:]
prefill = gen = None
for line in open(path):
    line = line.strip()
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("n_prompt", 0) > 1:
        prefill = r
    elif r.get("n_gen", 0) > 1:
        gen = r
# The workload's own storage traffic is its total minus the load-only baseline.
own_dev = int(work_dev) - int(load_dev)
gb = lambda b: b / 1e9
rec = {
    "ncmoe": int(ncmoe),
    "prefill_tok_s": prefill and round(prefill["avg_ts"], 2),
    "prefill_seconds": prefill and round(prefill["avg_ns"] / 1e9, 2),
    "gen_tok_s": gen and round(gen["avg_ts"], 2),
    "gen_seconds": gen and round(gen["avg_ns"] / 1e9, 2),
    "load_wall_seconds": None,  # filled from the baseline by the caller if wanted
    "baseline_read_gb_proc": round(gb(int(load_proc)), 2),
    "baseline_read_gb_dev": round(gb(int(load_dev)), 2),
    "workload_read_gb_proc": round(gb(int(work_proc)), 2),
    "workload_read_gb_dev": round(gb(int(work_dev)), 2),
    "inference_read_gb_dev": round(gb(own_dev), 2),
    "mb_read_per_gen_token": gen and round(own_dev / max(1, gen["n_gen"]) / 1e6, 1),
    "memory_peak_gib_workload": float(peak_work),
    "memory_events_max_workload": int(ev_work) if ev_work.isdigit() else ev_work,
    "memory_peak_gib_baseline": float(peak_load),
    "memory_events_max_baseline": int(ev_load) if ev_load.isdigit() else ev_load,
}
print(json.dumps({"config": rec}, allow_nan=False))
PYEOF
  echo "# ncmoe=$ncmoe load_reads proc=$(gb $load_proc) dev=$(gb $load_dev) GB | workload_reads proc=$(gb $work_proc) dev=$(gb $work_dev) GB | peak=$(cg_peak) GiB max_events=$(cg_max_events)"
}

echo "==> bin        $BIN"
echo "==> gguf       $GGUF ($(du -h "$GGUF" | cut -f1))"
echo "==> ncmoe      $NCMOE_LIST"
echo "==> workload   prefill $PROMPTS tokens, decode $NGEN tokens, ub $UB, reps $REPS"
echo "==> cap        ${CAP:-none}   threads ${THREADS:-llama-bench default}"
echo "==> output     $OUTDIR"
[[ -x "$BIN" ]]  || { echo "missing $BIN" >&2; exit 1; }
[[ -f "$GGUF" ]] || { echo "missing $GGUF" >&2; exit 1; }

if [[ -n "$CAP" && -z "${PROBE_ACTIVE:-}" ]]; then
  evict
  echo "==> page cache after eviction: $(fincore "$GGUF" 2>/dev/null | tail -1)"
  echo "==> re-exec under MemoryMax=$CAP MemorySwapMax=0"
  export PROBE_ACTIVE=1
  exec systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 -- "$0" "$@"
fi
# Gate on PROBE_ACTIVE, never on CAP: a child inherits CAP, so gating on CAP alone
# re-execs itself forever (found the hard way, ~40 nested scopes).
if [[ -n "${PROBE_ACTIVE:-}" ]]; then
  echo "==> inside scope: $(cgdir) memory.max=$(cat "$(cgdir)/memory.max")"
else
  echo "==> NOT capped (warm run)"
fi

: > "$OUTDIR/configs.jsonl"
echo '{"note":"device read deltas are /sys/block/'"$(basename "$(dirname "$DEV")")"'/stat sectors*512, whole device; proc deltas are /proc/self/io read_bytes, this shell and its reaped children"}' >> "$OUTDIR/configs.jsonl"
for ncmoe in $NCMOE_LIST; do
  run_config "$ncmoe" >> "$OUTDIR/configs.jsonl"
done

echo "==> wrote $OUTDIR"
python3 - "$OUTDIR/configs.jsonl" <<'PYEOF'
import json, sys
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith('{"config"'):
        continue
    rows.append(json.loads(line)["config"])
if not rows:
    print("no rows"); raise SystemExit(0)
hdr = f"{'ncmoe':>6} {'prefill t/s':>12} {'gen t/s':>9} {'inf read GB':>12} {'MB/token':>9} {'peak GiB':>9} {'cap hits':>10}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['ncmoe']:>6} {str(r['prefill_tok_s']):>12} {str(r['gen_tok_s']):>9} "
          f"{r['inference_read_gb_dev']:>12} {str(r['mb_read_per_gen_token']):>9} "
          f"{r['memory_peak_gib_workload']:>9} {str(r['memory_events_max_workload']):>10}")
PYEOF
