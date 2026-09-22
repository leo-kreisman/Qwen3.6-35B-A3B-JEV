#!/usr/bin/env bash
# How much of our SSD traffic during PREFILL is redundant re-reads of the same
# expert, and how much of it does row-grouping remove?
#
# THE DISEASE (SSD-LLaMA, arXiv:2609.18110 section 4.3): during prefill, many
# prompt tokens are processed together, but they are distributed unevenly among
# the routed experts. If a forward pass is split into several ubatches, the same
# expert's weights get pulled off the disk once per ubatch that touches it. The
# fix is to group all rows assigned to an expert and load that expert once.
#
# In llama.cpp there is no cross-ubatch expert scheduling yet, so the only lever
# available today is the ubatch size itself: one ubatch spanning the whole prompt
# is the degenerate case where every expert IS loaded once. So this probe measures
# the same thing from the other end -- what the redundant reads cost, by watching
# storage traffic fall as the pass is split into fewer ubatches.
#
#   ub = 512, prompt 2048  -> 4 ubatches  -> up to 4x redundant expert reads
#   ub = 1024              -> 2 ubatches
#   ub = 2048              -> 1 ubatch    -> every expert read exactly once
#
# METHOD (inherited from probe_ncmoe_capped.sh, which exists because of a trap):
#   * systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- a real cap.
#   * posix_fadvise(DONTNEED) on the GGUF before EVERY configuration. A cgroup
#     does not re-charge an already-cached page, so without eviction the cap never
#     binds and the run measures the page cache instead of the disk.
#   * TWO counters per run, deltas over exactly that run:
#       /proc/self/io read_bytes   -- this shell + reaped children
#       /sys/block/<dev>/stat      -- whole device, nobody excluded
#     read_bytes counts only reads that reached storage, so a page-cache hit
#     contributes nothing.
#   * a LOAD-ONLY baseline per configuration (`-p 1 -n 1`), because the model load
#     itself reads the whole checkpoint; without it the workload's own storage
#     traffic is unattributable. inference_read = workload_read - baseline_read.
#   * TRAP: /proc/self/io read inside "$(...)" reads the *subshell*, which has read
#     nothing, so the whole sweep reports 0.00 GB and looks plausible. Counters are
#     read with shell builtins in this shell.
#
# Usage:
#   ./probe_readamp_capped.sh                      # capped 8G, default sweep
#   CAP= ./probe_readamp_capped.sh                 # warm run, no cap
#   UB_LIST="512 2048" PROMPTS=1024 ./probe_readamp_capped.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-/home/scribe/Projects/llama.cpp-fresh/build/bin/llama-bench}"
GGUF="${GGUF:-$HOME/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf}"
DEV="${DEV:-/sys/block/nvme0n1/stat}"
UB_LIST="${UB_LIST:-512 1024 2048}"
PROMPTS="${PROMPTS:-2048}"
NGEN="${NGEN:-1}"
CAP="${CAP-8G}"
THREADS="${THREADS:-6}"
REPS="${REPS:-1}"

OUTDIR="${OUTDIR:-$REPO_ROOT/results/probe-readamp-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
if [[ -n "${PROBE_ACTIVE:-}" ]]; then
  exec >> "$OUTDIR/run.log" 2>&1
else
  exec > >(tee -a "$OUTDIR/run.log") 2>&1
fi

PROC_BYTES=0
DEV_BYTES=0

snap() {
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
cg_peak() { awk -v b="$(cat "$(cgdir)/memory.peak" 2>/dev/null || echo 0)" 'BEGIN{printf "%.2f", b/1073741824}'; }
cg_max_events() { awk '/^max /{print $2}' "$(cgdir)/memory.events" 2>/dev/null || echo "?"; }

evict() {
  python3 - "$GGUF" <<'PYEOF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PYEOF
}

run_config() {
  local ub="$1"
  echo "--- ubatch=$ub (prompt $PROMPTS) ---"

  evict
  local b_proc b_dev a_proc a_dev
  snap b_proc b_dev
  "$BIN" -m "$GGUF" -ngl 0 -p 1 -n 1 -ub "$ub" -b "$PROMPTS" -t "$THREADS" -r 1 -o jsonl \
      > "$OUTDIR/.baseline.jsonl" 2>/dev/null || true
  snap a_proc a_dev
  local load_proc=$((a_proc-b_proc)) load_dev=$((a_dev-b_dev))
  local peak_load ev_load; peak_load="$(cg_peak)"; ev_load="$(cg_max_events)"

  evict
  snap b_proc b_dev
  "$BIN" -m "$GGUF" -ngl 0 -p "$PROMPTS" -n "$NGEN" -ub "$ub" -b "$PROMPTS" -t "$THREADS" -r "$REPS" -o jsonl \
      > "$OUTDIR/.workload.jsonl" 2>/dev/null || true
  snap a_proc a_dev
  local work_proc=$((a_proc-b_proc)) work_dev=$((a_dev-b_dev))
  local peak_work ev_work; peak_work="$(cg_peak)"; ev_work="$(cg_max_events)"

  python3 - "$ub" "$PROMPTS" "$OUTDIR/.workload.jsonl" "$load_proc" "$load_dev" \
             "$work_proc" "$work_dev" "$peak_load" "$peak_work" "$ev_load" "$ev_work" \
    <<'PYEOF'
import json, sys
(ub, prompts, path, load_proc, load_dev, work_proc, work_dev,
 peak_load, peak_work, ev_load, ev_work) = sys.argv[1:12]
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
own_dev = int(work_dev) - int(load_dev)
own_proc = int(work_proc) - int(load_proc)
n_prompt = prefill["n_prompt"] if prefill else int(prompts)
rec = {
    "ubatch": int(ub),
    "prompt_tokens": n_prompt,
    "ubatches_per_pass": round(n_prompt / int(ub), 2),
    "prefill_tok_s": prefill and round(prefill["avg_ts"], 2),
    "prefill_seconds": prefill and round(prefill["avg_ns"] / 1e9, 2),
    "gen_tok_s": gen and round(gen["avg_ts"], 2),
    "load_read_gb_proc": round(int(load_proc)/1e9, 2),
    "load_read_gb_dev": round(int(load_dev)/1e9, 2),
    "workload_read_gb_proc": round(int(work_proc)/1e9, 2),
    "workload_read_gb_dev": round(int(work_dev)/1e9, 2),
    "inference_read_gb_dev": round(own_dev/1e9, 2),
    "inference_read_gb_proc": round(own_proc/1e9, 2),
    "mb_read_per_prefill_token": round(own_dev / max(1, n_prompt) / 1e6, 1),
    "effective_gb_s": (prefill and own_dev and
                       round(own_dev / (prefill["avg_ns"]/1e9) / 1e9, 2)),
    "memory_peak_gib": float(peak_work),
    "memory_events_max": int(ev_work) if ev_work.isdigit() else ev_work,
    "memory_peak_gib_baseline": float(peak_load),
}
print(json.dumps({"config": rec}, allow_nan=False))
PYEOF
  echo "# ub=$ub load=$(gb $load_dev)GB inference=$(gb $((work_dev-load_dev)))GB peak=$(cg_peak)GiB max_events=$(cg_max_events)"
}

echo "==> bin      $BIN"
echo "==> gguf     $GGUF ($(du -h "$GGUF" | cut -f1))"
echo "==> ubatch   $UB_LIST   prompt $PROMPTS   threads $THREADS"
echo "==> cap      ${CAP:-none}"
echo "==> output   $OUTDIR"
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
echo '{"note":"device read deltas from /sys/block stat sectors*512 (whole device); proc deltas from /proc/self/io read_bytes (this shell + reaped children). inference = workload - load-only baseline."}' >> "$OUTDIR/configs.jsonl"
for ub in $UB_LIST; do
  run_config "$ub" >> "$OUTDIR/configs.jsonl"
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
hdr = f"{'ubatch':>7} {'ub/pass':>8} {'prefill t/s':>12} {'inf GB':>8} {'MB/tok':>8} {'GB/s':>7} {'peak GiB':>9} {'cap hits':>9}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['ubatch']:>7} {str(r['ubatches_per_pass']):>8} {str(r['prefill_tok_s']):>12} "
          f"{str(r['inference_read_gb_dev']):>8} {str(r['mb_read_per_prefill_token']):>8} "
          f"{str(r['effective_gb_s']):>7} {r['memory_peak_gib']:>9} {str(r['memory_events_max']):>9}")
if rows:
    best = max(rows, key=lambda r: r['prefill_tok_s'] or 0)
    worst = min(rows, key=lambda r: r['prefill_tok_s'] or 0)
    if worst['prefill_tok_s']:
        print(f"\nrow-grouping gain (ub {worst['ubatch']} -> {best['ubatch']}): "
              f"{best['prefill_tok_s']/worst['prefill_tok_s']:.2f}x prefill, "
              f"{worst['inference_read_gb_dev']} -> {best['inference_read_gb_dev']} GB inference read")
PYEOF
