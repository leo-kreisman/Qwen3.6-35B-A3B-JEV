#!/usr/bin/env bash
# Sequential resident sweep: one process per (SEQ_MAX, LOAD_MODE) pair.
#
# SEQ_MAX and LOAD_MODE are model/context-creation parameters, so they are fixed
# for the life of a llama.cpp context and cannot be swept inside one process.
# The runs are also strictly serial on purpose: two capped runs at once contend
# for device bandwidth, and the whole measurement is bytes-read-over-time, so
# overlapping them would corrupt both.
#
# What each run answers:
#   token-scale  SEQ_MAX=8   n1 -> n4 -> n8   one process, three calls
#       Is per-pass read a fixed expert sweep (-> ~24 GB even at 253 tokens) or
#       proportional to tokens routed (-> ~7 GB)? n4 and n8 alone cannot tell,
#       because any two points fit a line. This is the discriminating run.
#   n16-s16      SEQ_MAX=16  n16              the batching win
#       16 rows past n_seq_max=8 is two full model sweeps; at SEQ_MAX=16 it is
#       one. Expected saving is exactly one sweep's worth of bytes.
#   direct-io    LOAD_MODE=direct_io
#       O_DIRECT is present in this build but has never been measured here.
#
# Usage:  ./scripts/sweep_resident.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
STAMP="$(date +%Y%m%d-%H%M%S)"

run() {  # run <tag> <seq_max> <load_mode> <fixture...>
  local tag="$1" seq="$2" mode="$3"
  shift 3
  echo
  echo "############ $tag   seq_max=$seq   load_mode=$mode"
  OUTDIR="$REPO_ROOT/results/sweep-$STAMP-$tag" \
  REPEAT=1 SEQ_MAX="$seq" LOAD_MODE="$mode" \
    ./run_resident_35b_ssd.sh --simulate-8g "$@"
}

run token-scale 8  mmap \
  examples/decisions.one.jsonl \
  examples/decisions.sample.jsonl \
  examples/decisions.n8.jsonl

run n16-s16 16 mmap examples/decisions.n16.jsonl
run direct-io 8 direct_io examples/decisions.sample.jsonl

echo
echo "############ sweep complete: $STAMP"
