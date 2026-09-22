"""Fit the SSD read model from resident-scorer ledger lines.

The question this answers: at a fixed 8 GiB cap, what does one scoring pass
actually read, and what part of it is avoidable?

Every call reports ``call_read_bytes`` from ``/proc/self/io``, and the pass
reports how many tokens it forwarded. ``batched_tokens`` -- not
``true_suffix_tokens`` -- is the right denominator: ``branch_logits_batched``
gives every branch the *whole* prompt as its own sequence, so a 4-criterion call
forwards 908 tokens even though only 368 of them are suffix.

Two things are then worth knowing, and they are different claims:

1. ``read_bytes / batched_tokens`` -- how many bytes each forwarded token costs.
   If this were constant, read would be purely proportional and a bigger batch
   would save nothing per criterion. It is not constant: the marginal cost per
   token *falls* as the batch grows, because a batch shares experts.

2. The fixed part per pass. Fitting ``read = a + k * tokens`` separates a term
   that every pass pays regardless of batch size from a term that scales. The
   fixed term is the one batching amortizes, and the one SEQ_MAX >= batch width
   avoids paying twice.

Neither number is an estimate of what a better engine could reach. For that,
compare against the routed-expert set (18.33 GB, from the GGUF tensor table) and
against the same workload at a larger cap -- see README's memory-cap table.

Usage:  analyze_sweep.py results/sweep-*/run.log [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERT_SET_BYTES = 18.33e9  # routed experts only; scripts/probes/gguf_tensor_layout.py


def ledger(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Collect ledger records from run.log lines and from call*.jsonl rows.

    Both forms are accepted on purpose. ``run.log`` is gitignored (``*.log``), so
    a fresh clone only has the ``call*.jsonl`` files -- and those are the better
    evidence anyway, because the resident scorer embeds the whole call report in
    every result row, which makes them self-describing.

    That embedding is also why this dedupes: one call writes N rows, each holding
    an identical copy of the report, so a four-row call would otherwise be counted
    four times.
    """
    loads, calls = [], []
    seen = set()
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            for key, sink in (("resident_load", loads), ("resident_call", calls)):
                if f'"{key}": ' not in line:
                    continue
                try:
                    record = json.loads(line)[key]
                except (json.JSONDecodeError, KeyError):
                    continue
                identity = (record.get("call_index"), record.get("rows"),
                            record.get("wall_seconds"), record.get("call_read_bytes"))
                if key == "resident_call":
                    if identity in seen:
                        continue
                    seen.add(identity)
                sink.append(record)
    return loads, calls


def fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares a + k*x. Returns None if the x values do not spread."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denominator = n * sxx - sx * sx
    if denominator == 0:
        return None
    k = (n * sxy - sx * sy) / denominator
    return (sy - k * sx) / n, k


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        # run.log carries the load lines and is gitignored; call*.jsonl is what a
        # fresh clone has. Both, when both exist -- the dedupe handles the overlap.
        paths = sorted(Path("results").glob("sweep-*/run.log")) + \
            sorted(Path("results").glob("sweep-*/call*.jsonl"))
    if not paths:
        print("no run.log or call*.jsonl given or found under results/sweep-*/")
        return 1

    loads, calls = ledger(paths)
    print(f"{len(paths)} log(s)   {len(loads)} loads   {len(calls)} calls\n")

    for load in loads:
        print(f"LOAD  {load['load_seconds']:.1f} s   "
              f"read {load['process_read_bytes'] / 1e9:.2f} GB   "
              f"rss {load['rss_mib']} MiB   seq_max {load.get('n_seq_max')}   "
              f"ubatch {load.get('n_ubatch')}   {load.get('load_mode_name')}")

    print(f"\n{'#':>2}  {'rows':>4}  {'passes':>6}  {'tokens':>7}  {'wall s':>7}  "
          f"{'read GB':>8}  {'GB/tok':>7}  {'x experts':>9}  {'s/crit':>7}")
    points = []
    for call in calls:
        timing = call.get("shared_timing") or {}
        tokens = timing.get("batched_tokens")
        read = call["call_read_bytes"]
        rows = call["rows"]
        if not tokens:
            continue
        points.append((float(tokens), float(read)))
        print(f"{call['call_index']:>2}  {rows:>4}  {timing.get('batched_passes'):>6}  "
              f"{tokens:>7}  {call['wall_seconds']:>7.2f}  {read / 1e9:>8.2f}  "
              f"{read / tokens / 1e6:>7.1f}  {read / EXPERT_SET_BYTES:>9.2f}  "
              f"{call['wall_seconds'] / rows:>7.2f}")

    print(f"\nper-criterion read: ", end="")
    print("  ".join(f"{c['rows']}crit={c['call_read_bytes'] / 1e9 / c['rows']:.1f}GB"
                    for c in calls if c["rows"]))

    result = fit(points)
    if result:
        fixed, per_token = result
        print(f"\nleast-squares read = {fixed / 1e9:.2f} GB + {per_token / 1e6:.1f} MB x token")
        print(f"  fixed term      {fixed / EXPERT_SET_BYTES:.2f}x the routed-expert set")
        print(f"  marginal cost   {per_token / 1e6:.1f} MB/token, falling as the "
              f"batch grows")
    else:
        print("\nnot enough distinct token counts to fit a line")

    # Bytes over time, which is the whole reason read_bytes is worth measuring:
    # if throughput is flat then time is a read counter, and cutting bytes is the
    # only way to cut time.
    rates = [c["call_read_bytes"] / 1e9 / c["wall_seconds"] for c in calls if c["wall_seconds"]]
    if rates:
        print(f"\nthroughput {min(rates):.2f}-{max(rates):.2f} GB/s "
              f"(mean {sum(rates) / len(rates):.2f}) -- flat means time tracks bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
