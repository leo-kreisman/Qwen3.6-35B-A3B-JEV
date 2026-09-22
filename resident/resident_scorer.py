"""Resident JEV decision scorer: load the checkpoint once, serve many calls.

``semif_phase1.cli`` is one-shot by design -- ``load -> score -> exit`` -- so the
20.9 GB checkpoint is read from disk again on *every* call. On this host that is
~19.8 s of the ~60 s per-call wall clock (33%), and because the process dies
between calls nothing carries over: call 2 and call 3 read the same 61.6 GB as
call 1. That is the one cost in the whole stack that is pure overhead rather
than physics.

This module keeps the process alive instead. It reuses SemIf's *verified* path
unchanged -- the reference-tokenizer prompt render, the GGUF/HF agreement check
that guards ``prompt_sha256``, the shared-state fan-out, and the option-slot
readout -- and pays the load once. Nothing about the arithmetic changes, so
option scores are expected to be bit-identical to the CLI; anything else would
be a bug, not a trade.

What it deliberately does *not* do is touch the weight-access path. The decode
still goes through llama.cpp's mmap, so read amplification (41 GB against an
18.33 GB routed-expert set) is untouched. Fixing that needs `pread` over exact
expert extents with an engine-owned slot pool, which means a llama.cpp buffer
type; this wheel ships prebuilt ``.so`` files, so that is a rebuild-and-patch
project, not a flag. Kept separate on purpose: residency is measurable today
and lossless, the weight rewrite is neither yet.

Two ways in:

    # measurement: one call per fixture, in one process
    resident_scorer.py --gguf ... --model TOK fixture1.jsonl fixture2.jsonl

    # serving: newline-delimited JSON requests on stdin, one response per line
    resident_scorer.py --gguf ... --model TOK --serve
    {"call_id": "a", "rows": [ {"id": ..., "state": ..., "question": ..., "options": [...]} ]}

Every call reports its own ``read_bytes`` delta from ``/proc/self/io``, which is
the counter that matters here: it includes page-fault-driven reads, so it is the
honest measure of what mmap streaming actually pulled from the device. Wall
clock is not -- a warm run can beat a disk run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _io_counters() -> dict:
    """Read this process's I/O accounting; empty dict if unavailable."""
    counters = {}
    try:
        with open("/proc/self/io", encoding="ascii") as handle:
            for line in handle:
                key, _, value = line.partition(": ")
                if value:
                    counters[key] = int(value)
    except OSError:
        pass
    return counters


def _rss_kib() -> int:
    """Resident set size in KiB from /proc/self/status."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _load_fixture(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


class ResidentScorer:
    """One loaded checkpoint, many scoring calls."""

    def __init__(self, args) -> None:
        from semif_phase1 import llamacpp_backend

        self.backend = llamacpp_backend
        self.max_tokens = args.max_tokens
        self.calls = 0
        started = time.perf_counter()
        self.model, self.tokenizer, self.metadata = llamacpp_backend.load_model(
            args.model, args.revision, args.gguf,
            threads=args.llama_threads, context_tokens=args.max_tokens,
        )
        self.load_seconds = time.perf_counter() - started

    def call(self, rows: list[dict]) -> tuple[list[dict], dict]:
        """Score one shared-state batch and return (results, call report)."""
        if not rows:
            raise ValueError("Refusing to score an empty request")
        before = _io_counters()
        rss_before = _rss_kib()
        wall = time.perf_counter()
        results, timing = self.backend.score_shared(
            self.model, self.tokenizer, rows, self.metadata, self.max_tokens
        )
        wall = time.perf_counter() - wall
        after = _io_counters()
        self.calls += 1
        report = {
            "call_index": self.calls,
            "rows": len(rows),
            "wall_seconds": wall,
            "seconds_per_criterion": wall / len(rows),
            # Deltas, not totals: the point of a resident process is that the
            # totals stop growing at the same rate. read_bytes is cumulative for
            # the process, so a per-call number only exists as a difference.
            "call_read_bytes": after.get("read_bytes", 0) - before.get("read_bytes", 0),
            "call_rchar": after.get("rchar", 0) - before.get("rchar", 0),
            "process_read_bytes": after.get("read_bytes", 0),
            "rss_mib_before": rss_before // 1024,
            "rss_mib_after": _rss_kib() // 1024,
            "load_seconds": self.load_seconds,
            "shared_timing": timing,
        }
        return results, report


def _emit(report: dict) -> None:
    """One machine-readable ledger line per call, on stderr."""
    print(json.dumps({"resident_call": report}, allow_nan=False), file=sys.stderr, flush=True)


def _serve(scorer: ResidentScorer, destination: Path | None) -> int:
    """Serve newline-delimited JSON requests from stdin until EOF."""
    stream = destination.open("x") if destination else None
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            results, report = scorer.call(request["rows"])
            report["call_id"] = request.get("call_id")
            payload = {"call_id": request.get("call_id"), "results": results, "resident": report}
            print(json.dumps(payload, allow_nan=False), flush=True)
            if stream:
                stream.write(json.dumps(payload, allow_nan=False) + "\n")
                stream.flush()
            _emit(report)
    finally:
        if stream:
            stream.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Reference tokenizer directory")
    parser.add_argument("--revision", default="local")
    parser.add_argument("--llama-threads", type=int, help="default: 4/3 of nproc, the measured knee")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, help="write one result file per call")
    parser.add_argument("--repeat", type=int, default=1, help="run the fixture list this many times")
    parser.add_argument("--serve", action="store_true", help="read requests from stdin instead")
    parser.add_argument("--serve-output", type=Path, help="also append served responses here")
    parser.add_argument("fixtures", nargs="*", type=Path, help="input JSONL fixtures, one call each")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if not args.serve and not args.fixtures:
        parser.error("give at least one fixture, or --serve")

    scorer = ResidentScorer(args)
    print(json.dumps({
        "resident_load": {
            "load_seconds": scorer.load_seconds,
            "process_read_bytes": _io_counters().get("read_bytes", 0),
            "rss_mib": _rss_kib() // 1024,
            "n_seq_max": scorer.metadata.get("n_seq_max"),
            "n_ubatch": scorer.metadata.get("n_ubatch"),
            "load_mode_name": scorer.metadata.get("load_mode_name"),
            "threads": scorer.metadata.get("threads"),
        }
    }, allow_nan=False), file=sys.stderr, flush=True)

    if args.serve:
        return _serve(scorer, args.serve_output)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for repeat in range(args.repeat):
        for fixture in args.fixtures:
            rows = _load_fixture(fixture)
            results, report = scorer.call(rows)
            report["fixture"] = fixture.name
            report["repeat"] = repeat
            reports.append(report)
            if args.output_dir:
                destination = args.output_dir / f"call{report['call_index']:02d}.{fixture.stem}.jsonl"
                with destination.open("x") as handle:
                    for result in results:
                        handle.write(json.dumps({**result, "resident_call": report}, allow_nan=False) + "\n")
            _emit(report)

    summary = {
        "calls": len(reports),
        "load_seconds": scorer.load_seconds,
        "total_seconds": scorer.load_seconds + sum(r["wall_seconds"] for r in reports),
        "total_read_bytes": _io_counters().get("read_bytes", 0),
        "mean_call_seconds": sum(r["wall_seconds"] for r in reports) / len(reports),
    }
    print(json.dumps({"resident_summary": summary}, allow_nan=False), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
