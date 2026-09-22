"""First real generation measurement for this project.

Every published number so far is prompt-prefill throughput for a scorer that
emits nothing.  This probe decodes autoregressively — one token-forward per
generated token, greedy, through the same streaming engine under test — and
reports tok/s with the storage reads attributed to it.

Greedy because it needs no sampler to be comparable across runs; the point is
how many tokens this hardware can push per second from SSD, not sample quality.

    python -m probes.probe_decode --gguf <path> --ntok 64
"""

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "patch" / "files" / "src"))

import numpy  # noqa: E402

from semif_phase1 import llamacpp_backend  # noqa: E402

PROMPT = (
    "Write a Python function that returns the nth Fibonacci number, "
    "iteratively, with a docstring.\n\ndef fib(n):\n"
)


def io_counters() -> dict:
    counters = {}
    try:
        with open("/proc/self/io") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                counters[key.strip()] = int(value.strip())
    except OSError:
        pass
    return counters


def peak_rss_mib() -> int:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/scribe/models/Qwen3.6-35B-A3B-tokenizer")
    parser.add_argument("--revision", default="local")
    parser.add_argument(
        "--gguf",
        default="/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/"
        "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
    )
    parser.add_argument("--ntok", type=int, default=64)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    report: dict = {"prompt": PROMPT, "ntok_requested": args.ntok}

    load_started = time.perf_counter()
    backend, _tokenizer, metadata = llamacpp_backend.load_model(
        args.model, args.revision, args.gguf,
        threads=args.threads, context_tokens=args.ctx,
    )
    report["load_seconds"] = time.perf_counter() - load_started
    report["metadata"] = metadata

    engine = backend.engine
    vocab = engine.lib.llama_model_get_vocab(engine.model)
    prompt_tokens = llamacpp_backend._gguf_tokenize(engine.lib, vocab, PROMPT)
    report["prompt_tokens"] = len(prompt_tokens)
    report["threads"] = args.threads
    report["ctx"] = engine.context_tokens

    # --- prefill: fills the KV, returns the first next-token logits ---
    engine.clear()
    before = io_counters()
    prefill_started = time.perf_counter()
    logits = engine._decode(prompt_tokens, 0, 0, True)
    report["prefill_seconds"] = time.perf_counter() - prefill_started
    report["prefill_read_bytes"] = io_counters().get("read_bytes", 0) - before.get("read_bytes", 0)

    # --- decode: one token-forward per generated token ---
    generated: list[int] = []
    step_seconds: list[float] = []
    decode_read_before = io_counters().get("read_bytes", 0)
    decode_started = time.perf_counter()
    position = len(prompt_tokens)
    for _ in range(args.ntok):
        token = int(numpy.argmax(logits))
        generated.append(token)
        if token == 0:  # never observed on this vocab; a hard stop guard
            break
        step = time.perf_counter()
        logits = engine._decode([token], position, 0, True)
        step_seconds.append(time.perf_counter() - step)
        position += 1
        if position + 1 >= engine.context_tokens:
            break
    decode_seconds = time.perf_counter() - decode_started
    decode_read_bytes = io_counters().get("read_bytes", 0) - decode_read_before

    report.update({
        "generated_tokens": len(generated),
        "decode_seconds": decode_seconds,
        "decode_read_bytes": decode_read_bytes,
        "decode_gb": decode_read_bytes / 1e9,
        "tok_per_s": len(generated) / decode_seconds if decode_seconds else None,
        "mb_read_per_token": (decode_read_bytes / len(generated) / 1e6) if generated else None,
        "step_seconds": step_seconds,
        "step_seconds_median": float(numpy.median(step_seconds)) if step_seconds else None,
        "step_seconds_min": min(step_seconds) if step_seconds else None,
        "peak_rss_mib": peak_rss_mib(),
    })

    text = b"".join(llamacpp_backend._gguf_piece(engine.lib, vocab, token) for token in generated)
    report["generated_text"] = text.decode("utf-8", errors="replace")

    engine.close()

    print(json.dumps(report, indent=2, default=str))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
