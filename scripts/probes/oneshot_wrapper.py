"""Measure the true one-shot invocation path, including its own /proc/self/io.

Runs the shipped CLI in-process (runpy) so the process's block-device reads are still
readable after the CLI's sys.exit. This is what a code agent actually pays when it
shells out: argparse + tokenizer + GGUF digest + the full model load + one scoring pass.
"""
import runpy
import sys

sys.argv = ["semif_phase1.cli"] + sys.argv[1:]
try:
    runpy.run_module("semif_phase1.cli", run_name="__main__", alter_sys=True)
except SystemExit as exit_code:
    if exit_code.code not in (0, None):
        raise

with open("/proc/self/io") as fh:
    for line in fh:
        if line.startswith("read_bytes:"):
            print(f"READ_BYTES={line.split()[1].strip()}")
