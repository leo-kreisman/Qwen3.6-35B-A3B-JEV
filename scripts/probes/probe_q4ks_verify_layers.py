#!/usr/bin/env python3
"""Verify the q4_K_S stream-compressibility result is not a file-head artifact.

The main probe samples the FIRST N expert tensors, i.e. blk.0/blk.1. Early-layer
experts can be more skewed than late layers, so this re-samples the same three
streams from tensors spread across the whole file (first, middle, last layers)
and prints per-tensor ratios side by side. If the payload ratio holds across
layers the measurement is trustworthy; if it collapses, the headline number is
a layer-0 artifact and must not be quoted.
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/scribe/Projects/JEV_experiment/scripts/probes")
from probe_q4ks_streams import (BLOCK_BYTES, QK_K, Reader, entropy_bytes,
                                zstd_size, byte_plane)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--mb-per-tensor", type=float, default=2.0)
    ap.add_argument("--zstd-level", type=int, default=6)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    r = Reader(a.gguf)
    r.read_metadata()
    tensors = r.read_tensors()
    exps = sorted([t for t in tensors if "exps" in t["name"] and t["type"] in BLOCK_BYTES],
                  key=lambda t: t["name"])
    n = len(exps)
    print(f"# {n} quantized expert tensors", file=sys.stderr)
    # spread picks: first, 1/4, 1/2, 3/4, last
    picks = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    rows = []
    for i in picks:
        t = exps[i]
        bb = BLOCK_BYTES[t["type"]]
        max_blocks = int(a.mb_per_tensor * 1e6 // bb)
        d, p6, pl, nb, nbt, bb = r.tensor_streams(t, max_blocks)
        row = {"tensor": t["name"], "layer": t["name"].split(".")[1],
               "blocks_sampled": int(nb), "blocks_total": int(nbt)}
        for key, s in (("fp16_d_dmin", d), ("packed6", p6), ("payload", pl)):
            raw = s.tobytes()
            z = zstd_size(raw, a.zstd_level)
            e = entropy_bytes(s)
            row[key] = {"raw": int(s.size), "ent_bits": round(e, 4),
                        "entropy_floor_ratio": round(8.0 / e, 4) if e else None,
                        "zstd_ratio": round(s.size / z, 4),
                        "zstd_pct": round(100.0 * z / s.size, 2)}
        rows.append(row)
        print(f"# {t['name']}: payload zstd_ratio={row['payload']['zstd_ratio']}", file=sys.stderr)

    out = {"gguf": a.gguf, "zstd_level": a.zstd_level,
           "sample_mb_per_tensor": a.mb_per_tensor, "rows": rows}
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
