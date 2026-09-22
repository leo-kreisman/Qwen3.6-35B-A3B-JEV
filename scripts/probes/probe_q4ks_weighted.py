#!/usr/bin/env python3
"""Full-file-weighted estimate of stream compressibility across ALL expert tensors.

The first probe sampled only blk.0/blk.1 experts, which turned out to be the most
compressible layers in the file (payload ratio 1.11 at layer 0 vs 1.03 at layer 18+).
This samples EVERY quantized expert tensor with a small per-tensor budget and
weights each stream's ratio by that tensor's real byte size, so the headline
number is file-weighted rather than head-weighted.

Read-only. Prints a JSON summary.
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
    ap.add_argument("--kb-per-tensor", type=float, default=384.0)
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    r = Reader(a.gguf)
    r.read_metadata()
    tensors = r.read_tensors()
    exps = [t for t in tensors if "exps" in t["name"] and t["type"] in BLOCK_BYTES]
    exps.sort(key=lambda t: t["name"])

    # real byte size per stream, file-wide
    def bytes_of(t):
        nw = int(np.prod(t["dims"]))
        return nw // QK_K * BLOCK_BYTES[t["type"]]

    tot = {"fp16_d_dmin": 0, "packed6": 0, "payload": 0}
    # compressed-size accumulators, computed at the sampled ratio for each tensor
    comp = {"fp16_d_dmin": 0.0, "packed6": 0.0, "payload": 0.0}
    ent_num = {"fp16_d_dmin": 0.0, "packed6": 0.0, "payload": 0.0}
    per_kind = {}
    rows = []

    for t in exps:
        bb = BLOCK_BYTES[t["type"]]
        max_blocks = max(1, int(a.kb_per_tensor * 1000 // bb))
        d, p6, pl, nb, nbt, bb = r.tensor_streams(t, max_blocks)
        tb = bytes_of(t)
        nw = int(np.prod(t["dims"]))
        # sizes of the three streams in the whole tensor
        nblk = nw // QK_K
        sizes = {"fp16_d_dmin": nblk * 4, "packed6": nblk * 12, "payload": nblk * (bb - 16)}
        kind = t["name"].split(".")[2]
        per_kind.setdefault(kind, {"raw": 0, "comp": 0.0})
        row = {"tensor": t["name"], "sampled_blocks": int(nb)}
        for key, s in (("fp16_d_dmin", d), ("packed6", p6), ("payload", pl)):
            raw = s.tobytes()
            z = zstd_size(raw, a.zstd_level)
            ratio = s.size / z
            e = entropy_bytes(s)
            tot[key] += sizes[key]
            comp[key] += sizes[key] / ratio
            ent_num[key] += e * sizes[key]
            per_kind[kind]["raw"] += sizes[key]
            per_kind[kind]["comp"] += sizes[key] / ratio
            row[key] = {"zstd_ratio": round(ratio, 4), "ent_bits": round(e, 4)}
        rows.append(row)

    total_raw = sum(tot.values())
    out = {
        "gguf": a.gguf,
        "zstd_level": a.zstd_level,
        "kb_per_tensor": a.kb_per_tensor,
        "n_expert_tensors": len(exps),
        "expert_bytes_GB": round(total_raw / 1e9, 3),
        "streams": {},
        "by_kind": {},
        "rows": rows,
    }
    for k in tot:
        out["streams"][k] = {
            "raw_bytes": tot[k],
            "size_weighted_pct_of_file": round(100.0 * tot[k] / total_raw, 3),
            "size_weighted_entropy_bits_per_byte": round(ent_num[k] / tot[k], 4),
            "size_weighted_zstd_ratio": round(tot[k] / comp[k], 4),
            "size_weighted_zstd_pct_of_raw": round(100.0 * comp[k] / tot[k], 2),
            "bytes_saved_GB": round((tot[k] - comp[k]) / 1e9, 3),
        }
    for k, v in per_kind.items():
        out["by_kind"][k] = {
            "raw_bytes": v["raw"],
            "zstd_pct_of_raw": round(100.0 * v["comp"] / v["raw"], 2),
            "zstd_ratio": round(v["raw"] / v["comp"], 4),
        }
    tot_comp = sum(comp.values())
    out["whole_routed_set_zstd_pct_of_raw"] = round(100.0 * tot_comp / total_raw, 2)
    out["whole_routed_set_bytes_saved_GB"] = round((total_raw - tot_comp) / 1e9, 3)

    print(json.dumps(out, indent=2))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
