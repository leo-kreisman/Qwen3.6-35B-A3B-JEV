#!/usr/bin/env python3
"""Measure the byte budget and compressibility of each stream inside a q4_K_S GGUF.

Per K-quant super-block (256 weights, 144 bytes for q4_K):
  4 bytes  : d (fp16) + dmin (fp16)          -> the "block scale/min" stream
  12 bytes : 8 x 6-bit scales + 8 x 6-bit mins (packed)  -> the "sub-scale" stream
 128 bytes : 256 x 4-bit payload (nibbles)   -> the "payload" stream

Reports, per stream: byte share of the file, order-0 entropy (=> best-case
lossless floor), zstd/lz4 ratio on a bounded sample, and the ratio after
byte-plane transposition (the bitshuffle/shuffle family of transforms).

Read-only, no model load. Prints a JSON summary.
"""
import argparse
import json
import math
import mmap
import struct
import sys
from collections import Counter

import numpy as np

QK_K = 256
BLOCK_BYTES = {12: 144, 13: 176, 14: 210}   # q4_K / q5_K / q6_K
GGUF_MAGIC = b"GGUF"


class Reader:
    """Minimal GGUF reader: header, KV metadata (walked + skipped), tensor table."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.buf = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", self.buf, 0)
        assert magic == GGUF_MAGIC, magic
        self.version, self.n_tensors, self.n_kv = ver, n_tensors, n_kv
        self.pos = 24

    def _str(self):
        n, = struct.unpack_from("<Q", self.buf, self.pos)
        self.pos += 8
        s = self.buf[self.pos:self.pos + n].decode("utf-8", "replace")
        self.pos += n
        return s

    def _skip_value(self, t):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sizes:
            self.pos += sizes[t]
        elif t == 8:
            self._str()
        elif t == 9:
            et, = struct.unpack_from("<I", self.buf, self.pos)
            n, = struct.unpack_from("<Q", self.buf, self.pos + 4)
            self.pos += 12
            for _ in range(n):
                self._skip_value(et)
        else:
            raise ValueError(f"unknown gguf value type {t}")

    def read_metadata(self):
        for _ in range(self.n_kv):
            self._str()
            t, = struct.unpack_from("<I", self.buf, self.pos)
            self.pos += 4
            self._skip_value(t)

    def read_tensors(self):
        tensors = []
        for _ in range(self.n_tensors):
            name = self._str()
            nd, = struct.unpack_from("<I", self.buf, self.pos)
            self.pos += 4
            dims = struct.unpack_from("<" + "Q" * nd, self.buf, self.pos)
            self.pos += 8 * nd
            ttype, = struct.unpack_from("<I", self.buf, self.pos)
            self.pos += 4
            offset, = struct.unpack_from("<Q", self.buf, self.pos)
            self.pos += 8
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        self.data_start = (self.pos + 31) // 32 * 32
        return tensors

    def tensor_streams(self, t, max_blocks):
        bb = BLOCK_BYTES[t["type"]]
        nweights = int(np.prod(t["dims"])) if t["dims"] else 0
        nblocks_total = nweights // QK_K
        nb = min(nblocks_total, max_blocks)
        off = self.data_start + t["offset"]
        raw = self.buf[off:off + nb * bb]
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(nb, bb)
        d = np.ascontiguousarray(arr[:, 0:4]).reshape(-1)
        packed6 = np.ascontiguousarray(arr[:, 4:16]).reshape(-1)
        payload = np.ascontiguousarray(arr[:, 16:bb]).reshape(-1)
        return d, packed6, payload, nb, nblocks_total, bb


def entropy_bytes(a):
    """Order-0 empirical entropy of a uint8 array, in bits/byte."""
    n = a.size
    if n == 0:
        return 0.0
    cnt = np.bincount(a, minlength=256)
    p = cnt[cnt > 0] / n
    return float(-(p * np.log2(p)).sum())


def zstd_size(data, level):
    import zstandard as zstd
    return len(zstd.ZstdCompressor(level=level).compress(data))


def lz4_size(data):
    import lz4.block as lz4b
    return len(lz4b.compress(data, store_size=False))


def byte_plane(a, elem_bytes):
    """Deinterleave into elem_bytes planes (byte-shuffle / bitshuffle-style)."""
    n = (a.size // elem_bytes) * elem_bytes
    m = a[:n].reshape(-1, elem_bytes)
    return np.ascontiguousarray(m.T).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tensors", type=int, default=6)
    ap.add_argument("--mb-per-tensor", type=float, default=5.0)
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    r = Reader(a.gguf)
    r.read_metadata()
    tensors = r.read_tensors()

    tc = Counter(t["type"] for t in tensors)
    print(f"# GGUF v{r.version} tensors={r.n_tensors} kv={r.n_kv}", file=sys.stderr)
    print(f"# tensor type histogram: {dict(tc)}", file=sys.stderr)

    all_bytes = {t["name"]: int(np.prod(t["dims"])) * (BLOCK_BYTES[t["type"]] / QK_K)
                 for t in tensors if t["type"] in BLOCK_BYTES and t["dims"]}
    exps = [t for t in tensors
            if "exps" in t["name"] and t["type"] in BLOCK_BYTES]
    exps.sort(key=lambda t: t["name"])
    exps_gb = sum(all_bytes[t["name"]] for t in exps) / 1e9
    other_gb = (sum(all_bytes.values()) - sum(all_bytes[t["name"]] for t in exps)) / 1e9
    print(f"# quantized expert tensors: {len(exps)}  expert bytes={exps_gb:.3f} GB  "
          f"non-expert quant bytes={other_gb:.3f} GB", file=sys.stderr)

    streams = ["fp16_d_dmin", "packed6_scales_mins", "q4_payload"]
    acc = {k: {"raw": 0, "zstd": 0, "lz4": 0, "zstd_plane": 0, "ent_bits": 0.0, "n": 0}
           for k in streams}
    measured = []

    for t in exps[: a.tensors]:
        bb = BLOCK_BYTES[t["type"]]
        max_blocks = int(a.mb_per_tensor * 1e6 // bb)
        d, p6, pl, nb, nbt, bb = r.tensor_streams(t, max_blocks)
        d_s = d.tobytes(); p6_s = p6.tobytes(); pl_s = pl.tobytes()

        for key, s in (("fp16_d_dmin", d), ("packed6_scales_mins", p6), ("q4_payload", pl)):
            e = acc[key]
            e["raw"] += s.size
            e["ent_bits"] += entropy_bytes(s) * s.size
            e["n"] += s.size
            e["zstd"] += zstd_size(s.tobytes(), a.zstd_level)
            e["lz4"] += lz4_size(s.tobytes())

        # byte-plane transposes: d/dmin is 4 bytes -> 4 planes; payload is 2 nibbles
        acc["fp16_d_dmin"]["zstd_plane"] += zstd_size(byte_plane(d, 4).tobytes(), a.zstd_level)
        acc["packed6_scales_mins"]["zstd_plane"] += zstd_size(byte_plane(p6, 2).tobytes(), a.zstd_level)
        # nibble split: high and low nibble planes of the 4-bit payload
        plr = pl.reshape(-1, 2)
        nl = np.ascontiguousarray(plr[:, 0]).tobytes()
        nh = np.ascontiguousarray(plr[:, 1]).tobytes()
        acc["q4_payload"]["zstd_plane"] += zstd_size(nl, a.zstd_level) + zstd_size(nh, a.zstd_level)

        measured.append({"tensor": t["name"], "type": t["type"],
                         "blocks_sampled": int(nb), "blocks_total": int(nbt)})
        print(f"#   {t['name']} type={t['type']} blocks {nb}/{nbt}", file=sys.stderr)

    total = sum(acc[k]["raw"] for k in streams)
    out = {
        "gguf": a.gguf,
        "gguf_version": r.version,
        "tensor_type_histogram": {str(k): v for k, v in tc.items()},
        "n_quantized_expert_tensors": len(exps),
        "expert_quant_bytes_GB": round(exps_gb, 3),
        "nonexpert_quant_bytes_GB": round(other_gb, 3),
        "zstd_level": a.zstd_level,
        "sample_mb_per_tensor": a.mb_per_tensor,
        "tensors_measured": measured,
        "bytes_sampled_total": total,
        "streams": {},
    }
    for k in streams:
        e = acc[k]
        if e["raw"] == 0:
            continue
        out["streams"][k] = {
            "raw_bytes": e["raw"],
            "pct_of_total": round(100.0 * e["raw"] / total, 3),
            "order0_entropy_bits_per_byte": round(e["ent_bits"] / e["n"], 4),
            "entropy_floor_ratio": round(8.0 / (e["ent_bits"] / e["n"]), 4),
            "zstd_ratio": round(e["raw"] / e["zstd"], 4),
            "zstd_pct_of_raw": round(100.0 * e["zstd"] / e["raw"], 2),
            "lz4_ratio": round(e["raw"] / e["lz4"], 4),
            "byte_plane_zstd_ratio": round(e["raw"] / e["zstd_plane"], 4),
            "byte_plane_zstd_pct_of_raw": round(100.0 * e["zstd_plane"] / e["raw"], 2),
        }

    tot_z = sum(acc[k]["zstd"] for k in streams)
    tot_p = sum(acc[k]["zstd_plane"] for k in streams)
    out["whole_sample_zstd_pct"] = round(100.0 * tot_z / total, 2)
    out["whole_sample_plane_zstd_pct"] = round(100.0 * tot_p / total, 2)

    print(json.dumps(out, indent=2))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
