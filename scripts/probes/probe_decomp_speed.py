#!/usr/bin/env python3
"""Decompression-throughput probe on REAL q4_K_S expert bytes.

Question it answers: if the expert streams are stored compressed on NVMe and
decompressed in RAM before the kernel consumes them, does the decoder outrun
the device (~2.4 GB/s here)? If not, the byte saving is paid back in CPU time.

Reads a bounded slice straight out of the GGUF, then measures decompression
throughput of several container layouts (whole-buffer, independent 4 MiB
frames, block-parallel) plus per-stream splits.
"""
import argparse
import json
import mmap
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

QK_K = 256
BLOCK_BYTES = {12: 144, 13: 176, 14: 210}
GGUF_MAGIC = b"GGUF"


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench(fn, out_bytes, reps=5):
    fn()  # warm
    best = min(_timed(fn) for _ in range(reps))
    return best, out_bytes / best / 1e6  # MB of DECOMPRESSED output per second


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.buf = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", self.buf, 0)
        assert magic == GGUF_MAGIC, magic
        self.pos = 24
        self.n_tensors, self.n_kv = n_tensors, n_kv

    def _str(self):
        n, = struct.unpack_from("<Q", self.buf, self.pos); self.pos += 8
        s = self.buf[self.pos:self.pos + n].decode("utf-8", "replace"); self.pos += n
        return s

    def _skip(self, t):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sizes:
            self.pos += sizes[t]
        elif t == 8:
            self._str()
        elif t == 9:
            et, = struct.unpack_from("<I", self.buf, self.pos)
            n, = struct.unpack_from("<Q", self.buf, self.pos + 4); self.pos += 12
            for _ in range(n):
                self._skip(et)
        else:
            raise ValueError(t)

    def tensors(self):
        for _ in range(self.n_kv):
            self._str(); t, = struct.unpack_from("<I", self.buf, self.pos); self.pos += 4
            self._skip(t)
        out = []
        for _ in range(self.n_tensors):
            name = self._str()
            nd, = struct.unpack_from("<I", self.buf, self.pos); self.pos += 4
            dims = struct.unpack_from("<" + "Q" * nd, self.buf, self.pos); self.pos += 8 * nd
            ttype, = struct.unpack_from("<I", self.buf, self.pos); self.pos += 4
            off, = struct.unpack_from("<Q", self.buf, self.pos); self.pos += 8
            out.append({"name": name, "dims": dims, "type": ttype, "offset": off})
        self.data_start = (self.pos + 31) // 32 * 32
        return out

    def raw_slice(self, t, nbytes):
        off = self.data_start + t["offset"]
        return self.buf[off:off + nbytes]


def zstd_c(data, lvl):
    import zstandard as zstd
    return zstd.ZstdCompressor(level=lvl).compress(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--mb", type=float, default=48.0)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    import zstandard as zstd
    import lz4.block as lz4b

    r = Reader(a.gguf)
    ts = r.tensors()
    exps = sorted([t for t in ts if "exps" in t["name"] and t["type"] == 12],
                  key=lambda t: t["name"])
    print(f"# {len(exps)} q4_K expert tensors", file=sys.stderr)

    target = int(a.mb * 1e6)
    chunks, got = [], 0
    for t in exps:
        if got >= target:
            break
        want = min(target - got, 8 << 20)
        want = (want // 144) * 144
        chunks.append(bytes(r.raw_slice(t, want)))
        got += want
    raw = b"".join(chunks)
    n = len(raw)
    print(f"# sampled {n/1e6:.2f} MB of real q4_K expert bytes", file=sys.stderr)

    res = {"input_MB": round(n / 1e6, 2), "source": a.gguf}

    res["raw_bytearray_copy_MBs"] = round(bench(lambda: bytearray(raw), n)[1], 1)

    for lvl in (3, 9):
        comp = zstd_c(raw, lvl)
        csize = len(comp)
        dctx = zstd.ZstdDecompressor()
        b, mb = bench(lambda c=comp: dctx.decompress(c, max_output_size=n), n)
        res[f"zstd{lvl}_whole"] = {
            "compressed_pct_of_raw": round(100.0 * csize / n, 2),
            "decomp_MBs": round(mb, 1),
            "sec_per_212.6MB_out": round(212.6 / mb, 4),
        }

    CH = 4 << 20
    for lvl in (3, 9):
        frames = [zstd_c(raw[i:i + CH], lvl) for i in range(0, n, CH)]
        tot_c = sum(len(f) for f in frames)

        def seq(frames=frames):
            return [zstd.ZstdDecompressor().decompress(f, max_output_size=CH + 65536)
                    for f in frames]
        b, mb = bench(seq, n)
        res[f"zstd{lvl}_frames4M"] = {
            "compressed_pct_of_raw": round(100.0 * tot_c / n, 2),
            "decomp_MBs": round(mb, 1),
            "sec_per_212.6MB_out": round(212.6 / mb, 4),
        }
        for nt in (4, 8):
            def par(frames=frames, nt=nt):
                with ThreadPoolExecutor(max_workers=nt) as ex:
                    return list(ex.map(lambda f: zstd.ZstdDecompressor().decompress(
                        f, max_output_size=CH + 65536), frames))
            b, mb = bench(par, n, reps=3)
            res[f"zstd{lvl}_frames4M_par{nt}"] = {
                "compressed_pct_of_raw": round(100.0 * tot_c / n, 2),
                "decomp_MBs": round(mb, 1),
                "sec_per_212.6MB_out": round(212.6 / mb, 4),
            }

    frames4 = [lz4b.compress(raw[i:i + CH], store_size=False) for i in range(0, n, CH)]
    tot_c = sum(len(f) for f in frames4)
    outlens = [len(raw[i:i + CH]) for i in range(0, n, CH)]

    def lz4seq():
        return [lz4b.decompress(f, uncompressed_size=L)
                for f, L in zip(frames4, outlens)]
    b, mb = bench(lz4seq, n)
    res["lz4_frames4M"] = {
        "compressed_pct_of_raw": round(100.0 * tot_c / n, 2),
        "decomp_MBs": round(mb, 1),
        "sec_per_212.6MB_out": round(212.6 / mb, 4),
    }
    for nt in (4, 8):
        def par4(nt=nt):
            with ThreadPoolExecutor(max_workers=nt) as ex:
                return list(ex.map(
                    lambda p: lz4b.decompress(p[0], uncompressed_size=p[1]),
                    zip(frames4, outlens)))
        b, mb = bench(par4, n, reps=3)
        res[f"lz4_frames4M_par{nt}"] = {
            "compressed_pct_of_raw": round(100.0 * tot_c / n, 2),
            "decomp_MBs": round(mb, 1),
            "sec_per_212.6MB_out": round(212.6 / mb, 4),
        }

    # per-stream split
    m = (n // 144) * 144
    arr = np.frombuffer(raw, dtype=np.uint8)[:m].reshape(-1, 144)
    d = np.ascontiguousarray(arr[:, 0:4]).tobytes()
    p6 = np.ascontiguousarray(arr[:, 4:16]).tobytes()
    pl = np.ascontiguousarray(arr[:, 16:144]).tobytes()
    for tag, stream in (("fp16_d_dmin", d), ("packed6", p6), ("payload", pl)):
        comp = zstd_c(stream, 9)
        dctx = zstd.ZstdDecompressor()
        b, mb = bench(lambda c=comp, L=len(stream):
                      dctx.decompress(c, max_output_size=L), len(stream))
        res[f"stream_{tag}"] = {
            "share_pct_of_file": round(100.0 * len(stream) / m, 3),
            "compressed_pct_of_raw": round(100.0 * len(comp) / len(stream), 2),
            "decomp_MBs": round(mb, 1),
        }
    meta = d + p6
    comp = zstd_c(meta, 9)
    res["meta_streams_combined"] = {
        "share_pct_of_file": round(100.0 * len(meta) / m, 3),
        "compressed_pct_of_raw": round(100.0 * len(comp) / len(meta), 2),
        "file_size_reduction_if_payload_raw_pct": round(
            100.0 * (1 - (len(comp) + len(pl)) / m), 2),
    }
    print(json.dumps(res, indent=2))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
