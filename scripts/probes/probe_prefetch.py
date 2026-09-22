"""Layer-ordered readahead of expert tensors, to convert random faults into sequential reads.

Measured facts this attacks:
  - The GGUF is on an NVMe that does 2.4 GB/s with O_DIRECT sequential reads.
  - The decode only achieves 750 MB/s, because expert access arrives as small demand
    faults (read_ahead_kb is 128, fault-around is 64 KB). That is a 3.2x loss to
    access pattern, not to the device.
  - Suppressing readahead (MADV_RANDOM) makes it far worse (71 MB/s), so the fix is
    MORE sequential readahead, not less.

llama.cpp walks layers 0..39 in order within a pass, so a prefetcher that walks the
same layers in the same order never needs to know exactly where the decoder is: the
page-cache victim is always an already-consumed layer. The only real risk is outrunning
the decoder far enough to evict unconsumed layers, so the thread is byte-rate limited.

Run as:  SEMIF_PREFETCH_GBS=1.3 python probe_prefetch.py
"""
import ctypes
import json
import os
import struct
import sys
import threading
import time

sys.path.insert(0, "/home/scribe/Projects/JEV_experiment/semif/src")
from semif_phase1 import llamacpp_backend as L

GGUF = "/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_S.gguf"
if not os.path.exists(GGUF):
    GGUF = "/home/scribe/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
TOK = "/home/scribe/models/Qwen3.6-35B-A3B-tokenizer"
IN = "/home/scribe/Projects/JEV_experiment/decisions.sample.jsonl"
RATE_GBS = float(os.environ.get("SEMIF_PREFETCH_GBS", "1.3"))
CHUNK = int(os.environ.get("SEMIF_PREFETCH_CHUNK_MB", "16")) * 1024 * 1024

_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd():
    with open("/proc/self/io") as fh:
        for line in fh:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return -1


def gguf_layout(path):
    """Parse the GGUF tensor table -> (data_start, alignment, [(name, offset, nbytes)])."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {magic!r}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        alignment = 32
        for _ in range(n_kv):
            key = fh.read(struct.unpack("<Q", fh.read(8))[0]).decode("utf-8", "replace")
            kind = struct.unpack("<I", fh.read(4))[0]
            if kind == 8:  # string
                fh.read(struct.unpack("<Q", fh.read(8))[0])
            elif kind == 9:  # array
                elem = struct.unpack("<I", fh.read(4))[0]
                count = struct.unpack("<Q", fh.read(8))[0]
                if elem == 8:
                    for _ in range(count):
                        fh.read(struct.unpack("<Q", fh.read(8))[0])
                else:
                    fh.read(_SCALAR[elem] * count)
            else:
                fh.read(_SCALAR[kind])
            if key == "general.alignment":
                pass  # alignment is read from the value when it is an integer; rare
        tensors = []
        for _ in range(n_tensors):
            name = fh.read(struct.unpack("<Q", fh.read(8))[0]).decode("utf-8", "replace")
            n_dims = struct.unpack("<I", fh.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims))
            kind = struct.unpack("<I", fh.read(4))[0]
            offset = struct.unpack("<Q", fh.read(8))[0]
            tensors.append((name, offset, kind, dims))
        data_start = fh.tell()
        data_start = (data_start + alignment - 1) // alignment * alignment
    return data_start, alignment, tensors


def expert_extents(path):
    """Byte extents of the routed-expert tensors, grouped in layer order.

    Returns [(layer, [(abs_offset, nbytes), ...]), ...] sorted by layer. Uses the
    tensor sizes implied by the next tensor's offset, since GGUF stores offsets only.
    """
    data_start, _align, tensors = gguf_layout(path)
    by_offset = sorted(tensors, key=lambda item: item[1])
    size_of = {}
    for index, (name, offset, _kind, _dims) in enumerate(by_offset):
        if index + 1 < len(by_offset):
            size_of[name] = by_offset[index + 1][1] - offset
        else:
            size_of[name] = os.path.getsize(path) - data_start - offset
    layers = {}
    for name, offset, _kind, _dims in tensors:
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "blk" or "exps" not in name:
            continue
        try:
            layer = int(parts[1])
        except ValueError:
            continue
        layers.setdefault(layer, []).append((data_start + offset, size_of[name]))
    return sorted(layers.items())


class Prefetcher(threading.Thread):
    """Walk expert extents in layer order, warming the page cache at a capped rate.

    Two modes, because they are not equivalent. ``advise`` issues
    ``POSIX_FADV_WILLNEED``, which the kernel caps to a small readahead window — measured
    here as *no effect at all* (read_bytes moved by 0.14 GB across a whole run).
    ``read`` performs a real ``preadv`` into a reused buffer, which forces genuine
    sequential I/O at device speed; that is the q36 "prewarm pthread" pattern.
    """

    def __init__(self, path, extents, rate_bytes_per_s, mode="read"):
        super().__init__(daemon=True)
        self.path = path
        self.extents = extents
        self.rate = rate_bytes_per_s
        self.mode = mode
        self.issued = 0
        self.layers_done = 0
        self.stop = threading.Event()
        self.fd = None

    def run(self):
        self.fd = os.open(self.path, os.O_RDONLY)
        buffer = bytearray(CHUNK)
        view = memoryview(buffer)
        started = time.perf_counter()
        try:
            for layer, parts in self.extents:
                if self.stop.is_set():
                    return
                for offset, length in parts:
                    done = 0
                    while done < length:
                        if self.stop.is_set():
                            return
                        span = min(CHUNK, length - done)
                        try:
                            if self.mode == "read":
                                os.preadv(self.fd, [view[:span]], offset + done)
                            else:
                                os.posix_fadvise(
                                    self.fd, offset + done, span, os.POSIX_FADV_WILLNEED
                                )
                        except OSError:
                            pass
                        done += span
                        self.issued += span
                        # Rate limit on bytes issued, so the prewarmer cannot outrun
                        # the decoder far enough to evict layers it has not reached.
                        target = started + self.issued / self.rate
                        lag = target - time.perf_counter()
                        if lag > 0:
                            time.sleep(lag)
                self.layers_done += 1
        finally:
            os.close(self.fd)


rows = [json.loads(line) for line in open(IN)]
base = rd()
model, tokenizer, metadata = L.load_model(TOK, "local", GGUF, threads=6, context_tokens=8192)
after_load = rd()
print(f"LOAD: read_bytes = {after_load-base:,} ({(after_load-base)/1e9:.1f} GB)")

extents = expert_extents(GGUF)
total = sum(size for _layer, parts in extents for _off, size in parts)
print(f"expert tensors: {len(extents)} layers, {total/1e9:.1f} GB total")
print(f"prefetch rate = {RATE_GBS} GB/s, chunk = {CHUNK//1024//1024} MB")

prefetch = Prefetcher(GGUF, extents, RATE_GBS * 1e9,
                      os.environ.get("SEMIF_PREFETCH_MODE", "read"))
t0 = time.perf_counter()
prefetch.start()
results, timing = L.score_shared(model, tokenizer, rows, metadata, max_tokens=8192)
elapsed = time.perf_counter() - t0
prefetch.stop.set()
print(f"DECODE: {elapsed:.2f}s  read_bytes = {rd()-after_load:,} ({(rd()-after_load)/1e9:.1f} GB)")
print(f"prefetcher issued {prefetch.issued/1e9:.1f} GB across {prefetch.layers_done}/{len(extents)} layers")
print(f"TOTAL read_bytes = {rd()-base:,} ({(rd()-base)/1e9:.1f} GB)")
print("answers:", [(r["id"], round(max(r["probabilities"]), 6)) for r in results])
