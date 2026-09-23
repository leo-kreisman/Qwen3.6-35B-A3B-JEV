"""Dump GGUF tensor extents and look for duplicated expert storage.

Written to test one hypothesis: the 2.24x read amplification on the SSD path
(41 GB read against an 18.33 GB routed-expert set) may not be an access-pattern
problem at all. If the checkpoint stores both a fused ``ffn_gate_up_exps`` and
its separate ``ffn_gate_exps`` / ``ffn_up_exps`` components as *distinct file
extents*, then any engine reading the fused tensor and also the split tensors
reads expert bytes twice per layer, and 41 / 18.33 = 2.24 is explained without
invoking page-cache behaviour at all.

If instead the tensors overlap (same offsets), they are views of one extent, the
bytes are read once, and amplification has some other cause -- which is equally
useful to know, because it means a pread slot pool is the only remaining lever.

Reads only the header and tensor table (a few hundred KB), never the weights.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

# (bytes per block, elements per block) for the qtypes that appear in this file.
BLOCK = {
    0: (4, 1),      # F32
    1: (2, 1),      # F16
    2: (18, 32),    # Q4_0
    3: (20, 32),    # Q4_1
    8: (34, 32),    # Q8_0
    12: (144, 256),  # Q4_K
    13: (176, 256),  # Q5_K
    14: (210, 256),  # Q6_K
}
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K"}

_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
           5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
           12: ("<d", 8)}


def _read_exact(handle, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise EOFError(f"wanted {count} bytes, got {len(data)}")
    return data


def _string(handle) -> str:
    (length,) = struct.unpack("<Q", _read_exact(handle, 8))
    return _read_exact(handle, length).decode("utf-8", "replace")


def _value(handle, kind: int):
    if kind == 8:
        return _string(handle)
    if kind == 9:  # array
        (element_kind,) = struct.unpack("<I", _read_exact(handle, 4))
        (length,) = struct.unpack("<Q", _read_exact(handle, 8))
        if element_kind == 8:
            for _ in range(length):
                _string(handle)
            return f"<{length} strings>"
        fmt, size = _SCALAR[element_kind]
        _read_exact(handle, size * length)
        return f"<{length} values>"
    fmt, size = _SCALAR[kind]
    return struct.unpack(fmt, _read_exact(handle, size))[0]


def parse(path: Path):
    with path.open("rb") as handle:
        magic, version = struct.unpack("<II", _read_exact(handle, 8))
        if magic != 0x46554747:
            raise ValueError(f"{path} is not a GGUF file")
        tensor_count, kv_count = struct.unpack("<QQ", _read_exact(handle, 16))
        alignment = 32
        for _ in range(kv_count):
            key = _string(handle)
            (kind,) = struct.unpack("<I", _read_exact(handle, 4))
            value = _value(handle, kind)
            if key == "general.alignment":
                alignment = int(value)
        tensors = []
        for _ in range(tensor_count):
            name = _string(handle)
            (n_dims,) = struct.unpack("<I", _read_exact(handle, 4))
            dims = struct.unpack(f"<{n_dims}Q", _read_exact(handle, 8 * n_dims))
            (qtype,) = struct.unpack("<I", _read_exact(handle, 4))
            (offset,) = struct.unpack("<Q", _read_exact(handle, 8))
            elements = 1
            for dim in dims:
                elements *= dim
            if qtype not in BLOCK:
                raise ValueError(f"{name}: unhandled qtype {qtype}")
            per_block, block_elements = BLOCK[qtype]
            blocks = (elements + block_elements - 1) // block_elements
            tensors.append({
                "name": name, "qtype": qtype, "type": TYPE_NAMES[qtype],
                "elements": elements, "bytes": blocks * per_block, "offset": offset,
                "dims": dims,
            })
        data_start = handle.tell()
        data_start += (-data_start) % alignment
        return version, alignment, data_start, tensors


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                Path.home() / "models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")
    version, alignment, data_start, tensors = parse(path)
    file_size = path.stat().st_size
    print(f"file            {path.name}")
    print(f"version         {version}   alignment {alignment}")
    print(f"tensor table    {len(tensors)} tensors, data starts at {data_start}")
    total = sum(t["bytes"] for t in tensors)
    print(f"sum of extents  {total / 1e9:.2f} GB   file {file_size / 1e9:.2f} GB")
    print(f"unaccounted     {(file_size - data_start - total) / 1e6:.1f} MB")

    # "exps" and "shexp" are distinct substrings ("shexp" has no "exps"), so the
    # two filters do not overlap -- but name them apart anyway, because the
    # obvious `len(experts) - len(shared)` reads as a negative count.
    experts = [t for t in tensors if "exps" in t["name"] and "shexp" not in t["name"]]
    shared = [t for t in tensors if "shexp" in t["name"]]
    expert_bytes = sum(t["bytes"] for t in experts)
    print(f"\nrouted expert tensors  {len(experts)}   bytes {expert_bytes / 1e9:.2f} GB")
    print(f"shared expert tensors  {len(shared)}   bytes "
          f"{sum(t['bytes'] for t in shared) / 1e9:.2f} GB")

    names = sorted({t["name"].split(".", 2)[-1] for t in experts})
    print(f"distinct routed expert suffixes: {names}")

    # The hypothesis: does each layer's fused gate_up cover the same file range
    # as its separate gate/up, or a different one?
    print("\nper-layer extents (layer 0, 3 and 39):")
    for layer in (0, 3, 39):
        prefix = f"blk.{layer}."
        rows = [t for t in experts if t["name"].startswith(prefix)]
        span = 0
        for tensor in sorted(rows, key=lambda t: t["offset"]):
            start = data_start + tensor["offset"]
            end = start + tensor["bytes"]
            span += tensor["bytes"]
            print(f"  {tensor['name']:<34} {tensor['type']:<6} "
                  f"{start:>14,} .. {end:>14,}  {(end - start) / 1e6:>8.1f} MB")
        print(f"  {'layer total':<34} {'':<6} {'':>14}    {'':>14}  {span / 1e6:>8.1f} MB")

    # Overlap analysis over the whole file: are any two tensors' byte ranges
    # identical (views of one extent) or partially overlapping?
    ranges = []
    for tensor in tensors:
        start = data_start + tensor["offset"]
        ranges.append((start, start + tensor["bytes"], tensor["name"]))
    ranges.sort()
    exact = 0
    partial = 0
    for (s1, e1, n1), (s2, e2, n2) in zip(ranges, ranges[1:]):
        if s1 == s2 and e1 == e2:
            exact += 1
            if "exps" in n1 and "shexp" not in n1:
                print(f"  EXACT DUPLICATE RANGE: {n1} == {n2}")
        elif s2 < e1:
            partial += 1
            if "exps" in n1 or "exps" in n2:
                print(f"  PARTIAL OVERLAP: {n1} [{s1},{e1}) vs {n2} [{s2},{e2})")
    print(f"\nidentical ranges {exact}   partial overlaps {partial}")

    # If every routed-expert suffix is distinct storage, the sum of extents
    # accounts for the file; if some are views, the sum overshoots.
    print(f"\nexpert extents sum {expert_bytes / 1e9:.2f} GB vs "
          f"file {file_size / 1e9:.2f} GB "
          f"({expert_bytes / file_size:.2%} of the file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
