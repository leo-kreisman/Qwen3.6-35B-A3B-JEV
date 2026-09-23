#!/usr/bin/env python3
"""Repack the routed-expert tensors of a Qwen3.6-35B-A3B GGUF into one
contiguous, 16 KiB-aligned slab per (layer, expert).

Why
---
In the original GGUF the three tensors that make up one routed expert
(``ffn_gate_exps``, ``ffn_up_exps``, ``ffn_down_exps``) live in three separate
file regions roughly 305 MB apart.  Reading one routed expert is therefore
three scattered ``pread``s instead of one contiguous read, and the request count
is 3x higher than the routed set requires.  Measured on this device at exactly
this extent size (589,824 B): 3,384 MiB/s with io_uring + O_DIRECT at qd=8
versus 1,658 MiB/s with sync pread.

What this changes
-----------------
Only the physical layout.  Every byte is copied verbatim -- there is no
dequantisation, no requantisation, and no reordering *within* a row.
Quantization formats, routing decisions and model computation are untouched by
construction, because this tool never parses a weight.  The losslessness claim
is not an argument, it is checkable: ``--verify`` reads slabs back and compares
them against the source slices byte for byte.

Container format
----------------
Prior art for this layout (docs/ASSEMBLED-DESIGN.md section 1):

    qwen-fieldfare  Q4EXP02   4 KiB header, uniform 1,769,472 B stride, 16 KiB align
    flash-moe       fixed 7,077,888 B stride, 9 components at fixed offsets
    apus            gate_up + down packed into a 6,291,456 B slab
    slipstream      PGRN v1, PGRN_ALIGN=16384
    siphon.cpp      16 KiB-aligned sidecar manifest

Where our checkpoint differs from that prior art: it is **not** stride-uniform.
117 of the 120 routed-expert tensors are Q4_K with a 589,824 B per-expert row,
but ``blk.34/38/39.ffn_down_exps`` are Q6_K with an 860,160 B row.  A single
global stride cannot express that, so the header carries an explicit
**per-layer table** (stride and the three component offsets) plus the cumulative
base offset of each layer.  Slabs stay uniform *within* a layer, which is all a
loader needs.

    [ 16 KiB header ] [ blk.0 slabs ] [ blk.1 slabs ] ... [ blk.39 slabs ]
    slab = gate_row(e) || up_row(e) || down_row(e) [ || zero pad to 16 KiB ]

    slab_offset(layer, expert) = header + layer_base[layer] + expert * stride[layer]

Sidecars written beside the pack:
    <out>.manifest.json        parameters, per-layer geometry, padding declared
    <out>.source_regions.tsv   per-layer source offsets, for the read benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import struct
import sys
import time
from pathlib import Path

# The GGUF tensor-table parser is shared with the measurement probes, so the
# table this tool acts on is the same table those probes reported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
from gguf_tensor_layout import parse as gguf_parse  # noqa: E402

HEADER_MAGIC = b"JEVXPACK"
HEADER_VERSION = 2
HEADER_BYTES = 16384
ALIGN = 16384
COMPONENTS = ("gate", "up", "down")
LAYER_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")

# header: magic, version, align, n_layers, n_experts, n_components, reserved,
#         src_size, body_bytes, layer_table_off, layer_base_off
HEADER_FIXED = struct.Struct("<8sIIIIIIQQQQ")
LAYER_ENTRY = struct.Struct("<4Q")  # stride, gate_off, up_off, down_off

DEFAULT_SRC = (Path.home() / "models/unsloth/Qwen3.6-35B-A3B-GGUF"
               / "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")
DEFAULT_OUT = Path.home() / "models/jev-pack/qwen3.6-35b-a3b-expertpack.bin"


def align_up(value, multiple=ALIGN):
    return value + (-value) % multiple


def discover(tensors):
    """Group the routed-expert tensors by layer: {layer: {component: tensor}}.

    Raises if a layer is incomplete: a partial layer would produce a pack that
    silently mis-addresses experts, which is far worse than failing here.
    """
    found = {}
    for tex in tensors:
        match = LAYER_RE.match(tex["name"])
        if match:
            found.setdefault(int(match.group(1)), {})[match.group(2)] = tex
    if not found:
        raise SystemExit("no routed-expert tensors matched ffn_{gate,up,down}_exps")
    for layer, parts in sorted(found.items()):
        missing = [c for c in COMPONENTS if c not in parts]
        if missing:
            raise SystemExit(f"blk.{layer}: missing component(s) {missing}")
    return found


def plan(found):
    """Compute per-layer slab geometry and validate what must hold.

    Requirements, checked rather than assumed:
      * every layer carries all three components (discover)
      * all layers agree on the expert count
      * within a layer, each component divides evenly into per-expert rows
    A stride is NOT required to be uniform across layers -- this checkpoint has
    Q6_K ``ffn_down_exps`` in three layers, which is why the header carries a
    per-layer table instead of one global stride.
    """
    layers = sorted(found)
    counts = {found[layers[0]][c]["dims"][-1] for c in COMPONENTS}
    if len(counts) != 1:
        raise SystemExit(f"expert counts disagree across components: {counts}")
    n_experts = counts.pop()

    per_layer = {}
    for layer in layers:
        rows = {}
        for component in COMPONENTS:
            tex = found[layer][component]
            if tex["dims"][-1] != n_experts:
                raise SystemExit(f"{tex['name']}: expert count {tex['dims'][-1]} "
                                 f"!= {n_experts}")
            if tex["bytes"] % n_experts:
                raise SystemExit(f"{tex['name']}: {tex['bytes']} B not divisible by "
                                 f"{n_experts} experts; rows are not uniform within "
                                 f"the tensor")
            rows[component] = tex["bytes"] // n_experts
        offsets, cursor = {}, 0
        for component in COMPONENTS:
            offsets[component] = cursor
            cursor += rows[component]
        stride = align_up(cursor)
        per_layer[layer] = {
            "rows": rows, "component_offset": offsets,
            "content_bytes": cursor, "stride": stride, "pad_bytes": stride - cursor,
            "types": {c: found[layer][c]["type"] for c in COMPONENTS},
        }

    base, cursor = {}, 0
    for layer in layers:
        base[layer] = cursor
        cursor += n_experts * per_layer[layer]["stride"]

    classes = {}
    for layer in layers:
        key = (per_layer[layer]["stride"],
               tuple(per_layer[layer]["rows"][c] for c in COMPONENTS),
               tuple(per_layer[layer]["types"][c] for c in COMPONENTS))
        classes.setdefault(key, []).append(layer)

    return {
        "layers": layers, "n_layers": len(layers), "n_experts": n_experts,
        "per_layer": per_layer, "layer_base": base,
        "body_bytes": cursor, "total_bytes": HEADER_BYTES + cursor,
        "classes": classes,
        "pad_bytes": sum(n_experts * per_layer[l]["pad_bytes"] for l in layers),
    }


def header_bytes(geom, src_size):
    """Fixed 16 KiB header: a loader reads 16 KiB and knows every offset in the
    container without consulting the manifest."""
    table_off = HEADER_FIXED.size
    base_off = table_off + geom["n_layers"] * LAYER_ENTRY.size
    blob = HEADER_FIXED.pack(
        HEADER_MAGIC, HEADER_VERSION, ALIGN, geom["n_layers"], geom["n_experts"],
        len(COMPONENTS), 0, src_size, geom["body_bytes"], table_off, base_off,
    )
    for layer in geom["layers"]:
        info = geom["per_layer"][layer]
        blob += LAYER_ENTRY.pack(
            info["stride"], info["component_offset"]["gate"],
            info["component_offset"]["up"], info["component_offset"]["down"])
    blob += b"".join(struct.pack("<Q", geom["layer_base"][l]) for l in geom["layers"])
    if len(blob) > HEADER_BYTES:
        raise SystemExit(f"header overflow: {len(blob)} > {HEADER_BYTES}")
    return blob + b"\0" * (HEADER_BYTES - len(blob))


def read_header(path):
    """Parse the container header back.  Used by --verify and by the benchmark,
    so the format has exactly one reader."""
    with open(path, "rb") as handle:
        head = handle.read(HEADER_BYTES)
    if len(head) < HEADER_FIXED.size:
        raise SystemExit(f"{path}: too short to contain a header")
    (magic, version, align, n_layers, n_experts, n_comp, _res,
     src_size, body_bytes, table_off, base_off) = HEADER_FIXED.unpack(
        head[:HEADER_FIXED.size])
    if magic != HEADER_MAGIC:
        raise SystemExit(f"{path}: bad magic {magic!r}, expected {HEADER_MAGIC!r}")
    if version != HEADER_VERSION:
        raise SystemExit(f"{path}: container version {version}, expected {HEADER_VERSION}")
    if align != ALIGN:
        raise SystemExit(f"{path}: alignment {align}, expected {ALIGN}")
    per_layer = {}
    for i in range(n_layers):
        off = table_off + i * LAYER_ENTRY.size
        stride, gate, up, down = LAYER_ENTRY.unpack(head[off:off + LAYER_ENTRY.size])
        per_layer[i] = {"stride": stride,
                        "component_offset": {"gate": gate, "up": up, "down": down}}
    base = {}
    for i in range(n_layers):
        off = base_off + i * 8
        (base[i],) = struct.unpack("<Q", head[off:off + 8])
    return {"n_layers": n_layers, "n_experts": n_experts, "n_components": n_comp,
            "src_size": src_size, "body_bytes": body_bytes,
            "per_layer": per_layer, "layer_base": base}


def build(src, out, data_start, found, geom, group):
    """Stream the source into the pack.

    Reads are batched: for `group` consecutive experts we issue one large read
    per component (group * row_bytes) rather than one per expert, which keeps
    the repack itself from becoming a scatter-gather benchmark.
    """
    n_experts = geom["n_experts"]
    src_fd = os.open(src, os.O_RDONLY)
    out_fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.pwrite(out_fd, header_bytes(geom, os.fstat(src_fd).st_size), 0)
        for layer in geom["layers"]:
            info = geom["per_layer"][layer]
            rows, offsets = info["rows"], info["component_offset"]
            stride = info["stride"]
            base = {c: data_start + found[layer][c]["offset"] for c in COMPONENTS}
            for start in range(0, n_experts, group):
                count = min(group, n_experts - start)
                blobs = {}
                for component in COMPONENTS:
                    span = count * rows[component]
                    blob = os.pread(src_fd, span, base[component] + start * rows[component])
                    if len(blob) != span:
                        raise SystemExit(f"short read: blk.{layer} {component} "
                                         f"wanted {span} got {len(blob)}")
                    blobs[component] = blob
                buf = bytearray(count * stride)  # zero-filled, so padding is explicit
                for i in range(count):
                    for component in COMPONENTS:
                        row = rows[component]
                        dst = i * stride + offsets[component]
                        buf[dst:dst + row] = blobs[component][i * row:(i + 1) * row]
                slab0 = geom["layer_base"][layer] // stride + start
                os.pwrite(out_fd, buf, HEADER_BYTES + geom["layer_base"][layer]
                          + start * stride)
            print(f"  blk.{layer:<2} written ({geom['layers'].index(layer) + 1}"
                  f"/{geom['n_layers']}) stride {stride:,}"
                  f"{' (+%d pad)' % info['pad_bytes'] if info['pad_bytes'] else ''}",
                  flush=True)
    finally:
        os.close(src_fd)
        os.close(out_fd)


def verify(src, out, data_start, found, geom, samples, seed=1234):
    """Prove losslessness by reading slabs back and comparing to the source.

    This is the point of the tool, so it compares real bytes rather than
    trusting the write path: random slabs are read from the pack and from the
    three source regions, and any difference is reported with its byte offset.
    """
    head = read_header(out)
    src_fd, out_fd = os.open(src, os.O_RDONLY), os.open(out, os.O_RDONLY)
    rng = random.Random(seed)
    bad = 0
    checked = 0
    try:
        for _ in range(samples):
            layer = rng.choice(geom["layers"])
            expert = rng.randrange(geom["n_experts"])
            info = geom["per_layer"][layer]
            stride = info["stride"]
            offset = HEADER_BYTES + geom["layer_base"][layer] + expert * stride
            slab = os.pread(out_fd, stride, offset)
            if len(slab) != stride:
                raise SystemExit(f"short slab read at {offset}")
            for component in COMPONENTS:
                row = info["rows"][component]
                tex = found[layer][component]
                want = os.pread(src_fd, row, data_start + tex["offset"] + expert * row)
                start = info["component_offset"][component]
                got = slab[start:start + row]
                checked += 1
                if got != want:
                    bad += 1
                    where = next((i for i in range(row) if got[i] != want[i]), -1)
                    print(f"  MISMATCH blk.{layer} expert {expert} {component} "
                          f"first differing byte at +{where}")
        # the header's own view must agree with the geometry we built from
        for layer in geom["layers"]:
            if head["per_layer"][layer]["stride"] != geom["per_layer"][layer]["stride"]:
                raise SystemExit(f"header stride disagrees for blk.{layer}")
            if head["layer_base"][layer] != geom["layer_base"][layer]:
                raise SystemExit(f"header base disagrees for blk.{layer}")
        if head["n_layers"] != geom["n_layers"] or head["n_experts"] != geom["n_experts"]:
            raise SystemExit("header dimension mismatch")
    finally:
        os.close(src_fd)
        os.close(out_fd)
    if bad:
        raise SystemExit(f"VERIFY FAILED: {bad}/{checked} component(s) mismatch")
    print(f"  VERIFY OK: {samples} random slabs, {checked} components "
          f"bit-identical to source; header geometry round-trips")


def write_sidecars(out, src, geom, data_start, found, src_size, elapsed):
    manifest = {
        "container": HEADER_MAGIC.decode(), "version": HEADER_VERSION,
        "alignment": ALIGN, "header_bytes": HEADER_BYTES,
        "source": str(src), "source_bytes": src_size,
        "layers": geom["n_layers"], "experts_per_layer": geom["n_experts"],
        "slabs": geom["n_layers"] * geom["n_experts"],
        "components": list(COMPONENTS),
        "body_bytes": geom["body_bytes"], "total_bytes": geom["total_bytes"],
        "padding_bytes": geom["pad_bytes"],
        "padding_note": ("slabs are padded to a 16 KiB multiple with zeros; padding "
                         "carries no content and is not read as weights"),
        "stride_uniform_across_layers": len(geom["classes"]) == 1,
        "distinct_layer_geometries": [
            {"stride": k[0], "component_rows": dict(zip(COMPONENTS, k[1])),
             "component_types": dict(zip(COMPONENTS, k[2])), "layers": v}
            for k, v in sorted(geom["classes"].items())
        ],
        "build_seconds": round(elapsed, 1),
        "slab_offset_formula": ("header_bytes + layer_base[layer] + expert * "
                               "stride[layer]"),
        "lossless": ("bytes copied verbatim; no dequantisation, no requantisation, "
                     "no reordering within a row"),
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = ["# layer\tgate_off\tup_off\tdown_off\tgate_row\tup_row\tdown_row\tstride",
             "# absolute byte offsets in the source; rows are per-expert"]
    for layer in geom["layers"]:
        info = geom["per_layer"][layer]
        offs = [data_start + found[layer][c]["offset"] for c in COMPONENTS]
        lines.append("\t".join(str(v) for v in (
            layer, *offs, *(info["rows"][c] for c in COMPONENTS), info["stride"])))
    Path(str(out) + ".source_regions.tsv").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the geometry and free space, write nothing")
    parser.add_argument("--verify", action="store_true",
                        help="read slabs back and compare against the source")
    parser.add_argument("--verify-samples", type=int, default=200)
    parser.add_argument("--verify-only", action="store_true",
                        help="skip the build, verify an existing pack")
    parser.add_argument("--group", type=int, default=128,
                        help="experts per batched read/write")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")
    print(f"source      {args.src}")
    version, alignment, data_start, tensors = gguf_parse(args.src)
    src_size = args.src.stat().st_size
    print(f"gguf        version {version}, alignment {alignment}, data at "
          f"{data_start}, {len(tensors)} tensors, {src_size:,} B")

    found = discover(tensors)
    geom = plan(found)
    print(f"layout      {geom['n_layers']} layers x {geom['n_experts']} experts, "
          f"{len(geom['classes'])} distinct layer geometries")
    for key, layers in sorted(geom["classes"].items()):
        stride, rows, types = key
        print(f"  stride {stride:,} B ({stride // ALIGN} x {ALIGN})  "
              f"rows {[f'{c}={rows[i]:,}' for i, c in enumerate(COMPONENTS)]}  "
              f"types {[f'{c}={types[i]}' for i, c in enumerate(COMPONENTS)]}")
        print(f"    layers ({len(layers)}): {layers if len(layers) <= 6 else str(layers[:6])[:-1] + ', ...]'}")
    print(f"container   {geom['total_bytes']:,} B ({geom['total_bytes'] / 1e9:.3f} GB) "
          f"vs source {src_size / 1e9:.3f} GB; padding {geom['pad_bytes'] / 1e6:.1f} MB "
          f"({geom['pad_bytes'] / geom['total_bytes']:.3%})")

    if args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(args.out.parent).free
        print(f"target      {args.out}")
        print(f"free        {free / 1e9:.1f} GB "
              f"({'OK' if free > geom['total_bytes'] else 'INSUFFICIENT'})")
        return 0

    if not args.verify_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print("building...")
        started = time.monotonic()
        build(args.src, args.out, data_start, found, geom, args.group)
        elapsed = time.monotonic() - started
        write_sidecars(args.out, args.src, geom, data_start, found, src_size, elapsed)
        wrote = args.out.stat().st_size
        if wrote != geom["total_bytes"]:
            raise SystemExit(f"size mismatch: wrote {wrote}, expected {geom['total_bytes']}")
        print(f"written     {args.out} ({wrote:,} B) in {elapsed:.1f} s "
              f"({geom['body_bytes'] / elapsed / 1e9:.2f} GB/s)")

    if args.verify:
        print("verifying...")
        verify(args.src, args.out, data_start, found, geom, args.verify_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
