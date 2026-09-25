#!/usr/bin/env python3
"""Build a profile-guided cold-expert sidecar for native Q4_K streaming.

Hot experts and non-Q4_K tensors stay in the original GGUF. This file reduces
read traffic, not total installed disk space or expanded expert-cache memory.
"""
import argparse
import concurrent.futures
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probes'))
from gguf_tensor_layout import parse

HEADER = struct.Struct('<8sQQQ')
ENTRY = struct.Struct('<QQQQQ')
MAGIC = b'JEVTIER1'

def fingerprint(path):
    size = path.stat().st_size
    h = 14695981039346656037
    with path.open('rb') as f:
        for offset, count in [(0, min(size, 65536))] + [
                ((size - 4096) * i // 15, 4096) for i in range(16) if size >= 4096]:
            f.seek(offset)
            data = f.read(count)
            if len(data) != count:
                raise ValueError('short fingerprint read')
            for x in data:
                h = ((h ^ x) * 1099511628211) & ((1 << 64) - 1)
    return h

def load_profile(paths, n_expert, layers, coverage, model=None):
    counts = {l: [0] * n_expert for l in layers}
    for path in paths:
        if model is not None:
            with path.open() as f:
                model_line=next((line for line in f if line.startswith('# model=')),None)
            if model_line is None or Path(model_line[8:].split(' arch=',1)[0]).resolve()!=model.resolve():
                raise ValueError(f'trace source does not match model: {path}')
        with path.open() as f:
            for row in csv.DictReader(line for line in f if not line.startswith('#')):
                l, e = int(row['layer']), int(row['expert'])
                if l not in counts or not 0 <= e < n_expert:
                    raise ValueError(f'trace geometry mismatch: {l}, {e}')
                if int(row.get('dropped', 0)):
                    raise ValueError('profile must come from an unmodified, non-dropping baseline')
                counts[l][e] += 1
    hot, report = {}, {}
    for l, freq in counts.items():
        total = sum(freq)
        if not total:
            raise ValueError(f'no routing observations for layer {l}')
        chosen, covered = [], 0
        for e in sorted(range(n_expert), key=lambda e: (-freq[e], e)):
            chosen.append(e); covered += freq[e]
            if covered >= coverage * total:
                break
        hot[l] = set(chosen)
        report[l] = dict(hot=chosen, counts=freq, observed_coverage=covered/total)
    return hot, report

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--trace', type=Path, action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--codec', type=Path, required=True)
    ap.add_argument('--coverage', type=float, default=.9)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if not 0 < args.coverage <= 1 or not 1 <= args.workers <= 16:
        ap.error('coverage must be in (0,1]; workers in [1,16]')
    _, _, base, tensors = parse(args.model)
    import re
    experts=[]
    for t in tensors:
        m=re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', t['name'])
        if m:
            if len(t['dims']) != 3 or t['bytes'] % t['dims'][2]:
                raise ValueError('unsupported tensor geometry')
            experts.append((int(m[1]), t))
    if not experts:
        raise ValueError('no split expert tensors')
    counts={t['dims'][2] for _,t in experts}
    if len(counts)!=1: raise ValueError('inconsistent expert counts')
    hot, profile = load_profile(args.trace, counts.pop(), {l for l,_ in experts}, args.coverage,args.model)
    jobs=[]
    for l,t in experts:
        # The original Q6_K down projections remain higher precision.
        if t['qtype'] != 12: continue
        size=t['bytes']//t['dims'][2]
        if size%144: raise ValueError('invalid Q4_K extent')
        for e in range(t['dims'][2]):
            if e not in hot[l]:
                jobs.append((base+t['offset']+e*size,size,l,e,t['name']))
    jobs.sort()
    lib=ctypes.CDLL(str(args.codec.resolve()))
    lib.jev_pack.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    lib.jev_pack.restype=ctypes.c_int
    args.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=args.output.with_suffix(args.output.suffix+'.json')
    if args.output.exists() or manifest.exists():
        raise FileExistsError('output or manifest already exists; choose a new path')
    started=time.monotonic()
    fd=os.open(args.model,os.O_RDONLY)
    def encode(job):
        off,size,*_=job
        raw=os.pread(fd,size,off)
        if len(raw)!=size: raise ValueError('short source read')
        buf=ctypes.create_string_buffer(size//144*81)
        if not lib.jev_pack(raw,size,buf): raise ValueError('codec rejected extent')
        return buf.raw
    sha=hashlib.sha256()
    with args.model.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): sha.update(chunk)
    entries=[]
    # Keep only one small batch of encoded slices in memory.
    try:
        with args.output.open('xb') as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            f.write(HEADER.pack(MAGIC,args.model.stat().st_size,fingerprint(args.model),len(jobs)))
            f.write(bytes(ENTRY.size*len(jobs)))
            for start in range(0,len(jobs),args.workers*2):
                group=jobs[start:start+args.workers*2]
                for job,data in zip(group,pool.map(encode,group)):
                    pos=(f.tell()+4095)//4096*4096
                    f.write(bytes(pos-f.tell()))
                    entries.append((job[0],pos,job[1],len(data),1))
                    f.write(data)
                if start % 1000 < args.workers*2:
                    print(f'{min(start+len(group),len(jobs))}/{len(jobs)} slices',flush=True)
            end=(f.tell()+4095)//4096*4096
            f.write(bytes(end-f.tell()))
            f.seek(HEADER.size)
            for entry in entries: f.write(ENTRY.pack(*entry))
    finally:
        os.close(fd)
    raw=sum(x[2] for x in entries); packed=sum(x[3] for x in entries)
    report=dict(format='JEVTIER1',model=str(args.model.resolve()),source_sha256=sha.hexdigest(),
                source_bytes=args.model.stat().st_size,coverage=args.coverage,
                traces=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.trace],
                cold_slices=len(entries),cold_raw_bytes=raw,cold_stored_bytes=packed,
                logical_expert_bytes_before=sum(t['bytes'] for _,t in experts),
                logical_expert_bytes_after=sum(t['bytes'] for _,t in experts)-raw+packed,
                sidecar_bytes=args.output.stat().st_size,build_seconds=time.monotonic()-started,
                lossy=True,expanded_cache_unchanged=True,profile=profile)
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='profile'},indent=2))

if __name__=='__main__': main()
