#!/usr/bin/env python3
"""Expand a sidecar into a separate GGUF for same-weights transport A/B tests."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from build import HEADER, ENTRY, MAGIC, fingerprint

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--pack',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--codec',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    manifest=json.loads(a.pack.with_suffix(a.pack.suffix+'.json').read_text())
    sha=hashlib.sha256()
    with a.model.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): sha.update(chunk)
    if sha.hexdigest()!=manifest['source_sha256']: raise ValueError('source SHA256 mismatch')
    lib=ctypes.CDLL(str(a.codec.resolve()))
    lib.jev_expand.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
    with a.pack.open('rb') as f:
        magic,size,fp,n=HEADER.unpack(f.read(HEADER.size))
        if magic!=MAGIC or size!=a.model.stat().st_size or fp!=fingerprint(a.model):
            raise ValueError('pack header mismatch')
        entries=[ENTRY.unpack(f.read(ENTRY.size)) for _ in range(n)]
        # Reflinks when supported; otherwise an independent file copy.
        subprocess.run(['cp','--reflink=auto','--',str(a.model),str(a.output)],check=True)
        with a.output.open('r+b') as out:
            for source,offset,raw,stored,codec in entries:
                if codec!=1 or raw%144 or stored!=raw//144*81 or source+raw>size:
                    raise ValueError('invalid pack entry')
                f.seek(offset); data=f.read(stored)
                if len(data)!=stored: raise ValueError('truncated pack')
                decoded=ctypes.create_string_buffer(raw)
                if not lib.jev_expand(data,stored,decoded,raw): raise ValueError('invalid block')
                out.seek(source); out.write(decoded.raw)
    print(json.dumps(dict(output=str(a.output),cold_slices=n,source_sha256=sha.hexdigest())))

if __name__=='__main__': main()
