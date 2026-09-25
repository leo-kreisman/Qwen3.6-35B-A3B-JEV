#!/usr/bin/env python3
"""Lossless gate/up/down bundles and CPU replay on captured, actual routed rows.

Numerical reference uses dequantized original GGUF weights, not GGML's quantized
activation kernel. This is a layer experiment, never an end-to-end speed claim.
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/probes'))
from gguf_tensor_layout import parse, BLOCK


def geometry(model, layer):
    _,_,base,tensors=parse(model)
    out={}
    for kind in ('gate','up','down'):
        t=next(t for t in tensors if t['name']==f'blk.{layer}.ffn_{kind}_exps.weight')
        if t['qtype'] not in (12,13,14): raise ValueError('only K-block quantized experts supported')
        out[kind]={**t,'absolute':base+t['offset']}
    d,m,e=out['gate']['dims']
    if out['up']['dims']!=(d,m,e) or out['down']['dims']!=(m,d,e):
        raise ValueError('unsupported gated expert geometry')
    return d,m,e,out


def slice_tile(raw, dims, qtype, start, width, down=False):
    columns,rows=dims
    block_bytes,block_values=BLOCK[qtype]
    if columns%block_values or start%block_values or width%block_values:
        raise ValueError('tile cuts quantization blocks')
    if down:
        return np.frombuffer(raw,dtype=np.uint8).reshape(rows,columns//block_values,block_bytes)[:,start//block_values:(start+width)//block_values,:].copy().tobytes()
    rowbytes=columns//block_values*block_bytes
    return raw[start*rowbytes:(start+width)*rowbytes]


def build(a):
    d,m,e,ts=geometry(a.model,a.layer)
    if a.tile<256 or a.tile%256 or m%a.tile: raise ValueError('tile must divide intermediate width and align to 256')
    a.output.mkdir(exist_ok=False)
    manifest=dict(format='jev-executable-tiles-v1',model=str(a.model.resolve()),layer=a.layer,
                  hidden=d,intermediate=m,experts=e,tile=a.tile,tensors=ts,blocks=[])
    digest=hashlib.sha256()
    with a.model.open('rb') as src,(a.output/'weights.bin').open('xb') as dst:
        for expert in range(e):
            raw={}
            for kind,t in ts.items():
                size=t['bytes']//e
                src.seek(t['absolute']+expert*size);raw[kind]=src.read(size)
                if len(raw[kind])!=size:raise ValueError('short GGUF expert')
                digest.update(raw[kind])
            for start in range(0,m,a.tile):
                offset=(dst.tell()+4095)//4096*4096
                dst.write(bytes(offset-dst.tell())); lengths=[]
                for kind in ('gate','up','down'):
                    t=ts[kind]
                    b=slice_tile(raw[kind],t['dims'][:2],t['qtype'],start,a.tile,kind=='down')
                    dst.write(b);lengths.append(len(b))
                manifest['blocks'].append(dict(expert=expert,start=start,offset=offset,lengths=lengths))
        dst.write(bytes((-dst.tell())%4096))
    manifest['source_layer_payload_sha256']=digest.hexdigest()
    manifest['stored_bytes']=(a.output/'weights.bin').stat().st_size
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('tensors','blocks')}))


class Decoder:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path.resolve()))
        self.fn=self.lib.resume_dequant
        self.fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p]
        self.fn.restype=ctypes.c_int

    def decode(self,raw,kind,shape):
        out=np.empty(shape,dtype=np.float32)
        if not self.fn(kind,raw,out.size,out.ctypes.data):raise ValueError('decoder rejected block')
        return out


class DirectReader:
    """Bounded page-aligned bounce reader; every read bypasses file page cache."""
    def __init__(self,path):
        self.fd=os.open(path,os.O_RDONLY|os.O_DIRECT)
        self.buffer=None;self.capacity=0;self.position=0;self.physical_bytes=0

    def __enter__(self):return self

    def __exit__(self,*args):
        if self.buffer is not None:self.buffer.close()
        os.close(self.fd)

    def seek(self,offset):self.position=offset

    def read(self,size):
        start=self.position//4096*4096;delta=self.position-start
        count=(delta+size+4095)//4096*4096
        if count>self.capacity:
            if self.buffer is not None:self.buffer.close()
            self.buffer=mmap.mmap(-1,count);self.capacity=count
        view=memoryview(self.buffer)[:count]
        try:n=os.preadv(self.fd,[view],start)
        finally:view.release()
        if n<delta+size:raise ValueError('short direct read')
        self.physical_bytes+=n;self.position+=size
        return self.buffer[delta:delta+size]


def load_groups(trace,layer,phase):
    groups=[]
    for line in trace.read_text().splitlines():
        r=json.loads(line)
        if r['layer']!=layer or (phase and r['phase']!=phase):continue
        d,one,n,last=r['input_shape']
        if one!=1 or last!=1:raise ValueError('expected common expert input per token')
        x=np.fromfile(trace.parent/r['input_file'],dtype=np.float32).reshape(n,d)
        ids=np.array(r['ids'],dtype=np.int32).reshape(n,-1)
        if len(set(ids[0]))!=ids.shape[1] or any(len(set(v))!=ids.shape[1] for v in ids):
            raise ValueError('duplicate expert in token route')
        groups.append((x,ids))
    if not groups:raise ValueError('no captured groups')
    return groups


def silu(x):
    # Stable sigmoid: never overflow on a large negative gate.
    sigmoid=np.empty_like(x);pos=x>=0
    sigmoid[pos]=1/(1+np.exp(-x[pos])); ex=np.exp(x[~pos]);sigmoid[~pos]=ex/(1+ex)
    return x*sigmoid


def replay(a):
    manifest=json.loads((a.pack/'manifest.json').read_text())
    layer=manifest['layer'];d=manifest['hidden'];m=manifest['intermediate'];e=manifest['experts']
    groups=load_groups(a.trace,layer,a.phase)
    if a.schedule=='grouped':groups=[(np.concatenate([x for x,_ in groups]),np.concatenate([ids for _,ids in groups]))]
    dec=Decoder(a.decoder);ts=manifest['tensors']
    if str(a.model.resolve())!=manifest['model']:raise ValueError('source path mismatch')
    index={i:[] for i in range(e)}
    for b in manifest['blocks']:index[b['expert']].append(b)
    arrays=[];logical=0;calls=0;weights_peak=0;started=time.monotonic()
    # Direct reads prevent this isolated single layer fitting in the OS cache
    # from hiding the rereads observed between full-model sweeps in the trace.
    with DirectReader(a.model) as src,DirectReader(a.pack/'weights.bin') as pack:
        for x,ids in groups:
            result=np.zeros((len(x),ids.shape[1],d),dtype=np.float32)
            for expert in np.unique(ids):
                rows,slots=np.where(ids==expert);selected=x[rows]
                if a.representation=='original':
                    weights={}
                    for kind,t in ts.items():
                        size=t['bytes']//e;src.seek(t['absolute']+int(expert)*size);raw=src.read(size)
                        if len(raw)!=size:raise ValueError('short original read')
                        logical+=size;calls+=1
                        shape=(d,m) if kind=='down' else (m,d)
                        weights[kind]=dec.decode(raw,t['qtype'],shape)
                    weights_peak=max(weights_peak,sum(w.nbytes for w in weights.values()))
                    h=silu(selected@weights['gate'].T)*(selected@weights['up'].T)
                    result[rows,slots]=h@weights['down'].T
                else:
                    acc=np.zeros((len(rows),d),dtype=np.float32)
                    for b in index[int(expert)]:
                        count=sum(b['lengths']);pack.seek(b['offset']);raw=pack.read(count)
                        if len(raw)!=count:raise ValueError('short bundle')
                        logical+=count;calls+=1;weights={};offset=0
                        for kind,size in zip(('gate','up','down'),b['lengths']):
                            shape=(d,manifest['tile']) if kind=='down' else (manifest['tile'],d)
                            weights[kind]=dec.decode(raw[offset:offset+size],ts[kind]['qtype'],shape);offset+=size
                        weights_peak=max(weights_peak,sum(w.nbytes for w in weights.values()))
                        h=silu(selected@weights['gate'].T)*(selected@weights['up'].T)
                        acc+=h@weights['down'].T
                    result[rows,slots]=acc
            arrays.append(result)
        physical=src.physical_bytes+pack.physical_bytes
        io_buffers=src.capacity+pack.capacity
    elapsed=time.monotonic()-started
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # The same token order is retained across grouping so direct comparisons work.
    with a.output.with_suffix('.npy').open('xb') as f:np.save(f,np.concatenate(arrays))
    result=dict(schedule=a.schedule,representation=a.representation,seconds=elapsed,
                logical_read_bytes=logical,read_calls=calls,decoded_weight_bytes_peak=weights_peak,
                direct_read_bytes=physical,io_buffer_bytes=io_buffers,
                input_bytes=sum(x.nbytes+ids.nbytes for x,ids in groups),
                output_bytes=sum(x.nbytes for x in arrays),rows=sum(len(x) for x,_ in groups),
                note='one routed layer; float32 dequantized reference; no gate-weighted merge or end-to-end speed claim')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


def verify(a):
    manifest=json.loads((a.pack/'manifest.json').read_text())
    d,m,e,ts=geometry(a.model,manifest['layer'])
    if (d,m,e)!=(manifest['hidden'],manifest['intermediate'],manifest['experts']):
        raise ValueError('geometry differs')
    payload=hashlib.sha256();count=0
    with a.model.open('rb') as src,(a.pack/'weights.bin').open('rb') as pack:
        for expert in range(e):
            reconstructed={k:[] for k in ts}
            blocks=[b for b in manifest['blocks'] if b['expert']==expert]
            if [b['start'] for b in blocks]!=list(range(0,m,manifest['tile'])):
                raise ValueError('missing or unordered tiles')
            for b in blocks:
                pack.seek(b['offset'])
                for kind,size in zip(('gate','up','down'),b['lengths']):
                    raw=pack.read(size)
                    if len(raw)!=size:raise ValueError('short bundle')
                    reconstructed[kind].append(raw)
            for kind,t in ts.items():
                parts=reconstructed[kind]
                if kind=='down':
                    merged=np.concatenate([np.frombuffer(b,dtype=np.uint8).reshape(d,-1) for b in parts],axis=1).tobytes()
                else:merged=b''.join(parts)
                size=t['bytes']//e;src.seek(t['absolute']+expert*size);original=src.read(size)
                if len(original)!=size or merged!=original:raise ValueError(f'byte mismatch expert {expert} {kind}')
                payload.update(original);count+=size
    if payload.hexdigest()!=manifest['source_layer_payload_sha256']:raise ValueError('source payload hash differs')
    pack_hash=hashlib.sha256()
    with (a.pack/'weights.bin').open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):pack_hash.update(chunk)
    result=dict(experts=e,verified_original_bytes=count,byte_identical=True,
                source_layer_payload_sha256=payload.hexdigest(),pack_sha256=pack_hash.hexdigest())
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    b=sub.add_parser('build');b.add_argument('--model',type=Path,required=True);b.add_argument('--layer',type=int,default=0)
    b.add_argument('--tile',type=int,default=256);b.add_argument('--output',type=Path,required=True)
    v=sub.add_parser('verify')
    for name in ('model','pack','output'):v.add_argument('--'+name,type=Path,required=True)
    r=sub.add_parser('replay')
    for name in ('model','pack','trace','decoder','output'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--phase');r.add_argument('--schedule',choices=('separate','grouped'),required=True)
    r.add_argument('--representation',choices=('original','tiles'),required=True)
    a=p.parse_args()
    if a.cmd=='build':build(a)
    elif a.cmd=='verify':verify(a)
    else:replay(a)


if __name__=='__main__':main()
