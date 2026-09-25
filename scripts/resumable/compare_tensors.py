#!/usr/bin/env python3
"""Compare corresponding prefix tensors, retaining repeated-name occurrences."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def tensors(directory, phase):
    counts=defaultdict(int);out={}
    for line in (directory/'routes.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r.get('kind')!='tensor' or r['phase']!=phase:continue
        key=(r['name'],counts[r['name']]);counts[r['name']]+=1
        out[key]=r
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('full','split','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();full=tensors(a.full,'full');split=tensors(a.split,'prefix-0')
    nr=json.loads((a.full/'result.json').read_text())['phases'][0]['tokens']
    nprefix=json.loads((a.split/'result.json').read_text())['phases'][0]['tokens']
    result=[]
    for key,f in full.items():
        if key not in split:continue
        b=split[key];slices=[]
        for x,y in zip(f['shape'][::-1],b['shape'][::-1]):
            if x==y:slices.append(slice(None))
            elif x==nr and y==nprefix:slices.append(slice(0,nprefix))
            else:break
        if len(slices)!=4:continue
        x=np.fromfile(a.full/f['input_file'],dtype=np.float32).reshape(f['shape'][::-1])[tuple(slices)]
        y=np.fromfile(a.split/b['input_file'],dtype=np.float32).reshape(b['shape'][::-1])
        equal=x==y
        if np.isnan(x).any() or np.isnan(y).any():raise ValueError('nonfinite comparison input')
        difference=np.zeros_like(x);np.subtract(x,y,out=difference,where=~equal)
        result.append(dict(name=key[0],occurrence=key[1],full_shape=f['shape'],prefix_shape=b['shape'],
                           prefix_max_abs=float(np.abs(difference).max()),exact=bool(equal.all())))
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
