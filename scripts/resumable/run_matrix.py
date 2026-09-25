#!/usr/bin/env python3
"""Run sequential, cold-start, memory-capped comparisons; never overlap engines."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--modes',nargs='+',default=['full','full-padded','shared-padded'])
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    for repeat in range(a.rounds):
        modes=a.modes if repeat%2==0 else list(reversed(a.modes))
        for mode in modes:
            name=f'{mode}-{repeat+1}';guard=a.output/(name+'.guard.json');out=a.output/name
            cmd=['systemd-run','--user','--scope','-p','MemoryMax=8G','-p','MemorySwapMax=0','--',
                 sys.executable,str(ROOT/'scripts/tiered/run_guarded.py'),'--output',str(guard),
                 '--evict',str(a.model),'--',str(a.probe),str(a.model),str(a.input),str(out),mode]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            r=json.loads((out/'result.json').read_text())
            print(json.dumps(dict(run=name,seconds=r['scoring_seconds'],read_bytes=r['scoring_read_bytes'])),flush=True)


if __name__=='__main__':main()
