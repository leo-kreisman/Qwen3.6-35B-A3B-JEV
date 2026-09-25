#!/usr/bin/env python3
"""Replay measured expert demand with an explicit conservative 8 GiB ledger.

Simulation estimates expert payload traffic, not mmap physical reads or latency.
Grouping is an offline opportunity bound, not an implemented graph scheduler.
"""
import argparse
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/probes'))
from gguf_tensor_layout import parse


def lru(demands,sizes,budget):
    cache=OrderedDict();used=0;read=0;hits=0;misses=0
    for layer,ids in demands:
        for expert in ids:
            key=(layer,expert);size=sizes[layer]
            if key in cache:
                hits+=1;cache.move_to_end(key);continue
            read+=size;misses+=1
            if size>budget:continue
            while used+size>budget:
                _,old=cache.popitem(last=False);used-=old
            cache[key]=size;used+=size
    return dict(expert_payload_bytes=read,hits=hits,misses=misses,final_cache_bytes=used)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--snapshot-bytes',type=int,required=True);p.add_argument('--branches',type=int,required=True)
    p.add_argument('--context-bytes',type=int,required=True,
                   help='allocated KV plus recurrent state bytes from the engine log; not serialized prefix size')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=[json.loads(l) for l in a.trace.read_text().splitlines()]
    _,_,_,ts=parse(a.model);sizes=defaultdict(int);expert_counts={};dense=0;intermediate=0;hidden=0
    for t in ts:
        m=re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight',t['name'])
        if m:
            sizes[int(m[1])]+=t['bytes']//t['dims'][2]
            expert_counts[int(m[1])]=t['dims'][2]
            if m[2]=='gate':hidden=t['dims'][0];intermediate=t['dims'][1]
        else:dense+=t['bytes']
    if not sizes:raise ValueError('no experts')
    demands=[];grouped={};token_positions=defaultdict(int)
    for r in records:
        if r['layer'] not in sizes:raise ValueError('trace geometry mismatch')
        if any(not 0<=e<expert_counts[r['layer']] for e in r['ids']):raise ValueError('invalid expert ID')
        ids=list(dict.fromkeys(r['ids']));demands.append((r['layer'],ids))
        key=(r['phase'],r['layer'])
        grouped.setdefault(key,[])
        grouped[key]=list(dict.fromkeys(grouped[key]+ids))
        token_positions[key]+=r['input_shape'][2]
    # Never reorder prefix computation with suffix work: separate phase fences.
    phases=list(dict.fromkeys(r['phase'] for r in records))
    frontier=[(layer,grouped[(phase,layer)]) for phase in phases for layer in sorted(sizes) if (phase,layer) in grouped]
    peak_rows=max(token_positions.values())
    ledger=dict(dense_weights=dense,index_and_runtime_reserve=512*1024**2,
                graph_scratch_reserve=256*1024**2,
                branch_states=a.context_bytes+a.snapshot_bytes,
                activations=peak_rows*4*(6*hidden+3*intermediate),
                io_buffers=2*max(sizes.values()),deployment_margin=512*1024**2)
    mandatory=sum(ledger.values());cap=8*1024**3
    reports=[]
    for cache_mib in (0,512,1024,2048,3072,4096):
        cache=cache_mib*1024**2
        if mandatory+cache>cap:
            reports.append(dict(cache_mib=cache_mib,feasible_in_model=False,required=mandatory+cache));continue
        old=lru(demands,sizes,cache);new=lru(frontier,sizes,cache)
        reports.append(dict(cache_mib=cache_mib,feasible_in_model=True,total_budget=mandatory+cache,
                            captured_order=old,layer_frontier=new,
                            payload_reduction=1-new['expert_payload_bytes']/old['expert_payload_bytes']))
    result=dict(trace=str(a.trace),branches=a.branches,captured_groups=len(demands),frontier_groups=len(frontier),
                peak_ready_rows=peak_rows,ledger=ledger,scenarios=reports,
                unique_expert_payload_bytes=sum(sizes[l]*len(ids) for l,ids in frontier),
                caveats=['offline route knowledge; no latency prediction',
                         'ledger is conservative design accounting, not a measured allocator proof',
                         'no cross-phase reordering; within-phase layer-frontier assumes explicit graph support',
                         'expert payload excludes page readahead, dense rereads and filesystem amplification'])
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
