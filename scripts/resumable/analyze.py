#!/usr/bin/env python3
"""Summarize measured runs without treating tracing or simulation as timing proof."""
import argparse
import json
from pathlib import Path
import statistics


def compare(reference,candidate):
    base={r['id']:r for r in reference['answers']}
    other={r['id']:r for r in candidate['answers']}
    if base.keys()!=other.keys():raise ValueError('different decision sets')
    flips=[];maximum=0.;logit_max=0.
    for key,a in base.items():
        b=other[key]
        if len(a['probabilities'])!=len(b['probabilities']):raise ValueError('different options')
        maximum=max(maximum,max(abs(p-q) for p,q in zip(a['probabilities'],b['probabilities'])))
        logit_max=max(logit_max,max(abs(p-q) for p,q in zip(a['logits'],b['logits'])))
        if max(range(len(a['probabilities'])),key=a['probabilities'].__getitem__)!=max(range(len(b['probabilities'])),key=b['probabilities'].__getitem__):flips.append(key)
    return dict(flips=flips,max_probability_drift=maximum,max_logit_drift=logit_max)


def summarize(directory):
    runs=[]
    for file in sorted(directory.glob('*/result.json')):
        r=json.loads(file.read_text());guard=file.parent.parent/(file.parent.name+'.guard.json')
        if not guard.exists():raise ValueError('missing guard')
        g=json.loads(guard.read_text())
        if g['exit_code'] or g['memory_max']>8*1024**3 or g['memory_events']['oom_kill']:
            raise ValueError('invalid guarded run')
        runs.append((file.parent.name,r,g))
    if not runs:raise ValueError('no results')
    reference=next(r for _,r,_ in runs if r['mode']=='full')
    result={}
    for mode in sorted({r['mode'] for _,r,_ in runs}):
        chosen=[(name,r,g) for name,r,g in runs if r['mode']==mode]
        result[mode]=dict(runs=len(chosen),median_scoring_seconds=statistics.median(r['scoring_seconds'] for _,r,_ in chosen),
                          median_scoring_read_bytes=statistics.median(r['scoring_read_bytes'] for _,r,_ in chosen),
                          measurements=[dict(name=name,seconds=r['scoring_seconds'],read_bytes=r['scoring_read_bytes'],
                                             peak=g['memory_peak'],**compare(reference,r)) for name,r,g in chosen])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=summarize(a.directory)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
