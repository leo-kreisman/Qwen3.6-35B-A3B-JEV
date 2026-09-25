#!/usr/bin/env python3
"""Compare final-label scoring logs without interpreting raw confidence as accuracy."""
import argparse
import json
import math
from pathlib import Path
import re

def read_log(path):
    rows=[]; current={}
    for line in path.read_text().splitlines():
        if line.startswith('ppl-file: '): current={'path':line[10:]}
        elif line.startswith('choices-only: '):
            m=re.search(r'(\d+) input tokens, ([\d.]+) s, (\d+) expert read bytes',line)
            if not m: raise ValueError(f'invalid scoring telemetry: {line}')
            current.update(tokens=int(m[1]),seconds=float(m[2]),expert_read_bytes=int(m[3]))
        elif line.startswith('ppl-choices: '):
            values={k:float(v) for k,v in (x.split('=') for x in line[13:].split())}
            if not values or not all(math.isfinite(v) for v in values.values()):
                raise ValueError('non-finite or empty answer log-probabilities')
            high=max(values.values()); p={k:math.exp(v-high) for k,v in values.items()}
            total=sum(p.values()); p={k:v/total for k,v in p.items()}
            current.update(logp=values,probabilities=p,winner=max(p,key=p.get))
            rows.append(current);current={}
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixtures',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    fixtures=json.loads(a.fixtures.read_text()); base=read_log(a.baseline); candidate=read_log(a.candidate)
    if not len(fixtures)==len(base)==len(candidate): raise ValueError('missing scored rows')
    rows=[]
    for f,b,c in zip(fixtures,base,candidate):
        if b['path']!=f['path'] or c['path']!=f['path']: raise ValueError('fixture order mismatch')
        rows.append(dict(id=f['id'],expected=f['expected'],baseline=b,candidate=c,
                         flipped=b['winner']!=c['winner'],
                         max_probability_drift=max(abs(b['probabilities'][k]-c['probabilities'][k]) for k in b['probabilities'])))
    baseline_seconds=sum(r['seconds'] for r in base); candidate_seconds=sum(r['seconds'] for r in candidate)
    result=dict(count=len(rows),baseline_correct=sum(r['baseline']['winner']==r['expected'] for r in rows),
                candidate_correct=sum(r['candidate']['winner']==r['expected'] for r in rows),
                flips=sum(r['flipped'] for r in rows),max_probability_drift=max(r['max_probability_drift'] for r in rows),
                baseline_scoring_seconds=baseline_seconds,candidate_scoring_seconds=candidate_seconds,
                speedup=baseline_seconds/candidate_seconds,
                baseline_expert_read_bytes=sum(r['expert_read_bytes'] for r in base),
                candidate_expert_read_bytes=sum(r['expert_read_bytes'] for r in candidate),rows=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
