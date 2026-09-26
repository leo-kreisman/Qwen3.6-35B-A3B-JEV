#!/usr/bin/env python3
"""Cold, capped baseline/bypass comparison with exact logits and operation counts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from analyze import compare

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','input','pack','probe','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',default='full-padded')
    p.add_argument('--protect-weights',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    fixture=json.loads(a.input.read_text());fixture['protect_bypassed_weights']=a.protect_weights
    inp=a.output/'input.json';inp.write_text(json.dumps(fixture,indent=2)+'\n')
    results={}
    for name in ('baseline','bypass'):
        out=a.output/name
        cmd=['systemd-run','--user','--scope','-p','MemoryMax=8G','-p','MemorySwapMax=0','--',
             sys.executable,str(ROOT/'scripts/tiered/run_guarded.py'),'--output',str(a.output/(name+'.guard.json')),
             '--evict',str(a.model),'--evict',str(a.pack/'weights.bin'),'--',str(a.probe),str(a.model),str(inp),str(out),a.mode,'-1','6']
        if name=='bypass':cmd += [str(a.pack),'bypass','staged']
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        results[name]=json.loads((out/'result.json').read_text())
        print(name,results[name]['scoring_seconds'],'seconds',flush=True)
    reference=results['baseline'];candidate=results['bypass']
    differences=[compare({'answers':x},{'answers':y}) for x,y in zip(reference['iterations'],candidate['iterations'])]
    layer=json.loads((a.pack/'manifest.json').read_text())['layer']
    groups=sum(json.loads(line).get('layer')==layer for line in (a.output/'baseline/routes.jsonl').read_text().splitlines())
    counts=candidate['native_bypass']
    exact=(len(reference['iterations'])==len(candidate['iterations']) and all(d['max_logit_drift']==0 for d in differences))
    count_match=counts['groups']==groups and all(counts[k]==groups for k in ('skipped_gate','skipped_up','skipped_activation','skipped_down'))
    protection=not a.protect_weights or counts['protected_original_weight_bytes']>0
    passed=exact and count_match and protection and counts['operators_restored']
    report=dict(passed=passed,iteration_comparisons=differences,baseline_expert_groups=groups,bypass=counts,
                measurements={name:{k:r[k] for k in ('scoring_seconds','scoring_read_bytes')} for name,r in results.items()},
                caveat='one ordered pair; not a statistically established speedup; only packed layer bypassed')
    with (a.output/'comparison.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit('bypass fidelity/operation-count gate failed')


if __name__=='__main__':main()
