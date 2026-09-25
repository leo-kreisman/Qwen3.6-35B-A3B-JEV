#!/usr/bin/env python3
"""Render the existing JEV examples and separate synthetic UI decision checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'semif/src'))
from semif_phase1.core import direct_messages
from transformers import AutoTokenizer

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenizer',required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    calibration=[json.loads(l) for l in (ROOT/'examples/decisions.sample.jsonl').read_text().splitlines()]
    checks=[
        ('Origin Lisbon; destination Madrid. Goal: Lisbon to Madrid.','Does the origin match the goal?',True),
        ('Origin Lisbon; destination Madrid. Goal: Porto to Madrid.','Does the origin match the goal?',False),
        ('Departure 2026-11-04. Goal departure 2026-11-14.','Is the departure date correct?',False),
        ('Departure 2026-11-14. Goal departure 2026-11-14.','Is the departure date correct?',True),
        ('A return-trip radio button is selected; one-way is not. Goal: a one-way flight.','Is the trip type correct?',False),
        ('A one-way radio button is selected. Goal: a one-way flight.','Is the trip type correct?',True),
        ('Checkbox Subscribe is unchecked. Goal: do not subscribe.','Should Subscribe be clicked to achieve the goal?',False),
        ('Checkbox Subscribe is checked. Goal: do not subscribe.','Should Subscribe be clicked to achieve the goal?',True),
        ('Button Save is disabled. Button Cancel is enabled.','Can Save currently be activated?',False),
        ('Button Save is enabled. Button Cancel is disabled.','Can Save currently be activated?',True),
        ('The destination field contains Paris, not London. The goal is to travel to London.','Does the destination match the goal?',False),
        ('The destination field contains London. A nearby advertisement mentions Paris. Goal: London.','Does the destination match the goal?',True),
    ]
    validation=[dict(id=f'ui-{i}',state=s,question=q,options=[dict(id='yes',description='Yes.'),
                   dict(id='no',description='No.')],expected='A' if yes else 'B') for i,(s,q,yes) in enumerate(checks)]
    for split,rows in [('calibration',calibration),('validation',validation)]:
        d=a.output/split;d.mkdir()
        records=[]
        for i,row in enumerate(rows):
            p=d/f'{i:02}.txt'
            text=tok.apply_chat_template(direct_messages(row),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            p.write_text(text)
            records.append(dict(row,path=str(p.resolve()),prompt_sha256=__import__('hashlib').sha256(text.encode()).hexdigest()))
        (a.output/f'{split}.json').write_text(json.dumps(records,indent=2)+'\n')
        (a.output/f'{split}.list').write_text(''.join(r['path']+'\n' for r in records))
        (a.output/f'{split}.requests.jsonl').write_text(''.join(json.dumps(dict(cmd='generate',id=i,
                 prompt=Path(r['path']).read_text(),n_predict=1,clear_kv=True))+'\n' for i,r in enumerate(records)))

if __name__=='__main__': main()
