#!/usr/bin/env python3
"""Score JEV JSONL using the experimental streaming backend, under an 8 GiB cap.

Uses SemIf's prompt renderer. Each question is a separate prefill in one loaded
process. This adapter does not implement shared-prefix batching or the HTTP API.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from analyze import read_log

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'semif/src'))
from semif_phase1.core import direct_messages, LETTERS
from transformers import AutoTokenizer

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--pack',type=Path)
    ap.add_argument('--cli',type=Path,default=ROOT/'vendor/BigMoeOnEdge/build/cli/bmoe-cli')
    ap.add_argument('--context',type=int,default=1024)
    a=ap.parse_args()
    outputs=[a.output,a.output.with_suffix('.run.json'),a.output.with_suffix('.log')]
    if any(p.exists() for p in outputs): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in a.input.read_text().splitlines() if l.strip()]
    if not rows: raise ValueError('empty input')
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    maximum=max(len(r['options']) for r in rows)
    answers=[]
    with tempfile.TemporaryDirectory(prefix='jev-tiered-') as tmp:
        d=Path(tmp);paths=[]; hashes=[]
        for i,r in enumerate(rows):
            text=tok.apply_chat_template(direct_messages(r),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            p=d/f'{i}.txt';p.write_text(text);paths.append(p)
            hashes.append(hashlib.sha256(text.encode()).hexdigest())
        listing=d/'inputs.list';listing.write_text(''.join(str(p)+'\n' for p in paths))
        cmd=[str(a.cli.resolve()),'-m',str(a.model.resolve()),'--moe-stream','--dense-weights','anon',
             '--cache-mb','3000','--io-threads','4','-t','6','-c',str(a.context),'--ubatch',str(min(a.context,512)),
             '--ppl-list',str(listing),'--ppl-choices',','.join(LETTERS[:maximum]),'--choices-only']
        if a.pack:cmd+=['--tiered-pack',str(a.pack.resolve())]
        run=d/'run.json'
        completed=subprocess.run(['systemd-run','--user','--scope','-p','MemoryMax=8G','-p','MemorySwapMax=0','--',
                        sys.executable,str(HERE/'run_guarded.py'),'--output',str(run),'--',*cmd],
                       stdout=subprocess.DEVNULL)
        if completed.returncode:
            if run.with_suffix('.log').exists():
                sys.stderr.write(run.with_suffix('.log').read_text())
            raise RuntimeError(f'scoring failed with exit code {completed.returncode}')
        scored=read_log(run.with_suffix('.log'))
        if len(scored)!=len(rows):raise ValueError('missing scorer outputs')
        for row,result,digest in zip(rows,scored,hashes):
            labels=LETTERS[:len(row['options'])]
            logits=[result['logp'][label] for label in labels]
            high=max(logits);p=[math.exp(v-high) for v in logits];total=sum(p);p=[v/total for v in p]
            answers.append(dict(id=row['id'],option_ids=[o['id'] for o in row['options']],probabilities=p,
                prompt_sha256=digest,input_tokens=result['tokens'],forward_seconds=result['seconds'],
                expert_read_bytes=result['expert_read_bytes'],model=str(a.model.resolve()),
                tiered_pack=str(a.pack.resolve()) if a.pack else None,
                probability_status='conditional option scores; uncalibrated; tiered weights are lossy when enabled'))
        contents=[''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in answers),
                  run.read_text(),run.with_suffix('.log').read_text()]
        for path,content in zip(outputs,contents):
            with path.open('x') as f:f.write(content)

HERE=Path(__file__).resolve().parent
if __name__=='__main__':main()
