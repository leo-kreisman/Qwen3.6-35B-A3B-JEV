#!/usr/bin/env python3
"""Freeze SemIf requests, token IDs and exact common prefixes for native replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'semif/src'))
from semif_phase1.direct import encode_prompt
from semif_phase1.core import direct_messages
from transformers import AutoTokenizer


def common_prefix(sequences):
    if not sequences or any(not s for s in sequences):
        raise ValueError('nonempty sequences required')
    n = 0
    for items in zip(*sequences):
        if len(set(items)) != 1:
            break
        n += 1
    return min(n, min(map(len, sequences)) - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    rows = [json.loads(s) for s in a.input.read_text().splitlines() if s.strip()]
    if not rows or any(r['state'] != rows[0]['state'] for r in rows):
        raise ValueError('one exact shared state per fixture required')
    encoded = []
    for row in rows:
        ids, slots, digest = encode_prompt(tok, row, 4096)
        prompt = tok.apply_chat_template(direct_messages(row), tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        encoded.append(dict(id=row['id'], tokens=ids, slots=slots, prompt=prompt,
                            prompt_sha256=digest, option_ids=[o['id'] for o in row['options']]))
    prefix = common_prefix([r['tokens'] for r in encoded])
    report = dict(format='jev-resume-input-v1', rows=encoded, prefix_tokens=prefix,
                  source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                  full_token_positions=sum(len(r['tokens']) for r in encoded),
                  shared_token_positions=prefix + sum(len(r['tokens'])-prefix for r in encoded))
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
