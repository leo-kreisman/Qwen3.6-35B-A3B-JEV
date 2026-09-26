#!/usr/bin/env python3
"""Balanced native layer replay with byte-identical output and equal-I/O gates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pack', 'trace', 'control', 'candidate', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.rounds < 1 or a.repeats < 1:
        p.error('rounds and repeats must be positive')
    a.output.mkdir(exist_ok=False)
    report = dict(scope='single-layer saved-input replay, not end-to-end latency', complete=False,
                  runs=[], binaries={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for name, path in [('control', a.control), ('candidate', a.candidate)]})
    reference = None
    for r in range(a.rounds):
        order = ['control', 'sync', 'overlap']
        if r % 2:
            order.reverse()
        for variant in order:
            tag = f'{r+1}-{variant}'
            out = a.output / tag
            binary = a.control if variant == 'control' else a.candidate
            cmd = ['systemd-run', '--user', '--scope', '-p', 'MemoryMax=8G', '-p', 'MemorySwapMax=0', '--',
                   sys.executable, str(ROOT / 'scripts/tiered/run_guarded.py'), '--output', str(a.output / (tag + '.guard.json')),
                   '--evict', str(a.pack / 'weights.bin'), '--', str(binary), str(a.pack), str(a.trace), str(out),
                   str(a.repeats), 'overlap' if variant == 'overlap' else 'sync']
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            result = json.loads((out / 'result.json').read_text())
            guard = json.loads((a.output / (tag + '.guard.json')).read_text())
            checksum = hashlib.sha256()
            with (out / 'outputs.f32').open('rb') as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    checksum.update(chunk)
            digest = checksum.hexdigest()
            identity = (digest, result['native_tiles']['direct_read_bytes'], result['native_tiles']['read_calls'])
            if reference is None:
                reference = identity
            if identity != reference:
                raise RuntimeError('output bytes or I/O changed')
            if guard['exit_code'] or guard['memory_events'].get('oom', 0) or guard['memory_events'].get('oom_kill', 0):
                raise RuntimeError('invalid guarded run')
            report['runs'].append(dict(round=r+1, variant=variant, median_seconds=statistics.median(result['seconds']),
                                       seconds=result['seconds'], output_sha256=digest, memory_peak=guard['memory_peak'],
                                       native_tiles=result['native_tiles']))
            (a.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
            print(tag, report['runs'][-1]['median_seconds'], flush=True)
    report['summary'] = {v: statistics.median(r['median_seconds'] for r in report['runs'] if r['variant'] == v)
                         for v in ('control', 'sync', 'overlap')}
    report.update(complete=True, passed=True)
    (a.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
