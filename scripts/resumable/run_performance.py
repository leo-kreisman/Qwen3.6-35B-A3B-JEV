#!/usr/bin/env python3
"""Balanced, cold CPU comparisons: stock graph, control tiles, candidate tiles.

Control and candidate must both implement diagnostics=false. This isolates tile
changes from logging changes. Each variant runs in its own 8 GiB/no-swap scope.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from analyze import compare

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'input', 'pack', 'control', 'candidate', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--mode', choices=('full-padded', 'shared-padded'), default='full-padded')
    p.add_argument('--protect-weights', action='store_true')
    a = p.parse_args()
    if a.rounds < 1:
        p.error('rounds must be positive')
    a.output.mkdir(exist_ok=False)
    fixture = json.loads(a.input.read_text())
    fixture.update(diagnostics=False, protect_bypassed_weights=a.protect_weights)
    inp = a.output / 'input.json'
    inp.write_text(json.dumps(fixture, indent=2) + '\n')
    expected_groups = fixture.get('repetitions', 1) * (2 if a.mode == 'shared-padded' else 1)
    report = dict(mode=a.mode, rounds=a.rounds, diagnostics=False,
                  binaries={name: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                            for name, path in [('control', a.control), ('candidate', a.candidate)]},
                  runs=[], pairs=[], passed=True,
                  caveat='Only the packed layer is replaced. Timings include callback control overhead; no diagnostic route/tensor logging. Small repeated fixture, not production p95.')
    def save():
        (a.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for round_id in range(a.rounds):
        results = {}
        order = ['baseline', 'control', 'candidate']
        if round_id % 2:
            order.reverse()
        for name in order:
            tag = f'{round_id + 1}-{name}'
            out = a.output / tag
            binary = a.control if name == 'control' else a.candidate
            cmd = ['systemd-run', '--user', '--scope', '-p', 'MemoryMax=8G', '-p', 'MemorySwapMax=0', '--',
                   sys.executable, str(ROOT / 'scripts/tiered/run_guarded.py'), '--output', str(a.output / (tag + '.guard.json')),
                   '--evict', str(a.model), '--evict', str(a.pack / 'weights.bin'), '--',
                   str(binary), str(a.model), str(inp), str(out), a.mode, '-1', '6']
            if name != 'baseline':
                cmd += [str(a.pack), 'bypass', 'staged']
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            r = json.loads((out / 'result.json').read_text())
            guard = json.loads((a.output / (tag + '.guard.json')).read_text())
            if r['tracing'] or (out / 'routes.jsonl').exists():
                raise RuntimeError('diagnostic logging enabled during timing')
            if guard['exit_code'] or guard['memory_events'].get('oom', 0) or guard['memory_events'].get('oom_kill', 0):
                raise RuntimeError('invalid guarded execution')
            if name != 'baseline':
                counts = r['native_bypass']
                if counts['groups'] != expected_groups or not counts['operators_restored'] or any(
                        counts[k] != expected_groups for k in ('skipped_gate', 'skipped_up', 'skipped_activation', 'skipped_down')):
                    raise RuntimeError('operation count/restoration mismatch')
                if a.protect_weights and not counts['protected_original_weight_bytes']:
                    raise RuntimeError('weight protection missing')
            results[name] = r
            report['runs'].append(dict(round=round_id + 1, variant=name, seconds=r['scoring_seconds'],
                                       read_bytes=r['scoring_read_bytes'], memory_peak=guard['memory_peak'],
                                       native_tiles=r.get('native_tiles')))
            save()
            print(tag, r['scoring_seconds'], 'seconds', flush=True)
        for name in ('control', 'candidate'):
            ref, cur = results['baseline']['iterations'], results[name]['iterations']
            if len(ref) != len(cur):
                raise RuntimeError('iteration count mismatch')
            diffs = [compare({'answers': x}, {'answers': y}) for x, y in zip(ref, cur)]
            exact = all(d['max_logit_drift'] == 0 and d['max_probability_drift'] == 0 for d in diffs)
            report['passed'] &= exact
            report['pairs'].append(dict(round=round_id + 1, variant=name, exact=exact, differences=diffs))
        save()
        if not report['passed']:
            raise RuntimeError('fidelity gate failed')
    report['summary'] = {name: dict(median_seconds=statistics.median(r['seconds'] for r in report['runs'] if r['variant'] == name),
                                  median_read_bytes=statistics.median(r['read_bytes'] for r in report['runs'] if r['variant'] == name))
                         for name in ('baseline', 'control', 'candidate')}
    report['paired_candidate_minus_control_seconds'] = [
        next(r['seconds'] for r in report['runs'] if r['round'] == i and r['variant'] == 'candidate') -
        next(r['seconds'] for r in report['runs'] if r['round'] == i and r['variant'] == 'control')
        for i in range(1, a.rounds + 1)]
    save()
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
