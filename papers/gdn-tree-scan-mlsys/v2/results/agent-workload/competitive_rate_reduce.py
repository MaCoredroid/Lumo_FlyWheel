#!/usr/bin/env python3
"""Combine all recovered shared-task system comparisons using verified reducers.

CPU only. Retains earlier SGLang's empty-patch failure and all tree observations.
The child reducers check original metrics, populations, boundaries and manifests.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(script, receipt):
    output = subprocess.check_output([sys.executable, str(HERE / script)])
    if output != (HERE / receipt).read_bytes():
        raise ValueError('child replay differs from saved audit: ' + script)
    return json.loads(output)


def main():
    current = replay('shared_task_reduce.py', 'shared-task-rate-audit.json')
    earlier = replay('sglang_step2_reduce.py', 'sglang-step2-rate-audit.json')
    arms = dict(current['arms'])
    arms['SGLang_later'] = arms.pop('SGLang_EAGLE')
    arms['SGLang_earlier'] = {
        'tasks': earlier['tasks'], 'totals': earlier['totals'],
        'task_count': len(earlier['tasks']),
        'resolved': sum(t['eval_verdict'] == 'resolved' for t in earlier['tasks']),
    }
    shared = ('Sr12', 'Cqc16', 'Cqc15', 'SGLang_earlier', 'SGLang_later')
    ids = sorted(current['shared_task_ids'])
    for name in shared:
        if sorted(t['task'] for t in arms[name]['tasks']) != ids:
            raise ValueError('shared task IDs differ: ' + name)
    rate = lambda name: arms[name]['totals']['pooled_post_first_token_tokens_per_s']
    tree_best = max(('Sr12', 'Cqc16', 'Cqc15'), key=rate)
    sg_best = max(('SGLang_earlier', 'SGLang_later'), key=rate)
    inputs = {**current['input_sha256'], **earlier['input_sha256']}
    manifest = HERE / 'raw/sglang-step2-20260924/manifest.json'
    inputs[str(manifest.relative_to(REPO))] = digest(manifest)
    for row in json.loads(manifest.read_text())['files']:
        path = manifest.parent / row['path']
        inputs[str(path.relative_to(REPO))] = digest(path)
    projections = dict(current['configuration_and_behavior_projections'])
    for name in ('sglang-step2-config-projection.json', 'sglang-step2-behavior-projection.json'):
        path = HERE / name
        projections[str(path.relative_to(REPO))] = digest(path)
    scripts = ['competitive_rate_reduce.py', 'shared_task_reduce.py',
               'sglang_step2_reduce.py', 'shared_rate_reduce.py']
    result = {
        'schema': 'competitive-shared-task-pooled-decode-v1',
        'scope': 'Complete decoding systems on the shared two-task subset; all three recovered tree observations and both SGLang observations retained, including the earlier empty-patch failure. No common draft-budget tuning sweep or general superiority established.',
        'formula': current['formula'], 'shared_task_ids': ids, 'arms': arms,
        'best_recorded_tree': tree_best, 'best_recorded_sglang': sg_best,
        'tree_relative_to_sglang_percent': {
            name: 100 * (rate(tree_best) / rate(name) - 1)
            for name in ('SGLang_earlier', 'SGLang_later')
        },
        'input_sha256': dict(sorted(inputs.items())),
        'configuration_and_behavior_projections': dict(sorted(projections.items())),
        'implementation_sha256': {s: digest(HERE/s) for s in scripts},
        'child_audits_sha256': {s: digest(HERE/s) for s in ('shared-task-rate-audit.json', 'sglang-step2-rate-audit.json')},
        'new_inference_launched': False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
