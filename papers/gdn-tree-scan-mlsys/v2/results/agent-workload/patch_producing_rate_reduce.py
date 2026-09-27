#!/usr/bin/env python3
"""Apply a retrospective, symmetric patch-production rule to recovered runs.

Does not change original rates or evidence. Every task in a compared run-pair
must complete and produce a nonempty patch; test failure does not exclude it.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SHARED = ('Sr12', 'Cqc16', 'Cqc15', 'SGLang_earlier', 'SGLang_later')
MARKERS = {
    'Sr12': '/Sr12/', 'Sr12_full_four_tasks': '/Sr12/',
    'Cqc16': '/Cqc16/', 'Cqc15': '/Cqc15/', 'Cqc10': '/cqc10/',
    'SGLang_earlier': '/sglang-step2-20260924/',
    'SGLang_later': '/sglang16_evidence/',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = subprocess.check_output([sys.executable, str(HERE/'competitive_rate_reduce.py')])
    if raw != (HERE/'competitive-rate-audit.json').read_bytes():
        raise ValueError('unfiltered competitive replay differs from saved audit')
    audit = json.loads(raw)
    eligibility = {}
    for name, arm in audit['arms'].items():
        tasks = []
        for row in arm['tasks']:
            paths = [k for k in audit['input_sha256']
                     if MARKERS[name] in k and k.endswith(row['task']+'/runner_metadata.json')]
            if len(paths) != 1:
                raise ValueError('ambiguous task metadata: '+name+' '+row['task'])
            path = paths[0]
            if digest(REPO/path) != audit['input_sha256'][path]:
                raise ValueError('metadata hash mismatch: '+path)
            meta = json.loads((REPO/path).read_text())
            if meta['instance_id'] != row['task']:
                raise ValueError('metadata task mismatch')
            n = meta['patch_bytes']
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError('invalid patch size')
            agent = meta['agent']
            completed = (agent.get('exit_code') == 0 and not agent.get('timed_out', True)
                         and not agent.get('budget_capped', False) and bool(meta.get('ended_at')))
            reasons = []
            if not completed:
                reasons.append('incomplete_attempt')
            if n == 0:
                reasons.append('empty_patch')
            tasks.append({'task': row['task'], 'patch_bytes': n, 'completed': completed,
                          'eval_verdict': row['eval_verdict'], 'eligible': not reasons,
                          'reasons': reasons, 'metadata_path': path})
        eligibility[name] = {'eligible': all(t['eligible'] for t in tasks), 'tasks': tasks}
    included = [n for n in SHARED if eligibility[n]['eligible']]
    excluded = [n for n in SHARED if not eligibility[n]['eligible']]
    trees = [n for n in included if n in ('Sr12', 'Cqc16', 'Cqc15')]
    sg = [n for n in included if n.startswith('SGLang_')]
    if not trees or not sg:
        raise ValueError('no eligible shared-task comparison')
    rate = lambda n: audit['arms'][n]['totals']['pooled_post_first_token_tokens_per_s']
    best_tree, best_sg = max(trees, key=rate), max(sg, key=rate)
    result = {
        'schema': 'patch-producing-shared-task-pooled-decode-v1',
        'scope': 'Retrospective comparison conditional on both shared tasks completing and producing nonempty patches. Same rule applied to both systems; failed tests remain eligible. Excludes whole run-pairs, not individual requests or only their failed tasks. Does not estimate speed over all attempts or establish quality equivalence.',
        'selection_timing': 'Retrospective user-requested rule after inspecting the archived outcomes.',
        'eligibility_rule': 'Each task: agent exit0, ended_at present, no timeout/budget cap, patch_bytes>0. No condition on test success or decode rate.',
        'shared_task_ids': audit['shared_task_ids'], 'formula': audit['formula'],
        'included_shared_task_runs': included, 'excluded_shared_task_runs': excluded,
        'eligibility': eligibility,
        'eligible_shared_task_arms': {n:audit['arms'][n] for n in included},
        'context_arms': {n:audit['arms'][n] for n in ('Cqc10', 'Sr12_full_four_tasks')},
        'best_eligible_tree': best_tree, 'best_eligible_sglang': best_sg,
        'tree_relative_to_eligible_sglang_percent': 100*(rate(best_tree)/rate(best_sg)-1),
        'unfiltered_audit': {'path':'competitive-rate-audit.json', 'sha256':digest(HERE/'competitive-rate-audit.json')},
        'input_sha256': audit['input_sha256'],
        'configuration_and_behavior_projections': audit['configuration_and_behavior_projections'],
        'implementation_sha256': audit['implementation_sha256'] | {'patch_producing_rate_reduce.py':digest(Path(__file__))},
        'new_inference_launched': False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
