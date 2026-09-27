#!/usr/bin/env python3
"""Reproduce best and later shared-task pooled rates from original safe records.

CPU only. Reuses the frozen population/clock checks without changing the prior
48-input reduction. Different builds and caps are not treated as replicates.
"""
import argparse
import hashlib
import json
from pathlib import Path

from shared_rate_reduce import Audit, TREE_IDS, SG_TASKS, rates

HERE = Path(__file__).resolve().parent
PAIR = ('12907', '13033')
FIELDS = ('output_tokens', 'completed_requests', 'request_latency_sum_s',
          'ttft_sum_s', 'agent_elapsed_s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=HERE.parents[4])
    repo = parser.parse_args().repo.resolve()
    work = repo/'papers/gdn-tree-scan-mlsys/v2/results/agent-workload'
    raw = work/'raw/shared-tasks-20260924'
    audit = Audit(repo)
    manifest = json.loads(audit.read(raw/'MANIFEST.json'))
    for row in manifest['raw_files'] + manifest['audit_projections']:
        path = raw/row['path']
        assert path.resolve().is_relative_to(raw.resolve())
        data = path.read_bytes()
        assert len(data) == (row['size'] if row['projection'] else row['copied_size']), path
        assert hashlib.sha256(data).hexdigest() == row['sha256'], path
    arms = {}

    def add(name, paths, engine):
        tasks = [audit.task(path, engine) for path in paths]
        total = {key: sum(row[key] for row in tasks) for key in FIELDS}
        total.update(rates(total))
        arms[name] = dict(task_count=len(tasks),
                         resolved=sum(row['eval_verdict'] == 'resolved' for row in tasks),
                         tasks=tasks, totals=total)

    for run in ('Sr12', 'Cqc16', 'Cqc15'):
        root = raw/run/'swe_out/verified/per_task'
        add(run, [root/('astropy__astropy-'+task) for task in PAIR], 'tree')
        if run == 'Sr12':
            add('Sr12_full_four_tasks',
                [root/('astropy__astropy-'+task) for task in PAIR+('13236', '13398')],
                'tree')
    sg = repo/'results/fr14_nvfp4_port_20260816/sglang16_evidence'
    add('SGLang_EAGLE', [sg/run/'swe_out/verified/per_task'/('astropy__astropy-'+task)
                        for run, task in SG_TASKS], 'sglang')
    cq = work/'raw/cqc10/hydra27_fixed32_promoab_Cqc10/swe_out/verified/per_task'
    add('Cqc10', [cq/('astropy__astropy-'+task) for task in TREE_IDS], 'tree')
    key = 'pooled_post_first_token_tokens_per_s'
    reference = arms['SGLang_EAGLE']['totals'][key]
    best = max(('Sr12', 'Cqc16', 'Cqc15'), key=lambda name: arms[name]['totals'][key])
    for name in ('Sr12', 'Cqc16', 'Cqc15'):
        assert [row['task'] for row in arms[name]['tasks']] == [
            row['task'] for row in arms['SGLang_EAGLE']['tasks']]
    report = {
        'schema': 'shared-task-pooled-decode-v1',
        'scope': 'Best and later recorded tree runs on the two completed SGLang task IDs; different builds, caps and trajectories; separate four-task Sr12 and ten-task Cqc10 populations retained.',
        'formula': '(sum output tokens - completed requests) / (sum E2E request latency - sum TTFT)',
        'selection': 'Select shared task IDs by completed comparator availability; report all three recovered split-K runs and explicitly identify the maximum among them.',
        'shared_task_ids': ['astropy__astropy-'+task for task in PAIR],
        'best_recorded_shared_task_run': best,
        'latest_recorded_shared_task_run': 'Cqc15',
        'tree_response_cap': 32768,
        'sglang_response_cap': 24000,
        'arms': arms,
        'relative_to_sglang_percent': {
            name: 100*(arms[name]['totals'][key]/reference-1)
            for name in ('Sr12', 'Cqc16', 'Cqc15')},
        'input_sha256': dict(sorted(audit.hashes.items())),
        'implementation_sha256': {name: hashlib.sha256((work/name).read_bytes()).hexdigest()
                                  for name in ('shared_task_reduce.py', 'shared_rate_reduce.py')},
        'configuration_and_behavior_projections': {
            str((raw/row['path']).relative_to(repo)): row['sha256']
            for row in manifest['audit_projections']},
        'new_inference_launched': False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
