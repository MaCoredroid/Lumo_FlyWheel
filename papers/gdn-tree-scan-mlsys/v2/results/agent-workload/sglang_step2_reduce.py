#!/usr/bin/env python3
"""CPU-only replay of recovered SGLang step-2 shared-task request metrics.

Retains the empty-patch failure; no filtering on task outcome or response cap.
Uses the unchanged common-estimator boundary/population audit.
"""
import hashlib
import json
from pathlib import Path
from shared_rate_reduce import Audit, rates

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'raw/sglang-step2-20260924'
REPO = HERE.parents[4]


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for f in manifest['files']:
        raw = (ROOT/f['path']).read_bytes()
        if len(raw) != f['bytes'] or hashlib.sha256(raw).hexdigest() != f['sha256']:
            raise ValueError('raw manifest mismatch: '+f['path'])
    audit = Audit(REPO)
    tasks = [audit.task(ROOT/('astropy__astropy-'+t), 'sglang') for t in ('12907','13033')]
    for row in tasks:
        meta = json.loads((ROOT/row['task']/'runner_metadata.json').read_text())
        ev = json.loads((ROOT/row['task']/'eval/eval_report.json').read_text())
        row['patch_bytes'] = meta['patch_bytes']
        row['failure_mode'] = ev['failure_mode']
        row['synthetic_no_patch'] = ev.get('synthetic_no_patch', False)
        row['harness_invoked'] = ev.get('harness_invoked', not ev.get('synthetic_no_patch', False))
        row['agent_exit_code'] = meta['agent']['exit_code']
        row['agent_budget_capped'] = meta['agent'].get('budget_capped', False)
    fields = ('output_tokens','completed_requests','request_latency_sum_s','ttft_sum_s','agent_elapsed_s')
    totals = {k:sum(row[k] for row in tasks) for k in fields}
    totals.update(rates(totals))
    result = {
        'schema':'sglang-step2-shared-task-rate-v1', 'status':'PASS',
        'scope':'Two shared tasks12907/13033 from the earlier SGLang EAGLE3/1/4 run. Descriptive completed-request pooled rate, not agent wall throughput or quality-adjusted throughput. Failed empty-patch task13033 retained. Original notes identify a capped runaway response; agent budget_capped false does not mean response cap absent.',
        'formula':'(sum output tokens - completed requests)/(sum E2E request latency - sum TTFT)',
        'population':'Both brackets have idle running/queue boundaries, aligned generated/completed token and request histograms, and zero nonstreaming metric deltas. Task13236 omitted: nonstreaming contribution makes its all-request TTFT clock unsuitable for this estimator;task13398 unfinished. No selection on response finish reasons.',
        'tasks':tasks, 'totals':totals,
        'raw_manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
        'input_sha256':dict(sorted(audit.hashes.items())),
        'common_reducer_sha256':hashlib.sha256((HERE/'shared_rate_reduce.py').read_bytes()).hexdigest(),
        'reducer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_inference_launched':False,
    }
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
