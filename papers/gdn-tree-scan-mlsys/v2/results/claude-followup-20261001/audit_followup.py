#!/usr/bin/env python3
"""Reduce preserved experiment records; does not run inference or unit tests."""
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'claude-replay-20261001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    vals = sorted(values)
    x = (len(vals) - 1) * p / 100
    lo, hi = math.floor(x), math.ceil(x)
    return vals[lo] + (vals[hi] - vals[lo]) * (x - lo)


def audit_timings(obj):
    count = 0
    if isinstance(obj, dict):
        if 'raw_ms' in obj and 'total' in obj:
            raw = obj['raw_ms']
            for values, summary in [(raw['total'], obj['total'])] + [
                    (values, obj['regions'][k]) for k, values in raw['regions'].items()]:
                assert len(values) == summary['n'] and values
                assert all(math.isfinite(v) and v > 0 for v in values)
                for key, p in [('median_ms', 50), ('p10_ms', 10), ('p90_ms', 90)]:
                    assert math.isclose(percentile(values, p), summary[key], abs_tol=1e-9), key
            count += 1
        for k, v in obj.items():
            if k != 'raw_ms':
                count += audit_timings(v)
    elif isinstance(obj, list):
        count += sum(audit_timings(v) for v in obj)
    return count


def scratch(method, acc):
    if method == 'lumo_fixed32':
        return acc['cut_state_buffer']['allocated_bytes'] + acc['replay_rings']['bytes'] + acc['committer_fixed16_staging']['bytes']
    if method.startswith('naive'):
        return acc['node_state_bank']['allocated_bytes'] + acc['commit_staging']['bytes']
    if method.startswith('treewy'):
        return acc['stashes_allocated']['bytes'] + acc['dfs_staging_and_remap']['bytes']
    return acc['stashes']['bytes'] + acc['verifier_workspace_cache']['bytes'] + acc['topology']['bytes']


def main():
    synced = json.loads((ROOT / 'SYNC-MANIFEST.json').read_text())
    for row in synced:
        f = ROOT / row['local_path']
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], row['local_path']
    ids_path = ROOT / 'raw/corpus/vllm_prompt_ids.json'
    ids = json.loads(ids_path.read_text())
    manifest = json.loads((ROOT / 'code/results/vllm_prompt_ids_manifest.json').read_text())
    assert sha(ids_path) == manifest['sha256'] and len(ids) == 43
    for rec in ids.values():
        assert rec['count'] == len(rec['ids'])
        assert hashlib.sha256(json.dumps(rec['ids']).encode()).hexdigest() == rec['sha256']
    selection = json.loads((OLD / 'RUN-SELECTION.json').read_text())
    for arm in [a for a in selection['arms'] if not a.startswith('sglang')]:
        for name in selection['arms'][arm]:
            for row in map(json.loads, (OLD / 'raw/replay' / name / 'replay.jsonl').read_text().splitlines()):
                assert row['prompt_tokens'] == ids[row['request']]['count']
    groups = {
        'stockfa2': [OLD / 'raw/replay' / n for n in selection['arms']['stockfa2']] + sorted((ROOT / 'raw/replay').glob('tree-fa2stock-*')),
        'sglang7_ids': sorted((ROOT / 'raw/replay').glob('sglang-s7k1d8-ids-*')),
    }
    replay = {}
    for arm, dirs in groups.items():
        runs = []
        for d in dirs:
            f = d / 'replay.jsonl'
            rows = list(map(json.loads, f.read_text().splitlines()))
            assert len(rows) == 43 and len({r['request'] for r in rows}) == 43
            assert {r['request'] for r in rows} == set(ids)
            for r in rows:
                assert not r.get('error') and r['max_tokens'] == 1024
                assert r['completion_tokens'] > 0 and math.isfinite(r['t_decode_s']) and r['t_decode_s'] > 0
                assert r['prompt_tokens'] == ids[r['request']]['count']
                if arm == 'sglang7_ids':
                    assert r['mode'] == 'sampled-ids' and r['prompt_ids_sha256'] == ids[r['request']]['sha256']
            n = sum(r['completion_tokens'] - 1 for r in rows)
            t = sum(r['t_decode_s'] for r in rows)
            runs.append({'name': d.name, 'sha256': sha(f), 'requests': 43, 'interval_tokens': n,
                         'decode_seconds': t, 'rate': n / t, 'cap_hits': sum(r['completion_tokens'] == 1024 for r in rows)})
        replay[arm] = {'runs': runs, 'pooled_rate': sum(r['interval_tokens'] for r in runs) / sum(r['decode_seconds'] for r in runs),
                       'run_min': min(r['rate'] for r in runs), 'run_max': max(r['rate'] for r in runs)}
    kernel = []
    timing_count = 0
    kp = ROOT / 'raw/kernel/20261001T205557Z'
    for f in sorted((kp / 'methods').glob('*.json')):
        d = json.loads(f.read_text())
        assert d['status'] == 'complete' and not d['smoke']
        assert d['args']['layers'] == 48 and d['args']['warmup'] == 20 and d['args']['repeats'] == 100
        assert d['pinned_sources']['pinned_ok'] and not d['pinned_sources']['mismatches']
        timing_count += audit_timings(d['timing'])
        assert d['numerics']['verify_output']['invalid_cells'] == 0
        for mode in ['graph', 'eager']:
            st = d['timing'][mode]['step']
            assert st['total']['n'] == 100
            kernel.append({'method': d['method'], 'mode': mode, 'sha256': sha(f), 'step_ms': st['total'],
                           'regions_ms': st['regions'], 'scratch_named_bytes': scratch(d['method'], d['memory']['accounting']),
                           'framework_peak_allocated_bytes': st['memory']['peak_allocated_bytes'],
                           'output_max_abs': d['numerics']['verify_output']['max_abs_max'],
                           'committed_n31_max_abs': d['numerics']['committed_state']['n31']['max_abs_max'],
                           'derived': d['derived'][mode], 'numerical_gate': False})
    sweep_count = 0
    sweep_summaries = 0
    for f in sorted((kp / 'sweep').glob('*.json')):
        d = json.loads(f.read_text())
        assert d['status'] == 'complete'
        sweep_count += audit_timings(d)
        def count_summaries(obj):
            if isinstance(obj, dict):
                return int('total' in obj and 'regions' in obj and obj['total'].get('n') == 50) + sum(count_summaries(v) for v in obj.values())
            if isinstance(obj, list):
                return sum(count_summaries(v) for v in obj)
            return 0
        sweep_summaries += count_summaries(d)
    distribution = []
    for f in sorted((ROOT / 'raw/dist').glob('*/dist.jsonl')):
        rows = list(map(json.loads, f.read_text().splitlines()))
        assert len(rows) == 800 and not any(r.get('error') for r in rows)
        counts = collections.Counter(r['request'] for r in rows)
        assert len(counts) == 20 and set(counts.values()) == {40}
        assert len({(r['request'], r['sample']) for r in rows}) == 800
        distribution.append({'name': f.parent.name, 'sha256': sha(f), 'samples': len(rows), 'requests': 20})
    verdict = json.loads((ROOT / 'code/results/dist_v2_verdict.json').read_text())
    assert verdict['rule1_validity_NULL_p_ge_0.01'] is False
    result = {'schema': 'lumo.followup.audit.v1', 'synced_files': len(synced), 'replay': replay,
              'kernel': kernel, 'main_step_timing_cells': len(kernel), 'all_main_timing_surfaces': timing_count,
              'verify_only_sweep_cells_raw_reduced': sweep_count, 'verify_only_sweep_cells_summary_only': sweep_summaries,
              'distribution': distribution, 'distribution_verdict': verdict,
              'cross_stack_timer_boundary': 'vLLM first parsed nonempty delta; SGLang first completion-count increase; not identical API observables',
              'fullmodel_qualification_completed': 0}
    (ROOT / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['synced_files', 'main_step_timing_cells', 'all_main_timing_surfaces', 'verify_only_sweep_cells_summary_only']}, indent=2))
    for k, v in replay.items():
        print(k, len(v['runs']), v['pooled_rate'], v['run_min'], v['run_max'])


if __name__ == '__main__':
    main()
