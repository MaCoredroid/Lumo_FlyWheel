#!/usr/bin/env python3
"""CPU-only reconstruction of the Cqc10 task ledger and request-mean TPOT.

Usage: python3 cqc10_reduce.py [path-to-raw/cqc10/hydra27_fixed32_promoab_Cqc10]
Prints deterministic JSON; never edits inputs. Does not infer per-request latency
samples or a causal speedup from aggregate histograms. Requires original metric
brackets, task metadata, evaluator reports, and stored deploy reduction.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / 'raw/cqc10/hydra27_fixed32_promoab_Cqc10'
METRICS = ('request_time_per_output_token_seconds_count',
           'request_time_per_output_token_seconds_sum', 'generation_tokens_total',
           'fr13_decode_forward_gpu_seconds_total', 'fr13_decode_forward_gpu_steps_total',
           'fr13_decode_step_wall_seconds_total', 'fr13_decode_step_wall_steps_total',
           'fr13_decode_step_wall_attempts_total', 'fr13_decode_step_wall_rejected_total')
KEYS = tuple('vllm:' + x for x in METRICS)
FILE_HASHES = {}


def load(path, as_json=True):
    data = path.read_bytes()
    FILE_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data) if as_json else data.decode('utf-8')


def snapshot(path):
    found = {}
    for line in load(path, False).splitlines():
        if not line or line.startswith('#'):
            continue
        m = re.fullmatch(r'([^\s{]+)(\{[^}]*\})?\s+(\S+)(?:\s+\S+)?', line)
        if not m or m.group(1) not in KEYS:
            continue
        key, labels, raw = m.group(1), m.group(2), m.group(3)
        if labels is not None and labels != '{engine="0",model_name="qwen3.8-27b-nvfp4-radixark"}':
            raise ValueError(f'unexpected metric labels in {path.name}: {key} {labels}')
        if key in found:
            raise ValueError(f'duplicate metric in {path.name}: {key}')
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid metric in {path.name}: {key}')
        found[key] = value
    if set(found) != set(KEYS):
        raise ValueError(f'missing metric in {path}: {set(KEYS)-set(found)}')
    return found


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError(f'{label}: {a} != {b}')


def main():
    deploy = load(ROOT / 'deploy_speed_promoab_C.json')
    health = load(ROOT / 'health.json')
    if health['swe_orchestrator_rc'] != 0 or deploy['n_tasks'] != 10:
        raise ValueError('not the completed ten-task cohort')
    expected_ids = ['astropy__astropy-' + x for x in ('13977','14096','14182','14309','14365','14369','14508','14539','14598','14995')]
    if deploy['task_instance_ids'] != expected_ids:
        raise ValueError('cohort or order changed')
    rows, totals, previous = [], dict.fromkeys(KEYS, 0.0), None
    normal = compact = completed = stop = length = failed_compact = 0
    for task in expected_ids:
        p = ROOT / 'swe_out/verified/per_task' / task
        before, after = snapshot(p/'vllm_metrics_pre.txt'), snapshot(p/'vllm_metrics_post.txt')
        if previous is not None and before != previous:
            raise ValueError(f'non-contiguous selected metric origins at {task}')
        previous = after
        delta = {k:after[k]-before[k] for k in KEYS}
        if any(v < 0 for v in delta.values()):
            raise ValueError(f'counter reset at {task}')
        for k,v in delta.items():
            totals[k] += v
        meta, evaluation = load(p/'runner_metadata.json'), load(p/'eval/eval_report.json')
        if meta['instance_id'] != task or evaluation['instance_id'] != task:
            raise ValueError('task identity mismatch')
        ev = meta['fixed32_real_task_provenance']['qwen_compaction_metric_evidence']
        count = delta[KEYS[0]]
        close(count, ev['completed_engine_requests'], 'TPOT vs completed engine requests')
        close(count, ev['normal_requests']+ev['total_compaction_requests'], 'request classes')
        close(count, ev['request_success_stop']+ev['request_success_non_stop'], 'completion classes')
        if ev['normal_visible_max_output_tokens'] != 24000 or ev['compaction_max_output_tokens'] != 20000:
            raise ValueError('output ceiling changed')
        normal += ev['normal_requests']; compact += ev['successful_compaction_requests']
        completed += ev['completed_engine_requests']; stop += ev['request_success_stop']
        length += ev['request_success_length']; failed_compact += ev['failed_compaction_requests']
        if meta['agent']['timed_out'] or meta['agent'].get('budget_capped', False):
            raise ValueError('unexpected incomplete task')
        rows.append({'instance_id':task, 'agent_elapsed_s':meta['agent']['elapsed_s'],
                     'eval_verdict':evaluation['verdict'], 'tpot_observation_count':count,
                     'tpot_observation_sum_seconds':delta[KEYS[1]],
                     'normal_requests':ev['normal_requests'],
                     'successful_compaction_requests':ev['successful_compaction_requests'],
                     'raw_metric_delta':delta})
    for k,v in totals.items():
        if k in deploy['raw_counter_delta_aggregate']:
            close(v, deploy['raw_counter_delta_aggregate'][k], 'stored raw delta '+k)
    inverse = totals[KEYS[0]]/totals[KEYS[1]]
    wall_ms = 1000*totals[KEYS[5]]/totals[KEYS[6]]
    close(inverse,deploy['per_request_decode_tps'],'inverse request mean TPOT')
    close(wall_ms,deploy['step_wall_ms'],'mean retained physical interval')
    result={'schema':'cqc10-independent-metric-audit-v1',
            'scope':'One completed ten-task Astropy continuation segment; descriptive, no matched baseline or causal speedup.',
            'source_relative_root':'raw/cqc10/hydra27_fixed32_promoab_Cqc10',
            'reducer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'metric_support':'Completed engine TPOT observations, including internal compaction; request-weighted, not token-weighted.',
            'all_selected_metric_brackets_contiguous':True,
            'completed_engine_requests':completed, 'normal_requests':normal,
            'successful_compaction_requests':compact, 'failed_compaction_requests':failed_compact,
            'completion_reasons':{'stop':stop,'length':length},
            'mean_request_tpot_ms':1000/inverse,'inverse_mean_request_tpot_tokens_per_s':inverse,
            'agent_elapsed_sum_s':sum(t['agent_elapsed_s']for t in rows),
            'resolved':sum(t['eval_verdict']=='resolved'for t in rows),
            'failed':sum(t['eval_verdict']=='failed'for t in rows),
            'all_attempts':10,'raw_metric_totals':totals,
            'retained_physical_interval_mean_ms':wall_ms,
            'tasks':rows,'source_file_sha256':dict(sorted(FILE_HASHES.items()))}
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
