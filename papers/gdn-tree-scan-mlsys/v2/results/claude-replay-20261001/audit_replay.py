#!/usr/bin/env python3
"""Recompute the paper's replay rates from a frozen run selection and raw JSONL.

This is a data reducer, not a synthetic test. No inference or networking occurs.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    selection = json.loads((root / 'RUN-SELECTION.json').read_text())
    corpus = json.loads((root / 'CORPUS-MANIFEST.json').read_text())
    expected = {Path(r['path']).name for r in corpus}
    assert len(expected) == 43
    sync = json.loads((root / 'SYNC-MANIFEST.json').read_text())
    for rec in sync:
        path = root / rec['path']
        assert path.stat().st_size == rec['bytes'], rec['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256'], rec['path']
    arms, prompts = {}, {}
    for arm, names in selection['arms'].items():
        runs = []
        for name in names:
            p = root / 'raw/replay' / name / 'replay.jsonl'
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            assert len(rows) == 43, name
            assert {r['request'] for r in rows} == expected, name
            assert len({r['request'] for r in rows}) == len(rows), name
            assert all(not r.get('error') and r['mode'] == 'sampled'
                       and r['max_tokens'] == 1024 for r in rows), name
            assert all(isinstance(r['completion_tokens'], int) and r['completion_tokens'] > 0
                       and isinstance(r['t_decode_s'], (int, float))
                       and math.isfinite(r['t_decode_s']) and r['t_decode_s'] > 0 for r in rows), name
            drafts = accepted = 0.0
            negatives = []
            for r in rows:
                prompts.setdefault(r['request'], {})[name] = r['prompt_tokens']
                for key, value in r.get('spec_delta', {}).items():
                    if key.startswith('vllm:') and value < 0:
                        negatives.append([r['request'], key, value])
                    if key.startswith('vllm:spec_decode_num_drafts{'):
                        drafts += value
                    if key.startswith('vllm:spec_decode_num_accepted_tokens{'):
                        accepted += value
            assert not negatives, (name, negatives)
            nt = sum(r['completion_tokens'] for r in rows)
            ts = sum(r['t_decode_s'] for r in rows)
            runs.append({'name': name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'requests': len(rows), 'completion_tokens': nt, 'interval_tokens': nt - len(rows),
                         'decode_seconds': ts, 'pooled_rate': (nt - len(rows)) / ts,
                         'drafts': drafts, 'accepted_drafts': accepted,
                         'output_acceptance_plus_one': 1 + accepted / drafts if drafts else None,
                         'negative_vllm_counter_deltas': negatives,
                         'cap_hits': sum(r['completion_tokens'] == 1024 for r in rows),
                         'prompt_token_min': min(r['prompt_tokens'] for r in rows),
                         'prompt_token_max': max(r['prompt_tokens'] for r in rows),
                         'first_wall_utc': rows[0]['wall_utc'], 'last_wall_utc': rows[-1]['wall_utc']})
        d = sum(r['drafts'] for r in runs)
        arms[arm] = {'runs': runs,
                     'pooled_rate': sum(r['interval_tokens'] for r in runs) / sum(r['decode_seconds'] for r in runs),
                     'run_mean': statistics.mean(r['pooled_rate'] for r in runs),
                     'run_sd': statistics.stdev(r['pooled_rate'] for r in runs) if len(runs) > 1 else None,
                     'acceptance_plus_one': 1 + sum(r['accepted_drafts'] for r in runs) / d if d else None}
    result = {'schema': 'lumo.replay.audit.v1', 'arms': arms, 'requests_matched_per_run': 43,
              'accepted_primary_runs': sum(len(v['runs']) for v in arms.values()),
              'prompt_token_mismatches': {k: v for k, v in prompts.items() if len(set(v.values())) > 1},
              'ratios': {'tree_over_mtp5': arms['lumotree']['pooled_rate'] / arms['mtp5']['pooled_rate'],
                         'sglang7_over_tree': arms['sglang7']['pooled_rate'] / arms['lumotree']['pooled_rate'],
                         'tree_over_stock': arms['lumotree']['pooled_rate'] / arms['stockfa2']['pooled_rate']}}
    (root / 'AUDITED-RATES.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Verified {len(sync)} synced files, {result['accepted_primary_runs']} runs, 43 requests per run")
    for k, v in arms.items():
        print(f"{k}: {v['pooled_rate']:.4f} tokens/s ({len(v['runs'])} runs)")


if __name__ == '__main__':
    main()
