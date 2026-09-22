#!/usr/bin/env python3
"""Export the complete frozen E8 aggregate for the paper; CPU only, no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = '8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.run.resolve()
    stage = root / 'stage'
    assert sha(stage / 'timing_manifest.json') == EXPECTED, 'wrong timing freeze'
    assert not (root / 'FAILED.json').exists(), 'failed campaign'
    terminal = json.loads((root / 'CAMPAIGN_COMPLETE.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['timing_manifest_sha256'] == EXPECTED, 'campaign not terminal'
    assert terminal['aggregate_sha256'] == sha(root / 'aggregate.json'), 'terminal aggregate binding'
    assert not args.output.exists(), 'use a new output directory'
    sys.path.insert(0, str(stage))
    from aggregate import aggregate
    fresh = aggregate(stage, root)
    recorded = json.loads((root / 'aggregate.json').read_text())
    # The copied qualification root is the only absolute path remapped offline.
    mapped = json.loads(json.dumps(recorded))
    mapped['qualification']['root'] = fresh['qualification']['root']
    assert mapped == fresh, 'frozen raw reconstruction differs'
    assert fresh['status'] == 'COMPLETE_FROZEN_ANALYSIS', 'incomplete campaign'
    assert len(fresh['cells']) == 6 and fresh['n_complete_valid_blocks'] == 3
    args.output.mkdir(parents=True)
    (args.output / 'aggregate.json').write_text(json.dumps(recorded, indent=2) + '\n')
    cells = []
    for name, item in fresh['cells'].items():
        c, r = item['cell'], item['result']
        cells.append(dict(cell=name, block=c['block'], arm=c['arm'], rate=r['rate'],
                          api_bound_tokens=r['api_bound_tokens'], wall_s=r['unique_wall_s'],
                          intervals=r['n_intervals'], min_prefix_intervals=min(x['intervals'] for x in r['per_prefix'].values()),
                          head_proposals=r['head_census']['proposals'], primary_head_calls=r['head_census']['primary_head_calls'],
                          legacy_head_calls=r['head_census']['legacy_head_calls']))
    for filename, rows in [('cells.csv', cells), ('blocks.csv', fresh['paired_blocks'])]:
        with (args.output / filename).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    comparisons = fresh['stream_diagnostic']['comparisons']
    groups = {}
    for key in ('within_pair_across_arms', 'across_blocks_within_arm'):
        rows = [x for x in comparisons if x['kind'] == key]
        groups[key] = {'equal': sum(x['equal'] for x in rows), 'total': len(rows)}
    summary = dict(source_run=str(root), source_aggregate_sha256=sha(root / 'aggregate.json'),
                   timing_manifest_sha256=EXPECTED, qualification_pass_sha256=fresh['qualification']['pass_sha256'],
                   primary_mean_paired_relative_difference=fresh['primary_mean_paired_relative_difference'],
                   primary_95pct_percentile_interval=fresh['primary_95pct_percentile_interval'],
                   diagnostic_ratio_of_arm_means_minus_one=fresh['diagnostic_ratio_of_arm_means_minus_one'],
                   arm_mean_rates={a: sum(x['rate'] for x in cells if x['arm'] == a)/3 for a in ('on', 'off')},
                   stream_comparisons=groups,
                   offline_reconstruction='exact semantic match except explicit local qualification root mapping',
                   limits='fixed B1 prefixes, three paired blocks, retained-decode rate, sampled ownership; no general quality or full-model equivalence claim')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [r'% Generated only from the complete independently reviewable frozen E8 aggregate.',
             r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Block & Reuse OFF & Reuse ON & Paired change \\', r'\midrule']
    for row in fresh['paired_blocks']:
        lines.append(f"{row['block']} & {row['rate_off']:.3f} & {row['rate_on']:.3f} & {100*row['paired_relative_difference']:+.2f}\\% \\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (args.output / 'paired-rates.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
