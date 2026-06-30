#!/usr/bin/env python3
"""Reduce the block-sweep TSV -> resolve-rate + char-8-rate by block + Fisher exact."""
import sys, csv
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
agg = defaultdict(lambda: {'n': 0, 'resolved': 0, 'char8': 0, 'block': '?'})
for r in rows:
    a = agg[r['arm']]
    a['n'] += 1
    a['block'] = r['block']
    a['resolved'] += (r['verdict'] == 'resolved')
    a['char8'] += (int(r['char8']) > 0)

order = ['OFF', 'ON1024', 'ON2048', 'ON4096', 'ON8192']
print("\n=== FR13 APC BLOCK-SWEEP: resolve-rate by block (the 'is 1024 a bad zone' test) ===")
for arm in order:
    if arm in agg:
        a = agg[arm]
        blk = 'native~816' if a['block'] in ('0', '?') else a['block']
        print(f"  {arm:8s} block={blk:11s}: resolved {a['resolved']}/{a['n']} ({a['resolved']/a['n']:.0%})  "
              f"char8 {a['char8']}/{a['n']} ({a['char8']/a['n']:.0%})")
try:
    from scipy.stats import fisher_exact
    def fish(x, y, key='resolved'):
        if x in agg and y in agg and agg[x]['n'] and agg[y]['n']:
            ax, ay = agg[x], agg[y]
            _, p = fisher_exact([[ax[key], ax['n'] - ax[key]], [ay[key], ay['n'] - ay[key]]])
            return p
        return None
    p = fish('ON1024', 'ON8192')
    if p is not None:
        print(f"  Fisher resolve ON1024 vs ON8192: p={p:.3f} => {'BLOCK LEVER REAL (bigger block helps)' if p < 0.05 else 'no significant block effect (1024 not a bad zone)'}")
    p = fish('ON8192', 'OFF')
    if p is not None:
        print(f"  Fisher resolve ON8192 vs OFF:    p={p:.3f} => {'8192-cache MATCHES no-cache' if p > 0.05 else 'big-block cache still below OFF'}")
    p = fish('ON1024', 'OFF')
    if p is not None:
        print(f"  Fisher resolve ON1024 vs OFF:    p={p:.3f} => {'1024-cache differs from OFF' if p < 0.05 else 'even 1024 not sig below OFF'}")
except Exception as e:
    print(f"  (scipy unavailable: {e})")
print("=" * 60)
