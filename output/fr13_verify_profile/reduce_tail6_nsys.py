#!/usr/bin/env python3
"""Single-arm nsys reducer for the tail6 21-node verify-forward profile.

Buckets per-kernel GPU time (within the probe walls) into the categories the
chain+leaf rewrite decision needs:
  gdn_tree_scan      _tree_gdn_kernel        -- the verify GDN scan (rewrite target 1)
  gdn_replay         _tree_gdn_replay        -- committer replay (overlap target)
  tree_conv_aux      fused_tree_conv / ex2_silu / conv taps / linear_remap
  attention          flash / fmha / attn (FA2 tree-bias fork)
  gemm_mlp_proj      cutlass fp8 blockwise + bf16 wmma (MLP/proj GEMMs)
  lm_head            cublas gemvx class
  norm_elementwise   rmsnorm / layernorm / elementwise / act
  sampler_committer  multidraft / rejection / sample
  other              everything else
Also reports: gpu_busy vs probe-wall time (launch/host gap), runtime API totals,
memcpy traffic. Normalizes per draft (spec_drafts summed over window probes).

Usage: python3 reduce_tail6_nsys.py [armdir]   (default: nsys_tail6_21node)
Derived from output/fr13_b1_kernel_attrib/analyze_kernels.py (two-arm variant).
"""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMDIR = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'nsys_tail6_21node')

CATS = [
    ('gdn_replay', ('_tree_gdn_replay', 'tree_gdn_replay')),
    ('gdn_tree_scan', ('_tree_gdn', 'tree_gdn')),
    ('tree_conv_aux', ('fused_tree_conv', 'ex2_silu', 'conv_taps',
                       'linear_remap', 'causal_conv')),
    ('attention', ('flash', 'fmha', 'attn')),
    ('lm_head', ('gemvx',)),
    ('gemm_mlp_proj', ('cutlass', 'wmma', 'gemm', 'nvjet')),
    ('norm_elementwise', ('rmsnorm', 'layernorm', 'norm', 'elementwise',
                          'act_and_mul', 'silu', 'rotary', 'vectorized')),
    ('sampler_committer', ('multidraft', 'rejection', 'sample', 'topk',
                           'softmax', 'argmax')),
]


def bucket(sn: str, dn: str) -> str:
    s = (sn + ' ' + dn).lower()
    for cat, pats in CATS:
        if any(p in s for p in pats):
            return cat
    return 'other'


def ensure_sqlite(rep: Path) -> Path:
    sql = rep.with_suffix('.sqlite')
    if not sql.exists():
        subprocess.run(['nsys', 'export', '--type', 'sqlite', '-o', str(sql),
                        str(rep)], check=True)
    return sql


def main() -> None:
    walls = [json.loads(l) for l in
             (ARMDIR / 'probe_walls.jsonl').read_text().splitlines() if l.strip()]
    assert walls, 'no probe walls recorded'
    drafts = 0.0
    for w in walls:
        p = ARMDIR / f"window_probe_r{w['r']}.json"
        drafts += json.load(open(p))['summary']['spec_drafts']
    assert drafts > 0, 'no spec drafts in window probes'

    rep = next((ARMDIR / 'logs').glob('*.nsys-rep'))
    con = sqlite3.connect(ensure_sqlite(rep))
    (epoch_ns,) = con.execute(
        'SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()

    cat_tot: dict[str, float] = {}
    cat_cnt: dict[str, int] = {}
    kern_tot: dict[str, float] = {}
    gpu_busy = 0.0
    wall_total_s = 0.0
    api_tot: dict[str, tuple[int, float]] = {}
    mem: dict[str, tuple[int, float, int]] = {}

    for w in walls:
        t0 = (w['wall_start'] - 0.5) * 1e9 - epoch_ns
        t1 = (w['wall_end'] + 0.5) * 1e9 - epoch_ns
        t0 = max(t0, 0.0)
        wall_total_s += w['wall_end'] - w['wall_start']
        q = '''SELECT s.value, d.value, COUNT(*), SUM(k.end-k.start)
               FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON s.id = k.shortName
               JOIN StringIds d ON d.id = k.demangledName
               WHERE k.start >= ? AND k.end <= ?
               GROUP BY s.value, d.value'''
        for sn, dn, cnt, tot in con.execute(q, (t0, t1)):
            cat = bucket(sn, dn)
            cat_tot[cat] = cat_tot.get(cat, 0.0) + tot
            cat_cnt[cat] = cat_cnt.get(cat, 0) + cnt
            key = f'{cat} :: {sn[:70]}'
            kern_tot[key] = kern_tot.get(key, 0.0) + tot
            gpu_busy += tot
        qa = '''SELECT s.value, COUNT(*), SUM(r.end-r.start)
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON s.id = r.nameId
                WHERE r.start >= ? AND r.end <= ? GROUP BY s.value'''
        for name, cnt, tot in con.execute(qa, (t0, t1)):
            k = name.split('_v')[0]
            c0, t00 = api_tot.get(k, (0, 0.0))
            api_tot[k] = (c0 + cnt, t00 + tot)
        tabs = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tabs:
            qm = '''SELECT copyKind, COUNT(*), SUM(end-start), SUM(bytes)
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    WHERE start >= ? AND end <= ? GROUP BY copyKind'''
            for kind, cnt, tot, byts in con.execute(qm, (t0, t1)):
                c0, t00, b0 = mem.get(f'kind{kind}', (0, 0.0, 0))
                mem[f'kind{kind}'] = (c0 + cnt, t00 + tot, b0 + byts)

    out = {
        'drafts_in_window': drafts,
        'probe_wall_s_total': wall_total_s,
        'gpu_kernel_busy_ms': gpu_busy / 1e6,
        'launch_host_gap_ms': wall_total_s * 1e3 - gpu_busy / 1e6,
        'per_draft_ms_by_category': {
            k: round(v / 1e6 / drafts, 4)
            for k, v in sorted(cat_tot.items(), key=lambda kv: -kv[1])},
        'launch_count_per_draft_by_category': {
            k: round(cat_cnt[k] / drafts, 2)
            for k in sorted(cat_cnt, key=lambda k: -cat_tot[k])},
        'memcpy': {k: dict(count=c, ms=round(t / 1e6, 3), bytes=b)
                   for k, (c, t, b) in mem.items()},
    }
    json.dump(out, open(ARMDIR / 'verify_split.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
    print('\ntop kernels (ms/draft):')
    for k, v in sorted(kern_tot.items(), key=lambda kv: -kv[1])[:30]:
        print(f'  {v / 1e6 / drafts:9.4f}  {k}')
    print('\ntop runtime APIs (ms/draft):')
    for k, (c, t) in sorted(api_tot.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f'  {t / 1e6 / drafts:9.4f}  n/draft={c / drafts:8.2f}  {k}')


if __name__ == '__main__':
    main()
