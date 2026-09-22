#!/usr/bin/env python3
"""Frozen six-cell analysis. No subsets, replacements, precision extension or GPU."""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
from timing_verify import check, sha, stage, qualification, sealed_cell


def calculate(cells):
    """Cells keyed by (block,arm); missing/insufficient support stays unavailable."""
    expected={(b,a) for b in (1,2,3) for a in ('on','off')}
    check(set(cells)<=expected,'unexpected or extra analysis cell')
    rows=[];relative=[]
    for b in (1,2,3):
        on=cells.get((b,'on'),{});off=cells.get((b,'off'),{})
        valid=all(c.get('status')=='VALID' and isinstance(c.get('rate'),(float,int)) and math.isfinite(c['rate']) and c['rate']>0 for c in (on,off))
        d=(on['rate']-off['rate'])/off['rate'] if valid else None
        rows.append({'block':b,'rate_on':on.get('rate'),'rate_off':off.get('rate'),'paired_relative_difference':d,
                     'on_status':on.get('status','MISSING'),'off_status':off.get('status','MISSING')})
        if valid:relative.append(d)
    out={'paired_blocks':rows,'replication_unit':'paired block','n_prespecified_blocks':3,'n_complete_valid_blocks':len(relative),
         'primary_mean_paired_relative_difference':None,'primary_95pct_percentile_interval':None,
         'diagnostic_ratio_of_arm_means_minus_one':None,'bootstrap_draws':10000,'bootstrap_seed':20260921,
         'status':'UNAVAILABLE_MISSING_FAILED_OR_INSUFFICIENT_CELL','no_extension_or_replacement':True}
    if len(relative)!=3:return out
    out['primary_mean_paired_relative_difference']=sum(relative)/3
    out['diagnostic_ratio_of_arm_means_minus_one']=sum(cells[(b,'on')]['rate'] for b in (1,2,3))/sum(cells[(b,'off')]['rate'] for b in (1,2,3))-1
    rng=random.Random(20260921)
    draws=sorted(sum(relative[rng.randrange(3)] for _ in range(3))/3 for _ in range(10000))
    def percentile(p):
        x=(len(draws)-1)*p;lo=math.floor(x);hi=math.ceil(x)
        return draws[lo]+(draws[hi]-draws[lo])*(x-lo)
    out['primary_95pct_percentile_interval']=[percentile(.025),percentile(.975)]
    out['status']='COMPLETE_FROZEN_ANALYSIS'
    return out


def stream_comparisons(cells):
    """Retrospective descriptive diagnostic; never a correctness/retention gate."""
    pairs=[]
    for b in (1,2,3):pairs.append(((b,'on'),(b,'off'),'within_pair_across_arms'))
    for arm in ('on','off'):
        for x,y in itertools.combinations((1,2,3),2):pairs.append(((x,arm),(y,arm),'across_blocks_within_arm'))
    rows=[]
    for a,b,kind in pairs:
        x=cells.get(a,{}).get('streams',{});y=cells.get(b,{}).get('streams',{})
        for prefix in sorted(set(x)&set(y)):
            s,t=x[prefix],y[prefix];n=min(len(s),len(t));first=next((i for i in range(n) if s[i]!=t[i]),None)
            if first is None and len(s)!=len(t):first=n
            rows.append({'kind':kind,'left':list(a),'right':list(b),'prefix':prefix,'equal':s==t,'left_tokens':len(s),
                         'right_tokens':len(t),'first_divergence_zero_based':first,'common_prefix_tokens':n if first is None else first})
    return {'scope':'descriptive only; matched prefixes do not imply identical continuations','comparisons':rows,
            'n':len(rows),'n_equal':sum(x['equal'] for x in rows),'n_different':sum(not x['equal'] for x in rows)}


def aggregate(source, root):
    source,root=Path(source),Path(root);m=stage(source);q=qualification(source,root/'qualification_evidence')
    started=json.loads((root/'CAMPAIGN_STARTED.json').read_text())
    check(started['timing_manifest_sha256']==sha(source/'timing_manifest.json') and started['cells']==m['cells'],'campaign start/freeze binding')
    check(started['qualification']['pass_sha256']==q['pass_sha256'],'campaign start/qualification binding')
    cells={};receipts={};missing=[]
    for cell in m['cells']:
        r=root/cell['name']
        if not (r/'cell_result.json').exists():missing.append(cell['name']);continue
        result=sealed_cell(source,r,cell,q['pass_sha256'])
        cells[(cell['block'],cell['arm'])]=result
        receipts[cell['name']]={'cell':cell,'result':result,'terminal_receipt_sha256':sha(r/'cell_result.json')}
    out=calculate(cells)
    if (root/'FAILED.json').exists():
        out.update(status='UNAVAILABLE_CAMPAIGN_FAILURE',primary_mean_paired_relative_difference=None,primary_95pct_percentile_interval=None,diagnostic_ratio_of_arm_means_minus_one=None)
    out.update(schema='e8.timing.aggregate.v1',timing_manifest_sha256=sha(source/'timing_manifest.json'),qualification=q,
               missing_cells=missing,cells=receipts,stream_diagnostic=stream_comparisons(cells))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--run',required=True);ap.add_argument('--output');a=ap.parse_args()
    out=aggregate(a.stage,a.run)
    if a.output:
        with Path(a.output).open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
