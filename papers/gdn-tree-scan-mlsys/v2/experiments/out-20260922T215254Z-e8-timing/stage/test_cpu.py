#!/usr/bin/env python3
"""Bounded no-GPU controls for admission, exact support, seals and frozen analysis."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import timing_verify as tv
from aggregate import calculate, stream_comparisons
from ownership import assess, verify as ownership_verify

HERE=Path(__file__).resolve().parent


def main():
    results={}
    def rejected(name,fn):
        try:fn()
        except (ValueError,FileNotFoundError,KeyError):results[name]='REFUSED_AS_REQUIRED'
        else:raise AssertionError(name)
    m=tv.stage(HERE);results['source_manifest']='PASS'
    assert [c['arm'] for c in m['cells']]==['off','on','on','off','off','on']
    assert m['qualification_manifest_sha256']=='bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030'
    results['six_cells_corrected_qualification_pin']='PASS'
    good={'schema':'e8.head.v1','closed':True,'failures':[],'qualify':False,'arm':'on','proposals':10,'owner_pid':22,
          'primary_head_calls':50,'legacy_head_calls':0,'qualified_proposals':0,'checked_heads':0,'root_heads':0,'loop_heads':0,'dispatch_ops_checked':0}
    tv.validate_head(good,'on');off=dict(good,arm='off',legacy_head_calls=50);tv.validate_head(off,'off');results['clean_5_vs_10_census']='PASS'
    for key,val in [('closed',False),('qualify',True),('primary_head_calls',49),('legacy_head_calls',1),('proposals',0),('owner_pid',None),('dispatch_ops_checked',1),('checked_heads',1),('failures',['bad'])]:
        d=dict(good);d[key]=val;rejected('head_'+key,lambda d=d:tv.validate_head(d,'on'))
    d=dict(off,legacy_head_calls=49);rejected('off_incomplete_legacy',lambda:tv.validate_head(d,'off'))
    cells={(b,a):{'status':'VALID','rate':r,'streams':{'p':[1,2,b if a=='on' else 4]}} for b,pair in enumerate([(10,11),(20,24),(40,36)],1) for a,r in zip(('off','on'),pair)}
    z=calculate(cells);assert abs(z['primary_mean_paired_relative_difference']-1/15)<1e-14
    assert abs(z['diagnostic_ratio_of_arm_means_minus_one']-(71/70-1))<1e-14
    assert z==calculate(cells) and z['primary_95pct_percentile_interval'][0]<=1/15<=z['primary_95pct_percentile_interval'][1]
    results['paired_estimator_distinct_from_ratio_of_means_and_deterministic_ci']='PASS'
    for name,change in [('missing',None),('insufficient',{'status':'INSUFFICIENT_SUPPORT','rate':None}),('invalid',{'status':'FAILED','rate':40})]:
        d=copy.deepcopy(cells)
        if change is None:del d[(3,'off')]
        else:d[(3,'off')]=change
        out=calculate(d);assert out['primary_mean_paired_relative_difference'] is None and out['primary_95pct_percentile_interval'] is None
        results['no_subset_'+name]='PASS'
    sd=stream_comparisons(cells);assert sd['n']==9 and sd['n_different']>0;results['streams_diagnostic_only']='PASS'
    raw={'inspect':{'returncode':0,'stdout':json.dumps([{'Id':'a'*64,'State':{'Running':True},'Config':{'Labels':{'lumo.e8.run':'run'}}}])},
         'containers':{'returncode':0,'stdout':'a'*64+'\n'},'gpu':{'returncode':0,'stdout':'123\n'},'top':{'returncode':0,'stdout':'PID\n123\n124\n'}}
    assert assess(raw,'a'*64,'run')['gpu_host_pids']==[123];results['ownership_host_pid_positive']='PASS'
    for name,key,field,value in [('foreign_gpu','gpu','stdout','999\n'),('foreign_container','containers','stdout','a'*64+'\n'+'b'*64+'\n'),
                                ('query_error','gpu','returncode',1),('empty_gpu','gpu','stdout',''),('namespace_pid_mismatch','top','stdout','PID\n1\n2\n')]:
        d=copy.deepcopy(raw);d[key][field]=value;rejected('ownership_'+name,lambda d=d:assess(d,'a'*64,'run'))
    rejected('ownership_wrong_label',lambda:assess(raw,'a'*64,'other'))
    with tempfile.TemporaryDirectory(prefix='e8-timing-cpu-') as tmp:
        t=Path(tmp);q=t/'qualification';q.mkdir()
        own=t/'ownership';own.mkdir()
        rejected('ownership_missing_receipt',lambda:ownership_verify(own))
        oi={'status':'PASS','container_id':'a'*64,'run_id':'run','workload_start_monotonic':2,'workload_end_monotonic':3,'visibility_limit':'sampled'}
        osamples=[{'status':'PASS','start_monotonic':a,'end_monotonic':b,'raw':raw,'identity':assess(raw,'a'*64,'run')} for a,b in [(0,1),(4,5)]]
        (own/'ownership_receipt.json').write_text(json.dumps(oi));(own/'ownership_samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in osamples))
        (own/'docker_inspect.json').write_text(raw['inspect']['stdout'])
        assert ownership_verify(own)['samples']==2;results['ownership_bracketed_receipt']='PASS'
        oi['workload_end_monotonic']=6;(own/'ownership_receipt.json').write_text(json.dumps(oi))
        rejected('ownership_nonbracketing_receipt',lambda:ownership_verify(own))
        rejected('missing_live_qualification',lambda:tv.qualification(HERE,q))
        (q/'FAILED.json').write_text('{}');rejected('failed_qualification',lambda:tv.qualification(HERE,q));(q/'FAILED.json').unlink()
        (q/'QUALIFICATION_PASS.json').write_text(json.dumps({'status':'QUALIFICATION_PASS','arms':['on']}))
        rejected('one_arm_qualification',lambda:tv.qualification(HERE,q))
        (q/'QUALIFICATION_PASS.json').write_text(json.dumps({'status':'QUALIFICATION_PASS','arms':['on','off'],'stage_manifest_sha256':'0'*64}))
        rejected('wrong_qualification_source',lambda:tv.qualification(HERE,q))
        # Real archived E1 token/interval fixture; this tests only the pure CPU
        # reducer, not its E8 configuration/head/ownership gates. Their positive
        # qualification remains live and separate; do not present old timing as E8.
        old=HERE.parents[1]/'experiments/out-20260922T100028Z-e1-18cells/cell_05_b1_tree_B1_a1'
        r=t/'cell';r.mkdir();(r/'logs').mkdir();shutil.copytree(old/'cohort',r/'cohort')
        for src,dst in [('e1_join.json','join.json'),('e1_api_tokens.json','api_tokens.json'),('logs/e1_events.jsonl','logs/e1_events.jsonl')]:shutil.copyfile(old/src,r/dst)
        (r/'logs/e8_head_gate.json').write_text(json.dumps(good));(r/'logs/e8_head_owner.json').write_text(json.dumps({'pid':22}))
        with patch.object(tv,'config',return_value=m),patch('ownership.verify',return_value={'test_fixture_only':True}):
            reduced=tv.reduce_cell(HERE,r,'on');j=json.loads((r/'join.json').read_text())
            assert reduced['rate']==j['tokens_per_wall_second'] and reduced['api_bound_tokens']==j['sum_emitted_tokens_api_bound_pure_support']
            assert sum(x['intervals'] for x in reduced['per_prefix'].values())==j['n_usable'];results['archived_raw_API_interval_reconstruction']='PASS'
            zero=dict(j,steps=[],sum_emitted_tokens_api_bound_pure_support=0,sum_wall_s_unique_physical_steps=0,n_usable=0)
            (r/'join.json').write_text(json.dumps(zero));z=tv.reduce_cell(HERE,r,'on')
            assert z['status']=='INSUFFICIENT_SUPPORT' and z['rate'] is None and len(z['per_prefix'])==8
            results['zero_support_preserved_with_full_API']='PASS';(r/'join.json').write_text(json.dumps(j))
            cell=m['cells'][1];tv.seal_cell(HERE,r,cell,reduced,'q'*64)
            assert tv.sealed_cell(HERE,r,cell,'q'*64)==reduced;results['terminal_seal_raw_reconstruction']='PASS'
            rejected('duplicate_terminal_seal',lambda:tv.seal_cell(HERE,r,cell,reduced,'q'*64))
            (r/'new_unsealed_file').write_text('changed');rejected('unsealed_raw_change',lambda:tv.sealed_cell(HERE,r,cell,'q'*64));(r/'new_unsealed_file').unlink()
            (r/'FAILED.json').write_text('{}');rejected('failure_after_valid_prefix',lambda:tv.sealed_cell(HERE,r,cell,'q'*64));(r/'FAILED.json').unlink()
            seal=json.loads((r/'cell_result.json').read_text());seal['terminal_seal']=False;(r/'cell_result.json').write_text(json.dumps(seal))
            rejected('preliminary_is_not_terminal',lambda:tv.sealed_cell(HERE,r,cell,'q'*64))
    # Plan mode and shell syntax cannot create any GPU process or artifact root.
    subprocess.run([sys.executable,'-B',str(HERE/'timing_driver.py'),'--repo','/tmp/mock','--run','/tmp/e8-never-created-plan'],check=True,stdout=subprocess.DEVNULL)
    assert not Path('/tmp/e8-never-created-plan').exists();results['default_plan_no_execution']='PASS'
    subprocess.run(['bash','-n',str(HERE/'timing_launch.v1.sh')],check=True)
    for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    results['source_compile']='PASS'
    out={'status':'PASS','count':len(results),'results':results,'qualification_live_gate':'NOT_RUN; absent receipt refusal tested'}
    dest=HERE/'tests_out/cpu_controls.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
