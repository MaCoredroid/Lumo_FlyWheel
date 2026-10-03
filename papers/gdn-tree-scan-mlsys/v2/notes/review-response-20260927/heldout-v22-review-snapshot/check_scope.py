#!/usr/bin/env python3
"""Pure-stdlib source/metadata checks; never load fixture tensors or execute a candidate."""
import argparse,ast,collections,copy,hashlib,json,os,pathlib,subprocess
BASE=pathlib.Path(__file__).resolve().parent
V2=BASE.parents[2]
CAMP=V2/'experiments/review-response-20260927'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
P=BASE/'tools/q1_component_runner_v2_2.py'
f=ast.parse(P.read_text());node=next(n for n in f.body if isinstance(n,ast.FunctionDef) and n.name=='block_authorization')
ns={'__file__':str(P),'os':os,'HELD_OUT_BLOCK':'evaluation','SUPPORTED_BLOCKS':('calibration','evaluation'),'sha256_file':sha}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(P),'exec'),ns)
fm=json.loads((CAMP/'fixtures/q1_2b/manifest.json').read_text());pol=json.loads((CAMP/'policy/q1_component_numerical_policy.v2.1.json').read_text());obs=json.loads((CAMP/'fixtures/q1_2b/expected_observations.json').read_text())
ids=sorted(e['fixture_id'] for e in fm['fixtures'] if e['block']=='evaluation')
g={'reviewed_hashes':{'runner':sha(P)},'reviewed_scope':{'block':'evaluation','negatives':0,'held_out_fixture_ids':ids,'calibration_power_run_id':'q12b-calibration-retry-20260928T010045Z'}}
args=argparse.Namespace(block='evaluation',negatives=0,limit=0)
checks=[]
def test(name,mut=None,want=True):
 a,b,c,d=copy.deepcopy((g,pol,fm,args))
 if mut:mut(a,b,c,d)
 pr,facts=ns['block_authorization'](a,b,c,d)
 assert (not pr)==want,(name,pr)
 assert facts['complete_coverage']==want,(name,facts)
 checks.append({'name':name,'accepted':not pr,'problems':pr})
test('complete_frozen_evaluation_scope')
test('unsupported_block',lambda g,p,f,a:setattr(a,'block','test'),False)
test('wrong_gate_block',lambda g,p,f,a:g['reviewed_scope'].update(block='calibration'),False)
test('candidate_negatives_nonzero',lambda g,p,f,a:setattr(a,'negatives',1),False)
test('gate_negatives_nonzero',lambda g,p,f,a:g['reviewed_scope'].update(negatives=1),False)
test('candidate_limit_partial',lambda g,p,f,a:setattr(a,'limit',1),False)
test('gate_missing_fixture',lambda g,p,f,a:g['reviewed_scope'].update(held_out_fixture_ids=ids[:1]),False)
test('gate_duplicate_fixture',lambda g,p,f,a:g['reviewed_scope'].update(held_out_fixture_ids=ids+ids[:1]),False)
test('manifest_missing_fixture',lambda g,p,f,a:f.update(fixtures=[e for e in f['fixtures'] if e['fixture_id']!=ids[0]]),False)
test('manifest_duplicate_fixture',lambda g,p,f,a:f['fixtures'].append(next(e for e in f['fixtures'] if e['fixture_id']==ids[0])),False)
test('policy_missing_fixture',lambda g,p,f,a:p['domain']['q1_2b']['fixture_ids'].update(evaluation_held_out=ids[:1]),False)
test('wrong_runner_sha',lambda g,p,f,a:g['reviewed_hashes'].update(runner='0'*64),False)
test('missing_power_run',lambda g,p,f,a:g['reviewed_scope'].pop('calibration_power_run_id'),False)
test('empty_power_run',lambda g,p,f,a:g['reviewed_scope'].update(calibration_power_run_id=''),False)
test('nonempty_power_run_is_parent_authority_not_validated_here',lambda g,p,f,a:g['reviewed_scope'].update(calibration_power_run_id='does-not-exist'),True)
unchanged={}
for stem in ('q1_component_runner','q1_2b_reduce'):
 p1=BASE/'tools'/f'{stem}_v2_1.py';p2=BASE/'tools'/f'{stem}_v2_2.py'
 a1={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p1.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 a2={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p2.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 same=sorted(k for k in a1 if a1[k]==a2.get(k));changed=sorted(k for k in a1 if a1[k]!=a2.get(k));new=sorted(set(a2)-set(a1))
 assert changed==(['main'] if stem=='q1_component_runner' else ['reduce_run'])
 assert new==(['block_authorization'] if stem=='q1_component_runner' else [])
 unchanged[stem]={'unchanged_functions_classes':same,'changed':changed,'added':new}
cs=[c for c in obs['cases'] if c['fixture_id'] in ids]
assert len(cs)==5760 and len({c['case_id'] for c in cs})==5760
count=dict(collections.Counter(c['kind'] for c in cs));assert count=={'output':3072,'state':2688}
assert sum(c['heads']*len(c['depths'])*len(c['surfaces']) for c in cs)==276480
r=subprocess.run(['bash','-n',str(BASE/'tools/run_q1_2b_component_v2_2.sh')],capture_output=True,text=True);assert r.returncode==0,r.stderr
out={'scope':'pure stdlib controls and frozen metadata only, no Torch/GPU/fixture tensor/candidate','source_snapshot_sha256':sha(BASE/'SNAPSHOT.json'),'checks':checks,'functions':unchanged,'expected_ids':ids,'expected_cases':len(cs),'case_counts':count,'head_observations':276480,'bash_syntax_exit':r.returncode,'parent_power_authority_condition':'Parent must bind/copy/verify accepted calibration summary, terminal receipt and acceptance receipt in the trusted held-out gate; runner alone checks only a nonempty run ID.'}
print(json.dumps(out,indent=2))
