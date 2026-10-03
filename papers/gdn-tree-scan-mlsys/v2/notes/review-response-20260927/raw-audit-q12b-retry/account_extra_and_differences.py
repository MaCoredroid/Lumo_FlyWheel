#!/usr/bin/env python3
"""Read-only metadata accounting: no tensor reads, Torch, Docker, or GPU."""
import collections, hashlib, json, pathlib
R=pathlib.Path('/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/runs/q1.2b/q12b-calibration-retry-20260928T010045Z')
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
a=load(R/'procA/result.json'); b=load(R/'procB/result.json'); pol=load(R/'policy.snapshot.json')
obs={c['case_id']:c for c in load(R/'expected_observations.snapshot.json')['cases']}
refs=set(); differing=[]; byfixture=collections.Counter(); bydepth=collections.Counter(); bynode=collections.Counter()
k,u,eta=(pol['constants'][x] for x in ('kappa','u32','eta32'))
for e in map(json.loads,(R/'procA/raw_inventory.jsonl').read_text().splitlines()):
 rec=load(R/'procA'/e['path'])
 for field in ('candidate_tensor','native_tensor'):
  if field in rec: refs.add(rec[field]['sha256'])
 if e['kind']=='metrics' and rec['candidate_sha256']!=rec['native_sha256']:
  c=obs[e['case_id']]; assert c['kind']=='output'
  byfixture[c['fixture_id']]+=1; bydepth[str(c['depths'][0])]+=1; bynode[str(c['node'])]+=1
  rr=[v/(k*w+u*x+eta) for v,w,x in zip(rec['rms_C0_C2'],rec['rms_C1_C2'],rec['rms_C2'])]
  mr=[v/(k*w+u*x+eta) for v,w,x in zip(rec['maxabs_C0_C2'],rec['maxabs_C1_C2'],rec['maxabs_C2'])]
  assert max(rr)<=1 and max(mr)<=1
  differing.append({'case_id':e['case_id'],'depth':c['depths'][0], 'candidate_sha256':rec['candidate_sha256'],'native_sha256':rec['native_sha256'], 'raw_record_path':e['path'],'raw_file_sha256':e['file_sha256'],'max_rms_fraction_of_frozen_bound':max(rr),'max_maxabs_fraction_of_frozen_bound':max(mr)})
objects={p.stem:p.stat().st_size for p in (R/'tensors').glob('*.bin')}
extra=set(objects)-refs; native_extra=[]
for f in a['fixtures']:
 for l,states in enumerate(f['native_repeats'][0]['state_sha256']):
  for node,h in states.items():
   if h in extra: native_extra.append({'fixture_id':f['fixture_id'],'instance':l,'node':int(node),'sha256':h,'bytes':objects[h]})
assert {x['sha256'] for x in native_extra}==extra
assert len(extra)==len(native_extra)==384
assert all(x['bytes']==3145728 for x in native_extra)
assert len(differing)==68
assert {c['case_id']:c['candidate_sha256'] for f in a['fixtures'] for c in f['cases']}=={c['case_id']:c['candidate_sha256'] for f in b['fixtures'] for c in f['cases']}
print(json.dumps({'run_id':R.name,'input_sha256':{'procA/result.json':sha(R/'procA/result.json'),'procB/result.json':sha(R/'procB/result.json'),'procA/raw_inventory.jsonl':sha(R/'procA/raw_inventory.jsonl'),'policy.snapshot.json':sha(R/'policy.snapshot.json'),'expected_observations.snapshot.json':sha(R/'expected_observations.snapshot.json')},'scope':'metadata only; no tensor byte rehash or numerical tensor recomputation','referenced_objects':len(refs),'store_objects':len(objects),'extra_objects':len(extra),'extra_bytes':sum(objects[h] for h in extra),'extra_native_nodes':sorted({x['node'] for x in native_extra}),'extra_native_states':native_extra,'nonbitwise_output_case_count':len(differing),'by_fixture':dict(byfixture),'by_depth':dict(bydepth),'by_node':dict(bynode),'max_rms_fraction_of_frozen_bound':max(x['max_rms_fraction_of_frozen_bound'] for x in differing),'max_maxabs_fraction_of_frozen_bound':max(x['max_maxabs_fraction_of_frozen_bound'] for x in differing),'all_nonbitwise_output_cases':differing},indent=2))
