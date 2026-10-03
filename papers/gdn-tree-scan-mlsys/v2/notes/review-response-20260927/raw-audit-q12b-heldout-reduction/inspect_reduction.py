#!/usr/bin/env python3
"""Read-only receipt audit of one authorized CPU reduction; no numerical/tensor/Docker work."""
from pathlib import Path
import datetime,hashlib,json,sys
REPO=Path('/home/mark/lumotree-review-20260927');V2=REPO/'papers/gdn-tree-scan-mlsys/v2';C=V2/'experiments/review-response-20260927'
RUN_ID='q12b-heldout-20260928T013539Z';RED_ID='q12b-heldout-reduction-20260928T020837Z'
O=C/'runs/q1.2b-heldout'/RUN_ID;R=C/'runs/q1.2b-heldout-reduction'/RED_ID
AUTH_SHA='2a732de1ad7b18485d73638446f5d42e1f1b3ae0cdaf70d2028acec8071e2f42'
ORIGINAL={
'RUN-RECEIPT.json':'7746dd7ba4eeb74c61e9e0c328a24653dfaf54f59667fcca67d27e726bf6e427',
'summary.v2.json':'646a07b7c57e540e512042d2dcf0f3bba04bba228564806d0a0fb9a693b46745',
'LAUNCH-BINDING.json':'cb67d9e996b81078160d8828e3b052c28ec5387f039b1de3ebe61b8e7385d3d8',
'GATE-Q1.2b.snapshot.json':'4ef2a52886edfdccea5812bbdc6dd3d05a4237ec41a6d9c8f5d58627519e883e',
'procA/result.json':'1097fd0b4a1423325c79954e8c88d98fcdca1d27357dd621dea32044b02cd71d',
'procB/result.json':'4509588c3c7a9f98fcd39f262a66d4af07364e4703dcd40c40bff23d894427ca',
'procA/raw_inventory.jsonl':'49e91f9b53e0d7a87e012ff19e42f146b545e4cdda8523763ac84a6551f20a94',
'procB/raw_inventory.jsonl':'93254a23fd66dff12c39506d597589ab546c1493759d07b94c821545bce7c9d0'}
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(v,m):
 if not v:raise RuntimeError(m)
out={'scope':'independent receipt/source/metadata audit only; no tensor rehash/recompute or numerical execution','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_run_id':RUN_ID,'reduction_id':RED_ID}
if not (R/'REDUCTION-RECEIPT.json').exists():
 out['status']='PENDING_TERMINAL_RECEIPT';print(json.dumps(out,indent=2));sys.exit(0)
receipt=load(R/'REDUCTION-RECEIPT.json');members={}
for rel,meta in receipt['files'].items():
 p=R/rel;h=sha(p);check(h==meta['sha256'] and p.stat().st_size==meta['bytes'],'receipt member '+rel);members[rel]=meta
check(sha(R/'AUTHORIZATION.snapshot.json')==AUTH_SHA,'authorization exact')
auth=load(R/'AUTHORIZATION.snapshot.json');b=load(R/'REDUCTION-BINDING.json');s=load(R/'summary.v2_2_1.json')
check(auth['approved'] is True and auth['authorizes']=='CPU_REDUCTION_ONLY' and auth['run_id']==RUN_ID and auth['reduction_id']==RED_ID,'authorization scope')
check(b['authorization']['sha256']==AUTH_SHA and b['reduction_id']==RED_ID and b['original_run']==str(O) and b['original_run_mounted']=='read-only','binding scope')
check(b['reducer']['sha256']==auth['repaired_reducer_sha256']=='a7decd00863f69b6970f7e1015a1d725681f046ebc65c63a535d5bc42da00cc8','reducer source binding')
check(b['launcher_sha256']==auth['launcher_sha256']=='6aa6a9c3f374840b728d64f565c73bbd445e8843105d0bc06259fbe121c383c5','launcher source binding')
for rel,h in ORIGINAL.items():check(sha(O/rel)==h and b['original_evidence'][rel]==h,'original preserved '+rel)
source_map={'repaired_reducer_sha256':C/'tools/q1_2b_reduce_v2_2_1.py','launcher_sha256':C/'tools/run_q1_2b_reduction_repair_v2_2_1.sh','source_acceptance_sha256':V2/'p0/monitor/review-response-20260927/q12b-heldout-reduction-source-acceptance.json','freeze_sha256':C/'FREEZE-Q1_2B-HELDOUT-REDUCTION-v1.json','review_sha256':V2/'notes/review-response-20260927/q1-2b-heldout-reducer-repair-review.md'}
for key,p in source_map.items():check(sha(p)==auth[key],'source authority '+key)
check(s['reduction']['original_run_id']==RUN_ID and s['reduction']['original_reducer_exit']==2 and s['reduction']['original_reducer_sha256']=='6475182d9e0996dd4e7c228340c57299665cd1ea641ed0b61bf291d631af9d73','original failure preserved')
check(s['reduction']['binding']['authorization_sha256']==AUTH_SHA,'summary authorization')
rt=s['reducer_runtime'];cid=(R/'reduce.cid').read_text().strip()
check(rt['hostname']==cid[:12] and rt['torch']=='2.11.0+cu130' and rt['torch_threads']==1 and rt['OMP_NUM_THREADS']=='1' and rt['MKL_NUM_THREADS']=='1' and rt['CUDA_VISIBLE_DEVICES']=='' and rt['cuda_available'] is False,'CPU runtime')
cmd=(R/'command_reduce.txt').read_text();check('--gpus' not in cmd and '--recompute all' in cmd and '--block evaluation' in cmd and '-v '+str(O)+':/runs/'+RUN_ID+':ro' in cmd,'command CPU/readonly/all')
img=load(R/'image_inspect.json')[0];check(img['Id']==b['image_id']=='sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc','image ID')
check('vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776' in img['RepoDigests'],'image digest')
for tag in ('A','B'):check(s['attestation_'+tag]==load(O/f'proc{tag}/result.json')['attestation'],'unchanged original attestation '+tag)
pass_counts=(s.get('cases_expected')==5760 and s.get('cases_evaluated')=={'A':5760,'B':5760} and s.get('case_verdicts')=={'A':{'PASS':5760},'B':{'PASS':5760}} and s.get('recomputed_records')==11520 and s.get('tensors_verified')==5814 and all(not v for v in s['findings'].values()) and s.get('negative_power')=={'A':{},'B':{}})
out.update({'status':receipt['status'],'reduce_exit':receipt['reduce_exit'],'receipt_sha256':sha(R/'REDUCTION-RECEIPT.json'),'binding_sha256':sha(R/'REDUCTION-BINDING.json'),'summary_sha256':sha(R/'summary.v2_2_1.json'),'authorization_sha256':AUTH_SHA,'receipt_members_verified':members,'original_evidence_rehashed_unchanged':ORIGINAL,'authorized_sources_current_hashes':{k:sha(p) for k,p in source_map.items()},'matches_independent_raw_census_and_expected_PASS_counts':pass_counts,'summary':{k:v for k,v in s.items() if k not in ('attestation_A','attestation_B')},'cpu_container_id':cid,'command_reduce':cmd,'image_id':img['Id'],'image_repo_digests':img['RepoDigests'],'started_utc':(R/'reduce_started_utc.txt').read_text().strip(),'ended_utc':(R/'reduce_ended_utc.txt').read_text().strip(),'original_attestations_unchanged':True})
print(json.dumps(out,indent=2))
