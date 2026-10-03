#!/usr/bin/env python3
"""Read-only final source, process and command receipts; no GPU/Torch/tensor work."""
from pathlib import Path
import json,hashlib
REPO=Path('/home/mark/lumotree-review-20260927');C=REPO/'papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927';R=C/'runs/q1.2b-heldout/q12b-heldout-20260928T013539Z'
def j(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
g=j(R/'GATE-Q1.2b.snapshot.json'); lb=j(R/'LAUNCH-BINDING.json')
paths={'runner':C/'tools/q1_component_runner_v2_2.py','reducer':C/'tools/q1_2b_reduce_v2_2.py','launcher':C/'tools/run_q1_2b_component_v2_2.sh','native_module':C/'identity/native_source/fla_ops__fused_sigmoid_gating.py'}
for name,path in paths.items():assert sha(path)==g['reviewed_hashes'][name]==lb['expect'][name]
for name,h in lb['helpers'].items():
 p=REPO/name if '/' in name else C/'tools'/name
 assert sha(p)==h
A=j(R/'procA/result.json');B=j(R/'procB/result.json')
for a,b in zip(A['fixtures'],B['fixtures']):
 assert a['fixture_id']==b['fixture_id']
 assert a['native_repeats']==b['native_repeats']
 assert a['repeat_hashes']==b['repeat_hashes']
for tag in ('A','B'):
 cid=(R/f'proc{tag}.cid').read_text().strip();r=A if tag=='A' else B
 assert r['attestation']['hostname_in_container']==cid[:12]
assert (R/'procA.cid').read_text()!=(R/'procB.cid').read_text()
cmd=(R/'command_reduce.txt').read_text()
assert '--gpus' not in cmd and 'CUDA_VISIBLE_DEVICES= ' in cmd and '--recompute all' in cmd and 'OMP_NUM_THREADS=1' in cmd and 'MKL_NUM_THREADS=1' in cmd and g['reviewed_hashes']['image_id'] in cmd
s=j(R/'summary.v2.json');assert s['reducer_runtime']['cuda_available'] is False and s['reducer_runtime']['torch_threads']==1
rt=j(R/'RUN-RECEIPT.json');assert rt['status']=='COMPLETED_reduce_rc=2'
logs={}
for tag in ('A','B'):
 p=R/f'proc{tag}.log';txt=p.read_text();assert '[FR13_SUBTREE_PARALLEL ENGAGED] n_actual=32 schedule=fixed32 critical=12' in txt and '[FR13_FIXED32_COMMIT_DEVICE_FILL ENGAGED] mode=hydra27_fixed32 B=1 fixed16 one-replay' in txt
 logs[f'proc{tag}.log']={'sha256':sha(p),'engagement_messages_present':True}
print(json.dumps({'run_id':R.name,'scope':'read-only source/metadata check; original reducer failure preserved','source_bytes_verified':{k:sha(p) for k,p in paths.items()},'all_launch_helper_source_bytes_verified':lb['helpers'],'full_native_and_candidate_repeat_censuses_AB_identical':True,'process_container_ids':{tag:(R/f'proc{tag}.cid').read_text().strip() for tag in ('A','B')},'command_reduce':cmd,'pinned_cpu_command_and_runtime_verified':True,'original_summary':s,'terminal_receipt_sha256':sha(R/'RUN-RECEIPT.json'),'logs':logs},indent=2))
