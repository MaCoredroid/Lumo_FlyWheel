#!/usr/bin/env python3
"""Verify current production component evidence and runtime identity; no inference."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;R=D.parents[4]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=json.loads((D/'inputs.json').read_text())['inputs']
for name,sha in inputs.items():
 assert H(R/name)==sha,name
B=R/'results/fr14_nvfp4_port_20260816'
Q=R/'papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10'
t=json.loads((B/'fr14_fused_draft_topk_probe_result.json').read_text())
f=json.loads((B/'fr14_splitk_tierb_credential.json').read_text())
e=json.loads((Q/'logs/fr13_fa2_qrow32_b1_production_engagement.json').read_text())
env=dict(line.split('=',1) for line in (Q/'container_env.SANITIZED.txt').read_text().splitlines() if '=' in line)
assert t['so_sha256']==env['FR14_FUSED_DRAFT_TOPK_SHA256']
assert H(B/'fr14_splitk_tierb_credential.json')==e['tier_b_credential_sha256']
assert f['identity']['so_sha256']==e['candidate_so_sha256']
assert f['identity']['patch_source_sha256']==e['patch_source_sha256']
assert f['identity']['bounds_sha256']==H(B/'fr14_splitk_tierb_bounds.json')
assert t['gate']['gate_pass'] and t['gate']['mismatch_total']==0 and t['graph_gate']['mismatching_replays']==0
assert f['bounds_evaluation']['bounds_passed'] and all(x['passed'] for x in f['bounds_evaluation']['bounds'])
assert e['status']=='ENGAGED' and e['candidate_served'] and not e['fallback_allowed'] and e['layer_count']==16 and e['num_splits']==4
required={'FR13_DRAFTER_SINGLE_LOGITS':'1','FR13_DRAFTER_GRAPH':'1','FR13_COMMITTER_GRAPH':'1','FR13_COMMITTER_NATIVE':'1','FR13_TAW':'1','FR13_SUBTREE_PARALLEL':'1','FR13_SLOT_REORDER':'1','FR13_ATTN_KV_REMAP':'1','FR13_ENABLE_APC':'1','FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION':'0','FR14_FUSED_DRAFT_TOPK':'1','FR13_DRAFT_VOCAB_K':'0','FR13_DRAFT_VOCAB_ROOT':'0','FR13_SCAN_ALIGN':'0','FR13_STEP_GRAPH':'0'}
for k,v in required.items():assert env[k]==v,k
out={'schema':'lumo.current-production-audit.v1','input_sha256':inputs,'reducer_sha256':H(Path(__file__)),'current_method':'Hydra27 fixed32; patched FA2 GQA-pair splitK4; graph/cache-enabled B1 NVFP4 agent deployment','recorded_source_commit':e['source_commit'],'fused_selection':{'cases':t['gate']['cases_evaluated'],'configurations':t['gate']['total_configs'],'byte_mismatches':t['gate']['mismatch_total'],'negative_controls_fire':t['gate']['negative_control_all_fire'],'graph_replays':t['graph_gate']['replays'],'graph_mismatches':t['graph_gate']['mismatching_replays'],'binary_matches_deployment':True},'attention':{'binary_sha256':e['candidate_so_sha256'],'binary_and_credential_match_deployment':True,'determinism_cases':f['determinism']['cases'],'determinism_processes':f['probe']['determinism_processes'],'determinism_repeats':f['probe']['determinism_reps'],'determinism_pass':f['measurements']['all_cases_bitwise_identical'] and f['measurements']['cross_process_digests_identical'],'declared_checks':len(f['bounds_evaluation']['bounds']),'all_checks_pass':True,'output_within_two_ulp_percent':100*f['measurements']['output_ulp_le_2_fraction'],'output_max_abs_difference':f['measurements']['output_max_abs_delta'],'lse_max_ulp':f['measurements']['lse_max_ulp'],'engaged_layers':e['layer_count'],'num_splits':e['num_splits']},'verified_runtime_flags':required,'performance_claims':[],'new_inference_launched':False}
print(json.dumps(out,sort_keys=True,indent=2))
