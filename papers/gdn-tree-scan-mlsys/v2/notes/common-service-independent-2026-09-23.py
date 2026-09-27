from pathlib import Path
import json,re,hashlib,math
R=Path('/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel');V=R/'papers/gdn-tree-scan-mlsys/v2'; hashes={}
def load(p,text=False):
 b=p.read_bytes();hashes[str(p.relative_to(R))]=hashlib.sha256(b).hexdigest();return b.decode()if text else json.loads(b)
def metrics(p):
 d={}
 for l in load(p,True).splitlines():
  if not l or l.startswith('#'):continue
  m=re.fullmatch(r'([^\s{]+)(\{[^}]*\})?\s+(\S+)(?:\s+\S+)?',l)
  if m:
   key=(m[1],m[2]or'');assert key not in d;d[key]=float(m[3])
 return d
out=[]
for cohort in ['sglang','cqc10']:
 if cohort=='sglang':
  root=R/'results/fr14_nvfp4_port_20260816/sglang16_evidence';paths=[root/'run1/swe_out/verified/per_task/astropy__astropy-12907',root/'r1/swe_out/verified/per_task/astropy__astropy-13033'];ns='sglang';gh='generation_tokens_histogram';req='num_requests_total';gauges=['num_running_reqs','num_queue_reqs','num_grammar_queue_reqs','num_prefill_bootstrap_queue_reqs','num_prefill_inflight_queue_reqs','num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs']
 else:
  root=V/'results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10';paths=sorted((root/'swe_out/verified/per_task').iterdir());ns='vllm';gh='request_generation_tokens';req='request_success_total';gauges=['num_requests_running','num_requests_waiting','num_requests_waiting_by_reason']
 names=['generation_tokens_total',gh+'_sum',gh+'_count','e2e_request_latency_seconds_sum','e2e_request_latency_seconds_count','time_to_first_token_seconds_sum','time_to_first_token_seconds_count',req]
 previous=None;rows=[]
 for p in paths:
  a,b=metrics(p/'vllm_metrics_pre.txt'),metrics(p/'vllm_metrics_post.txt');keys={k for k in set(a)|set(b)if k[0]in [ns+':'+n for n in names]}; delta={k:b.get(k,0)-a.get(k,0) for k in keys};assert all(v>=-1e-7 and math.isfinite(v)for v in delta.values())
  if cohort=='cqc10' and previous is not None:assert all(a.get(k,0)==previous.get(k,0)for k in keys)
  previous=b
  def val(n):return sum(v for k,v in delta.items()if k[0]==ns+':'+n)
  counts={n:val(n)for n in [gh+'_count','e2e_request_latency_seconds_count','time_to_first_token_seconds_count',req]};assert len(set(counts.values()))==1 and next(iter(counts.values()))>0
  N=val('generation_tokens_total');count=val(req);E=val('e2e_request_latency_seconds_sum');T=val('time_to_first_token_seconds_sum');assert N==val(gh+'_sum') and N>=count and E>T>=0
  boundary={side:{k[0]+k[1]:v for k,v in snap.items()if k[0]in [ns+':'+g for g in gauges]}for side,snap in [('pre',a),('post',b)]};assert all(boundary.values())and all(v==0 for x in boundary.values()for v in x.values())
  row={'task':p.name,'tokens':N,'requests':count,'e2e_seconds':E,'ttft_seconds':T,'count_alignment':counts,'idle_boundaries':boundary,'tokens_per_summed_e2e_s':N/E,'post_first_token_pooled_rate':(N-count)/(E-T)}
  if cohort=='sglang':
   tr=[json.loads(x)for x in load(p/'qwen_trace.jsonl',True).splitlines()if x.strip()];res=[x for x in tr if x.get('type')=='result'];assert len(res)==1;assert res[0]['usage']['output_tokens']==N
   paid=[x['message']['usage']for x in tr if x.get('type')=='assistant' and x.get('message',{}).get('usage',{}).get('output_tokens',0)>0]
   row['trace_reconciliation']={'result_output_tokens':res[0]['usage']['output_tokens'],'visible_nonzero_usage_messages':len(paid),'visible_output_tokens':sum(x['output_tokens']for x in paid),'additional_engine_requests':count-len(paid),'additional_output_tokens':N-sum(x['output_tokens']for x in paid),'note':'Additional requests are included in server counters and final trace usage; their class is not explicitly bound by this older metadata.'}
  else:
   md=load(p/'runner_metadata.json');ev=md['fixed32_real_task_provenance']['qwen_compaction_metric_evidence'];assert ev['completed_engine_requests']==count and ev['generation_tokens']==N
   bd=load(p/'fixed32_task_boundary.json');assert all(bd[s]['status']=='ok'and all(bd[s]['counters'][k]==0 for k in ['cfwd_pending','dfwd_pending','sfwd_pending'])for s in ['pre','post'])
   row['compaction']={'normal':ev['normal_requests'],'successful_internal':ev['successful_compaction_requests'],'failed_internal':ev['failed_compaction_requests']}
  rows.append(row)
 tot={k:sum(x[k]for x in rows)for k in ['tokens','requests','e2e_seconds','ttft_seconds']};tot['tokens_per_summed_e2e_s']=tot['tokens']/tot['e2e_seconds'];tot['post_first_token_pooled_rate']=(tot['tokens']-tot['requests'])/(tot['e2e_seconds']-tot['ttft_seconds']);out.append({'cohort':cohort,'tasks':rows,'total':tot})
print(json.dumps({'schema':'independent-common-service-estimator-v1','cohorts':out,'file_sha256':hashes},indent=2))
