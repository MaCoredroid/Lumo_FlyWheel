#!/usr/bin/env python3
"""Reduce preserved existing observations; no inference or unit tests."""
import hashlib,json,math,runpy,statistics,re
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'claude-results-20261002'
FOLLOW=HERE.parent/'claude-followup-20261001'
REPLAY=HERE.parent/'claude-replay-20261001'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(HERE/'SYNC-MANIFEST.json')
for f in manifest['files']:assert sha(HERE/f['path'])==f['sha256'],f['path']
freeze=read(HERE/'code/q1v3v3/FREEZE.json')
for n,h in freeze['files'].items():assert sha(HERE/'code/q1v3v3'/n)==h,n
cases=read(HERE/'code/q1v3v3/cases.json')
def digest(d):return hashlib.sha256(json.dumps({k:v for k,v in d.items() if k!='record_sha256'},sort_keys=True,default=str).encode()).hexdigest()
assert digest(cases)==cases['record_sha256']
records={}
for arm in ['A','B','V','P','CAND']:
 files=list((HERE/'raw/q1v3v3'/arm/'cases').glob('*.json')); records[arm]=[read(p) for p in files]
 assert len(files)==(29 if arm=='CAND' else 18)
 for d in records[arm]:
  assert digest(d)==d['record_sha256'] and d['valid'] and not d['problems']
  # The planted stale-root fault deliberately violates this input check.
  if d['structural_mismatches']:
   assert d['control']['mutation']['mutation']=='NC_STALE' and all(x['check']=='root_token' for x in d['structural_mismatches'])
  assert d['cases_sha256']==freeze['files']['cases.json']
  assert d['hooks_sha256']==freeze['files']['q1v3_hooks.py']
first=min(d['sealed_utc'] for vals in records.values() for d in vals);assert freeze['utc']<first
v=read(HERE/'summaries/q1v3_v3_20261002/VERDICT.json')
assert v['verdict']=='EQUIVALENT' and all(v['gates'].values()) and not v['candidate']['failing_cells'] and not v['candidate']['structural_failures']
surfaces=['gdn_rel','conv_rel','kv_new_rel','kv_hist_rel','next_kl']; floors={s:.02 for s in surfaces};floors['next_kl']=.15
worst={s:0 for s in surfaces};absolute={s:0 for s in surfaces};flips={a:0 for a in ['CAND','B','V','P']}; count=0;amb=0
for cid, cells in v['cells_candidate'].items():
 for ce in cells:
  k=ce['k']; ns=[v['cells_native'][a][cid][k] for a in ['B','V','P']]
  for s in surfaces:
   ratio=ce[s]/max(floors[s],*(n[s] for n in ns));worst[s]=max(worst[s],ratio);absolute[s]=max(absolute[s],ce[s]);assert ratio<=2
  band=max(.25,2*max(n['next_maxabs'] for n in ns));unstable=any(n['greedy_x']!=n['greedy_ref'] for n in ns) and ce['margin_ref']>band
  amb+=int(unstable);assert ce['greedy_x']==ce['greedy_ref'] or ce['margin_ref']<=band or unstable
  for a,x in [('CAND',ce),*[(a,v['cells_native'][a][cid][k]) for a in ['B','V','P']]]:flips[a]+=int(x['greedy_x']!=x['greedy_ref'])
  count+=1
assert count==126 and amb/count<=.1
assert len(v['negative_controls'])==10 and all(n['detected'] and n['executed'] for n in v['negative_controls'])
for n in v['negative_controls']:
 for x in n['target'].values():assert x['detected']==(x['value']>x['bound']) and math.isclose(x['ratio'],2*x['value']/x['bound'])
 assert any(x['detected'] for x in n['target'].values())
assert len(v['cache_reuse']['observations'])==30
for o in v['cache_reuse']['observations']:assert o['cached_tokens']>=.9*o['P'] and o['hit']
assert all(r['bitwise_equal_all_cycles'] for r in v['candidate_repeat'])

def replay(p):
 rows=[json.loads(l) for l in p.read_text().splitlines()];assert len(rows)==43 and not any(r.get('error') for r in rows)
 assert len({r['request'] for r in rows})==43 and all(r['max_tokens']==1024 for r in rows)
 n=sum(r['completion_tokens'] for r in rows); t=sum(r['t_e2e_s']-r['t_ttft_s'] for r in rows)
 accepted=sum(x for r in rows for k,x in r.get('spec_delta',{}).items() if 'num_accepted_tokens{' in k)
 events=sum(x for r in rows for k,x in r.get('spec_delta',{}).items() if 'num_drafts{' in k)
 return {'run':p.parent.name,'requests':43,'output_tokens':n,'decode_seconds':t,'tokens_s':(n-43)/t,'accepted':accepted,'events':events,'accepted_per_event':accepted/events if events else None,'prompt_counts':{r['request']:r['prompt_tokens'] for r in rows}}
def pool(rr):return {'runs':len(rr),'requests':sum(r['requests'] for r in rr),'output_tokens':sum(r['output_tokens'] for r in rr),'decode_seconds':sum(r['decode_seconds'] for r in rr),'pooled_tokens_s':sum(r['output_tokens']-r['requests'] for r in rr)/sum(r['decode_seconds'] for r in rr),'run_rates':[r['tokens_s'] for r in rr],'accepted_per_event':sum(r['accepted'] for r in rr)/sum(r['events'] for r in rr) if sum(r['events'] for r in rr) else None}
confirm={}
for arm,markers in [('tree',['cfT1','cfT2','cfT3']),('mtp5',['cfM1','cfM2','cfM3'])]:
 rr=[]
 for marker in markers:
  p=next(p for base in [OLD,HERE] for p in (base/'raw/replay').glob('*/replay.jsonl') if marker in p.parent.name);rr.append(replay(p))
 confirm[arm]=pool(rr)
 if arm=='tree':tree=rr
 else:
  for a,b in zip(tree,rr):assert a['prompt_counts']==b['prompt_counts']
confirm['pooled_ratio']=confirm['tree']['pooled_tokens_s']/confirm['mtp5']['pooled_tokens_s']
confirm['all_tree_rates_above_all_mtp']=min(confirm['tree']['run_rates'])>max(confirm['mtp5']['run_rates'])

def timers(p):
 totals={}
 for f in p.glob('*.json.*'):
  if '.samples.' in f.name:continue
  d=read(f)
  if d.get('schema','').startswith('fr13.sfwd_gpu_timer'):
   totals['target']=[d['decode_forward_gpu_seconds'],d['n_pure_decode_steps_timed']]; totals['wall']=[d['decode_step_wall_seconds'],d['n_wall_steps']]
  elif d.get('label') in ['drafter','committer']:totals[d['label']]=[d['gpu_seconds'],d['n_spans']]
 return totals
cts=[timers(p.parent) for base in [OLD,HERE] for p in (base/'raw/replay').glob('*cfT*/replay.jsonl')]
assert len(cts)==3 and all(set(t)=={'target','wall','drafter','committer'} for t in cts)
confirm['tree']['timer_means_ms']={k:1000*sum(t[k][0] for t in cts)/sum(t[k][1] for t in cts) for k in cts[0]}
attrib={}
for label,doc in read(HERE/'summaries/attribution_20261002.json').items():
 reps=[]; ts=[]; mem=[]
 for r in doc['replicates']:
  name=r['run']; p=next(p for base in [OLD,FOLLOW,REPLAY,HERE] for p in (base/'raw/replay').glob('*/replay.jsonl') if p.parent.name==name)
  reps.append(replay(p)); t=timers(HERE/'raw/attribution'/name);assert set(t)=={'target','wall','drafter','committer'};ts.append(t)
  m={k:r[k] for k in ['weights_gib','kv_avail_gib','graph_pool_gib','kv_tokens']}
  log=(HERE/'raw/attribution'/name/'boot-memory.txt').read_text()
  pats={'weights_gib':r'Model loading took ([\d.]+) GiB','kv_avail_gib':r'Available KV cache memory: ([\d.]+) GiB','graph_pool_gib':r'CUDA graph pool memory: ([\d.]+) GiB \(actual\)','kv_tokens':r'GPU KV cache size: ([\d,]+) tokens'}
  for k,pat in pats.items():assert float(re.search(pat,log).group(1).replace(',',''))==m[k]
  mem.append(m)
 attrib[label]={**pool(reps),'timer_means_ms':{k:1000*sum(t[k][0] for t in ts)/sum(t[k][1] for t in ts) for k in ts[0]},'boot_memory':mem}

# Independent ancestry/history reconstruction from passive rows.
T=runpy.run_path(str(HERE/'code/fr13_fixed32_topology.py')); parents=list(T['DRAFT_PARENT'])
trace=next((HERE/'raw/replay').glob('tree-pd1-*/pen_trace.jsonl'))
calls=[json.loads(l) for l in trace.read_text().splitlines()];calls=[c for c in calls if c.get('nrows')==31]
assert len(calls)==80;flat=correct=total=0
for i in range(0,len(calls),2):
 target,selfr=calls[i:i+2];spec=target['spec'][0];base=set(target['rows'][0]['pen_ids'])
 for kind,call in [('target',target),('self',selfr)]:
  assert call['penalties']['presence_penalties']==[1.0]
  for j,row in enumerate(call['rows']):
   u=parents[j] if kind=='target' else j; path=set()
   while u>=0:path.add(spec[u]);u=parents[u]
   total+=1;flat+=set(row['pen_ids'])==(base|set(spec[:j]));correct+=set(row['pen_ids'])==(base|path)
assert (total,flat,correct)==(2480,2480,63)
a=read(HERE/'raw/replay'/trace.parent.name/'pen_analysis_v2.json');mc=[s for s in a['per_step'] if 'faults' in s]
# Independently reconstruct node TV from captured rounded top-256 logits.
def row_probs(row,pen,exact):
 ids=np.array(row['top_ids']); x=np.array(row['top_vals'],dtype=np.float32)
 pres=-min(row['pen_vals']) if row['pen_vals'] else 1.0
 x=(x-np.array([pres if int(t) in pen else 0 for t in ids],dtype=np.float32))/np.float32(.6)
 order=np.lexsort((ids,-x)); keep=order[:20] if exact else np.flatnonzero(x>=x[order[19]])
 p=np.exp(x[keep].astype(float)-float(x[keep].max()));p/=p.sum()
 lo=np.argsort(p);n=int((np.cumsum(p[lo])<=.05).sum())
 tied=n>0 and n<len(lo) and p[lo[n-1]]==p[lo[n]]
 p[lo[:n]]=0;p/=p.sum()
 return {int(ids[j]):float(q) for j,q in zip(keep,p) if q>0},bool(tied)
children=T['active_child_lists']('hydra27_fixed32');tvmaxdiff=0;tv_tie_differences=0;tv_independent=[]
for si in range(40):
 target,selfr=calls[2*si:2*si+2];spec=target['spec'][0];base=set(target['rows'][0]['pen_ids'])
 for node,value in a['per_step'][si]['node_tv'].items():
  u=int(node);kids=children.get(u,[]);row=target['rows'][kids[0]] if kids else selfr['rows'][u]
  path=set();z=u
  while z>=0:path.add(spec[z]);z=parents[z]
  pd,dt=row_probs(row,set(row['pen_ids']),True);pc,ct=row_probs(row,base|path,False)
  tv=.5*sum(abs(pd.get(t,0)-pc.get(t,0)) for t in pd.keys()|pc.keys());tvmaxdiff=max(tvmaxdiff,abs(tv-value))
  if abs(tv-value)>.00015:assert dt or ct;tv_tie_differences+=1
  tv_independent.append((u,tv))
assert round(max(x for _,x in tv_independent),3)==.421
tv_recheck={'node_steps':1120,'non_tied_node_values_match_within':.00015,'top_p_tie_order_differences':tv_tie_differences,'maximum_difference_at_ties':tvmaxdiff,'maximum_TV_independent':max(x for _,x in tv_independent),'deeper_share_over_0_01_independent':sum(x>.01 for u,x in tv_independent if u!=-1)/1080,'scope':'Top-256 captured logits rounded to five decimals; ties make some TV values implementation-order dependent; describe the aggregate as approximate.'}
assert len(mc)==6
nt=sum(s['walk_vs_used_rows']['nodes_tested'] for s in mc);nr=sum(s['walk_vs_used_rows']['nodes_rejected'] for s in mc)
faults={n:sum(s['faults'][n]['detected'] for s in mc) for n in mc[0]['faults']}
assert nt==54 and nr==0 and all(c>0 for c in faults.values())
node_tv=[x for s in a['per_step'] for n,x in s['node_tv'].items() if n!='-1'];root_tv=[s['node_tv']['-1'] for s in a['per_step']];assert max(root_tv)==0
sampling={'steps_captured':40,'rows':total,'flat_history_matches':flat,'correct_path_matches':correct,'node_steps':len(node_tv)+len(root_tv),'deeper_share_TV_over_0_01':sum(x>.01 for x in node_tv)/len(node_tv),'maximum_node_TV':max(node_tv),'MC_steps':6,'walks_per_step':200000,'node_tests':nt,'used_row_rejections':nr,'correct_path_rejected_steps':sum(s['walk_vs_correct_rows']['nodes_rejected']>0 for s in mc),'fault_detected_steps':faults,'expected_sum_TV_per_step_range':[min(s['expected_tv_per_step'] for s in mc),max(s['expected_tv_per_step'] for s in mc)],'scope':'CPU deployed float-oracle walk on captured compressed supports; Monte Carlo non-rejection does not prove universal exactness; history-independent exactness remains conditional on correct target rows and matching filters.'}
audit={'schema':'lumotree.closure-audit.v1','remote_files_sha_verified':len(manifest['files']),'remote_head':manifest['remote_head'],'freeze_utc':freeze['utc'],'first_sealed_observation_utc':first,'fullmodel':{'numerical_verdict':v['verdict'],'prefixes':6,'cases':18,'cycles':count,'faults_detected':10,'max_ratio':worst,'max_absolute':absolute,'greedy_flips':flips,'cache_reuse_observations':30,'cache_hit_fraction_min':min(o['cached_tokens']/o['P'] for o in v['cache_reuse']['observations']),'candidate_repeat_bitwise':True,'scope':'Tolerance-based comparison against native controls; not bitwise model equality or sampling-distribution equality; binary tensor norms not independently recomputed in this paper update.'},'confirmation':confirm,'attribution':attrib,'sampling':sampling,'TV_independent_recheck':tv_recheck}
(HERE/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
