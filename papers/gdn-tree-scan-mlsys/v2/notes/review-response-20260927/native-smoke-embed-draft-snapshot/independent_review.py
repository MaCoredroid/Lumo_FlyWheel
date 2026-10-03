#!/usr/bin/env python3
"""Read-only, stdlib-only seam/anchor/config checks; no torch/vLLM/model imports."""
import ast, hashlib, importlib.util, json, pathlib, subprocess, sys, types
V2=pathlib.Path(__file__).resolve().parents[3]
C=V2/'experiments/review-response-20260927'
T=C/'tools'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pins={
 'q1_reference_hooks_v2_1.py':'1ac939405de01a4c22e5d73decb9994cfb3a3214fd838ed2a734b39110b2b8cf',
 'q1_patch_reference_runner_v2_1.py':'ee61c5d779cc39dce9757bae944f51d991bf2b15ef89b04bf69e9a58c91b315a',
 'q1_spec_off_engine_config_v2_1.py':'cd7fbc1322c9c4206990f2223728c2b49ae551c1a644a99fbafd2576380deb14',
 'run_q1_native_smoke_v2_4.sh':'df5e0c1982f12060a4277bcd79d9762c371ed009c8a5ed47d3cfc7062903a721'}
assert all(sha(T/k)==v for k,v in pins.items())
checks=[]
def check(name, fn):
 fn(); checks.append(name)
def load(p):
 spec=importlib.util.spec_from_file_location(p.stem,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
# These modules are stdlib configuration renderers/patchers, not runtime GPU modules.
sys.dont_write_bytecode=True
P=load(T/'q1_patch_reference_runner_v2_1.py')
CFG=load(T/'q1_spec_off_engine_config_v2_1.py')
CFG2=load(T/'q1_spec_off_engine_config_v2.py')
source=(C/'identity/native_source/vllm__v1__worker__gpu_model_runner.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()==P.EXPECTED_RUNNER_SHA
rendered,counts=P.patch_text(source)
def render_check():
 assert counts==dict.fromkeys(['A1','A2','A3','A4','A5'],1)
 compile(rendered,'<rendered-runner>','exec')
 assert rendered.index(') = self._preprocess(')<rendered.index('_q1_ref_hooks.instance().on_pre_forward(')<rendered.index('            model_output = self._model_forward(')
 assert 'import q1_reference_hooks_v2_1 as _q1_ref_hooks' in rendered
check('pinned stock source and 5 anchors; emitted runner compiles; prehook after preparation before forward',render_check)
def double_refuse():
 try:P.patch_text(rendered)
 except RuntimeError:return
 raise AssertionError('repatch admitted')
check('already patched source refuses',double_refuse)
def anchor_refuse():
 try:P.patch_text(source.replace(P.A2_ANCHOR,'# removed\n'))
 except RuntimeError:return
 raise AssertionError('missing anchor admitted')
check('missing anchor refuses',anchor_refuse)
def cfg_check():
 for arm in CFG.ARMS:
  a=CFG.arm_config(arm,'CPU','/logs','a'*64); b=CFG2.arm_config(arm,'CPU','/logs','a'*64)
  for k in ('env','serve_argv','docker_argv_without_script'):assert a[k]==b[k],k
  assert 'q1_patch_reference_runner_v2_1.py --apply' in a['in_container_script']
  assert '--allow-hash-mismatch' not in a['in_container_script']
 assert CFG.REF_HOOKS=='tools/q1_reference_hooks_v2_1.py'
check('both arms preserve env/serve/docker argv and invoke new strict patcher',cfg_check)
# Extract ONLY actual method ASTs; replace IO/boot/state collaborators with CPU fakes.
module=ast.parse((T/'q1_reference_hooks_v2_1.py').read_text())
cl=next(x for x in module.body if isinstance(x,ast.ClassDef) and x.name=='Q1RefHooks')
methods=[x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name in ('on_pre_forward','on_sampled')]
class Scalar:
 def __init__(self,t=None,key=None,value=None):self.t,self.key,self.value=t,key,value
 def item(self):return self.t.values[self.key] if self.t else self.value
 def copy_(self,other):self.t.values[self.key]=other.item()
class Tensor:
 dtype='int32';device='cpu'
 def __init__(self,vals,shape):self.values=dict(vals);self.shape=shape
 def __getitem__(self,k):return Scalar(self,k)
 def dim(self):return len(self.shape)
class Case:
 def __init__(self):
  self.sealed=False;self.invalid_reasons=[];self.prefix_len=21;self.chain=[3,22];self.positions=[21,22];self.z_step_num_computed=22;self.consumed_trace=[];self.natural_sampled=[];self.req_id='r';self.forced=0;self.request_active=True;self.o0=None;self.o1=None
 def invalidate(self,s):self.invalid_reasons.append(s)
class Probe:
 def __init__(self):self.enabled=True;self.case=Case();self.fault=None;self.seq=0;self.seals=0;self.job={'terminal_token_id':7}
 def _attest_boot(self,r):pass
 def _bind(self,r):return 'r'
 def _seal_case(self):self.seals+=1;self.case.sealed=True
 def _snapshot_state(self,r,rid,attn,tag,n):return {'tag':tag,'materialized_tokens':n}
ns={'torch':types.SimpleNamespace(tensor=lambda v,**kw:Scalar(value=v))}
exec(compile(ast.Module(body=methods,type_ignores=[]),'<actual-hook-methods>','exec'),ns)
for x in methods:setattr(Probe,x.name,ns[x.name])
def runner(n=21,qsl=2,tok=3):
 return types.SimpleNamespace(requests={'r':types.SimpleNamespace(num_computed_tokens=n)},input_batch=types.SimpleNamespace(req_id_to_index={'r':0}),query_start_loc=types.SimpleNamespace(np=[qsl]),input_ids=types.SimpleNamespace(gpu=Tensor({0:999,qsl:tok},[qsl+1])))
def pos(n=21,qsl=2,dim=1):return Tensor({qsl:n} if dim==1 else {(0,qsl):n,(1,qsl):n,(2,qsl):n},[qsl+1] if dim==1 else [3,qsl+1])
def step(h,r,ids=None,position='default',n_sched=1):
 if position=='default':position=pos(r.requests['r'].num_computed_tokens,r.query_start_loc.np[0])
 so=types.SimpleNamespace(num_scheduled_tokens={'r':n_sched})
 h.on_pre_forward(r,so,None,ids,position,None);return so
def offset():
 h=Probe();r=runner();step(h,r);assert not h.case.invalid_reasons and h.case.consumed_trace[0]['token']==3 and h.case.o0['materialized_tokens']==21
check('embed buffer uses nonzero query_start_loc rather than stale slot zero',offset)
def primary_arg():
 h=Probe();r=runner(tok=999);step(h,r,Tensor({2:3},[3]));assert not h.case.invalid_reasons and h.case.consumed_trace[0]['token_source']=='input_ids_arg'
check('non-None argument takes precedence over runner buffer',primary_arg)
def invalid(kind):
 h=Probe();r=runner();p=pos()
 if kind=='buffer':del r.input_ids
 if kind=='position':p=None
 if kind=='wrong-token':r.input_ids.gpu.values[2]=999
 if kind=='wrong-position':p=pos(n=20)
 if kind=='wrong-mrope-row0':p=pos(n=20,dim=3)
 step(h,r,position=p);step(h,r,position=p)
 assert h.case.invalid_reasons and h.case.sealed and h.seals==1 and h.case.o0 is None
for kind in ('buffer','position','wrong-token','wrong-position','wrong-mrope-row0'):check('invalid '+kind+' seals once before state capture',lambda k=kind:invalid(k))
def mrope():
 h=Probe();step(h,runner(),position=pos(dim=3));assert not h.case.invalid_reasons and h.case.consumed_trace[0]['positions_shape']==[3,3] and h.case.consumed_trace[0]['position']==21
check('mRoPE-shaped position accepts matching row0 at request offset',mrope)
def forcing():
 h=Probe();r=runner(n=0,qsl=0,tok=100);so=step(h,r,n_sched=21)
 output=Tensor({(0,0):99},[1,1]);h.on_sampled(r,types.SimpleNamespace(sampled_token_ids=output),so);assert output.values[0,0]==3
 for n,t in ((21,3),(22,22)):
  r.requests['r'].num_computed_tokens=n;r.input_ids.gpu.values[0]=t;so=step(h,r)
  h.on_sampled(r,types.SimpleNamespace(sampled_token_ids=output),so)
 assert [x['token'] for x in h.case.consumed_trace if 'token' in x]==[3,22]
 assert h.case.forced==3 and output.values[0,0]==7 and h.case.o0['materialized_tokens']==21 and h.case.o1['materialized_tokens']==22 and h.seals==1 and not h.case.invalid_reasons
check('actual sampled-token method forces root,z,terminal and embed observer consumes chain',forcing)
# Every other method AST remains identical; added diagnostic boot fields are reviewed separately.
a=ast.parse((T/'q1_reference_hooks_v2.py').read_text()); a=next(x for x in a.body if isinstance(x,ast.ClassDef) and x.name=='Q1RefHooks')
old={x.name:ast.dump(x,include_attributes=False) for x in a.body if isinstance(x,ast.FunctionDef)}
changed=[x.name for x in cl.body if isinstance(x,ast.FunctionDef) and old.get(x.name)!=ast.dump(x,include_attributes=False)]
assert changed==['_attest_boot','on_pre_forward'],changed
checks.append('only boot facts and preforward method AST differ; forcing/numerical/sealing methods identical')
# Real shell parses and exact dry-run exits before any model/Docker/mount action.
l=T/'run_q1_native_smoke_v2_4.sh'
assert subprocess.run(['bash','-n',str(l)],capture_output=True).returncode==0
r=subprocess.run(['bash',str(l),'--dry-run'],capture_output=True,text=True)
assert r.returncode==0 and '(dry-run: nothing launched, nothing written)' in r.stdout
checks.append('actual shell syntax and no-write/no-launch dry-run pass')
freeze=C/'FREEZE-Q1-FULLMODEL-DESIGN-v2.5.json'
if not freeze.exists():freeze=C/'FREEZE-Q1-FULLMODEL-DESIGN-v2.4.json'
f=json.loads(freeze.read_text());bad=[]
for rel,v in f['files'].items():
 p=C/rel
 if sha(p)!=v['sha256'] or p.stat().st_size!=v['bytes']:bad.append(rel)
repo=C.parents[4]
for rel,v in f['repo_files'].items():
 p=repo/rel
 if sha(p)!=v['sha256'] or p.stat().st_size!=v['bytes']:bad.append(rel)
assert not bad,bad
checks.append('all frozen local campaign and repository member hashes/sizes match')
result={'schema':'independent.native-smoke-embed-repair.cpu.v1','status':'PASS_BOUNDED_CPU_SOURCE_REVIEW','checks':checks,'count':len(checks),'source_sha256':pins,'stock_runner_sha256':P.EXPECTED_RUNNER_SHA,'emitted_runner_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'freeze':{'name':freeze.name,'file_sha256':sha(freeze),'local_campaign_members':len(f['files']),'repo_members':len(f['repo_files']),'mismatches':bad},'limits':['No torch, pytest, vLLM, model or GPU imports/execution. Author Torch controls were reviewed as source-bound evidence, not rerun here.','Seam probes stub state/boot/IO; they do not establish actual runtime O0/O1/O2 validity.','mRoPE check observes row0 and shape; it is not an independent all-axis or multimodal qualification.','No launch authorization or numerical qualification.']}
print(json.dumps(result,indent=2))
