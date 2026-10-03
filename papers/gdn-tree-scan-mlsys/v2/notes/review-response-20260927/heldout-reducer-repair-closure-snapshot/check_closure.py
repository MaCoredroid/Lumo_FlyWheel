#!/usr/bin/env python3
"""Bounded stdlib checks of F1/F2/auth only; no reducer, Torch, GPU or Docker."""
import argparse,ast,contextlib,copy,hashlib,io,json,os,pathlib,sys,tempfile,subprocess
BASE=pathlib.Path(__file__).resolve().parent;V2=BASE.parents[2];P=BASE/'tools/q1_2b_reduce_v2_2_1.py';src=P.read_text();tree=ast.parse(src)
fn={'main','output_guard','bind_original_run','sha256_file'};const={'ORIGINAL_RUN','BOUND_RUNNER_SHA256','BOUND_LAUNCHER_SHA256'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in fn or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in const for t in n.targets)]
ns={'__file__':str(P),'argparse':argparse,'hashlib':hashlib,'json':json,'os':os,'sys':sys}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P),'exec'),ns)
def forbidden(*a,**kw):raise AssertionError('A real reduction must never run in this review')
ns['reduce_run']=forbidden
checks=[]
# Compile the exact F1 identity predicate from reduce_run, using only its expression.
pred=next(n.test for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and 'original[\'original_reducer_sha256\']' in ast.unparse(n.test) and "exp.get('runner')" in ast.unparse(n.test))
expression=compile(ast.Expression(body=pred),str(P),'eval')
good={'runner':ns['BOUND_RUNNER_SHA256'],'launcher':ns['BOUND_LAUNCHER_SHA256'],'reducer':ns['ORIGINAL_RUN']['original_reducer_sha256']}
for name,exp,want in [('F1_original_binding_accepted',good,False),('F1_repair_hash_in_original_binding_rejected',{**good,'reducer':ns['sha256_file'](P)},True),('F1_wrong_original_hash_rejected',{**good,'reducer':'0'*64},True)]:
 assert eval(expression,{**ns,'exp':exp,'original':ns['ORIGINAL_RUN']}) is want;checks.append(name)
with tempfile.TemporaryDirectory(prefix='q12b-independent-closure-') as tmp:
 root=pathlib.Path(tmp);run=root/'original';run.mkdir();(run/'sentinel').write_text('preserve')
 (root/'link').symlink_to(run,target_is_directory=True);file=root/'file';file.write_text('preserve-file')
 cases={'same':run,'inside':run/'new/subdir','symlink_to_run':root/'link','symlink_into_run':root/'link/new','ancestor':root,'existing_file':file}
 def census():return sorted((str(p.relative_to(root)), 'symlink:'+os.readlink(p) if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'directory') for p in root.rglob('*'))
 before=census()
 for name,out in cases.items():
  args=sys.argv;sys.argv=['reducer','--run',str(run),'--out',str(out),'--authorization','unused']
  err=io.StringIO();code=None
  try:
   with contextlib.redirect_stderr(err): ns['main']()
  except SystemExit as e:code=e.code
  finally:sys.argv=args
  assert code==2 and 'before any filesystem mutation' in err.getvalue() and census()==before,name
  checks.append('F2_no_write_'+name)
 out=root/'separate';assert ns['output_guard'](str(run),str(out)) is None;checks.append('separate_output_allowed')
 # Read only the original local small receipts via actual bind function, never raw tensors or reducer.
 original=ns['ORIGINAL_RUN'];real=V2/'experiments/review-response-20260927/runs/q1.2b-heldout'/original['run_id']
 auth={'approved':True,'authorizes':'CPU_REDUCTION_ONLY','run_id':original['run_id'],'reduction_id':'separate','repaired_reducer_sha256':ns['sha256_file'](P),'original_evidence':{k:original[k] for k in ['run_receipt_sha256','original_summary_sha256','launch_binding_sha256','gate_snapshot_sha256','procA_result_sha256','procB_result_sha256']}}
 pr,_=ns['bind_original_run'](str(real),'evaluation',original,str(out),auth);assert pr==[],pr;checks.append('synthetic_authorization_binds_real_original_receipts')
 mutations={'missing':None,'not_approved':{**auth,'approved':False},'wrong_purpose':{**auth,'authorizes':'GPU'},'wrong_run':{**auth,'run_id':'OTHER'},'wrong_repair_sha':{**auth,'repaired_reducer_sha256':original['original_reducer_sha256']},'wrong_reduction_id':{**auth,'reduction_id':'OTHER'},'wrong_original_receipt':{**auth,'original_evidence':{**auth['original_evidence'],'run_receipt_sha256':'0'*64}}}
 for name,a in mutations.items():
  pr,_=ns['bind_original_run'](str(real),'evaluation',original,str(out),a);assert pr,name;checks.append('auth_reject_'+name)
 assert not out.exists() and census()==before
launcher=BASE/'tools/run_q1_2b_reduction_repair_v2_2_1.sh';assert subprocess.run(['bash','-n',str(launcher)]).returncode==0
print(json.dumps({'scope':'independent local stdlib F1/F2/auth controls; synthetic in-memory authority is not launch approval; actual reduce_run is forbidden','source_sha256':ns['sha256_file'](P),'launcher_sha256':ns['sha256_file'](launcher),'checks':checks,'passed':len(checks),'shell_syntax':'PASS','no_original_or_candidate_output_mutation':True},indent=2))
