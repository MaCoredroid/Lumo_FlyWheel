#!/usr/bin/env python3
"""Tiny local AST probes of the draft only; no production imports, tensors, GPU, or Docker."""
import argparse,ast,hashlib,json,os,pathlib,sys,tempfile,contextlib,io
P=pathlib.Path(__file__).resolve().parent/'q1_2b_reduce_v2_2_1.py'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
t=ast.parse(P.read_text()); names={'main','reduce_run','bind_original_run','sha256_file'}; constants={'ORIGINAL_RUN','SCHEMA_SUMMARY'}
selected=[n for n in t.body if (isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in constants for x in n.targets))]
ns={'__file__':str(P),'argparse':argparse,'hashlib':hashlib,'json':json,'os':os,'sys':sys}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(P),'exec'),ns)
# Exact inherited self-hash predicate: immutable exp.reducer is the original reducer.
old=ns['ORIGINAL_RUN']['original_reducer_sha256'];new=sha(P)
assert old!=new
with tempfile.TemporaryDirectory(prefix='q12b-review-no-original-write-') as tmp:
 run=pathlib.Path(tmp)/ns['ORIGINAL_RUN']['run_id'];run.mkdir()
 oldargv=sys.argv;sys.argv=['reducer','--run',str(run),'--block','evaluation','--out',str(run),'--authorization',str(pathlib.Path(tmp)/'absent-auth.json')]
 code=None;buf=io.StringIO()
 try:
  with contextlib.redirect_stdout(buf): ns['main']()
 except SystemExit as e:code=e.code
 finally:sys.argv=oldargv
 p=run/'summary.v2_2_1.json'
 assert code==2 and p.exists()
 r=json.loads(p.read_text());assert any('OUTSIDE' in x for x in r['findings']['malformed'])
 output={'snapshot_reducer_sha256':new,'original_immutable_reducer_sha256':old,'F1_original_self_hash_comparison_refuses':old!=new,'F2_real_main_writes_inside_original_even_when_outside_guard_refuses':True,'F2_exit':code,'F2_written_file':'<temporary-original>/summary.v2_2_1.json','F2_outside_refusal':[x for x in r['findings']['malformed'] if 'OUTSIDE' in x],'scope':'local temporary metadata tree only; actual AST main/reduce/bind; no production imports or numerical execution'}
 print(json.dumps(output,indent=2))
