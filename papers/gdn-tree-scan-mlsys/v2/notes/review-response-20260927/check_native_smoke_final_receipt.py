"""Review-only CPU mocks: actual library + actual receipt function + actual EXIT trap."""
import hashlib,json,pathlib,runpy,sys,tempfile
campaign=pathlib.Path(sys.argv[1]).resolve()
ns=runpy.run_path(str(campaign/'tools/tests/test_q1_native_smoke_launcher_v2_3_static.py'))
launcher=(campaign/'tools/run_q1_native_smoke_v2_3.sh').read_text()
receipt=launcher[launcher.index('write_receipt() {'):launcher.index("trap 'rc=$?;")]
trap=[x for x in launcher.splitlines() if x.startswith("trap '")][0]
checks=[]
for label,kwargs,primary,status,want in [
 ('success',{},0,'COMPLETED_driver_rc=0',0),
 ('stop_failed',dict(stop_rc='1',status='running',running='true'),0,'COMPLETED_driver_rc=0',8),
 ('driver_and_stop_failed',dict(stop_rc='1',status='running',running='true'),2,'COMPLETED_driver_rc=2',2),
 ('query_failed',dict(ps_rc='1'),0,'COMPLETED_driver_rc=0',8),
 ('boot_and_stop_failed',dict(stop_rc='1',status='running',running='true'),6,'ENGINE_NOT_HEALTHY',6),
]:
 with tempfile.TemporaryDirectory(prefix='review-smoke-receipt-') as d:
  env,out=ns['_env'](d,**kwargs)
  pre='set -uo pipefail; . "'+str(campaign/'tools/q1_native_smoke_cleanup_v2_3.sh')+'"; RECEIPT_WRITTEN=0\n'
  r=ns['_sh'](env,pre+receipt+'\n'+trap+'\nfinalize "'+status+'" '+str(primary)+'; exit $?\n')
  assert r.returncode==want,(label,r.returncode,r.stdout,r.stderr)
  out=pathlib.Path(out);doc=json.loads((out/'RUN-RECEIPT.json').read_text()); assert doc['engine']['finalized']==(out/'FINALIZED.txt').read_text().strip(),label
  assert 'FINALIZED.txt' in doc['files_recursive_excluding_object_store'],label
  mismatches=[]
  for name,obj in doc['files_recursive_excluding_object_store'].items():
   raw=(out/name).read_bytes()
   if len(raw)!=obj['bytes'] or hashlib.sha256(raw).hexdigest()!=obj['sha256']:mismatches.append(name)
  assert not mismatches,(label,mismatches)
  checks.append(dict(case=label,exit=r.returncode,files_verified=len(doc['files_recursive_excluding_object_store']),finalized=doc['engine']['finalized']))
print(json.dumps(checks,indent=2))
