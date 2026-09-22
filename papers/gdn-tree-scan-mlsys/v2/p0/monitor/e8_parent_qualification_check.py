"""Independent B1 qualification ledger reconstruction; no inference or timing estimate."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, re
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--arm',choices=['on','off'],required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
r=a.root/('qualification_'+a.arm)
receipt=json.loads((a.root/('qualification_'+a.arm+'.json')).read_text())
for rel,digest in receipt['evidence_sha256'].items():
 assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==digest,rel
events=[json.loads(s) for s in (r/'logs/e1_events.jsonl').read_text().splitlines() if s.strip()]
assert [e['n'] for e in events]==list(range(1,len(events)+1))
assert events[-1]['event']=='run_close' and events[-1]['errors']==events[-1]['sink_failures']==0 and not events[-1]['sink_failed']
seq=collections.defaultdict(list);pure=set();discarded=0
for e in events:
 if e['event']=='output_rows':
  assert len(e['rows'])==e['num_reqs']==1
  for row in e['rows']:
   assert row['n_emitted']==len(row['emitted_ids'])
   if row['discarded']:
    assert e['kind']=='nonpure' and row['n_emitted']==0
    discarded+=1
    continue
   seq[row['request_id']].extend(row['emitted_ids'])
   if e['kind']=='pure':pure.add(row['request_id'])
requests=[json.loads(f.read_text()) for f in sorted((r/'cohort').glob('*/capture_request.json'))]
assert len(requests)==8
frozen=json.loads((a.stage/'frozen/e1/frozen_prefixes.json').read_text())
expected={x['id']:x['prefix_sha256'] for x in frozen['pilot']}
assert set(x['prefix_id'] for x in requests)==set(expected)
matched={};trunc={}
for q in requests:
 assert q['prompt_sha256']==q['prefix_sha256']==expected[q['prefix_id']]
 assert q['seed']==q['request']['seed']==20260921 and q['request']['temperature']==0
 assert q['request']['max_tokens']==q['max_tokens']==32 and q['error'] is None
 candidates=[rid for rid in seq if rid==q['response_id'] or rid.startswith(q['response_id']+'-')]
 assert len(candidates)==1,candidates
 rid=candidates[0];api=[]
 for token in q['response_logprobs_tokens']:
  assert re.fullmatch(r'token_id:\d+',token)
  api.append(int(token.split(':')[1]))
 assert len(api)==q['usage']['completion_tokens'] and 0<len(api)<=32 and seq[rid][:len(api)]==api
 if q['finish_reason']=='length': assert len(api)==32
 else: assert q['finish_reason']=='stop'
 matched[rid]=len(api);trunc[q['prefix_id']]=len(seq[rid])-len(api)
assert set(matched)==set(seq)==pure
head=json.loads((r/'logs/e8_head_gate.json').read_text());n=sum(e['event']=='forward_entry' for e in events)
assert head['proposals']==head['qualified_proposals']==n
assert head['primary_head_calls']==head['legacy_head_calls']==head['checked_heads']==5*n
assert head['root_heads']==n and head['loop_heads']==4*n and head['closed'] and not head['failures']
join=json.loads((r/'join.json').read_text());assert join['token_evidence']==matched and join['n_usable']==0 and not join['invalid'] and not join['refused']
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_'+a.arm.upper()+'_ARM_ONLY','scope':'Independent reconstruction of complete API streams and ledger/head counts; full-logit comparison is executed by the pinned gate, not re-derived from exported logits','receipt_files_verified':len(receipt['evidence_sha256']),'requests':8,'direct_api_tokens':sum(matched.values()),'proposals':n,'paired_head_checks':5*n,'untimed_physical_intervals':sum(e['event']=='physical_step' for e in events),'timed_support_intervals':0,'discarded_empty_nonpure_rows':discarded,'terminal_api_clipping_by_prefix':trunc,'ledger_events':len(events),'source_qualification_receipt_sha256':hashlib.sha256((a.root/('qualification_'+a.arm+'.json')).read_bytes()).hexdigest(),'timing_authorized':False}
assert not a.out.exists()
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
