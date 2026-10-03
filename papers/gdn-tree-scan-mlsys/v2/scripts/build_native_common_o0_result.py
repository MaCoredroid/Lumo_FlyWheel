from pathlib import Path
import argparse,json,hashlib
B=Path(__file__).resolve().parents[1];O=B/'results/review-response-20260927';p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--review',required=True);a=p.parse_args();src=Path(a.result).resolve();review=Path(a.review).resolve();d=json.loads(src.read_bytes())
assert d['status']=='COMPLETE_SINGLE_RUN_AUTHENTICATED' and d['observations']==168 and d['r2']['complete_pairs']==84 and not d['cross_process_qualified'] and not d['full_q1_qualified'];assert review.is_file()
q=d['r2'];pairs=q['cases'];assert len(pairs)==84 and len({x['case_id'] for x in pairs})==84
for key,rowkey in [('decision_stable_pairs','decision_stable'),('o0_identical_pairs','o0_identical'),('o1_identical_pairs','o1_identical'),('full_logits_identical_pairs','full_logits_identical')]:assert q[key]==sum(x[rowkey] is True for x in pairs)
assert q['tie_observations']==sum(t>1 for x in pairs for t in x['ties'])
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
rec={'schema':'lumo.paper.native-common-o0-single-process.v1','source':str(src.relative_to(B)),'source_sha256':sha(src),'independent_review':str(review.relative_to(B)),'independent_review_sha256':sha(review),'run_id':d['run_id'],'counts':{k:v for k,v in q.items() if k!='cases'},'observations':168,'scope':'One target-only process,84 predetermined paths,two repeats; no A/B,candidate,MTP,lifecycle or workload qualification','status':'SINGLE_PROCESS_RESULT_REVIEWED','full_q1_qualified':False}
text=f'''% Bound source and independent review are in native-common-o0.json.
A subsequent controlled reference run imports the same archived starting state
for each path before native continuation. In its first process, all 168
observations (84 predetermined paths, two repetitions each) pass the source,
raw-data, finiteness, and ownership checks. The two repetitions reproduce
identical imported state for {q['o0_identical_pairs']} of 84 paths, identical published target
state for {q['o1_identical_pairs']}, and identical full-vocabulary logits for
{q['full_logits_identical_pairs']}; {q['decision_stable_pairs']} retain the same greedy decision.
There are {q['tie_observations']} exact-tie observations. This is a within-process result;
independent-process repeatability remains pending. The import controls the
starting target state, but does not yet establish compatible MTP state,
candidate continuation, or lifecycle behavior. The earlier natural-prefill
result remains a separate adverse observation.
'''
for path,value in [(O/'native-common-o0.json',json.dumps(rec,indent=2)+'\n'),(O/'native-common-o0.tex',text)]:
 with path.open('x') as f:f.write(value)
main=B/'main.tex';s=main.read_text();needle='\\input{results/review-response-20260927/native-reference}';assert s.count(needle)==1;main.write_text(s.replace(needle,needle+'\n\\input{results/review-response-20260927/native-common-o0}'))
print(json.dumps({'paper_source':str(O/'native-common-o0.tex'),'result_sha256':sha(src),'counts':rec['counts']}))
