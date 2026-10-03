"""Integrate an independently audited A/B result; preserve single-process files."""
from pathlib import Path
import argparse,json,hashlib
B=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-dir',required=True);p.add_argument('--review',required=True);a=p.parse_args()
 root=Path(a.audit_dir).resolve();review=Path(a.review).resolve();m=json.loads((root/'MANIFEST.json').read_text())
 identities={r['path']:r for r in m['files']}
 assert len(identities)==len(m['files']) and {'RESULT.json','TARGET-ONLY-REPEAT-SUMMARY.json','FROZEN-REDUCTION.json'} <= set(identities)
 for r in m['files']:
  path=root/r['path'];assert path.resolve().is_relative_to(root) and path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
 d=json.loads((root/'RESULT.json').read_text());summary=json.loads((root/'TARGET-ONLY-REPEAT-SUMMARY.json').read_text());rows=json.loads((root/'FROZEN-REDUCTION.json').read_text())['rows']
 assert d['status']=='PAIRED_RAW_AUTHENTICATED' and d['observations']==336 and d['complete_cases']==84 and review.is_file()
 assert all(d[k] is False for k in ('full_q1_qualified','mtp_qualified','candidate_qualified','heldout_qualified','workload_qualified'))
 assert len(rows)==336 and len({(x['case_id'],x['process'],x['repeat']) for x in rows})==336
 assert summary['denominator']==summary['complete_cases']==84 and len(summary['cases'])==84
 croot=B/'experiments/review-response-20260927'
 freeze_path=croot/'FREEZE-Q1-NATIVE-COMMON-O0-v2.json';assert sha(freeze_path)==d['freeze_sha256']=='ceb210445886cd334c5b266a171d16a8877ad77dde599cd28f451413608371cd'
 freeze=json.loads(freeze_path.read_text());sources_path=croot/'fullmodel/native-common-o0-v1/SOURCES.json'
 assert sha(sources_path)==freeze['source_hashes']['shared_sources']
 sources=json.loads(sources_path.read_text());fixture_path=croot/'fullmodel/fixtures/token-fixtures.v1.json';assert sha(fixture_path)==sources['fixtures_sha256']
 fixtures=json.loads(fixture_path.read_text());source_map={x['case_id']:x for x in sources['sources']}
 assert len(source_map)==len(sources['sources'])==84
 case_ids=[c['case_id'] for c in fixtures['cases'] if c['block']=='calibration']
 assert len(case_ids)==len(set(case_ids))==84 and {x['case_id'] for x in rows}==set(case_ids)
 assert {x['case_id'] for x in summary['cases']}==set(case_ids)==set(source_map)
 for c in summary['cases']:
  group=[x for x in rows if x['case_id']==c['case_id']]
  assert len(group)==4 and {(x['process'],x['repeat']) for x in group}=={('A',0),('A',1),('B',0),('B',1)}
  assert c['complete'] is True and all(x['arm']=='aligned_nonpacked' and x.get('valid') is True and x.get('target_only') is True for x in group)
  assert c['winners']==sorted({x['winner'] for x in group})
  checks=dict(common_o0=all(x['o0']==source_map[c['case_id']]['logical_digest'] for x in group),o1_identical=len({x['o1'] for x in group})==1,logits_identical=len({x['raw_sha256'] for x in group})==1,decision_stable=len({x['winner'] for x in group})==1)
  assert all(c[k] is v for k,v in checks.items())
  assert c['target_decision_prerequisite'] is (checks['common_o0'] and checks['decision_stable'])
 for output,key in [('common_o0_cases','common_o0'),('o1_identical_cases','o1_identical'),('full_logits_identical_cases','logits_identical'),('decision_stable_cases','decision_stable')]:assert d[output]==sum(x[key] is True for x in summary['cases'])
 assert d['tie_observations']==sum(x['ties']>1 for x in rows)
 assert d['target_decision_prerequisite_cases']==summary['target_decision_prerequisite_cases']==sum(x['target_decision_prerequisite'] is True for x in summary['cases'])
 assert d['all_target_decisions_pass'] is summary['all_target_decisions_pass'] is all(x['target_decision_prerequisite'] is True for x in summary['cases'])
 assert summary['full_q1_qualified'] is summary['mtp_qualified'] is False
 rec=dict(schema='lumo.paper.native-common-o0-paired.v1',fixtures_sha256=sha(fixture_path),sources_sha256=sha(sources_path),freeze_sha256=sha(freeze_path),source=str((root/'RESULT.json').relative_to(B)),source_sha256=sha(root/'RESULT.json'),audit_manifest=str((root/'MANIFEST.json').relative_to(B)),audit_manifest_sha256=sha(root/'MANIFEST.json'),independent_review=str(review.relative_to(B)),independent_review_sha256=sha(review),observations=336,cases=84,processes=2,repeats=2,counts={k:d[k] for k in ['common_o0_cases','o1_identical_cases','full_logits_identical_cases','decision_stable_cases','tie_observations','target_decision_prerequisite_cases']},scope=d['scope'],full_q1_qualified=False)
 text=f'''% Source/raw audit and independent review are bound in native-common-o0-paired-v1.json.
A subsequent controlled reference experiment imports the same archived target
starting state for each path before native continuation. Across two independent
processes and two repetitions per process, all 336 observations (84 predetermined
paths) pass the source, raw-data, finiteness, and ownership checks.
All four observations use identical imported target state for
{d['common_o0_cases']} of 84 paths; published target state is identical for
{d['o1_identical_cases']}, full-vocabulary logits for {d['full_logits_identical_cases']},
and the greedy decision for {d['decision_stable_cases']}. There are
{d['tie_observations']} exact-tie observations.
These checks establish the target decision prerequisite for
{d['target_decision_prerequisite_cases']} of 84 paths under the controlled starting state.
They do not establish compatible MTP state, candidate continuation, continuous
cycles, or lifecycle behavior. The earlier natural-prefill result remains a
separate adverse observation; no performance or workload claim follows.
'''
 out=B/'results/review-response-20260927'
 for path,value in [(out/'native-common-o0-paired-v1.json',json.dumps(rec,indent=2)+'\n'),(out/'native-common-o0-paired-v1.tex',text)]:
  with path.open('x') as f:f.write(value)
 main=B/'main.tex';s=main.read_text();old='\\input{results/review-response-20260927/native-common-o0}';assert s.count(old)==1
 main.write_text(s.replace(old,'\\input{results/review-response-20260927/native-common-o0-paired-v1}'))
 print(json.dumps(rec))
if __name__=='__main__':main()
