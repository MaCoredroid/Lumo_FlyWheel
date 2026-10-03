"""Integrate the independently sealed A-only result without A/B qualification."""
from pathlib import Path
import hashlib
import json

B = Path(__file__).resolve().parents[1]
ROOT = B / 'p0/monitor/review-response-20260927/native-joint-common-o0-A-full-independent'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    seal = json.loads((ROOT / 'REVIEW-SEAL.json').read_text())
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    review = B / manifest['report_path']
    assert sha(ROOT / 'REVIEW-SEAL.json') == '2047ecc8d1aef269ede30a25507bc8bcc5a553ef50581aad17d22bf99ea04372'
    assert seal['status'] == 'PASS_COMPLETE_A_RAW_AND_TERMINAL'
    assert sha(ROOT / 'MANIFEST.json') == seal['manifest_sha256']
    assert sha(review) == seal['report_sha256'] == manifest['report_sha256']
    for name, row in manifest['files'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    summary_path = ROOT / 'A-FINAL-SUMMARY.json'
    assert sha(summary_path) == seal['summary_sha256']
    d = json.loads(summary_path.read_text())
    corpus = B / 'experiments/review-response-20260927/fullmodel/native-joint-common-o0-v1/CORPUS.json'
    assert sha(corpus) == d['plan_sha256'] == 'ac3f1248aeb5343714334b402c9f0cd6ecaca856c68cae8c625a5c9b5b7b8f61'
    ids = json.loads(corpus.read_text())['case_ids']
    assert [c['case_id'] for c in d['cases']] == ids and len(ids) == len(set(ids)) == 84
    assert d['denominator_cases'] == d['complete_pairs'] == 84
    assert d['audited_valid_observations'] == d['expected_observations'] == d['unique_req_ids'] == 168
    assert not d['missing_observations'] and not d['failed_observations'] and d['standalone_A_raw_valid_complete'] is True
    assert all(d[k] is seal[k] is False for k in ('A_B_baseline_qualified', 'candidate_qualified', 'full_Q1_qualified'))
    for case in d['cases']:
        assert case['complete_valid_pair'] is True and len(case['observations']) == 2
        assert all(o['valid'] is True for o in case['observations'])
        for repeat, observation in enumerate(case['observations']):
            assert observation['obs_id'].endswith('__pA__r' + str(repeat) + '__' + case['case_id'])
        for name in ('o0', 'o1', 'winner', 'raw_sha256', 'mtp_initial_digest', 'joint_source_digest'):
            assert case['target'][name] is True
        for phase in ('before_z_first', 'before_z_follow', 'after_z_first', 'target_o2'):
            check = case['phase_checks'][phase]
            assert all(check[k] is True for k in ('inputs_matched', 'categorical_stable', 'raw_logits_identical'))
            points = check['points']
            assert len(points) == 2 and points[0]['raw_sha256'] == points[1]['raw_sha256']
            assert points[0]['spine'] == points[1]['spine'] and points[0].get('ordered_top3') == points[1].get('ordered_top3')
    text = r'''% A-only raw review; source identities and scope are bound in native-joint-A-v1.json.
A separate reference experiment imports both target and MTP starting state.
Its first process completes two repetitions of all 84 predetermined paths
(168 independently authenticated observations). Within that process, imported
target state, published target state, and the complete next-target logit vector
are identical across repetitions for all 84 paths. The three recorded MTP phases
---the first and self-fed follow-up before the pending target token, and the first
after that token---have matched inputs and identical spine tokens, ordered
top-three selections, and full logit vectors for all 84 paths.
This establishes within-process reference repeatability only. The second-process
comparison, tree-candidate qualification, continuous cycles, and lifecycle checks
remain pending; this result supplies no performance or workload claim.
'''
    result = dict(schema='lumo.paper.native-joint-A.v1', summary=str(summary_path.relative_to(B)),
                  summary_sha256=sha(summary_path), review=str(review.relative_to(B)), review_sha256=sha(review),
                  seal=str((ROOT / 'REVIEW-SEAL.json').relative_to(B)), seal_sha256=sha(ROOT / 'REVIEW-SEAL.json'),
                  corpus_sha256=sha(corpus), cases=84, observations=168, processes=1, repeats=2,
                  target_counts=d['target_counts'], phase_counts=d['phase_counts'],
                  A_B_baseline_qualified=False, candidate_qualified=False, full_Q1_qualified=False,
                  performance_claim=False, tex_sha256=hashlib.sha256(text.encode()).hexdigest())
    for name, value in [('native-joint-A-v1.json', json.dumps(result, indent=2) + '\n'), ('native-joint-A-v1.tex', text)]:
        with (B / 'results/review-response-20260927' / name).open('x') as stream:
            stream.write(value)
    path = B / 'main.tex'
    source = path.read_text()
    anchor = r'\input{results/review-response-20260927/native-common-o0-paired-v1}'
    new = r'\input{results/review-response-20260927/native-joint-A-v1}'
    assert source.count(anchor) == 1 and new not in source
    path.write_text(source.replace(anchor, anchor + '\n\n' + new))
    print(json.dumps(dict(cases=84, observations=168, A_B_baseline_qualified=False)))


if __name__ == '__main__':
    main()
