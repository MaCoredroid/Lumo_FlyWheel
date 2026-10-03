"""Derive a paper result from a reviewed complete native joint A/B reduction.

No launch or qualification authority. Preparation can precede a result; output
cannot: the caller must supply a hash-bound independent review and parent receipt.
The current manuscript is not edited by this script.
"""
from pathlib import Path
import argparse
import hashlib
import json

B = Path(__file__).resolve().parents[1]
C = B / 'experiments/review-response-20260927'
CORPUS = C / 'fullmodel/native-joint-common-o0-v1/CORPUS.json'
POLICY = B / 'p0/monitor/review-response-20260927/Q1-JOINT-MTP-CATEGORICAL-CONTRACT-v1.json'
CORPUS_SHA = 'ac3f1248aeb5343714334b402c9f0cd6ecaca856c68cae8c625a5c9b5b7b8f61'
POLICY_SHA = '4dad3bf5918c31d734214d8d2579ca1bb5b75500b575dbf2c0718287a287c593'
PHASES = ('before_z_first', 'before_z_follow', 'after_z_first', 'target_o2')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound_file(item):
    path = (B / item['path']).resolve()
    require(path.is_relative_to(B.resolve()) and path.is_file(), 'evidence outside paper root or absent')
    require(sha(path) == item['sha256'], 'evidence hash mismatch: ' + item['path'])
    return path


def derive(receipt_path):
    receipt = read(receipt_path)
    require(receipt['schema'] == 'lumo.paper.native-joint-result-review.v1' and
            receipt['reviewed'] is True, 'review receipt required')
    reduction_path = bound_file(receipt['reduction'])
    review_path = bound_file(receipt['independent_review'])
    seal_path = bound_file(receipt['review_seal'])
    # A parent must explicitly bind these files after reviewing the raw audit.
    # Merely finding an existing note or preparation receipt is not sufficient.
    require(receipt['scope'] == 'complete native joint A/B raw and categorical review' and
            receipt['observations'] == 336 and receipt['cases'] == 84,
            'review does not cover the complete frozen cohort')
    require(sha(CORPUS) == CORPUS_SHA and sha(POLICY) == POLICY_SHA,
            'frozen population or prospective criterion changed')
    cm, policy, reduced = read(CORPUS), read(POLICY), read(reduction_path)
    ids = cm['case_ids']
    require(len(ids) == len(set(ids)) == 84 and ids == policy['population']['case_ids'],
            'case population differs')
    require(reduced['schema'] == 'lumo.q1.native-joint-common-o0-reduction.v1' and
            reduced['corpus_sha256'] == CORPUS_SHA and reduced['complete'] is True and
            reduced['observations'] == 336, 'incomplete or foreign reduction')
    sources_path = C / cm['joint_sources_rel']
    require(sha(sources_path) == cm['joint_sources_sha256'] == reduced['joint_sources_sha256'],
            'joint source identity differs')
    require(all(reduced[k] is False for k in
                ('candidate_qualified', 'full_Q1_qualified', 'launch_authority')),
            'native result overstates its scope')
    rows = reduced['rows']
    expected = {(cid, process, rep) for cid in ids for process in ('A', 'B') for rep in (0, 1)}
    require(len(rows) == 336 and {(r['case_id'], r['process'], r['repeat']) for r in rows} == expected,
            'missing, duplicated or foreign observations')
    require(all(r['arm'] == 'aligned_nonpacked' and r['valid'] is True for r in rows),
            'invalid or foreign observation')
    target_summary = reduced['arms']['aligned_nonpacked']
    categorical = reduced['native_categorical']
    require(target_summary['denominator'] == categorical['denominator'] == 84 and
            categorical['policy_sha256'] == POLICY_SHA and
            [r['case_id'] for r in target_summary['cases']] == ids and
            [r['case_id'] for r in categorical['cases']] == ids,
            'summary population differs')
    counts = dict(target_o0_identical=0, target_o1_identical=0, target_logits_identical=0,
                  target_greedy_stable=0, target_baseline_eligible=0, joint_baseline_eligible=0)
    phases = {p: dict(inputs_matched=0, categorical_stable=0, raw_logits_identical=0,
                     exact_tie_observations=0) for p in PHASES}
    cases = []
    for i, cid in enumerate(ids):
        group = [r for r in rows if r['case_id'] == cid]
        target = target_summary['cases'][i]
        cat = categorical['cases'][i]
        require(target['complete'] is True and cat['complete'] is True, 'incomplete case')
        target_checks = {name: len({r[key] for r in group}) == 1 for name, key in
                         [('o0_identical', 'o0'), ('o1_identical', 'o1'),
                          ('raw_logits_identical', 'raw_sha256'), ('decision_stable', 'winner')]}
        require(all(target[k] is value for k, value in target_checks.items()),
                'target summary differs from observations')
        eligible_target = target_checks['o0_identical'] and target_checks['decision_stable']
        require(target['baseline_qualified'] is eligible_target, 'target eligibility differs')
        same_joint = len({r['joint_source_digest'] for r in group}) == 1
        require(cat['joint_o0_identical'] is same_joint, 'joint source summary differs')
        phase_checks = {}
        for phase in PHASES:
            projections = [r['native_categorical'] for r in group]
            require(all(p['policy_sha256'] == POLICY_SHA and p['qualification'] is False and
                        p['candidate_selector_observed'] is False for p in projections),
                    'invalid categorical projection')
            points = [p['phases'][phase] for p in projections]
            matched = len({(p['logical_extent'], p['input_token'], p['input_position'],
                            p['native_dtype'], p['operator'], p.get('k')) for p in points}) == 1
            stable = matched and len({(p['spine'], tuple(p.get('ordered_top3', []))) for p in points}) == 1
            fields = dict(inputs_matched=matched, categorical_stable=stable,
                          raw_logits_identical=len({p['raw_sha256'] for p in points}) == 1,
                          exact_tie_observations=sum(p['ties'] > 1 for p in points))
            require(all(cat['phases'][phase][k] == v for k, v in fields.items()) and
                    cat['phases'][phase]['points'] == points, 'categorical summary differs from observations')
            for name, value in fields.items():
                phases[phase][name] += int(value)
            phase_checks[phase] = fields
        eligible_joint = same_joint and all(p['categorical_stable'] for p in phase_checks.values())
        require(cat['native_categorical_baseline_qualified'] is eligible_joint, 'joint eligibility differs')
        for name, key in [('target_o0_identical', 'o0_identical'), ('target_o1_identical', 'o1_identical'),
                          ('target_logits_identical', 'raw_logits_identical'), ('target_greedy_stable', 'decision_stable')]:
            counts[name] += int(target_checks[key])
        counts['target_baseline_eligible'] += int(eligible_target)
        counts['joint_baseline_eligible'] += int(eligible_joint)
        cases.append(dict(case_id=cid, target=target_checks, phases=phase_checks,
                          target_baseline_eligible=eligible_target, joint_baseline_eligible=eligible_joint))
    require(counts['target_baseline_eligible'] == target_summary['qualified'] and
            counts['joint_baseline_eligible'] == categorical['baseline_qualified'], 'aggregate counts differ')
    require(reduced['target_native_baseline_qualified'] is (counts['target_baseline_eligible'] == 84) and
            reduced['joint_native_categorical_baseline_qualified'] is (counts['joint_baseline_eligible'] == 84),
            'aggregate eligibility differs')
    return dict(schema='lumo.paper.native-joint-result.v1', observations=336, cases=84,
                processes=2, repeats=2, corpus_sha256=CORPUS_SHA, policy_sha256=POLICY_SHA,
                joint_sources_sha256=sha(sources_path), reduction=receipt['reduction'],
                independent_review=receipt['independent_review'], review_seal=receipt['review_seal'],
                parent_receipt_sha256=sha(receipt_path), counts=counts, phases=phases, case_rows=cases,
                candidate_qualified=False, full_q1_qualified=False, performance_claim=False)


def render(result):
    c, phases = result['counts'], result['phases']
    phase_rows = '\n'.join(label + ' & ' + str(phases[key]['categorical_stable']) + '/84 & ' +
                           str(phases[key]['raw_logits_identical']) + '/84 \\\\' for key, label in
                           [('before_z_first', 'MTP root, before $z$'),
                            ('before_z_follow', 'MTP follow-up, before $z$'),
                            ('after_z_first', 'MTP root, after $z$'),
                            ('target_o2', 'Next target decision')])
    return f'''% Generated from reviewed complete native joint A/B evidence; see companion JSON.
A separate reference experiment imports both target and MTP starting state and
records native continuation on the same 84 predetermined paths, with two repetitions
in each of two independent processes (336 observations). All observations pass
the recorded raw-data, ownership, finiteness, and fixed-population audit.
Across the four observations per path, target starting state is identical for
{c['target_o0_identical']}/84 paths, published target state for {c['target_o1_identical']}/84,
full target logits for {c['target_logits_identical']}/84, and greedy decisions are
stable for {c['target_greedy_stable']}/84.

\\begin{{center}}
\\begin{{tabular}}{{lrr}}
\\toprule
Native continuation phase & Categorical & Raw logits \\\\
\\midrule
{phase_rows}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
Here $z$ is the pending target token. Categorical agreement requires matched inputs and identical actual spine and
ordered top-three selections at each MTP phase, and the same next target decision.
Raw-logit equality is reported separately. The joint categorical reference
prerequisite holds for {c['joint_baseline_eligible']}/84 paths under the frozen
criterion. This native reference experiment does not qualify the tree candidate,
continuous cycles, lifecycle behavior, or workload performance.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-receipt', required=True)
    parser.add_argument('--output-stem', required=True)
    args = parser.parse_args()
    result = derive(args.review_receipt)
    stem = Path(args.output_stem)
    outputs = {stem.with_suffix('.json'): json.dumps(result, indent=2) + '\n',
               stem.with_suffix('.tex'): render(result)}
    require(all(not p.exists() for p in outputs), 'paper result already exists')
    for path, value in outputs.items():
        with path.open('x') as stream:
            stream.write(value)
    print(json.dumps(dict(outputs=[str(p) for p in outputs], counts=result['counts'])))


if __name__ == '__main__':
    main()
