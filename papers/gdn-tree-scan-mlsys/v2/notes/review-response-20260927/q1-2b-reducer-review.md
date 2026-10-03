# Q1.2b reducer and evidence-seal review — 2026-09-27

**Verdict: BLOCKED for qualification.** This is a CPU-only preparation review, not permission to launch. The frozen metadata currently has the intended denominator, but the reducer accepts materially incomplete or inconsistent evidence as `PASS`. No GPU, model, API, remote write, fixture tensor generation, or production change was performed.

## Reviewed identity and positive checks

All 19 files in `FREEZE-Q1_2B.json.files` are locally present and match their recorded SHA-256 values. Both actual metadata canonical hashes recompute correctly. There are exactly 11,520 unique declared cases overall: 5,760 calibration and 5,760 held out. Each of the four fixtures has 2,880 cases (1,536 output, 1,344 state), representing 138,240 head observations; the calibration denominator is **276,480**, per numerical pass. The current Q1.2b freeze declares **two repeats in each of A and B**, not the separate earlier eight-repeat native campaign.

| Reviewed file (relative to experiment root) | SHA-256 |
|---|---|
| `tools/q1_2b_reduce.py` | `684103fd28684defceaa73d32e6960b584de5f55cbcb4bde99bcdf9e6ff27fc9` |
| `tools/tests/test_q1_2b_reduce.py` | `3cbfe69b04a1a3bf0e932c671393a421b3e57dc0e303c98c0b90e71590a9cf1c` |
| `tools/q1_component_runner.py` | `e0eb4cb7eae9ed224ff03618d0fd4a31cc84a78f537278d39118526fe23d54d2` |
| `tools/q1_policy_evaluator_v2.py` | `a65f492b0b576221de328852c84b5b367a89dca2d542b9dde7062c410ed3ff0f` |
| `FREEZE-Q1_2B.json` | `00a3a0a2239fdcca3ca10832d2086eaa17a520e14fe75919fcdabfe9febd19bb` |
| `fixtures/q1_2b/manifest.json` | `2dc0a42921e6e9eda9d5b636060a246b6d4af98b88a3b2f78c9a0dbc61197522` |
| `fixtures/q1_2b/expected_observations.json` | `c5df3aefd4b838ef2098b03cfc54775b1cb467510c5956edb9e868c725ef7448` |

The 18 supplied test cases (one positive plus 17 parameterized failures) passed when called directly. Local Python environments have no `pytest`; I used a minimal marker-registration shim and executed every original assertion. This is explicitly **not a pytest invocation**. Boolean numerical metrics, false structural flags, and missing process-A raw records correctly refuse. Malformed JSON refuses operationally via an exception, but currently does not yield the documented structured exit-2 report.

## Material blockers and minimum repair

### R1. Frozen denominator and provenance are not consumed by the reducer

`tools/q1_2b_reduce.py:38–52,61–75` checks equality of two *claimed* canonical strings, filters a caller-supplied case list, and compares source fields only between A and B. It never rehashes either metadata input against the gate/freeze or recomputes the canonical payload. Missing identity fields in both attestations compare equal. Process block, repeat count, fixture counters and `integrity_ok` are unused.

Independent controls returned PASS after truncating four cases to one; adding a duplicate case (count reported as five); changing the fixture payload SHA while preserving its claimed canonical; removing every source/library/hardware identity field except policy/time/tag in both processes; and setting both process results to `integrity_ok=false`, `block=evaluation`, `repeats=-9` and negative fixture counts. The real metadata is not defective: **the missing enforcement is in replay/reduction**. The launcher checks gate hashes before execution, but that does not make later standalone reduction source-bound.

Minimum fix: accept a hash-bound run/freeze or approved gate snapshot; validate its expected block, exact repeats, process set and all source/image/metadata identities; recompute canonical payloads; require typed, unique fixture/case/observation keys and exact full selected-block products. Pin both evaluator modules (`q1_policy_evaluator_v2.py` imports `q1_policy_evaluator.py`). Bind mandatory actual identities to expected values, not merely to each other. Reject unexplained omissions, extras and inconsistent counters before numerical reduction. Preserve held-out exclusion.

### R2. “Sealed” metric files and process-B numerical references are not sealed or reconciled

`q1_2b_reduce.py:110–123` reads only A raw metrics and never uses its own `sha256_file`. `q1_component_runner.py:78–87,244–266` records candidate/native tensor hashes but does not hash each JSON file into its case index. The index only says `sealed: true`; the reducer does not even check this flag or index membership. The launch receipt at `run_q1_2b_component.sh:64–68` hashes root-level files, omitting nested process results/raw records. Thus no retained receipt currently authenticates the metric byte set.

PASS controls: remove **all** `procB/raw`; set a B candidate metric invalid; empty both case indexes; replace every A raw candidate/native hash and dtype with nonsense; delete `case_id`, `stratum` and `reference_valid` from raw records. The evaluator treats the IDs as optional and defaults missing reference validity. Changing raw B’s references is equally invisible because B raw is never read. Candidate aggregate hashes alone cannot establish B’s C1/C2 references or its recorded metrics.

Minimum fix: seal an exact per-process metric inventory with path, case identity, byte count and file SHA; bind it from the terminal result/receipt, with missing/extra/duplicate refusal. Read/hash both A and B. Require raw case, fixture/instance/path/node, stratum, surface/depth/head shape, dtype and C0/C1/C2 identity fields and reconcile them against expected metadata/repeat evidence. Bind C2 to the expected reference hash. Require the complete per-process numerical/reference observations (or explicitly prove their identity before reducing one copy). Preserve tensor-retention/recomputation decisions from the parent runner audit; this review does not prescribe an additional GPU campaign.

### R3. Repeat/determinism evidence is incomplete and accepts vacuous evidence

`q1_2b_reduce.py:84–96` trusts `within_process_bitwise` and compares lists with `zip`; it does not recompute within-process equality, validate repeat IDs, require nonempty expected scan/path maps, or check native repeat identity. It only demands equal repeat-list lengths of at least two. `q1_component_runner.py:235–241` recomputes native C1 in each repeat, but `:244–266` retains raw metrics only for repeat zero and `:272–279` checks candidate hashes alone. No native repeat hash census is retained to prove the comment’s native-repeat stability.

PASS controls: both repeat lists contain empty scan hashes, empty publication maps, Boolean duplicate repeat IDs, and false per-repeat ring flags, while summaries remain true; or both A and B change their second-repeat scan output identically, contradicting within-process determinism. The native/candidate hashes in raw records are unchecked as in R2.

Minimum fix: retain C1 output/state hashes for every expected instance/node/path and repeat; validate exact repeat indices `{0,1}` under this freeze, exact scan and accepted-path products, and hash syntax/shape. Independently derive within- and cross-process candidate **and native** determinism from raw hashes. Reconcile per-publication ring, pointer, control, tail and replay-count evidence with structural summaries rather than accepting vacuous maps. Do not infer C1 identity from C0 identity.

### R4. Negative-control denominator is not exact; N5 tests the wrong equality

`q1_2b_reduce.py:98–106,129–143` inspects structural tags only in A, accepts duplicate tags by overwrite, discovers numerical negatives from directory contents, pools power across fixtures, and makes N5 raw records optional. Negative record identity/depth/head count is self-declared. It accepts Boolean `leaked_instances=false` as integer zero. Actual `run_negatives` intends, **per fixture/process**, N1=48, N3=2 affected instances, N4=48 and N5=48 numerical records, plus the N7 engagement control.

PASS controls: omit all B negative tags; omit all N5 raw files; omit every negative raw file from a second fixture while first-fixture negatives remain; use Boolean N5 leak count. These are missing controls, not evidence of negative power.

Also independently confirmed the parent’s source finding: `q1_component_runner.py:315–323` declares N5 “leak” by comparing poisoned publication with native C1 bitwise. A valid C0 that differs harmlessly from C1 under the approved paired rule can therefore fail N5, and a corruption that moves C0 toward C1 could be misclassified. The causal invariant is **poisoned C0 equals the same unpoisoned C0**, holding inputs/path/initial state fixed; ordinary C1/C2 numerical eligibility remains a separate gate.

Minimum fix: freeze exact negative IDs, affected instances, paths and expected outcomes for both processes and each fixture; verify that product and seals before checking power. Require real integer counters, strict Boolean flags, no duplicate/extra tags, and mandatory N5 evidence. Retain and compare unpoisoned/poisoned C0 bytes or hashes for N5, separately enforcing the ordinary paired C1/C2 rule. Keep calibration-only controls out of the held-out block.

## Targeted CPU outcomes

Every row below used a fresh temporary fixture; real evidence and source files were not edited. Invalid-evidence PASS rows are blockers, not successful scientific tests.

| Independent case | Observed outcome |
|---|---|
| `baseline` | exit 0, PASS |
| `all_procB_raw_omitted` | exit 0, PASS |
| `procB_candidate_metrics_invalid` | exit 0, PASS |
| `expected_cases_truncated_4_to_1` | exit 0, PASS |
| `expected_case_duplicated` | exit 0, PASS |
| `fixture_sha_changed_keep_claimed_canonical` | exit 0, PASS |
| `source_attestation_fields_omitted_both` | exit 0, PASS |
| `empty_scan_publish_and_bool_repeat_ids` | exit 0, PASS |
| `both_processes_same_within_process_drift` | exit 0, PASS |
| `empty_case_inventory_both` | exit 0, PASS |
| `integrity_false_wrong_block_negative_repeats` | exit 0, PASS |
| `candidate_native_hashes_wrong_and_dtypes_wrong` | exit 0, PASS |
| `case_stratum_reference_valid_fields_omitted` | exit 0, PASS |
| `procB_negative_tags_omitted` | exit 0, PASS |
| `N5_raw_omitted` | exit 0, PASS |
| `N5_false_leaked_count` | exit 0, PASS |
| `second_fixture_all_negative_raw_omitted` | exit 0, PASS |
| `malformed_process_json` | JSONDecodeError |
| `malformed_raw_json` | JSONDecodeError |
| `boolean_metric_control` | exit 2, REFUSED_OR_MALFORMED |
| `false_structural_flag_control` | exit 5, None |
| `missing_procA_raw_control` | exit 2, None |

## Exact reproducer

Executed with the bundled Python using `-B`; temporary script `/tmp/q1_2b_independent_review.py`, results `/tmp/q1_2b_independent_review.json`. The script is reproduced below so `/tmp` retention is not required. It imports only the CPU reducer/evaluator/test helper, never the GPU runner. Its locally approved **synthetic temporary** policy is used only to reach reducer branches; the actual frozen policy remains unapproved and unchanged.

Reproducer SHA-256: `a892408e6763d1c1157a6ceec4b4746fdd2f31a1acde3740859bddf5a6f0c317`.

```python
import sys, types, importlib.util, pathlib, tempfile, json, copy, hashlib, shutil
ROOT=pathlib.Path('/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927')
params=[]
class Marks:
    def skipif(self,*a,**kw): return lambda f:f
    def parametrize(self,names,values):
        params.extend(values)
        return lambda f:f
sys.modules['pytest']=types.SimpleNamespace(mark=Marks())
spec=importlib.util.spec_from_file_location('qtests',ROOT/'tools/tests/test_q1_2b_reduce.py'); T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
T.test_valid_run_passes()
for a,b in params:T.test_failure_classes(a,b)
rows=[]
def execute(name, mutate=None, after=None):
    with tempfile.TemporaryDirectory(prefix='q1-2b-redteam-') as td:
        b=T.build(td)
        if mutate:mutate(b)
        T._write(b)
        if after:after(b)
        try:
            s,c=T._reduce(b)
            rows.append({'case':name,'exit':c,'aggregate':s.get('aggregate',{}).get('aggregate'),'cases_expected':s.get('cases_expected'),'findings':s['findings']})
        except Exception as e:rows.append({'case':name,'exception':type(e).__name__,'message':str(e)})
def edit(path,fn):
    j=json.loads(pathlib.Path(path).read_text());fn(j);pathlib.Path(path).write_text(json.dumps(j))
def fixtures(b):return [b['res'][p]['fixtures'][0] for p in ('A','B')]
def rawp(b,proc,cid):return pathlib.Path(T.RD._raw_path(b['run'],proc,cid))
def mutate_raw(b):
    for r in b['raw'].values():r.update(candidate_sha256='bogus',native_sha256='wrong',candidate_dtype='garbage',native_dtype='garbage')
def invalid_reps(b):
    for f in fixtures(b):
        for r in f['repeat_hashes']:r['scan_out_sha256']=[];r['publish']={};r['repeat']=False;r['ring_bytes_exact']=False

def within_drift(b):
    for f in fixtures(b):f['repeat_hashes'][1]['scan_out_sha256']=['changed-in-second-repeat']
def no_identity(b):
    for p in ('A','B'):b['res'][p]['attestation']={k:v for k,v in b['res'][p]['attestation'].items() if k in ('utc','process_tag','policy_sha256')}
def nonsense_summary(b):
    for p in ('A','B'):b['res'][p].update(integrity_ok=False,block='evaluation',repeats=-9,fixtures_done=-1,fixtures_expected=-1)
def no_inventory(b):
    for f in fixtures(b):f['cases']=[]
def noids(b):
    for r in b['raw'].values():r.pop('case_id');r.pop('stratum');r.pop('reference_valid')
def second_fixture_missing_negs(b):
    old=T.FID;new='calibration/fx1_ordinary-random'
    edit(b['fm'],lambda j:j['fixtures'].append({'fixture_id':new,'block':'calibration','sha256':'a'*64}))
    def obs(j):
        extra=copy.deepcopy(j['cases'])
        for c in extra:c['case_id']=c['case_id'].replace(old,new);c['fixture_id']=new
        j['cases']+=extra
    edit(b['obs'],obs)
    for p in ('A','B'):
        f=copy.deepcopy(b['res'][p]['fixtures'][0]);f['fixture_id']=new
        for c in f['cases']:c['case_id']=c['case_id'].replace(old,new)
        b['res'][p]['fixtures'].append(f)
    b['_second_fid']=new

def add_second_raw(b):
    for p in ('A','B'):
        for cid,r in b['raw'].items():
            r=copy.deepcopy(r);r['case_id']=cid.replace(T.FID,b['_second_fid']);path=rawp(b,p,r['case_id']);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r))
execute('baseline')
execute('all_procB_raw_omitted',after=lambda b:shutil.rmtree(pathlib.Path(b['run'])/'procB/raw'))
execute('procB_candidate_metrics_invalid',after=lambda b:edit(rawp(b,'B',next(iter(b['raw']))),lambda r:r.update(metrics_valid=False)))
execute('expected_cases_truncated_4_to_1',lambda b:edit(b['obs'],lambda j:j.update(cases=j['cases'][:1])))
execute('expected_case_duplicated',lambda b:edit(b['obs'],lambda j:j['cases'].append(copy.deepcopy(j['cases'][0]))))
execute('fixture_sha_changed_keep_claimed_canonical',lambda b:edit(b['fm'],lambda j:j['fixtures'][0].update(sha256='0'*64)))
execute('source_attestation_fields_omitted_both',no_identity)
execute('empty_scan_publish_and_bool_repeat_ids',invalid_reps)
execute('both_processes_same_within_process_drift',within_drift)
execute('empty_case_inventory_both',no_inventory)
execute('integrity_false_wrong_block_negative_repeats',nonsense_summary)
execute('candidate_native_hashes_wrong_and_dtypes_wrong',mutate_raw)
execute('case_stratum_reference_valid_fields_omitted',noids)
execute('procB_negative_tags_omitted',lambda b:b['res']['B']['fixtures'][0].update(negatives=[]))
execute('N5_raw_omitted',lambda b:b['negraw'].pop('N5_offpath_sentinels'))
execute('N5_false_leaked_count',lambda b:next(n for n in b['res']['A']['fixtures'][0]['negatives'] if n['tag']=='N5_offpath_sentinels').update(leaked_instances=False))
execute('second_fixture_all_negative_raw_omitted',second_fixture_missing_negs,add_second_raw)
execute('malformed_process_json',after=lambda b:(pathlib.Path(b['run'])/'procB/result.json').write_text('{'))
execute('malformed_raw_json',after=lambda b:rawp(b,'A',next(iter(b['raw']))).write_text('{'))
execute('boolean_metric_control',lambda b:next(iter(b['raw'].values()))['rms_C0_C2'].__setitem__(0,True))
execute('false_structural_flag_control',lambda b:b['res']['A']['fixtures'][0]['structural'].update(pointer_identity_unchanged=False))
execute('missing_procA_raw_control',lambda b:b['raw'].pop(next(iter(b['raw']))))
out={'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'tools/q1_2b_reduce.py',ROOT/'tools/tests/test_q1_2b_reduce.py',ROOT/'tools/q1_component_runner.py',ROOT/'tools/q1_policy_evaluator_v2.py',ROOT/'FREEZE-Q1_2B.json',ROOT/'fixtures/q1_2b/manifest.json',ROOT/'fixtures/q1_2b/expected_observations.json']},'supplied_test_functions_passed':1+len(params),'note':'pytest unavailable; decorators replaced by minimal local registration and all original test functions called directly. No GPU imports or production mutation. Each adversarial case uses its own temporary fixture.','repro_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'results':rows}
pathlib.Path('/tmp/q1_2b_independent_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
```
