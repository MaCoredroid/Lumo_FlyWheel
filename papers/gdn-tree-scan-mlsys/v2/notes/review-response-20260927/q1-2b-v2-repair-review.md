# Q1.2b v2 pipeline — bounded independent repair review

27 September 2026. **Verdict: remaining material repairs required before launch.** The major v1 omissions are substantially repaired, but the reviewed v2 has a real two-container PID rejection, skipped helper binding, incomplete negative-witness verification and a reducer-runtime mismatch. This report reviews the original v2 snapshot only; the worker began further repairs during review. No gate is approved here.

## Exact reviewed snapshot

The parent preserved these bytes under `p0/monitor/review-response-20260927/reviewed-v2-snapshot-20260927T235500Z/SNAPSHOT.json`, relative to paper v2. Source references below use `experiments/review-response-20260927/tools/`.

| File | SHA-256 |
|---|---|
| `q1_2b_reduce_v2.py` | `68aac40e04362095435c23c8a658e51c1b732dfc8b0a393437747c9211966b0f` |
| `q1_component_runner_v2.py` | `3843b7dcc74e39595e9164deecce78cc19aa0e44847572db91a72eaec1390c15` |
| `run_q1_2b_component_v2.sh` | `5892f42647e993298821dc5b1e8ba0422da50b54d36d7246317d25cad3307a1d` |
| `tests/test_q1_2b_v2_pipeline.py` | `1233569d8753182974e575e148fa23e38cd392580ac5e6300e8d2a39e9876d75` |

The original v1 review `q1-2b-reducer-review.md` and its counterexamples remain unchanged.

## Verification performed and limits

- Reviewed the full reducer, reference/candidate/negative runner loops, receipts and launcher; paired-rule details remain the separate policy reviewer's remit.
- Ran the supplied CPU pipeline and launcher-static tests on the existing DGX host Python, with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, `python3 -B -m pytest -q -p no:cacheprovider tests/test_q1_2b_v2_pipeline.py tests/test_q1_2b_v2_launcher_static.py`: **62 passed in 158.66 seconds**. Only synthetic CPU stub work and temporary test files were used; no GPU/container/model/workload or production edit. The requested source hashes were printed and matched before that command started. Remote source subsequently changed while review was underway, so this is not a substitute for the final immutable-freeze rerun.
- A separate independent full-pipeline probe stopped at its initial source-hash assertion because remote bytes had changed. It did **not** generate a stage or run its mutations. No independent full-pipeline invalid-PASS count is claimed for v2.
- Four bounded local standard-library checks against the still-pinned original local reducer confirm the PID predicate, skipped helper loop, acceptance of a sealed negative record without tensor references, and failure to flag an extra negative ID. Local Torch is unavailable. The exact checks appear below.

## What R1–R4 now repair

R1: the reducer hashes snapshots, recomputes the fixture canonical payload, uses policy-bound expected cases, enforces a per-fixture `L*32 + L*28` product and checks process completion/block/counters. Truncated/duplicate observations, omitted identities and wrong summaries are covered by supplied failure tests.

R2: both processes now need exact ordinary `ref`, `eligibility` and `metrics` inventories. File hashes, sizes and canonical record hashes are checked. Ordinary metrics require case/reference/dtype/execution identities, and `--recompute all` derives ordinary arrays from retained C0/C1 tensors and regenerated C2. Missing B raw data or altered ordinary metrics no longer rely on A's candidate identity.

R3: `q1_component_runner_v2.py:312–378` computes C2, every C1 chain and both native repeats, retains their content-addressed tensors, seals reference-only eligibility and then starts each fixture's candidate phase. The loop does not read C0 to construct reference eligibility. Native census and candidate repeat maps are retained and the reducer derives within/across-process determinism and connects ordinary metric hashes to repeat-zero node/path hashes. These are real source changes, not just revised labels.

R4/N5: `q1_component_runner_v2.py:478–490` now restores the same seed-17 O0 as the clean repeat, poisons off-path rows, and compares poisoned C0 with the corresponding clean C0 hash. `q1_2b_reduce_v2.py:371–380` binds clean hashes back to ordinary n14 publication. The ordinary n14 paired numerical gate remains independent. This fixes the previous mistaken poisoned-C0-versus-native-bitwise causal criterion. Supplied tests exercise a real stub leak and unbound clean hashes. Negative tag presence, per-fixture expected counts and mandatory records are now checked in both processes, subject to the residual gaps below.

## Remaining material issues: one repair pass

### V2-1. Namespace-local PID comparison rejects the actual launch architecture

Reducer line **216** rejects if `A.pid == B.pid`. Runner line **517** records `os.getpid()`. Launcher lines **29–30** use separate ordinary Docker namespaces with `--entrypoint python3`; each main runner is PID **1**. Fresh valid containers therefore fail reduction even if both GPU processes succeed. The stub assigns 1001/1002 and misses this.

**Required repair:** bind each process to the actual fresh container identity/hostname and A/B tag plus launch/start evidence. Compare container+PID identities, not PID alone across namespaces. Test two valid distinct containers both PID 1 (must admit), repeated same container/boot despite differing caller tags (must refuse), and mismatched A/B process tags. This does not require a model or GPU.

### V2-2. Helper/source identity checks do not consume the actual launch binding

Launcher lines **51–64,80** write the helper map as `LAUNCH-BINDING.helpers`. Reducer lines **131,208** instead read `exp.get('helpers')`, where `exp = lb['expect']`; the comparison loop executes **zero iterations** on actual launcher output. Merely requiring a nonempty attestation helper map does not detect loaded helper drift.

Also, reducer lines **137–143** take only policy/expected-observations/run authorization from the gate. They do not reconcile the complete `lb.expect` source/image map or `lb.scope` with `gate.reviewed_hashes`/`reviewed_scope`; lines **201–206** skip source checks when the corresponding expected key is absent. This is a source-consistency problem even with an unchanged approved gate: a partial or stale launch binding should not disable or redefine its identities. It does not require an attacker to forge a parent approval. Required native-function/runtime fields are mostly presence checks; the previously used A/B comparison of Torch/Triton/GPU identity has disappeared.

**Required repair:** consume and require the correctly located complete helper map, reconcile it against the gate-pinned helper/source entries and actual process helper maps, and require every prescribed source/image expectation. Reconcile scope and the approved run with the gate. Require A/B runtime identity consistency and bind expected runtime values where declared. Add a wrong-helper-value test preserving all approved gate bytes, a deleted required LB field, a stale LB source value with unchanged gate, and scope disagreement. No new external signing system or broad adversarial threat model is requested.

### V2-3. Negative raw tensors and metric arrays are not fully verified even in `--recompute all`

`verify_record`, lines **99–102**, silently continues when `candidate_tensor` or `native_tensor` is absent. Ordinary records later hit mandatory fields/recomputation, but the negative loop does not. A locally sealed negative record without either tensor reference returns successfully with no malformed finding (direct CPU check below).

Numerical recomputation at lines **390–405** iterates only ordinary cases. Negative power at **413–428** consumes declared negative arrays directly. Thus `--recompute all` does not establish that N1/N3/N4's recorded bound violations actually follow from their retained candidate tensors; a stale or wrongly computed negative error array can still authorize negative power, even though ordinary metric corruption is detected. Native/reference fields are bound by the evaluator, but that does not derive negative C0 error arrays from its bytes.

**Required repair:** require and validate C0/C1 tensor references for every numerical negative; validate dtype/shape/byte length and record/inventory case identity. Recompute its six arrays against the correct n14 C2 and candidate-blind C1 record on the approved CPU path, just as for ordinary cases. Missing tensors, corrupt bytes, resealed incorrect negative arrays, and an otherwise correctly retained negative that fails to exceed the paired bounds must refuse. N5 keeps its exact poisoned-versus-clean invariant; its paired metric remains separately labelled diagnostic, while ordinary n14 retains the numerical requirement.

### V2-4. Negative product and some structural corroboration still accept inconsistencies

The `extra_negs` condition at line **384** requires both `not cid.startswith('NEG|'+fid+'|')` and `cid.split('|')[1] == fid`. Every normally formatted negative for that fixture already starts with the prefix, so an extra `NEG|fid|L999|N1_sibling_substitution` is not flagged. The loop at **417–421** obtains the target instance from `record.execution.instance`, not the inventory ID; it also does not require `record.case_id == inventory.case_id` before remapping to n14. Required negatives may all exist while extra/misindexed negatives remain admissible.

The ordinary raw inventories are now authoritative for the numerical denominator, which is an improvement over v1. Nevertheless, the reducer still accepts some summary corroboration without deriving it: `reference_phase_before_candidate` is only checked as a Boolean, not reconciled with retained phase times; `replays_enqueued == 1` at line **307** accepts Boolean `true`. Empty secondary case indexes are not themselves evidence loss when the new complete raw inventory remains present, so they should be reconciled or explicitly classified as nonauthoritative—not treated as a new missing-observation discovery.

**Required repair:** build the exact negative-ID set per fixture/process (N1 L, N3 instances 0/1, N4 L, N5 L), compare inventory and record sets exactly, and bind each negative's instance/target/path/reference to its frozen key before evaluation. Reject duplicates/extras/misindexed records. Derive phase-order corroboration and require integer replay counts; reconcile or remove unused redundant indexes. Tests should add an extra negative alias pointing to an existing good file, an out-of-range inventory instance with in-range execution instance, reversed reference/candidate phase timestamps, and Boolean replay count.

### V2-5. The final reducer runs in an unqualified host numeric environment

Launcher line **147** invokes host `python3 ... --recompute all`. `_recompute_metrics` uses host Torch to regenerate C2 and compare its tensor SHA and all six metric arrays exactly. Fixture C2 generation and runner references are bound to the pinned image's Torch 2.11 plus OMP/MKL=1. The parent separately observed that the host currently uses Torch 2.4.1 with default 20 threads. The earlier Q1.2a campaign already demonstrated that the host/image reference path cannot be assumed byte-identical. Our supplied CPU stubs generate and reduce within one host runtime, so their passing tests do not establish actual-image parity.

**Required repair:** run final reduction in the same pinned image **CPU-only, without `--gpus`**, with OMP/MKL=1 and pinned source/fixture/tensor mounts; record its actual runtime identities. Before the GPU job, use this approved CPU path to regenerate an actual frozen calibration fixture's C2 and verify its existing hashes. This is a runtime-parity preparation check, not a new numerical-policy experiment, threshold tuning or additional GPU axis. Preserve previous failures and versioned logs.

## Minimal local source checks

Run from the reviewed `tools` directory with standard-library Python. These checks establish the specified source gaps; they do not claim an end-to-end numerical PASS.

```python
import tempfile, pathlib, json
import q1_2b_reduce_v2 as R
assert R.sha256_file('q1_2b_reduce_v2.py') == '68aac40e04362095435c23c8a658e51c1b732dfc8b0a393437747c9211966b0f'
with tempfile.TemporaryDirectory() as t:
    p = pathlib.Path(t) / 'procA/raw/n.json'; p.parent.mkdir(parents=True)
    rec = {'case_id':'NEG|calibration/fx0_ordinary-random|L0|N1_sibling_substitution',
           'negative':'N1_sibling_substitution', 'candidate_sha256':'a'*64,
           'native_sha256':'b'*64}
    rec['record_sha256'] = R.canonical_sha(rec); p.write_text(json.dumps(rec))
    entry = {'path':'raw/n.json','file_sha256':R.sha256_file(str(p)),
             'bytes':p.stat().st_size,'record_sha256':rec['record_sha256']}
    F = {'malformed':[]}
    assert R.verify_record(t,'A',entry,F,t,set()) is not None
    assert F == {'malformed':[]}  # both tensor references missing
fid = 'calibration/fx0_ordinary-random'
extra = 'NEG|' + fid + '|L999|N1_sibling_substitution'
assert [cid for cid in [extra] if not cid.startswith(f'NEG|{fid}|') and cid.split('|')[1] == fid] == []
lb = {'expect':{}, 'helpers':{'q1_policy_evaluator_v2_1.py':'a'*64}}
assert list((lb['expect'].get('helpers') or {}).items()) == []
a = {'utc':'2026-09-27T23:00:00Z','process_tag':'A','pid':1,'hostname_in_container':'container-A'}
b = {'utc':'2026-09-27T23:01:00Z','process_tag':'B','pid':1,'hostname_in_container':'container-B'}
assert a.get('utc') == b.get('utc') or a.get('process_tag') == b.get('process_tag') or a.get('pid') == b.get('pid')
```

The next review should use one newly settled source/freeze snapshot and targeted controls for these five groups. No claim is made about the moving repaired version until its hashes and results are reviewed.
