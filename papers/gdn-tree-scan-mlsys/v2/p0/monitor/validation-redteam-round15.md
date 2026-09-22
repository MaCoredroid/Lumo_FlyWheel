# Independent validation red-team, round 15

Scope: current remote files under `/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2`, completed p072 validation batch `experiments/out-20260922T063019Z-e7b-r13-p072-batch`, corrected E7a checker and retrospective outputs. CPU/read-only inspection only: no inference, container launch, tmux interaction, remote writes, or shared-source edits. The only output is this local report. Source findings refer to runtime SHA-256 `d6468150e17e6a5a55476821565f3c29e279ee6ef089fc3d8c21a13051319a8c`; later repairs require their own verification.

## Decision

**The corrected three-boot batch supports a narrow verifier-propagation observation. It does not qualify compact commitment or general continuation.** A new, concrete wrong-destination-row defect must be fixed before a commit-only/both model boot. Checker v2 closes the identified omitted-rule audit finding for the retained E7a data; original E7a remains failed.

## What the three boots establish

I independently loaded the raw tensor captures on CPU, comparing the first12 forward calls in both pairs, inspected all39 before/after-state files, checked the actual substitution records/specs, and hashed the evidence. This is additional to reading `e7b_reduce_v3.json`.

| Observation | None → matched B-verifier sham | Matched sham → B verify-only |
| --- | --- | --- |
| API response |32 token IDs, same sequence |32 token IDs, same sequence |
| Acceptance trace |Both `[1,3,1,0,3,3,1,2,0,0,4,5]` |Same |
| Comparable forward calls |12 calls, rows0–9 each |12 calls, rows0–9 each |
| Row IDs / full3-row mrope positions / input embeddings |Equal |Equal |
| Hidden/residual differences |None in captured layers0–63 |First differences at selected layer62, then layer63; calls3 and7 only |
| Final logits |Zero difference |Max absolute0.15625 at call3 and0.125 at call7; other compared calls zero |
| Argmax changes |0 |0 across120 compared rows |
| Layer62 replay-before / saved-after terminal states |Bitwise equal |Bitwise equal |
| Verifier records |Sham13 computed,0 applied |13 computed,13 applied |
| Compact-commit records |0 |0 |

For call3, the B output perturbation is followed by hidden/residual maximum differences0.0625/0.015625 at layer62 and0.75/0.25 at layer63. For call7, corresponding values are0.046875/0.125 and0.5/0.5. Layers0–61 and the model input embeddings match in both cases. The saved reduction reports reference-argmax conditional-log-probability differences of0.0161861 and0.00137675; unchanged greedy tokens do not imply unchanged conditional probabilities.

The largest logged B-verifier delta versus the production output is6.103515625e-5; the minimum logged bf16 agreement is0.9997884631. This fraction is a float32 reduction, not an exact integer mismatch fraction. These logs measure local replacement size, not an independently sampled noise distribution.

**Interpretation:** the matched sham removes the earlier demonstrated cross-arm input confound for this one observed prefix. Stage-local appearance at layer62 supports attribution of these particular downstream numerical changes to verifier substitution in this batch. With one sham and one substituted boot, do not generalize a noise floor, stability probability, harmlessness, or a distribution guarantee. No extra repetitions are needed merely to publish this bounded observation.

## P1 — Commit substitution targets a row the next forward does not read

At `experiments/e2/e7b_runtime.py:168–189`, the hook selects `final_col=max(accepted_len-1,0)`, captures that row, and writes the compact state there. It labels the capture `published_after` / “what the next forward reads.” That assertion is wrong for the selected stateless route when accepted_len>1:

- The runtime's own verifier hook chooses column0 when `FR13_TREE_RUNROW_INIT=1` (`e7b_runtime.py:104–111`).
- The actual scan is passed `h0_use_accepted_column=False` under that flag (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:17172–17176`).
- The replay kernel writes the terminal state to its accepted column AND explicitly republishes it to column0 (`src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:11915–11929`; documented at11770–11775).
- The served committer requires both running-row flags to be1; the alternative route is retired (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:21191–21218`).

Actual batch example: call1 accepts3 drafts, captures `final_row=23`, while the following verifier reads `h0_row=21,h0_col=0`. Six of the12 compared events have accepted_len>1; their captures are terminal-copy rows rather than explicitly observed authoritative running rows. Replay copies are expected equal before any compact replacement; the existing verifier-only data are not invalidated by this defect.

Independent in-memory CPU reproduction of the actual runtime, with accepted_len3 and spec indices[0,1,2,3]:

```json
{"captured_final_row":2,"authoritative_running_row":0,
 "bank_values":[1.0,1.0,1.0000100135803223,1.0],
 "captured_published_after":1.0000100135803223,
 "next_forward_would_read":1.0,"applied":true}
```

The reproduction used a fake compact kernel and intercepted save/log/mkdir calls in memory, so no model or filesystem state was changed.

**Required repair:** derive the authoritative destination from the same active lifecycle policy the served next-forward uses. Under this route, compact commitment must replace and capture column0 after replay. Preserve any extra mirrors only if the active route actually requires them. Capture both the terminal-copy row and running row under distinct names if both are useful; do not silently relabel old files. Add a CPU control with distinct column0/accepted-column addresses, accepted_len>1, plus zero acceptance, and assert that the **next-forward source row** contains the substitute. A sanity comparison of two terminal rows cannot catch this bug.

## Boundary accounting:12 forward comparisons, not13 qualified continuations

Each arm has13 logit captures,13 hidden captures and13 state captures; sham/verify each have13 verifier records. Only12 request acceptance events are traced. The reducer correctly excludes call12 from comparisons because its request/path binding is incomplete.

There is an additional end-of-request qualification: cumulative event accounting is `[1,3,7,9,10,14,18,20,23,24,25,30,36]`, while the API returned32 tokens. Thus call11's input boundary is known (30 tokens), but its reported five-draft acceptance plus bonus extends beyond the truncated response. It remains a valid matched-input forward comparison; it is not proof of a completed36-token user-visible continuation or a separately qualified32-token state horizon.

`e7b_reduce.py:134` compares sliced API token lists without requiring the expected end boundary to be present; two equally truncated slices compare equal. For state-continuation summaries, label that event terminal/truncated and distinguish it from complete emitted-path evidence. Preserve its internal state and selected-path measurements as such. Do not count call12 or this truncation as full fixed-input horizon coverage.

## Checker v2: bounded audit closure

Reviewed `check_confirmation_vs_freeze.v2.py`, `test_freeze_checker_v2.py`, `PILOT_FREEZE.erratum.md`, saved audits, original freeze and original confirmation verdict, plus the parent's round14 audit. Reexecuted v2 using an in-memory output stream, without changing files.

| Check | Independently reproduced result |
| --- | --- |
| Immutable freeze |SHA-256 remains `21f3eb5c3f9ce52aa2492d62c5be4dbc0f3d0df236aec717f70420c2ab2b36ce`; status FROZEN |
| Original confirmation `freeze_check.json` |SHA-256 remains `f2b8579cf2a84c1c51ce92e895ef1c5e92fccfa89b3671660f1f7d7dbea8780c`; original failed verdict retained |
| Pilot-v2 |D6c padding fails8/8; P1 half-width fails7 cells across6 prefixes; other exercised rules pass |
| Confirmation |12 failed rule IDs: T2,T3,D1,D2,D3,D6c,D7a,D7b,D8b,D8d,P1,P3 |
| Omitted padding criterion |Now evaluated for output AND factors;31/31 confirmation failures |
| Omitted compact-state ordering |Now evaluated; p031 fails |
| Omitted verify timing half-width |Now evaluated;14 failed cells across12 prefixes |
| Missing/NaN/changed-state/widened-timing controls |Six independent in-memory mutations detected as intended |
| Existing test artifact |17/17 controls marked passing, including exact comparison to the parent's independent round14 audit |

The checker has32 rules without capture-root provenance and33 when that check is supplied; this is why pilot/confirmation counts differ. These counts overlap logically (e.g. tolerance and decision rules), so12 failed IDs are not12 distinct physical defects.

The erratum correctly preserves the failed original analysis, calls out the false pilot padding/timing rationale, and separates superseded pilot timing from pilot-v2. It also reconstructs33 actual confirmation boots, not34; the freeze-MD gate refusal launched no container. No new GPU run is needed to close this audit issue.

Scope of closure: v2 is a retrospective evaluator of these retained, already provenance-audited records, not a deployment qualification or a replacement provenance inspector. Its static D9 statement confirms no tf32 tolerance class was promoted in the freeze; it is not evidence that a served deployment uses any particular arithmetic. Minor erratum cleanup: line29 prints `pilot_harness_summary.run_dir=None` because the record uses `path`; quote the actual `path` field rather than presenting this as missing evidence.

## Minimum next work

1. **Keep the three-boot result as one-prefix verifier-only evidence.** Write the bounded observations above; do not resume the obsolete28-boot sweep or promote B based on unchanged greedy tokens.
2. **Fix authoritative-row publication and capture first.** Run the two-row CPU control before any commit-only/both inference. Also label terminal truncation in the reducer; existing data suffice to repair that label.
3. **If compact-commit propagation remains necessary, run only a corrected matched pair initially:** B commit-only sham versus B commit-only on p072, with identical source/configuration and authoritative running-state capture, then inspect the next forward. Stop on a wrong/missing row binding or failed sanity/provenance. This is diagnostic work under a new explicit protocol; the original E7a remains failed. Add a combined arm only if the stage-isolated result justifies it. No C or large prefix sweep is necessary to resolve the current wrong-row defect.
4. **E1 stays gated by baseline route correctness.** A candidate-commit failure does not require indefinite candidate research before measuring a separately qualified sequential tree. B4/graph, conv/KV publication and actual fixed-input continuation remain separate coverage gaps if those claims are retained.
5. **No experiment is needed for frozen-checker closure.** Preserve the original artifacts, side erratum and corrected retrospective outputs.

## Evidence hashes

All paths below are relative to the remote v2 root. For each arm, `RAW_CALL_AND_STATE_HASH_INDEX` is SHA-256 of the sorted index containing `sha256 + two spaces + path-relative-to-arm + newline` for the26 `*.call*.pt` files and13 state files. The actual files were hashed and the compared tensor data independently read. This compact index does not claim GPU reexecution.

```text
85b9703feb7248513ef63ad469a3df2116c73eff7f42f8af45f460b0dac47954  experiments/e7a/check_confirmation_vs_freeze.v2.py
42619c200ba690f4d09706b8e252f3d00ab0e2ea2c43e61f56e6aa594e2dab24  experiments/e7a/test_freeze_checker_v2.py
051a6f9de10aa809a8096083ef34f8cbad5a081a846fcba3688222511b69bb29  experiments/e7a/PILOT_FREEZE.erratum.md
21f3eb5c3f9ce52aa2492d62c5be4dbc0f3d0df236aec717f70420c2ab2b36ce  experiments/e7a/PILOT_FREEZE.json
575cc9fa6aab45c0420d88c916b8db2fa87ddaee24362813b18acb10e4e0cf63  experiments/e7a/PILOT_FREEZE.md
15e651a60c93aac0d3ffb14b45536ea6aefbbef022d0f13967c402e5db57e20b  experiments/e7a/tests_out/freeze_checker_v2_controls.json
63556f9ba5a70c0268af981bc8401ecf7fcd10361a1428c15ceb93408ed0d11b  p0/monitor/2026-09-22-review-14-frozen-audit.json
a92ebe3d611877dc03593693bd82a555bc4bbb08d4105cb51aaab22bde8db434  experiments/out-20260922T064200Z-e7a-freeze-retrospective-audit/pilot_v2_retrospective.json
5baeb4c6261c5c9366592afb9aee2fcbd41e4e9bf3fb427b5fb1f547f5c8e9b8  experiments/out-20260922T064200Z-e7a-freeze-retrospective-audit/confirmation_retrospective.json
d6468150e17e6a5a55476821565f3c29e279ee6ef089fc3d8c21a13051319a8c  experiments/e2/e7b_runtime.py
ebfaa9ef86fdd6a8f8dd4cb551e4c24bfdd177e757ecc9f3b3e36054f2fbd1c1  experiments/e2/e7b_reduce.py
40063932a729d22eb12370c185b2f01b4979db013a5872e539c26e739eff3c5e  experiments/e2/e7b_substitute_shim.py
f2b8579cf2a84c1c51ce92e895ef1c5e92fccfa89b3671660f1f7d7dbea8780c  experiments/out-20260922T051611Z-e7a-fresh-confirmation/freeze_check.json
ffc7fff41b3e42878825bf623f1cc7568b3b2731c2d40b8505601e516742a537  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/e7b_spec.json
491c9909a28630ca37eb432440b79ebf4ded5d33b00f9bb34225b848be2e2539  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/capture_request.json
d2b9b81b866b5c48f1e31d0d48da3a518c1ae82862a35d12f38f53f178452cc8  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/capture_provenance.json
27f0b7655f53eece6a129c092dd13c5a07260b911da5aa6c7e307f54e45db315  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/capture_expected.json
879e67e3f69c2c94886ab309c54f14b7cf771e879aa2d4cec069def94bd934fa  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/logs/e7b_substitute.jsonl
b7acfd47e10e085aba4b6e2cd9543cf50963076588755a545386fd67cb1eb489  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/logs/per_req_spec_trace.jsonl
2bd12a30e817232c5bb3968cbcf6ae062fd1a5644c86ced0428c33ce26f8ea29  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/logs/fr10_mtp_draft_trace.jsonl
d6468150e17e6a5a55476821565f3c29e279ee6ef089fc3d8c21a13051319a8c  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/script_snapshot/e7b_runtime.py
40063932a729d22eb12370c185b2f01b4979db013a5872e539c26e739eff3c5e  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/script_snapshot/e7b_substitute_shim.py
15c3536252183281c6ff1dd5fc9fcc068e5d056bf6fa5ec8b199cc302ac21a70  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/script_snapshot/e7a_kernels.py
a82c9ec7abf5a0bc8bdf29f7e400422cbe7a2d188f7a3dcbc0cf5a5fea8a33a6  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all/RAW_CALL_AND_STATE_HASH_INDEX
e7f23ad2ea274f1e1c863b0d35e7bf2726f76d1df96ade6b9ffb9312ac9c8d90  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/e7b_spec.json
da3c46a06bba342fa11f5e320b5bf8665344a0f595155c41542ce41ee238ca16  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/capture_request.json
8ab5eaf13e412f27f8333b746e278035ec6a9772ddccd75da263dad42893c7ca  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/capture_provenance.json
27f0b7655f53eece6a129c092dd13c5a07260b911da5aa6c7e307f54e45db315  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/capture_expected.json
4bf792ab5b62dc89b9a9a96cec6bfe6ece92013d7cd16c1df83f786eabb20407  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/logs/e7b_substitute.jsonl
a553d2fee5af9742f7a67ae4d11d5d41a6f1631201769c6397f8588be7115b0f  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/logs/per_req_spec_trace.jsonl
8992bfa4dde894bc51bfe93cb11af2016090a2316eac88b179dc202e794269e0  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/logs/fr10_mtp_draft_trace.jsonl
d6468150e17e6a5a55476821565f3c29e279ee6ef089fc3d8c21a13051319a8c  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/script_snapshot/e7b_runtime.py
40063932a729d22eb12370c185b2f01b4979db013a5872e539c26e739eff3c5e  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/script_snapshot/e7b_substitute_shim.py
15c3536252183281c6ff1dd5fc9fcc068e5d056bf6fa5ec8b199cc302ac21a70  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/script_snapshot/e7a_kernels.py
fa9c9be44905adc2e5a79dec3eeea70491bba3640a0ac4b378feba2b760a88b4  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_002_p072_arm2_sham-B_fs_ieee-all-verify_only/RAW_CALL_AND_STATE_HASH_INDEX
d8e3743a5d7ef0d6c5f1d105285a5e84fd14529c796ce85a60501dcc4e863c89  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/e7b_spec.json
d13bbe3f8bf15ac629db7df9d22bcd79cb87309408842c1c82a66fe8e0af97fd  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/capture_request.json
6bddeea0da63b3e67893b76051b7baf6394e70a6ede5d4e83b5530758c3f2118  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/capture_provenance.json
27f0b7655f53eece6a129c092dd13c5a07260b911da5aa6c7e307f54e45db315  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/capture_expected.json
d26dfd9190b9f125c6e3846efe72c3b6e1b476aca35a94ab13933cd7b2486072  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/logs/e7b_substitute.jsonl
e9468fe3e86717fe2f6f30f3014e962ee115b8aae91cba64f72acd24be2b9303  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/logs/per_req_spec_trace.jsonl
95bcbe4b6085b0cce7a7b309fc1da43334896d872955618d9adbae34f6781375  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/logs/fr10_mtp_draft_trace.jsonl
d6468150e17e6a5a55476821565f3c29e279ee6ef089fc3d8c21a13051319a8c  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/script_snapshot/e7b_runtime.py
40063932a729d22eb12370c185b2f01b4979db013a5872e539c26e739eff3c5e  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/script_snapshot/e7b_substitute_shim.py
15c3536252183281c6ff1dd5fc9fcc068e5d056bf6fa5ec8b199cc302ac21a70  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/script_snapshot/e7a_kernels.py
a09e9f0c8aa53274b75ef5e6af9c3031059816f482aeda279af0898f803b4229  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_003_p072_arm3_verify_only-B_fs_ieee-all/RAW_CALL_AND_STATE_HASH_INDEX
582c96f7799e237273894f0f43ff71465b93c573f8ccbb71703510c75776ac7d  experiments/out-20260922T063019Z-e7b-r13-p072-batch/e7b_reduce_v3.json
d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8  ../../../src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281  ../../../scripts/fr10_phase4_patch_vllm_tree_gdn.py
```

## Follow-up: column-zero repair independently verified

The wrong-row finding above applies to the preserved three-boot runtime snapshot `d6468150…`. Claude subsequently repaired the current and staged runtimes. I inspected both revisions and reran the actual distinct-row controls against each, capturing their output in memory and using only temporary CPU fixtures. No model, container, GPU or stored result was modified. All 12 controls pass for each runtime. An initial attempt to direct the JSON to `/dev/stdout` completed every assertion but failed at output-file creation; the in-memory rerun exited 0 and supplied the final result.

Current `e2/e7b_runtime.py:168–208` and staged `e2/staging_r14/e7b_runtime.py:187–227` now select column zero as authoritative under the served runrow setting, save `replay_before` from it, publish the replacement into both column zero and the accepted-column copy, and save `published_after` from column zero. The sham computes its compact state into scratch and changes neither live row. Zero/one acceptance writes a single row. The recorded `rows_written` is the set of publication destinations, including in sham/none records; use `applied` to distinguish actual writes.

I also added an independent two-step CPU observation in the review process: with distinct bank sentinels, accepted length 3 and spec rows `[21,22,23,…]`, commit changes rows 21 and 23 to `21.000009536743164`. The following verifier's copied `h0` equals the saved authoritative `published_after` bitwise and differs bitwise from its original `21.0`. This passes for both runtimes and specifically closes the defect that the previous accepted-column-only implementation concealed. Fake kernels establish hook wiring, not numerical or served-route qualification.

The three-boot diagnostic may proceed with the staged capture runtime: **none + B commit-only matched sham + B commit-only**, one p072 prefix and the same layer/configuration/instrumentation in all arms. This replaces the earlier two-arm minimum above because the new operand capture deserves its own none/sham comparison. Capture the actual scan operands and both authoritative pre/postcommit states; begin with one complete, identity-bound transition after accepted length greater than one. Require the next forward's recorded `h0` to equal its own previous `published_after` bitwise. Compare none/sham first, then sham/commit at the same input and publication boundary. Reject attribution on failed provenance, differing inputs, inconsistent publication rows, sanity failure or an incomplete boundary. A successful horizon-1 witness closes this bounded publication/consumption question; longer horizons or more prefixes are needed only for a broader claim or an observed unresolved divergence. No combined arm, C arm or large sweep is required now. Sequential E2 qualification before E1 remains a separate route.

```text
806aa091b24dfa282c0541a7ccd6e77fd367a452fc74e02f6d580b9551171cc9  experiments/e2/e7b_runtime.py
9c3a6d94b2e1c6ccb07e4aedbbc29d812b9cbf4d102789b5f8a262b04b99b5ca  experiments/e2/staging_r14/e7b_runtime.py
c563c0980107e140659623bcccd6f4e8058d7092fc09541809fbb80d47da4d02  experiments/e2/test_e7b_commit_rows_cpu.py
a37249d044b8124407048edef16709c93f72cd2ea54640ab0a6a0eaf028f88cf  experiments/e2/tests_out/e7b_commit_rows_cpu_controls.json
a37249d044b8124407048edef16709c93f72cd2ea54640ab0a6a0eaf028f88cf  experiments/e2/tests_out/e7b_commit_rows_cpu_controls_staging_r14.json
```

## Follow-up: layer continuation fixture, before its pending repair

Reviewed `experiments/e2/e7b_state_continuation.py`, SHA-256 `d62da9e78af36165f4e1d4c926124e67aae1dfb95f9f139d8679925c3807d856`. Its stated scope at lines 2–17 is appropriate: an offline recurrence through one layer's fixed recorded operands, with no cross-layer propagation. It can measure layer sensitivity to an injected recurrent state. It cannot qualify full-model E2, downstream logits, convolution/KV publication or a served fixed-token continuation. The following are concrete prerequisites before interpreting *real-arm* continuation outputs from this revision; they can be repaired and tested on CPU.

1. **Reject malformed or unbound paths instead of changing them.** `path_nodes:37–38` clamps every node into the padded range; callers at 53 and 78 then silently drop nodes outside the real tree. A CPU probe with `path=[-3,8]`, accepted length 2, `n_pad=10`, `n=3` yields `[0,0,8]` and then replays `[0,0]`, while counting three committed tokens. Require a nonnegative bounded accepted length, enough path entries, real-node bounds and the actual parent-chain relation. Do not report a valid continuation for an altered path. Verify contiguous recorded steps rather than letting sorted sparse operands silently skip a transition.
2. **Make comparability fail closed and bind actual input identity.** At 60–68 only path indices/lengths are required; missing API token IDs skip the token check, and equally truncated API slices pass without covering the requested boundary. Independent CPU probes return `(True,"ok")` both with no API IDs and with only `[10]` in each arm when the required cumulative boundary is five tokens. Require complete prefix/boundary identity (or an equivalently complete token source), recorded draft-token ancestry, positions, tree topology, and layer/request/step binding through injection. Path indices alone can name different tokens in different trees. The full API response is not itself enough to bind rejected draft nodes; use the existing draft and forward-capture records. Terminal truncation must remain excluded from full postcommit continuation.
3. **Enforce reference fidelity before treating the result as actual state continuation.** `run_fidelity:42–58` emits measurements, but `main:94–104` continues regardless. The docstring's required invariants are not gates. Require complete relevant reference state/operand pairs, bitwise `h0(k)==published_after(k-1)`, and a declared replay/output numerical class; refuse or clearly return diagnostic-only on failure. An oracle-to-production bf16 mismatch is not automatically a semantic error—use the already defined numerical class rather than inventing bitwise equivalence. But a missing or unequal next-read state invalidates the claimed state flow.
4. **Disclose the precision boundary.** `payload_of:34–36` casts every incoming state to fp32 even with `--dtype float64`; continuation:75 invokes it at every step. A CPU probe maps fp64 states `[1.0,1.0000000009313226]` to identical `[1.0,1.0]` before `lift`. This may deliberately represent a stored fp32 state boundary, but the result must say “fp64 recurrence with fp32 state rounding between steps.” `state_delta_in` at 77 currently measures the value before that rounding; report the state actually consumed too, or a ratio can appear to attenuate a perturbation erased by quantization. Pure-fp64 recurrence would be a separate mode, not a silent reinterpretation of this run.

For the next diagnostic, collect the corrected three-arm captures once and use these CPU identity/fidelity gates before publishing continuation results. If horizon 1 has no state delta, report that observation for the chosen event; do not manufacture propagation or require a large campaign to force a nonzero effect. If a correctly bound nonzero state produces a downstream change, preserve the earliest witness and stop expanding until the narrower causal result is understood. These repairs do not alter the failed E7a freeze or promote any compact method to deployment qualification.
