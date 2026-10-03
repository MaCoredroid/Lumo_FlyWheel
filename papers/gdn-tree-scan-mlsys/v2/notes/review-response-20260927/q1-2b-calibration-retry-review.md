# Independent Q1.2b calibration retry raw-result review

Reviewed 2026-09-28 UTC. **Bounded PASS for the integrity and reported calibration-component result; no material result blocker found. This is not a qualification approval, a held-out result, or full-model qualification.** Parent owns gates. No GPU, Docker, model, or expensive reducer was launched by this reviewer. No frozen source, gate, policy, run record, or tensor was modified.

Run: `q12b-calibration-retry-20260928T010045Z` at `mark@100.103.10.122:/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/runs/q1.2b/`. All paths below are relative to paper v2 unless explicitly stated.

## Outcome and denominator

The completed runner/reducer receipts and independently reconstructed records agree:

| Per fresh process | Output | Published recurrent state | Total |
|---|---:|---:|---:|
| Ordinary cases | 3,072 | 2,688 | 5,760 |
| Head observations | 147,456 | 129,024 | 276,480 |
| Frozen paired verdict | all PASS | all PASS | all PASS |
| Candidate/native tensor identity | 3,004 equal; 68 different | all 2,688 equal | 5,692 equal; 68 different |

The denominator is exactly two calibration fixtures (`fx0_ordinary-random`, `fx1_mixed-stress`) × 48 recurrent instances × (32 output nodes + 28 publication paths), with 48 heads per case. Neither the two repeats nor the two fresh processes creates additional independent fixture coverage. Ordinary records are retained from repeat 0; repeat 1 checks determinism. A/B each contain exactly 5,760 reference, 5,760 eligibility, 5,760 metric and 292 negative records: **17,572 records per process**. No missing, duplicate, malformed, uncovered, or numerically failing ordinary case was found.

The 68 BF16 output differences split **32 ordinary-random / 36 mixed-stress**, at recorded depths 1–12. Every head of every differing case satisfies both frozen bounds. The maximum observed fraction of its frozen RMS bound among these cases is `0.9090678024409594`; the corresponding maximum fraction of a max-absolute bound is `0.9090780908477382`. These are descriptive checks on the recorded six metric arrays, not a new criterion or a claim of bitwise output equivalence. The supplemental JSON retains all 68 IDs, source-record hashes and their bounds, without selecting favorable cases. All A/B ordinary candidate/native/C2 identities match; native and candidate repeat censuses also match within each process.

The frozen rule used independently was, per head and for both RMS and max-absolute error, `error(C0,C2) <= kappa*error(C1,C2) + u32*magnitude(C2) + eta32`, with constants read from the hash-bound policy. The independent standard-library audit evaluates the saved arrays; the already-completed production reducer separately regenerated those arrays from retained tensors and regenerated C2 in the pinned CPU runtime. I did not rerun that expensive computation.

## Negative power, reference-first execution and actual publication

Each process has the exact negative product:

| Negative | Records/process | Required result | Observed |
|---|---:|---|---|
| N1 sibling substitution | 96 | paired FAIL | 96 FAIL |
| N3 ring swap | 4 | paired FAIL | 4 FAIL |
| N4 stale metadata | 96 | paired FAIL | 96 FAIL |
| N5 off-path sentinels | 96 | poisoned C0 equals the same clean C0 | 96 exact hash matches |

N5's paired PASS (96 per process) is only a diagnostic; causal acceptance uses clean/poisoned candidate equality for path `n14`, not equality to the native reference. N7 is a structural zero-replay control, recorded as rejected; it is not an additional tensor-record negative. The reducer enforces N1/N3/N4 power in both processes (`tools/q1_2b_reduce_v2_1.py:547–576`).

For both fixtures and processes, sealed references/eligibility precede the first C0 scan, independently checked in phase timestamps and record-inventory timestamps. For example, A ordinary seals at `01:02:35.935644Z` before scan `01:02:35.942888Z`; B ordinary seals at `01:04:56.822245Z` before scan `01:04:56.829779Z`. The runner independently computes/repeats all C1 native chains before sealing those records (`tools/q1_component_runner_v2_1.py:524–579`). Candidate/reference linkage agrees with the frozen observation IDs and C2 hashes.

The measured publication invokes the production `launch_tree_gdn_replay_all_layers` entry, uses root-exclusive draft indices plus their explicit accepted length, and reads the actual running row (`tools/q1_component_runner_v2_1.py:397–412,433–434`). All 28 paths × 2 repeats × 2 fixtures in each process have exactly one replay, unchanged pointer identities, intact control rows, neutral staging tails, and stored state hashes matching the corresponding ordinary metric records. This is captured recurrent publication across 48 instances, not one fused native update kernel for all layers.

The public boot sequence is attested: subtree preseed → three conv SSI registrations → committer preseed → conv pregather preseed → SSI source selfcheck → lease audit → warm. All restoration flags are true and measured replay counters start at zero. That establishes the repaired boot lifecycle for this component harness; **conv numerical cases are explicitly outside this run's scope**.

## Source, process and tensor integrity

A and B ran in distinct containers (`6a55e7d148a1…`, `3f2443598c1a…`), each with namespace PID 1; identical PID values are correctly not treated as a reused process. Attestation UTCs are respectively `01:01:36.343027Z` and `01:03:59.289539Z`. Helper identities equal the launch binding, and snapshot hashes equal the gate's reviewed identities. The launch is the approved calibration block with repeats 2, negatives enabled and process set A/B.

The pinned image is `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`; actual recorded runtime is Torch `2.11.0+cu130`, Triton `3.6.0`, vLLM `0.19.2rc1.dev134+gfe9c3d6c5`, GB10. Relevant source pins:

| Source/authority | SHA-256 |
|---|---|
| `tools/q1_component_runner_v2_1.py` | `b7241e8b925a9b5d9e921673d84f84b701f9b1404ed71d727bc5ade37d15cb65` |
| `tools/q1_2b_reduce_v2_1.py` | `1d58c83d4cf96d755d12822ed23f4557e5d037f6f9aeadbb8a135dfe5d59449c` |
| `tools/run_q1_2b_component_v2_1.sh` | `84a377973c3cf68504b964fae7082e8db4d4daba6335bb9878f02bc3cb2456e2` |
| `q1_policy_evaluator_v2_1.py` | `2d25e4c9bb669a4c66a5a3caa5093b614e5aba136eb424efb0e410f0b0b61061` |
| repository `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` | `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8` |
| policy snapshot | `4a103df013439ae02977d10a42c3cf2eeb916a7dcb1a61c19e6b929da620f7b3` |
| observation snapshot | `c5df3aefd4b838ef2098b03cfc54775b1cb467510c5956edb9e868c725ef7448` |
| launch binding | `e6f535271b821d9171bf7e23dffe5f7c7334930c907234bba02694658efef625` |
| approved gate snapshot | `1878e07b63e1a4a32b9abbb84560dd793d712321dfe3cee20c90eb1aabb83117` |

**Tensor-count reconciliation:** the store has 6,216 unique objects / 9,714,843,648 bytes. Exactly 5,832 unique objects / 8,506,884,096 bytes are referenced by the scored/reference/negative records. The other **384 objects / 1,207,959,552 bytes are precisely native FP32 states at nodes 18, 23, 25 and 27 across 2 fixtures × 48 instances**. They appear in the native repeat census but are not publication endpoints in the scored observation set. The runner saves native states for all 32 nodes (`runner:538–554`); the reducer hashes tensors reached through record references (`reducer:134–171,377`), so its `tensors_verified=5832` is correct. This is not missing scored evidence. Do not describe all 6,216 archived objects as reducer-verified. This reviewer checked extra-object names, sizes and their exact native-census mapping, not their bytes.

The reducer's `recomputed_records=12104` is exactly `2 × (5760 ordinary + 292 negatives)`. Source `reducer:591–628` loads candidate/native bytes, verifies and regenerates fixture C2, checks the C2 hash, and requires exact equality of all six recomputed metric arrays. The recorded reducer command is pinned-image **CPU-only**, no `--gpus`, `CUDA_VISIBLE_DEVICES=''`, OMP/MKL/torch threads 1, `--recompute all` (`launcher:153–158`). Runtime receipt confirms `cuda_available=false`. Its 62-second completed execution is not merely a metadata-only PASS.

The parent separately reported a full read-only SHA/size check of all 35,184 receipt-listed non-tensor members with zero mismatches. This is parent verification, not an additional independent tensor rehash by this reviewer. Independently, I read and verified all 35,144 inventory-listed record files, canonical record seals, source/gate bindings, referenced tensor existence/lengths, count products and result identities.

## Route evidence caveat and result boundary

The route stamp has `schedule=fixed32`, `critical=12`, but `last_executed_gdn=null`. This field is initialized null at pinned kernel line 6347 and is filled by the single-launch branch at 17218, not by the selected ordinary fixed32 two-level branch at 17263–17352. Both A/B logs have the actual `[FR13_SUBTREE_PARALLEL ENGAGED] n_actual=32 schedule=fixed32 critical=12` and `[FR13_FIXED32_COMMIT_DEVICE_FILL ENGAGED] ... B=1 fixed16 one-replay` messages. Frozen flags have single-launch 0, group3 production 0, scan-align 0, BV 8 and ring export 1. Source, flags, logs, saved outputs and publication counters jointly support the executed route; the null field alone must not be advertised as a positive execution stamp. No new capture is required to interpret this completed component run.

A completed `01:01:33–01:03:56Z`, B `01:03:56–01:06:16Z`, and CPU reduction `01:06:16–01:07:18Z`, all exit 0. The terminal receipt says `COMPLETED_reduce_rc=0`. This supports the stated two-fixture **calibration component** outcome: verification outputs satisfy the frozen paired numerical rule, and recurrent publication is bitwise native for the enumerated paths with powered negatives. It does not qualify held-out fixtures, full-model logits/sampling, complete conv/KV continuation, end-to-end serving or timing. The earlier pre-fixture boot failure remains a separate preserved failed attempt. No thresholds or qualifications are changed by this review.

## Durable independent evidence and exact commands

Scripts are standard-library only and read remote files without writing them or importing production code. From paper v2:

```sh
ssh mark@100.103.10.122 'python3 -B -' < notes/review-response-20260927/raw-audit-q12b-retry/inspect_raw.py > notes/review-response-20260927/raw-audit-q12b-retry/independent-raw-audit.json
ssh mark@100.103.10.122 'python3 -B -' < notes/review-response-20260927/raw-audit-q12b-retry/account_extra_and_differences.py > notes/review-response-20260927/raw-audit-q12b-retry/extra-tensors-and-output-differences.json
```

Both commands completed exit 0. Important receipt/data hashes:

| Artifact | SHA-256 |
|---|---|
| A result | `a745b55a9d3c590d2d0d2181d73e548244634737290fef57e230645985de3448` |
| B result | `b4ebc28e91e9a1f0b018f574792963a1bdccb19124ee43792da54ac3b57c3c03` |
| A inventory | `b67a86822603156c6db6d79d6f1a957782fe55d158a1b637b3c9af2bf65f8c10` |
| B inventory | `5dd5231822db5bc9281715baa6ed8131db367c681ef434386181e08a8709aab3` |
| terminal summary | `68408fc1f530ca7f8de37eb4689f5becc541adfea0b2a21905838ce60e1141f6` |
| terminal run receipt | `4deaafb73996aca5260ce0b2d1aeccd45edf6dc7daa42c3799eab47d68262583` |
| `raw-audit-q12b-retry/inspect_raw.py` | `a303bc79f27cf397da56d24bb0ed127fa7d8aaa7527f2898df018bd0f7cac8c3` |
| `raw-audit-q12b-retry/independent-raw-audit.json` | `67d9afd81d6f72fb5508f2326ea276fb6e7ebb718523d06a18c877405aad825b` |
| `raw-audit-q12b-retry/account_extra_and_differences.py` | `f61d498dc2a296e40448a0b246481e39e564525e002717b66ce3cb4642973654` |
| `raw-audit-q12b-retry/extra-tensors-and-output-differences.json` | `037917fed988550f0bbe8a81996e9f1d8e6c1c6930cac3a927069e461fe9fb7f` |
