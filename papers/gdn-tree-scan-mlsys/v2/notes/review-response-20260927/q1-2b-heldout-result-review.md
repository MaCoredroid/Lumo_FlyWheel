# Independent held-out Q1.2b raw-result review

2026-09-28 UTC. Run `q12b-heldout-20260928T013539Z`.

**Disposition: collection completed, result UNACCEPTED.** Both fresh GPU processes exited 0, but the original frozen CPU reducer exited **2** before raw numerical recomputation because the runner's helper map names v2.1 instead of v2.2. This is a concrete metadata-production defect in the reviewed package. Independently inspected execution identities match the approved v2.2 source, and the raw evidence is intact. A narrowly versioned, separately approved CPU-only reduction compatibility repair is justified; no GPU recollection is required by the evidence inspected here. This review neither repairs the run nor approves a new reduction or qualification.

Remote root: `mark@100.103.10.122:/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/runs/q1.2b-heldout/q12b-heldout-20260928T013539Z`. Source paths below are under paper v2's `experiments/review-response-20260927/` unless otherwise stated. No run output, source, gate, or tensor was changed. No GPU, Docker, Torch, candidate execution or numerical reducer was run by this reviewer.

## Terminal outcome and exact defect

Actual receipts show A `01:36:19–01:38:32Z` exit 0, B `01:38:32–01:40:47Z` exit 0, CPU reduction `01:40:47–01:40:48Z` exit 2. `RUN-RECEIPT.json` records `COMPLETED_reduce_rc=2` at `01:40:49.090453Z`. Thus this is no longer pending, and the run is not numerically accepted.

The original `summary.v2.json` has exactly these findings, no aggregate/case verdicts, and no completed tensor-recomputation census:

```
procA: loaded helper q1_component_runner_v2_2.py hash != launch-bound helper map
procB: loaded helper q1_component_runner_v2_2.py hash != launch-bound helper map
```

`tools/q1_component_runner_v2_2.py:72` leaves `HELPER_FILES[0] = "q1_component_runner_v2_1.py"`. Its real `helper_hashes()` at lines 750–757 emits that old key/hash, and `attest()` uses it at 783. The launcher/reducer correctly require `q1_component_runner_v2_2.py`. Both real attestations therefore contain old helper v2.1 SHA `b7241e8b…` and omit the v2.2 helper key.

This does **not** show execution of the wrong candidate code: both `attestation.runner_sha256` values are the correct v2.2 SHA `ab27fc9c…`; every ordinary metric's `execution.runner_sha256` also equals that gate/launch-bound SHA. Direct read-only hashing confirmed the actual runner, launcher, reducer, all launch helper sources, and native source copy match the gate. Every other actual helper-map member matches. The initial inherited helper entry names an extra old source, rather than replacing the separately recorded executed-runner identity.

The preparation review missed this cross-file rename: the author's static test explicitly required `HELPER_FILES` to remain unchanged (`tests/test_q1_2b_v2_2_heldout.py:84`), while synthetic fixture setup hand-constructed the expected v2.2 helper map at 216 and injected it into the attestation at 256. Those tests bypass actual `helper_hashes()`. My initial AST review likewise treated the unchanged helper function/constant as a preserved path. The present real-source AST reproducer executes `helper_hashes()` without importing Torch and reproduces the missing v2.2 key. This failed outcome supersedes the earlier source-only conclusion about end-to-end identity wiring; the earlier report and frozen run remain preserved.

## Independent raw support and numerical description

A and B each have the exact held-out product, without denominator inflation by repeats:

| Per process | BF16 verification output | FP32 published state | Total |
|---|---:|---:|---:|
| Ordinary cases | 3,072 | 2,688 | 5,760 |
| Head observations | 147,456 | 129,024 | 276,480 |
| Candidate/native identities | 3,018 equal, 54 different | all 2,688 equal | 5,706 equal, 54 different |
| Saved-array paired verdict | all PASS | all PASS | all PASS |

The two exact fixtures are `evaluation/fx0_ordinary-random` and `evaluation/fx1_mixed-stress`. Each has 48 instances × (32 output nodes + 28 publication paths); every case has 48 heads. There are two repeats per fresh process. Ordinary arrays are sealed for repeat 0, while repeat 1 checks determinism; neither repeats nor process replication supplies new independent fixture inputs.

Each process has exactly **5,760 reference + 5,760 eligibility + 5,760 metric records = 17,280 records**, with **zero negatives**. I read all 34,560 A/B record files, verified inventory/file hashes and canonical record seals, exact case ID sets, C2-reference IDs, eligibility flags, per-record process/executed-runner binding, publication accepted paths, and candidate/reference hash links. No missing, duplicate, extra or corrupt record was found.

Native and candidate repeat censuses are exact within each process and across A/B, including native states outside the scored publication endpoint set. All ordinary/reference candidate/native/C2 identities agree across processes. Both container IDs are distinct (`c8b27d825b28…`, `b7e6337ac476…`), each namespace PID 1. The helper-map exception is explicitly preserved in the independent JSON rather than silently normalized.

References and eligibility records precede each first candidate scan in both phase timestamps and inventory order. A ordinary reference seal is `01:37:17.957429Z`, before scan `01:37:17.964923Z`; A stress `01:38:11.689850Z` before `01:38:11.697367Z`. B ordinary `01:39:31.525240Z` before `01:39:31.532468Z`; B stress `01:40:26.113231Z` before `01:40:26.120488Z`.

All 28 accepted paths in each fixture/repeat have one actual publication replay, unchanged pointers, intact control rows, neutral staging tails and raw published-state hash links. The recorded public boot lifecycle restores its state and starts with zero measured replay counters. Actual A/B logs include the fixed32 two-level scan and fixed16 one-replay engagement messages. As in calibration, the null `last_executed_gdn` field is not itself positive execution evidence; the pinned source's selected branch does not populate it.

The **54 non-bitwise BF16 output cases** split **31 ordinary / 23 stress**, at depths 2–12. Independently evaluating every saved per-head metric array against the unchanged policy gives no failed or uncovered head. Among those 54 cases, maximum fractions of the frozen bounds are RMS `0.9090686576928357` and max-absolute `0.9090782118052655`. All IDs, raw-record hashes and case bound fractions are retained in the supplemental JSON. These are descriptive saved-array checks, **not a substitute for the not-yet-completed tensor-to-C2 numerical recomputation**.

## Tensor and calibration-power accounting

The shared store contains **6,198 unique objects / 9,702,088,704 bytes**. I independently rehashed every **5,814 record-referenced tensor object / 8,494,129,152 bytes**, verified its byte length and record link, and found no mismatch. This was one read-only SHA pass, not numerical recomputation.

The remaining **384 objects / 1,207,959,552 bytes** are exactly the native FP32 states for nodes **18, 23, 25, 27** across 2 fixtures × 48 instances. They are in the repeated native census but not in the scored publication endpoint set. Their identities and sizes were reconciled, not rehashed here. Do not claim that all 6,198 stored objects were scored or independently rehashed.

The actual gate SHA is `4ef2a52886edfdccea5812bbdc6dd3d05a4237ec41a6d9c8f5d58627519e883e`; the launch binding SHA is `cb67d9e996b81078160d8828e3b052c28ec5387f039b1de3ebe61b8e7385d3d8`. The gate names evaluation only, repeats 2, both processes, zero negatives, the exact two held-out fixture IDs and accepted calibration run `q12b-calibration-retry-20260928T010045Z`.

I independently verified the remote gate-bound calibration-power manifest `a9c1cdc3…` and three copied receipt hashes: summary `68408fc1…`, terminal `4deaafb7…`, parent acceptance `9d56a48c…`. Every `same_numerical_authorities` member in that manifest equals the held-out launch expectation, including policy `4a103df0…`, observations `c5df3aef…`, fixture manifest `2dc0a429…`, kernel `d9dd0c69…`, oracle/evaluators/native module/topology. Thus the parent-approved calibration-power reuse is bound; no mutation negatives ran on held-out inputs.

The recorded reducer command correctly uses the same pinned image, no `--gpus`, `CUDA_VISIBLE_DEVICES=''`, OMP/MKL/torch threads 1 and `--recompute all`. Runtime records Torch `2.11.0+cu130`, `cuda_available=false`. **Its early identity refusal prevented that requested all-tensor numerical recomputation from occurring.** Merely having `--recompute all` in the command must not be reported as completion.

## Minimal next step, subject to separate parent approval

Preserve original summary, logs, terminal receipt, helper maps and tensors byte-for-byte. A separately versioned CPU-only reducer/compatibility receipt may recognize **only this exact frozen v2.2 runner's known helper-map defect**, under an explicit parent authority record. It must require the original run/gate/source/result hashes, actual runner SHA `ab27fc9c…`, exact old v2.1 helper value, every other helper/source match, and per-metric v2.2 runner identities; it must refuse any other absent, changed, extra or wrong helper and any genuine executed-source mismatch. Record the exception explicitly, use new output files, and perform the unchanged full tensor/C2 recomputation. No threshold, evaluator, fixture, kernel or metric change is needed.

Regression coverage must exercise the **actual helper producer**, rather than inject a hand-built expected map; retain controls for wrong runner scalar/per-record runner, wrong old helper, missing/changed unrelated helper and original failure. The result remains unaccepted until that separately reviewed computation completes. Full-model, convolution/KV numerical continuation, sampling and serving performance remain outside Q1.2b regardless of its later component verdict.

## Durable audit artifacts

All independent files are under `notes/review-response-20260927/raw-audit-q12b-heldout/`. Commands executed remotely use only standard-library reads and emit JSON locally:

```sh
ssh mark@100.103.10.122 'python3 -B -' < notes/review-response-20260927/raw-audit-q12b-heldout/inspect_raw.py > notes/review-response-20260927/raw-audit-q12b-heldout/independent-raw-audit.json
ssh mark@100.103.10.122 'python3 -B -' < notes/review-response-20260927/raw-audit-q12b-heldout/account_extra_and_differences.py > notes/review-response-20260927/raw-audit-q12b-heldout/extra-tensors-and-output-differences.json
ssh mark@100.103.10.122 'python3 -B -' < notes/review-response-20260927/raw-audit-q12b-heldout/check_identity_receipts.py > notes/review-response-20260927/raw-audit-q12b-heldout/identity-receipts.json
python3 -B notes/review-response-20260927/raw-audit-q12b-heldout/helper_identity_repro.py > notes/review-response-20260927/raw-audit-q12b-heldout/helper-identity-repro.json
```

All four completed exit 0. Exact principal hashes:

| Artifact/source | SHA-256 |
|---|---|
| Original A result | `1097fd0b4a1423325c79954e8c88d98fcdca1d27357dd621dea32044b02cd71d` |
| Original B result | `4509588c3c7a9f98fcd39f262a66d4af07364e4703dcd40c40bff23d894427ca` |
| A inventory | `49e91f9b53e0d7a87e012ff19e42f146b545e4cdda8523763ac84a6551f20a94` |
| B inventory | `93254a23fd66dff12c39506d597589ab546c1493759d07b94c821545bce7c9d0` |
| Original failed summary | `646a07b7c57e540e512042d2dcf0f3bba04bba228564806d0a0fb9a693b46745` |
| Original terminal receipt | `7746dd7ba4eeb74c61e9e0c328a24653dfaf54f59667fcca67d27e726bf6e427` |
| Frozen runner v2.2 | `ab27fc9cf680bdd2b7c3ad994fc9f7bffe0f9ff1b0f35dc2cc36018f364dea29` |
| Frozen reducer v2.2 | `6475182d9e0996dd4e7c228340c57299665cd1ea641ed0b61bf291d631af9d73` |
| Frozen launcher v2.2 | `5cc0fdd7fc0895560de77e8046c73c7eac3f010bba8eee137b3573d3eaf050b5` |
| Independent main audit JSON | `49fe8fbe22863646e3bd186d71a05ada72f78bd823a55e5a588b2acbb4963d2b` |
| Independent extra-tensor/output JSON | `b8cfc11d22d4028f21279f3a9d36c149d04580f9178cdd1265a1bbf787423003` |
| Actual-helper reproducer JSON | `bdc64b5b215e3267f6db271d2d37bffe4521187696b4d53290ecbdd6fdaa0c45` |

`AUDIT-MANIFEST.json` in that directory records every independent script/JSON hash and size. It is separate from the original run and is not a qualification receipt.
