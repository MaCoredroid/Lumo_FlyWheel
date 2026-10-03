# Repaired M1-Q C0 result — independent audit

**Per-policy verdict: Lumo PASS; Weaver author-default FAIL; Weaver aligned-local FAIL; TreeWY author-default FAIL under the unchanged v4.2 criteria.** All eight cycles now completed with clean collection, yielding all 571,392 expected repeated observations. The combined four-policy outcome remains `FAIL`. This accepts a scoped Lumo recurrent-component numerical result, not a speed, full-model, agentic-task or universal author-correctness claim.

Scope: repaired run `experiments/review-response-20260927/runs/m1-q-c0/m1-q-c0-20260928T222959Z`, reduction `reduction-repair-20260928T223230Z`, and the unchanged accepted held-out pack/seal. The original failed run and its independent review remain preserved. All reviewer work was read-only saved-evidence inspection or CPU arithmetic on existing tensors; no GPU, candidate execution, reference generation, threshold adjustment or gate operation occurred.

## Reproduced numerical result

Each repeat has 64,512 output cells and 6,912 state cells. Both repeats were evaluated separately.

| Named policy | Failed output cells in each repeat | Failed state cells in each repeat | Verdict |
|---|---:|---:|---|
| Lumo fixed32 with native replay | 0 | 0 | PASS |
| Weaver author preparation, TF32, `bf16_mode=none` | 53,745 | 1,154 | FAIL |
| Weaver with declared local FP32 preparation | 57,707 | 388 | FAIL |
| TreeWY author, `dot_bf16=True` | 37,528 | 6,912 | FAIL |

The independent CPU reducer recomputed and compared **4,608 layer metric/mask arrays bitwise** against the saved reduction: RMS, maximum error, finite, pass, fail and uncovered arrays across both surfaces and all eight cycles. Every evaluated tensor is finite; every reference cell is eligible; missing observations are zero. No per-head pooling, repeat averaging or repeat selection was used. Frozen per-cell RMS **and** maximum-error bounds remain exactly those in the pre-C0 reference seal.

All four implementations' repeat-0 and repeat-1 saved outputs and state captures are byte-identical. All **384** layer-level initial states (48 × 8 complete cycles) match the archived pristine S0. Exact domain is seed 20260929, layers 0–47, 28 active first-verification nodes, three publications over the uninterrupted 1/6/11-update history, and 48 value heads. This gives **285,696 unique cells and 571,392 repeated observations**, with zero omissions or duplicates.

The audit also authenticated **1,632** recorded first-step operand argument digests and **96** TreeWY inverse-remap raw-output digests. Publication order/leaf IDs and state-capture boundaries are unchanged and correct. The source-grounded mapping and comparator qualifications in `m1-q-c0-original-result-review.md` remain applicable.

## Original-result preservation and repair isolation

For every previously completed cycle, both its 48 first-output digests and four full-state-capture digests match the original failed run exactly:

- all six author cycles (both repeats for all three author policies);
- Lumo original repeat 0 and repaired repeat 0.

The repaired Lumo repeat 1 also matches repaired repeat 0. This closes the previously missing data without substituting new reference operands, changing numerical rules or discarding author failures. The original run remains `FAILED_PRESERVED`; it is not relabeled a successful run.

The original raw run has already been independently authenticated. The cross-run comparison authenticates its original launch manifest plus each referenced diagnostics/endpoint file, then matches those recorded raw digests to the newly hashed repaired raw tensors. No alternate author setting or method was executed by this audit.

## Backend lifetime and reset evidence

The repaired source uses one process-owned Lumo Backend with fresh per-cycle executor/capture bookkeeping. Observed evidence agrees with the accepted helper's assertions:

- Backend-boot log events are `[1, 0]` across the two Lumo cycles; cycle tokens are 0 and 1; `reused` is `[false, true]`.
- Both resets record `reset_complete`, unchanged persistent-pointer identities and in-place reset of `accepted_paths`, `accepted_lens` and `prev_lens`. The accepted helper checks boot-fixed storage, static SSI values and inter-cycle `prev_lens` identity before running the unchanged base reset.
- Global replay counters are cumulative **0 → 3 → 6**, with a delta of three in each cycle. The second cycle's full before-counter snapshot equals the first cycle's after snapshot. Capture count and preseeded-graph count both remain one, with zero recapture delta. Counters are not reset to hide repeated initialization.
- Each Lumo cycle executes three scans and three publications, with no failed call. State-zero equality is independently checked from the raw captures, rather than accepted solely from the boolean lifecycle flags.

Pointer equality itself is a source-bound runtime assertion and recorded result; the receipt does not archive the complete pointer tuple values for a separate address-level replay. The one boot event, cumulative counters, raw reset equality and unchanged scientific outputs are independent corroborating evidence at this stage. This run establishes two repeats within one process, not cross-process determinism.

## Frozen identity and clean termination

| Artifact | SHA-256 |
|---|---|
| Repaired approved gate snapshot | `92d1caae313b624bbd33b3b4194532232fbafbce9b7f228b9d174f2db5f4470c` |
| Repaired launch receipt | `32be4401846eb2f8b736e549adb3799f5ec9250a9babd78dc95e9ab8cd592241` |
| Repaired raw C0 receipt | `d90e62c27f4cb4b6c69e6bfe4f67376f873cdcc0151cad5e08d61f4f8e4039f1` |
| Repaired numerical receipt | `22b5a831277f91c5f1b15f7a86f9cfa5ba0f198a3f0d1dfadd71c2ddaa3edaec` |
| Lifetime helper | `c2a40888373a9de0a002dd49a8c86dc819c514f6948a6d50f362021f5ebe5d59` |
| v1.1 collector | `736ee7b49be82ebed4cbb37e10002a766461ff73cfda4b183446059bb50b8a08` |
| v1.1 launcher | `7c226d00905855c57aac633b36dc3a1022f63ec98d6d707decb962e40f5b5087` |
| v1.1 CPU reducer | `d41cf4249b382d2d7250768f0bece9c5fa00848f9a850b934eb9ad944c497191` |

All gate/authorization source hashes and the 33 method-receipt source identities match their actual saved source bytes. The v1.1 reducer is the reviewed dependency-filename-only derivative of original scorer `4e2aa7da…`; no numerical equation, classification or failure rule changed. It was explicitly bound by the repaired gate before C0 execution. Reference seal remains `a2521810…`, and pack remains `5f6157fd…`.

Exactly one owned container is recorded, CID `bb8ead8f366c837f1df81235c4f8299cc2ef409b5e0508bb4601c8735dfefbad`, with authenticated name/nonce/image, restart count zero, immutable image `ffa30d66…`, Torch 2.11.0+cu130 and Triton 3.6.0. Both container and attach exit codes are zero. Exact-CID removal and the subsequent successful full-ID absence check match. The one-use gate marker binds the same repaired authority. Collection has no stage/finalization failures and `collection_admissible=true`.

The owned launch lasted 73.403 seconds. CPU numerical reduction took 57.417 seconds; its outer process exited zero after 58.127 seconds at 22:34:15Z. These are operational durations and must not be reported as comparative kernel/decode speed.

## Supported paper narrowing

A defensible statement is: **On the frozen held-out recurrent-component population, Lumo satisfies the paired per-head RMS and maximum-error limits against the specified native reference at first verification and all three continuation publications, in both repeats.** The precise scope is the fixed topology, seed, dtype/configuration and GB10 implementation above.

The three author policies fail that predeclared sequential-comparator envelope in both repeats. Report these as configuration-specific numerical qualification failures if included; do not characterize them as mathematically incorrect algorithms or use this untimed run to claim a speed advantage. TreeWY's BF16 compact commit and the declared C1 FP32 outer-product convention remain different floating-point computations, as documented in the original audit. End-to-end continuation across convolution/attention/drafter state and workload quality still require separate evidence.

## Reproduction record

Reviewer directory: `p0/monitor/review-response-20260927/m1-q-c0-repaired-result-independent-review-20260928/`. CPU raw audit authenticated **427 unique source/evidence files totaling 12,252,915,678 bytes** in 16.15 seconds; metadata/lifecycle/source checks have their own 109-member hash index. These counts include reused reference/source bindings and are not new experimental observations.

| Reviewer artifact | SHA-256 |
|---|---|
| `audit_raw_cpu.py` | `f4905c6ee97ec7d30a2d728ca3d868815eb8b56a4eb8ad97632a0d3ada3a83ae` |
| `RAW-AUDIT.json` | `8e5704efeec781fc239c5becc3abfad750323fcc87ba77925d37272474d7a7fd` |
| `audit_metadata.py` | `6fcfb97001f758c0d88658c1a0fcd49926132327d74712f8a0de52580a48247f` |
| `METADATA-AUDIT.json` | `e89a1f0501a99928691b89143bdc38511c388130da07b2ccb29587ba51fcfe27` |
