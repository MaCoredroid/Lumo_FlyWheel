# Workload launch source-binding successor: bounded review

2026-09-29. **PASS for launch-packet builder v2 and the exact immutable LAUNCH-DRAFT manifest successor.** This is source/evidence admission preparation only. No experiment, model-file read, GPU, Docker, SSH, network, evaluator or agent operation occurred, and no implementation/gate was edited.

## Exact findings

1. `workload-plan/tools/attempt-runtime/assemble_common_controls_v1.py:10` still selects `sole_executor_v3_10.py` and `prepare_launch_packet_v1.py`. Its lines 30–36 overwrite common controls and append new digest→same-path aliases to the original evidence map. It should remain a historical assembler, not be rerun against the active drafts. Original `COMMON-CONTROLS.DRAFT.json` also binds v3.10 at its source-binding entry. An old correctly hashed historical source is not itself invalid evidence; the issue is using it as the selected current caller.
2. `prepare_launch_packet_v1.py:92` imports v3.10. Its `preflight` body is AST-identical to v3.11, but the called `prior_evidence_map` differs: v3.11 calls `NETBOUND.verify_prior` at line 399. Thus retaining v1 for current preparation would use the old prior-attempt evidence path. The actual v3.11 executor still checks independently; this is not a demonstrated live-admission bypass.
3. Original `EVIDENCE.DRAFT.json` has 181 entries and exactly one hash mismatch: `4f816890c482e1120ea5f083a84799ef25b7ba625bf75b9d290535309e606459` points at mutable `COMMON-CONTROLS.DRAFT.json`, whose actual SHA is `af37030b6b8b7ee299cf310519840a1d354718dcf87d760f004f84d7252ad7e8`. `prepare_launch_packet_v1.py:19–20` verifies **all** supplied remote-map entries before observations, so an unused invalid alias still refuses. Line 21's later merge does not rescue it.
4. **That map defect is already repaired in existing preserved successors.** `EVIDENCE.REPAIRED-DRAFT.json` has 180/180 matching local entries, archives the valid common-control bytes at a digest-named history path, retains the original map unchanged, and documents the omitted unreachable invalid alias. `assemble_network_controls_v1.py:12–23,26–35` starts from this repaired map and creates exclusive new files. `EVIDENCE.NETWORK-DRAFT.json` has 184/184 matching entries and matching local/remote keysets. Its common controls already replace v3.10 with reviewed v3.11 and add the network helper/review. It correctly retains old, valid v3.10/v1 files as historical evidence. Do not classify those residual map entries as new blockers.
5. The remaining selected-builder binding in `COMMON-CONTROLS.NETWORK-DRAFT.json:94–95` is still v1. Parent's new `prepare_launch_packet_v2.py` is **byte-identical to v1 except** `import sole_executor_v3_11 as S` at line 92. No collection/clock/source/evaluator/model rule changes. Its exact reviewed SHA is `72cd6755b4168b9a2963da0ef5507c702efcbecdba4526862447e5b56a00574c`.

## Recommended successor, subsequently implemented below

Start from the reviewed NETWORK draft triplet, not the original mutable assembler. Verify every input byte and local/remote keyset first. Create fresh, exclusive LAUNCH filenames. In the successor common-control source bindings replace precisely the selected `prepare_launch_packet_v1.py` path/hash with reviewed v2; leave v3.11/network-helper bindings and all scientific/task/schedule/settings policies unchanged. Add explicit parent lineage identifying the input triplet and new builder. Repoint only the six common-control receipt fields in the copied freeze to the new common-control SHA. Add v2 and the new immutable common-control bytes to both evidence maps with matching keys and repository-relative remote paths. Preserve all 184 valid NETWORK entries and all historical files; valid v1/v3.10 evidence need not be removed.

Do not edit a historical map's hash key to match changed bytes, ignore bad entries, or weaken the all-entry verifier. A future genuinely obsolete invalid alias may be excluded only in an explicitly versioned successor after proving it is unreachable from the selected freeze/source graph and retaining the original mapping and reason, as the existing repair already does. Status remains DRAFT; qualification, four-arm settings, request seed mode, schedule, clocks, and WP approval are untouched by this small source repair.

## CPU controls and acceptance conditions

**17 bounded controls passed:** twelve source/map/actual-collect controls plus five actual-v2-main injected controls. These are preparation controls, not experiments.

- Hashed all listed local source/template/config evidence: original 181 entries has exactly the stated mismatch, repaired 180 and network 184 pass. No payload/model files were read; inspected source members are under this checkout and at most 2,118,510 bytes. The remote paths were not queried; their local mirrors were used for CPU replay.
- AST-extracted the actual `collect` function and injected a preflight stub plus a first-clock stop sentinel. Original map refuses at the stale alias; repaired and network maps reach the sentinel only after all entries verify. An additional unreferenced bad alias also refuses. No clock/model/evaluator action is called.
- AST-extracted actual v2 `main`: failed parent preflight stops before transport import, Docker/requests import, component creation or collection. An approved stub flow imports only the v3.11 stub, passes that same object to collect, then closes both clients. Injected collection failure also closes both clients. These controls do not claim real v3.11 runtime dependencies were imported or live approval exercised.
- Checked exact one-import byte delta and unchanged reviewed sources. Initial reviewer-only runs stopped at an overly narrow source suffix allowlist (`.js`, then `.jinja`); those attempts are retained. Adding the stat-confirmed source/template/config extensions yielded the final controls; no product-source failure was hidden.

The manifest acceptance conditions were: (a) all 184 prior entries remain byte-valid and unchanged; (b) exact builder-v2 source binding plus current helper/caller pins; (c) no selected source binding to v1/v3.10; historical entries may remain; (d) exactly the six common-control receipt hashes change in the freeze, with all pending qualification/time/approval fields unchanged; (e) local/remote keyset and path parity; (f) corruption/missing-member refusal, including an unused bad alias; (g) exclusive outputs refuse overwrite; and (h) predecessor bytes still match their preserved hashes. There is no need to rerun unchanged GPU/component work.

## Exact identities and records

- `experiments/review-response-20260927/workload-plan/tools/attempt-runtime/assemble_common_controls_v1.py`: `a879f49d8645060e8545a228e3501acbed36db1fb06f92c326306badef18bd30`.
- `experiments/review-response-20260927/workload-plan/tools/attempt-runtime/prepare_launch_packet_v1.py`: `0ce035762173064f5d3f81d23dcf3c201a0e41f70753379c0a9cf90b0428a793`.
- `experiments/review-response-20260927/workload-plan/tools/attempt-runtime/assemble_network_controls_v1.py`: `80bc08b77fb77efe4bbec1357bd6eefc8d0ba669bb3418f30180d6a1fa644332`.
- `experiments/review-response-20260927/workload-plan/tools/attempt-runtime/sole_executor_v3_11.py`: `76334a811892e6572aadbd670dda8ee6b99fe78598ea4eebe54ea48891bdfe9a`.
- `experiments/review-response-20260927/workload-plan/tools/attempt-runtime/network_boundary_recheck_v2.py`: `9793e44c059831045dd5dc07588ae9fa2545bb17b2dbcfb9e5fd7fe7969c7572`.
- `experiments/review-response-20260927/workload-plan/runtime-package-v1/EVIDENCE.DRAFT.json`: `1fc0dff8f611bccd2587651d8b73b0b0cf1a3919af2881be08caaf22e0878b58`.
- `experiments/review-response-20260927/workload-plan/runtime-package-v1/EVIDENCE.REPAIRED-DRAFT.json`: `75ddafb44ad3a9c906d2b37c6e2079198a5cd507a312331a15e480d126e7f831`.
- `experiments/review-response-20260927/workload-plan/runtime-package-v1/EVIDENCE.NETWORK-DRAFT.json`: `17b004714142b2925a09774d920243e45b8520f8a9cf95bc5bc04264fe26ad25`.
- `experiments/review-response-20260927/workload-plan/runtime-package-v1/COMMON-CONTROLS.NETWORK-DRAFT.json`: `4cf68c99f34ee51b76e55a219b46273f71e6f6bcfe29d0b4de8608ff0761079a`.
- `experiments/review-response-20260927/workload-plan/runtime-package-v1/runtime-freeze.NETWORK-DRAFT.json`: `a4c1a760edcb7750b3dd63b39f7e61c60b56c47c4871e777a5400be91718f93d`.

Reviewer evidence is under `p0/monitor/review-response-20260927/workload-launch-source-binding-review/`: exact source/map snapshots, controls, results, preserved harness setup failures, and the final seal. This review grants no launch authority.

## Delivered LAUNCH successor: closure

Parent delivered `assemble_launch_successor_v1.py` SHA `1c16f31ae2ed4a288ef48e202ca08fb7d8bf2f2b262732880b147f22fe1304dd`. **The bounded closure passes with 16 additional controls (33 total across this review).** Exact snapshots and `LAUNCH-MANIFEST-RESULTS.json` preserve the checks.

- All **186/186** local evidence members authenticate, local/remote keysets agree, and every remote path preserves the exact repository-relative path. All prior 184 NETWORK mappings remain unchanged. Only the new builder and new common-control artifact are added.
- The common-control diff is exactly one selected-builder path/hash replacement plus explicit non-authorizing lineage. All 14 selected source bindings authenticate, with reviewed v3.11 and its network helper still pinned. No selected binding names v3.10 or builder v1; their valid historical evidence remains intact.
- The freeze diff is exactly the six common-control receipt hashes. All task/order/configuration/numerical/seed/timing/measurement fields and pending qualification/clock/approval fields remain unchanged. The successor remains DRAFT, with zero workload attempts and no authority/qualification claim.
- Independently exercised the actual assembler in temporary reviewer-owned directories with copied immutable input documents: a positive creates the same common-control and freeze bytes. Pre-existing output, dangling output symlink, changed pinned input, corrupted builder, corrupted otherwise-unused source member, and missing source member all refuse before creating a new output. No real runtime boundary or implementation mutation is involved.
- The assembler authenticates the full input triplet (lines 32–34), builder (35–37), and all prior evidence (41–43), then enforces the single predecessor binding (45–48), all six expected old receipt references (62–65), and exact other-field equality (66–70). Fresh-path refusal (57–58) plus exclusive writes (81–83) preserve old records. Since the input map itself is pinned to its reviewed bytes, its already-verified keyset cannot silently drift.

Final artifact identities:

| Artifact | SHA-256 |
|---|---|
| `COMMON-CONTROLS.LAUNCH-DRAFT.json` | `631ce8ad342c72f8f1c85d770e07a7d68eb1b1b6ffca26d1e9e1c2d491156e0b` |
| `runtime-freeze.LAUNCH-DRAFT.json` | `f5399dc1e7e5afcab06f14b8fce9d0ac8e60db23393c6fd1e144dd31f14bacd2` |
| `EVIDENCE.LAUNCH-DRAFT.json` | `d2cc25a184a0276fa0b14fb3e926be35045f252bc9d988708807842eada1ac03` |

The source/package binding gap is closed for these exact bytes. Parent still owns final route qualification, clock binding, WP/freeze approval and live launch-packet collection. No scientific gate or experiment count advances from this review.
