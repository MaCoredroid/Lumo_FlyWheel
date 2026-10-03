# Joint-native common-O0 outer integration: independent source/CPU review

Initial disposition: HOLD for the two concrete integration corrections below. No joint `SOURCES.json`, frozen prospective plan, launch gate, GPU execution or result is claimed. This review covers the new outer wiring, not a repeat review of accepted tensor/importer/auditor predicates.

## Findings

**R1 — Hook source paths are outside the container's mounted namespace.** `q1_native_joint_common_o0_plan_v1.py:43–44` writes the joint source manifest and raw source root under `/home/mark/lumotree-review-20260927/...`. The host driver needs host-readable paths, but `q1_reference_hooks_joint_common_o0_v1.py:26–27` consumes these same fields directly inside the engine. `q1_spec_off_joint_common_o0_config_v1.py:8–13` retains the base Docker mounts: the repository is visible at `/workspace:ro`, not its original host path. A static check using the actual renderer confirms none of its four mount destinations contains either source path. The hook would fail opening its source manifest before continuation. Repair with explicit, bound host/container resolution or a narrowly declared read-only source mount; do not simply change the job to container paths and break the host auditor.

**R2 — Recursive source hashes are evaluated after the imports they are meant to authenticate.** `run_q1_native_joint_common_o0_v1.sh:40–49` checks the plan module and manifest, then imports the plan. Its top-level imports execute `q1_native_corpus_v1`, `q1_reference_job`, `q1_joint_source_plan_v1`, `q1_joint_common_o0_sources_v1` and further helpers. The full hash comparison occurs only at lines 71–73, after `build` has already executed. The plan's static closure discovery is useful, but does not make this a pre-import check. The actual first heredoc was tested with the captured exact plan and a deliberately mismatched, harmless corpus-module sentinel: that dependency executed before rejection. Move the complete static closure authentication ahead of every campaign import/build call. A closed-gate control alone would not cover this admitted-gate/mismatched-dependency path.

**R3 — Receipt description still names the prior three-source stage.** The new wrapper's receipt `note` says “three joint target/MTP prefix-source preparations only,” while the driver runs 84 paths × R2 for one selected A/B process. Update the description to the actual common-O0 continuation scope, retaining no timing/qualification claim. This is a record-label correction, not a numerical criterion change.

## Bounded controls and source conclusions

Evidence is retained at `p0/monitor/review-response-20260927/native-joint-outer-review-20260929/`. `check_outer.py` executes the captured gate heredoc only in a temporary directory with a harmless mismatched dependency; it never reaches shell contention, Docker or GPU commands. Driver tests execute the captured actual `run` function over the real fixture's 84 calibration cases and prefix bytes, with HTTP, outer seals and the joint raw-audit body injected. These test scheduling/connection, not raw tensor correctness.

- A and B positive driver controls each produce 168 mock requests and 168 ordered raw-audit calls. All 336 observation salts are distinct and match both request and returned control. The next request is forbidden by the mock unless the prior audit ran.
- A failing third joint raw-audit retains exactly three per-observation driver records and produces an incomplete verdict; no fourth request is sent. A returned salt mismatch stops after one request, before accepting a joint audit.
- Actual config comparison preserves target serve arguments, engine environment, image, model, FA2 settings and Docker arguments. The sampler payload retains temperature 0, top-p 1 and seed 0; wall-time recording remains diagnostic and adds no timing qualification.
- Applying the actual new patch transform to the pinned runner produces SHA256 `604a184bd11979357411046c7946f74ad762b3bfbc5d27df5d8c5b188127965d`. Its delta from the reviewed joint-source worker patch is exactly the hook-module import substitution. The apply adapter retains pinned input/readback, fresh output/receipt and source-alias refusal logic.
- Plan scope is explicitly aligned-nonpacked only, A/B, R2, 84 fixed case IDs, 168 requests per process and 336 total observations. Job generation rebuilds the plan, invokes `source_for_case` for every case, and uses the predetermined three-prefix source map. That map binds each destination prefix/root and source record; it has no outcome-based selection branch. No genuine three-source manifest exists yet, so this is a source conclusion, not a successful real-source preparation claim.
- Wrapper cleanup/readiness logic is inherited from the reviewed joint-source launcher: exact created-CID/name/image proof, an ambiguous post-creation/no-CID state remains a cleanup failure, and readiness consumption does not itself run a recovery operation. No additional operational defect was identified in this narrow delta; none of those paths were exercised live.

## Captured source identities

| File | SHA256 |
|---|---|
| `q1_native_joint_common_o0_plan_v1.py` | `15d55be408df590421634d2a97f9816aacefe356afd756d9371fa07e58460c5c` |
| `q1_reference_driver_joint_common_o0_v1.py` | `bcb975c8750ed0b59dc8de7b1aca224b605476dae5be174ec840743020ade19f` |
| `q1_patch_native_mtp_joint_common_o0_v1.py` | `d4b05bc7e400c856788a27513737e8a8939fa8cf1539ea8ca4b24d900cfc82c7` |
| `q1_apply_native_mtp_joint_common_o0_v1.py` | `a87bda8293721a395cfda8656c036cea8e0a581217593e65d07419421fe743f2` |
| `q1_spec_off_joint_common_o0_config_v1.py` | `9c8dfe06430cecaf8b647dce56f2525a972ae464a4c792469400ee4a7dca3d7b` |
| `run_q1_native_joint_common_o0_v1.sh` | `baf95c4948a01693420f9cd7cc3078c175a7b20470161d14ea9cd76464a97d01` |
| `q1_reference_hooks_joint_common_o0_v1.py` | `6d4a95e06a95f8f00f708800a146ca92a6264a0e5d80e3210a97061866c18809` |

These are independent review snapshots, not an execution freeze. No implementation, scientific gate, source inventory or experimental result was modified.
