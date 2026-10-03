# Cycle0 v5 launch wiring: bounded independent source review

Reviewed 2026-09-29 UTC. **No concrete blocker found in the new selection/wiring delta.** Accepted as prospective source preparation only. This review neither opens a gate nor approves a GPU launch, and does not re-review the publication observers' scientific implementations. No full launcher, Docker, GPU, cache, remote or model operation was run.

## Exact reviewed files

Paths are relative to `experiments/review-response-20260927/`.

| File | SHA-256 |
|---|---|
| `tools/run_q1_candidate_cycle0_v3.sh` | `77dc89459a9b3bef1db1d5c19e3ef9d83788fa19a834f6bc8422438db0a06c4e` |
| `tools/q1_candidate_cycle0_gate_v3.py` | `004da986417c8222bf23e57debc9b88d7df40e473f112c8db7723445c4011038` |
| `tools/q1_make_diag_launcher_v3_5.py` | `c86ec3ca13168fbfa1fe9f405613757e81f7bc0143e9b86c3172db87f9af0a66` |
| `tools/generated/fr14_leg3_launch_nomiddleware.q1diag.v3_5.sh` | `c943ff2f1060376e55857c6b0ed8a9b5c076e2f75749ee019a1463dbfa7b70c4` |
| `identity/launcher-diagnostic.v3_5-from-v3_2.diff` | `8b48046188c45eeed378e96ee1054789e120fc7eadc052574538637d24c557ad` |
| `tools/tests/test_q1_candidate_cycle0_gate_v3.py` | `b7f2e851cc767e4ee97574ecc41f0e5d111cfca13c8bfdac4a80ae74706e018c` |

## Verified delta

- **Selection is consistent.** Wrapper lines 24–37 select diagnostic v3.5, hooks/patcher/job v5, raw audit v4 and both convolution/replay witness sources. The host patcher hash export and launch-binding record select the same v5 patcher/hooks. Patcher v5 imports hooks v5. The offline reducer is pinned for later use; this wrapper does not itself execute that audit or infer success from it.
- **No hidden operational change.** Reversing only the documented version/path/schema substitutions, new helper pins and scope comment restores wrapper v2 byte for byte. Atomic namespace claim, contention checks, producer-owned readiness verification, staged-copy authentication, CID/name/image ownership proof, exact-owned cleanup, and the post-creation unknown-ownership latch are consequently unchanged. Model/serving/numerical settings are unchanged by this delta.
- **The generated launcher has exactly five replacements.** Rendering matches the saved file. Replacing only `q1_patch_candidate_v5.py` with the old `q1_patch_candidate_v1.py` restores the original v3.2 bytes, SHA `db89386fb76db8594c00b575bddf369f6f8b05a6183f425734d1679f81de5352`. The five sites are host path/hash validation, its error description, in-container presence, actual patcher invocation and provenance. The saved unified diff matches independent reconstruction exactly. Base-byte drift and occurrence-count drift both refuse generation.
- **Campaign imports on this route are pinned before import.** Wrapper lines 45–64 verify every `REVIEWED` path before loading the gate module. Static transitive traversal found all 25 campaign modules reachable from the selected gate/builder, hook, patcher, driver, reducer, renderer, readiness and host-verification paths represented in the pin set. The repository topology import is covered by `topology_module`. Native corpus `engine_view`/`reduce`/`main` contain three further deferred imports; the candidate route does not call those functions. Candidate job construction calls `N.build`, whose native-source hash manifest must equal the already approved corpus. This is not a claim that every file in a future remote freeze has been inspected or that third-party runtime identity was newly qualified.
- **Scientific admission is preserved.** Inverse substitution of the builder import and collector-revision check restores gate v2 exactly. Gate v3 still requires accepted root instrumentation, rebuilds the prepared job from the accepted full native corpus/reduction, requires the fixed same-case native A/r0 reference and accepts only the declared calibration cycle0 scope with R2 in one explicitly selected A or B process. Packed/native alternate-reference selection and held-out or continuous-cycle admission are not added. The new job/helper hashes are part of the rebuilt canonical job, so the prior collector cannot silently pass as v5.
- **Root retry remains separate.** The root wrapper v2.7, root hooks/patcher/job v1.2 retain the exact hashes recorded in the earlier accepted root-repair review. No root source or authority was edited by this review; the cycle0 successors do not select those root files for their collector.

## CPU controls and boundary

Independently ran the four supplied gate test methods: all pass. Their native builder is deliberately mocked; they provide admission-logic controls, not a fabricated accepted native result. Separately executed only the actual wrapper's extracted Python pre-import block with temporary sentinel modules. A closed gate and changed convolution/replay helper hashes each refused before any campaign import or output namespace creation. No shell launcher was executed.

Independent audit: `q1-cycle0-v5-launch-wiring-independent-audit.json`, SHA `19e75f7daa7e5e1f15c8802ace011cd80f4584a08b564e2bf40a5bea50bcbfb7`. It records the exact inverse checks, generated delta, dependency traversal, sentinel refusals and preserved root hashes. Existing operational/result prerequisites and parent-owned gates still govern any future execution.
