# Candidate ownership-unknown repair: independent closure

Reviewed 2026-09-29 UTC. **R1 is closed for both exact successor sources below.** No remaining blocker was found in the bounded cycle0 integration review or this root-wrapper repair. This is source/CPU acceptance only; it opens no gate and does not authorize a boot. The initial failure and reproduction remain in `candidate-cycle0-launcher-v1-source-review.md` and its separate CPU audit.

| Source | Reviewed SHA-256 |
|---|---|
| `tools/run_q1_candidate_cycle0_v1.sh` | `6c0b8482a44695be180a7674d42071a9d7c72f1c643ea29697a06a5d6ed46531` |
| `tools/run_q1_candidate_stage1_v2_6.sh` | `d54550b4ab98508b33baa95cfed9514cda38b4d29034b4fc351ac4c7196206ba` |

Both wrappers initialize `CREATION_ATTEMPTED=0`, set it immediately before the sole staged-launcher invocation, and preserve unproven resources. The wrapper-local sealer now writes `UNKNOWN_CREATION_OWNERSHIP_UNPROVEN` and `cleanup_rc=8` after that latch; it no longer labels post-attempt uncertainty as `no_owned_container`. Pre-attempt refusal retains the previous disposition. These changes do not alter proven-owner cleanup, model/engine settings, job population, native-reference selection, or request settings.

Independently ran **10 extracted-function controls**, using a recording receipt stub and no actual launcher or operational command: both wrappers × pre-attempt readiness refusal, missing CID, failed CID ownership, timeout, and an exit-143 trap after the attempt latch. Pre-attempt returns 4 with the absence suffix. Every post-attempt direct call returns 8 with explicit unknown cleanup and no absence suffix. The exit trap retains unknown cleanup and the original nonzero shell status 143. The full wrapper's explicit failure branches continue to exit 6; `cleanup_rc=8` identifies the sealer's cleanup disposition, not a promise that the wrapper process always exits 8.

Inverse-delta checks restore the exact prior cycle0 SHA `2e8d739ec3f32da9b564e7835db028868647d1c62bf67919c804e71fafc59c81` and preserved root-v2.5 SHA `8b475bffbae5087a62580b8bf17f6d9115349021d9a59b6f286447e5611f9de2` after removing only the latch/sealer repair and root header change. The root original remains intact. The three passing cycle0 gate tests are unaffected by this shell-only repair.

Independent raw controls: `candidate-ownership-unknown-closure-cpu-audit.json`, SHA `932d4c895cf33f3bae1270c91fb9b2b33375be9fb152209e6604953832cc58be`. The separately inspected parent target-Bash syntax/control record is `tools/test_log.candidate_ownership_unknown_closure.json`, SHA `054a965d870db9d7c86012511d0fcf2965490241ae5e09c58b55067b1b677c87`; it reports successful target-shell parsing for both bound files. No live Docker, SSH, GPU, model, cache, or workload operation was run by this reviewer, and no implementation/gate was changed.

Root retries should bind the reviewed v2.6 successor. Cycle0 still requires actual parent-accepted full-native-corpus results, exact final source/gate binding and operational readiness. No full-model qualification, performance, or task outcome follows from these CPU controls.
