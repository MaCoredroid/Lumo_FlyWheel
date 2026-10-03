# Native-only M1-C v1: bounded draft source review

**Disposition: native collection control flow is consistent with the declared population; repair launcher F1 before launch.** This is a three-file draft snapshot, not acceptance of a final freeze, topology/declaration package, numerical reducer or runtime result. No Torch import, GPU operation, container operation, source edit or gate change was performed.

Snapshot: `p0/monitor/review-response-20260927/m1-native-c1-draft-independent-review-20260928/`. Exact reviewed hashes:

| Source | SHA-256 |
| --- | --- |
| `m1_native_reference_pack_v1.py` | `098de87f5b8c1e9ac61be2b02746a34897bc7d7f8da6e64703c8b3ef2a40e257` |
| `m1_native_c1_collect_v1.py` | `698ce5d2afca2bc628e1a9f8acd460cc58f4dde7c402409f4cdde4a6815d90c4` |
| `run_m1_native_c1_owned_v1.py` | `a6e1abe73e62785450ead2a1011592dd311f1ffd31cf4ac2ced3d07b505d5db4` |

## F1 — failed create/first inspect can strand the owned container

Launcher lines 130–136 obtain/authenticate the CID only after `cmd(docker create)` returns; cleanup runs only when `owned` is already true (159–160). If create times out after writing its CID file, `subprocess.TimeoutExpired` skips all CID recovery. Likewise, a transient first inspection failure leaves `owned=False`. Both go straight to a failed receipt without inspecting/removing the container they just created.

Injected standard-library controls reproduced both cases against these exact bytes. The valid control completed and removed its CID. Create-timeout and first-inspect-failure controls both produced `FAILED_PRESERVED`, with `created=true`, `removed=false`, and no cleanup field/error. No real command was run. Reproducer: `reproduce_lifecycle.py`; exact receipts of the mocked command sequences: `lifecycle-control-results.json` in the snapshot.

**Minimal closure:** record that create was attempted; in finalization recover the exact CID file even if the command raised, authenticate CID/image/name/nonce, then stop/remove only the proven owned container. A failed/unknown inspection must be recorded as cleanup-unproven, not treated as absence. Preserve the create/inspect primary failure even if cleanup succeeds. Capture command timeout evidence, not merely commands that returned. After successful `docker rm`, add successful exact-ID absence enumeration (enumeration failure is not absence) before the terminal receipt. Keep one launch/no retry; cleanup recovery must never start the container. Focused controls should cover normal exit, create timeout after CID creation, first inspect failure, failed stop/remove and failed absence enumeration. Broader launcher redesign is unnecessary.

## Accepted source-level behavior

The pack exporter pins the accepted receipt, frozen baseline source, CPU Torch 2.4.1/no CUDA, NumPy 2.3.4 and four threads, matching the accepted receipt runtime. It checks original operand and both C2 surface-value/preparation hashes before saving each member (`:31–76`). This is materialization of accepted reference values, not new calibration or changed arithmetic. It preserves incomplete status on exceptions.

The native call (`collector:97–107`) matches the existing reviewed `Backend.native_chain` signature: token slice then batch dimension; BF16 q/k/v/a/b; FP32 A_log/dt_bias; FP32 scratch state `[2,48,128,128]`; `cu_seqlens=[0,1]`, int32 state index `[[1]]`; in-place state; native in-kernel q/k normalization; scale `1/sqrt(128)`; BF16 stored output. Native module/function hashes match the previously accepted native identity. The scratch is independent of any Lumo/author candidate. Snapshot/reset/restore operations copy data, so the alternate sibling cannot mutate saved history.

`collect_repeat` resets each root-to-node path; collects exactly 28 first-verification outputs; resets again for the uninterrupted 11-update history; and restores state10 before applying alternate final node14. Two repeats also reset after the alternate sibling. Eight independent injected control-flow checks passed: exact output paths, cumulative states, sibling divergence only at the last update, repeat reset, call count and duplicate/cycle/inactive-ancestor rejection. They used the frozen topology SHA `14c5b5659f8cb66346367bbb27d123afc122fbe964c179cf9ae0f7fb7288f374` and imported no Torch.

Expected count is **310 native update calls per layer-instance across two repeats**, hence **29,760 calls** for 2 seeds × 48 layers. The scientific domain remains 142,848 unique Lumo cells (129,024 output + 13,824 publication cells). Raw intermediate states and repeats are supporting evidence, not extra scientific cells. Collector fields record finite/repeat equality; `COLLECTED_REFERENCE_ONLY` deliberately supplies no resolved numerical verdict. Launcher `COMPLETE_REFERENCE_COLLECTION_REVIEW_PENDING` must retain that meaning even if raw finite/repeat checks are false. The later reducer must derive exact record keys, shapes, hashes, finite/repeat masks and witness coverage rather than treating 96 records alone as qualification.

## Before freezing the package

- Many pack/collector identity and domain checks are Python `assert` statements (pack 31–38,55,63–65,78; collector 136–149,163–168). These disappear under optimized Python. Use explicit exceptions, or at minimum an explicit non-assert rejection of `sys.flags.optimize != 0` before the checks and bind the actual invocation. This is a small fail-closed execution prerequisite; the ordinary nonoptimized code inspected above performs the intended checks.
- Bind the exact new topology/declaration to the existing manifest/source, native six-field arithmetic declaration, stage scope and all consumed sources. These were still being prepared when this snapshot was taken; absence from this draft is not a new regression. The new topology must preserve the complete 28-node order/parents, not only the three paths checked by collector assertions.
- Retain the existing 12 GiB in-process floor and separate host/CUDA observation semantics. The pinned image, fresh one-use gate, output namespace, exclusive executor lock, stopped/owned-CID policy, source/pack hashes and no-reclaim/no-separate-query settings are explicit in the launcher. No full-model memory floor or model boot is introduced.

Final launch review can be bounded to F1/optimization closure plus equality of the completed frozen package and the accepted population; unchanged native call/history semantics need no broad re-review.
