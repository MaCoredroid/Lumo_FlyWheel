# Cycle0 v9 downstream propagation and native-preparer pin review

**Bounded source-readiness PASS. No new runtime integration or fail-open defect found in this delta.** No GPU, model, container, remote, cache, gate, authorization or implementation action was performed. This is neither numerical qualification nor permission to launch. The separate v9 live-connection review remains applicable to the identical hooks/patcher/MTP helper bytes; its internals were not reopened.

## V9 propagation

The job builder changes the collector revision to 9, selects hooks/patcher v9 and binds `mtp_kv_witness_sha256`. The raw auditor checks that pin before reading records and requires the v9 case schema. After the existing authenticated hidden-event/case join, its new `MW.audit(...)` call uses the owner reconstructed from that same `hidden_event`, the exact case nodes and the case-bound prefix length. The real witness auditor re-derives its map, authenticates both raw byte captures and validates ownership, restoration/index readbacks and the captured nondestination rows; the new downstream call cannot silently skip a missing witness. Its result is retained in each raw-audit row. No numerical MTP claim is introduced.

The chain is consistently job v9 → gate v7 → wrapper v7 → hooks/patcher v9, with raw auditor v8 → authenticator v3 → driver v2.4. The authenticator retains the native per-case A/r0 corpus binding and turns raw-audit exceptions into authentication failure. The driver loop remains 84 calibration cases × R2, null request seed, and explicit 168-request denominator even after a partial failure. Exact raw disagreement stays visible independently of collection completeness.

The wrapper contains 79 distinct source-pin keys, including the new MTP helper and all active successor files. Its launch-binding and exported patcher digest select v9. The generated v3.9 launcher is byte-identical to v3.8 after exactly five patcher-path substitutions; the accepted runner/rejection/Eagle mandatory prehash flow, model/settings and cleanup behavior are unchanged. Re-running the pure generator reproduces the saved launcher and saved diff exactly.

## Native preparer v2

The `prepare_native_corpus_recovery_v2.py` diff is exactly two additions: the current model-config path is included in local/remote source-hash preflight and `reviewed_hashes`, then its digest is required to equal `corpus['model_config_sha256']`. Both checks occur before timestamps, gate construction and authorization saves. The actual local model config matches the frozen corpus digest.

The unchanged native launcher also rebuilds the complete corpus from the same source/config before comparing it to the pinned manifest; the new preparer pin supplies the missing explicit remote preparation check. This review does not execute the preparer, issue the next B authority or assert present machine readiness. Its inherited normal-Python invocation/operational policy is unchanged.

## Independent bounded CPU controls

All controls passed using standard-library Python without Torch/CUDA or remote calls:

- Four successor gate test methods (accepted native-builder stub), plus actual authenticator routing with eight malformed stage/case/native/receipt/seal controls.
- Actual 168-request driver loop with synthetic authenticated responses. An injected failure on request 2 retained both responses/receipts, one valid authentication and `expected_requests=168`; null request seed and valid disagreement handling were checked.
- Missing and wrong MTP witness source pins refused even with recomputed job canonical hashes.
- The exact new MTP audit assignment was AST-extracted from the real raw auditor and executed with the real pure byte oracle. A consistent synthetic record passed; 13 owner, prefix, path, restoration, missing/corrupt raw and resealed guard-byte controls refused. This exercises downstream joins, not live cache capture or a model.
- Exact v3.9 generator/diff reproduction and all 79 wrapper pin entries passed.
- The actual preparer hash-preflight block was AST-extracted and run with a read-only SSH stub. Correct binding passed, while remote model-config mismatch and corpus model-config mismatch refused before any authority code could run. The complete v1→v2 text delta was checked against the two declared additions.

The parent's preserved connected Torch log reports 10 passing methods in 3.032 seconds; it is copied only as parent-produced supporting evidence, not counted as independently executed here. The prior live-connection review SHA is `10102666da1aac807ba15203255af8d79d04c3574d7cbcba6a21c8de27ae4e8f`.

## Evidence

Review time: `2026-09-29T10:25:34.485125+00:00`. Reproducer, result JSON, gate-control log, immutable source copies and hashes are in `p0/monitor/review-response-20260927/cycle0-v9-propagation-review/`. All snapshotted implementation bytes still matched at the end of the controls. No final freeze or run gate is granted by this note.

| Source | SHA-256 |
| --- | --- |
| `q1_candidate_hooks_v9.py` | `905e35898a504357b8a96f2315395e0bec37d5b2509d2ae0a0709b827271317c` |
| `q1_candidate_job_v9.py` | `881232bc8934d6a99879ea909d69c3f9bd1255ddea80ed648c48abe258f8376e` |
| `q1_patch_candidate_v9.py` | `93894ce2758c8283f60dd90f64b364fd1efd6f310ab63ff3f5e25814928ab202` |
| `q1_candidate_raw_audit_v8.py` | `85e7e0aad47ed850e7edb3c742af06eaf72d153bc8a3aa732f312eb5748f8753` |
| `q1_candidate_driver_v2_4.py` | `a84e57c8801dc14ab52d25680cd51324cb6525454a937f55992c12194e9306f7` |
| `q1_candidate_cycle0_seal_auth_v3.py` | `1299c8882ee58d487fa18392f60445941b8188f226d604b9bbb95c0b06219da5` |
| `q1_candidate_cycle0_gate_v7.py` | `ae4dc6524a2cba8e27a23ce4726a34d03a891d97ebb23c2eb6e66d01e1979eb1` |
| `run_q1_candidate_cycle0_v7.sh` | `b76e650491b5ac6d4eed5e5275316347ddade8a96f767ec8ef8cb793afc44c61` |
| `q1_make_diag_launcher_v3_9.py` | `2277a14620e6218318eddeac975543d6e7f3cdbddbfd16008526876c5575fe06` |
| `generated/fr14_leg3_launch_nomiddleware.q1diag.v3_9.sh` | `2fd766f3b4e82d1bd50d6972f17de2aaf0c235c23869c2b6c240287fe60e5480` |
| `q1_hidden_publication_bridge_v1.py` | `00d30bc67e34027b09eae5a0386017ffc7f8766d0a7399218abb5920a15622b8` |
| `q1_hidden_publication_live_v1.py` | `e634c349434f29cbc5d5045fdd2098144a56871a4a66721a1ff85dfd0bab3eae` |
| `q1_hidden_publication_witness_v1.py` | `92ac01369e8e337716183172ec8f64ec266b6601bca21384abef834361f2cd18` |
| `q1_hidden_event_binding_v1.py` | `9ea8e10f4eb290840a908f4b78f38c5f29c0144d0e40dcab5b3750650c1550e4` |
| `q1_mtp_kv_publication_witness_v1.py` | `9239e35fa10395cd01e7e2a9a7c4a34b42fea753b1f0b064ac16909937ace976` |
| `identity/launcher-diagnostic.v3_9-from-v3_2.diff` | `3bf6f6c3fc8f7f67404558d8a4178d9f852dea0b7cf61288a315821f32f78cee` |
| `monitor/prepare_native_corpus_recovery_v2.py` | `95de54e3dbb12ea2c57313e312aa0fb6b57dc2d83e06703e9b8bd217de4f6bc5` |
