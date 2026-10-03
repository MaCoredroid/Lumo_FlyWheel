# Native MTP imported-prefix history: bounded source/CPU review

Date: 2026-09-29. Disposition: **PASS for this bookkeeping helper under its explicit caller contract; runtime integration and qualification remain pending.** No material implementation defect remains from this bounded review. No model, GPU, container, network, campaign execution, or implementation changes occurred.

## Reviewed source and semantics

All paths below are relative to `experiments/review-response-20260927/tools/`; exact dependency hashes are in the accompanying CPU JSON and review seal.

| Source | SHA-256 |
|---|---|
| `q1_native_mtp_history_joint_v1.py` | `ed0742f6a544fd7a5c3058ad581cbdf6038bfd00dcb5cc5799e0acd829f5b6fc` |
| `tests/test_q1_native_mtp_history_joint_v1.py` | `a101628ea22bc95972f026aa652ae61b602c86f421e907802ce7d04c8d82f3ce` |
| Accepted `q1_native_mtp_history_v2.py` | `83dc891ca4200cbd3dfb77a92269574105dca1ee2569a919c63367f0214b066c` |

- **Boundary and destination ownership:** new helper lines 23–29 require a live accepted History-v2 owner/generation, same prefix extent, no target step, deferred followup or live program context, and a destination-only first-pass bootstrap ending at the designated root. Its final bootstrap allocation and positive metadata serial must remain present. Base-v2 `check` (57–60) binds the case/request and owner generation; actual owner `assert_live_cache` (139–142) also checks readiness and allocation identity.
- **Source/receipt join:** lines 30–37 use the accepted structural source audit, bind the actual destination prompt-prefix token hash and root, require the single latched union-import receipt, and match source observation, joint/target/MTP digests and extent. The supplied post-import MTP snapshot must use the destination group, block unit and logical-to-physical block mapping. `q1_joint_source_v1.py:35–53` checks the source's prefix record/history digests and boundary structure; the single union primitive returns the relevant receipt at `q1_joint_common_o0_import_v1.py:64–67`.
- **History semantics:** lines 40–42 copy the destination bootstrap separately and reset only `records`. They preserve the destination owner, current lease, generation, `last_serial`, `materialized`, program, and future real model work. The import record retains source record/job/history hashes, destination bootstrap digest, and copied receipt/snapshot. No source hidden tensor, source address, source lease or generation replaces a destination object. Base-v2 `pre` (64–76) therefore continues from P with a fresh serial and authentic destination allocation; its later deferred-followup index is relative to the new record segment (132), and terminal checks (140–146) apply to that segment.
- **Fail-stop and reuse:** any exception, including `BaseException`, poisons both history and owner and clears owner readiness (44–46). An adoption attempt that fails before reset leaves the destination bootstrap records intact; repairing inputs does not resurrect that history. A successful history cannot adopt again. This is refusal/stop behavior, not a rollback assertion about already imported model bytes.

## CPU evidence

Executed locally with `/Users/zhiyuanma/miniforge3/bin/python`, Torch 2.8.0, `CUDA_VISIBLE_DEVICES=''`, CPU thread limits 2 and bytecode writes disabled. The supplied six tests pass. Fifteen independent controls pass: separate quiescent wrong-source/joint-digest/target-digest/prefix refusals; absent mutation latch and nonunion receipt; tampered source history; wrong destination group/block unit; changed generation; pending followup/live program; missing serial; interruption poisoning; and successful identity preservation/evidence-copy isolation. Every independent ordinary refusal also refuses a retry on the same history.

The supplied test at lines 36–39 combines a wrong receipt with a pending step, so it alone does not isolate receipt validation. The independent quiescent wrong-receipt control resolves that coverage ambiguity; no implementation repair is needed.

Reproducible evidence: `p0/monitor/review-response-20260927/native-mtp-history-joint-v1-review/{independent_controls.py,independent-controls.json,supplied-controls.log,REVIEW-SEAL.json}`. These are synthetic bookkeeping controls, not actual tensor import or numerical/model execution tests. Source hashes were rechecked unchanged before sealing.

## Exact integration boundary

The method explicitly requires the caller to authenticate the full sealed source/raw audit **before mutation**, perform one successful union import, and export `destination_kv` from the **actual post-import live cache** before either route consumes the root (17–21, 38). This helper does not independently authenticate `record_sha256`/`job_sha256`, read raw object files, rerun the importer, or establish target-state correctness from an MTP digest. A copied source snapshot or a receipt alone cannot meet its actual-snapshot precondition. The receipt used by the hook must be the return from that destination's union transaction.

The unwired runtime hook/final seal must preserve all three evidence segments: authenticated source provenance, natural destination bootstrap, and destination post-import continuation; inherited `finish()` returns only the current `records`, so the import/bootstrap fields must be emitted separately. This is the already-declared integration obligation, not a new experiment or a defect in the scoped helper. No launch, numerical qualification, next-draft equality, or full-model result follows from this review.
