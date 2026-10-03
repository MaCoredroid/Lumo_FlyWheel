# Agent output preparation / executor v3.10 independent review

**Disposition: bounded CPU/source acceptance on the final hashes below.** One concrete caller receipt-binding defect was reproduced and repaired during review. No Docker, SSH, network, GPU, model, agent, evaluator, or task execution was performed. Local temporary-directory Python programs and injected SDK/lifecycle controls only; WP and all execution gates remain parent-owned.

| Final source, campaign `workload-plan/tools/attempt-runtime` | SHA-256 |
| --- | --- |
| `agent_output_v1.py` | `83d3f68a215112bcd65f55614b0ee77017a695efaa78e9861be62031efbe1f7c` |
| `sole_executor_v3_10.py` | `82b56406c23565897dd4ee8d663c555bba449389c72e65b52dba2f02a7d15b68` |
| `test_agent_output_v1.py` | `8771564b64877c48a8045265bbffdbbd99225ec35398b4f3d6e7f2e1d692595b` |

## Source and integration

`AO.validate` checks the exact preparation spec/source hash, immutable task image command, root writer, and unique expected writable `/out` mount for each of the four frozen arms. The real rendered commands pass; cross-arm destinations and wrong helper hashes refuse. The caller validates before writing STARTED/reserving the attempt, then prepares inside ownership, after agent-host preflight and before the probe or agent creation (`sole_executor_v3_10.py:574,789–792`). No scientific, engine, numerical, agent/evaluator timeout, or task-population setting is changed by this delta.

The actual `PREPARE` program uses directory descriptors and `O_NOFOLLOW` through each descendant, exclusively creates the arm leaf, then exclusively creates `OWNER.json` and the empty, single-link regular `qwen_trace.jsonl`. Existing leaves refuse even if empty. No delete, truncate-existing-output, replacement, or reuse path exists in this program. The existing agent wrapper's trace-file precondition is satisfied by successful preparation. The mount is the existing base `/home/mark/swe_eval_offload`, with the fixed three-part descendant path; no model or agent bundle is mounted into this helper.

The helper requests the pinned amd64 task image, `runc`, network `none`, read-only container root, explicit GPU-visibility environment, 128 MiB memory, and one CPU, with only the declared writable output-base mount (`agent_output_v1.py:82–86`). This is a CPU/file-only helper. The final record appropriately distinguishes **no GPU operations requested** from **GPU visibility not observed**; it does not claim a live device-isolation inspection occurred.

Cleanup proves exact returned CID, container name, image ID and all expected labels before stop/remove; typed absence verifies completion (`agent_output_v1.py:114–122`). Failure or cancellation is re-raised after retaining the record. Missing create-return identity never authorizes speculative deletion. Failure logs are now retained best-effort before cleanup, with a bounded accepted payload and an explicit `log_error` when unavailable; log failure does not suppress cleanup.

## Reproduced defect and closure

The original caller SHA `98f4b0c158ebd02143bc216e6048de63957844fa38e34deb2a38be9ccc632c21` verified the output-preparation receipt only when its sequence included `create_agent`. Preparation followed by a probe/setup failure could therefore leave `agent_output_prepared` without `create_agent`, and the next-arm output check ignored a missing receipt. A raising `AO.prepare` also did not assign its retained failure receipt into the RESULT field.

An exact AST control of the original added prior-validation statements reproduced **acceptance with a missing receipt** for sequence `['agent_output_prepared']`; adding `create_agent` made the same control refuse. These are caller validation slices, not a claim of running the entire admission stack. The parent preserved the failing caller under `p0/monitor/review-response-20260927/agent-output-before-receipt-binding-fix/`; its SHA was independently verified.

The final caller records `agent_output_preparation_attempted` before the call, binds a retained receipt in `retain_result` on success and failure, and forces terminal false for missing/bad binding or unverified returned-CID cleanup. Prior validation now checks every attempted, issued, retained, or ledger-named preparation (`:388–396,701–722`). `PREPARED` is required when agent creation occurred; a genuine retained REFUSED setup result remains a failure. If create returned no CID, the separately required verified empty agent-host sweep can establish absence; a returned CID still requires its own verified cleanup. No failure becomes an implicit retry or disappears from the attempt population.

## Independent CPU controls

- **5 supplied tests pass** on final bytes: actual preparation success/no reuse, empty existing leaf refusal, symlink-parent refusal with target preservation, traversal refusal, and all four real rendered-command bindings plus negative controls. Command: `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_agent_output_v1`.
- **Actual helper with injected SDK:** success retains PREPARED and verified exact-CID removal; wait cancellation retains REFUSED, stops/removes only its proven CID, verifies absence, and re-raises. Wrong cleanup identity refuses stop/remove and remains unverified. Ambiguous create retains REFUSED with no CID and performs no speculative cleanup. The source-approval check alone was stubbed for the fake runtime; real helper checks, file retention and SDK call ordering executed.
- **Final log/isolation delta:** cancellation with available partial logs retains them; an injected log-read error records `log_error`. Both still verify exact-CID cleanup and re-raise cancellation. The fake SDK checks the actual final `runc`, network, read-only-root and visibility environment arguments.
- **Exact final `retain_result` and prior-validation AST controls:** PREPARED/clean, REFUSED/clean, and REFUSED/no-CID with verified empty sweep retain their receipt and are admitted by this component. Returned-CID/unverified cleanup, missing receipt, and wrong attempt identity force terminal false and refuse the next-arm output check. REFUSED receipts are rejected if the sequence says an agent was created. These tests exercise the actual final functions/statements with surrounding lifecycle seams, not replacement validation logic.

No remaining blocker was found in this bounded wiring delta. Real helper/container state and filesystem effects remain prospective until the separately authorized runtime; this source review does not establish their execution. The previously reviewed deadline implementation is unchanged and was not requalified as a workload experiment.
