# M1 v3.2 late-evidence closure review

**Disposition: APPROVE this bounded collector correction.** The remaining v3.1 F1 failure-lifecycle finding is closed for the reviewed source. No further blocker was identified in this delta. Final-freeze hash binding and host-launcher review remain parent-owned; this note is not launch authority or a GPU result.

Reviewed early snapshot:
`p0/monitor/review-response-20260927/m1-v32-early-reviewed-20260928T0527Z/repo/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927`.

Independently verified collector `tools/m1/m1_stage_collector_v3_2.py` SHA256:
`e6b4b62b575b18c222210b515dfab2ecf5b420070620d0e1832526d9a4ab43f1`.

## Source closure

- `demote_on_secondary`, lines 41–47, demotes an otherwise complete method and preserves an earlier primary error/disposition.
- `run_method`, lines 113–121, applies this rule after receipt writing/hashing and again after a failed status write. A status write gets one evidence-only retry; the method itself is never rerun. If both status writes fail, the demoted status remains available to the stage manifest.
- `run_stage`, lines 171–188, retains incremental and finalization failures and applies `_finalize_exit` after hashing and after a failed final manifest write. The fallback failure manifest does not restore success. `_finalize_exit`, lines 192–199, forces an otherwise-zero exit to 3 when stage evidence fails.
- `main`, lines 209–215, returns 3 for an uncaught stage exception and otherwise returns the finalized stage code.

Earlier path resolution, protected construction, reset premise, raw completeness, device, buffer and adapter closures remain closed. This review added no protocol or numerical criterion.

## Independent focused reproduction

Executed only the actual ASTs of `demote_on_secondary`, `run_method`, `_write_manifest`, `run_stage`, `_finalize_exit` and `main`, with stdlib dependencies, an in-memory `StringIO` filesystem, and injected cycle/completeness stubs. Thus the real method and stage disposition paths ran, while no tensors, kernels or actual collector process ran. The clean control supplied the receipt fields used by serialization and reached complete validation/reduction before writing. Failure controls raised `OSError` at the specific write/hash seam.

| Control | Observed disposition |
| --- | --- |
| Clean cycle and all evidence saved | `complete`; exit 0; `finalization_ok=true` |
| Receipt write failure | `evidence_serialization_failed`; exit 3; exception retained |
| Receipt hash failure | `evidence_serialization_failed`; exit 3; hash failures retained |
| First status write fails, retry succeeds | Demoted status saved on second write; exit 3 |
| Both status writes fail | Demoted status retained in stage manifest; exactly two write attempts; exit 3 |
| Earlier constructor error plus failed status writes | Original `failed/executor_construction` and primary exception preserved; secondary failures retained; exit 3 |
| Tensor-index hash failure | Stage exit 3; `finalization_ok=false` |
| Other evidence-file hash failure | Stage exit 3; `finalization_ok=false` |
| Incremental manifest write failure | Retained in final manifest; exit 3; `finalization_ok=false` |
| Final manifest write fails, fallback succeeds | Failure manifest saved with exit 3 and `finalization_ok=false` |
| Final and fallback manifests both fail | Both exceptions retained in returned state; exit 3 |
| Uncaught stage exception through actual CLI function | Exit 3 |

**12 focused controls passed.** Previously written receipt/raw files remain preserved; final stage disposition governs whether the attempt succeeded when later mandatory evidence fails. The tests included in the snapshot were read, but the full author test suite was not independently rerun.

No Torch import, GPU/CUDA execution, SSH, container, model, source edit, gate change, or real experiment occurred. Only this review note was written. Approval is limited to the remaining late-evidence repair at the exact source hash above and the already specified untimed initialization/diagnostic scope.
