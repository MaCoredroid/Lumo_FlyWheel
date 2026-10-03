# M1v2 draft: parent stage and evidence review

Reviewed the 14-file hash-verified parent snapshot `p0/monitor/review-response-20260927/m1-v2-draft-20260928T0428Z` and its proposed first GPU stage. This is preparation, not a final worker freeze or a launch gate. Source-call correctness and v1 finding closures are independently reviewed separately.

## Prospective scope decision

Retain the proposed untimed, one-process, serial four-variant initialization stage at B1, 48 layers, ordinary-random seed 20260928, unchanged pinned image. Freeze three accepted paths **root-only, n14, n15**, with the source-authoritative node arrays in `M1-INITIALIZATION-STAGE-DESIGN.json`. The proposal's n01=[0,1] lies on the same spine as n14=[0,1,4,9,14], so it does not exercise off-spine publication. Replacing it prospectively with sibling branch n15=[0,1,4,9,15] preserves the three-step count while covering a real branch choice. This is before any M1 execution/output. TreeWY's final deferred commit is still mandatory; author-default and aligned-local labels remain separate. No optional second dot_bf16 setting in this first stage.

## Complete the proposed collector before requesting a gate

The proposal promises first verification versus C2, nonfinite stop rules, and inspectable durable states. The actual `run_cycle` and `ImageExecutor` currently record tensor hashes/shapes and state digests; they do not call `per_head_errors`/C2, persist verification outputs or endpoint-state bytes, or test these outputs/states for finiteness. A digest is not a numerical check. Record **all 48 layers**, with all 28 active first-step node outputs, and durable endpoints after each logical publication including final flush. Save these before buffers/results are overwritten; bind the shared input/S0 bytes and declared reference computation. Use raw per-head diagnostics only at this stage, with nonfinite/structural failure stopping. A change in state digest is informative, not a correctness proof or a substitute for raw comparison. Keep prior qualification criteria unchanged; this stage alone qualifies no method for timing.

There is no executable runtime stage caller in the reviewed package: driver `__main__` calls `DryRunExecutor`; executor `__main__` prints import/CUDA availability. Add the narrow proposed caller and its owned launch/finalization wrapper. Persist the actual full call log whose hash is cited, plus per-method/step partial evidence on an exception before `run_cycle` returns; `_dispatch` currently appends successful records after the callable, so failed calls otherwise disappear. This is implementation of the proposed stage, not an added experiment.

## Runtime identity must be enforced from the frozen package

The executor's default CUDA device is unindexed whereas tensors become indexed; independent review is checking this concrete dispatch defect. The current runner hash check compares a current file hash to `lumo_binding()`'s freshly recomputed hash of that same file. It needs the independently frozen expected value `ab27fc9cf680bdd2b7c3ad994fc9f7bffe0f9ff1b0f35dc2cc36018f364dea29` and applicable kernel/topology/native bindings before import. The proposal says only the pinned image is accepted by runtime reducers, but the current image check is regex-format validation. The final caller must bind the actual inspected image plus expected runtime versions/device and exact source freeze; a self-reported environment value is not image proof. Preserve v1 files and issue a separately hashed v2 freeze.

Disposition: continue CPU completion/repair and review the settled executable package. GATE-M1 remains closed. No GPU, recovery, new workload or timing authorization is provided.
