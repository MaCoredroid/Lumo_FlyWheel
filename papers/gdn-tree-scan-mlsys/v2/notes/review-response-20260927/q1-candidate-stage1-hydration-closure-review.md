# Candidate stage 1: hydration closure review

Reviewed `p0/monitor/review-response-20260927/candidate-stage1-v1-reviewed`, with parent snapshot timestamp 2026-09-28 04:38:28 UTC. All 49 local files match the parent's observed hashes; 46 match the executor freeze. The three known drifts are `launch_env`, `launch_env_renderer`, and `tests_hooks`. The two implementation files reviewed here **match the freeze**:

- `tools/q1_candidate_hooks_v1.py`: `1ead3b4d977df7640b238e78fc4dd2e07cbe5f26e8d4823e4bc68cebef2f4b93`
- `tools/q1_candidate_hydration_v2.py`: `136e1cf36992958a9e88f52efd3c3d16d69af780481533cb201f7c3b006349cb`

The parent snapshot binds freeze SHA-256 `dfc06e952109796db5efe994b65daa89f781e4cd89bac2c7d70d769832ae9b76`. This review covers only the prior hydration/identity/fail-stop findings and relevant hook behavior. It is not whole-package freeze acceptance or gate approval. No Torch import, GPU/container/SSH operation, source edit or gate edit occurred.

## Closed findings

**Fail-stop now closes the three reproduced error paths.** `apply_all` latches `attempted=True` before its first copy statement (`hydration:168–178`) and propagates that status on exceptions (`204–207`). The caller receives an in-memory transaction latch before copying (`hooks:288–289`). The S0 transaction includes post-import O0 snapshot/digest verification; errors after mutation latch the process unusable and raise in the current pre-forward (`317–338`). `_mark_unusable` sets its in-memory latch before marker/receipt I/O, with both writes best effort (`224–237`).

AST-only CPU controls using standard-library stubs passed: a first copy that mutates then raises reports stage `copy` and unusable; normal S0 returns without a failure latch; a post-import snapshot exception raises `ProcessUnusable`; the same exception still raises and latches when both marker and receipt writes fail. No inference about CUDA execution is made from these controls.

**Alias planning and final readback repairs remain present.** The planner now groups exact bank data pointers, checks matching conv/SSM alias partitions, requires each layer's conv/SSM shared base and consistent alias-class view geometry, and rejects colliding selected rows (`hydration:63–94`). Source preauthentication precedes mutation; the global live-region reread and complete native-order digest remain enforced (`129–207`). The explicitly bounded complementary guard scope remains unchanged.

**Tuple/signature observation is repaired.** The actual runner tensor signature now includes object ID, pointer, storage base/size, shape, strides, offset, dtype and device (`hooks:122–125`). Required bank tuples and their IDs/member signatures are recorded (`135–143`). A standard-library control confirmed that replacing a bank tuple with a new tuple containing the same tensors changes the snapshot. Normal completion now calls `_seal_case` at the second TAW event (`381–383`); the relevant lifecycle tests also assert that path, although this snapshot's hooks test file is one of the known freeze drifts and was not executed here.

## One remaining narrow identity repair

`_identity_snapshot` currently calls the layer-to-preseed relationship valid when the two views share only their **storage base** (`hooks:144–148`). This accepts different regions of one allocation. Reproduced with the exact AST method: runner SSM data pointer 2000 and preseed bank pointer 2100, both with base 1000, yield `_layer_bank_relationship = {'L': [True, True]}`. The full signatures are recorded but are only compared before/after; a consistently wrong preexisting relationship does not change during import.

This does **not** show that the present production banks are wrong. It shows the new attestation does not establish the relationship it claims: final O0 readback checks the runner view, while replay/commit consumes the bank view. Bind each bank using a validated complete, unique `fixed32_order` name-to-index mapping. Require the SSM bank's address, shape, strides and offset to describe the runner SSM view, and require the convolution bank to describe the expected transpose of the runner SD convolution view. Compare expected view geometry/address rather than demanding Python-object identity for a transpose. The already accepted production binding is SSM directly and conv through its DS transpose; do not change that layout.

A focused same-base/different-offset negative control plus the existing valid binding control is sufficient. No additional experiment or numerical-policy change is requested. **Verdict: fail-stop and tuple-replacement closures pass; exact layer-to-bank view binding remains to fix.** Package-wide approval also remains separate because of the three documented freeze drifts.
