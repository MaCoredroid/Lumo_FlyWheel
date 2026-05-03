# L0c CUTLASS Agent Guidance Learnings

Date: 2026-05-03

Scope: CUTLASS-only FP8 GEMM auto-research.

## What Changed

Agents now receive a local editable CUTLASS source copy under `cutlass_source_workspace`. The controller stages this workspace from the live vLLM container, keeps an immutable `cutlass_source_base`, and owns apply/restart/parity/measurement.

Agent-side checks are cheap:

- `patch --dry-run -p0 < candidates/<NNN>/mutation.patch`
- `cd cutlass_source_workspace && python3 -m py_compile $(find . -name '*.py' -print)`
- `lumoserve auto-research preflight-patch ...`

Agents must not run `apply-and-test`.

## What Failed This Round

The orchestration worked, but the proposer produced candidates that were either too small to matter or violated vLLM CUTLASS semantics:

- A reshape fast path failed downstream logit parity.
- A contiguity-only `process_weights_after_loading` override discarded important CUTLASS loading behavior.
- A control-flow cleanup failed parity and had no strong performance mechanism.

## Next Prompt Guidance

Give agents this explicit bar before allowing a patch:

1. State the expected speed mechanism in one sentence.
2. Preserve all existing CUTLASS loading semantics unless the mutation is specifically about those semantics.
3. Avoid reshape-only, variable-cache-only, and cosmetic control-flow mutations.
4. Do not replace `process_weights_after_loading` wholesale.
5. Prefer guarded fast paths with exact predicates over broad behavior changes.
6. Read `mutations_rejected.tsv` and avoid repeating rejected mutation families.

## Next Loop Recommendation

Run one larger-budget loop after adding the guidance above to the candidate brief. A larger attempt cap is useful only if the proposer is steered away from wrapper no-ops and toward changes that can affect CUTLASS kernel dispatch, scale handling, shape gating, or schedule selection.
