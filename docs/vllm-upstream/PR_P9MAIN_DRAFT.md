# Own test-only PR (item N) — v1 (Codex GO for submission; AWAITING MARK GO: push p9-main-aligned-state-indices @bc414ae723 to fork, open PR)
> Codex N_review.md: GO on bc414ae723; no blocking code finding; nonblocking nits left unapplied (docstring wording; 'no model weights'; replay description) — applying them would change the verified commit. NO-GO on any new #55506 comment now. CI gate: pre-commit requires verified/ready labels or ≥4 merges; maintainer pickup possible, not promised; no label/CI request.

## Title
[Test] Cover aligned Mamba state-index kernel invariants

## Body

## Purpose

Add four model-free GPU tests for `MambaSpecDecodeGPUContext.compute_aligned_state_indices` in the existing `test_mamba_utils.py`: exact index arithmetic across two groups, `num_reqs` bounds including zero, capture/replay after in-place length updates, and idempotent block-table binding. Synthetic marker tables stay alive while the context retains their raw pointers. No production changes.

These identity-order cases pass both before and after #55506. They do not validate request-slot mapping, its padding clamp, or the runner's capture-time context lifecycle, and take no position on that PR's indexing change.

## Test Plan

Run `python -m pytest tests/v1/worker/test_mamba_utils.py -q`; select `-k TestAlignedStateIndicesKernel` for the four added cases. Check their compute-sanitizer output and repeat with arithmetic, store-mask and initialization-guard mutations.

## Test Result

GB10/sm_121, source base `379e9a1ea8`, test commit `bc414ae723`: 4 new tests and 47 whole-file tests passed; the same counts passed after locally applying #55506 through `5f71d6f`. Compute-sanitizer reported 0 errors for the new cases. All three mutations were detected; the arithmetic mutation fails two cases. Ruff 0.14.0 passed; full pre-commit was not run.

Python/Triton sources came from the tested checkout; imported native extensions were reused from `23ab0cfdb`. No model, KDA consumer, PP/async-serving or multi-program (>32-row) validation is claimed.

AI assistance was used for test development and review.
