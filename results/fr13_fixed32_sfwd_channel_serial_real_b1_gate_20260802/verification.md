# Verification

- The terminal gate validator returned `status=pass` and the runner exited 0.
- One authenticated real SWE-Verified B1 task completed under K64/root1.
- All 35,808 records passed; zero records mismatched.
- All 48 unique layers were covered.
- Both `conv_out` and `commit_source_stage` were byte-equal on every record.
- The final task boundary, engine ledger, traffic ledger, runtime/external
  manifests, process attestation, and terminal snapshot passed validation.
- The source manifests captured at launch and end are byte-identical.
- The incumbent reference was served and no fallback engaged.
- Post-run Docker and GPU compute-owner counts were both zero.
- `133 passed` in the focused SFWD/ingress/wall-timer/preseed test suite before
  launch; Python compilation, shell syntax, and `git diff --check` passed.
- Offline SM121a codegen for the repaired source reports 66 registers and zero
  shared memory, barriers, spills, local memory, stack, or calls for B1/B4.

The gate explicitly marks timing, floor acceptance, and production eligibility
false. It contains no performance result.
