# Fixed32 divisor-balanced CUTLASS B1 K64 gate readiness

Status: the immutable divisor-balanced candidate is wired for the authenticated
one-task SWE-Verified B1 K64/root raw-byte gate. The diagnostic selector is
installable, while the production selector remains blocked until that byte gate
passes. This artifact contains reduced readiness evidence only.

## Candidate identity

- selector: `divisor_static_stocktile`
- diagnostic selector: `divisor_static_stocktile_byte_ab`
- binary SHA256: `338e89d062c2b1ac40909dbc8d64d4ab6b0def9fd86988c9e395e8244606a9f6`
- binary bytes: `113837288`
- source/gate commit: `65725c7504d003ea3778e36e8fd11bbd4f875aac`
- K64 live schema: `fr13.fixed32.cutlass_divisor_static_k64_root_live_gate.v1`
- result name: `cutlass_divisor_static_k64_root_byte_gate.json`

The gate keeps stock serving active, forces eager execution for the diagnostic,
and requires the pinned K64/root block map, all five real projection shapes,
320 comparisons, and exact byte equality. It does not enable production.

## Verification

- all focused CUTLASS tests: 91 passed;
- selected launcher, runner, ingress, attestation, and B1 route tests: 115 passed;
- shell syntax, Python compilation, focused Ruff, and diff checks: passed;
- one unrelated baseline assertion was deselected because it expects an older
  equality-form expression while the unchanged source uses set membership.

No GPU kernel, synthetic probe, Docker workload, or real task was run while
creating this artifact. This is gate readiness, not byte qualification, timing,
or acceptance evidence. B1 remains diagnostic only; acceptance still requires
the canonical exact4 B4 or exact16 real SWE-Verified campaign.
