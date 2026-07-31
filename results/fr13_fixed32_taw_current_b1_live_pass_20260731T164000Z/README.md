# FR13 exact-current TAW B1 live PASS

This artifact packages the real SWE-Verified B1 diagnostic run at
`output/fr13_b1_kernel_live_gate_taw_current_20260731T164000Z`.

## Result

The exact TAW native-precompute source passed its full-graph byte A/B gate on
real task `astropy__astropy-12907`. The immutable `live_pass.json` has SHA-256
`4bfb971f4e9808069d67c4896d9664ecee19542767867157a21f66b0c22f79e5`
and binds:

- executed commit `c8d8bda914af632741d3f2bd9ff0980256b3e897`;
- source contract `fr13-fixed32-taw-exact-commit-v3` /
  `42b92d872d2324bf618b35fdd71c22d0e68e5c00e25ad2a43ae553c8ab1f92da`;
- `hydra27_fixed32`, B1, 32 physical rows, 31 physical drafts;
- one real full-graph replay with zero probability and product mismatches;
- reference returned and candidate not returned during the diagnostic.

The run then completed 968 contiguous B1 events. All 968 work-census rows are
complete, use the exact source contract and diagnostic reference-return route,
and report zero fallback, graph-dead, overflow, mixed-pseudo, or TAW-cache-miss
failures. The runtime log has 968 root-check PASS records, ending at
`root_checks=968`, with zero nonzero mismatch records.

The SWE-Verified task resolved with harness exit 0. Final flush generation 3
reported 968 complete events, forward steps 0 through 967, and zero SFWD,
DFWD, or CFWD pending work. Runtime and external manifests were byte-identical
at launch and end. The launcher recorded `serve_rc=0`, stopped the offload
proxy, ran its attested container-removal teardown, and recovered 105 GiB free
memory.

## Production credential

TAW does not define a separate production-sidecar issuer. Its production
launcher copies the exact live-PASS JSON to the production PASS path, and the
runtime validates that file directly. On artifact base
`d8f25f10032d7769ff024cc1d9cbe5e5e7fdccc4`, the unmodified
`live_pass.json` is accepted by
`_fr13_fixed32_taw_native_production_pass` for `hydra27_fixed32`, B1. It is
rejected for B4 because its coverage is exactly `[1]`. No derived sidecar was
created.

The executed `c8d8bda91` run used an external atomic writer synchronized to
the real task-start log to create the task marker. Its archived 0400 marker is
still present in the source runroot. The artifact base `d8f25f1` adds the
supported task-bracket arm and rotation lifecycle for future gates; this
artifact does not claim that the executed run used that later hook.

## Scope

This is B1 kernel-correctness and source-bound production-selector evidence.
It is not B4 evidence, a production-return timing run, an exact4/exact16
campaign, a TPS measurement, or hardware-floor acceptance. The run is
explicitly classified `b1_diagnostic`, `gate_eligible=false`, and
`floor_acceptance_eligible=false`.

`verification.json` contains the reduced checks and provenance hashes.
`source_evidence.sha256` binds the retained source-run files without copying
the large logs. `SHA256SUMS` covers every file in this artifact except itself.
