# FR13 TAW B1 diagnostic byte-equivalence PASS

This artifact packages the completed real SWE-Verified B1 TAW-only run at
`output/fr13_b1_kernel_live_gate_taw_only_20260731T160322Z`.

The diagnostic verdict is PASS: 762 complete Hydra27 B1 events used the
native-precompute byte-A/B reference-return route, all 762 work-census rows
bound the executed source contract `fe73ad35...`, five periodic PASS markers
reported zero probability and product mismatches, and no TAW mismatch marker
occurred. The single real SWE-Verified task `astropy__astropy-12907` resolved.

## Production status

This is not a production-arming PASS.

The run executed commit `850355982` with source contract
`fe73ad35a916e41532575e29a5f9f6442d1081d0d1c0d0fc18210fdc8f0f56f8`.
The current production selector requires the exact all-parent-v7 payload
schema and source contract
`694a3f4cd6e36ff1b6503ff19b2968b94a1ac226535a6efb44dcea1bb8a9a57b`.
It also adds an in-runtime real-task arm and atomic live-PASS emitter that the
older run source did not contain.

`diagnostic_pass.json` uses a dedicated diagnostic schema, deliberately
retains the executed `fe73ad35...` hash, sets `production_eligible=false`, and
is rejected by the current production validator on both schema and source.
No validator relaxation or compatibility exception is added. An
exact-current-source real gate is still required before TAW production can
arm.

## Source equivalence

`source_equivalence.json` compares normalized ASTs between the run source and
the current physical-slot integration. The projection now differs because the
served all-parent path and executed physical committer have changed. The old
diagnostic remains valid evidence for its recorded source only; it is neither
math-equivalent to nor production-informative for the current candidate.

## Exact evidence

- Run source: `850355982cf747d2a960e7dae5f769edb660d772`
- Run source file SHA-256: `8a870b9f432074cbf851a1d457de5fa6e0987e56190b972931d9cda2694fdb73`
- Docker log SHA-256: `3ce52d60f0c19b8463fd870bed35148e724fdf194d2bdf0cf8c7e78573a159b9`
- Work-census SHA-256: `9f9028696be6dc50ffe517962675c6a026296541b052f1850df7eaf40856c834`
- Work-census terminal event digest:
  `e5423fbf0475713abe2a698ff3eaef27ca81dc14665e95488024153478eef9f1`
- Periodic PASS markers: 5 at root checks 128, 256, 384, 512, and 640
- TAW mismatch markers: 0
- Current-schema runtime live-PASS markers: 0
- Task verdict: resolved, harness exit 0, tests passed
- Serve return code: 0

This artifact contains no new GPU execution, performance, acceptance, TPS, or
hardware-floor result. B1 diagnostics are not floor-acceptance eligible.
