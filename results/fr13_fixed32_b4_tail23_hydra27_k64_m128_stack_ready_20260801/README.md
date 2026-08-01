# B4 Tail23/Hydra27 K64 all-parent + persistent-M128 route

This artifact is static readiness evidence. It contains no GPU timing,
acceptance, TPS, or hardware-floor result.

## Bound stack

- source branch: `agent/fixed32-b4-tail23-hydra27-k64-m128`
- source commit: `ccec920494cad2b356a53d2c01b75be30cd12fd8`
- reviewed K64 persistent-M128 base: `531da75343987948dd844ce22a23481355a80674`
- fixed32 node32/all-parent source: `d383ec46d03a08b5138b86471aecec1199a14ae3`
- exact task set: canonical real SWE-Verified exact4
- batch size and concurrency: `4`
- draft vocabulary: root enabled, `K=65536`
- target verifier vocabulary: full
- block-map SHA-256: `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`
- physical rows per request: `32` (`1` root + `31` draft slots)
- B4 SFWD projection rows: `128`
- Tail23: mode `tail6_fixed32`, active drafts `23`, mask `0x7a9ce7ff`
- Hydra27: mode `hydra27_fixed32`, active drafts `27`, mask `0x7abdffff`
- mandatory weight bytes: `32666638208`
- optimistic floor: `119.658015414 ms/step`
- one-sided 1.15x cap: `137.6067177261 ms/step`

## Launch

Run only after the active B4 campaign has fully torn down:

```bash
cd /home/mark/lumoFlyWheel-b4-tail23-hydra27-k64-m128
bash scripts/fr13_run_b4_tail23_hydra27_k64_m128_stack.sh
```

The fail-closed order is:

1. Tail23 all-parent shadow byte gate on exact4.
2. Tail23 persistent-M128 shadow byte gate on exact4.
3. Tail23 paired full-step timing: stock CUTLASS vs persistent-M128, with the
   qualified all-parent committer enabled in both arms.
4. Hydra27 all-parent shadow byte gate on exact4.
5. Hydra27 persistent-M128 shadow byte gate on exact4.
6. Hydra27 paired full-step timing under the same common-committer contract.

The two timing arms per topology differ only in stock CUTLASS versus
persistent-M128. The final `paired_summary.json` reports accepted drafts/event,
committed tokens/event, full-wall TPS, wall milliseconds/step,
SFWD/DFWD/CFWD/other-wall milliseconds/step, and floor ratio for both trees.
Exact4 is a tuning screen, not the formal exact16/U95 acceptance gate.

## Static verification

- focused topology/credential/attestation suite: `94 passed`
- broader CUTLASS, all-parent, provenance, floor, ingress suite:
  `345 passed, 1 skipped`
- shell syntax and embedded Python compilation: pass
- Ruff and Python byte compilation: pass
- work-census and depth-acceptance self-tests: pass
- runtime-manifest preflight: pass, canonical SHA-256
  `3cb8cd901f65372e421e8747f62b339403a530345e8a2e716f43cda1fc46e989`
- external-manifest preflight: pass, canonical SHA-256
  `c42fb16d2ea932f0e819c81172c49b69b312b75dd446146b6d7b6ef78af00a9e`

No Docker container, GPU kernel, SWE-Verified task, timing arm, or synthetic
probe was launched while preparing this artifact.
