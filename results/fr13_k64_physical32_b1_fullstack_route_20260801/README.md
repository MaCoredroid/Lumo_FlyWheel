# K64 physical-32 B1 full-stack route

Status: prepared and statically verified. No GPU campaign was run from this
branch, so this directory contains no timing or acceptance measurements.

The immutable implementation commit is
`7064f094e43c6b1f14ac358f31bfc8858de3b136`. It is based on the reviewed B1
full-stack route `c3ee2fece6daa17927ec216ff0135c5cf3ebb1e0`. The route consumes only the
corrected TAW B4 review contract from the current route tip
`080c417ed627e155c98e715327e0fdeb48d542ab` (terminal-census code fix
`9f30d84dc68f97bfd871862db829b7048e921847`, source commit
`07a0d0e4613bb4a7ef750120700ba9a2387f58d7`, phase-math fix
`6ed4a55df803c5a7b9190e9c0de0498085a9b9d0`, and direct-file trace integration
`4d0c57617e6a3675dddf8f76ecbee376b710220e`). It rejects the pre-review B4
bundle format.

The B4 attempt launched from `68336f72ada43aa1e9681329e58dc031d2a69491`
is classified `aborted_known_invalid_source`: its timing reducer treated the
mandatory v9 terminal census record as an event. Only the Tail23 all-parent
stage launched, and the operator stopped it before any formal production pass,
byte verdict, M128 gate, timing summary, TPS, acceptance, or hardware-floor
result. That reduced abort metadata is not a B1 prerequisite and cannot be
substituted for the corrected Tail23/Hydra27 pass and verdict inputs.

The fixed32 remote-agent trace route includes direct in-container Qwen capture
from `31de2814e10c1097bb5e6c18a32378e27b14cd47`, integrated here as
`8754b8c77db2872ab22b42c29de596ab9d1a3c10`. Qwen writes directly to the
bind-mounted regular trace file; Docker stdout is not a second writer. The
observer attests and strictly validates the exact pulled JSONL before a
campaign can finalize, closing the prior 258048-byte pipe-drain cutoff.

The GDN coefficient implementation/review lineage is
`1ca12bc2611d87432d36caa563efa5c8d795942b` then
`f5ccbdfdd1b7244cb551bca69d1ff099b9ab2c70`. The reviewed offline SM121
artifact records `COUNT_INVOCATION=False` and zero stack bytes, local bytes,
and spill instructions for all eight B1/B4 stock/candidate level
specializations. That artifact is static evidence only: it is not a live B4
qualification and does not make the GDN candidate B4-deployable.

## Fixed configuration

- K64 drafter head with ROOT=1 and the pinned block map.
- 32 physical rows, root inclusive.
- Tail23: 23 logical drafts, mask `0x7a9ce7ff`.
- Hydra27: 27 logical drafts, mask `0x7abdffff`.
- B1, concurrency 1, real SWE-bench Verified traffic.
- qrow16, SFWD state-fusion, and GDN level-0 coefficient production in both
  timing arms.
- GDN production uses `COUNT_INVOCATION=False` and the exact BV8 candidate in
  both arms.
- Both arms must emit a source/PASS-bound
  `fixed32_gdn_level0_coeff_production` engagement sidecar. The reducer validates
  its route, candidate, mode, graph identity, physical geometry, and no-fallback
  contract, then hash-binds the exact sidecar into the timing summary.
- The only timing-arm delta is source-v7 all-parent committer production.

## Campaign sequence

1. Validate each reviewed B4 production bundle plus its corrected exact4 gate
   verdict.
2. Run a separate one-real-task TAW B1 full-graph byte gate for Tail23 and
   Hydra27. The reference is served and the candidate remains shadow-only.
3. Bind each fresh, mode-specific B1 credential to its reviewed B4 verdict and
   preserve the reviewed B2/B3/B4 records.
4. Before timing, run a separate one-real-task, mode-specific GDN B1 byte gate.
   It compares 48 records and `4,725,178,944` raw bytes across output, all 31
   non-scratch export rows, K/V/A/B rings, flags, and counter; scratch row 31 is
   contained and the served state is restored.
5. Run stock then candidate on the canonical exact4 set for each mode. Both
   arms consume the same source-bound GDN PASS, enable the same GDN candidate,
   and must prove production engagement.
6. Validate every work-census event and bind the census hash/count to the
   authenticated traffic audit before reducing the timing pair.

The output reports full-step wall milliseconds and TPS, accepted and committed
tokens per event, SFWD/DFWD/CFWD milliseconds, residual wall overhead, and the
corrected `119.658015414 ms` mandatory-weight floor ratio. SFWD is reduced from
`s_per_fwd_gpu_per_forward`; B1 requires exactly one event per physical step.

This exact4 run is a screening measurement. It is not the formal hardware-floor
acceptance gate; exact16 and a one-sided U95 at or below `1.15x` the corrected
floor remain required.

Run from a clean launch worktree with a project Python environment, no existing
Docker containers, the two corrected TAW B4 pass/verdict pairs, and the pinned
stock/qrow16 binaries. The route performs fresh GPU/Docker GDN and TAW B1 gates;
none was run while preparing this artifact.

```bash
TAG=<unique-tag> \
STOCK_FA2_SO=<absolute-stock-so> \
QROW16_FA2_SO=<absolute-qrow16-so> \
TAIL23_REVIEWED_B4_TAW_PASS=<absolute-tail-pass> \
TAIL23_REVIEWED_B4_TAW_VERDICT=<absolute-tail-verdict> \
HYDRA27_REVIEWED_B4_TAW_PASS=<absolute-hydra-pass> \
HYDRA27_REVIEWED_B4_TAW_VERDICT=<absolute-hydra-verdict> \
bash results/fr13_k64_physical32_b1_fullstack_route_20260801/prepared_campaign.sh
```
