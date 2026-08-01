# K64 physical-32 B1 full-stack route

Status: prepared and statically verified. No GPU campaign was run from this
branch, so this directory contains no timing or acceptance measurements.

The immutable implementation commit is
`402481c24a8872f3ddc65859ca7da93f69b3eef9`. It is based on the reviewed B1
full-stack route `c3ee2fece6daa17927ec216ff0135c5cf3ebb1e0`. The route consumes only the
corrected TAW B4 review contract from
`e0ac403c22525265525957ff15e118ca291e68fa` (code fix
`6ed4a55df803c5a7b9190e9c0de0498085a9b9d0`). It rejects the pre-review B4
bundle format.

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
   arms consume the same source-bound GDN PASS and enable the same GDN candidate.
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
