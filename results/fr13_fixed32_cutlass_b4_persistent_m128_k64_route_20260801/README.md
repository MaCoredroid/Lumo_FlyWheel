# Persistent M128 B4 K64-root route

This is a prepared route, not timing or acceptance evidence. It qualifies and
screens the existing persistent-M128 CUTLASS SFWD binary on canonical real
SWE-Verified exact4 B4 traffic under the deployment K64-root workload.

## First live attempt

The 2026-08-01 exact4 attempt is recorded in
`rejected_live_gate_20260801.json`. The persistent-M128 comparator was exact on
all 320 bounded real-event invocations and all five projection shapes, covering
1,436,811,264 output bytes with zero differing bytes. The diagnostic continued
to serve the stock result.

The attempt is rejected. One completed remote Qwen trace was truncated at the
SSH capture boundary: it ended without a newline and its final JSONL record was
incomplete. Campaign provenance therefore remained unpublished, no live PASS
or production credential was issued, and the timing stage did not start. The
comparator result is kernel-equivalence evidence only; it is not performance or
acceptance evidence.

## Bound workload

- topology: `hydra27_fixed32`
- batch size and concurrency: `4`
- physical SFWD rows: `128`
- draft vocabulary: root enabled, `K=65536`
- block map: `/workspace/scripts/fr13_dvk_subset_blocks.json`
- block-map SHA-256: `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`
- target verifier vocabulary: full
- mandatory weight bytes: `32666638208`
- optimistic weight floor: `119.658015414 ms/step`
- one-sided 1.15x cap: `137.6067177261 ms/step`

The live diagnostic serves stock results, performs at most 320 exact byte
comparisons on real K64 traffic, and is neither timing nor acceptance evidence.
The timing stage runs the stock arm once and the persistent-M128 arm once in the
same exact4 campaign. Do not run a separate K64 baseline before this pair.

## Pinned binaries

- persistent-M128 candidate SHA-256:
  `895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f`
  (`112698512` bytes)
- stock FA2 SHA-256:
  `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`
  (`299183936` bytes)
- canonical exact4 subset SHA-256:
  `0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5`

## Run

From a clean `agent/fixed32-k64-m128-b4` worktree with no Docker containers:

```bash
bash results/fr13_fixed32_cutlass_b4_persistent_m128_k64_route_20260801/prepared_campaign.sh
```

The script pins the live PASS SHA and qualification source commit into the
candidate timing arm. The final `timing_summary.json` reports full-wall step
latency and TPS, acceptance/commit rate, and reconciled
SFWD/DFWD/CFWD/other-wall time for both arms. SFWD is converted from the
measurement harness's seconds-per-forward field to milliseconds-per-step.
It remains an exact4 candidate screen, not the formal exact16/U95 acceptance
gate.

## Static verification

- focused K64 credential, production, ingress, and CUTLASS suite: `58 passed`
- broader CUTLASS, floor, subset, manifest, process, and measurement suite:
  `141 passed`
- `bash -n`: pass
- Python byte compilation: pass
- Ruff: pass
