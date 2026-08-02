# SFWD real-B1 geometry rejection

This artifact records the failed authenticated real SWE-Verified B1 SFWD byte
gate launched at `2026-08-01T11:00:08Z`.

- Source commit: `443344789dd4df0bda671832c1badd4f1bd04d4d`
- Task: `astropy__astropy-12907`
- Run root:
  `output/fr13_b1_sfwd_state_fusion_live_gate_20260801T110008Z`
- Wrapper result: `FR13_SFWD_GATE_RC=15`
- First real decode: rejected before the candidate/reference comparator
- Runtime geometry: fixed32 tree rows 32, conv width 4, conv state length 34
- Rejected source contract: conv state length 12

The runtime boot needle reported
`FR13_TREE_CONV_FUSED ... tree_n=32 width=4 state_len=34`. The first
authenticated decode then failed closed with
`FR13_FIXED32_SFWD_STATE_FUSION dependency/geometry contract drifted`.
There are no byte-comparison, timing, production, or acceptance results from
this run.

The stopped container was removed after its host-mounted evidence and Docker
state were inspected. No candidate output was served.
