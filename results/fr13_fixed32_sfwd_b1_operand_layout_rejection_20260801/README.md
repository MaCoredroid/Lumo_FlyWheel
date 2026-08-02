# SFWD real-B1 operand-layout rejection

This artifact records the failed authenticated real SWE-Verified B1 SFWD byte
gate launched at `2026-08-01T11:16:54Z`.

- Source commit: `a8ad851a7aa7a458c7d5cf3036279e89745465a4`
- Task: `astropy__astropy-12907`
- Run root:
  `output/fr13_b1_sfwd_state_fusion_live_gate_20260801T111654Z`
- Wrapper result: exit code 15
- First real decode: rejected before the candidate/reference comparator
- Confirmed runtime geometry: fixed32 tree rows 32, conv width 4, conv state
  length 34

The live boot completed and reported
`FR13_TREE_CONV_FUSED ... tree_n=32 width=4 state_len=34`. The first
authenticated decode entered `launch_fixed32_sfwd_state_fusion` and then
failed closed with
`FR13_FIXED32_SFWD_STATE_FUSION operand geometry/layout drift`.

This is a second, distinct rejection after the state-length fix. The aggregate
guard did not identify which shape, dtype, or contiguity predicate failed.
There are no byte-comparison, timing, production, or acceptance results from
this run. No candidate output was served.

