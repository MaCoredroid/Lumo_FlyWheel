# SFWD real-B1 padded-input-stride rejection

This artifact records the authenticated real SWE-Verified B1 SFWD byte gate
launched at `2026-08-01T11:34:06Z`.

- Source commit: `62db58a7a31153cf4240c158e8dba17c9e9b12a2`
- Task: `astropy__astropy-12907`
- Run root:
  `output/fr13_b1_sfwd_state_fusion_live_gate_20260801T113406Z`
- Wrapper result: exit code 15
- First real decode: rejected before the candidate/reference comparator

The detailed fail-closed guard isolated one failed predicate:
`x_contiguous`. The live pre-conv input is a valid row-padded view with shape
`[32, 10240]` and element strides `[16384, 1]`. The candidate incorrectly
indexed each row using the logical channel count `10240`.

All other observed operands matched the fixed32 contract:

- output: shape `[32, 10240]`, strides `[10240, 1]`, BF16
- conv state: shape `[665, 10240, 34]`, strides
  `[2097152, 1, 10240]`, BF16
- conv weights: shape `[10240, 4]`, strides `[4, 1]`, BF16
- state indices: shape `[1, 32]`, strides `[32, 1]`, INT32
- source descriptor: shape `[128]`, stride `[1]`, INT64
- commit source stage: shape `[36, 10240]`, strides `[10240, 1]`, BF16

There are no byte-comparison, timing, production, or acceptance results from
this run. No candidate output was served.

