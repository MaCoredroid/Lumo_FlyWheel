# Fixed32 GDN path-BV live gate

Source-only candidate artifact based on `e5a4029249e5b9483919056bae6542e47bbce75b`.

The fixed32 FULL serving graph remains pinned to `BLOCK_V=8`. Setting
`FR13_FIXED32_GDN_PATH_BV_CANDIDATE` to `16`, `32`, `64`, or `128` arms a
one-shot gate on the first measured real SWE graph replay. For every captured
fixed32 GDN call, the gate runs an explicit BV8 reference and then the selected
candidate on the same persistent tensors. It compares raw bytes for output,
export, all four replay rings, flags, and a counter, then restores the bytes
produced by the served BV8 graph.

The gate requires exactly `48 * batch_size` captured calls and binds them to the
signed FULL graph identity. It rejects a candidate launch that reports BV8 or
the same launch key as the reference. Triton compilation, launch, or resource
failure raises and leaves the candidate unserved.

BV64 and BV128 reduce the V-grid to two and one CTAs per value head. Their
logical fp32 state tiles are 32 KiB and 64 KiB respectively for `DIM_K=128`;
this is not a claim about compiled register allocation or occupancy. Those
properties must pass on the actual target GPU during the real replay gate.

No GPU benchmark, synthetic performance probe, SWE task, throughput result, or
acceptance claim was produced by this source-only artifact.

Validation completed:

- `9 passed` in the dedicated CPU/static live-gate suite.
- `58 passed, 1 skipped` across the live gate plus existing fixed32 GDN exact
  I/O, schedule, final-capture, and observer-lifecycle suites. The skip is the
  existing CUDA-only exact-I/O test.
- Python compilation, shell syntax, and `git diff --check` passed.
