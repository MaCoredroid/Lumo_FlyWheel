# B4 M128 block-map install fix

The Tail23 exact4 timing candidate stopped before serving because the pinned
container starts in `/vllm-workspace`, while production-sidecar verification
used the relative host default `scripts/fr13_dvk_subset_blocks.json`.

Commit `5bf4267ace0c2c1234f257f0b9142157d922a23c` passes the already pinned
absolute `FR13_DRAFT_VOCAB_BLOCKS` path through the container launcher and the
candidate installer into B4 sidecar verification. The K64 block map, M128
binary, qualification sidecar, and kernel selector are unchanged.

The failed arm ended before the server became ready and produced no real task
measurements. It is not a candidate timing result. The exact production install
was replayed in the pinned image from its native `/vllm-workspace` directory and
passed. A fresh all-parent exact4 credential is required because the timing
harness commit changed; the prior M128 byte credential remains valid for its
explicit ancestor qualification commit.

## Verification

- 60 focused tests passed.
- Ruff, Python compilation, shell syntax, and `git diff --check` passed.
- Pinned-image production installer replay passed with K64/root1 and Tail23.
- No performance probe, timing claim, or raw task data is included.

