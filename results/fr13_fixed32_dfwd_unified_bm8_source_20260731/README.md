# FR13 fixed32 DFWD unified-attention BM8 candidate

This artifact publishes a default-off exact-math candidate for the largest
uncovered DFWD kernel group. It does not publish a GPU speed result or a byte
parity result.

The current valid Hydra27 SWE-Verified exact4 arm measures 38.1293 ms/event in
DFWD and 20.6866 ms/event in CFWD. The provenance-bound real-SWE Nsight capture
attributes 6.9676 ms/event to four `kernel_unified_attention_2d` calls. The
larger groups are already covered by the BF16 head-padding candidate or are the
mandatory MTP FP8 GEMMs. The remaining named CFWD groups are already covered by
TAW/GDN work or touch sampling distribution.

For B1 MTP, stock uses `BLOCK_M=16`, `BLOCK_Q=2` for 24 query heads and four KV
heads. Each KV CTA has six live query-head lanes and ten masked lanes. The BM8
candidate keeps those same six live lanes and their exact K/V tile traversal,
while reducing masked lanes from ten to two. It also fixes the exact candidate
grid at one q-block so the smaller `BLOCK_Q` does not add empty CTAs.

`FR13_DFWD_UNIFIED_BM8_LIVE_AB=1` is diagnostic only. During the four-call
drafter graph capture, it records per-call copies of the live query and sequence
length because the model reuses intermediate buffers across iterations. After
the first measured real SWE-Verified replay, the stock graph result has already
been served. The gate recalls stock BM16 and candidate BM8 into private outputs,
compares every output byte for all four calls, writes an atomic sidecar, and
requires four candidate-dispatch counter increments before it can pass. It
writes an atomic sidecar and raises on compile failure, false stock-vs-stock, or
mismatch. The internal candidate selector cannot be supplied through the
launcher.

The optimistic group ceiling is 6.9676 ms/event. Even deleting the entire group
would leave the current GPU component at about 215.59 ms/event, 1.802x the
corrected 119.658 ms floor, so this candidate is useful but not sufficient for
the 1.15x goal.

Verification performed without GPU execution:

- emitted patched vLLM source compiles as Python;
- launcher passes `bash -n`;
- 49 focused ownership, launcher, FA2, sidecar, and BM8 tests pass.
