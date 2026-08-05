# Fixed32 DFWD K64 M1 exact-order R32 checkpoint

Status: **default off, fail closed, source and runtime wiring complete; no new
GPU execution, byte qualification, timing, or production admission**.

This checkpoint ports the exact-order BF16 K64 B1 drafter-head kernel onto
pushed `main` base `4dc01e59f1c29e57192ea2e0341c4b18b95a8714` without changing
the active six-way stack. The candidate maps 32 output rows to each 512-thread
CTA and launches 2,048 CTAs. Each row retains the incumbent-aligned 16-lane K
partition, 320 dependent scalar FP32 FMAs per lane, width-16 `8+4+2+1` FP32
shuffle-add tree, alpha-one/beta-positive-zero epilogue, and BF16 rounding.

The selector `FR13_DRAFT_HEAD_M1_R32=1` is accepted only for B1 fixed32
Tail23/Hydra27, physical32, K64/root1, the pinned gathered-vocabulary map,
single logits, and FULL graphs. The launcher requires the immutable 113,648
byte shared object with SHA-256
`c389bf5e01b942cfe73b2e4fc05db7b158f16b61205c9f3e9988cbd8a82474dd`,
mounts it read only, prepares it before capture, and treats any setup or runtime
error as fatal. It remains mutually exclusive with all other kernel candidates
and has no production credential path.

## Why this target

Historical real SWE-Verified B1 attribution measured the five BF16 K64 heads
at `26.227316014 ms/event` against a `12.291000733 ms/event` mandatory-weight
floor, leaving `13.936315281 ms/event` excess. The later acceptance-valid Hydra
anchor measured aggregate DFWD at `36.813368134 ms/event`. The candidate keeps
mandatory weight traffic unchanged, so it targets scheduling, barriers, and
reduction overhead only. No latency improvement is claimed.

The rejected pair8 diagnostic is not reused as a speed claim: it reduced its
single-task wall latency but regressed accepted drafts from `4.89` to `3.65`
per event and was dominated in wall TPS. Exact-order R32 is selected to avoid
that arithmetic-partition quality change, subject to a future byte gate.

## Static evidence

The current CUDA and builder bytes exactly match the prior no-GPU pinned build.
The historical binary was re-read from Git and audited without GPU exposure:
one SM121a cubin, 18 registers/thread, zero stack/local/shared bytes, four SHFL,
four FADD, and no BAR, LDL, STL, or CALL. Its size and SHA-256 match the new
launcher pins.

Focused source/runtime and adjacent-stack tests passed. Python compilation,
shell syntax, and `git diff --check` passed. The current host Python has PyTorch
2.4.1 rather than the pinned 2.11.0+cu130 build toolchain, so this checkpoint
uses the byte-identical historical build instead of starting a competing
container while the live Gate A preflight is running.

## Required qualification

1. Add or run an authenticated stock-serving real SWE-Verified B1 byte gate
   for root and all four MTP head sites. Compare all 65,536 BF16 logits bitwise
   at every site and require zero mismatch.
2. Only after byte PASS, add a separate source-bound production credential and
   compose R32 with mapped K64 top3.
3. Run the standing exact-four real-task timing with DFWD, SFWD, CFWD, full wall,
   acceptance, and TPS. Exact16 remains conditional on the floor gate.

This package contains reduced source hashes, static build facts, and test
summaries only. It excludes binaries, cubins, PTX, SASS, compiler caches, raw
tasks, prompts, responses, patches, logs, credentials, secrets, process IDs,
and container IDs.
