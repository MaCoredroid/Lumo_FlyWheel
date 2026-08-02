# Fixed32 B4 FA2 qrow32 source kernel

Status: **default off; source/static verified; pinned-image compile, resource
inspection, exact4 real-byte qualification, and timing pending**.

## Change

The exact-safe B4 tree-attention route currently launches one FA2 `BM64/BN64`
four-warp CTA for each batch/query-head pair. Each fixed32 batch slot contains
exactly 32 root-inclusive query rows, so half of every stock query tile is
outside the physical query extent.

This source candidate adds a private `BM32/BN64`, two-warp specialization.
It preserves one CTA per batch/query-head pair, one complete ordered K loop per
real query row, `num_splits=1` semantics, and no combine kernel. Tail23 and
Hydra27 therefore retain the same physical32 launch geometry. For rows 0..31,
the warp index and warp-local row coordinate match stock BM64 exactly.

The candidate is emitted as a dedicated hidden CUDA translation unit selected
only by `--fixed32-query-tile32`. There is intentionally no runtime selector;
the production route cannot call it before binary-resource and real-byte
qualification succeed.

## Static work model

These are launch/trait counts, not performance measurements.

| B4 fixed32 tree attention | Stock BM64 | Candidate BM32 |
| --- | ---: | ---: |
| CTAs/layer | 96 | 96 |
| Threads/CTA | 128 | 64 |
| Threads/layer | 12,288 | 6,144 |
| Query rows/tile | 64 | 32 |
| Out-of-extent query rows/tile | 32 | 0 |
| Dynamic shared bytes/CTA | 98,304 | 81,920 |
| K/V passes per batch/head | 1 | 1 |

Across the 16 target attention layers, CTA count stays 1,536 while launched
threads fall from 196,608 to 98,304. The candidate changes the cooperative
paged K/V copy layout from four to eight rows per thread; it does not change
the ordered per-row attention reduction.

No speedup or byte equivalence is claimed. The private symbol must compile for
the pinned SM121 image with zero stack/local/spill traffic, preserve ABI/ELF
parity, and pass a same-EngineCore raw-byte comparison on retained live paged
operands for the canonical real SWE-Verified exact4 Tail23 and Hydra27 sets.

## Static verification

- FA2 candidate source tests: `6 passed`.
- Python byte compilation: pass.
- `git diff --check`: pass.
- No GPU command or synthetic performance probe was run.

This directory contains source-level aggregate metadata only. It contains no
prompts, responses, patches, traces, raw logs, process/container identities,
credentials, or timing samples.
