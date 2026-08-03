# Fixed32 K64 FP8 draft-head plus mapped top3 source candidate

Status: `SOURCE_READY_UNBUILT`

This artifact contains a default-off, source-only SM121a candidate for the
physical32 K64 drafter head. It consumes the existing block-FP8 activation,
weight, and scale layouts; computes the 65,536 draft scores; rounds each score
to BF16; and retains only rowwise top3 partials. A second kernel merges 512
partials and writes mapped spine ID, mapped top3 IDs, and BF16 top3 scores.

There are no runtime, launcher, or main-branch changes in this candidate. It has
not run a GPU equality gate, a real SWE-Verified task, or a timing campaign.

## Consumer proof

At base commit `a4b32a0df8128cd3fef99146253a8ff46134f507`, exact fixed32
routes through `_fr10_is_wide`. That route empties `_fr10_leaf_steps`, does not
set `_fr10_consumes_root_leaf`, and admits the current reducer only when all five
depth widths are `(3,3,3,3,3)`. Root and four loop sites then source the spine
and siblings exclusively from mapped top3 results. No full-logit consumer
remains inside that exact guard.

The selector order matches the existing K64 mapped-top3 source: NaNs first,
then descending BF16 score, then lower K64 subset index. The ID map is applied
only after subset order is final, so arbitrary map order cannot change ties.

## Kernel geometry

- Stage 1: 512 blocks x 256 threads, one 128-vocabulary-row block per FP8
  weight-scale row.
- Stage 1 B4: each qweight byte is loaded once and reused across all four
  activation rows. There is no batch grid dimension.
- Workspace: BF16 values plus int32 subset IDs, `[B,512,3]` each.
- Stage 2: B blocks x 256 threads, one exact rowwise mapped top3 reduction.
- Full BF16 `[B,65536]` logits are never written.

CUDA 13.0 `nvcc` compiled the translation unit for `sm_121a` using host Torch
2.10 headers as a syntax/resource check. This is not the pinned Torch 2.11
build. The resulting object was not retained.

| Kernel | Registers/thread | Static shared bytes | Stack/local bytes |
| --- | ---: | ---: | ---: |
| partial FP8 head + top3 | 51 | 26,240 | 0 |
| final mapped top3 | 28 | 1,216 | 0 |

## Closed physical32 model

Five head calls occur per physical32 event for both B1 and B4. The stage-1 B4
layout keeps qweight reads fixed at 335,544,320 bytes/head instead of multiplying
them by four.

| Quantity | B1 | B4 |
| --- | ---: | ---: |
| MACs/head | 335,544,320 | 1,342,177,280 |
| FP8 qweight bytes/head | 335,544,320 | 335,544,320 |
| removed full-logit write+read/head | 262,144 | 1,048,576 |
| partial workspace write+read/head | 18,432 | 73,728 |
| net intermediate bytes removed/event | 1,218,560 | 4,874,240 |
| candidate launches/head | 2 | 2 |

At the existing 273 GB/s floor basis, the B1 intermediate-byte delta is only
about 0.0045 ms/event before launch effects. Therefore this candidate cannot by
itself close the current 95.17 ms acceptance gap. Its value is removing a known
materialize/rescan edge without increasing B4 weight traffic; it still requires
a pinned build, exact B1/B4 kernel gate, and real-task measurement.

