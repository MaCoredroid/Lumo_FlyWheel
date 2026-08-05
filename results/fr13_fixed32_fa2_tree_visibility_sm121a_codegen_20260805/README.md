# Fixed32 FA2 tree-visibility masks, SM121a

Status: **default-off CPU/codegen admission pass; real B1 and B4 byte gates
and full-step timing are pending**.

## Fixed contract

This candidate is limited to the current qrow32 `num_splits=0` kernels for
physical32 Tail23/Hydra27, K64/root1, B1 and B4. The two modes share the same
32-node parent topology and differ only in downstream valid-node masks. All 32
rows, including inactive physical slots, retain their exact self-plus-ancestor
visibility.

The source replaces each dense fp32 tree-bias value load with one cached
32-bit visibility row and materializes the same `0.0f` or `-INFINITY` value.
The existing branch and `bias / scale_softmax` expression remain in place.
QK, ordered K traversal, masking, softmax, PV, output stores, launch geometry,
and commit-time valid-node ownership are unchanged. The candidate is
source-bound, fail-closed, and selected only by the new
`--fixed32-tree-visibility-mask` build flag.

## SM121a codegen

Fresh CUDA 13.0.88 objects were built twice from pristine FA2 commit
`29210221863736a08f71a866459e368ad1ac4a95`, with `CUDA_VISIBLE_DEVICES` empty,
CUDA cache disabled, GCC 11.4, and target `sm_121a`. Both independent builds
produce identical full cuobjdump SASS in each lane.

| lane | variant | regs | stack/local/spills | SASS instructions | `LDG.E` | `LDC` |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| B1 qrow32 | qualified no-split incumbent | 252 | 0/0/0 | 4,008 | 68 | 68 |
| B1 qrow32 | visibility candidate | 252 | 0/0/0 | 3,640 | 4 | 72 |
| B4 qrow32 | admitted incumbent | 252 | 0/0/0 | 3,992 | 68 | 68 |
| B4 qrow32 | visibility candidate | 252 | 0/0/0 | 3,632 | 4 | 72 |

The visibility table occupies 128 bytes in constant bank 3. Both candidates
retain 1,024 bytes static shared memory and one barrier. The 252-register
source cap is required: a first 254-cap build allocated 254 registers. With
the 252 cap, B1 and B4 remain spill-free and exactly match their incumbents'
resource envelope. Static attention work is invariant at 512 HMMAs, 132
FFMAs, 264 FMULs, 176 `LDGSTS`, 288 `LDSM`, and 38 global stores.

The exact `LDG.E` reduction is 64 static sites in both lanes. Total static
SASS falls by 368 instructions (9.182%) on B1 and 360 (9.018%) on B4. This is
compiler/disassembler evidence, not a dynamic instruction or speed claim.

## Per-event bias demand

Across 16 target layers and 24 query heads per sequence, dense physical32 bias
consumes 1,024 logical fp32 values per head. A row mask consumes 32 uint32
values per overlapping `BlockN=64` suffix tile; the 32-token suffix spans one
or two tiles depending on alignment.

| batch | dense bytes/event | mask bytes/event | reduction bytes/event | ratio |
| --- | ---: | ---: | ---: | ---: |
| B1 | 1,572,864 | 49,152 to 98,304 | 1,474,560 to 1,523,712 | 16x to 32x |
| B4 | 6,291,456 | 196,608 to 393,216 | 5,898,240 to 6,094,848 | 16x to 32x |

These are logical source-memory values before lane mapping, predication,
constant broadcast, and cache behavior. They are not measured HBM traffic.

## Prior-lane audit

- The qrow16 `num_splits=0` reference remains untouched. Applying this mask
  transform to its RU3/static-stride source rose from 212 to 216 registers;
  forcing 212 produced a 40-byte stack frame and ptxas reported spill stores
  and loads. That experiment is rejected and no qrow16 source is retained.
- B1 qrow32 no-split Gate A attempt14 matched qrow16 with zero BF16 output and
  FP32 LSE byte mismatches on all 16 target layers of a real SWE-Verified task.
- B1 qrow32 split2 is untouched. Its first real replay differed by 3,104,943
  output bytes and 9,551 LSE bytes, so it cannot be used as a reduction-order
  substitute for the qualified no-split kernel.
- The earlier paired-page-load edit was rejected because the compiler already
  coalesced the adjacent load and the remaining pairs span long register-live
  regions. The B4 qrow32 incumbent remains host-admitted but live-byte pending.

## Qualification boundary

No GPU, Docker container, synthetic probe, task payload, timing, TPS, or
acceptance run was used here. This package does not authorize production or
claim progress against the hardware floor. The next required evidence is the
standing real SWE-Verified raw-byte gate followed by full-step measurements:
B1 on the standing task set and exact4 B4, both Hydra27 and Tail23 where the
campaign gate requires them. Raw SASS, objects, binaries, task data, and
runtime identifiers are excluded from Git.
