# FR13 fixed32 current kernel-time ledger

Status: `CURRENT_PHASE_TIMING_VALID_SYMBOL_TIMING_STALE`.

This ledger does not add a measurement. It reconciles the latest valid real
SWE-Verified Qrow16 exact4 B1 Hydra arm with the latest available real-workload
Nsight symbol attribution. The Qrow16 run measured current phase envelopes but
did not run Nsight or Kineto. Therefore no individual kernel duration in this
artifact is labeled current.

## Current measured point

The valid Hydra arm at B1 measured 5,148 complete fixed32 events:

| Quantity | ms/event |
| --- | ---: |
| SFWD GPU | 159.619263 |
| DFWD GPU | 36.813368 |
| CFWD GPU | 20.677391 |
| GPU components | 217.110022 |
| Non-component wall residual | 15.669768 |
| Full-step wall | 232.779790 |

It delivered `24.718147` full-wall TPS at `4.753885` accepted drafts/event.
The corrected optimistic weight-read floor is `119.658015 ms/event`; the
`1.15x` cap is `137.606718 ms/event`. The point is `95.173072 ms/event` above
the cap. No one-sided U95 was produced because the paired Tail arm was invalid.

Current work census proves Qrow16 on all 16 target-attention layers, 16 tree
attention calls and 512 query rows/event, 96 tree-GDN launches/event, one
committer graph replay with 48 layer calls, and one direct conv-commit launch.
These are current counts, not current kernel durations.

## Latest symbol timing

The latest committed symbol attribution is the earlier B1 Tail6 Nsight
capture at source `1a7a765447c8ce6068e0dd5d3a344d58ace85f2b`. It used real
SWE-Verified exact4 traffic and contains 881 complete step ranges, but it
predates Qrow16 and the current integrated kernel stack and was not an
acceptance run. All numbers below are historical attribution only:

| Phase | Historical symbol/group | ms/event |
| --- | --- | ---: |
| SFWD | CUTLASS FP8 blockwise target GEMMs | 112.312954 |
| SFWD | FA2 split-KV tree attention | 24.708601 |
| SFWD | `_tree_gdn_path_kernel` | 14.019520 |
| SFWD | BF16 `indexSelectSmallIndex` | 5.938124 |
| SFWD | BF16 `index_copy` | 4.591679 |
| SFWD | `_fr13_conv_wb_fused_batched_kernel` | 4.484287 |
| DFWD | cuBLAS BF16 `gemvx` draft heads | 26.227316 |
| DFWD | CUTLASS FP8 blockwise MTP GEMMs | 8.514285 |
| DFWD | `kernel_unified_attention_2d` | 6.967564 |
| DFWD | `nvjet_sm121_tst_mma_64x112x64...` | 2.556116 |
| DFWD | FA2 causal split-KV | 1.249600 |
| CFWD | `fused_sigmoid_gating_delta_rule_update_kernel` | 4.082147 |
| CFWD | ATen float softmax | 2.304176 |
| CFWD | ATen float add | 1.600615 |
| CFWD | ATen scalar compare | 1.355676 |
| CFWD | ATen long fill | 1.252878 |
| CFWD | ATen float multiply | 1.176678 |
| CFWD | `_topk_topp_kernel` | 1.107767 |

The historical phase envelopes were SFWD `174.813673`, DFWD `47.435717`, and
CFWD `22.755077 ms/event`. They must not be subtracted from the current phase
timers to claim a kernel speedup: the profiler was Tail6, the current arm is
Hydra27, and source/runtime paths changed. No post-Qrow Kineto trace was found.

## Next runnable kernel

After Qrow16, BM8, and the B4 BV64 work, the next runnable candidate is the
fixed32 CUTLASS `streamk_coop128` target-GEMM path. It is the largest quantified
candidate that is already built and has an exhaustive real-task byte gate:

- deployment binary SHA-256
  `fa9395754b13de26dbed38dfc551614dbb109058764426564dcbb3c77fdd6ea9`
- build and additive ABI audit complete
- B1 same-process stock/candidate all-byte gate ready
- GPU correctness and speed not run
- modeled maximum recovery `10.923627 ms/event`, based on the stale Nsight
  target-CUTLASS group and an ideal traffic-equivalent model

That model is not timing evidence. Even realizing all of it would leave a
modeled `221.856163 ms/event` wall point, `1.854085x` the floor and
`84.249445 ms/event` above the cap.

Run order must stay on real SWE-Verified traffic: first the allowed real B1
byte gate, then, only if byte-clean, a control/candidate exact4 B1 full-wall
measurement with fresh symbol profiling. Do not use synthetic GEMM timing and
do not promote to exact16 before a meaningful floor-ratio breakthrough.
