# Round-F Tree-Delta GDN State-Copy Closeout - 2026-05-30

## Result

Primary long-sequence correctness is fixed for the Round-F unique-node GDN tree-delta verifier.
The best uncapped real SWE-Verified run reached `mean_acc_per_event = 1.995`, matching the E3
target band around `1.989` and clearing the prior uncapped drift (`1.43`).

The CUDA graph speed target is partially met. The same lossless config reached
`warm_decode_tps = 14.060876`, beating path-row K2 (`12.35`) but not E3-FULL (`15.56`).

Implementation commit:

- `0e4f29cc Fix tree delta GDN rollback commit`

Primary patch surface:

- `scripts/swe_x86_helpers/relaunch_qwen36_round.py`

## Root Cause

The drift was the known hybrid GDN/Mamba speculative-decode state rollback failure mode:
spec decode advances recurrent state through all proposed tokens, but only the accepted prefix
is kept. The next iteration must therefore restore the recurrent cache to the exact state after
the accepted token, not after the rejected suffix.

The prior activation-replay commit recomputed the accepted path and left room for long-sequence
divergence from the canonical recurrence. The fix stops recomputing the accepted path and instead
copies the already-materialized accepted row state back into the prefix slot.

The copy covers both state families:

- `conv_state`
- `ssm_state`

The accepted row is selected as:

```text
final_row = base + accepted_count - 1
```

This matches vLLM's own postprocess formula:

```text
new_num_computed_tokens =
    num_computed_tokens + num_scheduled_tokens - num_draft_tokens
    + num_accepted_tokens - 1
```

Container-side inspection of `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py`
confirmed the generic Mamba/GDN postprocess rollback uses that same `-1` accepted-position
convention. The relevant upstream family is vLLM Issue #39273 and PR #40738:

- https://github.com/vllm-project/vllm/issues/39273
- https://github.com/vllm-project/vllm/pull/40738

## Validation Matrix

All rows below are uncapped real SWE-Verified measurements from
`scripts/measure_track_b_real_workload.py --model qwen3.6-27b`.

| Config | Workload | mean_acc_per_event | Events | acc_dist | warm_decode_tps | ms/generated tok | accepted/draft | Invariants |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Eager state-copy | 2 completions, conc=1 | 2.237 | 316 | 0:35, 1:48, 2:40, 3:193 | 15.068788 | 66.362339 | 0.745781 | clean |
| Eager state-copy | 5 completions, conc=4 | 1.917 | 877 | 0:172, 1:153, 2:128, 3:424 | 12.175196 | 82.134202 | 0.626348 | clean |
| CUDA full + packed + GDN core unsafe | 5 completions, conc=4 | 1.995 | 855 | 0:146, 1:145, 2:131, 3:433 | 14.060876 | 71.119325 | 0.663747 | clean |
| CUDA full + packed + GDN core captured | 5 completions, conc=4 | 1.905 | 882 | 0:181, 1:149, 2:125, 3:427 | 13.980225 | 71.529608 | 0.652237 | clean |

Best artifact:

- `output/real_workload/F_tree_delta_state_copy_cgfull_swe_verified_conc4_20260530T0629Z.json`

Other artifacts:

- `output/real_workload/F_tree_delta_state_copy_eager_swe_verified_2inst_20260530T0615Z.json`
- `output/real_workload/F_tree_delta_state_copy_eager_swe_verified_conc4_20260530T0618Z.json`
- `output/real_workload/F_tree_delta_state_copy_cgfullfull_swe_verified_conc4_20260530T0642Z.json`

The measurement script returned non-zero for these runs because its hardcoded pass threshold is
well above this experiment's comparison target. The JSON artifacts are valid and invariant-clean:
`path_rows_zero_rate = 1.0` and `invariant_failures = {}`.

## CUDA Graph Attempt

The best speed/correctness point is:

```bash
unset LUMO_ENFORCE_EAGER
export LUMO_GPU_MEMORY_UTILIZATION=0.86
export LUMO_FA_UNIQUE_NODES=1
export LUMO_FA_TREE_DELTA_TRITON=1
export LUMO_FA_ACTIVATION_REPLAY_COMMIT=1
export LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1
export LUMO_FA_PACKED_CUDAGRAPH_SIZES=1
export LUMO_CUDAGRAPH_MODE=full
.venv/bin/python scripts/swe_x86_helpers/relaunch_qwen36_round.py \
  --config F --mtp 3 --tree '[(0,), (0, 0), (0, 0, 0)]'
```

An additional FULL/FULL attempt without `LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1` reached READY, but
vLLM downgraded to `FULL_AND_PIECEWISE` because `GDNAttentionBackend` is not full-capture safe in
this stack. It did not improve acceptance or throughput.

## Remaining Risk

The fix removes the lossy recompute path and validates the uncapped real workload target. The
remaining gap is speed versus E3-FULL. The next useful speed work is not more rollback debugging;
it is reducing verifier overhead after state-copy commit, especially GDN-core capture support and
event-level timing instrumentation for `verify_us`, `gdn_parent_gather_us`, and `commit_us`.
