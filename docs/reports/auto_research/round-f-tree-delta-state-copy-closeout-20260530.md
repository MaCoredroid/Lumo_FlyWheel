# Round-F Tree-Delta GDN State-Copy Closeout - 2026-05-30

## Result

Primary long-sequence correctness is fixed for the Round-F unique-node GDN tree-delta verifier.
The best uncapped real SWE-Verified run reached `mean_acc_per_event = 1.995`, matching the E3
target band around `1.989` and clearing the prior uncapped drift (`1.43`).

The CUDA graph speed target is partially met. The same lossless config reached
`warm_decode_tps = 14.060876`, beating path-row K2 (`12.35`) but not E3-FULL (`15.56`).
This is a research deliverable, not a production replacement for E3-FULL: the best spine
configuration is about 10% slower at the event level (`87.608 ms/event`) and remains below
E3-FULL throughput.

⚠️ COMPARISON CAVEAT (must resolve before trusting the speed ranking): the tree-delta number
(`14.06`) was measured on `workload_distribution_id = swe_bench_verified_subset:swe-bench-concprobe4-verified-instances-20260522.json`
(4 real SWE-Verified instances, ~819 prompt tokens/req), but the E3-FULL (`15.56`) and path-row K2
(`12.35`) baselines came from the earlier matrix on a DIFFERENT workload (`632cac0c…`, ~660 prompt
tokens/req, no recorded instance_ids). So the speed ranking is **cross-workload, not apples-to-apples**.
The lossless correctness result (`acc 1.995` on SWE-Verified) stands on its own; the speed comparison
must be re-established with a single frozen paired run (E3-FULL / spine tree-delta / branched tree-delta
on the SAME SWE-Verified subset, same container/runtime) before the "beats path-row K2 / ~1.10×E3"
claims can be trusted.

Primary implementation commit:

- `0e4f29cc Fix tree delta GDN rollback commit`

Branched follow-up commits:

- `44cfc6f6 Fix branched tree rejection sampling ratio`
- `6365355f Commit branched GDN state from accepted tree row`
- `f6eeb856 Order branch accepted-row state-copy patch`
- `d3ac2030 Harden accepted tree row sampler handoff`
- `e025297e Fix accepted row sampler vocab scope`
- `55a8a0a0 Record branched accepted rows in tree sampler`

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

## Branched Tree Follow-Up

The branched K=2 tree was tested after the spine state-copy fix using:

```text
[(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)]
```

This tree contains the top-1 spine path plus alternate branches, so its acceptance should not be
below the spine if the branch verifier and state commit are fully correct. It did not clear that
bar on the uncapped real SWE-Verified workload.

| Branched config | mean_acc_per_event | Events | acc_dist | warm_decode_tps | mean_event_ms | Invariants |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Initial branched PIECEWISE | 1.134 | 1200 | 0:460, 1:324, 2:211, 3:205 | 8.629945 | 92.621 | clean |
| Accepted branch-row state copy | 1.605 | 985 | 0:247, 1:238, 2:157, 3:343 | 10.104572 | 96.483 | clean |
| Kernel-sourced accepted branch row | 1.608 | 985 | 0:251, 1:236, 2:146, 3:352 | 10.386313 | 98.703 | clean |

Final branched artifact:

- `output/real_workload/F_tree_delta_state_copy_branched_default_triton_eager_kernelrow_swe_verified_conc4_20260530T0933Z.json`

The branched attempt improved from `1.134` to `1.608` but remained below the lossless spine
run (`1.995`). The proposer was ruled out: the branched top-1 draft matched the spine draft, and
short diagnostics accepted the top-1 path. The residual gap is in the long-sequence,
accepted-branch verifier/state-copy path. Work stops here by directive: on this real-code
workload, alternate MTP branches rarely add enough accepted tokens to exceed the top-1 spine,
so even a fully corrected branched verifier has a ceiling near the spine while paying extra
tree/GDN overhead.

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

The branched path also cannot become the E3-beating path in this vLLM 0.19.0 stack without more
substantial backend work: `GDNAttentionBackend` and `TreeAttentionBackend` are not full-capture
safe, forcing eager-GDN and PIECEWISE-tree overhead. With the measured branch acceptance ceiling
on real code, that overhead dominates.

## Remaining Risk

The fix removes the lossy recompute path and validates the uncapped real workload target. The
remaining gap is speed versus E3-FULL. The next useful speed work is not more rollback debugging;
it is reducing verifier overhead after state-copy commit, especially GDN-core capture support and
event-level timing instrumentation for `verify_us`, `gdn_parent_gather_us`, and `commit_us`.

Branch handling remains a non-shipping research path. The work was merged to `main` via
`--no-ff` merge commit `9fc08ae7` (flag-gated behind `LUMO_FA_*`, so the default serving path is
unchanged); the source branch `round-f-tree-delta-kernel` @ `45c56c10` is preserved.
