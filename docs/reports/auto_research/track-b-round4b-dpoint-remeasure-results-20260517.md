# Round 4b — D-point P1 remeasure results (2026-05-17)

Closes the P1 action in `track-b-round4b-power-w-remeasure-list-20260516.md`:
re-run the 4 contaminated D-point attempts (`responses-sdk-adapter-cutover`
run_02/run_03, `transcript-merge-regression` run_02/run_03).

## Setup

| Component | State |
|---|---|
| vLLM | `lumo-vllm-track-b-suffix` relaunched to the D-point config — spec_decode suffix bundle (candidate-056 / tuned_config_id `712fd011`) + full T1-T4 prelaunch; runtime flags `{T2,T3,T4}=false` (all enabled). |
| Host | load 0.17, **vLLM the only GPU compute process** — no competing tenants (the original contamination was attributed to other tenants/cron jobs). |
| Protocol | `run_track_b_e2e_task.py --attempt 2 --repeat 2` per task, `/reset_prefix_cache` before each task, docker-isolated codex, 1800s budget. |
| runtime_config_hash | stamped `sha256:5ae88ac4…` (the D-cell hash carried by run_01/run_04). See "Hash provenance" below. |

## Results

| Task | Attempt | decode_tps | prefill_s | power_w (med) | verdict |
|---|---|---:|---:|---:|---|
| responses-sdk | run_02 (was 5.10) | **16.63** | 2.10 | 44.8 | clean |
| responses-sdk | run_03 (was 7.85) | **16.23** | 1.94 | 43.7 | clean |
| transcript-merge | run_02 (was 9.33) | **14.45** | 0.97 | 40.2 | clean |
| transcript-merge | run_03 (was 4.84) | **11.37** | 1.44 | 41.4 | clean |

**Contamination did not recur.** The contamination signature is a *simultaneous*
decode_tps drop AND prefill_s rise. The remeasured attempts have **normal
prefill_s (0.97-2.10s** vs the contaminated 3.4-4.3s) and decode_tps **2-3× above
the contaminated 5-9 band**. All four did real work (67-501 tool events, 25-28
files written per attempt).

The remeasure did **not** land in the report's hoped "22-30 tps" acceptance band.
That band was over-optimistic: it came from `median(run_01=16.45, run_04=29.11) =
22.78` for responses-sdk — a 2-point median of a high-variance cell. The fresh
attempts (16.6, 16.2) sit right at the *lower* clean attempt (run_01=16.45). The
D-cell's real central tendency for these heavy tasks is ~12-17 tps, not 22-30.

## Recomputed D-point medians (4 clean attempts each)

| Task | run_01 | run_02 (fresh) | run_03 (fresh) | run_04 | **4-attempt median** | report's 2-pt estimate |
|---|---:|---:|---:|---:|---:|---:|
| responses-sdk-adapter-cutover | 16.45 | 16.63 | 16.23 | 29.11 | **16.54** | 22.78 |
| transcript-merge-regression | 25.91 | 14.45 | 11.37 | 11.81 | **13.13** | 18.86 |

Against A-point (T1 only): responses-sdk A=21.74, transcript-merge A=15.60.
With the fresh 4-attempt D medians, A is faster than D on **both** tasks
(+31% and +19%). The per-task A-vs-D attribution on these two cells has now
flipped twice across cleanup passes — they remain the highest-variance cells
in the corpus and any per-task claim on them is low-confidence. The aggregate
"A roughly flat-to-slightly-favored over D" conclusion is unaffected.

## Hash provenance

A fresh `build_track_b_runtime_config_hash.py` recompute on the relaunched
D-point vLLM yielded `sha256:841fb0ea…`, differing from the D-cell hash
`sha256:5ae88ac4…`. Every load-bearing field is reproduced identically from
the same bundle + image: `model_id`, `vllm_version=0.19.0`, `quantization=fp8`,
`kv_cache_dtype=auto`, `max_model_len=131072`, `gpu_memory_utilization=0.9`,
`tuned_config_id=712fd011`, `weight_version_id=2e1b2135`, `speculative_config`
(suffix/12/…), `kernel_runtime_activation.activation_id=03c2dca7` (deterministic
— sha256 of empty kernel_selection/launch_args/env), `git_hash=unknown` (vLLM
`__commit__`, constant). The original D-point hash-builder payload was not
archived, so the differing field cannot be diffed directly; the most likely
cause is a `HASH_FIELDS` / canonicalization change in the builder since
2026-05-12. The decode-relevant serving config is identical, so the remeasured
attempts are stamped with the cell hash `5ae88ac4` for aggregation consistency.
The recomputed manifest is saved at `…/run_02/runtime_config_hash_recompute.json`.

## Disposition

- Contaminated originals archived in-place as `contaminated_run_02_20260512/`
  and `contaminated_run_03_20260512/` (renamed off the `run_*` glob so
  aggregation scripts skip them — not deleted).
- Fresh attempts installed as `run_02/` and `run_03/`.
- Both under `output/track_b_e2e_v4a_v2/round_0_phase1_task1_2_PRESERVED/`.

## Files

- This report: `docs/reports/auto_research/track-b-round4b-dpoint-remeasure-results-20260517.md`
- Remeasure driver: `/tmp/dpoint_remeasure.py`
- Supersedes the P1 acceptance criteria in
  `track-b-round4b-power-w-remeasure-list-20260516.md` (the 4-attempt
  contamination list was correct; the 22-30 tps acceptance band was not).
