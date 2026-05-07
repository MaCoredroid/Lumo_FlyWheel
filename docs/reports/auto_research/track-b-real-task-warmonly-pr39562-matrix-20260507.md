# Track B Real-Task Warm-Only Matrix - PR39562 Stop-Gap - 2026-05-07

## Scope

This report reruns every Track B candidate whose original synthetic first-five artifact exceeded `10 tok/s`:

- `020`: original `15.753922 tok/s`
- `025`: original `14.506594 tok/s`
- `028`: original `14.581565 tok/s`
- `051`: original `17.087062 tok/s`

The rerun uses a content-bearing repo benchmark task instead of the token-count proxy prompt. The live vLLM runtime used the PR39562 allocator stop-gap in `single_type_kv_cache_manager.py`, applied through the `ModelServer` prelaunch hook before `vllm serve`.

## Measurement Shape

- Agentic task: `release-note-to-plan-translation/v1-clean-baseline`
- Prompt source: `benchmark_blueprints/families/release-note-to-plan-translation/report/attempt_01_live_probe/metadata.json`
- Context included inline: `AGENTS.md`, `.scenario_variant`, release notes, and repo inventory from the same benchmark workspace bundle.
- Output cap: `2048`
- Policy: one cold completion was discarded; metrics were sampled immediately before and after the warm window only.
- Concurrency cases: `warm_concurrency=1` and `warm_concurrency=4`
- Speed metric: vLLM Prometheus decode-time throughput, `generation_tokens / decode_sum_s`
- Gate shown here: `9.0 decode tok/s`

## Results

| Candidate | Agentic task | Warm concurrency | Warm completions measured | Warm output tokens total | Warm metric generation tokens | Decode tok/s | Wall output tok/s | 9 tok/s gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `020` | release-note-to-plan-translation | 1 | 1 | 2048 | 2048 | 11.322390 | 11.307890 | pass |
| `020` | release-note-to-plan-translation | 4 | 4 | 8192 | 8192 | 9.861393 | 38.746154 | pass |
| `025` | release-note-to-plan-translation | 1 | 1 | 2048 | 2048 | 10.328952 | 10.316675 | pass |
| `025` | release-note-to-plan-translation | 4 | 4 | 8192 | 8192 | 9.214660 | 35.801341 | pass |
| `028` | release-note-to-plan-translation | 1 | 1 | 2048 | 2048 | 10.547060 | 10.534431 | pass |
| `028` | release-note-to-plan-translation | 4 | 4 | 7233 | 7233 | 9.860336 | 31.874462 | pass |
| `051` | release-note-to-plan-translation | 1 | 1 | 2048 | 2048 | 8.009177 | 8.001952 | fail |
| `051` | release-note-to-plan-translation | 4 | 4 | 7511 | 7511 | 8.304864 | 28.374132 | fail |

## Artifacts

Summary artifact:

- `output/auto_research/track_b/qwen3.5-27b-track-b-round0-real-workload-5x-20260506T000000Z/real_task_warmonly_pr39562_release_plan_matrix_summary.json`

Per-candidate artifacts:

- `candidates/020/real_task_warmonly_pr39562_release_plan_c1_run_01.json`
- `candidates/020/real_task_warmonly_pr39562_release_plan_c4_run_01.json`
- `candidates/025/real_task_warmonly_pr39562_release_plan_c1_run_01.json`
- `candidates/025/real_task_warmonly_pr39562_release_plan_c4_run_01.json`
- `candidates/028/real_task_warmonly_pr39562_release_plan_c1_run_01.json`
- `candidates/028/real_task_warmonly_pr39562_release_plan_c4_run_01.json`
- `candidates/051/real_task_warmonly_pr39562_release_plan_c1_run_01.json`
- `candidates/051/real_task_warmonly_pr39562_release_plan_c4_run_01.json`

## Interpretation

Candidates `020`, `025`, and `028` clear the `9.0 tok/s` speed gate on this real content task for both c1 and c4. Candidate `051` does not reproduce its original synthetic c4 speed and fails both c1 and c4.

The c4 wall-output throughput is much higher than c1 because four requests are in flight together, but the acceptance-relevant decode metric remains `generation_tokens / decode_sum_s`. The rerun therefore does not support treating the original synthetic c4 numbers as direct 4x decode speedups.

The next production-grade step is to run equivalence/correctness gates against `020`, `025`, and `028` under the same PR39562-patched runtime and real-task measurement shape. Among these, `020` has the best c1 decode result and `028` has the best c4 decode result by a small margin.
