# L0c Memory Prior-Art Alignment

Date: 2026-05-04

## Online Prior Art Checked

- Apache TVM AutoTVM blog: the tuner repeatedly chooses promising candidates, profiles them on hardware, trains a prediction model from profiling results, and uses that model to choose the next candidates. Source: https://tvm.apache.org/2018/10/03/auto-opt-all
- TVM MetaSchedule docs: the loop stores schedule traces and measured run times in a persistent database, then uses measured results to update the cost model. Source: https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html
- Ansor OSDI 2020: evolutionary search uses high-quality programs from previous measurements as population seeds, measures selected programs on hardware, and feeds profiling data back into the cost model. Source: https://www.usenix.org/system/files/osdi20-zheng.pdf
- OpenTuner PACT 2014: search techniques share results through a common database; stronger techniques receive more test budget, while poor techniques receive less. Source: https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf

## Alignment Decision

The L0c loop should not remember only rejected mutation hashes. It should maintain a measured-trial memory that records:

- workload/kernel identity,
- changed surface and source region,
- compact mutation features and schedule/config trace,
- expected low-level mechanism,
- preflight/compile/parity/measurement gate,
- objective measurements,
- measurement policy and cold/warm context,
- parity diagnostic overshoot where present,
- relation to prior failure families,
- next-search implication and explicit search-budget bias.

This matches the common prior-art pattern: preserve workload/config identity, hardware-measured records, feature tags, and compact traces, then bias later search using those records. It intentionally avoids turning history into blind syntax bans.

## Implemented Structure

- `prior_research_memory.tsv`: cross-round measured-trial memory imported at round start.
- `research_memory.tsv`: current-round memory updated as candidates are rejected, demoted, fail compile/parity, or complete measurement.
- `research_memory.md`: agent-readable explanation and compact recent/prior row summary.
- `strategy_brief.md`: now names the prior-art memory contract and tells agents to compare new candidates against the measured-trial ledger.
- `iteration_brief.md`: now instructs agents to read `prior_research_memory.tsv`, `research_memory.tsv`, and `research_memory.md` before proposing mutations.

The TSV schema now includes `workload_key`, `mutation_features`, `schedule_trace`, `measurement_policy`, and `search_bias` in addition to the outcome/objective/parity fields. This makes the memory closer to AutoTVM/MetaSchedule-style tuning records: future candidates can compare the actual tried configuration and decide whether to exploit, avoid, repair, or treat a row as context-only.

## CUTLASS-Specific Guidance

For the current FP8 GEMM CUTLASS-only loop, memory rows should steer agents away from repeated wrapper-only, SM-count, SM120-guard-only, and already-measured M<=16 tile variants unless the agent has a new dispatch, scale, shape, or schedule fact that changes the expected outcome.

The controller still owns expensive apply-and-test, vLLM restart, parity, and measurement. Authoring agents own local patch dry-run, Python compile, and targeted C++/CUDA preflight for touched CUTLASS files.
