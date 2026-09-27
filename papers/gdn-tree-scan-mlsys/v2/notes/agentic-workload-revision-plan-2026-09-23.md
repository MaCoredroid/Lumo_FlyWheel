# Author-directed workload evaluation revision — 23 September 2026

The author requires performance results from named long-running agent workloads such as SWE-bench Verified. Local-document continuation E1/E8 performance numbers and their rate figure must leave the manuscript, abstract and conclusion. Original data, frozen protocols, adverse outcomes and previous paper snapshots remain immutable. Numerical and continuation correctness evidence remains supporting mechanism evidence, clearly separate from agent-performance evaluation.

## Available evidence under review

- Latest Qwen3.8-27B NVFP4 coding-agent records in results/fr14_nvfp4_port_20260816, including the full campaign through its final sections rather than only the Pages headline.
- Exact4 split-K tree/control workload comparison: request-weighted inverse mean TPOT, physical step timing, task outcomes and complete run lineage need exact labels; this is not a native-versus-tree contrast or an end-to-end task speedup.
- Exact16 lineage: raw evaluators disagree with the website and later narrative; verify task census and configuration parity before publishing any score. Never discard incomplete tasks from its intended denominator.
- Four native MTP-5 NVFP4 task probes: actual same-model decode route with declared loader-only shim; targeted degeneration probes do not establish full-workload performance parity.

## Required performance table fields

Named benchmark and task IDs/subset; task count and original incomplete outcomes; pinned model/revision/quantization; agent harness and tool/network policy; sampler and budget; concurrency; implementation/backend; completed/evaluated/resolved counts; total agent-task wall times and completion limits; emitted tokens and full request timing where available; the exact decode-rate estimator; repeated-run count and matched comparator.

## Missing comparison to prepare, not launch during this evidence audit

A current, matched native-MTP versus selected-tree SWE-bench Verified campaign must pin one Qwen3.8 NVFP4 checkpoint, one task cohort, identical agent/tool/sampling/budget/network conditions and the actual selected optimizations. Each arm needs independent repeated complete task attempts, no favorable replacement, all terminal outcomes, task-latency and success reporting, plus API-bound serving measures. Use an official disjoint broader cohort for a broader benchmark claim; an Astropy-only subset must be labeled as such. Add multi-agent concurrency only after the single-agent route is qualified and only if that generalization is claimed. Freeze the full protocol and inspect the concrete experiment list before launching long GPU work. Existing microbenchmark data cannot fill these missing agent-workload cells.
