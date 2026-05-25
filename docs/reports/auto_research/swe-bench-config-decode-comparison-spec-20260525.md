# SWE-Bench Config Decode Comparison Spec

Date: 2026-05-25

## Purpose

Define the math used to compare Config D and Config E SWE-Bench runs, especially
for concurrent `B=4` campaigns where per-task proxy metrics can overlap.

This spec is for speed and telemetry comparison only. Correctness is still the
SWE-Bench eval verdict in each task's `eval_report.json`.

## Inputs

For each run tag, such as `q36a_D_b4`, `q36a_E1_b4`, `q36a_E2_b4`, or
`q36a_E3_b4`, use:

- Task metadata:
  `output/<tag>/<tag>/per_task/<instance_id>/runner_metadata.json`
- Eval verdict:
  `runner_metadata.json.eval_report.verdict`, or
  `output/<tag>/<tag>/per_task/<instance_id>/eval/eval_report.json`
- Proxy request metrics:
  `output/<tag>/<tag>/per_task/<instance_id>/vllm_request_metrics.jsonl`
- Run-level vLLM step trace:
  `output/<tag>/dgx_steptrace.jsonl`
- Run-level speculative step trace:
  `output/<tag>/per_req_spec_trace.jsonl`

## Correctness Metrics

For a run with task set `T`, define:

```text
resolved_count = count(t in T where verdict(t) == "resolved")
failed_count   = count(t in T where verdict(t) == "failed")
resolved_rate  = resolved_count / |T|
```

Timeouts are diagnostic, not direct correctness:

```text
timeout_count = count(t in T where runner_metadata.codex.timed_out == true)
```

## Why Task-Local Decode TPS Is Unsafe At B=4

`vllm_request_metrics.jsonl` rows contain per-request token counts, but their
`decode_sum_s` and `prefill_sum_s` come from deltas of global vLLM Prometheus
counters. Under `B=1`, a task-local slice is usually isolated enough to treat as
per-task. Under `B=4`, multiple task runners overlap, so a task-local capture can
include decode counter movement caused by sibling tasks.

Therefore:

- Use task-local `vllm_request_metrics.jsonl` for request shape diagnostics.
- Do not sum task-local `decode_sum_s` as the primary `B=4` run speed.
- Use run-level `dgx_steptrace.jsonl` as the primary `B=4` decode speed source.

## Request Deduplication

When using proxy request rows for diagnostics, deduplicate by `request_id` across
all per-task files in the run:

```text
R_raw = concat(all task vllm_request_metrics rows)
R = latest row per request_id in R_raw, plus rows without request_id
R_usable = {r in R where completion_tokens(r) > 0 and decode_sum_s(r) > 0}
```

Deduplicated request-level diagnostic aggregate:

```text
request_tokens = sum(completion_tokens(r) for r in R_usable)
request_decode_sum_s = sum(decode_sum_s(r) for r in R_usable)
request_agg_tps = request_tokens / request_decode_sum_s
request_median_tps = median(completion_tokens(r) / decode_sum_s(r) for r in R_usable)
```

These values are useful for trend checks, but remain secondary under `B=4`
because global counter overlap still exists.

## Primary B=4 Decode Speed

Let `S` be `dgx_steptrace.jsonl` rows clipped to the run wall-clock window:

```text
run_start = min(task.started_at for all tasks)
run_end   = max(task.ended_at for all tasks)
S = {s in steptrace where run_start <= s.ts <= run_end}
```

Each steptrace row contains cumulative vLLM counters:

```text
gen     = vllm:generation_tokens_total
dec_sum = vllm:request_decode_time_seconds_sum
```

Primary run-level decode speed:

```text
steptrace_gen_delta = last(S).gen - first(S).gen
steptrace_dec_delta = last(S).dec_sum - first(S).dec_sum
steptrace_decode_tps = steptrace_gen_delta / steptrace_dec_delta
```

This is the headline speed metric for `B=4`.

## Dormant Window Math

A step interval `(s_i, s_{i+1})` is decode-dormant when all are true:

```text
s_{i+1}.gen == s_i.gen
s_{i+1}.dec_sum == s_i.dec_sum
s_i.running == 0 and s_i.waiting == 0
s_{i+1}.running == 0 and s_{i+1}.waiting == 0
```

Dormant totals:

```text
dormant_s = sum(s_{i+1}.ts - s_i.ts for dormant intervals)
dormant_pct = dormant_s / (run_end - run_start)
```

Dormant time is not subtracted from headline decode TPS. It is reported to
explain scheduler gaps or tail/wrap-up windows.

## Speculative-Decoding Math

Use `per_req_spec_trace.jsonl` as the source of truth for speculative events.
Each row has:

```text
draft = number of draft tokens proposed for that speculative event
acc   = number of draft tokens accepted for that speculative event
```

For all speculative events `E` in the run:

```text
draft_total = sum(e.draft for e in E)
acc_total   = sum(e.acc for e in E)
event_count = |E|

accept_ratio = acc_total / draft_total
draft_per_event = draft_total / event_count
acc_per_event = acc_total / event_count
```

Interpretation:

- `accept_ratio` alone is not a speed metric.
- `acc_per_event` is a better speculative-work metric because it includes depth.
- A high `accept_ratio` with `draft_per_event = 1` can still be slower than a
  lower `accept_ratio` with larger `draft_per_event`.

## Current D/E B=4 Snapshot

As of the completed `q36a_D_b4`, `q36a_E1_b4`, `q36a_E2_b4`, and `q36a_E3_b4`
runs:

| run | depth / mode | resolved | steptrace decode TPS | accept ratio | draft/event | accepted/event |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q36a_D_b4` | suffix/D | 6/16 | 13.246 | 0.256 | 5.761 | 1.473 |
| `q36a_E1_b4` | MTP 1 | 7/16 | 10.795 | 0.877 | 1.000 | 0.877 |
| `q36a_E2_b4` | MTP 2 | 7/16 | 12.051 | 0.818 | 2.000 | 1.635 |
| `q36a_E3_b4` | MTP 3 | 7/16 | 15.058 | 0.751 | 3.000 | 2.254 |

Conclusion from this slice:

- E1 has high acceptance but too little accepted work per event.
- E2 improves accepted work per event but remains slower than D on primary
  steptrace speed.
- E3 is the current leader on this slice: same best resolved count and the
  highest run-level decode TPS.

## Reporting Rules

When reporting a run comparison, include:

1. `resolved_count / task_count`
2. `steptrace_decode_tps`
3. `accept_ratio`
4. `draft_per_event`
5. `acc_per_event`
6. `timeout_count`
7. `dormant_s` and whether it is material

Do not promote a config from one 16-task slice alone. A promotion-quality claim
needs at least one repeat run or a larger paired set with the same task subset,
temperature, concurrency, agent wall clock, eval timeout, and config relaunch
method.
