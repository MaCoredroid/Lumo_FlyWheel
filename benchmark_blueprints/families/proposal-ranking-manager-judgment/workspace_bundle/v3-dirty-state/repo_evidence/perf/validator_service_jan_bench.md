# Validator-service benchmark — 2026-01-22

**STALE PRE-FP8 BASELINE.** See footer for why the headline is misleading.

## Setup

- Service under test: prototype `schema-validator` microservice
- Traffic source: replay of 2026-01-15 production capture
- Baseline respproxy: **v0.17.4 (pre-FP8 quantization)**. Model forward
  on the tool-call path was 95 ms p95 at that time.

## Results

| metric (p95)         | baseline (2026-01-22) | with validator svc |
| -------------------- | --------------------- | ------------------ |
| normalize_request    |  50 ms                |  50 ms             |
| validate_tool_args   |  84 ms                |   0 ms (async)     |
| compile_tool_schema  | 128 ms                | 128 ms             |
| dispatch_skill_pack  |  61 ms                |  61 ms             |
| model_forward        |  95 ms                |  95 ms             |
| end-to-end           | 420 ms                | 125 ms             |

Headline: 70% reduction.

## Why this is misleading if consumed today

The baseline's `model_forward=95ms` row reflects the **pre-FP8** transformer
runtime. Production model_forward on the tool-call path has been 42 ms since
the FP8 rollout on 2026-03-11 (see `rollout_history.md`). The current
end-to-end baseline is 420ms total but with a much smaller model_forward slice;
the "savings" this proposal captures are the same ~84ms of validation time
we'd save with or without the microservice. The extra 211ms of "savings" in
the table above are already captured by the production runtime and are not
attributable to the validator service.

**Today's realistic delta from this proposal: 420ms → ~290ms**, not → 125ms.

## Authoring note

This file was labeled `stale` in-repo in the 2026-04-03 evidence sweep.
Kept in tree for historical context. Do not cite the headline without the
correction above.
