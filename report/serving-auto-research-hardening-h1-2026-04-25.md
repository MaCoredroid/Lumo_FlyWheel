# Serving Auto-Research Hardening + H1 Report — 2026-04-25

## Summary

Implemented the Serving Auto-Research hardening plan through the H1 replay measurement. The final H1 result does **not** support promoting the imported C003 configuration as a defensible improvement.

Plan decision case: **Case beta** — H1 confidence is `exploratory`; the measured improvement is negative and the 95% CI crosses zero.

## Repository State

- Final `main` commit before this report: `b750b04c341ca86b8edd83037bfb7fa52bb18bcf`
- H1 round branch created by replay mechanics: `autoresearch/qwen3.5-27b/multi-family-v5/sprint-0/20260425T085916Z`
- H1 finalize commit on that round branch: `14efe15d0b25003b256f207600a496dc00e51ca2`

## Implementation Commits

- `9fab279` — `auto-research: clean up v0.1.12 loop contract`
- `24681eb` — `auto-research: align iteration brief harness contract`
- `94fb1f0` — `Implement multi-family v5 workload hardening`
- `08e1104` — `Fix A.1 thinking probe bug classification`
- `2fda68c` — `Enforce AR.26 per-file thinking thresholds`
- `0b8d1a2` — `Enforce serving thinking probe precondition`
- `f9d376b` — `Harden auto-research statistics and bundle metadata`
- `ff28967` — `Verify auto-research hardening metadata`
- `a29b0e8` — `Enforce L2 request shaping in proxy`
- `dce07e2` — `Keep L2 advisory shaping out of action space`
- `1478beb` — `Tighten L2 enforcement verification metadata`
- `2f3013e` — `Fix live vLLM reasoning probe accounting`
- `a4cf979` — `Implement auto-research replay round`
- `b750b04` — `Generate multi-family v5 workload artifacts`

## Live vLLM Verification

Live vLLM was verified at:

- Base URL: `http://127.0.0.1:8100/v1`
- Model: `qwen3.5-27b`
- Proxy/admin port: `8101`

The live thinking probe passed with outcome `row-3`. The probe also exposed a real repo-side accounting gap: vLLM Responses returned `output` items of type `reasoning` while usage fields reported zero reasoning tokens. Commit `2f3013e` fixed the probe and seed-capture accounting.

Live verification after the fix:

- Probe outcome: `row-3`
- Case A reasoning: `1798`
- Case B reasoning: `2200`
- Live seed-capture smoke: 2 rows, `thinking_tokens=[128, 128]`
- Required regression suite after live fix: `95 passed`

## Generated Workload Artifacts

Generated and committed the checked-family multi-family v5 workload in `b750b04`.

Artifact paths:

- `benchmark_blueprints/workloads/multi-family-v5/pool.yaml`
- `benchmark_blueprints/workloads/multi-family-v5/workload.yaml`
- `benchmark_blueprints/workloads/multi-family-v5/seed_trace.jsonl`
- `benchmark_blueprints/workloads/multi-family-v5/holdout_trace.jsonl`

Per-family v5 traces:

- `benchmark_blueprints/families/codex-provider-rollover/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/codex-skill-runtime-v2-split/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/esm-plugin-loader-modernization/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/nightly-regression-watch/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/objective-driven-repo-improvement/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/policy-aware-request-resolution/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/release-manifest-v2-modernization/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl`
- `benchmark_blueprints/families/sqlalchemy-2-session-modernization/seed_trace_v5.jsonl`

Workload descriptor summary:

- `workload_distribution_id`: `8d3e9038afd0c6f6be23c57e47d3a2ee8d619433d3c2e7c4127376a10ad456e8`
- Included pool size: 9 families
- Excluded pool size: 19 families, all `wire_api_missing`
- Seed rows: 27
- Holdout rows: 9
- Thinking probe ref: `reports/thinking-probe-20260424.md`
- Thinking probe outcome: `row-3`

AR.26 checked-pool validation:

- Per-file family coverage: pass
- Seed per family: 3 rows
- Holdout per family: 1 row
- Seed thinking positive: `27/27`
- Seed thinking greater than response: `27/27`
- Seed large-thinking rows: 5
- Holdout thinking positive: `9/9`
- Holdout thinking greater than response: `9/9`
- Holdout large-thinking rows: 8

The original 28-family strict minima (`84` seed / `28` holdout) do not apply to the generated checked-family descriptor because the current eligible pool has 9 included families and 19 excluded families.

## H1 Replay Round

H1 was run with the descriptor-supported holdout count of 9:

```bash
VLLM_BASE_URL=http://127.0.0.1:8100/v1 \
VLLM_API_KEY=EMPTY \
OPENAI_API_KEY=EMPTY \
./.venv/bin/lumoserve --port 8100 --proxy-port 8101 auto-research replay-round \
  --workload-file benchmark_blueprints/workloads/multi-family-v5/workload.yaml \
  --baselines 5 \
  --import-candidate output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260424T072126Z/candidates/003/candidate.yaml \
  --rescreens-screen 3 \
  --rescreens-full 1 \
  --holdout-rows 9 \
  --round-root output/auto_research
```

Round artifacts:

- Round path: `output/auto_research/qwen3.5-27b-multi-family-v5-sprint-0-20260425T085916Z`
- Results: `output/auto_research/qwen3.5-27b-multi-family-v5-sprint-0-20260425T085916Z/results.tsv`
- Run log: `output/auto_research/qwen3.5-27b-multi-family-v5-sprint-0-20260425T085916Z/run_log.json`
- Bundle: `output/tuned_configs/multi-family-v5/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/20260425T1020520000_08792ff0.yaml`

H1 output:

- Outcome: `ROUND_BUNDLE_READY`
- Round type: `replay`
- Imported candidate: C003 candidate YAML from Sprint 0
- Holdout validation: `pass`
- Confidence: `exploratory`
- Baseline mean screen: `0.058766`
- Winner screen mean: `0.05574825`
- Improvement over baseline: `-0.0030177499999999996 req/s`
- 95% CI: `[-0.014273117497117722, 0.008237617497117723]`
- Noise floor: `0.015228949799641473`
- Latency above SLO: `true`
- Screen/full consistency: `consistent`
- L2 enforcement coverage: `mode: not_l2`

Measurement rows:

- 5 Screen baselines: `baseline_a` through `baseline_e`
- 1 imported candidate Screen measurement: `import_001`
- 3 Screen rescreens
- 1 Full rescreen

Replay integrity checks:

- Imported candidate byte-compare: identical
- No Codex agent sessions in replay round: `agent_session.jsonl` count was `0`
- `git diff --check`: passed

## Decision

H1 maps to **Case beta** in the hardening plan:

- `confidence` is `exploratory`
- Improvement is negative
- CI crosses zero, so this is not a defensible positive result
- CI upper bound is positive, so this is not Case gamma

Per the hardening plan, the next technical direction is to treat C003 as reference-only and move to L0 substrate scoping/work rather than running more L1/L2 search on this substrate.

