# Serving Auto-Research Round Report - 2026-04-23

## Scope

This report covers the Sprint 0 serving auto-research round run for:

- model: `qwen3.5-27b`
- family: `proposal-ranking-manager-judgment`
- workload: `benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml`
- HLD target: the first `L1-only` auto-research loop from `docs/HLD-Serving-Backend-AutoResearch-v0_1.md`

The user direction for this round was:

- use a manager skill to run the round
- use spawned subagents with `gpt-5.4` and `high` reasoning
- prioritize live verification over adding extra corner-case tests
- verify against a real `vllm + qwen` backend on the `proposal-ranking-manager-judgment` family

## How The Round Was Run

I first wrote a local orchestration skill at `skills/auto-research-round-manager/SKILL.md`. That skill fixed the round contract to one active research worker at a time and required `gpt-5.4` with `high` reasoning for every spawned subagent.

The round itself was executed as a closed loop with four steps:

1. Run the offline Sprint 0 auto-research sweep to produce a tuned bundle.
2. Load that bundle into serving state.
3. Start the real local vLLM/Qwen serving path and wait for `/health`.
4. Run the real live family verifier through the `/v1/responses` path.

The concrete commands used by the skill were:

```bash
.venv/bin/lumoserve \
  --registry model_registry.yaml \
  --state-root "$STATE_DIR" \
  --tuned-config-root "$BUNDLE_DIR" \
  auto-research run qwen3.5-27b \
  --family-id proposal-ranking-manager-judgment \
  --workload-file benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml
```

```bash
.venv/bin/lumoserve \
  --registry model_registry.yaml \
  --state-root "$STATE_DIR" \
  load-tuned-config "$BUNDLE"
```

```bash
LUMO_HOST_MEMORY_RECOVERY=0 \
LUMO_SKIP_VRAM_WAIT=1 \
LUMO_MIN_GPU_MEMORY_UTILIZATION=0.05 \
VLLM_API_KEY=EMPTY \
.venv/bin/lumoserve \
  --registry model_registry.yaml \
  --container-name lumo-vllm-sprint0 \
  --port 8100 \
  --proxy-port 8101 \
  --logs-root "$LOGS_DIR" \
  --triton-cache-root "$TRITON_DIR" \
  --state-root "$STATE_DIR" \
  serve qwen3.5-27b
```

```bash
VLLM_API_KEY=EMPTY \
.venv/bin/python scripts/run_live_proposal_family.py \
  --base-url http://127.0.0.1:8101/v1 \
  --model qwen3.5-27b \
  --json
```

## Winning Bundle

The round produced bundle:

- `output/sprint0_live/tuned_configs/proposal-ranking-manager-judgment/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/20260423T0449110000_48cb401b.yaml`

Bundle contents:

- `max_num_seqs: 1`
- `max_num_batched_tokens: 6144`
- `enable_chunked_prefill: true`
- `enable_prefix_caching: false`
- `gpu_memory_utilization: 0.3`
- `max_model_len: 32768`
- `kv_cache_dtype: fp8_e5m2`

Recorded objective:

- `metric: sustained_concurrent_eval_threads_at_L_ceiling`
- `value: 1`
- `L_ceiling_ms: 35000`
- `measurement_window_minutes: 25`

The corresponding run log says:

- `status: produced_bundle`
- `stopping_reason: hard_infeasibility_oom`
- `baseline_value: 0`
- `best_value: 1`
- `best_candidate_label: candidate-01`

The search trace shows six evaluated configurations total:

- `baseline`
- `candidate-01`
- `candidate-02`
- `candidate-03`
- `candidate-04`
- `candidate-05`

Only `candidate-01` was feasible. The baseline and three larger candidates were marked OOM. One smaller candidate failed the rollout floor.

## Live Verification Result

After loading the bundle, the real vLLM/Qwen serve path reached healthy state and the live family verification run completed against the proxy endpoint.

Passing live verification artifact:

- `output/live_proposal_family/proposal-ranking-manager-judgment/v1-clean-baseline/20260423T045455Z/result.json`

Verifier result:

- `pass: true`
- `score: 78`
- `codex_result.returncode: 0`

Important interpretation:

- `78` was not the optimization target
- it was a correctness gate for the family run
- the actual optimization target for this round was concurrency, recorded as `best_value: 1`

Observed live serve behavior from `output/sprint0_live/logs/vllm_qwen3.5-27b.log` was roughly:

- prompt throughput around `1.9k` to `2.3k tokens/s`
- generation throughput around `7.4` to `7.5 tokens/s` during steady single-request sections
- active concurrency effectively `1`

## What Was Actually Optimized

This round optimized for:

- sustained concurrent eval threads under the configured latency ceiling

This round did not optimize for:

- family benchmark score
- best-of-N answer quality
- corner-case unit coverage

The live family score was used only as a live end-to-end gate after the bundle had already been chosen.

## What Matched The HLD

This run did match several important HLD requirements:

- it was run for the Sprint 0 scope
- it tuned `L1` vLLM config only
- it emitted a frozen tuned-config bundle
- it loaded that bundle into serving state
- it brought up a real local vLLM/Qwen serving path
- it performed a live family verification through the real responses endpoint

## Where This Round Still Diverged From The HLD

This round was not a full implementation of the HLD auto-research agent.

Current implementation evidence:

- `src/lumo_flywheel_serving/auto_research.py` uses `SyntheticMeasurementHarness`
- the harness computes feasibility and objective values from fixed formulas over candidate flags
- it does not run the full HLD measurement harness against real `/metrics` deltas for each candidate

The HLD describes:

- an empirical eval-workload distribution captured from a seed run
- a synthetic load generator that mirrors that distribution
- per-iteration restart and measured latency windows
- optimization over measured concurrency at the latency ceiling

The current Sprint 0 implementation instead uses:

- a fixed workload yaml with summary values
- a synthetic evaluator in Python
- a small candidate sweep with synthetic feasibility and capacity calculations
- one final live verifier run after bundle selection

So the honest summary is:

- this round was a real closed loop for bundle production and live serving verification
- but the inner measurement loop was still a Sprint 0 proxy, not the full document-defined measurement harness

## Artifacts

- Skill: `skills/auto-research-round-manager/SKILL.md`
- Workload: `benchmark_blueprints/families/proposal-ranking-manager-judgment/serving_workload.yaml`
- HLD: `docs/HLD-Serving-Backend-AutoResearch-v0_1.md`
- Run log: `output/sprint0_live/tuned_configs/proposal-ranking-manager-judgment/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/run_1776919751_qwen3_5-27b/run_log.json`
- Search trace: `output/sprint0_live/tuned_configs/proposal-ranking-manager-judgment/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/run_1776919751_qwen3_5-27b/search_trace.json`
- Measurement trace: `output/sprint0_live/tuned_configs/proposal-ranking-manager-judgment/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/run_1776919751_qwen3_5-27b/measurement_trace.json`
- Tuned bundle: `output/sprint0_live/tuned_configs/proposal-ranking-manager-judgment/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/20260423T0449110000_48cb401b.yaml`
- Serve log: `output/sprint0_live/logs/vllm_qwen3.5-27b.log`
- Live result: `output/live_proposal_family/proposal-ranking-manager-judgment/v1-clean-baseline/20260423T045455Z/result.json`
- Live verify result: `output/live_proposal_family/proposal-ranking-manager-judgment/v1-clean-baseline/20260423T045455Z/verify_result.json`

## Bottom Line

This time, the auto-research loop was run as:

- an orchestrated Sprint 0 `L1-only` tuning pass
- followed by real bundle activation
- followed by real local serve bring-up
- followed by a real live family gate

It did produce a working bundle and pass the live family gate, but the optimization core was still the repo's synthetic Sprint 0 harness, so this should be treated as a live-verified proxy round rather than a full HLD-complete auto-research implementation.
