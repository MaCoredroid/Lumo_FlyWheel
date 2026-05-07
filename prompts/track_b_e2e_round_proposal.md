# Round {{round}} Proposal

## Hypothesis
{{hypothesis}}

## Targeted regime / diagnosis
{{targeted_regime_diagnosis}}

## Config delta (YAML diff vs Round {{prior_round}})
```yaml
{{config_delta_yaml_diff}}
```

## Predicted impact
- Regime: {{predicted_regime}}
- Predicted accepted/draft change: {{accepted_per_draft_baseline}} -> {{accepted_per_draft_expected}}
- Predicted DRAM_ACTIVE change: {{dram_active_baseline}} -> {{dram_active_expected}}
- Predicted median wallclock change: {{median_wallclock_baseline}} -> {{median_wallclock_expected}}

## Cheap preflight commands
1. `curl -sSf http://127.0.0.1:9950/health`
2. `curl -sSf http://127.0.0.1:9950/metrics | grep -E 'spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total'`
3. `.venv/bin/python scripts/preflight_track_b_e2e.py --out output/track_b_e2e/round_{{round}}/preflight_audit.json`
4. `.venv/bin/python scripts/run_track_b_e2e_task.py transcript-merge-regression/v1-clean-baseline --round {{round}} --attempt 1 --runtime-config-hash {{runtime_config_hash}} --codex-command-template "{{codex_command_template}}"`
5. `.venv/bin/python scripts/build_track_b_e2e_summary.py task --round {{round}} --task-dir output/track_b_e2e/round_{{round}}/transcript-merge-regression__v1-clean-baseline/run_01 --family transcript-merge-regression --variant v1-clean-baseline --runtime-config-hash {{runtime_config_hash}} --cold-completion-discarded --cache-reset-verified --protocol-hash-match --generation-volume-within-band --sample-hash-match --clock-skew-ms-p99 {{clock_skew_ms_p99}} --trace-emitter-correctness-verified-at {{trace_emitter_correctness_verified_at}} --write-untrusted-diagnostic`
6. `.venv/bin/python scripts/run_track_b_tool_call_gate.py --family policy-aware-request-resolution --variant v1-clean-baseline --mode auto --cases 4`

## Cheap preflight pass criteria
- vLLM health check returns 200.
- spec_decode metrics are exposed and increment on non-prefill turns.
- The smoke task exits 0 and emits `task_start`, `turn_start`, `turn_end`, and `task_end`.
- Every `turn_start` has `vllm_request_id`.
- The smoke summary is produced as an untrusted diagnostic unless it has explicit 3-run wallclock evidence.
- Tool-call XML auto mode remains 4/4.

## Full measurement command
`.venv/bin/python scripts/run_track_b_e2e_round.py --round {{round}} --runtime-config-hash {{runtime_config_hash}} --codex-command-template "{{codex_command_template}}" --clock-skew-ms-p99 {{clock_skew_ms_p99}} --trace-emitter-correctness-verified-at {{trace_emitter_correctness_verified_at}} --protocol-hash-match`

## Correctness caveat checklist
- [ ] B-1 batch equivalence retained
- [ ] B-2 workload equivalence retained
- [ ] B-3 longer-prefix equivalence retained
- [ ] Tool-call XML auto-mode 4/4 retained
- [ ] All 13 tasks complete (exit_code 0 + task_score recorded)
- [ ] Aggregate task_score does not regress more than 5% vs Round {{prior_round}}
- [ ] No new spec_decode crashes

## Truthful measurement contract
- [ ] Rule 1: Cold completion discarded
- [ ] Rule 2: Output cap hits counted and reviewed
- [ ] Rule 3: Median of N >= 3 runs, not single-run headline
- [ ] Rule 4: Workspace and prompt hashes match Round 0
- [ ] Rule 5: Prefix cache reset verified before each run
- [ ] Rule 6: DCGM sampler dropout < 1%
- [ ] Rule 7: Codex/vLLM clock skew p99 < 100 ms
- [ ] Rule 8: Codex task completion verified by `task_end`
- [ ] Rule 9: Wallclock is wall-to-wall `task_end.ts - task_start.ts`
- [ ] Rule 10: Measurement protocol hash matches prior comparable round
- [ ] Rule 11: Generation-token volume guard passed
- [ ] Rule 12: Spec_decode accepted/draft metrics captured per eligible turn
- [ ] Rule 13: No silent fallback to vanilla decode
- [ ] Rule 14: Trace emitter byte-equality correctness is verified
- [ ] Rule 15: 13-task sample hash matches Round 0
