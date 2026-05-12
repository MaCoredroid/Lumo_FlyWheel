# Track B v4a_v2 — Task 1 closeout: responses-sdk-adapter-cutover

First complete task under the docker-isolated harness. All 4 attempts
finished; per-attempt artifacts (excluding 80 MB dcgm samples each)
plus grader scores are committed. P2 continues on task 2; this report
freezes task 1.

## Run context

| Field | Value |
|---|---|
| Round | `output/track_b_e2e_v4a_v2/round_0/` |
| Task | `responses-sdk-adapter-cutover/v1-clean-baseline` |
| Per-attempt budget | 1800 s (30 min) |
| Repeats | 4 |
| Codex CLI | 0.128.0, inside `codex-runner:v1` docker container |
| Container args | `--rm --network=host -u <uid:gid> -v <ws>:/workspace -e HOME=/tmp -w /workspace` |
| Codex sandbox flag | `--dangerously-bypass-approvals-and-sandbox` (container IS the sandbox) |
| Proxy | `127.0.0.1:8022` (LUMO_PROXY_NONSTREAM_BYPASS=1, LUMO_PROXY_AUTO_CONTINUE=1, retries=5) |
| Runtime config hash | `sha256:5ae88ac4…` |

## Per-attempt wallclock + exit code

| Attempt | elapsed | codex_rc | normalized rows | outcome |
|---|---:|---:|---:|---|
| run_01 | 1800 s (30:00) | 124 | 110 | wall-budget timeout |
| run_02 | 1800 s (30:00) | 124 | 127 | wall-budget timeout |
| run_03 | 1800 s (30:00) | 124 | 201 | wall-budget timeout |
| run_04 | 1479 s (24:39) | 0   | 149 | clean exit |

Mean: 28.2 min/attempt. 3 of 4 hit the 1800 s wall. The `TimeoutExpired`
catch added in this session ensured all 4 attempts still wrote
`runner_metadata.json` + `codex_trace.jsonl` + `vllm_request_metrics.jsonl`.

## Milestone-grader scores

| Attempt | M_aggregate | P_benchmark | M_training | Integrity | Ceilings applied |
|---|---:|---:|---:|---:|---|
| run_01 | 0.30 | 0  | 0.00 | **1** | `responses_alias_blindness` |
| run_02 | 0.30 | 0  | 0.00 | **1** | `responses_alias_blindness` |
| run_03 | 0.70 | 35 | 0.35 | 0 | `responses_alias_blindness` |
| run_04 | 0.70 | 20 | 0.20 | 0 | `compatibility_shim_left_live`, `responses_alias_blindness`, `visible_only_cutover` |

- Pass threshold = `P_benchmark >= 65`. **0 / 4 pass** for this task.
- Mean `P_benchmark` = 13.75; median = 10.
- Two attempts (runs 01, 02) tripped an integrity flag.
- Even on the "clean exit" attempt (run_04), three score-ceiling rules
  fired, including `compatibility_shim_left_live` and
  `visible_only_cutover` — the model did the visible/declarative edits
  but left a compatibility shim in place that the hidden grader catches.

Score-component view (run_04 representative breakdown — M = milestone-bound):

| Component | Pts | Band |
|---|---:|---|
| config.responses_mode | 5 | M |
| config.responses_wire | 10 | M |
| docs.event_ordering | 5 | M |
| docs.tool_result_correlation | 5 | M |
| docs.variant_complete | 5 | M |
| hidden.interleaved_order | 15 | M |
| hidden.replay_roundtrip | 15 | M |
| visible.pytest | 10 | M |

Milestone vector (run_04):

| Milestone | Weight | Passed |
|---|---:|:---:|
| M1_localization | 0.10 | ✓ |
| M2_primary_fix | 0.20 | ✓ |
| M3_invariants | 0.20 | ✓ |
| M4_functional | 0.20 | ✓ |
| M5_e2e | 0.30 | ✗ |

M5_e2e is the binding milestone the model is missing — it requires the
end-to-end variant-complete behavior (legacy path removed, alias
normalization).

## What the model actually changed (run_04 changed_paths)

- `config/runtime.toml`
- `docs/migrations/responses-cutover.md`
- `src/incident_handoff/adapter.py`
- `src/incident_handoff/render.py`
- `src/incident_handoff/replay.py`

The model touched the right files. The pattern across attempts is
consistent: it makes the visible / declarative changes (config + docs +
adapter wiring) but leaves the compatibility shim in `legacy.py` /
function-alias normalization paths, which the hidden grader cares
about.

## Comparison: validity now vs the §18 file-presence claim

§18 reported `responses-sdk-adapter-cutover` as one of the highest
"real file output" producers (64 files written in the validation
single-attempt sweep). The proxy stack is working — codex is plainly
doing real cross-file edits. The §18 file-count signal said "model can
do work in this workspace under the working harness." It did **not**
say "model passes the grader." Now we have both: structural-output
proof (still strong) AND milestone-grader proof (0/4 pass on this
task).

That gap — model writes plausible-looking edits but misses hidden
correctness — is the binding constraint for qwen3.5-27b on this corpus
under this harness, exactly as the §19 closeout warned. Re-baseline
arithmetic now uses real `M_aggregate` per attempt, not deferred-
to-exit-code defaults.

## What's committed for this task

Per `run_NN/`:
- `prompt.md` — built from AGENTS.md + .scenario_variant
- `codex_stdout.log` — full JSONL trace of codex turns (216–292 KB)
- `codex_stderr.log` — stderr (negligible)
- `codex_trace.jsonl` — synthesized task_start/task_end frames
- `runner_metadata.json` — elapsed_s, codex_exit_code,
  vllm_request_metrics_capture summary, warmup_pass, etc.
- `vllm_per_turn.json` — joined per-request decode metrics
- `vllm_request_metrics.jsonl` — sliced proxy capture (128–232 KB)
- `vllm_metrics_pre.txt` / `vllm_metrics_post.txt` — Prometheus
  snapshots bracketing the attempt
- `workspace/` — the codex-modified workspace (168 KB)
- `grader_result.json` — the verifier output for this attempt

Excluded from git: `dcgm_samples.jsonl` (~80 MB each, 320 MB total).
Recover from `/tmp/lumo-l0c-fp8-cutlass-run30-logs/` if needed for
DCGM-driven post-mortem.

Round-level prefix in `round_0/`:
- `preflight_audit.json`, `codex_system_prompt.json`,
  `codex_system_prompt_decomposition.json`, `round_warmup_pass.json`.

## Open at task close

- P2 continues on task 2 (transcript-merge-regression). 10 tasks
  pending after this.
- The grading was post-hoc here. Wiring graders into the round driver
  (so `runner_metadata.json` carries `task_score`) is task #51 / loop
  item 2; still deferred to a closeout sweep after P2 lands.
- 0/4 pass for the **heaviest** task is a data point, not the task
  score for the corpus. Lighter tasks (transcript-merge, dead-flag,
  security-audit) will dominate the final v4a_v2 pass-rate number.
