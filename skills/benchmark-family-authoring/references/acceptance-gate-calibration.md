# Acceptance gates & calibration loop

§10.1 of the CNB-55 authoring spec defines four gates a family must satisfy before the freeze. This file explains what each gate measures, how to run the probe that evaluates them, and how to decide between "keep hardening" and "accept the signal as-is".

## The four gates

For a family with five variants and N runs per variant:

| Gate                            | Default value      | What it measures                                                                |
|---------------------------------|--------------------|---------------------------------------------------------------------------------|
| `family_mean ∈ [lo, hi]`        | `[15, 25]`         | The family discriminates — capable models score in the partial-credit zone.    |
| `max_variant_mean ≤ cap`        | `40`               | No single variant is a freebie.                                                 |
| `min_variant_mean ≤ hard_floor` | `10`               | At least one variant genuinely fails — confirms the rubric has a real hard end. |
| Monotonic V1 ≥ V2 ≥ V3 ≥ V4 ≥ V5 | `±3 tolerance`    | The variants are progressing in difficulty, not just scrambled.                 |

Plus three baseline gates (enforced by the regen script, not the probe):

- Oracle ≥ 90 for every variant.
- Empty brief = 0 for every variant.
- Shortcut brief ≤ 30 for every variant.

## Probe invocation

```bash
# From the FlyWheel repo root, on the calibration Mac (codex CLI installed).
N=3 FAMILY=<family-id> bash scripts/probe_family.sh
python3 scripts/probe_report.py report/probe_runs.jsonl --probe-run-id <timestamp>
```

`probe_family.sh`:
- Stages a fresh copy of each variant's workspace bundle to `/tmp/cnb55_probe/<run-tag>/workspace/`.
- Runs `codex exec --cd <ws> --skip-git-repo-check --sandbox workspace-write --color never --ephemeral <PROMPT>` with `CODEX_TIMEOUT=1800` per run.
- Scores via the family's scorer and appends one JSONL record per run to `report/probe_runs.jsonl`.
- Logs per-run codex output to `report/probe_run_logs/<run-tag>.log`.

`probe_report.py`:
- Filters `probe_runs.jsonl` to the given `--probe-run-id`.
- Emits the per-variant mean/stdev/min/max/scores/ceilings table.
- Evaluates the four §10.1 gates and prints `overall: READY` or `overall: HARDEN NEEDED`.

The full loop (regen → probe → report → harden → regen) takes ~45 min wall per iteration. Budget at least 3-5 iterations before declaring a calibration stalled.

## Fire-and-forget probe launcher

The probe is long-running (~38 min wall for 15 runs) and sometimes needs to survive an AppleScript/osascript session. Use the `_run_probe_02b.sh` pattern:

```bash
#!/usr/bin/env bash
# scripts/_run_probe_XX.sh — self-daemonizing launcher.
# Usage: bash scripts/_run_probe_XX.sh start
set -u
MODE="${1:-start}"
LOG=/tmp/cnb55_probe_XX.log
DONE=/tmp/cnb55_probe_XX.done
if [ "$MODE" = "start" ]; then
  rm -f "$DONE" "$LOG"
  nohup "$0" _inner >/dev/null 2>&1 &
  echo $! > /tmp/cnb55_probe_XX.pid
  exit 0
fi
cd /path/to/Lumo_FlyWheel
{
  echo "=== probe start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  N=3 bash scripts/probe_family.sh
  rc=$?
  echo "EXIT=$rc" > "$DONE"
} >"$LOG" 2>&1
```

AppleScript `do shell script` can't embed a literal `&` in a string without triggering the reserved-operator parser, so the launcher self-daemonizes inside the child process. Poll `/tmp/cnb55_probe_XX.done` for completion; read `/tmp/cnb55_probe_XX.log` for streaming progress.

## Accept-or-widen decision tree

After a probe run fails a gate, decide whether to harden or widen before touching anything:

```
Is oracle < 90 for any variant?
├── YES → scorer or oracle bug; fix. This is not a §10.1 question.
└── NO  → continue

Does at least one variant mean ≤ 10?
├── NO  → V5 (or the hardest variant) is not pressuring enough. Check whether
│          ceilings are firing. If ceilings fire but score is high anyway,
│          check ceiling values (may be too generous). If ceilings don't
│          fire, the trigger conditions don't match the agent's failure mode.
└── YES → continue

Is max_variant_mean ≤ cap?
├── NO  → V1-V3 cluster too high. Check for rubric leakage in AGENTS.md and
│          in evidence files (see pitfalls.md). If no leakage, this is a
│          mechanical-floor case — go to the last question below.
└── YES → continue

Is family_mean in [lo, hi]?
├── YES → all gates pass; ship.
└── NO  → one of:
          (a) widen the window for this family (legitimate if docs show the
              family is at the frontier of the calibration model),
          (b) add one new judgment ceiling targeting a specific miss,
          (c) re-check V4/V5 design — they carry the bulk of the mean-down
              pressure and may need more ceiling weight.
```

## When to stop hardening

Stop and widen the window (or accept the signal) when:

1. The `benchmark_run.md` log shows two consecutive hardening attempts moved the family mean by < 5 points.
2. Every remaining hardening idea requires a ceiling that fails the legitimate-difficulty test in `SKILL.md`'s "one hard rule".
3. Spot-checking agent briefs shows them doing legitimate manager work — not cheesing the rubric, not missing honest judgment calls.

In that state, the family's honest signal is its current calibration curve. Document this explicitly in `benchmark_run.md` and either:

- Widen `[lo, hi]` for this specific family in the family's `evaluator_contract.md`, recording the frontier-difficulty rationale.
- Drop the family from the calibration-model freeze gate and treat it as a probe-only family for stronger models.
- Accept the family as-is and move on, noting that this family's signal lives in V4/V5 variance rather than in V1-V3 mean.

Widening is not a failure mode. A family where a capable model scores 65 across variants is still measuring something real — it just isn't measuring "can this model do the task at all", which is what the default window is calibrated for.

## Probe result JSONL format

One line per run:

```json
{
  "probe_run_id": "20260421T022540Z",
  "variant": "v1-clean-baseline",
  "run_index": 1,
  "codex_exit": 0,
  "codex_seconds": 148,
  "workspace_path": "/tmp/cnb55_probe/<run-tag>/workspace",
  "score": 88,
  "raw_score_pre_ceiling": 88,
  "pass": true,
  "shortcut_detected": false,
  "ceilings_applied": [],
  "milestones": {...},
  "breakdown": {...},
  "errors": []
}
```

`score` is the final capped value; `raw_score_pre_ceiling` is the pre-ceiling sum. When those differ, read `ceilings_applied` to see which cap bound the score.

## Spot-checking briefs

Every `attempt_NN` section in `benchmark_run.md` should include a spot-check of at least one agent brief. Read:

- `/tmp/cnb55_probe/<run-tag>/workspace/brief/manager_brief.json` — what the agent submitted.
- `report/probe_run_logs/<run-tag>.log` — what codex printed during the run.

Look for: which citations the agent picked, whether the primary_risk statement mentions the variant-specific constraints, whether the assumption_ledger has genuine "missing" rows, and whether ranking rationales name-check the real evidence files. This qualitative read is how you distinguish "agent is doing the task well" from "agent is hitting the rubric without really understanding" — both can produce the same score.
