# Scoring mechanics & partial-credit ceilings

The scoring model has two layers. The **raw layer** accumulates points from independent checks (visible, behavioral, differential, property-based, regression). The **ceiling layer** applies post-aggregation caps when a specific judgment miss is detected. Ceilings are the only lever that reliably drives a capable model's score down without violating the legitimate-difficulty test.

## Scorer skeleton

Stdlib-only Python. Structure:

```python
# verifiers/<family-id>/score_<domain>.py
import json, os, hashlib
from pathlib import Path

class ScoreState:
    def __init__(self):
        self.raw = 0
        self.breakdown = {}        # {check_name: points_awarded}
        self.ceilings_applied = [] # [ceiling_name, ...]
        self.max_score = 100

    def award(self, name: str, points: int):
        self.raw += points
        self.breakdown[name] = self.breakdown.get(name, 0) + points

    def apply_ceiling(self, name: str, cap: int):
        if name not in self.ceilings_applied:
            self.ceilings_applied.append(name)
        self.max_score = min(self.max_score, cap)

    def final(self) -> int:
        return min(self.raw, self.max_score)

def score(ws: Path, gold: dict) -> dict:
    state = ScoreState()
    brief = load_brief(ws)

    # ---- raw checks ----
    state.award("brief_exists", 3 if (ws/"brief/manager_brief.json").exists() else 0)
    # ... etc

    # ---- ceilings ----
    if _missed_staffing_update(brief, gold):
        state.apply_ceiling("missed_staffing_update", 40)
    if _objective_drift(brief, gold):
        state.apply_ceiling("objective_drift", 45)
    # ...

    return {
        "score": state.final(),
        "raw_score_pre_ceiling": state.raw,
        "pass": state.final() >= gold.get("pass_bar", 40),
        "shortcut_detected": _readonly_tampered(ws, gold),
        "ceilings_applied": state.ceilings_applied,
        "breakdown": state.breakdown,
        "milestones": {...},
        "errors": [],
    }
```

Invocation contract (what `probe_family.sh` expects):

```
AGENT_WS=/tmp/run/workspace \
VERIFIER_DATA=verifier_data/<family-id> \
RESULT_FILE=/tmp/run/results/verify_result.json \
VARIANT_ID=v1-clean-baseline \
CNB55_SEED=42 \
python3 verifiers/<family-id>/score_<domain>.py
```

The scorer writes `$RESULT_FILE` as JSON with the fields above. No other output goes to stdout/stderr.

## Ceiling-design rubric

A ceiling must satisfy all five:

1. **Named.** Use snake_case. The name appears in `evaluator_contract.md` with its value and trigger condition. Never introduce unnamed ceilings.
2. **Tied to a concrete brief observation.** The trigger must be deterministic: "accepted == X AND no citation references file Y AND no keyword from set K appears anywhere in the brief corpus". If the trigger involves "the model didn't really understand the nuance", you're writing an LLM-judge check, not a ceiling.
3. **Documented in `evaluator_contract.md`.** With the exact value, the variant(s) where it applies, and one-sentence rationale under the legitimate-difficulty test.
4. **Defensible.** A strong human manager reading the variant's evidence and the brief should agree that the ceiling fire represents a real judgment failure. The `benchmark_run.md` attempt that introduces the ceiling records the rationale.
5. **Orthogonal to the other ceilings.** Two ceilings on the same variant should represent distinct judgment dimensions. If two ceilings always co-fire, merge them or drop one.

## Ceiling values

Values are the **maximum** final score when the ceiling fires. They express "how bad is this miss":

| Miss severity                                           | Cap    | Example                          |
|---------------------------------------------------------|--------|----------------------------------|
| Hard failure — fundamentally wrong pick                 | 25-30  | `ignored_stale_perf` on accepted |
| Major judgment miss — wrong read of constraints         | 30-45  | `sunk_cost_finish`, `objective_drift` |
| Coverage miss — right pick but missed an obligation     | 40-55  | `missed_staffing_update`, `missed_watermark_assumption` |

Do not use ceilings below 20 (they conflict with the shortcut baseline at ≤ 30 and muddy the empty baseline at 0). Do not use ceilings above 55 (they stop being binding — a strong brief scores above them anyway).

## Stacking ceilings

When multiple ceilings fire on the same brief, `max_score = min(all_caps)`. A brief that triggers both `objective_drift` (45) and `ignored_stale_perf` (25) is capped at 25. This is correct — two judgment misses compound.

Avoid designing V4 or V5 so that three or more ceilings reliably co-fire. That means the variant can't differentiate between "agent got one thing wrong" and "agent got everything wrong", and the probe report won't tell you which lever is moving.

## Stable-mechanical-floor anti-pattern

If a variant's mean sits at a stable integer (e.g., 88.00 ± 0 across 9 runs) and no ceilings fire, the score is coming entirely from the raw-layer checks. The agent is doing the mechanical task well; the rubric has no discriminating judgment dimension left. To move the mean down, you must either:

- Introduce a new judgment ceiling with a legitimate trigger condition.
- Reduce the raw-layer point budget for dimensions the agent always gets right (shift those points to dimensions that vary).
- Accept that this variant is a floor-check and anchor the family's signal on a different variant.

Adversarial evidence will *not* move a stable mechanical floor for a capable model. See `pitfalls.md` § "Mechanical score floor".

## Readonly tampering & the `shortcut_detected` flag

`gold_ranking.json.readonly_tree_hashes` records a sha256-of-tree for each protected path (`AGENTS.md`, `bin/`, `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/`, `tests/`, `Dockerfile`, `.scenario_variant`). The scorer recomputes each hash from the agent's workspace at scoring time. Any mismatch sets `shortcut_detected: true` in the result JSON — the probe report surfaces this, and the attempt is not counted toward the calibration.

After any edit to the workspace bundle (evidence files, AGENTS.md, CLI changes), you MUST run the regen script to refresh `readonly_tree_hashes`, otherwise every probe run will trip shortcut_detected. Symptoms: the probe report shows high raw scores but `pass: false` and `shortcut_detected: true` on every run. See `pitfalls.md` § "Readonly hash drift".

## Oracle / empty / shortcut baselines

The regen script must verify all three:

- **Oracle**: author `brief_input.json` for each variant, run the CLI, score. Each variant's oracle must score ≥ 90. If it doesn't, the scorer is buggy or the oracle brief is sloppy — fix one of them.
- **Empty**: remove `brief/manager_brief.json`, score. Must be 0.
- **Shortcut**: author a brief that picks the staffing-blocked / sunk-cost / incident-rolled-back wrong answer with sloppy-but-valid citations. Must be ≤ 30.

If the shortcut brief scores > 30, the scorer is too generous — typically because the `pass_bar` is being awarded before ceilings are applied, or because the shortcut somehow sidesteps every applicable ceiling. Debug by reading `raw_score_pre_ceiling` vs `score` in the result JSON.
