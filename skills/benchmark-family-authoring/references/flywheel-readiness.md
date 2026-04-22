# Flywheel readiness — Layer B (HLD Family-Test-Requirements §4)

Layer A answers "is this a well-calibrated benchmark?". Layer B answers "can LLD-06 ingest this family's rows and train against them without re-reading the scorer?". Both are required before a family ships.

This file is the field guide for Layer B. Read it alongside `benchmark_blueprints/families/proposal-ranking-manager-judgment/` — every section below has a concrete instance in that reference family.

## Why Layer B exists

The event store (HLD §3.1) receives one row per trajectory, and downstream training views read the rows without loading Python. If a family emits only a raw score, the trainer cannot tell:

- whether the score comes from a deterministic check or an LLM-judge (DPO/RL hate LLM-judge noise in the reward);
- which 5-slot milestone fired, so the SFT view can filter to complete-chain trajectories;
- which capability the trajectory exercised, so the curriculum can balance inspect / modify / verify;
- whether the trajectory is integrity-clean (H=0) or should be excluded entirely;
- whether the state transition was a corrective turn or a regression.

Layer B is the wire protocol between the scorer and LLD-06. If you skip it, the rows are scored-but-unusable — the probe benchmarks correctly but the training flywheel starves.

## The 14 items, in author order

Work through these roughly sequentially; later items build on earlier ones.

### 1. Dual-band scorer (§4 item 10, Decision A)

The scorer emits **two** numerical surfaces:

- `P_benchmark` — full 0-100 score, includes every check the rubric declares. This is what the probe and leaderboard read.
- `M_training` — deterministic-only subtotal, normalized to [0, 1]. LLD-06 reads this into SFT, auto-preference, and RL views.

Pattern (lifted from `score_ranking.py`):

```python
@dataclass
class ScorerState:
    breakdown: dict[str, int] = field(default_factory=dict)
    breakdown_bands: dict[str, str] = field(default_factory=dict)  # key → "M" | "P_only"
    raw_score: int = 0       # total (both bands)
    raw_M_score: int = 0     # M band only

    def add(self, key: str, points: int, band: str = "M") -> None:
        self.breakdown[key] = self.breakdown.get(key, 0) + points
        self.breakdown_bands[key] = band
        self.raw_score += points
        if band == "M":
            self.raw_M_score += points

    def final_M_training(self) -> float:
        capped = max(0, min(self.raw_M_score, self.ceiling_cap))
        return round(capped / MAX_M_POINTS, 4)
```

Every `add(key, pts)` call that derives from a regex, JSON field read, sha256 compare, or structured-data assertion defaults to `band="M"`. Every `add(key, pts, band="P_only")` signals "this check is an LLM-judge or LLM-judge-flavored signal — invisible to training loss".

`MAX_M_POINTS` is the sum of the *maximum* possible M-band award across every M-band `add` call. Compute it once, pin it as a module constant with a comment derivation, and never change it after the probe freezes (changing it rescales every historical `M_training` value and poisons longitudinal comparisons).

**Emit both fields.** Keep `score` as a backward-compat alias for `P_benchmark` so downstream consumers reading the old schema don't break:

```python
result = {
    "score": int(final_score),
    "P_benchmark": int(final_score),
    "M_training": state.final_M_training(),
    "raw_score_pre_ceiling": int(state.raw_score),
    "raw_M_pre_ceiling": int(state.raw_M_score),
    "breakdown": {**sorted_breakdown, "__bands": dict(sorted(state.breakdown_bands.items()))},
    # ...
    "schema_version": "cnb55.verify_result.v3",
}
```

### 2. 5-slot milestone vector (§4 items 1–4, HLD §7.5)

Emit five milestones with fixed weights:

| Slot | Name | Weight | Typical semantics |
|---|---|---:|---|
| M1 | Localization | 0.10 | Agent localized the right evidence before acting |
| M2 | Primary fix | 0.20 | Agent produced the artifact in the correct schema |
| M3 | Invariants / anti-shortcut | 0.20 | No readonly mutation, no test tampering, no shim |
| M4 | Functional | 0.20 | Structural rubric gates all pass |
| M5 | E2E integration | 0.30 | Top-line alignment with gold |

Rules that the scorer **must** enforce:

- **No LLM-judge in the milestone layer.** Each slot is L1 (regex), L2 (structured), or L3 (executable-or-learned) per HLD §7.8. If a milestone's only honest check is "ask a model if the agent understood", keep it in the P_only band and pass the slot on a deterministic proxy until L3 is available.
- **H=1 force-fails M3/M4/M5** (HLD §7.7.5). Integrity violations zero the higher milestones regardless of other signals. M1/M2 are allowed to still pass — a bad-faith agent can still localize and produce an artifact.
- **M4 depends on M2, M5 depends on M2.** Declare the dependency in `family.yaml#milestone_dependencies`. A broken artifact cannot produce a valid functional or e2e signal.
- **Emit both shapes.** `milestones` as a flat bool dict (legacy consumers), and `milestone_vector` as `{slots: [{milestone_id, passed_bool, weight}, ...], M_aggregate: float}` (HLD §3.1 shape for LLD-06).

Reference implementation: `compute_5slot_milestones()` + `compute_milestone_vector()` in `verifiers/proposal-ranking-manager-judgment/score_ranking.py`.

### 3. Integrity-flag detector (§4 item 8)

Five H=1 rules are the minimum bar. Declare them in `family.yaml#integrity_rules` and call them 1:1 from the scorer:

```yaml
integrity_rules:
  - id: write_outside_whitelist
    rule: "any file write whose relpath is not under brief/* and is not brief_input.json at root"
    detector: trusted_final_state:wrote_outside_brief
  - id: immutable_slice_mutated
    rule: "any change to sha256 of proposals/, repo_evidence/, ..."
    detector: trusted_final_state:readonly_tree:*
  - id: pytest_shim
    rule: "presence of sitecustomize.py, usercustomize.py, or pytest.py in workspace"
    detector: trusted_final_state:shim:*
  - id: tests_modified
    rule: "sha256 of tests/test_*.py changed"
    detector: trusted_final_state:tests_modified
  - id: network_egress
    rule: ".network_egress_detected marker present in workspace"
    detector: trusted_final_state:network_egress
```

On the scorer side, add the corresponding method:

```python
def raise_integrity(self, rule_id: str) -> None:
    self.integrity_flag = 1
    if rule_id not in self.integrity_rules_fired:
        self.integrity_rules_fired.append(rule_id)
    self.shortcut = True  # flip shortcut_detected so LLD-06 filters the run
```

Every `trusted_final_state` detector calls `state.raise_integrity("<matching id>")`. Verify the mapping by grepping for `raise_integrity(` in the scorer and diffing the `rule_id` strings against the YAML — the count and spelling must match.

**Emit in result:** `integrity_flag: 0|1`, `integrity_rules_fired: [sorted rule ids]`.

### 4. Milestone scripts (§4 item 4)

`verifier_data/<family>/<variant>/milestones/m{1..5}_*.sh` — one script per slot per variant. Each reads `$RESULT_FILE`, exits 0 (pass) / 1 (fail) / 2 (indeterminate).

Share the scripts across variants via symlinks to `verifier_data/<family>/_milestones_shared/` (milestone semantics are family-wide, not variant-specific). Put a `README.md` in the shared dir explaining the exit codes and the H=1 override.

These scripts are the **declarative** surface LLD-06's milestone grader reads. The scorer's embedded `milestone_vector` must agree with what the scripts produce; the CI check runs both and asserts equality.

### 5. Capability tags (§4 item 6, HLD §8.3 + §17.5)

Shared core: `{localize, inspect, modify, verify, respect_invariants}`.

Extended sub-tags (per HLD §17.5) for families that test specific capabilities:

- `inspect:prioritize` — does the agent prioritize relevant evidence?
- `inspect:evidence_triage` — does it distinguish fresh from stale evidence?
- `modify:policy_tradeoff` — does it weigh competing constraints?
- `verify:assumption_honesty` — does it flag unresolved assumptions rather than fabricate?

Declare shared_core + extended blocks as YAML anchors, override per-variant:

```yaml
capability_tags:
  shared_core: &shared_core
    - localize
    - inspect
    - modify
    - verify
    - respect_invariants
  extended: &extended
    - inspect:prioritize
    - inspect:evidence_triage
    - modify:policy_tradeoff
    - verify:assumption_honesty
  per_variant:
    v1-clean-baseline:
      tags: *shared_core
      variant_notes: "Floor check; does not specifically test extended tags."
    v2-noisy-distractor:
      tags:
        <<: *shared_core
      extended_tags:
        - inspect:evidence_triage
      variant_notes: "Adds stale-perf-marker trap; tests evidence triage."
```

### 6. Tool-call overrides (§4 item 6)

If the family ships a CLI, map each subcommand to the 5-capability taxonomy so LLD-06 can tag turn records:

```yaml
tool_call_overrides:
  cnb55-brief:
    schema: inspect
    validate: verify
    submit: modify_terminal   # marks the commit point of the trajectory
```

`modify_terminal` is the sentinel for "this action is the agent's final artifact emission" — LLD-06 uses it to delimit trajectories.

### 7. State-delta rules (§4 item 7, Decision B)

Declare the deliverable's state machine. For a JSON deliverable:

```yaml
state_delta:
  kind: json_deliverable
  path: brief/manager_brief.json
  aggregate_clamp: [0, 1]
  transitions:
    - from: absent
      to: absent
      interpretation: "no progress"
      corrective_turn_bool: false
    - from: absent
      to: present_and_invalid
      interpretation: "first submit, schema-invalid"
      corrective_turn_bool: false
    - from: absent
      to: present_and_valid
      interpretation: "first submit, clean"
      corrective_turn_bool: true
    - from: present_and_invalid
      to: present_and_invalid
      interpretation: "resubmit, still invalid"
      corrective_turn_bool: false
    - from: present_and_invalid
      to: present_and_valid
      interpretation: "fix-up turn"
      corrective_turn_bool: true
    - from: present_and_valid
      to: present_and_invalid
      interpretation: "regression"
      corrective_turn_bool: false
```

`corrective_turn_bool` is read by LLD-06 into `turn_scoring.corrective_turn_bool`. It decides whether a fix-up turn contributes a positive gradient (yes if state transitioned forward, no if it regressed).

### 8. LLM-judge quarantine (§4 item 10, Decision A)

Any check whose correctness requires an LLM (partial_progress rubric, assumption-ledger padding, prose-coherence signals) goes in the `P_only` band. Document it in `family.yaml`:

```yaml
llm_judge_quarantine:
  partial_progress_rubric:
    max_points: 10
    source: verifier_data/<family>/_rubrics/partial_progress.md
    band: P_benchmark_only
    today_implementation: |
      Current scorer uses a deterministic regex floor. Once the LLM-judge
      replaces it, the band assignment is unchanged.
  total_quarantined_points: 10   # of 100
```

Two invariants:

- **Total quarantined points are visible to probe, invisible to training loss.** P_benchmark sees them; M_training does not. This keeps the leaderboard calibrated on the full 100-point rubric while giving RL a clean deterministic reward.
- **The quarantine is permanent.** Even when the LLM-judge is implemented later, the band stays `P_benchmark_only`. LLM-judge contributions never enter `M_training` under any circumstance.

### 9. Seeds + variance escalation (§4 item 11, HLD §17.8)

```yaml
seeds:
  base_count: 2
  variance_escalation:
    stdev_threshold_to_4: 0.10   # M_training stdev at 2 seeds → re-run at 4
    stdev_threshold_to_8: 0.20   # stdev at 4 seeds → re-run at 8
    stdev_flag_high_variance: 0.15   # stdev still ≥ 0.15 at 8 seeds → flag
  current_observed_stdev_M_training: 0.08   # from latest probe
  escalation_currently_active: false
```

Record the stdev observed at the most recent probe. If it crosses the threshold, escalate; record `escalation_currently_active: true` until the stdev drops back under.

### 10. Initial state (§4 item 12)

```yaml
initial_state:
  type: manifest_locked
  ref: manifest.lock.json
  determinism_check: |
    scripts/regen_cnb55_v2.py replays the oracle input under CNB55_SEED=42 and
    verifies byte-identical manager_brief.json.
```

The family's `manifest.lock.json` pins every file hash in the workspace bundle. Any drift invalidates the lock; the probe fires `shortcut_detected: true` on the offending variant.

### 11. Verification matrix (§4 item 5, HLD §5)

Six trajectories × five metrics:

| Trajectory | What it tests | Expected (V1) |
|---|---|---|
| Oracle | Full pass path | P≥90, M≥0.90, G≥0.75, R≥0.98, S_TTC≥1175 |
| Empty | No artifact written | P=0, M=0, pass=False |
| RAWR grounding_stripped | Right pick, fabricated citations | P≈25, ranking_without_grounding ceiling |
| Pick-ceiling (e.g. P3) | Shortcut lever fires | P≈30, named ceiling |
| Top1-wrong | Close-call miss | P in 50–70 band |
| Delete-tests | Integrity trip | P=0, M=0, H=1, M3-M5 zero |

Implement in `scripts/run_verification_matrix.py` (see reference family). Write results to `verification_matrix.md` next to `task_spec.md`.

**Run on V1 AND at least one stress variant.** HLD §8 box 8: "Repeated on at least one stress variant where the family has variant-gated traps." A V1-only matrix does not satisfy acceptance. For the reference family, this is V3; for your family, pick the earliest variant that introduces a variant-gated ceiling. Write results to `verification_matrix_<variant>.md` so both runs are in the tree.

**All six rows should match their bands after the synthesizer is correct.** Earlier drafts of this guide claimed "only three rows are load-bearing; the others may drift". That was true when the synthesizer had field-name bugs (see `references/pitfalls.md` §"Verification-matrix synthesizer mutates the wrong field"). Once the synthesizer mutates exactly the keys the scorer reads, Oracle / Empty / RAWR / Pick-ceiling / Top1-wrong / Delete-tests all land inside their expected bands — the reference family does. Treat drift as a bug to diagnose, not a caveat to document, unless the drift traces to a scorer-side proxy limitation (G / S_TTC against the LLD §8 full formula, see below).

**G, R, S_TTC** are scorer-side proxies. The real values come from turn-level data only LLD-06 sees; the scorer approximates using `integrity_flag` for H, `M_training` as the S-component, and `M1_localization` as the I-component. This is why S_TTC can land below the HLD example (e.g. 1110 vs ≥ 1175) without a rubric bug — it's a proxy artifact. LLD-06 overrides the proxies with live values at ingestion time.

### 12. Multi-mode RAWR (§4 item 5)

Declare at least `grounding_stripped` as implemented. The other two modes — `citation_fabricated` (citations reference real paths but misrepresent contents) and `constraint_named_not_respected` (brief names the constraint but the decision violates it) — can be declared as `declared_not_yet_implemented` with a rationale. Don't ship a family that tests only one RAWR mode without stating it explicitly.

```yaml
rawr_modes:
  grounding_stripped:
    status: implemented
    detector: scorer ceiling `ranking_without_grounding`
  citation_fabricated:
    status: declared_not_yet_implemented
    blocker: "requires file-content LLM-judge; deferred to post-saturation pass"
  constraint_named_not_respected:
    status: declared_not_yet_implemented
    blocker: "requires NLI classifier; not in base stack"
```

### 13. Saturation + renewal plan (§4 item 14, HLD §17.10)

Add a `## Saturation and renewal plan` section to `task_spec.md`. Trigger:

> Mean `P_benchmark` > 80 for 2 consecutive probe rounds at the default model tier → flag `saturation_renewal_due` in `benchmark_run.md`.

Queue **at least two** renewal mechanisms:

- A new variant (e.g. mid-run staffing change, contradictory-evidence cross-corpus) that adds a capability dimension the current V1-V5 don't test.
- Retire the floor-check variant (usually V1) and replace with a harder floor derived from V2 minus one trap.

Downstream consumers filter on `family_status_at_write_time=active`; retired variants remain in the event store for longitudinal regression checks.

### 14. Grader_ref + milestone_config_ref (§4 item 13)

```yaml
grader_ref: verifiers/<family>/score_<domain>.py
milestone_config_ref: verifier_data/<family>/_milestones_shared/
```

LLD-06 uses these to lazy-load the scorer at ingestion time. If either path is stale, the event-store row fails to ingest and ends up in the quarantine queue. Re-run `scripts/refresh_manifest_lock.py` whenever either path moves.

## Validation checklist (run before declaring Layer B done)

- [ ] `python3 -c "import ast; ast.parse(open('verifiers/<family>/score_*.py').read())"` succeeds.
- [ ] `python3 scripts/regen_cnb55_v2.py` passes (oracle ≥ 90, empty = 0, shortcut ≤ 30 for all variants).
- [ ] Scorer result JSON against an oracle brief has keys: `P_benchmark`, `M_training`, `raw_M_pre_ceiling`, `milestones`, `milestone_vector`, `integrity_flag`, `integrity_rules_fired`, `breakdown.__bands`, `schema_version: "cnb55.verify_result.v3"`.
- [ ] Same run with tests/ tampered: `integrity_flag == 1`, `M_training == 0.0`, `M3_invariants / M4_functional / M5_e2e` all False.
- [ ] `verifier_data/<family>/<variant>/milestones/m{1..5}_*.sh` exist (via symlink) and agree with `milestone_vector.slots[*].passed_bool` for at least the Oracle and H=1 cases.
- [ ] `scripts/run_verification_matrix.py --variant <v1-id>` writes `verification_matrix.md` with all 6 trajectories; **every row** (Oracle / Empty / RAWR / Pick-ceiling / Top1-wrong / Delete-tests) matches HLD §5 expected band. If a row drifts, diagnose — usually a synthesizer field-name bug (see `pitfalls.md`).
- [ ] `scripts/run_verification_matrix.py --variant <stress-variant-id>` writes `verification_matrix_<stress>.md` with all 6 rows in band. HLD §8 box 8 requires this stress-variant rerun; V1-only matrix is incomplete.
- [ ] Every `raise_integrity("<id>")` call site in the scorer has a matching `integrity_rules[].id` in `family.yaml`.
- [ ] Every `add(key, pts, band="P_only")` call in the scorer has a matching `llm_judge_quarantine.<name>` entry.
- [ ] `family.yaml` declares all 14 sections and validates as YAML (`python3 -c "import yaml; yaml.safe_load(open('family.yaml'))"`).
- [ ] `benchmark_run.md` has an `attempt_NN — Layer B flywheel-readiness upgrade` section enumerating the shipped changes.

If any box fails, the family is not Layer B done — it is Layer A done. Do not ship until Layer B passes too; the training flywheel is the other half of the product.
