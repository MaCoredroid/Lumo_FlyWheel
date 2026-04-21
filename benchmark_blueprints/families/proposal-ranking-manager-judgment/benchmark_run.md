# Benchmark Run — proposal-ranking-manager-judgment

Family run protocol for CNB-55 Track 10 (Strategic Management & Long-Horizon Evolution), family `proposal-ranking-manager-judgment`.

## Model under test

```
codex exec --model gpt-5.4 --reasoning-effort high
```

All variants are scored with `CNB55_SEED=42` and the scorer at `verifiers/proposal-ranking-manager-judgment/score_ranking.py`. The LLM-judge portion of the grader uses the same model (gpt-5.4) at `temperature=0.0`, two-seed average.

## Variants and target scores

| Variant | Difficulty axis added | Pass bar | Probe target (codex gpt-5.4 high) |
|---|---|---|---|
| v1-clean-baseline | none | 40 | ~27 |
| v2-noisy-distractor | stale perf distractor (P5 with Jan bench) | 35 | ~21 |
| v3-dirty-state | sunk-cost trap (abandoned in-progress Rust patch) | 35 | ~17 |
| v4-multi-corpus-objective | current objective flips (streaming reliability per INC-2411/19/27) | 30 | ~15 |
| v5-recovery-in-thread | prior accepted proposal was rolled back in INC-2481 | 30 | ~11 |

**Family mean target:** `[15, 25]` with a central value of ~20. No single variant may exceed 40 on the probe run. At least one variant must probe <=10. Scores must be monotonically non-increasing V1 -> V5 on the probe (strict monotonicity not required, ties allowed).

## Probe-and-harden calibration loop

1. **Oracle sanity** (runs on every change): run scorer against `verifier_data/<variant>/oracle/manager_brief.md`. All 5 variants must score >=90. Empty brief must score 0. Shortcut brief (accept P3) must be <=30. Any oracle regression blocks the family.
2. **Probe**: run codex gpt-5.4 --reasoning-effort=high 3 times per variant, fresh workspace, no carryover context. Record per-variant mean and std.
3. **Check targets**:
   - probe family mean in `[15, 25]`
   - V1 <= 40
   - at least one variant probe mean <= 10
   - probe ordering V1 >= V2 >= V3 >= V4 >= V5 within +/-3 tolerance
4. **Harden if off-target**:
   - if family mean > 25: raise pass bars, tighten ceilings, add a hidden check against the shortcut the probe exploited
   - if family mean < 15: soften one partial-credit cap OR lower pass_bar by 5 on the variant that is starving the mean
   - if not monotonic: isolate the inversion pair and add a variant-specific ceiling or reduce reward for the easier variant
5. **Re-probe.** Repeat until 3 consecutive probe runs are inside tolerances.

## Current calibration state

The repo ships with oracle briefs at score 90 / 90 / 90 / 94 / 99 for V1-V5 and empty-brief score 0. A naive "always pick P4 with evidence citations" probe brief lands at 90 / 80 / 80 / 45 / 45 — confirming the objective-drift ceiling at V4+ does the intended work. The shortcut "accept P3" probe lands at 30 due to `ignored_staffing_constraint`. A probe run under codex gpt-5.4 --reasoning-effort=high has not yet been recorded in this tree — that is the next calibration step.

## Attempt history

- `attempt_00` (design): per-variant ceilings, hidden tests, and oracle briefs authored. Determinism verified under CNB55_SEED=42. Oracle >=90, empty=0 on all 5 variants.
- `attempt_01` (2026-04-20, probe_run_id `20260420T064244Z`): codex gpt-5.4 --reasoning-effort=high x 3 per variant under `scripts/probe_family.sh` on MacBook, `codex exec v0.120.0`, fresh ephemeral workspace per run. Full report at `report/attempt_01_probe_report.txt`; per-run records at `report/probe_runs.jsonl`.

  | Variant | n | mean | stdev | raw scores (pre-ceiling) | post-ceiling scores | ceilings hit |
  |---|---|---|---|---|---|---|
  | v1-clean-baseline | 3 | 25.00 | 0.00 | 75, 75, 79 | 25, 25, 25 | ranking_without_grounding x3 |
  | v2-noisy-distractor | 3 | 25.00 | 0.00 | 75, 75, 77 | 25, 25, 25 | ranking_without_grounding x3 |
  | v3-dirty-state | 3 | 25.00 | 0.00 | 75, 77, 75 | 25, 25, 25 | ranking_without_grounding x3 |
  | v4-multi-corpus-objective | 3 | 25.00 | 0.00 | 51, 51, 51 | 25, 25, 25 | objective_drift x3 + ranking_without_grounding x3 |
  | v5-recovery-in-thread | 3 | 25.00 | 0.00 | 81, 49, 83 | 25, 25, 25 | objective_drift x1 + ranking_without_grounding x3 |

  Acceptance checks:
  - `[PASS] family_mean in [15, 25]`: 25.00 (at upper edge)
  - `[PASS] max variant <= 40`: 25.00
  - `[FAIL] at least one variant <= 10`: min variant mean = 25.00
  - `[PASS] monotonic V1>=V2>=V3>=V4>=V5 +/-3`: flat at 25 satisfies within tolerance

  **Overall: HARDEN NEEDED** — one hard-floor acceptance check failed.

  Key observations:
  - `ranking_without_grounding` (cap 25) fires on 15/15 runs. Codex gpt-5.4 writes plausible prose rationales but consistently fails the "≥2 rejection rationales cite a real evidence file path" check. This ceiling dominates the post-ceiling score across the entire family, collapsing the raw-score gradient (75-83 for easy variants, ~50 for V4) onto a flat 25.
  - `objective_drift` (cap 45) fires correctly on V4 (3/3) and intermittently on V5 (1/3), confirming the current-objective release_context / incident_context gate does the intended work.
  - Raw-score gradient is preserved and monotonic: V1-V3 clusters ~76, V4 at 51, V5 at 49-83. The harness is producing signal; the post-ceiling compression is the only blocker.
  - Zero shortcut_detected events, zero trusted-final-state violations. Manifest-guarded readonly trees held.

  **Root-cause re-read (superseded the variant-scaled-cap plan):** reading the actual codex brief at `/tmp/cnb55_probe/.../workspace/brief/manager_brief.md` showed the model *did* cite 10+ real evidence files in its rejection prose (`repo_evidence/perf/schema_compile_bench.md`, `repo_evidence/meta/staffing.md`, `repo_evidence/meta/rollout_history.md`, etc.). But its prose layout was flat paragraphs ("P2 ranks second because…"), not the `- **P2 …` bullets or `### P2` headers that the v1 grader's `_rejection_prose_section` regex recognized. The extractor therefore returned an empty slice and the citation check only saw compressed YAML slugs like "material-win-cross-team-risk" — none of which match a real file path. `ranking_without_grounding` fired on formatting, not content. A regex loosen would paper over the format-ambiguity class of bug rather than fix it.
- `attempt_02` (CLI rewrite, 2026-04-19/20): switch the family from "agent writes markdown → grader extracts with regex" to "agent writes structured JSON via a family-shipped CLI → grader reads the JSON". Scope:
  1. **Family-specific CLI shipped at `bin/cnb55-brief`** inside every variant's `workspace_bundle/`. Subcommands `schema`, `validate FILE`, `submit FILE`. Zero-dep Python stdlib. Validates schema v2 (required fields, enum ranges, ranks contiguous 1..N, accepted is the rank=1 entry, at least one assumption with `status=missing`, every citation resolves to a real file under `proposals/`, `repo_evidence/`, `release_context/`, or `incident_context/`, `variant_id` matches `.scenario_variant`). `submit` writes canonical `brief/manager_brief.json` (grader reads) and `brief/manager_brief.md` (human rendering).
  2. **AGENTS.md rewritten and copied identically into all 5 variants** (md5 `c6a8f8d624d685b0b8d9f26035e49758`). Instructs the agent to author `brief_input.json`, run `./bin/cnb55-brief validate`, then `./bin/cnb55-brief submit`. Explicitly forbids hand-writing `brief/manager_brief.md` — the CLI is the only supported path.
  3. **Scorer rewritten** (`verifiers/proposal-ranking-manager-judgment/score_ranking.py`, 730 lines). Removed: `MACHINE_BLOCK_RE`, `yaml` import, `_rejection_prose_section`, `parse_machine_block`, all regex prose extraction. Added: `read_brief()` (reads `brief/manager_brief.json` + checks `schema_version == cnb55.manager_brief.v2`), `phase2_structural()` (structural gates from JSON fields), `score_grounding()` (iterates `ranking[].citations` against `evidence_file_set()` with zero substring heuristics), `score_risk_mitigation()` (reads `primary_risk.mitigations` list), `score_variant_ceilings()` (builds a keyword-scan corpus from structured summaries + primary_risk + assumption_ledger notes). All ceiling caps preserved (`ignored_staffing_constraint`=30, `ignored_stale_perf`=35, `objective_drift`=45, `sunk_cost_finish`=40, `incident_blind_reselect`=30, `ranking_without_grounding`=25). `malformed_machine_block` replaced by `malformed_brief` (10) / `no_brief_file` (0). Result-JSON schema unchanged.
  4. **Manifests and gold_ranking.json regenerated** for all 5 variants. `bin/` added to `readonly_tree_hashes` so the CLI itself is protected from tampering. New AGENTS.md hash (`d..` once recomputed) locked into gold. `bin/cnb55-brief` added to `workspace_manifest.json.files` so trusted-final-state's `wrote_outside_brief` check does not trip on it.
  5. **Oracle briefs regenerated via the CLI.** For each variant, authored `verifier_data/.../oracle/brief_input.json`, ran `cnb55-brief submit` against a temp copy of `workspace_bundle/{variant}/`, and copied the resulting canonical `brief/manager_brief.{json,md}` into `verifier_data/.../oracle/`.
  6. **Sanity re-check** (`regen_cnb55.py`, run 2026-04-20):
     - Oracle scores: **V1=90, V2=90, V3=90, V4=96, V5=99** — all ≥90 ✓
     - Empty brief (no `brief/manager_brief.json`): **0 / 0 / 0 / 0 / 0** — all =0 ✓
     - Shortcut brief (accept staffing-blocked P3 via CLI): **30 / 30 / 30 / 30 / 30** — all ≤30 ✓ (caps `ignored_staffing_constraint`=30 as designed)
     - `malformed_brief` and `no_brief_file` ceilings fire when expected; zero regex-extraction paths remain on the grading path.
  7. **Re-probe on codex gpt-5.4 high** (completed, `probe_run_id=20260420T172913Z`, N=3 × 5 variants = 15 runs):

     | variant | n | mean | stdev | min | max | scores | ceilings |
     | --- | --- | --- | --- | --- | --- | --- | --- |
     | v1-clean-baseline | 3 | 88.00 | 0.00 | 88 | 88 | [88,88,88] | — |
     | v2-noisy-distractor | 3 | 88.00 | 0.00 | 88 | 88 | [88,88,88] | — |
     | v3-dirty-state | 3 | 87.33 | 1.15 | 86 | 88 | [88,88,86] | — |
     | v4-multi-corpus-objective | 3 | 45.00 | 0.00 | 45 | 45 | [45,45,45] | objective_drift × 3 |
     | v5-recovery-in-thread | 3 | 92.00 | 0.00 | 92 | 92 | [92,92,92] | — |

     Acceptance vs §10.1 freeze gate:
     - `[FAIL] family_mean in window [15,25]` — observed **80.07**
     - `[FAIL] max variant ≤ 40` — observed **92**
     - `[FAIL] ≥1 variant ≤ 10` — observed min variant mean **45**
     - `[FAIL] monotonic V1≥V2≥V3≥V4≥V5 ±3` — V5 (92) > V4 (45)
     - `[PASS] oracle ≥ 90 / empty = 0 / shortcut ≤ 30` (90/90/90/96/99, 0, 30)
     - `[PASS] deterministic under CNB55_SEED=42` — stdev 0 on 4/5 variants, 1.15 on v3

     Overall verdict: **HARDEN NEEDED**. Report at `report/attempt_02_probe_report.txt`.

     Bug caught mid-probe (before any variant completed): the first live run scored 0 because the scorer's `wrote_outside_brief` trusted-final-state check was not whitelisting `brief_input.json` at workspace root (the CLI's required input file). Fixed by adding `brief_input.json` to the allowed-write set in `trusted_final_state()`, rebumped `last_regen_utc` in `manifest.lock.json`, documented the whitelist in `task_spec.md`, `evaluator_contract.md`, and `SKILL.md`. Oracle/empty/shortcut sanity still clears 90+/0/30 after fix.

     Findings:
     - V4's `objective_drift` cap (45) fires on 3/3 runs as designed. Scorer milestones + ceilings on V4 are working.
     - V5's `incident_blind_reselect` cap does NOT fire on 0/3 — codex correctly anchors on INC-2481 and selects P5. V5's raw 92 is above V4's capped 45, violating the required V4≥V5 monotonicity.
     - V1–V3 all cluster at 87–88 post-ceiling. The family is currently too easy for codex gpt-5.4 high: the Kendall-τ thresholds (V1 ≥ 0.50; V2–V3 ≥ 0.60) and the grounding / mitigation floors are all cleared trivially, and the non-V4/V5 variants have no active per-variant ceilings biting. That leaves the entire V1–V3 surface living at close to the oracle max (90).

     Hardening plan (to execute in a follow-up pass):
     - Tighten τ thresholds: V1 ≥ 0.67, V2 ≥ 0.75, V3 ≥ 0.80, V4 ≥ 0.85, V5 ≥ 0.85.
     - V1 gains a new staffing-ambiguity trap (Priya's Q3 availability flagged as "TBD" in a second staffing doc that supersedes the first); accepted proposals that don't cite BOTH staffing notes cap at 55.
     - V2 gets a second stale-perf marker on a file that codex currently cites as "fresh"; `ignored_stale_perf` cap drops from 35 → 25 and trigger expands.
     - V3 gains a deeper sunk-cost surface (a second abandoned patch under `repo_evidence/code/in_progress_patch_v2/`); `sunk_cost_finish` cap drops from 40 → 30.
     - V5 gains a subtle watermark-redesign-timing ambiguity: accepted briefs that don't list a watermark-redesign assumption as `status: missing` cap at 55 via a new `unanchored_recovery` ceiling.
     - Reduce the "free" Phase 2 trust bonuses (no_stray_files +4, no_shim +5) from 9 → 4 total, keeping the visible gates without giving trivial headroom.
     - Re-probe expected post-harden: V1≈30, V2≈25, V3≈20, V4≈15, V5≈18 → family_mean ≈ 21 (in window), max ≤ 30, V4 the floor.

- `attempt_02b` (harden, 2026-04-20): realistic, evidence-grounded hardening pass driven by the attempt_02 findings. Each change creates a discoverable failure mode — the agent has everything it needs to succeed, but it has to actually read the evidence and reason about it.

  **New evidence files (discoverable by the agent; documented in AGENTS.md and SKILL.md):**
  - `workspace_bundle/v{1..5}/repo_evidence/meta/staffing_update_2026_06_15.md` — mid-quarter staffing revision. Priya (owner of P4 in every variant) moves to 40% Q3 due to parental leave; Ravi covers the ramp-down. Document explicitly lists the keywords a competent brief should use (`40%`, `parental`, `reduced availability`, `mid-quarter`, `handoff`, `Ravi cover`) and the staged-rollout expectation.
  - `workspace_bundle/v3-dirty-state/repo_evidence/code/in_progress_patch_v2/README.md` — a second abandoned prototype (Kenji's validator scaffolding, transferred to an unrelated team). Document states explicitly this is sunk cost, not a head start, and lists the language (`abandoned`, `sunk cost`, `maintainability`, `in-progress`, `not a head start`) a competent brief should use if it discusses it.

  **Scorer changes (`verifiers/proposal-ranking-manager-judgment/score_ranking.py`):**
  - τ thresholds tightened: V1 ≥ 0.67 (one inversion allowed at N=4), V2–V5 ≥ 0.80 (one inversion allowed at N=5).
  - Phase 2 trust bonuses trimmed: `no_stray_files` 4 → 3, `no_shim` 5 → 3 (total 9 → 6 points freed for substantive work).
  - `primary_risk.statement` minimum length raised from 20 → 40 characters (one dense sentence, not a label).
  - New property check `accepted_citations_depth` (3 pts): accepted entry must list ≥ 3 citations.
  - Mitigation-keyword scoring made graded: ≥ 3 keywords = 8 pts (full), 2 = 4 pts (partial), 1 = 2 pts (weak), 0 = 0.
  - `ignored_stale_perf` cap tightened 35 → 25 **and** trigger tightened: staleness must be flagged in the SAME entry's summary OR in `primary_risk` text (ledger aside is no longer enough). Qualifying keywords: `stale`, `outdated`, `superseded`, `pre-streaming`, `jan 2026`, `january 2026`, `jan_bench`, `january bench`, `jan bench`.
  - `sunk_cost_finish` cap tightened 40 → 30 **and** trigger expanded to include the regex phrase `not a head start`.
  - NEW `missed_staffing_update` ceiling at cap 40 (V1+). Fires when `accepted == P4` (Priya) AND the brief neither cites `staffing_update_2026_06_15.md` nor uses any of the update keywords (`40%`, `parental`, `reduced availability`, `reduced capacity`, `mid-quarter`, `handoff`, `ramp-down`, `staffing_update`, `staffing update`, `priya`).
  - NEW `missed_watermark_assumption` ceiling at cap 55 (V5 only). Fires when no `assumption_ledger` row has both `status: "missing"` AND `watermark` in its `topic` or `note`. The V5 incident doc notes a watermark redesign is required but attaches no target quarter — a competent brief must flag that unknown explicitly.

  **Gold-file updates (`verifier_data/<variant>/gold_ranking.json`):**
  - All 5 variants: `staffing_update_owner_proposal: "P4"`, `staffing_update_file: "repo_evidence/meta/staffing_update_2026_06_15.md"`.
  - V5 only: `requires_watermark_assumption: true`.

  **Oracle-input updates (`verifier_data/<variant>/oracle/brief_input.json`, regenerated via `scripts/regen_cnb55_v2.py`):**
  - V1: extended primary_risk.statement ≥ 40 chars, added staffing_update citation to P4's ranking entry, added "Ravi cover during Priya ramp-down" mitigation, added 4th assumption_ledger row.
  - V2: the staleness flag for the `jan_bench` evidence is now inside P5's *rejection summary* (where the file is cited), not only in the ledger.
  - V3: P5 rejection cites `in_progress_patch_v2/README.md` and the ledger includes an "abandoned validator scaffolding" row with `status: missing`.
  - V4: primary_risk.statement extended to ≥ 40 chars referencing watermark/INC-2427; "staged rollout with Mei pre-warm" added to mitigations.
  - V5: primary_risk.statement references watermark redesign timeline explicitly; P5's accepted summary flags the stale / superseded pre-streaming comparison.

  **Documentation updates:**
  - `benchmark_blueprints/families/proposal-ranking-manager-judgment/skills/proposal-ranking-brief/SKILL.md` — documents the new τ thresholds, grounding-depth floors, mitigation-keyword scoring, and both new ceilings with their exact keyword lists. Adds staffing_update_2026_06_15.md to the "reading order" section (step 5).
  - `workspace_bundle/v{1..5}/AGENTS.md` — new "Things the hardened grader specifically looks for" section listing staffing-update / stale-perf / sunk-cost / objective-keyword / watermark-assumption gates in agent-readable language.
  - `benchmark_blueprints/families/proposal-ranking-manager-judgment/evaluator_contract.md` — new τ thresholds, tightened caps, both new ceilings in the ceiling table. Visible-check trust bonuses reflect 3 + 3 split.

  **Sanity re-check (`regen_cnb55_v2.py`, run 2026-04-20 post-harden):**
  - Oracle scores: **V1=90, V2=90, V3=90, V4=96, V5=99** — all ≥ 90 ✓
  - Empty brief: **0 / 0 / 0 / 0 / 0** — all = 0 ✓
  - Shortcut brief (accept staffing-blocked P3): **30 / 30 / 30 / 30 / 30** — all ≤ 30 (cap `ignored_staffing_constraint`) ✓
  - Synthetic trap checks (isolated runs against scrubbed oracle briefs):
    - V1 staffing-update-scrubbed brief → **40**, ceiling `missed_staffing_update` fires ✓
    - V2 stale-perf-unflagged brief → **25**, ceiling `ignored_stale_perf` fires ✓
    - V5 watermark-assumption-scrubbed brief → **55**, ceiling `missed_watermark_assumption` fires ✓
  - Thin-but-correct brief vs oracle: thin = 78, oracle = 90 → 12-point spread preserved (grounding depth / mitigation depth rewarding detail).

  **Manifest refresh.** `manifest.lock.json` regenerated: new scorer sha, skill sha, oracle_scores V1=90 / V2=90 / V3=90 / V4=96 / V5=99, empty=0, shortcut=30 across all variants.

  Expected probe on codex gpt-5.4 --reasoning-effort=high: V1 ≈ 30, V2 ≈ 22, V3 ≈ 18, V4 ≈ 14, V5 ≈ 10 → family_mean ≈ 19 (in [15,25] window), max ≤ 40, V4 or V5 as the floor.

  **Re-probe on codex gpt-5.4 --reasoning-effort=high** (probe_run_id `20260420T221304Z`, N=3 × 5 = 15 runs, ~38 min wall). Full report at `report/attempt_02b_probe_report.txt`.

  | variant | n | mean | stdev | min | max | scores | ceilings |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | v1-clean-baseline | 3 | 88.67 | 1.15 | 88 | 90 | [88,88,90] | — |
  | v2-noisy-distractor | 3 | 88.00 | 0.00 | 88 | 88 | [88,88,88] | — |
  | v3-dirty-state | 3 | 88.67 | 1.15 | 88 | 90 | [88,90,88] | — |
  | v4-multi-corpus-objective | 3 | 45.00 | 0.00 | 45 | 45 | [45,45,45] | objective_drift × 3 |
  | v5-recovery-in-thread | 3 | 81.00 | 31.18 | 45 | 99 | [99,99,45] | objective_drift × 1 |

  Acceptance vs §10.1 freeze gate:
  - `[FAIL] family_mean in window [15,25]` — observed **78.27**
  - `[FAIL] max variant ≤ 40` — observed **88.67** (V1, V3 tied)
  - `[FAIL] ≥1 variant ≤ 10` — observed min variant mean **45.00**
  - `[FAIL] monotonic V1≥V2≥V3≥V4≥V5 ±3` — V5 (81) > V4 (45)
  - `[PASS] oracle ≥ 90 / empty = 0 / shortcut ≤ 30` (90/90/90/96/99, 0, 30)
  - `[PASS] deterministic under CNB55_SEED=42` — within-variant stdev ≤ 1.15 on V1–V4; V5 stdev 31.18 reflects one of three runs tripping objective_drift, not nondeterminism in the scorer (the scorer is deterministic; the codex agent's reasoning trace varies across runs).

  Overall verdict: **HARDEN NEEDED** by the §10.1 gate, but the hardening is doing the meaningful work it was designed to do. Detailed read:

  - All new evidence files were discovered and used. Per-run brief inspection (one V1 run sampled; oracle/non-oracle path divergence < 5 chars on staffing keywords): codex cited `staffing_update_2026_06_15.md` in 3/3 V1 runs, used `40%` / `parental` / `mid-quarter` keywords, listed ≥ 3 citations on the accepted entry, produced ≥ 40-char primary_risk statements, and used ≥ 3 mitigation keywords. The new `missed_staffing_update` ceiling (cap 40) did not fire on any V1–V5 run — confirmed via `report/probe_runs.jsonl` (zero `missed_staffing_update` entries across 15 runs).
  - The `missed_watermark_assumption` ceiling (cap 55) did not fire on any V5 run — codex added an `assumption_ledger` row with `status: missing` mentioning `watermark` in 3/3 V5 runs.
  - The tightened τ thresholds (V1 ≥ 0.67, V2–V5 ≥ 0.80), the `accepted_citations_depth` ≥ 3 floor, the ≥ 40-char primary_risk floor, and the ≥ 3 mitigation-keyword full-credit floor are all being cleared by codex without any clamp.
  - V4 still trips `objective_drift` cap 45 on 3/3 runs. Codex reads proposals first, anchors on the proposer-stated objective, and does not re-frame to the current `release_context/` objective even with the explicit AGENTS.md callout. This is a real reasoning failure, not a calibration artifact — the cap is doing its intended job.
  - V5 trips `objective_drift` 1/3 (down from attempt_02's incident-blind 0/3) and is otherwise around 99. The watermark / INC-2481 anchoring lands cleanly when the agent reframes the objective, and falls to 45 when it does not.
  - Sanity-test traps still fire on synthetic bad briefs (verified post-probe with hand-scrubbed oracle copies): V1 staffing-update-scrubbed → 40 / `missed_staffing_update`; V2 stale-perf-unflagged → 25 / `ignored_stale_perf`; V5 watermark-scrubbed → 55 / `missed_watermark_assumption`. The traps work; codex just doesn't fall into them.

  Honest read on the §10.1 gate: the user's directive for this hardening pass was **"make the test hard in a meaningful way and provide all the context the agent needs — but do NOT lower the score by adding inconcrete tests or unrealistic / ambiguous requirements."** Concretely-documented, well-grounded requirements at this difficulty level are inside gpt-5.4 high's reasoning envelope: when the agent is told *exactly* what to look for and the evidence is on disk, it produces a strong brief. The hardening did legitimately raise the bar — it added five new substantive requirements and tightened five caps — but a strong reasoning model meets the bar.

  Two of the §10.1 gate axes appear to be in tension with that directive:
  - `family_mean ∈ [15,25]` and `max variant ≤ 40` would require either (a) introducing requirements the agent cannot satisfy from documented evidence (violates the user's directive — adds unrealistic / ambiguous gates), or (b) hidden grading rules not mirrored in AGENTS.md / SKILL.md (violates the user's directive — adds inconcrete gates).
  - `≥ 1 variant ≤ 10` would require a near-zero-floor variant. The only mechanism in the current grader that produces ≤ 10 is `no_brief_file` (cap 0) or `malformed_brief` (cap 10), and codex consistently produces a valid CLI submission.

  Recommended follow-up (deferred, requires user direction): either (1) accept the family at this difficulty band and revise the §10.1 freeze-gate target window upward for "instructions-fully-disclosed" mode, (2) introduce a separate "evidence-only" run mode where AGENTS.md and SKILL.md are stripped to bare task language so the agent must derive the gates from the evidence, or (3) add adversarial evidence (e.g. an internally inconsistent staffing memo where the agent must reconcile contradictions). Options 2 and 3 are legitimate hardness levers that do not violate the user's directive; option 1 is a calibration concession.

- `attempt_02c` (decouple rubric from task docs, 2026-04-20): per user direction, take option (2) from the attempt_02b follow-up list. Principle: the task description stays complete and honest, but the exact grader rubric (keyword lists, char-count thresholds, cap values, reference to `evaluator_contract.md` / `SKILL.md`) is not mirrored into AGENTS.md. A competent manager should infer what concrete mitigations look like from the phrase "concrete operational levers (rollout shape, observability, reversibility)" without being told the grader scans for `{mitigate, gate, pre-warm, shadow, staged rollout, rollback, observability, SLO, kill switch, feature flag, canary}`.

  **AGENTS.md rewrite (identical across all 5 variants, md5 `24c183d9...`):**
  - Removed: explicit keyword lists for staffing-update / stale-perf / sunk-cost / objective / watermark traps; explicit `≥ 40 chars` and `≥ 3 citations` thresholds; explicit cap values (30/40/45/55); references to `evaluator_contract.md` and `.claude/skills/proposal-ranking-brief/SKILL.md` (neither file is staged into the run workspace anyway).
  - Kept: task framing, input list (including pointer to `repo_evidence/meta/` as where staffing artifacts live), CLI workflow, schema minimal example, enum lists for `constraint_tags` and `assumption_ledger.status`, structural rules enforced by the trusted-final-state check, principle-level guidance in a "Things to pay attention to" section.
  - Principle-level framing example: instead of "your brief MUST cite `staffing_update_2026_06_15.md` OR mention the update keywords (`40%`, `parental`, ...)", AGENTS.md now says "staffing can change mid-quarter. `repo_evidence/meta/` may contain more than one staffing artifact written at different dates. If the accepted pick's owner is the one whose availability changed, that reality should shape how you reason about risk, rollout pacing, and mitigations — not sit as a footnote."
  - Grader section now simply tells the agent the grader exists, lists a few named anti-pattern categories (staffing-blocked, stale perf, sunk cost, objective drift, incident-blind re-select) without enumerating the regex keywords or caps, and ends with: "The specific thresholds and caps are grader-internal. Produce a good brief by the standard above."
  - All grader logic (scorer, gold files, evaluator_contract.md, SKILL.md — which is family-maintainer documentation, not staged into the agent workspace) is unchanged. This is a **task-doc-only** change, not a rubric change.

  Verified: `regen_cnb55_v2.py` re-run end-to-end after the rewrite → oracle V1=90 / V2=90 / V3=90 / V4=96 / V5=99, empty=0, shortcut=30 (no rubric regression).

  **Re-probe on codex gpt-5.4 --reasoning-effort=high** (probe_run_id `20260421T011218Z`, N=3 × 5 = 15 runs, ~38 min wall). Report at `report/attempt_02c_probe_report.txt`.

  | variant | n | mean | stdev | scores | ceilings (probe-wide) |
  | --- | --- | --- | --- | --- | --- |
  | v1-clean-baseline | 3 | 88.67 | 1.15 | [88, 88, 90] | — |
  | v2-noisy-distractor | 3 | 88.00 | 0.00 | [88, 88, 88] | — |
  | v3-dirty-state | 3 | 87.33 | 1.15 | [88, 86, 88] | — |
  | v4-multi-corpus-objective | 3 | 45.00 | 0.00 | [45, 45, 45] | objective_drift × 3 |
  | v5-recovery-in-thread | 3 | **25.00** | 0.00 | [25, 25, 25] | ignored_stale_perf × 3, missed_watermark_assumption × 1 |

  **Delta vs attempt_02b (coupled AGENTS.md):**

  | variant | coupled mean | decoupled mean | Δ |
  | --- | ---: | ---: | ---: |
  | V1 | 88.67 | 88.67 | 0 |
  | V2 | 88.00 | 88.00 | 0 |
  | V3 | 88.67 | 87.33 | -1.33 |
  | V4 | 45.00 | 45.00 | 0 |
  | V5 | 81.00 | 25.00 | **-56.00** |
  | family_mean | 78.27 | 66.80 | -11.47 |

  V5 collapsed from 81 → 25. All three V5 runs tripped `ignored_stale_perf` (cap 25), and one additionally tripped `missed_watermark_assumption`. The interpretation is direct: in attempt_02b, AGENTS.md listed the staleness keyword set and explicitly instructed the agent to add a `status: missing` row with the word `watermark`. Codex was reliably pattern-matching to that instruction. With those pointers removed, codex now cites post-incident streaming-era evidence alongside the January 2026 benchmarks *without flagging* the age gap, and does not spontaneously produce a watermark-redesign-timing assumption row. Both behaviors are real manager-judgment failures — the dates are on the files, the incident_context doc mentions the watermark redesign — the agent just does not connect them without an explicit cue.

  V1-V3 were almost unchanged: the staffing-update cue survives removal of the keyword list because the staffing_update_2026_06_15.md document's filename, date-in-filename, and contents are themselves sufficient to make a competent brief cite or acknowledge it. This is the expected outcome — the decoupling does not hide evidence; it hides the grader's regex.

  Acceptance vs §10.1 freeze gate:
  - `[FAIL] family_mean in window [15,25]` — observed **66.80** (better than 78.27, still outside)
  - `[FAIL] max variant ≤ 40` — observed **88.67** (V1)
  - `[FAIL] ≥ 1 variant ≤ 10` — observed min **25.00** (V5)
  - `[PASS]` **monotonic V1≥V2≥V3≥V4≥V5 ±3** — V1 88.67 ≥ V2 88 ≥ V3 87.33 ≥ V4 45 ≥ V5 25. First pass on monotonicity in the whole calibration history.
  - `[PASS]` oracle ≥ 90 / empty = 0 / shortcut ≤ 30 (90/90/90/96/99, 0, 30)

  Overall verdict: **HARDEN NEEDED** by the §10.1 gate, but the decoupling did the meaningful work it was supposed to do — it turned V5's 81 into a 25 by forcing real reasoning instead of pattern-matching. The remaining gap is the V1-V3 cluster at ~88, which stays high because the easy-variant gates (citation depth, mitigation depth, staffing-update acknowledgment) are things gpt-5.4 high infers correctly from the evidence without needing a cheat sheet. To bring V1-V3 down would require either (a) adding adversarial evidence (the inconsistent-staffing-memo proposal from earlier), (b) widening the §10.1 window, or (c) accepting that on this family the signal lives in V4 / V5 and the easy variants are a floor-check rather than a discriminator.

## attempt_02d — adversarial staffing evidence on V1-V3 (probe_run_id `20260421T022540Z`, 15 runs, gpt-5.4 high)

**Hypothesis.** attempt_02c moved V5 from 81 → 25 by decoupling AGENTS.md from the rubric, but V1-V3 stayed clustered at ~88 because the easy-variant gates (citation depth, staffing-update acknowledgment, mitigation-depth) are inferable from the evidence itself without needing the AGENTS.md cheat sheet. Hypothesis: make the evidence *itself* harder to read by inserting an internal contradiction the agent has to reconcile. Real-world analogue: a plan-of-record doc superseded mid-quarter by a memo, where the filename no longer carries the date cue. Managers deal with this all the time.

**What changed in V1-V3 only (V4/V5 untouched).**

- `repo_evidence/meta/staffing.md` rewritten as an authoritative 2026-04-10 Q3 plan of record, "Approved by VP Engineering," with an explicit **`Priya — 100% Q3`** allocation in the headcount table. V1 keeps baseline (no Kenji row); V2/V3 preserve the Kenji-departing row + handoff-plan footnote.
- `repo_evidence/meta/staffing_update_2026_06_15.md` — the dated-filename update memo — deleted in V1/V2/V3.
- Replaced by `repo_evidence/meta/roster_memo.md` (neutral filename, no date cue) containing the same 2026-06-15 in-document date, the `Supersedes: staffing.md` language, the Priya 40% / Ravi cover / 2026-07-08 handoff facts. **Stripped the prior "How to reflect this in a manager brief" section** that was leaking the rubric (literal keyword list: `40%`, `parental leave`, `Ravi cover`, etc.) — the agent now only sees facts.
- `verifier_data/v{1,2,3}/gold_ranking.json` → `staffing_update_file = repo_evidence/meta/roster_memo.md`.
- `scripts/regen_cnb55_v2.py` oracle P4 citations updated to cite `roster_memo.md`.
- Ran `regen_cnb55_v2.py` + `refresh_manifest_lock.py`. Oracles unchanged: V1/V2/V3 = 90/90/90, V4 = 96, V5 = 99; empty = 0; shortcut = 30. `readonly_tree_hashes` refreshed for the new `repo_evidence` layout.

**Results (15 runs, 3 per variant).**

| Variant | n | mean  | stdev | scores     | ceilings                  |
|---------|---|-------|-------|------------|---------------------------|
| V1      | 3 | 88.00 | 0.00  | 88,88,88   | —                         |
| V2      | 3 | 88.00 | 0.00  | 88,88,88   | —                         |
| V3      | 3 | 87.33 | 1.15  | 88,88,86   | —                         |
| V4      | 3 | 45.00 | 0.00  | 45,45,45   | `objective_drift` × 3     |
| V5      | 3 | 49.67 | 42.72 | 25,99,25   | `ignored_stale_perf` × 2  |

`family_mean = 71.60`, `max = 88.00`, `min = 45.00`.

Acceptance against §10.1:
- `[FAIL]` family_mean in `[15.0, 25.0]` — observed **71.60**
- `[FAIL]` max variant ≤ 40 — observed **88.00** (V1 & V2)
- `[FAIL]` ≥ 1 variant ≤ 10 — observed min **45.00** (V4)
- `[FAIL]` monotonic V1 ≥ V2 ≥ V3 ≥ V4 ≥ V5 ±3 — V4 45.0 < V5 49.7 beyond ±3 (V5 variance flipped the ordering; three-sample stdev 42.72 on V5 is the real story, not a V4/V5 ordering change)
- `[PASS]` oracle ≥ 90 / empty = 0 / shortcut ≤ 30 (unchanged)

**What actually happened on the adversarial evidence.** Spot-checked V1 run 1 `manager_brief.json`: the agent cited `roster_memo.md`, set `primary_risk` = "Priya at 40% capacity … Ravi covers execution," tagged the P4 entry with `constraint_tags: [perf, staffing, rollout]`. It silently resolved the 100% vs 40% contradiction by trusting the later in-document date, did not hit `missed_staffing_update`, and did not regress any other dimension. In other words: gpt-5.4 high reconciled the contradiction cleanly and kept its 88. The adversarial-evidence lever did not move V1-V3 because reconciling dated docs is a skill the model already has.

**Why V1-V3 are stuck at 88.** The 88 floor is remarkably stable across every attempt since 02b: 88.67 / 88 / 88.67 (02b), 88.67 / 88 / 87.33 (02c), 88 / 88 / 87.33 (02d). That's a mechanical scoring floor, not a judgment score — the agent hits every concrete rubric item (Kendall τ, accepted match, primary_risk match, citation count, ledger rows, pass_bar) and the 12-point gap is spread across minor dimensions none of which this hardening lever touches. No ceiling fires because there's no honest ceiling to fire — a good manager reading this evidence would do exactly what gpt-5.4 did.

**V5 regressed from 25→49.67.** The 99 run 2 slipped the `ignored_stale_perf` ceiling — same behavior attempt_02c showed once (its 99/99/45 pattern). V5's high variance under decoupled AGENTS.md is real: sometimes the model catches the jan_bench staleness cue, sometimes it doesn't. That's a genuine judgment dimension behaving stochastically, which is actually what we want — but it means V5 can't anchor the "one variant ≤ 10" gate without more reliable pressure.

**Takeaway.** Three calibration attempts (02b → 02c → 02d) have established that the §10.1 acceptance window cannot be hit for gpt-5.4 high on this family via rubric hardening alone. The family as configured is within the model's competence envelope on V1-V3 and genuinely partial on V4/V5. That is a real finding about model capability, not a benchmark authoring failure — and hardening further without a concrete judgment lever would cross into the "fake ambiguity" zone that the directive explicitly rules out. Options to consider before re-running: (a) widen the §10.1 acceptance window for this family (treat it as a "this family is at the frontier" family), (b) add a new legitimate judgment ceiling keyed on explicit contradiction-acknowledgment in the brief (fires when accepted == P4 AND brief does NOT reference both `staffing.md` and `roster_memo.md` — a good manager would flag the conflict, not silently resolve it), or (c) accept this as the family's honest signal and move on. Pending user decision.

## Hardening decisions already applied

- Full-ranking scoring via Kendall tau rather than single-pick top-1, preventing memorize-top-1 shortcuts.
- Structured-output CLI (`bin/cnb55-brief`) is the only path to `brief/manager_brief.{json,md}`; the grader reads the canonical JSON, eliminating the regex-extraction class of format bugs that collapsed the attempt_01 signal.
- Evidence-grounding floor: <2 rejection rationales citing real files caps score at 25 — now driven off `ranking[].citations` paths, not prose heuristics.
- `objective_drift` ceiling (V4+) prevents a solver from memorizing "always pick P4".
- `incident_blind_reselect` ceiling (V5) forces explicit INC-2481 reference before re-selecting P2.
- `sunk_cost_finish` ceiling (V3+) guards against finishing the abandoned Rust patch.
- `ignored_stale_perf` ceiling (V2+) forces explicit acknowledgement when any accepted reason cites the January 2026 numbers.
- Workspace manifest + readonly tree hashes prevent mutation of proposals/, repo_evidence/, release_context/, incident_context/, tests/, AGENTS.md, Dockerfile, `.scenario_variant`, and `bin/` (the CLI itself).

## Run commands (local)

```bash
# Fresh workspace per variant
for v in v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread; do
  rm -rf /tmp/run && mkdir -p /tmp/run/workspace /tmp/run/results
  cp -a benchmark_blueprints/families/proposal-ranking-manager-judgment/workspace_bundle/$v/. /tmp/run/workspace/
  mkdir -p /tmp/run/workspace/brief
  (cd /tmp/run/workspace && codex exec --model gpt-5.4 --reasoning-effort high)
  AGENT_WS=/tmp/run/workspace \
  VERIFIER_DATA=verifier_data/proposal-ranking-manager-judgment \
  RESULT_FILE=/tmp/run/results/verify_result.json \
  VARIANT_ID=$v \
  CNB55_SEED=42 \
  python3 verifiers/proposal-ranking-manager-judgment/score_ranking.py
done
```
