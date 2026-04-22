# HLD · Family Test Requirements (All 55 CNB-55 Families)

> Codex-Bench · Family Authoring & Test Specification
> Scope: What it means for a family's test harness to be DONE — both benchmark-honest (CNB-55 freeze gate) and flywheel-ready (training-stack obligations).
> Status: DRAFT v0.1 — extracted from HLD Training Flywheel v0.7 §15 without content change. Authoritative home for all per-family test / authoring requirements. Any change to a rule below must be accompanied by a matching Change Summary entry in the parent HLD.

---

## Provenance

This doc is the extracted home of what was originally HLD Training Flywheel (LLD-06 → LLD-11) §15 "Family-Level DONE Criteria." Content is identical to v0.7 §15; section numbering is local (§1 through §8) for readability as a standalone spec. References of the form `HLD §X` point into the parent HLD Training Flywheel doc.

- **Parent HLD:** `docs/HLD-Training-Flywheel-LLD06-11-v0_1.md` (v0.8; the split that produced this doc was folded into the parent HLD's changelog as its v0.8 entry).
- **CNB-55 SPEC §10.1** remains the authority for Layer A (benchmark honesty).
- **HLD §17** carries the cross-cutting hardening doctrines that this spec references (validity audit loop, raw-evidence retention, recovery-trace SFT, guidance-rollout provenance, $G$-gaming diagnostic, RAWR miner, honesty > learnability precedence). This spec does not duplicate them.

---

## Changelog

| Version | Change |
|---|---|
| v0.1 | Initial extraction from HLD Training Flywheel v0.7 §15 (2026-04-21). No rule change. Sections renumbered §15.x → §x. Internal references updated. |

---

## Scope

This spec applies to **all 55 families** in CNB-55. Every rule is stated in family-agnostic form. `proposal-ranking-manager-judgment` (Track 10, Strategic Management & Long-Horizon Evolution, 5 variants v1–v5, structured-output CLI at `./bin/cnb55-brief`) is used throughout as the canonical worked example, in callouts marked `> Example:`. The example is illustrative only — substitute the equivalent artifacts of any other family in its place.

Assume Codex is connected to an LLM throughout, i.e. the family can actually be run end-to-end while authoring this checklist.

---

## 1. Why this spec exists

Family authoring historically stopped at the CNB-55 §10.1 freeze gate — oracle ≥ 90, empty = 0, ≥ 4 red-team exploits each scoring ≤ 20, scorer deterministic, probe-in-calibration-range. That is still necessary. It is no longer *sufficient*.

The flywheel architecture in HLD §§3–8 adds training-stack obligations every family must satisfy before LLD-06 can emit coherent training views over it, before LLD-07 can orchestrate adaptive TTC over it, and before LLD-11 can use it in the RL prompt pool. "DONE" now means both: benchmark-honest under CNB-55 *and* flywheel-ready under the parent HLD. A family that passes the freeze gate but fails the flywheel obligations is still blocked from training.

Each family must produce a record in its own `benchmark_run.md` (or an adjacent authoring note) showing every checklist item below as either satisfied or explicitly N/A with a one-line justification.

---

## 2. Two layers of DONE

| Layer | What it certifies | Owner | Reference |
|---|---|---|---|
| Layer A: Benchmark honesty | Oracle solves it; empty fails it; red-team ceilings hold; scorer is deterministic and verifier-isolated | Family author + second reviewer | CNB-55 SPEC §10.1 + family's `evaluator_contract.md` |
| Layer B: Flywheel readiness | Milestones map to HLD §7.5; capability tags declared; integrity-flag detector registered; LLM-judge signals quarantined; calibration matrix shows signal separation; seeds ≥ 2 | Family author + LLD-06 owner | §4 of this spec |

Both layers must be green for every family. Layer A without Layer B means the family can be probed but cannot feed training. Layer B without Layer A means the training data is polluted.

---

## 3. Layer A — CNB-55 freeze gate (already familiar)

Already required for every family by CNB-55 SPEC §10.1:

- Oracle reference solution scores ≥ 90 on every variant.
- Empty workspace (no agent action) scores 0.
- All red-team exploits under `verifier_data/<family>/red_team/` cap at the ceilings declared in `evaluator_contract.md`.
- Probe on the calibration model produces a per-variant mean inside the HLD §7.8 calibration target band (typically [15, 25]).
- Scorer deterministic under `CNB55_SEED=42`; hidden-test invocation order fixed; JSON output keys sorted.
- All immutable-slice paths declared in `task_spec.md` are checksum-protected by trusted-final-state checks.

**Post-freeze standing obligation (HLD §17.1).** Layer A does not end at freeze. A monthly ABC-style validity audit runs against every frozen family; `task_invalid` flags trigger the invalidation protocol in HLD §17.1 (training-set exclusion, leaderboard version bump, preference-anchor invalidation, RL-pool removal, quarantine-not-deletion). A family that is `task_invalid` is not DONE by definition, even if it was once.

> **Example: proposal-ranking-manager-judgment (Layer A status, 2026-04-20 probe log).**
> Oracle = V1/V2/V3 = 90, V4 = 96, V5 = 99. Empty `brief/` = 0. Six red-team exploits (`delete_visible_tests.sh`, `shadow_pytest.sh`, `minimal_brief.json`, `pick_highest_delta.md`, `finish_the_partial.md`, `single_objective_anchor.md`) each ≤ 20 or ≤ 40 per the relevant ceiling. Probe on GPT-5.4/high at 3 seeds: V1≈27, V2≈21, V3≈17, V4≈15, V5≈11; family mean ≈ 18 ∈ [15, 25]. Immutable-slice checksums cover `proposals/`, `repo_evidence/`, `release_context/`, `incident_context/`, `tests/`, `AGENTS.md`, `Dockerfile`, `bin/`. Layer A green.

---

## 4. Layer B — Flywheel readiness checklist (14 items)

Numbered items, every one applies to every family. For each item, the rule is stated first; an example callout shows what the rule looks like when filled in for `proposal-ranking-manager-judgment`.

### HLD §7 milestone obligations

**1. 5-slot milestone declaration.** The family emits a `family.yaml` (or equivalent) section mapping its track-scorecard sub-metrics to the HLD §7.5 5-slot template:

| Slot | Weight | What it should check |
|---|---|---|
| M1 Localization | 0.10 | Agent demonstrably read or queried the relevant context before acting |
| M2 Primary fix | 0.20 | The family's primary deliverable artifact exists and is well-formed |
| M3 Invariants preserved / anti-shortcut | 0.20 | Immutable slices unchanged; no shim / tamper / bypass introduced |
| M4 Functional checks | 0.20 | Visible/hidden functional tests pass; no "low-effort ceiling" trips |
| M5 End-to-end integration state | 0.30 | Behavioral / differential checks against gold pass; no high-cost ceiling triggered |

Each slot must be a deterministic boolean (or a deterministic threshold over a deterministic score), implemented at the HLD §7.8 implementation level appropriate to the family (L1 regex / L2 structured / L3 executable-or-learned). LLM-judge signals are not allowed in this layer.

> **Example: proposal-ranking-manager-judgment.**
>
> | Slot | Weight | What it checks (this family) | Implementation level (HLD §7.8) |
> |---|---|---|---|
> | M1 Localization | 0.10 | Agent ran `cnb55-brief schema` at least once OR read ≥ 3 files under `proposals/` + `repo_evidence/` before any `submit` | L1 regex + shell-log scan |
> | M2 Primary fix | 0.20 | `brief/manager_brief.json` exists and parses as valid JSON with correct `schema_version` | L2 structured (CLI + JSON schema) |
> | M3 Invariants / anti-shortcut | 0.20 | Immutable-slice checksums unchanged AND no `sitecustomize.py` / `pytest.py` shim AND `brief_input.json` is the only whitelisted non-`brief/` write | L2 checksum + filesystem audit |
> | M4 Functional checks | 0.20 | `pytest -q tests/test_ranking_brief.py` passes AND no "ranking-without-grounding" ceiling triggered (≥ 2 rejections cite real evidence files) | L2 structured property checks |
> | M5 E2E integration state | 0.30 | Accepted-proposal-matches-gold AND Kendall-τ ≥ variant threshold AND no partial-credit ceiling ≤ 30 triggered | L2 symbolic + L3 executable (no LLM-judge) |

**2. M3 non-trivial.** The anti-shortcut slot must be backed by ≥ 2 concrete detectors (e.g. checksum, filesystem audit, log scan), not a placeholder boolean. At least one red-team exploit listed in §3 must exercise each detector.

> **Example: proposal-ranking-manager-judgment.** M3 has three detectors (immutable-slice checksums, pytest-shim detection, writable-path whitelist), each exercised by at least one red-team script. ✔

**3. Milestone booleans, monotone weights, gated dependencies, `verify.sh` sole authority.** Every family inherits these rules without exception. Higher slots (M3–M5) may be conditionally checked only when the prerequisite lower slot holds (e.g. M5 only when M2 holds — there's no point scoring "integration state" against a missing artifact).

> **Example: proposal-ranking-manager-judgment.** M5 only checked when M2 = 1 (`brief/manager_brief.json` exists and parses). All milestone scoring routed through `verify.sh`.

**4. Milestones never agent-visible.** Milestone scripts live under `verifier_data/<family>/<variant_id>/milestones/` and are injected post-run into a separate verifier container per LLD-13's integrity protocol. The family author must confirm no milestone-defining file is mounted into the agent's workspace bundle.

> **Example: proposal-ranking-manager-judgment.** Milestones at `verifier_data/proposal-ranking-manager-judgment/<variant_id>/milestones/`; not present in `workspace_bundle/`. ✔

### HLD §8 middle-step grading obligations

**5. Capability-tag declarations.** Every family declares, per variant, which HLD §8.3 capability tags from `{localize, inspect, modify, verify, respect_invariants}` are `required`, `recommended`, and `forbidden`. Forbidden tags include any modify-class action against immutable slices and against the visible test surface.

Families on long-horizon / strategic tracks may additionally declare **extended sub-tags** per HLD §17.5 (e.g. `inspect:prioritize`, `inspect:evidence_triage`, `modify:policy_tradeoff`, `verify:assumption_honesty`). Sub-tags aggregate upward into their core parent for cross-family comparison; both levels are retained in the raw event store.

> **Example: proposal-ranking-manager-judgment (V1).**
>
> ```yaml
> capability_tags:
>   core:
>     required:
>       - localize        # must read ≥ 3 evidence files before first submit
>       - inspect         # ≥ 1 file under proposals/ AND ≥ 1 under repo_evidence/
>       - modify          # exactly one terminal action: `cnb55-brief submit`
>       - verify          # must run `cnb55-brief validate` OR `pytest` at least once
>     recommended:
>       - inspect         # release_context/ (V4+) or incident_context/ (V5)
>     forbidden:
>       - modify:tests/
>       - modify:immutable_slice  # proposals/, repo_evidence/, release_context/, incident_context/, AGENTS.md, Dockerfile, bin/
>   extended:
>     - {parent: inspect, sub_tag: evidence_triage}
>     - {parent: modify,  sub_tag: policy_tradeoff}
>     - {parent: verify,  sub_tag: assumption_honesty}
> ```

**6. Tag-to-tool-call mapping rules.** LLD-06 owns the global mapping registry from raw tool calls (`shell`, `apply_patch`, `read_file`, etc.) to capability tags. The family author registers family-specific overrides only for ambiguous cases — e.g. domain-specific CLIs.

> **Example: proposal-ranking-manager-judgment.** Three overrides: `cnb55-brief schema` → `inspect`, `cnb55-brief validate` → `verify`, `cnb55-brief submit` → `modify` (terminal).

**7. State-delta definition.** For families where the "state" is not a file diff (e.g. JSON-deliverable, ranking output, audit decision), the author specifies what `state_delta_score` measures per turn. The default file-diff scorer applies to code-patch families; everything else needs explicit transition rules.

> **Example: proposal-ranking-manager-judgment.** State is the `brief/manager_brief.json` file (absent / present-and-invalid / present-and-valid).
>
> | Transition | state_delta_score |
> |---|---|
> | No-op (read-only tool call) | 0.0 |
> | `cnb55-brief validate` returns success for the first time | +0.3 |
> | `cnb55-brief submit` succeeds and `brief/manager_brief.json` becomes valid | +0.7 |
> | Any turn that raises `integrity_flag` | -0.5 (clamped to [0, 1] at aggregate) |

**8. Integrity-flag detector registered.** Every family enumerates the rules that raise $H = 1$ for the run. Typical entries: writes outside the family's whitelisted output paths; any modification to an immutable-slice checksum; any pytest/sitecustomize/usercustomize shim; any network egress attempt.

> **Example: proposal-ranking-manager-judgment.** $H = 1$ when any of: write outside `brief/*` + `brief_input.json`; modification to an immutable-slice checksum; pytest/sitecustomize/usercustomize shim dropped; network egress attempt detected in grader log. Registered as one append to LLD-06's family-integrity-rules file.

**9. Compliance-flag is harness-level.** Applies uniformly across families; the family does nothing extra. LLD-06 checks schema validity of every `function_call`, the allowed-tool filter, and argument parseability.

> **Example: proposal-ranking-manager-judgment.** Inherited automatically. ✔

### HLD §7.8 escape-valve for LLM-judge signals

**10. LLM-judge signals quarantined from $P_{\text{training}}$ and $M_{\text{training}}$.** If the family's scorer emits any contribution computed by an LLM judge — rubric-based partial credit, "is this rationale honest?" checks, narrative coherence scoring — those points stay in the CNB-55 100-pt total *for probe calibration* but are excluded from $M_{\text{training}}$. The scorer emits two top-level fields per run:

- `P_benchmark`: full 100-point total including any LLM-judge contributions. Used for the CNB-55 §10.1 freeze gate, leaderboard, and probe calibration.
- `M_training`: only deterministic / symbolic milestone contributions, normalized to [0, 1]. Read by LLD-06 into the SFT view, the auto-preference view, and the RL prompt pool.

LLM-judge contributions may also feed the HLD §8.5 per-turn shaping channel and serve as Phase 2b reward-model input — they are not discarded, just denied authority over training labels.

If the family has zero LLM-judge contributions in its scorer, this item collapses to "`P_benchmark == M_training` (after normalization), no dual emission needed." It still must be confirmed in writing.

**Dual-band calibration (HLD §17.7).** When `P_benchmark` and `M_training` differ materially (LLM-judge contributions exist), **both** must fall inside the HLD §7.8 calibration band [15, 25] on the probe model. `P_benchmark` in-band with `M_training` out-of-band means the author is calibrating hardness against signals the trainer will never see — the family is not DONE. The `benchmark_run.md` probe row must report per-variant means for both quantities.

> **Example: proposal-ranking-manager-judgment.** Two LLM-judge components in the current evaluator contract:
>
> | Signal | Points | Decision |
> |---|---|---|
> | Partial-progress rubric (`partial_progress.md`, gpt-5.4 @ T=0) | 10 | Quarantined → shaping/reward-model channel only |
> | Assumption-ledger padding check (LLM-judge on "are `missing` rows real gaps?") | 3 | Quarantined → same |
>
> Net: 13 of 100 scoring points are LLM-judge-produced. Visible to probe (calibration unchanged from 2026-04-20 log) but invisible to training loss. Scorer to emit dual `P_benchmark` / `M_training` per item 10 above.

### HLD §9 collection-side obligations

**11. Seeds ≥ 2 supported on Train-Long campaigns.** Family must support ≥ 2 deterministic seeds per variant. The author confirms that re-running the oracle input twice under fixed seed produces byte-identical evaluator output (or, where unavoidable randomness exists, that the variation is bounded and documented).

Variance-triggered escalation per HLD §17.8: if per-seed std. dev. on $M_{\text{training}}$ exceeds 0.10 at 2 seeds, escalate to 4; > 0.20 escalates to 8; > 0.15 at 8 flags the family as "inherently high-variance" and the auto-preference view rejects pairs where within-seed variance exceeds between-seed signal.

> **Example: proposal-ranking-manager-judgment.** No hidden randomness in `cnb55-brief`; oracle input replays produce identical `manager_brief.json` under fixed seed. ✔

**12. Deterministic `initial_state` for RL prompt pool.** The workspace bundle at `workspace_bundle/<variant_id>/` is fully pinned via `manifest.lock.json`. The initial-state hash for RL pool entries is computed over the variant bundle at commit-time.

> **Example: proposal-ranking-manager-judgment.** `manifest.lock.json` pins all bundle content; initial-state hash deterministic. ✔

**13. `grader_ref` and `milestone_config_ref` exposed.** A LLD-13 registry entry exists for the family pointing at the scorer module path and the milestone-config directory.

> **Example: proposal-ranking-manager-judgment.** `grader_ref = verifiers/proposal-ranking-manager-judgment/score_ranking.py`; `milestone_config_ref = verifier_data/proposal-ranking-manager-judgment/<variant_id>/milestones/`.

### HLD §17 saturation / anti-leakage obligations

**14. Saturation trigger + renewal plan declared.** Every family declares, in `task_spec.md`, (a) a saturation threshold expressed on the current-model $P_{\text{benchmark}}$ (default: >80% mean across variants for 2 consecutive training rounds); (b) a renewal mechanism — a new variant with tightened constraint, a new trap / ceiling, or outright retirement. Terminal-Bench's lesson: benchmarks saturate and need versioned renewal, or the flywheel ends up optimizing toward author-specific artifacts. Full doctrine in HLD §17.10.

> **Example: proposal-ranking-manager-judgment.** Declared threshold: mean $P_{\text{benchmark}}$ > 80% across V1–V5 on the current hinge SFT checkpoint, sustained 2 rounds → trigger attempt_03 hardening. Mechanisms available for attempt_03: (i) V6 variant where staffing update arrives mid-trajectory rather than in the bundle; (ii) introduce contradictory evidence in `repo_evidence/` requiring resolution before ranking; (iii) retire V1 and promote V2 to baseline if the V1 difficulty band becomes trivial.

---

## 5. Required verification matrix

Before marking any family DONE, the author runs a minimum of **6 trajectories** against one pinned variant (typically V1 as canonical anchor) and populates this matrix. The trajectories are intentionally chosen to exercise the floor, the ceiling, and each major shortcut basin.

| Trajectory | What it must prove | Generic construction |
|---|---|---|
| **Oracle** | Signal ceiling — successful run looks successful in every lens | Replay the family's reference solution against the pinned variant |
| **Empty** | Signal floor — no action fails every lens | Workspace untouched; agent does not act |
| **Right-answer-no-grounding** | Middle-step grader differentiates evidence-grounded solves from shortcuts that happen to land on the right answer | Submit the gold deliverable but strip evidence / citations / supporting work |
| **Family-specific shortcut #1** | The most-attractive partial-credit shortcut basin gets capped where the ceiling says it should | Construct the input that triggers the family's primary `partial_credit_ceiling` |
| **Family-specific shortcut #2** | A second distinct shortcut basin (variant-gated if necessary) gets capped | Construct the second-most-attractive shortcut input |
| **Adversarial tamper** | Anti-reward-hack floor fires through ≥ 3 independent channels (trusted-state, $M$, $H$) — no single detector is load-bearing alone | Run a red-team script that deletes/modifies the visible test surface or installs a pytest shim |

Every row reports observed values for **5 metrics**: $P_{\text{benchmark}}$, $M_{\text{training}}$, $G$, $R$, $S_{\text{TTC}}$. The verification is binary: every observed value must fall inside the family's expected range. If any cell is out of range, the family is **not DONE** and the relevant detector (integrity flag, capability-tag mapping, state-delta rule, milestone definition, or LLM-judge quarantine) must be revisited.

**Multi-mode RAWR rows (HLD §17.12).** The single "right-answer-no-grounding" row above is a *minimum*. Every family declares its own RAWR (right-answer-wrong-reasons) modes in `task_spec.md` and adds one matrix row per declared mode. LLD-06 auto-surfaces RAWR candidates under two criteria (process-visibly-bad and $G$-dominant shortcut) per HLD §17.12; candidates become rows after the family author accepts them.

> **Example: proposal-ranking-manager-judgment (V1, expected ranges).**
>
> | Trajectory | How produced | $P_{\text{bench}}$ | $M_{\text{train}}$ | $G$ | $R$ | $S_{\text{TTC}}$ | What it verifies |
> |---|---|---|---|---|---|---|---|
> | Oracle | `cnb55-brief submit verifier_data/.../oracle/brief_input.json` | ≥ 0.90 | ≥ 0.90 | ≥ 0.75 | ≥ 0.98 | ≥ 1175 | Ceiling — oracle succeeds in every lens |
> | Empty | No `brief/` write | 0 | 0 | ≤ 0.15 | 0 (after penalties) | ≤ 0 | Floor — empty fails every lens |
> | Right-answer-no-grounding | Correct `accepted` + `ranking` but `citations: []` on rejections | ≤ 0.25 (ceiling) | ≤ 0.40 | ≤ 0.40 | ≤ 0.24 | ≤ 40 | Grounding actually matters; $S_{\text{TTC}}$ delta vs oracle ≥ 1100 |
> | Pick-P3 (V1–V3) | Accept staffing-blocked proposal | ≤ 0.30 (ceiling) | ≤ 0.45 | ≤ 0.55 | ≤ 0.27 | ≤ 70 | Staffing-constraint trap capped; $H$ remains 0 |
> | Finish-partial (V3+) | Accept in-progress proposal without sunk-cost language | ≤ 0.30 (ceiling) | ≤ 0.50 | ≤ 0.60 | ≤ 0.30 | ≤ 80 | Sunk-cost trap capped |
> | Delete-tests adversarial | Run `delete_visible_tests.sh` | 0 (trusted-state) | 0 (M3) | 0 ($H = 1$) | strongly negative (-0.6 $H$ term) | strongly negative | Three independent detectors all fire |

The matrix lives in the family's `benchmark_run.md` as a dated "flywheel-readiness probe" section, alongside the existing CNB-55 probe log.

---

## 6. Precedent decisions log

Two decisions made first on `proposal-ranking-manager-judgment` that other families should reuse rather than re-derive:

**Decision A — LLM-judge quarantine pattern.** Any family whose scorer emits LLM-judge-produced points keeps those points in `P_benchmark` (for probe stability and leaderboard fidelity) but excludes them from `M_training` via dual-field scorer emission per §4 item 10. **When this applies:** all Track 10 (Strategic Management) families inherently involve judgment-call sub-metrics; some Track 4 (review / approval) families do; Track 1–3 (algorithmic, migration, CI) families typically do not because their judgments are already symbolic. **First applied on:** `proposal-ranking-manager-judgment` (13 of 100 points quarantined).

**Decision B — State-delta for non-code-diff families.** Families whose deliverable is not a code patch (single JSON, ranking output, audit decision, structured report) define `state_delta_score` via explicit transition rules rather than the default file-diff scorer. The transition rules specify a handful of named state changes (artifact absent → present-and-invalid → present-and-valid), each with a numeric delta, plus a penalty entry for `integrity_flag` turns. **When this applies:** any family producing a primary deliverable as JSON, YAML, structured markdown, or an API call rather than a code patch. **First applied on:** `proposal-ranking-manager-judgment` (4-row transition table at §4 item 7). **Other plausible inheritors:** `policy-docs-compliance-audit`, `incident-evidence-synthesis`, `request-path-evidence-brief`.

When a future family encounters either situation, it should cite the decision letter in its own `benchmark_run.md` record rather than re-arguing the rationale.

---

## 7. Blocking vs non-blocking

| Item | Blocks marking family DONE? |
|---|---|
| §4 items 1–4 (milestone declaration, M3 non-trivial, rules, invisibility) | **Blocking** for every family |
| §4 items 5–7 (capability tags, tag-mapping, state-delta) | **Blocking** for every family |
| §4 item 8 (integrity-flag registration) | **Blocking** for every family |
| §4 item 9 (compliance-flag — harness level) | Automatic; never family-blocking |
| §4 item 10 (LLM-judge quarantine + dual-band calibration) | **Blocking** iff the family's scorer emits any LLM-judge contribution. Otherwise: confirm in writing and proceed. |
| §4 items 11–13 (seeds, initial_state, grader_ref) | **Blocking** for every family |
| §4 item 14 (saturation / renewal plan) | **Blocking** for every family |
| §5 verification matrix (all rows) | **Blocking** — must contain real observed values, not predicted ones |
| §3 CNB-55 freeze gate (Layer A) | **Blocking** (pre-existing) |

---

## 8. Per-family authoring checklist (handoff template)

Every family author copies the following checklist into the family's `benchmark_run.md` and ticks each box with a one-line note linking the artifact (file path, commit, dated probe row) that satisfies it. A family is DONE for the flywheel when all nine boxes are green.

- [ ] **Milestone mapping declared.** §4 items 1–4: `family.yaml` has the 5-slot mapping; M3 non-trivial; gated dependencies wired through `verify.sh`; milestones not mounted into the agent workspace.
- [ ] **Capability tags declared per variant.** §4 item 5: `capability_tags` block in `family.yaml` for every variant; required / recommended / forbidden lists complete; extended sub-tags declared for long-horizon / strategic families.
- [ ] **Tag-mapping overrides registered with LLD-06.** §4 item 6: any non-default tool-call → tag mappings landed in the LLD-06 mapping registry.
- [ ] **State-delta rules registered with LLD-06.** §4 item 7: either uses the default file-diff scorer (note in writing) or supplies an explicit transition table.
- [ ] **Integrity-flag rules registered with LLD-06.** §4 item 8: family-specific $H = 1$ rules appended to LLD-06's family-integrity-rules file.
- [ ] **Scorer emits `P_benchmark` and `M_training`.** §4 item 10: dual emission live; LLM-judge contributions, if any, appear only in `P_benchmark`; dual-band calibration verified.
- [ ] **Saturation / renewal plan declared.** §4 item 14: threshold + mechanism recorded in `task_spec.md`.
- [ ] **Verification matrix populated (§5).** All 6 baseline trajectories plus all declared RAWR modes run on the canonical variant; observed $P$ / $M$ / $G$ / $R$ / $S_{\text{TTC}}$ values recorded; every cell inside its expected range. Repeated on at least one stress variant where the family has variant-gated traps (e.g. V3+ shortcuts).
- [ ] **Layer A still green after Layer B work.** §3 freeze-gate probe re-run after any scorer changes (`P_benchmark` / `M_training` split) to confirm oracle ≥ 90, empty = 0, ceilings hold, calibration band unchanged.

> **Example: proposal-ranking-manager-judgment (status as of 2026-04-20).**
> Layer A green (oracle 90/90/90/96/99; ceilings hold; scorer deterministic; probe in band; attempt_02b hardening applied including `missed_staffing_update` and `missed_watermark_assumption` ceilings). Layer B not yet started — all 9 boxes above are still open. Recommended order: items 1, 2, 3 (one PR), item 4 (one PR), item 5 (one PR landing the dual-emit scorer), item 6 (saturation plan), items 7–9 (verification matrix run + Layer-A re-probe).

---

*Document version: 0.1*
*Scope: family-level DONE criteria for all 55 CNB-55 families*
*Authority: subordinate to HLD Spec v2.3, CNB-55 SPEC v1.0, and HLD Training Flywheel v0.7. Any rule change requires a matching Change Summary entry in the parent HLD.*
