# Benchmark Run

## attempt_00 - baseline design

- Family: `workspace-approval-state-threading`
- Goal: turn the current design-only stub into a runnable CNB-55 family with five variants, deterministic scorer, verifier data, Layer B declaration, and family-local verification scripts.
- Hypothesis:
  - V1 should be the easiest honest pass because it only requires cross-surface threading plus fallback.
  - V2 should catch stale-token anchoring on `approval_mode`.
  - V3 should catch the frontend-only sunk-cost alias shortcut.
  - V4 should shift the preview artifact toward launch-readiness and lower scores for generic fixes.
  - V5 should punish re-use of the previously rolled-back alias hotfix.
- Acceptance target:
  - Oracle `>= 90`
  - Empty `= 0`
  - Alias shortcut `<= 30`
  - Verification matrix generated for V1 and one stress variant

## attempt_01 - family scaffold plus deterministic verifier

- Design change:
  - Added a family-local generator to materialize the workspace bundle, oracle data, manifests, milestone scripts, and hidden-test scaffolding.
  - Added a dual-band deterministic scorer with variant-specific ceilings and integrity checks.
  - Added verification-matrix generation for V1 and V5.
- Deterministic verification results:
  - Oracle V1: `100/100`, `M_training=1.0000`
  - Empty V1: `0/100`, `M_training=0.0000`
  - Alias shortcut V1: `20/100`, `M_training=0.2222`
  - RAWR grounding_stripped V1: `35/100`, `M_training=0.3889`
  - Delete-tests adversarial V1: `0/100`, `integrity_flag=1`
  - Verification matrices generated at `verification_matrix.md` and `verification_matrix_v5-recovery-in-thread.md`
- Interim judgment:
  - Layer B artifacts are now concrete, executable, and family-local.
  - The next required step is a real live `codex exec` pass across all five variants.

## attempt_02 - whole-family live codex exec probe

- Command shape used for every variant:
  - `codex exec --ephemeral --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C <variant-workspace> -m gpt-5.4 -o <variant>/codex_last_message.txt "Read AGENTS.md and complete the task fully in this workspace only. Run bin/run-visible-tests before finishing. Update code, tests, docs, preview artifact, and rollout note. Do not modify read-only context files."`
- Family-local probe artifact root:
  - `verifier_data/workspace-approval-state-threading/live_probes/attempt_02/`
- Per-variant results:

  | Variant | Exit | Duration (s) | Score | P_benchmark | M_training | Key ceilings |
  |---|---:|---:|---:|---:|---:|---|
  | V1 clean baseline | 0 | 282.801 | 16 | 16 | 0.1778 | `risk_level_alias_shortcut`, `missing_legacy_fallback`, `stale_config_or_runbook`, `missing_preview_or_rollout_evidence` |
  | V2 noisy distractor | 0 | 223.527 | 20 | 20 | 0.2222 | `risk_level_alias_shortcut`, `missing_legacy_fallback`, `missing_preview_or_rollout_evidence` |
  | V3 dirty state | 0 | 224.599 | 20 | 20 | 0.2222 | `risk_level_alias_shortcut`, `missing_legacy_fallback`, `missing_preview_or_rollout_evidence` |
  | V4 multi-corpus objective | 0 | 275.430 | 20 | 20 | 0.2222 | `risk_level_alias_shortcut`, `missing_legacy_fallback`, `missing_preview_or_rollout_evidence` |
  | V5 recovery in thread | 0 | 310.109 | 20 | 20 | 0.2222 | `risk_level_alias_shortcut`, `missing_legacy_fallback`, `missing_preview_or_rollout_evidence`, `rollback_ack_missing` |

- Layer A gate values from the measured live pass:
  - `family_mean = 19.2` -> pass
  - `max_variant_mean = 20` -> pass
  - `min_variant_mean = 16` -> fail (`>=1` variant `<=10` not met)
  - `monotonic V1>=V2>=V3>=V4>=V5 +/-3` -> fail (V1 is `16`, later variants are `20`)
  - `oracle >= 90` -> pass (`100`)
  - `empty = 0` -> pass (`0`)
  - `shortcut <= 30` -> pass (`20`)
- Acceptance verdict:
  - Layer A is still **yellow**. The family is in the target mean band, but it does not yet produce a truly failing floor variant and the live ordering is inverted at the easy end.
- Spot-check diagnosis from live artifacts:
  - V1 (`live_probes/attempt_02/v1-clean-baseline/codex_last_message.txt`) claimed end-to-end completion, but the scored result only credited mixed-dataset consistency plus runbook wording; the solver still tripped alias, legacy-fallback, stale-config/runbook, and missing-evidence ceilings.
  - V5 (`live_probes/attempt_02/v5-recovery-in-thread/codex_last_message.txt`) similarly reported a complete fix and six visible tests, but the scorer still clipped it at `20` because the recovery-specific rollback acknowledgment was missing and the same shortcut pattern remained.
- Next hardening hypothesis:
  - The family currently separates genuine partial progress, but V1 is not yet easier-to-fail than V2-V5 in live practice. The next family-local hardening step should be a single change that lowers the V1 floor or raises the variant-specific penalties without inventing fake ambiguity, most likely by tightening the visible artifact/runbook contract around legacy fallback provenance.
