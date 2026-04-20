# Evaluator Contract

- `family_id`: `annotated-mockup-component-port`
- `task_id`: `annotated-mockup-component-port/shared-comparison-card-port`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Required Solver Inputs

- `task_spec.md`
- `codex/config.toml`
- `skills/mockup-port-fidelity/SKILL.md`

## Trusted Runtime Assumptions

The official runtime instance provides the held-out `repo/` and artifact bundle plus hidden verifier assets. The public family directory alone is not enough for a full score. Official grading is execution-backed and inspection-backed; reviewer prose is secondary evidence only.

## 100-Point Breakdown

- `20` Shared implementation ownership
  - solver repairs the shared component or true shared dependency path used by consumers
  - diff inspection rejects preview-only, story-only, or one-off local overrides
- `20` Multistate visual fidelity
  - repaired component matches the annotated state family rather than one easiest screenshot
  - structure, spacing, and typography remain coherent across hidden state combinations
- `15` Responsive and geometry robustness
  - narrow-width and long-label behavior remains stable
  - hidden DOM or geometry assertions reject overlap, clipping, or label-specific hacks
- `15` Alternate-presentation robustness
  - density, theme, or equivalent alternate presentation path remains correct
  - hidden checks reject fixes that only work in the default presentation mode
- `15` Compatibility preservation
  - downstream consumer behavior implied by integration and history artifacts still works
  - stale-note-driven removals or prop regressions fail
- `10` Verification quality
  - visible tests cover one width-stress case and one alternate-presentation case
  - hidden tests extend those checks beyond the obvious visible examples
- `5` Story and docs alignment
  - stories and docs describe the repaired shared behavior rather than local overrides or stale semantics

## Milestone Checks

The official verifier exposes milestone-style checks in the `scenario_families` pattern. The exact scripts are held out, but the milestone semantics are fixed:

- `m1_shared_path_repaired`
  - shared component path or true shared dependency was changed
  - no preview-only replacement or story-only implementation exists
  - partial credit intent: `0.20`
- `m2_state_matrix_holds`
  - annotated state family behaves correctly under visible and hidden state combinations
  - no one-state or one-label shortcut survives
  - partial credit intent: `0.25`
- `m3_responsive_and_alt_mode_holds`
  - width-stress behavior and alternate-presentation behavior both pass
  - hidden geometry or DOM checks confirm robustness
  - partial credit intent: `0.25`
- `m4_consumer_contract_preserved`
  - downstream integration path still works
  - hidden compatibility assertions reject stale-note-driven regressions
  - partial credit intent: `0.20`
- `m5_stories_docs_and_tests_aligned`
  - visible regression coverage exists
  - stories and docs align with the repaired shared path
  - partial credit intent: `0.10`

## Trusted Evaluation Phases

### Phase 1: Integrity Gate

- diff inspection confirms the patch lands in the real shared path
- verifier rejects:
  - story-only, docs-only, or baseline-only changes
  - preview-only substitutes
  - snapshot churn without corresponding implementation repair
  - benchmark-specific string matching or width hardcoding
  - modified verifier hooks, skipped checks, or deleted regression coverage

### Phase 2: Visible Command Slice

The held-out runtime executes the visible verification command for the repo. Full points require both:

- the component or unit test slice passes
- the visible story, visual, or DOM-regression slice passes

If visible commands are skipped, stubbed, or only nominally updated, verification credit is capped even if the prose sounds plausible.

### Phase 3: Hidden Functional Checks

The hidden verifier then runs trusted checks that are not exposed in this family directory, including:

- width and long-label stress cases beyond the visible fixture
- density, theme, or equivalent alternate-presentation cases
- downstream integration or prop-contract checks
- DOM or geometry assertions for overlap, clipping, wrapping, and hierarchy
- story or docs coherence checks against the repaired implementation
- mutation or exploit slices that ensure the tests are not trivially satisfiable

## Score Caps

- Cap at `20/100` if the attempt is family-bundle-only or otherwise lacks the held-out runtime bundle needed for code changes and trusted verification.
- Cap at `25/100` if no shared component or true shared dependency path is repaired.
- Cap at `30/100` if no trusted automated verification covers both the width-stress case and the alternate-presentation case.
- Cap at `25/100` if downstream compatibility is not evidenced or is treated as optional.
- Cap at `10/100` for any banned shortcut, including story-only, preview-only, docs-only, baseline-only, or verifier-bypass behavior.

## Automatic Failure Conditions

- `0-5` if the submission proposes only stories, docs, screenshots, or baseline updates.
- `0-5` if the submission disables or removes the live compatibility path to satisfy a stale note.
- `0-5` if the submission tampers with tests, verifier hooks, or gating logic instead of repairing the implementation.
- `0-5` if the submission fabricates passing commands, screenshots, or compatibility evidence.

## Evidence Rules

- Official score comes from trusted command execution, hidden tests, and diff inspection rather than LLM-as-judge prose.
- Hidden tests should validate behavior, geometry, and compatibility; screenshots alone are insufficient.
- The hidden suite must avoid narrow tests that require arbitrary symbol names and wide tests that demand behavior absent from the provided artifacts.
- A family-bundle-only attempt can still earn partial credit for correctly identifying ownership, state families, anti-shortcut risks, and verification strategy, but it cannot earn full grounding points.

## Judging Procedure

1. Verify the solver used the required inputs and targeted the real shared component path.
2. Run the visible command slice on the held-out runtime instance.
3. Run milestone inspections and hidden tests.
4. Apply raw category scoring, then score caps.
5. Record raw score, applied caps, and final score in `benchmark_run.md`.

## Hardness And Validity Checks

- The held-out runtime must include at least one compatibility expectation that is only inferable from repeated evidence, not from one explicit note.
- The hidden suite must include at least one width-stress case, one alternate-presentation case, and one downstream integration check.
- If a family-bundle-only attempt can exceed `30/100`, the benchmark is under-hardened and should be revised.
