# CNB-40 Track Folders

This directory organizes the Linux Codex CLI subset into the 8 tracks described in [benchmark_deisgn.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_deisgn.md).

The canonical family bundles still live under [benchmark_blueprints/families](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families). Each entry here is a symlink so the repo has both:

- a flat family-owned layout for the retained benchmark artifacts
- a track-based layout for the Linux CLI track plan

App-only tracks were removed from this Linux CLI subset:

- browser preview workflows
- computer use workflows
- scheduled automation maintenance

## Track Index

### 01 Core Implementation

- `responses-sdk-adapter-cutover`
- `responses-tool-schema-cutover`
- `sandbox-policy-ci-drift`
- `transcript-merge-regression`
- `workspace-approval-state-threading`

### 02 Codebase Understanding

- `codex-surface-workflow-mapping`
- `dead-flag-reachability-audit`
- `parallel-explorer-root-cause-map`
- `request-path-evidence-brief`
- `runbook-code-reconciliation`

### 03 Refactor Modernization

- `codex-provider-rollover`
- `codex-skill-runtime-v2-split`
- `esm-plugin-loader-modernization`
- `release-manifest-v2-modernization`
- `sqlalchemy-2-session-modernization`

### 04 Review Remediation

- `pr-intent-regression-review`
- `pr-review-fanout-remediation`
- `pr-thread-contract-remediation`
- `review-thread-ui-hardening`
- `security-audit-hotfix-remediation`

### 05 Frontend Multimodal Build

- `annotated-mockup-component-port`
- `design-token-dashboard-refresh`
- `hero-asset-launch-page`
- `responsive-checkout-visual-regression`
- `support-playbook-ui-migration`

### 06 Skills Tooling

- `codex-config-skill-remediation`
- `skill-driven-handoff-command`
- `skill-entrypoint-drift`
- `skill-pack-contract-drift`
- `skill-router-contract-upgrade`

### 07 Plugin Integration Work

- `connector-tool-registry-bridge`
- `plugin-marketplace-validation`
- `plugin-scaffold-alignment`
- `review-driven-plugin-release`
- `tool-catalog-fallback-wiring`

### 08 Subagents Orchestration

- `candidate-worktree-shootout`
- `delegation-merge-salvage`
- `fanout-fullstack-release-blocker`
- `incident-retro-runbook-closure`
- `nightly-regression-watch`
