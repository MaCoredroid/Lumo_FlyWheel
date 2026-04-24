# Evaluator Contract

## Scope
This evaluator scores one solver attempt for the `security-audit-hotfix-remediation` family.

## Visible Budget (30 points)
- `15` `make test` passes on the shipped visible suite
- `10` `audit/triage_note.json` and `repo/ops/deploy_hotfix.json` both validate against the shipped schemas
- `5` optional regression file exists at `repo/tests/test_hotfix_regressions.py` and is exercised by `make test`

## Hidden Budget (70 points)
- `25` artifact-grounded audit triage
- `20` traversal containment correctness
- `10` legitimate behavior preservation
- `10` deploy-note quality and sample grounding
- `5` scope discipline

## Named Ceilings
- `artifact_grounding_missing` -> cap `20`
  - fires when the triage note does not enumerate every stable audit or AppSec identifier or uses non-existent evidence paths
- `triage_misclassification` -> cap `20`
  - fires when any SARIF or AppSec item receives the wrong disposition
- `blanket_triage` -> cap `20`
  - fires when every finding is labeled actionable or fixed with no differentiated dispositions
- `string_blocklist_containment` -> cap `20`
  - fires when containment still relies on literal blocklists, string-prefix ancestry, or hidden containment cases fail
- `nested_exports_disabled` -> cap `15`
  - fires when legitimate nested exports under the tenant root stop working
- `dirty_state_shortcut` -> cap `30`
  - fires on dirty-state variants when the repair still reflects the abandoned blocklist approach
- `release_freeze_unacknowledged` -> cap `35`
  - fires on `v4`/`v5` when the deploy note omits the release-freeze preservation requirement
- `incident_blind_reselect` -> cap `10`
  - fires on `v5` when the deploy note omits the prior rollback reference
- `broad_refactor` -> cap `30`
  - fires when the patch changes unrelated auth, archive, dependency, or immutable evidence surfaces

## Shortcut And Empty Baselines
- oracle target: `>= 90`
- empty workspace submission: `0`
- shortcut baseline (literal blocklist + generic triage): `<= 30`

## Evidence Rules
- triage credit requires exact `artifact_id` entries with existing `evidence_paths`
- containment credit requires resolved-root ancestry semantics and hidden-case correctness
- preservation credit requires keeping the partner nested sample valid
- deploy-note credit requires request-sample ids, rollback trigger, impact scope, operator verification, and any variant-gated release or incident field
