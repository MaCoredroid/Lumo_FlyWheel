# Audit Hotfix Triage

Use this skill when solving the `security-audit-hotfix-remediation` family.

## Workflow
1. Enumerate every SARIF and AppSec item by stable identifier.
2. Differentiate actionable hotfix work from false positives, monitor-only items, and out-of-scope follow-ups.
3. Repair the traversal bug with resolved-root ancestry semantics rather than string filters.
4. Preserve valid nested exports under the tenant root.
5. Write `audit/triage_note.json` and `repo/ops/deploy_hotfix.json`, then validate them with `python3 bin/cnb55-hotfix-artifacts validate`.
6. If `release_context/` or `incident_context/` exists, acknowledge it in the deploy note.

## Avoid
- treating every audit warning as fixed
- blocking all subdirectories
- using bare `".."` checks or string-prefix ancestry
- broad refactors in auth, dependencies, archive code, or unrelated config
