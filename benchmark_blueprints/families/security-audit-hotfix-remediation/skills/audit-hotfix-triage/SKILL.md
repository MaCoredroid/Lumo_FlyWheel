# Audit Hotfix Triage

Use this skill when solving the `security-audit-hotfix-remediation` family.

## Workflow
1. Enumerate every audit item by stable identifier or rule id.
2. Decide which item is actionable for the hotfix and justify every non-actionable item.
3. Prefer a resolved-root containment fix over string filters.
4. Preserve valid nested export behavior under the tenant root.
5. Write a deploy note with rollback trigger and concrete verification.

## Avoid
- treating every audit warning as fixed
- blocking all subdirectories
- using bare `".."` checks
- broad refactors in auth, dependencies, or unrelated config
