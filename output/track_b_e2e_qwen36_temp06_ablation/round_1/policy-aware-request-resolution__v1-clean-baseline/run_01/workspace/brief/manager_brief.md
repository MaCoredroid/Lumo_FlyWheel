# Policy Resolution Brief — v1-clean-baseline

Accepted resolution: `P2`

## Ranked Options

### 1. P2
Redacted sandbox export with dual approval matches the narrow path allowed by access-policy v2.1 and satisfies all operational constraints.

- Constraint tags: policy, privacy, security, operational
- Citations: repo_evidence/policies/access_policy_v2_1.md, repo_evidence/runbooks/request_resolution_runbook_current.md, repo_evidence/tooling/audit_log_rules.md

### 2. P4
Sanitized sample bundle is policy-safe but less useful than the sandbox path that policy v2.1 explicitly allows when no freeze or incident constraints exist.

- Constraint tags: policy, operational
- Citations: repo_evidence/policies/access_policy_v2_1.md, repo_evidence/runbooks/request_resolution_runbook_current.md

### 3. P1
Denying outright over-corrects when policy v2.1 permits a narrower compliant path through sandbox redaction with approvals.

- Constraint tags: policy, operational
- Citations: repo_evidence/policies/access_policy_v2_1.md, repo_evidence/runbooks/request_resolution_runbook_current.md

### 4. P3
Direct local override token is forbidden by policy v2.1 which prohibits raw production replay access and bypasses required approvals and audit controls.

- Constraint tags: policy, security, privacy
- Citations: repo_evidence/policies/access_policy_v2_1.md, repo_evidence/tooling/local_admin_tool.md, repo_evidence/policies/data_handling_matrix.md

## Primary Risk

Granting sandbox replay access to C3 customer data creates exposure risk if redaction or approval controls fail.

Mitigations:
- 24-hour expiry enforces automatic revocation of the exception window
- Dual privacy and security approval gates required before activation
- Audit ticket linkage required before exception activation per audit_log_rules

## Assumption Ledger

- `to_verify` Privacy review sign-off: Privacy team has not yet confirmed sign-off for this exception; required before activation per access-policy v2.1.
- `to_verify` Security review sign-off: Security team has not yet confirmed sign-off for this exception; required before activation per access-policy v2.1.
- `missing` Audit ticket creation: No audit ticket has been created yet for this exception; must be created before activation per audit_log_rules.

## Resolution Note

The selected path is `P2` because it best satisfies the current policy and operating context for `v1-clean-baseline`.
