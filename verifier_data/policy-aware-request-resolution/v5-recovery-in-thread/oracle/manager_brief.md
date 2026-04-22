# Policy Resolution Brief — v5-recovery-in-thread

Accepted resolution: `P4`

## Ranked Options

### 1. P4
After INC-2471, the right current answer is the sanitized fallback because replay exceptions are temporarily suspended.

- Constraint tags: incident, policy, operational
- Citations: proposals/P4.md, incident_context/incident_INC-2471.md, incident_context/prior_resolution.md, release_context/freeze_notice.md

### 2. P1
Outright denial is compliant but less useful than the post-incident sanitized fallback.

- Constraint tags: incident, policy
- Citations: proposals/P1.md, incident_context/incident_INC-2471.md

### 3. P5
Waiting for the isolated mirror is a future follow-up, but the timeline is missing and it does not resolve the current request now.

- Constraint tags: incident, operational
- Citations: proposals/P5.md, incident_context/mirror_followup.md, incident_context/incident_INC-2471.md

### 4. P2
The prior sandbox exception path was rolled back under INC-2471 and cannot be reselected without ignoring the incident record.

- Constraint tags: incident, policy
- Citations: proposals/P2.md, incident_context/incident_INC-2471.md, incident_context/prior_resolution.md

### 5. P3
Direct override is still the fastest but remains a clear policy violation after the incident rollback.

- Constraint tags: policy, incident, security
- Citations: proposals/P3.md, repo_evidence/tooling/local_admin_tool.md, incident_context/incident_INC-2471.md

## Primary Risk

Re-selecting the rolled-back replay exception would repeat the post-incident audit gap instead of honoring the temporary suspension.

Mitigations:
- 24h expiry recorded on the ticket
- audit log linked before activation
- least privilege scope only to the vendor sandbox
- rollback to the sanitized fallback if approval or controls are missing

## Assumption Ledger

- `to_verify` Vendor deadline flexibility: The request does not say whether a one-day delay for approvals is acceptable.
- `missing` Future mirror timeline: The isolated mirror or broader replay redesign schedule is not pinned down in the current evidence.

## Resolution Note

The selected path is `P4` because it best satisfies the current policy and operating context for `v5-recovery-in-thread`.
