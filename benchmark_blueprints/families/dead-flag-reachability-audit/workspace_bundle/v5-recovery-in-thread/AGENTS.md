# Dead Flag Reachability Audit

You are auditing preview rollout controls inside this workspace copy.

Variant: `v5-recovery-in-thread`

Do:
- inspect defaults, env parsing, runtime branching, tests, docs, and any extra context dirs
- author `brief_input.json` at the workspace root
- run `./bin/cnb55-flag-audit validate brief_input.json`
- run `./bin/cnb55-flag-audit submit brief_input.json`

Required outputs:
- `artifacts/flag_audit.md`
- `artifacts/reachability_matrix.json`
- `artifacts/cleanup.patchplan.md`

Guardrails:
- classify each named flag as `live`, `partial`, or `dead`
- distinguish parser presence from runtime reachability
- distinguish a live alias from a live standalone flag
- name one disproved false-positive path per flag
- do not modify code, tests, docs, or evidence files
- do not use evidence outside this workspace copy
