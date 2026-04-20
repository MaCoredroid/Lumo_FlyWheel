# Policy Resolution Operator

Use this skill when a local request must be resolved under explicit policy constraints.

## Workflow
1. Determine the allowed disposition before touching tools.
2. Separate technically possible actions from policy-permitted actions.
3. Apply the minimal allowed local change, if any.
4. Explain the outcome with a policy-grounded note.

## Anti-Patterns
- Performing a forbidden change because the tool allows it.
- Treating stale runbook language as current policy.
- Returning a disposition without a policy basis.

## Done Signal
- The request is resolved correctly and policy is obeyed throughout.
