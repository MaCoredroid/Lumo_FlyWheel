# Review Thread Remediation

Use this skill when solving the `pr-thread-contract-remediation` family.

## Workflow
1. Treat review-thread state as authoritative before reading summaries.
2. Separate actionable, resolved, and outdated threads.
3. Propose the minimum patch touching the narrowest plausible files.
4. Preserve backward compatibility unless the thread explicitly asks for a contract break.
5. Draft reviewer replies that cite concrete evidence: changed files, tests, and docs.

## Avoid
- replying `"fixed"` without evidence
- implementing resolved or outdated suggestions
- satisfying only one serializer path
- replacing stable ordering with a global sort
