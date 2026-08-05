# B1 Gate A no-split attempt10 invalid result

Status: **qrow exact32 real-event byte comparison passed; task and combined
Gate A invalid**.

Attempt10 ran from repository commit
`0cb1664cba90c20b2f4e2b5ebae2544d876a0c6d` with the truthful-reference v5
FA2 binary. On the authenticated real SWE-Verified event, the qrow32 B1
no-split candidate matched the qrow16 reference across all 16 target
tree-attention layers. The comparison reported zero BF16 output byte
mismatches and zero FP32 LSE byte mismatches. The captured qrow16 reference
output remained the served result.

That component result does not make the overall attempt valid. After the
exact32 replay completed, the diagnostic shadow gate remained armed. A later
legitimate query with 25 rows entered the exact32-only registration path and
was rejected as geometry drift. The engine stopped, the task result became
invalid, and the launcher ended with exit code 15. Therefore the combined
Gate A did not pass and this attempt provides no timing or acceptance result.

The lifecycle repair is commit
`23074f17f684a3fe11c1bd61f48173a124afa075` (`fix(fr13): disarm qrow shadow
gate after replay`). It is recorded only as the required follow-up; it is not
part of this attempt's source and was not cherry-picked or merged by this
evidence branch.

This directory is a sanitized summary. It excludes prompts, responses, task
outputs, request identifiers, credentials, environment dumps, raw logs, and
container identifiers.
