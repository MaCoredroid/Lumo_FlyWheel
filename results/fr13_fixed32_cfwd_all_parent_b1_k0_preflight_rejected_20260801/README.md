# CFWD all-parent B1 full-vocab preflight rejection

Status: **rejected before task ingress; not a measurement**.

The real SWE-Verified one-task B1 diagnostic was launched from source
`f637aeec60694e4113095a74bc05d631fbb691ff` with full vocabulary
`FR13_DRAFT_VOCAB_K=0`, `FR13_DRAFT_VOCAB_ROOT=0`. The server became healthy
after 697 seconds and passed runtime identity checks. The runner then stopped on
the stale required-flags assertion
`FAIL: env pin missing: FR13_DRAFT_VOCAB_K=65536`.

No task request reached the engine. The ready acknowledgement records zero pure
decode steps and the engine ingress ledger is an empty, root-owned file. There
was therefore no incumbent/candidate live comparison, no timing sample, and no
qualification result for `fixed32_all_parent_commit_v2`.

Commit `1a82491f56cff5750706c6ab5239397c5124c8c7` fixes only this preflight
assertion by explicitly admitting the full-vocabulary override. The preserved
container was removed and host/GPU memory was recovered before retry.

The applicable corrected mandatory-weight floor is 153.938384645 ms/step and
the one-sided 1.15x cap is 177.029142341 ms/step. This rejected pre-task run
provides no evidence against either value and is ineligible for floor
acceptance.
