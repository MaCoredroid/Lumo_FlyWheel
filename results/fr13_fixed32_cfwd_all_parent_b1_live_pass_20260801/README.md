# CFWD all-parent B1 full-vocab live PASS

Status: **B1 real-task byte gate PASS; production timing pending**.

`fixed32_all_parent_commit_v2` ran beside the incumbent committer on the
canonical SWE-Verified task `astropy__astropy-12907`. Across the complete
authenticated stream of 1,051 pure decode events, the diagnostic compared
causally consumed FP32 probability rows and all integer products, returned the
incumbent result, and reported zero probability and product mismatches. The
task resolved and its evaluator passed.

This run used full vocabulary (`K=0`, `root=0`), one physical 31-draft tree plus
root, and B1. The final flush closed all 1,051 events with zero pending SFWD,
DFWD, or CFWD samples. All 17 authenticated model requests completed; there
were no campaign rejects, failed attempts, aborted requests, or traffic outside
the task bracket. Launch and end runtime/external manifests are byte-identical.

The launcher returned 16 after the container had been safely removed because
its post-teardown audit call omitted the newly required `concurrency` argument.
The exact deterministic audit was reconstructed from the already finalized
engine/proxy ledgers, task boundary, runner metadata, and work census with
`concurrency=1`; every audit check passed. Commit
`1d804fa4249c875e9ca71ae5aa0b03ca64e72c4e` fixes the call site.

This recovered closeout qualifies the source-bound B1 live PASS for a
production timing diagnostic. It is not a throughput measurement, is not B4
evidence, does not enable the candidate by default, and is ineligible for
hardware-floor acceptance.
