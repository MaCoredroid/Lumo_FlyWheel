# B1 Gate A no-split attempt11 invalid result

Status: **real task resolved and component evidence passed; combined Gate A
invalid because of a post-run validator defect**.

Attempt11 ran from clean, pushed source commit
`23074f17f684a3fe11c1bd61f48173a124afa075`. The fixed contract was Hydra27,
physical32, K64/root1, B1, FULL graph mode, and the canonical SWE-Verified task
`astropy__astropy-12907`.

The task resolved cleanly. The qrow32 no-split candidate matched the truthful
qrow16 reference at all 16 target tree-attention layers, with zero BF16 output
byte mismatches and zero FP32 LSE byte mismatches. The GDN GQA-group3
credential passed with raw-byte equality on ten authenticated comparator
events across the 368-step work census. DFWD K64 top3 emitted its ready,
engaged, and full-graph-captured markers exactly once each.

The combined credential issuer then rejected the run. Its old validator read
`exact_proxy_engine_attempt_parity` at the audit document root. The v3 audit
stores that verdict under `ingress`, where it is `true`. All other validator
predicates passed: all ten audit checks were true, proxy and engine each
completed ten requests, and there were no failed or aborted requests. This is
a harness schema-path defect, not a traffic authentication or coverage defect.
The abort occurred before the composed qrow and DFWD credentials were written,
so the combined Gate A remains invalid.

The validator repair is commit
`6b13ff859cb5e532d43b5ab34ea83e764acd5fe9` (`fix(fr13): read traffic parity
from ingress audit`). It is referenced only and is not included in this
evidence branch.

This attempt makes no timing, TPS, acceptance, production, or hardware-floor
claim. A new real-task Gate A run from the repaired frozen source is required.

This directory is sanitized. It excludes prompts, responses, task outputs,
request identifiers, credentials, environment dumps, raw logs, process and
container identifiers, binaries, tensors, and source/runtime manifests. The
manifest retains SHA-256 bindings to the excluded local evidence.
