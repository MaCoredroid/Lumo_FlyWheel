# Fixed32 K64 M1 R64 qualification source readiness

Status: source-ready for one real SWE-Verified B1 byte qualification; not run.

This artifact binds the exact-order R64 draft-head candidate to the same
authenticated five-site qualification contract used by R32. The live comparator
executes the candidate at `root` and `mtp_depth_1` through `mtp_depth_4`, compares
all 65,536 BF16 logits at every site by raw 16-bit value, and continues serving
the incumbent BF16 K64 reference logits.

The gate is restricted to `astropy__astropy-12907`, B1, K64/root1, the canonical
fixed32 drafter graph signature, and the pinned SM121a R64 source and binary. Its
credential issuer requires the final fixed32 flush, matching boundary snapshot,
and a rebuilt authenticated chat-traffic audit.

No GPU or Docker runtime was used for this source-readiness step. It makes no
byte-equality, timing, performance, or production-admission claim. A PASS exists
only after `scripts/fr13_run_b1_draft_head_m1_r64_live_ab.sh` completes a real
task and `scripts/fr13_draft_head_m1_r64_pass.py` issues and re-verifies the
resulting credential.

Source commit: `264134619fd7380fbf961a5c1d596f69bc80ec06`

Offline verification: 76 focused R32/R64 qualification and R64 source/codegen/
artifact tests passed. Shell syntax, Python syntax, embedded observed-runtime
source compilation, and `git diff --check` also passed.
