# DFWD K64/root1 R64-U8 production-selector readiness

This artifact records CPU-only readiness of the default-off U8 production
selector at source commit `877fbe54abdb1171a462e5bb96cdbe456a02ed50`.

The selector remains unavailable unless the launcher and in-container validator
authenticate an exact, same-build, real SWE-Verified B1 shadow PASS covering the
root plus MTP depths 1-4 with zero raw BF16 mismatches. Qualification continues
to return incumbent logits. Credentialed production serves the candidate with
one root call and four captured loop calls while preserving the fixed graph
lifecycle; runtime engagement must report zero fallbacks and zero incumbent
head calls.

The timing runner accepts only the pinned real exact4 or exact16 task sets at
B1. It emits full-step wall TPS and GPU-component timing records, but this
readiness artifact contains no timing result.

No GPU run, Docker run, real SWE task, production enablement, speed claim, or
hardware-floor acceptance claim was performed for this artifact. A fresh U8
shadow B1 qualification on the exact source commit remains mandatory.
