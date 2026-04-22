
# INC-2471 — Vendor Replay Audit Gap

A previous vendor sandbox replay exception was rolled back after audit hooks were found missing on one export job.

Temporary policy overlay:
- replay exceptions are suspended until follow-up controls land
- sanitized bundles remain allowed
