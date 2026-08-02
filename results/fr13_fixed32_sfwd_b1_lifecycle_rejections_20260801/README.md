# FR13 fixed32 SFWD B1 lifecycle rejections

## Verdict

REJECTED_LIFECYCLE. Neither attempt is a timing measurement or an acceptance
result.

Both attempts used the real SWE-Verified task `astropy__astropy-12907` at batch
size 1 and concurrency 1. They failed before a model forward or speculative
decode step completed:

1. Source `456db364351d020d8ffd11bcc085aeb9edfc5043` failed while building GDN
   attention metadata because `PAD_SLOT_ID` was not imported. Commit
   `3c45640ae623bdd1826498412d53b85698fbbeb2` fixes that import.
2. Source `3c45640ae623bdd1826498412d53b85698fbbeb2` advanced into the GDN forward
   path, then failed closed because the eager SFWD pregather consumer ran before
   its state was preseeded. Commit
   `cff5ad03c99e4518dcd132f6f8123cdcd841cd36` fixes eager preseed setup.

For each attempt, the enabled marker was present but
`fr13_fixed32_sfwd_state_fusion.byte_ab.jsonl` was absent. Therefore each has
zero comparator records. Both final timer sidecars report zero forward starts,
zero pure-decode steps, zero wall steps, and empty sample arrays. No TPS,
latency, acceptance, confidence bound, or hardware-floor ratio may be derived
from these runs.

## Evidence

- PAD failure runroot:
  `output/fr13_b1_sfwd_state_fusion_live_gate_20260801T094600Z`
- Pregather-before-preseed runroot:
  `output/fr13_b1_sfwd_state_fusion_live_gate_20260801T101754Z`
- Sanitized failure excerpts: `evidence/`
- Immutable original log bindings: `source_checksums.sha256`
- Machine-readable rejection: `verdict.json`
- Packaged artifact checksums: `checksums.sha256`

Only the minimal traceback excerpts are copied here. Full logs, container and
process identity files, and environment captures are intentionally excluded.
