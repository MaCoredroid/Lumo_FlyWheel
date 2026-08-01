# Fixed32 B4 batched-GDN K64 diagnostic

## NONFORMAL K64/ROOT1 EVIDENCE ONLY

This campaign used `FR13_DRAFT_VOCAB_K=65536` and
`FR13_DRAFT_VOCAB_ROOT=1`. It is not the required full-vocabulary `K=0`,
`root=0` workload, contains no candidate timing, and is not eligible for the
hardware-floor acceptance decision. Its 119.658015 ms optimistic K64
weight-read floor must not be compared with the current 153.938385 ms
full-vocabulary floor as if they were the same workload.

The real SWE-Verified exact4 B4 graph diagnostic completed all four tasks and
closed 3,943 pure decode steps. Traffic provenance audit v3 found 135 successful
engine requests: 127 with pure decode and 8 authenticated successful requests
without pure decode. There were no rejected, failed, or aborted campaign
requests and no fixed32 traffic outside task brackets.

Task outcomes were:

| Task | Agent | Evaluation |
| --- | --- | --- |
| `astropy__astropy-12907` | exit 0 | resolved |
| `astropy__astropy-13033` | exit 0 | failed |
| `astropy__astropy-13236` | exit 0 | failed |
| `astropy__astropy-13398` | exit 0 | failed |

The post-replay shadow gate compared all 48/48 target GDN layers and all listed
output/state surfaces byte-for-byte. The B4 reference path issued eight
physical GDN launches per layer; the batch-folded candidate issued two, reducing
the structural count from 384 to 96 launches per event. The gate passed, but the
reference was always served, the candidate remained default-off, and no timing
was eligible.

The original serving wrapper returned 16 after inference because its post-task
traffic-audit call lacked the newly required concurrency argument. The four
tasks and final GPU timer flush had already completed. The post-run v3 audit and
gate finalization then succeeded; their exact raw SHA-256 identities are bound
in `summary.json` and `source_audit.json`. Raw `output/**/per_task/**` data
remains on disk and is intentionally not published.

`summary.json` is the curated machine-readable result. `source_audit.json`
separates execution-source identity from the later post-run audit/finalization
source and records every retained raw-evidence checksum.
