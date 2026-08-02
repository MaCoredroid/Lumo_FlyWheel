# Fixed32 SFWD descriptor prior-reuse B1 timing diagnostic

## Verdict

Reject `fixed32_sfwd_prior_reuse_rowgroup32_c64_v1` as a performance
candidate.  On the matched real SWE-Verified B1 diagnostic, step wall time
regressed from 239.09383569184118 ms to 243.36520258768618 ms (+1.786481%),
while SFWD regressed from 167.98448456422247 ms to 171.99758979742455 ms
(+2.388974%).

Wall TPS increased from 21.640909853758078 to 24.306848775305966 only because
accepted drafts/event changed from 4.17420814479638 to 4.915441176470588 in
the two real-task executions.  That acceptance shift does not turn the slower
per-step kernel path into a kernel speedup.

## Phase breakdown

| Phase (ms/step) | Stock | Candidate | Candidate - stock |
| --- | ---: | ---: | ---: |
| SFWD verify GPU | 167.98448456422247 | 171.99758979742455 | +4.013105233202083 |
| DFWD drafter GPU | 35.1336487682997 | 35.36089980516505 | +0.2272510368653471 |
| CFWD committer GPU | 27.74713703571819 | 26.187714597304108 | -1.5594224384140816 |
| Other wall | 8.228565323600833 | 9.818998387792488 | +1.5904330641916555 |
| Full step wall | 239.09383569184118 | 243.36520258768618 | +4.271366895845006 |

## Floor distance

The 119.658015414 ms reference is only an optimistic mandatory-weight-read
lower bound, not a complete physical hardware floor.  Its nominal 1.15x cap is
137.6067177261 ms.

- Stock is 1.9981430818872428x the lower bound and 101.48711796574119 ms above
  the cap.  Reaching the cap requires a 42.44656399111185% step-wall reduction.
- Candidate is 2.0338395363292348x the lower bound and 105.7584848615862 ms
  above the cap.  Reaching the cap requires a 43.45669953512794% reduction.

## Evidence boundary

Both arms ran the same one authenticated real SWE-Verified task and each
resolved it with tests passing and clean agent/evaluation exits.  The candidate
was the sole convolution source producer on all 48 layers, with one candidate
launch per layer, zero incumbent launches, and no fallback.  Source, runtime,
and external manifests were independently byte-identical at launch and end.
The runner completed attested container cleanup for both arms; Docker was empty
between arms and the post-pair observation found zero containers and zero GPU
compute processes.

This is a B1 diagnostic only.  It is not timing-, production-, or
floor-acceptance-eligible, and no one-sided U95 was computed.  Exact4 B4 or
exact16 is required for acceptance.  This directory contains aggregate values
and hashes only; it excludes raw logs, task/model/request/response/patch data,
environment values, secrets, process/container identifiers, task identifiers,
and raw sidecar paths.
