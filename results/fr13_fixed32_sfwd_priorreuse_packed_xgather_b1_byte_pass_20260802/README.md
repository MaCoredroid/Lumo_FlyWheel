# Packed x-gather SFWD real B1 byte pass

Status: **PASS_SOURCE_ONLY**.

The K64/root1 packed x-gather candidate completed a one-task authenticated
real SWE-Verified B1 diagnostic. The task resolved and its evaluation reported
`tests_passed`. This run is a correctness diagnostic only: it is not timing or
hardware-floor acceptance evidence.

Across 22,080 candidate invocations and all 48 model layers, both `conv_out`
and `commit_source_stage` were compared against the reference. All 44,160
surface comparisons were byte-equal, with zero differing bytes and zero shape
or dtype mismatches. The comparison covered 30,749,491,200 bytes. The reference
was always served, the candidate had no fallback path, and production remained
disabled.

The source, runtime, and external manifests were each byte-identical between
launch and end. After completion and teardown, the host census found zero
running Docker containers, zero GPU compute processes, and zero MiB GPU memory
in use.

Corrected analytical traffic accounting decodes 23, 28, and 31 historical
`x` rows across the three taps, for 82 total. The baseline is 24,196 logical
global bytes per CTA and the candidate is 13,700 bytes per CTA. The reduction
is 10,496 bytes per CTA, or 43.3790709208134%, equal to 1,679,360 bytes per
request-layer and 80,609,280 bytes per 48-layer forward. This traffic model is
analytical and does not establish runtime speed.

This reduced package excludes raw logs, task/model/request/response/patch/env
values, secrets, process identifiers, and container identifiers.
