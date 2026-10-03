# Unfinished workload rate v3: parent review

HOLD. The helper stopped at the Fable rate limit before delivering a connected runtime package. This is a review of the saved reducer only, not a final freeze or an executed experiment.

Source SHA256: e2e02e2133aed28e081fdd4d55f12c278907b34588999ac9266e3792567a83a4. CPU counterexample is retained in `p0/monitor/review-response-20260927/workload-rate-draft-20260928T0558Z/PARENT-COUNTEREXAMPLE.json`.

1. Aggregate generation tokens are pooled over completed intervals without proving the aggregate excludes aborted requests. A permitted record with two completed requests (10 tokens each), one abort contributing100 tokens, and one visible agent response passes and reports59tokens/s ((120-2)/2), versus9tokens/s for the completed subset. Hidden compaction bypasses the optional visible-total equality. The live producer is unfinished, so its actual counter semantics remain unverified; the reducer must refuse any aborted population unless per-request token totals or a proven completed-only counter align the numerator and denominator. Preserve failures in attempt outcomes; mark rate missing where necessary.

2. `ts_first_byte` is treated as first generated token without a token-bearing SSE-event requirement. A role-only, empty, usage or heartbeat event can arrive before the first token. Subtracting one token then using that timestamp is not the agreed token/time boundary. Record first token-bearing event with defined reasoning/content/tool token handling and clarify batched token events; bind the actual producer parser. Network-byte arrival alone is insufficient.

3. The claimed three-way population equality permits hidden/unidentified requests and relies on optional aggregate trace counts. Require a complete independent request-start inventory or request IDs covering compaction/retries/failed requests and reconcile the actual connected producer. No connected runtime/caller or v3 boot lifecycle exists in this captured package, so those previous findings remain open.

No source edit, remote mutation, inference or scientific gate occurred during this review. Existing v2 schedule and source acceptances are unaffected.
