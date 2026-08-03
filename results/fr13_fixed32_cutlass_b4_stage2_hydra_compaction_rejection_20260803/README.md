# B4 Stage2 Hydra27 compaction-provenance rejection

Status: **kernel comparison clean; qualification rejected**.

This reduced artifact records a canonical real SWE-Verified exact4 B4
Hydra27 K64/root diagnostic of the rebuilt stock-shape Stage2 CUTLASS
candidate. All four tasks and their evaluators completed. The stock-serving
comparator reached its 320-call ceiling with no differing bytes across all
five qualified projection shapes.

The run is not a PASS. Campaign finalization rejected a Qwen trace/task-auth
count mismatch before publishing task metadata: authenticated ingress counted
130 completed model requests while visible trace reconstruction counted 128.
The same immutable campaign metrics prove exactly two 20,000-token hidden
compaction requests and 128 normal 32,768-token requests:

```
128 * 32768 + 2 * 20000 = 4234304
```

All 130 engine requests completed with stop; abort, error, length, and
repetition completions were zero. Aggregate result token usage also reconciles
with campaign metrics, and hidden prompt/output usage is positive. The current
validator rejects before applying that stronger global algebra when a task has
hidden compactions but no trace-visible successful-compaction marker.

No production credential, timing result, throughput result, or hardware-floor
claim is derived from this run. The raw task prompts, responses, patches,
traces, logs, credentials, environment, process/container identities, and raw
metrics remain unpublished.

## Bound candidate

- Source commit: `3295f4d38045486244b8cea1b1f647edc5617cc0`
- Diagnostic selector: `identity_stockshape_stage2_b4_byte_ab`
- Candidate family: `identity_stockshape_stage2_b4`
- Binary SHA-256: `c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29`
- Binary bytes: `117488608`
- Physical rows: `128` (`B4 * 32`)
- Logical topology: `Hydra27`
- Draft vocabulary: K64/root1

See `summary.json` for the reduced machine-readable record.
