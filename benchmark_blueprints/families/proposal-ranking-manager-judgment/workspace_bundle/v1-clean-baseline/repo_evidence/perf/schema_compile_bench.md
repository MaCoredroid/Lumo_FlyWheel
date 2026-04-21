# Schema compile microbenchmark

**Authored:** Priya, 2026-04-05

## Observation

Every call to `validate_tool_call_args(args, schema)` re-derives the
compiled jsonschema validator because the normalizer passes the schema
dict by copy (see `src/respproxy/normalizer.py:112-140`).

Compilation of a representative skill-pack schema:

```
jsonschema Draft7Validator: 128 ms    <-- per call today
cached compiled validator:    1.2 ms  <-- per call under P4
```

## Replay against 2026-04-02 flamegraph

Substituting the cached path in for the per-call compile reduces p95
end-to-end from 420ms to a projected 294ms. Adding the warm-start cache
to eliminate the first-N-requests cold cost brings the working
estimate to **220ms p95**, matching the range a similar change made on
the internal staging proxy last quarter.

## Why it is real

The schema compile cost is structural, not a profiler artifact. The
proof is that `normalize_request.self_time` is 52ms but
`compile_tool_schema.self_time` is 128ms and happens every request;
the two sum to more than half the hot-path budget.

## Risks

- Schema-version-change invalidation must be exact. I propose keying
  the cache on `(skill_pack_version, tool_name)` and invalidating on
  skill-pack reload. A differential fixture pack verifies the
  invalidation is correct under rolling reloads.
