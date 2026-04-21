# respproxy — test inventory, 2026-04 snapshot

## Streaming-reliability test suite

`tests/streaming/` contains 48 end-to-end streaming-tool-call tests.

| age     | flaky this cycle | reopened | net     |
| ------- | ---------------- | -------- | ------- |
| < 30d   | 5 (+5 from 0)    | —        | +5      |
| 30-90d  | 2                | 1        | +3      |
| > 90d   | 0                | —        | 0       |

**Headline:** 5 new flaky streaming-reliability tests opened in the last
30 days. None resolved. Three directly traced to `INC-2411`,
`INC-2419`, `INC-2427`.

## Latency-path test suite

`tests/hot_path/` contains 91 tool-call latency tests (microbenchmark
replay + end-to-end). All 91 passing and stable this cycle.

## Suggested allocation for Q3

Given the streaming-reliability test churn vs. latency-path stability,
the current priority order is inverted from the Q2 close-out plan. The
team should be spending cycles where the tests are moving, not where
they are stable.
