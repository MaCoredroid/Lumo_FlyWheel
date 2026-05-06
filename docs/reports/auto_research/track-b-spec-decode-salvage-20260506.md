# Track B Spec Decode Salvage

Generated: 2026-05-06

## Objective

Try to reproduce any viable candidate from the fast speculative-decode set:

| Candidate | Original speed | Spec decode | Prior gate state |
|---|---:|---|---|
| `020` | `15.753922 tok/s` | ngram, 3 draft tokens, lookup `2-8` | Failed B-1 at concurrency 4 |
| `025` | `14.506594 tok/s` | ngram, 2 draft tokens, lookup `2-16` | Not gated |
| `028` | `14.581565 tok/s` | ngram, 2 draft tokens, lookup `2-8` | Not gated |
| `051` | `17.087062 tok/s` original; `7.677189 tok/s` c1 recheck mean | ngram, 4 draft tokens, lookup `7-8` | Passed B-1/B-2/B-3 at c1; failed B-3 at c4 |

Production criterion used here: a candidate must clear the `9.0 tok/s` 20% speed threshold over the fixed `7.5 tok/s` baseline at `warm_concurrency=1`, and must have correctness evidence at concurrency 1.

## Results

| Candidate | Concurrency-1 reproduction result | Decision |
|---|---|---|
| `020` | First c1 speed recheck crashed vLLM EngineCore with `AssertionError: num_required_blocks 5 < len(req_blocks) 6`; no valid speed artifact. | Not produced |
| `025` | First c1 speed recheck crashed vLLM EngineCore with `AssertionError: num_required_blocks 4 < len(req_blocks) 5`; no valid speed artifact. | Not produced |
| `028` | Three c1 speed repeats completed: `8.323741`, `8.311742`, `8.454847 tok/s`; mean `8.363443 tok/s`; `0/3` cleared `9.0 tok/s`. | Not produced |
| `051` | Five c1 speed repeats completed: `7.658967-7.713841 tok/s`; mean `7.677189 tok/s`; `0/5` cleared `9.0 tok/s`. B-1/B-2/B-3 passed at c1. | Correct at c1, not speed-produced |

## Artifacts

- `candidates/020/reproduction_attempt_c1_result.json`
- `candidates/025/reproduction_attempt_c1_result.json`
- `candidates/028/reproduction_attempt_c1_result.json`
- `candidates/051/speed_recheck_c1_summary.json`
- `candidates/051/validation_recheck_result.json`

## Conclusion

No candidate in this set was reproduced as both speed-positive and correctness-acceptable at concurrency 1.

The original high numbers for `020`, `025`, `028`, and `051` all came from speculative decoding runs at `warm_concurrency=4` with much longer generated-output windows. Under the requested concurrency-1 validation shape, `028` was the best surviving candidate, but it reached only `8.454847 tok/s` best-case and averaged `8.363443 tok/s`, below the `9.0 tok/s` acceptance threshold.

Spec decode remains the only observed source of material speed, but the useful speed region is unstable here: aggressive shapes crash EngineCore or fail equivalence under stress, while safer shapes do not clear the speed threshold at concurrency 1.
