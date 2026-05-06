# Track B Concurrency Measurement Audit - 2026-05-06

## Scope

This note records the latest Track B speculative-decode measurement discussion:

- `warm_concurrency=4` is unstable, not impossible to run.
- The observed c4 highs are not evidence of a real 4x speedup.
- Current c1 evidence does not reproduce the earlier >10 tok/s results.

## Key conclusion

The measurement formula is not the obvious bug. The harness reports decode-time throughput from vLLM metrics as:

```text
generation_tokens / decode_sum_s
```

The risk is the measurement protocol. The original c4 high numbers were single completed runs with uncontrolled output length, including a 4096-token warm completion. Later reruns did not reproduce that token volume and either slowed down, failed correctness, or crashed the engine.

## Concurrency 4 status

`warm_concurrency=4` means four warm requests are submitted concurrently to one vLLM server. It is not four Codex agents and it is not a guaranteed 4x multiplier.

The c4 path can complete, which is how the original high artifacts were produced. It is still not reliable acceptance evidence for the current ngram speculative-decode family:

- Candidate `051` passed B1/B2 at c4 but failed B3 at c4.
- Candidate `028` crashed during fresh c4 heavy and alternate-workload attempts with a vLLM KV allocator assertion:

```text
AssertionError: num_required_blocks 3 < len(req_blocks) 4
```

So c4 is best treated as exploratory until runtime stability and equivalence are both fixed.

## Latest measurement evidence

| Run | Workload | warm_concurrency | decode tok/s | Warm generation tokens | Warm output tokens | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `051/throughput.json` | heavy | 4 | 17.087062 | 5335 | `512,491,4096,236` | Original high, not reproduced |
| `051/speed_recheck_result_concurrency4.json` | heavy | 4 | 7.561378 | 1752 | `512,512,375,353` | Failed speed gate |
| `051/speed_recheck_c1_heavy_fresh_run_01.json` | heavy | 1 | 7.669483 | 1673 | `512,512,273,376` | Failed 9.0 tok/s gate |
| `051/speed_recheck_c1_multifamily_run_01.json` | multi-family-v5 | 1 | 7.737550 | 1377 | `107,512,512,246` | Failed 9.0 tok/s gate |

The original 051 c4 artifact showed about `2.28x` versus the fixed `7.5 tok/s` baseline, not 4x. The fresh c1 runs are about `1.02x` and `1.03x`.

## Interpretation

The original >10 tok/s numbers are likely tied to the combination of:

- c4 scheduling,
- speculative decoding,
- synthetic repeated-token prompts,
- uncontrolled completion length,
- and at least one long cap-hit warm output.

That makes those artifacts useful as a research clue, but not as a stable candidate acceptance basis.

## Recommended gate policy

- Compare candidates only against a baseline measured with the same `warm_concurrency`.
- Do not promote a mutation baseline unless speed and equivalence both pass under that same shape.
- Require repeated speed runs, preferably median of 3 or 5.
- Add a generated-token-volume guard so a 4096-token outlier run is not compared directly against a short-output rerun without being flagged.
- Keep c1 as the acceptance path until c4 speculative-decode correctness and runtime stability are repaired.
