# Experiment card — controlled GB10 reproduction of #54928 (Plan v3 item P5)

> For Mark's FUNDING decision. Nothing runs and nothing posts without it.
> Funding the run is separate from approving its report.

## Question
On a pinned stock vLLM build on one GB10 (sm_121), does
Qwen3.8-27B-FP8 + DFlash2 greedy output diverge from target-only, and at the
first divergence is it "E == V != A" (verifier block forward ranks a
different token than the single-token forward) or run-to-run instability?

## Why it adds information
The original report is BF16 TP=4 on 4×3090 (SM86); Windless84's data is FP8
on one RTX PRO 6000 (sm_120); jschmied's bit-for-bit result needed three
kernel fixes on sm_121. A pinned, controlled sm_121 run with the same
attribution method is the missing datapoint; positive or negative is useful.
Labelled a RELATED reproduction (FP8, single device), not the original.

## Pins (proposed; Mark confirms)
- vLLM: stock `v0.28.0` wheel in an isolated venv (campaign install untouched).
  Alternate arm if cheap: current `main` at a recorded SHA.
- Target: `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a`.
- Draft: `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df6` (bf16).
- `num_speculative_tokens`: 7 (as Windless84) and 1 (as the original K=1 arm).
- Sampling: temperature 0, top_p 1, top_k −1, fixed seed; `logprobs` top-5 on
  both sides; `max_tokens` 256.
- Prefix caching OFF; `max-num-seqs` 4; thinking ON for the original prompt
  (`reasoning_effort: medium`, `preserve_thinking: true`) and OFF as control.
- Prompts: the original Spanish "tetris de gatos" prompt (issue body) and
  Windless84's "Write a git command sequence to squash the last 3 commits."

## Controls
- Target-only ×5 within one launch (must be 5/5 identical) → A-side logprobs.
- DFlash2 ×5 within one launch → E and V-side logprobs.
- Second launch of each identical argv → separates launch-sensitivity from
  path difference.
- `--enforce-eager` arm for the DFlash2 side.
- Read argv back from `/proc/<pid>/cmdline` so the two sides differ only by
  `--speculative-config`.

## Attribution at first divergence p
A = target-only argmax at p; V = verifier argmax at p (DFlash2 run's top-5);
E = emitted token at p. Report the pattern (E==V!=A, A==V!=E, …) and the
target-only top-2 margin at p in nats. Report first-divergence index per
arm and per launch.

## Budget and stop rules
- One reserved half-day of GB10 time (download ≈ 30 GB FP8 + draft; isolated
  venv; two launches per arm). One more half-day only if the first run
  exposes a named uncertainty. No competing workloads during measurement.
- Stop if the draft checkpoint or DFlash2 path does not load on the stock
  wheel; report that as the result rather than patching around it.
- No instrumentation beyond logprobs; if attribution needs more, that is
  new-code and a separate decision.

## Output
A compact table (arm × launch × first-divergence × pattern × margin), raw
JSONL responses, exact commands and SHAs, what did and did not execute.
Mark GO before any comment on #54928. If posted, it goes in #54928 as a
reproduction datapoint — no tree mention, no oracle-floor argument.
