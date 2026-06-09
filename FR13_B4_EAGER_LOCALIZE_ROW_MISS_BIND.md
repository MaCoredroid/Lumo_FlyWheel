# FR13 B4 Eager Localization Row-Miss Bind

Date: 2026-06-09

Run root: `output/fr13_b4_eager_localize_20260609T210539Z`

## Objective

Localize the eager B=4 co-residency verify divergence using per-layer hidden/final-logit captures, targeting the bisection's first real-loss row: prompt `1`, position `11`, tree token `12182` vs native token `26622`.

## Executed Arms

All arms were eager B=4 (`MAX_NUM_SEQS=4`, `ENFORCE_EAGER=1`), same four SWE prompts, seed `1313`, `temperature=0.6`, `top_p=0.95`.

1. `tree`: first narrow capture, `max_tokens=20`, row filter `0,9,18,27`, `FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS=36`.
   - Reproduced prompt1 pos11 loss class but wrote no `.pt` capture files; the `num_tokens=36` filter was too strict.
2. `tree2` + `native`: second narrow capture, `max_tokens=20`, rows `0..9,18,27`, no token-count filter.
   - `tree2` reproduced prompt1 pos11: tree `12182` vs native `26622`.
   - Capture files were written, but the failing prompt1 rows were not in the row list. Prompt1 rows around the failure occupy interior rows of the 40-row verify call, not just roots/early rows.
3. `tree3` + `native3`: full-row short capture, `max_tokens=14`, rows `0..39`, no token-count filter.
   - Capture files were written for the needed 40-row calls.
   - This run did **not** reproduce prompt1 pos11. Both tree3 and native3 emitted `26622` at prompt1 position `11`.

## Reduction Result

No first-diverging layer is bound for prompt1 pos11.

Reason: the run that reproduced prompt1 pos11 (`tree2`) did not capture the failing row; the run that captured all candidate rows (`tree3/native3`) did not reproduce the prompt1 pos11 failure.

This is a row/token selection miss, not evidence of a clean ladder. Per the marathon instruction, no fifth capture was started.

## Useful Observations

- The eager hooks work outside CUDA graph capture. `tree2/native` and `tree3/native3` produced per-layer hidden and final-logit `.pt` captures.
- Full-row capture has substantial overhead and can perturb the sampled trajectory enough that the exact prompt1 pos11 failure is not stable in the short `max_tokens=14` run.
- `tree3/native3` still showed a visible served-token mismatch at prompt3 position `13` (`tree=321`, `native=310`), but that is not the user-requested prompt1 pos11 row and was not localized in this turn.
- Host memory was recovered after the final arm; no `fr13-b4loc-eager-*` containers remained running.

## Next Scope

Fresh turn recommendation: target a reproduced failure with row discovery first, then run the ladder on that exact `(call,row)`. Do not rely on root-row assumptions under B=4; the failing prompt rows can land inside the 40-row verify pack.

