# FR13 All-8 Branch Oracle Binding

Date: 2026-06-09

Commit target: FR13 decisive test branch-oracle stage after Seam 1 and Seam 2.

## Scope

This is the targeted native-on-branch-path oracle for the eight first
outside-native-self-noise positions from `FR13_DECISIVE_TEST.md`:

- prompt0 position 46
- prompt1 position 28
- prompt2 position 35
- prompt3 position 10
- prompt4 position 1
- prompt5 position 1
- prompt6 position 1
- prompt7 position 15

The B4 temp-0.6 superset e2e deliverable was intentionally not started in this
stage.

## Artifacts

Run root: `output/fr13_decisive_final_20260609T182455Z`

- Tree B1 greedy branch capture:
  `tree_b1_greedy_branch/tree_greedy_probe.json`
- Tree path-LCP log:
  `tree_b1_greedy_branch/logs/tree_path_lcp.jsonl`
- Native B1 greedy reference:
  `native_b1/native_greedy_probe.json`
- Native B1 temp0.6 self-noise baseline:
  `native_b1/native_temp06_probe.json`
- Targeted branch oracle:
  `all8_branch_oracle.json`

## Reducer Update

`scripts/fr13_branch_token_oracle.py` now supports:

- `--targets prompt:position,...` to select the requested flip events.
- `--winner-only` to check only the committed winner branch path.
- skipped leading overrun rows when a previous request hit `max_tokens` and
  vLLM logged an extra unserved tree event. The existing mismatch test still
  fails closed for a first-record emitted-token mismatch.

Guard checks:

- `pytest -q tests/test_fr13_branch_token_oracle.py` -> `3 passed`
- `python3 -m py_compile scripts/fr13_branch_token_oracle.py` -> passed

## Result

The oracle wrote `all8_branch_oracle.json` and failed closed:

- events aligned: `143`
- skipped overrun rows: `5`
- targeted events checked: `8`
- winner-only path checks: `17`
- tree/native-on-branch matches: `13/17`
- first mismatch: prompt0, event13, path `[0,1,3,5,8]`, depth4
  parent target, tree token `198`, native-on-branch token `1358`

Mismatching checks:

| prompt | event | path | check | tree token | native-on-branch token |
| ---: | ---: | --- | --- | ---: | ---: |
| 0 | 13 | `[0,1,3,5,8]` | depth4 parent target | 198 | 1358 |
| 0 | 13 | `[0,1,3,5,8]` | leaf self target | 262 | 71093 |
| 3 | 4 | `[0,1,3,6]` | leaf self target | 265 | 1302 |
| 7 | 6 | `[0,1,3,5,8]` | depth0 parent target | 417 | 7620 |

## Verdict

The all-8 branch oracle does **not** confirm lossless-by-branch for HEAD.
Several targeted winner-path checks differ from native-on-their-branch-path,
so the branch-flip surface cannot be dismissed as only stale-comparator or
native-self-noise alignment.

This binding does not start or claim the B4 temp0.6 superset e2e result.
