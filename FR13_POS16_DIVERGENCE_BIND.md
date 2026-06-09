# FR13 Pos-16 / Next Divergence Bind

Date: 2026-06-09
Run: `output/fr13_pos16_substate_20260609T081638Z`

## Scope

Targeted prompt0 greedy diagnostic after the conv prior-slot fix. The original
same8 CUDA-graph token gate failed first at prompt0 position 16; this eager
substate diagnostic reproduced the same frontier class, with the first served
token mismatch shifted to position 18 under capture/eager mode.

The eager shift matters: do not bind a literal "position 16" row from this run.
Bind the class: the next failure is not another row0 layer-0 conv bank read.

## Deployed-Regime Same8 Delta

Artifact:
`output/fr13_conv_fix_same8_e2e_20260609T071944Z/tree_vs_native_temp06_compare.json`

Temp `0.6`, top-p `0.95`, same8 current-schema comparison with the conv fix:

- Exact token sequence matches: `1/32`.
- First mismatch: prompt `1`, sample `0`, position `1`, native token `248068`, tree token `846`.
- Bag-TV: `0.24993977147577093`.
- Native accept/event: `3.0483558994197293`.
- Tree accept/event: `2.053067993366501`.

This improves over the pre-fix same8 bag-TV `0.4258`, but remains above the
`0.059` floor.

## Prompt0 Greedy Eager Reproduction

Artifact:
`output/fr13_pos16_substate_20260609T081638Z/tree_vs_native_prompt0_token_compare.json`

Shape: prompt0 only, `max_tokens=24`, `temperature=0`, `top_p=1`,
`ENFORCE_EAGER=1`, subkernel capture enabled.

- First mismatch in the eager diagnostic: position `18`.
- Native prefix around the mismatch:
  `[... 10088, 364, 264, 82546, 52589, 1103, ...]`
- Tree prefix around the mismatch:
  `[... 10088, 364, 264, 82546, 10278, 1103, ...]`
- Bag-TV: `0.041666666666666664` for this one prompt / 24-token diagnostic.

The earlier prompt0 conv-fix run stopped at 16 tokens, so it ended before this
next mismatch.

## Layer-0 GDN Substate Result

Artifact:
`output/fr13_pos16_substate_20260609T081638Z/fr13_pos16_gdn_substate_compare.json`

Row0 call-by-call result:

| call | first diverging row0 stage | notes |
| ---: | --- | --- |
| 0 | none | clean through `o_proj_out` |
| 1 | none | only `h0_state_in=3.7e-09` |
| 2 | none | only `h0_state_in=9.5e-07` |
| 3 | `input_hidden` | downstream drift; not a clean-input conv root |
| 4 | none | clean through `o_proj_out`; this is the served branch-flip event |
| 5 | `input_hidden` | downstream/path drift |
| 6 | `input_hidden` | downstream/path drift |

The clean calls confirm the prior conv fix held at the next frontier. The
diverging row0 calls diverge before layer-0 GDN, so another row0 conv/bank
writeback tweak is not supported by this capture.

## Branch / Selector Finding

Artifact:
`output/fr13_pos16_substate_20260609T081638Z/fr13_pos16_branch_l0_compare.json`

At the served flip event (`tree_path_lcp_max` row 4), the tree selects an
off-spine branch path:

- Winner path: `[0, 1, 3, 5, 7]`
- Emitted tokens: `[364, 264, 82546, 10278, 1103, 2357]`
- Path0 LCP: `1`; winner branch LCP: `5`
- The wrong served token is `10278`, tree branch node `3` self-target.
- Native greedy at the same served prefix expects `52589`.

Layer-0 GDN at this event:

- Spine rows are clean:
  - tree row0 vs native row0: all checked stages `0.0`
  - tree row1 vs native row1: all checked stages `0.0`
  - tree row2 vs native row2: all checked stages `0.0`
  - tree row4 vs native row3: all checked stages `0.0`
- Off-spine branch rows are not validated by the native spine arm:
  - tree row3 vs native row2 starts divergent at `input_hidden`
  - tree row5 vs native row3 starts divergent at `input_hidden`
  - tree row7 vs native row4 starts divergent at `input_hidden`

Interpretation: this is a branch/selector-oracle front, not another systematic
row0 conv/bank num-accepted-class issue. The current native arm is a spine arm;
it cannot validate off-spine branch rows. The next decisive measurement must be
the real native-on-branch-path oracle for the selected leaf paths, with served
policy/log alignment fixed first.

## Verdict

Systematic conv bank issue: **not supported** for this frontier.

New front: **branch selector / native-on-branch-path oracle alignment**. The
served tree can choose an off-spine branch that changes greedy output tokens.
Before any kernel whack-a-mole, the branch path must be checked against real
native `/v1/completions` on that branch prefix and the diagnostic
`tree_path_lcp_max` rows must be aligned with the served policy.
