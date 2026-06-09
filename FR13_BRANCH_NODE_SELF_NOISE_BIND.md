# FR13 branch-node red-team + native self-noise bind (2026-06-09)

## Scope

This note supersedes any losslessness inference from stale pairings around prompt0 position 18. The stale artifacts disagreed on native's own greedy token (`10278` vs `52589`), so the gate was rerun from fresh current arms and interpreted against a native self-noise mask.

## Artifacts

- Run root: `output/fr13_branch_node_redteam_20260609T090932Z`
- Branch-prefix probe: `branching_node_redteam.json`
- Current native1 same8: `current_native_same8/native_greedy_probe.json`
- Current tree same8: `current_tree_same8/tree_greedy_probe.json`
- Native2 self-noise same8: `native2_b4/native_greedy_probe.json`
- A/B/C gate summary: `current_self_noise_loss_gate.json`

## Branch-node red-team

The exact branch prefix probe confirmed the prior native-on-path oracle is not enough to prove the served branch decision:

- Saved prompt0 native artifact at the branch point: `52589`.
- Fresh native1 exact-prefix completion at the same generated prefix: `10278`.
- Fresh native1 text-prompt replay also emitted `10278`.

This is native/config self-noise at a low-margin position, not evidence that stale tree/native artifacts can be compared directly.

## Fresh current gate

Fresh arms:

- Tree: `TREE_ATTN`, forked FA2, `tree_mtp`, 9-node tree, `MAX_NUM_SEQS=1`, probe batch size `1`.
- Native1: `FLASH_ATTN`, `naive_mtp`, 5 MTP tokens, `MAX_NUM_SEQS=1`, probe batch size `1`.
- Native2 self-noise: `FLASH_ATTN`, `naive_mtp`, 5 MTP tokens, `MAX_NUM_SEQS=4`, probe batch size `4`.

Prompt identity guard passed for all `8/8` records.

### A/B/C counts

- (a) Tree-vs-native1 mismatching positions: `368`.
- (b) Native self-noise mismatching positions: `85`.
- (c) Tree mismatches outside the native self-noise mask: `287`.

Sequence-level:

- Tree/native exact sequences: `0/8`.
- Tree/native prefix sequences: `0/8`.
- Tree/native ordered-subsequence sequences: `0/8`.
- Native1/native2 exact sequences: `4/8`.

## Interpretation

The prompt0 position-18 branch flip is within native self-noise:

- native1: `10278`
- native2: `52589`
- tree: `52589`

That specific flip should not be counted as a real tree loss.

However, the current tree still has substantial mismatches outside native self-noise. First real outside-self-noise positions:

- prompt0: position `46`, tree `2168`, native1/native2 `4381`
- prompt1: position `28`, tree `17`, native1/native2 `256`
- prompt2: position `35`, tree `1031`, native1/native2 `2315`
- prompt3: position `10`, tree `3`, native1/native2 `15495`
- prompt4: position `1`, tree `846`, native1/native2 `248068`
- prompt5: position `1`, tree `846`, native1/native2 `32`
- prompt6: position `1`, tree `846`, native1/native2 `248068`
- prompt7: position `15`, tree `33`, native1/native2 `58950`

Verdict: native self-noise explains the branch-node contradiction but does **not** explain the current tree's same8 token divergence. The remaining loss is real under the accepted "minus native-self-noise" gate and needs localization from the first outside-self-noise divergence, not from the stale pos18 branch artifact.
