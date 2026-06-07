# FR-13 source finding (Claude, 2026-06-07): the deficit is in the tree COMMITTER dispatch

Boot-free read of the tree commit path (scripts/fr10_phase4_patch_vllm_tree_gdn.py + live rejection_sampler.py), since the kernels are proven clean (GDN bit-exact, full-attn FA2-equiv 1-ULP) and drafts native-identical.

## Native accept rule (temp 0.6, NO_DRAFT_PROBS) — the bar
`rejection_random_sample_kernel`: `draft_prob=1`, accept iff `target_prob[draft] >= uniform`, i.e. **accept probability = target_prob[draft]** (temp-0.6 + top_p scaled, via apply_sampling_constraints). On reject -> recovered token from residual. is_greedy = (temperature==0) = False at temp 0.6.

## Tree committer dispatch (patch ~L3441)
```
if sampling_metadata.all_greedy:
    _lumo_tree_path_lcp_max_greedy_sample(...)   # GREEDY: accept while draft == target_argmax (L3500 argmax), emit argmax on reject. policy='greedy_tree_lcp_max'
else:
    _lumo_tree_canonical_multidraft_sample(...)  # STOCHASTIC: sample_deterministic_multidraft_rejection_step, spine accept_prob = p[draft]  == native rule
```
- The tree path DOES apply apply_sampling_constraints (temp 0.6 + top_p) to target_logits before the commit (L3468-3492), and softmaxes those -> temp-0.6 probs. So temperature is NOT mis-applied.
- The canonical (stochastic) committer's SPINE rule is provably identical to native (accept_prob = p[draft]); multi-child nodes give overlap_mass >= p[spine] (the superset). So if canonical fires at temp 0.6, accept should be >= native — yet measured eager-B1 = 0.737 vs E5 ~2.6.

## Therefore the 0.737 deficit is EITHER:
- **(a) `sampling_metadata.all_greedy` is wrongly True at temp 0.6** -> the GREEDY argmax-match committer fires -> under-accepts (argmax agreement << stochastic accept) AND lossy (emits argmax, not a temp-0.6 sample -> explains bag_TV 0.558 + spurious EOS). Leading suspect — explains BOTH symptoms.
- **(b) canonical fires but mis-counts** the per-node accepted path/length.

## Decisive cheap check (codex, eager-B1 server already up)
Set FR10_METRICS=1 (+ LUMO_TREE_PATH_LCP_LOG), one request, read which committer policy fires + log `sampling_metadata.all_greedy`.
- greedy fires @ temp 0.6 -> THE BUG: force the stochastic/canonical path (or fix the all_greedy plumbing for the spec-verify metadata).
- canonical fires -> instrument its per-node accept_prob (spine must = p[draft]) + accepted-length counting.
Verify SPINE and BRANCH. The fix is likely a one-line dispatch/metadata correction, not a kernel change.
