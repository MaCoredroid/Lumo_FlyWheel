# Exact fixed32 logits-consumer proof

Pinned patcher SHA256:
`494b6d4c1475204a7bc5148ac386d8455ecca575a2d0b643a74967c9e68a4046`

1. `_fr10_is_wide` is true when `_fr13_is_fixed32` is true and none of the
   disjoint legacy exact shapes match.
2. The wide branch sets `_fr10_leaf_steps = frozenset()`, so the legacy loop
   `torch.topk` consumer is absent.
3. `_fr10_consumes_root_leaf` includes only cat3w, cat6root, cat10, and 333; it
   excludes wide/fixed32, so the separate root `torch.topk` consumer is absent.
4. `FR13_DFWD_K64_TOP3` fails closed unless fixed32, K64, root reduction,
   single-logits, wide mode, and widths `(3,3,3,3,3)` all hold.
5. Under that guard, the root stores `_fr13_root_top3` in
   `_fr10_wide_topk[0]`; all four loop depths store `_fr13_step_top3` in the
   corresponding wide slot.
6. The wide packer consumes those mapped IDs. It does not consume scores or a
   full logits tensor.

This proves semantic feasibility only for the exact guarded physical32 K64
path. It is not a general replacement for other topologies or vocabulary sizes.
