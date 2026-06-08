# FR13 WY next front — localize the layer-0 GDN outlier divergence (the live ladder ROOT)

## WHERE WE ARE (committed 1e9aee57, valid paired ladder, prompt "Explain hash tables.")
Drift trajectory: 3.32 (pre-state-fix) → **1.25** (state-fix 8a975837, argmax-lossless-by-LUCK: all 6 spine depths match but max_abs 1.25 >> E5 floor 0.0593) → **1.027** (+ #6 readout tap, flag FLA_BF16_OUTPUT_SPLIT ON).

**The #6 readout tap is PARTIAL and is NOT the root.** Per-layer ladder (`gateA_spine_ladder_{baseline,tap}.json`):
- `first_nonzero = stage=layer_residual, layer=0, max_abs=0.0625` in **BOTH** baseline and tap — the tap did NOT change layer-0.
- layer-0 GDN: `hidden` max_abs **0.015625** (mean 1.8e-4), `residual` **0.0625** — IDENTICAL baseline vs tap.
- The tap only trims the downstream bulk (final_norm 2.625 → 2.1875). It addresses the readout floor (1.22e-4), not the layer-0 root.

**Diagnosis:** layer-0 GDN output is at the 1.22e-4 floor on the MEAN but **0.015625 on OUTLIER elements** (~128×). Those outliers are UPSTREAM of the readout (the tap leaves them untouched) → a non-readout WY GDN sub-op: in_proj / causal_conv1d / l2norm / the WY scan / RMSNormGated gate / o_proj. Top-down rule ([[feedback_top_down_per_layer_lossless_gate]]): the first nonzero layer IS the root — fix layer-0 before anything downstream.

## OPEN QUESTION (decide first): is 0.015625 a WY REGRESSION or a standing issue?
The OLD per-subkernel GDN gate (62516997) drove scan/gate/o_proj/conv/in_proj to 0.0 at every GDN layer — but for a DIFFERENT kernel. The WY one-pass kernel is new. Determine: did WY REGRESS layer-0 from 0.0 → 0.015625 (a WY-specific bug to fix in the kernel), or was layer-0 never actually 0.0 live (standing)? Compare the WY path vs the old per-subkernel path at layer-0.

## THE TASK: layer-0 SUB-OP ladder (NOT another readout tap)
Capture each GDN sub-op OUTPUT at layer-0, tree vs native-on-path, B=1 eager, same pinned-prompt paired harness (save request.json):
`in_proj → causal_conv1d → l2norm(q,k) → WY scan(state+output) → RMSNormGated → o_proj`.
Find which sub-op FIRST shows the 0.015625 outliers (mean ~1.8e-4 OK, the MAX outliers are the target). Localize WHICH elements (channels/heads/tree-rows) — outliers concentrated on branch rows ⇒ a tree-mask/ancestry bug; on specific heads/channels ⇒ a numeric/dtype seam. Then fix that sub-op to 0.0 (bit-exact-or-bust), re-ladder layer-0 → 0.0, propagate.

## DISCIPLINE
Keep the #6 tap committed (flag OFF default; it is real, just not the root). ONE GPU, recover between arms, pinned-prompt paired runs (request.json saved), commit+push+bind FR13_LADDER_LOG.md, NO copy/splice/reroute, branch oracle must be the TRUE native-on-branch-path (the proxy in the last run is insufficient). No self-declared PASS. The monitor runs a parallel CPU localizer workflow ahead of you.
