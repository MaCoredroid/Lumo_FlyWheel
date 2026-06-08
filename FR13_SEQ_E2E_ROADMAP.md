# FR13 sequential tree-scan — e2e GRIND ROADMAP (workflow wnylt7mhs + monitor red-team, 2026-06-08)

Sequential-scan thesis PROVEN (layers 0,1 bit-exact 0.0). This closes the remaining drift grind, not the algorithm. Standing policy: per divergence LOCATE wiring-vs-kernel; bit-exact-or-bust spine+branch; proceed continuously; ask only before close/pass-fail or banned shortcut; NO self-declare.

## ORDERED FRONTS (clear in residual-stream order; each gates everything downstream — do NOT re-measure late layers until upstream is 0.0)

### FRONT 1 — GDN layer-2 seam — THIS KERNEL (the first_nonzero; gates everything) — IN PROGRESS
- first_nonzero = {layer 2, layer_hidden, 0.015625}; single channel (h31/d26/flat-3994), exactly 1 bf16 ULP (-2.1875 vs -2.203125), same channel as WY-L0 migrated up via state.
- **LOCATION RED-TEAM (monitor):** wnylt7mhs said `fr10_gdn_tree_kernel.py:539-540` — that is the **WY kernel** (`_tree_gdn_wy_kernel` L354+, FLA_BF16_BOUNDARIES block); the SEQUENTIAL path (`_tree_gdn_kernel` L207-353) NEVER executes it. Likewise the ww7rx446u beta-bf16 fix is WRONG (verify oracle fused_sigmoid_gating.py:150 beta is fp32 = our L332 matches). **DO NOT touch WY taps or add bf16 to the seq scan.** The real seq-path source is the **h_cache parent-resume vs native register-carry at depth-2 (L280-283)** — found via the LIVE L2 subop ladder (codex fixing the capture tooling + drilling, dir 48c261b6). See FR13_SEQ_LAYER2_REDTEAM.md.
- Gate: L2 hidden+residual -> 0.0 vs pinned native, spine AND branch.

### FRONT 2 — remaining 47 GDN layers — THIS KERNEL, verify-only (expect FREE)
Once Front 1 = 0.0, re-run the full ladder; layers 4,5,6,8… were nonzero ONLY by inherited L2 contamination (same kernel proven bit-exact at L0/L1). Do NOT grind independently until a NEW first_nonzero is a GDN layer with CLEAN input.

### FRONT 3 — full_attention layers (16: idx 3,7,…,63) — OTHER SUBSYSTEM (forked FA2 .so), do NOT touch fr10_gdn_tree_kernel.py
- Forked `_vllm_fa2_C.abi3.so` (additive -inf ancestry-mask). Isolated (tree3/): FA2-LSE-vs-dense 7.6e-6 = accepted MMA floor (DONE); attn-output/value-mix ~0.0077 = the real seam (the ~0.00195 TREE_ATTN-vs-FLASH_ATTN + depth-RoPE wiring front, MEMORY). Buried under inherited GDN drift until Fronts 1-2 = 0.0.
- depth-RoPE wiring already recorded clean (mrope `[0,1,2,2,3,3,4,4,5,5]`, base num_computed_tokens-1/num_computed_tokens). For the ~0.0077: LOCATE the exact divergent op in tree_attn.py vs flash_attn.py (softmax-scale placement / accum dtype / tiling-reduction / online-softmax rescale / qk cast), then alignable-vs-algorithmic. USER: do NOT patch FLASH_ATTN until TREE_ATTN confirmed dead (cuda-captures + B=4). Deliverable vs E5 (FLASH_ATTN MTP-5): within floor -> TREE_ATTN deploys; beyond -> FLASH_ATTN+tree-mask.

### FRONT 4 — final_norm + logits — verify-only (NO independent grind)
final_norm 1.25, logits 0.727 are PURE inheritance; collapse to 0.0 (or composed irreducible floor) when Fronts 1-3 = 0.0. A nonzero logit after upstream 0.0 = a masked upstream miss -> go back up the ladder.

## BRANCH ORACLE the next ladder MUST add (true native-on-path, NOT the tree_self_logits proxy)
Current ladder = SPINE only ([0,1,2,4,6,8]); branch leaves {3,5,7,9} only have the tree's own proxy (`fr10_phase4_patch...:4016`) = the verification GAP.
- **Route A (PRIMARY, no patch edit, ~sub-second, no new server):** on the already-running native arm, send **4 `/v1/completions`** with integer-token `prompt = context(0..4) + path_node_tokens`, max_tokens=1 temp=0, for leaf paths node3 `[0,1,3]` / node5 `[0,1,2,5]` / node7 `[0,1,2,4,7]` / node9 `[0,1,2,4,6,9]`. Capture hooks fire on token-count in `FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS:6737` (set =8,9,10,11) + rows. Oracle per branch node = the LAST row of its leaf-path prefill. Tree branch rows already captured (tree_layer_hidden.pt rows [0..9]) — no tree rerun.
- **Route B (fallback, 0 server cost):** `scripts/fr12_branch_path_oracle_probe.py` (`_node_path:49-55`, `_native_path_scan:236-260`) = boot-free L0 path-replay cross-check.
- **GATE = per-depth ARGMAX 4/4** (per-node marginal, NOT max_abs, NOT path-joint) + temp-0.6 MSS residual within E5 floor. A per-node argmax LAG at the first branch row = cross-branch state-bleed MASK bug -> fix the SEQ ancestry-resume `tl.where` (L280-283) (same code as the Front-1 suspect — NOT rounding).

## CLEAN e2e-vs-E5 + TPS recipe (when the ladder passes)
- **Two E5 refs, do NOT conflate:** superset bar = accept/event **3.076**; lossless floor = bag-TV **0.0593** (`native_e5_self_compare.json`). MANDATORY: run a FRESH native MTP-5 arm in the SAME run/regime/seed-shape as the SEQ arm + bag-TV vs THAT (both have token records); static 3.076 has no token records (superset bar only).
- **SEQ server (`scripts/fr13_launch_forked_fa2_tree_server.sh`):** `FR10_TREE_GDN_WY=0`, `FR10_METRICS=0` (default 1 skews TPS), CUDA-graph (ENFORCE_EAGER unset), diagnostics/FR12 hooks OFF (break capture + contaminate TPS), B=4 temp0.6 top_p0.95 mtp=5, same 8 prompts.
- **Metrics:** lossless = bag-TV vs fresh-native within 0.0593 + accept/event >= 3.076; speed = decode TPS vs native (warm ~16.5) + vs WY. Compute from /metrics + steptrace + existing scripts (reuse, don't hand-roll). Register-residence guard (n_spills==0, no HBM state-load in walk) = the no-tax proof.

## Kernel ownership: Front 1 = THIS kernel (seq, L280-283/subop ladder — NOT the WY L539 taps). Front 3 = OTHER subsystem (forked FA2 .so + tree_attn.py). Fronts 2,4 = verify-only. No input/wiring front (input 0.0).
