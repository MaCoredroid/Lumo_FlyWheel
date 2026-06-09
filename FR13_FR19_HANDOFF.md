# FR13 -> codex_fr19 handoff (fr18 stood down 2026-06-09 after a 47m marathon boot-churning the conv-fix validation). Fresh context, the SEQ GDN tree-scan front.

## WHERE WE ARE (sequential rank-1 GDN tree-scan, use_wy=False; WY is archived to branch fr13-wy-archive + docs/archive/wy/)
The top-down per-layer ladder (pinned prompt "Explain hash tables.", paired native-vs-tree, gateA threshold 0.0) has peeled off the GDN scan; the conv is the current front:
- **Scan fix DONE + committed (e4a6a2f2):** the 2-D `axis=1` reduction (BV=16, re-applying FR12's dropped tile) drives **layers 0-3 = 0.0**. BLOCK_V=1 collapses the degenerate [1,128] (DO NOT use BV=1). Spill at N_PAD=16/BV=16 is a TPS-opt deferred to the TPS gate (num_warps=8 keeps BV=16, or BV=8 — see FR13_BV_SPILL_VERDICT.md).
- **Conv fix APPLIED + committed (37a349f2) but UNVALIDATED + has a RED-TEAM concern (your first job):**

## TASK 1 (FIRST): validate + harden the conv fix
**Root (PROVEN, FR13_L4_CONV_VERDICT.md / 76eeb452):** the tree conv prior-window READ used HEAD cols [0,1,2] but native reads the rolled TAIL (`num_accepted-1`/`state_len-1`). Staircase proof (depth0=3taps 0.0556 -> depth3=0taps 0.0 = only the 3 history columns differ; numeric + source-index ruled out as the first injector).
**What fr18 applied (37a349f2, scripts/fr10_phase4_patch_vllm_tree_gdn.py ~L873):** `_fr10_use_rolled_tail_prior = (layer_idx >= 4)` — tail for layer>=4, head for <4. **THIS IS A BAND-AID, NOT the root fix.** It special-cases layer>=4 (assumes L0-3=prefill-state/head-correct, L4+=tree-handoff/tail-correct — true this step, fragile). The verdict's proper root = fix the **write-back roll (`:1254-1312`)** so the tree handoff stores native's exact `[history…, accepted…]` ordering, then ALL layers read native's tail convention UNIFORMLY (no layer condition).
**Do:** (1) ONE clean filtered re-ladder to confirm the band-aid gives conv1d_out 0.0556 -> 0.0 at L4 (it likely does). (2) THEN replace the layer>=4 band-aid with the uniform write-back fix (or prove the layer-conditional is genuinely correct, not luck). (3) Re-ladder: expect the conv to clear across ALL 48 GDN layers (per-layer convention), jumping first_nonzero to a new op. Commit+bind the hardened fix.

## TASK 2: the next front (likely full_attention)
After the conv clears the GDN conv, the new first_nonzero is likely the **full_attention** subsystem (16 layers, idx 3,7,11,…) = the forked-FA2 `.so` / TREE_ATTN-vs-FLASH_ATTN ~0.0077 seam + the depth-RoPE wiring (MEMORY). That is a DIFFERENT subsystem — do NOT grind the GDN kernel for it. See FR13_SEQ_E2E_ROADMAP.md (ordered fronts + the TRUE branch oracle (4 native /v1/completions on leaf paths, NOT the proxy) + the clean e2e-vs-E5 recipe).

## RECURRING FRICTION (both bit fr18 repeatedly)
- **num_tokens capture filter:** the FR12 subkernel/layer-hidden capture fires on the 2048-token PREFILL/profile pass, not the verify forward -> invalid "all 0.0" captures. Set `FR10_LAYER_HIDDEN_CAPTURE_NUM_TOKENS` / the subop filter to the VERIFY token count (the spine tree size), EXCLUDING 2048. Verify the captured stages are the spec stages (conv1d_out present, not None) before trusting a 0.0.
- **capture-once-native:** pin native ONCE (save request.json), re-run only the tree arm after each fix. The pinned prompt is "Explain hash tables." (request.json in the paired run dirs).

## DISCIPLINE (standing, user)
ONE GPU (no concurrent --gpus; relaunch WITHOUT --rm; recover_host_memory between arms — forked exit wedges ~90GB, sudo pw .lumo.local.env). Bit-exact-or-bust (gateA threshold 0.0, spine AND branch). WIRING vs KERNEL: locate precisely (the conv is WIRING; the scan was KERNEL). NO copy/dense/reroute/splice (verify vs native-on-path oracle, splice OFF, OUR code computing). Commit+push+bind EVERY step (pathspec `git commit -m .. -- <file>` if the monitor may also commit). NO self-declare; the live ladder (conv1d_out->0.0, then all-layers-0.0 spine+branch) is the gate. The monitor runs parallel CPU workflows + red-teams ahead of you (and can be wrong — push back like fr18 did on the BLOCK_V=1 misread).
