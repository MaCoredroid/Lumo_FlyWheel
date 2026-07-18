All checks green. Final report follows.

---

# FR13_PIGGYBACK PHASE-3 + S1 — APPLY REPORT

**Files modified (only these two):**
- `/home/mark/shared/lumoFlyWheel/scripts/fr10_phase4_patch_vllm_tree_gdn.py` (P): **+566 / −15** lines
- `/home/mark/shared/lumoFlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` (K): **+180 / −8** lines

No commits made. No GPU/docker/tmux commands run. No `.sh` files were part of the plan (`bash -n` N/A).

## Per-edit status table

| Wave | Edit | Status | Location (current file:line) |
|---|---|---|---|
| 0 | B1a ring-row buffer alloc | APPLIED (verbatim) | P:218-225 (metadata-builder inject) |
| 0 | B1b ring-row publish | APPLIED (verbatim) | P:231 |
| 0 | B2 prev spec-order stash | APPLIED (verbatim) | P:10508-10510 (before the spec publish) |
| 0 | B3 ring-row map + desync fail-loud | APPLIED **with 1 flagged deviation** (raise pb-gated, see below) | P:10684-10724 |
| 0 | A3 akr ≥9 guard | APPLIED (verbatim) | P:18712-18728 |
| 1 | A1 RoPE `np.maximum(offsets-8,0)` + n≠18 raise | APPLIED (verbatim) | P:9928-9958 (depth-positions inject) |
| 1 | A2 bias ghosting rules [1][2][3] | APPLIED (verbatim; pre-req resolved by S1 spec §3: fp32 / true `-inf` / build-once PASS) | P:14712-14764 (`new_return` head) |
| 2 | B4 hoist + ROUTE=1 raise + None-inits | APPLIED (verbatim) | P:4495, 4514-4532 |
| 2 | B5 row-8-rooted cross-step gather | APPLIED (verbatim, re-indented) | P:5229-5290 (inside `fr10_b == 0`, before ring overwrite) |
| 2 | B6 k/v/a/b scatter rows 0..7 ← ring | APPLIED (verbatim) | P:5322-5351 |
| 2 | B7 launch k=/v= pb-conditional | APPLIED (verbatim) | P:5353-5362 |
| 3 | CONV-1a′ (`conv_parent[8] = -1`, bundle-fenced) | APPLIED (bundle code) | P:290-307 |
| 3 | CONV-1b walk → `conv_parent` | APPLIED (verbatim; `path_node_tensors` + strict/visible walks untouched) | P:311-314 |
| 3 | CONV-2a′ (`_fr10_conv_parent[8] = -1`, LIVE-8 comment, lockstep note retained) | APPLIED (bundle-modified) | P:2971-2986 |
| 3 | CONV-2b (`parent=` + `_fr13_tcf_key`) | APPLIED (verbatim). **Rider NOT applied** — FR13_TCF_SELFCHECK twin decision is open (bundle unresolved #7); no edit was specced | P:3078, 3096 |
| 3 | S2 zero-accept conv leaf 0→8 | APPLIED (bundle's exact replacement) | P:7544-7556 |
| 4 | C-INT-3 prev-bonus stash → gdn module | APPLIED (verbatim) | P:13469-13479 |
| 4 | C-INT-1 marker set + NODE-column APC leaf publish (len==0→`spec_idx[b,8]`) | APPLIED (implemented from Scout-C prose spec — code was prose-specified, not literal) | P:8497-8572 |
| 4 | C-INT-2′ marker check / catch-up / invalidate / seq / eviction | APPLIED (implemented from Scout-C prose spec + bundle `root_node=8` modification) | P:10416-10432 (eviction ext), P:10441-10527 (check/catch-up), P:10559-10570 (seq bump) |
| S1§4 | (a) `ROOT_NODE: tl.constexpr = 0` | APPLIED | K:1018 |
| S1§4 | (b) `node = ROOT_NODE` + amended comment | APPLIED | K:1083-1090 |
| S1§4 | (c) launcher `root_node: int = 0` | APPLIED | K:1298 |
| S1§4 | (d) launcher validation | APPLIED | K:1338-1339 |
| S1§4 | (e) `ROOT_NODE=int(root_node)` in Triton call | APPLIED | K:1440 |
| S1§4 | (f) native-committer threading (call + 2 sigs + layout call + `torch.full` root row) | APPLIED (all 5 sub-edits) | K:1140 (layout sig), K:1151-1154 (root row), K:1170/1173 (replay sig), K:1197-1200 (layout call), K:1402 (route call) |
| S1§4 | (g) `piggyback_catchup_replay` helper (calls `launch_tree_gdn_replay(root_node=8)`) | APPLIED (implemented from Scout-C spec, mirrors the E17 serial call shape P:8694-8714-old) | K:1487-1587 |
| S1§4 | (h) all-layers pb fail-loud (twin stays stock `node = 0`, K:1723) | APPLIED | K:2030-2035 |
| S1 | K1.a `pb_bonus_src` kwarg | APPLIED (verbatim) | K:460 |
| S1 | K1.b 4 fail-loud exits (+`_max_off` extension) | APPLIED (verbatim; branch names `empty-batch/kv`, `no-path-cols`, `qsl-too-short`, `nonuniform-spans-or-span<=max-off`) | K:486-491, 495-500, 506-511, 540-552 |
| S1 | K1.c bonus pair 8→slot-C (before `foreign.any()`) | APPLIED (verbatim) | K:554-588 |
| S1 | P2.a armed-but-remap-off raise | APPLIED (verbatim) | P:18682-18690 |
| S1 | P2.b pb gate + zero-accept cols floor | APPLIED (verbatim) | P:18757-18764 |
| S1 | P2.c `pb_bonus_src=(8 if _fr13_akr_pb else None)` | APPLIED (verbatim) | P:18802 |
| S1 | P3 row-0 full attention ghost | APPLIED (verbatim), textually AFTER A2's rules in the same `new_return`, before the capture `try:` — ordering requirement verified (A2 writes only `-inf`, never 0.0) | P:14765-14789 |
| S1 | P4 `_fr12_native_spine_conv_out` fail-loud | APPLIED (verbatim) | P:3169-3176 |

**REJECTED edits confirmed NOT landed:** Scout C's row-0 cross-seam contract and `conv_parent[9] = 0` reroot — `grep 'conv_parent\[9\]' = 0` hits; both conv reroots are `[8] = -1`.

## Compile results per wave
- Baseline: OK. Wave 0: OK. Wave 1: OK. Wave 2: OK. Wave 3: OK. Wave 4: OK. Step B (root_node): OK. S1-K1: OK. S1-P2/P3/P4 + final: OK (both files).
- **Template-string audit** (py_compile does not check generated code inside the patcher's template strings): AST-extracted every ≥300-char template from baseline and edited files, compiled each (raw / def-wrapped, f-string placeholders substituted). 149 templates checked, **zero compile regressions vs baseline**; every template containing new code (metadata-builder, gdn blob ×2, rejection-sampler committer, depth-positions, REQKEY, E5 packer, `new_return`, KV-remap apply) PASSES as generated code. The two "FAIL" rows are pre-existing folded-constant artifacts identical in the baseline (the full assembled JoinedStr injects PASS in both).

## STOP conditions / deviations
No anchor failures — all anchors matched exactly and uniquely (twins in K disambiguated per the S1 spec's prescribed disambiguators; verified `k_ring.shape` vs `k_rings.shape`, `spec_state_indices` vs `spec_layer` before editing).

**One flagged deviation (B3):** Scout B's payload had the desync raise **ungated**. Reachability proof that it violates flag-off byte-identity: with pb OFF, the E11 chain-lens scatter still computes `1+L` for any spec row with a `_LUMO_FA_TREE_ACCEPT_BY_REQ` entry; on a spec→nonspec→spec interleave (max-len trim, resume — real in the live campaign) the nonspec step publishes `_LUMO_FA_SPEC_ROW_REQ_IDS = []`, so the next spec step sees chain>0 + rid-not-in-prev → the payload's bare raise kills the engine on the **unarmed default path**. This is exactly the E11-ungated-raise class the task orders audited, and it contradicts the bundle's own Wave-0 contract ("Value-inert until pb armed"). Applied fix: the raise is wrapped in `_fr13_piggyback_on()` (identical import-in-branch pattern as the adjacent E11 `>7` clamp hotfix); when unarmed it records ring-row 0 (buffer unread). **Armed behavior is byte-identical to the payload.** If you want the payload's literal ungated raise instead, it is a one-line revert at P:10695-10704.

Minor implementation notes (within spec): C-INT-1, C-INT-2′ and the catch-up helper were prose-specified by Scout C (no literal code existed); implemented exactly per the prose including the seq-discipline (`marker_seq <= current_seq` fail-loud, catch-up only on equality), dead-col-1 doctoring **on a clone**, invalidation clearing marker+by_req+leaf-map+prev-bonus together, and the helper asserting the runrow triple + `flags[0]==1` per layer. The helper reads the pre-overwrite `_LUMO_FA_SPEC_ROW_REQ_IDS` for row→rid mapping (the hook calls it before the publish). C-INT-2′ inherits the REQKEY try/except's fail-loud re-raise (no inner swallow added), per S1 spec §5.

**Not applied (explicitly open in the specs, no edit existed):** CONV-2b's SELFCHECK rider (unresolved 7 — note: `FR13_TCF_SELFCHECK=1` diagnostic would fail under cat9_pb until decided); E12 arming-triple /tmp alignment (unresolved 8); committer-row/spec-row order assert (unresolved 9, required before B>1 agentic gates).

## Flag-gating audit of the default (pb-unarmed) path — every new raise checked
All new raises: B3 (gated by deviation fix), B4 ×2 (`if _fr13_pb_fwd`), A1 (`if _fr10_pb_arm`), A2 ×2 (`if _fr13_pb_on`), A3 (`_fr13_akr_pb_on()`), S1-P2.a (`probe() and env!=1`), S1-K1.b ×4 (`pb_bonus_src is not None`; caller passes None unarmed), S1-P4 (`enabled and _fr13_piggyback_on()`), C-INT-1 ×2 (`if _fr13_pb_drop_replay`), C-INT-2 ×2 (inside `if _fr13_rk_pb_pend` — pend is only ever populated by the pb-gated committer marker), helper raises (only called from the catch-up arm), K(h) (`_fr13_piggyback_on()`), K(d) root_node range (default 0 always passes). **Zero raises reachable unarmed** — the one leak the payloads contained (B3) was caught and gated at apply, and is the only place the applied text differs from a payload.

Unarmed executable-path deltas (all value-inert): B1a buffer alloc+publish (unread); B2/B3 stash + ring-row writes into an unread buffer; A1 cached-False probe; A2/S1-P3 boot-time probe → no mutation (base tree n=10 also fails S1-P3's shape gate); B7/S2 flag-off expressions value-identical (`torch.full_like(x,0)` ≡ `zeros_like`); CONV tables built from `conv_parent == parent` when n≠18 → byte-identical tables and equal `_fr13_tcf_key`; C-INT-2 seq-bump attribute write (unread unarmed); eviction extension no-ops on absent/empty dicts; S1-K1 default `pb_bonus_src=None` preserves all `return 0` exits bit-for-bit. **One acknowledged non-inert effect:** the `ROOT_NODE` constexpr changes the Triton JIT source hash → one boot-time recompile; `ROOT_NODE=0` constexpr-folds to stock `node = 0` (bundle-sanctioned; V0(b) codegen-identity byte A/B must be re-run before any live boot — see gates). Gate asymmetry note: A2 gates on the arm-triple, S1-P3 on shape (n==18); consistent because E1 raises enforce armed↔tree bidirectionally.

## V-gate checklist (copied inline from both specs — none run here; apply-only task)

**V0(b)** — re-run the codegen-identity byte A/B for `_tree_gdn_replay_kernel` (ROOT_NODE constexpr changed the JIT hash; `ROOT_NODE=0` must fold to stock bytes) before any live boot. RED-gate for boot.

**V0(d) — offline CPU/fixtures, BEFORE any GPU run** (extends V0(c)):
- Bias-ghost equivalence + masked-row no-op through forked FA2: apply A2(+S1-P3) to a captured [18,18] bias; live rows {8..17} visibility graph isomorphic to base cat9 under 8↔0, 9..17↔1..9; chain cols 1..7 unreachable from every row; no fully-masked row; push through `apply_tree_bias` (fr13_patch_fa2_tree_bias.py:26-74) — `-inf` hard-mask + finite ghost-row outputs.
- Position-remap unit: armed offsets == `[0,0,0,0,0,0,0,0,0,1,2,2,3,3,4,4,5,5]`; raise on tree_n≠18; mrope twin covered.
- **TWO-STEP ring induction (decisive, settles risk 2 / row-8-vs-0)**: forward N stages rings under E9 identity mask; forward N+1's chain via B5's row-8-rooted gather must byte-equal the base-cat9 replay deposit (extend `scripts/fr13_native_committer_validate.py`). Single-step V0(c) pass ≠ acceptance (false-passes with row 0).
- Row-position invariance probe (FA2 M-row 0 vs 8) — quantify any ULP floor first.
- Conv table CPU gate (retargeted): node-8 window == `(prior ++ x_8)`; nodes 9..17 == base-cat9 windows modulo +8 rank shift with node-8 splice, L∈{0..5}; `build_tree_conv_state_src_indices` twin lockstep; commit-leaf window incl. zero-accept col 8 byte-equal pre-piggyback replay-lifecycle window.
- Leaf-publish law fixture (SSM): len>0→`path[len-1]`, len==0→row 8, vs pre-pb leaf state.
- Catch-up fixture: doctored spec_idx + `root_node=8` deposit == dropped-replay deposit byte-exact; non-target rows deposit into dead col 1 only; **include an all-zero-accept step** (S1 new-risk 1).
- REQKEY units: ring-row map correctness, desync fail-loud, seq-marker partition arms (spec / nonspec+valid / resumed), eviction of marker+stash.

**V1**: flag-OFF same-boot IN-PROCESS byte-identity after the full bundle (never cross-boot).

**V2**: mechanical/CFWD flag-ON with engagement asserts fail-loud (n==18 raise path exercised, ghost applied, B5 gather fired>0, catch-up counters; hit>0/fired>0 before trusting any pass); graph AND eager rows, labeled.

**V2.5 — before V3, EVERY carrier restored-vs-oracle**: GDN col-0 (post-export AND post-catch-up); conv col-0 (post-commit incl. zero-accept); attention paged KV **including bonus slot C** — this is S1's acceptance gate, RED-expected before S1 landed, must now FLIP with S1 in the tree; APC leaf snapshot (cache-ON restore == oracle). Live token gates: same-boot cat9_pb vs base cat9 at identical committed state, per-token argmax probe (never scalar-only), temp 0.6 + fixed seed (never greedy), depth-matched vs E5, no-spec = ground truth, equal A/B sample sizes. Interleave stress: forced nonspec decode between commits, preemption/resume, APC hit/miss boundary — marker/catch-up counters fire, then byte gates re-run. Engagement-needle caveat (S1 new-risk 2): `_fr13_akr_foreign_seen` still counts only foreign rows — do NOT reuse it as "bonus copy alive"; liveness proof = K1.b raises + the slot-C gate.

**V3**: live SWE-Verified only (nudge-free qwen-code, temp 0.6, no AGENT_WALL_S — trace-inactivity watchdog), only after all V2.5 green; graph+eager rows labeled.

**Ship rule reminder (bundle §2/§5)**: Waves 0-4 + S1 = one flag-gated commit series; never arm before V0(d)+V1 green; ATTN edits never ship without variant-B (both are now in-tree together); the tree is left py_compile-clean and default-path byte-identical for the live campaign's mid-apply boots.
