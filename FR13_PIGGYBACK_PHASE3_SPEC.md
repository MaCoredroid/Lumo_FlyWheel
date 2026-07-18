All anchors verified against the current tree. Assembling the bundle now.

---

# FR13_PIGGYBACK PHASE-3 (KV surgery) — assembled bundle

Patcher `P` = `/home/mark/shared/lumoFlyWheel/scripts/fr10_phase4_patch_vllm_tree_gdn.py`. Kernel lib `K` = `/home/mark/shared/lumoFlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`. All scout anchors were re-grepped at assembly time; every anchor cited below is **unique in the current tree** at the line given.

## 1. VERDICT

**Governing insight ("chain tokens already committed ⇒ attention-ghost the chain") — VERIFIED for the attention/KV half, REFUTED as sufficient on its own.**
All three scouts independently confirm the premise: the chain tokens are committed before the forward; their durable KV is already in the paged context at correct positions (stream-0 pos-0 write + FR13_ATTN_KV_REMAP at commit, P:18105-18244); slot mapping runs on flat positions before the depth remap (P:9644-9661). So ghosting chain visibility + clamping positions is correct and necessary. It is NOT sufficient: the GDN scan consumes the chain streams' LIVE k/v/a/b (P:5213-5226; only pos-0/pad rows identity-masked at P:5200-5212), and in-forward recompute of the committed chain hiddens is structurally impossible (each chain row's true causal prefix excludes LATER chain tokens that sit in the paged region, which the [18,18] bias cannot mask and FA2 varlen cannot bound per-row) and would fail the bit-exact bar anyway (~1-ULP/layer × ~492x same-sign amplifier, FR13_AMPLIFICATION_PHYSICS).

**Ring-injection (variant B) REQUIRED — YES** (Scout B item 3): (i) ghosted rows' recomputed projections are garbage, not drift; (ii) even ℝ-correct recompute is a different kernel route and fails bit-exactness; (iii) the only byte-exact carry proof (V0(c)) was obtained feeding ring bytes, and recompute lacks the chain's original conv1d prior windows. **Ship rule: the ATTN edits must NEVER land without the variant-B GDN edits.**

**ONE DECLARED DESIGN — "LIVE-8"** (required by Scout C risk 1; resolves the cross-scout conflict):
- **Stream 8** = live root twin: attends paged+self at RoPE base+0 (A1 clamp + A2 rule [3]); scan-ACTIVE bonus application. This is already the baked E9 law — P:5190-5193 says stream 0's state update is "deferred to the ACTIVE duplicate at stream 8". Its own projections are base-correct; no input remap.
- **Streams 1..7** = full ghosts: attention-ghosted (A2 rules [1][2]), scan inputs ring-fed (B5/B6/B7) with **root source = prev ring row 8** (Scout B CORRECTION 1).
- **Stream 0** = transitional live self-only KV writer — **poisoned after the first GDN layer** (Scout B Risk 2): its GDN output is `q·h_prev` without the bonus update, so its later-layer hidden ≠ base root, so its full-attn K/V writes to the canonical bonus slot C are wrong bytes. This is blocker **S1** (no edit exists in any scout).
- **REJECTED**: Scout C's cross-seam contract ("source stream 8's scan inputs from ROW 0") and its `conv_parent[9] = 0` reroot. Both are contradicted by Scout B's verified row-0 post-L0 divergence — row 0's later-layer x/k/v are wrong bytes, so anything tapping row 0 past layer 0 inherits the poison. Scout C wrote its contract under the assumption stream 8 is attention-ghosted; under LIVE-8 it is not. Conv edits are **retargeted to node 8** (see Wave 3): `conv_parent[8] = -1` gives node 8 the exact window `(prior ++ x_8)` and subtree nodes `(prior ++ x_8 ++ path)` — same tokens as base cat9's windows, uncorrupted bytes, static table, exact for every chain length. Do not hand-flip any of these back without re-deriving (Scout B risk 1 discipline).
- **Assembly-discovered defect in C-INT-2(3)**: the catch-up helper calls `launch_tree_gdn_replay`, whose kernel replays **ring node 0 as root unconditionally** (K:1020-1027). Under LIVE-8 that deposits the diverged pos-0 row. The helper needs a `root_node` arg (default 0 = stock; pass 8) — see C-INT-2′.
- **Assembly-discovered gap S2 (now an apply-ready edit)**: zero-accept conv commit sources node col 0 (P:7388-7391 verified) — diverged bytes at all layers past L0-GDN. Redirect to col 8 under pb (Wave 3, edit 15).

## 2. Dependency-ordered, conflict-checked edit list

All edits are flag-gated (arm triple / `_fr13_piggyback_on()` / `_fr13_pb_fwd` / shape-keyed n==18 detect); flag-off is byte-identical (verified per-edit below; V1 asserts it). For edits marked **verbatim**, apply exactly the code in the named scout payload (the caller holds it); anchor + verified line are restated. Full code is given only where assembly modified or added an edit.

### Wave 0 — infra + guards (no behavior until later waves)
| # | Edit | Anchor (verified) | Notes |
|---|------|-------------------|-------|
| 1 | **B1a+B1b** verbatim (ring-row map buffer + publish) | `self.fr13_pb_chain_lens = torch.zeros(` P:213; publish line P:222 | New buffer, nothing reads it yet. |
| 2 | **B2** verbatim (prev spec-order stash) | insert immediately before `_fr13_rk_gdn._LUMO_FA_SPEC_ROW_REQ_IDS = _fr13_rk_spec_rids` P:10224 | Observe-only. |
| 3 | **B3** verbatim (ring-row map write + desync fail-loud) | after the `_fr13_rk_pb_lens[:_fr13_rk_n].copy_(...)` block P:10368-10374 | Depends on 1, 2. Value-inert until pb armed. |
| 4 | **A3** verbatim (akr accepted-rows ≥9 guard) | `_fr13_akr_rows.append(` P:18168 | Defensive; `_fr13_akr_pb_on()`-gated. |

### Wave 1 — attention + position surgery
| # | Edit | Anchor (verified) | Notes |
|---|------|-------------------|-------|
| 5 | **A1** verbatim (RoPE `np.maximum(offsets-8,0)` + n==18 raise) | `_fr10_depth_offsets = np.array(` P:9691 | Boot-cached arm; covers mrope twin. |
| 6 | **A2** verbatim (bias ghosting [1][2][3] in `new_return` head) | `new_return = f"""    {mask_sentinel}: dump the runtime root/bonus attention bias row.` P:14267 | **Pre-req: read LIVE container tree_attn.py first** (bias dtype/-inf convention + build-once) — unresolved item 2. |

### Wave 2 — GDN variant-B ring feed (licenses the Wave-1 ghosting)
| # | Edit | Anchor (verified) | Notes |
|---|------|-------------------|-------|
| 7 | **B4** verbatim (hoist: paths/ring-row tensors + ROUTE=1 raise + None-inits) | `"_LUMO_FA_PB_CHAIN_LENS_TENSOR"` raise block P:4469 region (:4460-4479) | Depends on 1. |
| 8 | **B5** verbatim (row-8-rooted cross-step gather in the `fr10_b == 0` snapshot branch) | insert after `self._fr13_replay_conv_state = conv_state` P:5172, before ring overwrite P:5173 | Depends on 3, 7. Root source = ring row 8 (declared design). |
| 9 | **B6** verbatim (k/v/a/b scatter, rows 0..7 ← ring bytes) | replaces the two `torch.where(_fr13_pb_ident, ...)` blocks P:5207-5212 (`_fr13_pb_ident` at P:5200 stays) | Depends on 8. Finite bytes on all rows 0..7 = NaN hygiene for the identity step. |
| 10 | **B7** verbatim (launch k=/v= pb-conditional) | `tree_out, _ = launch_tree_gdn_prepared(` P:5213-5216 | Depends on 9 (uses `_fr13_pb_k/_fr13_pb_v`). Flag-off expressions identical. |

### Wave 3 — conv tables, RETARGETED to node 8
| # | Edit | Anchor (verified) | Code |
|---|------|-------------------|------|
| 11 | **CONV-1a′** (modified) | between `"                path_node_tensors.append(...)\n"` P:279 and `"            for width in range(2, 7):\n"` P:280 | Scout C's CONV-1a with the reroot line and comment replaced: |

```python
            "            conv_parent = list(parent)\n"
            "            _fr13_pb_ext_tree = (\n"
            "                n == 18\n"
            "                and all(tree_choices[_pbk] == tuple([0] * (_pbk + 1)) for _pbk in range(8))\n"
            "            )\n"
            "            if _fr13_pb_ext_tree:\n"
            "                # FR13_PIGGYBACK conv ancestry (LIVE-8): chain streams 1..7 are\n"
            "                # conv-GHOSTS (windows dead: scan inputs ring-fed) and stream 8 is\n"
            "                # the LIVE root twin. Conv col-0 is advanced AT COMMIT (post-chain),\n"
            "                # so node 8 must NOT re-append the chain: make node 8 a conv-ROOT\n"
            "                # (window = prior ++ x_8, the exact bonus window from the properly-\n"
            "                # attending row 8). Subtree keeps parent 9->8, so subtree windows =\n"
            "                # prior ++ x_8 ++ path == base cat9's windows byte-for-byte intent.\n"
            "                # NOTE: NOT node 0 -- row 0 is GDN-identity-masked and its post-L0\n"
            "                # x rows diverge from the true root (scout-B row-0 divergence).\n"
            "                # SCAN parent/strict/visible masks + path_node_tensors keep FULL\n"
            "                # extended ancestry (the GDN scan NEEDS the chain).\n"
            "                conv_parent[8] = -1\n"
```

| # | Edit | Anchor (verified) | Code |
|---|------|-------------------|------|
| 12 | **CONV-1b** verbatim (`source_rows` walk uses `conv_parent`) | `ancestry = []` walk block P:283-287 (unique; the `path_node_tensors` walk at P:274/277 uses `cur = node` and must NOT change) | Depends on 11. |
| 13 | **CONV-2a′** (modified) | after the parent build `_fr10_index = {_p: _i + 1 ...` / `_fr10_parent = [-1]` P:2935-2943 | Scout C's CONV-2a with `_fr10_conv_parent[9] = 0` → `_fr10_conv_parent[8] = -1` (comment updated to the LIVE-8 rationale, "both twins MUST stay in lockstep" retained). |
| 14 | **CONV-2b** verbatim (`parent=_fr10_conv_parent` at the `build_tree_conv_state_src_indices(` call P:3052 + `tuple(_fr10_conv_parent)` in `_fr13_tcf_key` P:3034) | Depends on 13. Plus its rider: gate `FR13_TCF_SELFCHECK` off under cat9_pb (or update its twin) — decision open (unresolved 7). |
| 15 | **S2 (NEW, assembly)** — zero-accept conv leaf 0→8 | plain code (not inject-string) in `_fr13_conv_commit_to_col0`, P:7388-7391: |

Anchor (exact, verified):
```python
        # zero-accept rows commit the ROOT (col 0) window (acc_len==0 -> col0).
        _leaf_node = torch.where(
            _alen > 0, _leaf_node, torch.zeros_like(_leaf_node)
        ).clamp(0, _spec_cols - 1).to(torch.long)
```
Replace with:
```python
        # zero-accept rows commit the ROOT window. Base cat9: node col 0.
        # FR13_PIGGYBACK (LIVE-8): node col 8 -- row 0 is GDN-identity-masked
        # so its post-L0 conv x rows diverge; stream 8 is the ACTIVE bonus row
        # whose retargeted window (prior ++ x_8, conv_parent[8] = -1) carries
        # the SAME token with uncorrupted bytes. Matches the E12 row-8 SSM law.
        from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
            _fr13_piggyback_on as _fr13_cc_pb_on,
        )
        _fr13_cc_zero_col = 8 if _fr13_cc_pb_on() else 0
        _leaf_node = torch.where(
            _alen > 0, _leaf_node,
            torch.full_like(_leaf_node, _fr13_cc_zero_col),
        ).clamp(0, _spec_cols - 1).to(torch.long)
```
Rationale: on zero-accept, col-0's post-commit invariant is "window through committed except trailing bonus" = `(prior ++ x_bonus)`; node 8's retargeted window is exactly that from the live row. pb-off → 0 → byte-identical (base tree_n=10 unaffected). Depends on 11/13/14 (node-8 write-back row must be retargeted first).

### Wave 4 — interleave lifecycle (Risk 4 control/data plane)
| # | Edit | Anchor (verified) | Notes |
|---|------|-------------------|-------|
| 16 | **C-INT-3** verbatim (E5 prev-bonus stash → gdn module) | `global _FR13_PB_PREV_BONUS` block P:13241-13245 (`_fr13_pb_gdn` in scope, P:13205) | Apply BEFORE 18 so the gdn-hosted dict exists for invalidation. |
| 17 | **C-INT-1** verbatim (commit marker set + NODE-column APC leaf publish; len==0 → `spec_idx[b, 8]`) | after the `_FR13_PB_DROP_ANNOUNCED` announce block P:8374-8380 | Row-8 zero-accept law already consistent with LIVE-8. |
| 18 | **C-INT-2′** (modified) — pre-forward marker check / one-shot catch-up / invalidation / seq / eviction | insert after `_fr13_rk_gdn._LUMO_FA_NONSPEC_ROW_REQ_IDS = (...)` P:10196-10198, BEFORE the P:10224 spec publish; eviction extension at `_fr13_rk_live = set(_fr13_rk_req_ids)` P:10173-10176 | **Modification**: `piggyback_catchup_replay(gdn_mod, catchup_rids)` (new helper in K next to `launch_tree_gdn_replay`) must pass a new `root_node=8` arg — the stock kernel replays ring node 0 as root (K:1020-1027), which under LIVE-8 is the diverged pos-0 row. `root_node` defaults 0 so the stock path is byte-identical. Everything else per Scout C spec (dead-col-1 spec_idx doctoring, seq marker, resumed→invalidate-only, clears marker + by_req + leaf-map + prev-bonus together). Depends on 16, 17; coexists with B2/B3 (disjoint insertion points :10198+ / :10223+ / :10368+ — no anchor overlap, both prev-spec reads precede the :10224 overwrite). |

### Wave 5 — BLOCKER, spec-only (needs its own scout before any accept/quality gate)
**S1 — bonus durable-KV fix**: commit-time copy of stream 8's per-layer scratch KV (flat offset 8) → the canonical bonus slot C (flat offset 0's slot), all full-attn layers; pure slot copy — stream 8's K is already RoPE'd at the base position via A1's clamp, no re-rotation; extend `launch_attn_kv_linear_remap` (K:508-537 region) with the extra (src=sm[qsl+8] → dst=sm[qsl+0]) pair; then make row 0 a full attention ghost in A2 (`tree_attn_mask[..., 0:1, 1:] = -inf` equivalent — row 0 attends paged only) and audit remaining row-0 consumers. Until S1 lands, cat9_pb is NOT byte-exact one step later (stale bonus KV) — only mechanical/CFWD gates are meaningful; the V2.5 slot-C carrier gate is S1's acceptance test.

**Ship rule**: Waves 0-4 land as ONE flag-gated commit series (each step committed+pushed); never arm before V0(d)+V1 green; never run accept/quality gates before S1.

## 3. Ranked residual risks

1. **S1 blocker — stream-0 bonus-KV poisoning** (Scout B risk 2): unfixed; every quality gate blocked on it. The slot-C restored-vs-oracle gate must stay RED-expected until landed, then flip.
2. **Ring root-source convention (row 8 vs 0)**: declared row 8; MUST be fixture-settled by the two-step induction gate — a single-step V0(c) run **false-passes with row 0** (its fixtures came from base-cat9 rings). Applies equally to C-INT-2′'s `root_node=8`. Export-index bug class: wrong pick = silent per-step garble.
3. **Cross-step ring lifetime / seq discipline** (B risk 3 + C risk 4): B3 guards set/order changes but not same-order stale-chain interleaves; C-INT-2′'s seq marker is the single authority — assert `marker_seq <= current_seq`, catch-up only on equality. An off-by-one = rare unreproducible garble (bug-class 12).
4. **Zero-accept / leaf-law semantics fixture-unproven** (C risk 5 + S2): len>0→`path[len-1]`, len==0→col/row 8 (conv AND SSM) must byte-match the pre-piggyback replay-lifecycle committed state on captured fixtures before any cache-ON gate.
5. **Stream-8-as-root FA2 tile/M-position ULP** (B risk 4): row 8 vs base row 0 may differ at the single-ULP floor (FA2-fork precedent, MMA grouping); ring bytes staged for the next chain inherit it. Quantify offline (row-position invariance probe) before the induction gate; may require an explicit accepted-floor decision.
6. **Bias convention + boot-static assumption** (A risk 1): -inf vs finfo.min, fp32, build-once — read the LIVE container source before landing A2 (feedback_read_vllm_source_first).
7. **NaN hygiene residual** (B risk 5): state protected by B6's finite scatter; ghost rows never fully masked (paged cols unbiased, context_len ≥ 1); residual = ghost-row q/out NaN if any future all-row consumer appears.
8. **Committer-row vs spec-row order under mixed batches** (C risk 2): leaf publish + catch-up key off `_fr13_spec_req_ids`; add order assert (or req-keyed join) before any B>1 agentic gate.
9. **Capture safety** (B risk 6 + C risk 8): new per-step device ops have in-block precedent; catch-up is an eager launch between graph replays (legal); marker machinery host-side. Re-gate and label graph vs eager on every row.
10. **Secondary consumers of the mutated bias** (A risk 6): build_for_drafting slices, qq_bias fallback, FR13_FA2_SPINE_REORDER all--inf suffix rows — verify unconsumed/compatible before arming those paths.
11. **Arming split-brain + K=8 hard-coding** (A risks 4-5): E12 lacks /tmp in its triple; the -8 clamp / 1:8 / col-8 constants must move with any future cap change (fail-louds make partial change loud).
12. **SNAP_FIX-off silent stock snapshot** (C risk 6): add fail-loud "pb armed + APC on ⇒ FR13_APC_SNAP_FIX=1".
13. **Catch-up dead-col-1 future readers** (C risk 3) and **flags[0] discipline** (B risk 8 vs C-INT-2′'s `flags[0].fill_(0)`): variant-B depends on unconditional ring staging — keep B4's ROUTE=1 raise; declare one owner for flags semantics (unresolved 6).

## 4. V-gate additions

**V0(d) — offline, CPU/fixtures, BEFORE any GPU run** (extends V0(c)):
- **Bias-ghost equivalence + masked-row no-op through the forked FA2**: apply A2 to a captured [18,18] bias; assert live rows {0,8..17} visibility graph isomorphic to base cat9 under 8↔0, 9..17↔1..9; chain cols 1..7 unreachable from every row; no fully-masked row; push a fixture through `apply_tree_bias` (fr13_patch_fa2_tree_bias.py:26-74) confirming -inf hard-mask + finite ghost-row outputs.
- **Position-remap unit**: armed offsets == `[0,0,0,0,0,0,0,0,0,1,2,2,3,3,4,4,5,5]`; raise on tree_n≠18; mrope twin covered.
- **TWO-STEP ring induction (decisive, settles risk 2)**: extended forward N under the E9 identity mask stages rings; forward N+1's chain fed via B5's row-8-rooted gather must byte-equal the base-cat9 replay deposit (extend `scripts/fr13_native_committer_validate.py`). Single-step pass ≠ acceptance.
- **Row-position invariance probe** (risk 5): same token/position at FA2 M-row 0 vs 8 → byte-compare; quantify any ULP floor first.
- **Conv table CPU gate (retargeted)**: node 8 window == `(prior ++ x_8)`; nodes 9..17 == base-cat9 windows modulo +8 rank shift with node-8 splice, L∈{0..5}; `build_tree_conv_state_src_indices` twin lockstep; commit-leaf window (incl. zero-accept col 8) byte-equal the pre-piggyback replay-lifecycle window.
- **Leaf-publish law fixture (SSM)**: len>0→`path[len-1]`, len==0→row 8, vs pre-pb leaf state.
- **Catch-up fixture**: doctored spec_idx + `root_node=8` deposit == dropped-replay deposit byte-exact; non-target rows deposit into dead col 1 only.
- **REQKEY units**: ring-row map correctness, desync fail-loud, seq-marker partition arms (spec / nonspec+valid / resumed), eviction of marker+stash.

**V1**: flag-OFF same-boot IN-PROCESS byte-identity after the full bundle (never cross-boot).

**V2**: mechanical/CFWD flag-ON with engagement asserts fail-loud (arm actually reached: n==18 raise path exercised, ghost applied, B5 gather fired>0, catch-up counters; hit>0/fired>0 before trusting any pass); graph AND eager rows, labeled.

**V2.5 — before V3** (per feedback_verify_notstale_all_carriers_before_swe):
- **Per-carrier restored-vs-oracle, EVERY carrier**: GDN col-0 (post-export AND post-catch-up), conv col-0 (post-commit incl. zero-accept), attention paged KV **including bonus slot C** (RED-expected until S1; = S1's acceptance gate), APC leaf snapshot (cache-ON restore == oracle).
- **Live token gates**: same-boot cat9_pb vs base cat9 at identical committed state — per-token argmax probe (never scalar-only), temp 0.6 + fixed seed (never greedy), depth-matched vs E5 bar, no-spec = ground truth, equal A/B sample sizes.
- **Interleave stress**: forced nonspec decode between commits, preemption/resume, APC hit/miss boundary — marker/catch-up counters fire, then byte gates re-run.

**V3**: live SWE-Verified only (nudge-free qwen-code, temp 0.6, no AGENT_WALL_S — trace-inactivity watchdog), only after S1 landed + all V2.5 green; graph+eager rows labeled.

## 5. UNRESOLVED

1. **S1** — bonus slot-C KV copy + row-0 full ghost: no edit produced by any scout; needs its own scout (K remap-kernel extension + A2 delta + row-0 consumer audit, incl. asserting `_fr12_native_spine_conv_out` off under cat9_pb).
2. **Live-container tree_attn.py bias convention / build-once** — asserted from the SR perm-cache contract, never read (blocks A2 landing).
3. **`piggyback_catchup_replay` + `root_node` kernel-lib anchors** — K:1020-1036/1148-1151 are read-refs only; the helper and kernel arg have no scouted anchors.
4. **`scheduler_output.scheduled_cached_reqs.resumed_req_ids` scope at the REQKEY hook** — the pattern exists in a different inject (P:11485); `scheduler_output` is in the surrounding function (P:10094 region) but the exact attribute path at :10198 is verify-at-apply.
5. **Drafting-path consumers of the mutated bias under cat9_pb** (A risk 6) — unverified.
6. **`_fr13_replay_flags[0]` ownership** — latched under pb (dropped consumers) vs catch-up zeroing vs variant-B's need for every-forward staging: one declared discipline missing.
7. **FR13_TCF_SELFCHECK twin** — gate-off under cat9_pb vs update to walk `_fr10_conv_parent`: decision open (either satisfies CONV-2b's rider).
8. **E12 arming-triple alignment** (/tmp missing) — follow-up edit not specced.
9. **Committer-row vs spec-row order assert** (C risk 2) — specced as ride-along, no anchor; required before B>1 agentic gates.
10. **Scout C's Q3 "multi-turn moot" verdict** is conditional on the INT-1 NODE-column leaf publish being fixture-proven (item 4 of §3 risks) — not yet proven.
