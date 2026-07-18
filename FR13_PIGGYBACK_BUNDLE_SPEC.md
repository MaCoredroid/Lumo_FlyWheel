# FR13_PIGGYBACK INTEGRATION BUNDLE — seams 1b/1c/1d + 2 + 4 + 5 + serve wiring

Assembled from PACKER, CALLER_REPLAY, COMMITTER, TOPOLOGY. All load-bearing anchors and the two factual disputes were re-verified against the working tree (refs below are fresh `file:line` as of now). Kernel-side pieces confirmed landed: `_fr13_piggyback_on`/`_cap` (src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:22/34), `PIGGYBACK_EXPORT`/`CHAIN_END_IDX` constexprs (:739-740), export block (:909-911), launch threading (:2218-2219).

## 0. BINDING SCHEME RESOLUTION (governs three edits — read before applying)

The scouts split on the chain/bonus composition. The bundle adopts **Scheme A (PACKER's), end-to-end**, because it is the only one specified at every seam (packer tokens + 1c mask + export index + induction) and it preserves **today's col-0 invariant** (state through all committed EXCEPT the trailing bonus = exactly `fr13_native_committer_validate.py`'s replay deposit), making the offline gate a direct byte-compare:

- Chain streams 1..L+1 = `[prev pos-0 root token] + prev accepted drafts` (real chain len = 1+prev_accept_len).
- Stream 0 (vLLM's pos-0 bonus copy) = **GDN-identity-masked**; the bonus is applied ONCE, at stream 8 = `(0,)^8` = the ACTIVE subtree root.
- **`chain_end_idx = 7`** (scan-STREAM space, `(0,)^7`) — export is the PRE-bonus state. CALLER_REPLAY/TOPOLOGY's `cap()=8` is correct only under their (unbuilt) drafts-only chain convention; **under Scheme A, 8 double-applies the bonus every step**. All three scouts agree on the indexing facts (root=stream 0, choice i=stream i+1); the disagreement was scheme, not arithmetic. Guard: E8 asserts `_fr13_piggyback_cap()==8`; committer walk root stays `cap()-1 = 7` in CHOICES-rank space (same number, different space — deliberate, documented in code comments).
- **Chain token source AMENDED** (dispositive verification): the merged-drafter lifecycle runs at `_prepare_inputs` (patcher:17472 docstring, ingest at :17504), so `_COMMITTED` at propose-time is one commit stale — PACKER's `_COMMITTED[-(L+2):-1]` source would pack last step's window and its "bonus" `_COMMITTED[-1]` is not b_N. Fresh sources verified: `_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS` = drafts-only accepted rows (built patcher:8013, published :8127-8129, sampler-row-ordered, before propose), `_LUMO_FA_TREE_ACCEPT_BY_REQ` = `(paths, len)` only (:8154-8157). E5 uses: accepts from the publish, prev-root from a self-priming per-req stash of the previous pack's bonus, current bonus from the propose-scope `next_token_ids`. This also **dissolves the `_COMMITTED` dependency entirely** (TOPOLOGY edit [2] dropped, PACKER risk R8 gone).

Unified flag name for the drafter shape: **`_fr10_is_cat9_pb`** (PACKER's `_fr13_pb_cat9` renamed in E4/E5/E6).

## 1. ORDERED, APPLY-READY EDIT LIST

Apply top-to-bottom. `SCOUT.edits[i]` = apply that scout's edit verbatim (the orchestrator holds the scout JSONs); modifications are stated exhaustively. Edits E1–E17 + E22 target `/home/mark/shared/lumoFlyWheel/scripts/fr10_phase4_patch_vllm_tree_gdn.py`; E18–E21 target `/home/mark/shared/lumoFlyWheel/scripts/fr13_device_multidraft_kernel.py`; E23–E24 target `/home/mark/shared/lumoFlyWheel/scripts/fr13_bigdenom_swe_serve_variant.sh`.

**Group 1 — drafter (eagle blob; E1 defines the name E2–E6 use; apply atomically)**

| # | Seam | Edit |
|---|------|------|
| E1 | 1b-0 detect + bidirectional fail-loud | `TOPOLOGY.edits[3]` **verbatim**. Anchor `_fr10_is_333 = (` block verified at :11849. Defines `_fr13_pb_choices` (canonical sorted 17-tuple), `_fr13_pb_armed` (/logs sidecar OR env — `os` confirmed in scope, cf. :11960), `_fr10_is_cat9_pb`, and BOTH raises (tree-without-arm, arm-without-tree). |
| E2 | 1b-i wide exclusion | `TOPOLOGY.edits[4]` **verbatim**: add `or _fr10_is_cat9_pb` to the `_fr10_is_wide` not-list. Anchor verified :11883-11887; the `\n            )\n        )` tail makes it unique vs the dispatch (:11998-12001, 12-space, has `or _fr10_is_wide`) and disengage guards. **PACKER.edits[0] (force `_fr10_is_wide=False`) is superseded — do NOT also apply.** |
| E3 | 1b spine-steps | `TOPOLOGY.edits[5]`: explicit `elif _fr10_is_cat9_pb: _fr10_spine_steps = 4` between :11950 and :11951 (anchor verified). Redundant with the `else: 4` at :11964-11965 but locked-in explicit; leaf-steps needs NO edit (cat9_pb correctly falls to the `else: frozenset({1,2,3,4})` at :11984-11985 — verified it is NOT in the :11970-11975 empty-leaf list). |
| E4 | 1b-ii dispatch | `PACKER.edits[1]` with rename `_fr13_pb_cat9` → `_fr10_is_cat9_pb`. Anchor verified :11993-12002. |
| E5 | 1b-iii chain packer | **AMENDED** — PACKER.edits[3] structure with swapped token sources. Full replacement code below. Anchor (the cat9 else-stack) verified byte-exact at :12934-12948, insert AFTER it at 12-space indent. |
| E6 | 1b-ii' disengage guard | `PACKER.edits[2]` with rename → `_fr10_is_cat9_pb`. Anchor at the guard above the "FR10 caterpillar drafter disengaged" raise (:13420). |

**E5 code** (replaces PACKER.edits[3]'s change block; rationale for every deviation in §0):

```python
            if _fr10_is_cat9_pb:
                # FR13_PIGGYBACK seam 1b: prepend the 8 chain columns
                # ((0,)^1..(0,)^8 = packed cols 0..7) ahead of the 9 base-cat9
                # columns. ORDER-CRITICAL (fr13_native_committer_validate.py:
                # replay path = [prev pos-0 root token] + prev accepted
                # drafts): cols 0..L = [prev root] + THIS commit's accepted
                # drafts; cols len..6 repeat-pad (GDN-identity via seam 1c,
                # token value inert); col 7 = the CURRENT bonus (vLLM feeds
                # the same token at stream 0 next step; that copy is
                # identity-masked by seam 1c — the ACTIVE application is at
                # stream 8 = the subtree root). Sources are all FRESH at
                # propose-time; _COMMITTED is NOT used (runner lifecycle
                # ingests at _prepare_inputs = one commit stale here).
                from vllm.model_executor.layers.mamba import (
                    gdn_linear_attn as _fr13_pb_gdn,
                )
                if int(_fr10_packed.shape[1]) != 9:
                    raise RuntimeError(
                        "FR13_PIGGYBACK: expected the base-cat9 9-col pack, "
                        "got " + str(int(_fr10_packed.shape[1]))
                    )
                _fr13_pb_B = int(_fr10_packed.shape[0])
                _fr13_pb_ids = getattr(
                    _fr13_pb_gdn, "_LUMO_FA_SPEC_ROW_REQ_IDS", None
                )
                if _fr13_pb_ids is None or len(_fr13_pb_ids) != _fr13_pb_B:
                    _fr13_pb_ids = getattr(
                        _fr13_pb_gdn, "_LUMO_FA_SAMPLER_ROW_REQ_IDS", None
                    )
                if _fr13_pb_ids is None or len(_fr13_pb_ids) != _fr13_pb_B:
                    raise RuntimeError(
                        "FR13_PIGGYBACK: no row-aligned request ids for the "
                        "chain packer (B=" + str(_fr13_pb_B) + ")"
                    )
                _fr13_pb_by_req = getattr(
                    _fr13_pb_gdn, "_LUMO_FA_TREE_ACCEPT_BY_REQ", None
                ) or {}
                _fr13_pb_tok_by_req = {
                    str(_r): _t
                    for _r, _t in zip(
                        getattr(
                            _fr13_pb_gdn,
                            "_LUMO_FA_SAMPLER_ROW_REQ_IDS", None,
                        ) or [],
                        getattr(
                            _fr13_pb_gdn,
                            "_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS", None,
                        ) or [],
                    )
                }
                global _FR13_PB_PREV_BONUS
                try:
                    _FR13_PB_PREV_BONUS
                except NameError:
                    _FR13_PB_PREV_BONUS = {}
                # APPLY-TIME BIND: next_token_ids is the propose() input of
                # sampled tokens, row-aligned with _fr10_packed. Verify the
                # parameter name in the live-container eagle propose signature
                # before landing (feedback_read_vllm_source_first).
                _fr13_pb_bonus_t = next_token_ids[:_fr13_pb_B].tolist()
                _fr13_pb_rows = []
                _fr13_pb_live = set()
                for _fr13_pb_b in range(_fr13_pb_B):
                    _fr13_pb_rid = str(_fr13_pb_ids[_fr13_pb_b])
                    _fr13_pb_live.add(_fr13_pb_rid)
                    _fr13_pb_entry = _fr13_pb_by_req.get(_fr13_pb_rid)
                    _fr13_pb_prev = _FR13_PB_PREV_BONUS.get(_fr13_pb_rid)
                    _fr13_pb_bonus = int(_fr13_pb_bonus_t[_fr13_pb_b])
                    if (_fr13_pb_entry is None) != (_fr13_pb_prev is None):
                        raise RuntimeError(
                            "FR13_PIGGYBACK: accept-publish/prev-bonus stash "
                            "desync for req " + _fr13_pb_rid
                        )
                    if _fr13_pb_entry is None:
                        _fr13_pb_toks = []
                    else:
                        _fr13_pb_L = int(_fr13_pb_entry[1])
                        _fr13_pb_acc = [
                            int(_x) for _x in
                            _fr13_pb_tok_by_req.get(_fr13_pb_rid, [])
                        ]
                        if len(_fr13_pb_acc) != _fr13_pb_L:
                            raise RuntimeError(
                                "FR13_PIGGYBACK: accepted-token row len "
                                + str(len(_fr13_pb_acc)) + " != by_req L "
                                + str(_fr13_pb_L) + " for req "
                                + _fr13_pb_rid
                            )
                        if _fr13_pb_L + 1 > 7:
                            raise RuntimeError(
                                "FR13_PIGGYBACK: chain does not fit (L="
                                + str(_fr13_pb_L) + ")"
                            )
                        _fr13_pb_toks = [int(_fr13_pb_prev)] + _fr13_pb_acc
                    _fr13_pb_fill = (
                        _fr13_pb_toks[-1] if _fr13_pb_toks else _fr13_pb_bonus
                    )
                    _fr13_pb_rows.append(
                        _fr13_pb_toks
                        + [_fr13_pb_fill] * (7 - len(_fr13_pb_toks))
                        + [_fr13_pb_bonus]
                    )
                    _FR13_PB_PREV_BONUS[_fr13_pb_rid] = _fr13_pb_bonus
                for _fr13_pb_dead in [
                    _k for _k in _FR13_PB_PREV_BONUS
                    if _k not in _fr13_pb_live
                ]:
                    _FR13_PB_PREV_BONUS.pop(_fr13_pb_dead, None)
                _fr10_packed = torch.cat(
                    [
                        torch.tensor(
                            _fr13_pb_rows,
                            dtype=_fr10_packed.dtype,
                            device=_fr10_packed.device,
                        ),
                        _fr10_packed,
                    ],
                    dim=1,
                )
                logger.info_once(
                    "FR13_PIGGYBACK cat9_pb drafter engaged: 17 cols "
                    "(8 chain + 9 MTP)"
                )
```

Self-priming induction (verified): propose(prefill) — no by_req, no stash → chain len 0, col7 = first sampled token, stash primed; 1d-ii mask writes lens=0 for the first tree forward (by_req absent) → all-identity chain ✓. Every later step: chain len = 1+L on both packer and mask sides from the same `by_req` state (unchanged between propose(N) and prepare(N+1) since commit(N+1) is later). The engage log line is the E24 needle.

**Group 2 — tree-scan forward (gdn_linear blob)**

| # | Seam | Edit |
|---|------|------|
| E7 | 1c/2-import | Merge of PACKER.edits[4] + CALLER_REPLAY.edits[0]: replace the injected import line (verified unique, :786 region) with `"from lumo_flywheel_serving.fr10_gdn_tree_kernel import _fr13_piggyback_cap, _fr13_piggyback_on, gather_committed_path_conv_prior, launch_tree_gdn_prepared, launch_tree_state_linear_remap\\n"`. |
| E8 | 1c-hoist | `PACKER.edits[5]` **plus** insert, immediately after the `tree_n != 18` raise inside `if _fr13_pb_fwd:`: `if int(_fr13_piggyback_cap()) != 8: raise RuntimeError("FR13_PIGGYBACK: prefix cap " + str(int(_fr13_piggyback_cap())) + " != 8; cat9_pb layout/mask/export hard-code K=8 (chain streams 1..7, export@7, subtree root stream 8)")` (same 20-space indent). Anchor (`tree_n_pad = int(...)` line) verified unique. This closes COMMITTER R3 / cap-desync. |
| E9 | 1c-mask + seam-2 rider | `PACKER.edits[6]` **verbatim** (`chain_end_idx=7` per §0). Anchor (`tree_out, _ = launch_tree_gdn_prepared(` + q/k head) verified unique. **Supersedes CALLER_REPLAY.edits[1]+[2] and TOPOLOGY.edits[7] — do NOT apply those**: their `invocation_counter` tail anchor still exists after E9, and applying either on top would produce duplicate `piggyback_export=`/`chain_end_idx=` kwargs (TypeError at first tree forward) or, worse, a mixed 7/8 export. |

**Group 3 — chain-lens buffer + pre-forward scatter + zero-accept row map**

| # | Seam | Edit |
|---|------|------|
| E10 | 1d-i buffer | `PACKER.edits[7]` **verbatim**. Anchor (`self.fr10_tree_accepted_lens = torch.zeros(` block) verified unique. |
| E11 | 1d-ii scatter | `PACKER.edits[8]` **verbatim**, WITH its `change_note` enforced: the insert must land dedented to the `if _fr13_rk_spec_rids:` body level (28-space body, first line aligned with `_ep_pack = getattr(`) so it executes on BOTH the eager-pack and fallback arms. Caution: the short anchor line `_fr13_rk_lens[:_fr13_rk_n].copy_(` occurs 2x in the file — use PACKER's full multi-line anchor (ends at `except Exception as _fr13_rk_exc:`), which is unique. |
| E12 | 1d-tsr (bundle-added gap fix) | The TSR zero-accept fallback maps `len==0 → row 0` (verified :11706-11711); under piggyback row 0 is the identity-masked stale root — the bonus-equivalent row is stream 8. Replace the verified anchor block:<br>`_fr13_tsr_leaf = torch.where(\n    _fr13_tsr_len_n > 0,\n    _fr13_tsr_leaf,\n    torch.zeros_like(_fr13_tsr_leaf),\n)`<br>with:<br>`_fr13_tsr_pb_os = __import__("os")`<br>`_fr13_tsr_pb = (_fr13_tsr_pb_os.path.exists("/logs/fr13_piggyback.arm") or _fr13_tsr_pb_os.environ.get("FR13_PIGGYBACK") == "1")`<br>`_fr13_tsr_leaf = torch.where(_fr13_tsr_len_n > 0, _fr13_tsr_leaf, torch.full_like(_fr13_tsr_leaf, 8) if _fr13_tsr_pb else torch.zeros_like(_fr13_tsr_leaf))`<br>(keep original wrapping/indent). Sourced from TOPOLOGY risk 6 — not in any scout's apply list but anchor-verified and required; without it every zero-accept step conditions the drafter on the stale root row. Accepted len>0 rows need no translation (paths already hold extended stream ids 9..17). |

**Group 4 — replay drop (seam 5, committer helper string)** — `CALLER_REPLAY.edits[3],[4],[5],[6],[7]` **verbatim**, in that order (E13 5-import → E14 5-guard → E15 5-sbr-gate → E16 5-ms-gate → E17 5-serial-loop). Anchors `_fr13_sbr_active = (`, `_fr13_ms_on = (`, `for _fr13_prefix in (` verified unique; E14's long anchor uniquely selects the `_fr13_conv_commit_to_col0(` CALL (the bare name occurs 2x: def + call). The conv col-0 commit, accepted-paths/lens refill, and host publishes remain live per CALLER_REPLAY's inventory.

**Group 5 — committer walk (seam 4)** — `COMMITTER.edits[0],[1],[2],[3]` **verbatim** on fr13_device_multidraft_kernel.py (anchors verified: `current_parent = -1` :480, `cur_parent = [-1] * nreq` :638, `_row_fn = ...` unique) = E18–E21a; then `COMMITTER.edits[4]` (host-reference-walk fail-loud in the patcher, anchor `_fr13_dm_counts = counts` verified unique) = E21b. **TOPOLOGY.edits[8] superseded** — its `accepted_row = max(0, walk_root)` variant is NOT applied (accepted_row is diagnostics-only; the functional zero-accept consumer is fixed at E12; remaining len==0 consumers → Risk 6).

**Group 6 — serve wiring** — E23 = `TOPOLOGY.edits[0]` (cat9_pb kind; anchor `cat55221)` verified :219; XFLAGS `FR13_PIGGYBACK=1 FR13_TREE_GDN_GEOM_OVERRIDE=BV=8`; launcher sidecar/forwarding confirmed, no launcher edit). E24 = `TOPOLOGY.edits[1]` (engagement needle; anchor verified :472; greps E5's `info_once` line and should ALSO grep E14's `[FR13_PIGGYBACK] committer GDN replay DROPPED` stderr needle — append a second `docker logs | grep -m1` for it in the same `if [[ "$KIND" == "cat9_pb" ]]` block).

**DROPPED (superseded/dissolved):** PACKER.edits[0] (force-False detect → E1+E2), PACKER.edits[3] original sources (→ E5), CALLER_REPLAY.edits[1],[2] (→ E9), TOPOLOGY.edits[2] (lifecycle widen — E5 no longer reads `_COMMITTED`), TOPOLOGY.edits[6] (placeholder packer → E5), TOPOLOGY.edits[7] (→ E9), TOPOLOGY.edits[8] (→ E18-E21+E12).

Per feedback_commit_push_every_step: commit the bundle flag-gated on main with pathspec commits (`git commit -m .. -- <file>`), one commit per group.

## 2. CONFLICT CHECK

1. **Injected import line (:786 region)** — 3 scouts edit the same unique line (PACKER 1c-import, CALLER_REPLAY 2-import, TOPOLOGY seam-2 rider). Merged into E7. Applying any two sequentially would fail anchor-match (benign) or fork the import.
2. **`launch_tree_gdn_prepared` call site (single occurrence, verified)** — PACKER 1c-mask (head replace, `chain_end_idx=7`) vs CALLER_REPLAY 2-params + TOPOLOGY seam-2 (tail replace, `chain_end_idx=cap()=8`). **Substantive contradiction, not just overlap**: 7-vs-8 is scheme-dependent (§0). Resolved: E9 only. The dropped tail edits' anchors survive E9, so a naive "apply everything" produces duplicate-kwarg TypeError or a mixed scheme — the bundle's single most dangerous mis-apply.
3. **Detect/wide-exclusion region (:11849-11887)** — PACKER.edits[0] and TOPOLOGY.edits[3]+[4] occupy the same region with different mechanisms and names (`_fr13_pb_cat9` vs `_fr10_is_cat9_pb`). Resolved: TOPOLOGY mechanism, unified name, PACKER renamed at E4/E5/E6. PACKER's tree-without-flag raise is retained inside E1; E1 adds the reverse raise PACKER lacked.
4. **cat9 else-stack packer (:12934-12948)** — PACKER.edits[3] (insert-after, concrete) vs TOPOLOGY.edits[6] (elif-before-wide, contains a `<placeholder>` — not apply-ready). Resolved: PACKER structure at E5. **Factual conflict resolved by code read**: PACKER's `_COMMITTED` source is one commit stale (lifecycle at `_prepare_inputs`, patcher:17472; `by_req` carries no tokens :8154-8157; accepted rows are drafts-only :8013) — TOPOLOGY risk 5 confirmed, E5 amended accordingly. TOPOLOGY's engage-log (needed by E24) folded into E5.
5. **Committer walk anchors (:480, :638)** — COMMITTER.edits[2]/[3] vs TOPOLOGY.edits[8] touch identical lines with different `accepted_row` policy. Resolved per Group 5; the disagreement's functional content moved to E12 + Risk 6.
6. **Walk-root helper** — COMMITTER.edits[0] and TOPOLOGY.edits[8] both define `_fr13_pb_walk_root` (different bodies). Only COMMITTER's lands (E18).
7. **Duplicate-anchor hazards inside one file** — `_fr13_rk_lens[:_fr13_rk_n].copy_(` (2x → use PACKER's full block anchor, E11); `_fr13_conv_commit_to_col0(` (2x → E14's long anchor); the three `or _fr10_is_wide` or-lists (:11974 leaf-steps — deliberately NOT edited; :12001 dispatch — E4; :~13410 disengage — E6) are distinguished by trailing context; E2's exclusion-list anchor is unique via its `)\n        )` tail.
8. **Cross-edit contracts (tri-consistency)** — chain length rule `0 | 1+by_req_L` is encoded three times: E5 (tokens), E11 (mask lens), E9 (mask consumption). All three derive from the same `_LUMO_FA_TREE_ACCEPT_BY_REQ` dict state; E5's stash-desync raise makes divergence loud. Walk numbers: committer `cap()-1=7` is CHOICES-rank; kernel `chain_end_idx=7` is SCAN-stream `(0,)^7` — equal by coincidence, different meanings; comments in E9/E18 state both spaces explicitly to prevent a "fix" that breaks one.

## 3. UNRESOLVED RISKS (ranked)

1. **KV double-write / attention side of chain re-processing (BLOCKER for any accept/lossless gate)** — chain tokens' KV already exists at committed positions (FR13_ATTN_KV_REMAP wrote it last commit); re-processing writes it AGAIN at tree-block slots with wrong RoPE positions (base+depth remap, patcher:9544-9547), the auto tree visible-mask lets subtree rows attend chain slots while paged context holds the same keys (double count), repeat-pad slots contribute garbage K/V, and chain row j can attend existing KV of later drafts (future leak). 1c fixes ONLY the GDN recurrent state. Until phase-3 resolves this (variant-B ring-reuse — feed the chain from `_fr13_replay_ring_k/v/a/b` and give chain slots no attention role — or mask+position surgery), cat9_pb output WILL be wrong; only mechanical/CFWD gates are meaningful (§4 V2).
2. **Export-index/ordering scheme** — resolved on paper to Scheme A / `chain_end_idx=7` (§0), but two scouts independently derived 8 under another convention. A wrong pick is silent garble-class (double-applied or missing bonus every step). MUST be settled by the offline validator assert (V0-c) before any GPU run; never hand-edit 7↔8 without re-deriving the scheme.
3. **17-vs-9 drafter mismatch / vacuous wide fallback** — the extended tree satisfies the generic wide predicate (:11868-11875): with any of E1/E2/E4 missing, the wide drafter serves 17 cols via 12 autoregressive MTP forwards with speculation packed into chain slots, and the `EXPECT_RATIO=17` metrics assert passes vacuously. Mitigated by E1's bidirectional raises + E24's needle — but ONLY if Group 1 lands atomically. vLLM never asserts drafter-forward-count vs num_speculative_tokens (tail6 precedent), so nothing else catches it.
4. **Non-tree step interleave + stale req-keyed state (PACKER R3/R4)** — col-0 permanently lags between commit and the next TREE forward; any interleaved non-tree forward (chunked prefill of a tool turn, non-spec decode, APC restore) reads/overwrites stale col-0; afterwards `by_req` + `_FR13_PB_PREV_BONUS` would replay an OLD path. No catch-up replay is in this bundle. Same boundary class as project_fr13_apc_spec_specific_carrier — must be an explicit live-gate scenario (agentic multi-turn), and entries need a step-marker invalidation once the policy is decided.
5. **`next_token_ids` bind in E5** — the propose-scope name was not confirmed in the blob range (no occurrence :11700-13400; it is a propose() parameter upstream). Verify in the live-container eagle source before landing (feedback_read_vllm_source_first); a wrong bind fail-louds (NameError), not silently.
6. **Remaining zero-accept `len==0 → row 0` consumers** — E12 fixes TSR; the conv committer and col-restore twins were flagged (TOPOLOGY risk 6) but not verified/edited. Under piggyback their row-0 fallback reads the identity node. Audit before V2; conv is phase-3 anyway (Risk 8).
7. **Graph-capture / arming contract** — flag is host-read, baked at capture; `PIGGYBACK_EXPORT` is constexpr. Arm the sidecar before launch ONLY; never touch `/logs/fr13_piggyback.arm` on a live serve (mid-run arming double-applies state on already-advanced col-0). Label graph-vs-eager on every gate row. E8/E1 raises execute at capture/eager time only — acceptable because topology is boot-static.
8. **Conv1d recurrent half untouched (PACKER R9)** — conv prior-window/committed-path machinery still relies on the conv col-0 commit; the chain gives conv no advance/identity treatment. Gate conv byte-exact separately (project_fr13_conv_priorwindow_root history: this is the delicate half).
9. **Geometry/cap pinning** — n_pad=32 requires BV=8 (register wall; launch raises without the override — E23 sets it); cap≠8 now raises (E8); stale `/tmp/fr13_piggyback.arm` on the dev host would arm offline CPU gates — gate scripts must clear/assert the sidecar in setup.
10. **Secondary**: depthsync byte gate needs a pb-armed fixture rerun; piggyback+greedy diagnostics unsupported (host-ref walk raises by design, E21b); merged/Arctic/tail modes don't engage for cat9_pb (accept relies on pure MTP cat9 — fine for the mechanism gate); replay staging flags left armed are inert today but a future `flags[0]==1` consumer would misread; pre-existing conv sampler-vs-spec row-order mismatch on mixed batches (CALLER_REPLAY risk 5) will be leaned on harder.

## 4. VALIDATION PLAN

**V0 — offline/CPU (parallel to GPU work; clear `/tmp|/logs/fr13_piggyback.arm` in setup):**
(a) run the prelaunch patcher against live-container sources; assert every sentinel/anchor hit exactly once and the generated modules import.
(b) Triton codegen identity: `piggyback_export=False, chain_end_idx=0` equals the kernel-signature defaults → same specialization key, no recompile (bug-class #10).
(c) **extend `scripts/fr13_native_committer_validate.py`**: on captured fixtures, run the extended-tree scan with the E9 identity mask for L∈{0..5} and assert (i) export@stream-7 == the `launch_tree_gdn_replay` deposit byte-exact (settles Risk 2), (ii) identity-masked rows leave h byte-unchanged (raw_a=raw_b=−1e9 no-op), (iii) `fr13_device_multidraft_commit` on the extended fixture == base-cat9 commit on the stripped subtree modulo the +8 rank shift, and E5's packed chain matches the validator's `[0]+accepted_paths` token sequence.
(d) rerun `scripts/fr13_dm_depthsync_byte_gate.py` with a pb-armed fixture.

**V1 — flag-OFF byte-identity (default path, gate before commit):** locked-pipeline launch of deployed cat9; IN-PROCESS same-boot same-seed gate (no cross-boot byte gate on GB10), temp 0.6 + fixed seed (never greedy), byte-identical decode vs pre-bundle build; /metrics accept + `derived_tps_gpu` unchanged via the canonical fr13_measure module; rows labeled graph+eager; verify zero new sidecars/env armed.

**V2 — cat9_pb ON, MECHANICAL gates only (Risk 1 unresolved → no accept/quality claims):** boot `fr13_bigdenom_swe_serve_variant.sh cat9_pb` (graph and eager rows). Needles (all fail-loud): `FR13_PIGGYBACK cat9_pb drafter engaged` (E24), `[FR13_PIGGYBACK] committer GDN replay DROPPED` (E14), draft_tokens/drafts == 17, container-env grep, no `RuntimeError` from E1/E8/E9/E14 guards, tree_n=18 served. **CFWD collapse**: committer-forward span 99ms → ~16ms expected (48-kernel serial replay ≈72ms + per-layer sync overhead gone) measured with the existing splits + /metrics on matched prefill_frac; report per-committed-token cost (speed-gap = our diff). GPU-mem hygiene between runs; MAX 2 concurrent workflows, GPU serialized.

**V3 — after phase-3 (KV/conv) lands:** live SWE-Verified agentic multi-turn at temp 0.6, nudge-free qwen-code, no AGENT_WALL_S (trace-inactivity watchdog only); depth-matched accept vs native **E5** (cat9→E5; depth axis matched, big-denom ≥13% lossless bar vs no-spec ground truth); per-token argmax probe, not scalar-only, equal A/B sample sizes; garble measured at codex_trace agent_messages (not docker logs); rows labeled graph/eager; then the end-goal cell: tree + EXACT_SEED cache, nudge-free.