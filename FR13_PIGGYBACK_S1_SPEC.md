All evidence gathered. Here is the report.

---

# FR13_PIGGYBACK Phase-3 — S1 resolution + unresolved items 2-4

**Tree state verified first**: the phase-1/2 bundle (E1–E24) is LANDED (seam-1c mask + `piggyback_export`/`chain_end_idx=7` at P:5185-5235, E5 packer + `_FR13_PB_PREV_BONUS` at P:13241-13298, E11 pb-lens scatter at P:10325-10374, E12 TSR row-8 at P:11908-11913, Group-5 walk-root in `scripts/fr13_device_multidraft_kernel.py:97-108` used at :547/:743, pb-lifecycle raises at P:8355-8381). Phase-3 Waves 0–4 are NOT landed (no A1/A2/B5-B7/CONV/S2/C-INT code in the tree). All anchors below are exact and unique in the **current** tree. `P` = `/home/mark/shared/lumoFlyWheel/scripts/fr10_phase4_patch_vllm_tree_gdn.py`, `K` = `/home/mark/shared/lumoFlyWheel/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`.

---

## 1. S1 edit list (dependency-ordered, flag-gated)

Ship order: **S1-K1 → S1-P2 → S1-P3 → S1-P4**, all AFTER Waves 0–4 (S1-P3 composes with A2; S1(a) is only *correct* once A1's RoPE clamp is live — stream 8's K is rotated at base+0 only under A1, which is what licenses the pure slot copy). S1-K1+P2 without P3 is strictly-better bytes (safe intermediate); P3 without K1/P2 would leave the bonus slot holding stale bytes with no live writer — never land P3 first. Flag-off is byte-identical for every edit (`pb_bonus_src=None` default; shape-keyed n==18 detect; env/default-off oracle). Acceptance gate = V2.5 slot-C restored-vs-oracle (RED until this lands, then flips).

### S1-K1 — `launch_attn_kv_linear_remap`: bonus-slot pair + strict pb mode (K)

There is **no Triton kernel** here — the remap is pure advanced-indexing torch (K:451-551); "extend the kernel" = extend this function. It has exactly **one** call site: the patcher inject built by `_patch_gpu_model_runner_attn_kv_remap_apply` (P:18113+, call at P:18244-18252), running post-`_sample` while `slot_mapping` is still this step's verify mapping (the required timing window). The capture half (P:18090-18106) collects **all non-Mamba/GDN builders** = all full-attn layers, so one flag covers "ALL full-attn layers".

**K1.a — signature.** Anchor (K:451-459, unique):
```python
def launch_attn_kv_linear_remap(
    *,
    kv_caches,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    dst_pi: torch.Tensor | None = None,
```
Add after the `dst_pi` line:
```python
    pb_bonus_src: int | None = None,
```
`None` (default) = shipped behavior bit-for-bit.

**K1.b — convert the four silent-skip exits to fail-loud under pb** (feedback_fail_loud_assert_engagement: a skipped bonus copy = stale bonus KV next step = silent garble). The exits:

1. Anchor (K:484-486):
```python
    b = int(num_spec_decodes)
    if b <= 0 or not kv_caches:
        return 0
```
2. Anchor (K:488-490):
```python
    path_cols = int(accepted_paths.shape[1])
    if path_cols <= 0:
        return 0
```
3. Anchor (K:496-497 with the preceding comment line for uniqueness):
```python
    # batch would put a spec row at a higher batch position and qsl[:b] would
    # index a foreign request -> wrong-slot copy: skip (safe no-op) instead.
    if int(query_start_loc.shape[0]) < b + 1:
        return 0
```
4. Anchor (K:525-527):
```python
    _max_off = torch.maximum(ap.max(), torch.maximum(acc.max(), dst_off.max()))
    if not (bool((spans == spans[0]).all()) and bool(spans[0] > _max_off)):
        return 0
```
In each, replace the bare `return 0` with:
```python
        if pb_bonus_src is not None:
            raise RuntimeError(
                "FR13_PIGGYBACK S1(a): bonus-slot KV copy would be skipped "
                "(<branch name>) -- stale bonus KV is silent garble; refusing"
            )
        return 0
```
(one distinct `<branch name>` string per site: `empty-batch/kv`, `no-path-cols`, `qsl-too-short`, `nonuniform-spans-or-span<=max-off`). For exit 4 also extend `_max_off` first so the span guard covers offset 8:
```python
    if pb_bonus_src is not None:
        _max_off = torch.maximum(
            _max_off,
            torch.tensor(int(pb_bonus_src), device=device, dtype=torch.long),
        )
```
(under pb spans[0]==18 > 8, so this only bites on real breakage).

**K1.c — the bonus pair itself.** Anchor (K:528-531, unique via the trailing comments):
```python
    qsl = qsl_full[:b].view(-1, 1)                                     # [b,1]
    foreign = (m_idx < acc) & (ap != dst_off)                          # [b,path]
    if not bool(foreign.any()):
        return 0
```
Insert between the `qsl = ...` line and the `foreign = ...` line:
```python
    if pb_bonus_src is not None:
        # FR13_PIGGYBACK S1(a) (LIVE-8): stream 0's later-layer hiddens lack
        # the GDN bonus update, so its full-attn K/V write at the canonical
        # bonus slot (flat verify offset 0 = linear position base+0) is wrong
        # bytes on EVERY full-attn layer (all sit past L0-GDN). Copy the
        # ACTIVE root twin's per-layer scratch KV (verify offset
        # pb_bonus_src == 8) -> the canonical bonus slot, every commit step
        # INCLUDING zero-accept (this runs before the foreign.any() early
        # return on purpose). Pure slot copy: stream 8's K is already RoPE'd
        # at base+0 via the A1 offset clamp -- no re-rotation. SOURCE
        # auto-threads under FR13_SLOT_REORDER (node k wrote via
        # sm_perm[qsl+k]); DEST is the FLAT slot of offset 0 =
        # sm_perm[pi[0]], and pi[0] == 0 by construction, so dst ==
        # slot_mapping[qsl+0] with or without the perm.
        pb_ns = int(slot_mapping.shape[0])
        pb_src_off = torch.full(
            (b, 1), int(pb_bonus_src), device=device, dtype=torch.long
        )
        pb_dst_off = torch.zeros((b, 1), device=device, dtype=torch.long)
        if dst_pi is not None:
            pb_dst_off = dst_pi.to(device=device, dtype=torch.long)[pb_dst_off]
        pb_src_slot = slot_mapping.reshape(-1)[
            (qsl + pb_src_off).clamp(0, pb_ns - 1).reshape(-1)
        ].to(torch.long)
        pb_dst_slot = slot_mapping.reshape(-1)[
            (qsl + pb_dst_off).clamp(0, pb_ns - 1).reshape(-1)
        ].to(torch.long)
        for kv in kv_caches:
            if not torch.is_tensor(kv) or kv.dim() < 3 or int(kv.shape[0]) != 2:
                continue
            bs = int(kv.shape[2])
            pb_gathered = kv[:, pb_src_slot // bs, pb_src_slot % bs].clone()
            kv[:, pb_dst_slot // bs, pb_dst_slot % bs] = pb_gathered
```
Rationale for placement: it must run **before** the `foreign.any()` early return (zero-accept steps have no foreign accepted rows but still need the bonus copy) and after `qsl`. Src/dst never collide with the accepted-path copies (main dsts = offsets 1..acc ≤ 6; main srcs = subtree rows 9..17; bonus is 8→0), and the pair itself is disjoint (8≠0), so ordering vs. the main gather-then-scatter is a non-issue. **Return value intentionally unchanged** (foreign accepted-row count only) so the existing branching-engagement needle keeps its meaning; bonus-copy engagement is guaranteed by the fail-loud exits instead.

### S1-P2 — patcher remap inject: gate, cols floor, kwarg, armed-guard (P)

**P2.a — armed-but-remap-off fail-loud.** Anchor (P:18130-18133, unique via the sentinel f-string):
```python
    inject = anchor + (
        f"\n        {sentinel}: re-linearize committed-path full-attn KV.\n"
        "        if __import__(\"os\").environ.get(\"FR13_ATTN_KV_REMAP\", \"0\") == \"1\":\n"
        "            try:\n"
```
Insert between the sentinel line and the `if ... == "1"` line:
```python
        "        from lumo_flywheel_serving.fr10_gdn_tree_kernel import (\n"
        "            _fr13_piggyback_on as _fr13_akr_pb_probe,\n"
        "        )\n"
        "        if _fr13_akr_pb_probe() and __import__(\"os\").environ.get(\"FR13_ATTN_KV_REMAP\", \"0\") != \"1\":\n"
        "            raise RuntimeError(\n"
        "                \"FR13_PIGGYBACK requires FR13_ATTN_KV_REMAP=1 \"\n"
        "                \"(S1 bonus slot-C copy rides the commit-time remap)\"\n"
        "            )\n"
```
(The locked launcher exports `FR13_ATTN_KV_REMAP=1` — `scripts/fr13_launch_locked.sh:38` — this guard makes a mis-configured pb boot loud instead of silently stale.)

**P2.b — pb gate + zero-accept cols floor.** Anchor (P:18198-18200, unique):
```python
        "                _fr13_akr_cols = max((len(_r[0]) for _r in _fr13_akr_rows), default=0)\n"
        "                _fr13_akr_paths = None\n"
        "                _fr13_akr_lens = None\n"
```
Replace with:
```python
        "                _fr13_akr_pb = _fr13_akr_pb_probe()\n"
        "                _fr13_akr_cols = max((len(_r[0]) for _r in _fr13_akr_rows), default=0)\n"
        "                if _fr13_akr_pb and _fr13_akr_n > 0 and _fr13_akr_nrows > 0:\n"
        "                    # S1(a): the bonus copy must fire on ALL-zero-accept\n"
        "                    # steps too; force the paths tensor to build (padded\n"
        "                    # zeros, lens 0 -> no foreign copies, bonus pair only).\n"
        "                    _fr13_akr_cols = max(_fr13_akr_cols, 1)\n"
        "                _fr13_akr_paths = None\n"
        "                _fr13_akr_lens = None\n"
```
This matters: without it, an all-zero-accept step leaves `_fr13_akr_cols == 0` → `_fr13_akr_paths is None` → the remap call is skipped entirely and the bonus slot goes stale. The `_fr13_akr_nrows > 0` freshness gate (commit fired this step) is kept — non-tree steps must not copy.

**P2.c — pass the kwarg.** Anchor (P:18244-18252, unique):
```python
        "                            _fr13_akr_tot += _fr13_akr_fn(\n"
        "                                kv_caches=_fr13_akr_kvs,\n"
        "                                slot_mapping=_fr13_akr_sm,\n"
        "                                query_start_loc=_fr13_akr_qsl,\n"
        "                                accepted_paths=_fr13_akr_paths,\n"
        "                                num_accepted_tokens=_fr13_akr_lens,\n"
        "                                num_spec_decodes=_fr13_akr_n,\n"
        "                                dst_pi=_fr13_akr_dstpi,\n"
        "                            )\n"
```
Add after the `dst_pi=` line:
```python
        "                                pb_bonus_src=(8 if _fr13_akr_pb else None),\n"
```
Compose note: Wave-0 **A3** (accepted-rows ≥9 guard, `_fr13_akr_pb_on()`-gated, anchored at `_fr13_akr_rows.append(` P:18168 — still exact) edits this same inject. Disjoint lines, no anchor overlap; if A3's payload defines its own `_fr13_akr_pb_on()`, keep both names or unify at apply — P2 is self-contained either way.

### S1-P3 — A2 delta: row 0 = full attention ghost (P → tree_attn.py)

Anchor (P:14267-14269, unique; this is the `new_return` head A2 also edits — **apply after A2**, placing this block after A2's rules [1][2][3], still before `    try:` — last-writer-wins ordering matters if A2 writes any zeros into row/col 0):
```python
        new_return = f"""    {mask_sentinel}: dump the runtime root/bonus attention bias row.
    try:
        _fr10_mask_path = os.environ.get("FR10_ROOT_HIDDEN_CAPTURE")
```
Insert between the sentinel line and `    try:` (4-space indent inside the f-string; no `{}` braces used, f-string-safe):
```python
    _fr13_s1_pb = (len(sorted_tree_choices) + 1 == 18) and all(
        tuple(sorted_tree_choices[_s1k]) == tuple([0] * (_s1k + 1))
        for _s1k in range(8)
    )
    if _fr13_s1_pb:
        # FR13_PIGGYBACK S1(b) (LIVE-8): stream 0 (the pos-0 bonus copy) is
        # poisoned after the first GDN layer, so every full-attn layer's
        # row-0 K/V is wrong bytes past L0. Full attention ghost: row 0
        # attends the PAGED CONTEXT ONLY (finite -- the tree bias covers only
        # the [18,18] suffix block and context_len >= 1 in tree decode), and
        # NO row reads its K column (the ACTIVE root twin is stream 8; its
        # durable KV lands in the canonical bonus slot via the S1(a)
        # commit-time copy in launch_attn_kv_linear_remap). Convention must
        # be hard -inf: the FA2 fork special-cases bias == -INFINITY as a
        # hard mask; torch.finfo.min would take the += branch instead.
        # Boot-static bias => applies to every step, drafting slices
        # ([1:,1:]) exclude row/col 0, and the FR13_SLOT_REORDER column perm
        # (pi[0] == 0) preserves both writes.
        tree_attn_mask[0, :] = -torch.inf
        tree_attn_mask[:, 0] = -torch.inf
        logger.info(
            "FR13_PIGGYBACK S1(b): row 0 attention-ghosted "
            "(row+col 0 = -inf; live root twin = stream 8)"
        )
```
`sorted_tree_choices`, `tree_attn_mask`, `torch`, `logger` all verified in scope at the live insertion point (container `tree_attn.py:1030-1080`). Resulting visibility graph: live rows {8..17} see exactly {8} ∪ subtree-ancestors ∪ self = base cat9 under the 8↔0 / 9..17↔1..9 isomorphism (subtree ancestor lists include cols 1..8; A2 kills 1..7, this kills 0, leaving 8) — matches the V0(d) fixture spec.

### S1-P4 — row-0 consumer fail-loud: `_fr12_native_spine_conv_out` (P)

Anchor (P:3136-3141, plain template code, unique):
```python
                    _fr12_native_spine_conv_enabled = (
                        _fr12_native_spine_oracle_enabled
                        and os.environ.get("FR12_TREE_CONV_NATIVE_SPINE", "0") == "1"
                    )
                    _fr12_native_spine_conv_out = None
```
Insert after the `_fr12_native_spine_conv_out = None` line:
```python
                    if _fr12_native_spine_conv_enabled and _fr13_piggyback_on():
                        raise RuntimeError(
                            "FR12_TREE_CONV_NATIVE_SPINE oracle assumes the "
                            "base tree (path-0 spine incl. row 0 = live "
                            "root); under FR13_PIGGYBACK row 0 is the "
                            "identity-masked stale root -- disarm the oracle "
                            "for cat9_pb"
                        )
```
`_fr13_piggyback_on` is already in the blob's injected import (P:799). Zero cost when the oracle is off (default).

---

## 2. Row-0 consumer audit under cat9_pb (S1(c))

| # | Consumer | Site | Disposition |
|---|----------|------|-------------|
| 1 | Full-attn K/V write to canonical bonus slot (flat offset 0) | verify forward; remap at P:18113+ / K:451 | **FIXED by S1-K1/P2** (stream 8 → slot C, every commit step incl. zero-accept, all full-attn groups) |
| 2 | In-forward attention reads of tree col 0 (base bias sets `[:,0]=0`, live tree_attn.py:1049-1050) | FA2 fork + qq_bias fallback | **FIXED by S1-P3** (col 0 -inf for all rows; row 0 paged-only) |
| 3 | Row-0 logits as verify/walk root | committer walk | **SAFE — landed**: `_fr13_pb_walk_root()=cap()-1=7` (choices-rank) `scripts/fr13_device_multidraft_kernel.py:97-108`, used :547/:743, + extended-tree validation :130-165 |
| 4 | Drafter sample row, zero-accept (`token_indices_to_sample`) | TSR, P:11899-11921 | **SAFE — landed** (E12: `torch.full_like(_fr13_tsr_leaf, 8)` under pb) |
| 5 | Conv zero-accept commit (`_fr13_conv_commit_to_col0`) | P:7388-7391 | **OPEN — covered by S2** (Wave-3 #15, apply-ready in the phase-3 spec, NOT landed; anchor re-verified exact at current P:7388-7391). Required before any pb quality gate |
| 6 | Conv prior-window read, zero-accept (`gather_committed_path_conv_prior` K:596-607; fused twin `prepare_committed_path_conv_rows` src/lumo_flywheel_serving/fr13_tree_conv_fused.py:304-322) | tree forward P:2626-2689 | **SAFE under the lock**: `FR13_TREE_RUNROW_INIT=1` (mandatory under pb — raise at P:8355-8364) short-circuits both to spec col 0 = the durable running row, not stream-0 scratch. Latent only if RUNROW_INIT=0 (dead branch len==0→node col 0) — do not arm that combination |
| 7 | SSM h0 seed + `PIGGYBACK_EXPORT` dst (col 0) | K:995-1001, K:909-920 | **SAFE by design** — col 0 is the durable running row (export target), not stream 0 |
| 8 | `_fr12_native_spine_conv_out` splice over path-0 nodes (incl. node 0) | P:3140-3182, P:3510-3515 | Default OFF (needs FR12_NATIVE_SPINE_ORACLE=1 **and** FR12_TREE_CONV_NATIVE_SPINE=1); **fail-loud added by S1-P4** |
| 9 | APC SSM/conv leaf publish, zero-accept (`_fr13_publish_apc_ssm_leaf`, ZEROACCEPT arm) | P:7411-7470+ | **OPEN — covered by C-INT-1** (Wave-4 #17, len==0 → `spec_idx[b,8]`, NOT landed) |
| 10 | Replay/catch-up root = ring node 0 | K:1020-1027 (+ native layout K:1148-1151; all-layers twin K:1547-1554) | **FIXED by item-3 spec below** (`root_node=8` threading + all-layers pb guard) |
| 11 | GDN scan row-0 identity mask | P:5200-5206 | **SAFE — deliberate** (E9 law: stream 0 identity, active bonus at stream 8) |
| 12 | Durable-AB / boundary replay taps ("root node 0 + accepted path", P:7650-7666) | observe-only | **SAFE — landed fail-loud**: pb + taps raises at P:8365-8372 |
| 13 | Diagnostics reading path-0/row-0 (`_fr10_path0_x` conv diag P:3222-3224/3723-3736, `_fr13_conv_subop_mab` P:3517-3531, `root_row` bias capture, branch-accept hist) | FR10_METRICS / capture flags | **Inert to serving**; pb-armed diagnostic runs will see ghost-row values — do not interpret path-0/row-0 taps as "root" under cat9_pb |
| 14 | `build_for_drafting` bias slices | live tree_attn.py:917-937 | Slices start at `[1:, 1:]` → **S1(b)'s row/col-0 mutation does NOT leak**; chain-ghost rows 1..7 DO sit inside the slice = pre-existing unresolved item 5 (unchanged by S1) |
| 15 | `hybrid_reorder_decode` (FR13_FA2_SPINE_REORDER) | live tree_attn.py:1274-1297 | Env absent in the lock → off. **Keep off under pb** until the all--inf-suffix-row-0 merge path is verified (risk 10) |

---

## 3. Item 2 — live-container tree_attn.py bias facts (A2 pre-req)

Read from `fr13-bigdenom-tail6_async_lad2:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/tree_attn.py` (path exactly as the phase-3 spec guessed).

| Fact | Value | Cite (live file) |
|---|---|---|
| Bias dtype | **fp32** (`dtype=torch.float32` at the builder call; `torch.full(..., dtype=dtype)`) | :844-849, :1037-1040 |
| Mask convention | **true `-torch.inf`**, visible cells `0.0`; NOT finfo.min | :1038-1040, :1044-1050 |
| Consumer semantics | FA2 fork `apply_tree_bias` special-cases `bias == -INFINITY` → hard mask; any finite bias goes through `score += bias/scale`. Bias applies to **suffix (tree) columns only** (`k_rel = col - context_len - k_offset`, negative for context cols → untouched) → an all--inf tree row still attends paged context, finite output | `scripts/fr13_patch_fa2_tree_bias.py:41-71` |
| Build cadence | **Built ONCE** at `TreeAttentionMetadataBuilder.__init__`; every step's metadata carries the same tensor by reference; cudagraph-capture build reuses it. No per-step rebuild | :843-851, :907-909, :853-858 |
| Per-step consumption | `decode_meta.tree_attn_bias` → optional `_fr13_sr_bias_perm` (col perm, cached by `(id, data_ptr)`, `pi[0]==0` asserted) → `flash_attn_varlen_func(tree_bias=...)`; qq_bias fallback passes the **unpermuted** tensor with `causal=True` | :1253-1265, :1299-1338, :975-992 |
| A2 edit points vs reality | The patcher's sort+SPEC_CONFIG override and the `new_return` capture inject exist **verbatim** in the live file, with `sorted_tree_choices`/`depth_counts`/`tree_attn_mask` in scope at the insertion point | :829-841, :1066-1080 ↔ P:14244-14256, P:14267-14283 |

**VERDICT: PASS** for A2's assumptions, with two binding requirements: (1) ghost writes MUST use exact `-torch.inf` (finfo.min takes the `+=` branch — not a hard mask); (2) mutation MUST happen at construction time (inside `_prepare_tree_attn_bias`), never post-boot — the SR perm cache is keyed on the tensor identity and would go stale on a later mutation. Caveats already in the risk ledger and unchanged: drafting slices `[1:, 1:]` share the mutated tensor (item 5), and `hybrid_reorder_decode` is unverified for fully-ghosted rows (off in the lock).

---

## 4. Item 3 — `root_node` extension anchors (C-INT-2′)

The two replay kernels' root blocks are **textually identical twins** (K:1013-1029 vs K:1540-1556) — naive short anchors are non-unique; the anchors below are disambiguated and verified unique.

**(a) Kernel signature** — anchor (K:949-958; unique via `RING_B_STRIDE_AB`+`RING_N_STRIDE_AB` directly followed by `OUTPUT_SCALE` — the all-layers twin interposes `SPEC_L_STRIDE/PREV_L_STRIDE/GATE_L_STRIDE`):
```python
    RING_B_STRIDE_AB: tl.constexpr,
    RING_N_STRIDE_AB: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
    SCAN_ALIGN: tl.constexpr = False,
    RUNROW_COMMIT: tl.constexpr = False,
    RUNROW_INIT: tl.constexpr = False,
    BURN_NODE_BANK: tl.constexpr = False,
):
```
→ add `    ROOT_NODE: tl.constexpr = 0,` before `):`.

**(b) Kernel root branch** — anchor (K:1001 through K:1027; the K:1001 line is unique — the twin reads `spec_layer`, not `spec_state_indices`):
```python
    h0_row = tl.load(spec_state_indices + pid_b * SPEC_COLS + h0_col).to(tl.int64)
```
…extend the anchor through:
```python
            active = acc_len >= 0
            node = 0
```
→ replace the final `            node = 0` with `            node = ROOT_NODE` (and amend the root comment: "gdn node ROOT_NODE (stock 0; FR13_PIGGYBACK catch-up passes 8 = the LIVE-8 bonus twin's ring row — ring row 0 is the diverged pos-0 copy)").

**(c) Launcher signature** — anchor (K:1232-1236; unique via the docstring line):
```python
    runrow_commit: bool = False,
    runrow_init: bool = False,
    burn_node_bank: bool = False,
) -> None:
    """Launch the FR13 accepted-path replay (the durable-state publish).
```
→ add `    root_node: int = 0,` after `burn_node_bank`.

**(d) Launcher validation** — anchor (K:1272-1275):
```python
    if n_pad > 32 or n_pad & (n_pad - 1):
        raise ValueError(f"ring n_pad must be a power of two <=32, got {n_pad}")
    if ring_dim_k != dim_k:
        raise ValueError(f"ring k dim {ring_dim_k} != bank dim_k {dim_k}")
```
→ insert between the two ifs: `if not (0 <= int(root_node) < n_pad): raise ValueError(f"root_node {root_node} outside ring n_pad {n_pad}")`.

**(e) Launcher → Triton call** — anchor (K:1364-1374; unique via `a_ring.stride(1)` — twin uses `a_rings.stride(2)`):
```python
        RING_B_STRIDE_AB=a_ring.stride(0),
        RING_N_STRIDE_AB=a_ring.stride(1),
        OUTPUT_SCALE=output_scale,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        RAW_GATING=True,
        SCAN_ALIGN=scan_align_on(),
        RUNROW_COMMIT=runrow_commit,
        RUNROW_INIT=runrow_init,
        BURN_NODE_BANK=burn_node_bank,
        num_warps=8,
    )
```
→ add `        ROOT_NODE=int(root_node),` after `BURN_NODE_BANK=...`.

**(f) Native-committer route MUST be threaded too** — `launch_tree_gdn_replay` reroutes to `_fr13_native_committer_replay` when `_fr13_committer_native_on() and runrow_init` (K:1325), and its layout hard-codes root node 0 at **K:1148-1151** (this is the second read-ref from unresolved item 3):
- K:1329-1337 call (anchor unique via `state_bank=state_bank,` — the all-layers twin passes `banks_list[_L]`): add `root_node=root_node,`.
- `_fr13_native_committer_replay` signature (anchor K:1167-1172, unique): add `root_node=0,` to the keyword list, and thread into its `layout is None` call (anchor K:1193-1197): `root_node=root_node,`.
- `_fr13_prepare_committer_layout` signature (anchor K:1137 one-liner, unique): add `root_node=0`; and replace in the anchor (K:1148-1151, unique):
```python
        nodes = torch.cat([
            torch.zeros(1, dtype=torch.long, device=dev),
            accepted_paths[b, :L].to(torch.long),
        ])
```
→ `torch.zeros(1, ...)` becomes `torch.full((1,), int(root_node), dtype=torch.long, device=dev)`.
Note the callers that pass a **precomputed** `layout` (all-layers ship path) stay stock root-0 — correct, see (h).

**(g) `piggyback_catchup_replay` helper insertion anchor** — after the wrapper rebind (anchor K:1412-1416, unique via the rebind line):
```python
        return _r

    return _w


launch_tree_gdn_replay = _fr13_replay_gpu_timed(launch_tree_gdn_replay)
```
→ insert the Scout-C helper after this line, calling `launch_tree_gdn_replay(..., root_node=8)`.

**(h) Defensive twin guard** — the all-layers replay (root twin at K:1547-1554, launcher call K:1883) is left stock: under pb the committer replay is DROPPED (E14-E17), so it never runs pb-armed. Make that loud — anchor (K:1859-1862, `if _fr13_committer_native_on() and runrow_init and banks_list is not None:` line with the two preceding raises, unique): insert before it:
```python
    if _fr13_piggyback_on():
        raise RuntimeError(
            "FR13_PIGGYBACK: all-layers committer replay must be DROPPED "
            "under piggyback (it replays ring node 0 = the diverged pos-0 "
            "row); catch-up uses launch_tree_gdn_replay(root_node=8)"
        )
```

Codegen note: adding the `ROOT_NODE` constexpr changes the JIT source hash → one boot-time recompile; `ROOT_NODE=0` constexpr-folds to the exact stock `node = 0`, but per bug-class #10 discipline the V0(b) codegen-identity byte A/B must be re-run before any live boot.

---

## 5. Item 4 — `resumed_req_ids` at the REQKEY hook: CONFIRMED

- The REQKEY inject (P anchor at :10141-10145) lands inside `def _prepare_inputs(self, scheduler_output: "SchedulerOutput", num_scheduled_tokens)` — live container `gpu_model_runner.py:1937-1940`, sentinel at :2517. **`scheduler_output` is in scope** at the C-INT-2′ insertion point (after P:10196-10198, before the P:10224 spec publish — both re-verified exact in the current tree, as are the eviction anchor P:10173-10176 and B2/B3's insertion points P:10224/P:10368-10374).
- The exact attribute path **`scheduler_output.scheduled_cached_reqs.resumed_req_ids` is valid**: `CachedRequestData` declares `resumed_req_ids: set[str]` (live `vllm/v1/core/sched/output.py:110-115`, default `set()` at :169), and gpu_model_runner itself uses the identical pattern at live :1103. The P:11485 mamba-utils precedent is the same path. It is a `set[str]` → direct `in` membership on req-id strings, matching the REQKEY code's `str(...)` keying.
- One property worth encoding in C-INT-2′: the whole REQKEY block sits in a `try/except` that **re-raises** unless `FR10_ALLOW_LINEAR_FALLBACK=1` (P:10375-10382) — catch-up failures inherit fail-loud by default; do not add an inner swallow.

**C-INT edit anchor: CONFIRMED as written — no correction needed.**

---

## 6. NEW risks discovered (beyond the §3 ledger)

1. **All-zero-accept remap skip (closed by P2.b)** — pre-existing gap not in any scout: with every row at acc==0, `_fr13_akr_cols==0` skips the entire remap call; under pb that is a guaranteed stale bonus slot. P2.b + K1.b close it; the V0(d) catch-up/zero-accept fixtures should include an all-zero-accept step.
2. **Engagement-needle semantics shift** — under pb the remap now performs work on every commit step, but the return/needle still counts only foreign accepted rows (deliberate). Any gate that reads `_fr13_akr_foreign_seen` as "remap alive" must not be reused as "bonus copy alive"; the bonus copy's liveness proof is the K1.b raises plus the V2.5 slot-C gate.
3. **A2-payload composition ordering** — S1-P3 assumes A2 inserts strictly between the sentinel line and `try:` in the `new_return` head; if A2's rule [3] writes any `0.0` into row/col 0 (e.g., a row-8 rewrite touching cell [8,0]), S1-P3 must be textually AFTER it. Verify at apply; the V0(d) isomorphism fixture catches any mis-order.
4. **Row-0 capture semantics change** — `FR10_ROOT_HIDDEN_CAPTURE`'s `root_row` dump will record the all--inf ghost row under pb (the dump runs after the mutation). Offline tools that treat `root_row` as "the live root's visibility" must switch to row 8 under cat9_pb (cosmetic, but a bug-class-9 trap for gate scripts).
5. **`_fr13_sr_bias_perm` cache + mutation timing** — safe today only because S1-P3 mutates at construction; flagging explicitly so nobody "optimizes" the ghosting into a runtime hook (the `(id, data_ptr)` cache would serve the stale unghosted perm).
6. **No contradiction with LIVE-8 found** — specifically re-verified: all Qwen3-Next full-attn layers sit after L0-GDN (so "ALL full-attn layers" in S1(a) is complete); `pi[0]==0` is asserted at boot (live :966) so the dst formulation `slot-map[qsl+0]` vs `dst_pi[0]` are identical; conv-bank vs bias-row index "8" coincide because (0,)^8 is sorted choice index 7 (+1 in both spaces) — the two "8"s in S2/TSR/C-INT-1 and S1 are consistent, not a lucky collision (documented so nobody "fixes" one).

Everything above stays within the declared LIVE-8 design; no scout conclusion was overturned.
