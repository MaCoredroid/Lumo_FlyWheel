# FR13 Cache-Fix Catalog (KT) — which fix applies to native-MTP+cache vs tree+cache, by flag

**Purpose (2026-07-05):** knowledge-transfer catalog of every prefix-cache-related fix in the FR13 stack,
categorized by which serving config it mechanically engages on and which config was observably defective
without it. Companion to `FR13_TREECACHE_CAMPAIGN_20260704.md` (§ refs below).

**The organizing pattern:** allocator/prefill/restore-layer fixes are **universal** (native inherits them
for free, mostly as latent-risk insurance — native never showed route-level damage). Everything **tree-only**
lives in *leaf/row selection among co-resident states* — a concept native's single spine doesn't have.

| Flag / fix | Layer | Native+cache | Tree+cache | Status | Ref |
|---|---|---|---|---|---|
| `FR13_APC_BLOCK_ALIGN_45477` + `max_num_batched_tokens=block_size` | align chunked-prefill overshoot guard (prefill writing past a block boundary poisoned the next pool block) | **✓ fixes it** | **✓ fixes it** | BAKED (`d228c76b`) | §9-§10 |
| `FR13_APC_EXACT_SEED` | cache capture/restore, chunked-lineage (replaces the restart-fold gross-corruption carrier) | **✓ engages** (the `native_exseed` arm runs it; native's realization mismatch was ε-level) | **✓ engages** (tree's was gross) | ON for cache arms | §16 |
| **E5 `FR13_APC_ZERO_MAMBA_ON_ALLOC`** | allocator invariant: zero freshly-(re)allocated mamba/GDN conv+ssm pool rows (vLLM stock zeroes only full-attention blocks) | **✓ applies** (hooks `MambaManager.allocate_new_blocks` itself — every mamba boot incl. none-mode) | **✓ applies**; tree was the observably bitten config (spec node-bank / conv prior-window residue reads) | default-OFF, SHIP-verified; rp3: kills the accumulating carrier (P1 spread 1.31→0.0019 nats, garble gone); bake pending gates | §48-§50 |
| `FR13_APC_SNAP_FIX_ZEROACCEPT` | zero-accept (accepted_len==0) snapshot publishes committed-root row — the "shared spine+tree carrier" | **✓** (all-drafts-rejected happens in native MTP too) | **✓** | BAKED 06-27 | launcher :283 |
| `FR13_APC_CONV_FIX` / `FR13_APC_CONV_SNAPSHOT` | base conv snapshot machinery for the cache path | **✓ engages** | **✓ engages** | BAKED defaults | launcher :280-281 |
| `FR13_APC_SNAP_FIX` | SSM snapshot source = **committed leaf** row (not base col-0) | engages but ~no-op (one row — no leaf to mispick) | **✓ fixes tree defect** (wrong-leaf among co-resident rows) | BAKED 06-24 (FAITHFUL 240/240) | §16, launcher :282 |
| `FR13_APC_CONV_SNAP_FIX` | conv twin of SNAP_FIX (conv window from accepted leaf) | benign no-op (single row) | **✓** (temp-0.6 TV gate proved the pre-fix conv restore lossy) | BAKED 07-03 | launcher :284 |
| `FR13_APC_CONV_LEAF_COMPLETE` | complete conv leaf-map publish (every snapshot boundary incl. zero-accept/commit gaps) | **auto-no-op on native** (no tree leaf map) | **✓** (killed the conv wrong-row give-up class: 2-6 → 16+ turn engagement) | BAKED 07-04 | §17-§19 |
| Tree committer stack: `FR13_TREE_PER_REQ_GEN/REQKEY/REMAP_SEQ/BONUS_SELF`, `FR13_CONV_COMMITTED_PATH`, `FR13_FA2_TREE_BIAS`, `FR13_FA2_PREFILL_NATIVE`, scheduler-state-sync, GDN-parent-depth | spec-tree decode path (class 1/2/3 committer fixes + FA2 fork + B4 crash fixes) | **✗** (native arms explicitly set these 0) | **✓** | always-on for tree | §fr9/fr13 binds |
| `FR13_APC_HIT_RECURRENT_SUFFIX` (+ `HIT_SUFFIX_CAP`) | hit-restore realization (roll nearest checkpoint forward through the recurrent kernel; SGLang-style) | ✓ when on | ✓ when on (17/48→5/48 mismatched layers) | OFF — superseded by EXACT_SEED; **slated for DELETION** (user 07-05) | §14-§16 |
| R1 context fix (`QWEN_CODE_MAX_OUTPUT_TOKENS` → 75k hard limit), `GPU_UTIL` 0.82→0.78 fence, Responses-API tool_choice parser hook | harness/serving | **✓** | **✓** | SHIPPED | §21, §2 |
| Residual fix (in flight: null-row write-once vs stale carry index at the unguarded prefill SSM carry `gdn_linear_attn.py:984/:1004`) | align chunked-prefill recurrent carry | mechanically likely shared; exposure may require the num_spec offsets (native num_spec=0 spared behaviorally) | **✓ target** (the deterministic `The`@−0.0111 cold fixed point) | E4CAP trace localization in flight | §51 |

## Off-by-policy / dead (DELETE after tree+cache behavior gate passes — user 2026-07-05; git preserves)

REFOLD stack (`FR13_APC_BLOCK_REFOLD`, `FR13_APC_REFOLD_TO_SNAPSHOT`, v1-v4 fold/consume-hop — never consumed,
§36), RECOMPUTE (`FR13_SCAN_ALIGN=recompute`, `FR13_RECOMPUTE_NODE_PARALLEL`, §10), dead flags
(`FR13_APC_FIXED_BUFFER` no-consumer, `FR13_APC_REQUIRE_SHADOW` never-passable, `FR13_APC_PRE_SNAP_FIX`),
slot-pin (`FR13_TREE_GDN_SLOT_PIN` — proven numerical no-op §42), HRS (above; verify no EXACT_SEED dependency
first), `FR13_DECODE_GDN_CAPTURE` (broken instrument, never fired ×2) if still dead post-fix.

KEEP: conv stack, EXACT_SEED/align, BLOCK_ALIGN_45477, tree committer fixes, E5 (+ residual fix),
`FR13_FORK_MARGIN_DUMP` (active instrument).

## Root-cause summary for readers landing here cold

The tree+cache give-up campaign resolved into **write-side** and **read-side** defect families:
1. **Write-side (fixed by the SNAP_FIX/conv/leaf stack):** what the tree committer *stored* into the mamba
   cache picked the wrong co-resident row/realization. Native immune (one row).
2. **Read-side (found 2026-07-05, §47-§51):** recycled mamba pool rows are **never zeroed** by stock vLLM
   (only full-attn blocks are) — request N read request N-1's residue on the cold path. Order-keyed, not
   seed-keyed; every prior "bit-exact" instrument measured request #1 of a boot (vacuous). E5 fixes the
   recycled-row family; one deterministic write-once residual (null-row/carry-index suspect) remains, in
   localization. The launcher now auto-forwards all `FR<N>_*/LUMO_*/VLLM_*` env into the container
   (whitelist trap closed).
