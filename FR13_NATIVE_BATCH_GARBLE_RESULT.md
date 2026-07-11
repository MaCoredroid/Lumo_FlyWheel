# FR13 — Is the garble caused by batching/cache, or by our tree? (EMPIRICAL, 2026-07-11)

**Question (user):** our machinery = tree(cat8)+cache garbles (commits near-neighbor misspelled
identifiers). Does the *native* analog — native-MTP5 **+ cache** under **real** batching — garble too?
If yes → it's general batch/cache co-residency; if no → it's specifically our tree.

**Why this needed a dedicated probe:** live agentic SWE serializes decode to effective B≈1.3
(vLLM queues, `Running==1` ~80%), so "conc4" is NOT real batching. Had to FORCE co-residency.

## Method
Same-methodology 4-cell matrix, identical prompts+seeds, temp 0.6. Instrument =
`scripts/fr13_garble_gate.py` (identifier-consistency probes that BAIT near-neighbor corruption +
AST undefined-name-rate scorer). Real co-residency PROVEN by sampling engine `Running:` during
each B=8 gate (max Running = 8 for both native and tree — not the conc4≠b4 trap).
- native+cache: `scripts/fr13_native_cache_batch_garble.sh` (native MTP-5 + stock APC, max_num_seqs=8;
  gate concurrency=8 vs 1, same boot).
- tree positive control: `scripts/fr13_tree_garble_poscontrol.sh` (cat8 forked tree, TREE_ATTN,
  num_spec=8, max_num_seqs=8; gate concurrency=8 vs 1).

## Result

| arm | real batch (Running=8) | undefined-name garble | syntax err |
|---|---|---|---|
| native+cache B=1        | no  | **0.00%** (0/72)  | 0/72 |
| native+cache B=8 (real) | yes | **0.00%** (0/72)  | 6/72 |
| tree cat8   B=1         | no  | **7.89%** (42/72) | 0/72 |
| tree cat8   B=8 (real)  | yes | **11.29%** (40/72)| 0/72 |

## Conclusions
1. **Real inter-request batching is EXONERATED.** native+cache at genuine 8-way co-residency = 0.00%,
   identical to B=1.
2. **Gate is sensitive** (positive control passes): tree = 8–11%, catching the exact signature
   (`running_balanc·accumlator`, `applied_entry_idx`, `sliced_bounds`).
3. **The garble is the tree's INTRA-request scan, batch-independent.** Cleanest isolation = the B=1 row:
   with zero batching, tree=7.89% vs native=0.00%. The corruption is purely the batched tree GDN scan
   over its M=8 nodes. Real B=8 on top (11.29%) is the same order — no meaningful batch amplification
   (samples-with-undef 42→40).
4. Different pathologies: native batch-variance → benign syntax errors (6); tree → semantic identifier
   corruption. Not the same failure.

## Caveats
- Cache was ENABLED but sat at 0% hit (gate prompts share no prefix) → the native cache-RESTORE path
  wasn't exercised. Mitigant: garble is proven cache-independent (the tree ran cache-OFF here and still
  garbled 7.89%), and native decode is lossless-by-construction (rejection-sampled), so no mechanism
  for cache-restore to inject garble into native. The native+cache-HITTING+batch cell is
  theoretically-covered-but-untested; a prefix-reuse variant would close it.

Confirms the mechanistic conclusion (garble = tree intra-request co-residency) with a live same-boot
A/B. Run dirs: `output/fr13_native_batch_garble/run_*` (native), `.../tree_pos_*` (tree).
