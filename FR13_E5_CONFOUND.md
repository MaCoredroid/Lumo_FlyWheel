# FR13 — the "E5 / spine ≠ native-MTP" naming confound
### READ THIS before trusting ANY tree-vs-"native" drift / lossless / amplification conclusion.

## The confound in one line
In FR13 drift/lossless docs, **"E5", "native-E5", "spine5", "chain5"** have been used loosely as "the native baseline" — but they are **NOT** stock native + MTP-5. **`E5` = `chain5` = a FORKED, TREE_ATTN linear spine.** Any conclusion of the form *"the tree's X is shared with native"* or *"native does Y"* is **suspect** until you check WHICH baseline it used. (User caught this 2026-07-08; confirmed from code.)

## What each label ACTUALLY is (from `scripts/fr13_bigdenom_swe_serve_variant.sh` KIND table)
| label used in docs | KIND | launcher | attention | decode | tree? | = true native? |
|---|---|---|---|---|---|---|
| stock native+MTP5 | `nativemtp5` | **native** (stock .so) | FLASH_ATTN | naive_mtp | no | **YES — the only fully-stock baseline** |
| forked native+MTP5 | `nativemtp5_exseed`, `flash_ns5_nocache` | **forked** | FLASH_ATTN | naive_mtp | no | native *decode*, forked kernel/cache |
| "E5" / "spine5" / "chain5" | `chain5` | **forked** | **TREE_ATTN** | tree (linear spine, M=1) | **YES (linear)** | **NO — a forked TREE_ATTN spine** |
| the tree | `cat6root`, `cat8`, `cat9` | forked | TREE_ATTN | tree (branched, M>1) | yes | the tree |

Evidence: `fr13_apc_3way_gate.sh:6` "chain5 = ... the **E5/spine baseline**"; `chain5) LAUNCHER=forked; TREEARG=$CHAIN5_TREE` (serve-variant L140). `NAT_SPEC='{...qwen3_5_mtp, num_speculative_tokens:5}'` in the finalizer just sets the MTP-5 **spec**, not a stock launcher — the `nativeE5_*` arms were served forked-with-TREE_ATTN too.

## Which conclusions this CONFOUNDS
- **"the ~1.166×/layer amplification is SHARED with native/E5" (spine-drift study `wsvy4vn5k`)** — measured `cat9` (TREE_ATTN tree) vs `chain5`≡E5 (TREE_ATTN spine). BOTH are TREE_ATTN, so of course they share the amplification. **It does NOT establish the amplification is shared with native-MTP-decode (FLASH_ATTN).** ⇒ the derived ranking *"amplification-reduction is only a secondary margin-buffer"* is **UNSUPPORTED** for our tree-vs-native garble question.
- Any "spine5/E5 behaves like native" claim: `chain5` shares the TREE_ATTN kernel + co-residency machinery with the tree, so it is **not** a clean native control — only a "tree without branches (M=1)" control (which is still useful for isolating *branches/M*, just not for isolating *TREE_ATTN vs FLASH_ATTN*).

## What IS clean (the ground truth we now have)
- The **garble differential** (tree garbles, "native" doesn't) is `cat6`/`cat8` (TREE_ATTN tree) vs `flash_ns5_nocache`/`nativemtp5_exseed` (FLASH_ATTN naive_mtp) — **both forked**. So it cleanly isolates **tree-decode vs native-MTP-decode** (NOT launcher). That comparison is VALID.
- The truly-stock baseline (`nativemtp5`, LAUNCHER=native) exists but was **not run** in this B=4 matrix (we chose `flash_ns5_nocache` to keep the `s_per_fwd_gpu` timer). If a "vs fully-stock" number is ever needed, run `nativemtp5`.

## THE RULE going forward
When you write **"tree vs native,"** always name WHICH native:
- **(a) stock** `nativemtp5` (LAUNCHER=native), or
- **(b) forked native-MTP-decode** `flash_ns5_nocache` / `nativemtp5_exseed` (FLASH_ATTN naive_mtp), or
- **(c) chain5 / spine5 / "E5"** — a **TREE_ATTN spine, NOT native.**

They have different numerics. The clean **tree-vs-native-MTP-decode** per-layer drift comparison (**G0** in `FR13_TREE_GARBLE_GATE_AND_FIX.md`) has **never** been run against baseline (b); do it before ranking the garble-fix levers — do not inherit the E5=chain5 study's ranking.
