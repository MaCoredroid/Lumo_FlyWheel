# FR13 — "E5" is an OVERLOADED label (naming hazard). Always name WHICH native.
### Read before trusting any tree-vs-"native"/"E5" drift, lossless, or amplification claim.

## The hazard in one line
**"E5" means different things in different FR13 docs.** In some it is `chain5` (a FORKED TREE_ATTN linear spine — NOT native); in others it is a genuine **native-MTP-5 capture**. So "shared with E5" / "E5 = the bar" is only meaningful once you check which construction that doc used. (User flagged 2026-07-08; I first *over*-corrected — see below.)

## What "E5" resolves to, per context (verified from code + captures)
| where | "E5" actually is | KIND / capture | launcher | attention | = native-MTP-decode? |
|---|---|---|---|---|---|
| APC 3-way gate (`fr13_apc_3way_gate.sh:6`) | **chain5** | `chain5` | forked | **TREE_ATTN** | **NO** — forked TREE_ATTN spine |
| **spine-drift study** (`FR13_E5_VS_CAT9_SPINE_DRIFT.md`, `wsvy4vn5k`) | **native MTP-5 capture** | `output/fr10_native_mtp5_same8_*` (num_spec=5) | (unconfirmed from json) | (unconfirmed) | **likely YES** (labeled "E5 native MTP-5 capture") |
| lossless gate bar (`fr13_b1_lossless_prescore.sh`, most docs) | the **depth-5 within-floor bar** | `nativeE5_*` (MTP-5 spec) | — | — | a valid *reference frame* (legit use) |

Our current matrix arms (all forked): `nativemtp5_exseed`/`flash_ns5_nocache` = native-MTP-decode (FLASH_ATTN, naive_mtp); `cat6`/`cat8` = tree (TREE_ATTN). The truly-stock baseline is `nativemtp5` (LAUNCHER=native), not run in this matrix.

## Is the amplification / M-dependence conclusion confounded? — NO (probably), with a caveat
**I initially claimed it was — that was an overstatement.** The spine-drift study that concluded *"the ~1.166×/layer amplification is shared with native; the differential is co-residency/branches"* actually used a **native-MTP-5 capture** as E5 (not chain5), and its ladder is native-referenced:
`native(E5) ≈ 3 flips  <  chain5 (tree kernels, NO branches) ≈ 5  <  cat9 (branches) ≈ 17`.
So the **branches (co-residency)** are the differential (+~12–14), the tree KERNELS add only ~2 over native, and amplification is genuinely shared. That conclusion **stands as the leading hypothesis** ⇒ the garble-fix ranking **M-invariance / batch-invariant PRIMARY, amplification-reduction SECONDARY** is *plausible and native-referenced*, not confounded.

**The only open caveat:** the fr10 capture's exact launcher/attention isn't confirmed from its json, and it predates our current clean arms. So **re-confirm with G0** (per-layer drift `tree` vs our known-config `flash_ns5_nocache`, in `FR13_TREE_GARBLE_GATE_AND_FIX.md`) before betting the fix on it — cheap insurance, not a debunk.

## THE RULE going forward
When you write **"tree vs native" / "E5"**, name WHICH: (a) stock `nativemtp5` (LAUNCHER=native), (b) forked native-MTP-decode `flash_ns5_nocache`/`nativemtp5_exseed`, (c) chain5/spine5 (TREE_ATTN spine — NOT native), or (d) the depth-5 lossless *bar* (a reference frame). Don't let "E5" silently stand in for "native" — check the construction. G0 removes the ambiguity by measuring against a baseline whose config we fully control.
