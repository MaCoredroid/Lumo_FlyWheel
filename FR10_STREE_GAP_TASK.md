# FR10 — STree-gap task (fresh session, focused)

You are the worker. Claude (monitor/red-team) runs a /loop 10m and will steer + verify. Read
this brief + `FR10_STATUS.md` (full history) before acting. **Do NOT re-litigate the 7
contamination layers from the prior session — start from the STree-gap diagnosis below.**

## The situation (decisive, research-grounded)
FR10 = lossless GDN/STree token-tree verifier for Qwen3.6-27B (hybrid: 48 GDN linear_attn +
16 full_attn). The STree **scan** kernel is BUILT and **L1-proven lossless** (packed ==
serial-per-path, 7.45e-9 output / 7.63e-6 state). **DO NOT TOUCH the scan kernel.**

But in serving, tree_mtp **path0 ACCEPTANCE degrades**: tree_mtp-only path0 survival cliffs
at depth2 `{0:220,1:36,2:1}` / **0.39 accept/event** vs native gradual
`{0:453,1:362,2:300,3:230,4:178}` / **~2.6/event**. This was on a tree_mtp-ONLY server
(no naive, own captured graph) so it is NOT the naive/CUDA-graph-specialization confound —
it is genuine tree_mtp path0 contamination.

## THE KEY INSIGHT (STree paper arXiv:2505.14969, researched 2026-06-04) — why NO COPYING is needed
STree's `A_tree := L·A_log` accumulates state-transition matrices per tree path via the tree
mask L, and **is lossless for path0 without copying** ("when L is a lower-triangular causal
mask, our method == Mamba2 with non-zero initial state"). That is exactly our L1-proven scan
kernel. **The scan is correct.** BUT the paper is explicit about TWO gaps it does NOT cover:
1. **"The paper does NOT explicitly address causal_conv1d"** — STree covers only the SSM
   scan, NOT the short causal convolution (kernel=4) in the GDN layer.
2. **"STree addresses only the verifier"** — the DRAFTER uses standard AR generation; STree
   does not modify how the draft tree is produced.

**So path0 degrades from the conv and/or the drafter — the two things STree doesn't cover —
NOT the scan. Both can be closed ancestry-correctly WITHOUT copying recurrent state.**

**IMPORTANT — the two gaps differ in NATURE (prioritize the conv):**
- **DRAFTER gap = OUTSIDE the verifier, ACCEPTANCE-ONLY.** Spec-decode is lossless for ANY
  draft; a bad path0 deep draft (drafter sharing recurrent state across the tree) is just
  REJECTED → low acceptance/cliff (lossless, only slow). The prior session's overlay
  `f727928a` already gives path0 a clean SEPARATE native draft — so if that overlay actually
  drives path0, the drafter gap is already closed.
- **CONV gap = INSIDE the verifier (target GDN layer, before the scan), ACCEPTANCE + CORRECTNESS.**
  STree's losslessness assumes the scan gets CORRECT inputs. The conv (uncovered by STree) over
  flattened tree order feeds CONTAMINATED inputs to the correct scan → verifier computes WRONG
  path0 logits → rejects correct drafts AND risks mis-commit. This is a hole IN the verifier.
- **=> Because the overlay already cleans path0's draft yet the cliff PERSISTS, the remaining
  cliff is most likely the CONV gap (verify-side). Diagnose/fix the conv FIRST.**

## STEP 1 — localize: conv-gap vs drafter-gap (tree_mtp-ONLY, no full SWE)
On a tree_mtp-only server (FR10_DECODE_MODE_DEFAULT=tree_mtp, B4/temp0.6/top_p0.95,
FR10_METRICS=1), capture for path0 on identical context:
- **path0 DRAFT tokens** vs the native-spine (mtp5) draft tokens at each depth.
- **path0 VERIFY acceptance** of those tokens.

Decision:
- path0 draft tokens **CORRUPTED** (≠ native spine at depth ≥2) → **DRAFTER gap** (MTP drafter
  shares recurrent state across the tree → corrupts path0 deep draft). Note: the prior
  session's overlay `f727928a` (separate clean spine draft → overlay onto path0) was supposed
  to fix this — verify whether the overlaid path0 tokens actually reach the verifier as path0,
  or are being re-corrupted/ignored.
- path0 draft tokens **CORRECT but REJECTED** → **VERIFY-side**: the causal conv feeds wrong
  (tree-contaminated) inputs into the correct scan kernel → **CONV gap**. We have a tree-aware
  conv (`1f5fa025`) + parity gates (tree==serial, linear==stock) that PASS in isolation — so
  if this is the cause, the SERVING integration is wrong: confirm the tree-aware conv is
  actually applied to path0 inside the captured tree_mtp graph and its output feeds the scan.

## STEP 2 — close the identified gap, NO COPYING
- **Conv gap:** each node's conv window (current + 3 prev) AND carried conv_state must come
  from its 3 root-path ANCESTORS, using the SAME tree mask L as the scan (ancestry, not
  flattened order). Keep the existing parity gates green; add/confirm the SERVING path uses it.
- **Drafter gap:** path0 deep draft must come from the PURE causal (path0) recurrent state.
  Preferred: a STree-style accumulated-transition drafter (A_tree on the draft pass, no copy)
  so the whole tree drafts in one pass with per-path-correct states. Acceptable interim: make
  the overlay (separate native path0 draft) actually drive path0 in the verifier.

## GATE (the exit criterion) — superset Tier1
DONE only when **tree_mtp path0 survival == native gradual** (~`{1:362,2:300,3:230}`, ~2.6/ev,
no depth-2 cliff) on the tree_mtp-only server, AND:
- L1 scan kernel still 7.45e-9 (you must not regress it),
- tree-aware conv parity gates still green (tree==serial AND linear==stock byte-identical),
- the lossless RULE gates still pass (canonical multidraft + deterministic one-hot).
Then re-measure branch recovery + tree_mtp decode-TPS vs recorded E5 (39.9 / 8-16). Recorded
E5 is the baseline — do NOT chase naive_mtp on the FR10 server (CUDA-graph specialization
makes it replay the tree graph; it's a confound we don't need).

## CONSTRAINTS
- B=4, temp=0.6, top_p=0.95, gpu-mem 0.88, fp8, num_speculative_tokens matching the caterpillar
  tree `[(0,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0),(0,1),(0,0,1),(0,0,0,1),(0,0,0,0,1)]`.
- **Memory: run the recovery cycle (sync; drop_caches; swapoff -a; swapon -a via
  LUMO_SUDO_PASSWORD) before EVERY server reboot** — direct-docker launch wedges ~100GiB on
  GB10 (the host-OOM danger). One server at a time. oom_score_adj-protect this codex stack.
- Commit + push every step on branch `fr10-gdn-tree-kernel`. Record numbers in committed docs.
- All math/parity through committed tests, never hand-rolled one-offs.
- Iterate the conv/drafter logic OFFLINE (parity tests, no boot) where possible; boot once to
  confirm. Stop the boot-per-bug cycle.
