# FR13 tree spec-decode — pipeline speed breakdown & tree-scaling

**Source:** tail6 arm (MTP head + 16-node arctic tail, 21 nodes, n_pad=32), LIVE B=4 SWE-Verified,
`deploy_speed_tailg4c.json` GPU timer sidecars (FR13_SFWD/DFWD/CFWD_GPU_TIMER). Committed 2026-07-15.

Legend: **[M]** = directly measured this run · **[P]** = measured in a prior probe (S1/S2/S3, memory) ·
**[A]** = architecture-derived (not yet isolated by a timer).

---

## 0. Top-level (per decode STEP, GPU-active time) — [M]

| stage | ms/step | share | basis |
|---|---|---|---|
| **DRAFTER** | 97.7 | 35% | FR13_DFWD_GPU_TIMER: the MTP head forwards |
| **VERIFY**  | 85.8 | 30% | FR13_SFWD_GPU_TIMER: the one tree forward (s_per_fwd_gpu) |
| **COMMITTER** | 97.5 | 35% | FR13_CFWD_GPU_TIMER: path pick + GDN state replay |
| **total GPU** | **281** | 100% | committed 5.28 → derived_tps_fullstep_gpu = **18.81** |

Other bases: verify-only `derived_tps_gpu`=61.85; wall-clock w/ prefill+idle `aggregate_decode_tps`=10.9,
`per_request_decode_tps`=4.8. The `_gpu` numbers EXCLUDE host gaps (arctic trie walk, assembly, H2D, prefill,
idle) — those live in the ~40% wedge between fullstep_gpu (18.81) and the wall-clock aggregate.

**Headline:** the three GPU stages are ~1/3 each. The committer being **as expensive as the drafter** is the
non-obvious result (one would guess the committer is cheap).

---

## 1. DRAFTER — 97.7 ms GPU (35%)

What runs, per step:
- **MTP head forwards** — [M gpu] the ~97.7 ms is the autoregressive MTP head: root is produced by the base
  decode forward; then the MTP head runs for depths 1..4 (spine_steps capped at 4 in tail mode) = **4 light
  forwards** (~24 ms each — the MTP head is a few layers on the shared hidden state, far cheaper than the 85.8
  ms base forward). This is the GPU cost the DFWD timer captures.
- **Arctic `.speculate()`** — [P host] the tail-chain retrieval (trie walk). HOST-bound, NOT in the 97.7 ms
  GPU. Prior probe S3 measured the drafter host-side at ~117–140 ms (host-bound, FR-Spec refuted) — that host
  time is the arctic walk + assembly + H2D, and it sits in the wall-clock gap, overlappable with GPU.
- **Assembly + H2D** — [A host] `build_tail_columns`/`build_cat33333_columns` build the per-node token columns
  on CPU, one H2D per depth. Small but host.

### Scales with tree:
- **width (more branches/depth):** MTP head forwards are FIXED (the head always drafts its 5 depths regardless
  of how many branch slots the tree has — branches are extra `torch.topk` reads off the SAME step logits, ~0).
  So **drafter GPU ~FLAT vs width.**
- **depth of the TAIL (arctic chain length):** adds HOST time only (more arctic tokens to retrieve + assemble),
  ~O(tail_len), **not GPU.** The MTP GPU part stays fixed.
- **Net:** drafter GPU is ~**FIXED** in tree size (it's the MTP head, which is a constant 5-depth draft).
  Growing the tree does NOT grow the drafter GPU — only a mild host cost. => **the drafter is a per-STEP fixed
  cost, so higher accept amortizes it directly.**

---

## 2. VERIFY — 85.8 ms GPU (30%)

What runs: ONE base-model forward over all n_pad tree nodes (TREE_ATTN backend + the GDN tree scan kernel),
producing the verify logits for every node in parallel.

Sub-costs [A, design §4]:
- **Weight read (HBM):** the base model weights load ONCE per forward (~98.6 ms HBM floor). This dominates and
  is **independent of tree size** (16 vs 32 nodes read the same weights).
- **Tree attention intra-block:** O(tree_n²) but tiny (256 → 1024 flops for 16 → 32 nodes).
- **GDN tree scan:** node-independent grid; extra nodes pile into the same CTAs as an O(N_PAD) `static_range`
  loop with an O(N²) ancestor gather (small base, ~8× ALU going 16→32 — but ALU, not HBM).

### Scales with tree:
- **width / depth / node-count:** ~**FLAT** — the weight-read floor dwarfs the node-dependent ALU. Design §4:
  "node-dependent work is sub-few-percent of the weight floor → verify stays flat." GATE-2 confirmed n_pad=32
  boots + serves at the same register budget (BV=8).
- **This is the whole reason a bigger tree is cheap:** you pay the 85.8 ms weight-read regardless, so extra
  nodes (more accept opportunity) are ~free on verify.

---

## 3. COMMITTER — 97.5 ms GPU (35%)

What runs, per step:
- **Path selection** — [A] pick the accepted path (longest prefix of a tree path whose tokens match the
  verify sample), per row. O(tree paths) enumeration.
- **GDN state replay** — [A, the big one] recompute the SSM/conv recurrence along the ACCEPTED path so the KV/
  mamba state advances correctly for the committed tokens (launch_tree_gdn_replay). This recomputes the
  recurrence for each accepted token. Cost ~O(accepted-path length) per row.
- Sampling/bookkeeping — small.

### Scales with tree — THE ONE THAT GROWS:
- **width (paths):** path enumeration grows ~O(#paths) — more branches = more candidate paths to score.
- **accept (replay):** GDN replay grows with the **accepted-path length** = the accept itself. Deeper accept
  (a tail arm accepting to depth 10) replays a longer recurrence than a depth-5 baseline. This is why tail6's
  committer (97.5 ms) is inflated vs a depth-5 baseline's would be.
- **Net:** committer scales with BOTH tree size (enumeration) AND accept (replay) => it is the **eventual
  ceiling** as you scale accept. Known lever: S1 "sampled-committer port" (memory) — a cheaper commit path.

---

## 4. Putting it together — how the pipeline scales, and the levers

Per-COMMITTED-TOKEN time = (drafter + verify + committer) / accept:
- **drafter (fixed) / accept** → DROPS as accept rises.
- **verify (fixed, HBM) / accept** → DROPS as accept rises.
- **committer (grows with accept) / accept** → ~FLAT (replay scales with accept, so per-token ~constant).

=> **Bigger tree → higher accept → drafter+verify (65% of the step) amortize → net faster**, until the
committer's accept-linear replay becomes the dominant term. CONFIRMED live: tail6 +19% accept → **+13%
fullstep-TPS** vs the depth-5 baseline.

**Speed levers, ranked (fall directly out of the scaling):**
1. **Scale accept (bigger/richer tree, union, deeper tail).** Amortizes the fixed 65% (drafter+verify). Same
   lever as the accept work — accept and speed align here. Verify is HBM-flat so the tree is ~free to grow.
2. **Cheaper committer (35%, and rising with accept).** Sampled/faster GDN replay (S1). Pairs with (1): the
   more you scale accept, the more the committer matters, so this is the structural unlock for deep trees.
3. **Overlap arctic `.speculate` (host) with the verify forward (GPU).** Hides part of the drafter's host cost.
4. **Fewer/parallel MTP head forwards** (the drafter's fixed GPU 97.7 ms = 4 autoregressive MTP forwards).

**To measure the sub-breakdowns precisely** (currently [A]/[P]): add per-sub-stage timers — split DFWD into
(MTP-forward vs arctic-host vs assembly-H2D), split CFWD into (path-pick vs GDN-replay). Queued; needs a GPU run.

## 5. MTP↔arctic parallelism — the sequential trap and the union fix (user 2026-07-15)
CURRENT TAIL = SEQUENTIAL: decide_tail seeds arctic with `pattern = _COMMITTED + MTP head tokens` (MTP-guided
suffix). Arctic can't start until the MTP forwards finish, and the suffix match uses the RECENT tokens (incl.
MTP) => FULL dependency, not partial. So the GPU IDLES during the ~100-140ms host arctic walk (head-loop GPU ->
decide_tail host -> verify GPU) -- a big chunk of the ~40% wall gap (aggregate 10.9 vs fullstep_gpu 18.81).
FIX = the UNION (independent arctic tree from _COMMITTED ONLY, not MTP-seeded): arctic host walk has NO dep on
the MTP head -> runs IN PARALLEL with the MTP head forwards (GPU) -> the arctic host is HIDDEN behind MTP GPU.
=> the union wins on BOTH axes: complementarity (accept) AND pipeline parallelism (speed). The MTP-guided tail
traded parallelism for a confident seed. Two interleave levels: (1) within-step arctic-host || MTP-GPU (needs
union); (2) across-step: prefetch step N+1 arctic while step N verify+committer GPU runs (hides arctic behind
committer even for the seeded variant). CAVEATS: parallelism gain is INFERRED from the sequential structure,
not yet measured (needs union + stage timer); MTP-seed DOES buy tail accept (arctic continues MTP's confident
prefix) so the independent union may draft a weaker deep spine -> accept-vs-speed tradeoff is the measurement.
suffonly arm (running) = first data point on arctic-from-committed-alone quality.

## 6. CORRECTION (2026-07-15, audit): per-step timers reverse the TPS verdict — tail6 is SLOWER
The §0 table used the deploy_speed derived_tps_fullstep_gpu (18.81) which MIXES per-DRAFT verify (s_per_fwd_gpu
85.8ms) with per-STEP drafter/committer (fr13_measure.py:1568) -- at B=4 there are ~4 draft-events/step, so
verify is understated ~4x. The 3 timers DON'T overlap (SFWD=execute_model verify, DFWD=propose drafter, CFWD=
committer replay). CONSISTENT per-STEP (raw sidecars):
  baseline t33333: verify 258.8 + drafter 101.9 + committer 74.0 = 434.6ms/step, committed/step 18.36 -> aggregate GPU-TPS ~42.2 (per-req 10.56)
  tail6 (depth-11): verify 339.7 + drafter 100.6 + committer 113.5 = 553.8ms/step, committed/step 21.12 -> aggregate GPU-TPS ~38.1 (per-req 9.53)
=> tail6 is ~10% SLOWER (GPU compute) despite +19% accept. ROOT: the DEEP tail's VERIFY (+31%) + COMMITTER
(+53%) scale with tree DEPTH and exceed the accept gain; the drafter is NOT the cost (100.6 vs 101.9). So:
- **VERIFY is NOT HBM-flat for deep trees** (design §4 was wrong): 258ms at depth-5 >> 98.6ms weight floor ->
  the GDN tree-scan is depth-dependent compute, not weight-read-bound. A deeper tail costs real verify.
- **The tail as built (depth-11) is a COST-GATE, not a ship** (accept up +19%, GPU-TPS down -10%).
- LEVERS for a NET speed win: (1) SHALLOWER tail (depth 6-8: less verify+committer, still >baseline accept) --
  find the accept/depth-cost sweet spot; (2) cheaper committer (the +53% that scales with depth, S1 sampled);
  (3) the union's arctic-parallelism helps the DRAFTER, which is NOT the deep-tail cost -> won't fix this.
CAVEAT: baseline sidecar is mid-run (16-task batch-1, n=413 steps) but per-step GPU is workload-independent so
robust. Depth is the speed axis to sweep next.

## 7. native MTP-5 vs t33333 vs tail6 (user comparison 2026-07-15)
SHAPE/SOURCE: native = 5-node depth-5 CHAIN, MTP only (5 heads), NO branches, NO FR13 tree committer (stock
linear commit) -- the fr9 baseline (~40 TPS, accept ~3.4). t33333 = 15-node cat33333 (depth-5 spine + 2
branches/depth), MTP only (spine=heads, branches=MTP topk), n_pad=16. tail6 = cat33333 head (15) + 6-node
arctic CHAIN tail (depths 6-11) = 21 nodes, n_pad=32; head=MTP, tail=ARCTIC.
STAGE (per-step GPU, ms): native (draft n/a, verify ~150-200, committer NATIVE-linear ~cheap) acc~3.4 ~40TPS;
t33333 (draft 101.9 / verify 258.8 / commit 74.0 = 434.6) acc 3.59 ~42TPS; tail6 (draft 100.6 / verify 339.7 /
commit 113.5 = 553.8) acc 4.28 ~38TPS. (native drafter/committer NOT FR13-timed = stock path.)
STORY: monotone accept-for-speed trade. The FR13 TREE itself is the first tax: t33333 vs native adds ~+150ms/
step (tree verify 258 vs ~180 + a NEW 74ms GDN tree committer native lacks) for only +0.2 accept (~wash on
speed). tail6's arctic tail is the real accept jump (3.59->4.28) but n_pad=32 pushes verify 339 + committer 113
=> -10% vs native. KEY: the COMMITTER is where tree/tail diverge from native (native=cheap linear; tree=74ms;
tail=113ms GDN replay, depth-scaling) -> a sampled/faster tree committer is the unlock to keep the accept while
closing the speed gap to native.
