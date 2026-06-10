# FR13 Conv-Fix A/B — Bind (2026-06-10)

Findings bind only — **NO pass/fail close**; verdicts are for the user. This bind reduces the decisive A/B
specified in `FR13_S1S2S3_DISCRIMINATE_BIND.md` (m1 seam 1 = conv prior-window, SHARED machinery) after the fix
landed at `c0b53f5d` ("FR13 conv fix: committed-path conv window (branch-valid) + forced-spine diag flag").

## Header — flag state + seeds (per boot)

**Substrate**: HEAD `c0b53f5d` (conv fix `FR13_CONV_COMMITTED_PATH`, default ON; diagnostic
`FR13_FORCE_SPINE_COMMIT`, default OFF) on top of `4d45be27` (S1 bonus-row fix) + `cc008587` (determinism flags ON).
B=1 strictly sequential, 4 pinned SWE prompts (`output/fr13_acceptance_ladder/prompts_swe4.json`, byte-identical
reuse), max_tokens=128, probe `scripts/fr10_quick_decode_tps_probe.py` samples_per_prompt=1 warmup=0
`--require-tree-engagement`. **Sampling**: greedy temp 0.0 / top_p 1.0 / seed 1313 (both boots).
**Run dir**: `output/fr13_convfix_ab/` (`run_header.json` = authoritative per-boot flag state). Serial boots,
teardown + `recover_host_memory` between, `docker ps` empty before each; post-run 105G available.

| boot | topology | FR13_CONV_COMMITTED_PATH | FR13_FORCE_SPINE_COMMIT | other flags | windows |
|---|---|---|---|---|---|
| A (DIAGNOSTIC) | 9-node caterpillar | **0** | **1** (diag-only) | BI=1, FR13_BI_TREE_ATTN=1, FR13_TREE_BONUS_SELF=1, FR10_METRICS=1, GPU_UTIL 0.82, fp32 capture NUM_TOKENS=10 LIMIT=600 | `a_forced_spine_greedy` |
| B | 9-node caterpillar | **1** | 0 | same | `b_greedy`, `b_greedy_rep2` |

**Boot A is DIAGNOSTIC-ONLY** (`FR13_FORCE_SPINE_COMMIT=1`, same class as `FR10_ALLOW_LINEAR_FALLBACK`): its
numbers must NEVER be cited as serving/gate results. Flag engagement verified live: 154/154 committer rows
`forced_spine_commit=true`, policy `greedy_tree_lcp_max_FORCED_SPINE_DIAGNOSTIC`, 0 non-spine winners, 5 paths
still scored on every event. Boot B conv-fix engagement verified via in-container env + patched
`gdn_linear_attn.py` (6 committed-path markers) + the behavioral accept delta (the
`FR10_TREE_GDN_COUNTER_DUMP` file never materialized — see Problems).

References: chain boot `output/fr13_s1s2s3_discriminate/chain_greedy` (accept/event 2.277, 159 events); pre-fix
caterpillar `tree_greedy` (1.819, 182 events; S2 corrupt event = p0 gen_pos 16); `native_bi1_greedy` (BI=1, fp32
captures, accept/event 3.047 greedy); `output/fr13_acceptance_ladder/native_greedy` (BI=0 fork reference, 3.154).

---

## A — Decisive A/B (forced-spine caterpillar vs chain): NOT token-identical

The bind-specified discriminator: caterpillar topology with commits FORCED to the spine path (alts verified but
never winning). If next-event drafts then matched the chain boot token-for-token, m1 contamination would be
entirely in branch-commit state advance. **They do not match.**

Lockstep identical-committed-prefix drafter comparison (artifact `a_forced_spine_vs_chain.json`):

| comparison | pairs | identical | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|---|---|
| **Boot A (forced-spine) vs chain** | 17 | **7** | 4/17 | 6/17 | 7/17 | 9/17 | 9/17 |
| pre-fix caterpillar (normal commits) vs chain | 15 | 4 | 3/15 | 6/15 | 8/15 | 9/15 | 11/15 |
| **cross-boot floor**: forced-spine vs pre-fix caterpillar (same topology) | 23 | **14** | 5/23 | 6/23 | 7/23 | 6/23 | 9/23 |

Served streams fork from chain at pos 15/6/6/21 (p0..p3). Forced-spine **helps but does not converge** to chain.
DIAGNOSTIC-ONLY accept/event 1.987 (306/154).

**Two confound-breaking facts** (`a_forced_spine_addendum.json`):

1. **Commit-independent divergence exists**: at the boot's FIRST event (p0 ev0, zero commits ever), BOTH
   caterpillar boots reproducibly propose spine d2=364 where the chain boot proposes 271. A drafter divergence
   that precedes any commit cannot be commit-state contamination ⇒ **m1 contamination is NOT entirely
   branch-commit state advance**; a component lives in the 9-vs-5-row verify-forward shape itself.
2. **Cross-boot drift floor is the same order as the signal**: forced-spine vs pre-fix caterpillar (identical
   topology) is only 14/23 identical at matched prefixes, with pure pre-commit drift (p0 ev0 node-8 alt 3274 vs
   1046; p1 ev0 full divergence from d0 despite identical prefix+token0 — boot1-internal rep2 was byte-exact, so
   this is cross-boot numeric drift surviving BI=1, not history leakage). **The token-identical bar was
   unreachable by construction**; attribution beyond "not entirely branch-commit" is not supported by this A/B.

**A/B verdict**: m1 is at most PARTIALLY branch-commit state advance; a commit-independent caterpillar-vs-chain
forward-shape component is proven; finer attribution is confounded by the cross-boot drift floor (autotune /
cross-session class; Boot A also ran HEAD `c0b53f5d` vs reference boots at `4d45be27`, inseparable here).

## B — S2 at the bound trigger: GONE (logit evidence)

With `FR13_CONV_COMMITTED_PATH=1` (normal commits), the deterministic whole-forward corruption bound at
**prompt-0 gen_pos 16** (trigger = follows acc=2 BRANCH commit, winner [0,1,4]) **no longer exists**:

- p0 matches native BI=0 token-for-token to **pos 35** (pre-fix fork was 16). No verify event exists at gen_pos 16
  anymore: the pos-13 [0,1,4] branch-commit site (the old trigger) now **accepts 3 (was 2)** and the continuation
  matches native.
- Lockstep fp32 logits vs `native_bi1_greedy` (`s2_bootB_vs_nativeBI1_greedy.json`): 27 pairs, 0 missing captures,
  capture-offset alignment strict 130/130. p0 events bracketing the old corruption all **argmax-match**:
  pos 12 mean|d| 0.292, pos 13 0.365, pos 24 0.405, pos 30 0.581 — against baseline median 0.388. Pre-fix the
  corrupt event sat at mean|d| 3.32 / max|d| 15.75 / root flip margin 7.25 (~7x baseline).
- Caveat: events directly following branch commits never lockstep-paired with native (native event boundaries
  differ), so the branch-trigger class is cleared via stream-match-to-pos-35 rather than direct paired logits.

**REMAINING divergence — different class**: a weaker gross-flip at **p0 pos 35**: root argmax 8445 vs native 44675
at native margin 6.0, event mean|d| 1.341 (3.5x baseline, below the 8x outlier bar), max|d| 10.76. Trigger context
= follows an **acc=4 SPINE commit** ([0,1,3,5,7], reject_parent_target) — NOT a branch commit. fp32 banked:
`b_greedy/logs/tree_final_logits.call11.pt` vs the native capture per the align map. Also flagged: p2 pos 18,
a single d2 verify-row argmax flip (1970 vs 3425, margin 2.25) with event mean|d| 0.47 ≈ baseline — S3
drafter-flip class, not whole-forward corruption (it materializes as p2's serve fork at pos 21).

**S2 verdict**: the bound branch-commit-class corruption is HEALED by the conv fix. The residual p0-pos-35 flip
follows a SPINE commit — and under the fix, spine winners' conv reads are byte-identical to legacy (proven,
`tests/test_fr13_conv_committed_path.py` test 3) — so its root is NOT the conv prior read: it localizes to the
machinery the conv fix deliberately did not touch (legacy ssm publish/remap/h0 handoff) or to verify-forward
numerics. Per the bind's directive: **the REPLAY ROUTE rebuild is the next discriminator — go straight to the
replay campaign.**

## C — Accept progression (greedy B=1, caterpillar)

| arm | accept/event | events | note |
|---|---|---|---|
| pre-fix caterpillar (4d45be27) | 1.819 | 182 | the "alts NET-NEGATIVE ~0.46" datum |
| **post-conv-fix caterpillar (Boot B)** | **2.215** | 130 walked (raw 2.187/134 incl. trailing clipped; 288/130 probe) | branch winners ACTIVE: 32 events ([0,1,4]/[0,2]/[0,1,3,6]) |
| chain boot | 2.277 | 159 | alt-free reference |
| native MTP-5 BI=1 / BI=0 | 3.047 / 3.154 | 127 / — | the FR13 floor bar is native 3.076 (E5) |

The conv fix moves the caterpillar **+0.397** to within **0.061** of the chain boot — the pre-fix "alts
net-negative ~0.46" deficit is essentially **erased**. Boot B serve-streams vs native BI=0: first-fork p0 35 (was
16), p1 21 (11), p2 21 (25 — regressed 4, the pos-18 d2 flip), p3 68 (31); total match length 145 vs 83 pre-fix.
Same-seed repeat (`b_greedy_rep2`) **byte-identical on all 4 prompts**, same aggregates. Per-request decode TPS
mean 6.81 (informational only; speed deferred per policy).

**Trajectory caveat (applies to every row)**: these are NOT like-for-like trajectories — the fix changes served
streams from early events, so deltas mix trajectory change with acceptance, the same caveat carried by the 1.819
reference itself. The remaining −0.83/−0.94 gap to native is the S3/forward-drift share, unchanged in kind.

## D — Scope: what was deliberately NOT fixed

**The legacy ssm publish / remap / h0 handoff (m1 seam 2) was NOT touched — per the hybrid decision.** The conv
fix is SHARED machinery (the replay route keeps the conv half); the replay-route rebuild (branch
`fr13-replay-route`, CPU-cleared at `50ac5f5a`, GPU-gate TODO) **deletes** the legacy ssm publish/remap/h0
machinery entirely, so fixing it on main is waste (user decision, bound in `FR13_S1S2S3_DISCRIMINATE_BIND.md`).
The residual p0-pos-35 spine-commit-class flip is exactly the discriminator the replay route should face: if it
heals under replay, the legacy ssm/remap/h0 side is convicted; if it survives, the verify-forward numerics
(commit-independent component from §A) own it.

`FR13_FORCE_SPINE_COMMIT` remains DIAGNOSTIC-ONLY (never a serving config; sampled committer fails loud on it).
`FR13_CONV_COMMITTED_PATH=1` is the new default on main, revertible via `=0`; under the flag the
`fr12.tree_conv_detail.v1` capture's `read_cols` holds NODE columns (`prior_read_mode` self-describes).

## Problems / riders

1. Boot A's decisive A/B is confounded by cross-boot numeric drift at the same order as the chain-vs-caterpillar
   delta (§A fact 2); only the two pre-commit facts are clean conclusions. The HEAD mismatch (c0b53f5d vs
   4d45a27-era references) is a plausible drift contributor and cannot be separated from autotune drift here.
2. The conv fix does NOT eliminate all whole-forward divergence (p0 pos 35, spine-commit class — own
   localization needed: replay-route discriminator, or an FR13_POS16-style ladder on this event).
3. p2's fork vs native regressed 25→21 under the fix (d2 row flip, margin 2.25, S3 class).
4. `FR10_TREE_GDN_COUNTER_DUMP` never materialized; conv-fix engagement was verified by env + patched-file
   markers + behavioral delta instead.
5. At HEAD the legacy post-remap read and the committed-path read coincide mathematically when the remap executes
   exactly (test 4 of the fix's suite); the fix's causal value — confirmed by this A/B — is removing the conv
   read's dependence on the in-place remap machinery.
6. The pristine `gdn_linear_attn` snapshot at `/tmp/vllm_pristine_019` does not accept `_patch_gdn_linear`
   (pre-existing needle mismatch); patch application end-to-end validated only at container re-patch/boot (it
   booted and engaged in this campaign — that validation is now done).
7. All raw artifacts live under gitignored `output/fr13_convfix_ab/` (analysis scripts
   `analyze_bootA_forced_spine.py` / `analyze_bootB_convfix.py` in the run dir); this bind is the committed record.
