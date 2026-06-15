# FR13 — Committer fork-margin probe: forks are BIMODAL (≈half genuine leaf-wins); spine-bonus is a partial lossless lever, not a native-3 fix

Date 2026-06-15. GPU workflow `wlg8a66g8`, ProbeClassify→Verify **verify HOLDS=True**. Raw:
`research/fr13_workflows/fr13_fork_margin_probe_raw.json`. Instrument committed default-OFF (82bf2162 dump +
launcher passthrough; bd5a9e50 classify alignment fix). Answers the user's "are all flips near-tie?" → NO.

## The decisive measurement (non-vacuous, all green)
At each clear-margin fork, the deciding LCP-divergence node's VERIFY top1−top2 margin (nat), all 23 sorted:
`0.125, 0.25, 0.25, 0.25, 0.25, 0.4375, 0.5, 0.5, 0.625, 0.875 | 1.25, 1.375, 1.375, 1.5, 1.5, 2.25, 3.125,
3.25, 3.625, 3.625, 7.125, 8.5, 9.125`. **Bimodal at the 1.0-nat realization floor:**
- **B = 10 sub-1nat near-ties** (verify nearly indifferent → leaf-vs-spine within float noise → FIXABLE losslessly).
- **A = 13 confident ≥1nat** (verify strongly preferred the leaf-matched token, heavy 7–9 nat tail = genuine
  leaf-LCP accept-edge wins → suppressing them = rejecting a real accept = LOSSY).
- Restricted to the **19 TRUE committer leaf-forks** (is_fork=True, best_leaf≠spine_leaf): **A=11 / B=8**.
- The other **4 are is_fork=False** spine-served verify-vs-recurrent **realization flips** (kernel-numerics,
  the diffuse L0–L58 GDN accumulation) — an LCP margin-damp does NOT touch them.

## Non-vacuity (all four legs proven; a real measurement-trap caught + fixed)
DUMP FLAG LIVE (FR13_FORK_MARGIN_DUMP=1 in 3 worker /proc/environ) + dump non-empty (250 recs, per-node
margins); K1 live (FR13_SCAN_ALIGN=1 MODE=body); RECURRENT engaged (44352 calls); tok/draft=9 (cat9 engaged);
det [T,T,T,T]; accept/event 3.004 (matches banked K1). JOIN non-vacuous: all 4 prompts coverage=1.0, 23/23
forks joined, 0 miss, 19/23 land on is_fork=True; all 23 served fork-tokens present in their joined
committed_row (same stream). **Caught + fixed (#12):** the served stream carries a 1-token chat-template prefix
absent from the raw committer dump (+ warmup/rep2) — head-skip+offset segmentation slides to a contiguous
coverage=1.0 alignment; WITHOUT the fix the JOIN would have been a disjoint-capture artifact (vacuous). Dump is
READ-ONLY (after out_rows.append; default-OFF byte-identical); native = A/B oracle only; no copy/dense/force-spine.

## VERDICT: spine-bonus (margin-damp) is a PARTIAL lossless lever, NOT a native-3 fix
A deterministic rank-2 LCP **near-tie margin-damp** (stop a leaf winning the LCP boundary when the deciding
parent_target margin <1.0 nat; **preserve** >1nat leaf wins; NOT force-spine) cleanly removes the **8 sub-1nat
true-leaf-fork near-ties (B)** losslessly, taking residual **23 → ~15**. But it does **NOT reach native-E5=3**:
the residual ~15 = **11 fundamental confident leaf wins** (the accept edge — the verify forward strongly, 2–9
nat, preferred the leaf token; suppressing = lossy) **+ 4 spine verify-vs-recurrent realization flips**
(kernel-numerics, diffuse, K2–K5 dead). So **lossless-to-native-3 WHILE KEEPING LEAVES is not cleanly
achievable**: the genuine leaf wins ARE simultaneously the accept edge AND the lossy-ness (coupled, as the
reshape A/B already showed). The deployable arbiter (accept/event 3.004 ≈ native 3.076) is already at parity.

## The close/pass-fail (user's call; literature pending)
1. **Implement margin-damp** (committer-only, FREE, no-copy/no-HBM) — a real lossless improvement 23→~15, keeps
   leaves + speed. Worth taking regardless.
2. **The residual ~15 is the accept-vs-lossless tension** (11 genuine leaf wins) + a small diffuse kernel
   residual (4). To go below ~15 with leaves you must either (a) accept it = relax to accept/event-parity, or
   (b) find a committer that handles the confident leaf wins losslessly — the **Traversal Verification**
   (arXiv 2505.12398) bottom-up lead, under evaluation in the concurrent literature workflow `wtl0kz4wf`.
WY stays parked. Links: [[reference_scalar_metric_per_token_blindspot]], [[feedback_depth_matched_accept_compare]],
[[reference_multispine_not_lossless_closed_nonship]].
