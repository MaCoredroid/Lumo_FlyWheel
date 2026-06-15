# FR13 — K1 (per-node bf16 b_h store-boundary): a VERIFIED PARTIAL lever, accept held; does NOT reach native alone

Date 2026-06-15. GPU workflow `waao62oj0`, BootRescore→Verdict **verify HOLDS=True**. Raw:
`research/fr13_workflows/fr13_k1_store_boundary_raw.json`. Kernel seam committed b91c1bc0 (default-OFF);
forked-launcher flag passthrough committed 365da33b.

## What K1 is
Adopt native packed-decode oracle (2)'s per-token bf16 `b_h` store-reload into our fp32-carried tree-scan:
`_gdn_node_step` (fr10_gdn_tree_kernel.py) inserts `if SCAN_ALIGN: state_i = state_i.to(tl.bfloat16).to(fp32)`
on the CARRIED state (after `out_i` taken from precise fp32, before return), matching (2) (b_o from fp32 b_h
fused_recurrent.py:331, store bf16 L336, reload L303). Combined with body seams d (l2norm div) + e (beta
round-trip) under MODE=body. Keeps EXACT cat9 geometry/h_cache — NO path fork (unlike recompute). Default-OFF =
constexpr-dead = byte-identical locked path (reward-hack-clean).

## Non-vacuity PROVEN (the first K1 boot FAILED-LOUD, recovered)
- First boot: `FLAG NOT LIVE` (locked launcher had no `-e FR13_SCAN_ALIGN` passthrough → bare env curated out
  of mp/spawn worker, bug-class #9). Root-caused + fixed 365da33b (forked launcher forwards `${FR13_SCAN_ALIGN
  :-0}` default-OFF). Re-boot: flag live in worker /proc/1/environ (PID 1/175/556 carry FR13_SCAN_ALIGN=1
  MODE=body). (i) flag live, (ii) served diverges from OFF, (iii) RECURRENT_PATH_ENGAGED=True, det [T,T,T,T].

## The numbers (recurrent-oracle frame, cat9-vs-E5 = depth-matched, both depth-5)
| arm | raw clear flips | de-cascaded (FR13_PLUS2 gap≤2) | accept/event |
|---|---|---|---|
| native-E5 (BAR) | 3 [0,0,2,1] | 3 [0,0,2,1] | 3.076 |
| cat9 OFF | 23 [5,4,5,9] | 18 [3,4,4,7] | ~3.15 |
| **cat9 + K1** | **20 [6,5,4,5]** | **12 [2,4,3,3]** | **3.004 (held)** |
| recompute (separate, non-lossless) | 32 | 23 | — |

Verify INDEPENDENTLY re-derived 12/18/3 via the canonical FR13_PLUS2 cluster-collapse rule (gap from the
immediately-PRECEDING position; naive "gap-from-last-KEPT" wrongly gives 14).

## VERDICT: PARTIAL lever (holds=True), NOT "dead/no-collapse" (monitor's first report was too strong)
K1 closed **~33% (6 of 15) of the OFF→native de-cascaded gap (18→12) WITHOUT cratering accept** (3.004 vs OFF
~3.15, native 3.076; cat9 tree intact, tok/draft=9) — so it is NOT the "incumbent-dead / flips stay ~23 or
rise" branch. BUT it did NOT collapse to native-3 (still **+9 de-cascaded above native**). So the diffuse floor
is **PARTLY** this per-node store-boundary op-order (K1's ~1/3) and **PARTLY** trajectory/topology-intrinsic
(the committer leaf-fork carrier, the other ~2/3, FR13_LEAF_CORESIDENCY_PATH).

**RED-TEAM caveat (#12 cross-boot):** raw −3 (23→20) is WITHIN the cross-boot ±9 floor (the recompute prior's
resolution); de-cascaded −6 (18→12) is verified-ARITHMETIC but a cross-trajectory comparison (K1 boot vs OFF
boot, different streams), so a single boot cannot fully separate a weak-real-lever from trajectory noise. Honest
read: K1 is a **weak-to-moderate partial lever that holds accept**, NOT a native-reaching fix.

## Consequence for the lossless+fast path
K1 holds accept (3.004) = a **drop-in-able partial** that can COMBINE with the committer-fork fix: K1 cuts the
kernel ~1/3, and the rank-2 LCP near-tie margin-damp (FR13_LEAF_CORESIDENCY_PATH) targets the remaining ~2/3
committer forks. **K1+margin-damp is the lossless+fast candidate** — evaluate the margin-damp WITH K1 ON. Next:
the committer-replay margin probe (classify the residual 12 forks fundamental-vs-fixable). User chose
"probe then margin-damp" (2026-06-15). Links: [[project_fr13_tree_reshape_unifying_lever]],
[[reference_diffuse_gdn_accumulation_explained]], [[feedback_depth_matched_accept_compare]].
