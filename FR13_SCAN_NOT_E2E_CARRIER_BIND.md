# FR13 — DECISIVE: the GDN scan state-feed is NOT the e2e carrier (non-vacuous, re-open)

Date 2026-06-15. Re-run w7wr68z06 (verify holds=false = correctly "NOT a lossless win"). The FIRST fully
non-vacuous e2e measurement this session — all three instruments independently re-verified non-vacuous
(the recurring playbook-#9 trap finally avoided).

## Non-vacuity (ALL proven)
- STATE gate negative_control_powered=True (hardened): neg-control flips int_view→False, native_state_norm
  =730.31>0, our_state_norm=730.34>0 — cannot re-vacuum off a zeros ref (the e428db3a fix).
- Recurrent oracle RECURRENT_PATH_ENGAGED=True on ALL 3 arms (decode_calls native/recompute 48768, OFF
  41376); flips are GOLD-MARGIN (deviation_nat>1.0 + full oracle_topk), NOT streamed top_logprobs.
- Recompute engaged TRIPLE-proven: bridge needle FR13_SCAN_ALIGN=1+MODE=recompute on worker pid=175 +
  /proc/175/environ + served stream diverges from OFF (LCP 15-69, ~369 token diffs). within-boot det [T,T,T,T].

## The numbers (recurrent oracle = deployment-correct single-step decode, SAME oracle all arms)
| arm | clear-margin flips | per-prompt | rate/token |
|---|---|---|---|
| **native-E5 (BAR)** | **3** | [0,0,2,1] | 0.0059 |
| cat9 OFF (deployed) | 23 | [5,4,5,9] | 0.0529 |
| cat9 RECOMPUTE (fix) | **32 (ROSE)** | [10,9,7,6] | 0.0625 |

STATE gate (kernel): OFF int_view=False max_abs **0.0289** (REAL scan-vs-native-packed gap); RECOMPUTE
int_view=True **0.0** (bit-exact); BODY_SEAMS@deployed-geom 2.86e-6 (bf16-ULP floor), @native-geom 0.0.
Artifact-checked (playbook #12): OFF EOS'd early (denoms 435 vs 512); common-prefix-normalized recompute=25
(STILL up vs 23); per-position rate rose 0.0529→0.0625 — the rise is REAL, not a length artifact.

## CONCLUSION
The per-node GDN scan STATE alignment makes our scan bit-exact (int-view 0.0) to the native packed-decode
incumbent SASS, **yet e2e clear-margin flips did NOT drop toward native 3 — they ROSE (23→32).** Therefore
**the rank-1 GDN scan state-feed is NOT the dominant e2e carrier of the cat9 flip gap.** The kernel-level
state gap (0.0289) is REAL but NON-CAUSAL for the flips. native-E5=3 (clean, frame-invariant per the prior
CLOSED bind) is the existence proof that a ~3-flip realization exists at the same model/fp8/frame — so the
23 is a real defect, NOT irreducible, but its carrier is elsewhere.

## What this challenges
- Recompute-from-spine REMOVES leaf co-residency (each node replayed from the spine independently) AND aligns
  to native geometry — and it made flips WORSE. So the banked decomposition's "+17 leaf co-residency carrier"
  is itself in question: removing co-residency did not help. Either co-residency is not the carrier, or
  recompute introduces a compensating trajectory shift.
- Recompute produces a DIFFERENT deterministic stream (~369 tok diffs vs OFF) = NOT byte-lossless, NOT a free
  drop-in. **Do NOT bake recompute.**

## RE-OPEN — candidate carriers (none yet e2e-tested non-vacuously)
The cat9-vs-native 23-vs-3 gap is the difference between OUR tree spec-decode (9-node caterpillar + MTP drafter
+ our tree-verify) and native linear MTP-5. With the scan ruled out e2e, candidates: (1) TRAJECTORY-FORK
cascade (cat9 served stream forks early from native's → downstream high-entropy crossings cascade; the
reanalysis already saw tree flips near-disjoint from native's = a superset of crossings, not a per-forward
seam); (2) the tree-verify ACCEPT/COMMIT logic (rejection_sampler ↔ committer vs native MTP-5 accept); (3) the
DIFFUSE multi-layer bf16-ULP accumulation (FR13_DIFFUSE_GDN_EXPLAINED) — but it must explain the recompute-
worse result; (4) the DRAFTER / tree topology itself. The kernel-seam framing (single per-forward op) is
weakened: the scan was the strongest per-forward candidate and it's not causal.

## Reward-hack / hygiene
CLEAN: native packed-decode + recurrent oracle = A/B oracle only (no served-path splice); committed kernel
fr10_gdn_tree_kernel.py zero git diff; recompute is OUR kernel; sitecustomize bridge + flag + container all
torn down (no leaks); MemAvailable 106-112 GiB every boot. HEAD=e428db3a.
