# FR13 — oracle-frame CLOSED: recurrent re-score = ours-only reward-hack signature, KEEP chunked, flips are REAL

Date 2026-06-14. GPU workflow `wf_44100bad-1f3` (task wd0ok777i), rescore.ok=True. Raw:
`research/fr13_workflows/recurrent_rescore_wd0ok777i.raw.json`. Resolves the candidate lineage change
(FR13_PLUS2_NOT_WALL_ORACLE_FRAME_BIND / FR13_ORACLE_FRAME_DECISION).

## Result: both-arm recurrent re-score (deployment-correct in-process single-step decode oracle)
| arm | chunked | recurrent |
|---|---|---|
| native E5 | 3 | **3** (unchanged) |
| spine chain5 | 5 | **5** (byte-identical) |
| cat9 | 22 | **20** (−2, ours-only) |

The recurrent oracle is genuine (forced single-token greedy decode, query_len==1 → recurrent rank-1 roll,
conv/ssm carried in KV cache; recurrent_decode_calls native/spine 48768, cat9 43968; class-8 det True;
reproduces the q1 pinned deep-row flip exactly: p2 pos21 served 1970 vs recurrent argmax 3425).

## Verdict: REWARD-HACK SIGNATURE → do NOT adopt recurrent; the +2 (and the frontier) is a REAL ours-defect
Per the user's adopt criterion (both-arms-consistent = frame; ours-only = suspect): native + spine are
EXACTLY frame-invariant; only cat9 dropped 2 (non-proportional). Adopting recurrent would shave 2 of OUR
flips while leaving the reference arms untouched = partly hiding an ours-defect. The q1 deep-row flip
SURVIVES the recurrent reframe with the same clear margin. So the chunk-vs-recurrent gap is NOT the
flip-count carrier (q1's 9.14x L0-magnitude reduction does not translate to e2e flips — the diffuse
accumulation crosses the same high-entropy boundaries). **KEEP the chunked oracle** (instrument valid; the
directive's "oracle=no-spec NOT prefill" matters for L0 magnitude, not the binding flip count). The q1
single-row finding was a red herring at the e2e level.

## Disposition
Oracle front CLOSED as a no-op. cat9 22 = native-floor 3 + REAL +2 spine (our-kernel-vs-native realization,
small, alignment territory) + REAL +17 leaf co-residency. The +17 is the sole large target → now LOCALIZED
to the bf16 in_proj_ba GEMM (FR13_WIDTH_CARRIER_INPROJ_BA_BIND.md). Pairs with
[[reference_scalar_metric_per_token_blindspot]], [[feedback_check_artifact_before_concluding]],
[[feedback_research_before_deadend]] (research dissolved the WY-wall but the re-score then confirmed the
defect is real, not frame — both were needed).
