# FR13 — the +2 spine floor is NOT a WY-wall; it is largely a CHUNKED-vs-RECURRENT oracle-frame artifact (candidate lineage change)

Date 2026-06-14. CPU research workflow `wf_843ad889-4e3` (task w1ee7q0q0), **verify holds=True**. Raw:
`research/fr13_workflows/spine_floor_plus2_wf_843ad889.raw.json`. Doc: FR13_SPINE_FLOOR_PLUS2_RESEARCH.md
(c368bc5f). Research-before-deadend DISSOLVED the candidate reshape wall.

## Finding 1: the +2 (our pure spine 5 vs native 3) is the GDN scan STATE-FEED, measured vs the WRONG oracle
Banked, VERIFIED-TO-EXIST q1 evidence (`output/fr13_verify_decisive/q1_recur_vs_chunked.json` +
q1_summary.json, boot4 flip call6 row6 pos849, input_hidden byte-exact 0.0):
- L0 GDN first-nonzero = **0.0078125 vs the CHUNKED-prefill oracle** but only **0.0008544921875 (~1 bf16
  ULP, at floor) vs the TRUE per-path RECURRENT (non-MTP) oracle = 9.14× ratio**.
- ~89% of the chunked gap is **chunk-vs-recurrent state-construction frame**, not op-divergence.
Mechanism: the live tree-verify builds the spine state via a rank-1 SEQUENTIAL tree-scan (recurrent); the
binding flip oracle `fr13_oracle_stream_teacher_force.py` re-prefills the prefix per position (max_tokens=1)
= the CHUNKED scan (chunk_gated_delta_rule). Two realizations of one recurrent state = the documented
chunk-vs-recurrent ULP gap (fp non-associative; Yang 2406.06484 ℝ-equal not bit-exact). Native E5's gap vs
chunked is smaller because its verify is MTP-5's recurrence in the SAME kernel family vLLM tunes for
prefill+decode; ours is the fr10 tree-scan re-index (different Triton codegen/load-order).

## Finding 2 (CANDIDATE LINEAGE CHANGE — verify before adopting): the flip oracle frame
The directive states **"oracle=no-spec NOT prefill."** The binding instrument
`fr13_oracle_stream_teacher_force.py` implements the "no-spec oracle" via **per-position chunked
re-prefill** — which is the prefill path the directive warns against. If the deployment-correct lossless
reference is the **recurrent no-spec DECODE** (single-step state roll, how vLLM actually generates without
speculation), then the entire flip frontier (native 3 / our-spine 5 / cat9 22) is measured against a
mismatched chunked frame, inflating every count by the chunk-vs-recurrent gap. Our tree-verify is ~1 ULP
from the recurrent oracle (q1) → re-scored vs the recurrent oracle, our-spine is PREDICTED to drop toward
native 3. **This is foundational + a lineage-change candidate → NOT adopted unilaterally; it requires (a) a
hard reward-hack red-team [reframe-to-pass vs deployment-correct] and (b) the decisive e2e re-score.**

## Finding 3: FA2 tree-bias is ARITHMETICALLY INERT on a branchless spine (proven, contributes ~0 to +2)
`_prepare_tree_attn_bias(chain5) == lower-triangular causal` EXACTLY (a spine node's only ancestors are the
prior spine nodes = the causal triangle); bias adds 0.0f where ancestor/self (IEEE x+0==x) and -inf above
diag (= stock causal). The residual 2-ULP MMA-grouping floor (project_fr13_fa2_fork_nocopy_floor: 2 ULP in
~983k, 15× below the E5 noise floor, no depth growth) cannot cross 1.1–9.75-nat boundaries. So FA2 ≈ 0 to
the +2. The "FLASH-for-branchless-spine" route is LEGITIMATE (not a reward-hack reroute) but a structural
NULL (bias already inert) → do NOT pursue. L60/L61 = confirmed AMPLIFIER not origin (don't per-layer patch).

## nonWYClosable=TRUE — WY stays correctly parked
WY (chunked-WY kernel) is the WRONG tool: the verify path is RECURRENT, not chunked
([[reference_gdn_verify_sequential_dispatch]]). fp32-state is exhausted (ours already MORE precise than
native bf16). The non-WY levers: (1) recurrent-vs-recurrent scan codegen align (class-10, the conv-bf16-tap
/ scan static_range→tl.range family) — but q1 shows ~1 ULP headroom, so little to gain; (4) **re-frame the
oracle to recurrent** = the likely REAL resolution, pending the reward-hack red-team + e2e re-score.

## NEXT (decisive)
1. cat3w (GPU, running) = the deployable shallow-width test — read as (leaf-co-residency removed?) ×
   (accept recovered?) sitting ON TOP of the +2 frame issue.
2. HARD red-team the oracle reframe (legitimate deployment-correct vs reward-hack) + design the recurrent
   no-spec-DECODE full-stream re-score instrument (CPU, running as `wf` this tick).
3. DECISIVE e2e test: re-score chain5/chain3 (and cat9) flip counts vs the recurrent no-spec DECODE oracle.
   If our-spine → ~3, the +2 is the frame. THEN surface the lineage-change + the oracle-frame decision to
   the user (foundational; affects every flip number). Pairs with [[project_fr13_22flip_carrier_l0gdn]],
   [[reference_scalar_metric_per_token_blindspot]], [[feedback_check_artifact_before_concluding]],
   [[feedback_research_before_deadend]], [[feedback_no_reroute_reward_hacking]],
   [[reference_capture_once_native_pin_prompt]].
