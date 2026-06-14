# FR13 — Oracle-frame decision package (VERIFY + RECOMMEND, no close/pass-fail)

Date 2026-06-14 UTC. Read-only verify pass on top of the committed research
`FR13_PLUS2_NOT_WALL_ORACLE_FRAME_BIND.md` (ec342d86, verify holds=True) and the
VERIFIED-TO-EXIST `output/fr13_verify_decisive/q1_summary.json`. This doc is the
USER-DECISION package: it does **not** adopt a frame, re-baseline, or modify the
instrument. A GPU workflow (cat3w) is running concurrently; no kernel/patcher
touched. **VERDICT holds=True on the chunked-vs-recurrent claim (proven by exact
code below); the reward-hack red-team is CONDITIONAL — the reframe is legitimate
ONLY under a discipline the user must authorize (re-score BOTH arms; adopt
regardless of whether our number improves; do not declare "+2 was frame" until a
full-stream re-score, not a single row).**

---

## 1. The finding (one paragraph)

The cat9 22-flip / our-pure-spine 5 / native 3 frontier is measured by the binding
flip instrument `scripts/fr13_oracle_stream_teacher_force.py`, which re-prefills the
prefix per served position (`prompt_ids + served_ids[:i]`, `max_tokens=1`). On
Qwen3-Next this routes the served-prefix GDN recurrent state through the **CHUNKED**
scan (`chunk_gated_delta_rule`, block-parallel WY). The live tree-verify builds the
SAME recurrent state through a **rank-1 SEQUENTIAL roll** (`fused_sigmoid_gating`,
spec path). Two different *algorithms* (not two codegens) for one recurrent state →
an fp-nonassociative chunk-vs-recurrent ULP gap. q1 (boot4 flip call6 row6 pos849,
input byte-exact) measures this directly at L0 GDN: **0.0078125 vs the chunked oracle
but 0.0008544921875 (~1 bf16 ULP, at the per-op floor) vs the TRUE per-path RECURRENT
non-MTP oracle = 9.14×.** So ~89% of the L0 carrier magnitude is a chunk-vs-recurrent
**measurement FRAME**, not an op defect. The directive's own anchor —
**"oracle = no-spec NOT prefill"** — is therefore VIOLATED by the current instrument:
it implements the no-spec oracle via per-position chunked **prefill**, exactly the
path the directive names against.

---

## 2. Instrument IS chunked — exact code path (CONFIRMED)

**Instrument** (`scripts/fr13_oracle_stream_teacher_force.py`):
- `_force_one()` L82-93: POST `/v1/completions` `{"prompt": context_ids,
  "max_tokens": 1, "temperature": 0.0, "vllm_xargs": {"fr10_decode_mode": mode}}`.
- loop L165-166: `context_ids = prompt_ids + served_ids[:i]` — a **fresh full-context
  request per served position**. One `/reset_prefix_cache` before the loop (L146).

**vLLM dispatch** (`/tmp/vllm_live_019/vllm/model_executor/layers/mamba/gdn_linear_attn.py`,
`_forward_core`):
- L1142 `if attn_metadata.num_prefills > 0:` → `self.chunk_gated_delta_rule(...)`
  (L1148) = **CHUNKED** (block-parallel WY).
- L1163 `elif attn_metadata.num_decodes > 0:` → `fused_sigmoid_gating_delta_rule_update`
  (L1165) = **RECURRENT** (sequential rank-1 roll).
- Non-spec decode fast path `_forward_core_decode_non_spec` L1261-1307:
  `causal_conv1d_update` (L1285) + `fused_recurrent_gated_delta_rule_packed_decode`
  (L1295) = single-step **RECURRENT** roll.

**The prefill/decode split is by query_len only.**
`v1/attention/backends/gdn_attn.py` L213-216: `split_decodes_and_prefills(m,
decode_threshold=1)`. `v1/attention/backends/utils.py` L524-529: a request is a
*decode* only when `max_query_len <= 1`; otherwise (L531-534, beyond) it is a prefill.

**Why every teacher-force position is a prefill (the linchpin — verified):**
`Qwen3NextForCausalLM` (`models/qwen3_next.py` L691-698) inherits `IsHybrid` but
**NOT** `SupportsMambaPrefixCaching`. So `interfaces.py` L971 `getattr(model,
"supports_mamba_prefix_caching", False)` = **False** → `models/config.py` L439/L451
forces `mamba_cache_mode = "align"` (block-granular). L717-720 of qwen3_next.py even
hard-`raise`s if `all` is requested. In `align` mode the recurrent SSM state is only
cached at **block granularity** (`MambaManager.find_longest_cache_hit`,
`single_type_kv_cache_manager.py` L778-824: a prefix hit must be block-aligned,
L810-814 skips non-block-aligned hits). A teacher-force request for `prompt+served[:i]`
has an **uncached partial-block suffix** for nearly every `i`, so `query_len >> 1` →
`num_prefills > 0` → the served[:i] state is rebuilt through `chunk_gated_delta_rule`.
=> **The instrument IS chunked.** The live tree arm instead builds spine state via the
SPEC recurrent path (gdn_linear_attn.py L1117-1137, `IS_SPEC_DECODING`).

**Recurrent kernel body** (the algorithm that chunked does NOT reproduce bit-for-bit),
`fla/ops/fused_sigmoid_gating.py` L136-168: `for i_t in range(0, T)` sequential roll,
decay `b_h *= exp(b_g)`, rank-1 update `b_h += b_v[:,None]*b_k[None,:]`,
`b_o = sum(b_h*b_q)`. Block-parallel WY (`chunk_gated_delta_rule`) computes the same
ℝ-valued recurrence with a **different reduction order** → bit-different (Yang
2406.06484: ℝ-equal, not bit-exact). **class-10** discriminator: these are not two
SASS realizations of one body — they are **different algorithms**; any "bit-exact by
re-execution" claim across them is invalid (playbook row 10).

---

## 3. Which oracle is deployment-correct (CONFIRMED, with one asymmetry)

When vLLM serves **without speculation**, every generated token is a single-token
step: `query_len == 1` → `split_decodes_and_prefills` returns it as a DECODE
(decode_threshold=1) → gdn_linear_attn.py L1163 `elif num_decodes>0` →
`fused_sigmoid_gating_delta_rule_update` (or `_forward_core_decode_non_spec`
L1261-1307: `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`),
the **RECURRENT single-step roll**. So at no-spec decode the served GDN state is built
recurrently and is **never re-chunked**. The teacher-force oracle, by re-prefilling
`prompt+served[:i]` as a fresh multi-token prefill, routes that same state through the
CHUNKED kernel — a different reduction order than deployment uses.

**=> CONFIRMED: the chunked re-prefill is NOT what deployment does at decode; it is a
stricter-than-reality proxy** that adds a chunk-vs-recurrent realization gap on top of
any real defect. The lossless reference that matches no-spec deployment = **recurrent
no-spec decode.**

**Asymmetry the reframe must NOT hide (deployment-faithful nuance):** the LIVE TREE
arm builds spine state via the **SPEC** recurrent kernel (`fused_sigmoid_gating`,
`IS_SPEC_DECODING`, L1117-1137), which is a **third** realization — same recurrence
algebra as the non-spec packed decode, ~1 ULP apart, and itself distinct from chunked.
So "recurrent" is two kernels (spec vs packed-decode), both ≈1 ULP, both ≠ chunked.
The deployment-correct lossless reference for a *no-spec serve* is the **packed-decode
recurrent** kernel; the tree's serve is the **spec recurrent** kernel. They agree to
~1 ULP (within self-noise) but are not byte-identical. The decision must pick the
no-spec packed-decode recurrent as the lossless reference (that is what "deploy without
spec" produces), and accept that the tree arm is scored against it within the
self-noise floor — the same within-floor bar the user already set
(`project_fr13_active_worker_codex_fr15`, FR13 bar = per-depth-argmax + within-floor).

---

## 4. HARD reward-hack red-team (the decisive section)

**The question:** is switching the flip oracle from chunked-re-prefill to
recurrent-no-spec-decode a LEGITIMATE deployment-correct correction, or a banned
reframe-to-pass?

**Distinguish from the banned pattern.** `feedback_no_reroute_reward_hacking`: the
banned hack is changing **what computes the deliverable** — routing path0 through
native's `causal_conv1d_update`/FLA so a parity metric drops while OUR kernel is
unchanged/wrong. That is NOT what is on the table here. The deliverable kernel (our
tree-scan) is untouched. What changes is **what we measure against** (the reference
oracle). That can STILL be a hack — the note's spirit and the user's "softer reference
chosen to hide a real defect" — so the deliverable-vs-reference distinction does NOT by
itself exonerate the reframe. The decisive criterion is the user's own:

> **LEGITIMATE iff the recurrent oracle measures what DEPLOYMENT actually does AND
> native is re-scored vs the SAME oracle AND the reframe is adopted whether or not OUR
> number improves. HACK iff the recurrent oracle is applied selectively to soften our
> count, or chosen because it hides a real ours-defect.**

**Criterion part A — does it measure deployment? YES (proven §3).** Recurrent no-spec
decode IS the kernel deployment runs without speculation. So the *reference* is
deployment-correct by construction. This part is settled by code, not by our numbers.

**Criterion part B — does native re-scored vs the SAME oracle also collapse, or does
the gap persist?** THIS IS THE UNSETTLED, DECISIVE EMPIRICAL QUESTION, and the honest
answer from the evidence is **MIXED — we must NOT claim "+2 was frame" yet:**

- **Frame component is REAL and partly removes our inflation (legitimate):** q1 proves
  the L0 carrier magnitude is ~89% chunk-vs-recurrent. Re-scored vs the recurrent
  oracle, our-spine L0 drops to ~1 ULP. Native E5's gap vs chunked is already smaller
  (its MTP-5 verify lives in the SAME kernel family vLLM tunes for prefill/decode),
  so re-scoring native vs recurrent moves native LESS than it moves ours. **If the gap
  is purely L0-input-frame, the re-score shrinks our-spine 5 toward native 3.** This is
  the *predicted* direction and would make the reframe a clean deployment-correct
  correction.

- **But the q1 deep-row flip PERSISTS vs the recurrent oracle (NOT erased):**
  q1_summary.json — vs the recurrent oracle the divergence STILL accumulates diffusely
  (L30~0.12, L58~0.19), STILL explodes at deep full-attn (L59~1.3, L63~33.75,
  final_norm max_abs 3.125 cos 0.986), and STILL **flips the deep-row argmax (recurrent
  oracle=3425 vs tree=1970).** For THIS carrier the flip is carried by **diffuse fp
  accumulation that is real vs the non-MTP ground truth** — the recurrent oracle does
  NOT hide it. So the reframe legitimately removes an inflated L0 magnitude but is
  **NOT guaranteed to collapse our-spine 5 → native 3.**

- **Independent proof the chunked oracle does NOT blanket-hide loss (kills the "softer
  reference" worry at the high-margin end):** `FR13_GOLD_MARGIN_BIND.md` L50-64 — using
  the *same* `max_tokens=1` chunked teacher-force (`fr13_gold_margin_probe.py` L411-424,
  identical pattern), 2 of 4 forks are **large-margin tree-COMMITTER serve deviations**
  (p2 gap 2.125, p3 gap 5.125) where BOTH backends agree on the clean argmax and the
  tree serves a rank-2 token its own clean greedy clearly rejects. A 5-logprob swing is
  a **logic/commit-path defect, NOT a chunk-vs-recurrent ULP** — it would survive ANY
  oracle reframe. So the recurrent oracle cannot be a blanket softener: real defects
  remain real under it. The reframe only removes the ~1-ULP inflation on the **near-tie
  band**, exactly where fp-frame mismatch should and does dominate.

**VERDICT (reward-hack red-team):** the reframe is **LEGITIMATE-BY-CONSTRUCTION ONLY
UNDER DISCIPLINE**, not unconditionally. It is deployment-correct (part A proven). It
is NOT a blanket softener (gold-margin proves real defects survive it; q1 proves the
deep-row flip survives it). It becomes a **HACK** if any of: (i) it is applied to our
arm but native is not re-scored vs the identical recurrent oracle; (ii) "+2 was frame"
is declared from a single row instead of the full-stream re-score; (iii) it is adopted
*because* our number improved rather than because it is what deployment does.
**Class-12 discipline (playbook row 12, measurement traps):** re-score BOTH arms, do
NOT conclude "+2 was frame" until the full-stream (not single-row) re-score shows the
gap shrinks. The honest state today: **part A is settled (legitimate reference); part B
is OPEN** — the gap MAY shrink (frame) or MAY persist (a real diffuse defect the
recurrent oracle would NOT hide). The re-score is genuinely needed to settle it; it is
not a foregone conclusion.

---

## 5. Feasibility — can a recurrent no-spec-DECODE oracle be captured for a
full-stream flip re-score?

The streamed-off-by-one note and the chunked-instrument choice are **causally linked**
(directive's two notes ARE related): per `FR13_GOLD_MARGIN_BIND.md` L29-37 the TREE
arm's live-streamed `top_logprobs` is misaligned at spec-decode accept positions
(~12/128: served-token != reported-argmax) — a vLLM logprob-REPORTING quirk on the
recurrent spec-decode serve stream. Distrusting that stream, the author switched to
CLEAN `max_tokens=1` teacher-force on the byte-identical prefix (L35-36). That is
correct for a clean per-position distribution, but its side effect is rebuilding GDN
state via the CHUNKED prefill (query_len>1) instead of the recurrent roll — trading an
off-by-one **alignment** bug for a chunk-vs-recurrent **frame** mismatch. **The
off-by-one is an alignment bug in stream-READING, not an intrinsic blocker to a
recurrent oracle.** Options:

- **(a) single streaming no-spec decode with corrected off-by-one alignment.**
  Cheapest IF the off-by-one in the stream reader can be pinned. BLOCKER: the
  streamed-logprob quirk is in vLLM's reporting and was explicitly distrusted; fragile.

- **(b) sequential single-token no-spec DECODE, RECOMMENDED.** Drive the server
  forcing the NON-spec regular-decode path (`fr10_decode_mode` non-spec) and feed
  served tokens one at a time so each step has `query_len == 1` → `num_decodes>0` →
  recurrent roll (gdn_linear_attn.py L1163 / `_forward_core_decode_non_spec`). Read each
  step's `max_tokens=1` distribution directly per step (no streamed-array indexing → no
  off-by-one). This reproduces exactly the no-spec deployment kernel and gives
  per-position argmax+top-k for the full-stream re-score. **BLOCKER / cost:** vLLM
  continuous-batching does not natively expose "advance one token and return the
  decode-step distribution with **persisted recurrent state**" via `/v1/completions` —
  a single `/v1/completions` for `prompt+served[:i]` re-prefills (chunked), because
  (i) Qwen3-Next lacks `SupportsMambaPrefixCaching` so cross-request recurrent state is
  not preserved at sub-block granularity (§2), and (ii) the API has no "resume from
  saved SSM state" hook. So a TRUE recurrent re-score needs an **in-process harness**
  (load the model, run a genuine single-step decode loop over served_ids carrying
  `ssm_state`/`conv_state` forward) — NOT an HTTP `/v1/completions` loop.

- **(c) the q1 per-path recurrent oracle generalized to every position.** q1 already
  built a per-path recurrent oracle for ONE row (boot4 row6 pos849) via the
  prefill_gdn_state_replay-style capture; scaling it to every served position is the
  straightforward (GPU-heavy) path and needs NO API change. This is essentially (b)
  done as an in-process replay.

**Cleanest + blocker:** **(b)/(c) in-process recurrent single-step loop** is the robust
recommendation — it is byte-faithful to no-spec deployment and sidesteps the off-by-one
entirely (read each step directly, no streamed array). The blocker is that it cannot be
an HTTP re-prefill loop; it must be an in-process model harness carrying recurrent state
forward. Option (a) is cheapest but rests on the distrusted streamed-logprob alignment.

---

## 6. Recommendation (single; NO close/pass-fail)

Build the **in-process recurrent no-spec single-step DECODE oracle** (option b/c:
load model once, decode served_ids one token at a time carrying `ssm_state`/`conv_state`
forward, read each step's distribution directly) and re-score **BOTH arms** (our pure
spine AND native E5, AND cat9) against it for the **full stream** — not a single row.
This is the only instrument that (i) measures what no-spec deployment actually does,
(ii) sidesteps the off-by-one, and (iii) re-scores native vs the identical oracle so
the part-B question (does the native-vs-ours gap shrink to ~0 = frame, or persist =
real defect) is answered empirically. Adopt the recurrent frame as the lossless flip
oracle **iff** that re-score is run on both arms and the frame is adopted regardless of
whether our count improves. Until that re-score exists, do NOT re-state the frontier
numbers (5/3/22) as "frame-corrected" and do NOT retire the +2 — the deep-row flip and
the gold-margin committer defects show real, non-frame loss remains in the mix.

---

## 7. THE PRECISE QUESTION FOR THE USER

Two coupled decisions:

1. **Oracle frame (lineage-affecting):** The binding flip oracle
   (`fr13_oracle_stream_teacher_force.py`) measures every flip count (cat9 22 /
   our-spine 5 / native 3) against a **chunked per-position re-prefill**, which VIOLATES
   the directive's own anchor "oracle = no-spec NOT prefill" and inflates the near-tie
   band by a ~9× chunk-vs-recurrent ULP frame (q1-proven). **Should the lossless flip
   oracle be switched to the recurrent no-spec DECODE — the kernel deployment actually
   runs without speculation?** This is a candidate **lineage change**: it would re-frame
   ALL flip numbers (native, our-spine, cat9) and per the GDN-kernel-lineage policy it
   STOPS for your ruling before adoption.

2. **Build the instrument?** **Should I build the in-process recurrent no-spec
   single-step decode oracle (option b/c) and re-score BOTH arms + cat9 on the full
   stream** to settle empirically whether the native-vs-ours gap **shrinks to ~0 (the
   +2 was a frame artifact → reframe is a clean deployment-correct correction)** or
   **persists (ours has a real diffuse defect the recurrent oracle does NOT hide →
   reframe legitimate but does not retire the +2)**? The reframe is legitimate ONLY if
   this both-arm re-score is run and adopted regardless of whether our number improves;
   adopting it on our arm alone, or declaring "+2 was frame" from the single q1 row,
   would be a class-12 measurement trap / soft-reference hack.

(My recommendation: YES to building the both-arm in-process recurrent re-score
[decision 2] BEFORE ruling on decision 1 — the re-score is what makes the lineage change
defensible-or-refuted; the q1 deep-row flip and the gold-margin p2/p3 committer defects
prove the answer is not foregone.)

---

## Relevant files (absolute)
- Instrument: `/home/mark/shared/lumoFlyWheel/scripts/fr13_oracle_stream_teacher_force.py`
- Same-pattern probe that caught the REAL defect:
  `/home/mark/shared/lumoFlyWheel/scripts/fr13_gold_margin_probe.py`
- vLLM GDN dispatch:
  `/tmp/vllm_live_019/vllm/model_executor/layers/mamba/gdn_linear_attn.py`
  (L1102-1198 prefill/decode split; L1261-1307 non-spec packed-decode recurrent)
- query_len split: `/tmp/vllm_live_019/vllm/v1/attention/backends/gdn_attn.py`
  (L213-216); `.../v1/attention/backends/utils.py` (L489-534)
- recurrent kernel body:
  `/tmp/vllm_live_019/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`
  (L136-168)
- mamba prefix-caching gate:
  `/tmp/vllm_live_019/vllm/model_executor/models/qwen3_next.py` (L691-720);
  `.../model_executor/models/interfaces.py` (L946-971);
  `.../model_executor/models/config.py` (L436-472);
  `.../v1/core/single_type_kv_cache_manager.py` (L763-824 align block-granular hit)
- q1 evidence: `/home/mark/shared/lumoFlyWheel/output/fr13_verify_decisive/q1_summary.json`
- corroboration: `/home/mark/shared/lumoFlyWheel/FR13_GOLD_MARGIN_BIND.md` (L29-64)
- prior research: `/home/mark/shared/lumoFlyWheel/FR13_PLUS2_NOT_WALL_ORACLE_FRAME_BIND.md`
