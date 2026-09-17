# vLLM upstream plan v3 — "close GDN and quantized-serving defects; keep trees as a record"

Converged 2026-09-10 between Claude and Codex (gpt-6-astra high), rounds
1–4, artifact red-team round 1 applied; full Codex texts in the session
scratch (`impact_plan_codex.md`, `impact_plan_codex_r2.md`,
`artifacts_r1_report.md`). Supersedes EXECUTION_PLAN.md (v1/v2).

## Why
benchislett asked that DDTree remain a draft in his May 27 reply on
[#42910](https://github.com/vllm-project/vllm/pull/42910#issuecomment-4555281194).
#42121 removed unsupported tree-attention machinery; #42449's refactor
proposal remains open. The later #46105 tracker welcomed custom-mask kernel
integration, including Hopper XQA through FlashInfer, and is now closed;
that is a qualified integration signal, not approval of a complete tree
feature. #54080 is our agreed public design home.
Our existing evidence does not establish a general tree throughput win:
historical results need the sampling erratum, and whole-region capture
showed no detectable additional gain after prerequisite host work.
Our near-term bet is to close specific GDN and quantized-serving defects
through an owned patch, a focused review, and one controlled reproduction.

## Ranked plays (cost class)
1. **P11 — #53651 to a review decision**: zero-code audit of the residual
   diff and existing tests; rewrite the description, then ONE status ping.
   Any needed code/test correction is a separate new-code funding decision;
   an additional GPU smoke requires its own run budget.
   `PR_53651_DESCRIPTION_v2.md`.
2. **P3 — review #53798** (zero-code; their tests run locally on the GB10:
   56 passed at `af5357c2b`; merges clean onto main): `REVIEW_53798_DRAFT.md`.
   #54076 only with a distinct finding (its author reports v0.29.0
   normalization may make it a no-op for some configs).
3. **P5 — one funded controlled GB10 reproduction of #54928**: GPU-time
   only if existing scripts and API outputs suffice; any new harness or
   instrumentation is new-code. `EXPERIMENT_CARD_54928.md`.
   Next-ranked alternative: **P4′**, one bounded missing case with jschmied.
   Choose one; neither is funded by listing it here.
4. **P1 — #55688 question** (zero-code): `COMMENT_55688_DRAFT.md` v2, GO.
5. **P7 — status ping** (zero-code): after P11; counts with it.
6. **P8 — coverage/oracle investigation** (unfunded zero-code investigation;
   any resulting test or instrumentation is new-code): upstream's
   prefix-cache parity helpers compare outputs with tolerance; kernel-level
   precopy tests are zero-tolerance; whether a state-level gap exists under
   heterogeneous geometry (#53142/#54076 cluster) is an investigation, not a
   test spec.
7. **P2 — #54080 design addendum** (zero-code): `COMMENT_54080_ADDENDUM.md`;
   ends with one question to benchislett. Then `SLACK_CONTRIBUTORS_POINTER.md`.
8. P4 broad testing is not scheduled. P10 arXiv keeps its independent,
   separately chosen editorial budget. P6's fork-anatomy note and P9's
   custom-mask integration remain deferred.

## Not filed
New tree RFC (contingency title only: "[RFC]: Tree verification and
accepted-state publication for GDN hybrids"); Phase-0 draft PR
(`feat/phase0-tree-state-interfaces` stays local); fixtures port (campaign
fixtures are bound to our endpoint/kernels); comments on #42910/#40809/
#42449; the retired `COMMENT_54080_DRAFT.md` v8 and `COMMENT_54928_DRAFT.md`.

## Rules
- Six distinct threads in 30 days: #53651, #55688, #53798, #54080, one of
  #54928|#55122, one reserve (#54076 or a P8 PR). Two unsolicited first
  contacts per rolling 7 days. Two active workstreams.
- Pre-commit eligibility: `ready`, `ready-run-all-tests`, or `verified`,
  or at least four merged PRs. This is not the full-CI policy; one merge is
  evidence, not an automatic unlock.
- Every public artifact: Claude drafts → Codex red-team → Mark reviews →
  Mark posts. Funding a run ≠ approving its report. Public numbers trace to
  the site and its errata; never "43.57 vs ~43.7 = 0.92×"; "0.087
  tok/event" is not public.

## 30-day sequence (Sep 10 – Oct 9)
- d1: P1, P11 (+ping), P3, P2 addendum POSTED 2026-09-10 (see file headers); Slack pointer ready. d8–14:
  P2 + Slack; choose P5|P4′ and approve its card. d15–21: the one funded
  run (half-day + one more only for a named uncertainty). d22–26: respond;
  reserve-slot decision. d27–30: ledger; Mark decides next month.

## Metrics (private ledger, weekly)
#53651 review decision; external reviews the author acts on;
reproductions that resolve a stated uncertainty; distinct maintainer
decisions; cost (sessions, GPU time, threads, open commitments).

## Phase 2 — days 2–30 (Sep 11 – Oct 9). Authored by Codex, red-teamed by Claude, converged 2026-09-11.

Principle: do not wait for replies to do useful work. Convert what is posted
into reproducible evidence and mergeable changes. Success = a merged fix, an
author acting on a review, or a resolved maintainer question.

ACT-NOW, in priority order:
1. **#54928 matrix → evidence audit → report** (GPU-time funded; analysis
   in-scope, no extra GO). Analyzer corrections applied (all repetitions,
   per-token ID verification, tied margin = 0, failures reported). Draft the
   report Sep 11 once the audit is complete; one independent (Codex) review
   Sep 11–12; Mark GO; publish on #54928 (evidence reports are exempt from
   the ask quota). Per outcome: reproduces → publish first-divergence
   evidence, then fund ONE discriminator; non-reproduction → publish exact
   scope, not equivalence; unstable → preserve within/between-launch
   variation; unattributable → report the limit, never infer acceptance
   failure. E = unique V ≠ A localizes a ranking discrepancy, not its cause.
2. **#53651: own the missing re-tying regression** — DONE 2026-09-11 (38f7bcef2, ~40 min) (new-code, ≤2 h cap —
   MARK GO). `tests/model_executor/model_loader/test_weight_tying.py` at
   main already holds three CPU tests; add `test_excluded_lm_head_is_retied`
   (quant_method = `UnquantizedLinearMethod()`), fail-before/pass-after with
   the existing controls. Push to our PR (Mark GO per update). Inspect
   #55494's composition privately; if it lands first, rebase/reconcile
   (compatibility, not activity).
3. **P8 investigation** (zero-code, half-day of reading — credit is Mark's
   call). Pin main; trace scheduler boundary → stored state → cache hash →
   resumed request across the heterogeneous-geometry cluster; inventory
   assertions and reachability. Deliver one missing invariant with its
   smallest upstream test location, or a documented non-gap. A test is a
   separate bounded new-code GO. #54076 review only with a distinct finding.
4. **P4′ prep** (zero-code reading). Inspect #55122's kernel suite for one
   uncovered GB10 case; dense targets cannot validate MoE-finalize #54948.
   One concrete offer only if it resolves missing evidence (Mark GO before
   contact); a run needs a named case + ~2 h GPU GO.

WAIT only at: #53651/#53798 review decisions (monitor; no status-only ping
this week); #55688/#54080 answers (monitor; no tree port/interface/custom-
mask project without a consumer + separate GO); P4′ execution (free GPU,
distinct case, run GO). Answered-thread branch: within 48 h of a
substantive reply, bring Mark the reply, a proposed response, and any new
scope/budget decision; private triage and funded work continue; only
unfunded implementation/GPU waits. Every public response needs Mark GO.

Rules (replace Plan v3 §Rules): drop the six-thread ceiling — count
outstanding promises; ONE new unsolicited ask per week (evidence reports and
substantive reviews exempt); day-1 overrun acknowledged — no further
unsolicited ask before Sep 17; one author + one independent review for
substantive artifacts (routine factual replies: Mark's review only, to
conserve credit); Mark GO per public item; isolation, one GPU workload,
immutable evidence, funding ≠ publication; two active deliverables (one
measurement, one code/review).

Calendar: Sep 11 audit matrix + draft report + specify re-tying test →
Sep 12 implement regression (if funded) → Sep 13 validate, prepare PR
update → Sep 14 P8 inventory → Sep 15–16 resolve feedback/composition →
Sep 17 score outcomes; choose one P4′ case or a P8 regression → Sep 18–24
finish the selected contribution → Sep 25–Oct 1 pursue merge/review
decisions → Oct 2–9 close commitments, record outcomes; no new speculative
project.

### Phase 2 items 3–4 — findings (2026-09-11, agents read-only; Codex independent review; Mark decides)
- **P8 (item 3): gap confirmed.** Unit mismatch verified at main 8359e15a
  (core.py:345–349 resets the global block size; mamba_hybrid.py:121–122
  seeds in global units; mamba_utils.py:534–545 advances in spec units). No
  unequal-geometry admission→preprocess→state-content test exists upstream
  or in #54076/#53798/#55688/#53803. Agent report corrected by Codex
  (#53803 binds the same spec size, not a third divisor; #55688's
  mamba_block_size is FlashInfer-ReplaySSM-only; a tensor oracle cannot
  distinguish equal divisors). Concrete test spec + ask: P8_REVIEW_codex.md.
  Funding ask: ≤1 engineering day incl. ≤1 h exclusive GPU after #54928.
- **P4′ (item 4): one zero-code review finding GO** (#55122 head 7cfd04a3:
  RADIX_THRESHOLD=22016 but test_persistent_topk_path_transition still
  brackets 16384 — exact text in P4PRIME_REVIEW_codex.md; needs Mark posting
  GO). The GB10 low-smem fallback case is real but the agent's coverage and
  exclusivity claims were overstated and its script is unfit; corrected
  ≤2 h feasibility/validation ask + replacement offer text in the review.

### 2026-09-11 23:18 UTC — Mark ruled on the four Phase-2 decisions: 1 GO, 2 FUND, 3 GO, 4 FUND
- #54928 report POSTED: https://github.com/vllm-project/vllm/issues/54928#issuecomment-5641733441 (thread 5 of the month).
- #55122 review finding POSTED (Comment): https://github.com/vllm-project/vllm/pull/55122#pullrequestreview-5184175591 (substantive review, quota-exempt).
- P8 regression test FUNDED (≤1 day incl. ≤1 h GPU): Opus agent on branch p8-restore-fidelity off pr-53798; spec = P8_REVIEW_codex.md; no push, no PR — Codex review + Mark GO before any submission.
- P4′ kernel check FUNDED (≤2 h): Opus agent in /home/mark/shared/p4prime-55122; protocol = P4PRIME_REVIEW_codex.md; result recorded only; offer/comment needs Codex review + Mark GO.

### 2026-09-12 — funded work delivered (both under budget; Codex GO on both artifacts)
- **P4′ (#55122 low-smem fallback on GB10): POSITIVE.** Routing window measured (n ∈ [355588, 474112] at vec_size 4), branch entry proven, 324 fallback launches + 108 cooperative controls exact and repeatable, 18 expected >64-CTA rejections. Artifacts results/upstream/55122 @a4c38b9c1. Result comment (Codex text, 137 words): COMMENT_55122_RESULT_DRAFT.md — awaiting Mark GO.
- **P8 (align-mode restore-fidelity regression): GO as an artifact.** Commit ca1d410ae on vllm-head branch p8-restore-fidelity (off #53798 head af5357c2b), test-only +358/−1; Codex independently re-ran fail-before/pass-after. Trim pass in progress (one layer, shorter docstrings, consolidated negative hooks, framing fix). #55688 finding: its ordinary align path still seeds with cache_config.block_size — inherited defect, closed by #53798 or an equivalent spec-unit fix; not a new ReplaySSM regression. Routes (Mark decides): 1 (preferred) offer the test commit + evidence to ptorsten on #53798; 2 fallback test-only PR after the seed fix merges; 3 at most a dependency/rebase pointer on #55688.

### 2026-09-12 08:45 UTC — first upstream reply: jschmied on #55122
- Accepted our review finding ("Good catch, and it was our own doing"), fixed in a7188289e (seq_len 22015/22016/22017), verified on his GB10 (old widths exercised 1 path, new widths 2), and folded our rows-64/FilteredTopK note into the docstring. **Metric: author acted on our review.**
- His second comment (to the other reviewer, LopezCastroRoberto, re opt-in backend #55872): the defect is unreachable in his 16k-context census; he already runs the deterministic kernel opt-in out-of-tree; on GB10 the deterministic path hard-fails at ~100k context (dynamic smem 98080 > 97120) in HIS wrapper; routing note: long rows on GB10 fall to top_k_per_row_decode only when three conditions hold (row > RADIX_THRESHOLD, cooperative oversubscription, optin < 128 KiB).
- Relevance: our P4′ result (force_single_cta fallback exact/repeatable at 355588–474112 on GB10, pinned 7cfd04a3) speaks directly to that routing discussion. Result comment pending Mark GO; bridge sentence to be Codex-checked.

### 2026-09-13 18:34 UTC — jschmied acknowledged the P4′ result on #55122
Quoted: "the case we could argue for but not demonstrate … says more about the fallback than anything in the PR body. Publishing the harness and source hashes is what makes it checkable, and your scope note is right." PR rebased to ef5d2d953 with kernel sources byte-identical (our pinned result still applies). **Metric: evidence used / acknowledged by the author (second confirmed engagement on #55122).** No reply needed; no action.

### 2026-09-17 — day-7 checkpoint
- Upstream quiet since Sep 15 on all six threads; three of four target PRs now `needs-rebase`; scoreboard unchanged (1 review acted on + 1 evidence acknowledged on #55122; 1 reproduction on #54928; 1 test offered on #53798; 0 merges).
- #53651 second route POSTED in Slack #pr-reviews (sanctioned 7-day ping; Codex-corrected text — format checks are gated, not green): https://vllm-dev.slack.com/archives/C07QT0LUF4K/p1789603743165569
- Slack watch re-armed at 3-hour cadence (Mark's decision). Contribution scan running (Codex + independent Opus agent) for the next bounded play.
