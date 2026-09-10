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
2. **#53651: own the missing re-tying regression** (new-code, ≤2 h cap —
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
