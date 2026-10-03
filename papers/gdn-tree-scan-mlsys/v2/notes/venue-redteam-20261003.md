# Red team and venue decision — 3 October 2026

Inputs: `main.pdf` (15 pp, 2 Oct 23:57 build), Codex paper session (final message 2 Oct 23:59 PDT: "final v2 at its current scope; more experiments only for a stronger lossless-speedup claim"), Claude experiment session `4b42a19b` (final 2 Oct 19:04 PDT: continuation v3 EQUIVALENT; accept walk exact on its inputs; penalty-history defect confirmed; split-K ≈15 ms/step, fused top-k ≈0.8 ms, prep ≈0; +11.7% vs MTP-5 on the disjoint corpus, every run above every run), `MarkResume/eb1a_claim_strategy.md` rev. j and `eb1a_data/research/venues_pb_2026-09-08.md`, live CFP checks today.

## 1. Reviewer-grade findings, ranked

1. **The system loses to the strongest baseline in the fairest comparison.** Table IV: SGLang EAGLE 29.69 vs LumoTree 28.79 pooled tokens/s on exact-token-ID replay. Table III: LumoTree is the slowest arm (26.19 vs 27.17 vs 29.89), the longest (144 vs 99.5 vs 103 min) and resolves fewer tasks than MTP-5 (5 vs 6). The only wins are +10.2%/+11.7% over vLLM's own MTP-5 and a 1.7% edge over one SGLang run with non-matched inputs (Table VII). A top-tier reviewer summarises this as "slower than the open-source baseline, 10% faster than its own stack."
2. **Every measured run contains a known sampling defect that was not fixed.** Section VII-F: flattened penalty history; all deployed/SWE runs use presence penalty 1.0; the fix is a contained patch (Claude's estimate: about half a day including re-qualification and replays). Reviewers will ask why it was disclosed instead of repaired. Worse, it is an unaddressed confound for the task study: LumoTree emitted 187,137 output tokens vs 134,643 for MTP-5 (+39%); a weakened repetition penalty on deeper nodes is a plausible cause of longer, more repetitive outputs, which would inflate tokens/s and explain the longer wall time and lower resolution. The paper does not test this.
3. **Scope: one GPU, one model, one quantisation, batch size 1.** No batch sweep, which is where tree verification cost diverges from chain MTP. MLSys/OSDI reviewers treat this as incomplete; ICPE reviewers accept a scoped study if it is framed as one.
4. **Bole (arXiv 2608.01651, Aug 2026) is the direct competitor and is not measured.** Same hardware class (GB10), same model family (Qwen3.5-27B), SGLang integration, OpenHands replay, 4 models × 2 GPUs, up to 2.03× over the strongest tree baseline. The paper cites it in Related Work only. Expect "compare against Bole" from at least one reviewer.
5. **Tables XIII–XV are net negative.** They show LumoTree's isolated GDN verify+publish at 17.69 ms vs Weaver 3.45 ms and TreeWY 11.01 ms, and 132.5 MiB of buffers vs 25.1 MiB, then disclaim the ranking because the comparators use different precision policies. Readers keep the 5× number and discard the disclaimer. Cut to one paragraph or move to the supplement.
6. **Only one of the five Table II optimisations has an isolated measured effect** (split-K, ≈6.5%). Fused selection and operand prep have one run each; spine-first layout has no ablation; precomputed histories none. The paper itself says the layout effect "remains an ablation question."
7. **The prose argues against itself.** "Descriptive", "does not establish", "not isolated", "approximate" appear in nearly every results paragraph and again in a one-page Limitations section. Reviewers read this as the authors' own low confidence. State each result once, plainly; put every caveat in Limitations once.
8. **Length.** 15 pages, 17 tables. MLSys allows 10 + references; ICPE allows 10 + references and appendices (ACM double-column). Roughly a third must go regardless of venue.

What survives review well: the measurement methodology (pooled completed-request decode-rate estimator with explicit exclusions, two frozen replay corpora, disjoint-task generalisation check), the fault-injected numerical-agreement protocol (126 cycles, 10/10 planted faults detected, frozen bounds), the Monte-Carlo sampler audit with planted faults, the dated prior-art table, and the artifact. These are performance-engineering contributions, which is what ICPE rewards.

## 2. Verdict on "finalize and call it a day"

- As an MLSys/OSDI/EuroSys paper: not ready, and not fixable in weeks. Findings 1, 3 and 4 require a Bole comparison, batch scaling and ideally a second GPU; that is a new campaign.
- As an ICPE research-track paper: ready after one bounded experiment and an editorial pass. The one experiment worth running is the penalty-history fix plus re-measurement of the two 43-request replays and the LumoTree arm of the task study (about half a day of fix/re-qualification per Claude's estimate, roughly one GPU day of replays). It removes finding 2 entirely, lets the abstract claim exact sampling for the deployed configuration, and tests the token-inflation confound. Everything else is editorial.
- Codex's "final v2 at current scope" is correct for arXiv. It is not sufficient for a double-blind conference submission as written.

## 3. Venue

| Venue | Deadline | Decision | Published with DOI | Fit for this paper | Verdict |
|---|---|---|---|---|---|
| MLSys 2027 (research) | 30 Oct 2026 20:00 UTC (OpenReview profile needed ~2 weeks ahead) | 28 Feb 2027 | No DOI (proceedings.mlsys.org; deck verified on 2025 listing) | Exact field; 10 pp; arXiv allowed | Long odds (findings 1–4); a rejection on 28 Feb lands after ICPE/CCGrid; excluded as the (vi) anchor by the deck |
| **ICPE 2027 research track** | abstract 9 Nov, paper 16 Nov 2026 (AoE) | 25 Jan 2027 | ACM DL at conference, 24–28 May 2027, Gothenburg | Performance-engineering venue; methodology is the paper's strength; 10 pp + refs/appendices; double-anonymous; arXiv permitted with "a sufficiently different title"; artifact track for badges | **Primary** |
| CCGrid 2027 | abstract 24 Nov, paper 1 Dec 2026 | 1 Feb 2027 | IEEE Xplore ~mid-2027 | Edge of scope (single node) | Backup only if the ICPE deadline is missed (no dual submission) |
| ACM TACO | rolling | ~5 months | Gold OA, DOI on publication | Extended version | Submit the week after the ICPE decision |
| EuroMLSys 2027 (EuroSys workshop) | ~late Feb 2027 (unverified; 2026 was 24 Feb) | ~Mar 2027 | ACM DL, DOI, 6 pp | Archival floor | Only if ICPE rejects |

Decision: ICPE 2027 research track, paper due 16 November 2026. This is also what the EB-1A deck already chose on 8 September; nothing learned since changes it.

## 4. ICPE-specific mechanics to get right

- Different title from the arXiv preprint (ICPE double-blind FAQ requires it). Keep arXiv 2609.23900 as is; do not post v3 under a new title.
- Remove the artifact citation [37] (GitHub URL under the author's name), the author e-mail, and any first-person reference to earlier volumes; anonymise the artifact if it is submitted to the artifact track.
- arXiv v2 (uploaded 1 Oct) is one revision behind the current manuscript (it lacks the 30-attempt study, the disjoint replay, continuation v3 and the sampling disclosure). Post v3 under the existing title before the ICPE submission so the public record matches what reviewers may find.
- 10-page cut: keep Sections I–V compressed to ~4 pages, Setup + Results ~4.5, Discussion/Limitations ~1; move Tables V, VI, VIII, XIII–XV, XVI, XVII and Appendices to the supplement.

## 5. EB-1A implications (from the deck, not re-litigated)

- Criterion (vi) is pass/fail on papers published with a DOI by 31 Jul 2028, with the deck's own gate "≥1 published and ≥1 accepted by 31 Dec 2027." ICPE May 2027 satisfies both; MLSys would not count even if accepted (no DOI), and a February rejection would push the first DOI to Euro-Par (Aug 2027) or TACO (Q3 2027) at best.
- The paper does nothing for criterion (v) by itself. The (v) lever attached to this work is the vLLM tracker: the TreeWY #54080 comment and the state-interface RFC have been drafted since 28 Aug and marked NOW since 8 Sep, and the git log still shows "RFC filing gated on comment reply." That is the EB-1A-critical action in this project, not the venue.
- Deck finding 12 (PIIA §2870): v2 work was done during Scale employment and inference serving sits next to Scale's open-weight program. If disclosure through Scale's process has not happened, do it before the ICPE submission.
- Deck finding 11 (public materials must match the record): the resume says "a 3.04× measured decode speedup." The paper's largest ratio is 30.74/10.54 = 2.92× over plain autoregressive decoding, and the comparison a reviewer will quote is +10–12% over MTP-5. Source the 3.04× to a dated record or change the sentence before anything is filed.

## 6. Actions in order

1. Decide on the penalty-history fix (recommended: yes, one bounded run; freeze the rest).
2. Post the TreeWY comment / RFC on vllm-project (independent of the paper).
3. Post arXiv v3 of the current manuscript under the existing title.
4. Cut and reframe to 10 pages ACM double-column with a different title; anonymise.
5. ICPE abstract 9 Nov, paper 16 Nov 2026.
6. TACO extended version the week after 25 Jan 2027; EuroMLSys as floor if rejected.

No inference, push, tag or arXiv action was performed for this note. Session-start commit excludes three raw GPU-run trees totalling ~125 GB via `.git/info/exclude` (`experiments/review-response-20260927`, `artifacts/machine-backup-20260929`, `p0/monitor/review-response-20260927`).
