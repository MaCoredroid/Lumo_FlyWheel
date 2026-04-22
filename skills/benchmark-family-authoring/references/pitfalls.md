# Pitfalls

These are the failure modes that burned calibration time on the `proposal-ranking-manager-judgment` family (attempts 01 → 02d). Read the ones relevant to your current phase *before* making the change, not after.

## The disclosure paradox

**Symptom.** AGENTS.md tells the agent what the scorer rewards ("cite the staffing update file", "mention 40% in primary_risk", "include an assumption ledger with at least one missing row"). Capable models score very high — V1-V3 clustered near the pass bar. There's no signal at the top of the curve because the agent knows the answer template.

**Why.** You can't simultaneously (a) tell the agent exactly how they'll be graded and (b) measure whether they'd exercise that judgment unprompted. Either the task description doubles as a cheat sheet or the scorer measures something different from what was described.

**Fix.** Decouple. Rewrite AGENTS.md with **principle-level** guidance only:

- "Staffing can change mid-quarter. Evidence may contain more than one staffing artifact at different dates. If the accepted owner's availability changed, that reality should shape how you reason about risk."
- NOT: "Cite `roster_memo.md`. Mention `40%`, `parental leave`, `Ravi cover` in `primary_risk`."

Then end AGENTS.md's grader section with: "The specific thresholds and caps are grader-internal. Produce a good brief by the standard above." The principle survives across variants; the thresholds and keyword lists do not appear in agent-visible text anywhere.

**Reference attempt.** `proposal-ranking-manager-judgment` attempt_02c moved V5 from 81 → 25 on this change alone. That delta was the real signal — V5's prior 81 was rubric leakage, not reasoning.

## Rubric leakage in evidence files

**Symptom.** Decoupling AGENTS.md didn't move the scores. The 88 floor on V1-V3 is stable.

**Why.** Evidence files sometimes contain their own rubric-leakage sections — e.g., a memo that ends with "How to reflect this in a manager brief: keywords X, Y, Z; citations should include this file". The agent doesn't need AGENTS.md to cheat; the memo itself is the cheat sheet.

**Fix.** Audit every evidence file in `repo_evidence/`, `release_context/`, `incident_context/`, `proposals/`. Strip any section that reads like scorer instructions. Keep the facts: dates, numbers, named people, concrete constraints. Let the agent build the brief structure from the principles in AGENTS.md and the facts in evidence, not from template instructions embedded in evidence.

A good smell test: if a section could plausibly be titled "Brief-writing guide", delete it.

## Adversarial evidence without adversarial scoring

**Symptom.** You add contradictory evidence (e.g., `staffing.md` says Priya 100%, `roster_memo.md` says 40%, no filename precedence cue). You expect V1-V3 to drop to 40-60. Scores don't move.

**Why.** Adversarial evidence hardens the *task* (the agent has to read more, reconcile more). It does not automatically harden the *score*. A capable model reads both docs, notices the 2026-06-15 in-document date vs the 2026-04-10 date, trusts the "Supersedes" language, and silently picks the correct value. The scorer — which only checks that the brief uses the correct number — is happy. No ceiling fires because no ceiling was set up to fire on "silently resolved contradictory evidence without flagging it".

**Fix options.**

- **If silent resolution is legitimate**: accept that the adversarial evidence didn't move the score. It made the task more realistic but didn't create a new judgment dimension. That's fine — not every hardening lever has to move the score.
- **If silent resolution is a real judgment failure**: add a ceiling. For example, a `contradiction_unflagged = 55` ceiling that fires when `accepted == <priya-owned proposal>` AND the brief citations reference only one of `staffing.md` / `roster_memo.md` (a conscientious manager flags the conflict rather than silently picking a side). Run the legitimate-difficulty test: would a strong human manager agree this ceiling represents a real failure? If yes, ship it. If you're unsure, don't.

**Reference attempt.** `proposal-ranking-manager-judgment` attempt_02d confirmed this: the adversarial staffing memo was cleanly handled by gpt-5.4 high on every run. Family mean was 71.60 vs attempt_02c's 66.80 — no meaningful change, and the slight regression came from V5 variance, not from the new adversarial evidence.

## Mechanical score floor

**Symptom.** A variant's mean sits at a stable integer (e.g., 88.00 ± 0 across 9 runs). No ceilings fire. Every probe report looks identical.

**Why.** The raw-layer checks are each giving the agent their full points. The remaining 12 points (or whatever the gap is) are scattered across tiny-weight dimensions the agent never fully nails but never badly fails either. This is the rubric's **mechanical floor** — the score a competent agent gets for doing the concrete task correctly, before any judgment ceiling fires.

The mechanical floor is not movable by adversarial evidence, decoupling, or rubric tightening. It represents "can the agent produce a valid brief for this family at all". The only ways to move it down:

1. **Add a new judgment ceiling** with a legitimate trigger that the capable model reliably misses. The ceiling must pass the legitimate-difficulty test — don't invent a trap to close the gap.
2. **Redistribute raw points** away from dimensions the agent always nails toward dimensions that vary. This is a scorer refactor, not a hardening pass. Do it when `breakdown` shows 10+ checks all awarding full points on every run.
3. **Accept the floor** and anchor the family's signal on the variants that do vary (usually V4/V5). Document this in `benchmark_run.md` and widen the §10.1 window for this family.

**Do not** try to force a capable model below its mechanical floor by adding vague "judgment" requirements or by weighting LLM-judge rubrics upward. That crosses into fake ambiguity.

**Reference attempt.** `proposal-ranking-manager-judgment` V1-V3 stabilized at 88 across attempts 02b, 02c, 02d. Three different hardening levers applied. No movement. The honest conclusion: the family's V1-V3 signal is a floor-check, not a discriminator. V4/V5 carry the signal. The §10.1 window was widened for this family and the family shipped.

## Readonly hash drift → `shortcut_detected: true`

**Symptom.** Every probe run reports `shortcut_detected: true` and `pass: false` despite high raw scores. The oracle scores ≥ 90 in the regen script but probe runs don't.

**Why.** `gold_ranking.json.readonly_tree_hashes` hasn't been refreshed after a change to `AGENTS.md`, `bin/`, or anything under `repo_evidence/`, `proposals/`, `release_context/`, `incident_context/`, or `tests/`. The scorer recomputes the hashes at scoring time, sees they don't match the stored values, and flags the run as a shortcut (i.e., "the agent must have tampered with readonly trees").

**Fix.**

```bash
python3 scripts/regen_cnb55_v2.py     # refreshes readonly_tree_hashes, regens oracle briefs, verifies scores
python3 scripts/refresh_manifest_lock.py   # rebuilds manifest.lock.json
```

Run both after any edit under `workspace_bundle/<variant>/`. The regen script also re-verifies the oracle/empty/shortcut baselines — if one fails, the regen exits non-zero and you'll notice before the probe.

## AppleScript `&` reserved operator

**Symptom.** An osascript call with `do shell script "<cmd> &"` returns successfully but no background process starts. The probe launch silently dropped.

**Why.** AppleScript's string parser treats `&` as the string-concatenation operator. Inside `do shell script`'s argument, the `&` is either parsed as a concatenation with the next string (which isn't there) or silently stripped depending on the surrounding quoting. The shell never sees it, so no nohup detach happens.

**Fix.** Use a wrapper script that self-daemonizes internally. See `_run_probe_02b.sh` in `acceptance-gate-calibration.md`. The AppleScript caller invokes `_run_probe_XX.sh start` — no `&` in the AppleScript layer at all.

## osascript MCP timeout (~45s)

**Symptom.** An osascript call with `sleep 90` or `sleep 120` inside the `do shell script` body times out. The long-running work doesn't happen.

**Why.** The osascript MCP caps individual call duration around 45 seconds.

**Fix.** Never embed long sleeps in osascript. Put the waiting in the Bash tool (which has its own timeout you can set to 10 minutes) and keep osascript calls short. For polling a long-running probe: `Bash(sleep 300)`, then a short osascript call to `tail` the log and check the DONE marker. Iterate until DONE appears.

## AppleScript `%` character parsing

**Symptom.** An osascript call containing `grep '40%'` returns a parse error, though the output sometimes appears before the error.

**Why.** AppleScript treats `%` as special inside some quoting contexts.

**Fix.** Escape with `%%` in AppleScript string literals, or restructure the grep to avoid the character (match on `"40"` only, or on `"parental"` instead of `"40%"`).

## One-variant shortcut: updating only some variants

**Symptom.** You change `staffing.md` in V1-V3 to add an authoritative "100% Q3" claim and leave V4-V5 alone. Probe reports V1-V3 oracles fail with `missed_staffing_update` firing.

**Why.** The regen script refreshed oracle briefs for V4/V5 too, but the oracle brief for V4/V5 still cites `staffing_update_2026_06_15.md` (the old dated filename) while V1-V3 now have `roster_memo.md`. If the scorer reads `staffing_update_file` from `gold_ranking.json` and that per-variant path doesn't match the oracle's citations, the oracle brief triggers its own miss.

**Fix.** When splitting a change across a subset of variants, audit:

- `verifier_data/<family>/<variant>/gold_ranking.json` for each affected variant.
- `scripts/regen_cnb55_v2.py` oracle functions for each affected variant.
- The evidence files themselves under `workspace_bundle/<variant>/repo_evidence/`.

All three must agree on the new evidence layout per variant. Re-run the regen script and check that oracle scores are still ≥ 90 for every variant before launching the probe.

## Attempts that change two levers at once

**Symptom.** `benchmark_run.md` attempt_NN shows a big score movement, but the next attempt undoes part of the change and the scores barely move. Diagnosis is impossible.

**Why.** The attempt applied two hardening levers simultaneously (e.g., decoupled AGENTS.md AND added adversarial evidence). When the probe shows the score moved, you can't tell which lever caused it, so you can't build a mental model.

**Fix.** One lever per attempt. Always. Even if you're confident both are needed. The `benchmark_run.md` log is only useful if each attempt has a single attributable cause.

The one exception: a scorer bug fix that unblocks a previous lever. Document the bug in the attempt description and apply both changes as one attempt — but note explicitly that this is an exception.

## Verification-matrix synthesizer mutates the wrong field

**Symptom.** The Pick-ceiling row (e.g. "Pick-P3 staffing-blocked") in `verification_matrix.md` scores well above the HLD §5 expected band — often in the 60–80s instead of ≈ 30 — and the expected ceiling does not appear in `ceilings_applied`. Oracle, Empty, and Delete-tests rows all look fine.

**Why.** The synthesizer in `scripts/run_verification_matrix.py` is mutating a field the scorer does not read. Typical version of the bug: the schema has `accepted_proposal_id` somewhere in the ranking entries but the scorer's ceiling logic reads top-level `brief["accepted"]`, and the synthesizer sets `accepted_proposal_id` because that name "looks more complete". The scorer never sees the mutation; the oracle's real `accepted` stays, ceiling never fires, score stays high.

**Fix.** Grep the scorer for `brief.get(...)` and `brief[...]` — the field names the synthesizer has to mutate are exactly those. Do the same for `gold.get(...)` when a synthesizer has to align a gold-side field. Running the Pick-ceiling row against the Oracle brief with only `accepted` changed is enough to prove the ceiling fires before wiring the full matrix.

**Debug recipe.** Print `result["ceilings_applied"]` for the Pick row. If it's empty, the condition did not trigger — the synthesizer is editing the wrong field.

## Ceiling stacking: hard gate always beats soft gate

**Symptom.** You added a new soft ceiling (acknowledgement-aware) for a trap that happens to sit on the same proposal ID as an existing hard ceiling. The verification matrix only ever reports the hard ceiling, never the new soft one, even with synthesizers that should trip it.

**Why.** When two ceilings fire on the same brief, the scorer applies the minimum cap (lowest value wins). Hard gates (no acknowledgement escape, e.g. `ignored_staffing_constraint`) generally cap lower or equal to soft gates (acknowledgement-aware, e.g. `sunk_cost_finish`). Even if both fire, only the hard ceiling shows in `ceilings_applied` because the soft ceiling's apply-step is suppressed by a stricter earlier cap. And the soft ceiling's acknowledgement-escape may already be satisfied by language the synthesizer unintentionally carries over from the Oracle brief, so it never fires to begin with.

**This is intended, not a bug.** Do not add logic to force the soft ceiling to fire — you would weaken the harder constraint. Two legitimate responses:

1. **Document the stacking** in `verification_matrix_<variant>.md`. Note that on this variant the soft ceiling is subsumed by the hard one, and that isolated testing would require a future variant where the traps split.
2. **Split the traps in a new variant.** If the family needs to exercise the soft ceiling as a distinct training signal, author a new variant where the soft-trap proposal is different from the hard-trap proposal.

**Do NOT** tune down the hard ceiling's cap to let the soft ceiling show through. That is rubric corruption — the hard ceiling is the higher-confidence signal and has to stay authoritative.
