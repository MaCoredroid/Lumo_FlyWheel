# FR-13 — the current FA2-fork build's SPINE is argmax-lossless (the accepted gate is met); the 1.77 is STALE

Monitor red-team, 2026-06-08. Offline check on **committed** artifacts (no GPU), recomputed from raw `.pt` logits in `output/fr13_ex2_live_ladder_20260608T021853Z/`.

## Finding 1 — spine ARGMAX-LOSSLESS 6/6 on the current build
`tree10/logs/tree_final_logits.pt` (10 tree rows) vs `native/logs/native_final_logits.pt` (6 spine rows), spine row map `[0,1,2,4,6,8] -> [0,1,2,3,4,5]` (ladder-log scheduler map):

| spine row -> native | tree argmax | native argmax | match | max_abs logit drift | native top1 gap (1st-2nd) |
| --- | --- | --- | --- | ---: | ---: |
| 0 -> 0 | 579 | 579 | ✅ | 1.5547 | 11.75 |
| 1 -> 1 | 264 | 264 | ✅ | 1.8750 | 11.12 |
| 2 -> 2 | 7047 | 7047 | ✅ | 1.5000 | 12.12 |
| 4 -> 3 | 1817 | 1817 | ✅ | 1.9062 | 10.75 |
| 6 -> 4 | 25 | 25 | ✅ | 1.3750 | 7.25 |
| 8 -> 5 | 271 | 271 | ✅ | 0.5938 | 9.19 |

**SPINE ARGMAX MATCH = 6/6.** The drift (≤1.9) is **4–12× below** the native top-1 gap on every row ⟹ it cannot flip the argmax. The row map matching argmax 6/6 is itself a consistency check that the map is correct.

## Finding 2 — the "1.77 vs 3.076" e2e number is STALE (FR10 pre-fork), NOT the current build
`grep` provenance: 1.77 originates in `FR10_CLOSEOUT.md` / `FR10_STATUS.md` — the **FR10-era** no-copy tree (TreeAttention/depth-position, **before the CUTLASS FA2 fork existed**). FR10 closed no-go @ 1.77 citing *"diffuse per-layer numerical drift."* **FR13's entire CUTLASS fork was the fix for that** — and it worked for full-attn (floor verdict: 14/16 calls whole-tree byte-exact, 2 single-ULP in ~1M). **The current FA2-fork build's e2e accept/event has never been measured.** Do not cite 1.77 as the current verdict.

## Finding 3 — the ladder log's "NOT a pass" used the SUPERSEDED bar
`FR13_LADDER_LOG.md` marks the run "not a pass" because final-logits max_abs = 1.9 ≠ 0. But the user's **accepted gate** (`FR13_FLOOR_WORKFLOW_VERDICT.md`, USER DECISION 2026-06-07) is **within-E5-floor / argmax-lossless**, NOT literal-0/max_abs. Under the accepted gate, the **spine passes** (argmax 6/6). The two codex workers (fr14, fr15) hung for hours chasing GDN L8 `h0_state_in` to *literal 0.0* — a bar the user already relaxed. That grind is not required by the accepted gate.

## Implication — RUN THE E2E (the real gate), stop the literal-0 GDN micro-grind
The remaining GDN per-layer drift (L8 `h0_state_in`=7e-4 → compounds to 1.9 final logits) does **not** flip spine argmax. The accepted gate is the **e2e**: bag-TV vs E5 ≤ floor (~0.059) **+** accept/event ≥ 3.076 at **B=4 CUDA-graphed**. The single-event argmax-lossless spine is strong evidence the e2e will pass; **measure it**.

## Honest caveats (this is evidence to RUN the e2e, NOT a self-declared pass)
1. **One event** (call0), **B=1 eager**, **6 spine rows**. The gate is B=4 + CUDA-graph + SWE-4. argmax-lossless may shift under B=4 co-residency / more events.
2. **argmax-lossless ≠ sample-distribution-lossless.** e2e uses rejection sampling at temp 0.6 / top_p 0.95; a 1.9 logit drift shifts softmax mass on the selected token, so accept/event could differ from native even with argmax preserved. Direction not obviously down (drift is small vs gap). The read-only workflow predicts the temp-0.6 acceptance from these committed logits before the GPU run.
3. **Branches** are within the FA2 floor (ladder: `tree_vs_fa2_branch=0.0` for 15/16 full-attn layers; one 4.9e-4 residual at L55) — adds ≥0 accepts (superset).

## Next action
1. Read-only workflow: adversarially verify the 6/6 argmax-lossless claim + predict temp-0.6 e2e acceptance from committed logits + audit stale-1.77 → go/no-go.
2. codex e2e: boot forked-FA2 server **B=4 CUDA-graph** (confirm FULL capture + Gate-2 hooks-off), pin E5's exact launch config + align both launchers, run the timed e2e (accept/event via /metrics + bag-TV vs E5 `output/fr10_native_mtp5_same8_20260604T210257Z`). ONE GPU, no `--rm`, ModelServer sync+drop_caches between arms.
