# FR13 — Unresolved Failure-Mode Classification (cap=500)

**Question:** Are the UNRESOLVED cases DEGENERATION (cache/spec lossiness), do they NEED A HIGHER THINKING CAP (500 too tight), or are they just genuine task difficulty (real wrong fixes)?

**Data:** Two arms adjudicated per-task from eval_report.json / runner_metadata.json / codex_trace*.jsonl / codex_stdout*.log.
- `m_e5_ON`  = cache-ON (spec + cache engaged)
- `m_cat8_OFF` = cache-OFF (control)

> READ-ONLY analysis. A live inference arm was running during classification; the running task (`astropy__astropy-14096` on the OFF arm) is IN-PROGRESS and excluded from the failure denominators (`klass=OTHER`).

---

## (1) Combined tally per class, split by cache-ON vs cache-OFF

| Class | Meaning / action bucket | cache-ON (m_e5_ON) | cache-OFF (m_cat8_OFF) |
|---|---|---:|---:|
| **CHAR8** | Transport/JSON "Unterminated string (char N)" truncation on the acting turn — cache-independent flake | **9** | **3** |
| **DEGENERATE** | Blank-decode stall / repetition / hallucinated garble — LOSSINESS signal | **2** | **0** |
| **TESTS_FAILED** | Real applied patch, eval ran, failed correctness — genuine wrong/near-miss fix | **3** | **3** |
| **GIVE_UP** | Plenty of turns, inspected, never committed — behavior/prompt issue | **1** | **0** |
| **CAP_TRUNCATED** | Reasoning hit the 500 cap mid-thought — would benefit from higher cap | **0** | **0** |
| **BAD_DIFF** | Malformed diff that engine rejected | **0** | **0** |
| **RESOLVED** | tests_passed | 1 | 1 |
| **OTHER** | In-progress / not adjudicated (excluded) | 0 | 1 |
| **Total adjudicated** | | 16 | 8 |
| **Unresolved (excl. RESOLVED+OTHER)** | | **15** | **6** |

---

## (2) Failure-fraction by action bucket

Denominators = adjudicated unresolved failures (cache-ON=15, cache-OFF=6, combined=21).

| Action bucket | Classes | cache-ON | cache-OFF | Combined |
|---|---|---:|---:|---:|
| **LOSSINESS** (act on cache / `.cpu()`-drop) | DEGENERATE | 2/15 = **13%** | 0/6 = **0%** | 2/21 = **10%** |
| **HIGHER CAP** (raise cap ~10–12k) | CAP_TRUNCATED + higher_cap_would_help | 0/15 = **0%** | 0/6 = **0%** | 0/21 = **0%** |
| **GENUINE difficulty** (pipeline healthy) | TESTS_FAILED + GIVE_UP + BAD_DIFF | 4/15 = **27%** | 3/6 = **50%** | 7/21 = **33%** |
| **TRANSPORT flake** (char-8; not cache, not cap, not difficulty) | CHAR8 | 9/15 = **60%** | 3/6 = **50%** | 12/21 = **57%** |

**Not a single task is flagged `higher_cap_would_help=true`.** In every CHAR8 case the model *completed its reasoning* and stated an explicit intent to edit ("Let me apply the fix", "I'll implement...", "The fix has been applied via sed, let me verify") — the turn then died on `Unterminated string ... (char N)` BEFORE or DURING the tool-call emission. That is transport/JSON truncation, not cap exhaustion. Cap=500 was not the binding constraint on any adjudicated failure.

---

## (3) cache-ON vs cache-OFF DEGENERATE comparison (the lossiness tell)

- **DEGENERATE: cache-ON = 2, cache-OFF = 0.**
- Both DEGENERATE cases are on the cache-ON arm:
  - `astropy-13453`: 18 consecutive blank/whitespace-only agent_messages then empty completion (blank-decode stall).
  - `astropy-14369`: `</think>` repetition loop + hallucinated digit-run garble ("123456789101112...") + fake HTML ("I had a dream ... someone handed me an apple" / "Tang Lang").
- Both are textbook lossy-decode signatures (repetition / blank-decode / garble). They appear ONLY when cache is ON, and 0/6 on cache-OFF.

**Caveat weakening this signal:** the arms do NOT cover the same adjudicated task set, and cache-OFF has far fewer adjudicated failures (6 vs 15) with one still in-progress. 2-vs-0 out of small, non-matched denominators is a weak-to-moderate tell, not proof. The dominant failure mode (CHAR8) is present on BOTH arms at ~50–60%, so it is clearly cache-independent and is NOT evidence of lossiness.

---

## (4) VERDICT + recommended action

**VERDICT: The unresolved cases are NOT primarily a thinking-cap problem, and only marginally a lossiness problem. The dominant blocker (~57%) is the CHAR8 transport/JSON-truncation flake, which is cache-independent (present on both arms). Of the remainder, ~33% are genuine task difficulty (real wrong/near-miss fixes — pipeline healthy) and ~10% (2 tasks, both cache-ON) are DEGENERATE lossiness.**

- **Higher cap: NO.** 0/21 failures are cap-truncated; `higher_cap_would_help=false` on every task. Raising the cap will not recover any of these specific failures. (See standing caveat below — a real SWE *score* still needs a higher cap for headroom, but it is not what is losing these cases.)
- **Lossiness (act on cache / drop the `.cpu()` sync): WEAK-MODERATE signal, worth acting on but not the main lever.** 2 DEGENERATE on cache-ON vs 0 on cache-OFF is directionally consistent with cache lossiness, but small non-matched denominators and the cache-independent CHAR8 dominance mean this is not the primary driver of the low resolve rate.
- **Genuine difficulty: real and expected.** TESTS_FAILED/GIVE_UP show full engagement with wrong or incomplete fixes — neither a cap nor a cache fix helps these; they are the honest floor of model capability on these tasks.

**Recommended action (priority order):**
1. **FIX CHAR8 FIRST (highest leverage, ~57% of failures).** Chase the "Unterminated string (char N)" transport/JSON truncation on the acting turn. This is cache-independent (both arms), blocks the most tasks, and masks the true underlying resolve rate — no cache or cap change touches it. Every unblocked CHAR8 turn had a completed diagnosis and an in-flight edit.
2. **Then re-measure at a higher cap (~10–12k) to get a real SWE score** — not because cap is losing current cases, but because cap=500 is a headroom-starved measurement regime; the true resolve rate is unknown until CHAR8 is fixed AND cap is raised.
3. **Keep the cache-ON DEGENERATE tell on the watchlist.** 2-vs-0 warrants a matched-task-set cache-ON/OFF re-run (identical instances, both arms) to confirm or clear the lossiness hypothesis. Do NOT green-light a cache/`.cpu()`-drop change on this evidence alone — it is suggestive, not conclusive.

**Confidence: MEDIUM.**
- HIGH that cap is not the binding constraint (0/21, unanimous `higher_cap_would_help=false`).
- HIGH that CHAR8 is the dominant, cache-independent blocker (present on both arms).
- MEDIUM-LOW on the lossiness verdict: the 2-vs-0 DEGENERATE split is directionally real but rests on small, non-matched denominators with one OFF-arm task still in progress.

---

## Standing caveat (per memory)

This is **cap=500 proxy data**, not a real SWE-Verified score. The 1-task/small-N astropy slice is a cheap gate, not truth. A genuine SWE measurement requires (a) CHAR8 fixed and (b) a higher thinking cap, regardless of this classification. Do not read the resolve rate here as the model's true capability.
