# FR13 — char-8 tool-call regression: fr9 8/16 → current 1–2/16 (consolidated findings)

**Date:** 2026-07-01
**Scope:** consolidates this session's regression investigation. Supersedes the "current run is in-line,
not a regression" conclusion in FR13_SOLVERATE_HISTORY.md (that workflow missed the fr9 record).

## TL;DR
On the **identical 16 astropy tasks**, **fr9 (2026-06-02) resolved 8/16**; the current pipeline gets **1–2/16**.
This is a **real ~6-task regression**, and the carrier is **char-8** — the model generating malformed tool-call
JSON args that get stored in the transcript and re-parsed by vLLM → **terminal 400 → the `apply_patch` never
lands → 0-byte patch → `patch_apply_failed`**. It is **decode-side, not transport** (the tunnel was tested and
cleared). The fix is char-8 hardening (json_repair + guided_json), not the cache, cap, or tunnel.

---

## 1. The regression is real (earlier "in-line" verdict was WRONG)

`output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z` (native MTP-5 = "spine-5" = chain5/e5, temp 0.6,
**NO thinking cap**, on-GB10 harness, concurrency=4, cache OFF) resolved **8/16**:

| resolved in fr9 | 12907, 13453, 14096, 14309, 14365, 14508, 14539, 14995 |
|---|---|
| failed in fr9 | 13033, 13236, 13398, 13579, 13977, 14182, 14369, 14598 |

Current run `run_20260701T072605Z`: m_e5_ON (cache-ON) = **1/16**; m_cat8_OFF = **~3/16**. The w5t1wnjn8 history
workflow reported "high-water 4/16 / not a regression" **because it never scanned `fr9_*` dirs** — it missed
this record. (It was right that 13033/13236/13398 are hard — those failed in fr9 too — but the FULL-16 clearly
discriminates and shows regression.)

## 2. Mechanism — char-8 kills `apply_patch` (verified)

The tasks fr9 solved now fail like this in the current run:

| task | fr9 | current | how it dies |
|---|---|---|---|
| 13453 | ✅ | ❌ | patch_apply_failed, **0-byte** patch, 40 turns, char-8 present |
| 14508 | ✅ | ❌ | patch_apply_failed, 0-byte, char-8 |
| 14539 | ✅ | ❌ | patch_apply_failed, 0-byte, 44 turns, char-8 |
| 14995 | ✅ | ❌ | patch_apply_failed, 0-byte, char-8 |
| **12907** | ✅ | ✅ *(cat8_OFF)* | **char-8 did NOT fire → RESOLVED, 504B patch, tests_passed** |

The `12907` control is the smoking gun: same task, **char-8 off → resolves**. The agent grinds 40+ turns but the
long diff sits in the tool-call JSON args → truncated → 400 → the patch never applies → empty patch.

## 3. It is DECODE-side, not the tunnel (tunnel tested + cleared)

No-GPU echo-server test, alienware→GB10 over the tailscale tunnel (the path the request travels):

| payload | 1 KB → 1 MB (7 sizes) |
|---|---|
| result | **3/3 round-trips INTACT each** (received == sent, sha match) |

TCP/WireGuard doesn't truncate a well-formed request. So char-8 is **not** the transport cutting the payload —
it's the **model generating malformed args**, which are **stored in the transcript** and re-parsed on every
later turn → **terminal** (retries re-hit the SAME frozen bad args; this is why fr9's char-8 was survivable but
this pipeline's is fatal). Running on-GB10 would NOT fix it. Amplifier vs fr9 is decode-side (tree/APC/spec/
forked-fa2 or the offload proxy's request-shaping) — isolating which needs a GPU A/B (the queued e5_OFF arm).

## 4. char-8 is NOT the cap and NOT the cache (per the classification)

- **Cap:** 0/21 unresolved cases were CAP_TRUNCATED; the regressed tasks run 40+ turns (not truncated into
  giving up). fr9 had NO cap, but the cap is not the carrier. (Cap=500 still depresses a *real* score; drop it
  for score runs — bounded by MAX_OUTPUT 32768.)
- **Cache:** char-8 fires on BOTH cache-ON (15/16) and cache-OFF (7/9) traces = cache-independent; and history
  shows the ON-vs-OFF gap vanishes under EXACT_SEED. (The queued e5_OFF arm confirms cache-off spine-5.)
- **Dominance:** char-8 ≈ 57% primary cause; genuine task difficulty ≈ 33%; degeneration/garble ≈ 10% (2/21,
  cache-ON only, watchlist). See FR13_UNRESOLVED_FAILURE_MODES.md.

## 5. The fix — grounded + research-backed, but strict-flag NOT in our build

- **Research confirms** malformed/truncated Qwen3 tool-call JSON is a known failure mode (vLLM #21711, #19419,
  #27921, #18819, #39056) and the documented fix is grammar-constrained (strict) tool calling.
- **BUT our forked vLLM is `0.19.2rc1.dev134` — it has NO `VLLM_ENFORCE_STRICT_TOOL_CALLING`** (verified: no refs;
  `strict:true` on a tool is not consumed). So `LUMO_PROXY_TOOL_STRICT` is a **no-op in our build** — a flag flip
  will NOT fix char-8.
- **What works in 0.19.2rc1:** the structured-outputs / `structural_tag` machinery IS present
  (`responses/serving.py:481-489`). Real fix = a proxy implementation, chosen (user, 2026-07-01) = **BOTH**:
  1. **json_repair** on the outgoing transcript — repair stored malformed args **before** vLLM re-parses them →
     breaks the terminal-400 loop. **Distribution-neutral** (doesn't touch decode; safe for lossless work).
     `json_repair` pip-installed this session.
  2. **guided_json** — attach a JSON grammar to tool-call args per request to prevent malformed generation
     (changes distribution; apply equally to both arms if used in an A/B).

## 6. Corrections to prior conclusions

- **FR13_SOLVERATE_HISTORY.md "not a regression" = WRONG** (missed fr9). It IS a regression.
- **project_fr13_b4_solve_ceiling** "full-16 high-water 4/16" = WRONG (fr9 = 8/16); corrected in memory.
- **char-8 attribution OFF-control gate = now DELIVERED** (cache-OFF 7/9) → cache-independent CONFIRMED.

## 7. Queued next experiment + fix

- **e5_OFF arm** (auto-launches after the current matrix; waiter script
  `$CLAUDE_JOB_DIR/tmp/launch_e5off_after.sh`): spine-5, **cache OFF**, cap=500, subset_b4_sixteen. Isolates
  whether APC cache is the carrier: recovers toward 8/16 ⇒ cache; stays ~1/16 ⇒ cap/harness/char-8.
- **Resume the char-8 fix** (json_repair transcript-repair first; then guided_json), then rerun to watch
  char-8 + empty-patches vanish and the solve-rate climb back toward fr9's 8/16.

## Artifacts
- fr9 record: `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/` (8/16)
- current run: `output/fr13_tree_cache_matrix/run_20260701T072605Z/`
- companion docs: FR13_UNRESOLVED_FAILURE_MODES.md, FR13_SPEC_CACHE_DEGENERATION_LITERATURE.md,
  FR13_DEGENERATION_INVESTIGATION.md, FR13_SOLVERATE_HISTORY.md (see §6 correction)
- memory: `project_fr13_char8_attribution_open`, `project_fr13_b4_solve_ceiling`
