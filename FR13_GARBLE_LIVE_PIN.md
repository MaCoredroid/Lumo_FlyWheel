# FR13 — LIVE garble pin on a real SWE task (from_geodetic → from_geodentic)

**Date:** 2026-07-09. **Config:** cat8 (8-node tree spec-decode) + APC cache, pad-block
ON (baked `LUMO_FB_KERNEL_ROWS=1`), qwen-code agent, temp 0.6, the official per-instance
image. Arm `output/fr13_garble_fix_v1/cat8_nofix_g5`, task `astropy__astropy-13398`.
**Made visible by the stream-json trace fix** (commit 30212e4e) — before it, this
evidence lived in the >64KB tail that the old buffered `qwen --output-format json` clamp
threw away.

## The pin (this is the user's "spelling miss in a tool-call parameter")

Within ONE agentic session, the tree emitted the astropy API identifier
`EarthLocation.from_geodetic` in the agent's generated Python — and intermittently
CORRUPTED it to near-neighbors:

| spelling            | total | in agent's code | verdict                       |
|---------------------|-------|-----------------|-------------------------------|
| `from_geodetic`     | 63    | 51              | ✅ correct (real API)          |
| `to_geodetic`       | 11    | 9               | ✅ correct (related real API)  |
| `from_geodentic`    | 8     | 7               | 🔴 GARBLE — extra `n` (geo**den**tic) |
| `from_geodec`       | 2     | 1               | 🔴 GARBLE — truncation          |

~13% of this identifier's emissions were near-neighbor corruptions. Each corruption →
`AttributeError`/`NameError` traceback → the agent must notice and recover → a measured
**9× repeated-traceback "stuck" loop** on this task. The garble both wastes turns
(meander) and risks a give-up if the agent can't recover.

This is a DETERMINISTIC, inspectable single decision (exactly what the user predicted:
"this should be easy to pin"), not a rate statistic. It is consistent with the pinned
MECHANISM = tree-verify FORWARD DRIFT wrongly accepting a drafted near-neighbor token
that the true (no-spec) model would reject; the correct spelling still dominates (51/59)
because most positions aren't drift-flipped.

## Instrument

`scripts/fr13_garble_watch.py` (commit f3e4e8e7 + this update) now auto-flags it:
```
python3 scripts/fr13_garble_watch.py trace <qwen_trace.jsonl>
  -> NEAR-NEIGHBOR GARBLE: undefined 'from_geodentic' ~ established ['from_geodetic']
```
It mines the established-identifier pool from BOTH agent tool inputs AND tool results
(the source the agent read), so a garbled call matches the correct name; verdict is
error-anchored (an actual traceback names the undefined identifier) so normal
exploration does not false-positive. `compare --tree <cat8> --native <native>` diffs
tree vs native on shared instance_ids so common env noise cancels.

## Still to nail (differential + fix)

1. **Differential vs native** — confirm native/no-spec on 13398 does NOT emit
   `from_geodentic` (mechanism says it won't). Needs a native trace on 13398 with the
   same infra, OR a focused same-boot reproducer.
2. **Focused reproducer** — a deterministic identifier-emission probe forcing
   `EarthLocation.from_geodetic` (tree vs native vs no-spec, same seed) is a far faster
   fix-selection instrument than full live-SWE arms.
3. **Fix lever** — pad-block is already ON and the garble persists, so the drift is NOT
   (only) in the in_proj_ba tiling. Next levers: targeted fp32 hotspots / rms-clamp /
   residual re-anchor (project_fr13_amplification_levers_queued). A fix must eliminate
   the `from_geodentic` emission (and the accept-time p_target drift behind it).

## Gate (unchanged, user-binding)

Re-run the same 16 tasks (subset_b4_sixteen.json) cat8+cache-ON; give-ups drop toward
native AND `fr13_garble_watch.py arm` shows no NEAR-NEIGHBOR flags.
