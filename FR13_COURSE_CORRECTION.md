# FR-13 course-correction (2026-06-07) — the spine full-attn is NOT the 0.92 cause; regime mismatch

**Trigger:** user red-team — "accept/event gap seems large vs the attention gap; are we measuring the same code path (cuda-graph vs our measurement)? error propagate 64 layers?"

## The inconsistency (user is right)
- Deliverable (accept 0.92, bag_TV 0.558, LOSSY): **B=4, CUDA-graph CAPTURED**, TREE_ATTN exp2.
- ALL per-layer/op diagnostics (attn_out_raw 0.0077; propagate 0.004->7.25): **eager, B=1.** Different code path. Never measured per-layer in the captured/B4 regime.
- codex boot-free FA2 op-diff: **TREE_ATTN matches FA2 on the SPINE** (LSE 7.6e-6; only a tiny P@V/accum gap, within floor).
- Eager propagate 203056Z: spine final_norm **depths 0,1,2 = 0.0** (bit-exact), only depths 3,4 diverge.

**=> The spine full-attn is within floor and does NOT reject the spine. Bit-exact shallow depths predict accept ~2.6, NOT 0.92. So the full-attn op is NOT the cause of the accept deficit.** My earlier "full-attn is THE lossy root / drive it to 0" was an over-claim: the full-attn drift is real but concentrated in deep depths (3,4) and within-floor on the spine.

## Extra red flag: our eager propagates DISAGREE
- 203056Z spine final_norm depth0 = **0.0**.
- 221323Z (treeattn) spine final_norm depth0 = **0.75**.
Internal inconsistency = measurement-validity problem (exactly the user's "same code path?"). Do not trust per-layer hidden diffs to explain accept; measure accept e2e in a controlled regime.

## The 0.92 lives in a path we never measured. Decisive experiments (codex, ONE GPU):
1. **REGIME:** run fr12_deliverable_swe4_probe.py **EAGER B=1**, same prompts/seed. If e2e accept ~0.92 -> eager reproduces it (localize in eager). If ~E5 (~2.6) -> the 0.92 is **captured/B4-specific** (cuda-graph numerics, B=4 co-residency, or the captured commit path) and ALL eager op-localization was chasing a ghost.
2. **DECOMPOSE (standing directive #3):** spine-only accept (branches OFF) vs branches-on vs E5.
   - spine-only ~= E5 -> verify is lossless; deficit is branches/drafter (branch bonus ~ 0 = drafter topology).
   - spine-only ~= 0.92 with spine attn proven fine -> the **commit / rejection-sampler** is the bug, not the attention.

## Status of the "drive full_attn drift to 0" task
Demoted. The full-attn op-fix is at most a minor lossless contributor (deep depths); it is NOT the lever for the 0.92 accept deficit. Resolve the regime + decomposition FIRST; only return to the full-attn op if it proves to matter e2e.
