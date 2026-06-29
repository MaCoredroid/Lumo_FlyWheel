# FR13 — Post-Sweep Test Plan (do AFTER the running 1800s B=4 sweep finishes)

**Hard rule (user 2026-06-16):** do NOT edit/build/re-reduce/commit on the GB10 host while the
performance sweep is running — host CPU/disk activity contaminates the GB10 unified-memory timing.
The reconcile workflow (`w13rzd5zn`) is read-only and may finish first; **record its result, do not
apply the fix on-machine until the sweep is done.** Everything below is deferred to post-sweep.

---

## A. What the CURRENT 1800s sweep already covers (the baseline)
Driver `fr13_b4_campaign_driver.sh` @ WALL=1800, B=4, OFFLOAD_CODEX=1, sequence:
**E5 → cat9 → OPT-1 → cat6root → cat10 → E3 → 3-3-3 → cat9-contam**, each:
- **Speed**: `deploy-speed` → s/fwd, accept/event, committed, derived_tps, **per_request_decode_tps**,
  **aggregate_decode_tps** (the new fields, commit 585b1f02).
- **Lossless (finalizer `fr13_b4_finalizer.sh`)**: ON-mode rescore for the DECISIVE pairs only —
  cat9-vs-native-E5 (depth-5) and 3-3-3-vs-native-E3 (depth-3): temp-0 flip-rate (`deploy-lossless`)
  + temp-0.6 per-position id-aligned TV (`deploy-temp06-drift`), each arm vs its own no-spec
  recurrent oracle, within native floor.
- **Contamination contrast**: cat9 OFFLOAD_CODEX=0 (codex co-located) vs the clean cat9.
- Depth-match: 3-3-3→E3; cat6root/cat9/cat10→E5. (E4 dropped = not a depth-match bar.)

### Gaps in the current sweep to fill next
1. **Lossless only covers cat9 + 3-3-3.** cat6root, cat10, OPT-1 get speed but NOT the temp-0/temp-0.6
   lossless gate. If any of them is a speed contender, it MUST clear lossless too → extend the
   finalizer to the contending arms.
2. **No B=1 arm** on this 1800s deployment regime → the clean B=1↔B=4 reconciliation stays analytic.

---

## B. Accounting fix (from the workflow `w13rzd5zn` + the user's catch) — APPLY POST-SWEEP
- **`aggregate_decode_tps` (15.66) is mislabeled.** Its union window (earliest-pre→latest-post)
  INCLUDES tool-call + inter-turn agentic idle → it's **effective end-to-end throughput**, NOT GPU
  decode capacity. RELABEL it honestly (or drop).
- **Add the idle-EXCLUDED decode-busy aggregate** = total generation_tokens ÷ decode-active wall
  (≈ per_request_decode_tps × avg concurrency-while-decoding). This is the apples-to-apple basis vs
  the historical ~40. **Offline-computable from saved data — NO GPU re-run** (E5 1800s docker_full.log
  carries the `Running: N reqs` histogram: 126×4 / 82×3 / 158×2 / 22×1 → avg active concurrency ~2.8;
  brackets carry gen_tokens, decode_seconds, request_time_per_output_token).
- **Reconcile vs history ~40** (41.266 "returned-token wall TPS" / ct/ds / 39.9): per the workflow,
  pin whether ~40 was flat-4 synthetic vs SWE-deployment, and which wall it used; the 1800s run only
  held ~2.8 streams (agentic desync), a real chunk of the gap. Report decode-busy + the closing
  arithmetic; do NOT hand-roll a number.

---

## C. New tree shapes to BUILD + measure (16-node budget)
1. **5-5-5** (user): depth-3, **5 candidates per depth = 15 nodes** (≤16, pad16). Compares to **E3**.
   - Needs drafter **rank-4 (top-5)** support: read `torch.topk(logits, 5)[:, 1..4]` from the same
     spine logits (3-3-3 added rank-2/top-3; this extends to top-5). Lossless-by-construction
     (runner-up logit reads, never enter a forward/recurrent state). Build like cat3w/3-3-3:
     exact-match guard `_fr10_is_555`, default-OFF byte-identical, fail-loud unbuilt.
   - Hypothesis: HBM-bound ⇒ accept-per-forward is the only lever; a WIDER shallow tree (15 nodes,
     5-wide d3) tests whether more candidates → enough accept gain to beat its ~2%/node state cost vs
     cat9 (d5,9) and 3-3-3 (d3,9). Decisive: 5-5-5 accept/event vs E3, net TPS vs E3.
2. **Width sweep at fixed budget** to map the accept/node curve: 4-4-4 (12), 5-5-5 (15), and a
   depth-4 variant (e.g. 4-4-4-4 = 16) — find the accept-maximizing topology under the 16-node cap.
3. Each new shape: deploy-speed (all 3 TPS bases) + deploy-lossless + deploy-temp06-drift vs its
   depth-matched native (d3→E3, d4→E4 [re-measure E4 if a d4 shape ships], d5→E5).

---

## D. Other things worth testing (Claude's adds)
- **Suffix-decode drafter arm** (user mentioned "even suffix decode"; FR13_SUFFIX_FUSION): graft
  suffix-decoded guess tokens onto the cat caterpillar branches (narrow MTP spine at root/1/2 +
  suffix-decode guesses on branches). Drafter-agnostic verifier already consumes a standard
  candidate-tree descriptor, so this is the natural next drafter. Measure accept/event vs the
  depth-matched native + the lossless gate. The real test = does the suffix tail's conditional accept
  gain on the agentic-SWE stream exceed the cat9 leaf accepts.
- **B=1 native-E5 + cat9 twin** at WALL=1800 on the SAME deployment content → airtight B=1↔B=4
  reconciliation (per-forward 1.030× clean-regime confirm) + the pure-decode per-request bar.
- **OPT-1 (committer sync-kill)** at B=4 clean: run-ahead census (block % OFF vs ON), byte-identical
  OFF==ON, s/fwd/wall benefit now the box is uncontended (it's in the current sweep — read its result).
- **Co-residency sensitivity**: the 1800s run held only ~2.8 streams (modal 2). Test whether the
  aggregate/per-request picture changes with more concurrent codex agents (SWE_CONCURRENCY > 4 feeding
  MAX_NUM_SEQS=4) so the batch stays fuller, or whether agentic desync caps real co-residency.
- **Lossless headline = the temp-0.6 Tier-A number** for cat9-vs-E5 (the gap we've never had a clean
  read of) — make sure the finalizer's TV is real id-aligned per-position (n_scored>0), not the
  string/id artifact or a temp-0 stand-in.

---

## E. Process
After the sweep ends: prelaunch (recover_host_memory + free ≥95GiB + docker empty) before any GPU
arm. Build phase (5-5-5/4-4-4 + suffix drafter) is CPU/off-machine; the measurement phase is the
single serialized GPU. Apply the accounting fix (B) offline first (no GPU). Commit+push each step on
fr13-speedfix. Keep MAX 2 concurrent workflows.
