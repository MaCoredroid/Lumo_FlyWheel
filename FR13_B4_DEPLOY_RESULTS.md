# FR13 B=4 deploy-speed campaign results (CLEAN offloaded harness)

Branch `fr13-speedfix`. Regime: B=4 co-residency (MAX_NUM_SEQS=4, SWE_CONCURRENCY=4),
OFFLOAD_CODEX=1 (codex+proxy on alienware, GB10 vLLM-only = clean s/fwd), 4-task
SWE-Verified subset (astropy 12907/13033/13236/13398), WALL=600 codex wall.

deploy-speed basis = `d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total)`
(per-DRAFT s/fwd, the canonical HBM-bound basis). per-TOKEN ms/tok = s_fwd /
committed_per_event (committed = accept+1). derived TPS = committed/s_fwd (NOT measured).
output/ is gitignored — numbers recorded here.

DEPTH-MATCH: E3 <- 3-3-3/cat3w/chain3; E5 <- cat6root/cat9/cat10. native E3/E4
captured at B=4 as the bars FIRST.

## NATIVE BARS (B=4, CLEAN)

| arm | depth | s/fwd (per-draft) | accept/event | committed | ms/tok | derived TPS | n_tasks |
|-----|-------|-------------------|--------------|-----------|--------|-------------|---------|
| native E5 | 5 | 0.6263 | 3.177 | 4.177 | 149.9 | 6.67 | 4 |
| native E4 | 4 | (pending) | | | | | |
| native E3 | 3 | 0.5628 | 2.350 | 3.350 | 168.0 | 5.95 | 4 |

native E3: swerc=0, 4/4 brackets. NOTE: E3 ms/tok (168.0) WORSE than E5 (149.9) and
E3 TPS (5.95) < E5 TPS (6.67) — at B=4 deeper native MTP wins (more committed/fwd
amortizes the ~B-invariant HBM read). depth-matched bar for depth-3 trees (3-3-3).

native E5: swerc=0, wall=2146s, 142 pair-dumps, resolved_rate=0.25.

### STATUS (this session)
Native bars E5 + E3 CAPTURED + COMMITTED (CLEAN offloaded B=4). E4 + all candidates +
cat9-contam + ON-mode (lossless + temp-0.6 drift) running AUTONOMOUSLY:
- speed driver pid 1568868: E4 -> cat9 -> OPT-1 -> cat6root -> cat10 -> 3-3-3 -> cat9-contam
- finalizer pid 1586106 (scripts/fr13_b4_finalizer.sh): waits for the driver, dumps all
  speed bars, runs ON-mode for the decisive depth-5 pair (cat9 vs native-E5):
  reduce pair-dump -> recurrent p oracle (temp-0 flip) + capture-q-deploy q (temp-0.6 TV)
  -> deploy-lossless (Wilson CI flip vs native floor) + deploy-temp06-drift (TV p95).
Each arm ~35-40 min (codex retry-on-empty-patch doubles the wall); full campaign ~5-6h.
Results land in output/fr13_bigdenom_swe/<arm>/deploy_speed_b4.json and
output/fr13_b4_onmode/. Monitor delivers each arm's deploy-speed incrementally.

KEY FINDING SO FAR (native bars, B=4 CLEAN): deeper native MTP is FASTER at B=4.
E5 (6.67 TPS, accept 3.18) > E3 (5.95 TPS, accept 2.35). The ~B-invariant HBM weight
read (98.6ms floor) is amortized over MORE committed tokens at higher N, so the
depth-matched BAR for depth-5 trees (cat9/cat6root/cat10) is the demanding E5 6.67 TPS.

## CANDIDATE ARMS (B=4, CLEAN) — depth-matched

| arm | depth->bar | s/fwd | accept/event | committed | ms/tok | derived TPS | vs bar TPS |
|-----|-----------|-------|--------------|-----------|--------|-------------|-----------|
| cat9 | 5->E5 | (pending) | | | | | |
| cat9+OPT-1 | 5->E5 | (pending) | | | | | |
| cat6root | 5->E5 | (pending) | | | | | |
| cat10 | 5->E5 | (pending) | | | | | |
| 3-3-3 | 3->E3 | (pending) | | | | | |

## CONTAMINATION CONTRAST (cat9, OFFLOAD_CODEX=0, codex co-located on GB10)

| arm | s/fwd | ms/tok | accept | contamination delta vs cat9 CLEAN |
|-----|-------|--------|--------|-----------------------------------|
| cat9 contam | (pending) | | | |

## LOSSLESS + temp-0.6 drift (ON-mode rescore) — depth-matched

(pending: per-arm no-spec recurrent oracle rescore; flip-rate vs native floor + temp-0.6 TV p95)
