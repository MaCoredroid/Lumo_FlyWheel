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
