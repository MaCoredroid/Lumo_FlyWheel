# FR13 B=1 depth-5 data: VERIFIED CLEAN (not B=4-confounded) — 2026-06-17

A false alarm I raised ("the b1 run was actually B=4, num_running ~1.79-2.16") was WRONG: I aggregated the
GITIGNORED `output/fr13_bigdenom_swe/cat6root_b1/offload_request_metrics.jsonl`, which ACCUMULATED multiple
campaigns (an earlier B=4 run + the b1 run) into one file. Separating by run window settles it.

## The committed b1 arms ran SEQUENTIALLY at TRUE B=1 (num_running ~0)
| arm | run window (UTC, 06-16->06-17) | offload num_running mean (max) | accept/event | per_request_decode_tps | s_per_fwd_gpu |
|---|---|---|---|---|---|
| nativeE5_b1 | 22:11:48 -> 01:17:21 | **0.007 (1)** | 3.112 | 17.80 | 0.1370 |
| cat9_b1     | 01:24:45 -> 04:18:43 | **0.015 (1)** | 3.641 | 18.44 | 0.1440 |
| cat6root_b1 | 04:26:06 -> 07:10:59 | **0.022 (1)** | 3.821 | 18.51 | 0.1377 |

All three at num_running ~0 (max 1) => genuine B=1 single-stream, no co-residency. The runner_metadata
timestamps map exactly into the num_running~0 offload windows. The committed deploy_speed comparison
(cat6 18.51/3.82 > cat9 18.44/3.64 > E5 17.80/3.11, all lossless within floor) is a VALID clean B=1 result.

## The contamination is NOT on main
The B=4 records (06-16 00:44->21:00, num_running mean **2.38**, 1059 reqs) are a SEPARATE earlier campaign,
present ONLY in the gitignored `output/.../offload_request_metrics.jsonl` (accumulated across runs). NONE of
the committed b1_depth5_raw/ data (per-task brackets, deploy_speed, sfwd sidecar, lossless rescores) carries it
— those are per-arm, from the sequential B=1 windows above. So nothing checked into main needs separating.
(ACTION: the contaminated output/ file is deleted locally to prevent re-aggregation.)

## RETRACTION: speed_gap_localization.md "agent-side / consumer-paced" localization is WITHDRAWN
That doc concluded the +28ms cat6-vs-E5 step gap was "consumer/agent-loop idle, our-code ~6ms". It was derived
from (a) the MIXED offload file (B=4+B=1 aggregated) and (b) a SYNTHETIC generic-text prompt (unrepresentative —
accept flipped, E5 won). BOTH inputs were invalid. On the clean B=1 data the +28ms is REAL (cat6 step 0.260 vs
E5 0.231 = committed/TPS) but UNLOCALIZED. The proper test = an INSTRUMENTED B=1 SWE-Verified re-run with
per-phase decode timers (forward / drafter / committer), pending. The VERDICT (cat6 +4% TPS + higher accept +
lossless within floor, all at true B=1) HOLDS.
