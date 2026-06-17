# FR13 B=1 depth-5 LOSSLESS verdict (2026-06-17)

Clear-margin argmax flip-rate vs the no-spec RECURRENT decode oracle (40-turn sample/arm,
deterministic seed). 12.9% is the temp-0.6 shared baseline (sampled != greedy argmax), NOT
a native-lossy finding; the gate is comparative (within E5's floor).

| arm | total_positions | clear_margin_flips | rate | verdict |
|---|---|---|---|---|
| nativeE5_b1 | 4883 | 630 | 12.90% +/-0.48 | FLOOR |
| cat9_b1 | 4858 | 645 | 13.28% +/-0.49 | LOSSLESS (within floor, +0.38% = 0.5 SE) |
| cat6root_b1 | ~4860 | ~636 | 13.09% +/-0.48 | LOSSLESS (within floor, +0.18% = 0.3 SE) |

**Both cat9 and cat6 are within native E5's floor (<1 SE) => lossless-equivalent.**

FULL DEPTH-5 VERDICT (speed x lossless), clean B=1 temp-0.6 4-task:
- nativeE5: 17.8 tps / 0.137 / 3.11  (bar)
- cat9: 18.4 tps (+3.4%) / 0.144 (+5% verify) / 3.64 ; LOSSLESS -> WINS
- cat6: 18.51 tps (+4.0%) / 0.138 (~native) / 3.82 ; LOSSLESS -> WINS, CLEANER (zero verify tax, highest accept)

Both tree shapes BEAT native E5 on realized deploy-TPS AND are lossless within floor at clean B=1
(overturns the concurrency-confounded B=4 read). cat6 is the cleaner winner. The +~0.026s/step that
eats most of the accept gain is the committer/tree-propose overhead (ours) -> Stage D tuning target.
Raw rescore jsons in b1_depth5_raw/lossless/. Caveat: 40-turn sample (~4.9k pos/arm); margins <1 SE.
