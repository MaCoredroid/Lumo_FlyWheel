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

## DEVICE-COMMITTER LOSSLESS GATE (FR13_DEVICE_MULTIDRAFT=1, t0.6, B=4) — 2026-06-16

Pair: `dm_device` (FR13_DEVICE_MULTIDRAFT=1, device committer engaged — needle
`rejection_sampler:3066 FR13_DEVICE_MULTIDRAFT engaged`, log-once guard; fires on
every temp>0 commit, no silent host fallback, class-9 raises on disengagement) vs
`dm_hostref` (=0, host reference `_lumo_tree_canonical_multidraft_sample`). Same
cat9 tree (`num_speculative_tokens:9`, TREE_ATTN), same subset_b4_four, same seed.

**STREAM BYTE-IDENTITY (the decisive on-distribution evidence).** The device-committed
served stream is BYTE-IDENTICAL to the host-reference AND to native-E5 across ALL
1166/675 shared codex turns (verified token-for-token incl. every drift-window turn).
This is exactly what distribution-equivalence + paired per-request seeding predicts:
same residual-mix accept distribution + same RNG generator => same draws at t0.6.
=> device commits the SAME tokens as host/native; lossless by realized identity.

**BINDING per-token clear-margin gate (deploy-lossless, the within-floor BAR).**
`output/fr13_b4_onmode/deploy_lossless_device.json` (drift window: 8 generative
turns, 1422 forced positions; no-spec RECURRENT oracle p, RECURRENT_PATH_ENGAGED,
recurrent_decode_calls=135744, 96 GDN layers, within-proc deterministic):

| arm | clear-margin flips | rate | Wilson95 | verdict |
|-----|--------------------|------|----------|---------|
| dm_device | 110/1422 | 7.736% | [6.458%, 9.241%] | — |
| native-E5 (== device stream, byte-identical) | 110/1422 | 7.736% | [6.458%, 9.241%] | BAR |
| | | | `cat9_above_native_separated=false` | **LOSSLESS_within_floor** |

`within_floor=true`. native == device because the served stream is byte-identical
(constructive, not circular: both scored vs the same no-spec recurrent oracle on the
same forced ids). This is THE binding lossless gate per FR13_MEASURE_DEPLOYMENT_REGIME
(the per-token clear-margin instrument, NOT bag-TV; reference_scalar_metric_per_token_blindspot).

**Offline distribution-equivalence (ground truth).** `fr13_device_multidraft_offline_gate.py`
fresh re-run: A 22/22, B 22/22, C 22/22 (per-node accept probs match host reference
<=1e-9; closed-form output dist match; sampled freq within 6-sigma band, N=20000).

**temp-0.6 distributional drift TV (deploy-temp06-drift) — INSTRUMENT BLOCKED on the
pinned image (root-caused, not hand-waved).** The TV needs the SPEC-VERIFY q forced-
and-step-aligned to the deployment served stream. The canonical producer
`fr13_recurrent_decode_oracle.py capture-q-deploy` builds the spec engine with a custom
forced-decode logits processor (`_build_spec_llm` registers `logits_processors=[ForcedRecurrentAdapter]`
+ `speculative_config`). vLLM 0.19.2 (pinned cu130 image) REJECTS this:
`v1/sample/logits_processor/__init__.py:203-205` -> `ValueError: STR_SPEC_DEC_REJECTS_LOGITSPROCS`
("Custom logits processors are not supported when speculative decoding is enabled"),
unconditional, no env bypass. The HTTP `top_logprobs` capture-q paths (cmd_capture_q,
fr13_gold_margin_probe) ARE spec-compatible but FREE-RUN (off-distribution, NOT forced/
aligned to the pinned stream) so q!=p step-alignment is a fiction. No banked
capture_q_deploy artifact exists anywhere -> this producer never ran on this image.
The forced-aligned spec-verify q is only recoverable via a serve-side verify-q dump in a
re-boot of the locked TREE_ATTN serve (future); the binding lossless verdict does NOT
depend on it (flip-rate gate above is the BAR).

Artifacts: `output/fr13_b4_onmode/{rescore_dm_device.json, consolidated_device.json,
deploy_lossless_device.json}`; speed pair `output/fr13_device_multidraft/deploy_speed_{device,hostref}.json`
(device s/fwd 0.5875 vs host 0.7562 = ~22% faster, accept 3.53 vs 3.31).
