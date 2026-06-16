# FR13 codex-offload — ValidateClean results (GB10 vLLM-only proof + clean s/fwd)

Validates the contamination fix from commit a2ff61b6 (codex docker + inference_proxy
moved to alienware; GB10 runs ONLY vLLM). Two SHORT cat9 runs on the SAME boot
conditions, single-task subset (astropy__astropy-12907), AGENT_WALL_S bounded.
output/ is gitignored so the numbers are recorded here.

GB10 = gx10-edb9 (tailscale 100.103.10.122). alienware = mark-Alienware (100.83.202.36).

## 1. GB10 is vLLM-ONLY during the offloaded codex run (the proof the user wanted)

DURING-RUN GB10 snapshot taken at 00:43:51Z while the alienware codex container
`swe-codex-astropy__astropy-12907-1781570629` was LIVE (clean arm, OFFLOAD_CODEX=1):

- GB10 `docker ps`: ONLY `fr13-bigdenom-validate_clean (vllm/vllm-openai)` — no swe-codex-* container.
- GB10 `pgrep inference_proxy`: NONE (the proxy is on alienware).
- GB10 docker containers matching codex/swe: NONE.
- `nvidia-smi` compute apps: ONLY `VLLM::EngineCore` (89790 MiB). No codex/proxy on the GPU.
- The only GB10 process referencing "codex" is the thin SSH client forwarding
  `docker run codex-runner:v1 codex exec ...` to alienware (argv text, not a local process).

Cross-check ALIENWARE snapshot (same instant): the `swe-codex-...:codex-runner:v1`
docker AND the `inference_proxy` python (pid 2192653, listen 127.0.0.1:8023 ->
100.103.10.122:9950) are BOTH there. Contamination components (2)+(3) genuinely moved.

CONTRAST — contaminated arm (OFFLOAD_CODEX=0) GB10 snapshot at 01:01:07Z: GB10 shows
BOTH `swe-codex-...:codex-runner:v1` + `fr13-bigdenom-validate_contam:vllm` containers,
the inference_proxy python (pid 1549631) co-resident, and top-CPU shows codex + node +
proxy contending with VLLM::EngineCore. This is the contamination the offload removes.

## 2. codex genuinely ran on alienware + the measurement works through the offload

- Clean arm completed end-to-end: swerc=0, 15 pair-dumps fetched back (rsync resilient,
  OFFLOAD_FETCH_OK pair_dumps_back=15), all 15 non-empty (22087 served chars, 6902 out tok).
  10 served requests with per-request decode metrics captured on alienware + rsynced back.
- deploy-speed reads the GB10 vLLM /metrics LOCALLY through the offload (clean arm):
  s/fwd = d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total) = 0.24359
  draft_tokens/draft = 9.0 (cat9 tree engaged), accepted/draft = 3.34.
- deploy-lossless pair-dump path is intact: the 15 pair-dumps are the recurrent-oracle
  rescore input (a SEPARATE GB10 vLLM-only GPU phase), captured on alienware + fetched back.

## 3. Network resilience — blip injected mid-run, run SURVIVED

25s blip: alienware iptables-DROP egress to GB10:9950, 00:43:52Z -> 00:44:17Z (rule
cleared cleanly after, no leftover firewall on alienware).

- Link watchdog CLASSIFIED it correctly (offload_link_state.log):
  `LINK DOWN contig=15s (CLASSIFIED network-drop, not a model fork)` then `LINK up`.
  A blip is NOT mis-attributed as a degenerate #12 fork or a real failure.
- The codex container SURVIVED (still Up after the blip; the run continued to completion).
- The in-flight request spanned the blip: req0 received 00:43:50.486Z, completed 00:44:18Z
  (1s after recovery). wallclock_s=27.75 (inflated by the ~25s wire stall) but
  decode_sum_s=6.66 (NORMAL — same as comparable small requests). The wire stall
  inflated WALL but NOT the decode counter the s/fwd is derived from.
- Blip < OFFLOAD_LINK_DOWN_MAX_S (300s) => NO DEPLOY_SPEED_DISCARDED.flag, swerc=0:
  the window was recorded with a normal per-event s/fwd, the wire-stalled window was NOT
  recorded as a bad s/fwd. The local s/fwd held steady across the blip BY CONSTRUCTION
  (request_decode_time_seconds only advances while vLLM decodes; the blip just paused it).

## 4. Clean vs contaminated s/fwd (was the contamination real?)

Per-draft s/fwd (the canonical basis), MAX_NUM_SEQS=1 (B=1), short single-task window:
  clean (offloaded)    = 0.24359 s/fwd  (1588 drafts, accepted/draft 3.34)
  contaminated (co-loc) = 0.24296 s/fwd  (2085 drafts, accepted/draft 3.47)
  => per-draft basis is ~EQUAL (-0.3%, within noise) at B=1: the per-draft basis
     already normalizes out most of the contamination at B=1 single-seq.

Per-TOKEN decode rate (decode_sum_s / completion_tokens), contam-window-filtered:
  clean (offloaded)    = 59.36 ms/tok  (n=10)
  contaminated (co-loc) = 63.93 ms/tok  (n=11)
  => contaminated is ~7.7% slower per-token decode = the contamination IS real but
     MODEST at B=1. It shows at the per-token decode level; it is largely normalized
     out of the per-draft s/fwd. Expect a larger gap at B=4 (memory-bandwidth bound,
     273 GB/s unified Grace+Blackwell). The clean number is genuinely lower per-token.

HONEST CAVEAT: a single short single-task pair is suggestive, not a tight measurement.
The decisive contamination quantification is B=4 (MAX_NUM_SEQS_OVR=4 via the variant
vehicle), where the unified-memory bandwidth contention bites hardest; that is the
follow-up. What this run PROVES is the architecture: GB10 vLLM-only during the
offloaded codex run, codex on alienware, measurement (speed + lossless pair-dumps)
intact through the offload, and network resilience (blip survived, classified,
not mis-recorded).

## Artifacts (gitignored output/)
- output/fr13_bigdenom_swe/validate_clean/   (clean: during_run_gb10_snapshot.txt,
  during_run_alienware_snapshot.txt, during_run_blip.log, offload_link_state.log,
  deploy_speed_clean.json, proxy_pair_dumps/ x15, offload_request_metrics.jsonl)
- output/fr13_bigdenom_swe/validate_contam/  (contam: during_run_gb10_snapshot_contam.txt,
  deploy_speed_contam.json)

## Helpers added (committed)
- scripts/swe_x86_helpers/offload_validate_during_run.sh — DURING-RUN GB10/alienware
  snapshot + short blip injection + classification check (the ValidateClean prober).
- scripts/swe_x86_helpers/offload_sfwd_from_metrics.py — canonical s/fwd from a /metrics
  bracket pair (same basis as fr13_measure.py; does NOT change the measurement).
