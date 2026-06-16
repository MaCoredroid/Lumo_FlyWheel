# FR13 DEVICE MULTIDRAFT — GpuGate result (2026-06-16, branch fr13-speedfix)

Device-side temp>0 multidraft tree-rejection committer (HEAD device kernel
e377ff5c + launcher wiring fix 20638667). Boots, engages, distribution-lossless.

## BUILD
- Hook: scripts/fr10_phase4_patch_vllm_tree_gdn.py L8219-8346 (flag FR13_DEVICE_MULTIDRAFT,
  default OFF -> host reference `_lumo_tree_canonical_multidraft_sample` byte-identical to HEAD).
- Kernel: scripts/fr13_device_multidraft_kernel.py (SpecInfer/multi-draft residual-mix
  accept rule on-device: per-node source weights = overlaps/overlap_mass, accept =
  min(1,p[token]/q_mix_token), residual = max(p - q_mix_vocab,0)/mass, sampled from a
  torch DEVICE generator seeded per-request). NO [nodes x vocab] softmax DtoH (single-row
  on-device softmaxes + candidate gathers); NO numpy per-node interpreter loop.
  draft_probs!=None FAILS LOUD (no silent host fallback, bug-class 9).
- WIRING FIX (20638667): added `-e FR13_DEVICE_MULTIDRAFT` + `-e FR13_DEVICE_MULTIDRAFT_KERNEL`
  to scripts/fr13_launch_forked_fa2_tree_server.sh. HEAD had NOT passed the flag into the
  container, so the host-shell flag never reached the worker and the committer silently
  never engaged. The hook's kernel default was also a host path; pinned to
  /workspace/scripts/... (repo mount).

## OFFLINE DISTRIBUTION GATE (boot-free, CPU): PASS
scripts/fr13_device_multidraft_offline_gate.py
- A (per-node analytic objects: weights + accept_prob + residual): 22/22 within 1e-9
- B (closed-form node output distribution): 22/22 within 1e-9
- C (sampled token/accept frequencies within 6-sigma binomial band): 22/22
=> the device committer is distribution-equivalent to the host reference per-node.

## GPU BOOT GATE: PASS (prelaunch recover_host_memory + hygiene OK 106.8 GiB / swap 0)
Two paired cat9 boots, temp 0.6, top_p 0.95, seed 1313, MAX_NUM_SEQS=1 (deployment B=1),
identical 2-prompt SWE-4 probe, max_tokens 96. GB10, locked cat9 pipeline.

| arm                          | health | graph-capture | needle | crash | drafts | tok/draft | s/fwd     | accept/event |
|------------------------------|--------|---------------|--------|-------|--------|-----------|-----------|--------------|
| DEVICE  (FR13_DEVICE_MULTIDRAFT=1) | 432s   | OK 0.39GiB/7s | FIRED  | none  | 45     | 9.00      | 0.22896 s | 2.844        |
| HOSTREF (FR13_DEVICE_MULTIDRAFT=0) | 417s   | OK 0.25GiB/6s | n/a    | none  | 46     | 9.00      | 0.23228 s | 2.739        |

- Needle (device arm): "FR13_DEVICE_MULTIDRAFT engaged: device-side temp>0 multidraft
  committer (no [nodes x vocab] softmax DtoH, no per-node Python loop), n_req=1" -> ENGAGED,
  not a silent host fallback.
- s/fwd basis = d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total) (canonical,
  non-banned). before=0 both arms (fresh boot) -> clean delta.
- Per-position accept profiles device (36,34,25,18,15) vs hostref (38,35,21,18,14): consistent
  (cross-boot RNG-draw variation, distribution-lossless-not-byte; pos5+ = 0 = cat9 5-spine depth).

## DECISIVE vs HOST REFERENCE (probe trajectory)
device s/fwd 0.22896 < hostref s/fwd 0.23228 => DEVICE LOWER (faster) by ~1.4% on this arm.
Direction correct (the win). CAVEAT: 2-prompt probe, 45-46 events; a 1.4% delta is within the
GB10 cross-boot autotune/co-residency noise band. NOT a firm deployment number.

## NOT RUN (require multi-host deployment pipeline + a second oracle boot; GPU serialized)
- deploy-speed (cmd_deploy_speed): the CANONICAL deployment s/fwd. Reduces per-task
  vllm_metrics_pre/post brackets from a real codex SWE-Verified agent loop (offloaded to
  alienware, OFFLOAD_CODEX=1, DEPLOY_FORCE_TEMP=0.6). Not the 2-prompt probe.
- deploy-temp06-drift (cmd_deploy_temp06_drift): the binding distributional-lossless gate.
  Needs capture-q-deploy (spec verify q forced onto the codex served stream) + the no-spec
  RECURRENT oracle rescore of the SAME stream, then per-position TV vs the depth-5 native-E5
  floor. A multi-stage capture; not run here.

## INTERPRETATION
The device committer is distribution-equivalent to the host reference per-node (offline 22/22
within 1e-9), so on any served stream its temp-0.6 drift MUST land in the same within-floor band
as the host reference (it commits from the IDENTICAL distributions; only the RNG draws differ).
The probe s/fwd is lower in the right direction. The deployment-magnitude lever (toward chain5
~1.11x vs host-reference ~1.4x) is UNMEASURED -- it needs the full codex SWE deploy-speed run.
