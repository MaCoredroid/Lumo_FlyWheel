# FR14 constant-train audit — 2026-08-16

Mechanical record for the model-bound constant train that re-points the
fixed32 serving stack from `/models/qwen3.6-27b-fp8` (`qwen3.6-27b`) to
`/models/qwen3.8-27b-nvfp4` (`qwen3.8-27b-nvfp4`).

## Decisions

**Served-model-name: `qwen3.8-27b-nvfp4`** (not the shorter `qwen3.8-27b`).
The FP8-3.8 baseline arm is a structurally identical serve of the same base
model, so a bare `qwen3.8-27b` would let an FP8 boot satisfy every Prometheus
label check, trace `model` field and QC comparison meant for the NVFP4 arm —
exactly the decomposition the campaign exists to protect. With the suffix a
mis-pointed serve 404s on request one. Applied consistently across the serve
line, the contract argv, all Prometheus label builders, the chat `model`
field and the agent tag (`qwen3.8-27b-nvfp4::qwen-code-0.19.4::q38-a`).

**Floor**: re-derived by SUMMING real safetensors tensor spans, never by
scaling an FR13 constant. See `floor_ledger.json` / `floor_derivation.json`
in this directory and `scripts/fr13_hardware_floor_ledger.py
--derive-from-checkpoint`, which reproduces the arithmetic on demand and
exits 2 on drift.

| term | FR13 (3.6-fp8) | FR14 (3.8-nvfp4) | why |
|---|---:|---:|---|
| target_model_bytes | 24,382,399,488 | 17,831,788,928 | NVFP4 W4A4-g16 MLPs + FP8 attn/GDN + BF16 conv/norms |
| mtp_forward_bytes_per_pass | 477,199,744 | 849,398,784 | MTP shard is BF16 in every NVFP4 repack (was FP8) |
| FULL_HEAD_BYTES | 2,542,796,800 | 2,542,796,800 | lm_head is BF16 after the FR14 surgery — unchanged |
| SUBSET_HEAD_BYTES | 671,088,640 | 671,088,640 | unchanged |
| FIXED32 root_64k bytes | 32,666,638,208 | **27,977,022,848** | 0.856x, not 0.5x |
| FIXED32 floor_ms | 119.658015414 | **102.479937172** | |
| ONE_SIDED_U95_CAP_MS | 137.6067177261 | **117.8519277478** | 1.15x floor, PROVISIONAL |
| K64/root0 arm | 34,538,346,368 / 126.51408926 | 29,848,731,008 / 109.336011018 | |
| full-vocab arm | 42,025,179,008 / 153.9383846446886 | 37,335,563,648 / 136.7603064029304 | cap 157.27435236336996 |
| FR13_DRAFT_HEAD_FP8 arm | 30,989,326,208 / 113.514015414 | **RETIRED (exit 2)** | served lm_head is BF16 |
| FR13_COMPUTE_MS_PER_ROW | 0.54 | 0.54 | fp8-era MEASURED value, conservative under NVFP4 |

Derivation rule (stated so it can be re-run): `target_model_bytes` = sum of
tensor byte-spans of every `model.language_model.*` tensor EXCEPT
`embed_tokens` (i.e. the 64 decoder layers, 1,616 tensors = 17,831,778,688 B,
plus the final norm = 10,240 B). `lm_head` is counted separately as
FULL_HEAD_BYTES, `embed_tokens` is a row gather not a stream, and
`model.visual.*` (921,460,192 B) is not on the text decode path.
`mtp_forward_bytes_per_pass` = all 15 tensors of `model_mtp.safetensors`.

**FR13 fossil found while doing this**: the FR13 pin 24,382,399,488 was
layers-ONLY — it omitted `model.language_model.norm`. The same rule applied
to the 3.6 checkpoint yields 24,382,409,728 (+10,240 B = +0.0000375 ms of
floor). FR14 counts the final norm. Immaterial numerically; recorded so the
delta is never mistaken for a model effect.

## Files touched

87 modified, 3 new.

### New

- `results/fr14_nvfp4_port_20260816/floor_derivation.json`
- `results/fr14_nvfp4_port_20260816/floor_ledger.json`
- `scripts/fr14_gen_model_manifest.py`

### Modified

- `model_registry.yaml`
- `scripts/fr13_b4_campaign_driver.sh`
- `scripts/fr13_b4_clean_measure_workflow.js`
- `scripts/fr13_b4_deployment_sweep_workflow.js`
- `scripts/fr13_b4_honest_floor.py`
- `scripts/fr13_bigdenom_swe_serve.sh`
- `scripts/fr13_bigdenom_swe_serve_variant.sh`
- `scripts/fr13_build_wide_shapes_b4_workflow.js`
- `scripts/fr13_committer_bv64_real_result.py`
- `scripts/fr13_cutlass_b4_pass.py`
- `scripts/fr13_cutlass_streamk_pass.py`
- `scripts/fr13_depth_acceptance.py`
- `scripts/fr13_draft_head_fp8_gate.py`
- `scripts/fr13_draft_head_fp8_timing.py`
- `scripts/fr13_fixed32_b1_nsys_profile.sh`
- `scripts/fr13_fixed32_contract.py`
- `scripts/fr13_fixed32_floor_timers_seq.sh`
- `scripts/fr13_fixed32_sfwd_fusion_env.sh`
- `scripts/fr13_floor_gate.py`
- `scripts/fr13_hardware_floor_ledger.py`
- `scripts/fr13_launch_forked_fa2_tree_server.sh`
- `scripts/fr13_measure.py`
- `scripts/fr13_measure_orchestrate.sh`
- `scripts/fr13_qrow32_split2_timing.py`
- `scripts/fr13_run_b1_cfwd_logit_direct_live_gate.sh`
- `scripts/fr13_run_b1_cfwd_logit_direct_timing.sh`
- `scripts/fr13_run_b1_cutlass_streamk_live_gate.sh`
- `scripts/fr13_run_b1_cutlass_streamk_timing.sh`
- `scripts/fr13_run_b1_draft_head_fp8_timing.sh`
- `scripts/fr13_run_b1_draft_head_m32_timing.sh`
- `scripts/fr13_run_b1_fa2_qrow32_gqa_pair_timing.sh`
- `scripts/fr13_run_b1_fp8_quant_regcache_timing.sh`
- `scripts/fr13_run_b1_k64_physical32_fullstack_pair.sh`
- `scripts/fr13_run_b1_k64_qrow16_divfree_live_gate.sh`
- `scripts/fr13_run_b1_k64_qrow16_sfwd_stack_timing.sh`
- `scripts/fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh`
- `scripts/fr13_run_b1_k64_qrow32_split2_live_gate.sh`
- `scripts/fr13_run_b1_k64_taw_source_v7_gate.sh`
- `scripts/fr13_run_b1_kernel_live_gate.sh`
- `scripts/fr13_run_b1_sfwd_fusion_boot_diag.sh`
- `scripts/fr13_run_b1_sfwd_prior_reuse_gate.sh`
- `scripts/fr13_run_b1_sfwd_state_fusion_gate.sh`
- `scripts/fr13_run_b1_sfwd_state_fusion_timing.sh`
- `scripts/fr13_run_b1_target_sfwd_exact4_timing.sh`
- `scripts/fr13_run_b1_u8_cfwd_sfwd_stack_timing.sh`
- `scripts/fr13_run_b4_cutlass_persistent_m128_live_gate.sh`
- `scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh`
- `scripts/fr13_run_b4_draft_head_m32_timing.sh`
- `scripts/fr13_run_b4_fa2_qrow32_gqa_pair_timing.sh`
- `scripts/fr13_run_b4_gdn_bv8_timing.sh`
- `scripts/fr13_run_b4_gdn_single_launch_width4_timing.sh`
- `scripts/fr13_run_b4_gdn_wide_live_gate.sh`
- `scripts/fr13_run_b4_gqa_width4_timing.sh`
- `scripts/fr13_run_b4_mamba_narrow_within_run_pair.sh`
- `scripts/fr13_run_b4_sfwd_state_fusion_live_gate.sh`
- `scripts/fr13_run_b4_tail23_all_parent_live_gate.sh`
- `scripts/fr13_run_b4_tail23_hydra27_k64_m128_stack.sh`
- `scripts/fr13_run_b4_taw_width4_timing.sh`
- `scripts/fr13_run_committer_bv64_real.sh`
- `scripts/fr13_run_gdn_single_launch_live_gate.sh`
- `scripts/fr13_taw_b1_credential.py`
- `scripts/run_swe_bench_q36_a.py`
- `tests/test_fr13_b1_composed_stack.py`
- `tests/test_fr13_b4_gdn_bv8_production.py`
- `tests/test_fr13_b4_honest_floor_artifact.py`
- `tests/test_fr13_b4_width4_window.py`
- `tests/test_fr13_campaign_task_budget_cap.py`
- `tests/test_fr13_committer_bv64_real_runner.py`
- `tests/test_fr13_cutlass_b4_k64_profile.py`
- `tests/test_fr13_cutlass_b4_production.py`
- `tests/test_fr13_cutlass_streamk_gate_wiring.py`
- `tests/test_fr13_cutlass_wave_binary.py`
- `tests/test_fr13_draft_head_fp8.py`
- `tests/test_fr13_fa2_qrow32_gqa_pair_gate.py`
- `tests/test_fr13_fixed32_b4_campaign_provenance.py`
- `tests/test_fr13_fixed32_floor_propagation.py`
- `tests/test_fr13_fixed32_gdn_batch_wide_bv.py`
- `tests/test_fr13_fixed32_pretask_metrics.py`
- `tests/test_fr13_fixed32_qwen_completion_classes.py`
- `tests/test_fr13_fixed32_token_reconciliation.py`
- `tests/test_fr13_fixed32_trace_provenance.py`
- `tests/test_fr13_hardware_floor_ledger.py`
- `tests/test_fr13_nsys_launcher_wiring.py`
- `tests/test_fr13_qrow32_split2_timing.py`
- `tests/test_fr13_sfwd_fusion_boot_diag.py`
- `tests/test_fr13_treeconv_zero_tail_credential.py`
- `tests/test_inference_proxy.py`

## Grep audit — remaining `qwen3.6-27b` references

Zero remaining on the serving path. Every hit below is either an intentional
documentary/negative reference or an FR13-and-earlier one-off that FR14
deliberately did not re-point (they target a checkpoint that still exists on
disk, and re-pointing them would falsely imply they had been ported and
verified against NVFP4).

| file:lines | why skipped / retained |
|---|---|
| `model_registry.yaml:14,17` | the historical qwen3.6-27b Track-B entry, deliberately retained so the 3.6 checkpoint stays bootable for A/B |
| `scripts/fr10_draft_token_parity_probe.py:261` | FR10-era launcher/probe/reference |
| `scripts/fr10_gdn_tree_algebra_reference.py:19` | FR10-era launcher/probe/reference |
| `scripts/fr10_launch_speed_server.sh:347` | FR10-era launcher/probe/reference |
| `scripts/fr10_phase4_launch_tree_capture_probe.sh:24` | FR10-era launcher/probe/reference |
| `scripts/fr10_quick_decode_tps_probe.py:23` | FR10-era launcher/probe/reference |
| `scripts/fr12_branch_path_oracle_probe.py:333` | FR12-era probe |
| `scripts/fr12_deliverable_swe4_probe.py:278` | FR12-era probe |
| `scripts/fr12_fp8_full_gemm_batch_invariance_probe.py:320` | FR12-era probe |
| `scripts/fr13_accept_speed_probe.py:55,60` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_apc_gate0.sh:241` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_apc_lossless_ab.sh:241,369` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_apc_replay.py:49,373` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_apc_stale_or_not_gate.sh:206` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_apc_teacher_forced_logit_gate.py:182,184,200,201` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_apc_temp06_precheck.sh:65` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_attn_mgeom_bench.py:39` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_branch_seed_localize.sh:23` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_branch_token_oracle.py:408` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_build_prewarm_corpus.py:43` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_commit_trace_analyze.py:15,59` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_conv_compute_residual.py:50` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_e2e_measure.py:279` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_fixed32_contract.py:195,310` | documentary comments only (records the FR13 name and the 3.6/3.8 vocab.json byte-identity that carries the DVK block map) |
| `scripts/fr13_fork_margin_boot_capture.sh:15` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_fp8_inproj_mkey_microbench.py:16` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_g1_kernel_confirm.sh:18` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_garble_gate.py:20` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_garble_gate_shot.sh:30` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_garble_ladder_diff.py:13` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_garble_ladder_drive.py:46` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_garble_localizer.py:31` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_garble_prob_probe.py:25,112` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_garble_test.sh:59` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_gdn_subop_mab_drive.py:85` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_geodetic_reproducer.py:23` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_gold_margin_probe.py:31` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_inproj_ba_mkey_microbench.py:7` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_k1_boot_capture.sh:14` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_launch_native_mtp_server.sh:17,127` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_node5_ladder_drive.py:87` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_nospec_cache_swe.sh:25` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_npad_boot_capture.sh:16` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_oracle_capture.py:34` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_oracle_stream_teacher_force.py:33` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_pyspy_profile_cat6.sh:65` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_recurrent_decode_oracle.py:85,86` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_reshape_boot_capture.sh:17` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_s1_committer_capture_bench.py:17` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_serialization_shot.sh:55` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_shape_gate.sh:67` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_speed_phase0.sh:18,60` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_swe_stream_to_oracle_src.py:41` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_synth_realized_tps.sh:61` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_throughput_remeasure.sh:40,51` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_verify_bisect_probe.py:41` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr13_wa_capture.py:13` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/fr9_gate_b_greedy_probe.py:25` | FR9-era probe |
| `scripts/launch_qwen36_ablation_point.py:4,31,32,130` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/measure_spec_per_position.py:27,494` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/measure_spec_teacher_forced.py:486` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/probes/SEEDED2TURN_RUNBOOK.md:34` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/fr13_fb_probe.sh:27` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/fr13_garble_replay.sh:11` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/fr13_garble_scan.py:57` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/fr13_rp2_order_probe.sh:40` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/route_probe.sh:45` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/route_probe_payload.json:2` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/route_probe_payload.meta.json:4` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/route_probe_reduce.py:23,28,242` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/seeded2turn_reduce.py:40,49` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/probes/seeded2turn_run.sh:67,565` | FR13 probe kit (route/seeded2turn/garble/fb) -- one-off diagnostics |
| `scripts/run_codex_experiment.py:221` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/spec_speed_probe.py:189` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/swe_eval_offload.py:248` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/swe_eval_x86_worker.py:56` | FR13-era diagnostic / probe / microbench / capture one-off |
| `scripts/swe_x86_helpers/relaunch_qwen36_E.py:62` | x86 offload helpers; relaunch_qwen36_round.py in particular MUST NOT be re-pointed (recon checklist item 4: its quant_method 'fp8' + weight_block_size [128,128] would corrupt an NVFP4 config) |
| `scripts/swe_x86_helpers/relaunch_qwen36_round.py:1281,6157` | x86 offload helpers; relaunch_qwen36_round.py in particular MUST NOT be re-pointed (recon checklist item 4: its quant_method 'fp8' + weight_block_size [128,128] would corrupt an NVFP4 config) |
| `scripts/swe_x86_helpers/run_cnb_v4a_x86.py:119` | x86 offload helpers; relaunch_qwen36_round.py in particular MUST NOT be re-pointed (recon checklist item 4: its quant_method 'fp8' + weight_block_size [128,128] would corrupt an NVFP4 config) |
| `tests/test_fr13_nsys_launcher_wiring.py:95` | NEGATIVE assertion -- the old path must NOT appear in the launcher |

## Contract regen

`scripts/fr13_fixed32_contract.py` model block regenerated wholesale by
`scripts/fr14_gen_model_manifest.py` (committed, reproducible; `--check`
re-verifies an existing pin). Result:

- `MODEL_ROOT = /models/qwen3.8-27b-nvfp4`
- `MODEL_FILES`: **16 names, no `layers-N`** — this checkpoint is a single
  `model.safetensors` (23,839,093,880 B, sha256 `bbf67537…`) plus
  `model_mtp.safetensors` (849,400,392 B) and the config/tokenizer set.
- `MODEL_CANONICAL_SHA256 = a95cb7227fcccb335f5549b7df7264a332b03bdd906da69aeec3fc29e22a0fa8`
- `MODEL_TEXT_CONFIG_VOCAB_SIZE = 248_320` (unchanged)
- `MODEL_VOCAB_JSON_SHA256 = ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` — **new**

### dot-dirs vs dot-files (the question the port raised)

`_pinned_model_files` walks `model_root.iterdir() if path.is_file()`. A metadata
DIRECTORY (`.cache/huggingface/download/…`) is therefore excluded by
construction — the served 3.6 dir carries one too and it never appeared in the
pinned set — while dot-FILES ARE members; FR13 already pinned `.gitattributes`
on exactly that rule. Nothing about the rule changed, only the names. Verified
by calling `_pinned_model_files(MODEL_ROOT)` against the real directory: 16
files, digest matches.

The two surgery sidecars are pinned **deliberately**:
`.lumo_lmhead_surgery.json` (src/dst sha256 + sizes) and
`config.json.pre_lmhead_surgery.bak` (the upstream `quantization_config` with
`lm_head` still in `group_0`) are the provenance of the only mutation this
checkpoint carries over its pinned upstream revision
(`unsloth/Qwen3.8-27B-NVFP4 @ 16b6615af3548b88e2d8e382457bc705b00479cf`).
Deleting either must fail the contract, not pass quietly.

### Tokenizer-hash assertion (new)

`vocab_size` is a WIDTH check, not an IDENTITY check, and the K64 draft-vocab
block map indexes `lm_head` rows by token id — a reordered vocabulary of the
same width would pass every boot assertion and show up only as a silently
degraded accept rate. `tokenizer.json` is NOT comparable across 3.6/3.8 by file
bytes (the tokenizers library changed how merges serialise, 3.8 adds 7
audio/TTS specials inside the reserved padding range, and the pre_tokenizer
regex changed), but `vocab.json` is **byte-identical** across
`qwen3.6-27b-fp8`, `qwen3.8-27b-fp8` and `qwen3.8-27b-nvfp4` — verified
2026-08-16, sha `ce99b4cb…`, the same digest FR13 already pinned for the 3.6
dir. That sha is now asserted next to the vocab_size check inside
`_pinned_model_files`, and it is what carries the DVK block map
(`fr13_dvk_subset_blocks.json @85dffa58…`) across the model swap. The DVK block
map and `FORWARD_GRAPH_STRUCTURAL_SIGNATURES` are otherwise left pinned as-is
(topology identical: 64 layers, 48 GDN / 16 full-attn, 24 Q / 4 KV heads,
head_dim 256, vocab 248,320).

## New fail-loud guards

1. **fp8-lever refusal** (`fr13_launch_forked_fa2_tree_server.sh`, placed after
   the existing cross-kernel preflights so their more specific messages still
   win, and well before `mkdir LOG_DIR` / `docker run`). If any of
   `FR13_GB10_FP8_GEMV_CFG` != 0, `FR13_FIXED32_B1_FP8_QUANT_REGCACHE` != 0,
   `FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO` non-empty,
   `FR13_FIXED32_CUTLASS_WAVE` != `stock`, or `FR13_FIXED32_CUTLASS_WAVE_SO`
   non-empty is armed while the served `config.json`'s
   `quantization_config.quant_method` != `fp8`, the launcher exits 2 naming the
   armed levers. The levers are NOT deleted — re-point them at the nvfp4
   dispatch site, or serve an fp8 checkpoint, and they arm again unchanged.
   Behaviour-tested offline: NVFP4 + no levers → pass; NVFP4 + `CUTLASS_WAVE`
   → exit 2; NVFP4 + `GEMV_CFG=1` → exit 2; **fp8 checkpoint + `CUTLASS_WAVE`
   → pass**.
2. **FR13_DRAFT_HEAD_FP8 retirement**: the launcher case block, the floor-timer
   sequence, `fr13_run_b1_kernel_live_gate.sh` and `fr13_measure.py`'s deploy
   -speed ledger all refuse the arm with a named message. Rationale: the arm
   priced five FP8 qweight + FP32-scale draft-head reads; the served lm_head is
   BF16 (unsloth shipped FP8 per-channel, this vLLM builds `lm_head` as an
   unquantized `ParallelLMHead` and refused it, and the FR14 surgery dequantised
   it to BF16), so the arm would pin a floor the hardware cannot realise.
3. **Memory preflight** (launcher + `fr13_b4_campaign_driver.sh`): `sync` then
   `sudo -n sysctl vm.drop_caches=3 || true`, then refuse if
   `MemFree < gpu_memory_utilization x MemTotal`. `MemAvailable` counts
   reclaimable page cache as available; the GB10 unified-memory allocator does
   not — the first NVFP4 boot attempt was refused at 32/117.5 GiB free purely
   because of download page cache. `sudo -n` is sanctioned and deliberately
   non-interactive so an unattended campaign never stalls on a password prompt;
   if the sudoers rule is absent the drop is skipped and the gate still fires,
   loudly, before ~8 minutes of engine boot are spent.

## Offline verification performed

- `fr13_fixed32_contract.py external-manifest` **succeeded end to end** against
  the real model dir, the pinned docker image (`docker image inspect`, id
  `sha256:ffa30d66…`, linux/arm64) and the pinned baseline FA2 `.so`.
  `overall_canonical_sha256 = 414a4c8f94bb45cf64ec018ced4094a7e515f1466ac929711a36e31391ae58ea`.
  NOTE ON THE FA2 INPUT: the `.so` at
  `<repo>/output/auto_research/…/_vllm_fa2_C.abi3.so` inside the **main**
  checkout is 301,219,928 B / `97fa2519…`, which does NOT match the contract pin
  (299,183,936 B / `f51e23c5…`). The pinned bytes DO exist on this box, at
  `/home/mark/shared/lumoFlyWheel-fa2-suffix-only/output/fr13_fa2_suffix_fc855e59_build/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so`,
  and the manifest was produced with them hard-linked into a scratch tree at the
  expected relative path. This is a pre-existing local-artifact state issue,
  NOT a consequence of the model swap, and it must be reconciled before the next
  GPU boot because every instrument runs `external-manifest --repo $PWD`.
- `_pinned_model_files(MODEL_ROOT)` called directly: 16 files, digest match.
- `fr13_hardware_floor_ledger.py --derive-from-checkpoint`: PASS (all three
  pinned byte terms re-derived from the safetensors headers on disk).
- `bash -n` on every touched shell script: clean.
- `python -m py_compile` on every touched python file: clean.
- `node --check` on the three touched `.js` workflow files: clean.
- `yaml.safe_load(model_registry.yaml)`: clean.

## Targeted test runs

`TMPDIR=/home/mark/shared/tmp-scratch PYTHONPATH=$PWD/src` over 32 test modules
covering every floor constant, the contract, the launcher wiring, the metric
label parsers and the retired arm: **805 passed, 3 skipped, 19 failed**.

All 19 failures reproduce at HEAD (verified by stashing the entire change set
and re-running the same selection), i.e. **zero regressions**:

| test | pre-existing cause |
|---|---|
| `test_fr13_b1_composed_stack.py::test_combined_sfwd_gate_validates_target_pass_against_repo_patch` | fixture `SimpleNamespace` lacks `direct_nodegroup8`, which `fr13_sfwd_conv_postprep_gate.py:594` requires |
| `test_fr13_treeconv_zero_tail_credential.py` (12) | committer boundary-snapshot key set moved ahead of the fixture (`direct_metadata_*`, `metadata_fusion_*`) |
| `test_fr13_dfwd_k64_fp8_selector.py::test_selector_accepts_exact_b1_b4[1,4]` | the selector pins `fr10_phase4_patch_vllm_tree_gdn.py` at `0696bfc5…`; HEAD is `a61b1d73…` (stale FR13 credential) |
| `test_fr13_fixed32_trace_provenance.py::test_fixed32_runtime_manifest_includes_qwen_settings` | requires local `.cache/huggingface` + `csrc` artifacts absent from this worktree |
| `test_fr13_fixed32_ingress_wiring.py` (3) | two need a `.venv` inside this worktree; one asserts a heredoc shape unrelated to the model |

Never run (house rule): `tests/test_codex_long_assets.py`.

### Test fixtures re-placed with the floor (not silently retargeted)

Two synthetic fixtures encoded an operating point relative to the FR13 floor and
cap. Leaving them fixed would have turned an eligibility assertion into a
tautology, so they were moved to the same RELATIVE position under the FR14
numbers, with the reason recorded in-line:

- `tests/test_fr13_qrow32_split2_timing.py`: per-task wall 120–123 ms → 103–106 ms.
- `tests/test_fr13_b1_composed_stack.py`: phase breakdown 80/20/10/20 (wall 130,
  u95 135) → 68/18/9/16 (wall 111, u95 115).

## Deliberately NOT changed

- **`scripts/fr13_b4_honest_floor.py` + `tests/test_fr13_b4_honest_floor_artifact.py`**
  — frozen at FR13, with an explicit header saying so. The script is the EMITTER
  of the sha-pinned `results/fr13_b4_honest_floor_20260814/` artifact and every
  anchor in it (232.360 ms B1 wall, 4.33 ms FA2 roofline, the sealed ratios) is
  a Qwen3.6-FP8 MEASUREMENT. Re-pointing only its weight constants would produce
  a half-FR13/half-FR14 artifact that reads as an FR14 result. The FR14 honest
  floor needs new measured anchors; the geometry-only +7.117 ms/request term at
  C=18,031 does carry over (heads, head_dim, GDN dims and vocab are identical).
- **`results/fr13_fixed32_b1_nsys_20260731T013952Z_curated/floor_ledger.json`**
  (v1, sha `3507bc71…`) and its immutability test — untouched, as directed.
- **`results/fr13_hardware_floor_correction_20260731/floor_ledger.json`** (v2)
  — untouched. Its binding test was rewritten from `published == build_ledger()`
  to a historical-consistency check, because `build_ledger()` now emits the FR14
  ledger; the live binding moved to
  `results/fr14_nvfp4_port_20260816/floor_ledger.json`.
- **`scripts/fr10_phase4_patch_vllm_tree_gdn.py`** — one literal was swept and
  then REVERTED. The value lives inside the retired fp8-draft-head OPT block
  alongside two other FR13 constants; changing one of three made the block
  internally inconsistent, and the file's sha256 is pinned by
  `fr13_dfwd_k64_fp8_selector.py`. The whole block is FR13 history and stays so.
- **`FIXED32_B4_KV_CACHE_MEMORY_BYTES` (46 GiB)** — unchanged. KV geometry is
  identical (4 KV heads x head_dim 256, bf16) so the 176k-token pool still
  covers four ~40k-token working sets, and NVFP4 weights free ~4.7 GB rather
  than consuming more. Re-confirm against a real boot log's reported KV pool.
- **`scripts/swe_x86_helpers/relaunch_qwen36_round.py`** — must NOT be
  re-pointed (its `quant_method: 'fp8'` + `weight_block_size: [128,128]` config
  fix would corrupt an NVFP4 config).

## Must be verified on the next GPU boot

1. The pinned baseline FA2 `.so` must be restored to
   `<repo>/output/auto_research/…/_vllm_fa2_C.abi3.so` (299,183,936 B /
   `f51e23c5…`), or every instrument's `external-manifest --repo $PWD` fails.
2. PID1 argv equality: `vllm serve /models/qwen3.8-27b-nvfp4
   --served-model-name qwen3.8-27b-nvfp4 …` vs `expected_pid1_argv`.
3. The unified-memory preflight's arithmetic against what the engine actually
   demands at `GPU_UTIL=0.70` on a 117.5 GiB pool, and that
   `sudo -n sysctl vm.drop_caches=3` is permitted in the campaign's sudoers.
4. `FR13_COMPUTE_MS_PER_ROW` — re-measure on the first FR14 B1 profile; 0.54 is
   the fp8-era value, retained as a conservative (high) bound.
5. The provisional `ONE_SIDED_U95_CAP_MS = 1.15 x floor` — the FR14 objective
   bar is Mark's open ruling.
6. Prometheus label plumbing end to end: every counter bracket now reads through
   `model_name="qwen3.8-27b-nvfp4"`.
