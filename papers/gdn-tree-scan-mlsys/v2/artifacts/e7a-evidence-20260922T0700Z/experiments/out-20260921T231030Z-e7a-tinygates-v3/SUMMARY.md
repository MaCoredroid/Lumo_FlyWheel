# E7a run summary — `out-20260921T231030Z-e7a-tinygates-v3`

Manifest status **completed**; start/end source hashes match: **yes**; image `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`; torch 2.11.0+cu130 / triton 3.6.0; GPU NVIDIA GB10 cc [12, 1]; driver `NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  59`.
Production kernel flags: {"FIXED32_MODE": null, "scan_align_on": false, "npad_invariant_on": false, "parent_gather_on": false, "hc_internal_on": false, "BV_prod": 16, "num_warps_prod": 8, "tf32_matmul_allowed": false, "replay_h0_source_column": 15, "scan_h0_source_column": 0}. Triton cubins hashed: 76.
Source sha256: `prod_kernel`=d9dd0c697b46…, `legacy_wy`=e7c2a2a8b658…, `e7a_core`=868d55db26eb…, `e7a_kernels`=15c353625218…, `e7a_device`=70e475e3998c…, `native_fused_sigmoid_gating`=000ab8996af9…, `native_fused_recurrent`=3a2a3c5245ac…

**Timing attribution:** NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['2770990'] seen; sampled observation, not a guarantee) — window ['2026-09-21T23:10:30.574000', '2026-09-21T23:16:04.728000'], util samples 335, zero-util fraction 0.958, samples >5 %: 11, compute PIDs seen: {'2770990': {'samples': 324, 'names': ['python3']}}, container PIDs: ['2770990'], coverage gaps: 0.

All mechanisms B/C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems. Historical payloads are dependent historical operand sets (one topology, B1), not independent prefixes; synthetic rows are labeled. Rows marked **INVALID** contain non-finite values in candidate or reference and carry no error value.

## Operand sets

| id | label | provenance | topology | n | depth | g_min | cum_g_min | P_min | min visible decay ratio | fp32 underflow risk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | synthetic[tiny-gates] seed=20260921 n=12 | SYNTHETIC seed=20260921 regime=tiny-gates | `[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` | 12 | 11 | -97.553 | -911.894 | 0 | 0 | yes |
| 2 | synthetic[tiny-gates] seed=20260921 n=6 | SYNTHETIC seed=20260921 regime=tiny-gates | `[-1, 0, 1, 2, 3, 4]` | 6 | 5 | -108.963 | -548.486 | 6.24e-239 | 3.44e-195 | yes |
| 3 | synthetic[tiny-gates] seed=20260921 n=10 | SYNTHETIC seed=20260921 regime=tiny-gates | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -115.767 | -519.230 | 3.17e-226 | 2.12e-192 | yes |

## References vs fp64 oracle (same rounded operands)

| id | native_sg out32 | native_sg state | native_pk out32 vs oracle | native_pk state vs oracle | native_pk state vs ITS OWN oracle (beta bf16-rt + div/sqrt) | seam magnitude fp64 (oracle_pk vs oracle, state) | pk vs sg out16 exact | sg fp32-io vs bf16-io state max_abs |
|---|---|---|---|---|---|---|---|---|
| 1 | abs=4.38e-09 rel=9.2e-08 ulp=61 | abs=9.2e-08 rel=1.44e-07 ulp=5 | abs=6.56e-05 rel=0.001 ulp=63254 | abs=0.001 rel=0.002 ulp=63533 | abs=7.46e-08 rel=1.17e-07 ulp=4 | abs=0.001 rel=0.002 | 0.750 | 0 |
| 2 | abs=4.16e-09 rel=1.2e-07 ulp=77 | abs=6.87e-08 rel=1.17e-07 ulp=6 | abs=7.93e-05 rel=0.002 ulp=63150 | abs=0.001 rel=0.002 ulp=63529 | abs=7.6e-08 rel=1.29e-07 ulp=4 | abs=0.001 rel=0.002 | 0.758 | 0 |
| 3 | abs=4.49e-09 rel=1.46e-07 ulp=112 | abs=8.94e-08 rel=1.6e-07 ulp=5 | abs=5.7e-05 rel=0.002 ulp=62596 | abs=0.001 rel=0.002 ulp=63144 | abs=8.02e-08 rel=1.44e-07 ulp=4 | abs=0.001 rel=0.002 | 0.749 | 0 |

## Stage 1 — verifier outputs and correction factors (vs fp64 oracle; out16 exact fraction vs native spec-update kernel)

| id | mechanism | out32 vs oracle | out16 exact vs native_sg | factors U vs oracle u |
|---|---|---|---|---|
| 1 | A_prod | abs=4.38e-09 rel=9.2e-08 ulp=61 | 1.000 | — |
| 1 | B_legacyWY fp32-closed | abs=4.23e-09 rel=8.88e-08 ulp=70 | — | — |
| 1 | B_legacyWY bf16-bnd | abs=0.000163 rel=0.003 ulp=1329514 | — | — |
| 1 | B_fs[ieee] | abs=8.82e-09 rel=1.85e-07 ulp=123 | 1.000 | abs=1.96e-07 rel=8.52e-08 ulp=4 |
| 1 | C_nm[ieee] | abs=8.82e-09 rel=1.85e-07 ulp=123 | 1.000 | abs=1.96e-07 rel=8.52e-08 ulp=4 |
| 1 | B_fs[tf32] | abs=7.64e-05 rel=0.002 ulp=200910 | 0.691 | abs=1.96e-07 rel=8.52e-08 ulp=4 |
| 1 | C_nm[tf32] | abs=7.64e-05 rel=0.002 ulp=200910 | 0.691 | abs=1.96e-07 rel=8.52e-08 ulp=4 |
| 2 | A_prod | abs=4.16e-09 rel=1.2e-07 ulp=77 | 1.000 | — |
| 2 | B_legacyWY fp32-closed | abs=5.14e-09 rel=1.48e-07 ulp=97 | — | — |
| 2 | B_legacyWY bf16-bnd | abs=7.94e-05 rel=0.002 ulp=1121932 | — | — |
| 2 | B_fs[ieee] | abs=7.72e-09 rel=2.22e-07 ulp=43 | 1.000 | abs=1.78e-07 rel=7.54e-08 ulp=4 |
| 2 | C_nm[ieee] | abs=7.72e-09 rel=2.22e-07 ulp=43 | 1.000 | abs=1.78e-07 rel=7.54e-08 ulp=4 |
| 2 | B_fs[tf32] | abs=5.11e-05 rel=0.001 ulp=186679 | 0.697 | abs=1.78e-07 rel=7.54e-08 ulp=4 |
| 2 | C_nm[tf32] | abs=5.11e-05 rel=0.001 ulp=186679 | 0.697 | abs=1.78e-07 rel=7.54e-08 ulp=4 |
| 3 | A_prod | abs=4.49e-09 rel=1.46e-07 ulp=112 | 1.000 | — |
| 3 | B_legacyWY fp32-closed | abs=4.69e-09 rel=1.52e-07 ulp=96 | — | — |
| 3 | B_legacyWY bf16-bnd | abs=0.000113 rel=0.004 ulp=1550904 | — | — |
| 3 | B_fs[ieee] | abs=1.79e-08 rel=5.82e-07 ulp=104 | 1.000 | abs=1.83e-07 rel=7.93e-08 ulp=4 |
| 3 | C_nm[ieee] | abs=1.61e-08 rel=5.22e-07 ulp=104 | 1.000 | abs=1.83e-07 rel=7.93e-08 ulp=4 |
| 3 | B_fs[tf32] | abs=4.93e-05 rel=0.002 ulp=149093 | 0.698 | abs=1.83e-07 rel=7.93e-08 ulp=4 |
| 3 | C_nm[tf32] | abs=4.93e-05 rel=0.002 ulp=149093 | 0.698 | abs=1.83e-07 rel=7.93e-08 ulp=4 |

Production-scan identity checks (NUMERIC equality folds +0/-0; BITWISE is integer-view equality): id 1: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=—; id 2: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=—; id 3: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=—

Seam ablation for the one-token decode reference (native_pk state vs fp64 oracles applying ONE convention; max_rel): id 1: beta-bf16-rt only=rel=1.17e-07, div-sqrt only=rel=0.002, both=rel=1.17e-07; id 2: beta-bf16-rt only=rel=1.29e-07, div-sqrt only=rel=0.002, both=rel=1.29e-07; id 3: beta-bf16-rt only=rel=1.44e-07, div-sqrt only=rel=0.002, both=rel=1.44e-07

`A_legacy_state*` rows are excluded: the June-8 vendored sequential scan's export path skips the root update (source-inspected); when/which binary changed that path is NOT established. Its outputs are bit-identical to the production scan.

## Stage 2 — commit from IDENTICAL oracle factors (compact reconstruction kernel; vs oracle state / vs native_sg state)

| id | B_fs[ieee] vs oracle | vs native_sg | C_nm[ieee] vs oracle | B_fs[tf32] vs oracle | C_nm[tf32] vs oracle |
|---|---|---|---|---|---|
| 1 | abs=7.55e-08 rel=1.18e-07 ulp=4 | abs=8.94e-08 rel=1.4e-07 ulp=6 | abs=7.55e-08 rel=1.18e-07 ulp=4 | abs=0.000882 rel=0.001 ulp=24242 | abs=0.000882 rel=0.001 ulp=24242 |
| 2 | abs=6.33e-08 rel=1.08e-07 ulp=3 | abs=8.94e-08 rel=1.52e-07 ulp=6 | abs=6.33e-08 rel=1.08e-07 ulp=3 | abs=0.000683 rel=0.001 ulp=24099 | abs=0.000683 rel=0.001 ulp=24099 |
| 3 | abs=9.83e-08 rel=1.76e-07 ulp=4 | abs=1.19e-07 rel=2.14e-07 ulp=6 | abs=9.83e-08 rel=1.76e-07 ulp=4 | abs=0.000623 rel=0.001 ulp=24122 | abs=0.000623 rel=0.001 ulp=24122 |

## Stage 3 — own factors + own commit (all accepted prefixes incl. zero-accept; vs oracle / native_sg / native_pk)

| id | mechanism | vs oracle | vs native_sg (spec-update) | exact frac vs native_sg | vs native_pk (one-token decode) |
|---|---|---|---|---|---|
| 1 | A_prod replay | abs=9.2e-08 rel=1.44e-07 ulp=5 | abs=0 rel=0 ulp=0 | 1.000 | abs=0.001 rel=0.002 ulp=63531 |
| 1 | B_legacyWY fp32-closed (materialized) | abs=8.66e-08 rel=1.36e-07 ulp=5 | — | — | — |
| 1 | B_fs[ieee] compact | abs=8.66e-08 rel=1.36e-07 ulp=5 | abs=8.94e-08 rel=1.4e-07 ulp=4 | 0.803 | abs=0.001 rel=0.002 ulp=63531 |
| 1 | C_nm[ieee] compact | abs=8.66e-08 rel=1.36e-07 ulp=5 | abs=8.94e-08 rel=1.4e-07 ulp=5 | 0.796 | abs=0.001 rel=0.002 ulp=63531 |
| 1 | B_fs[tf32] compact | abs=0.000882 rel=0.001 ulp=24242 | abs=0.000882 rel=0.001 ulp=24243 | 0 | abs=0.002 rel=0.003 ulp=82441 |
| 1 | C_nm[tf32] compact | abs=0.000882 rel=0.001 ulp=24242 | abs=0.000882 rel=0.001 ulp=24243 | 0 | abs=0.002 rel=0.003 ulp=82441 |
| 2 | A_prod replay | abs=6.87e-08 rel=1.17e-07 ulp=6 | abs=0 rel=0 ulp=0 | 1.000 | abs=0.001 rel=0.002 ulp=63529 |
| 2 | B_legacyWY fp32-closed (materialized) | abs=7.02e-08 rel=1.2e-07 ulp=6 | — | — | — |
| 2 | B_fs[ieee] compact | abs=7.02e-08 rel=1.2e-07 ulp=6 | abs=5.96e-08 rel=1.02e-07 ulp=4 | 0.786 | abs=0.001 rel=0.002 ulp=63528 |
| 2 | C_nm[ieee] compact | abs=7.02e-08 rel=1.2e-07 ulp=6 | abs=5.96e-08 rel=1.02e-07 ulp=5 | 0.779 | abs=0.001 rel=0.002 ulp=63528 |
| 2 | B_fs[tf32] compact | abs=0.000683 rel=0.001 ulp=24099 | abs=0.000683 rel=0.001 ulp=24097 | 0 | abs=0.001 rel=0.003 ulp=82034 |
| 2 | C_nm[tf32] compact | abs=0.000683 rel=0.001 ulp=24099 | abs=0.000683 rel=0.001 ulp=24097 | 0 | abs=0.001 rel=0.003 ulp=82034 |
| 3 | A_prod replay | abs=8.94e-08 rel=1.6e-07 ulp=5 | abs=0 rel=0 ulp=0 | 1.000 | abs=0.001 rel=0.002 ulp=63141 |
| 3 | B_legacyWY fp32-closed (materialized) | abs=1.08e-07 rel=1.93e-07 ulp=5 | — | — | — |
| 3 | B_fs[ieee] compact | abs=1.08e-07 rel=1.93e-07 ulp=5 | abs=5.96e-08 rel=1.07e-07 ulp=4 | 0.811 | abs=0.001 rel=0.002 ulp=63141 |
| 3 | C_nm[ieee] compact | abs=1.08e-07 rel=1.93e-07 ulp=5 | abs=5.96e-08 rel=1.07e-07 ulp=5 | 0.804 | abs=0.001 rel=0.002 ulp=63141 |
| 3 | B_fs[tf32] compact | abs=0.000623 rel=0.001 ulp=24122 | abs=0.000623 rel=0.001 ulp=24123 | 0 | abs=0.001 rel=0.003 ulp=84922 |
| 3 | C_nm[tf32] compact | abs=0.000623 rel=0.001 ulp=24122 | abs=0.000623 rel=0.001 ulp=24123 | 0 | abs=0.001 rel=0.003 ulp=84922 |

Replay repeat-launch bitwise check (immutable h0 row): id 1: yes; id 2: yes; id 3: yes

## Committed-state max_abs vs oracle by accepted depth (finite rows only; any non-finite comparison in a row => n/a)


id 1 (synthetic[tiny-gates] seed=20260921 n=12):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 9.2e-08 | 6.6e-08 | 6.6e-08 | 4.81e-08 | 9.2e-08 | 0.000988 | 0.000476 |
| 1 | 7.53e-08 | 5.56e-08 | 5.56e-08 | 4.07e-08 | 7.53e-08 | 0.001 | 0.000523 |
| 2 | 8.11e-08 | 6.56e-08 | 6.56e-08 | 6.56e-08 | 8.11e-08 | 0.000883 | 0.000882 |
| 3 | 4.39e-08 | 6.3e-08 | 6.3e-08 | 7.55e-08 | 4.39e-08 | 0.000775 | 0.000414 |
| 4 | 5.95e-08 | 5.95e-08 | 5.95e-08 | 4.85e-08 | 5.95e-08 | 0.000893 | 0.000612 |
| 5 | 6.09e-08 | 6.09e-08 | 6.09e-08 | 5.73e-08 | 6.09e-08 | 0.000774 | 0.000503 |
| 6 | 6.7e-08 | 6.7e-08 | 6.7e-08 | 5.37e-08 | 6.7e-08 | 0.000848 | 0.000593 |
| 7 | 4.63e-08 | 5.06e-08 | 5.06e-08 | 5.71e-08 | 4.63e-08 | 0.001 | 0.000584 |
| 8 | 6.12e-08 | 6.12e-08 | 6.12e-08 | 3.86e-08 | 6.12e-08 | 0.000989 | 0.000459 |
| 9 | 7.3e-08 | 6.56e-08 | 6.56e-08 | 5.31e-08 | 7.3e-08 | 0.000843 | 0.000561 |
| 10 | 6.78e-08 | 6.78e-08 | 6.78e-08 | 5.44e-08 | 6.78e-08 | 0.000688 | 0.000554 |
| 11 | 6.5e-08 | 8.66e-08 | 8.66e-08 | 5.76e-08 | 6.5e-08 | 0.000946 | 0.00061 |

id 2 (synthetic[tiny-gates] seed=20260921 n=6):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 5.86e-08 | 5.86e-08 | 5.86e-08 | 4.99e-08 | 5.86e-08 | 0.001 | 0.000683 |
| 1 | 6.87e-08 | 5.89e-08 | 5.89e-08 | 4.86e-08 | 6.87e-08 | 0.001 | 0.000615 |
| 2 | 6.15e-08 | 6.15e-08 | 6.15e-08 | 6.33e-08 | 6.15e-08 | 0.000787 | 0.000509 |
| 3 | 6.86e-08 | 6.86e-08 | 6.86e-08 | 4.76e-08 | 6.86e-08 | 0.000867 | 0.000521 |
| 4 | 6.2e-08 | 7.02e-08 | 7.02e-08 | 5.57e-08 | 6.2e-08 | 0.001 | 0.000402 |
| 5 | 6.84e-08 | 6.84e-08 | 6.84e-08 | 4.67e-08 | 6.84e-08 | 0.001 | 0.000635 |

id 3 (synthetic[tiny-gates] seed=20260921 n=10):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 5.66e-08 | 6.79e-08 | 6.79e-08 | 6.66e-08 | 5.66e-08 | 0.000758 | 0.00036 |
| 1 | 7.5e-08 | 4.79e-08 | 4.79e-08 | 4.46e-08 | 7.5e-08 | 0.001 | 0.000361 |
| 2 | 8.94e-08 | 6.63e-08 | 6.63e-08 | 5.7e-08 | 8.94e-08 | 0.000964 | 0.000561 |
| 3 | 7.41e-08 | 7.41e-08 | 7.41e-08 | 9.83e-08 | 7.41e-08 | 0.000912 | 0.000501 |
| 4 | 6.8e-08 | 6.8e-08 | 6.8e-08 | 6.43e-08 | 6.8e-08 | 0.000951 | 0.000623 |
| 5 | 7.41e-08 | 1.08e-07 | 1.08e-07 | 7.79e-08 | 7.41e-08 | 0.000964 | 0.000526 |

## Controls

| id | Neumann d−1 terms (U err / out err) | B_fs N_PAD32 vs 16 (out exact / abs) | C_nm N_PAD32 vs 16 | B_fs sibling-reverse (out exact / U exact) | C_nm sibling-reverse | A_prod sibling-reverse out exact |
|---|---|---|---|---|---|---|
| 1 | 1.96e-07 / 8.82e-09 | 1.000 / 0 | 0.990 / 1.86e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 2 | 1.78e-07 / 7.72e-09 | 1.000 / 0 | 1.000 / 0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 3 | 1.83e-07 / 1.61e-08 | 1.000 / 0 | 1.000 / 0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |

## Transient / scratch memory (bytes, analytic from shapes; per request, per layer)

| quantity | id 1 | id 2 | id 3 |
|---|---|---|---|
| A_prod_scan_hbm_per_node_state_export | 0 | 0 | 0 |
| A_prod_scan_register_h_cache_per_program | 131,072 | 131,072 | 131,072 |
| A_prod_replay_hbm_written_rows(depth+1) | 37,748,736 | 18,874,368 | 18,874,368 |
| A_prod_activation_ring(k,v,a,b) | 265,216 | 265,216 | 265,216 |
| A_legacy_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 |
| B_legacyWY_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 |
| BC_compact_factors_U+cumg | 396,288 | 396,288 | 396,288 |
| BC_compact_commit_hbm_written_rows | 3,145,728 | 3,145,728 | 3,145,728 |
| one_full_state_row | 3,145,728 | 3,145,728 | 3,145,728 |

## Timing — NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['2770990'] seen; sampled observation, not a guarantee)

CUDA events, per-launch sync median / pipelined mean (µs), one layer, B1. Rows do DIFFERENT work (see WORK column); this is NOT an apples-to-apples kernel speedup table.

| kernel | WORK | id 1 | id 2 | id 3 |
|---|---|---|---|---|
| A_prod_scan_verify_out16 | bf16 out store; no per-node state export; preallocated | 133.5 / 109.6 | 59.4 / 34.9 | 89.5 / 65.6 |
| A_prod_replay_commit_deepest | writes depth+1 fp32 state rows to the bank; preallocated | 208.0 / 194.5 | 103.6 / 90.9 | 98.7 / 81.6 |
| A_prod_replay_commit_zero_accept | writes 1 fp32 state row (root); preallocated | 53.5 / 35.9 | 53.5 / 35.6 | 53.5 / 35.5 |
| A_legacy_scan_with_state_export | bf16 out + FULL per-node fp32 state export (n_pad x VH x DV x DK x 4 B); preallocated | 313.7 / 303.3 | 115.9 / 105.9 | 247.6 / 237.3 |
| B_legacyWY_fp32closed_with_state_export | bf16 out + FULL per-node fp32 state export; preallocated | 808.9 / 799.8 | 697.6 / 684.0 | 760.1 / 752.3 |
| B_legacyWY_bf16bnd_with_state_export | bf16 out (+bf16 boundary taps) + FULL per-node fp32 state export; preallocated | 2326.9 / 2320.0 | 1313.2 / 1305.2 | 1654.0 / 1644.5 |
| B_fs[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.7 / 9.6 | 18.0 / 9.6 | 18.6 / 9.6 |
| B_fs[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 110.9 / 98.1 | 112.0 / 98.2 | 110.8 / 96.7 |
| B_fs[ieee]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 106.7 / 92.7 | 106.8 / 92.9 | 105.9 / 92.2 |
| B_fs[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.7 / 9.8 | 16.8 / 9.7 | 16.9 / 9.7 |
| B_fs[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 42.9 / 28.7 | 41.2 / 28.7 | 41.8 / 28.7 |
| B_fs[tf32]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 42.3 / 30.0 | 41.2 / 28.7 | 42.3 / 30.5 |
| C_nm[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.8 / 9.9 | 18.7 / 9.8 | 18.8 / 9.8 |
| C_nm[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 96.6 / 88.2 | 92.4 / 95.5 | 94.6 / 86.9 |
| C_nm[ieee]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 98.6 / 85.1 | 95.4 / 81.9 | 96.6 / 83.5 |
| C_nm[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 17.0 / 9.5 | 16.8 / 9.8 | 16.8 / 10.6 |
| C_nm[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 34.9 / 20.5 | 31.4 / 18.3 | 31.2 / 18.5 |
| C_nm[tf32]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 35.1 / 20.6 | 31.0 / 21.3 | 32.8 / 18.5 |
| native_sg_chain_depth11_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | 227.6 / 209.3 | — | — |
| native_pk_one_token(context) | 1 token; helper allocates mixed/a/b/out per call (NOT matched work) | 24.8 / 15.6 | 24.0 / 15.5 | 24.1 / 15.5 |
| native_sg_chain_depth5_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | — | 115.8 / 100.4 | 117.0 / 100.7 |

In-band probe-drift criterion (4096² fp16 matmul before→after, pass iff drift ≤ 5 %): id 1: 1806.36158→1799.02077 µs drift=0.004 **PASS** [2026-09-21T23:12:54Z–2026-09-21T23:12:57Z]; id 2: 1792.72957→1841.73279 µs drift=0.027 **PASS** [2026-09-21T23:14:09Z–2026-09-21T23:14:12Z]; id 3: 1781.79836→1797.79510 µs drift=0.009 **PASS** [2026-09-21T23:16:02Z–2026-09-21T23:16:04Z]

iters=200, warmup=20. Not a serving throughput measurement.

## Validity

INVALID (non-finite) comparison cells in this summary: **0**. Probe-drift failures: **0**. Manifest status: **completed**; start/end source hashes match: **yes**.
