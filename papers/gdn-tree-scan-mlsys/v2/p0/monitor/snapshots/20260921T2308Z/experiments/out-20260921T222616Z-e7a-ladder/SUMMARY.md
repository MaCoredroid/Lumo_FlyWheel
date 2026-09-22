# E7a run summary — `out-20260921T222616Z-e7a-ladder`

Manifest status **completed**; start/end source hashes match: **yes**; image `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`; torch 2.11.0+cu130 / triton 3.6.0; GPU NVIDIA GB10 cc [12, 1]; driver `NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  59`.
Production kernel flags: {"FIXED32_MODE": null, "scan_align_on": false, "npad_invariant_on": false, "parent_gather_on": false, "hc_internal_on": false, "BV_prod": 16, "num_warps_prod": 8, "tf32_matmul_allowed": false, "replay_h0_source_column": 15, "scan_h0_source_column": 0}. Triton cubins hashed: 122.
Source sha256: `prod_kernel`=d9dd0c697b46…, `legacy_wy`=e7c2a2a8b658…, `e7a_core`=93c915d388d9…, `e7a_kernels`=15c353625218…, `e7a_device`=288bfcf072a6…, `native_fused_sigmoid_gating`=000ab8996af9…, `native_fused_recurrent`=3a2a3c5245ac…

**Timing attribution:** NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['2748559'] seen; sampled observation, not a guarantee) — window ['2026-09-21T22:26:16.768000', '2026-09-21T22:36:06.985000'], util samples 591, zero-util fraction 0.944, samples >5 %: 29, compute PIDs seen: {'2748559': {'samples': 575, 'names': ['python3']}}, container PIDs: ['2748559'], coverage gaps: 0.

All mechanisms B/C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems. Historical payloads are dependent historical operand sets (one topology, B1), not independent prefixes; synthetic rows are labeled. Rows marked **INVALID** contain non-finite values in candidate or reference and carry no error value.

## Operand sets

| id | label | provenance | topology | n | depth | g_min | cum_g_min | P_min | min visible decay ratio | fp32 underflow risk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | layers.62.linear_attn | HISTORICAL sha256 9f1ddb6e3cfa… (language_model.model.layers.62.linear_attn) | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.319 | -16.390 | 7.62e-08 | 5.3e-07 | no |
| 2 | layers.1.linear_attn | HISTORICAL sha256 855b41b460c0… (language_model.model.layers.1.linear_attn) | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -4.982 | -21.499 | 4.6e-10 | 1.88e-08 | no |
| 3 | layers.12.linear_attn | HISTORICAL sha256 77496f54e302… (language_model.model.layers.12.linear_attn) | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -23.834 | -55.126 | 1.15e-24 | 5.41e-23 | no |
| 4 | layers.0.linear_attn | HISTORICAL sha256 c174bd740565… (language_model.model.layers.0.linear_attn) | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -8.081 | -39.890 | 4.74e-18 | 3.5e-15 | no |
| 5 | synthetic[historical-like] seed=20260921 n=2 | SYNTHETIC seed=20260921 regime=historical-like | `[-1, 0]` | 2 | 1 | -2.606 | -4.941 | 0.007 | 0.074 | no |
| 6 | synthetic[historical-like] seed=20260921 n=6 | SYNTHETIC seed=20260921 regime=historical-like | `[-1, 0, 1, 2, 3, 4]` | 6 | 5 | -5.342 | -11.108 | 1.5e-05 | 1.5e-05 | no |
| 7 | synthetic[historical-like] seed=20260921 n=12 | SYNTHETIC seed=20260921 regime=historical-like | `[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` | 12 | 11 | -5.248 | -35.375 | 4.33e-16 | 6.48e-15 | no |
| 8 | synthetic[historical-like] seed=20260921 n=15 | SYNTHETIC seed=20260921 regime=historical-like | `[-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]` | 15 | 3 | -4.463 | -13.885 | 9.33e-07 | 1.44e-05 | no |
| 9 | synthetic[historical-like] seed=20260921 n=10 | SYNTHETIC seed=20260921 regime=historical-like | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.820 | -12.676 | 3.12e-06 | 5.22e-06 | no |

## References vs fp64 oracle (same rounded operands)

| id | native_sg out32 | native_sg state | native_pk out32 vs oracle | native_pk state vs oracle | native_pk state vs ITS OWN oracle (beta bf16-rt + div/sqrt) | seam magnitude fp64 (oracle_pk vs oracle, state) | pk vs sg out16 exact | sg fp32-io vs bf16-io state max_abs |
|---|---|---|---|---|---|---|---|---|
| 1 | abs=7.38e-09 rel=1.34e-07 ulp=43 | abs=4.6e-07 rel=1.27e-07 ulp=22 | abs=5.24e-05 rel=0.000948 ulp=341187 | abs=0.000799 rel=0.000221 ulp=145608 | abs=6.6e-07 rel=1.82e-07 ulp=30 | abs=0.000799 rel=0.00022 | 0.821 | 5.96e-08 |
| 2 | abs=5e-09 rel=1.35e-07 ulp=18 | abs=1.02e-07 rel=2.18e-07 ulp=82 | abs=2.19e-05 rel=0.000589 ulp=109742 | abs=0.000659 rel=0.001 ulp=499653 | abs=8.4e-08 rel=1.8e-07 ulp=88 | abs=0.000659 rel=0.001 | 0.751 | 7.45e-09 |
| 3 | abs=3.17e-09 rel=9.27e-08 ulp=13 | abs=8.81e-08 rel=1e-07 ulp=59 | abs=2.12e-05 rel=0.00062 ulp=186184 | abs=0.000565 rel=0.000641 ulp=257665 | abs=1.34e-07 rel=1.53e-07 ulp=59 | abs=0.000565 rel=0.000641 | 0.791 | 1.49e-08 |
| 4 | abs=3.42e-07 rel=1.46e-07 ulp=5 | abs=3.22e-06 rel=1.54e-07 ulp=97 | abs=0.003 rel=0.001 ulp=30853 | abs=0.034 rel=0.002 ulp=1438534 | abs=2.65e-06 rel=1.26e-07 ulp=135 | abs=0.034 rel=0.002 | 0.810 | 2.38e-07 |
| 5 | abs=3.82e-08 rel=1.43e-07 ulp=153 | abs=5.6e-07 rel=1.38e-07 ulp=53 | abs=8.37e-05 rel=0.000313 ulp=319115 | abs=0.001 rel=0.000349 ulp=529342 | abs=4.96e-07 rel=1.22e-07 ulp=74 | abs=0.001 rel=0.000349 | 0.937 | 2.38e-07 |
| 6 | abs=6.39e-08 rel=2.36e-07 ulp=172 | abs=1.02e-06 rel=2.71e-07 ulp=86 | abs=0.000126 rel=0.000465 ulp=334293 | abs=0.002 rel=0.000509 ulp=963217 | abs=1.05e-06 rel=2.79e-07 ulp=94 | abs=0.002 rel=0.000509 | 0.904 | 2.38e-07 |
| 7 | abs=8.08e-08 rel=3.13e-07 ulp=148 | abs=1.35e-06 rel=3.69e-07 ulp=169 | abs=0.000101 rel=0.000391 ulp=465189 | abs=0.002 rel=0.00055 ulp=1072356 | abs=1.45e-06 rel=3.96e-07 ulp=147 | abs=0.002 rel=0.00055 | 0.830 | 2.38e-07 |
| 8 | abs=6.91e-08 rel=2.45e-07 ulp=169 | abs=1.06e-06 rel=2.61e-07 ulp=103 | abs=0.000111 rel=0.000395 ulp=522970 | abs=0.002 rel=0.000424 ulp=948791 | abs=1.04e-06 rel=2.58e-07 ulp=96 | abs=0.002 rel=0.000424 | 0.904 | 2.38e-07 |
| 9 | abs=6.01e-08 rel=2.22e-07 ulp=173 | abs=1.19e-06 rel=3.19e-07 ulp=98 | abs=9.8e-05 rel=0.000363 ulp=479620 | abs=0.002 rel=0.000427 ulp=916226 | abs=1.19e-06 rel=3.2e-07 ulp=168 | abs=0.002 rel=0.000427 | 0.873 | 4.77e-07 |

## Stage 1 — verifier outputs and correction factors (vs fp64 oracle; out16 exact fraction vs native spec-update kernel)

| id | mechanism | out32 vs oracle | out16 exact vs native_sg | factors U vs oracle u |
|---|---|---|---|---|
| 1 | A_prod | abs=7.38e-09 rel=1.34e-07 ulp=43 | 1.000 | — |
| 1 | B_legacyWY fp32-closed | abs=8.82e-09 rel=1.6e-07 ulp=82 | — | — |
| 1 | B_legacyWY bf16-bnd | abs=0.000166 rel=0.003 ulp=1076431 | — | — |
| 1 | B_fs[ieee] | abs=1.61e-08 rel=2.92e-07 ulp=162 | 1.000 | abs=4.47e-07 rel=3.43e-07 ulp=277 |
| 1 | C_nm[ieee] | abs=1.66e-08 rel=3e-07 ulp=113 | 1.000 | abs=4.47e-07 rel=3.43e-07 ulp=277 |
| 1 | B_fs[tf32] | abs=7.6e-05 rel=0.001 ulp=273952 | 0.785 | abs=0.00066 rel=0.000506 ulp=388890 |
| 1 | C_nm[tf32] | abs=7.52e-05 rel=0.001 ulp=425186 | 0.788 | abs=0.00066 rel=0.000506 ulp=394106 |
| 2 | A_prod | abs=5e-09 rel=1.35e-07 ulp=18 | 1.000 | — |
| 2 | B_legacyWY fp32-closed | abs=6.09e-09 rel=1.64e-07 ulp=21 | — | — |
| 2 | B_legacyWY bf16-bnd | abs=7.54e-05 rel=0.002 ulp=328183 | — | — |
| 2 | B_fs[ieee] | abs=1.02e-08 rel=2.75e-07 ulp=17 | 1.000 | abs=1.14e-07 rel=1.71e-07 ulp=143 |
| 2 | C_nm[ieee] | abs=1.02e-08 rel=2.75e-07 ulp=17 | 1.000 | abs=1.14e-07 rel=1.71e-07 ulp=141 |
| 2 | B_fs[tf32] | abs=2.81e-05 rel=0.000756 ulp=65856 | 0.744 | abs=0.000353 rel=0.000526 ulp=204552 |
| 2 | C_nm[tf32] | abs=2.81e-05 rel=0.000756 ulp=56672 | 0.745 | abs=0.0004 rel=0.000596 ulp=184416 |
| 3 | A_prod | abs=3.17e-09 rel=9.27e-08 ulp=13 | 1.000 | — |
| 3 | B_legacyWY fp32-closed | abs=4.8e-09 rel=1.4e-07 ulp=20 | — | — |
| 3 | B_legacyWY bf16-bnd | abs=8.04e-05 rel=0.002 ulp=253577 | — | — |
| 3 | B_fs[ieee] | abs=9.86e-09 rel=2.88e-07 ulp=47 | 1.000 | abs=1.04e-07 rel=2.12e-07 ulp=83 |
| 3 | C_nm[ieee] | abs=9.86e-09 rel=2.88e-07 ulp=41 | 1.000 | abs=1.04e-07 rel=2.12e-07 ulp=83 |
| 3 | B_fs[tf32] | abs=2.36e-05 rel=0.00069 ulp=158138 | 0.763 | abs=0.000195 rel=0.000395 ulp=384718 |
| 3 | C_nm[tf32] | abs=2.22e-05 rel=0.000649 ulp=158138 | 0.765 | abs=0.000273 rel=0.000554 ulp=451954 |
| 4 | A_prod | abs=3.42e-07 rel=1.46e-07 ulp=5 | 1.000 | — |
| 4 | B_legacyWY fp32-closed | abs=4.93e-07 rel=2.1e-07 ulp=5 | — | — |
| 4 | B_legacyWY bf16-bnd | abs=0.007 rel=0.003 ulp=90453 | — | — |
| 4 | B_fs[ieee] | abs=6.88e-07 rel=2.94e-07 ulp=10 | 1.000 | abs=6.25e-06 rel=2.28e-07 ulp=55 |
| 4 | C_nm[ieee] | abs=6.88e-07 rel=2.94e-07 ulp=10 | 1.000 | abs=6.14e-06 rel=2.25e-07 ulp=61 |
| 4 | B_fs[tf32] | abs=0.002 rel=0.000875 ulp=30162 | 0.766 | abs=0.013 rel=0.00047 ulp=182132 |
| 4 | C_nm[tf32] | abs=0.002 rel=0.000847 ulp=30162 | 0.770 | abs=0.021 rel=0.000761 ulp=243157 |
| 5 | A_prod | abs=3.82e-08 rel=1.43e-07 ulp=153 | 1.000 | — |
| 5 | B_legacyWY fp32-closed | abs=4.01e-08 rel=1.5e-07 ulp=121 | — | — |
| 5 | B_legacyWY bf16-bnd | abs=0.000397 rel=0.001 ulp=3090391 | — | — |
| 5 | B_fs[ieee] | abs=1.31e-07 rel=4.9e-07 ulp=259 | 1.000 | abs=1.08e-06 rel=3.39e-07 ulp=156 |
| 5 | C_nm[ieee] | abs=1.31e-07 rel=4.9e-07 ulp=259 | 1.000 | abs=1.08e-06 rel=3.39e-07 ulp=156 |
| 5 | B_fs[tf32] | abs=0.000217 rel=0.000814 ulp=487885 | 0.779 | abs=0.002 rel=0.000584 ulp=667659 |
| 5 | C_nm[tf32] | abs=0.000217 rel=0.000814 ulp=487885 | 0.779 | abs=0.002 rel=0.000656 ulp=657283 |
| 6 | A_prod | abs=6.39e-08 rel=2.36e-07 ulp=172 | 1.000 | — |
| 6 | B_legacyWY fp32-closed | abs=4.57e-08 rel=1.69e-07 ulp=164 | — | — |
| 6 | B_legacyWY bf16-bnd | abs=0.000499 rel=0.002 ulp=3591086 | — | — |
| 6 | B_fs[ieee] | abs=1.11e-07 rel=4.1e-07 ulp=355 | 1.000 | abs=1.1e-06 rel=3.29e-07 ulp=340 |
| 6 | C_nm[ieee] | abs=1.11e-07 rel=4.1e-07 ulp=357 | 1.000 | abs=1.1e-06 rel=3.29e-07 ulp=340 |
| 6 | B_fs[tf32] | abs=0.000193 rel=0.000714 ulp=514355 | 0.783 | abs=0.002 rel=0.000609 ulp=948217 |
| 6 | C_nm[tf32] | abs=0.000193 rel=0.000714 ulp=514355 | 0.783 | abs=0.002 rel=0.000609 ulp=950750 |
| 7 | A_prod | abs=8.08e-08 rel=3.13e-07 ulp=148 | 1.000 | — |
| 7 | B_legacyWY fp32-closed | abs=3.65e-08 rel=1.42e-07 ulp=133 | — | — |
| 7 | B_legacyWY bf16-bnd | abs=0.000444 rel=0.002 ulp=2492220 | — | — |
| 7 | B_fs[ieee] | abs=1.05e-07 rel=4.08e-07 ulp=199 | 1.000 | abs=9.49e-07 rel=2.85e-07 ulp=264 |
| 7 | C_nm[ieee] | abs=1.05e-07 rel=4.08e-07 ulp=199 | 1.000 | abs=9.49e-07 rel=2.85e-07 ulp=266 |
| 7 | B_fs[tf32] | abs=0.000218 rel=0.000846 ulp=547660 | 0.755 | abs=0.002 rel=0.000588 ulp=532002 |
| 7 | C_nm[tf32] | abs=0.000218 rel=0.000845 ulp=538252 | 0.755 | abs=0.002 rel=0.000588 ulp=600587 |
| 8 | A_prod | abs=6.91e-08 rel=2.45e-07 ulp=169 | 1.000 | — |
| 8 | B_legacyWY fp32-closed | abs=4.73e-08 rel=1.68e-07 ulp=180 | — | — |
| 8 | B_legacyWY bf16-bnd | abs=0.000569 rel=0.002 ulp=3744239 | — | — |
| 8 | B_fs[ieee] | abs=1.47e-07 rel=5.22e-07 ulp=321 | 1.000 | abs=1.23e-06 rel=3.11e-07 ulp=176 |
| 8 | C_nm[ieee] | abs=1.47e-07 rel=5.22e-07 ulp=321 | 1.000 | abs=1.17e-06 rel=2.96e-07 ulp=174 |
| 8 | B_fs[tf32] | abs=0.000258 rel=0.000914 ulp=826510 | 0.786 | abs=0.002 rel=0.000584 ulp=516222 |
| 8 | C_nm[tf32] | abs=0.000258 rel=0.000914 ulp=822850 | 0.786 | abs=0.002 rel=0.000596 ulp=517164 |
| 9 | A_prod | abs=6.01e-08 rel=2.22e-07 ulp=173 | 1.000 | — |
| 9 | B_legacyWY fp32-closed | abs=5.33e-08 rel=1.97e-07 ulp=164 | — | — |
| 9 | B_legacyWY bf16-bnd | abs=0.000466 rel=0.002 ulp=3477288 | — | — |
| 9 | B_fs[ieee] | abs=1.31e-07 rel=4.85e-07 ulp=341 | 1.000 | abs=1.02e-06 rel=3.13e-07 ulp=318 |
| 9 | C_nm[ieee] | abs=1.39e-07 rel=5.13e-07 ulp=341 | 1.000 | abs=1.02e-06 rel=3.13e-07 ulp=318 |
| 9 | B_fs[tf32] | abs=0.000201 rel=0.000744 ulp=707545 | 0.774 | abs=0.002 rel=0.00065 ulp=1028051 |
| 9 | C_nm[tf32] | abs=0.000201 rel=0.000744 ulp=710321 | 0.775 | abs=0.002 rel=0.00065 ulp=1028051 |

Production-scan identity checks: id 1: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=yes; id 2: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=no; id 3: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=yes; id 4: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=no; id 5: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=—; id 6: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=—; id 7: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=—; id 8: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=—; id 9: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=—

`A_legacy_state*` rows are excluded: the June-8 vendored sequential scan's export path skips the root update (source-inspected); when/which binary changed that path is NOT established. Its outputs are bit-identical to the production scan.

## Stage 2 — commit from IDENTICAL oracle factors (compact reconstruction kernel; vs oracle state / vs native_sg state)

| id | B_fs[ieee] vs oracle | vs native_sg | C_nm[ieee] vs oracle | B_fs[tf32] vs oracle | C_nm[tf32] vs oracle |
|---|---|---|---|---|---|
| 1 | abs=4.58e-07 rel=1.26e-07 ulp=12 | abs=7.15e-07 rel=1.97e-07 ulp=25 | abs=4.58e-07 rel=1.26e-07 ulp=12 | abs=0.001 rel=0.000296 ulp=96135 | abs=0.001 rel=0.000296 ulp=96135 |
| 2 | abs=7.75e-08 rel=1.66e-07 ulp=66 | abs=8.94e-08 rel=1.92e-07 ulp=54 | abs=7.75e-08 rel=1.66e-07 ulp=66 | abs=0.000262 rel=0.000561 ulp=277754 | abs=0.000262 rel=0.000561 ulp=277754 |
| 3 | abs=8.86e-08 rel=1.01e-07 ulp=58 | abs=5.96e-08 rel=6.76e-08 ulp=41 | abs=8.86e-08 rel=1.01e-07 ulp=58 | abs=0.000321 rel=0.000364 ulp=138912 | abs=0.000321 rel=0.000364 ulp=138912 |
| 4 | abs=2.78e-06 rel=1.33e-07 ulp=97 | abs=3.81e-06 rel=1.82e-07 ulp=83 | abs=2.78e-06 rel=1.33e-07 ulp=97 | abs=0.019 rel=0.000894 ulp=562835 | abs=0.019 rel=0.000894 ulp=562835 |
| 5 | abs=4.53e-07 rel=1.11e-07 ulp=46 | abs=4.77e-07 rel=1.17e-07 ulp=70 | abs=4.53e-07 rel=1.11e-07 ulp=46 | abs=0.001 rel=0.000261 ulp=339865 | abs=0.001 rel=0.000261 ulp=339865 |
| 6 | abs=6.15e-07 rel=1.64e-07 ulp=143 | abs=1.19e-06 rel=3.18e-07 ulp=131 | abs=6.15e-07 rel=1.64e-07 ulp=143 | abs=0.001 rel=0.000325 ulp=597776 | abs=0.001 rel=0.000325 ulp=597776 |
| 7 | abs=7.71e-07 rel=2.11e-07 ulp=107 | abs=1.67e-06 rel=4.57e-07 ulp=176 | abs=7.71e-07 rel=2.11e-07 ulp=107 | abs=0.001 rel=0.00033 ulp=600234 | abs=0.001 rel=0.00033 ulp=600234 |
| 8 | abs=6.25e-07 rel=1.55e-07 ulp=83 | abs=9.54e-07 rel=2.36e-07 ulp=126 | abs=6.25e-07 rel=1.55e-07 ulp=83 | abs=0.001 rel=0.000307 ulp=498384 | abs=0.001 rel=0.000307 ulp=498384 |
| 9 | abs=6.9e-07 rel=1.85e-07 ulp=87 | abs=1.19e-06 rel=3.19e-07 ulp=111 | abs=6.9e-07 rel=1.85e-07 ulp=87 | abs=0.000963 rel=0.000258 ulp=487676 | abs=0.000963 rel=0.000258 ulp=487676 |

## Stage 3 — own factors + own commit (all accepted prefixes incl. zero-accept; vs oracle / native_sg / native_pk)

| id | mechanism | vs oracle | vs native_sg (spec-update) | exact frac vs native_sg | vs native_pk (one-token decode) |
|---|---|---|---|---|---|
| 1 | A_prod replay | abs=4.6e-07 rel=1.27e-07 ulp=22 | abs=5.96e-08 rel=1.65e-08 ulp=4 | 0.991 | abs=0.000799 rel=0.000221 ulp=145600 |
| 1 | B_legacyWY fp32-closed (materialized) | abs=4.8e-07 rel=1.32e-07 ulp=25 | — | — | — |
| 1 | B_fs[ieee] compact | abs=4.58e-07 rel=1.26e-07 ulp=60 | abs=4.77e-07 rel=1.32e-07 ulp=50 | 0.319 | abs=0.000799 rel=0.000221 ulp=145626 |
| 1 | C_nm[ieee] compact | abs=4.58e-07 rel=1.26e-07 ulp=60 | abs=4.77e-07 rel=1.32e-07 ulp=50 | 0.316 | abs=0.000799 rel=0.000221 ulp=145629 |
| 1 | B_fs[tf32] compact | abs=0.001 rel=0.000296 ulp=242824 | abs=0.001 rel=0.000296 ulp=242821 | 0.000443 | abs=0.001 rel=0.000392 ulp=200358 |
| 1 | C_nm[tf32] compact | abs=0.001 rel=0.000296 ulp=241842 | abs=0.001 rel=0.000296 ulp=241839 | 0.000439 | abs=0.001 rel=0.000392 ulp=199376 |
| 2 | A_prod replay | abs=1.02e-07 rel=2.18e-07 ulp=82 | abs=7.45e-09 rel=1.6e-08 ulp=9 | 0.993 | abs=0.000659 rel=0.001 ulp=499571 |
| 2 | B_legacyWY fp32-closed (materialized) | abs=6.38e-08 rel=1.37e-07 ulp=60 | — | — | — |
| 2 | B_fs[ieee] compact | abs=6.61e-08 rel=1.42e-07 ulp=83 | abs=1.19e-07 rel=2.55e-07 ulp=65 | 0.350 | abs=0.000659 rel=0.001 ulp=499587 |
| 2 | C_nm[ieee] compact | abs=7.75e-08 rel=1.66e-07 ulp=83 | abs=1.19e-07 rel=2.55e-07 ulp=66 | 0.346 | abs=0.000659 rel=0.001 ulp=499587 |
| 2 | B_fs[tf32] compact | abs=0.000232 rel=0.000497 ulp=334266 | abs=0.000232 rel=0.000497 ulp=334277 | 0.000133 | abs=0.000671 rel=0.001 ulp=637274 |
| 2 | C_nm[tf32] compact | abs=0.000248 rel=0.00053 ulp=315152 | abs=0.000248 rel=0.00053 ulp=315188 | 0.000134 | abs=0.000671 rel=0.001 ulp=618574 |
| 3 | A_prod replay | abs=8.81e-08 rel=1e-07 ulp=59 | abs=1.49e-08 rel=1.69e-08 ulp=7 | 0.988 | abs=0.000565 rel=0.000641 ulp=209492 |
| 3 | B_legacyWY fp32-closed (materialized) | abs=9.38e-08 rel=1.06e-07 ulp=59 | — | — | — |
| 3 | B_fs[ieee] compact | abs=8.86e-08 rel=1.01e-07 ulp=59 | abs=8.94e-08 rel=1.01e-07 ulp=50 | 0.302 | abs=0.000565 rel=0.000641 ulp=209484 |
| 3 | C_nm[ieee] compact | abs=8.86e-08 rel=1.01e-07 ulp=59 | abs=8.94e-08 rel=1.01e-07 ulp=51 | 0.303 | abs=0.000565 rel=0.000641 ulp=209488 |
| 3 | B_fs[tf32] compact | abs=0.00029 rel=0.000329 ulp=264736 | abs=0.00029 rel=0.000329 ulp=264743 | 0.00016 | abs=0.000754 rel=0.000855 ulp=473283 |
| 3 | C_nm[tf32] compact | abs=0.000283 rel=0.000321 ulp=264736 | abs=0.000283 rel=0.000321 ulp=264743 | 0.000169 | abs=0.000717 rel=0.000814 ulp=473283 |
| 4 | A_prod replay | abs=3.22e-06 rel=1.54e-07 ulp=97 | abs=2.38e-07 rel=1.14e-08 ulp=5 | 0.989 | abs=0.034 rel=0.002 ulp=1438566 |
| 4 | B_legacyWY fp32-closed (materialized) | abs=4.08e-06 rel=1.95e-07 ulp=91 | — | — | — |
| 4 | B_fs[ieee] compact | abs=3.53e-06 rel=1.69e-07 ulp=114 | abs=3.81e-06 rel=1.82e-07 ulp=163 | 0.327 | abs=0.034 rel=0.002 ulp=1438566 |
| 4 | C_nm[ieee] compact | abs=4.24e-06 rel=2.03e-07 ulp=112 | abs=4.77e-06 rel=2.28e-07 ulp=161 | 0.324 | abs=0.034 rel=0.002 ulp=1438565 |
| 4 | B_fs[tf32] compact | abs=0.017 rel=0.000812 ulp=562835 | abs=0.017 rel=0.000812 ulp=562849 | 0.000958 | abs=0.045 rel=0.002 ulp=1363439 |
| 4 | C_nm[tf32] compact | abs=0.017 rel=0.000812 ulp=562835 | abs=0.017 rel=0.000812 ulp=562849 | 0.000963 | abs=0.045 rel=0.002 ulp=1450927 |
| 5 | A_prod replay | abs=5.6e-07 rel=1.38e-07 ulp=53 | abs=2.38e-07 rel=5.85e-08 ulp=22 | 0.992 | abs=0.001 rel=0.000349 ulp=680512 |
| 5 | B_legacyWY fp32-closed (materialized) | abs=3.75e-07 rel=9.21e-08 ulp=69 | — | — | — |
| 5 | B_fs[ieee] compact | abs=4.53e-07 rel=1.11e-07 ulp=108 | abs=4.77e-07 rel=1.17e-07 ulp=129 | 0.571 | abs=0.001 rel=0.000349 ulp=680384 |
| 5 | C_nm[ieee] compact | abs=4.53e-07 rel=1.11e-07 ulp=108 | abs=4.77e-07 rel=1.17e-07 ulp=129 | 0.572 | abs=0.001 rel=0.000349 ulp=680384 |
| 5 | B_fs[tf32] compact | abs=0.001 rel=0.000347 ulp=384224 | abs=0.001 rel=0.000347 ulp=384247 | 0.003 | abs=0.002 rel=0.00052 ulp=837458 |
| 5 | C_nm[tf32] compact | abs=0.001 rel=0.000347 ulp=384224 | abs=0.001 rel=0.000347 ulp=384247 | 0.003 | abs=0.002 rel=0.00052 ulp=837458 |
| 6 | A_prod replay | abs=1.02e-06 rel=2.71e-07 ulp=75 | abs=2.38e-07 rel=6.36e-08 ulp=51 | 0.985 | abs=0.002 rel=0.000509 ulp=903215 |
| 6 | B_legacyWY fp32-closed (materialized) | abs=6.45e-07 rel=1.72e-07 ulp=105 | — | — | — |
| 6 | B_fs[ieee] compact | abs=6.15e-07 rel=1.64e-07 ulp=238 | abs=1.19e-06 rel=3.18e-07 ulp=186 | 0.344 | abs=0.002 rel=0.000509 ulp=903163 |
| 6 | C_nm[ieee] compact | abs=6.15e-07 rel=1.64e-07 ulp=272 | abs=1.19e-06 rel=3.18e-07 ulp=210 | 0.343 | abs=0.002 rel=0.000509 ulp=903163 |
| 6 | B_fs[tf32] compact | abs=0.002 rel=0.000448 ulp=997258 | abs=0.002 rel=0.000448 ulp=997320 | 0.002 | abs=0.003 rel=0.000752 ulp=1420745 |
| 6 | C_nm[tf32] compact | abs=0.002 rel=0.000448 ulp=997258 | abs=0.002 rel=0.000448 ulp=997320 | 0.002 | abs=0.003 rel=0.000752 ulp=1406849 |
| 7 | A_prod replay | abs=1.35e-06 rel=3.69e-07 ulp=169 | abs=2.38e-07 rel=6.53e-08 ulp=67 | 0.983 | abs=0.002 rel=0.00055 ulp=1072383 |
| 7 | B_legacyWY fp32-closed (materialized) | abs=8.15e-07 rel=2.23e-07 ulp=126 | — | — | — |
| 7 | B_fs[ieee] compact | abs=1.01e-06 rel=2.76e-07 ulp=162 | abs=1.67e-06 rel=4.57e-07 ulp=207 | 0.261 | abs=0.002 rel=0.00055 ulp=1072478 |
| 7 | C_nm[ieee] compact | abs=1.01e-06 rel=2.76e-07 ulp=189 | abs=1.67e-06 rel=4.57e-07 ulp=200 | 0.258 | abs=0.002 rel=0.00055 ulp=1072478 |
| 7 | B_fs[tf32] compact | abs=0.002 rel=0.000436 ulp=940461 | abs=0.002 rel=0.000436 ulp=940526 | 0.000673 | abs=0.003 rel=0.00086 ulp=1775117 |
| 7 | C_nm[tf32] compact | abs=0.002 rel=0.000426 ulp=932113 | abs=0.002 rel=0.000426 ulp=932178 | 0.00067 | abs=0.003 rel=0.000849 ulp=1773863 |
| 8 | A_prod replay | abs=1.06e-06 rel=2.61e-07 ulp=103 | abs=2.38e-07 rel=5.9e-08 ulp=28 | 0.987 | abs=0.002 rel=0.000424 ulp=774951 |
| 8 | B_legacyWY fp32-closed (materialized) | abs=5.98e-07 rel=1.48e-07 ulp=98 | — | — | — |
| 8 | B_fs[ieee] compact | abs=6.25e-07 rel=1.55e-07 ulp=105 | abs=9.54e-07 rel=2.36e-07 ulp=143 | 0.316 | abs=0.002 rel=0.000424 ulp=774891 |
| 8 | C_nm[ieee] compact | abs=6.25e-07 rel=1.55e-07 ulp=105 | abs=9.54e-07 rel=2.36e-07 ulp=143 | 0.316 | abs=0.002 rel=0.000424 ulp=774893 |
| 8 | B_fs[tf32] compact | abs=0.002 rel=0.000465 ulp=675878 | abs=0.002 rel=0.000465 ulp=675946 | 0.000787 | abs=0.003 rel=0.000801 ulp=1085504 |
| 8 | C_nm[tf32] compact | abs=0.002 rel=0.000465 ulp=675878 | abs=0.002 rel=0.000465 ulp=675946 | 0.000788 | abs=0.003 rel=0.000823 ulp=1085504 |
| 9 | A_prod replay | abs=1.19e-06 rel=3.19e-07 ulp=98 | abs=4.77e-07 rel=1.28e-07 ulp=35 | 0.985 | abs=0.002 rel=0.000427 ulp=1138882 |
| 9 | B_legacyWY fp32-closed (materialized) | abs=6.45e-07 rel=1.73e-07 ulp=140 | — | — | — |
| 9 | B_fs[ieee] compact | abs=6.9e-07 rel=1.85e-07 ulp=217 | abs=1.19e-06 rel=3.19e-07 ulp=174 | 0.308 | abs=0.002 rel=0.000427 ulp=1138804 |
| 9 | C_nm[ieee] compact | abs=6.9e-07 rel=1.85e-07 ulp=217 | abs=1.19e-06 rel=3.19e-07 ulp=174 | 0.307 | abs=0.002 rel=0.000427 ulp=1138804 |
| 9 | B_fs[tf32] compact | abs=0.001 rel=0.000387 ulp=738122 | abs=0.001 rel=0.000387 ulp=738139 | 0.000783 | abs=0.003 rel=0.000705 ulp=1864689 |
| 9 | C_nm[tf32] compact | abs=0.001 rel=0.000387 ulp=738122 | abs=0.001 rel=0.000387 ulp=738139 | 0.000784 | abs=0.003 rel=0.000705 ulp=1864689 |

Replay repeat-launch bitwise check (immutable h0 row): id 1: yes; id 2: yes; id 3: yes; id 4: yes; id 5: yes; id 6: yes; id 7: yes; id 8: yes; id 9: yes

## Committed-state max_abs vs oracle by accepted depth (finite rows only; any non-finite comparison in a row => n/a)


id 1 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.02e-07 | 2.14e-07 | 2.14e-07 | 2.02e-07 | 2.02e-07 | 0.000326 | 0.000262 |
| 1 | 2.84e-07 | 2.56e-07 | 2.56e-07 | 2.56e-07 | 2.84e-07 | 0.000678 | 0.000334 |
| 2 | 3.49e-07 | 3.49e-07 | 3.49e-07 | 3.49e-07 | 3.49e-07 | 0.000632 | 0.000446 |
| 3 | 4.6e-07 | 4.58e-07 | 4.58e-07 | 4.58e-07 | 4.6e-07 | 0.000451 | 0.000483 |
| 4 | 3.17e-07 | 3.96e-07 | 3.96e-07 | 3.96e-07 | 3.17e-07 | 0.000718 | 0.000623 |
| 5 | 4.43e-07 | 3.33e-07 | 3.33e-07 | 3.33e-07 | 4.43e-07 | 0.000799 | 0.001 |

id 2 (layers.1.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.26e-08 | 3.87e-08 | 3.87e-08 | 3.26e-08 | 3.26e-08 | 0.000485 | 0.000186 |
| 1 | 4.29e-08 | 4.03e-08 | 4.03e-08 | 4.75e-08 | 4.29e-08 | 0.000176 | 0.000117 |
| 2 | 5.34e-08 | 5.33e-08 | 5.33e-08 | 6.16e-08 | 5.34e-08 | 0.000659 | 0.000219 |
| 3 | 4.99e-08 | 5.45e-08 | 5.45e-08 | 6.11e-08 | 4.99e-08 | 0.000329 | 0.000193 |
| 4 | 8.23e-08 | 6.41e-08 | 6.41e-08 | 6.27e-08 | 8.23e-08 | 0.000356 | 0.000232 |
| 5 | 1.02e-07 | 6.61e-08 | 7.75e-08 | 7.75e-08 | 1.02e-07 | 0.000456 | 0.000219 |

id 3 (layers.12.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 4.41e-08 | 4.41e-08 | 4.41e-08 | 5.9e-08 | 4.41e-08 | 0.000211 | 0.000126 |
| 1 | 4.07e-08 | 8.86e-08 | 8.86e-08 | 8.86e-08 | 4.07e-08 | 0.000242 | 0.000124 |
| 2 | 6.39e-08 | 5.81e-08 | 5.81e-08 | 4.06e-08 | 6.39e-08 | 0.000319 | 0.000162 |
| 3 | 8.81e-08 | 6.3e-08 | 6.3e-08 | 5.15e-08 | 8.81e-08 | 0.000325 | 0.000198 |
| 4 | 6.83e-08 | 5.27e-08 | 5.27e-08 | 5.27e-08 | 6.83e-08 | 0.000565 | 0.000265 |
| 5 | 7.64e-08 | 6.4e-08 | 6.4e-08 | 5.42e-08 | 7.64e-08 | 0.000443 | 0.00029 |

id 4 (layers.0.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.36e-06 | 1.46e-06 | 1.46e-06 | 1.16e-06 | 1.36e-06 | 0.012 | 0.010 |
| 1 | 1.82e-06 | 2.11e-06 | 2.35e-06 | 1.89e-06 | 1.82e-06 | 0.024 | 0.010 |
| 2 | 2.36e-06 | 2.56e-06 | 2.56e-06 | 1.63e-06 | 2.36e-06 | 0.034 | 0.016 |
| 3 | 2.03e-06 | 3.05e-06 | 3.05e-06 | 2.25e-06 | 2.03e-06 | 0.034 | 0.017 |
| 4 | 2.26e-06 | 3.04e-06 | 3.63e-06 | 1.92e-06 | 2.26e-06 | 0.034 | 0.017 |
| 5 | 3.22e-06 | 3.53e-06 | 4.24e-06 | 2.78e-06 | 3.22e-06 | 0.034 | 0.016 |

id 5 (synthetic[historical-like] seed=20260921 n=2):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.82e-07 | 3.95e-07 | 3.95e-07 | 3.82e-07 | 3.82e-07 | 0.001 | 0.001 |
| 1 | 5.6e-07 | 4.53e-07 | 4.53e-07 | 4.53e-07 | 5.6e-07 | 0.001 | 0.001 |

id 6 (synthetic[historical-like] seed=20260921 n=6):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.86e-07 | 3.86e-07 | 3.86e-07 | 3.86e-07 | 3.86e-07 | 0.001 | 0.001 |
| 1 | 6.73e-07 | 4.89e-07 | 4.89e-07 | 4.89e-07 | 6.73e-07 | 0.001 | 0.001 |
| 2 | 8.16e-07 | 6.15e-07 | 6.15e-07 | 6.15e-07 | 8.16e-07 | 0.001 | 0.002 |
| 3 | 9.19e-07 | 5.72e-07 | 5.72e-07 | 5.72e-07 | 9.19e-07 | 0.001 | 0.002 |
| 4 | 1.02e-06 | 5.61e-07 | 5.95e-07 | 5.61e-07 | 1.02e-06 | 0.002 | 0.001 |
| 5 | 1e-06 | 5.96e-07 | 6.02e-07 | 6.02e-07 | 1e-06 | 0.002 | 0.001 |

id 7 (synthetic[historical-like] seed=20260921 n=12):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 5.27e-07 | 5.27e-07 | 5.27e-07 | 5.27e-07 | 5.27e-07 | 0.002 | 0.001 |
| 1 | 5.44e-07 | 4.47e-07 | 4.47e-07 | 4.3e-07 | 5.44e-07 | 0.002 | 0.001 |
| 2 | 8.15e-07 | 5e-07 | 5e-07 | 5.22e-07 | 8.15e-07 | 0.002 | 0.001 |
| 3 | 7.03e-07 | 5.91e-07 | 6.31e-07 | 6.31e-07 | 7.03e-07 | 0.002 | 0.001 |
| 4 | 7.92e-07 | 5.36e-07 | 5.36e-07 | 5.36e-07 | 7.92e-07 | 0.002 | 0.001 |
| 5 | 8.64e-07 | 7.01e-07 | 7.01e-07 | 6.69e-07 | 8.64e-07 | 0.001 | 0.001 |
| 6 | 9.36e-07 | 6.18e-07 | 6.18e-07 | 6.18e-07 | 9.36e-07 | 0.002 | 0.001 |
| 7 | 1.1e-06 | 7.38e-07 | 7.38e-07 | 7.38e-07 | 1.1e-06 | 0.002 | 0.002 |
| 8 | 1.15e-06 | 7.4e-07 | 7.4e-07 | 7.27e-07 | 1.15e-06 | 0.002 | 0.001 |
| 9 | 1.3e-06 | 7.32e-07 | 7.32e-07 | 7.3e-07 | 1.3e-06 | 0.002 | 0.001 |
| 10 | 1.34e-06 | 9.32e-07 | 9.32e-07 | 7.55e-07 | 1.34e-06 | 0.002 | 0.001 |
| 11 | 1.35e-06 | 1.01e-06 | 1.01e-06 | 7.71e-07 | 1.35e-06 | 0.002 | 0.001 |

id 8 (synthetic[historical-like] seed=20260921 n=15):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.46e-07 | 3.51e-07 | 3.51e-07 | 3.46e-07 | 3.46e-07 | 0.001 | 0.001 |
| 1 | 6.32e-07 | 5.45e-07 | 5.45e-07 | 5.45e-07 | 6.32e-07 | 0.002 | 0.002 |
| 2 | 8.04e-07 | 5.77e-07 | 5.77e-07 | 5.33e-07 | 8.04e-07 | 0.002 | 0.002 |
| 3 | 1.06e-06 | 6.25e-07 | 6.25e-07 | 6.25e-07 | 1.06e-06 | 0.002 | 0.002 |

id 9 (synthetic[historical-like] seed=20260921 n=10):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 4.53e-07 | 4.53e-07 | 4.53e-07 | 4.53e-07 | 4.53e-07 | 0.001 | 0.001 |
| 1 | 6.09e-07 | 5.13e-07 | 5.13e-07 | 5.13e-07 | 6.09e-07 | 0.001 | 0.001 |
| 2 | 7.53e-07 | 5.45e-07 | 5.45e-07 | 5.45e-07 | 7.53e-07 | 0.001 | 0.001 |
| 3 | 9.31e-07 | 6.35e-07 | 6.35e-07 | 6.35e-07 | 9.31e-07 | 0.001 | 0.001 |
| 4 | 1.12e-06 | 6.86e-07 | 6.86e-07 | 6.86e-07 | 1.12e-06 | 0.002 | 0.001 |
| 5 | 1.19e-06 | 6.9e-07 | 6.9e-07 | 6.9e-07 | 1.19e-06 | 0.002 | 0.001 |

## Controls

| id | Neumann d−1 terms (U err / out err) | B_fs N_PAD32 vs 16 (out exact / abs) | C_nm N_PAD32 vs 16 | B_fs sibling-reverse (out exact / U exact) | C_nm sibling-reverse | A_prod sibling-reverse out exact |
|---|---|---|---|---|---|---|
| 1 | 0.016 / 0.001 | 0.908 / 2.79e-09 | 0.930 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 2 | 7.22e-05 / 4.47e-07 | 0.876 / 9.31e-10 | 0.883 / 1.86e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 3 | 0.000627 / 4.77e-06 | 0.916 / 9.31e-10 | 0.931 / 9.31e-10 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 4 | 0.024 / 0.001 | 0.901 / 1.19e-07 | 0.909 / 1.19e-07 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 5 | 0.293 / 0.004 | 1.000 / 0 | 1.000 / 0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 6 | 1.15e-06 / 1.11e-07 | 0.909 / 2.98e-08 | 0.911 / 2.98e-08 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 7 | 9.49e-07 / 1.05e-07 | 0.770 / 2.24e-08 | 0.682 / 2.24e-08 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 8 | 0.001 / 7.26e-06 | 0.934 / 2.98e-08 | 0.934 / 2.98e-08 | 0.922 / 0.948 | 0.924 / 0.960 | 1.000 |
| 9 | 1.38e-06 / 1.39e-07 | 0.890 / 2.24e-08 | 0.892 / 1.49e-08 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |

## Transient / scratch memory (bytes, analytic from shapes; per request, per layer)

| quantity | id 1 | id 2 | id 3 | id 4 | id 5 | id 6 | id 7 | id 8 | id 9 |
|---|---|---|---|---|---|---|---|---|---|
| A_prod_scan_hbm_per_node_state_export | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_prod_scan_register_h_cache_per_program | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 |
| A_prod_replay_hbm_written_rows(depth+1) | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 6,291,456 | 18,874,368 | 37,748,736 | 12,582,912 | 18,874,368 |
| A_prod_activation_ring(k,v,a,b) | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 |
| A_legacy_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 |
| B_legacyWY_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 |
| BC_compact_factors_U+cumg | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 |
| BC_compact_commit_hbm_written_rows | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 |
| one_full_state_row | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 |

## Timing — NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['2748559'] seen; sampled observation, not a guarantee)

CUDA events, per-launch sync median / pipelined mean (µs), one layer, B1. Rows do DIFFERENT work (see WORK column); this is NOT an apples-to-apples kernel speedup table.

| kernel | WORK | id 1 | id 2 | id 3 | id 4 | id 5 | id 6 | id 7 | id 8 | id 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| A_prod_scan_verify_out16 | bf16 out store; no per-node state export; preallocated | 89.1 / 65.0 | 90.2 / 65.6 | 89.4 / 63.8 | 89.5 / 63.6 | 35.1 / 25.7 | 59.6 / 35.1 | 133.4 / 110.5 | 166.3 / 141.4 | 90.2 / 65.6 |
| A_prod_replay_commit_deepest | writes depth+1 fp32 state rows to the bank; preallocated | 103.7 / 89.8 | 102.8 / 88.7 | 103.6 / 91.2 | 103.7 / 90.6 | 55.5 / 37.0 | 103.7 / 91.5 | 207.1 / 191.2 | 58.8 / 49.5 | 104.0 / 86.2 |
| A_prod_replay_commit_zero_accept | writes 1 fp32 state row (root); preallocated | 53.6 / 35.6 | 53.5 / 37.6 | 53.6 / 35.7 | 53.6 / 35.6 | 53.5 / 35.4 | 53.5 / 35.8 | 53.4 / 35.7 | 53.6 / 35.6 | 53.6 / 35.9 |
| A_legacy_scan_with_state_export | bf16 out + FULL per-node fp32 state export (n_pad x VH x DV x DK x 4 B); preallocated | 245.1 / 235.7 | 245.1 / 234.3 | 244.9 / 231.7 | 245.1 / 231.8 | 29.0 / 17.7 | 116.7 / 107.0 | 306.3 / 297.3 | 518.3 / 510.0 | 247.0 / 237.5 |
| B_legacyWY_fp32closed_with_state_export | bf16 out + FULL per-node fp32 state export; preallocated | 762.9 / 749.2 | 761.1 / 745.8 | 760.4 / 747.0 | 759.1 / 749.4 | 652.7 / 638.9 | 699.6 / 687.6 | 810.2 / 795.1 | 1007.1 / 1002.2 | 773.5 / 757.7 |
| B_legacyWY_bf16bnd_with_state_export | bf16 out (+bf16 boundary taps) + FULL per-node fp32 state export; preallocated | 1645.7 / 1640.4 | 1645.7 / 1637.8 | 1644.9 / 1642.5 | 1643.6 / 1642.7 | 1272.2 / 1260.5 | 1313.0 / 1302.3 | 2343.1 / 2327.8 | 4465.8 / 4462.2 | 1671.6 / 1672.8 |
| B_fs[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.0 / 9.5 | 18.7 / 9.5 | 18.9 / 9.7 | 18.8 / 9.7 | 18.7 / 9.7 | 18.8 / 9.7 | 18.0 / 9.6 | 18.8 / 9.9 | 18.1 / 9.7 |
| B_fs[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 110.0 / 96.7 | 110.0 / 100.1 | 110.8 / 96.6 | 110.6 / 96.9 | 110.8 / 98.2 | 112.0 / 98.4 | 110.8 / 96.4 | 96.6 / 83.2 | 110.0 / 96.4 |
| B_fs[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.7 / 9.5 | 16.0 / 9.6 | 16.8 / 9.7 | 16.7 / 9.7 | 16.7 / 9.7 | 16.8 / 9.9 | 17.0 / 10.0 | 16.8 / 9.7 | 16.9 / 9.8 |
| B_fs[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 41.5 / 28.7 | 42.4 / 28.8 | 42.4 / 28.7 | 42.4 / 30.7 | 41.3 / 28.6 | 41.2 / 28.8 | 41.9 / 28.7 | 43.1 / 28.7 | 42.4 / 31.0 |
| C_nm[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.0 / 9.6 | 18.0 / 9.5 | 18.7 / 10.5 | 18.8 / 11.2 | 18.1 / 9.7 | 18.8 / 10.2 | 18.8 / 9.8 | 18.8 / 9.9 | 18.0 / 9.7 |
| C_nm[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 98.7 / 90.6 | 98.3 / 91.4 | 98.5 / 89.1 | 98.6 / 89.6 | 96.1 / 87.2 | 95.7 / 82.6 | 94.5 / 81.1 | 98.5 / 90.1 | 93.6 / 85.5 |
| C_nm[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.8 / 9.6 | 16.8 / 11.7 | 16.8 / 9.8 | 16.9 / 9.6 | 16.7 / 9.8 | 16.8 / 9.7 | 16.8 / 9.7 | 16.7 / 9.7 | 16.9 / 9.9 |
| C_nm[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 31.1 / 18.5 | 31.1 / 18.5 | 31.2 / 18.5 | 31.9 / 18.5 | 29.0 / 16.4 | 31.0 / 18.4 | 35.0 / 20.6 | 31.1 / 18.5 | 32.2 / 18.5 |
| native_sg_chain_depth5_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | 158.7 / 154.1 | 157.9 / 157.2 | 157.8 / 156.3 | 157.7 / 159.3 | — | 160.8 / 160.6 | — | — | 154.8 / 154.8 |
| native_pk_one_token(context) | 1 token; helper allocates mixed/a/b/out per call (NOT matched work) | 120.9 / 115.2 | 122.0 / 120.6 | 121.9 / 122.7 | 121.9 / 121.5 | 120.9 / 115.3 | 117.9 / 112.8 | 122.9 / 118.0 | 119.6 / 114.3 | 120.6 / 115.6 |
| native_sg_chain_depth1_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | — | — | — | — | 71.9 / 65.7 | — | — | — | — |
| native_sg_chain_depth11_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | — | — | — | — | — | — | 285.3 / 281.9 | — | — |
| native_sg_chain_depth3_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | — | — | — | — | — | — | — | 118.0 / 117.3 | — |

In-band probe-drift criterion (4096² fp16 matmul before→after, pass iff drift ≤ 5 %): id 1: 1825.07839→1791.74557 µs drift=0.018 **PASS** [2026-09-21T22:28:12Z–2026-09-21T22:28:14Z]; id 2: 1788.67359→1790.12642 µs drift=0.000812 **PASS** [2026-09-21T22:28:15Z–2026-09-21T22:28:17Z]; id 3: 1830.92804→1787.36153 µs drift=0.024 **PASS** [2026-09-21T22:28:18Z–2026-09-21T22:28:20Z]; id 4: 1822.83516→1861.96327 µs drift=0.021 **PASS** [2026-09-21T22:28:21Z–2026-09-21T22:28:22Z]; id 5: 1903.59039→1823.83366 µs drift=0.042 **PASS** [2026-09-21T22:29:11Z–2026-09-21T22:29:13Z]; id 6: 1970.63198→1972.76325 µs drift=0.001 **PASS** [2026-09-21T22:30:24Z–2026-09-21T22:30:27Z]; id 7: 1906.46400→1923.50235 µs drift=0.009 **PASS** [2026-09-21T22:32:44Z–2026-09-21T22:32:47Z]; id 8: 1834.72805→1865.42873 µs drift=0.017 **PASS** [2026-09-21T22:35:59Z–2026-09-21T22:36:03Z]; id 9: 1813.92002→1837.88471 µs drift=0.013 **PASS** [2026-09-21T22:36:04Z–2026-09-21T22:36:05Z]

iters=200, warmup=20. Not a serving throughput measurement.

## B4 arm (SYNTHETIC, 4 requests sharing layer params, batched factor kernels)

B4 synthetic caterpillar (shared layer params): n=10 depth=5
- B_fs: per-request out32/U/commit max_abs vs own oracle = req0: 1.31e-07/1.02e-06/6.05e-07(target 9); req1: 1.3e-07/1.11e-06/5.78e-07(target 7); req2: 1.42e-07/9.23e-07/4.94e-07(target 5); req3: 1.2e-07/1.15e-06/4.09e-07(target 0); B4-vs-B1 bitwise (request 2): {'out': True, 'U': True}; timing verify 372.9/357.6 µs, commit 84.5/77.1 µs (batched fp32-out verify; 4 requests)
- C_nm: per-request out32/U/commit max_abs vs own oracle = req0: 1.39e-07/1.02e-06/6.05e-07(target 9); req1: 1.3e-07/1.11e-06/5.78e-07(target 7); req2: 1.42e-07/1.04e-06/4.94e-07(target 5); req3: 1.2e-07/1.15e-06/4.09e-07(target 0); B4-vs-B1 bitwise (request 2): {'out': True, 'U': True}; timing verify 338.2/334.2 µs, commit 98.5/98.1 µs (batched fp32-out verify; 4 requests)
- A_prod_scan_4x_launches_timing_us: 280.4 / 259.0 µs (4 sequential B1 launches, bf16 out)

## Validity

INVALID (non-finite) comparison cells in this summary: **0**. Probe-drift failures: **0**. Manifest status: **completed**; start/end source hashes match: **yes**.
