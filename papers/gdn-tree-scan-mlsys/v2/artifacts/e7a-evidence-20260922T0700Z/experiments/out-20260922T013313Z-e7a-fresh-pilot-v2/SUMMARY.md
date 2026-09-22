# E7a run summary — `out-20260922T013313Z-e7a-fresh-pilot-v2`

Manifest status **completed**; start/end source hashes match: **yes**; image `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`; torch 2.11.0+cu130 / triton 3.6.0; GPU NVIDIA GB10 cc [12, 1]; driver `NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  59`.
Production kernel flags: {"FIXED32_MODE": null, "scan_align_on": false, "npad_invariant_on": false, "parent_gather_on": false, "hc_internal_on": false, "BV_prod": 16, "num_warps_prod": 8, "tf32_matmul_allowed": false, "replay_h0_source_column": 15, "scan_h0_source_column": 0}. Triton cubins hashed: 30.
Source sha256: `prod_kernel`=d9dd0c697b46…, `legacy_wy`=e7c2a2a8b658…, `e7a_core`=89feb49d2bf1…, `e7a_kernels`=15c353625218…, `e7a_device`=c6a4e1e9df0e…, `native_fused_sigmoid_gating`=000ab8996af9…, `native_fused_recurrent`=3a2a3c5245ac…

**Timing attribution:** NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['2883522'] seen; sampled observation, not a guarantee) — window ['2026-09-22T01:33:13.975000', '2026-09-22T01:35:33.061000'], util samples 140, zero-util fraction 0.821, samples >5 %: 24, compute PIDs seen: {'2883522': {'samples': 134, 'names': ['[No data]', 'python3']}}, container PIDs: ['2883522'], coverage gaps: 0.

All mechanisms B/C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems. Historical payloads are dependent historical operand sets (one topology, B1), not independent prefixes; synthetic rows are labeled. Rows marked **INVALID** contain non-finite values in candidate or reference and carry no error value.

## Operand sets

| id | label | provenance | topology | n | depth | g_min | cum_g_min | P_min | min visible decay ratio | fp32 underflow risk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | layers.62.linear_attn | VERIFIED-FRESH sha256 02b37fff1f85… (language_model.model.layers.62.linear_attn); prefix p072 hash 4022f519838d… tokens 256 req cmpl-948634bfbd59d29d; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.400 | -13.537 | 1.32e-06 | 1.33e-05 | no |
| 2 | layers.62.linear_attn | VERIFIED-FRESH sha256 dfafbe339d71… (language_model.model.layers.62.linear_attn); prefix p017 hash d75829a949cf… tokens 256 req cmpl-a8143cb9e0bfed98; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.353 | -14.020 | 8.15e-07 | 1.31e-05 | no |
| 3 | layers.62.linear_attn | VERIFIED-FRESH sha256 6742e2d92016… (language_model.model.layers.62.linear_attn); prefix p015 hash 90e85b57382a… tokens 256 req cmpl-88cd1629470301e1; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.302 | -12.794 | 2.78e-06 | 1.31e-05 | no |
| 4 | layers.62.linear_attn | VERIFIED-FRESH sha256 d8307e795dc7… (language_model.model.layers.62.linear_attn); prefix p021 hash 232827856584… tokens 256 req cmpl-83da9d139dd62fb0; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.095 | -12.328 | 4.43e-06 | 1.35e-05 | no |
| 5 | layers.62.linear_attn | VERIFIED-FRESH sha256 eb35f6708ad5… (language_model.model.layers.62.linear_attn); prefix p085 hash 3b2eb67c915e… tokens 4096 req cmpl-868f02cd5895aaf0; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.947 | -13.527 | 1.33e-06 | 7.2e-06 | no |
| 6 | layers.62.linear_attn | VERIFIED-FRESH sha256 e0b635931358… (language_model.model.layers.62.linear_attn); prefix p095 hash 0ec0f7292581… tokens 4096 req cmpl-bb16039e9d8ee7d8; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.779 | -17.438 | 2.67e-08 | 6.03e-07 | no |
| 7 | layers.62.linear_attn | VERIFIED-FRESH sha256 3b411271f4e3… (language_model.model.layers.62.linear_attn); prefix p058 hash a1be6d26630b… tokens 4096 req cmpl-84a57a1f782993cf; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.140 | -11.254 | 1.3e-05 | 9.42e-05 | no |
| 8 | layers.62.linear_attn | VERIFIED-FRESH sha256 f2672729ea95… (language_model.model.layers.62.linear_attn); prefix p083 hash 6dcdc6e0dd72… tokens 4096 req cmpl-89c5816c72d7f886; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.335 | -15.169 | 2.58e-07 | 4.3e-06 | no |

## References vs fp64 oracle (same rounded operands)

| id | native_sg out32 | native_sg state | native_pk out32 vs oracle | native_pk state vs oracle | native_pk state vs ITS OWN oracle (beta bf16-rt + div/sqrt) | seam magnitude fp64 (oracle_pk vs oracle, state) | pk vs sg out16 exact | sg fp32-io vs bf16-io state max_abs |
|---|---|---|---|---|---|---|---|---|
| 1 | abs=9.94e-09 rel=1.85e-07 ulp=43 | abs=5.06e-07 rel=1.55e-07 ulp=17 | abs=7.37e-05 rel=0.001 ulp=360847 | abs=0.000765 rel=0.000234 ulp=146067 | abs=4.34e-07 rel=1.33e-07 ulp=23 | abs=0.000765 rel=0.000234 | 0.843 | 2.98e-08 |
| 2 | abs=2.05e-08 rel=2.16e-07 ulp=47 | abs=5.6e-07 rel=1.85e-07 ulp=35 | abs=7.76e-05 rel=0.000818 ulp=374626 | abs=0.004 rel=0.001 ulp=527162 | abs=4.24e-07 rel=1.4e-07 ulp=28 | abs=0.004 rel=0.001 | 0.831 | 5.96e-08 |
| 3 | abs=1.03e-08 rel=1.34e-07 ulp=25 | abs=8.45e-07 rel=2.84e-07 ulp=28 | abs=6.76e-05 rel=0.000872 ulp=344296 | abs=0.00061 rel=0.000205 ulp=372360 | abs=4.61e-07 rel=1.55e-07 ulp=18 | abs=0.00061 rel=0.000205 | 0.846 | 5.96e-08 |
| 4 | abs=1.19e-08 rel=1.07e-07 ulp=32 | abs=6.82e-07 rel=2.61e-07 ulp=14 | abs=6.58e-05 rel=0.000592 ulp=177633 | abs=0.000888 rel=0.00034 ulp=207223 | abs=5.05e-07 rel=1.93e-07 ulp=14 | abs=0.000888 rel=0.000339 | 0.855 | 5.96e-08 |
| 5 | abs=1.58e-08 rel=1.13e-07 ulp=30 | abs=6.31e-07 rel=1.46e-07 ulp=9 | abs=5.14e-05 rel=0.000368 ulp=147735 | abs=0.001 rel=0.000277 ulp=188757 | abs=5.43e-07 rel=1.26e-07 ulp=18 | abs=0.001 rel=0.000277 | 0.858 | 2.98e-08 |
| 6 | abs=2.72e-08 rel=1.36e-07 ulp=29 | abs=1.17e-06 rel=2.94e-07 ulp=23 | abs=4.64e-05 rel=0.000231 ulp=137721 | abs=0.003 rel=0.000716 ulp=121459 | abs=1.17e-06 rel=2.93e-07 ulp=20 | abs=0.003 rel=0.000716 | 0.844 | 2.38e-07 |
| 7 | abs=1.51e-08 rel=1.75e-07 ulp=23 | abs=1.3e-06 rel=2.93e-07 ulp=27 | abs=4.64e-05 rel=0.000537 ulp=266996 | abs=0.000573 rel=0.00013 ulp=107235 | abs=1.16e-06 rel=2.63e-07 ulp=14 | abs=0.000573 rel=0.00013 | 0.805 | 5.96e-08 |
| 8 | abs=1.31e-08 rel=1.07e-07 ulp=33 | abs=9.09e-07 rel=1.94e-07 ulp=14 | abs=4.9e-05 rel=0.000401 ulp=133372 | abs=0.001 rel=0.000316 ulp=179279 | abs=6.65e-07 rel=1.42e-07 ulp=12 | abs=0.001 rel=0.000315 | 0.855 | 2.98e-08 |

## Stage 1 — verifier outputs and correction factors (vs fp64 oracle; out16 exact fraction vs native spec-update kernel)

| id | mechanism | out32 vs oracle | out16 exact vs native_sg | factors U vs oracle u |
|---|---|---|---|---|
| 1 | A_prod | abs=9.94e-09 rel=1.85e-07 ulp=43 | 1.000 | — |
| 1 | B_legacyWY fp32-closed | abs=9.94e-09 rel=1.85e-07 ulp=69 | — | — |
| 1 | B_legacyWY bf16-bnd | abs=0.000145 rel=0.003 ulp=843537 | — | — |
| 1 | B_fs[ieee] | abs=1.95e-08 rel=3.62e-07 ulp=71 | 1.000 | abs=1.87e-07 rel=2.53e-07 ulp=353 |
| 1 | C_nm[ieee] | abs=1.95e-08 rel=3.62e-07 ulp=76 | 1.000 | abs=1.94e-07 rel=2.62e-07 ulp=355 |
| 1 | B_fs[tf32] | abs=4.25e-05 rel=0.000789 ulp=303235 | 0.812 | abs=0.000334 rel=0.000452 ulp=576532 |
| 1 | C_nm[tf32] | abs=3.98e-05 rel=0.00074 ulp=313903 | 0.811 | abs=0.000711 rel=0.000961 ulp=584154 |
| 2 | A_prod | abs=2.05e-08 rel=2.16e-07 ulp=47 | 1.000 | — |
| 2 | B_legacyWY fp32-closed | abs=1.19e-08 rel=1.26e-07 ulp=113 | — | — |
| 2 | B_legacyWY bf16-bnd | abs=0.000152 rel=0.002 ulp=1173971 | — | — |
| 2 | B_fs[ieee] | abs=2.68e-08 rel=2.82e-07 ulp=217 | 1.000 | abs=3.4e-07 rel=1.17e-07 ulp=321 |
| 2 | C_nm[ieee] | abs=2.68e-08 rel=2.82e-07 ulp=175 | 1.000 | abs=3.38e-07 rel=1.17e-07 ulp=324 |
| 2 | B_fs[tf32] | abs=6.92e-05 rel=0.00073 ulp=670248 | 0.786 | abs=0.000914 rel=0.000315 ulp=248895 |
| 2 | C_nm[tf32] | abs=6.88e-05 rel=0.000726 ulp=670248 | 0.793 | abs=0.002 rel=0.000546 ulp=304479 |
| 3 | A_prod | abs=1.03e-08 rel=1.34e-07 ulp=25 | 1.000 | — |
| 3 | B_legacyWY fp32-closed | abs=9.29e-09 rel=1.2e-07 ulp=42 | — | — |
| 3 | B_legacyWY bf16-bnd | abs=0.000205 rel=0.003 ulp=650451 | — | — |
| 3 | B_fs[ieee] | abs=2.25e-08 rel=2.91e-07 ulp=109 | 1.000 | abs=1.77e-07 rel=1.57e-07 ulp=323 |
| 3 | C_nm[ieee] | abs=2.25e-08 rel=2.91e-07 ulp=109 | 1.000 | abs=2.07e-07 rel=1.84e-07 ulp=322 |
| 3 | B_fs[tf32] | abs=6.01e-05 rel=0.000776 ulp=227316 | 0.796 | abs=0.00038 rel=0.000338 ulp=339136 |
| 3 | C_nm[tf32] | abs=6.01e-05 rel=0.000776 ulp=293356 | 0.799 | abs=0.000369 rel=0.000328 ulp=398877 |
| 4 | A_prod | abs=1.19e-08 rel=1.07e-07 ulp=32 | 1.000 | — |
| 4 | B_legacyWY fp32-closed | abs=1.23e-08 rel=1.1e-07 ulp=36 | — | — |
| 4 | B_legacyWY bf16-bnd | abs=0.000288 rel=0.003 ulp=597148 | — | — |
| 4 | B_fs[ieee] | abs=2.73e-08 rel=2.46e-07 ulp=37 | 1.000 | abs=1.35e-07 rel=1.25e-07 ulp=115 |
| 4 | C_nm[ieee] | abs=2.73e-08 rel=2.46e-07 ulp=43 | 1.000 | abs=1.72e-07 rel=1.59e-07 ulp=116 |
| 4 | B_fs[tf32] | abs=7.86e-05 rel=0.000707 ulp=194840 | 0.803 | abs=0.000308 rel=0.000285 ulp=352150 |
| 4 | C_nm[tf32] | abs=7.82e-05 rel=0.000704 ulp=208065 | 0.805 | abs=0.000394 rel=0.000364 ulp=351772 |
| 5 | A_prod | abs=1.58e-08 rel=1.13e-07 ulp=30 | 1.000 | — |
| 5 | B_legacyWY fp32-closed | abs=1.51e-08 rel=1.08e-07 ulp=45 | — | — |
| 5 | B_legacyWY bf16-bnd | abs=0.000208 rel=0.001 ulp=961874 | — | — |
| 5 | B_fs[ieee] | abs=4.2e-08 rel=3e-07 ulp=57 | 1.000 | abs=3.71e-07 rel=3.53e-07 ulp=347 |
| 5 | C_nm[ieee] | abs=4.2e-08 rel=3e-07 ulp=57 | 1.000 | abs=3.71e-07 rel=3.53e-07 ulp=339 |
| 5 | B_fs[tf32] | abs=0.00012 rel=0.000855 ulp=147603 | 0.798 | abs=0.000414 rel=0.000394 ulp=683379 |
| 5 | C_nm[tf32] | abs=0.00012 rel=0.000855 ulp=144443 | 0.800 | abs=0.000431 rel=0.00041 ulp=755241 |
| 6 | A_prod | abs=2.72e-08 rel=1.36e-07 ulp=29 | 1.000 | — |
| 6 | B_legacyWY fp32-closed | abs=1.99e-08 rel=9.9e-08 ulp=33 | — | — |
| 6 | B_legacyWY bf16-bnd | abs=0.000242 rel=0.001 ulp=586767 | — | — |
| 6 | B_fs[ieee] | abs=4.45e-08 rel=2.21e-07 ulp=44 | 1.000 | abs=2.15e-07 rel=1.05e-07 ulp=176 |
| 6 | C_nm[ieee] | abs=4.45e-08 rel=2.21e-07 ulp=49 | 1.000 | abs=2.26e-07 rel=1.1e-07 ulp=176 |
| 6 | B_fs[tf32] | abs=0.000134 rel=0.000668 ulp=101877 | 0.794 | abs=0.000486 rel=0.000236 ulp=404595 |
| 6 | C_nm[tf32] | abs=0.000134 rel=0.000668 ulp=101877 | 0.796 | abs=0.000497 rel=0.000242 ulp=408342 |
| 7 | A_prod | abs=1.51e-08 rel=1.75e-07 ulp=23 | 1.000 | — |
| 7 | B_legacyWY fp32-closed | abs=1.47e-08 rel=1.7e-07 ulp=31 | — | — |
| 7 | B_legacyWY bf16-bnd | abs=0.000183 rel=0.002 ulp=1525923 | — | — |
| 7 | B_fs[ieee] | abs=2.75e-08 rel=3.18e-07 ulp=70 | 1.000 | abs=1.75e-07 rel=1.5e-07 ulp=148 |
| 7 | C_nm[ieee] | abs=2.75e-08 rel=3.18e-07 ulp=69 | 1.000 | abs=1.9e-07 rel=1.63e-07 ulp=147 |
| 7 | B_fs[tf32] | abs=5.96e-05 rel=0.000689 ulp=199973 | 0.769 | abs=0.000302 rel=0.000259 ulp=418533 |
| 7 | C_nm[tf32] | abs=5.96e-05 rel=0.000689 ulp=212299 | 0.770 | abs=0.000332 rel=0.000285 ulp=410650 |
| 8 | A_prod | abs=1.31e-08 rel=1.07e-07 ulp=33 | 1.000 | — |
| 8 | B_legacyWY fp32-closed | abs=1.28e-08 rel=1.05e-07 ulp=34 | — | — |
| 8 | B_legacyWY bf16-bnd | abs=0.000171 rel=0.001 ulp=612300 | — | — |
| 8 | B_fs[ieee] | abs=4.19e-08 rel=3.43e-07 ulp=56 | 1.000 | abs=1.62e-07 rel=1.22e-07 ulp=147 |
| 8 | C_nm[ieee] | abs=4.19e-08 rel=3.43e-07 ulp=56 | 1.000 | abs=2.4e-07 rel=1.8e-07 ulp=147 |
| 8 | B_fs[tf32] | abs=8.25e-05 rel=0.000674 ulp=137880 | 0.792 | abs=0.00054 rel=0.000406 ulp=443177 |
| 8 | C_nm[tf32] | abs=8.65e-05 rel=0.000707 ulp=134682 | 0.793 | abs=0.000959 rel=0.000722 ulp=502769 |

Production-scan identity checks (NUMERIC equality folds +0/-0; BITWISE is integer-view equality): id 1: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 2: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 3: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 4: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 5: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 6: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 7: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 8: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes

Seam ablation for the one-token decode reference (native_pk state vs fp64 oracles applying ONE convention; max_rel): id 1: beta-bf16-rt only=rel=1.33e-07, div-sqrt only=rel=0.000234, both=rel=1.33e-07; id 2: beta-bf16-rt only=rel=1.4e-07, div-sqrt only=rel=0.001, both=rel=1.4e-07; id 3: beta-bf16-rt only=rel=1.55e-07, div-sqrt only=rel=0.000205, both=rel=1.55e-07; id 4: beta-bf16-rt only=rel=1.93e-07, div-sqrt only=rel=0.00034, both=rel=1.93e-07; id 5: beta-bf16-rt only=rel=1.26e-07, div-sqrt only=rel=0.000277, both=rel=1.26e-07; id 6: beta-bf16-rt only=rel=2.93e-07, div-sqrt only=rel=0.000716, both=rel=2.93e-07; id 7: beta-bf16-rt only=rel=2.63e-07, div-sqrt only=rel=0.00013, both=rel=2.63e-07; id 8: beta-bf16-rt only=rel=1.42e-07, div-sqrt only=rel=0.000316, both=rel=1.42e-07

`A_legacy_state*` rows are excluded: the June-8 vendored sequential scan's export path skips the root update (source-inspected); when/which binary changed that path is NOT established. Its outputs are bit-identical to the production scan.

## Stage 2 — commit from IDENTICAL oracle factors (compact reconstruction kernel; vs oracle state / vs native_sg state)

| id | B_fs[ieee] vs oracle | vs native_sg | C_nm[ieee] vs oracle | B_fs[tf32] vs oracle | C_nm[tf32] vs oracle |
|---|---|---|---|---|---|
| 1 | abs=4.14e-07 rel=1.27e-07 ulp=16 | abs=7.15e-07 rel=2.18e-07 ulp=23 | abs=4.14e-07 rel=1.27e-07 ulp=16 | abs=0.000597 rel=0.000182 ulp=69474 | abs=0.000597 rel=0.000182 ulp=69474 |
| 2 | abs=3.94e-07 rel=1.3e-07 ulp=42 | abs=5.96e-07 rel=1.97e-07 ulp=50 | abs=3.94e-07 rel=1.3e-07 ulp=42 | abs=0.002 rel=0.000688 ulp=247966 | abs=0.002 rel=0.000688 ulp=247966 |
| 3 | abs=4.76e-07 rel=1.6e-07 ulp=14 | abs=1.19e-06 rel=4.01e-07 ulp=30 | abs=4.76e-07 rel=1.6e-07 ulp=14 | abs=0.000445 rel=0.00015 ulp=176743 | abs=0.000445 rel=0.00015 ulp=176743 |
| 4 | abs=4.2e-07 rel=1.61e-07 ulp=12 | abs=7.15e-07 rel=2.73e-07 ulp=15 | abs=4.2e-07 rel=1.61e-07 ulp=12 | abs=0.000571 rel=0.000218 ulp=65742 | abs=0.000571 rel=0.000218 ulp=65742 |
| 5 | abs=5.11e-07 rel=1.19e-07 ulp=7 | abs=7.15e-07 rel=1.66e-07 ulp=10 | abs=5.11e-07 rel=1.19e-07 ulp=7 | abs=0.000714 rel=0.000166 ulp=92813 | abs=0.000714 rel=0.000166 ulp=92813 |
| 6 | abs=4.59e-07 rel=1.15e-07 ulp=11 | abs=1.43e-06 rel=3.58e-07 ulp=22 | abs=4.59e-07 rel=1.15e-07 ulp=11 | abs=0.002 rel=0.000438 ulp=64688 | abs=0.002 rel=0.000438 ulp=64688 |
| 7 | abs=6.95e-07 rel=1.57e-07 ulp=14 | abs=1.91e-06 rel=4.31e-07 ulp=26 | abs=6.95e-07 rel=1.57e-07 ulp=14 | abs=0.000453 rel=0.000102 ulp=68888 | abs=0.000453 rel=0.000102 ulp=68888 |
| 8 | abs=7.87e-07 rel=1.68e-07 ulp=11 | abs=4.77e-07 rel=1.02e-07 ulp=12 | abs=7.87e-07 rel=1.68e-07 ulp=11 | abs=0.000616 rel=0.000131 ulp=43891 | abs=0.000616 rel=0.000131 ulp=43891 |

## Stage 3 — own factors + own commit (all accepted prefixes incl. zero-accept; vs oracle / native_sg / native_pk)

| id | mechanism | vs oracle | vs native_sg (spec-update) | exact frac vs native_sg | vs native_pk (one-token decode) |
|---|---|---|---|---|---|
| 1 | A_prod replay | abs=5.06e-07 rel=1.55e-07 ulp=17 | abs=2.98e-08 rel=9.1e-09 ulp=5 | 0.990 | abs=0.000765 rel=0.000234 ulp=146070 |
| 1 | B_legacyWY fp32-closed (materialized) | abs=3.92e-07 rel=1.2e-07 ulp=19 | — | — | — |
| 1 | B_fs[ieee] compact | abs=4.14e-07 rel=1.27e-07 ulp=41 | abs=7.15e-07 rel=2.18e-07 ulp=46 | 0.315 | abs=0.000765 rel=0.000234 ulp=146069 |
| 1 | C_nm[ieee] compact | abs=4.14e-07 rel=1.27e-07 ulp=38 | abs=7.15e-07 rel=2.18e-07 ulp=43 | 0.313 | abs=0.000765 rel=0.000234 ulp=146069 |
| 1 | B_fs[tf32] compact | abs=0.000455 rel=0.000139 ulp=127731 | abs=0.000455 rel=0.000139 ulp=127729 | 0.000382 | abs=0.000839 rel=0.000256 ulp=240232 |
| 1 | C_nm[tf32] compact | abs=0.000446 rel=0.000136 ulp=205619 | abs=0.000446 rel=0.000136 ulp=205617 | 0.000381 | abs=0.000839 rel=0.000256 ulp=318120 |
| 2 | A_prod replay | abs=5.6e-07 rel=1.85e-07 ulp=35 | abs=5.96e-08 rel=1.97e-08 ulp=6 | 0.988 | abs=0.004 rel=0.001 ulp=527155 |
| 2 | B_legacyWY fp32-closed (materialized) | abs=4.3e-07 rel=1.42e-07 ulp=76 | — | — | — |
| 2 | B_fs[ieee] compact | abs=3.97e-07 rel=1.31e-07 ulp=68 | abs=5.96e-07 rel=1.97e-07 ulp=76 | 0.295 | abs=0.004 rel=0.001 ulp=527160 |
| 2 | C_nm[ieee] compact | abs=3.97e-07 rel=1.31e-07 ulp=64 | abs=5.96e-07 rel=1.97e-07 ulp=72 | 0.292 | abs=0.004 rel=0.001 ulp=527158 |
| 2 | B_fs[tf32] compact | abs=0.002 rel=0.000693 ulp=336914 | abs=0.002 rel=0.000693 ulp=336922 | 0.000168 | abs=0.006 rel=0.002 ulp=858596 |
| 2 | C_nm[tf32] compact | abs=0.002 rel=0.000695 ulp=623102 | abs=0.002 rel=0.000695 ulp=623110 | 0.000166 | abs=0.006 rel=0.002 ulp=1144784 |
| 3 | A_prod replay | abs=8.45e-07 rel=2.84e-07 ulp=28 | abs=5.96e-08 rel=2.01e-08 ulp=5 | 0.990 | abs=0.00061 rel=0.000205 ulp=372367 |
| 3 | B_legacyWY fp32-closed (materialized) | abs=4.84e-07 rel=1.63e-07 ulp=29 | — | — | — |
| 3 | B_fs[ieee] compact | abs=4.76e-07 rel=1.6e-07 ulp=54 | abs=1.19e-06 rel=4.01e-07 ulp=58 | 0.319 | abs=0.00061 rel=0.000205 ulp=372314 |
| 3 | C_nm[ieee] compact | abs=4.76e-07 rel=1.6e-07 ulp=54 | abs=1.19e-06 rel=4.01e-07 ulp=58 | 0.315 | abs=0.00061 rel=0.000205 ulp=372332 |
| 3 | B_fs[tf32] compact | abs=0.000464 rel=0.000156 ulp=228715 | abs=0.000464 rel=0.000156 ulp=228709 | 0.000536 | abs=0.000738 rel=0.000248 ulp=600863 |
| 3 | C_nm[tf32] compact | abs=0.000604 rel=0.000203 ulp=274155 | abs=0.000604 rel=0.000203 ulp=274149 | 0.000539 | abs=0.000738 rel=0.000248 ulp=646303 |
| 4 | A_prod replay | abs=6.82e-07 rel=2.61e-07 ulp=14 | abs=5.96e-08 rel=2.28e-08 ulp=3 | 0.992 | abs=0.000888 rel=0.00034 ulp=207218 |
| 4 | B_legacyWY fp32-closed (materialized) | abs=4.15e-07 rel=1.59e-07 ulp=21 | — | — | — |
| 4 | B_fs[ieee] compact | abs=4.2e-07 rel=1.61e-07 ulp=21 | abs=7.15e-07 rel=2.73e-07 ulp=23 | 0.330 | abs=0.000888 rel=0.00034 ulp=207202 |
| 4 | C_nm[ieee] compact | abs=4.2e-07 rel=1.61e-07 ulp=21 | abs=7.15e-07 rel=2.73e-07 ulp=23 | 0.326 | abs=0.000888 rel=0.00034 ulp=207202 |
| 4 | B_fs[tf32] compact | abs=0.00053 rel=0.000202 ulp=65742 | abs=0.00053 rel=0.000203 ulp=65735 | 0.000622 | abs=0.001 rel=0.000459 ulp=266477 |
| 4 | C_nm[tf32] compact | abs=0.00053 rel=0.000202 ulp=69137 | abs=0.00053 rel=0.000203 ulp=69135 | 0.000613 | abs=0.001 rel=0.000459 ulp=272225 |
| 5 | A_prod replay | abs=6.31e-07 rel=1.46e-07 ulp=9 | abs=2.98e-08 rel=6.92e-09 ulp=2 | 0.993 | abs=0.001 rel=0.000277 ulp=188753 |
| 5 | B_legacyWY fp32-closed (materialized) | abs=6.98e-07 rel=1.62e-07 ulp=15 | — | — | — |
| 5 | B_fs[ieee] compact | abs=5.11e-07 rel=1.19e-07 ulp=26 | abs=7.15e-07 rel=1.66e-07 ulp=24 | 0.340 | abs=0.001 rel=0.000277 ulp=188748 |
| 5 | C_nm[ieee] compact | abs=5.11e-07 rel=1.19e-07 ulp=26 | abs=7.15e-07 rel=1.66e-07 ulp=24 | 0.336 | abs=0.001 rel=0.000277 ulp=188748 |
| 5 | B_fs[tf32] compact | abs=0.000683 rel=0.000159 ulp=112013 | abs=0.000683 rel=0.000158 ulp=112015 | 0.002 | abs=0.001 rel=0.000286 ulp=238312 |
| 5 | C_nm[tf32] compact | abs=0.00068 rel=0.000158 ulp=112013 | abs=0.00068 rel=0.000158 ulp=112015 | 0.002 | abs=0.001 rel=0.000286 ulp=234940 |
| 6 | A_prod replay | abs=1.17e-06 rel=2.94e-07 ulp=23 | abs=2.38e-07 rel=5.97e-08 ulp=4 | 0.992 | abs=0.003 rel=0.000716 ulp=128356 |
| 6 | B_legacyWY fp32-closed (materialized) | abs=4.53e-07 rel=1.13e-07 ulp=10 | — | — | — |
| 6 | B_fs[ieee] compact | abs=4.59e-07 rel=1.15e-07 ulp=18 | abs=1.43e-06 rel=3.58e-07 ulp=27 | 0.330 | abs=0.003 rel=0.000716 ulp=128364 |
| 6 | C_nm[ieee] compact | abs=4.59e-07 rel=1.15e-07 ulp=18 | abs=1.43e-06 rel=3.58e-07 ulp=27 | 0.326 | abs=0.003 rel=0.000716 ulp=128364 |
| 6 | B_fs[tf32] compact | abs=0.002 rel=0.000437 ulp=118903 | abs=0.002 rel=0.000437 ulp=118880 | 0.001 | abs=0.005 rel=0.001 ulp=211497 |
| 6 | C_nm[tf32] compact | abs=0.002 rel=0.000436 ulp=115350 | abs=0.002 rel=0.000436 ulp=115327 | 0.001 | abs=0.005 rel=0.001 ulp=215815 |
| 7 | A_prod replay | abs=1.3e-06 rel=2.93e-07 ulp=27 | abs=5.96e-08 rel=1.35e-08 ulp=5 | 0.991 | abs=0.000573 rel=0.00013 ulp=107229 |
| 7 | B_legacyWY fp32-closed (materialized) | abs=6.95e-07 rel=1.57e-07 ulp=16 | — | — | — |
| 7 | B_fs[ieee] compact | abs=6.95e-07 rel=1.57e-07 ulp=34 | abs=1.91e-06 rel=4.31e-07 ulp=45 | 0.311 | abs=0.000573 rel=0.00013 ulp=107228 |
| 7 | C_nm[ieee] compact | abs=6.95e-07 rel=1.57e-07 ulp=35 | abs=1.91e-06 rel=4.31e-07 ulp=47 | 0.310 | abs=0.000573 rel=0.00013 ulp=107225 |
| 7 | B_fs[tf32] compact | abs=0.000367 rel=8.29e-05 ulp=87251 | abs=0.000367 rel=8.29e-05 ulp=87224 | 0.000606 | abs=0.000752 rel=0.00017 ulp=181250 |
| 7 | C_nm[tf32] compact | abs=0.000348 rel=7.86e-05 ulp=87251 | abs=0.000348 rel=7.86e-05 ulp=87224 | 0.000608 | abs=0.000756 rel=0.000171 ulp=170848 |
| 8 | A_prod replay | abs=9.09e-07 rel=1.94e-07 ulp=12 | abs=2.98e-08 rel=6.36e-09 ulp=3 | 0.991 | abs=0.001 rel=0.000315 ulp=179279 |
| 8 | B_legacyWY fp32-closed (materialized) | abs=7.87e-07 rel=1.68e-07 ulp=12 | — | — | — |
| 8 | B_fs[ieee] compact | abs=7.87e-07 rel=1.68e-07 ulp=16 | abs=4.77e-07 rel=1.02e-07 ulp=17 | 0.339 | abs=0.001 rel=0.000316 ulp=179286 |
| 8 | C_nm[ieee] compact | abs=7.87e-07 rel=1.68e-07 ulp=16 | abs=4.77e-07 rel=1.02e-07 ulp=15 | 0.336 | abs=0.001 rel=0.000316 ulp=179293 |
| 8 | B_fs[tf32] compact | abs=0.000616 rel=0.000131 ulp=140358 | abs=0.000616 rel=0.000131 ulp=140350 | 0.000871 | abs=0.002 rel=0.000368 ulp=243306 |
| 8 | C_nm[tf32] compact | abs=0.000621 rel=0.000133 ulp=141338 | abs=0.000621 rel=0.000133 ulp=141330 | 0.000861 | abs=0.002 rel=0.000368 ulp=243306 |

Replay repeat-launch bitwise check (immutable h0 row): id 1: yes; id 2: yes; id 3: yes; id 4: yes; id 5: yes; id 6: yes; id 7: yes; id 8: yes

## Committed-state max_abs vs oracle by accepted depth (finite rows only; any non-finite comparison in a row => n/a)


id 1 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.78e-07 | 2.78e-07 | 2.78e-07 | 2.78e-07 | 2.78e-07 | 0.00047 | 0.000236 |
| 1 | 2.41e-07 | 3.1e-07 | 3.1e-07 | 3.1e-07 | 2.41e-07 | 0.000694 | 0.000302 |
| 2 | 3.55e-07 | 4.14e-07 | 4.14e-07 | 4.14e-07 | 3.55e-07 | 0.000509 | 0.000389 |
| 3 | 4e-07 | 2.65e-07 | 2.65e-07 | 2.65e-07 | 4e-07 | 0.000568 | 0.00035 |
| 4 | 5.06e-07 | 3.6e-07 | 3.6e-07 | 3.6e-07 | 5.06e-07 | 0.000765 | 0.000455 |
| 5 | 4.13e-07 | 3.83e-07 | 3.83e-07 | 3.83e-07 | 4.13e-07 | 0.000617 | 0.000408 |

id 2 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.38e-07 | 3.97e-07 | 3.97e-07 | 2.38e-07 | 2.38e-07 | 0.004 | 0.001 |
| 1 | 2.27e-07 | 2.69e-07 | 2.69e-07 | 2.33e-07 | 2.27e-07 | 0.004 | 0.002 |
| 2 | 1.98e-07 | 3.5e-07 | 3.5e-07 | 3.5e-07 | 1.98e-07 | 0.004 | 0.001 |
| 3 | 3.08e-07 | 3.94e-07 | 3.94e-07 | 3.94e-07 | 3.08e-07 | 0.004 | 0.002 |
| 4 | 3.95e-07 | 3.49e-07 | 3.49e-07 | 3.49e-07 | 3.95e-07 | 0.003 | 0.002 |
| 5 | 5.6e-07 | 2.67e-07 | 2.67e-07 | 2.67e-07 | 5.6e-07 | 0.003 | 0.002 |

id 3 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.8e-07 | 2.8e-07 | 2.8e-07 | 2.8e-07 | 2.8e-07 | 0.000314 | 0.000221 |
| 1 | 4.68e-07 | 2.3e-07 | 2.3e-07 | 2.3e-07 | 4.68e-07 | 0.000584 | 0.000224 |
| 2 | 5.61e-07 | 3.22e-07 | 3.22e-07 | 3.22e-07 | 5.61e-07 | 0.000604 | 0.000464 |
| 3 | 7.49e-07 | 3.06e-07 | 3.06e-07 | 3.06e-07 | 7.49e-07 | 0.00061 | 0.000309 |
| 4 | 7.64e-07 | 3.62e-07 | 3.62e-07 | 3.62e-07 | 7.64e-07 | 0.000536 | 0.000437 |
| 5 | 8.45e-07 | 4.76e-07 | 4.76e-07 | 4.76e-07 | 8.45e-07 | 0.000537 | 0.000444 |

id 4 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.83e-07 | 1.83e-07 | 1.83e-07 | 1.83e-07 | 1.83e-07 | 0.000212 | 0.000114 |
| 1 | 2.74e-07 | 1.48e-07 | 1.48e-07 | 1.48e-07 | 2.74e-07 | 0.000709 | 0.000185 |
| 2 | 4.42e-07 | 2.73e-07 | 2.73e-07 | 2.73e-07 | 4.42e-07 | 0.000794 | 0.00053 |
| 3 | 5.23e-07 | 2.03e-07 | 2.03e-07 | 2.03e-07 | 5.23e-07 | 0.000888 | 0.000371 |
| 4 | 6.82e-07 | 4.15e-07 | 4.15e-07 | 4.15e-07 | 6.82e-07 | 0.000386 | 0.000284 |
| 5 | 6.47e-07 | 4.2e-07 | 4.2e-07 | 4.2e-07 | 6.47e-07 | 0.000749 | 0.000264 |

id 5 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.68e-07 | 1.68e-07 | 1.68e-07 | 1.68e-07 | 1.68e-07 | 0.000479 | 0.000304 |
| 1 | 2.54e-07 | 2.54e-07 | 2.54e-07 | 2.54e-07 | 2.54e-07 | 0.000473 | 0.000336 |
| 2 | 3.99e-07 | 3.46e-07 | 3.46e-07 | 3.46e-07 | 3.99e-07 | 0.000906 | 0.000365 |
| 3 | 4.02e-07 | 3.76e-07 | 3.76e-07 | 3.76e-07 | 4.02e-07 | 0.001 | 0.000683 |
| 4 | 6.31e-07 | 4.04e-07 | 4.04e-07 | 4.04e-07 | 6.31e-07 | 0.000622 | 0.000485 |
| 5 | 4.21e-07 | 5.11e-07 | 5.11e-07 | 5.11e-07 | 4.21e-07 | 0.000982 | 0.00053 |

id 6 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.87e-07 | 2.87e-07 | 2.87e-07 | 2.87e-07 | 2.87e-07 | 0.000718 | 0.000398 |
| 1 | 4.64e-07 | 3.21e-07 | 3.21e-07 | 3.21e-07 | 4.64e-07 | 0.00064 | 0.000394 |
| 2 | 7.2e-07 | 2.95e-07 | 2.95e-07 | 2.95e-07 | 7.2e-07 | 0.001 | 0.000644 |
| 3 | 7.78e-07 | 3.38e-07 | 3.38e-07 | 3.38e-07 | 7.78e-07 | 0.003 | 0.002 |
| 4 | 9.8e-07 | 3.92e-07 | 3.92e-07 | 3.92e-07 | 9.8e-07 | 0.000474 | 0.000619 |
| 5 | 1.17e-06 | 4.59e-07 | 4.59e-07 | 4.59e-07 | 1.17e-06 | 0.000948 | 0.000592 |

id 7 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 4.45e-07 | 4.45e-07 | 4.45e-07 | 4.45e-07 | 4.45e-07 | 0.000262 | 0.000271 |
| 1 | 7.71e-07 | 3.06e-07 | 3.06e-07 | 3.06e-07 | 7.71e-07 | 0.000428 | 0.000235 |
| 2 | 1.07e-06 | 6.31e-07 | 6.31e-07 | 6.31e-07 | 1.07e-06 | 0.000573 | 0.000312 |
| 3 | 1.09e-06 | 6.57e-07 | 6.57e-07 | 6.57e-07 | 1.09e-06 | 0.000514 | 0.000289 |
| 4 | 1.3e-06 | 5.35e-07 | 5.35e-07 | 5.35e-07 | 1.3e-06 | 0.000514 | 0.000367 |
| 5 | 1.28e-06 | 6.95e-07 | 6.95e-07 | 6.95e-07 | 1.28e-06 | 0.000522 | 0.000314 |

id 8 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.81e-07 | 2.81e-07 | 2.81e-07 | 2.81e-07 | 2.81e-07 | 0.00039 | 0.000346 |
| 1 | 2.75e-07 | 2.79e-07 | 2.79e-07 | 2.79e-07 | 2.75e-07 | 0.001 | 0.000498 |
| 2 | 4.32e-07 | 4.32e-07 | 4.32e-07 | 4.32e-07 | 4.32e-07 | 0.00097 | 0.000513 |
| 3 | 4.69e-07 | 4.47e-07 | 4.47e-07 | 4.47e-07 | 4.69e-07 | 0.000974 | 0.000518 |
| 4 | 6.22e-07 | 6.22e-07 | 6.22e-07 | 6.22e-07 | 6.22e-07 | 0.000974 | 0.000569 |
| 5 | 9.09e-07 | 7.87e-07 | 7.87e-07 | 7.87e-07 | 9.09e-07 | 0.000966 | 0.000616 |

## Controls

| id | Neumann d−1 terms (U err / out err) | B_fs N_PAD32 vs 16 (out exact / abs) | C_nm N_PAD32 vs 16 | B_fs sibling-reverse (out exact / U exact) | C_nm sibling-reverse | A_prod sibling-reverse out exact |
|---|---|---|---|---|---|---|
| 1 | 0.023 / 0.001 | 0.924 / 3.73e-09 | 0.944 / 5.59e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 2 | 0.025 / 0.001 | 0.907 / 3.73e-09 | 0.945 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 3 | 0.025 / 0.002 | 0.912 / 3.73e-09 | 0.931 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 4 | 0.017 / 0.000746 | 0.928 / 3.73e-09 | 0.947 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 5 | 0.021 / 0.001 | 0.928 / 3.73e-09 | 0.944 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 6 | 0.038 / 0.002 | 0.917 / 7.45e-09 | 0.942 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 7 | 0.007 / 0.000419 | 0.873 / 3.73e-09 | 0.887 / 4.66e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 8 | 0.057 / 0.003 | 0.917 / 5.59e-09 | 0.931 / 7.45e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |

## Transient / scratch memory (bytes, analytic from shapes; per request, per layer)

| quantity | id 1 | id 2 | id 3 | id 4 | id 5 | id 6 | id 7 | id 8 |
|---|---|---|---|---|---|---|---|---|
| A_prod_scan_hbm_per_node_state_export | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_prod_scan_register_h_cache_per_program | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 |
| A_prod_replay_hbm_written_rows(depth+1) | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 |
| A_prod_activation_ring(k,v,a,b) | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 |
| A_legacy_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 |
| B_legacyWY_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 |
| BC_compact_factors_U+cumg | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 |
| BC_compact_commit_hbm_written_rows | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 |
| one_full_state_row | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 |

## Timing — NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['2883522'] seen; sampled observation, not a guarantee)

CUDA events, per-launch sync median / pipelined mean (µs), one layer, B1. Rows do DIFFERENT work (see WORK column); this is NOT an apples-to-apples kernel speedup table.

| kernel | WORK | id 1 | id 2 | id 3 | id 4 | id 5 | id 6 | id 7 | id 8 |
|---|---|---|---|---|---|---|---|---|---|
| A_prod_scan_verify_out16 | bf16 out store; no per-node state export; preallocated | 88.6 / 65.5 | 88.5 / 65.5 | 89.4 / 64.6 | 89.5 / 65.6 | 90.1 / 65.3 | 90.0 / 63.9 | 89.5 / 64.1 | 90.2 / 67.8 |
| A_prod_replay_commit_deepest | writes depth+1 fp32 state rows to the bank; preallocated | 102.6 / 89.2 | 102.7 / 88.3 | 103.5 / 87.1 | 103.3 / 88.8 | 103.6 / 88.3 | 102.8 / 89.9 | 102.8 / 85.9 | 103.5 / 90.4 |
| A_prod_replay_commit_zero_accept | writes 1 fp32 state row (root); preallocated | 53.6 / 35.7 | 53.6 / 37.3 | 53.6 / 35.7 | 53.6 / 35.7 | 53.6 / 37.7 | 53.6 / 36.0 | 53.6 / 38.1 | 53.6 / 36.1 |
| A_legacy_scan_with_state_export | bf16 out + FULL per-node fp32 state export (n_pad x VH x DV x DK x 4 B); preallocated | 249.2 / 238.9 | 242.9 / 232.3 | 243.1 / 233.4 | 243.9 / 240.3 | 244.0 / 229.5 | 243.0 / 231.1 | 242.1 / 232.1 | 243.1 / 230.1 |
| B_legacyWY_fp32closed_with_state_export | bf16 out + FULL per-node fp32 state export; preallocated | 756.9 / 745.4 | 756.0 / 743.7 | 756.0 / 744.7 | 756.1 / 744.3 | 756.9 / 745.5 | 756.0 / 742.5 | 755.1 / 742.2 | 756.1 / 743.9 |
| B_legacyWY_bf16bnd_with_state_export | bf16 out (+bf16 boundary taps) + FULL per-node fp32 state export; preallocated | 1638.5 / 1630.8 | 1646.8 / 1634.5 | 1646.0 / 1648.5 | 1646.5 / 1648.7 | 1646.8 / 1637.6 | 1647.0 / 1634.7 | 1646.9 / 1634.1 | 1646.8 / 1649.7 |
| B_fs[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.8 / 9.7 | 18.7 / 9.7 | 18.8 / 9.5 | 18.9 / 9.9 | 18.9 / 9.7 | 18.0 / 9.6 | 18.0 / 9.6 | 18.1 / 9.6 |
| B_fs[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 110.8 / 96.1 | 110.1 / 96.5 | 110.0 / 96.4 | 108.9 / 94.8 | 110.8 / 96.4 | 110.1 / 99.7 | 109.9 / 96.3 | 110.9 / 96.7 |
| B_fs[ieee]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 106.1 / 92.4 | 106.7 / 92.5 | 106.0 / 92.5 | 105.9 / 92.3 | 105.0 / 91.4 | 106.8 / 92.6 | 106.5 / 93.3 | 106.8 / 92.2 |
| B_fs[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.8 / 9.8 | 16.1 / 9.6 | 16.2 / 9.7 | 16.1 / 9.7 | 16.8 / 9.8 | 16.7 / 9.6 | 16.4 / 9.5 | 16.7 / 9.7 |
| B_fs[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 41.5 / 28.8 | 42.4 / 28.7 | 42.4 / 28.7 | 42.3 / 31.1 | 41.5 / 28.7 | 41.6 / 28.7 | 41.5 / 28.7 | 41.5 / 28.7 |
| B_fs[tf32]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 42.0 / 31.2 | 42.4 / 28.7 | 42.4 / 28.8 | 42.4 / 30.0 | 41.7 / 28.7 | 42.3 / 28.7 | 42.2 / 28.7 | 42.5 / 28.7 |
| C_nm[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.9 / 9.8 | 18.1 / 9.6 | 18.4 / 10.2 | 18.0 / 9.5 | 18.8 / 9.7 | 18.1 / 11.6 | 18.5 / 10.5 | 18.0 / 9.8 |
| C_nm[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 100.6 / 91.9 | 100.0 / 91.6 | 100.4 / 91.3 | 99.8 / 86.5 | 100.6 / 94.6 | 100.4 / 91.4 | 100.3 / 91.6 | 100.5 / 91.5 |
| C_nm[ieee]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 102.7 / 89.0 | 101.7 / 88.5 | 101.7 / 88.6 | 101.2 / 88.6 | 102.4 / 88.5 | 101.5 / 93.4 | 101.6 / 88.5 | 101.7 / 88.5 |
| C_nm[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.9 / 10.0 | 16.7 / 9.7 | 16.8 / 9.7 | 16.3 / 9.8 | 16.9 / 10.0 | 16.9 / 9.7 | 16.8 / 9.7 | 16.8 / 9.8 |
| C_nm[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 31.2 / 18.5 | 32.0 / 18.5 | 31.2 / 18.5 | 31.2 / 18.5 | 31.1 / 18.5 | 31.3 / 18.5 | 31.1 / 18.5 | 31.1 / 18.5 |
| C_nm[tf32]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 31.6 / 18.5 | 32.2 / 18.5 | 31.9 / 18.6 | 32.1 / 18.5 | 31.2 / 21.4 | 32.1 / 18.5 | 32.1 / 18.5 | 32.0 / 18.5 |
| native_sg_chain_depth5_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | 111.7 / 95.0 | 108.1 / 92.0 | 107.9 / 92.7 | 107.8 / 92.2 | 108.0 / 92.1 | 108.4 / 89.9 | 108.3 / 92.9 | 108.6 / 92.6 |
| native_pk_one_token(context) | 1 token; helper allocates mixed/a/b/out per call (NOT matched work) | 24.1 / 15.3 | 24.1 / 15.3 | 24.1 / 15.2 | 24.1 / 15.3 | 24.2 / 15.5 | 24.2 / 15.6 | 24.1 / 15.6 | 24.1 / 15.3 |

In-band probe-drift criterion (4096² fp16 matmul before→after, pass iff drift ≤ 5 %): id 1: 1746.73920→1733.77438 µs drift=0.007 **PASS** [2026-09-22T01:35:09Z–2026-09-22T01:35:12Z]; id 2: 1727.91042→1748.05279 µs drift=0.012 **PASS** [2026-09-22T01:35:13Z–2026-09-22T01:35:15Z]; id 3: 1808.58555→1766.30726 µs drift=0.023 **PASS** [2026-09-22T01:35:16Z–2026-09-22T01:35:18Z]; id 4: 1743.11676→1822.80483 µs drift=0.046 **PASS** [2026-09-22T01:35:19Z–2026-09-22T01:35:21Z]; id 5: 1793.37921→1736.87038 µs drift=0.032 **PASS** [2026-09-22T01:35:22Z–2026-09-22T01:35:24Z]; id 6: 1776.64471→1782.73125 µs drift=0.003 **PASS** [2026-09-22T01:35:25Z–2026-09-22T01:35:27Z]; id 7: 1745.12157→1737.60471 µs drift=0.004 **PASS** [2026-09-22T01:35:28Z–2026-09-22T01:35:29Z]; id 8: 1817.90562→1786.32164 µs drift=0.017 **PASS** [2026-09-22T01:35:31Z–2026-09-22T01:35:32Z]

iters=200, warmup=20. Not a serving throughput measurement.

## Validity

INVALID (non-finite) comparison cells in this summary: **0**. Probe-drift failures: **0**. Manifest status: **completed**; start/end source hashes match: **yes**.
