# E7a run summary — `out-20260921T222051Z-e7a-smoke`

Manifest status **completed**; start/end source hashes match: **yes**; image `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`; torch 2.11.0+cu130 / triton 3.6.0; GPU NVIDIA GB10 cc [12, 1]; driver `NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  59`.
Production kernel flags: {"FIXED32_MODE": null, "scan_align_on": false, "npad_invariant_on": false, "parent_gather_on": false, "hc_internal_on": false, "BV_prod": 16, "num_warps_prod": 8, "tf32_matmul_allowed": false, "replay_h0_source_column": 15, "scan_h0_source_column": 0}. Triton cubins hashed: 30.
Source sha256: `prod_kernel`=d9dd0c697b46…, `legacy_wy`=e7c2a2a8b658…, `e7a_core`=93c915d388d9…, `e7a_kernels`=15c353625218…, `e7a_device`=9b59b6c38fb5…, `native_fused_sigmoid_gating`=000ab8996af9…, `native_fused_recurrent`=3a2a3c5245ac…

**Timing attribution:** ATTRIBUTION UNVERIFIED (container host PID not recorded; PIDs seen: ['2746563']) — window ['2026-09-21T22:20:51.953000', '2026-09-21T22:22:49.012000'], util samples 118, zero-util fraction 0.966, samples >5 %: 2, compute PIDs seen: {'2746563': {'samples': 112, 'names': ['[No data]', 'python3']}}, container PIDs: None, coverage gaps: 0.

All mechanisms B/C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems. Historical payloads are dependent historical operand sets (one topology, B1), not independent prefixes; synthetic rows are labeled. Rows marked **INVALID** contain non-finite values in candidate or reference and carry no error value.

## Operand sets

| id | label | provenance | topology | n | depth | g_min | cum_g_min | P_min | min visible decay ratio | fp32 underflow risk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | layers.62.linear_attn | HISTORICAL sha256 9f1ddb6e3cfa… (language_model.model.layers.62.linear_attn) | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.319 | -16.390 | 7.62e-08 | 5.3e-07 | no |

## References vs fp64 oracle (same rounded operands)

| id | native_sg out32 | native_sg state | native_pk out32 vs oracle | native_pk state vs oracle | native_pk state vs ITS OWN oracle (beta bf16-rt + div/sqrt) | seam magnitude fp64 (oracle_pk vs oracle, state) | pk vs sg out16 exact | sg fp32-io vs bf16-io state max_abs |
|---|---|---|---|---|---|---|---|---|
| 1 | abs=7.38e-09 rel=1.34e-07 ulp=43 | abs=4.6e-07 rel=1.27e-07 ulp=22 | abs=5.24e-05 rel=0.000948 ulp=341187 | abs=0.000799 rel=0.000221 ulp=145608 | — | — | 0.821 | — |

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

Production-scan identity checks: id 1: out32 bit-equal native_sg=1.000, out16 bytes equal payload serving_out=yes

`A_legacy_state*` rows are excluded: the June-8 vendored sequential scan's export path skips the root update (source-inspected); when/which binary changed that path is NOT established. Its outputs are bit-identical to the production scan.

## Stage 2 — commit from IDENTICAL oracle factors (compact reconstruction kernel; vs oracle state / vs native_sg state)

| id | B_fs[ieee] vs oracle | vs native_sg | C_nm[ieee] vs oracle | B_fs[tf32] vs oracle | C_nm[tf32] vs oracle |
|---|---|---|---|---|---|
| 1 | abs=4.58e-07 rel=1.26e-07 ulp=12 | abs=7.15e-07 rel=1.97e-07 ulp=25 | abs=4.58e-07 rel=1.26e-07 ulp=12 | abs=0.001 rel=0.000296 ulp=96135 | abs=0.001 rel=0.000296 ulp=96135 |

## Stage 3 — own factors + own commit (all accepted prefixes incl. zero-accept; vs oracle / native_sg / native_pk)

| id | mechanism | vs oracle | vs native_sg (spec-update) | exact frac vs native_sg | vs native_pk (one-token decode) |
|---|---|---|---|---|---|
| 1 | A_prod replay | abs=4.6e-07 rel=1.27e-07 ulp=22 | abs=5.96e-08 rel=1.65e-08 ulp=4 | 0.991 | abs=0.000799 rel=0.000221 ulp=145600 |
| 1 | B_legacyWY fp32-closed (materialized) | abs=4.8e-07 rel=1.32e-07 ulp=25 | — | — | — |
| 1 | B_fs[ieee] compact | abs=4.58e-07 rel=1.26e-07 ulp=60 | abs=4.77e-07 rel=1.32e-07 ulp=50 | 0.319 | abs=0.000799 rel=0.000221 ulp=145626 |
| 1 | C_nm[ieee] compact | abs=4.58e-07 rel=1.26e-07 ulp=60 | abs=4.77e-07 rel=1.32e-07 ulp=50 | 0.316 | abs=0.000799 rel=0.000221 ulp=145629 |
| 1 | B_fs[tf32] compact | abs=0.001 rel=0.000296 ulp=242824 | abs=0.001 rel=0.000296 ulp=242821 | 0.000443 | abs=0.001 rel=0.000392 ulp=200358 |
| 1 | C_nm[tf32] compact | abs=0.001 rel=0.000296 ulp=241842 | abs=0.001 rel=0.000296 ulp=241839 | 0.000439 | abs=0.001 rel=0.000392 ulp=199376 |

Replay repeat-launch bitwise check (immutable h0 row): id 1: yes

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

## Controls

| id | Neumann d−1 terms (U err / out err) | B_fs N_PAD32 vs 16 (out exact / abs) | C_nm N_PAD32 vs 16 | B_fs sibling-reverse (out exact / U exact) | C_nm sibling-reverse | A_prod sibling-reverse out exact |
|---|---|---|---|---|---|---|
| 1 | 0.016 / 0.001 | 0.908 / 2.79e-09 | 0.930 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |

## Transient / scratch memory (bytes, analytic from shapes; per request, per layer)

| quantity | id 1 |
|---|---|
| A_prod_scan_hbm_per_node_state_export | 0 |
| A_prod_scan_register_h_cache_per_program | 131,072 |
| A_prod_replay_hbm_written_rows(depth+1) | 18,874,368 |
| A_prod_activation_ring(k,v,a,b) | 265,216 |
| A_legacy_state_export_all_nodes | 50,331,648 |
| B_legacyWY_state_export_all_nodes | 50,331,648 |
| BC_compact_factors_U+cumg | 396,288 |
| BC_compact_commit_hbm_written_rows | 3,145,728 |
| one_full_state_row | 3,145,728 |

## Timing — ATTRIBUTION UNVERIFIED (container host PID not recorded; PIDs seen: ['2746563'])

CUDA events, per-launch sync median / pipelined mean (µs), one layer, B1. Rows do DIFFERENT work (see WORK column); this is NOT an apples-to-apples kernel speedup table.

| kernel | WORK | id 1 |
|---|---|---|
| A_prod_scan_verify_out16 | bf16 out store; no per-node state export; preallocated | 90.7 / 64.0 |
| A_prod_replay_commit_deepest | writes depth+1 fp32 state rows to the bank; preallocated | 103.7 / 86.7 |
| A_prod_replay_commit_zero_accept | writes 1 fp32 state row (root); preallocated | 53.6 / 39.9 |
| A_legacy_scan_with_state_export | bf16 out + FULL per-node fp32 state export (n_pad x VH x DV x DK x 4 B); preallocated | 242.9 / 243.6 |
| B_legacyWY_fp32closed_with_state_export | bf16 out + FULL per-node fp32 state export; preallocated | 769.2 / 754.4 |
| B_legacyWY_bf16bnd_with_state_export | bf16 out (+bf16 boundary taps) + FULL per-node fp32 state export; preallocated | 1700.0 / 1688.4 |
| B_fs[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.8 / 10.0 |
| B_fs[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 110.9 / 96.7 |
| B_fs[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.9 / 10.1 |
| B_fs[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 42.0 / 28.9 |
| C_nm[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.8 / 10.4 |
| C_nm[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 96.2 / 83.0 |
| C_nm[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.7 / 10.1 |
| C_nm[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 31.5 / 18.7 |
| native_sg_chain_depth5_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | 157.6 / 166.6 |
| native_pk_one_token(context) | 1 token; helper allocates mixed/a/b/out per call (NOT matched work) | 120.9 / 115.3 |

In-band probe-drift criterion (4096² fp16 matmul before→after, pass iff drift ≤ 5 %): id 1: 1883.66871→1896.08326 µs drift=0.007 **PASS** [2026-09-21T22:22:46Z–2026-09-21T22:22:48Z]

iters=50, warmup=10. Not a serving throughput measurement.

## Validity

INVALID (non-finite) comparison cells in this summary: **0**. Probe-drift failures: **0**. Manifest status: **completed**; start/end source hashes match: **yes**.
