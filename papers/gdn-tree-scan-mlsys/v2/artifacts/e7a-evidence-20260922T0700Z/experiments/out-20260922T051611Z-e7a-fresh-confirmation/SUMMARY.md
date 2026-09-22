# E7a run summary — `out-20260922T051611Z-e7a-fresh-confirmation`

Manifest status **completed**; start/end source hashes match: **yes**; image `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`; torch 2.11.0+cu130 / triton 3.6.0; GPU NVIDIA GB10 cc [12, 1]; driver `NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  59`.
Production kernel flags: {"FIXED32_MODE": null, "scan_align_on": false, "npad_invariant_on": false, "parent_gather_on": false, "hc_internal_on": false, "BV_prod": 16, "num_warps_prod": 8, "tf32_matmul_allowed": false, "replay_h0_source_column": 15, "scan_h0_source_column": 0}. Triton cubins hashed: 30.
Source sha256: `prod_kernel`=d9dd0c697b46…, `legacy_wy`=e7c2a2a8b658…, `e7a_core`=89feb49d2bf1…, `e7a_kernels`=15c353625218…, `e7a_device`=c6a4e1e9df0e…, `native_fused_sigmoid_gating`=000ab8996af9…, `native_fused_recurrent`=3a2a3c5245ac…

**Timing attribution:** NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['3103483'] seen; sampled observation, not a guarantee) — window ['2026-09-22T05:16:12.104000', '2026-09-22T05:19:44.236000'], util samples 213, zero-util fraction 0.559, samples >5 %: 91, compute PIDs seen: {'3103483': {'samples': 199, 'names': ['python3']}}, container PIDs: ['3103483'], coverage gaps: 0.

All mechanisms B/C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems. Historical payloads are dependent historical operand sets (one topology, B1), not independent prefixes; synthetic rows are labeled. Rows marked **INVALID** contain non-finite values in candidate or reference and carry no error value.

## Operand sets

| id | label | provenance | topology | n | depth | g_min | cum_g_min | P_min | min visible decay ratio | fp32 underflow risk |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | layers.62.linear_attn | VERIFIED-FRESH sha256 1925f057acf4… (language_model.model.layers.62.linear_attn); prefix p063 hash 35ae47bf531b… tokens 256 req cmpl-9046d822fe161993; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.328 | -13.476 | 1.4e-06 | 1.43e-05 | no |
| 2 | layers.62.linear_attn | VERIFIED-FRESH sha256 6a8ec9dd815f… (language_model.model.layers.62.linear_attn); prefix p059 hash 7480d8c3955c… tokens 256 req cmpl-8fb8c51c9b07da28; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.133 | -13.260 | 1.74e-06 | 1.16e-05 | no |
| 3 | layers.62.linear_attn | VERIFIED-FRESH sha256 168224396e07… (language_model.model.layers.62.linear_attn); prefix p049 hash e39a3653f5a4… tokens 256 req cmpl-9af61982e8be6a7f; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.512 | -15.281 | 2.31e-07 | 3.65e-06 | no |
| 4 | layers.62.linear_attn | VERIFIED-FRESH sha256 966f41c418a7… (language_model.model.layers.62.linear_attn); prefix p052 hash ba06010c8676… tokens 256 req cmpl-98ba91f896aa9bd5; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.305 | -13.930 | 8.92e-07 | 6.76e-06 | no |
| 5 | layers.62.linear_attn | VERIFIED-FRESH sha256 a3b6363f2e67… (language_model.model.layers.62.linear_attn); prefix p046 hash 20b63352aea9… tokens 256 req cmpl-9f3872d3ffd1b917; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.285 | -15.430 | 1.99e-07 | 1.12e-06 | no |
| 6 | layers.62.linear_attn | VERIFIED-FRESH sha256 cf101bd0b81e… (language_model.model.layers.62.linear_attn); prefix p065 hash b77d631a346d… tokens 256 req cmpl-840d56271a481dc9; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.075 | -11.174 | 1.4e-05 | 8.58e-05 | no |
| 7 | layers.62.linear_attn | VERIFIED-FRESH sha256 7bd73c8e9b7a… (language_model.model.layers.62.linear_attn); prefix p025 hash fa17dcae865e… tokens 256 req cmpl-bcf2091e54776eeb; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.112 | -14.418 | 5.48e-07 | 6.93e-06 | no |
| 8 | layers.62.linear_attn | VERIFIED-FRESH sha256 4ed52fa2e916… (language_model.model.layers.62.linear_attn); prefix p031 hash 6e70bf724b24… tokens 256 req cmpl-9ecbd3f1680cb688; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.165 | -16.331 | 8.09e-08 | 1.92e-06 | no |
| 9 | layers.62.linear_attn | VERIFIED-FRESH sha256 319d8c8b349e… (language_model.model.layers.62.linear_attn); prefix p013 hash 79e545b81540… tokens 256 req cmpl-a862aacf23d3c9c9; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.522 | -15.399 | 2.05e-07 | 1.42e-06 | no |
| 10 | layers.62.linear_attn | VERIFIED-FRESH sha256 e795fe1d9cfd… (language_model.model.layers.62.linear_attn); prefix p040 hash 44a2bb7a5cd0… tokens 256 req cmpl-b91765d8ac7f0f44; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.668 | -9.765 | 5.75e-05 | 0.000182 | no |
| 11 | layers.62.linear_attn | VERIFIED-FRESH sha256 e84cafe571a0… (language_model.model.layers.62.linear_attn); prefix p033 hash 7ad0bf676e22… tokens 256 req cmpl-9a5717835b579178; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.217 | -10.680 | 2.3e-05 | 0.000477 | no |
| 12 | layers.62.linear_attn | VERIFIED-FRESH sha256 92ff89c588cf… (language_model.model.layers.62.linear_attn); prefix p038 hash cab8f09b3e0e… tokens 256 req cmpl-977e4f3ddcd783bb; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.736 | -12.481 | 3.8e-06 | 4.18e-05 | no |
| 13 | layers.62.linear_attn | VERIFIED-FRESH sha256 2d9eafd2c2fa… (language_model.model.layers.62.linear_attn); prefix p060 hash ef21c3cd7774… tokens 256 req cmpl-a1135269c7a79607; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.906 | -13.419 | 1.49e-06 | 5.83e-06 | no |
| 14 | layers.62.linear_attn | VERIFIED-FRESH sha256 841304e29647… (language_model.model.layers.62.linear_attn); prefix p067 hash f6fafd41491b… tokens 256 req cmpl-a49f5477a70384e4; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.420 | -13.099 | 2.05e-06 | 9e-06 | no |
| 15 | layers.62.linear_attn | VERIFIED-FRESH sha256 816500f268df… (language_model.model.layers.62.linear_attn); prefix p019 hash b37e0fce7c01… tokens 256 req cmpl-96f3b4faf610df9b; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.563 | -13.703 | 1.12e-06 | 5.23e-06 | no |
| 16 | layers.62.linear_attn | VERIFIED-FRESH sha256 0c467a0a6fc7… (language_model.model.layers.62.linear_attn); prefix p003 hash 6aa020a4c10a… tokens 256 req cmpl-9682bd6b5f80b4db; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.967 | -11.872 | 6.98e-06 | 6.56e-05 | no |
| 17 | layers.62.linear_attn | VERIFIED-FRESH sha256 25eb7fe18e52… (language_model.model.layers.62.linear_attn); prefix p077 hash 5235933d34f7… tokens 4096 req cmpl-92316eb02340fd6c; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.330 | -12.828 | 2.68e-06 | 2e-05 | no |
| 18 | layers.62.linear_attn | VERIFIED-FRESH sha256 3c6253d69b89… (language_model.model.layers.62.linear_attn); prefix p089 hash 42ddc634fe7e… tokens 4096 req cmpl-8866e3df066cfe8d; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.812 | -10.948 | 1.76e-05 | 8.3e-05 | no |
| 19 | layers.62.linear_attn | VERIFIED-FRESH sha256 c7d59f7ed8b2… (language_model.model.layers.62.linear_attn); prefix p030 hash f70235c652d5… tokens 4096 req cmpl-83c4738aa83f11fd; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.357 | -11.862 | 7.05e-06 | 4.09e-05 | no |
| 20 | layers.62.linear_attn | VERIFIED-FRESH sha256 30c73d9a4b15… (language_model.model.layers.62.linear_attn); prefix p081 hash c9151fcfcb9f… tokens 4096 req cmpl-a5ca149c7be2ede5; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.845 | -13.047 | 2.16e-06 | 3.43e-05 | no |
| 21 | layers.62.linear_attn | VERIFIED-FRESH sha256 a3d9501ab14e… (language_model.model.layers.62.linear_attn); prefix p012 hash f9b0d9cfcf12… tokens 4096 req cmpl-8a0c081b49920e89; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.913 | -14.572 | 4.69e-07 | 3.44e-06 | no |
| 22 | layers.62.linear_attn | VERIFIED-FRESH sha256 fc77d030cbc4… (language_model.model.layers.62.linear_attn); prefix p092 hash 163447d4a2d3… tokens 4096 req cmpl-bf80c4dc2f5434c1; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.486 | -9.567 | 7e-05 | 0.000323 | no |
| 23 | layers.62.linear_attn | VERIFIED-FRESH sha256 14b2cb821056… (language_model.model.layers.62.linear_attn); prefix p086 hash 711673a143e6… tokens 4096 req cmpl-b2a5111a709c4960; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.402 | -12.918 | 2.45e-06 | 2.71e-05 | no |
| 24 | layers.62.linear_attn | VERIFIED-FRESH sha256 ff93c2df9240… (language_model.model.layers.62.linear_attn); prefix p079 hash 38e383e20639… tokens 4096 req cmpl-82e2dea1b4aff81f; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.865 | -13.616 | 1.22e-06 | 1.15e-05 | no |
| 25 | layers.62.linear_attn | VERIFIED-FRESH sha256 2bcc0deb3dcd… (language_model.model.layers.62.linear_attn); prefix p027 hash a14079336008… tokens 4096 req cmpl-847e1872b50ea7f0; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.871 | -11.399 | 1.12e-05 | 7.75e-05 | no |
| 26 | layers.62.linear_attn | VERIFIED-FRESH sha256 99024065cbaf… (language_model.model.layers.62.linear_attn); prefix p091 hash f7657fc6c0b2… tokens 4096 req cmpl-93a3c1d3fff641e4; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.278 | -11.821 | 7.35e-06 | 2.13e-05 | no |
| 27 | layers.62.linear_attn | VERIFIED-FRESH sha256 e27ab1ca4acb… (language_model.model.layers.62.linear_attn); prefix p082 hash ee20f7189dcd… tokens 4096 req cmpl-ac2598738728c4cb; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.933 | -8.705 | 0.000166 | 0.00074 | no |
| 28 | layers.62.linear_attn | VERIFIED-FRESH sha256 fceef3e99595… (language_model.model.layers.62.linear_attn); prefix p088 hash 431454e7dbd0… tokens 4096 req cmpl-9b3e4120f9284106; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.891 | -11.652 | 8.71e-06 | 3.85e-05 | no |
| 29 | layers.62.linear_attn | VERIFIED-FRESH sha256 d5bf7a6152fc… (language_model.model.layers.62.linear_attn); prefix p005 hash 6593b0da0895… tokens 4096 req cmpl-8721124631381010; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.387 | -15.119 | 2.72e-07 | 1.48e-06 | no |
| 30 | layers.62.linear_attn | VERIFIED-FRESH sha256 89fe01500590… (language_model.model.layers.62.linear_attn); prefix p076 hash 6d7751fbd437… tokens 4096 req cmpl-a1bfee6b23a5a690; inspector 59c5b2b92d0d | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -2.713 | -6.913 | 0.000994 | 0.002 | no |
| 31 | layers.62.linear_attn | VERIFIED-FRESH sha256 199f79257df9… (language_model.model.layers.62.linear_attn); prefix p078 hash ce18f6e5d28a… tokens 4096 req cmpl-8d224846b702a706; inspector 6e71a24c5b7e | `[-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]` | 10 | 5 | -3.265 | -11.187 | 1.39e-05 | 8.28e-05 | no |

## References vs fp64 oracle (same rounded operands)

| id | native_sg out32 | native_sg state | native_pk out32 vs oracle | native_pk state vs oracle | native_pk state vs ITS OWN oracle (beta bf16-rt + div/sqrt) | seam magnitude fp64 (oracle_pk vs oracle, state) | pk vs sg out16 exact | sg fp32-io vs bf16-io state max_abs |
|---|---|---|---|---|---|---|---|---|
| 1 | abs=1.6e-08 rel=1.4e-07 ulp=22 | abs=5.66e-07 rel=1.93e-07 ulp=19 | abs=5.26e-05 rel=0.000461 ulp=184707 | abs=0.000842 rel=0.000286 ulp=265987 | abs=7.69e-07 rel=2.62e-07 ulp=21 | abs=0.000842 rel=0.000286 | 0.837 | 5.96e-08 |
| 2 | abs=1.25e-08 rel=1.55e-07 ulp=29 | abs=7.13e-07 rel=2.45e-07 ulp=36 | abs=5.38e-05 rel=0.000666 ulp=308172 | abs=0.000572 rel=0.000196 ulp=161622 | abs=5.47e-07 rel=1.88e-07 ulp=27 | abs=0.000572 rel=0.000196 | 0.817 | 2.98e-08 |
| 3 | abs=1.25e-08 rel=1.63e-07 ulp=31 | abs=4.71e-07 rel=1.69e-07 ulp=28 | abs=4.3e-05 rel=0.000558 ulp=266317 | abs=0.002 rel=0.000608 ulp=362964 | abs=4.39e-07 rel=1.58e-07 ulp=23 | abs=0.002 rel=0.000608 | 0.840 | 5.96e-08 |
| 4 | abs=1.36e-08 rel=1.7e-07 ulp=24 | abs=9.93e-07 rel=2.96e-07 ulp=18 | abs=4.08e-05 rel=0.00051 ulp=182034 | abs=0.00079 rel=0.000235 ulp=154543 | abs=5.52e-07 rel=1.64e-07 ulp=20 | abs=0.00079 rel=0.000235 | 0.847 | 2.98e-08 |
| 5 | abs=1.3e-08 rel=9.66e-08 ulp=18 | abs=5.07e-07 rel=1.41e-07 ulp=14 | abs=4.61e-05 rel=0.000342 ulp=341794 | abs=0.001 rel=0.000291 ulp=366736 | abs=5.59e-07 rel=1.56e-07 ulp=29 | abs=0.001 rel=0.000291 | 0.852 | 2.98e-08 |
| 6 | abs=1.52e-08 rel=2.13e-07 ulp=123 | abs=6.47e-07 rel=2e-07 ulp=29 | abs=0.000103 rel=0.001 ulp=602062 | abs=0.003 rel=0.000864 ulp=291761 | abs=6.17e-07 rel=1.91e-07 ulp=30 | abs=0.003 rel=0.000864 | 0.815 | 5.96e-08 |
| 7 | abs=1.65e-08 rel=1.85e-07 ulp=53 | abs=6.83e-07 rel=2.32e-07 ulp=28 | abs=5.37e-05 rel=0.000603 ulp=422727 | abs=0.001 rel=0.000347 ulp=237687 | abs=8.3e-07 rel=2.82e-07 ulp=21 | abs=0.001 rel=0.000347 | 0.830 | 5.96e-08 |
| 8 | abs=1.44e-08 rel=1.12e-07 ulp=19 | abs=4.49e-07 rel=1.57e-07 ulp=33 | abs=5.97e-05 rel=0.000462 ulp=229027 | abs=0.001 rel=0.000378 ulp=312872 | abs=7.92e-07 rel=2.77e-07 ulp=28 | abs=0.001 rel=0.000378 | 0.830 | 2.98e-08 |
| 9 | abs=1.77e-08 rel=1.2e-07 ulp=32 | abs=5.18e-07 rel=1.75e-07 ulp=29 | abs=0.000117 rel=0.000791 ulp=202953 | abs=0.001 rel=0.0005 ulp=386777 | abs=4.95e-07 rel=1.68e-07 ulp=19 | abs=0.001 rel=0.0005 | 0.833 | 2.98e-08 |
| 10 | abs=1.52e-08 rel=1.24e-07 ulp=22 | abs=3.97e-07 rel=1.27e-07 ulp=17 | abs=3.23e-05 rel=0.000262 ulp=163596 | abs=0.0007 rel=0.000224 ulp=295219 | abs=3.81e-07 rel=1.22e-07 ulp=24 | abs=0.0007 rel=0.000224 | 0.835 | 2.98e-08 |
| 11 | abs=2.12e-08 rel=2.27e-07 ulp=54 | abs=5.21e-07 rel=2.01e-07 ulp=27 | abs=3.94e-05 rel=0.000422 ulp=337986 | abs=0.000677 rel=0.000261 ulp=283208 | abs=6.85e-07 rel=2.64e-07 ulp=18 | abs=0.000677 rel=0.000261 | 0.825 | 5.96e-08 |
| 12 | abs=1.25e-08 rel=1.61e-07 ulp=22 | abs=5.11e-07 rel=1.57e-07 ulp=24 | abs=0.000108 rel=0.001 ulp=880773 | abs=0.003 rel=0.00089 ulp=520132 | abs=5.42e-07 rel=1.67e-07 ulp=31 | abs=0.003 rel=0.00089 | 0.812 | 2.38e-07 |
| 13 | abs=9.04e-09 rel=9.44e-08 ulp=25 | abs=7.43e-07 rel=2.28e-07 ulp=13 | abs=4.12e-05 rel=0.00043 ulp=217586 | abs=0.000559 rel=0.000171 ulp=108088 | abs=5.5e-07 rel=1.68e-07 ulp=11 | abs=0.000559 rel=0.000171 | 0.853 | 5.96e-08 |
| 14 | abs=1.55e-08 rel=1.8e-07 ulp=71 | abs=5.49e-07 rel=2.04e-07 ulp=38 | abs=0.000123 rel=0.001 ulp=377654 | abs=0.001 rel=0.000445 ulp=290114 | abs=5.41e-07 rel=2.01e-07 ulp=33 | abs=0.001 rel=0.000445 | 0.827 | 5.96e-08 |
| 15 | abs=1.3e-08 rel=1.2e-07 ulp=39 | abs=6.03e-07 rel=1.78e-07 ulp=16 | abs=6.86e-05 rel=0.000631 ulp=221855 | abs=0.000537 rel=0.000158 ulp=207141 | abs=5.3e-07 rel=1.56e-07 ulp=17 | abs=0.000537 rel=0.000158 | 0.843 | 5.96e-08 |
| 16 | abs=1.56e-08 rel=1.25e-07 ulp=28 | abs=6.15e-07 rel=2e-07 ulp=16 | abs=6.68e-05 rel=0.000536 ulp=259651 | abs=0.000956 rel=0.00031 ulp=336719 | abs=3.93e-07 rel=1.27e-07 ulp=27 | abs=0.000956 rel=0.00031 | 0.834 | 2.98e-08 |
| 17 | abs=5.26e-08 rel=2.36e-07 ulp=17 | abs=4.85e-07 rel=1.51e-07 ulp=25 | abs=3.64e-05 rel=0.000163 ulp=177239 | abs=0.000506 rel=0.000158 ulp=238245 | abs=6.97e-07 rel=2.17e-07 ulp=23 | abs=0.000506 rel=0.000158 | 0.905 | 2.98e-08 |
| 18 | abs=1.68e-08 rel=1.5e-07 ulp=34 | abs=6.8e-07 rel=1.7e-07 ulp=9 | abs=9.55e-05 rel=0.000857 ulp=257422 | abs=0.000856 rel=0.000214 ulp=94857 | abs=7.8e-07 rel=1.95e-07 ulp=19 | abs=0.000856 rel=0.000214 | 0.847 | 5.96e-08 |
| 19 | abs=2.07e-08 rel=1.3e-07 ulp=64 | abs=6.25e-07 rel=1.61e-07 ulp=12 | abs=5.6e-05 rel=0.000351 ulp=207860 | abs=0.000893 rel=0.00023 ulp=163253 | abs=7.15e-07 rel=1.84e-07 ulp=11 | abs=0.000893 rel=0.00023 | 0.846 | 2.98e-08 |
| 20 | abs=2.11e-08 rel=1.35e-07 ulp=28 | abs=5.99e-07 rel=1.53e-07 ulp=16 | abs=0.000215 rel=0.001 ulp=280292 | abs=0.002 rel=0.000586 ulp=319319 | abs=6.49e-07 rel=1.66e-07 ulp=28 | abs=0.002 rel=0.000586 | 0.850 | 5.96e-08 |
| 21 | abs=1.03e-08 rel=1.26e-07 ulp=41 | abs=6.14e-07 rel=1.75e-07 ulp=78 | abs=6.56e-05 rel=0.000806 ulp=262131 | abs=0.000706 rel=0.000201 ulp=175133 | abs=7.45e-07 rel=2.12e-07 ulp=20 | abs=0.000706 rel=0.000201 | 0.840 | 2.98e-08 |
| 22 | abs=2.35e-08 rel=1.34e-07 ulp=41 | abs=5.27e-07 rel=1.31e-07 ulp=10 | abs=4.05e-05 rel=0.00023 ulp=127822 | abs=0.000625 rel=0.000156 ulp=108400 | abs=5.92e-07 rel=1.47e-07 ulp=16 | abs=0.000625 rel=0.000156 | 0.859 | 5.96e-08 |
| 23 | abs=9.18e-08 rel=3.8e-07 ulp=20 | abs=8.82e-07 rel=2.84e-07 ulp=13 | abs=4.09e-05 rel=0.000169 ulp=77552 | abs=0.000472 rel=0.000152 ulp=85145 | abs=9.85e-07 rel=3.18e-07 ulp=15 | abs=0.000472 rel=0.000152 | 0.935 | 1.49e-08 |
| 24 | abs=2.23e-08 rel=1.24e-07 ulp=44 | abs=9.13e-07 rel=2.2e-07 ulp=13 | abs=4.84e-05 rel=0.000269 ulp=272213 | abs=0.000951 rel=0.000229 ulp=222317 | abs=7.41e-07 rel=1.79e-07 ulp=14 | abs=0.000951 rel=0.000229 | 0.853 | 5.96e-08 |
| 25 | abs=2.01e-08 rel=1.47e-07 ulp=35 | abs=1.08e-06 rel=2.17e-07 ulp=23 | abs=8.86e-05 rel=0.000647 ulp=261665 | abs=0.001 rel=0.000297 ulp=153790 | abs=9.02e-07 rel=1.81e-07 ulp=13 | abs=0.001 rel=0.000297 | 0.837 | 5.96e-08 |
| 26 | abs=1.48e-08 rel=1.37e-07 ulp=123 | abs=8.36e-07 rel=2.02e-07 ulp=11 | abs=7.42e-05 rel=0.000685 ulp=207855 | abs=0.000937 rel=0.000226 ulp=134236 | abs=8.07e-07 rel=1.95e-07 ulp=14 | abs=0.000937 rel=0.000226 | 0.829 | 5.96e-08 |
| 27 | abs=2.26e-08 rel=1.12e-07 ulp=61 | abs=6.11e-07 rel=1.44e-07 ulp=14 | abs=0.00019 rel=0.000941 ulp=525828 | abs=0.002 rel=0.000478 ulp=206092 | abs=1.02e-06 rel=2.4e-07 ulp=20 | abs=0.002 rel=0.000478 | 0.843 | 5.96e-08 |
| 28 | abs=1.33e-08 rel=1.21e-07 ulp=54 | abs=1.3e-06 rel=3.1e-07 ulp=19 | abs=5.22e-05 rel=0.000472 ulp=172012 | abs=0.000989 rel=0.000235 ulp=171361 | abs=1.22e-06 rel=2.9e-07 ulp=18 | abs=0.000989 rel=0.000235 | 0.845 | 5.96e-08 |
| 29 | abs=1.65e-08 rel=9.39e-08 ulp=39 | abs=1.34e-06 rel=2.67e-07 ulp=19 | abs=0.000104 rel=0.000591 ulp=195230 | abs=0.001 rel=0.000206 ulp=261765 | abs=6.85e-07 rel=1.37e-07 ulp=19 | abs=0.001 rel=0.000206 | 0.848 | 1.19e-07 |
| 30 | abs=1.85e-08 rel=1.73e-07 ulp=82 | abs=7.78e-07 rel=1.9e-07 ulp=7 | abs=3.48e-05 rel=0.000326 ulp=253341 | abs=0.000628 rel=0.000153 ulp=159784 | abs=6.97e-07 rel=1.7e-07 ulp=14 | abs=0.000628 rel=0.000153 | 0.841 | 5.96e-08 |
| 31 | abs=1.38e-08 rel=1.01e-07 ulp=34 | abs=1.03e-06 rel=2.48e-07 ulp=11 | abs=5.4e-05 rel=0.000394 ulp=100058 | abs=0.000987 rel=0.000236 ulp=133063 | abs=1.08e-06 rel=2.58e-07 ulp=14 | abs=0.000987 rel=0.000236 | 0.844 | 2.98e-08 |

## Stage 1 — verifier outputs and correction factors (vs fp64 oracle; out16 exact fraction vs native spec-update kernel)

| id | mechanism | out32 vs oracle | out16 exact vs native_sg | factors U vs oracle u |
|---|---|---|---|---|
| 1 | A_prod | abs=1.6e-08 rel=1.4e-07 ulp=22 | 1.000 | — |
| 1 | B_legacyWY fp32-closed | abs=1.6e-08 rel=1.4e-07 ulp=24 | — | — |
| 1 | B_legacyWY bf16-bnd | abs=0.000198 rel=0.002 ulp=495681 | — | — |
| 1 | B_fs[ieee] | abs=2.82e-08 rel=2.47e-07 ulp=49 | 1.000 | abs=3.77e-07 rel=3.3e-07 ulp=220 |
| 1 | C_nm[ieee] | abs=2.82e-08 rel=2.47e-07 ulp=45 | 1.000 | abs=3.77e-07 rel=3.3e-07 ulp=218 |
| 1 | B_fs[tf32] | abs=6.48e-05 rel=0.000568 ulp=224145 | 0.791 | abs=0.000727 rel=0.000635 ulp=529712 |
| 1 | C_nm[tf32] | abs=6.41e-05 rel=0.000562 ulp=273794 | 0.790 | abs=0.000742 rel=0.000648 ulp=529753 |
| 2 | A_prod | abs=1.25e-08 rel=1.55e-07 ulp=29 | 1.000 | — |
| 2 | B_legacyWY fp32-closed | abs=1.2e-08 rel=1.48e-07 ulp=39 | — | — |
| 2 | B_legacyWY bf16-bnd | abs=0.000121 rel=0.001 ulp=710082 | — | — |
| 2 | B_fs[ieee] | abs=2.8e-08 rel=3.47e-07 ulp=63 | 1.000 | abs=2.91e-07 rel=2.8e-07 ulp=274 |
| 2 | C_nm[ieee] | abs=2.8e-08 rel=3.47e-07 ulp=77 | 1.000 | abs=2.99e-07 rel=2.87e-07 ulp=274 |
| 2 | B_fs[tf32] | abs=5.75e-05 rel=0.000712 ulp=212954 | 0.776 | abs=0.000465 rel=0.000446 ulp=355192 |
| 2 | C_nm[tf32] | abs=5.75e-05 rel=0.000712 ulp=226530 | 0.779 | abs=0.000666 rel=0.00064 ulp=412131 |
| 3 | A_prod | abs=1.25e-08 rel=1.63e-07 ulp=31 | 1.000 | — |
| 3 | B_legacyWY fp32-closed | abs=1.03e-08 rel=1.34e-07 ulp=30 | — | — |
| 3 | B_legacyWY bf16-bnd | abs=0.000179 rel=0.002 ulp=797564 | — | — |
| 3 | B_fs[ieee] | abs=4.51e-08 rel=5.86e-07 ulp=69 | 1.000 | abs=3.1e-07 rel=2.59e-07 ulp=262 |
| 3 | C_nm[ieee] | abs=4.51e-08 rel=5.86e-07 ulp=76 | 1.000 | abs=2.87e-07 rel=2.4e-07 ulp=262 |
| 3 | B_fs[tf32] | abs=6.96e-05 rel=0.000903 ulp=337177 | 0.798 | abs=0.000891 rel=0.000745 ulp=542624 |
| 3 | C_nm[tf32] | abs=6.96e-05 rel=0.000903 ulp=337177 | 0.802 | abs=0.000905 rel=0.000757 ulp=675596 |
| 4 | A_prod | abs=1.36e-08 rel=1.7e-07 ulp=24 | 1.000 | — |
| 4 | B_legacyWY fp32-closed | abs=1.22e-08 rel=1.52e-07 ulp=40 | — | — |
| 4 | B_legacyWY bf16-bnd | abs=0.000217 rel=0.003 ulp=664981 | — | — |
| 4 | B_fs[ieee] | abs=3.03e-08 rel=3.78e-07 ulp=65 | 1.000 | abs=2.92e-07 rel=2.49e-07 ulp=253 |
| 4 | C_nm[ieee] | abs=3.03e-08 rel=3.78e-07 ulp=77 | 1.000 | abs=2.92e-07 rel=2.49e-07 ulp=250 |
| 4 | B_fs[tf32] | abs=7.62e-05 rel=0.000952 ulp=184058 | 0.793 | abs=0.000594 rel=0.000507 ulp=492488 |
| 4 | C_nm[tf32] | abs=7.62e-05 rel=0.000952 ulp=256314 | 0.794 | abs=0.000609 rel=0.00052 ulp=489864 |
| 5 | A_prod | abs=1.3e-08 rel=9.66e-08 ulp=18 | 1.000 | — |
| 5 | B_legacyWY fp32-closed | abs=1.8e-08 rel=1.33e-07 ulp=18 | — | — |
| 5 | B_legacyWY bf16-bnd | abs=0.000174 rel=0.001 ulp=1303830 | — | — |
| 5 | B_fs[ieee] | abs=3.87e-08 rel=2.87e-07 ulp=69 | 1.000 | abs=2.53e-07 rel=2.35e-07 ulp=322 |
| 5 | C_nm[ieee] | abs=3.87e-08 rel=2.87e-07 ulp=69 | 1.000 | abs=2.68e-07 rel=2.49e-07 ulp=326 |
| 5 | B_fs[tf32] | abs=7.3e-05 rel=0.000541 ulp=252510 | 0.792 | abs=0.000351 rel=0.000326 ulp=670214 |
| 5 | C_nm[tf32] | abs=7.25e-05 rel=0.000538 ulp=252510 | 0.795 | abs=0.000523 rel=0.000486 ulp=670214 |
| 6 | A_prod | abs=1.52e-08 rel=2.13e-07 ulp=123 | 1.000 | — |
| 6 | B_legacyWY fp32-closed | abs=1.98e-08 rel=2.77e-07 ulp=155 | — | — |
| 6 | B_legacyWY bf16-bnd | abs=0.000466 rel=0.007 ulp=783909 | — | — |
| 6 | B_fs[ieee] | abs=2.18e-08 rel=3.06e-07 ulp=71 | 1.000 | abs=2.47e-07 rel=1.18e-07 ulp=163 |
| 6 | C_nm[ieee] | abs=2.18e-08 rel=3.06e-07 ulp=118 | 1.000 | abs=2.65e-07 rel=1.27e-07 ulp=165 |
| 6 | B_fs[tf32] | abs=0.000146 rel=0.002 ulp=439178 | 0.784 | abs=0.000824 rel=0.000394 ulp=477711 |
| 6 | C_nm[tf32] | abs=0.000144 rel=0.002 ulp=439178 | 0.790 | abs=0.000824 rel=0.000394 ulp=477711 |
| 7 | A_prod | abs=1.65e-08 rel=1.85e-07 ulp=53 | 1.000 | — |
| 7 | B_legacyWY fp32-closed | abs=1.88e-08 rel=2.12e-07 ulp=29 | — | — |
| 7 | B_legacyWY bf16-bnd | abs=0.000183 rel=0.002 ulp=753342 | — | — |
| 7 | B_fs[ieee] | abs=2.39e-08 rel=2.68e-07 ulp=125 | 1.000 | abs=2.86e-07 rel=2.29e-07 ulp=309 |
| 7 | C_nm[ieee] | abs=2.39e-08 rel=2.68e-07 ulp=97 | 1.000 | abs=2.77e-07 rel=2.22e-07 ulp=309 |
| 7 | B_fs[tf32] | abs=5.35e-05 rel=0.000601 ulp=360757 | 0.795 | abs=0.000605 rel=0.000484 ulp=623440 |
| 7 | C_nm[tf32] | abs=5.37e-05 rel=0.000603 ulp=360757 | 0.799 | abs=0.000605 rel=0.000484 ulp=625377 |
| 8 | A_prod | abs=1.44e-08 rel=1.12e-07 ulp=19 | 1.000 | — |
| 8 | B_legacyWY fp32-closed | abs=1.76e-08 rel=1.37e-07 ulp=52 | — | — |
| 8 | B_legacyWY bf16-bnd | abs=0.0002 rel=0.002 ulp=649612 | — | — |
| 8 | B_fs[ieee] | abs=3.02e-08 rel=2.33e-07 ulp=43 | 1.000 | abs=3.69e-07 rel=2.83e-07 ulp=396 |
| 8 | C_nm[ieee] | abs=3.01e-08 rel=2.33e-07 ulp=51 | 1.000 | abs=3.73e-07 rel=2.86e-07 ulp=401 |
| 8 | B_fs[tf32] | abs=8.38e-05 rel=0.000649 ulp=177038 | 0.786 | abs=0.000631 rel=0.000484 ulp=984941 |
| 8 | C_nm[tf32] | abs=8.31e-05 rel=0.000643 ulp=174098 | 0.791 | abs=0.000691 rel=0.00053 ulp=1044863 |
| 9 | A_prod | abs=1.77e-08 rel=1.2e-07 ulp=32 | 1.000 | — |
| 9 | B_legacyWY fp32-closed | abs=2.5e-08 rel=1.68e-07 ulp=33 | — | — |
| 9 | B_legacyWY bf16-bnd | abs=0.00026 rel=0.002 ulp=585840 | — | — |
| 9 | B_fs[ieee] | abs=3.42e-08 rel=2.3e-07 ulp=65 | 1.000 | abs=2.39e-07 rel=2.05e-07 ulp=182 |
| 9 | C_nm[ieee] | abs=3.42e-08 rel=2.3e-07 ulp=65 | 1.000 | abs=2.54e-07 rel=2.18e-07 ulp=183 |
| 9 | B_fs[tf32] | abs=0.0001 rel=0.000676 ulp=168801 | 0.790 | abs=0.000506 rel=0.000435 ulp=520330 |
| 9 | C_nm[tf32] | abs=0.0001 rel=0.000676 ulp=134280 | 0.790 | abs=0.000506 rel=0.000435 ulp=817732 |
| 10 | A_prod | abs=1.52e-08 rel=1.24e-07 ulp=22 | 1.000 | — |
| 10 | B_legacyWY fp32-closed | abs=1.76e-08 rel=1.43e-07 ulp=22 | — | — |
| 10 | B_legacyWY bf16-bnd | abs=0.000126 rel=0.001 ulp=626933 | — | — |
| 10 | B_fs[ieee] | abs=4e-08 rel=3.24e-07 ulp=57 | 1.000 | abs=2.67e-07 rel=4.01e-07 ulp=210 |
| 10 | C_nm[ieee] | abs=4e-08 rel=3.24e-07 ulp=53 | 1.000 | abs=2.67e-07 rel=4.01e-07 ulp=210 |
| 10 | B_fs[tf32] | abs=8.17e-05 rel=0.000662 ulp=147111 | 0.785 | abs=0.000399 rel=0.000598 ulp=407649 |
| 10 | C_nm[tf32] | abs=7.81e-05 rel=0.000633 ulp=183497 | 0.787 | abs=0.000399 rel=0.000598 ulp=515219 |
| 11 | A_prod | abs=2.12e-08 rel=2.27e-07 ulp=54 | 1.000 | — |
| 11 | B_legacyWY fp32-closed | abs=1.19e-08 rel=1.27e-07 ulp=49 | — | — |
| 11 | B_legacyWY bf16-bnd | abs=0.000191 rel=0.002 ulp=549602 | — | — |
| 11 | B_fs[ieee] | abs=3.34e-08 rel=3.58e-07 ulp=59 | 1.000 | abs=1.25e-07 rel=2.46e-07 ulp=268 |
| 11 | C_nm[ieee] | abs=3.34e-08 rel=3.58e-07 ulp=59 | 1.000 | abs=1.26e-07 rel=2.49e-07 ulp=271 |
| 11 | B_fs[tf32] | abs=4.58e-05 rel=0.00049 ulp=218547 | 0.784 | abs=0.00038 rel=0.000751 ulp=575017 |
| 11 | C_nm[tf32] | abs=5.21e-05 rel=0.000558 ulp=197039 | 0.787 | abs=0.00038 rel=0.000751 ulp=575017 |
| 12 | A_prod | abs=1.25e-08 rel=1.61e-07 ulp=22 | 1.000 | — |
| 12 | B_legacyWY fp32-closed | abs=1.3e-08 rel=1.68e-07 ulp=77 | — | — |
| 12 | B_legacyWY bf16-bnd | abs=0.000171 rel=0.002 ulp=697657 | — | — |
| 12 | B_fs[ieee] | abs=1.97e-08 rel=2.55e-07 ulp=133 | 1.000 | abs=3.64e-07 rel=1.79e-07 ulp=391 |
| 12 | C_nm[ieee] | abs=2.12e-08 rel=2.74e-07 ulp=132 | 1.000 | abs=3.59e-07 rel=1.77e-07 ulp=386 |
| 12 | B_fs[tf32] | abs=6.88e-05 rel=0.000889 ulp=283199 | 0.785 | abs=0.001 rel=0.000579 ulp=376345 |
| 12 | C_nm[tf32] | abs=6.41e-05 rel=0.000829 ulp=283199 | 0.788 | abs=0.002 rel=0.000758 ulp=405379 |
| 13 | A_prod | abs=9.04e-09 rel=9.44e-08 ulp=25 | 1.000 | — |
| 13 | B_legacyWY fp32-closed | abs=1.04e-08 rel=1.08e-07 ulp=43 | — | — |
| 13 | B_legacyWY bf16-bnd | abs=0.000133 rel=0.001 ulp=716558 | — | — |
| 13 | B_fs[ieee] | abs=2.62e-08 rel=2.74e-07 ulp=86 | 1.000 | abs=1.66e-07 rel=2.28e-07 ulp=263 |
| 13 | C_nm[ieee] | abs=2.62e-08 rel=2.74e-07 ulp=100 | 1.000 | abs=1.64e-07 rel=2.26e-07 ulp=263 |
| 13 | B_fs[tf32] | abs=5.53e-05 rel=0.000577 ulp=183289 | 0.798 | abs=0.000255 rel=0.000349 ulp=430403 |
| 13 | C_nm[tf32] | abs=5.51e-05 rel=0.000576 ulp=306869 | 0.800 | abs=0.000363 rel=0.000499 ulp=430403 |
| 14 | A_prod | abs=1.55e-08 rel=1.8e-07 ulp=71 | 1.000 | — |
| 14 | B_legacyWY fp32-closed | abs=9.84e-09 rel=1.14e-07 ulp=39 | — | — |
| 14 | B_legacyWY bf16-bnd | abs=0.000135 rel=0.002 ulp=949049 | — | — |
| 14 | B_fs[ieee] | abs=3.22e-08 rel=3.74e-07 ulp=140 | 1.000 | abs=2.11e-07 rel=1.06e-07 ulp=227 |
| 14 | C_nm[ieee] | abs=3.22e-08 rel=3.74e-07 ulp=139 | 1.000 | abs=2.1e-07 rel=1.05e-07 ulp=226 |
| 14 | B_fs[tf32] | abs=0.000138 rel=0.002 ulp=312438 | 0.788 | abs=0.000593 rel=0.000298 ulp=379412 |
| 14 | C_nm[tf32] | abs=0.000138 rel=0.002 ulp=312438 | 0.793 | abs=0.000593 rel=0.000298 ulp=379412 |
| 15 | A_prod | abs=1.3e-08 rel=1.2e-07 ulp=39 | 1.000 | — |
| 15 | B_legacyWY fp32-closed | abs=1.51e-08 rel=1.39e-07 ulp=76 | — | — |
| 15 | B_legacyWY bf16-bnd | abs=0.000152 rel=0.001 ulp=597246 | — | — |
| 15 | B_fs[ieee] | abs=4.67e-08 rel=4.3e-07 ulp=76 | 1.000 | abs=3.1e-07 rel=3.81e-07 ulp=203 |
| 15 | C_nm[ieee] | abs=4.67e-08 rel=4.3e-07 ulp=77 | 1.000 | abs=3.1e-07 rel=3.81e-07 ulp=204 |
| 15 | B_fs[tf32] | abs=7.07e-05 rel=0.000651 ulp=127255 | 0.802 | abs=0.000436 rel=0.000536 ulp=654676 |
| 15 | C_nm[tf32] | abs=7.08e-05 rel=0.000651 ulp=126736 | 0.801 | abs=0.000437 rel=0.000537 ulp=659242 |
| 16 | A_prod | abs=1.56e-08 rel=1.25e-07 ulp=28 | 1.000 | — |
| 16 | B_legacyWY fp32-closed | abs=1.79e-08 rel=1.43e-07 ulp=25 | — | — |
| 16 | B_legacyWY bf16-bnd | abs=0.000171 rel=0.001 ulp=448307 | — | — |
| 16 | B_fs[ieee] | abs=3.4e-08 rel=2.73e-07 ulp=72 | 1.000 | abs=2.76e-07 rel=2.96e-07 ulp=495 |
| 16 | C_nm[ieee] | abs=3.4e-08 rel=2.73e-07 ulp=69 | 1.000 | abs=2.76e-07 rel=2.96e-07 ulp=495 |
| 16 | B_fs[tf32] | abs=6.33e-05 rel=0.000508 ulp=293297 | 0.790 | abs=0.000571 rel=0.000613 ulp=1225408 |
| 16 | C_nm[tf32] | abs=6.43e-05 rel=0.000516 ulp=261865 | 0.793 | abs=0.000571 rel=0.000613 ulp=1225408 |
| 17 | A_prod | abs=5.26e-08 rel=2.36e-07 ulp=17 | 1.000 | — |
| 17 | B_legacyWY fp32-closed | abs=2.82e-08 rel=1.26e-07 ulp=20 | — | — |
| 17 | B_legacyWY bf16-bnd | abs=0.000291 rel=0.001 ulp=251316 | — | — |
| 17 | B_fs[ieee] | abs=8.42e-08 rel=3.78e-07 ulp=38 | 1.000 | abs=1.7e-07 rel=2.78e-07 ulp=336 |
| 17 | C_nm[ieee] | abs=8.42e-08 rel=3.78e-07 ulp=32 | 1.000 | abs=1.83e-07 rel=2.98e-07 ulp=392 |
| 17 | B_fs[tf32] | abs=0.000136 rel=0.00061 ulp=85863 | 0.816 | abs=0.00034 rel=0.000556 ulp=295605 |
| 17 | C_nm[tf32] | abs=0.000136 rel=0.00061 ulp=86536 | 0.823 | abs=0.00034 rel=0.000556 ulp=344469 |
| 18 | A_prod | abs=1.68e-08 rel=1.5e-07 ulp=34 | 1.000 | — |
| 18 | B_legacyWY fp32-closed | abs=1.65e-08 rel=1.48e-07 ulp=38 | — | — |
| 18 | B_legacyWY bf16-bnd | abs=0.000174 rel=0.002 ulp=695380 | — | — |
| 18 | B_fs[ieee] | abs=3.74e-08 rel=3.36e-07 ulp=71 | 1.000 | abs=2.46e-07 rel=2.08e-07 ulp=133 |
| 18 | C_nm[ieee] | abs=3.74e-08 rel=3.36e-07 ulp=71 | 1.000 | abs=2.39e-07 rel=2.02e-07 ulp=133 |
| 18 | B_fs[tf32] | abs=8.22e-05 rel=0.000737 ulp=134899 | 0.790 | abs=0.000297 rel=0.000251 ulp=338649 |
| 18 | C_nm[tf32] | abs=8.22e-05 rel=0.000738 ulp=129645 | 0.795 | abs=0.000512 rel=0.000434 ulp=350185 |
| 19 | A_prod | abs=2.07e-08 rel=1.3e-07 ulp=64 | 1.000 | — |
| 19 | B_legacyWY fp32-closed | abs=1.92e-08 rel=1.2e-07 ulp=35 | — | — |
| 19 | B_legacyWY bf16-bnd | abs=0.000188 rel=0.001 ulp=1116576 | — | — |
| 19 | B_fs[ieee] | abs=4.9e-08 rel=3.07e-07 ulp=61 | 1.000 | abs=1.86e-07 rel=1.81e-07 ulp=199 |
| 19 | C_nm[ieee] | abs=4.9e-08 rel=3.07e-07 ulp=63 | 1.000 | abs=1.93e-07 rel=1.89e-07 ulp=202 |
| 19 | B_fs[tf32] | abs=0.000109 rel=0.000687 ulp=102926 | 0.793 | abs=0.000407 rel=0.000397 ulp=274915 |
| 19 | C_nm[tf32] | abs=0.000109 rel=0.000687 ulp=124136 | 0.793 | abs=0.000606 rel=0.000591 ulp=301119 |
| 20 | A_prod | abs=2.11e-08 rel=1.35e-07 ulp=28 | 1.000 | — |
| 20 | B_legacyWY fp32-closed | abs=3.39e-08 rel=2.16e-07 ulp=116 | — | — |
| 20 | B_legacyWY bf16-bnd | abs=0.000209 rel=0.001 ulp=731140 | — | — |
| 20 | B_fs[ieee] | abs=7.29e-08 rel=4.66e-07 ulp=342 | 1.000 | abs=7e-07 rel=3.34e-07 ulp=376 |
| 20 | C_nm[ieee] | abs=7.29e-08 rel=4.66e-07 ulp=245 | 1.000 | abs=7.16e-07 rel=3.42e-07 ulp=385 |
| 20 | B_fs[tf32] | abs=0.000248 rel=0.002 ulp=330013 | 0.787 | abs=0.001 rel=0.000664 ulp=223676 |
| 20 | C_nm[tf32] | abs=0.000248 rel=0.002 ulp=456846 | 0.787 | abs=0.003 rel=0.001 ulp=225943 |
| 21 | A_prod | abs=1.03e-08 rel=1.26e-07 ulp=41 | 1.000 | — |
| 21 | B_legacyWY fp32-closed | abs=9.02e-09 rel=1.11e-07 ulp=50 | — | — |
| 21 | B_legacyWY bf16-bnd | abs=0.000216 rel=0.003 ulp=1955556 | — | — |
| 21 | B_fs[ieee] | abs=2.2e-08 rel=2.7e-07 ulp=59 | 1.000 | abs=2.81e-07 rel=2.4e-07 ulp=321 |
| 21 | C_nm[ieee] | abs=2.2e-08 rel=2.7e-07 ulp=84 | 1.000 | abs=3.11e-07 rel=2.66e-07 ulp=319 |
| 21 | B_fs[tf32] | abs=6.4e-05 rel=0.000786 ulp=302216 | 0.781 | abs=0.000422 rel=0.000361 ulp=569793 |
| 21 | C_nm[tf32] | abs=6.4e-05 rel=0.000786 ulp=386437 | 0.782 | abs=0.000496 rel=0.000424 ulp=761858 |
| 22 | A_prod | abs=2.35e-08 rel=1.34e-07 ulp=41 | 1.000 | — |
| 22 | B_legacyWY fp32-closed | abs=1.61e-08 rel=9.16e-08 ulp=25 | — | — |
| 22 | B_legacyWY bf16-bnd | abs=0.00016 rel=0.000911 ulp=551444 | — | — |
| 22 | B_fs[ieee] | abs=5.26e-08 rel=2.99e-07 ulp=39 | 1.000 | abs=1.45e-07 rel=1.61e-07 ulp=123 |
| 22 | C_nm[ieee] | abs=5.26e-08 rel=2.99e-07 ulp=39 | 1.000 | abs=1.43e-07 rel=1.6e-07 ulp=124 |
| 22 | B_fs[tf32] | abs=0.000101 rel=0.000572 ulp=162216 | 0.801 | abs=0.000432 rel=0.00048 ulp=356974 |
| 22 | C_nm[tf32] | abs=0.000101 rel=0.000572 ulp=134883 | 0.802 | abs=0.000432 rel=0.00048 ulp=356974 |
| 23 | A_prod | abs=9.18e-08 rel=3.8e-07 ulp=20 | 1.000 | — |
| 23 | B_legacyWY fp32-closed | abs=5.14e-08 rel=2.12e-07 ulp=12 | — | — |
| 23 | B_legacyWY bf16-bnd | abs=0.000417 rel=0.002 ulp=303688 | — | — |
| 23 | B_fs[ieee] | abs=9.11e-08 rel=3.77e-07 ulp=24 | 1.000 | abs=1.14e-07 rel=2.24e-07 ulp=202 |
| 23 | C_nm[ieee] | abs=9.11e-08 rel=3.77e-07 ulp=24 | 1.000 | abs=1.14e-07 rel=2.24e-07 ulp=211 |
| 23 | B_fs[tf32] | abs=0.000227 rel=0.000937 ulp=66676 | 0.823 | abs=0.000284 rel=0.000559 ulp=470319 |
| 23 | C_nm[tf32] | abs=0.000227 rel=0.000937 ulp=72666 | 0.831 | abs=0.000284 rel=0.000559 ulp=470319 |
| 24 | A_prod | abs=2.23e-08 rel=1.24e-07 ulp=44 | 1.000 | — |
| 24 | B_legacyWY fp32-closed | abs=2.58e-08 rel=1.43e-07 ulp=33 | — | — |
| 24 | B_legacyWY bf16-bnd | abs=0.000265 rel=0.001 ulp=1010069 | — | — |
| 24 | B_fs[ieee] | abs=4.44e-08 rel=2.47e-07 ulp=69 | 1.000 | abs=2.6e-07 rel=2.1e-07 ulp=213 |
| 24 | C_nm[ieee] | abs=4.44e-08 rel=2.47e-07 ulp=67 | 1.000 | abs=2.6e-07 rel=2.1e-07 ulp=216 |
| 24 | B_fs[tf32] | abs=0.000142 rel=0.000789 ulp=156651 | 0.797 | abs=0.000458 rel=0.00037 ulp=547371 |
| 24 | C_nm[tf32] | abs=0.000142 rel=0.000789 ulp=156651 | 0.798 | abs=0.000458 rel=0.00037 ulp=507225 |
| 25 | A_prod | abs=2.01e-08 rel=1.47e-07 ulp=35 | 1.000 | — |
| 25 | B_legacyWY fp32-closed | abs=1.33e-08 rel=9.69e-08 ulp=54 | — | — |
| 25 | B_legacyWY bf16-bnd | abs=0.000178 rel=0.001 ulp=871134 | — | — |
| 25 | B_fs[ieee] | abs=3.74e-08 rel=2.73e-07 ulp=38 | 1.000 | abs=1.82e-07 rel=1.6e-07 ulp=357 |
| 25 | C_nm[ieee] | abs=3.74e-08 rel=2.73e-07 ulp=36 | 1.000 | abs=1.82e-07 rel=1.6e-07 ulp=356 |
| 25 | B_fs[tf32] | abs=9.71e-05 rel=0.00071 ulp=126228 | 0.783 | abs=0.00054 rel=0.000476 ulp=459629 |
| 25 | C_nm[tf32] | abs=9.71e-05 rel=0.00071 ulp=118063 | 0.784 | abs=0.000544 rel=0.00048 ulp=459792 |
| 26 | A_prod | abs=1.48e-08 rel=1.37e-07 ulp=123 | 1.000 | — |
| 26 | B_legacyWY fp32-closed | abs=1.48e-08 rel=1.37e-07 ulp=67 | — | — |
| 26 | B_legacyWY bf16-bnd | abs=0.000161 rel=0.001 ulp=709839 | — | — |
| 26 | B_fs[ieee] | abs=3.08e-08 rel=2.84e-07 ulp=85 | 1.000 | abs=1.69e-07 rel=1.43e-07 ulp=154 |
| 26 | C_nm[ieee] | abs=3.08e-08 rel=2.84e-07 ulp=85 | 1.000 | abs=2.27e-07 rel=1.92e-07 ulp=155 |
| 26 | B_fs[tf32] | abs=7.02e-05 rel=0.000648 ulp=244437 | 0.779 | abs=0.000322 rel=0.000272 ulp=245041 |
| 26 | C_nm[tf32] | abs=7.02e-05 rel=0.000648 ulp=243606 | 0.782 | abs=0.000413 rel=0.000349 ulp=245041 |
| 27 | A_prod | abs=2.26e-08 rel=1.12e-07 ulp=61 | 1.000 | — |
| 27 | B_legacyWY fp32-closed | abs=2.14e-08 rel=1.06e-07 ulp=36 | — | — |
| 27 | B_legacyWY bf16-bnd | abs=0.000252 rel=0.001 ulp=878502 | — | — |
| 27 | B_fs[ieee] | abs=4.39e-08 rel=2.17e-07 ulp=54 | 1.000 | abs=1.97e-07 rel=1.61e-07 ulp=282 |
| 27 | C_nm[ieee] | abs=4.39e-08 rel=2.17e-07 ulp=58 | 1.000 | abs=1.97e-07 rel=1.61e-07 ulp=280 |
| 27 | B_fs[tf32] | abs=0.00012 rel=0.000593 ulp=136725 | 0.795 | abs=0.000431 rel=0.000353 ulp=421451 |
| 27 | C_nm[tf32] | abs=0.000119 rel=0.000586 ulp=133181 | 0.798 | abs=0.000471 rel=0.000386 ulp=421451 |
| 28 | A_prod | abs=1.33e-08 rel=1.21e-07 ulp=54 | 1.000 | — |
| 28 | B_legacyWY fp32-closed | abs=1.31e-08 rel=1.19e-07 ulp=50 | — | — |
| 28 | B_legacyWY bf16-bnd | abs=0.000189 rel=0.002 ulp=569518 | — | — |
| 28 | B_fs[ieee] | abs=3.08e-08 rel=2.79e-07 ulp=51 | 1.000 | abs=2.12e-07 rel=2.05e-07 ulp=124 |
| 28 | C_nm[ieee] | abs=3.08e-08 rel=2.79e-07 ulp=51 | 1.000 | abs=2.05e-07 rel=1.98e-07 ulp=141 |
| 28 | B_fs[tf32] | abs=7.39e-05 rel=0.000669 ulp=191043 | 0.789 | abs=0.000398 rel=0.000385 ulp=389567 |
| 28 | C_nm[tf32] | abs=7.4e-05 rel=0.000669 ulp=280746 | 0.788 | abs=0.00038 rel=0.000368 ulp=382108 |
| 29 | A_prod | abs=1.65e-08 rel=9.39e-08 ulp=39 | 1.000 | — |
| 29 | B_legacyWY fp32-closed | abs=2.25e-08 rel=1.27e-07 ulp=55 | — | — |
| 29 | B_legacyWY bf16-bnd | abs=0.000306 rel=0.002 ulp=1070953 | — | — |
| 29 | B_fs[ieee] | abs=7.21e-08 rel=4.09e-07 ulp=55 | 1.000 | abs=4.46e-07 rel=4.45e-07 ulp=266 |
| 29 | C_nm[ieee] | abs=7.21e-08 rel=4.09e-07 ulp=45 | 1.000 | abs=4.42e-07 rel=4.41e-07 ulp=266 |
| 29 | B_fs[tf32] | abs=0.000122 rel=0.000691 ulp=331141 | 0.794 | abs=0.00036 rel=0.00036 ulp=651944 |
| 29 | C_nm[tf32] | abs=0.000114 rel=0.000647 ulp=331141 | 0.797 | abs=0.000567 rel=0.000565 ulp=927345 |
| 30 | A_prod | abs=1.85e-08 rel=1.73e-07 ulp=82 | 1.000 | — |
| 30 | B_legacyWY fp32-closed | abs=1.34e-08 rel=1.25e-07 ulp=81 | — | — |
| 30 | B_legacyWY bf16-bnd | abs=0.000131 rel=0.001 ulp=881304 | — | — |
| 30 | B_fs[ieee] | abs=4.81e-08 rel=4.5e-07 ulp=72 | 1.000 | abs=1.29e-07 rel=1.46e-07 ulp=137 |
| 30 | C_nm[ieee] | abs=4.81e-08 rel=4.5e-07 ulp=72 | 1.000 | abs=1.27e-07 rel=1.44e-07 ulp=141 |
| 30 | B_fs[tf32] | abs=8.23e-05 rel=0.00077 ulp=300041 | 0.786 | abs=0.000246 rel=0.000279 ulp=262436 |
| 30 | C_nm[tf32] | abs=8.33e-05 rel=0.00078 ulp=293875 | 0.786 | abs=0.000495 rel=0.00056 ulp=531633 |
| 31 | A_prod | abs=1.38e-08 rel=1.01e-07 ulp=34 | 1.000 | — |
| 31 | B_legacyWY fp32-closed | abs=1.71e-08 rel=1.25e-07 ulp=27 | — | — |
| 31 | B_legacyWY bf16-bnd | abs=0.000142 rel=0.001 ulp=401504 | — | — |
| 31 | B_fs[ieee] | abs=4.75e-08 rel=3.46e-07 ulp=33 | 1.000 | abs=1.63e-07 rel=1.77e-07 ulp=350 |
| 31 | C_nm[ieee] | abs=4.75e-08 rel=3.46e-07 ulp=33 | 1.000 | abs=1.62e-07 rel=1.76e-07 ulp=348 |
| 31 | B_fs[tf32] | abs=0.000102 rel=0.000745 ulp=118952 | 0.788 | abs=0.000649 rel=0.000703 ulp=691937 |
| 31 | C_nm[tf32] | abs=0.000102 rel=0.000745 ulp=111160 | 0.790 | abs=0.000617 rel=0.000669 ulp=697692 |

Production-scan identity checks (NUMERIC equality folds +0/-0; BITWISE is integer-view equality): id 1: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 2: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 3: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=0.999984 bitwise=0.999984, out16 bytes equal payload serving_out=yes; id 4: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 5: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 6: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 7: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 8: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 9: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 10: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 11: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 12: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 13: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 14: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 15: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 16: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 17: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 18: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 19: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 20: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 21: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 22: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 23: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 24: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 25: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 26: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 27: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 28: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 29: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=0.999984 bitwise=0.999984, out16 bytes equal payload serving_out=yes; id 30: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes; id 31: out32 vs native_sg numeric=1.000 bitwise=1.000000, out16 vs native_sg numeric=1.000000 bitwise=1.000000, out16 bytes equal payload serving_out=yes

Seam ablation for the one-token decode reference (native_pk state vs fp64 oracles applying ONE convention; max_rel): id 1: beta-bf16-rt only=rel=2.62e-07, div-sqrt only=rel=0.000286, both=rel=2.62e-07; id 2: beta-bf16-rt only=rel=1.88e-07, div-sqrt only=rel=0.000196, both=rel=1.88e-07; id 3: beta-bf16-rt only=rel=1.58e-07, div-sqrt only=rel=0.000608, both=rel=1.58e-07; id 4: beta-bf16-rt only=rel=1.64e-07, div-sqrt only=rel=0.000235, both=rel=1.64e-07; id 5: beta-bf16-rt only=rel=1.56e-07, div-sqrt only=rel=0.000291, both=rel=1.56e-07; id 6: beta-bf16-rt only=rel=1.91e-07, div-sqrt only=rel=0.000864, both=rel=1.91e-07; id 7: beta-bf16-rt only=rel=2.82e-07, div-sqrt only=rel=0.000347, both=rel=2.82e-07; id 8: beta-bf16-rt only=rel=2.77e-07, div-sqrt only=rel=0.000378, both=rel=2.77e-07; id 9: beta-bf16-rt only=rel=1.68e-07, div-sqrt only=rel=0.0005, both=rel=1.68e-07; id 10: beta-bf16-rt only=rel=1.22e-07, div-sqrt only=rel=0.000224, both=rel=1.22e-07; id 11: beta-bf16-rt only=rel=2.64e-07, div-sqrt only=rel=0.000261, both=rel=2.64e-07; id 12: beta-bf16-rt only=rel=1.67e-07, div-sqrt only=rel=0.00089, both=rel=1.67e-07; id 13: beta-bf16-rt only=rel=1.68e-07, div-sqrt only=rel=0.000171, both=rel=1.68e-07; id 14: beta-bf16-rt only=rel=2.01e-07, div-sqrt only=rel=0.000445, both=rel=2.01e-07; id 15: beta-bf16-rt only=rel=1.56e-07, div-sqrt only=rel=0.000158, both=rel=1.56e-07; id 16: beta-bf16-rt only=rel=1.27e-07, div-sqrt only=rel=0.00031, both=rel=1.27e-07; id 17: beta-bf16-rt only=rel=2.17e-07, div-sqrt only=rel=0.000158, both=rel=2.17e-07; id 18: beta-bf16-rt only=rel=1.95e-07, div-sqrt only=rel=0.000214, both=rel=1.95e-07; id 19: beta-bf16-rt only=rel=1.84e-07, div-sqrt only=rel=0.00023, both=rel=1.84e-07; id 20: beta-bf16-rt only=rel=1.66e-07, div-sqrt only=rel=0.000586, both=rel=1.66e-07; id 21: beta-bf16-rt only=rel=2.12e-07, div-sqrt only=rel=0.000201, both=rel=2.12e-07; id 22: beta-bf16-rt only=rel=1.47e-07, div-sqrt only=rel=0.000156, both=rel=1.47e-07; id 23: beta-bf16-rt only=rel=3.18e-07, div-sqrt only=rel=0.000152, both=rel=3.18e-07; id 24: beta-bf16-rt only=rel=1.79e-07, div-sqrt only=rel=0.000229, both=rel=1.79e-07; id 25: beta-bf16-rt only=rel=1.81e-07, div-sqrt only=rel=0.000297, both=rel=1.81e-07; id 26: beta-bf16-rt only=rel=1.95e-07, div-sqrt only=rel=0.000226, both=rel=1.95e-07; id 27: beta-bf16-rt only=rel=2.4e-07, div-sqrt only=rel=0.000478, both=rel=2.4e-07; id 28: beta-bf16-rt only=rel=2.9e-07, div-sqrt only=rel=0.000235, both=rel=2.9e-07; id 29: beta-bf16-rt only=rel=1.37e-07, div-sqrt only=rel=0.000206, both=rel=1.37e-07; id 30: beta-bf16-rt only=rel=1.7e-07, div-sqrt only=rel=0.000153, both=rel=1.7e-07; id 31: beta-bf16-rt only=rel=2.58e-07, div-sqrt only=rel=0.000236, both=rel=2.58e-07

`A_legacy_state*` rows are excluded: the June-8 vendored sequential scan's export path skips the root update (source-inspected); when/which binary changed that path is NOT established. Its outputs are bit-identical to the production scan.

## Stage 2 — commit from IDENTICAL oracle factors (compact reconstruction kernel; vs oracle state / vs native_sg state)

| id | B_fs[ieee] vs oracle | vs native_sg | C_nm[ieee] vs oracle | B_fs[tf32] vs oracle | C_nm[tf32] vs oracle |
|---|---|---|---|---|---|
| 1 | abs=4.46e-07 rel=1.52e-07 ulp=17 | abs=7.15e-07 rel=2.43e-07 ulp=19 | abs=4.46e-07 rel=1.52e-07 ulp=17 | abs=0.000562 rel=0.000191 ulp=99942 | abs=0.000562 rel=0.000191 ulp=99942 |
| 2 | abs=4.02e-07 rel=1.38e-07 ulp=15 | abs=7.15e-07 rel=2.45e-07 ulp=35 | abs=4.02e-07 rel=1.38e-07 ulp=15 | abs=0.000498 rel=0.000171 ulp=143103 | abs=0.000498 rel=0.000171 ulp=143103 |
| 3 | abs=4.11e-07 rel=1.48e-07 ulp=45 | abs=3.58e-07 rel=1.29e-07 ulp=36 | abs=4.11e-07 rel=1.48e-07 ulp=45 | abs=0.000841 rel=0.000302 ulp=285115 | abs=0.000841 rel=0.000302 ulp=285115 |
| 4 | abs=4.26e-07 rel=1.27e-07 ulp=14 | abs=1.19e-06 rel=3.55e-07 ulp=18 | abs=4.26e-07 rel=1.27e-07 ulp=14 | abs=0.000703 rel=0.000209 ulp=90634 | abs=0.000703 rel=0.000209 ulp=90634 |
| 5 | abs=4.87e-07 rel=1.36e-07 ulp=38 | abs=4.77e-07 rel=1.33e-07 ulp=52 | abs=4.87e-07 rel=1.36e-07 ulp=38 | abs=0.000559 rel=0.000156 ulp=142437 | abs=0.000559 rel=0.000156 ulp=142437 |
| 6 | abs=5.07e-07 rel=1.57e-07 ulp=42 | abs=7.15e-07 rel=2.21e-07 ulp=29 | abs=5.07e-07 rel=1.57e-07 ulp=42 | abs=0.002 rel=0.000504 ulp=291258 | abs=0.002 rel=0.000504 ulp=291258 |
| 7 | abs=4.09e-07 rel=1.39e-07 ulp=16 | abs=4.77e-07 rel=1.62e-07 ulp=44 | abs=4.09e-07 rel=1.39e-07 ulp=16 | abs=0.000679 rel=0.000231 ulp=99224 | abs=0.000679 rel=0.000231 ulp=99224 |
| 8 | abs=3.9e-07 rel=1.36e-07 ulp=29 | abs=7.15e-07 rel=2.5e-07 ulp=39 | abs=3.9e-07 rel=1.36e-07 ulp=29 | abs=0.000523 rel=0.000183 ulp=156837 | abs=0.000523 rel=0.000183 ulp=156837 |
| 9 | abs=3.58e-07 rel=1.21e-07 ulp=17 | abs=4.77e-07 rel=1.61e-07 ulp=40 | abs=3.58e-07 rel=1.21e-07 ulp=17 | abs=0.000431 rel=0.000146 ulp=147602 | abs=0.000431 rel=0.000146 ulp=147602 |
| 10 | abs=4.05e-07 rel=1.3e-07 ulp=24 | abs=4.77e-07 rel=1.53e-07 ulp=24 | abs=4.05e-07 rel=1.3e-07 ulp=24 | abs=0.000328 rel=0.000105 ulp=98247 | abs=0.000328 rel=0.000105 ulp=98247 |
| 11 | abs=3.54e-07 rel=1.36e-07 ulp=15 | abs=7.15e-07 rel=2.75e-07 ulp=25 | abs=3.54e-07 rel=1.36e-07 ulp=15 | abs=0.000241 rel=9.29e-05 ulp=138216 | abs=0.000241 rel=9.29e-05 ulp=138216 |
| 12 | abs=4.15e-07 rel=1.28e-07 ulp=30 | abs=7.15e-07 rel=2.2e-07 ulp=36 | abs=4.15e-07 rel=1.28e-07 ulp=30 | abs=0.001 rel=0.000435 ulp=170100 | abs=0.001 rel=0.000435 ulp=170100 |
| 13 | abs=4.28e-07 rel=1.31e-07 ulp=9 | abs=7.15e-07 rel=2.19e-07 ulp=15 | abs=4.28e-07 rel=1.31e-07 ulp=9 | abs=0.000313 rel=9.59e-05 ulp=57344 | abs=0.000313 rel=9.59e-05 ulp=57344 |
| 14 | abs=4.34e-07 rel=1.62e-07 ulp=22 | abs=7.15e-07 rel=2.66e-07 ulp=52 | abs=4.34e-07 rel=1.62e-07 ulp=22 | abs=0.000638 rel=0.000237 ulp=190368 | abs=0.000638 rel=0.000237 ulp=190368 |
| 15 | abs=4.41e-07 rel=1.3e-07 ulp=11 | abs=7.15e-07 rel=2.11e-07 ulp=19 | abs=4.41e-07 rel=1.3e-07 ulp=11 | abs=0.000308 rel=9.07e-05 ulp=63696 | abs=0.000308 rel=9.07e-05 ulp=63696 |
| 16 | abs=4.92e-07 rel=1.6e-07 ulp=28 | abs=7.15e-07 rel=2.32e-07 ulp=36 | abs=4.92e-07 rel=1.6e-07 ulp=28 | abs=0.000498 rel=0.000162 ulp=109977 | abs=0.000498 rel=0.000162 ulp=109977 |
| 17 | abs=4.74e-07 rel=1.48e-07 ulp=16 | abs=4.77e-07 rel=1.48e-07 ulp=25 | abs=4.74e-07 rel=1.48e-07 ulp=16 | abs=0.000351 rel=0.000109 ulp=130455 | abs=0.000351 rel=0.000109 ulp=130455 |
| 18 | abs=6.35e-07 rel=1.58e-07 ulp=8 | abs=4.77e-07 rel=1.19e-07 ulp=11 | abs=6.35e-07 rel=1.58e-07 ulp=8 | abs=0.000508 rel=0.000127 ulp=136356 | abs=0.000508 rel=0.000127 ulp=136356 |
| 19 | abs=5.34e-07 rel=1.37e-07 ulp=10 | abs=4.77e-07 rel=1.23e-07 ulp=18 | abs=5.34e-07 rel=1.37e-07 ulp=10 | abs=0.000422 rel=0.000109 ulp=40260 | abs=0.000422 rel=0.000109 ulp=40260 |
| 20 | abs=5.99e-07 rel=1.53e-07 ulp=34 | abs=7.15e-07 rel=1.83e-07 ulp=23 | abs=5.99e-07 rel=1.53e-07 ulp=34 | abs=0.002 rel=0.00039 ulp=128205 | abs=0.002 rel=0.00039 ulp=128205 |
| 21 | abs=4.8e-07 rel=1.37e-07 ulp=11 | abs=7.15e-07 rel=2.04e-07 ulp=78 | abs=4.8e-07 rel=1.37e-07 ulp=11 | abs=0.000587 rel=0.000167 ulp=62953 | abs=0.000587 rel=0.000167 ulp=62953 |
| 22 | abs=4.81e-07 rel=1.2e-07 ulp=9 | abs=4.77e-07 rel=1.19e-07 ulp=11 | abs=4.81e-07 rel=1.2e-07 ulp=9 | abs=0.000293 rel=7.28e-05 ulp=36117 | abs=0.000293 rel=7.28e-05 ulp=36117 |
| 23 | abs=4.54e-07 rel=1.46e-07 ulp=6 | abs=9.54e-07 rel=3.07e-07 ulp=13 | abs=4.54e-07 rel=1.46e-07 ulp=6 | abs=0.00037 rel=0.000119 ulp=35880 | abs=0.00037 rel=0.000119 ulp=35880 |
| 24 | abs=5.19e-07 rel=1.25e-07 ulp=11 | abs=9.54e-07 rel=2.3e-07 ulp=14 | abs=5.19e-07 rel=1.25e-07 ulp=11 | abs=0.000625 rel=0.000151 ulp=49772 | abs=0.000625 rel=0.000151 ulp=49772 |
| 25 | abs=7.81e-07 rel=1.57e-07 ulp=19 | abs=9.54e-07 rel=1.92e-07 ulp=28 | abs=7.81e-07 rel=1.57e-07 ulp=19 | abs=0.000713 rel=0.000143 ulp=60872 | abs=0.000713 rel=0.000143 ulp=60872 |
| 26 | abs=4.67e-07 rel=1.13e-07 ulp=8 | abs=9.54e-07 rel=2.3e-07 ulp=14 | abs=4.67e-07 rel=1.13e-07 ulp=8 | abs=0.000455 rel=0.00011 ulp=60679 | abs=0.000455 rel=0.00011 ulp=60679 |
| 27 | abs=7e-07 rel=1.66e-07 ulp=12 | abs=7.15e-07 rel=1.69e-07 ulp=21 | abs=7e-07 rel=1.66e-07 ulp=12 | abs=0.000669 rel=0.000158 ulp=72800 | abs=0.000669 rel=0.000158 ulp=72800 |
| 28 | abs=7.24e-07 rel=1.72e-07 ulp=13 | abs=9.54e-07 rel=2.27e-07 ulp=19 | abs=7.24e-07 rel=1.72e-07 ulp=13 | abs=0.000443 rel=0.000105 ulp=71496 | abs=0.000443 rel=0.000105 ulp=71496 |
| 29 | abs=6.19e-07 rel=1.23e-07 ulp=13 | abs=1.91e-06 rel=3.8e-07 ulp=16 | abs=6.19e-07 rel=1.23e-07 ulp=13 | abs=0.000878 rel=0.000175 ulp=83452 | abs=0.000878 rel=0.000175 ulp=83452 |
| 30 | abs=7.05e-07 rel=1.72e-07 ulp=9 | abs=9.54e-07 rel=2.33e-07 ulp=10 | abs=7.05e-07 rel=1.72e-07 ulp=9 | abs=0.00036 rel=8.79e-05 ulp=41495 | abs=0.00036 rel=8.79e-05 ulp=41495 |
| 31 | abs=7.29e-07 rel=1.75e-07 ulp=10 | abs=1.43e-06 rel=3.42e-07 ulp=10 | abs=7.29e-07 rel=1.75e-07 ulp=10 | abs=0.000575 rel=0.000138 ulp=40879 | abs=0.000575 rel=0.000138 ulp=40879 |

## Stage 3 — own factors + own commit (all accepted prefixes incl. zero-accept; vs oracle / native_sg / native_pk)

| id | mechanism | vs oracle | vs native_sg (spec-update) | exact frac vs native_sg | vs native_pk (one-token decode) |
|---|---|---|---|---|---|
| 1 | A_prod replay | abs=5.66e-07 rel=1.93e-07 ulp=19 | abs=5.96e-08 rel=2.03e-08 ulp=7 | 0.992 | abs=0.000842 rel=0.000286 ulp=265995 |
| 1 | B_legacyWY fp32-closed (materialized) | abs=4.46e-07 rel=1.52e-07 ulp=24 | — | — | — |
| 1 | B_fs[ieee] compact | abs=4.46e-07 rel=1.52e-07 ulp=37 | abs=7.15e-07 rel=2.43e-07 ulp=40 | 0.307 | abs=0.000842 rel=0.000286 ulp=265989 |
| 1 | C_nm[ieee] compact | abs=4.46e-07 rel=1.52e-07 ulp=37 | abs=7.15e-07 rel=2.43e-07 ulp=38 | 0.304 | abs=0.000842 rel=0.000286 ulp=265989 |
| 1 | B_fs[tf32] compact | abs=0.000598 rel=0.000203 ulp=114831 | abs=0.000598 rel=0.000203 ulp=114827 | 0.000409 | abs=0.001 rel=0.000385 ulp=286667 |
| 1 | C_nm[tf32] compact | abs=0.000605 rel=0.000206 ulp=138826 | abs=0.000605 rel=0.000206 ulp=138826 | 0.000406 | abs=0.001 rel=0.000383 ulp=286667 |
| 2 | A_prod replay | abs=7.13e-07 rel=2.45e-07 ulp=36 | abs=2.98e-08 rel=1.02e-08 ulp=6 | 0.991 | abs=0.000572 rel=0.000196 ulp=161630 |
| 2 | B_legacyWY fp32-closed (materialized) | abs=4.02e-07 rel=1.38e-07 ulp=29 | — | — | — |
| 2 | B_fs[ieee] compact | abs=4.02e-07 rel=1.38e-07 ulp=33 | abs=7.15e-07 rel=2.45e-07 ulp=36 | 0.317 | abs=0.000572 rel=0.000196 ulp=161612 |
| 2 | C_nm[ieee] compact | abs=4.02e-07 rel=1.38e-07 ulp=39 | abs=7.15e-07 rel=2.45e-07 ulp=36 | 0.314 | abs=0.000572 rel=0.000196 ulp=161606 |
| 2 | B_fs[tf32] compact | abs=0.000573 rel=0.000196 ulp=156199 | abs=0.000573 rel=0.000196 ulp=156183 | 0.000366 | abs=0.000808 rel=0.000277 ulp=216095 |
| 2 | C_nm[tf32] compact | abs=0.000573 rel=0.000196 ulp=156207 | abs=0.000573 rel=0.000196 ulp=156191 | 0.000368 | abs=0.000808 rel=0.000277 ulp=213343 |
| 3 | A_prod replay | abs=4.71e-07 rel=1.69e-07 ulp=28 | abs=5.96e-08 rel=2.14e-08 ulp=9 | 0.989 | abs=0.002 rel=0.000608 ulp=362960 |
| 3 | B_legacyWY fp32-closed (materialized) | abs=4.69e-07 rel=1.69e-07 ulp=44 | — | — | — |
| 3 | B_fs[ieee] compact | abs=4.11e-07 rel=1.48e-07 ulp=111 | abs=3.58e-07 rel=1.29e-07 ulp=103 | 0.302 | abs=0.002 rel=0.000607 ulp=362960 |
| 3 | C_nm[ieee] compact | abs=4.11e-07 rel=1.48e-07 ulp=112 | abs=3.58e-07 rel=1.29e-07 ulp=104 | 0.298 | abs=0.002 rel=0.000607 ulp=362973 |
| 3 | B_fs[tf32] compact | abs=0.002 rel=0.000589 ulp=436408 | abs=0.002 rel=0.000589 ulp=436416 | 0.000222 | abs=0.001 rel=0.000409 ulp=446455 |
| 3 | C_nm[tf32] compact | abs=0.002 rel=0.000589 ulp=398179 | abs=0.002 rel=0.000589 ulp=398188 | 0.000217 | abs=0.001 rel=0.000409 ulp=446455 |
| 4 | A_prod replay | abs=9.93e-07 rel=2.96e-07 ulp=18 | abs=2.98e-08 rel=8.88e-09 ulp=5 | 0.993 | abs=0.00079 rel=0.000235 ulp=154539 |
| 4 | B_legacyWY fp32-closed (materialized) | abs=4.26e-07 rel=1.27e-07 ulp=21 | — | — | — |
| 4 | B_fs[ieee] compact | abs=4.26e-07 rel=1.27e-07 ulp=44 | abs=1.19e-06 rel=3.55e-07 ulp=37 | 0.326 | abs=0.00079 rel=0.000235 ulp=154543 |
| 4 | C_nm[ieee] compact | abs=4.26e-07 rel=1.27e-07 ulp=44 | abs=1.19e-06 rel=3.55e-07 ulp=37 | 0.322 | abs=0.00079 rel=0.000235 ulp=154539 |
| 4 | B_fs[tf32] compact | abs=0.000657 rel=0.000196 ulp=159982 | abs=0.000657 rel=0.000196 ulp=159976 | 0.000754 | abs=0.001 rel=0.000344 ulp=276949 |
| 4 | C_nm[tf32] compact | abs=0.000706 rel=0.00021 ulp=152232 | abs=0.000706 rel=0.00021 ulp=152243 | 0.000756 | abs=0.000972 rel=0.000289 ulp=276949 |
| 5 | A_prod replay | abs=5.07e-07 rel=1.41e-07 ulp=14 | abs=2.98e-08 rel=8.3e-09 ulp=4 | 0.992 | abs=0.001 rel=0.000291 ulp=366744 |
| 5 | B_legacyWY fp32-closed (materialized) | abs=4.87e-07 rel=1.36e-07 ulp=43 | — | — | — |
| 5 | B_fs[ieee] compact | abs=4.87e-07 rel=1.36e-07 ulp=44 | abs=4.77e-07 rel=1.33e-07 ulp=50 | 0.327 | abs=0.001 rel=0.000291 ulp=366707 |
| 5 | C_nm[ieee] compact | abs=4.87e-07 rel=1.36e-07 ulp=44 | abs=4.77e-07 rel=1.33e-07 ulp=50 | 0.324 | abs=0.001 rel=0.000291 ulp=366707 |
| 5 | B_fs[tf32] compact | abs=0.000534 rel=0.000149 ulp=245261 | abs=0.000534 rel=0.000149 ulp=245275 | 0.000775 | abs=0.002 rel=0.000436 ulp=470658 |
| 5 | C_nm[tf32] compact | abs=0.000533 rel=0.000149 ulp=156017 | abs=0.000533 rel=0.000149 ulp=156015 | 0.000783 | abs=0.002 rel=0.000436 ulp=470658 |
| 6 | A_prod replay | abs=6.47e-07 rel=2e-07 ulp=29 | abs=5.96e-08 rel=1.85e-08 ulp=5 | 0.988 | abs=0.003 rel=0.000864 ulp=316121 |
| 6 | B_legacyWY fp32-closed (materialized) | abs=4.09e-07 rel=1.27e-07 ulp=42 | — | — | — |
| 6 | B_fs[ieee] compact | abs=5.07e-07 rel=1.57e-07 ulp=46 | abs=7.15e-07 rel=2.21e-07 ulp=53 | 0.304 | abs=0.003 rel=0.000864 ulp=316134 |
| 6 | C_nm[ieee] compact | abs=5.07e-07 rel=1.57e-07 ulp=46 | abs=7.15e-07 rel=2.21e-07 ulp=53 | 0.301 | abs=0.003 rel=0.000864 ulp=316132 |
| 6 | B_fs[tf32] compact | abs=0.002 rel=0.000511 ulp=524942 | abs=0.002 rel=0.000511 ulp=524968 | 0.000174 | abs=0.004 rel=0.001 ulp=816703 |
| 6 | C_nm[tf32] compact | abs=0.002 rel=0.000511 ulp=350618 | abs=0.002 rel=0.000511 ulp=350644 | 0.000174 | abs=0.004 rel=0.001 ulp=642379 |
| 7 | A_prod replay | abs=6.83e-07 rel=2.32e-07 ulp=28 | abs=5.96e-08 rel=2.02e-08 ulp=12 | 0.990 | abs=0.001 rel=0.000347 ulp=237676 |
| 7 | B_legacyWY fp32-closed (materialized) | abs=3.94e-07 rel=1.34e-07 ulp=35 | — | — | — |
| 7 | B_fs[ieee] compact | abs=4.09e-07 rel=1.39e-07 ulp=42 | abs=4.77e-07 rel=1.62e-07 ulp=42 | 0.302 | abs=0.001 rel=0.000347 ulp=237679 |
| 7 | C_nm[ieee] compact | abs=4.09e-07 rel=1.39e-07 ulp=41 | abs=4.77e-07 rel=1.62e-07 ulp=48 | 0.299 | abs=0.001 rel=0.000347 ulp=237679 |
| 7 | B_fs[tf32] compact | abs=0.000679 rel=0.000231 ulp=156144 | abs=0.000679 rel=0.000231 ulp=156139 | 0.000216 | abs=0.001 rel=0.000406 ulp=304283 |
| 7 | C_nm[tf32] compact | abs=0.000679 rel=0.000231 ulp=156144 | abs=0.000679 rel=0.000231 ulp=156139 | 0.000211 | abs=0.001 rel=0.000406 ulp=304283 |
| 8 | A_prod replay | abs=4.49e-07 rel=1.57e-07 ulp=33 | abs=2.98e-08 rel=1.04e-08 ulp=5 | 0.991 | abs=0.001 rel=0.000378 ulp=312868 |
| 8 | B_legacyWY fp32-closed (materialized) | abs=3.9e-07 rel=1.36e-07 ulp=42 | — | — | — |
| 8 | B_fs[ieee] compact | abs=2.81e-07 rel=9.84e-08 ulp=38 | abs=7.15e-07 rel=2.5e-07 ulp=44 | 0.305 | abs=0.001 rel=0.000378 ulp=312868 |
| 8 | C_nm[ieee] compact | abs=3.5e-07 rel=1.23e-07 ulp=38 | abs=7.15e-07 rel=2.5e-07 ulp=46 | 0.302 | abs=0.001 rel=0.000378 ulp=312868 |
| 8 | B_fs[tf32] compact | abs=0.000697 rel=0.000244 ulp=281499 | abs=0.000697 rel=0.000244 ulp=281532 | 0.00025 | abs=0.001 rel=0.000438 ulp=378360 |
| 8 | C_nm[tf32] compact | abs=0.000697 rel=0.000244 ulp=284868 | abs=0.000697 rel=0.000244 ulp=284901 | 0.000253 | abs=0.001 rel=0.000425 ulp=354372 |
| 9 | A_prod replay | abs=5.18e-07 rel=1.75e-07 ulp=29 | abs=2.98e-08 rel=1.01e-08 ulp=8 | 0.991 | abs=0.001 rel=0.0005 ulp=386777 |
| 9 | B_legacyWY fp32-closed (materialized) | abs=2.91e-07 rel=9.86e-08 ulp=35 | — | — | — |
| 9 | B_fs[ieee] compact | abs=3.58e-07 rel=1.21e-07 ulp=30 | abs=4.77e-07 rel=1.61e-07 ulp=35 | 0.317 | abs=0.001 rel=0.0005 ulp=386761 |
| 9 | C_nm[ieee] compact | abs=3.58e-07 rel=1.21e-07 ulp=39 | abs=4.77e-07 rel=1.61e-07 ulp=41 | 0.314 | abs=0.001 rel=0.0005 ulp=386773 |
| 9 | B_fs[tf32] compact | abs=0.000692 rel=0.000234 ulp=294527 | abs=0.000692 rel=0.000234 ulp=294522 | 0.000369 | abs=0.002 rel=0.000531 ulp=361087 |
| 9 | C_nm[tf32] compact | abs=0.000758 rel=0.000256 ulp=299485 | abs=0.000758 rel=0.000256 ulp=299480 | 0.000363 | abs=0.002 rel=0.000544 ulp=366023 |
| 10 | A_prod replay | abs=3.97e-07 rel=1.27e-07 ulp=17 | abs=2.98e-08 rel=9.54e-09 ulp=4 | 0.993 | abs=0.0007 rel=0.000224 ulp=295229 |
| 10 | B_legacyWY fp32-closed (materialized) | abs=5.02e-07 rel=1.61e-07 ulp=31 | — | — | — |
| 10 | B_fs[ieee] compact | abs=4.05e-07 rel=1.3e-07 ulp=65 | abs=5.96e-07 rel=1.91e-07 ulp=82 | 0.313 | abs=0.0007 rel=0.000224 ulp=295217 |
| 10 | C_nm[ieee] compact | abs=4.05e-07 rel=1.3e-07 ulp=64 | abs=5.96e-07 rel=1.91e-07 ulp=81 | 0.310 | abs=0.0007 rel=0.000224 ulp=295221 |
| 10 | B_fs[tf32] compact | abs=0.000438 rel=0.00014 ulp=120293 | abs=0.000438 rel=0.00014 ulp=120308 | 0.000842 | abs=0.000784 rel=0.000251 ulp=335689 |
| 10 | C_nm[tf32] compact | abs=0.000438 rel=0.00014 ulp=188386 | abs=0.000438 rel=0.00014 ulp=188396 | 0.000839 | abs=0.000806 rel=0.000258 ulp=275026 |
| 11 | A_prod replay | abs=5.21e-07 rel=2.01e-07 ulp=27 | abs=5.96e-08 rel=2.29e-08 ulp=12 | 0.990 | abs=0.000677 rel=0.000261 ulp=283224 |
| 11 | B_legacyWY fp32-closed (materialized) | abs=3.54e-07 rel=1.36e-07 ulp=21 | — | — | — |
| 11 | B_fs[ieee] compact | abs=3.54e-07 rel=1.36e-07 ulp=46 | abs=7.15e-07 rel=2.75e-07 ulp=27 | 0.324 | abs=0.000677 rel=0.000261 ulp=283216 |
| 11 | C_nm[ieee] compact | abs=3.54e-07 rel=1.36e-07 ulp=41 | abs=7.15e-07 rel=2.75e-07 ulp=27 | 0.322 | abs=0.000677 rel=0.000261 ulp=283223 |
| 11 | B_fs[tf32] compact | abs=0.000662 rel=0.000255 ulp=167570 | abs=0.000662 rel=0.000255 ulp=167569 | 0.000823 | abs=0.000841 rel=0.000324 ulp=361018 |
| 11 | C_nm[tf32] compact | abs=0.000664 rel=0.000256 ulp=167570 | abs=0.000664 rel=0.000256 ulp=167569 | 0.00082 | abs=0.00084 rel=0.000323 ulp=353536 |
| 12 | A_prod replay | abs=4.7e-07 rel=1.45e-07 ulp=24 | abs=2.38e-07 rel=7.34e-08 ulp=8 | 0.989 | abs=0.003 rel=0.00089 ulp=520135 |
| 12 | B_legacyWY fp32-closed (materialized) | abs=4.92e-07 rel=1.51e-07 ulp=36 | — | — | — |
| 12 | B_fs[ieee] compact | abs=4.15e-07 rel=1.28e-07 ulp=60 | abs=7.15e-07 rel=2.2e-07 ulp=62 | 0.304 | abs=0.003 rel=0.00089 ulp=520116 |
| 12 | C_nm[ieee] compact | abs=4.15e-07 rel=1.28e-07 ulp=60 | abs=7.15e-07 rel=2.2e-07 ulp=62 | 0.302 | abs=0.003 rel=0.00089 ulp=520125 |
| 12 | B_fs[tf32] compact | abs=0.001 rel=0.000445 ulp=319388 | abs=0.001 rel=0.000445 ulp=319377 | 0.00033 | abs=0.002 rel=0.000627 ulp=377149 |
| 12 | C_nm[tf32] compact | abs=0.001 rel=0.000447 ulp=320244 | abs=0.001 rel=0.000447 ulp=320233 | 0.000333 | abs=0.002 rel=0.0007 ulp=347191 |
| 13 | A_prod replay | abs=7.43e-07 rel=2.28e-07 ulp=13 | abs=5.96e-08 rel=1.83e-08 ulp=5 | 0.992 | abs=0.000559 rel=0.000171 ulp=108089 |
| 13 | B_legacyWY fp32-closed (materialized) | abs=3.76e-07 rel=1.15e-07 ulp=13 | — | — | — |
| 13 | B_fs[ieee] compact | abs=4.28e-07 rel=1.31e-07 ulp=22 | abs=7.15e-07 rel=2.19e-07 ulp=25 | 0.332 | abs=0.000559 rel=0.000171 ulp=108089 |
| 13 | C_nm[ieee] compact | abs=4.28e-07 rel=1.31e-07 ulp=22 | abs=7.15e-07 rel=2.19e-07 ulp=25 | 0.329 | abs=0.000559 rel=0.000171 ulp=108089 |
| 13 | B_fs[tf32] compact | abs=0.00038 rel=0.000117 ulp=75178 | abs=0.00038 rel=0.000117 ulp=75178 | 0.000687 | abs=0.000695 rel=0.000213 ulp=145645 |
| 13 | C_nm[tf32] compact | abs=0.000497 rel=0.000152 ulp=74271 | abs=0.000497 rel=0.000152 ulp=74276 | 0.000686 | abs=0.000757 rel=0.000232 ulp=145645 |
| 14 | A_prod replay | abs=5.49e-07 rel=2.04e-07 ulp=38 | abs=5.96e-08 rel=2.22e-08 ulp=9 | 0.989 | abs=0.001 rel=0.000445 ulp=290077 |
| 14 | B_legacyWY fp32-closed (materialized) | abs=3.09e-07 rel=1.15e-07 ulp=33 | — | — | — |
| 14 | B_fs[ieee] compact | abs=3.09e-07 rel=1.15e-07 ulp=79 | abs=7.15e-07 rel=2.66e-07 ulp=82 | 0.303 | abs=0.001 rel=0.000445 ulp=290129 |
| 14 | C_nm[ieee] compact | abs=3.09e-07 rel=1.15e-07 ulp=80 | abs=7.15e-07 rel=2.66e-07 ulp=83 | 0.301 | abs=0.001 rel=0.000445 ulp=290129 |
| 14 | B_fs[tf32] compact | abs=0.000638 rel=0.000237 ulp=296664 | abs=0.000638 rel=0.000237 ulp=296689 | 0.000256 | abs=0.002 rel=0.000599 ulp=571042 |
| 14 | C_nm[tf32] compact | abs=0.000638 rel=0.000237 ulp=287040 | abs=0.000638 rel=0.000237 ulp=287065 | 0.000258 | abs=0.002 rel=0.000599 ulp=562278 |
| 15 | A_prod replay | abs=6.03e-07 rel=1.78e-07 ulp=16 | abs=5.96e-08 rel=1.76e-08 ulp=4 | 0.992 | abs=0.000537 rel=0.000158 ulp=207144 |
| 15 | B_legacyWY fp32-closed (materialized) | abs=5.31e-07 rel=1.57e-07 ulp=20 | — | — | — |
| 15 | B_fs[ieee] compact | abs=4.41e-07 rel=1.3e-07 ulp=37 | abs=7.15e-07 rel=2.11e-07 ulp=32 | 0.330 | abs=0.000537 rel=0.000158 ulp=207130 |
| 15 | C_nm[ieee] compact | abs=4.41e-07 rel=1.3e-07 ulp=37 | abs=7.15e-07 rel=2.11e-07 ulp=32 | 0.327 | abs=0.000537 rel=0.000158 ulp=207135 |
| 15 | B_fs[tf32] compact | abs=0.000328 rel=9.67e-05 ulp=116558 | abs=0.000328 rel=9.67e-05 ulp=116551 | 0.000648 | abs=0.000576 rel=0.00017 ulp=283878 |
| 15 | C_nm[tf32] compact | abs=0.000307 rel=9.05e-05 ulp=125822 | abs=0.000307 rel=9.05e-05 ulp=125815 | 0.000636 | abs=0.000576 rel=0.00017 ulp=283878 |
| 16 | A_prod replay | abs=6.15e-07 rel=2e-07 ulp=16 | abs=2.98e-08 rel=9.67e-09 ulp=5 | 0.992 | abs=0.000956 rel=0.00031 ulp=336713 |
| 16 | B_legacyWY fp32-closed (materialized) | abs=4.92e-07 rel=1.6e-07 ulp=52 | — | — | — |
| 16 | B_fs[ieee] compact | abs=4.92e-07 rel=1.6e-07 ulp=51 | abs=7.15e-07 rel=2.32e-07 ulp=57 | 0.314 | abs=0.000956 rel=0.00031 ulp=336717 |
| 16 | C_nm[ieee] compact | abs=4.92e-07 rel=1.6e-07 ulp=56 | abs=7.15e-07 rel=2.32e-07 ulp=62 | 0.312 | abs=0.000956 rel=0.00031 ulp=336713 |
| 16 | B_fs[tf32] compact | abs=0.000497 rel=0.000161 ulp=156601 | abs=0.000497 rel=0.000161 ulp=156613 | 0.00044 | abs=0.001 rel=0.000416 ulp=352227 |
| 16 | C_nm[tf32] compact | abs=0.00048 rel=0.000156 ulp=134425 | abs=0.00048 rel=0.000156 ulp=134437 | 0.000432 | abs=0.001 rel=0.000416 ulp=331651 |
| 17 | A_prod replay | abs=4.85e-07 rel=1.51e-07 ulp=25 | abs=2.98e-08 rel=9.28e-09 ulp=2 | 0.993 | abs=0.000506 rel=0.000158 ulp=238220 |
| 17 | B_legacyWY fp32-closed (materialized) | abs=2.8e-07 rel=8.73e-08 ulp=14 | — | — | — |
| 17 | B_fs[ieee] compact | abs=4.74e-07 rel=1.48e-07 ulp=18 | abs=4.77e-07 rel=1.48e-07 ulp=14 | 0.326 | abs=0.000506 rel=0.000158 ulp=238232 |
| 17 | C_nm[ieee] compact | abs=4.74e-07 rel=1.48e-07 ulp=20 | abs=4.77e-07 rel=1.48e-07 ulp=18 | 0.325 | abs=0.000506 rel=0.000158 ulp=238232 |
| 17 | B_fs[tf32] compact | abs=0.000347 rel=0.000108 ulp=229847 | abs=0.000347 rel=0.000108 ulp=229822 | 0.006 | abs=0.000596 rel=0.000186 ulp=95978 |
| 17 | C_nm[tf32] compact | abs=0.000403 rel=0.000125 ulp=229847 | abs=0.000403 rel=0.000125 ulp=229822 | 0.006 | abs=0.000688 rel=0.000214 ulp=126022 |
| 18 | A_prod replay | abs=6.8e-07 rel=1.7e-07 ulp=9 | abs=5.96e-08 rel=1.49e-08 ulp=3 | 0.992 | abs=0.000856 rel=0.000214 ulp=94863 |
| 18 | B_legacyWY fp32-closed (materialized) | abs=6.35e-07 rel=1.58e-07 ulp=15 | — | — | — |
| 18 | B_fs[ieee] compact | abs=6.35e-07 rel=1.58e-07 ulp=19 | abs=4.77e-07 rel=1.19e-07 ulp=21 | 0.332 | abs=0.000856 rel=0.000214 ulp=94865 |
| 18 | C_nm[ieee] compact | abs=6.35e-07 rel=1.58e-07 ulp=18 | abs=4.77e-07 rel=1.19e-07 ulp=20 | 0.330 | abs=0.000856 rel=0.000214 ulp=94867 |
| 18 | B_fs[tf32] compact | abs=0.000905 rel=0.000226 ulp=178331 | abs=0.000905 rel=0.000226 ulp=178327 | 0.001 | abs=0.000984 rel=0.000246 ulp=143867 |
| 18 | C_nm[tf32] compact | abs=0.000905 rel=0.000226 ulp=177863 | abs=0.000905 rel=0.000226 ulp=177859 | 0.001 | abs=0.000984 rel=0.000246 ulp=143875 |
| 19 | A_prod replay | abs=6.25e-07 rel=1.61e-07 ulp=12 | abs=2.98e-08 rel=7.67e-09 ulp=3 | 0.993 | abs=0.000893 rel=0.00023 ulp=163245 |
| 19 | B_legacyWY fp32-closed (materialized) | abs=4.38e-07 rel=1.13e-07 ulp=18 | — | — | — |
| 19 | B_fs[ieee] compact | abs=5.34e-07 rel=1.37e-07 ulp=26 | abs=4.77e-07 rel=1.23e-07 ulp=31 | 0.325 | abs=0.000893 rel=0.00023 ulp=163267 |
| 19 | C_nm[ieee] compact | abs=5.34e-07 rel=1.37e-07 ulp=26 | abs=4.77e-07 rel=1.23e-07 ulp=31 | 0.324 | abs=0.000893 rel=0.00023 ulp=163267 |
| 19 | B_fs[tf32] compact | abs=0.000489 rel=0.000126 ulp=75189 | abs=0.000489 rel=0.000126 ulp=75183 | 0.002 | abs=0.001 rel=0.000284 ulp=189159 |
| 19 | C_nm[tf32] compact | abs=0.000477 rel=0.000123 ulp=103317 | abs=0.000477 rel=0.000123 ulp=103319 | 0.002 | abs=0.001 rel=0.000284 ulp=235527 |
| 20 | A_prod replay | abs=5.99e-07 rel=1.53e-07 ulp=16 | abs=5.96e-08 rel=1.52e-08 ulp=4 | 0.993 | abs=0.002 rel=0.000586 ulp=319326 |
| 20 | B_legacyWY fp32-closed (materialized) | abs=5.57e-07 rel=1.42e-07 ulp=62 | — | — | — |
| 20 | B_fs[ieee] compact | abs=5.99e-07 rel=1.53e-07 ulp=64 | abs=7.15e-07 rel=1.83e-07 ulp=67 | 0.322 | abs=0.002 rel=0.000586 ulp=319298 |
| 20 | C_nm[ieee] compact | abs=5.99e-07 rel=1.53e-07 ulp=66 | abs=7.15e-07 rel=1.83e-07 ulp=69 | 0.320 | abs=0.002 rel=0.000586 ulp=319298 |
| 20 | B_fs[tf32] compact | abs=0.002 rel=0.0004 ulp=151976 | abs=0.002 rel=0.0004 ulp=151984 | 0.001 | abs=0.002 rel=0.000565 ulp=354449 |
| 20 | C_nm[tf32] compact | abs=0.002 rel=0.0004 ulp=249128 | abs=0.002 rel=0.0004 ulp=249136 | 0.001 | abs=0.003 rel=0.000713 ulp=453265 |
| 21 | A_prod replay | abs=6.14e-07 rel=1.75e-07 ulp=78 | abs=2.98e-08 rel=8.49e-09 ulp=6 | 0.992 | abs=0.000706 rel=0.000201 ulp=175130 |
| 21 | B_legacyWY fp32-closed (materialized) | abs=3.91e-07 rel=1.11e-07 ulp=16 | — | — | — |
| 21 | B_fs[ieee] compact | abs=4.8e-07 rel=1.37e-07 ulp=33 | abs=7.15e-07 rel=2.04e-07 ulp=45 | 0.327 | abs=0.000706 rel=0.000201 ulp=175126 |
| 21 | C_nm[ieee] compact | abs=4.8e-07 rel=1.37e-07 ulp=32 | abs=7.15e-07 rel=2.04e-07 ulp=46 | 0.325 | abs=0.000706 rel=0.000201 ulp=175126 |
| 21 | B_fs[tf32] compact | abs=0.000587 rel=0.000167 ulp=200516 | abs=0.000587 rel=0.000167 ulp=200438 | 0.000587 | abs=0.000946 rel=0.00027 ulp=230744 |
| 21 | C_nm[tf32] compact | abs=0.000748 rel=0.000213 ulp=216484 | abs=0.000748 rel=0.000213 ulp=216406 | 0.00058 | abs=0.000817 rel=0.000233 ulp=241364 |
| 22 | A_prod replay | abs=5.27e-07 rel=1.31e-07 ulp=10 | abs=5.96e-08 rel=1.48e-08 ulp=2 | 0.992 | abs=0.000625 rel=0.000156 ulp=108400 |
| 22 | B_legacyWY fp32-closed (materialized) | abs=4.81e-07 rel=1.2e-07 ulp=17 | — | — | — |
| 22 | B_fs[ieee] compact | abs=4.81e-07 rel=1.2e-07 ulp=14 | abs=4.77e-07 rel=1.19e-07 ulp=14 | 0.328 | abs=0.000625 rel=0.000156 ulp=108386 |
| 22 | C_nm[ieee] compact | abs=4.81e-07 rel=1.2e-07 ulp=17 | abs=4.77e-07 rel=1.19e-07 ulp=17 | 0.326 | abs=0.000625 rel=0.000156 ulp=108388 |
| 22 | B_fs[tf32] compact | abs=0.00036 rel=8.97e-05 ulp=121242 | abs=0.00036 rel=8.97e-05 ulp=121252 | 0.001 | abs=0.000714 rel=0.000178 ulp=123650 |
| 22 | C_nm[tf32] compact | abs=0.000344 rel=8.55e-05 ulp=121242 | abs=0.000344 rel=8.55e-05 ulp=121252 | 0.001 | abs=0.000714 rel=0.000178 ulp=122870 |
| 23 | A_prod replay | abs=8.82e-07 rel=2.84e-07 ulp=13 | abs=1.49e-08 rel=4.8e-09 ulp=4 | 0.994 | abs=0.000472 rel=0.000152 ulp=85153 |
| 23 | B_legacyWY fp32-closed (materialized) | abs=3.65e-07 rel=1.18e-07 ulp=8 | — | — | — |
| 23 | B_fs[ieee] compact | abs=4.54e-07 rel=1.46e-07 ulp=9 | abs=9.54e-07 rel=3.07e-07 ulp=14 | 0.330 | abs=0.000472 rel=0.000152 ulp=85145 |
| 23 | C_nm[ieee] compact | abs=4.54e-07 rel=1.46e-07 ulp=9 | abs=9.54e-07 rel=3.07e-07 ulp=20 | 0.328 | abs=0.000472 rel=0.000152 ulp=85142 |
| 23 | B_fs[tf32] compact | abs=0.00033 rel=0.000106 ulp=40607 | abs=0.00033 rel=0.000106 ulp=40611 | 0.034 | abs=0.000487 rel=0.000157 ulp=118314 |
| 23 | C_nm[tf32] compact | abs=0.000347 rel=0.000112 ulp=42445 | abs=0.000347 rel=0.000112 ulp=42449 | 0.034 | abs=0.000451 rel=0.000146 ulp=124925 |
| 24 | A_prod replay | abs=9.13e-07 rel=2.2e-07 ulp=13 | abs=5.96e-08 rel=1.44e-08 ulp=4 | 0.993 | abs=0.000951 rel=0.000229 ulp=222305 |
| 24 | B_legacyWY fp32-closed (materialized) | abs=5.19e-07 rel=1.25e-07 ulp=11 | — | — | — |
| 24 | B_fs[ieee] compact | abs=5.19e-07 rel=1.25e-07 ulp=23 | abs=9.54e-07 rel=2.3e-07 ulp=24 | 0.319 | abs=0.000951 rel=0.000229 ulp=222321 |
| 24 | C_nm[ieee] compact | abs=5.19e-07 rel=1.25e-07 ulp=24 | abs=9.54e-07 rel=2.3e-07 ulp=25 | 0.316 | abs=0.000951 rel=0.000229 ulp=222317 |
| 24 | B_fs[tf32] compact | abs=0.000625 rel=0.000151 ulp=72401 | abs=0.000625 rel=0.000151 ulp=72402 | 0.002 | abs=0.001 rel=0.000261 ulp=232093 |
| 24 | C_nm[tf32] compact | abs=0.000606 rel=0.000146 ulp=71112 | abs=0.000606 rel=0.000146 ulp=71104 | 0.002 | abs=0.001 rel=0.000261 ulp=232093 |
| 25 | A_prod replay | abs=1.08e-06 rel=2.17e-07 ulp=23 | abs=5.96e-08 rel=1.2e-08 ulp=6 | 0.993 | abs=0.001 rel=0.000297 ulp=153793 |
| 25 | B_legacyWY fp32-closed (materialized) | abs=7.81e-07 rel=1.57e-07 ulp=22 | — | — | — |
| 25 | B_fs[ieee] compact | abs=7.81e-07 rel=1.57e-07 ulp=20 | abs=9.54e-07 rel=1.92e-07 ulp=24 | 0.330 | abs=0.001 rel=0.000297 ulp=153789 |
| 25 | C_nm[ieee] compact | abs=7.81e-07 rel=1.57e-07 ulp=22 | abs=9.54e-07 rel=1.92e-07 ulp=22 | 0.329 | abs=0.001 rel=0.000297 ulp=153788 |
| 25 | B_fs[tf32] compact | abs=0.000684 rel=0.000138 ulp=92542 | abs=0.000684 rel=0.000138 ulp=92519 | 0.002 | abs=0.002 rel=0.000332 ulp=180353 |
| 25 | C_nm[tf32] compact | abs=0.000657 rel=0.000132 ulp=113657 | abs=0.000657 rel=0.000132 ulp=113666 | 0.002 | abs=0.002 rel=0.000328 ulp=153857 |
| 26 | A_prod replay | abs=8.36e-07 rel=2.02e-07 ulp=11 | abs=5.96e-08 rel=1.44e-08 ulp=5 | 0.993 | abs=0.000937 rel=0.000226 ulp=134237 |
| 26 | B_legacyWY fp32-closed (materialized) | abs=4.46e-07 rel=1.08e-07 ulp=14 | — | — | — |
| 26 | B_fs[ieee] compact | abs=4.67e-07 rel=1.13e-07 ulp=27 | abs=9.54e-07 rel=2.3e-07 ulp=29 | 0.311 | abs=0.000937 rel=0.000226 ulp=134231 |
| 26 | C_nm[ieee] compact | abs=4.67e-07 rel=1.13e-07 ulp=27 | abs=9.54e-07 rel=2.3e-07 ulp=30 | 0.309 | abs=0.000937 rel=0.000226 ulp=134231 |
| 26 | B_fs[tf32] compact | abs=0.000455 rel=0.00011 ulp=137294 | abs=0.000455 rel=0.00011 ulp=137298 | 0.000882 | abs=0.001 rel=0.000254 ulp=194915 |
| 26 | C_nm[tf32] compact | abs=0.000446 rel=0.000108 ulp=134495 | abs=0.000446 rel=0.000108 ulp=134499 | 0.000878 | abs=0.001 rel=0.000252 ulp=194915 |
| 27 | A_prod replay | abs=6.11e-07 rel=1.44e-07 ulp=14 | abs=5.96e-08 rel=1.41e-08 ulp=5 | 0.991 | abs=0.002 rel=0.000478 ulp=206088 |
| 27 | B_legacyWY fp32-closed (materialized) | abs=5.27e-07 rel=1.25e-07 ulp=13 | — | — | — |
| 27 | B_fs[ieee] compact | abs=7e-07 rel=1.66e-07 ulp=22 | abs=7.15e-07 rel=1.69e-07 ulp=26 | 0.326 | abs=0.002 rel=0.000478 ulp=206096 |
| 27 | C_nm[ieee] compact | abs=7e-07 rel=1.66e-07 ulp=22 | abs=7.15e-07 rel=1.69e-07 ulp=26 | 0.325 | abs=0.002 rel=0.000478 ulp=206104 |
| 27 | B_fs[tf32] compact | abs=0.000669 rel=0.000158 ulp=158403 | abs=0.000669 rel=0.000158 ulp=158401 | 0.002 | abs=0.002 rel=0.00053 ulp=292806 |
| 27 | C_nm[tf32] compact | abs=0.000669 rel=0.000158 ulp=158403 | abs=0.000669 rel=0.000158 ulp=158401 | 0.002 | abs=0.002 rel=0.00053 ulp=292806 |
| 28 | A_prod replay | abs=1.3e-06 rel=3.1e-07 ulp=19 | abs=5.96e-08 rel=1.42e-08 ulp=6 | 0.992 | abs=0.000989 rel=0.000235 ulp=171367 |
| 28 | B_legacyWY fp32-closed (materialized) | abs=7.24e-07 rel=1.72e-07 ulp=13 | — | — | — |
| 28 | B_fs[ieee] compact | abs=7.24e-07 rel=1.72e-07 ulp=17 | abs=9.54e-07 rel=2.27e-07 ulp=28 | 0.334 | abs=0.000989 rel=0.000235 ulp=171367 |
| 28 | C_nm[ieee] compact | abs=7.24e-07 rel=1.72e-07 ulp=18 | abs=9.54e-07 rel=2.27e-07 ulp=30 | 0.331 | abs=0.000989 rel=0.000235 ulp=171362 |
| 28 | B_fs[tf32] compact | abs=0.000472 rel=0.000112 ulp=116849 | abs=0.000472 rel=0.000112 ulp=116846 | 0.001 | abs=0.001 rel=0.000318 ulp=239984 |
| 28 | C_nm[tf32] compact | abs=0.000472 rel=0.000112 ulp=134787 | abs=0.000472 rel=0.000112 ulp=134781 | 0.001 | abs=0.001 rel=0.000318 ulp=306148 |
| 29 | A_prod replay | abs=1.34e-06 rel=2.67e-07 ulp=19 | abs=1.19e-07 rel=2.38e-08 ulp=5 | 0.992 | abs=0.001 rel=0.000206 ulp=261767 |
| 29 | B_legacyWY fp32-closed (materialized) | abs=5.71e-07 rel=1.14e-07 ulp=18 | — | — | — |
| 29 | B_fs[ieee] compact | abs=6.19e-07 rel=1.23e-07 ulp=21 | abs=1.91e-06 rel=3.8e-07 ulp=22 | 0.319 | abs=0.001 rel=0.000206 ulp=261759 |
| 29 | C_nm[ieee] compact | abs=6.19e-07 rel=1.23e-07 ulp=19 | abs=1.91e-06 rel=3.8e-07 ulp=20 | 0.316 | abs=0.001 rel=0.000206 ulp=261751 |
| 29 | B_fs[tf32] compact | abs=0.000906 rel=0.000181 ulp=94915 | abs=0.000906 rel=0.000181 ulp=94914 | 0.001 | abs=0.000985 rel=0.000196 ulp=220755 |
| 29 | C_nm[tf32] compact | abs=0.000906 rel=0.000181 ulp=106854 | abs=0.000906 rel=0.000181 ulp=106850 | 0.001 | abs=0.000985 rel=0.000196 ulp=197538 |
| 30 | A_prod replay | abs=7.78e-07 rel=1.9e-07 ulp=7 | abs=5.96e-08 rel=1.46e-08 ulp=4 | 0.993 | abs=0.000628 rel=0.000153 ulp=159783 |
| 30 | B_legacyWY fp32-closed (materialized) | abs=7.05e-07 rel=1.72e-07 ulp=17 | — | — | — |
| 30 | B_fs[ieee] compact | abs=7.05e-07 rel=1.72e-07 ulp=17 | abs=9.54e-07 rel=2.33e-07 ulp=15 | 0.325 | abs=0.000628 rel=0.000153 ulp=159785 |
| 30 | C_nm[ieee] compact | abs=7.05e-07 rel=1.72e-07 ulp=22 | abs=9.54e-07 rel=2.33e-07 ulp=17 | 0.323 | abs=0.000628 rel=0.000153 ulp=159785 |
| 30 | B_fs[tf32] compact | abs=0.000362 rel=8.85e-05 ulp=49624 | abs=0.000362 rel=8.85e-05 ulp=49621 | 0.003 | abs=0.000655 rel=0.00016 ulp=177810 |
| 30 | C_nm[tf32] compact | abs=0.000364 rel=8.89e-05 ulp=81005 | abs=0.000364 rel=8.89e-05 ulp=81012 | 0.003 | abs=0.000658 rel=0.000161 ulp=171522 |
| 31 | A_prod replay | abs=1.03e-06 rel=2.48e-07 ulp=11 | abs=2.98e-08 rel=7.13e-09 ulp=4 | 0.993 | abs=0.000987 rel=0.000236 ulp=133061 |
| 31 | B_legacyWY fp32-closed (materialized) | abs=7.29e-07 rel=1.75e-07 ulp=16 | — | — | — |
| 31 | B_fs[ieee] compact | abs=7.29e-07 rel=1.75e-07 ulp=16 | abs=1.43e-06 rel=3.42e-07 ulp=16 | 0.344 | abs=0.000987 rel=0.000236 ulp=133068 |
| 31 | C_nm[ieee] compact | abs=7.29e-07 rel=1.75e-07 ulp=18 | abs=1.43e-06 rel=3.42e-07 ulp=16 | 0.341 | abs=0.000987 rel=0.000236 ulp=133068 |
| 31 | B_fs[tf32] compact | abs=0.000548 rel=0.000131 ulp=65991 | abs=0.000548 rel=0.000131 ulp=65990 | 0.001 | abs=0.001 rel=0.000273 ulp=117418 |
| 31 | C_nm[tf32] compact | abs=0.000548 rel=0.000131 ulp=64623 | abs=0.000548 rel=0.000131 ulp=64622 | 0.001 | abs=0.001 rel=0.000273 ulp=131482 |

Replay repeat-launch bitwise check (immutable h0 row): id 1: yes; id 2: yes; id 3: yes; id 4: yes; id 5: yes; id 6: yes; id 7: yes; id 8: yes; id 9: yes; id 10: yes; id 11: yes; id 12: yes; id 13: yes; id 14: yes; id 15: yes; id 16: yes; id 17: yes; id 18: yes; id 19: yes; id 20: yes; id 21: yes; id 22: yes; id 23: yes; id 24: yes; id 25: yes; id 26: yes; id 27: yes; id 28: yes; id 29: yes; id 30: yes; id 31: yes

## Committed-state max_abs vs oracle by accepted depth (finite rows only; any non-finite comparison in a row => n/a)


id 1 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.1e-07 | 1.41e-07 | 1.41e-07 | 1.41e-07 | 2.1e-07 | 0.000503 | 0.000395 |
| 1 | 2.84e-07 | 3.96e-07 | 3.96e-07 | 3.96e-07 | 2.84e-07 | 0.000526 | 0.000479 |
| 2 | 4.01e-07 | 4.01e-07 | 4.01e-07 | 4.01e-07 | 4.01e-07 | 0.000511 | 0.000581 |
| 3 | 5.66e-07 | 3.9e-07 | 3.9e-07 | 3.9e-07 | 5.66e-07 | 0.000774 | 0.000553 |
| 4 | 4.08e-07 | 3.41e-07 | 3.41e-07 | 3.41e-07 | 4.08e-07 | 0.000728 | 0.000567 |
| 5 | 4.89e-07 | 4.46e-07 | 4.46e-07 | 4.46e-07 | 4.89e-07 | 0.000842 | 0.000598 |

id 2 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.38e-07 | 2.38e-07 | 2.38e-07 | 2.38e-07 | 2.38e-07 | 0.000528 | 0.000343 |
| 1 | 2.32e-07 | 1.78e-07 | 1.78e-07 | 1.78e-07 | 2.32e-07 | 0.000476 | 0.000573 |
| 2 | 3.38e-07 | 2.87e-07 | 2.87e-07 | 2.87e-07 | 3.38e-07 | 0.000432 | 0.000408 |
| 3 | 4.02e-07 | 4.02e-07 | 4.02e-07 | 4.02e-07 | 4.02e-07 | 0.000388 | 0.000405 |
| 4 | 4.95e-07 | 2.69e-07 | 2.69e-07 | 2.69e-07 | 4.95e-07 | 0.000572 | 0.000408 |
| 5 | 7.13e-07 | 2.14e-07 | 2.14e-07 | 2.14e-07 | 7.13e-07 | 0.000569 | 0.000401 |

id 3 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.63e-07 | 1.63e-07 | 1.63e-07 | 1.63e-07 | 1.63e-07 | 0.000508 | 0.000529 |
| 1 | 2.13e-07 | 4.11e-07 | 4.11e-07 | 4.11e-07 | 2.13e-07 | 0.002 | 0.002 |
| 2 | 2.06e-07 | 2.86e-07 | 2.71e-07 | 2.71e-07 | 2.06e-07 | 0.002 | 0.001 |
| 3 | 2.77e-07 | 2.65e-07 | 2.65e-07 | 2.65e-07 | 2.77e-07 | 0.002 | 0.001 |
| 4 | 3.95e-07 | 2.54e-07 | 2.54e-07 | 2.27e-07 | 3.95e-07 | 0.001 | 0.001 |
| 5 | 4.71e-07 | 3.62e-07 | 3.62e-07 | 3.62e-07 | 4.71e-07 | 0.002 | 0.001 |

id 4 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.03e-07 | 3.03e-07 | 3.03e-07 | 3.03e-07 | 3.03e-07 | 0.000675 | 0.00042 |
| 1 | 4.12e-07 | 2.96e-07 | 2.96e-07 | 2.96e-07 | 4.12e-07 | 0.000415 | 0.000395 |
| 2 | 6.19e-07 | 2.73e-07 | 2.73e-07 | 2.73e-07 | 6.19e-07 | 0.000497 | 0.000567 |
| 3 | 8.36e-07 | 2.83e-07 | 2.83e-07 | 2.83e-07 | 8.36e-07 | 0.00079 | 0.000657 |
| 4 | 8.79e-07 | 4.26e-07 | 4.26e-07 | 4.26e-07 | 8.79e-07 | 0.000449 | 0.000568 |
| 5 | 9.93e-07 | 4.02e-07 | 4.02e-07 | 4.02e-07 | 9.93e-07 | 0.000452 | 0.000478 |

id 5 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.98e-07 | 1.98e-07 | 1.98e-07 | 1.98e-07 | 1.98e-07 | 0.000322 | 0.000221 |
| 1 | 2.5e-07 | 2.5e-07 | 2.5e-07 | 2.5e-07 | 2.5e-07 | 0.000604 | 0.000359 |
| 2 | 3.72e-07 | 4.27e-07 | 4.27e-07 | 4.27e-07 | 3.72e-07 | 0.001 | 0.000448 |
| 3 | 4.22e-07 | 3.22e-07 | 3.22e-07 | 3.22e-07 | 4.22e-07 | 0.001 | 0.000533 |
| 4 | 4.87e-07 | 4.87e-07 | 4.87e-07 | 4.87e-07 | 4.87e-07 | 0.001 | 0.000525 |
| 5 | 5.07e-07 | 2.68e-07 | 2.68e-07 | 2.6e-07 | 5.07e-07 | 0.000579 | 0.000534 |

id 6 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.09e-07 | 2.09e-07 | 2.09e-07 | 2.09e-07 | 2.09e-07 | 0.000799 | 0.000718 |
| 1 | 2.71e-07 | 2.13e-07 | 2.13e-07 | 2.13e-07 | 2.71e-07 | 0.000795 | 0.000704 |
| 2 | 2.99e-07 | 3.27e-07 | 3.27e-07 | 3.27e-07 | 2.99e-07 | 0.000801 | 0.000751 |
| 3 | 4.84e-07 | 4.84e-07 | 4.84e-07 | 4.84e-07 | 4.84e-07 | 0.003 | 0.002 |
| 4 | 6.27e-07 | 5.07e-07 | 5.07e-07 | 5.07e-07 | 6.27e-07 | 0.002 | 0.002 |
| 5 | 6.47e-07 | 3.93e-07 | 3.93e-07 | 3.93e-07 | 6.47e-07 | 0.002 | 0.002 |

id 7 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.59e-07 | 1.59e-07 | 1.59e-07 | 1.59e-07 | 1.59e-07 | 0.000862 | 0.000443 |
| 1 | 2.96e-07 | 1.39e-07 | 1.39e-07 | 1.81e-07 | 2.96e-07 | 0.000567 | 0.000543 |
| 2 | 3.53e-07 | 2.27e-07 | 2.27e-07 | 2.27e-07 | 3.53e-07 | 0.000903 | 0.0006 |
| 3 | 4.34e-07 | 2.77e-07 | 2.77e-07 | 2.77e-07 | 4.34e-07 | 0.000586 | 0.000583 |
| 4 | 5.48e-07 | 4.09e-07 | 4.09e-07 | 4.09e-07 | 5.48e-07 | 0.000638 | 0.000569 |
| 5 | 6.83e-07 | 3.94e-07 | 3.94e-07 | 3.94e-07 | 6.83e-07 | 0.001 | 0.000679 |

id 8 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.28e-07 | 2.28e-07 | 2.28e-07 | 2.28e-07 | 2.28e-07 | 0.000575 | 0.000697 |
| 1 | 1.65e-07 | 1.79e-07 | 1.79e-07 | 1.79e-07 | 1.65e-07 | 0.000552 | 0.000629 |
| 2 | 3.35e-07 | 1.62e-07 | 1.62e-07 | 1.82e-07 | 3.35e-07 | 0.000795 | 0.000636 |
| 3 | 4.12e-07 | 2.81e-07 | 2.81e-07 | 3.03e-07 | 4.12e-07 | 0.001 | 0.000608 |
| 4 | 4.47e-07 | 1.83e-07 | 2.93e-07 | 1.84e-07 | 4.47e-07 | 0.001 | 0.000587 |
| 5 | 4.49e-07 | 2.66e-07 | 3.5e-07 | 3.9e-07 | 4.49e-07 | 0.000899 | 0.000609 |

id 9 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.58e-07 | 2.58e-07 | 2.58e-07 | 2.58e-07 | 2.58e-07 | 0.000435 | 0.000309 |
| 1 | 3.22e-07 | 1.73e-07 | 1.73e-07 | 1.73e-07 | 3.22e-07 | 0.001 | 0.000415 |
| 2 | 4.31e-07 | 1.73e-07 | 1.73e-07 | 1.73e-07 | 4.31e-07 | 0.000682 | 0.000516 |
| 3 | 5.18e-07 | 3.58e-07 | 3.58e-07 | 3.58e-07 | 5.18e-07 | 0.001 | 0.000692 |
| 4 | 4.91e-07 | 2.92e-07 | 2.92e-07 | 2.91e-07 | 4.91e-07 | 0.001 | 0.000616 |
| 5 | 4.56e-07 | 2.39e-07 | 2.39e-07 | 2.39e-07 | 4.56e-07 | 0.001 | 0.000501 |

id 10 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.05e-07 | 1.69e-07 | 1.69e-07 | 1.05e-07 | 1.05e-07 | 0.000344 | 0.000438 |
| 1 | 1.68e-07 | 3.33e-07 | 3.33e-07 | 3.33e-07 | 1.68e-07 | 0.000284 | 0.000397 |
| 2 | 2.58e-07 | 2.68e-07 | 2.68e-07 | 2.68e-07 | 2.58e-07 | 0.000299 | 0.000393 |
| 3 | 2.81e-07 | 4.05e-07 | 4.05e-07 | 4.05e-07 | 2.81e-07 | 0.000385 | 0.000388 |
| 4 | 3.68e-07 | 3.27e-07 | 3.27e-07 | 3.27e-07 | 3.68e-07 | 0.0007 | 0.000368 |
| 5 | 3.97e-07 | 3.11e-07 | 3.11e-07 | 2.64e-07 | 3.97e-07 | 0.000697 | 0.000362 |

id 11 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.26e-07 | 2.26e-07 | 2.26e-07 | 2.26e-07 | 2.26e-07 | 0.000234 | 0.000181 |
| 1 | 1.9e-07 | 2.67e-07 | 2.67e-07 | 2.67e-07 | 1.9e-07 | 0.000337 | 0.000492 |
| 2 | 2.31e-07 | 1.91e-07 | 1.91e-07 | 1.91e-07 | 2.31e-07 | 0.000337 | 0.000519 |
| 3 | 3.51e-07 | 2.1e-07 | 2.1e-07 | 2.1e-07 | 3.51e-07 | 0.000677 | 0.000662 |
| 4 | 5.21e-07 | 2.41e-07 | 2.41e-07 | 2.41e-07 | 5.21e-07 | 0.000667 | 0.000612 |
| 5 | 4.88e-07 | 3.54e-07 | 3.54e-07 | 3.54e-07 | 4.88e-07 | 0.000551 | 0.000579 |

id 12 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.69e-07 | 2.69e-07 | 2.69e-07 | 2.69e-07 | 2.69e-07 | 0.000246 | 0.000186 |
| 1 | 2.99e-07 | 1.73e-07 | 1.73e-07 | 1.73e-07 | 2.99e-07 | 0.000639 | 0.00048 |
| 2 | 4.64e-07 | 2.99e-07 | 2.99e-07 | 2.72e-07 | 4.64e-07 | 0.000623 | 0.000666 |
| 3 | 4.25e-07 | 3.01e-07 | 3.01e-07 | 3.01e-07 | 4.25e-07 | 0.003 | 0.001 |
| 4 | 4.4e-07 | 3.68e-07 | 3.68e-07 | 3.68e-07 | 4.96e-07 | 0.003 | 0.001 |
| 5 | 4.7e-07 | 4.15e-07 | 4.15e-07 | 4.15e-07 | 5.11e-07 | 0.003 | 0.001 |

id 13 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.25e-07 | 1.25e-07 | 1.25e-07 | 1.25e-07 | 1.25e-07 | 0.000359 | 0.000204 |
| 1 | 2.23e-07 | 2.32e-07 | 2.32e-07 | 2.32e-07 | 2.23e-07 | 0.000433 | 0.000313 |
| 2 | 3.74e-07 | 2.95e-07 | 2.95e-07 | 2.95e-07 | 3.74e-07 | 0.000462 | 0.000287 |
| 3 | 4.67e-07 | 3.72e-07 | 3.72e-07 | 3.72e-07 | 4.67e-07 | 0.000559 | 0.00038 |
| 4 | 4.69e-07 | 3.76e-07 | 3.76e-07 | 3.76e-07 | 4.69e-07 | 0.000354 | 0.000374 |
| 5 | 7.43e-07 | 4.28e-07 | 4.28e-07 | 4.28e-07 | 7.43e-07 | 0.000328 | 0.00036 |

id 14 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.25e-07 | 1.25e-07 | 1.25e-07 | 1.25e-07 | 1.25e-07 | 0.001 | 0.000638 |
| 1 | 3.06e-07 | 2.65e-07 | 2.65e-07 | 2.65e-07 | 3.06e-07 | 0.001 | 0.000529 |
| 2 | 3.95e-07 | 1.94e-07 | 1.94e-07 | 1.94e-07 | 3.95e-07 | 0.001 | 0.000445 |
| 3 | 5.02e-07 | 2.47e-07 | 2.47e-07 | 2.47e-07 | 5.02e-07 | 0.001 | 0.000572 |
| 4 | 5.39e-07 | 3.01e-07 | 3.01e-07 | 3.01e-07 | 5.39e-07 | 0.001 | 0.000445 |
| 5 | 5.49e-07 | 3.09e-07 | 3.09e-07 | 4.34e-07 | 5.49e-07 | 0.001 | 0.000446 |

id 15 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.56e-07 | 1.56e-07 | 1.56e-07 | 1.56e-07 | 1.56e-07 | 0.000303 | 0.000123 |
| 1 | 2.86e-07 | 3.15e-07 | 3.15e-07 | 3.15e-07 | 2.86e-07 | 0.000157 | 0.000214 |
| 2 | 3.03e-07 | 2.08e-07 | 2.08e-07 | 2.08e-07 | 3.03e-07 | 0.000516 | 0.000286 |
| 3 | 4.2e-07 | 3.76e-07 | 3.76e-07 | 4.31e-07 | 4.2e-07 | 0.000536 | 0.000303 |
| 4 | 5.39e-07 | 4.41e-07 | 4.41e-07 | 4.41e-07 | 5.39e-07 | 0.000537 | 0.000316 |
| 5 | 6.03e-07 | 4.13e-07 | 4.13e-07 | 4.13e-07 | 6.03e-07 | 0.000536 | 0.000328 |

id 16 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.38e-07 | 1.38e-07 | 1.38e-07 | 1.38e-07 | 1.38e-07 | 0.000619 | 0.000418 |
| 1 | 4.24e-07 | 2.37e-07 | 2.37e-07 | 2.37e-07 | 4.24e-07 | 0.000621 | 0.000295 |
| 2 | 3.71e-07 | 2.89e-07 | 2.89e-07 | 2.89e-07 | 3.71e-07 | 0.00094 | 0.000437 |
| 3 | 4.67e-07 | 2.86e-07 | 2.86e-07 | 2.86e-07 | 4.67e-07 | 0.000956 | 0.000455 |
| 4 | 4.92e-07 | 4.92e-07 | 4.92e-07 | 4.92e-07 | 4.92e-07 | 0.000935 | 0.000497 |
| 5 | 6.15e-07 | 4.36e-07 | 4.36e-07 | 4.36e-07 | 6.15e-07 | 0.000819 | 0.000484 |

id 17 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.39e-07 | 1.39e-07 | 1.39e-07 | 1.39e-07 | 1.39e-07 | 0.000286 | 0.000261 |
| 1 | 2.03e-07 | 2.62e-07 | 2.62e-07 | 2.62e-07 | 2.03e-07 | 0.00034 | 0.00029 |
| 2 | 2.61e-07 | 2.52e-07 | 2.52e-07 | 2.52e-07 | 2.61e-07 | 0.000319 | 0.000284 |
| 3 | 3.47e-07 | 3.57e-07 | 3.57e-07 | 3.57e-07 | 3.47e-07 | 0.000402 | 0.000299 |
| 4 | 4.85e-07 | 2.93e-07 | 2.93e-07 | 2.93e-07 | 4.85e-07 | 0.000344 | 0.0003 |
| 5 | 4.74e-07 | 4.74e-07 | 4.74e-07 | 4.74e-07 | 4.74e-07 | 0.000506 | 0.000347 |

id 18 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.89e-07 | 1.89e-07 | 1.89e-07 | 1.89e-07 | 1.89e-07 | 0.000782 | 0.000178 |
| 1 | 4.14e-07 | 4.14e-07 | 4.14e-07 | 4.14e-07 | 4.14e-07 | 0.000856 | 0.000286 |
| 2 | 6.21e-07 | 6.21e-07 | 6.21e-07 | 6.21e-07 | 6.21e-07 | 0.000711 | 0.000471 |
| 3 | 6.67e-07 | 4e-07 | 4e-07 | 4e-07 | 6.67e-07 | 0.000728 | 0.000337 |
| 4 | 6.8e-07 | 5.16e-07 | 5.16e-07 | 5.16e-07 | 6.8e-07 | 0.000707 | 0.000905 |
| 5 | 5.79e-07 | 6.35e-07 | 6.35e-07 | 6.35e-07 | 5.79e-07 | 0.000433 | 0.000664 |

id 19 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.59e-07 | 2.59e-07 | 2.59e-07 | 2.59e-07 | 2.59e-07 | 0.000461 | 0.000251 |
| 1 | 4.58e-07 | 2.19e-07 | 2.19e-07 | 2.19e-07 | 4.58e-07 | 0.000893 | 0.000339 |
| 2 | 6.22e-07 | 3.36e-07 | 3.36e-07 | 3.36e-07 | 6.22e-07 | 0.000376 | 0.000413 |
| 3 | 5.45e-07 | 3.6e-07 | 3.6e-07 | 3.6e-07 | 5.45e-07 | 0.000537 | 0.000402 |
| 4 | 6.25e-07 | 3.1e-07 | 3.1e-07 | 3.71e-07 | 6.25e-07 | 0.000312 | 0.000412 |
| 5 | 6.07e-07 | 5.34e-07 | 5.34e-07 | 5.34e-07 | 6.07e-07 | 0.000504 | 0.000489 |

id 20 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.22e-07 | 2.22e-07 | 2.22e-07 | 2.22e-07 | 2.22e-07 | 0.000512 | 0.000266 |
| 1 | 3.09e-07 | 3.17e-07 | 3.17e-07 | 3.17e-07 | 3.09e-07 | 0.002 | 0.002 |
| 2 | 3.75e-07 | 4.27e-07 | 4.27e-07 | 3.43e-07 | 3.75e-07 | 0.002 | 0.001 |
| 3 | 5.77e-07 | 2.9e-07 | 2.96e-07 | 2.67e-07 | 5.77e-07 | 0.002 | 0.000861 |
| 4 | 4.48e-07 | 3.3e-07 | 3.3e-07 | 3.3e-07 | 4.48e-07 | 0.002 | 0.000485 |
| 5 | 5.99e-07 | 5.99e-07 | 5.99e-07 | 5.99e-07 | 5.99e-07 | 0.001 | 0.000653 |

id 21 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.01e-07 | 2.01e-07 | 2.01e-07 | 2.01e-07 | 2.01e-07 | 0.000561 | 0.000251 |
| 1 | 3.29e-07 | 2.67e-07 | 2.67e-07 | 2.67e-07 | 3.29e-07 | 0.000671 | 0.000405 |
| 2 | 5.74e-07 | 2.64e-07 | 2.64e-07 | 2.64e-07 | 5.74e-07 | 0.000706 | 0.000407 |
| 3 | 4.05e-07 | 3.48e-07 | 3.48e-07 | 3.48e-07 | 4.05e-07 | 0.000382 | 0.000416 |
| 4 | 5.05e-07 | 4.06e-07 | 4.06e-07 | 4.06e-07 | 5.05e-07 | 0.000457 | 0.000587 |
| 5 | 6.14e-07 | 4.8e-07 | 4.8e-07 | 4.8e-07 | 6.14e-07 | 0.000646 | 0.000544 |

id 22 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.13e-07 | 3.13e-07 | 3.13e-07 | 3.13e-07 | 3.13e-07 | 0.000262 | 0.000344 |
| 1 | 2.82e-07 | 1.94e-07 | 1.94e-07 | 1.94e-07 | 2.82e-07 | 0.000236 | 0.00036 |
| 2 | 4.81e-07 | 2.71e-07 | 2.71e-07 | 2.71e-07 | 4.81e-07 | 0.000625 | 0.000287 |
| 3 | 5.03e-07 | 2.69e-07 | 2.69e-07 | 2.69e-07 | 5.03e-07 | 0.000468 | 0.000254 |
| 4 | 5.27e-07 | 3.7e-07 | 3.7e-07 | 3.7e-07 | 5.27e-07 | 0.000397 | 0.000262 |
| 5 | 4.81e-07 | 4.81e-07 | 4.81e-07 | 4.81e-07 | 4.81e-07 | 0.000403 | 0.000281 |

id 23 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.13e-07 | 3.13e-07 | 3.13e-07 | 3.13e-07 | 3.13e-07 | 0.000292 | 0.000146 |
| 1 | 3.64e-07 | 1.97e-07 | 1.97e-07 | 1.97e-07 | 3.64e-07 | 0.000371 | 0.000174 |
| 2 | 5.4e-07 | 3.02e-07 | 3.02e-07 | 3.02e-07 | 5.4e-07 | 0.000346 | 0.000222 |
| 3 | 7.08e-07 | 4.07e-07 | 4.07e-07 | 4.07e-07 | 7.08e-07 | 0.000472 | 0.000247 |
| 4 | 8.16e-07 | 4.54e-07 | 4.54e-07 | 4.54e-07 | 8.16e-07 | 0.000288 | 0.000277 |
| 5 | 8.82e-07 | 3.37e-07 | 3.37e-07 | 3.37e-07 | 8.82e-07 | 0.000372 | 0.00033 |

id 24 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.37e-07 | 3.37e-07 | 3.37e-07 | 3.37e-07 | 3.37e-07 | 0.000196 | 0.00045 |
| 1 | 3.48e-07 | 2.43e-07 | 2.43e-07 | 2.43e-07 | 3.48e-07 | 0.000447 | 0.000442 |
| 2 | 4.64e-07 | 2.57e-07 | 2.57e-07 | 2.57e-07 | 4.64e-07 | 0.000543 | 0.000427 |
| 3 | 6.48e-07 | 3.78e-07 | 3.78e-07 | 3.78e-07 | 6.48e-07 | 0.000452 | 0.00045 |
| 4 | 7.61e-07 | 5.19e-07 | 5.19e-07 | 5.19e-07 | 7.61e-07 | 0.000828 | 0.000441 |
| 5 | 9.13e-07 | 4.94e-07 | 4.94e-07 | 4.94e-07 | 9.13e-07 | 0.000951 | 0.000625 |

id 25 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.56e-07 | 3.56e-07 | 3.56e-07 | 3.56e-07 | 3.56e-07 | 0.001 | 0.000331 |
| 1 | 3.86e-07 | 2.59e-07 | 2.59e-07 | 2.59e-07 | 3.86e-07 | 0.000419 | 0.000218 |
| 2 | 5.47e-07 | 3.43e-07 | 3.43e-07 | 3.43e-07 | 5.47e-07 | 0.000826 | 0.000371 |
| 3 | 5.77e-07 | 5.77e-07 | 5.77e-07 | 5.77e-07 | 5.77e-07 | 0.00075 | 0.000423 |
| 4 | 1.08e-06 | 6.11e-07 | 6.11e-07 | 6.11e-07 | 1.08e-06 | 0.001 | 0.000443 |
| 5 | 5.38e-07 | 7.81e-07 | 7.81e-07 | 7.81e-07 | 5.38e-07 | 0.000834 | 0.000684 |

id 26 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.65e-07 | 2.65e-07 | 2.65e-07 | 2.65e-07 | 2.65e-07 | 0.000505 | 0.000198 |
| 1 | 4.18e-07 | 2.8e-07 | 2.8e-07 | 2.8e-07 | 4.18e-07 | 0.000347 | 0.000316 |
| 2 | 4.23e-07 | 3.59e-07 | 3.59e-07 | 3.59e-07 | 4.23e-07 | 0.000934 | 0.000455 |
| 3 | 6.28e-07 | 4.67e-07 | 4.67e-07 | 4.67e-07 | 6.28e-07 | 0.000935 | 0.000398 |
| 4 | 8.36e-07 | 4.05e-07 | 4.05e-07 | 4.05e-07 | 8.36e-07 | 0.000936 | 0.000349 |
| 5 | 7.96e-07 | 4.12e-07 | 4.12e-07 | 4.12e-07 | 7.96e-07 | 0.000937 | 0.000396 |

id 27 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.45e-07 | 2.45e-07 | 2.45e-07 | 2.45e-07 | 2.45e-07 | 0.000295 | 0.000206 |
| 1 | 3.85e-07 | 5.27e-07 | 5.27e-07 | 5.27e-07 | 3.85e-07 | 0.000539 | 0.000267 |
| 2 | 6.11e-07 | 4.24e-07 | 4.24e-07 | 4.24e-07 | 6.11e-07 | 0.002 | 0.000425 |
| 3 | 5.81e-07 | 7e-07 | 7e-07 | 7e-07 | 5.81e-07 | 0.001 | 0.000669 |
| 4 | 5.87e-07 | 3.74e-07 | 3.74e-07 | 3.74e-07 | 5.87e-07 | 0.000431 | 0.000446 |
| 5 | 5.34e-07 | 4.26e-07 | 4.26e-07 | 4.26e-07 | 5.34e-07 | 0.000364 | 0.000477 |

id 28 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.32e-07 | 2.32e-07 | 2.32e-07 | 2.32e-07 | 2.32e-07 | 0.00016 | 0.000226 |
| 1 | 4.96e-07 | 4.96e-07 | 4.96e-07 | 4.96e-07 | 4.96e-07 | 0.000434 | 0.000328 |
| 2 | 8.43e-07 | 3.66e-07 | 3.66e-07 | 3.66e-07 | 8.43e-07 | 0.000576 | 0.000353 |
| 3 | 7.87e-07 | 5.81e-07 | 5.81e-07 | 5.81e-07 | 7.87e-07 | 0.000628 | 0.000472 |
| 4 | 1.2e-06 | 7.24e-07 | 7.24e-07 | 7.24e-07 | 1.2e-06 | 0.000989 | 0.000443 |
| 5 | 1.3e-06 | 3.49e-07 | 3.49e-07 | 3.49e-07 | 1.3e-06 | 0.000561 | 0.000462 |

id 29 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 3.4e-07 | 3.4e-07 | 3.4e-07 | 3.4e-07 | 3.4e-07 | 0.000302 | 0.000202 |
| 1 | 2.5e-07 | 2.98e-07 | 2.98e-07 | 2.98e-07 | 2.5e-07 | 0.000422 | 0.000523 |
| 2 | 5.1e-07 | 4.35e-07 | 4.35e-07 | 4.43e-07 | 5.1e-07 | 0.000859 | 0.000584 |
| 3 | 7.14e-07 | 5.97e-07 | 5.97e-07 | 5.97e-07 | 7.14e-07 | 0.000866 | 0.000906 |
| 4 | 1.08e-06 | 4.62e-07 | 4.62e-07 | 4.62e-07 | 1.08e-06 | 0.001 | 0.000767 |
| 5 | 1.34e-06 | 6.19e-07 | 6.19e-07 | 6.19e-07 | 1.34e-06 | 0.001 | 0.000714 |

id 30 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 2.47e-07 | 2.47e-07 | 2.47e-07 | 2.47e-07 | 2.47e-07 | 0.000338 | 0.000151 |
| 1 | 3.48e-07 | 3.23e-07 | 3.23e-07 | 3.23e-07 | 3.48e-07 | 0.000394 | 0.0002 |
| 2 | 4.48e-07 | 2.68e-07 | 2.68e-07 | 2.68e-07 | 4.48e-07 | 0.000416 | 0.000227 |
| 3 | 5.68e-07 | 4.6e-07 | 4.6e-07 | 4.6e-07 | 5.68e-07 | 0.000528 | 0.000237 |
| 4 | 7.78e-07 | 6.98e-07 | 6.98e-07 | 6.98e-07 | 7.78e-07 | 0.000474 | 0.000362 |
| 5 | 7.34e-07 | 7.05e-07 | 7.05e-07 | 7.05e-07 | 7.34e-07 | 0.000628 | 0.000268 |

id 31 (layers.62.linear_attn):

| depth | A_prod_replay | B_fs[ieee]_own | C_nm[ieee]_own | B_fs[ieee]_oraclefactors | native_sg | native_pk | B_fs[tf32]_own |
|---|---|---|---|---|---|---|---|
| 0 | 1.4e-07 | 1.4e-07 | 1.4e-07 | 1.4e-07 | 1.4e-07 | 0.000413 | 0.000159 |
| 1 | 3.76e-07 | 2.96e-07 | 2.96e-07 | 2.96e-07 | 3.76e-07 | 0.000324 | 0.00017 |
| 2 | 4.56e-07 | 4.91e-07 | 4.91e-07 | 4.91e-07 | 4.56e-07 | 0.000373 | 0.000187 |
| 3 | 4.82e-07 | 4.17e-07 | 4.17e-07 | 4.17e-07 | 4.82e-07 | 0.00068 | 0.000324 |
| 4 | 7.23e-07 | 4.59e-07 | 4.59e-07 | 4.59e-07 | 7.23e-07 | 0.000475 | 0.000253 |
| 5 | 1.03e-06 | 7.29e-07 | 7.29e-07 | 7.29e-07 | 1.03e-06 | 0.000987 | 0.000548 |

## Controls

| id | Neumann d−1 terms (U err / out err) | B_fs N_PAD32 vs 16 (out exact / abs) | C_nm N_PAD32 vs 16 | B_fs sibling-reverse (out exact / U exact) | C_nm sibling-reverse | A_prod sibling-reverse out exact |
|---|---|---|---|---|---|---|
| 1 | 0.030 / 0.001 | 0.915 / 1.86e-09 | 0.933 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 2 | 0.009 / 0.000552 | 0.909 / 3.73e-09 | 0.922 / 2.79e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 3 | 0.026 / 0.001 | 0.906 / 1.86e-09 | 0.942 / 1.86e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 4 | 0.018 / 0.001 | 0.931 / 3.73e-09 | 0.947 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 5 | 0.018 / 0.001 | 0.912 / 3.73e-09 | 0.922 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 6 | 0.015 / 0.000622 | 0.900 / 7.45e-09 | 0.930 / 7.45e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 7 | 0.026 / 0.001 | 0.908 / 3.73e-09 | 0.931 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 8 | 0.067 / 0.004 | 0.910 / 3.73e-09 | 0.947 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 9 | 0.051 / 0.002 | 0.909 / 3.73e-09 | 0.931 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 10 | 0.041 / 0.003 | 0.923 / 1.86e-09 | 0.940 / 4.66e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 11 | 0.005 / 0.000304 | 0.934 / 5.59e-09 | 0.962 / 5.59e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 12 | 0.024 / 0.001 | 0.905 / 7.45e-09 | 0.935 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 13 | 0.033 / 0.002 | 0.942 / 3.73e-09 | 0.964 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 14 | 0.024 / 0.001 | 0.914 / 1.86e-09 | 0.938 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 15 | 0.009 / 0.000466 | 0.933 / 3.73e-09 | 0.948 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 16 | 0.012 / 0.000515 | 0.914 / 7.45e-09 | 0.939 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 17 | 0.071 / 0.003 | 0.930 / 3.73e-09 | 0.950 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 18 | 0.034 / 0.002 | 0.925 / 7.45e-09 | 0.945 / 7.45e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 19 | 0.036 / 0.002 | 0.915 / 1.86e-09 | 0.934 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 20 | 0.010 / 0.000583 | 0.928 / 3.73e-09 | 0.943 / 5.12e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 21 | 0.024 / 0.002 | 0.919 / 3.73e-09 | 0.935 / 7.45e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 22 | 0.028 / 0.001 | 0.934 / 1.86e-09 | 0.959 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 23 | 0.037 / 0.002 | 0.941 / 3.73e-09 | 0.958 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 24 | 0.036 / 0.002 | 0.916 / 3.73e-09 | 0.935 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 25 | 0.009 / 0.000397 | 0.913 / 3.73e-09 | 0.922 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 26 | 0.044 / 0.002 | 0.936 / 1.86e-09 | 0.966 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 27 | 0.017 / 0.00084 | 0.921 / 3.73e-09 | 0.947 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 28 | 0.022 / 0.001 | 0.944 / 1.86e-09 | 0.967 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 29 | 0.055 / 0.003 | 0.939 / 3.73e-09 | 0.963 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 30 | 0.009 / 0.000537 | 0.933 / 3.73e-09 | 0.952 / 1.86e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| 31 | 0.007 / 0.000416 | 0.934 / 3.73e-09 | 0.954 / 3.73e-09 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |

## Transient / scratch memory (bytes, analytic from shapes; per request, per layer)

| quantity | id 1 | id 2 | id 3 | id 4 | id 5 | id 6 | id 7 | id 8 | id 9 | id 10 | id 11 | id 12 | id 13 | id 14 | id 15 | id 16 | id 17 | id 18 | id 19 | id 20 | id 21 | id 22 | id 23 | id 24 | id 25 | id 26 | id 27 | id 28 | id 29 | id 30 | id 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_prod_scan_hbm_per_node_state_export | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_prod_scan_register_h_cache_per_program | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 | 131,072 |
| A_prod_replay_hbm_written_rows(depth+1) | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 | 18,874,368 |
| A_prod_activation_ring(k,v,a,b) | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 | 265,216 |
| A_legacy_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 |
| B_legacyWY_state_export_all_nodes | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 |
| BC_compact_factors_U+cumg | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 | 396,288 |
| BC_compact_commit_hbm_written_rows | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 |
| one_full_state_row | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 | 3,145,728 |

## Timing — NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) ['3103483'] seen; sampled observation, not a guarantee)

CUDA events, per-launch sync median / pipelined mean (µs), one layer, B1. Rows do DIFFERENT work (see WORK column); this is NOT an apples-to-apples kernel speedup table.

| kernel | WORK | id 1 | id 2 | id 3 | id 4 | id 5 | id 6 | id 7 | id 8 | id 9 | id 10 | id 11 | id 12 | id 13 | id 14 | id 15 | id 16 | id 17 | id 18 | id 19 | id 20 | id 21 | id 22 | id 23 | id 24 | id 25 | id 26 | id 27 | id 28 | id 29 | id 30 | id 31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_prod_scan_verify_out16 | bf16 out store; no per-node state export; preallocated | 88.7 / 63.6 | 89.8 / 65.8 ⚠probe-drift | 90.1 / 65.7 | 90.0 / 64.1 | 90.3 / 64.9 | 90.2 / 65.4 | 89.5 / 65.6 | 89.9 / 65.5 | 89.5 / 64.5 | 90.0 / 65.6 | 90.3 / 65.9 | 89.6 / 68.2 | 89.5 / 66.1 | 90.3 / 64.1 ⚠probe-drift | 90.2 / 65.3 | 89.5 / 65.1 | 90.3 / 66.0 | 90.2 / 65.9 | 90.3 / 64.7 | 90.2 / 66.0 | 90.1 / 64.9 | 90.2 / 65.9 | 90.3 / 66.0 | 90.3 / 65.9 | 90.3 / 67.1 | 90.3 / 66.2 | 90.3 / 64.8 | 90.3 / 65.8 ⚠probe-drift | 90.3 / 64.0 | 90.3 / 65.9 | 90.3 / 66.0 |
| A_prod_replay_commit_deepest | writes depth+1 fp32 state rows to the bank; preallocated | 102.8 / 86.5 | 102.8 / 87.8 ⚠probe-drift | 102.5 / 87.7 | 102.5 / 87.7 | 102.6 / 88.3 | 102.8 / 89.8 | 102.6 / 87.7 | 102.6 / 89.7 | 102.4 / 87.3 | 102.7 / 90.0 | 102.7 / 87.8 | 102.6 / 87.7 | 102.6 / 87.6 | 102.6 / 85.9 ⚠probe-drift | 103.5 / 86.6 | 102.6 / 88.4 | 102.8 / 90.0 | 102.7 / 89.7 | 102.6 / 86.0 | 102.8 / 88.2 | 102.8 / 89.8 | 102.5 / 87.5 | 102.6 / 85.3 | 102.6 / 90.1 | 103.2 / 90.2 | 102.6 / 87.7 | 102.8 / 89.9 | 102.6 / 89.8 ⚠probe-drift | 102.7 / 87.5 | 102.9 / 85.1 | 102.8 / 89.3 |
| A_prod_replay_commit_zero_accept | writes 1 fp32 state row (root); preallocated | 52.8 / 35.7 | 53.6 / 35.7 ⚠probe-drift | 53.5 / 37.6 | 53.6 / 35.7 | 53.5 / 35.7 | 53.6 / 35.7 | 53.6 / 35.7 | 53.6 / 35.8 | 53.6 / 35.8 | 53.5 / 35.9 | 53.6 / 35.9 | 53.6 / 38.0 | 53.6 / 35.9 | 53.6 / 36.9 ⚠probe-drift | 53.5 / 36.0 | 53.6 / 35.9 | 53.5 / 35.8 | 53.6 / 36.9 | 53.6 / 38.8 | 53.5 / 36.8 | 53.6 / 36.9 | 53.6 / 36.9 | 53.6 / 38.4 | 53.6 / 36.9 | 53.5 / 36.9 | 53.5 / 38.5 | 53.6 / 36.9 | 53.5 / 36.9 ⚠probe-drift | 53.5 / 37.9 | 53.6 / 36.9 | 53.6 / 36.8 |
| A_legacy_scan_with_state_export | bf16 out + FULL per-node fp32 state export (n_pad x VH x DV x DK x 4 B); preallocated | 237.4 / 227.1 | 243.0 / 231.2 ⚠probe-drift | 243.7 / 231.7 | 244.0 / 231.1 | 243.1 / 231.9 | 243.7 / 236.9 | 244.0 / 235.6 | 243.1 / 229.0 | 243.1 / 231.4 | 244.0 / 231.8 | 244.0 / 232.1 | 244.9 / 232.0 | 244.1 / 235.8 | 244.0 / 236.2 ⚠probe-drift | 244.8 / 232.3 | 243.7 / 233.0 | 244.0 / 229.1 | 243.9 / 234.6 | 243.9 / 232.4 | 245.0 / 229.2 | 244.0 / 232.7 | 243.9 / 232.2 | 243.9 / 232.7 | 243.9 / 227.9 | 244.0 / 232.4 | 244.8 / 232.3 | 243.9 / 232.4 | 244.0 / 232.2 ⚠probe-drift | 244.3 / 241.6 | 244.1 / 228.4 | 243.9 / 235.0 |
| B_legacyWY_fp32closed_with_state_export | bf16 out + FULL per-node fp32 state export; preallocated | 778.4 / 763.9 | 757.9 / 750.7 ⚠probe-drift | 757.7 / 748.6 | 758.1 / 746.5 | 757.2 / 746.4 | 758.2 / 743.2 | 759.0 / 746.6 | 759.1 / 748.9 | 758.1 / 745.0 | 758.8 / 745.6 | 758.4 / 745.8 | 758.1 / 744.9 | 759.1 / 743.9 | 759.0 / 745.8 ⚠probe-drift | 758.1 / 743.4 | 759.0 / 746.6 | 763.3 / 746.1 | 759.1 / 745.6 | 763.7 / 753.6 | 763.6 / 749.5 | 763.3 / 749.3 | 763.2 / 749.5 | 763.2 / 750.3 | 765.0 / 751.1 | 763.2 / 757.6 | 763.2 / 750.5 | 763.1 / 749.3 | 764.0 / 748.8 ⚠probe-drift | 763.3 / 750.2 | 763.3 / 755.4 | 763.0 / 750.9 |
| B_legacyWY_bf16bnd_with_state_export | bf16 out (+bf16 boundary taps) + FULL per-node fp32 state export; preallocated | 1674.3 / 1667.2 | 1654.8 / 1648.3 ⚠probe-drift | 1642.6 / 1640.6 | 1651.0 / 1640.2 | 1654.0 / 1651.2 | 1654.8 / 1650.1 | 1639.6 / 1644.5 | 1642.6 / 1635.4 | 1640.7 / 1634.8 | 1639.7 / 1639.8 | 1639.8 / 1647.4 | 1640.7 / 1650.9 | 1652.6 / 1641.7 | 1647.7 / 1639.3 ⚠probe-drift | 1641.7 / 1640.7 | 1640.6 / 1634.1 | 1640.7 / 1645.7 | 1638.7 / 1642.4 | 1639.8 / 1641.6 | 1667.3 / 1650.4 | 1647.8 / 1639.9 | 1645.8 / 1646.7 | 1649.0 / 1656.5 | 1649.8 / 1649.0 | 1653.1 / 1651.5 | 1649.8 / 1651.4 | 1648.0 / 1647.3 | 1647.9 / 1655.3 ⚠probe-drift | 1652.9 / 1654.2 | 1651.1 / 1648.1 | 1650.7 / 1649.2 |
| B_fs[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.0 / 9.5 | 18.8 / 9.9 ⚠probe-drift | 18.8 / 9.9 | 18.9 / 9.8 | 18.8 / 9.8 | 18.8 / 9.8 | 18.9 / 9.9 | 18.1 / 9.7 | 18.8 / 9.7 | 18.8 / 9.9 | 18.8 / 9.9 | 18.8 / 9.7 | 18.8 / 9.7 | 18.8 / 9.9 ⚠probe-drift | 18.8 / 9.8 | 18.2 / 9.6 | 18.8 / 9.7 | 18.8 / 10.0 | 18.9 / 9.8 | 18.8 / 9.8 | 18.9 / 9.7 | 18.0 / 9.6 | 18.9 / 9.8 | 18.9 / 9.9 | 18.9 / 9.9 | 18.8 / 9.9 | 18.8 / 9.8 | 18.8 / 9.9 ⚠probe-drift | 18.8 / 9.9 | 18.8 / 9.9 | 18.9 / 9.8 |
| B_fs[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 110.0 / 106.8 | 108.9 / 95.8 ⚠probe-drift | 108.8 / 99.0 | 109.9 / 96.4 | 110.0 / 96.3 | 108.9 / 95.0 | 109.0 / 96.1 | 109.7 / 96.1 | 110.0 / 96.3 | 109.1 / 96.3 | 109.0 / 95.9 | 109.0 / 96.1 | 110.0 / 96.3 | 110.0 / 99.5 ⚠probe-drift | 110.0 / 96.3 | 109.0 / 95.8 | 110.0 / 97.0 | 108.8 / 94.5 | 108.9 / 94.7 | 110.6 / 100.6 | 110.0 / 96.4 | 109.1 / 96.3 | 110.0 / 96.3 | 110.8 / 96.4 | 110.1 / 100.3 | 108.9 / 95.2 | 110.8 / 96.3 | 110.8 / 99.7 ⚠probe-drift | 110.9 / 96.7 | 110.8 / 100.2 | 110.8 / 96.3 |
| B_fs[ieee]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 104.8 / 92.0 | 105.9 / 92.0 ⚠probe-drift | 106.6 / 92.1 | 106.8 / 92.5 | 104.9 / 91.9 | 106.6 / 94.4 | 106.8 / 98.6 | 105.9 / 94.9 | 106.0 / 92.3 | 106.8 / 92.3 | 106.8 / 92.5 | 106.8 / 91.9 | 106.8 / 92.5 | 106.7 / 91.8 ⚠probe-drift | 105.9 / 92.2 | 106.0 / 91.3 | 106.0 / 92.3 | 106.7 / 92.2 | 106.7 / 92.1 | 106.0 / 92.5 | 106.8 / 92.7 | 105.9 / 92.2 | 106.8 / 92.5 | 106.8 / 93.0 | 106.8 / 92.8 | 106.8 / 93.0 | 106.8 / 92.8 | 105.9 / 92.3 ⚠probe-drift | 106.0 / 92.4 | 106.8 / 93.5 | 106.7 / 92.2 |
| B_fs[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.8 / 9.8 | 16.8 / 9.7 ⚠probe-drift | 16.7 / 9.8 | 16.9 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.7 / 9.7 | 16.8 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.8 / 9.7 | 16.8 / 9.8 | 16.7 / 9.8 ⚠probe-drift | 16.9 / 9.8 | 16.8 / 9.8 | 16.9 / 9.8 | 16.8 / 9.9 | 16.8 / 10.0 | 16.9 / 9.7 | 16.8 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.9 / 10.2 | 16.8 / 9.8 | 16.9 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 ⚠probe-drift | 16.9 / 9.9 | 16.9 / 9.8 | 16.9 / 9.9 |
| B_fs[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 41.4 / 28.7 | 41.3 / 28.7 ⚠probe-drift | 41.4 / 28.7 | 41.4 / 28.7 | 42.4 / 30.7 | 41.3 / 28.7 | 41.5 / 28.7 | 42.4 / 28.7 | 42.4 / 28.9 | 41.3 / 28.7 | 41.4 / 28.7 | 42.4 / 28.7 | 42.3 / 28.7 | 41.5 / 30.1 ⚠probe-drift | 42.4 / 28.7 | 42.4 / 30.8 | 42.4 / 28.7 | 42.3 / 28.8 | 41.6 / 30.8 | 42.5 / 28.7 | 42.3 / 28.7 | 42.4 / 28.7 | 41.6 / 30.6 | 41.6 / 28.7 | 42.3 / 28.7 | 42.3 / 28.7 | 41.6 / 28.7 | 42.4 / 28.7 ⚠probe-drift | 42.5 / 31.2 | 42.4 / 28.7 | 42.3 / 28.8 |
| B_fs[tf32]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 42.2 / 31.1 | 41.4 / 30.1 ⚠probe-drift | 42.9 / 30.0 | 42.2 / 28.7 | 42.3 / 29.0 | 41.4 / 28.8 | 42.2 / 31.1 | 42.4 / 29.1 | 42.4 / 28.7 | 42.8 / 29.8 | 41.6 / 29.0 | 42.1 / 31.3 | 42.3 / 32.7 | 42.9 / 28.7 ⚠probe-drift | 42.3 / 28.7 | 42.4 / 28.7 | 42.4 / 28.7 | 42.4 / 29.8 | 41.7 / 30.5 | 42.4 / 28.7 | 42.5 / 28.7 | 42.4 / 31.1 | 43.2 / 28.9 | 42.3 / 28.7 | 42.4 / 28.7 | 42.8 / 28.8 | 42.8 / 28.7 | 42.5 / 28.7 ⚠probe-drift | 42.4 / 30.6 | 42.7 / 28.7 | 43.1 / 28.7 |
| C_nm[ieee]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 18.6 / 9.7 | 18.8 / 9.7 ⚠probe-drift | 18.8 / 9.9 | 18.8 / 9.8 | 18.8 / 9.7 | 18.7 / 9.9 | 18.8 / 9.8 | 18.0 / 9.7 | 18.6 / 9.7 | 18.8 / 9.8 | 18.8 / 9.9 | 18.8 / 9.6 | 18.8 / 9.8 | 18.8 / 9.8 ⚠probe-drift | 18.8 / 9.7 | 18.0 / 9.7 | 18.3 / 9.7 | 18.8 / 9.7 | 18.8 / 9.8 | 18.5 / 9.9 | 18.7 / 10.0 | 18.8 / 9.9 | 18.8 / 9.9 | 18.7 / 9.8 | 18.8 / 9.7 | 18.8 / 9.9 | 18.8 / 9.9 | 18.6 / 10.9 ⚠probe-drift | 18.0 / 9.7 | 18.8 / 10.6 | 18.8 / 9.9 |
| C_nm[ieee]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 94.4 / 86.1 | 94.7 / 91.8 ⚠probe-drift | 94.6 / 86.6 | 94.6 / 86.4 | 94.7 / 81.7 | 94.7 / 83.2 | 94.8 / 81.8 | 94.7 / 86.4 | 94.7 / 86.8 | 95.3 / 86.7 | 94.7 / 86.8 | 94.7 / 86.6 | 95.4 / 86.7 | 94.6 / 86.6 ⚠probe-drift | 94.6 / 86.8 | 94.7 / 86.8 | 95.5 / 84.0 | 94.7 / 86.9 | 94.7 / 86.8 | 94.7 / 84.0 | 95.2 / 86.7 | 94.7 / 86.7 | 94.7 / 86.8 | 95.4 / 86.6 | 94.8 / 86.7 | 94.7 / 85.9 | 94.7 / 86.7 | 94.8 / 81.9 ⚠probe-drift | 94.7 / 86.8 | 94.7 / 84.9 | 95.6 / 83.5 |
| C_nm[ieee]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 95.6 / 82.5 | 96.5 / 82.9 ⚠probe-drift | 96.4 / 87.4 | 96.4 / 82.5 | 95.9 / 82.7 | 96.4 / 82.9 | 96.4 / 82.6 | 96.1 / 82.4 | 95.7 / 82.9 | 96.5 / 82.8 | 96.2 / 90.4 | 96.4 / 82.7 | 95.7 / 82.8 | 96.5 / 87.4 ⚠probe-drift | 96.5 / 82.8 | 96.1 / 82.8 | 96.4 / 82.8 | 96.3 / 82.9 | 96.7 / 83.5 | 95.7 / 82.7 | 96.5 / 82.7 | 96.2 / 82.9 | 96.5 / 82.9 | 96.5 / 87.1 | 96.5 / 82.9 | 96.5 / 82.9 | 96.5 / 82.9 | 96.5 / 82.9 ⚠probe-drift | 96.5 / 82.9 | 96.6 / 85.3 | 96.5 / 82.8 |
| C_nm[tf32]_commit_deepest | writes 1 fp32 state row (dst bank); preallocated | 16.8 / 9.7 | 16.8 / 9.7 ⚠probe-drift | 16.7 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.7 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.1 / 9.6 | 16.7 / 10.1 | 16.8 / 9.8 | 16.9 / 9.8 | 16.8 / 9.7 | 16.8 / 9.8 ⚠probe-drift | 16.8 / 9.7 | 16.8 / 9.7 | 16.8 / 9.8 | 16.8 / 9.8 | 16.8 / 11.1 | 16.8 / 9.8 | 16.9 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.8 / 9.8 | 16.8 / 9.7 | 16.8 / 9.9 | 16.8 / 9.8 | 16.9 / 9.8 ⚠probe-drift | 16.9 / 9.9 | 16.8 / 9.8 | 16.8 / 9.8 |
| C_nm[tf32]_verify | fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store) | 31.0 / 18.5 | 31.1 / 18.5 ⚠probe-drift | 31.0 / 18.5 | 31.2 / 18.5 | 32.1 / 18.5 | 31.0 / 18.5 | 31.1 / 18.5 | 32.1 / 18.5 | 31.9 / 18.5 | 31.0 / 18.5 | 31.1 / 18.5 | 32.1 / 18.5 | 31.2 / 18.5 | 31.1 / 18.5 ⚠probe-drift | 31.4 / 18.5 | 32.2 / 18.5 | 32.1 / 18.5 | 31.4 / 18.5 | 31.1 / 18.5 | 31.1 / 18.5 | 32.1 / 18.5 | 32.1 / 18.5 | 31.1 / 18.5 | 31.1 / 18.5 | 31.2 / 18.5 | 31.1 / 18.5 | 31.1 / 18.5 | 32.2 / 18.5 ⚠probe-drift | 31.1 / 18.5 | 31.1 / 18.5 | 31.1 / 18.5 |
| C_nm[tf32]_verify_out16 | bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant) | 31.2 / 18.5 | 31.1 / 20.3 ⚠probe-drift | 31.3 / 18.5 | 31.2 / 18.5 | 32.1 / 18.6 | 31.0 / 18.5 | 31.2 / 18.5 | 32.2 / 18.5 | 32.0 / 18.5 | 31.0 / 18.5 | 31.2 / 20.2 | 31.4 / 18.5 | 32.0 / 21.3 | 31.3 / 18.5 ⚠probe-drift | 32.1 / 18.5 | 32.1 / 18.5 | 32.2 / 18.5 | 32.1 / 18.5 | 31.2 / 18.5 | 32.1 / 18.5 | 32.1 / 18.5 | 32.0 / 19.9 | 31.2 / 18.5 | 31.4 / 18.5 | 32.5 / 24.3 | 31.2 / 18.5 | 31.2 / 18.5 | 32.2 / 18.5 ⚠probe-drift | 32.2 / 18.5 | 32.3 / 18.5 | 31.3 / 18.5 |
| native_sg_chain_depth5_verify(context) | T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work) | 118.1 / 104.8 | 113.8 / 102.3 ⚠probe-drift | 112.9 / 96.3 | 113.3 / 98.5 | 112.9 / 97.4 | 113.5 / 99.0 | 112.7 / 96.3 | 112.9 / 102.3 | 113.0 / 97.0 | 113.6 / 99.3 | 113.2 / 94.4 | 113.6 / 96.4 | 113.0 / 97.8 | 113.6 / 97.8 ⚠probe-drift | 112.9 / 97.2 | 113.0 / 97.3 | 112.9 / 98.8 | 112.9 / 98.9 | 113.5 / 98.0 | 112.9 / 99.6 | 113.0 / 97.2 | 113.1 / 97.9 | 113.0 / 97.0 | 112.9 / 96.0 | 113.5 / 99.1 | 112.9 / 98.6 | 113.5 / 98.3 | 112.9 / 93.5 ⚠probe-drift | 112.9 / 98.5 | 113.6 / 98.1 | 113.3 / 103.4 |
| native_pk_one_token(context) | 1 token; helper allocates mixed/a/b/out per call (NOT matched work) | 24.1 / 15.3 | 24.8 / 15.6 ⚠probe-drift | 24.9 / 15.7 | 24.6 / 15.5 | 24.1 / 15.3 | 24.8 / 15.6 | 24.2 / 15.5 | 24.9 / 15.7 | 24.1 / 15.3 | 24.9 / 15.7 | 24.8 / 15.6 | 24.2 / 15.5 | 24.1 / 15.8 | 24.9 / 15.8 ⚠probe-drift | 24.1 / 15.4 | 24.1 / 15.4 | 24.3 / 15.4 | 24.6 / 15.3 | 24.9 / 15.6 | 24.1 / 15.4 | 24.1 / 15.5 | 24.1 / 15.4 | 24.9 / 15.6 | 24.9 / 15.6 | 24.1 / 15.5 | 24.8 / 15.7 | 24.9 / 15.6 | 24.5 / 15.4 ⚠probe-drift | 24.1 / 15.4 | 24.9 / 15.7 | 24.8 / 15.7 |

In-band probe-drift criterion (4096² fp16 matmul before→after, pass iff drift ≤ 5 %): id 1: 1810.60963→1819.93122 µs drift=0.005 **PASS** [2026-09-22T05:18:13Z–2026-09-22T05:18:15Z]; id 2: 1887.51049→1787.54082 µs drift=0.053 **FAIL** [2026-09-22T05:18:17Z–2026-09-22T05:18:18Z]; id 3: 1772.27688→1767.99526 µs drift=0.002 **PASS** [2026-09-22T05:18:20Z–2026-09-22T05:18:21Z]; id 4: 1810.81753→1786.30714 µs drift=0.014 **PASS** [2026-09-22T05:18:23Z–2026-09-22T05:18:24Z]; id 5: 1781.89926→1761.75194 µs drift=0.011 **PASS** [2026-09-22T05:18:26Z–2026-09-22T05:18:27Z]; id 6: 1765.30724→1811.64951 µs drift=0.026 **PASS** [2026-09-22T05:18:29Z–2026-09-22T05:18:30Z]; id 7: 1762.77752→1770.13435 µs drift=0.004 **PASS** [2026-09-22T05:18:32Z–2026-09-22T05:18:33Z]; id 8: 1772.18552→1812.39033 µs drift=0.023 **PASS** [2026-09-22T05:18:34Z–2026-09-22T05:18:36Z]; id 9: 1759.52969→1771.72318 µs drift=0.007 **PASS** [2026-09-22T05:18:37Z–2026-09-22T05:18:39Z]; id 10: 1760.35194→1822.12639 µs drift=0.035 **PASS** [2026-09-22T05:18:40Z–2026-09-22T05:18:42Z]; id 11: 1782.57446→1768.92643 µs drift=0.008 **PASS** [2026-09-22T05:18:43Z–2026-09-22T05:18:45Z]; id 12: 1766.59679→1789.76154 µs drift=0.013 **PASS** [2026-09-22T05:18:46Z–2026-09-22T05:18:48Z]; id 13: 1819.48166→1782.71351 µs drift=0.020 **PASS** [2026-09-22T05:18:49Z–2026-09-22T05:18:51Z]; id 14: 1760.97279→1876.40152 µs drift=0.066 **FAIL** [2026-09-22T05:18:52Z–2026-09-22T05:18:54Z]; id 15: 1822.91203→1770.40958 µs drift=0.029 **PASS** [2026-09-22T05:18:55Z–2026-09-22T05:18:57Z]; id 16: 1769.48795→1778.01914 µs drift=0.005 **PASS** [2026-09-22T05:18:58Z–2026-09-22T05:19:00Z]; id 17: 1808.16956→1819.40327 µs drift=0.006 **PASS** [2026-09-22T05:19:01Z–2026-09-22T05:19:02Z]; id 18: 1841.89758→1819.58408 µs drift=0.012 **PASS** [2026-09-22T05:19:04Z–2026-09-22T05:19:05Z]; id 19: 1769.45286→1855.91049 µs drift=0.049 **PASS** [2026-09-22T05:19:07Z–2026-09-22T05:19:08Z]; id 20: 1831.64482→1815.28969 µs drift=0.009 **PASS** [2026-09-22T05:19:10Z–2026-09-22T05:19:11Z]; id 21: 1771.20647→1751.45435 µs drift=0.011 **PASS** [2026-09-22T05:19:13Z–2026-09-22T05:19:14Z]; id 22: 1821.99993→1798.61279 µs drift=0.013 **PASS** [2026-09-22T05:19:15Z–2026-09-22T05:19:17Z]; id 23: 1773.86074→1757.03526 µs drift=0.009 **PASS** [2026-09-22T05:19:18Z–2026-09-22T05:19:20Z]; id 24: 1763.13591→1768.11199 µs drift=0.003 **PASS** [2026-09-22T05:19:21Z–2026-09-22T05:19:23Z]; id 25: 1796.88644→1771.41438 µs drift=0.014 **PASS** [2026-09-22T05:19:24Z–2026-09-22T05:19:26Z]; id 26: 1775.54245→1816.21609 µs drift=0.023 **PASS** [2026-09-22T05:19:27Z–2026-09-22T05:19:29Z]; id 27: 1762.64477→1799.38889 µs drift=0.021 **PASS** [2026-09-22T05:19:30Z–2026-09-22T05:19:32Z]; id 28: 1892.02728→1766.30554 µs drift=0.066 **FAIL** [2026-09-22T05:19:33Z–2026-09-22T05:19:35Z]; id 29: 1807.50713→1810.97126 µs drift=0.002 **PASS** [2026-09-22T05:19:36Z–2026-09-22T05:19:38Z]; id 30: 1774.27368→1768.96324 µs drift=0.003 **PASS** [2026-09-22T05:19:39Z–2026-09-22T05:19:40Z]; id 31: 1762.67204→1765.60802 µs drift=0.002 **PASS** [2026-09-22T05:19:42Z–2026-09-22T05:19:43Z]

iters=200, warmup=20. Not a serving throughput measurement.

## Validity

INVALID (non-finite) comparison cells in this summary: **0**. Probe-drift failures: **3**. Manifest status: **completed**; start/end source hashes match: **yes**.
