# Whole-draft independent red-team before E1 integration

Review date: 2026-09-22. **Reviewed main.tex SHA-256 `1857d76012642eb284ac8e1c4361a7af6bf4fa5d21697751d79d19edaabfe606`; abstract.tex `9a29ea33ffb9a3fdb84ee3daf980861535f0ab9414165e0516c6b67170d7b0b0`.** This is a fresh whole-paper evidence/logic pass integrating historical H1–H6, P0, E7a, stage-isolated E7b, and closed selected-route B1/B4 findings. E1 launched separately at 10:00:28 UTC; its pending measurements/results integration are known outstanding work, not a newly discovered finding. No inference, GPU operation, source mutation, or experiment expansion was performed.

**Verdict: two bounded E7a labeling corrections remain. The central scoped claims are supported; no additional experiment beyond the already planned E1 work is required.** Neither correction changes a numerical result or turns a failed frozen criterion into a pass.

## 1. P2 — Table E7a's verifier-error column must identify its output specialization

**Passages:** main.tex:323 (“Errors are maxima relative to the reference maximum” and “fp32/tf32 denote the precision of the dense products”), main.tex:328 (“Verify out”), main.tex:344, and abstract.tex:2 (“With fp32 products, errors against a float64 oracle are on the order of fp32 roundoff; tf32 increases error by roughly three orders of magnitude”).

The table's verifier-error numbers come from **fp32-output-store diagnostic specializations**, not the deployed bf16-output tensor compared with a float64 oracle. Dense-product precision alone does not identify that measurement. The adjacent bf16-agreement column correctly measures a different comparison.

Direct example: `out-20260921T230026Z-e7a-ladder-v3/result_01_layers_62_linear_attn.json → stage1_verify`:

| Raw key | max_rel_to_refmax |
|---|---:|
| `A_prod_out32_vs_oracle` | 1.335914729147904e-7 |
| `A_prod_out16_vs_oracle` | 2.206022704743479e-3 |
| `B_fs[ieee]_out32_vs_oracle` | 2.920470839432626e-7 |
| `B_fs[ieee]_out16_vs_oracle` | 2.206022704743479e-3 |

Across the nine historical/synthetic sets, A's diagnostic fp32 relative output error is 9.2698e-8–3.1318e-7, while the deployed-width bf16 error against the float64 oracle is 0.0018730–0.0035817. This is ordinary output quantization and does not contradict the close bf16 agreement between implementations. Main.tex:344 eventually describes separately compiled fp32 stores, but the standalone table and abstract do not carry that distinction.

**Minimum correction:** label the verifier-error column/caption “fp32-output-store diagnostic” and qualify the abstract's roundoff/three-orders statement accordingly. Keep deployed bf16 agreement separate. Say “diagnostic fp32 outputs” or “fp32-output-store specializations,” not “the deployed kernel before rounding”: changing output dtype recompiles the kernel, and the confirmation already demonstrates specialization-dependent differences. **No experiment is needed.**

Also clarify main.tex:348's “the bfloat16 output the server actually emitted ... is byte-identical to both” as “byte-identical to the corresponding scan and native bf16 outputs.” The parent's proposed clarification is supported by all eight fresh result files: `A_prod_out16_bytes_equal_payload_serving_out=True`, `A_prod_out16_vs_payload_serving_out_bitwise_frac=1.0`, and `native_sg_out16_bytes_equal_payload_serving_out=True`. It must not imply byte equality across fp32 and bf16 dtypes.

## 2. P2 — The fresh prefixes reproduce the pattern, not the exact table ranges

**Passage:** main.tex:323 says “the eight fresh, independently verified prefixes reproduce these ranges (text).”

Fresh extrema extend outside the table's historical/synthetic ranges:

| Quantity | Historical/synthetic raw extrema | Fresh pilot raw extrema |
|---|---|---|
| Native one-token/speculative-update state gap, relative to reference maximum | 2.2050e-4–1.6455e-3 | 1.2958e-4–1.3015e-3 |
| B tf32 bf16 agreement with native | 74.4206–78.5710% | 76.9434–81.1605% |
| B/C IEEE diagnostic fp32 output relative error | 2.7459e-7–5.2179e-7 | 2.2138e-7–3.6210e-7 |

Main.tex:348 reports the fresh pattern/extrema consistently. **Minimum correction:** “the eight fresh prefixes reproduce the precision pattern; their extrema are reported in the text.” **No experiment is needed.**

## Whole-paper claims checked and retained

| Area and passages | Evidence and disposition |
|---|---|
| State contract and H1 (86–151, 225–245) | The materialized-state/pending-token distinction is explicit; the logical recurrent equation is consistent with the stated state orientation. H1 archived report line14 supports 15/15→0/15; lines27–28 support 0/84 and0/61 script flags. These remain documentary fixture/observation claims, not a current whole-model or distribution proof. |
| Historical FA2 layout versus current TREE_ATTN (153–165, 370–380) | No contradiction in the detailed scope: the historical shared-spine permutation repair belongs to the forked FA2 route; the later flat-slot requirement is explicitly backend-specific. The superset statement at 163 is conditional on a longest-matching-path selector, whereas 380 correctly states the current duplicate-source sampler has no such guarantee. Neither theorem should be silently transferred to the other route. |
| Historical H3 accounting (202–220, 247–285) | Independently recomputed all three source JSONs. Rates are 42.7439324/39.9466257/32.8545038, event costs 103.448469/148.283588/160.879293 ms, with exactly the recorded mixed populations. All five derived break-even/gain values match the saved JSON. The table, figure caption, abstract and conclusion now consistently call this a proxy, avoid host-time attribution, and separate its support from emitted throughput. |
| H4/H5/H6 (169–173, 287–311, 418) | The hash-matched H4 closeout reports 27.03 ms, SD 13.77, one-sided lower bound 10.82; the paper preserves its historical, separate-campaign status. This pass did not reconstruct the individual four paired passes. H6's historical WY state drift and later correction are documented at FR13_REPLAY_CHASEDOWN_BANK.md:127–130 and the corrected WY verdict:3; the paper correctly treats them as a local prototype investigation, not a TreeWY refutation. H5 does not claim powered quality/non-inferiority. |
| E7a arithmetic/freeze (314–358) | Re-read and hashed all 9 historical/synthetic, 8 fresh-pilot, 31 primary confirmation result JSONs, plus the original freeze-audit evidence. Confirmation bf16 minimum C/p005 =0.9997721354166667; replay max absolute error 1.3365283662025718e-6; compact max 7.814217335422313e-7. The manuscript's rounded figures agree. It retains 31/32 provenance denominator, failed padding/order/timing/bitwise criteria, pilot-rationale errors, and dispersion-versus-confidence-interval distinction. No promotion is implied. The two labeling issues above are the remaining corrections. |
| E7b publication and horizon (360–368) | Matches prior independent raw reviews: 12 aligned verifier calls/120 rows; 13 captures/publications and 12 next-state links distinguished from 11 complete API post-boundaries; once-injected layer continuation has 10 fully API-bound future intervals. It is separate from repeated live substitution and cross-layer feedback. All six diagnostic boots' remap-off limitation remains disclosed. No new compact-method experiment is required to report these observations. |
| Selected route and exact API accounting (370–380) | The policy-A first-forward witness, policy-B B1 repair, helper/source composition, 53 B4 numerical steps/49 exact links, 32 blocks, 85 structural/83 API-visible captured tokens, 128 total API IDs and 7 complete intervals all match the closed reports. Derived API IDs are distinguished from direct ID capture. No live full-bank conv/KV oracle, full-model sequential proof or stochastic law is claimed. |
| Scope/novelty/conclusion (47–81, 382–420) | Contribution is failure witnesses, state-boundary analysis and scoped measurements. The paper denies first-system, general bitwise, distribution and external-system superiority claims. E3/E4/E5/E6/E7c remain conditional/deferred for corresponding stronger claims. Current E1 preflight/results remain outstanding and will be integrated later. |

The figure source and algorithm were also read. They are logical/mechanism illustrations, with no new measurement claim. The H3 residual figure uses “Residual estimate” and its caption preserves the unmatched-support limit.

## Primary-source related-work check

[Bole v1, §IV–VI](https://arxiv.org/html/2608.01651v1) supports the finite-depth series through power d, compact state reconstruction, SGLang/CUDA-graph integration, unquantized model weights, Qwen3.5-27B on GB10, and online agent-session replay. The draft explicitly distinguishes its local B/C implementations and avoids attributing their fp32/tf32 choice or measured runtime to Bole's authors. No external-system performance comparison is required for this scoped claim.

[TreeWY v1, §3–5](https://arxiv.org/html/2608.20961v1) supports a triangular solve with pseudo-value reconstruction, vLLM/Qwen3.5 35B/397B on B200, greedy evaluation, disabled prefix caching, finite-precision/nonidentical token streams, and wider-tree acceptance gains without an established width-throughput win. The draft's descriptions and limits are accurate.

[ReplaySSM RFC #47572](https://github.com/vllm-project/vllm/issues/47572) explicitly covers Mamba2 and GDN input caching/deferred materialization. The modest characterization at main.tex:68 is supported. These source checks do not establish independent replication of any author implementation.

The two author-site HTML URLs for H2/H4 were unavailable through the web tool in this pass. Their already-preserved archive evidence was read locally and hash-checked; this is not treated as a newly failed experiment or erased evidence. All14 entries in notes/evidence-sources.json matched the available local bytes.

## Necessary experiments and stopping rule

**No additional experiment is required by any presently asserted claim beyond the currently planned E1 work.** Finish the already specified native-route preflights and retain the initial 18 timing cells with the frozen support/precision rules, including failures or insufficient support. Then integrate actual E1 outcomes with the same distinction between matched decode support and full-service latency. Failed E7a criteria remain reportable findings; no compact arm needs to be made passing. Another tree boot, forced-path campaign, full-model sequential oracle, stochastic/graph/cache study, quality sweep, or author-system reproduction would answer a stronger deferred question.

The remaining release work is the two narrow wording repairs, intended clarification of the served bf16 comparison, and final evidence-companion/results synchronization already owned by the parent. This report does not reopen the earlier closed E2/selected-route findings or infer a completed E1 result from capture pilots.

## Reviewed identities

The following JSON retains the manuscript/evidence hashes and the raw E7a reductions used in this pass. Experiment-result paths are relative to the preserved `artifacts/e7a-evidence-20260922T0700Z/experiments/` snapshot; other paths are relative to the paper v2 directory. The hash-map digest is `f058ccd607d1842034e31a16cf82f9994e8f7e27d6eeacd44522abc5097ae7df`. Fresh source hashes were held fixed during this review.

<details>
<summary>Exact hashes and recomputed E7a extrema</summary>

```json
{
  "runs": {
    "historical_synthetic": {
      "n": 9,
      "A_out32_rel": [
        9.269778121188484e-8,
        3.1317697907837586e-7
      ],
      "A_out16_rel": [
        0.0018730207432038758,
        0.0035817310571393465
      ],
      "A_out32_native_bits": [
        1,
        1
      ],
      "A_out16_native_bits": [
        0.9999837239583333,
        1
      ],
      "native_state_gap_rel": [
        0.00022050351838155108,
        0.0016454703045688979
      ],
      "B_bf16_native": [
        0.9996744791666666,
        0.9999348958333333
      ],
      "C_bf16_native": [
        0.9996744791666666,
        0.9999348958333333
      ],
      "B_tf32_bf16_native": [
        0.7442057291666666,
        0.7857096354166667
      ],
      "C_tf32_bf16_native": [
        0.7445638020833333,
        0.7876627604166666
      ],
      "B_fs[ieee]_out32_rel": [
        2.7459077893898016e-7,
        5.217899231810301e-7
      ],
      "B_fs[ieee]_compact_vs_oracle_abs": [
        6.605215530086639e-8,
        0.000003531004297840923
      ],
      "C_nm[ieee]_out32_rel": [
        2.7459077893898016e-7,
        5.217899231810301e-7
      ],
      "C_nm[ieee]_compact_vs_oracle_abs": [
        7.745067998055077e-8,
        0.000004241275966876401
      ],
      "replay_vs_oracle_abs": [
        8.814532415080123e-8,
        0.0000032171737327502115
      ]
    },
    "fresh_pilot": {
      "n": 8,
      "A_out32_rel": [
        1.0686702522846833e-7,
        2.1637695540916573e-7
      ],
      "A_out16_rel": [
        0.001924957873228766,
        0.002966800670599147
      ],
      "A_out32_native_bits": [
        1,
        1
      ],
      "A_out16_native_bits": [
        1,
        1
      ],
      "native_state_gap_rel": [
        0.0001295800348589626,
        0.0013014720762023345
      ],
      "B_bf16_native": [
        0.9997884114583333,
        0.9999348958333333
      ],
      "C_bf16_native": [
        0.9998046875,
        0.9999186197916666
      ],
      "B_tf32_bf16_native": [
        0.76943359375,
        0.8116048177083334
      ],
      "C_tf32_bf16_native": [
        0.769580078125,
        0.8113932291666667
      ],
      "B_fs[ieee]_out32_rel": [
        2.2138191377352123e-7,
        3.6210376937940316e-7
      ],
      "B_fs[ieee]_compact_vs_oracle_abs": [
        3.971868167695902e-7,
        7.867071634137801e-7
      ],
      "C_nm[ieee]_out32_rel": [
        2.2138191377352123e-7,
        3.6210376937940316e-7
      ],
      "C_nm[ieee]_compact_vs_oracle_abs": [
        3.971868167695902e-7,
        7.867071634137801e-7
      ],
      "replay_vs_oracle_abs": [
        5.062497452357206e-7,
        0.0000012969487670488888
      ]
    },
    "confirmation": {
      "n": 31,
      "A_out32_rel": [
        9.386650944971413e-8,
        3.795789994970479e-7
      ],
      "A_out16_rel": [
        0.0016660991443321154,
        0.00313655271677231
      ],
      "A_out32_native_bits": [
        1,
        1
      ],
      "A_out16_native_bits": [
        0.9999837239583333,
        1
      ],
      "native_state_gap_rel": [
        0.00015221993377159657,
        0.0008897970621548502
      ],
      "B_bf16_native": [
        0.9998046875,
        0.9999674479166667
      ],
      "C_bf16_native": [
        0.9997721354166667,
        0.9999674479166667
      ],
      "B_tf32_bf16_native": [
        0.7761881510416666,
        0.8230794270833334
      ],
      "C_tf32_bf16_native": [
        0.7788411458333333,
        0.8314941406249999
      ],
      "B_fs[ieee]_out32_rel": [
        2.171600305206214e-7,
        5.856741550413446e-7
      ],
      "B_fs[ieee]_compact_vs_oracle_abs": [
        2.809754127142128e-7,
        7.814217335422313e-7
      ],
      "C_nm[ieee]_out32_rel": [
        2.171600305206214e-7,
        5.856741550413446e-7
      ],
      "C_nm[ieee]_compact_vs_oracle_abs": [
        3.0909358272879217e-7,
        7.814217335422313e-7
      ],
      "replay_vs_oracle_abs": [
        3.973203401841374e-7,
        0.0000013365283662025718
      ]
    }
  },
  "hashes": {
    "out-20260921T230026Z-e7a-ladder-v3/result_01_layers_62_linear_attn.json": "6bd9fd192ee132723ef7d388b7d4d33ebde3b89dda7c5a72b0192c89978d09d2",
    "out-20260921T230026Z-e7a-ladder-v3/result_02_layers_1_linear_attn.json": "a9c9842bbab2a5ab29a833f3936b68f5508f2117991048c743208f8e93289a5e",
    "out-20260921T230026Z-e7a-ladder-v3/result_03_layers_12_linear_attn.json": "1be5c2dbb71d7032f7f907a9dad53125298e384c5678220429577ee5497b63c4",
    "out-20260921T230026Z-e7a-ladder-v3/result_04_layers_0_linear_attn.json": "0fb2c31902dd7286c9b7229a14bd9c1d3af203c63e2baf23ada810ba6233ac48",
    "out-20260921T230026Z-e7a-ladder-v3/result_05_synthetic_chain1_historical-like.json": "2f6cd9da8f5f1aa181a51a106c89dd527926a9cc1f0063724c47abc25a7f84b6",
    "out-20260921T230026Z-e7a-ladder-v3/result_06_synthetic_chain5_historical-like.json": "4245c7aaca4f5975cd34cd3cbd73edfe94ccce6b770f43f828e143dbffa9d45a",
    "out-20260921T230026Z-e7a-ladder-v3/result_07_synthetic_chain11_historical-like.json": "74182edc70574db98925de56823d8821c044261ec184d4367ed611d6381ed883",
    "out-20260921T230026Z-e7a-ladder-v3/result_08_synthetic_binary3_historical-like.json": "d092e9bfbc9e2b4f01097eded70b862d8fc80b57f086a45b1f457ca72bea86cd",
    "out-20260921T230026Z-e7a-ladder-v3/result_09_synthetic_caterpillar_historical-like.json": "a428165226a1c232502537a4f4141105cece3d013e9383d275aeb833b2ccecc8",
    "out-20260921T230026Z-e7a-ladder-v3/result_b4_synthetic.json": "908a311e31a3ea71e708cc370bc0260ce884f5905fccec1e5b737f1dc2e24d95",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_01_layers_62_linear_attn.json": "4cb645cc14d3eb32763e4ea0a0dfe3c7da97f6705c266847748a3618959780e5",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_02_layers_62_linear_attn.json": "9d167c3f59fae6e87330e7f6d0247bbd0da5d8f7f7730b860853484e5d9cab4f",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_03_layers_62_linear_attn.json": "456ae619c8481fe6bca7c547e5388a50cee1c8cb141a4e30926fc80226ff4f93",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_04_layers_62_linear_attn.json": "999e6dcbc3d381c6bc5a2ff48e3cde423c3cb1743f3bc21d58e98b88bb333252",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_05_layers_62_linear_attn.json": "c5166c4f4d5d0289a4e19b261a21a77c6b2f36311dfd2567dea71ce2d5950e9d",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_06_layers_62_linear_attn.json": "0d0ea8369a22bb94d71b12bdb3b9fa087c13659611bee50a79b79d7cb958cc67",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_07_layers_62_linear_attn.json": "bce11c7d1b66fef06548ed538b1d1c0b549d93cec34d8648aba9a31bb7908444",
    "out-20260922T013313Z-e7a-fresh-pilot-v2/result_08_layers_62_linear_attn.json": "f57497b6ee804e4be9eaad0b3e505f1f300d14a1282fa1cbb0636a86cfe9d565",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_01_layers_62_linear_attn.json": "71deaaf2b745e6d145619108815f8b92fda9b1ca961fd1eaf5380678b4965b4a",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_02_layers_62_linear_attn.json": "bdcfac7d61923f436528ce013e3ed7d93628f0929fa99b55aaa62b1944791697",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_03_layers_62_linear_attn.json": "9ff508647c7a959d4621a548e5db3af7da10a98eedabff1109cbfee99788fc01",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_04_layers_62_linear_attn.json": "6e0b4985e8c7121c2944cbec48d0be48ea993fa7b64dd09dc748c391115a8733",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_05_layers_62_linear_attn.json": "1b72a4d9983f40273dadc1df277aa1048acdd08b2182aa4d222dadbb4f4cb37c",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_06_layers_62_linear_attn.json": "c80ad6a602ba43d5c4e2939063e4bcc5e8c1c56929de475a70c5a7bb3e7f34c4",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_07_layers_62_linear_attn.json": "52380838c678c3ca4420585982436a958bb54482eed6f18b0d29f9205e06b572",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_08_layers_62_linear_attn.json": "28cf2d0d92e5f87d8c8a1bb830960b60fcd99bd84f9b6ecc611e1cdb4011da24",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_09_layers_62_linear_attn.json": "f6f8160e48c55610f9cecb8a7a05c574f5a3a5fe1ff546aea41c4b7b271af0cb",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_10_layers_62_linear_attn.json": "60331a814056bb7cfe2fea221857c477d5543454fda8902737656d896998b7cc",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_11_layers_62_linear_attn.json": "77207af5dd8533bb530119435034ca43b16fbc99fd2d0d93f96840fa5fee3687",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_12_layers_62_linear_attn.json": "bb5bc1f38692ab75bad080fbf12a8d066d41b8a42c813bfdcffca73c8b7b422d",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_13_layers_62_linear_attn.json": "0a3730ec89fe9c83338a6df1a782c0fa4cf26a2ee2b8db19ce58de9beb1c487d",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_14_layers_62_linear_attn.json": "81aafb05cbea28767b1ba22ff101631a5838a2b652549d7ea9762b84cd1eda4f",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_15_layers_62_linear_attn.json": "a74480edf6af87e51bd64dd0e4503b06a391f40e5421ac7075ade10093d6d6f6",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_16_layers_62_linear_attn.json": "9499181dd412d548b1cb400bc305c30aa6e32c921cb0cf86184d6b9ee97f0862",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_17_layers_62_linear_attn.json": "41ec45209534429720c941494eece04fbdd7519a0e21d9dc3a48479d9214a3d3",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_18_layers_62_linear_attn.json": "f3d1af5bce2ecd0a3f066f2fd58a7c3cea5e02a9e716fa7af9bde18a2a0aab88",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_19_layers_62_linear_attn.json": "36984121c85b53b5a52c9447e944e37c3147a1dd2298e324594fd3c73ffbf63c",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_20_layers_62_linear_attn.json": "d9de498913dd8825746974f869d2a9df2c29d86dcbac880306bb5abf84d983df",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_21_layers_62_linear_attn.json": "4ba6fb3cfbd1afddff1ffd829b5f53a5c695a49fcb7e8b45c6f5d77708cf1e02",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_22_layers_62_linear_attn.json": "4d88947eb435568c8a6c95d1db6f5dd9420a1b112c77bf964e0f8031d0e82353",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_23_layers_62_linear_attn.json": "e8f3e51e54a91d5efe446b8726dc8a9a5fbdd72f1b5fc3a15f6a65bd7df6ecc3",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_24_layers_62_linear_attn.json": "1297fba96870cbbac699e51aac15ad24bfe89c8c7910107e77ad71d1136a392f",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_25_layers_62_linear_attn.json": "595ae338072ee1b433dca8a508762f8f77d80f36495a3c4b437592d74479b023",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_26_layers_62_linear_attn.json": "5f38d4ed1564964924b6c3ca0ba6ab8c2ae5bd97dbbf6d0caf3e64806ac154d2",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_27_layers_62_linear_attn.json": "fca828c47da8ffd1e99935f312e885b095b0bf46b263bb961b7b040f45f4ceb3",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_28_layers_62_linear_attn.json": "36597bf34b3060d1f5abe2234f5153fa7e43aa73d4d013523995371b97be7dec",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_29_layers_62_linear_attn.json": "433278d1ee85fd213b974f4d826a882286bc6ce3c016bd9df06901ed57a63613",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_30_layers_62_linear_attn.json": "3aaf6fdf42a3c0acf1b00c6e9054fb130212a01de8c9dd6ac05986c2e08ddc77",
    "out-20260922T051611Z-e7a-fresh-confirmation/result_31_layers_62_linear_attn.json": "9b6c306c35c90c4d4c531ae03d751e1439aac02210c1ad55f09e2b860909aa3e",
    "main.tex": "1857d76012642eb284ac8e1c4361a7af6bf4fa5d21697751d79d19edaabfe606",
    "abstract.tex": "9a29ea33ffb9a3fdb84ee3daf980861535f0ab9414165e0516c6b67170d7b0b0",
    "ref.bib": "51c788f1ffcaef1cc1b1a7846be1e862cbbc56e58d041c37bfc267d703e22905",
    "results/historical-macros.tex": "228e73de3c30abb84d728c7a86f814bb31f93f6f3b2970394956c7ab881b6c09",
    "results/derived-accounting.json": "0ae3f4c186da6ff19c4bbf07caad40ff88f18f87fc02537b460f4f9c01d41d95",
    "results/historical-supports.json": "2a85237b412479febae72886597e4f64aa0a6e18f8ac3dd3b66a1799837794e7",
    "notes/evidence-sources.json": "c8dbf013f7ba1c8fb3344bfe4191e99b77213c53460881772a421daee52bb41b",
    "scripts/audit_evidence.py": "e801bce52db5ecf761514b1aae6a346c4b50ed5ee368c459a2a3156693050433",
    "p0/monitor/2026-09-22-review-14-frozen-audit.json": "63556f9ba5a70c0268af981bc8401ecf7fcd10361a1428c15ceb93408ed0d11b",
    "p0/monitor/validation-redteam-round15.md": "c6aaf69802050e0a0d727f4e46d832a2465999379b5ea083b21bcd44e7b86826",
    "p0/monitor/validation-redteam-round18.md": "83b7e1669b9df258a5079efc8b60fbd7328cdf3ea0f08ae7adfdff911f3036b1",
    "p0/monitor/e2-b4-final-redteam.md": "105d5eb78029e70a864cbefa3cb71a2886f94733d9a107055be59ef77d654788",
    "p0/monitor/e2-policyB-b1-redteam.md": "dd7de6d504fdd4daf8d091b36cc8e02b9ebe7c89aee7e61e81fa518b51e1be8f",
    "p0/monitor/e2-closure-decision.md": "422b53087cc46fce0d131479a8839444b2bb473b7db6fa905c334e975c70c2af",
    "p0/monitor/e2-b4-api-inversion-redteam.md": "f3b37263801f0faeaf4ecd4faeaeb6382b0e014fac54501109e4367e1e0a388a",
    "p0/monitor/e2-fixture-final-recheck.md": "3d3c7343664a777cbc5e9df886b4026ea604751e4264cdf1a4fd4bfc28397f8f"
  },
  "hash_manifest_sha256": "f058ccd607d1842034e31a16cf82f9994e8f7e27d6eeacd44522abc5097ae7df"
}
```

</details>

## Delta closure — 2026-09-22 10:15 UTC

Both wording findings above are closed. This bounded recheck read the corrected E7a caption/header and interpretation paragraphs (`main.tex:323`, `328`, `344`, `348`) and the abstract, against the already audited raw evidence; it did not repeat the whole-paper review or launch an experiment. Local and remote source hashes agree:

- `main.tex`: `f52a828f4cb13aca925be7d081da0c4c438072762b673de9a2933b7396032212`.
- `abstract.tex`: `0eddd1990dfe88aa1b0251526448d24983c7d0f67821d2f2921bb63df18755ce`.

The caption now claims replication of the precision pattern and refers to the separately reported fresh extrema. The caption, header, main text, and abstract identify the diagnostic fp32-output-store specialization; they no longer imply that the deployed bf16 tensor has float64-oracle error near fp32 roundoff. The fresh served payload is explicitly compared with the corresponding scan and native **bf16** outputs, matching the eight raw pilot records. Committed-state error retains its own valid scope. Frozen failures and full-model limitations remain explicit.

No actionable issue remains from this whole-paper pass after these corrections, and no additional experiment is required by its findings. E1 result integration and a subsequent changed-section check remain pending under the existing plan; this disposition does not pre-approve unseen timing results, their native-control qualification, or the final PDF layout.
