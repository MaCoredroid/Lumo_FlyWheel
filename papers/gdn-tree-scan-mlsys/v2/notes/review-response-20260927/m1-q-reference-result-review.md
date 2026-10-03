# M1-Q held-out reference result — independent review

**Disposition: PASS for the held-out reference prerequisite only.** The recorded `REFERENCE_ELIGIBLE` result reproduces from the authenticated saved tensors under the unchanged v4.2 rule. All eight named reference surfaces are eligible on seed 20260929 across all 48 layers. This does not qualify a C0 implementation, establish performance, or open a gate. A separately authorized C0 phase may use these exact references once its own implementation, scorer and execution prerequisites are accepted.

The reviewer performed read-only remote filesystem access and CPU reductions of existing artifacts using `/usr/bin/python3 -B`, Torch 2.4.1 without CUDA, NumPy 2.3.4, four CPU threads and empty `CUDA_VISIBLE_DEVICES`. No recurrence/reference input was generated, no candidate outcome was inspected, and no GPU, container, query, reclaim or gate operation was performed.

## Exact records

Paths below are relative to `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927` on the owned remote repository `/home/mark/lumotree-review-20260927`.

| Record | SHA-256 |
|---|---|
| `runs/m1-q-reference/heldout-reference-pack-20260928T215324Z/MANIFEST.json` | `5f6157fd525ad8b465ddcdbcbd0a54f2c04936cc551bf504da238fe4817cdba9` |
| `runs/m1-q-native/m1-q-native-20260928T215723Z/LAUNCH-RECEIPT.json` | `e5c3edd48cc3348eae62adf45682753d0cdec1453f5b73ecb097ca41c820db8d` |
| Native run `out/native/RECEIPT.json` | `8c6f855948070f8a1798f51b44d2e4628c1e2d9a52ac4ff629e6df2c30694539` |
| `runs/m1-q-reference/heldout-reference-seal-20260928T220027Z/RECEIPT.json` | `a2521810df84d9e42751f1b1d8e5777f38f885cc56c7171ada3332db479e04b8` |
| Native approved gate snapshot | `edb7c369add84a348367b61e99a16a875358307e1fb40f909fc10163074f19ac` |

Executed source bindings are unchanged from their accepted reviews: pack `f1566c7c0a9301945f0223bae626bc076e4c9c3857eb9549cf9ffd84e5075fd2`, native collector `ce58ac5793b1c1547cbeabe3e374c91000c0b0cf9c609a9708adfac91df1163e`, launcher `1bf62cfc45993921a8ec13850e24c666a8ca252d8c141358492368117e488463`, seal `1d2009774c385c8a559ae7d975806125071ad7b626f8f15d07175ba4950a4692`. Both CPU authorizations' dependencies and all native gate source bindings still match the archived hashes. Contract `b98c3af2fd1d7cccf492cc10040a5847b82aefdc940c7570cb777b805ee4ee80` and observation manifest `c5dd3936ca2a94caea87b0f3e68794f892e01c5759204d149af10b7daa10c3f3` remain unchanged.

## Raw and numerical audit

The independent reducer imports neither the pack/seal implementations nor the baseline recurrence. It authenticates saved files, reads tensors, reconstructs the declared structural witnesses, and calculates the frozen RMS and maximum-error rules directly.

- Both manifest and native receipt contain exactly seed 20260929 × ordered layers 0–47, without missing, duplicated or extra records. All 48 operand files and all eight input tensor hashes per file match. Every native record binds the same operand file.
- The 48 native files hold 96 raw repeat records: BF16 output `[28,48,128]`, FP32 uninterrupted history `[12,48,128,128]`, and FP32 wrong-sibling final state `[48,128,128]`. All are finite, both repeats match bitwise, and both initial states equal the accepted S0. Recorded native update count is 14,880 (310 per layer), consistent with the reviewed two-repeat call schedule.
- Both author C1 and C2 recorded repeat hashes, shapes and dtypes authenticate on all surfaces. There are 288 author C1 and 384 C2 shared stored objects representing originally recorded bitwise-identical repeats; this is the disclosed deduplication format, not two distinct raw files. No divergent-repeat object is needed in this result. All recorded hashes agree, and all applicable history/witness tensors are finite. The audit performed 384 S0 equality checks across native and author state repeats.
- Independent calculations use each surface's own C1/C2 pair, `1.10 × native/reference error + 2^-24 × reference scale + 2^-149` separately for RMS and maximum error. There is no added BF16 allowance, pooled-head threshold or parameter adjustment.
- All **9,216 seal layer arrays** and **6,624 original author-pack layer arrays** reproduce bitwise, including the per-cell errors, bounds, witness applicability/difference/detection masks and sealed eligibility masks. Every summary, error range and witness count agrees.
- All **488,448 eligible witness cells** are detected, with **zero undetected** cells. This includes output head/sibling controls and state no-publication, stale-update, head-permutation and wrong-sibling controls. This is power for the declared witnesses and input population only.
- The independently enumerated manifest tuples give **71,424 cells per policy; 285,696 unique reference cells total**. Each policy's digest equals both the saved seal and the originally frozen observation manifest. Missing, extra and duplicate counts are zero; candidate observations remain zero.

All native/pack/seal member bytes needed for the audit were hashed remotely: 181 unique indexed files totaling 14,008,849,287 bytes, including source bindings and the saved raw tensors. The 37 metadata/NPZ/authority files copied locally also match their remote byte counts and hashes. The raw audit took 33.87 seconds on CPU. No raw GPU tensor bulk copy was required.

## Authorization and lifetime

The preserved authority chain is ordered: all-eight baseline resolution `7bbbcfeb…` at 21:39:18Z; CPU pack authorization `69940a63…` at 21:53:24Z; pack process 21:53:47–21:55:47Z, exit 0; native gate at 21:57:23Z; one native launch 21:58:40–21:58:54Z; seal authorization `6b92319c…` at 22:00:27Z; seal CPU process 22:01:22–22:01:52Z, exit 0. Generation and native execution each have a consumed marker matching their exact authority hashes. Neither authority can be reused. The seal is a read-only reduction into its own fresh output directory, not a second native execution.

Native ownership checks agree throughout on CID `9d49720b824908590ad1dd97ac8e2714e3b3bce0f4e57ca0ab09c2d03df3f721`, nonce `0bedcc48b72a4d8f816d6d36b5ae14f4`, and image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. One container create is recorded, restart count is zero, container and attach exit codes are zero, and no OOM is recorded. Exact-CID removal succeeded and the subsequent successful `docker ps -aq --no-trunc` output proves its absence. These are authenticated saved lifetime records, not a new live inventory.

The in-process 12 GiB resource check passed with 65,001,717,760 bytes free; it did not use the unrelated full-model 82.26 GiB setting. Native runtime attestation matches Torch 2.11.0+cu130, CUDA 13.0, the accepted native module `000ab899…` and function `fc0c2869…`. Native collection was 12.728 seconds; launcher lifetime 13.595 seconds. CPU seal computation was 29.137 seconds and its outer process lifetime 29.931 seconds. These are operational durations, not decode-throughput measurements.

## Reviewer artifacts and limits

Reproducers, original metadata, authority snapshots and audit results are under `p0/monitor/review-response-20260927/m1-q-reference-result-independent-review-20260928/`:

| Artifact | SHA-256 |
|---|---|
| `audit_raw_cpu.py` | `7097fdababf3016949d7492dc1f72b5e1121bb00c71569c53f2a38e004ba3739` |
| `RAW-AUDIT.json` | `21de2f23eba8c857702bd02ea941bc7ce4f440efe660421c89da6d599ae52db0` |
| `audit_metadata.py` | `d78d44680128413571518d0b484362b085f400300dcac7d868d343201f90d017` |
| `METADATA-AUDIT.json` | `9d58f3fcd5e9f1ac5c2c85b1f10bf18785498f6bf1196f0d040eeabebeb5e342` |

The retained `RESOLVED_CANDIDATE` per-surface field is the baseline rule's disposition label; it is not a C0 pass. This review accepts the reference prerequisite on its declared population only. It does not demonstrate cross-process native determinism, GPU comparator accuracy, full-model continuation, timing, or workload/task quality. Those claims require their separately declared evidence.
