# Native joint-O0 A/B adverse-result diagnosis — 2026-09-29

No supported implementation bug or changed operator route was found in this bounded read-only diagnosis. The retained evidence localizes process-dependent divergence to the target continuation before the differing MTP proposals. It does not establish the causal kernel or justify a code fix, retry, or altered admission criterion.

## Earliest observable divergence

Examined long-prefix `n01` and `n05`, all A/B × r0/r1 record metadata, with selected r0 raw comparisons. Both have prefix length 60,083 and begin with root token 7643 then token 3521. Their chains are respectively `[7643,3521,9923]` and `[7643,3521,25256,9923]`.

- Natural bootstrap states differ across processes, but after the common-state import the captured target O0 and MTP prefix bytes match exactly. The import independently reads back both components (`q1_reference_hooks_joint_common_o0_v1.py:66–77`).
- After root 7643 at position 60083, the target hidden vector and full MTP score bytes match exactly across A/B.
- After token 3521 at position 60084, **5,010 of 5,120 BF16 target-hidden elements differ**, maximum absolute difference **0.875**. The same A/B raw hidden hashes and difference occur in both cases. This is the first differing target-hidden checkpoint captured after common O0; full MTP scores also differ here, before the later ordered-top-three mismatches.
- At O1, both cases first have differing stored GDN state at layer **53**; differing GDN layers are 53,54,56,57,58,60,61,62. Attention KV differs only at layers **55,59,63**. Earlier stored GDN states and attention KV remain equal. At attention55, n01 differs only at new position **60084**; n05 additionally differs at **60085**. The root position and older prefix match in this inspected bank.

This identifies the earliest retained state surfaces, not the exact causal layer: intermediate per-layer activations and a complete recurrent-state snapshot immediately after the root were not captured. Layer53 is not proved to be a faulty kernel. At layer53, n01 convolution state differs only in stored row2 (8,741 elements; max 0.125); its FP32 recurrent state differs in 786,374/786,432 elements (max 0.00756574). n05 carries the same convolution-row difference shifted to stored row1.

For n01, the later after-Z MTP top three are A `[628,599,2908]` versus B `[628,599,1381]`. For n05, before-Z first proposals are A `[353,11,40]` versus B `[353,11,18770]`; the follow step swaps the secondary order `[628,635,2688]` / `[628,2688,635]`. Spines remain identical. Target O2 greedy decisions remain identical, although full target scores differ (maximum absolute difference 1.1875 for n01; 1.109375 for n05). These are diagnostics, not new tolerances.

## Important interpretation correction

`q1_joint_categorical_v1.py:108–109` calls phase inputs matched when extent, input token, position, dtype, operator and k match. It retains target-hidden hashes at lines61–63 but does **not** compare them in that predicate. Thus the adverse result supports **identical joint starting O0 and declared phase metadata**, not identical full MTP operands at every later phase. The recorded target hidden already differs; this is not a same-input demonstration that MTP arithmetic or replay itself is defective. The source really clones the actual target hidden (`q1_native_mtp_history_v2.py:82–86`) and feeds it to the native first pass (`:100–113`).

## Route/configuration audit

The authenticated actual image inspection is byte-identical. Patch and patched-FA2 receipts differ only in timestamp. Actual serving commands match after substituting the two run identities. Actual environment differs only in run ID, job hash and container name; the job differs only in process/run identity, construction timestamp and its dependent canonical hash. Host settings show own log/CID paths and bind-list ordering differences, not a distinct numerical route. Both use the aligned nonpacked recurrent operator and the same patched FA2 source; no packed/eager substitution was found.

One concrete uncontrolled physical difference remains: target and MTP KV banks have shape **A `[2,9776,64,4,256]` versus B `[2,9712,64,4,256]`**, and physical block IDs differ. Active block-table counts and imported logical bytes match. The configuration fixes `--gpu-memory-utilization 0.6`, not an exact KV-block count (`q1_spec_off_engine_config_v2_1.py:43–56`). This is a physical allocation difference worth retaining in interpretation; the existing evidence does **not** prove it caused the numerical divergence or make a specific allocation change a validated fix.

## Evidence and scope

The unchanged full reduction remains adverse: 336 valid observations; target greedy 84/84; joint categorical 68/84. This diagnosis does not rerun that audit or change its result. No new inference, tests, gate actions or source changes were performed. Existing raw comparison outputs include hashes, exact lengths, shapes and finite-value checks. The representative-case selection is explanatory and does not redefine the frozen population.

Evidence directory: `p0/monitor/review-response-20260927/native-joint-common-o0-AB-diagnostic-20260929/`.

A reviewer-helper correction is preserved in `CORRECTION.json`: the original helper wrongly compared two missing intermediate KV snapshots as equal (`None == None`). Corrected `PAIRED-METADATA.v2.json` marks these unavailable. There is **no intermediate MTP-KV snapshot after the first post-import root step**; equality is established for imported MTP O0, not that uncaptured seam. Original helper/output and the correction are retained. Campaign raw evidence and reducer were unaffected.

### Exact references

- full reduction: `p0/monitor/review-response-20260927/native-joint-common-o0-AB-full-independent/REDUCTION.json` — SHA256 `9ac73cd6664b9bec19fbf0cd6759d779b830bee743924acb57d533133f9bc8d3`.
- authenticated extracted metadata: `p0/monitor/review-response-20260927/native-joint-common-o0-AB-diagnostic-20260929/metadata.stdout.json` — SHA256 `42c6fd90e52cc17a07d0b8556ae4912d758248535c1590de13824c346157b0b3`.
- corrected paired metadata: `p0/monitor/review-response-20260927/native-joint-common-o0-AB-diagnostic-20260929/PAIRED-METADATA.v2.json` — SHA256 `2de3470c3c6ef7111ce1cf5187a6c4fcdc9f5b564e4446adf018c4fcf1c05223`.
- selected raw comparisons: `p0/monitor/review-response-20260927/native-joint-common-o0-AB-diagnostic-20260929/RAW-COMPARISONS.json` — SHA256 `5fb52017b82b0036c4b4f032b19fca65f7b0a5dec5c85343c49c04ec166ee282`.
- actual environment differences: `p0/monitor/review-response-20260927/native-joint-common-o0-AB-diagnostic-20260929/ACTUAL-ENV-DIFF.v2.json` — SHA256 `cd474b493f153a9be1404b4b68fd5ccccbf19bbfc5094bca293e7b1f5e0fa33c`.
- categorical projection/reduction: `experiments/review-response-20260927/tools/q1_joint_categorical_v1.py` — SHA256 `45187dfa879c79bbdcb2bffe963cce3ca913702628086c235bb98062ce3f40f4`.
- native MTP history: `experiments/review-response-20260927/tools/q1_native_mtp_history_v2.py` — SHA256 `83dc891ca4200cbd3dfb77a92269574105dca1ee2569a919c63367f0214b066c`.
- native MTP hook: `experiments/review-response-20260927/tools/q1_reference_hooks_v2_4_mtp.py` — SHA256 `3184550f37a6aabd6777753f1632e6f192a7fdd87b3d621cc7bf62c22eb8aac6`.
- joint O0 hook: `experiments/review-response-20260927/tools/q1_reference_hooks_joint_common_o0_v1.py` — SHA256 `6d4a95e06a95f8f00f708800a146ca92a6264a0e5d80e3210a97061866c18809`.
- native engine configuration: `experiments/review-response-20260927/tools/q1_spec_off_engine_config_v2_1.py` — SHA256 `cd7fbc1322c9c4206990f2223728c2b49ae551c1a644a99fbafd2576380deb14`.
- corpus: `experiments/review-response-20260927/fullmodel/native-joint-common-o0-v1/CORPUS.json` — SHA256 `ac3f1248aeb5343714334b402c9f0cd6ecaca856c68cae8c625a5c9b5b7b8f61`.
- joint source manifest: `experiments/review-response-20260927/fullmodel/joint-native-source-v1/SOURCES.json` — SHA256 `49ec5c4ddb97342cf2ca4db41be2027d1044366e9fa3cb08b0017961a38795b8`.
