# Native joint common-O0 preparation helper: bounded source review

**PASS for preparation-source compatibility. No blocking defect found in this successor. This is not an operational gate, source-result acceptance, recovery authorization, or qualification result.**

Reviewed `prepare_native_joint_common_o0_recovery_v1.py` SHA-256 `95ca6dd86a02aee2deffd34c3e0e61d847001ac8dbcd67253626c80bbd552280`, against preserved target-only predecessor `882f7536c1d39e7de33452f3f1a5856ac1e40ff23f0946ddd83e55bb505f2461`. No implementation changes or live commands were performed.

## Source and manifest checks

- Helper lines 15–18 require explicit three-case source-inventory acceptance with `qualification=false`, matching source-manifest and independent-review bytes, backup verification flags, and a matching pushed branch commit. Lines 23–34 reproduce the joint plan, require the selected run to belong to it, bind the accepted source SHA to that plan, and require the full 84-case / A+B / R2 / 336-observation denominator. Existing output namespaces are refused locally and remotely.
- Actual local `fullmodel/joint-native-source-v1/SOURCES.json` SHA `49ec5c4ddb97342cf2ca4db41be2027d1044366e9fa3cb08b0017961a38795b8` and `fullmodel/native-joint-common-o0-v1/CORPUS.json` SHA `ac3f1248aeb5343714334b402c9f0cd6ecaca856c68cae8c625a5c9b5b7b8f61` reproduce exactly through the actual plan. All **46 declared paths / 48 source hashes** match current local bytes. This is manifest/source parity; this review does not repeat the independent full raw source-inventory audit.
- Plan SHA `d4c47e9ed8b4701c052e7142fd7de4a91992d1295aed97832e0796acdd21f931`; launcher SHA `dd83f2863e18d82350d34adfb5e43d55e07cc3b8e234955139d46f76974186a1`.
- Helper lines 36–37 produce the exact `Q1-NATIVE-JOINT-COMMON-O0` scope consumed by launcher lines 82–98: one fresh process, one launch, 168 requests, R2, all 84 calibration cases, aligned/nonpacked, and raw KV saving. Each A or B run has a distinct bound run ID. The helper includes all **53 gate hash fields** required by the current source closure and external bindings.
- Helper lines 39–42 retain the predecessor’s authorization construction and recovery argv exactly (AST equality), including memory runner v2.8, its one-use run/gate/config binding, `global_clean_cache_readiness_v1`, and explicit **82.26 GiB** reservation. The selected runner SHA is `b85c13a77c9a6060d2f9b6bd221997851159f716efff8aa0c53751edfed22616`; readiness consumer SHA is `2230ba1dea4551b30d5efad88516a7aa6ad3487c61880b9bd0984d6ca2cc2eb9`. No prior consumed authority is reused.
- Helper lines 43–44 supply the actual launcher’s environment names and preserve the newly generated gate SHA in the launch receipt. The launcher authenticates its complete declared local closure before importing campaign modules. The earlier three-source launcher's external prelaunch guard is therefore not required for this already repaired joint-corpus launcher.

## Independent controls

**26 CPU controls passed: eight positives/invariants and eighteen targeted refusals.** These use the actual plan rebuild, actual helper assertion/gate AST, and the actual launcher Python preflight up to—but excluding—its binding-file write. Both A and B generated gates pass. Source acceptance with the wrong gate, false acceptance, wrong case count, qualification=true, stale source digest or stale review digest refuses. Invalid backup flags and pushed-commit mismatch refuse. Closed/foreign gates, multiple launches, altered or omitted dependencies, wrong request/repeat/case denominator, and an old recovery binding refuse.

The test acceptance/backup documents and Git output are synthetic. The locally absent FA2 binary is represented only by a hash/size stub, explicitly excluding binary/runtime validation. The preparation `main`, `save`, SSH, shell, recovery runner, GPU and containers are never called. No source inventory, gate or authorization was generated. Normal non-optimized Python remains the existing invocation convention for assertion-based preparation checks.

The first reviewer control attempt accidentally shared its synthetic hash-map object between negative cases, so later negatives stopped at an earlier mutation. That attempt is preserved; the corrected control deep-copies each generated gate. All final refusal paths reach their intended source/scope checks. This was a reviewer fixture defect, not an implementation change.

## Disposition and remaining action-time evidence

The parent may proceed with this exact preparation source once it supplies the real independent source-inventory acceptance and current verified/pushed backup receipt. Those live prerequisites, fresh recovery outcome, source/binary parity on the target, and launch dispatch remain the parent's responsibility. A successful source inventory or preparation does not establish native A/B repeatability, candidate qualification, workload completion, or a speed result. Original natural-prefill failures and separate packed controls remain preserved.

Evidence: `p0/monitor/review-response-20260927/native-joint-common-o0-preparation-review/` contains the immutable six-file snapshot, executable reviewer controls, preserved first attempt, final log and `RESULTS.json`. Canonical full raw reduction remains on the original remote paths; local metadata parity is not a claim of portable raw reduction.
